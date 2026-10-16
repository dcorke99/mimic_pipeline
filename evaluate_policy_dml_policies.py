"""Run the recursive Policy-DML estimator on existing fixed-period panels."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fit_nuisance_models as nuisance
import pipeline_config as run_config
from pipeline_config import BOOTSTRAP_MODES, NUISANCE_MODEL_TYPE
import policy_eval_common as pec
import policy_dml_common as dml
import policy_inference as inference

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260923
SETTINGS = dml.DMLSettings()
# Empty by default: no universal support/clipping threshold is imposed.
CLIPPING_SENSITIVITY = ()
PANEL_RUNS = tuple((name, *run_config.PANEL_PATHS[name]) for name in run_config.EVALUATION_PANELS)
SUMMARY_COLUMNS = {
    "cauti": "predicted_cauti_risk", "recatheterisation": "predicted_recatheterisation_risk",
    "death": "predicted_death_risk", "icu_exit_alive": "predicted_icu_exit_alive_risk",
    "catheter_exposure_days": "expected_mean_catheter_exposure_days",
}


def _normalise_keys(frame):
    frame = frame.copy()
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["inserted"] = pd.to_datetime(frame.inserted, format="mixed")
    return frame


def load_saved_behaviour(decisions, model_dir):
    """Load the existing patient-level OOF behaviour probabilities."""
    directory = Path(model_dir)
    score_path = directory / "nuisance_predictions.csv"
    scores = _normalise_keys(pd.read_csv(score_path, low_memory=False))
    dml.require_columns(scores, [*dml.KEYS, dml.FOLD, "p_remove_obs"], "saved predictions")
    output = decisions[[dml.ROW, *dml.KEYS]].merge(scores[dml.KEYS + [dml.FOLD, "p_remove_obs"]], on=dml.KEYS, how="left", validate="one_to_one")
    dml.probabilities(output.p_remove_obs, "required OOF prediction")
    dml.validate_folds(output)
    return output, {"source": str(score_path), "held_out_folds": sorted(output[dml.FOLD].unique().tolist())}


def policy_summary(episodes, decisions):
    rows = []
    for (name, kind, day), group in episodes.groupby(["policy_name", "policy_type", "policy_remove_day"], dropna=False, sort=False):
        row = {"policy_name": name, "policy_type": kind, "policy_remove_day": day,
               "estimator": "policy_dml", "prediction_mode": "recursive_patient_crossfit",
               "n_patients": group.subject_id.nunique(), "n_episodes": len(group),
               "n_complete_prediction_episodes": len(group), "n_incomplete_prediction_episodes": 0,
               "catheter_exposure_measure": pec.EXPOSURE_MEASURE}
        for outcome, col in SUMMARY_COLUMNS.items():
            row[col] = float(group[f"score_{outcome}"].mean())
            row[f"observed_event_count_{outcome}"] = float(decisions[f"reward_{outcome}"].sum())
            if outcome != "catheter_exposure_days":
                row[col + "_pct"] = row[col] * 100
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary.policy_name.eq("current_practice")]
    if len(baseline) != 1:
        raise ValueError("Exactly one current-practice reference required")
    comparison = {k + "_risk" if k != "catheter_exposure_days" else k: v for k, v in SUMMARY_COLUMNS.items()}
    return pec.add_standard_comparisons(summary, baseline_label="current_practice", comparison_map=comparison)


def patient_statistics(episodes):
    """Sum unnormalised recursive scores and episode counts within patients."""
    subjects = pd.Index(sorted(episodes.subject_id.unique()))
    entries, sums = [], []
    for (policy, day), frame in episodes.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        codes = subjects.get_indexer(frame.subject_id)
        for outcome in dml.OUTCOMES:
            entries.append({"estimator": "policy_dml", "policy_name": policy, "policy_remove_day": day, "outcome": outcome})
            sums.append(np.column_stack((np.bincount(codes, weights=frame[f"score_{outcome}"], minlength=len(subjects)),
                                         np.bincount(codes, minlength=len(subjects)))))
    return subjects, pd.DataFrame(entries), np.stack(sums, axis=1)


def estimate_from_counts(statistics, counts):
    totals = np.einsum("i,ijk->jk", counts, statistics)
    if (totals[:, 1] <= 0).any():
        raise ValueError("Bootstrap draw contains no eligible episodes")
    return totals[:, 0] / totals[:, 1]


def run_bootstrap(episodes, decisions, policies, output, *, mode, settings, learner, n_bootstrap, seed, regressor_factory=None):
    if mode not in ("fixed", "refit") or n_bootstrap < 0:
        raise ValueError("Invalid bootstrap mode/count")
    subjects, entries, statistics = patient_statistics(episodes)
    point = estimate_from_counts(statistics, np.ones(len(subjects)))
    values = np.empty((n_bootstrap, len(entries)))
    diagnostics = []
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    for replicate in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        multiplicity = pd.Series(counts, index=subjects)
        if mode == "fixed":
            values[replicate] = estimate_from_counts(statistics, counts)
        else:
            # Entire patient ownership, including duplicate copies, stays fixed.
            scored, _, _ = dml.crossfit_policy_scores(decisions, policies, settings=settings,
                behaviour_learner=learner, patient_counts=multiplicity, regressor_factory=regressor_factory)
            sampled = dml.episode_scores(scored)
            sampled_subjects, sampled_entries, sampled_stats = patient_statistics(sampled)
            if not sampled_entries.fillna(-1).equals(entries.fillna(-1)):
                raise ValueError("Refit bootstrap result entries changed")
            values[replicate] = estimate_from_counts(sampled_stats, multiplicity.reindex(sampled_subjects).to_numpy())
        diagnostics.append({"bootstrap_replicate": replicate + 1, "n_patient_draws": int(counts.sum()),
                            "n_unique_patients_resampled": int((counts > 0).sum()),
                            "n_episodes_resampled": int(counts @ statistics[:, 0, 1]), "bootstrap_mode": mode})
        if (replicate + 1) % 5 == 0 or replicate + 1 == n_bootstrap:
            print(f"[BOOTSTRAP] estimator=policy_dml; bootstrap={mode}; completed={replicate + 1}/{n_bootstrap}", flush=True)
    if n_bootstrap:
        inference.bootstrap_replicate_table(entries, values, pd.DataFrame(diagnostics)).to_csv(output / "policy_dml_bootstrap_replicate_estimates.csv", index=False)
        interval = inference.interval_table(entries, point, values, len(subjects))
        interval["bootstrap_mode"] = mode
        interval["nuisance_refitting_uncertainty"] = mode == "refit"
        interval.to_csv(output / "policy_dml_policy_value_confidence_intervals.csv", index=False)
        differences = inference.interval_table(entries, inference.paired_differences(point, entries), inference.paired_differences(values, entries), len(subjects))
        differences["bootstrap_mode"] = mode
        differences.to_csv(output / "policy_dml_policy_difference_confidence_intervals.csv", index=False)
    return values


def run_panel_estimation(panel_path, artefact_root, model_type, bootstrap_mode, panel_name=None, *, output_root=None,
                         settings=None, n_bootstrap=None, max_patients=None, regressor_factory=None):
    settings = SETTINGS if settings is None else settings
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    if bootstrap_mode not in ("fixed", "refit"):
        raise ValueError("Unknown bootstrap mode")
    panel = nuisance.load_panel(panel_path)
    decisions, _ = dml.prepare_decisions(panel, settings)
    policies = dml.configured_policies(run_config.POLICY_DAYS)
    saved, provenance = None, None
    model_dir = Path(artefact_root) / "nuisance_models" / model_type
    # Reuse suitable primary OOF action predictions in either mode. Full refit
    # draws always refit g and Q. Fixed mode explicitly requires saved g.
    if (model_dir / "nuisance_predictions.csv").exists() or bootstrap_mode == "fixed":
        saved, provenance = load_saved_behaviour(decisions, model_dir)
    if max_patients is not None:
        selected = sorted(decisions.subject_id.unique())[:max_patients]
        decisions = decisions.loc[decisions.subject_id.isin(selected)].copy()
        if saved is not None:
            saved = saved.loc[saved[dml.ROW].isin(decisions[dml.ROW])]
    rows, audit, folded = dml.crossfit_policy_scores(decisions, policies, settings=settings, behaviour_learner=model_type,
                                                   saved_behaviour=saved, regressor_factory=regressor_factory)
    episodes = dml.episode_scores(rows)
    summary = policy_summary(episodes, decisions)
    output = Path(output_root or artefact_root) / "policy_eval/policy_dml" / bootstrap_mode
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "policy_dml_decision_scores.csv", index=False)
    episodes.to_csv(output / "policy_dml_episode_scores.csv", index=False)
    summary.to_csv(output / "policy_dml_policy_outcomes_summary.csv", index=False)
    dml.support_diagnostics(rows).to_csv(output / "policy_dml_support_diagnostics.csv", index=False)
    (output / "policy_dml_model_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = {"estimator": "CoR Appendix B.2 recursive Policy-DML", "settings": asdict(settings),
                "panel": str(panel_path), "panel_name": panel_name, "behaviour_learner": model_type,
                "saved_behaviour_provenance": provenance, "bootstrap_mode": bootstrap_mode,
                "n_bootstrap": n_bootstrap, "bootstrap_seed": BOOTSTRAP_SEED,
                "evaluation_period": "complete panel episode" if settings.horizon_hours is None else f"{settings.horizon_hours} hours from insertion",
                "current_practice": "cross-fitted stochastic behaviour policy pi_current(a|S)=g_hat(a|S)",
                "censoring": "structural endpoints only; nonterminal censoring rejected",
                "exposure": pec.EXPOSURE_MEASURE, "scores_clipped": False,
                "fixed_day_off_path_extension": "remove on first available IN decision on/after the target day",
                "continuation_targets": "nested patient OOF recursive scores; auxiliary nuisance fits exclude outer and inner held-out patients"}
    (output / "policy_dml_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_bootstrap(episodes, folded.drop(columns="p_remove_obs", errors="ignore"), policies, output / "confidence_intervals",
                  mode=bootstrap_mode, settings=settings, learner=model_type, n_bootstrap=n_bootstrap,
                  seed=BOOTSTRAP_SEED, regressor_factory=regressor_factory)
    sensitivity = []
    for clip in CLIPPING_SENSITIVITY:
        alternative, _, _ = dml.crossfit_policy_scores(decisions, policies, settings=replace(settings, propensity_clip=clip),
                                                     behaviour_learner=model_type, saved_behaviour=saved, regressor_factory=regressor_factory)
        result = policy_summary(dml.episode_scores(alternative), decisions)
        result["clipping_epsilon"] = clip
        sensitivity.append(result)
    if sensitivity:
        pd.concat(sensitivity, ignore_index=True).to_csv(output / "policy_dml_clipping_sensitivity.csv", index=False)
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}: {len(episodes) // len(policies)} episodes; saved {output}", flush=True)
    return summary, episodes, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=list(run_config.PANEL_PATHS))
    parser.add_argument("--bootstrap-mode", choices=["fixed", "refit"])
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--horizon-hours", type=float)
    parser.add_argument("--propensity-clip", type=float)
    parser.add_argument("--continuation-learner", choices=["ridge", "extra_trees", "random_forest"], default=SETTINGS.continuation_learner)
    args = parser.parse_args()
    if args.max_patients is not None and (args.max_patients < 1 or args.output_root is None):
        parser.error("--max-patients requires a positive count and a separate --output-root")
    settings = replace(SETTINGS, horizon_hours=args.horizon_hours, propensity_clip=args.propensity_clip, continuation_learner=args.continuation_learner)
    runs = [(args.panel, *run_config.PANEL_PATHS[args.panel])] if args.panel else PANEL_RUNS
    modes = [args.bootstrap_mode] if args.bootstrap_mode else BOOTSTRAP_MODES
    for name, panel, nuisance_root in runs:
        for mode in modes:
            root = args.output_root / name if args.output_root else None
            run_panel_estimation(panel, nuisance_root.parent, NUISANCE_MODEL_TYPE, mode, name,
                                 output_root=root, settings=settings, n_bootstrap=args.n_bootstrap, max_patients=args.max_patients)


if __name__ == "__main__":
    main()
