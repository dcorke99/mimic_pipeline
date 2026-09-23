#!/usr/bin/env python3
"""Patient-clustered percentile intervals conditional on saved nuisance predictions.

Example: python calculate_policy_confidence_intervals.py --panel real --n-bootstrap 50 --seed 20260923
No nuisance models are refitted and no point-estimator artefacts are modified.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_aipw_policies as aipw
import evaluate_gformula_policies as gformula
import evaluate_ipw_policies as ipw
import policy_eval_common as pec
from panel_run_config import add_panel_argument, resolve_panel_run


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20260923
CURRENT_PRACTICE = aipw.CURRENT_PRACTICE_LABEL
EPISODE_ID = aipw.EPISODE_ID_COL
ESTIMATORS = {"gformula": gformula, "ipw": ipw, "aipw": aipw}
OUTCOMES = tuple(aipw.OUTCOME_SPECS)
GFORMULA_COLUMNS = dict(zip(OUTCOMES, [
    "predicted_any_cauti", "predicted_any_recatheterisation",
    "predicted_any_death", "predicted_icu_exit_alive",
    "expected_catheter_exposure_days",
]))
IPW_COLUMNS = {
    **{name: spec[0] for name, spec in ipw.OUTCOME_SPECS.items()},
    "catheter_exposure_days": "observed_catheter_exposure_days",
}
SUMMARY_COLUMNS = {
    "gformula": dict(zip(OUTCOMES, [
        "predicted_cauti_risk", "predicted_recatheterisation_risk",
        "predicted_death_risk", "predicted_icu_exit_alive_risk",
        "expected_mean_catheter_exposure_days",
    ])),
    "ipw": {
        name: f"ipw_weighted_{name}_risk" for name in ipw.OUTCOME_SPECS
    } | {"catheter_exposure_days": "ipw_weighted_mean_catheter_exposure_days"},
    "aipw": {
        name: f"aipw_{spec['summary_stub']}"
        for name, spec in aipw.OUTCOME_SPECS.items()
    } | {"catheter_exposure_days": "aipw_mean_catheter_exposure_days"},
}
# save_report_df writes three decimal places; allow half a reporting unit.
SUMMARY_ATOL = 0.0005 + 1e-12
REFERENCE_ATOL = 1e-12


def require_columns(df, columns, context):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def load_inputs(artefact_root):
    """Read the combined episode files, which already include current practice."""
    episodes, summaries, input_paths = {}, {}, {}
    for name, module in ESTIMATORS.items():
        directory = artefact_root / "policy_eval" / name
        paths = {
            kind: directory / module.OUTPUT_PATHS[kind].name
            for kind in ("episodes", "summary")
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Run evaluate_{name}_policies.py for this panel first."
                )
        episodes[name] = pd.read_csv(paths["episodes"], float_precision="round_trip")
        summaries[name] = pd.read_csv(paths["summary"], float_precision="round_trip")
        input_paths[name] = {
            kind: {"path": str(path), "size_bytes": path.stat().st_size,
                   "modified_time_ns": path.stat().st_mtime_ns}
            for kind, path in paths.items()
        }
    return episodes, summaries, input_paths


def validate_inputs(episodes, summaries):
    """Reject mismatched policy/cohort outputs before pairing bootstrap draws."""
    policy_cols = ["policy_name", "policy_remove_day"]
    for name, summary in summaries.items():
        require_columns(summary, policy_cols + list(SUMMARY_COLUMNS[name].values()), name)
        if summary.policy_name.isna().any() or summary.policy_name.duplicated().any():
            raise ValueError(f"{name}: policy summaries must have one row per named policy")
    policies = summaries["gformula"][policy_cols].copy()
    current = policies.policy_name.eq(CURRENT_PRACTICE)
    if current.sum() != 1 or not policies.loc[current, "policy_remove_day"].isna().all():
        raise ValueError("Expected one current_practice policy with no removal day")
    fixed_days = pd.to_numeric(policies.loc[~current, "policy_remove_day"], errors="coerce")
    if (fixed_days.empty or fixed_days.isna().any() or not np.isfinite(fixed_days).all()
            or (fixed_days <= 0).any() or (fixed_days % 1 != 0).any()):
        raise ValueError("Expected fixed-day policies with positive integer removal days")
    policies = policies.sort_values("policy_remove_day", na_position="first").reset_index(drop=True)
    expected_policy_days = policies.set_index("policy_name").policy_remove_day
    for name, summary in summaries.items():
        actual = summary.set_index("policy_name").policy_remove_day
        if set(actual.index) != set(expected_policy_days.index) or not np.allclose(
            actual.reindex(expected_policy_days.index), expected_policy_days,
            rtol=0, atol=0, equal_nan=True,
        ):
            raise ValueError(f"{name}: policies/removal days disagree across summaries")
    normalisation = summaries["aipw"].get("residual_normalisation")
    if (aipw.RESIDUAL_NORMALISATION != "hajek" or normalisation is None
            or not normalisation.eq("hajek").all()):
        raise ValueError("This bootstrap requires the existing selected Hajek AIPW estimates")

    for name, df in episodes.items():
        require_columns(df, ["subject_id", EPISODE_ID, "policy_type", *policy_cols], name)
        if df[["subject_id", EPISODE_ID, "policy_name"]].isna().any().any():
            raise ValueError(f"{name}: patient, episode and policy identifiers must be present")
        if df.duplicated(["policy_name", EPISODE_ID]).any():
            raise ValueError(f"{name}: duplicate policy/episode rows")
        if not set(df.policy_name).issubset(set(expected_policy_days.index)):
            raise ValueError(f"{name}: episode file contains policies absent from summary")
        days = df.policy_name.map(expected_policy_days)
        if not np.allclose(df.policy_remove_day, days, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"{name}: episode removal days disagree with summary")
        if "prediction_complete" in df:
            # Avoid astype(bool) treating a literal string 'False' as True.
            if df.prediction_complete.isna().any() or not df.prediction_complete.isin([True, False]).all():
                raise ValueError(f"{name}: prediction_complete must contain boolean values")
            df["prediction_complete"] = df.prediction_complete.astype(bool)

    baseline = episodes["gformula"].loc[
        episodes["gformula"].policy_name.eq(CURRENT_PRACTICE), [EPISODE_ID, "subject_id"]
    ].set_index(EPISODE_ID).subject_id.sort_index()
    if baseline.empty:
        raise ValueError("The current-practice patient cohort is empty")
    for name, df in episodes.items():
        for policy in policies.policy_name:
            group = df.loc[df.policy_name.eq(policy)]
            mapping = group.set_index(EPISODE_ID).subject_id.sort_index()
            # IPW saves only adherent fixed-policy episodes. Their missing rows
            # contribute zero; the draw population is still the entire cohort.
            if name == "ipw" and policy != CURRENT_PRACTICE:
                if not mapping.index.isin(baseline.index).all() or not mapping.equals(baseline.reindex(mapping.index)):
                    raise ValueError(f"{name}/{policy}: episodes do not belong to the common cohort")
            elif not mapping.equals(baseline):
                raise ValueError(f"{name}/{policy}: full episode/patient cohort differs")
    return policies, pd.Index(sorted(baseline.unique()), name="subject_id")


def reference_values(episodes, policies):
    """Recalculate with the original evaluator helpers as an independent check."""
    summaries = {
        "gformula": gformula.build_policy_summary(episodes["gformula"], gformula.PREDICTION_MODE),
        "aipw": aipw.build_policy_summary(episodes["aipw"], "hajek"),
    }
    rows = []
    for policy in policies.itertuples(index=False):
        group = episodes["ipw"].loc[episodes["ipw"].policy_name.eq(policy.policy_name)]
        rows.append(ipw.summarise_episode_estimates(
            group, policy.policy_name, policy.policy_remove_day, len(group), len(group),
        ))
    summaries["ipw"] = pd.DataFrame(rows)
    return np.array([
        summaries[name].set_index("policy_name").loc[policy, SUMMARY_COLUMNS[name][outcome]]
        for name in ESTIMATORS for policy in policies.policy_name for outcome in OUTCOMES
    ], dtype=float)


def numeric(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


@dataclass
class PatientStatistics:
    entries: pd.DataFrame
    statistics: np.ndarray
    episode_counts: np.ndarray

    def estimate(self, counts):
        """Counts multiply EVERY episode of a selected patient, including repeats."""
        counts = np.asarray(counts)
        if (counts.shape != (len(self.statistics),) or not np.isfinite(counts).all()
                or (counts < 0).any() or (counts % 1 != 0).any()):
            raise ValueError("Patient multiplicities must be nonnegative integer counts")
        totals = np.einsum("i,ijk->jk", counts, self.statistics)
        values = np.full(len(self.entries), np.nan)
        np.divide(totals[:, 0], totals[:, 1], out=values, where=totals[:, 1] > 0)
        is_aipw = self.entries.estimator.eq("aipw").to_numpy()
        correction = np.full(len(values), np.nan)
        np.divide(totals[:, 2], totals[:, 3], out=correction, where=totals[:, 3] > 0)
        values[is_aipw] += correction[is_aipw]
        risk = is_aipw & self.entries.outcome.ne("catheter_exposure_days").to_numpy()
        # Match bound_probability_estimate: clip the final AIPW aggregate only.
        values[risk] = np.clip(values[risk], 0.0, 1.0)
        return values


def build_patient_statistics(episodes, policies, subjects):
    """Pre-sum episode contributions by patient without changing episode weighting.

    Multiplying these sums by bootstrap patient counts is exactly equivalent to
    concatenating every episode for every sampled patient. A patient with three
    episodes selected twice contributes six episodes, not one patient mean.
    """
    entries, statistics = [], []
    for name, df in episodes.items():
        if name not in ESTIMATORS:
            raise ValueError(f"Unknown estimator: {name}")
    for name in ESTIMATORS:
        df = episodes[name]
        for policy in policies.itertuples(index=False):
            group = df.loc[df.policy_name.eq(policy.policy_name)]
            if name != "ipw":
                group = group.loc[group.prediction_complete.astype(bool)]
            codes = subjects.get_indexer(group.subject_id)
            if (codes < 0).any():
                raise ValueError(f"{name}: patient outside the bootstrap cohort")
            for outcome in OUTCOMES:
                terms = np.zeros((len(group), 4))
                if name == "ipw":
                    value, weight = numeric(group[IPW_COLUMNS[outcome]]), numeric(group[ipw.WEIGHT_COL])
                    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
                    terms[valid, 0] = value[valid] * weight[valid]
                    terms[valid, 1] = weight[valid]
                else:
                    column = GFORMULA_COLUMNS[outcome] if name == "gformula" else aipw.OUTCOME_SPECS[outcome]["plugin"]
                    value = numeric(group[column])
                    if np.isinf(value).any():
                        raise ValueError(f"{name}/{policy.policy_name}/{outcome}: infinite plug-in predictions")
                    valid = ~np.isnan(value)
                    terms[valid, 0], terms[valid, 1] = value[valid], 1.0
                    if name == "aipw":
                        weight = numeric(pd.to_numeric(
                            group[aipw.RESIDUAL_WEIGHT_COL], errors="coerce",
                        ).fillna(0.0))
                        residual = numeric(group[f"residual_{outcome}"])
                        with np.errstate(invalid="ignore", over="ignore"):
                            weighted_residual = weight * residual
                        # This mask and denominator deliberately match
                        # policy_summary_row; zero weights are retained.
                        valid = ~np.isnan(weighted_residual) & np.isfinite(weight)
                        terms[valid, 2] = weighted_residual[valid]
                        terms[valid, 3] = weight[valid]
                if not np.isfinite(terms).all():
                    raise ValueError(f"{name}/{policy.policy_name}/{outcome}: nonfinite estimator contributions")
                statistics.append(np.column_stack([
                    np.bincount(codes, weights=terms[:, j], minlength=len(subjects))
                    for j in range(4)
                ]))
                entries.append({
                    "estimator": name, "policy_name": policy.policy_name,
                    "policy_remove_day": policy.policy_remove_day, "outcome": outcome,
                })
    baseline = episodes["gformula"].loc[episodes["gformula"].policy_name.eq(CURRENT_PRACTICE)]
    episode_counts = np.bincount(subjects.get_indexer(baseline.subject_id), minlength=len(subjects))
    return PatientStatistics(pd.DataFrame(entries), np.stack(statistics, axis=1), episode_counts)


def paired_differences(values, entries):
    baseline = {
        (row.estimator, row.outcome): i
        for i, row in enumerate(entries.itertuples(index=False))
        if row.policy_name == CURRENT_PRACTICE
    }
    indices = [baseline[(row.estimator, row.outcome)] for row in entries.itertuples(index=False)]
    return np.asarray(values) - np.asarray(values)[..., indices]


def verify_point_estimates(point, design, episodes, policies, summaries):
    reference = reference_values(episodes, policies)
    if not np.allclose(point, reference, rtol=1e-12, atol=REFERENCE_ATOL, equal_nan=True):
        raise ValueError("Patient sufficient statistics disagree with original evaluator helpers")
    indexed = {name: df.set_index("policy_name") for name, df in summaries.items()}
    difference = paired_differences(point, design.entries)
    max_errors = {}
    for i, row in enumerate(design.entries.itertuples(index=False)):
        source = indexed[row.estimator].loc[row.policy_name]
        diff_col = f"{aipw.OUTCOME_SPECS[row.outcome]['summary_stub']}_difference_vs_current_practice"
        for kind, value, column in (
            ("value", point[i], SUMMARY_COLUMNS[row.estimator][row.outcome]),
            ("difference", difference[i], diff_col),
        ):
            if column not in source:
                raise ValueError(f"{row.estimator} summary is missing {column}")
            expected = float(source[column])
            if not np.isclose(value, expected, rtol=0, atol=SUMMARY_ATOL, equal_nan=True):
                raise ValueError(
                    f"{row.estimator}/{row.policy_name}/{row.outcome} {kind}: "
                    f"recomputed {value} differs from saved summary {expected}. "
                    "Check that episode outputs and summaries came from the same run."
                )
            key = f"{row.estimator}_{kind}"
            max_errors[key] = max(max_errors.get(key, 0.0), abs(value - expected) if np.isfinite(value) else 0.0)
    return {"passed": True, "original_helper_atol": REFERENCE_ATOL,
            "original_helper_rtol": 1e-12, "summary_csv_atol": SUMMARY_ATOL,
            "summary_csv_decimal_places": 3, "maximum_absolute_summary_errors": max_errors}


def draw_patient_counts(rng, n_patients):
    """Draw N patients with replacement; the count vector preserves clusters."""
    return np.bincount(rng.integers(0, n_patients, size=n_patients), minlength=n_patients)


def bootstrap_values(design, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n_patients = len(design.statistics)
    values = np.empty((n_bootstrap, len(design.entries)))
    diagnostics = []
    for replicate in range(n_bootstrap):
        # One shared draw for all estimators, policies and outcomes is essential
        # for paired differences from current practice.
        counts = draw_patient_counts(rng, n_patients)
        values[replicate] = design.estimate(counts)
        diagnostics.append({
            "bootstrap_replicate": replicate + 1,
            "n_patient_draws": int(counts.sum()),
            "n_unique_patients_resampled": int(np.count_nonzero(counts)),
            "n_episodes_resampled": int(counts @ design.episode_counts),
        })
    return values, pd.DataFrame(diagnostics)


def interval_table(entries, point, bootstrap, n_patients):
    table = entries.copy()
    table["point_estimate"] = point
    bounds, valid_counts = [], []
    for column in bootstrap.T:
        finite = column[np.isfinite(column)]
        valid_counts.append(len(finite))
        bounds.append(np.percentile(finite, [2.5, 97.5], method="linear") if len(finite) else [np.nan, np.nan])
    table[["ci_lower", "ci_upper"]] = np.asarray(bounds)
    table["n_bootstrap"] = len(bootstrap)
    table["n_valid_bootstrap"] = valid_counts
    table["n_undefined_bootstrap"] = len(bootstrap) - np.asarray(valid_counts)
    table["n_unique_patients"] = n_patients
    table["confidence_level"] = 0.95
    table["unit"] = np.where(table.outcome.eq("catheter_exposure_days"), "days", "probability")
    return table


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_panel_argument(parser)
    parser.add_argument("--n-bootstrap", type=positive_integer, default=1000,
                        help="Number of patient bootstrap replicates (default: 1000).")
    parser.add_argument("--seed", type=nonnegative_integer, default=DEFAULT_SEED,
                        help=f"Reproducible random seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--save-replicates", action=argparse.BooleanOptionalAction, default=True,
                        help="Save tidy full bootstrap values and paired differences (default: yes).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = resolve_panel_run(REPO_ROOT, args.panel)
    episodes, summaries, inputs = load_inputs(paths.artefact_root)
    policies, subjects = validate_inputs(episodes, summaries)
    design = build_patient_statistics(episodes, policies, subjects)
    point = design.estimate(np.ones(len(subjects), dtype=int))
    verification = verify_point_estimates(point, design, episodes, policies, summaries)
    print(f"Validated {len(point)} point estimates; bootstrapping {len(subjects):,} patients "
          f"({args.n_bootstrap:,} replicates, seed {args.seed}).", flush=True)
    values, draw_diagnostics = bootstrap_values(design, args.n_bootstrap, args.seed)
    differences = paired_differences(values, design.entries)
    value_table = interval_table(design.entries, point, values, len(subjects))
    difference_table = interval_table(
        design.entries, paired_differences(point, design.entries), differences, len(subjects),
    )
    difference_table["comparator"] = CURRENT_PRACTICE

    outdir = paths.artefact_root / "policy_eval" / "confidence_intervals"
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {kind: outdir / filename for kind, filename in {
        "policy_values": "policy_value_confidence_intervals.csv",
        "policy_differences": "policy_difference_confidence_intervals.csv",
        "metadata": "bootstrap_metadata.json",
    }.items()}
    # Keep full precision: three-decimal report rounding can erase narrow CIs.
    value_table.to_csv(outputs["policy_values"], index=False)
    difference_table.to_csv(outputs["policy_differences"], index=False)
    if args.save_replicates:
        outputs["replicates"] = outdir / "bootstrap_replicate_estimates.csv"
        tidy = pd.concat([design.entries] * args.n_bootstrap, ignore_index=True)
        for column in draw_diagnostics:
            tidy[column] = np.repeat(draw_diagnostics[column].to_numpy(), len(design.entries))
        tidy["policy_value"] = values.ravel()
        tidy["difference_vs_current_practice"] = differences.ravel()
        tidy["comparator"] = CURRENT_PRACTICE
        tidy.to_csv(outputs["replicates"], index=False)

    undefined = value_table.loc[value_table.n_undefined_bootstrap.gt(0), [
        "estimator", "policy_name", "outcome", "n_undefined_bootstrap",
    ]].to_dict("records")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": args.panel, "method": "patient-clustered non-parametric percentile bootstrap",
        "inference_scope": "Patient-clustered bootstrap intervals conditional on the existing cross-fitted nuisance predictions.",
        "nuisance_models_refitted": False, "cluster_column": "subject_id",
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "rng": "numpy.random.default_rng (PCG64)",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "confidence_level": 0.95, "percentiles": [2.5, 97.5], "percentile_method": "linear",
        "n_unique_patients": len(subjects), "n_episodes": int(design.episode_counts.sum()),
        "policies": [{"policy_name": row.policy_name,
                      "policy_remove_day": None if pd.isna(row.policy_remove_day) else int(row.policy_remove_day)}
                     for row in policies.itertuples(index=False)],
        "outcomes": list(OUTCOMES), "estimators": list(ESTIMATORS),
        "resampling": {
            "population": "Common current-practice cohort, sorted by subject_id",
            "patient_draws_per_replicate": len(subjects), "replacement": True,
            "all_episodes_retained": True,
            "implementation": "Patient multiplicities multiply sums of all episode contributions; equivalent to copying whole patient clusters.",
            "estimand": "Episode-weighted policy values, with patients as the resampling unit",
            "shared_draw_across_estimators_policies_and_outcomes": True,
            "ipw": "Fixed-policy episode files contain adherent episodes only; absent episodes contribute zero, and the full patient cohort is resampled.",
            "min_episodes_resampled": int(draw_diagnostics.n_episodes_resampled.min()),
            "max_episodes_resampled": int(draw_diagnostics.n_episodes_resampled.max()),
        },
        "estimator_definitions": {
            "gformula": "Mean saved prediction among prediction_complete episodes (including the modelled current-practice regime).",
            "ipw": "Sum(weight * observed outcome) / sum(weight), with the original finite, positive-weight mask; current-practice weights are one.",
            "aipw": "Selected Hajek: mean(plugin) + sum(residual_correction_weight * residual) / sum(residual_correction_weight), on complete episodes with the original outcome-specific residual mask. Recalculate both components and denominator in every replicate.",
            "aipw_probability_bounding": "Clip final aggregate risks to [0, 1] before differences; catheter exposure remains unbounded.",
            "differences": "Policy minus the same estimator's current_practice value within each replicate; probability differences and exposure-day differences.",
        },
        "undefined_replicate_handling": "Zero denominators produce NaN, never a plug-in fallback or a redraw. Percentiles use finite replicates only; valid/undefined counts are reported per interval. No finite replicates yields missing bounds.",
        "undefined_policy_value_replicates": undefined,
        "point_estimate_validation": verification,
        "input_files": inputs,
        "output_files": {key: str(path) for key, path in outputs.items()},
        "full_replicates_saved": args.save_replicates,
    }
    pec.save_json(metadata, outputs["metadata"])
    print(f"Complete: {len(value_table)} policy-value and {len(difference_table)} paired-difference "
          f"95% intervals; 3 estimators, {len(policies)} policies, {len(OUTCOMES)} outcomes.")
    print("Point estimates match original evaluator helpers and saved summaries (3-decimal tolerance).")
    if undefined:
        print(f"WARNING: {len(undefined)} value intervals have undefined replicates; see counts and metadata.")
    print(f"Saved confidence intervals{' and full replicates' if args.save_replicates else ''}: {outdir}")


if __name__ == "__main__":
    main()
