"""Patient-bootstrap nuisance refitting for the g-formula comparator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import fit_nuisance_models as nuisance
import policy_eval_common as pec
from policy_inference import interval_table, paired_differences

DEFAULT_SEED = 20260923
CURRENT_PRACTICE = "current_practice"
OUTCOMES = ("cauti", "recatheterisation", "death", "icu_exit_alive", "catheter_exposure_days")
VALUE_COLUMNS = dict(zip(OUTCOMES, (
    "predicted_any_cauti", "predicted_any_recatheterisation", "predicted_any_death",
    "predicted_icu_exit_alive", "expected_catheter_exposure_days",
)))


def refit_nuisance_predictions(panel, subjects, counts, n_splits=5, model_type=None):
    """Refit outcome models on a patient bootstrap sample and score source rows."""
    panel = nuisance.prepare_outcome_targets(panel.copy().reset_index(drop=True))
    panel["_source_row"] = panel.index
    multiplicity = counts[subjects.get_indexer(panel.subject_id)]
    sample = panel.loc[panel.index.repeat(multiplicity)].reset_index(drop=True)
    sample = nuisance.add_grouped_crossfit_folds(sample, n_splits=n_splits)
    scored = sample.drop_duplicates("_source_row").set_index("_source_row").sort_index()
    features = [
        nuisance.TIME_COL, nuisance.PERIODS_COL,
        *[column for column in panel if column.startswith(("itemid_", "sex_", "ethnicity_"))],
        "age",
    ]
    tasks = [
        *[("in", outcome, target, [*features, nuisance.ACTION_COL])
          for outcome, target in nuisance.IN_OUTCOMES.items()],
        *[("out", outcome, target,
           pec.reinsertion_feature_columns(features) if outcome == "reinsertion" else features)
          for outcome, target in nuisance.OUT_OUTCOMES.items()],
    ]
    initial = {column: np.nan for column in [*pec.PREDICTION_COLUMNS, *pec.reinsertion_policy_columns()]}
    initial.update({f"__rescored_{column}": False for column in pec.PREDICTION_COLUMNS})
    scored = pd.concat([
        scored.drop(columns=initial, errors="ignore"),
        pd.DataFrame(initial, index=scored.index),
    ], axis=1)

    fallback_folds = 0
    models = {"in_models": {}, "out_models": {}}
    for state, outcome, target, feature_columns in tasks:
        models[f"{state}_models"][outcome] = {"features": feature_columns, "fold_models": []}
        risk = nuisance.outcome_risk_mask(sample, state, outcome)
        for fold in range(n_splits):
            training = risk & sample[nuisance.CROSSFIT_FOLD_COL].ne(fold)
            held = scored.loc[scored[nuisance.CROSSFIT_FOLD_COL].eq(fold)]
            model = nuisance.fit_crossfit_fold_model(
                sample.loc[training, feature_columns], sample.loc[training, target],
                f"{state}_{outcome}", fold, model_type=model_type,
                groups=sample.loc[training, nuisance.ID_COL],
            )
            fallback_folds += int(model["fallback"])
            models[f"{state}_models"][outcome]["fold_models"].append(model)
            if outcome == "reinsertion":
                predictions = pec.predict_reinsertion_columns(
                    model, held, feature_columns, predictor=nuisance.predict_crossfit_fold,
                )
                for column, values in predictions.items():
                    scored.loc[held.index, column] = values
                continue
            for action in ((0, 1) if state == "in" else (None,)):
                inputs = held[feature_columns].copy()
                suffix = "out" if action is None else "remove" if action else "keep"
                if action is not None:
                    inputs[nuisance.ACTION_COL] = action
                values = nuisance.predict_crossfit_fold(model, inputs)
                outside_risk = held.catheter_state.eq(state) & ~nuisance.outcome_risk_mask(
                    held, state, outcome,
                )
                values[outside_risk.to_numpy()] = 0.0
                column = f"p_{outcome}_if_{suffix}"
                scored.loc[held.index, column] = values
                scored.loc[held.index, f"__rescored_{column}"] = held.catheter_state.ne(state)
    scored = pec.add_policy_predictions(scored, models, predictor=nuisance.predict_crossfit_fold)
    scored = pec.add_current_practice_cauti_predictions(
        scored, models, predictor=nuisance.predict_crossfit_fold,
    )
    return scored.reset_index(drop=True), fallback_folds


@dataclass
class PatientStatistics:
    entries: pd.DataFrame
    statistics: np.ndarray
    episode_counts: np.ndarray

    def estimate(self, counts):
        totals = np.einsum("i,ijk->jk", counts, self.statistics)
        values = np.full(len(self.entries), np.nan)
        np.divide(totals[:, 0], totals[:, 1], out=values, where=totals[:, 1] > 0)
        return values


def build_patient_statistics(episodes, policies, subjects):
    entries, statistics = [], []
    for policy in policies.itertuples(index=False):
        group = episodes.loc[
            episodes.policy_name.eq(policy.policy_name)
            & episodes.prediction_complete.astype(bool)
        ]
        codes = subjects.get_indexer(group.subject_id)
        for outcome, column in VALUE_COLUMNS.items():
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            statistics.append(np.column_stack([
                np.bincount(codes[valid], weights=values[valid], minlength=len(subjects)),
                np.bincount(codes[valid], minlength=len(subjects)),
            ]))
            entries.append({
                "estimator": "gformula", "policy_name": policy.policy_name,
                "policy_remove_day": policy.policy_remove_day, "outcome": outcome,
            })
    baseline = episodes.loc[episodes.policy_name.eq(CURRENT_PRACTICE)]
    episode_counts = np.bincount(subjects.get_indexer(baseline.subject_id), minlength=len(subjects))
    return PatientStatistics(pd.DataFrame(entries), np.stack(statistics, axis=1), episode_counts)


def run_bootstrap(episodes, policy_rows, evaluate, outdir, n_bootstrap,
                  refit_panel=None, seed=DEFAULT_SEED, refit_n_splits=5,
                  model_type=None, panel_name=None):
    baseline = episodes.loc[episodes.policy_name.eq(CURRENT_PRACTICE)]
    subjects = pd.Index(sorted(baseline.subject_id.unique()), name="subject_id")
    policies = pd.concat([
        baseline[["policy_name", "policy_remove_day"]],
        policy_rows[["policy_name", "policy_remove_day"]],
    ]).drop_duplicates().sort_values("policy_remove_day", na_position="first").reset_index(drop=True)
    design = build_patient_statistics(episodes, policies, subjects)
    point = design.estimate(np.ones(len(subjects), dtype=int))
    values = np.empty((n_bootstrap, len(design.entries)))
    rng = np.random.default_rng(seed)
    mode = "refit" if refit_panel is not None else "fixed"
    outdir.mkdir(parents=True, exist_ok=True)
    for replicate in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        draw_design = design
        if refit_panel is not None:
            scored, _ = refit_nuisance_predictions(
                refit_panel, subjects, counts, n_splits=refit_n_splits, model_type=model_type,
            )
            sampled_policies = policy_rows.loc[policy_rows.subject_id.isin(scored.subject_id)]
            sample_episodes = evaluate(sampled_policies, scored)[0]
            draw_design = build_patient_statistics(sample_episodes, policies, subjects)
        values[replicate] = draw_design.estimate(counts)
        completed = replicate + 1
        if completed % 5 == 0 or completed == n_bootstrap:
            print(
                f"[BOOTSTRAP] model={model_type or nuisance.MODEL_TYPE}; "
                f"panel={panel_name or outdir.parent.parent.name}; estimator=gformula; "
                f"bootstrap={mode}; completed={completed}/{n_bootstrap}", flush=True,
            )

    if n_bootstrap:
        interval_table(design.entries, point, values, len(subjects)).to_csv(
            outdir / "policy_values.csv", index=False, float_format="%.4f",
        )
        differences = interval_table(
            design.entries, paired_differences(point, design.entries),
            paired_differences(values, design.entries), len(subjects),
        )
        differences["comparator"] = CURRENT_PRACTICE
        differences.to_csv(
            outdir / "policy_differences.csv",
            index=False, float_format="%.4f",
        )
