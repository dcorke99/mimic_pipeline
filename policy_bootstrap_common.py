"""Shared patient-bootstrap calculations used by the three policy evaluators."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import fit_nuisance_models as nuisance
import policy_eval_common as pec

DEFAULT_SEED = 20260923
CURRENT_PRACTICE = "current_practice"
OUTCOMES = ("cauti", "recatheterisation", "death", "icu_exit_alive", "catheter_exposure_days")
GFORMULA_COLUMNS = dict(zip(OUTCOMES, (
    "predicted_any_cauti", "predicted_any_recatheterisation", "predicted_any_death",
    "predicted_icu_exit_alive", "expected_catheter_exposure_days",
)))
IPW_COLUMNS = dict(zip(OUTCOMES, (
    "any_cauti", "any_recatheterisation", "any_death", "observed_icu_exit_alive",
    "observed_catheter_exposure_days",
)))
AIPW_COLUMNS = {outcome: "plugin_" + column for outcome, column in GFORMULA_COLUMNS.items()}


def refit_nuisance_predictions(panel, subjects, counts, estimator, n_splits=5, model_type=None):
    """Refit on repeated patient rows, then score each sampled source row once.

    Original patient IDs keep all copies in the same cross-fit fold. Bootstrap
    multiplicity is applied to episode contributions after sequential weighting.
    """
    panel = panel.copy().reset_index(drop=True)
    panel["_source_row"] = panel.index
    if estimator != "ipw":
        panel = nuisance.prepare_outcome_targets(panel)
    multiplicity = counts[subjects.get_indexer(panel.subject_id)]
    sample = panel.loc[panel.index.repeat(multiplicity)].reset_index(drop=True)
    sample = nuisance.add_grouped_crossfit_folds(sample, n_splits=n_splits)
    scored = sample.drop_duplicates("_source_row").set_index("_source_row").sort_index()
    features = [
        nuisance.TIME_COL, nuisance.PERIODS_COL,
        *[column for column in panel if column.startswith(("itemid_", "sex_", "ethnicity_"))],
        "age",
    ]
    # Each task shares the full-model training risk set and fold preprocessing.
    tasks = []
    initial_columns = {}
    if estimator != "gformula":
        tasks.append(("in", "removal", nuisance.ACTION_COL, features))
        initial_columns.update({"p_remove_obs": np.nan, "p_keep_obs": np.nan})
    if estimator != "ipw":
        tasks.extend(("in", outcome, target, [*features, nuisance.ACTION_COL])
                     for outcome, target in nuisance.IN_OUTCOMES.items())
        tasks.extend(("out", outcome, target, features)
                     for outcome, target in nuisance.OUT_OUTCOMES.items())
        initial_columns.update({column: np.nan for column in pec.PREDICTION_COLUMNS})
        initial_columns.update({f"__rescored_{column}": False
                                for column in pec.PREDICTION_COLUMNS})
    scored = pd.concat([
        scored.drop(columns=initial_columns, errors="ignore"),
        pd.DataFrame(initial_columns, index=scored.index),
    ], axis=1)

    fallback_folds = 0
    for state, outcome, target, feature_columns in tasks:
        risk = (sample.catheter_state.eq("in") if outcome == "removal"
                else nuisance.outcome_risk_mask(sample, state, outcome))
        for fold in range(n_splits):
            training = risk & sample[nuisance.CROSSFIT_FOLD_COL].ne(fold)
            held_out = scored.loc[scored[nuisance.CROSSFIT_FOLD_COL].eq(fold)]
            model = nuisance.fit_crossfit_fold_model(
                sample.loc[training, feature_columns], sample.loc[training, target],
                f"{state}_{outcome}", fold, model_type=model_type,
                groups=sample.loc[training, nuisance.ID_COL],
            )
            fallback_folds += int(model["fallback"])
            if outcome == "removal":
                held_out = held_out.loc[held_out.catheter_state.eq("in")]
                if len(held_out):
                    probability = nuisance.predict_crossfit_fold(model, held_out[features])
                    scored.loc[held_out.index, "p_remove_obs"] = probability
                    scored.loc[held_out.index, "p_keep_obs"] = 1.0 - probability
                continue

            actions = (0, 1) if state == "in" else (None,)
            for action in actions:
                inputs = held_out[feature_columns].copy()
                suffix = "out" if action is None else "remove" if action else "keep"
                if action is not None:
                    inputs[nuisance.ACTION_COL] = action
                prediction = nuisance.predict_crossfit_fold(model, inputs)
                # Match full-model zeros outside the factual event risk set;
                # counterfactual states still receive model predictions.
                outside_risk = held_out.catheter_state.eq(state) & ~nuisance.outcome_risk_mask(
                    held_out, state, outcome,
                )
                prediction[outside_risk.to_numpy()] = 0.0
                column = f"p_{outcome}_if_{suffix}"
                scored.loc[held_out.index, column] = prediction
                scored.loc[held_out.index, f"__rescored_{column}"] = held_out.catheter_state.ne(state)
    return scored.reset_index(drop=True), fallback_folds


@dataclass
class PatientStatistics:
    entries: pd.DataFrame
    statistics: np.ndarray
    episode_counts: np.ndarray

    def estimate(self, counts):
        """Counts multiply EVERY episode of a selected patient, including repeats."""
        totals = np.einsum("i,ijk->jk", counts, self.statistics)
        values = np.full(len(self.entries), np.nan)
        np.divide(totals[:, 0], totals[:, 1], out=values, where=totals[:, 1] > 0)
        is_aipw = self.entries.estimator.eq("aipw").to_numpy()
        correction = np.full(len(values), np.nan)
        np.divide(totals[:, 2], totals[:, 3], out=correction, where=totals[:, 3] > 0)
        values[is_aipw] += correction[is_aipw]
        risk = is_aipw & self.entries.outcome.ne("catheter_exposure_days").to_numpy()
        # Clip the final AIPW aggregate only.
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
        for policy in policies.itertuples(index=False):
            group = df.loc[df.policy_name.eq(policy.policy_name)]
            if name != "ipw":
                group = group.loc[group.prediction_complete.astype(bool)]
            codes = subjects.get_indexer(group.subject_id)
            if name == "ipw":
                columns = list(IPW_COLUMNS.values()) + ["episode_ipw_weight"]
            elif name == "gformula":
                columns = list(GFORMULA_COLUMNS.values())
            else:
                columns = (list(AIPW_COLUMNS.values())
                           + [f"residual_{outcome}" for outcome in OUTCOMES]
                           + ["residual_correction_weight"])
            numeric = group[columns].apply(pd.to_numeric, errors="coerce")
            for outcome in OUTCOMES:
                terms = np.zeros((len(group), 4))
                if name == "ipw":
                    value = numeric[IPW_COLUMNS[outcome]].to_numpy(dtype=float)
                    weight = numeric["episode_ipw_weight"].to_numpy(dtype=float)
                    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
                    terms[valid, 0] = value[valid] * weight[valid]
                    terms[valid, 1] = weight[valid]
                else:
                    column = GFORMULA_COLUMNS[outcome] if name == "gformula" else AIPW_COLUMNS[outcome]
                    value = numeric[column].to_numpy(dtype=float)
                    valid = ~np.isnan(value)
                    terms[valid, 0], terms[valid, 1] = value[valid], 1.0
                    if name == "aipw":
                        weight = numeric["residual_correction_weight"].fillna(0.0).to_numpy(dtype=float)
                        residual = numeric[f"residual_{outcome}"].to_numpy(dtype=float)
                        with np.errstate(invalid="ignore", over="ignore"):
                            weighted_residual = weight * residual
                        # This mask and denominator deliberately match
                        # policy_summary_row; zero weights are retained.
                        valid = ~np.isnan(weighted_residual) & np.isfinite(weight)
                        terms[valid, 2] = weighted_residual[valid]
                        terms[valid, 3] = weight[valid]
                statistics.append(np.column_stack([
                    np.bincount(codes, weights=terms[:, j], minlength=len(subjects))
                    for j in range(4)
                ]))
                entries.append({
                    "estimator": name, "policy_name": policy.policy_name,
                    "policy_remove_day": policy.policy_remove_day, "outcome": outcome,
                })
    first = next(iter(episodes.values()))
    baseline = first.loc[first.policy_name.eq(CURRENT_PRACTICE)]
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


def bootstrap_replicate_table(entries, values, diagnostics):
    """One row per draw, with adjacent policy values and paired differences."""
    columns = [
        f"{row.policy_name}__{row.outcome}__{estimate_type}"
        for row in entries.itertuples(index=False)
        for estimate_type in ("policy_value", "difference_vs_current_practice")
    ]
    estimates = np.stack([values, paired_differences(values, entries)], axis=2)
    return pd.concat([
        diagnostics.reset_index(drop=True),
        pd.DataFrame(estimates.reshape(len(values), len(columns)), columns=columns),
    ], axis=1)


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


def run_bootstrap(estimator, episodes, policy_rows, evaluate, outdir, n_bootstrap,
                  refit_panel=None, seed=DEFAULT_SEED, refit_n_splits=5, model_type=None, panel_name=None):
    """Bootstrap one estimator, optionally rebuilding nuisance predictions."""
    baseline = episodes.loc[episodes.policy_name.eq(CURRENT_PRACTICE)]
    subjects = pd.Index(sorted(baseline.subject_id.unique()), name="subject_id")
    policies = pd.concat([
        baseline[["policy_name", "policy_remove_day"]],
        policy_rows[["policy_name", "policy_remove_day"]],
    ]).drop_duplicates().sort_values("policy_remove_day", na_position="first").reset_index(drop=True)
    design = build_patient_statistics({estimator: episodes}, policies, subjects)
    point = design.estimate(np.ones(len(subjects), dtype=int))
    values = np.empty((n_bootstrap, len(design.entries)))
    rng = np.random.default_rng(seed)
    mode = "refit" if refit_panel is not None else "fixed"
    output = outdir / "confidence_intervals"
    output.mkdir(parents=True, exist_ok=True)
    for replicate in range(n_bootstrap):
        counts = np.bincount(
            rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects),
        )
        draw_design = design
        diagnostic = {
            "bootstrap_replicate": replicate + 1,
            "n_patient_draws": int(counts.sum()),
            "n_unique_patients_resampled": int(np.count_nonzero(counts)),
            "n_episodes_resampled": int(counts @ design.episode_counts),
        }
        if refit_panel is not None:
            scored, fallback_folds = refit_nuisance_predictions(
                refit_panel, subjects, counts, estimator, n_splits=refit_n_splits, model_type=model_type,
            )
            sampled_policies = policy_rows.loc[policy_rows.subject_id.isin(scored.subject_id)]
            sample_episodes = evaluate(sampled_policies, scored)[0]
            draw_design = build_patient_statistics({estimator: sample_episodes}, policies, subjects)
            diagnostic["n_fallback_folds"] = fallback_folds
        values[replicate] = draw_design.estimate(counts)
        bootstrap_replicate_table(
            design.entries, values[replicate:replicate + 1], pd.DataFrame([diagnostic]),
        ).to_csv(output / f"{estimator}_bootstrap_replicate_estimates.csv", index=False, float_format="%.4f",
                 mode="w" if replicate == 0 else "a", header=replicate == 0)
        completed = replicate + 1
        if completed % 25 == 0 or completed == n_bootstrap:
            print(
                f"[BOOTSTRAP] model={model_type or nuisance.MODEL_TYPE}; "
                f"panel={panel_name or outdir.parent.parent.name}; estimator={estimator}; "
                f"bootstrap={mode}; completed={completed}/{n_bootstrap}",
                flush=True,
            )

    interval_table(design.entries, point, values, len(subjects)).to_csv(
        output / f"{estimator}_policy_value_confidence_intervals.csv", index=False, float_format="%.4f",
    )
    differences = interval_table(
        design.entries, paired_differences(point, design.entries),
        paired_differences(values, design.entries), len(subjects),
    )
    differences["comparator"] = CURRENT_PRACTICE
    differences.to_csv(output / f"{estimator}_policy_difference_confidence_intervals.csv", index=False, float_format="%.4f")
