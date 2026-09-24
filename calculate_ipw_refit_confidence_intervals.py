#!/usr/bin/env python3
"""Patient bootstrap for IPW, refitting cross-fitted propensities in every draw.

Run: python calculate_ipw_refit_confidence_intervals.py
Uses the existing nuisance learner settings; each replicate fits five models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_ipw_policies as ipw
import fit_nuisance_models as nuisance
import policy_eval_common as pec
from calculate_policy_confidence_intervals import (
    DEFAULT_SEED, IPW_COLUMNS, bootstrap_replicate_table, interval_table, paired_differences,
)


REPO_ROOT = Path(__file__).resolve().parent
N_BOOTSTRAP = 1000
SEED = DEFAULT_SEED
OUTDIR = REPO_ROOT / "artefacts" / "policy_eval" / "confidence_intervals" / "ipw_refit"


def refit_propensities(panel, subjects, counts, feature_columns):
    """Fit the original learner on repeated patient rows and predict held-out rows.

    Keep original subject IDs when splitting: every copy of an original patient
    must stay in one fold. Otherwise identical episodes could leak into training
    and validation. Repeating rows also refits preprocessing and constant-feature
    selection with the correct bootstrap frequencies.
    """
    multiplicity = counts[subjects.get_indexer(panel.subject_id)]
    sample = panel.loc[panel.index.repeat(multiplicity)].reset_index(drop=True)
    sample = nuisance.add_grouped_crossfit_folds(sample)
    scores = pd.DataFrame(index=panel.index, columns=[
        "p_remove_obs", "p_keep_obs", nuisance.CROSSFIT_FOLD_COL,
    ], dtype=float)
    eligible = sample[nuisance.STATE_COL].eq("in")
    fallback_folds = 0
    for fold in range(nuisance.N_CROSSFIT_FOLDS):
        training = eligible & sample[nuisance.CROSSFIT_FOLD_COL].ne(fold)
        held_out = sample.loc[
            sample[nuisance.CROSSFIT_FOLD_COL].eq(fold)
        ].drop_duplicates("_source_row")
        model = nuisance.fit_crossfit_fold_model(
            sample.loc[training, feature_columns],
            sample.loc[training, nuisance.ACTION_COL], "propensity_removal", fold,
        )
        fallback_folds += int(model["fallback"])
        scores.loc[held_out._source_row, nuisance.CROSSFIT_FOLD_COL] = fold
        held_out = held_out.loc[held_out[nuisance.STATE_COL].eq("in")]
        if len(held_out):
            probability = nuisance.predict_crossfit_fold(model, held_out[feature_columns])
            scores.loc[held_out._source_row, "p_remove_obs"] = probability
            scores.loc[held_out._source_row, "p_keep_obs"] = 1.0 - probability
    return scores, fallback_folds


def prepare_policy_evaluation(panel, policy_rows, scores):
    """Calculate observed outcomes/adherence once with the original IPW helpers."""
    scored_panel = panel.assign(**{column: scores[column] for column in scores})
    # Cache the natural-key join so each new propensity vector can be attached
    # to the same policy decision rows without repeating a large merge.
    policy_rows = policy_rows.merge(
        panel[[*ipw.ROW_JOIN_KEY_COLS, "_source_row"]],
        on=ipw.ROW_JOIN_KEY_COLS, how="left", validate="many_to_one",
    )
    joined = ipw.join_nuisance_predictions(policy_rows, scored_panel)
    joined = pec.add_period_duration_days(joined, context="refit-bootstrap IPW rows")
    joined = pec.add_observed_icu_exit_alive_period(joined)
    joined = ipw.add_ipw_row_quantities(joined, ipw.CLIP_LOWER, ipw.CLIP_UPPER)
    joined = ipw.add_adherence(joined)
    all_episodes, adherent_episodes = ipw.build_policy_episode_panel(joined)
    current = ipw.build_current_practice_episode_panel(scored_panel, policy_rows)
    episodes = pd.concat([current, adherent_episodes], ignore_index=True)
    policies = pd.concat([
        current[["policy_name", "policy_remove_day"]],
        all_episodes[["policy_name", "policy_remove_day"]],
    ], ignore_index=True).drop_duplicates()
    return policy_rows, episodes, policies


def evaluate_ipw_sample(policy_rows, episodes, subjects, counts, scores, policies):
    """Rebuild sequential IPW weights from refitted held-out predictions."""
    rows = policy_rows.loc[counts[subjects.get_indexer(policy_rows.subject_id)] > 0].copy()
    for column in ("p_remove_obs", "p_keep_obs"):
        rows[column] = scores[column].reindex(rows._source_row).to_numpy()
    rows = ipw.add_ipw_row_quantities(rows, ipw.CLIP_LOWER, ipw.CLIP_UPPER)
    weights = ipw.product_components_by_episode(rows, "ipw_component").rename(
        columns={"ipw_component": "refitted_weight"},
    )
    sample = episodes.loc[counts[subjects.get_indexer(episodes.subject_id)] > 0].merge(
        weights, on=["policy_name", "policy_remove_day", ipw.EPISODE_ID_COL], how="left",
    )
    # As in the original estimator, an adherent episode with no applicable
    # decision rows has weight one. Current practice also has weight one.
    weight_one = (sample.policy_name.eq(ipw.CURRENT_PRACTICE_LABEL)
                  | sample.n_applicable_policy_rows.eq(0))
    sample.loc[weight_one, "refitted_weight"] = 1.0
    # Multiply AFTER the within-episode product. Duplicating decision rows
    # before that product would incorrectly raise an episode's weight to the
    # patient multiplicity. This is equivalent to duplicating whole episodes.
    sample["refitted_weight"] *= counts[subjects.get_indexer(sample.subject_id)]
    values = []
    for policy in policies.policy_name:
        group = sample.loc[sample.policy_name.eq(policy)]
        for column in IPW_COLUMNS.values():
            values.append(ipw.weighted_mean(group[column], group.refitted_weight))
    return np.array(values)


def main():
    nuisance.configure_model_run(ipw.NUISANCE_MODEL_TYPE)
    panel = nuisance.load_panel().reset_index(drop=True)
    panel["_source_row"] = panel.index
    subjects = pd.Index(sorted(panel.subject_id.unique()), name="subject_id")
    feature_columns = [
        nuisance.TIME_COL, nuisance.PERIODS_COL,
        *[column for column in panel if column.startswith(("itemid_", "sex_", "ethnicity_"))],
        "age",
    ]
    policy_rows = pd.read_csv(ipw.POLICY_PANEL_PATH, dtype={"subject_id": str}, low_memory=False)

    print(f"Refitting full-sample {ipw.NUISANCE_MODEL_TYPE} propensities "
          f"({nuisance.N_CROSSFIT_FOLDS} patient-grouped folds).", flush=True)
    scores, _ = refit_propensities(
        panel, subjects, np.ones(len(subjects), dtype=int), feature_columns,
    )
    policy_rows, episodes, policies = prepare_policy_evaluation(panel, policy_rows, scores)
    policies = policies.sort_values(
        "policy_remove_day", na_position="first",
    ).reset_index(drop=True)
    entries = pd.DataFrame([
        {"estimator": "ipw", "policy_name": policy.policy_name,
         "policy_remove_day": policy.policy_remove_day, "outcome": outcome}
        for policy in policies.itertuples(index=False) for outcome in IPW_COLUMNS
    ])
    point = evaluate_ipw_sample(
        policy_rows, episodes, subjects, np.ones(len(subjects), dtype=int), scores, policies,
    )
    baseline = episodes.loc[episodes.policy_name.eq(ipw.CURRENT_PRACTICE_LABEL)]
    episode_counts = baseline.groupby("subject_id").size().reindex(subjects).to_numpy()
    values = np.empty((N_BOOTSTRAP, len(entries)))
    rng = np.random.default_rng(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    replicate_path = OUTDIR / "bootstrap_replicate_estimates.csv"
    for replicate in range(N_BOOTSTRAP):
        counts = np.bincount(
            rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects),
        )
        print(f"Bootstrap {replicate + 1}/{N_BOOTSTRAP}: refitting "
              f"{nuisance.N_CROSSFIT_FOLDS} folds on {np.count_nonzero(counts):,} distinct patients.",
              flush=True)
        scores, fallback_folds = refit_propensities(panel, subjects, counts, feature_columns)
        values[replicate] = evaluate_ipw_sample(
            policy_rows, episodes, subjects, counts, scores, policies,
        )
        diagnostic = {
            "bootstrap_replicate": replicate + 1,
            "n_patient_draws": int(counts.sum()),
            "n_unique_patients_resampled": int(np.count_nonzero(counts)),
            "n_episodes_resampled": int(counts @ episode_counts),
            "n_fallback_folds": fallback_folds,
        }
        replicate_row, _ = bootstrap_replicate_table(
            entries, values[replicate:replicate + 1], pd.DataFrame([diagnostic]),
        )
        # Save completed replicates as we go; long refitting runs need not lose
        # all estimates if interrupted. Each completed pass appends one CSV row.
        replicate_row.to_csv(replicate_path, index=False, mode="w" if replicate == 0 else "a",
                             header=replicate == 0)

    value_table = interval_table(entries, point, values, len(subjects))
    difference_table = interval_table(
        entries, paired_differences(point, entries), paired_differences(values, entries), len(subjects),
    )
    difference_table["comparator"] = ipw.CURRENT_PRACTICE_LABEL
    value_table.to_csv(OUTDIR / "policy_value_confidence_intervals.csv", index=False)
    difference_table.to_csv(OUTDIR / "policy_difference_confidence_intervals.csv", index=False)
    print(f"Saved bootstrap estimates and confidence intervals: {OUTDIR}")


if __name__ == "__main__":
    main()
