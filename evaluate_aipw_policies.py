from pathlib import Path

import numpy as np
import pandas as pd

import policy_eval_common as pec
import policy_bootstrap_common as bootstrap
import fit_nuisance_models as nuisance
from build_policy_panels import read_policy_collection
from policy_eval_common import (
    effective_sample_size,
    assign_policy_prediction,
    cumulative_event_probability,
    CROSSFIT_FOLD_COL,
    PREDICTION_COLUMNS,
    add_episode_day,
    fill_missing_counterfactual_predictions,
)


# Configuration

REPO_ROOT = Path(__file__).resolve().parent
NUISANCE_MODEL_TYPE = "xgboost"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260923
REFIT_CROSSFIT_FOLDS = 5
BOOTSTRAP_MODES = ("fixed",)  # Use ("fixed", "refit") to run both modes.


# Panels to process in this script only; comment out entries to skip them.
VALIDATION_DIR = REPO_ROOT / "artefacts/semi-synthetic_validation/semi_synthetic_measured_confounding"
PANEL_RUNS = (
    ("real", REPO_ROOT / "data/modelling_panel.csv", REPO_ROOT / "artefacts/nuisance_models"),
    ("semi_synthetic_with_confounding", VALIDATION_DIR / "semi_synthetic_panel.csv",
     VALIDATION_DIR / "pipeline_runs/semi_synthetic_with_confounding/nuisance_models"),
    ("confounder_omitted", VALIDATION_DIR / "semi_synthetic_panel_confounder_omitted.csv",
     VALIDATION_DIR / "pipeline_runs/confounder_omitted/nuisance_models"),
    ("randomised_action", VALIDATION_DIR / "semi_synthetic_panel_randomised_action.csv",
     VALIDATION_DIR / "pipeline_runs/randomised_action/nuisance_models"),
)

OUTPUT_FILENAMES = {
    "summary": "aipw_policy_outcomes_summary.csv",
    "episodes": "aipw_policy_episode_scores.csv",
    "support_diagnostics": "aipw_policy_support_diagnostics.csv",
    "weight_diagnostics": "aipw_weight_diagnostics.csv",
    "residual_diagnostics": "aipw_residual_diagnostics.csv",
    "clipping_sensitivity": "aipw_clipping_sensitivity.csv",
    "current_practice": "current_practice_aipw_episode_scores.csv",
}

CLIP_LOWER = 0.01
CLIP_UPPER = 0.99
RESIDUAL_NORMALISATION = "hajek"

CURRENT_PRACTICE_LABEL = "current_practice"
ESTIMATOR_NAME = "aipw"
POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS = 2

EPISODE_ID_COL = "catheter_episode_id"
POLICY_TYPE_COL = "policy_type"
WEIGHT_COL = "episode_ipw_weight"
UNCLIPPED_WEIGHT_COL = "episode_ipw_weight_unclipped"
RESIDUAL_WEIGHT_COL = "residual_correction_weight"
UNCLIPPED_RESIDUAL_WEIGHT_COL = "residual_correction_weight_unclipped"

EPISODE_KEY_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
]

ROW_JOIN_KEY_COLS = [
    *EPISODE_KEY_COLS,
    "period_start",
    "period_end",
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "removed_in_period",
]


NUISANCE_COLUMNS = [
    "p_keep_obs",
    *PREDICTION_COLUMNS,
    "cauti_in_period",
    "reinsertion_in_period",
    "death_in_period",
    "icu_exit_alive_in_period",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "episode_end_reason",
    "reinsertion_time",
    "_crossfit_fold",
]

EPISODE_FIRST_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
    "reinsertion_time",
    "_crossfit_fold",
    "episode_end_reason",
]

MU_COLUMNS = [
    "mu_cauti_under_policy",
    "mu_recatheterisation_under_policy",
    "mu_death_under_policy",
    "mu_icu_exit_alive_under_policy",
    "mu_no_event_under_policy",
]

OUTCOME_SPECS = {
    "cauti": {
        "plugin": "plugin_predicted_any_cauti",
        "observed": "observed_any_cauti",
        "mu": "mu_cauti_under_policy",
        "period_outcome": "cauti_in_period",
        "summary_stub": "cauti_risk",
    },
    "recatheterisation": {
        "plugin": "plugin_predicted_any_recatheterisation",
        "observed": "observed_any_recatheterisation",
        "mu": "mu_recatheterisation_under_policy",
        "period_outcome": "reinsertion_in_period",
        "summary_stub": "recatheterisation_risk",
    },
    "death": {
        "plugin": "plugin_predicted_any_death",
        "observed": "observed_any_death",
        "mu": "mu_death_under_policy",
        "period_outcome": "death_in_period",
        "summary_stub": "death_risk",
    },
    "icu_exit_alive": {
        "plugin": "plugin_predicted_icu_exit_alive",
        "observed": "observed_icu_exit_alive",
        "mu": "mu_icu_exit_alive_under_policy",
        "period_outcome": "icu_exit_alive_in_period",
        "summary_stub": "icu_exit_alive_risk",
    },
    "catheter_exposure_days": {
        "plugin": "plugin_expected_catheter_exposure_days",
        "observed": "observed_catheter_exposure_days",
        "summary_stub": "catheter_exposure_days",
    },
}

MISSING_COUNTERFACTUAL_MESSAGE = (
    "Missing counterfactual state/action outcome predictions were found. AIPW "
    "requires plug-in predictions for every eligible episode under every target "
    "policy. Re-run nuisance scoring with complete counterfactual predictions, "
    "or provide outcome_models.pkl for rescoring."
)


# Generic helpers


def load_nuisance_predictions(path):
    # Load and validate the nuisance predictions
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df[["p_remove_obs", "p_keep_obs"]] = df[
        ["p_remove_obs", "p_keep_obs"]
    ].apply(pd.to_numeric, errors="coerce")
    return df


def join_nuisance_predictions(policy_df, nuisance_df):
    nuisance_add_cols = [
        col
        for col in [
            "p_remove_obs",
            *NUISANCE_COLUMNS,
            *[f"__rescored_{col}" for col in PREDICTION_COLUMNS],
        ]
        if col not in ROW_JOIN_KEY_COLS
    ]

    merged = policy_df.merge(
        nuisance_df[[*ROW_JOIN_KEY_COLS, *nuisance_add_cols]],
        on=ROW_JOIN_KEY_COLS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, ROW_JOIN_KEY_COLS + ["policy_name"]].head(10)
        raise ValueError(
            "Some policy-panel rows did not match the nuisance predictions. "
            f"Examples:\n{examples}"
        )
    return merged.drop(columns="_merge")


# Policy timeline and nuisance prediction selection


def select_policy_predictions(df):
    # Select predictions implied by the target policy
    for col in MU_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df["policy_action"].eq("keep")
    remove_rows = df["policy_catheter_state"].eq("in") & df["policy_action"].eq("remove")
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    assign_policy_prediction(df, "mu_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    assign_policy_prediction(df, "mu_death_under_policy", "p_death_if_keep", keep_rows)
    assign_policy_prediction(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    assign_policy_prediction(df, "mu_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "mu_recatheterisation_under_policy"] = 0.0

    assign_policy_prediction(df, "mu_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    assign_policy_prediction(df, "mu_death_under_policy", "p_death_if_remove", remove_rows)
    assign_policy_prediction(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    assign_policy_prediction(df, "mu_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "mu_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "mu_cauti_under_policy"] = 0.0
    assign_policy_prediction(df, "mu_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    assign_policy_prediction(df, "mu_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    assign_policy_prediction(df, "mu_death_under_policy", "p_death_if_out", out_rows)
    assign_policy_prediction(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
    assign_policy_prediction(df, "mu_no_event_under_policy", "p_no_event_if_out", out_rows)

    missing_any = df[MU_COLUMNS].isna().any(axis=1)
    invalid_any = pd.Series(False, index=df.index)
    for col in MU_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_any |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))
    df["prediction_status"] = "complete"
    df.loc[missing_any, "prediction_status"] = "missing_prediction"
    df.loc[invalid_any, "prediction_status"] = "invalid_probability"
    return df


def validate_prediction_completeness(df):
    # Check prediction completeness and probability bounds
    invalid_rows = pd.Series(False, index=df.index)
    for col in MU_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_rows |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))
    if invalid_rows.any():
        examples = df.loc[invalid_rows, ["policy_name", "decision_row_id", *MU_COLUMNS]].head(10)
        raise ValueError(
            "AIPW plug-in predictions must be finite probabilities between 0 and 1 "
            f"where present. Examples:\n{examples}"
        )
    total_missing = int(df[MU_COLUMNS].isna().sum().sum())
    if total_missing:
        missing_counts = df.groupby("policy_name", dropna=False)[MU_COLUMNS].apply(
            lambda frame: frame.isna().sum()
        )
        raise ValueError(
            f"{MISSING_COUNTERFACTUAL_MESSAGE}\nMissing prediction counts by policy:\n{missing_counts}"
        )


# AIPW support, adherence and episode-level scores

def add_support_and_adherence(df, clip_lower, clip_upper):
    # Add support probabilities and adherence flags
    for col in ["p_remove_obs", "p_keep_obs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    applicable = df["policy_applicable"]
    missing_propensity = applicable & (df["p_remove_obs"].isna() | df["p_keep_obs"].isna())
    if missing_propensity.any():
        examples = df.loc[
            missing_propensity,
            ["policy_name", "decision_row_id", "p_remove_obs", "p_keep_obs"],
        ].head(10)
        raise ValueError(
            "Applicable rows are missing p_remove_obs or p_keep_obs. "
            f"Examples:\n{examples}"
        )
    for col in ["p_remove_obs", "p_keep_obs"]:
        invalid = applicable & (~np.isfinite(df[col]) | df[col].lt(0) | df[col].gt(1))
        if invalid.any():
            examples = df.loc[invalid, ["policy_name", "decision_row_id", col]].head(10)
            raise ValueError(
                f"Behaviour-policy probabilities in {col} must be finite and "
                f"between 0 and 1 for applicable rows. Examples:\n{examples}"
            )

    df["policy_support"] = np.nan
    remove_rows = applicable & df["policy_action"].eq("remove")
    keep_rows = applicable & df["policy_action"].eq("keep")
    df.loc[remove_rows, "policy_support"] = df.loc[remove_rows, "p_remove_obs"]
    df.loc[keep_rows, "policy_support"] = df.loc[keep_rows, "p_keep_obs"]
    support = pd.to_numeric(df["policy_support"], errors="coerce")

    invalid_support = applicable & (support.isna() | ~np.isfinite(support) | support.lt(0) | support.gt(1))
    if invalid_support.any():
        examples = df.loc[
            invalid_support,
            ["policy_name", "decision_row_id", "policy_support"],
        ].head(10)
        raise ValueError(f"Policy support must be finite and between 0 and 1. Examples:\n{examples}")

    df["policy_support_clipped"] = support.clip(lower=clip_lower, upper=clip_upper)
    df["row_ipw_component"] = np.nan
    df["row_ipw_component_unclipped"] = np.nan
    df["zero_support_matched_row"] = 0

    matched = applicable & df["policy_matches_observed_action_today"].eq(1)
    df.loc[matched, "row_ipw_component"] = 1.0 / df.loc[matched, "policy_support_clipped"]
    safe_unclipped = matched & support.gt(0)
    df.loc[safe_unclipped, "row_ipw_component_unclipped"] = 1.0 / support.loc[safe_unclipped]
    df.loc[matched & support.eq(0), "zero_support_matched_row"] = 1

    df["deviated_from_policy_today"] = 0
    df.loc[applicable & df["policy_matches_observed_action_today"].eq(0), "deviated_from_policy_today"] = 1

    sort_cols = ["policy_name", EPISODE_ID_COL, "period_start", "period_end", "decision_row_id"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    group_cols = ["policy_name", EPISODE_ID_COL]
    deviation_cummax = df.groupby(group_cols, sort=False)["deviated_from_policy_today"].cummax()
    df["followed_policy_so_far"] = (1 - deviation_cummax).astype(int)
    episode_deviation = df.groupby(group_cols, sort=False)["deviated_from_policy_today"].transform("max")
    df["episode_adherent_to_policy"] = episode_deviation.eq(0).astype(int)
    return df


def product_components_by_episode(df, component_col):
    # Multiply row components within each episode
    group_cols = ["policy_name", "policy_remove_day", EPISODE_ID_COL]
    return (
        df.groupby(group_cols, dropna=False, sort=False)[component_col]
        .prod(min_count=1)
        .reset_index()
    )


def build_policy_episode_scores(df):
    # Collapse row scores to policy-episode scores
    # Add observed outcomes to rows
    df = pec.add_observed_icu_exit_alive_period(df)
    df["_icu_exit_alive_period"] = df["observed_icu_exit_alive_in_period"]
    df["_applicable_int"] = df["policy_applicable"].astype(int)
    df["_matched_applicable_int"] = (
        df["policy_applicable"] & df["policy_matches_observed_action_today"].eq(1)
    ).astype(int)
    df["_catheter_in_row_int"] = df["catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_policy_catheter_in_row_int"] = df["policy_catheter_state"].eq("in").astype(int)
    df["_policy_remove_row_int"] = df[
        "policy_action"
    ].eq("remove").astype(int)
    df["_observed_catheter_exposure_days"] = df["_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    df["_policy_catheter_exposure_days"] = df["_policy_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    day = pd.to_numeric(df["episode_day"], errors="coerce")
    remove_day = pd.to_numeric(df["policy_remove_day"], errors="coerce")
    death_period = pd.to_numeric(
        df["death_in_period"],
        errors="coerce",
    ).fillna(0)
    icu_period = pd.to_numeric(
        df["icu_exit_alive_in_period"],
        errors="coerce",
    ).fillna(0)
    terminal_period = death_period.eq(1) | icu_period.eq(1)
    df["_terminal_period"] = terminal_period.astype(int)
    df["_observed_removed_before_policy_day"] = (
        day.lt(remove_day)
        & df["is_decision_row"].astype(bool)
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(1)
    ).astype(int)
    df["_failed_to_remove_on_policy_day"] = (
        day.eq(remove_day)
        & df["policy_applicable"].astype(bool)
        & df["policy_action"].eq("remove")
        & df["policy_matches_observed_action_today"].eq(0)
    ).astype(int)

    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day", EPISODE_ID_COL]
    episode_df = df.groupby(group_cols, as_index=False, dropna=False, sort=False).agg(
        episode_adherent_to_policy=("episode_adherent_to_policy", "max"),
        n_applicable_policy_rows=("_applicable_int", "sum"),
        n_matched_policy_rows=("_matched_applicable_int", "sum"),
        n_deviation_rows=("deviated_from_policy_today", "sum"),
        zero_support_matched_row=("zero_support_matched_row", "max"),
        n_policy_remove_rows=("_policy_remove_row_int", "sum"),
        n_policy_removal_day_extra_rows_treated_as_out=(
            "policy_removal_day_extra_row_treated_as_out",
            "sum",
        ),
        plugin_expected_catheter_in_interval_rows=("_policy_catheter_in_row_int", "sum"),
        plugin_expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
        observed_catheter_in_interval_rows=("_catheter_in_row_int", "sum"),
        observed_catheter_exposure_days=("_observed_catheter_exposure_days", "sum"),
        max_episode_day=("episode_day", "max"),
        episode_has_terminal_event=("_terminal_period", "max"),
        episode_observed_removed_before_policy_day=("_observed_removed_before_policy_day", "max"),
        episode_failed_to_remove_on_policy_day=("_failed_to_remove_on_policy_day", "max"),
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
    )
    episode_df["episode_has_more_than_one_policy_remove_row"] = (
        episode_df["n_policy_remove_rows"].gt(1)
    ).astype(int)
    max_day = pd.to_numeric(episode_df["max_episode_day"], errors="coerce")
    remove_day_episode = pd.to_numeric(episode_df["policy_remove_day"], errors="coerce")
    before_policy_day = max_day.lt(remove_day_episode)
    episode_df["episode_terminal_before_policy_removal"] = (
        before_policy_day & episode_df["episode_has_terminal_event"].eq(1)
    ).astype(int)
    episode_df["episode_censored_before_policy_removal"] = (
        before_policy_day & episode_df["episode_has_terminal_event"].ne(1)
    ).astype(int)

    for col in EPISODE_FIRST_COLS:
        values = df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[col].first()
        episode_df = episode_df.merge(values, on=group_cols, how="left")

    for outcome_name, spec in OUTCOME_SPECS.items():
        if outcome_name == "catheter_exposure_days":
            continue
        values = df.groupby(group_cols, as_index=False, dropna=False)[spec["mu"]].agg(
            cumulative_event_probability
        )
        values = values.rename(columns={spec["mu"]: spec["plugin"]})
        episode_df = episode_df.merge(values, on=group_cols, how="left")

        period_col = (
            "_icu_exit_alive_period"
            if outcome_name == "icu_exit_alive"
            else spec["period_outcome"]
        )
        observed_values = (
            pd.to_numeric(df[period_col], errors="coerce").gt(0)
            .groupby([df[col] for col in group_cols], dropna=False)
            .max().astype(int).reset_index(name=spec["observed"])
        )
        episode_df = episode_df.merge(
            observed_values,
            on=group_cols,
            how="left",
        )

    # Multiply row components within each episode
    weight_products = product_components_by_episode(df, "row_ipw_component")
    # Multiply row components within each episode
    unclipped_weight_products = product_components_by_episode(df, "row_ipw_component_unclipped")
    merge_cols = ["policy_name", "policy_remove_day", EPISODE_ID_COL]
    episode_df = episode_df.merge(weight_products, on=merge_cols, how="left")
    episode_df = episode_df.merge(
        unclipped_weight_products,
        on=merge_cols,
        how="left",
        suffixes=("", "_unclipped"),
    )
    episode_df = episode_df.rename(
        columns={
            "row_ipw_component": WEIGHT_COL,
            "row_ipw_component_unclipped": UNCLIPPED_WEIGHT_COL,
        }
    )

    episode_df["zero_applicable_rows_weight_assigned"] = 0
    zero_applicable_adherent = (
        episode_df["episode_adherent_to_policy"].eq(1)
        & episode_df["n_applicable_policy_rows"].eq(0)
    )
    if zero_applicable_adherent.any():
        n_zero = int(zero_applicable_adherent.sum())
        print(
            "WARNING: assigning weight 1 to adherent AIPW episodes with no "
            f"applicable decision rows: {n_zero}",
            flush=True,
        )
        episode_df.loc[zero_applicable_adherent, WEIGHT_COL] = 1.0
        episode_df.loc[zero_applicable_adherent, UNCLIPPED_WEIGHT_COL] = 1.0
        episode_df.loc[zero_applicable_adherent, "zero_applicable_rows_weight_assigned"] = 1

    missing_adherent_weight = (
        episode_df["episode_adherent_to_policy"].eq(1)
        & episode_df["n_applicable_policy_rows"].gt(0)
        & episode_df[WEIGHT_COL].isna()
    )
    if missing_adherent_weight.any():
        examples = episode_df.loc[
            missing_adherent_weight,
            ["policy_name", EPISODE_ID_COL, "n_applicable_policy_rows", "n_matched_policy_rows"],
        ].head(10)
        raise ValueError(
            "Some adherent policy episodes have applicable rows but no AIPW "
            f"residual-correction weight. Examples:\n{examples}"
        )

    episode_df[RESIDUAL_WEIGHT_COL] = np.where(
        episode_df["episode_adherent_to_policy"].eq(1),
        pd.to_numeric(episode_df[WEIGHT_COL], errors="coerce"),
        0.0,
    )
    episode_df[UNCLIPPED_RESIDUAL_WEIGHT_COL] = np.where(
        episode_df["episode_adherent_to_policy"].eq(1),
        pd.to_numeric(episode_df[UNCLIPPED_WEIGHT_COL], errors="coerce"),
        0.0,
    )
    residual_multiplier = pd.to_numeric(episode_df[RESIDUAL_WEIGHT_COL], errors="coerce").fillna(0.0)
    for outcome_name, spec in OUTCOME_SPECS.items():
        plugin_col = spec["plugin"]
        observed_col = spec["observed"]
        residual_col = f"residual_{outcome_name}"
        weighted_residual_col = f"weighted_residual_{outcome_name}"
        score_col = f"aipw_ht_score_{outcome_name}"
        episode_df[residual_col] = episode_df[observed_col] - episode_df[plugin_col]
        episode_df[weighted_residual_col] = residual_multiplier * episode_df[residual_col]
        episode_df[score_col] = episode_df[plugin_col] + episode_df[weighted_residual_col]

    # Order episode-level output columns
    return order_episode_columns(episode_df), df


# Current-practice comparator

def map_episode_ids_to_nuisance_predictions(nuisance_df, policy_df):
    # Map episode identifiers onto nuisance-prediction rows
    episode_map = policy_df[[*EPISODE_KEY_COLS, EPISODE_ID_COL]].drop_duplicates()
    out = nuisance_df.merge(
        episode_map,
        on=EPISODE_KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    if out[EPISODE_ID_COL].isna().any():
        examples = out.loc[out[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
        raise ValueError(
            "Some nuisance-prediction rows could not be mapped to episodes. "
            f"Examples:\n{examples}"
        )
    return out


def build_current_practice_rows(nuisance_df, policy_df):
    # Build rows for the observed current-practice regime
    # Map episode identifiers onto nuisance-prediction rows
    df = map_episode_ids_to_nuisance_predictions(nuisance_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice AIPW rows")
    # Add episode day since catheter insertion
    df = add_episode_day(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action"] = "out"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(0),
        "policy_action",
    ] = "keep"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(1),
        "policy_action",
    ] = "remove"
    df["policy_periods_in"] = np.where(df["policy_catheter_state"].eq("in"), df["periods_in_state"], np.nan)
    df["policy_periods_out"] = np.where(df["policy_catheter_state"].eq("out"), df["periods_in_state"], np.nan)
    df["policy_applicable"] = False
    df["policy_matches_observed_action_today"] = np.nan
    df["deviated_from_policy_today"] = 0
    df["followed_policy_so_far"] = 1
    df["episode_adherent_to_policy"] = 1
    df["policy_support"] = np.nan
    df["policy_support_clipped"] = np.nan
    df["row_ipw_component"] = np.nan
    df["row_ipw_component_unclipped"] = np.nan
    df["zero_support_matched_row"] = 0
    return df


def build_current_practice_episode_scores(current_rows):
    # Build episode scores for current practice
    # Add observed outcomes to rows
    df = pec.add_observed_icu_exit_alive_period(current_rows)
    df["_icu_exit_alive_period"] = df["observed_icu_exit_alive_in_period"]
    df["_catheter_in_row_int"] = df["catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_policy_catheter_in_row_int"] = df["policy_catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_observed_catheter_exposure_days"] = df["_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    df["_policy_catheter_exposure_days"] = df["_policy_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day", EPISODE_ID_COL]
    episode_df = df.groupby(group_cols, as_index=False, dropna=False, sort=False).agg(
        plugin_expected_catheter_in_interval_rows=("_policy_catheter_in_row_int", "sum"),
        plugin_expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
        observed_catheter_in_interval_rows=("_catheter_in_row_int", "sum"),
        observed_catheter_exposure_days=("_observed_catheter_exposure_days", "sum"),
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
    )
    for col in EPISODE_FIRST_COLS:
        values = df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[col].first()
        episode_df = episode_df.merge(values, on=group_cols, how="left")
    for outcome_name, spec in OUTCOME_SPECS.items():
        if outcome_name == "catheter_exposure_days":
            continue
        values = df.groupby(group_cols, as_index=False, dropna=False)[spec["mu"]].agg(
            cumulative_event_probability
        )
        values = values.rename(columns={spec["mu"]: spec["plugin"]})
        episode_df = episode_df.merge(values, on=group_cols, how="left")
        period_col = (
            "_icu_exit_alive_period"
            if outcome_name == "icu_exit_alive"
            else spec["period_outcome"]
        )
        observed_values = (
            pd.to_numeric(df[period_col], errors="coerce").gt(0)
            .groupby([df[col] for col in group_cols], dropna=False)
            .max().astype(int).reset_index(name=spec["observed"])
        )
        episode_df = episode_df.merge(
            observed_values,
            on=group_cols,
            how="left",
        )

    episode_df["episode_adherent_to_policy"] = 1
    episode_df[WEIGHT_COL] = 1.0
    episode_df[UNCLIPPED_WEIGHT_COL] = 1.0
    episode_df[RESIDUAL_WEIGHT_COL] = 1.0
    episode_df[UNCLIPPED_RESIDUAL_WEIGHT_COL] = 1.0
    episode_df["n_applicable_policy_rows"] = pd.NA
    episode_df["n_matched_policy_rows"] = pd.NA
    episode_df["n_deviation_rows"] = 0
    episode_df["zero_applicable_rows_weight_assigned"] = 0
    episode_df["zero_support_matched_row"] = 0

    for outcome_name, spec in OUTCOME_SPECS.items():
        plugin_col = spec["plugin"]
        observed_col = spec["observed"]
        residual_col = f"residual_{outcome_name}"
        weighted_residual_col = f"weighted_residual_{outcome_name}"
        score_col = f"aipw_ht_score_{outcome_name}"
        episode_df[residual_col] = episode_df[observed_col] - episode_df[plugin_col]
        episode_df[weighted_residual_col] = episode_df[RESIDUAL_WEIGHT_COL] * episode_df[residual_col]
        episode_df[score_col] = episode_df[observed_col].where(
            episode_df[observed_col].notna(),
            episode_df[plugin_col],
        )
        episode_df[f"current_practice_observed_{outcome_name}"] = episode_df[observed_col]
        episode_df[f"current_practice_plugin_predicted_{outcome_name}"] = episode_df[plugin_col]
        episode_df[f"current_practice_aipw_{outcome_name}"] = episode_df[score_col]
    # Order episode-level output columns
    return order_episode_columns(episode_df)


# Summaries, diagnostics and sensitivity

def policy_summary_row(
    policy_df,
    policy_name,
    policy_type,
    policy_remove_day,
    residual_normalisation,
    weight_col=RESIDUAL_WEIGHT_COL,
):
    # Build one policy summary row
    complete_df = policy_df.loc[policy_df["prediction_complete"].astype(bool)].copy()
    residual_weight = pd.to_numeric(complete_df[weight_col], errors="coerce").fillna(0.0)
    residual_weight_all = pd.to_numeric(policy_df[weight_col], errors="coerce").fillna(0.0)
    # Return positive finite weights
    positive_residual_weights = residual_weight_all[np.isfinite(residual_weight_all) & residual_weight_all.gt(0)]
    residual_weight_ess = effective_sample_size(positive_residual_weights)
    n_total = int(len(policy_df))
    n_complete = int(len(complete_df))
    n_adherent = int(policy_df["episode_adherent_to_policy"].fillna(0).sum())
    numeric_remove_day = pd.to_numeric(pd.Series([policy_remove_day]), errors="coerce").iloc[0]
    fixed_day_policy = pd.notna(numeric_remove_day) and policy_name != CURRENT_PRACTICE_LABEL
    n_policy_remove_rows = int(
        pd.to_numeric(
            policy_df.get("n_policy_remove_rows", pd.Series(0, index=policy_df.index)),
            errors="coerce",
        ).fillna(0).sum()
    )
    if fixed_day_policy and "max_episode_day" in policy_df.columns:
        n_episodes_reaching_policy_removal_day = int(
            pd.to_numeric(policy_df["max_episode_day"], errors="coerce")
            .ge(numeric_remove_day)
            .sum()
        )
        policy_remove_row_shortfall = int(n_episodes_reaching_policy_removal_day - n_policy_remove_rows)
    else:
        n_episodes_reaching_policy_removal_day = 0
        policy_remove_row_shortfall = pd.NA

    # Calculate the effective sample size
    row = {
        "policy_name": policy_name,
        POLICY_TYPE_COL: policy_type,
        "policy_remove_day": policy_remove_day,
        "estimator": ESTIMATOR_NAME,
        "residual_normalisation": residual_normalisation,
        "n_patients": int(policy_df["subject_id"].nunique()),
        "n_episodes": n_total,
        "n_complete_prediction_episodes": n_complete,
        "n_incomplete_prediction_episodes": int(n_total - n_complete),
        "n_adherent_episodes": n_adherent,
        "pct_adherent_episodes": n_adherent / n_total if n_total else np.nan,
        "n_policy_remove_rows": n_policy_remove_rows,
        "n_episodes_reaching_policy_removal_day": n_episodes_reaching_policy_removal_day,
        "n_policy_remove_row_shortfall_vs_reached_episodes": policy_remove_row_shortfall,
        "n_episodes_with_more_than_one_remove_row": int(
            pd.to_numeric(
                policy_df.get(
                    "episode_has_more_than_one_policy_remove_row",
                    pd.Series(0, index=policy_df.index),
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "n_policy_removal_day_extra_rows_treated_as_out": int(
            pd.to_numeric(
                policy_df.get(
                    "n_policy_removal_day_extra_rows_treated_as_out",
                    pd.Series(0, index=policy_df.index),
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "weight_diagnostic_type": "adherent_residual_correction_weight",
        "sum_weights": float(positive_residual_weights.sum()) if len(positive_residual_weights) else np.nan,
        "sum_residual_correction_weights": float(positive_residual_weights.sum()) if len(positive_residual_weights) else np.nan,
        "effective_sample_size": residual_weight_ess,
        "residual_correction_effective_sample_size": residual_weight_ess,
        "n_positive_residual_correction_weights": int(len(positive_residual_weights)),
    }

    for outcome_name, spec in OUTCOME_SPECS.items():
        plugin_col = spec["plugin"]
        residual_col = f"residual_{outcome_name}"
        summary_stub = spec["summary_stub"]

        mean_plugin = float(complete_df[plugin_col].mean()) if len(complete_df) else np.nan
        weighted_residual = residual_weight * pd.to_numeric(complete_df[residual_col], errors="coerce")
        ht_scores = pd.to_numeric(complete_df[plugin_col], errors="coerce") + weighted_residual
        ht_value = float(ht_scores.mean()) if len(ht_scores.dropna()) else np.nan
        valid_residual_rows = weighted_residual.notna() & residual_weight.notna() & np.isfinite(residual_weight)
        residual_weight_sum = float(residual_weight.loc[valid_residual_rows].sum())
        if residual_weight_sum > 0:
            hajek_correction = float(weighted_residual.loc[valid_residual_rows].sum() / residual_weight_sum)
        else:
            hajek_correction = np.nan
        ht_correction = float(weighted_residual.mean()) if len(weighted_residual.dropna()) else np.nan
        hajek_value = mean_plugin + hajek_correction if pd.notna(mean_plugin) and pd.notna(hajek_correction) else np.nan
        selected_value = ht_value if residual_normalisation == "ht" else hajek_value
        selected_correction = ht_correction if residual_normalisation == "ht" else hajek_correction

        if outcome_name == "catheter_exposure_days":
            row["plugin_expected_mean_catheter_exposure_days"] = mean_plugin
            row["aipw_ht_mean_catheter_exposure_days"] = ht_value
            row["aipw_hajek_mean_catheter_exposure_days"] = hajek_value
            row["aipw_mean_catheter_exposure_days"] = selected_value
            row["residual_correction_catheter_exposure_days"] = selected_correction
            row["plugin_expected_mean_catheter_in_interval_rows"] = float(
                complete_df["plugin_expected_catheter_in_interval_rows"].mean()
            )
            row["aipw_mean_catheter_in_interval_rows"] = row[
                "plugin_expected_mean_catheter_in_interval_rows"
            ]
        else:
            bounded_ht_value = float(np.clip(ht_value, 0.0, 1.0))
            bounded_hajek_value = float(np.clip(hajek_value, 0.0, 1.0))
            bounded_selected_value = float(np.clip(selected_value, 0.0, 1.0))
            row[f"plugin_predicted_{summary_stub}"] = mean_plugin
            row[f"aipw_ht_{summary_stub}"] = bounded_ht_value
            row[f"aipw_hajek_{summary_stub}"] = bounded_hajek_value
            row[f"aipw_{summary_stub}"] = bounded_selected_value
            row[f"aipw_{summary_stub}_pct"] = (
                bounded_selected_value * 100 if pd.notna(bounded_selected_value) else np.nan
            )
            row[f"residual_correction_{outcome_name}"] = selected_correction
            row[f"aipw_ht_unbounded_estimate_{outcome_name}"] = ht_value
            row[f"aipw_hajek_unbounded_estimate_{outcome_name}"] = hajek_value
            row[f"aipw_selected_unbounded_estimate_{outcome_name}"] = selected_value
            row[f"aipw_ht_was_bounded_{outcome_name}"] = bool(
                pd.notna(ht_value) and (ht_value < 0 or ht_value > 1)
            )
            row[f"aipw_hajek_was_bounded_{outcome_name}"] = bool(
                pd.notna(hajek_value) and (hajek_value < 0 or hajek_value > 1)
            )
            row[f"aipw_selected_was_bounded_{outcome_name}"] = bool(
                pd.notna(selected_value) and (selected_value < 0 or selected_value > 1)
            )
    return row


def build_policy_summary(episode_df, residual_normalisation):
    # Build policy-level summary estimates
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
    # Build one policy summary row
    for policy_values, policy_df in episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        # Build one policy summary row
        rows.append(policy_summary_row(policy_df, policy_name, policy_type, policy_remove_day, residual_normalisation))
    # Add comparisons against current practice
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary):
    # Add comparisons against current practice
    return pec.add_standard_comparisons(
        summary,
        baseline_label=CURRENT_PRACTICE_LABEL,
        comparison_map={
            "cauti_risk": "aipw_cauti_risk",
            "recatheterisation_risk": "aipw_recatheterisation_risk",
            "death_risk": "aipw_death_risk",
            "icu_exit_alive_risk": "aipw_icu_exit_alive_risk",
            "catheter_exposure_days": "aipw_mean_catheter_exposure_days",
        },
    )


def inverse_support_ess(support):
    # Calculate inverse-support effective sample size
    valid = pd.to_numeric(support, errors="coerce")
    valid = valid[valid.notna() & np.isfinite(valid) & valid.gt(0)]
    if valid.empty:
        return np.nan
    # Calculate the effective sample size
    return effective_sample_size(1.0 / valid)


def build_support_diagnostics(row_df):
    # Build support diagnostic output
    rows = []
    # Build one support diagnostic row
    for policy_values, policy_df in row_df.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        # Build one support diagnostic row
        for label, group_df in [("all", policy_df)]:
            # Build one support diagnostic row
            rows.append(support_diagnostic_row(group_df, label, policy_name, policy_remove_day))
        # Summarise support within each cross-fit fold
        for fold_value, fold_df in policy_df.groupby(
            CROSSFIT_FOLD_COL,
            dropna=False,
            sort=False,
        ):
            rows.append(
                support_diagnostic_row(
                    fold_df,
                    f"{CROSSFIT_FOLD_COL}={fold_value}",
                    policy_name,
                    policy_remove_day,
                )
            )
    return pd.DataFrame(rows)


def support_diagnostic_row(df, label, policy_name, policy_remove_day):
    # Build one support diagnostic row
    applicable = df["policy_applicable"]
    support = pd.to_numeric(df.loc[applicable, "policy_support"], errors="coerce")
    finite = support.notna() & np.isfinite(support)
    valid = support.loc[finite]
    # Calculate inverse-support effective sample size
    row = {
        "policy_name": policy_name,
        "policy_remove_day": policy_remove_day,
        "group": label,
        "n_applicable_rows": int(applicable.sum()),
        "mean_policy_support": float(valid.mean()) if len(valid) else np.nan,
        "median_policy_support": float(valid.median()) if len(valid) else np.nan,
        "min_policy_support": float(valid.min()) if len(valid) else np.nan,
        "pct_below_0_10": float(valid.lt(0.10).mean()) if len(valid) else np.nan,
        "pct_below_0_05": float(valid.lt(0.05).mean()) if len(valid) else np.nan,
        "pct_below_0_01": float(valid.lt(0.01).mean()) if len(valid) else np.nan,
        "row_inverse_support_effective_sample_size": inverse_support_ess(support),
        "n_missing_support_rows": int(support.isna().sum()),
        "n_non_finite_support_rows": int((support.notna() & ~np.isfinite(support)).sum()),
    }
    row["low_support_flag"] = bool(
        pd.notna(row["pct_below_0_05"])
        and row["pct_below_0_05"] > pec.LOW_SUPPORT_PCT_BELOW_005_THRESHOLD
    )
    return row


def build_weight_diagnostics(episode_df):
    # Build weight diagnostic output
    rows = []
    # Return positive finite weights
    for policy_values, policy_df in episode_df.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        if policy_name == CURRENT_PRACTICE_LABEL:
            continue
        # Return positive finite weights
        residual_weights = pd.to_numeric(policy_df[RESIDUAL_WEIGHT_COL], errors="coerce")
        residual_weights = residual_weights[np.isfinite(residual_weights) & residual_weights.gt(0)]
        # Return positive finite weights
        raw_episode_weights = pd.to_numeric(policy_df[WEIGHT_COL], errors="coerce")
        raw_episode_weights = raw_episode_weights[np.isfinite(raw_episode_weights) & raw_episode_weights.gt(0)]
        n_total = int(len(policy_df))
        n_adherent = int(policy_df["episode_adherent_to_policy"].eq(1).sum())
        pct_adherent = n_adherent / n_total if n_total else np.nan
        # Calculate the effective sample size
        ess = effective_sample_size(policy_df[RESIDUAL_WEIGHT_COL])
        p99 = float(residual_weights.quantile(0.99)) if len(residual_weights) else np.nan
        max_weight = float(residual_weights.max()) if len(residual_weights) else np.nan
        # Calculate the effective sample size
        rows.append({
            "policy_name": policy_name,
            "policy_remove_day": policy_remove_day,
            "weight_diagnostic_type": "adherent_residual_correction_weight",
            "n_total_policy_episodes": n_total,
            "n_adherent_episodes": n_adherent,
            "n_non_adherent_episodes": int(n_total - n_adherent),
            "pct_adherent_episodes": pct_adherent,
            "n_positive_residual_correction_weights": int(len(residual_weights)),
            "sum_residual_correction_weights": float(residual_weights.sum()) if len(residual_weights) else np.nan,
            "min_weight": float(residual_weights.min()) if len(residual_weights) else np.nan,
            "median_weight": float(residual_weights.median()) if len(residual_weights) else np.nan,
            "p90_weight": float(residual_weights.quantile(0.90)) if len(residual_weights) else np.nan,
            "p95_weight": float(residual_weights.quantile(0.95)) if len(residual_weights) else np.nan,
            "p99_weight": p99,
            "max_weight": max_weight,
            "effective_sample_size": ess,
            "residual_correction_effective_sample_size": ess,
            "raw_episode_sum_weights": float(raw_episode_weights.sum()) if len(raw_episode_weights) else np.nan,
            "raw_episode_effective_sample_size": effective_sample_size(raw_episode_weights),
            "low_adherence_flag": bool(pd.notna(pct_adherent) and pct_adherent < pec.LOW_ADHERENCE_THRESHOLD),
            "low_ess_flag": bool(
                pd.notna(ess)
                and (ess < pec.LOW_ESS_MIN or (n_total > 0 and ess < pec.LOW_ESS_FRACTION * n_total))
            ),
            "extreme_weight_flag": bool(
                (pd.notna(p99) and p99 > pec.EXTREME_WEIGHT_P99_THRESHOLD)
                or (pd.notna(max_weight) and max_weight > pec.EXTREME_WEIGHT_MAX_THRESHOLD)
            ),
            "n_zero_support_matched_episodes": int(policy_df["zero_support_matched_row"].fillna(0).sum()),
            "n_zero_applicable_rows_weight_assigned": int(policy_df["zero_applicable_rows_weight_assigned"].fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def build_residual_diagnostics(
    episode_df,
    residual_normalisation,
):
    # Build AIPW residual diagnostics
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
    # Return positive finite weights
    for policy_values, policy_df in episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        complete_df = policy_df.loc[policy_df["prediction_complete"].astype(bool)].copy()
        residual_weight = pd.to_numeric(
            complete_df[RESIDUAL_WEIGHT_COL],
            errors="coerce",
        ).fillna(0.0)
        adherent = pd.to_numeric(complete_df["episode_adherent_to_policy"], errors="coerce").fillna(0)
        residual_weight_all = pd.to_numeric(
            policy_df[RESIDUAL_WEIGHT_COL],
            errors="coerce",
        ).fillna(0.0)
        # Return positive finite weights
        positive_residual_weights = residual_weight_all[np.isfinite(residual_weight_all) & residual_weight_all.gt(0)]
        residual_weight_ess = effective_sample_size(positive_residual_weights)
        valid_residual_weight = residual_weight.notna() & np.isfinite(residual_weight)
        weight_sum = float(residual_weight.loc[valid_residual_weight].sum())

        # Calculate the effective sample size
        for outcome_name, spec in OUTCOME_SPECS.items():
            plugin_col = spec["plugin"]
            observed_col = spec["observed"]
            residual_col = f"residual_{outcome_name}"
            weighted_residual_col = f"weighted_residual_{outcome_name}"
            ht_score_col = f"aipw_ht_score_{outcome_name}"
            plugin = pd.to_numeric(complete_df[plugin_col], errors="coerce")
            observed = pd.to_numeric(complete_df[observed_col], errors="coerce")
            residual = pd.to_numeric(complete_df[residual_col], errors="coerce")
            weighted_residual = pd.to_numeric(complete_df[weighted_residual_col], errors="coerce")
            ht_score = pd.to_numeric(complete_df[ht_score_col], errors="coerce")
            mean_plugin = float(plugin.mean()) if len(plugin.dropna()) else np.nan
            ht_value = float(ht_score.mean()) if len(ht_score.dropna()) else np.nan
            hajek_correction = (
                float(weighted_residual.loc[valid_residual_weight].sum() / weight_sum)
                if weight_sum > 0 and len(weighted_residual.dropna())
                else np.nan
            )
            hajek_value = (
                mean_plugin + hajek_correction
                if pd.notna(mean_plugin) and pd.notna(hajek_correction)
                else np.nan
            )
            selected_value = ht_value if residual_normalisation == "ht" else hajek_value
            is_probability = outcome_name != "catheter_exposure_days"
            bounded_ht_value = float(np.clip(ht_value, 0.0, 1.0)) if is_probability else ht_value
            bounded_hajek_value = float(np.clip(hajek_value, 0.0, 1.0)) if is_probability else hajek_value
            bounded_selected_value = (
                float(np.clip(selected_value, 0.0, 1.0)) if is_probability else selected_value
            )
            # Calculate the effective sample size
            rows.append({
                "policy_name": policy_name,
                POLICY_TYPE_COL: policy_type,
                "policy_remove_day": policy_remove_day,
                "outcome": outcome_name,
                "residual_normalisation": residual_normalisation,
                "n_episodes": int(len(policy_df)),
                "n_complete_prediction_episodes": int(len(complete_df)),
                "n_adherent_episodes": int(complete_df["episode_adherent_to_policy"].eq(1).sum()),
                "weight_diagnostic_type": "adherent_residual_correction_weight",
                "sum_residual_correction_weights": float(positive_residual_weights.sum()) if len(positive_residual_weights) else np.nan,
                "effective_sample_size": residual_weight_ess,
                "residual_correction_effective_sample_size": residual_weight_ess,
                "plugin_mean": mean_plugin,
                "observed_mean_among_adherent": float(observed.loc[adherent.eq(1)].mean()) if len(observed.loc[adherent.eq(1)].dropna()) else np.nan,
                "mean_residual": float(residual.mean()) if len(residual.dropna()) else np.nan,
                "mean_weighted_residual": float(weighted_residual.mean()) if len(weighted_residual.dropna()) else np.nan,
                "p99_abs_weighted_residual": float(weighted_residual.abs().quantile(0.99)) if len(weighted_residual.dropna()) else np.nan,
                "max_abs_weighted_residual": float(weighted_residual.abs().max()) if len(weighted_residual.dropna()) else np.nan,
                "min_episode_aipw_ht_score": float(ht_score.min()) if len(ht_score.dropna()) else np.nan,
                "max_episode_aipw_ht_score": float(ht_score.max()) if len(ht_score.dropna()) else np.nan,
                "aipw_ht_estimate": bounded_ht_value,
                "aipw_hajek_estimate": bounded_hajek_value,
                "aipw_selected_estimate": bounded_selected_value,
                "aipw_ht_unbounded_estimate": ht_value,
                "aipw_hajek_unbounded_estimate": hajek_value,
                "aipw_selected_unbounded_estimate": selected_value,
                "aipw_ht_was_bounded": bool(
                    is_probability and pd.notna(ht_value) and (ht_value < 0 or ht_value > 1)
                ),
                "aipw_hajek_was_bounded": bool(
                    is_probability and pd.notna(hajek_value) and (hajek_value < 0 or hajek_value > 1)
                ),
                "aipw_selected_was_bounded": bool(
                    is_probability and pd.notna(selected_value) and (selected_value < 0 or selected_value > 1)
                ),
            })
    return pd.DataFrame(rows)


def build_clipping_sensitivity(
    episode_df,
    residual_normalisation,
    clip_lower,
    clip_upper,
):
    # Build clipping-sensitivity output
    rows = []
    target_df = episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
    # Return positive finite weights
    for policy_values, policy_df in target_df.groupby(["policy_name", "policy_type", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        default_weights = pd.to_numeric(policy_df[RESIDUAL_WEIGHT_COL], errors="coerce").fillna(0.0)
        unclipped_weights = pd.to_numeric(
            policy_df[UNCLIPPED_RESIDUAL_WEIGHT_COL],
            errors="coerce",
        ).fillna(0.0)
        # Return positive finite weights
        valid_default = default_weights[np.isfinite(default_weights) & default_weights.gt(0)]
        p99 = float(valid_default.quantile(0.99)) if len(valid_default) else np.nan
        # Build one policy summary row
        for rule, weights, support_low, support_high, upper in [
            ("unclipped_support_where_safe", unclipped_weights, np.nan, np.nan, np.nan),
            ("row_support_clipped", default_weights, clip_lower, clip_upper, np.nan),
            ("upper_episode_weight_clipped_at_p99", default_weights.clip(upper=p99) if pd.notna(p99) else default_weights, clip_lower, clip_upper, p99),
            ("upper_episode_weight_clipped_at_30", default_weights.clip(upper=30.0), clip_lower, clip_upper, 30.0),
            ("upper_episode_weight_clipped_at_20", default_weights.clip(upper=20.0), clip_lower, clip_upper, 20.0),
        ]:
            temp = policy_df.copy()
            temp["__sensitivity_weight"] = weights
            # Build one policy summary row
            summary_row = policy_summary_row(
                temp,
                policy_name,
                policy_type,
                policy_remove_day,
                residual_normalisation,
                "__sensitivity_weight",
            )
            numeric_weights = pd.to_numeric(weights, errors="coerce")
            valid_weights = numeric_weights[np.isfinite(numeric_weights) & numeric_weights.gt(0)]
            weight_ess = effective_sample_size(valid_weights)
            rows.append({
                "policy_name": policy_name,
                "policy_remove_day": policy_remove_day,
                "clipping_rule": rule,
                "weight_diagnostic_type": "adherent_residual_correction_weight",
                "support_clip_lower": support_low,
                "support_clip_upper": support_high,
                "clip_upper_weight": upper,
                "n_episodes": int(len(temp)),
                "n_adherent_episodes": int(temp["episode_adherent_to_policy"].eq(1).sum()),
                "sum_weights": float(valid_weights.sum()),
                "sum_residual_correction_weights": float(valid_weights.sum()),
                "effective_sample_size": weight_ess,
                "residual_correction_effective_sample_size": weight_ess,
                "aipw_cauti_risk": summary_row["aipw_cauti_risk"],
                "aipw_recatheterisation_risk": summary_row["aipw_recatheterisation_risk"],
                "aipw_death_risk": summary_row["aipw_death_risk"],
                "aipw_icu_exit_alive_risk": summary_row["aipw_icu_exit_alive_risk"],
                "aipw_mean_catheter_exposure_days": summary_row["aipw_mean_catheter_exposure_days"],
                "aipw_mean_catheter_in_interval_rows": summary_row["aipw_mean_catheter_in_interval_rows"],
            })
    return pd.DataFrame(rows)


# Output ordering

def order_episode_columns(df):
    # Order episode-level output columns
    preferred = [
        "subject_id",
        "hadm_id",
        "stay_id",
        EPISODE_ID_COL,
        "inserted",
        "removed",
        "policy_name",
        POLICY_TYPE_COL,
        "policy_remove_day",
        "episode_adherent_to_policy",
        WEIGHT_COL,
        UNCLIPPED_WEIGHT_COL,
        RESIDUAL_WEIGHT_COL,
        UNCLIPPED_RESIDUAL_WEIGHT_COL,
        "n_applicable_policy_rows",
        "n_matched_policy_rows",
        "n_deviation_rows",
        "n_policy_remove_rows",
        "n_policy_removal_day_extra_rows_treated_as_out",
        "episode_has_more_than_one_policy_remove_row",
        "plugin_predicted_any_cauti",
        "observed_any_cauti",
        "residual_cauti",
        "weighted_residual_cauti",
        "aipw_ht_score_cauti",
        "plugin_predicted_any_recatheterisation",
        "observed_any_recatheterisation",
        "residual_recatheterisation",
        "weighted_residual_recatheterisation",
        "aipw_ht_score_recatheterisation",
        "plugin_predicted_any_death",
        "observed_any_death",
        "residual_death",
        "weighted_residual_death",
        "aipw_ht_score_death",
        "plugin_predicted_icu_exit_alive",
        "observed_icu_exit_alive",
        "residual_icu_exit_alive",
        "weighted_residual_icu_exit_alive",
        "aipw_ht_score_icu_exit_alive",
        "plugin_expected_catheter_exposure_days",
        "observed_catheter_exposure_days",
        "residual_catheter_exposure_days",
        "weighted_residual_catheter_exposure_days",
        "aipw_ht_score_catheter_exposure_days",
        "plugin_expected_catheter_in_interval_rows",
        "observed_catheter_in_interval_rows",
        "prediction_complete",
        "n_missing_prediction_rows",
        "episode_terminal_before_policy_removal",
        "episode_censored_before_policy_removal",
        "episode_observed_removed_before_policy_day",
        "episode_failed_to_remove_on_policy_day",
        "_crossfit_fold",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


# Main

def print_console_summary(summary_df, row_df, episode_df, output_paths, policy_manifest_path, model_type):
    target_episode_df = episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)]
    print()
    print("--- AIPW POLICY EVALUATION COMPLETE ---")
    print(f"Policy panel index: {policy_manifest_path}")
    print(f"Nuisance model type: {model_type}")
    print(f"Number of policies: {row_df['policy_name'].nunique():,}")
    print(f"Number of patients: {row_df['subject_id'].nunique():,}")
    print(f"Number of episodes: {row_df[EPISODE_ID_COL].nunique():,}")
    print(f"Complete prediction episodes: {int(episode_df['prediction_complete'].astype(bool).sum()):,}")
    print(f"Adherent residual-correction episodes: {int(target_episode_df['episode_adherent_to_policy'].eq(1).sum()):,}")
    print()
    display_cols = [
        "policy_name",
        "aipw_cauti_risk",
        "aipw_recatheterisation_risk",
        "aipw_mean_catheter_exposure_days",
        "effective_sample_size",
        "pct_adherent_episodes",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print()
    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")


def evaluate_policy_episodes(policy_df, nuisance_df):
    model_feature_cols = [
        "episode_index",
        *[col for col in nuisance_df if col == "age" or col.startswith(("itemid_", "sex_", "ethnicity_"))],
    ]
    nuisance_df = nuisance_df.drop(columns=model_feature_cols)
    # Join nuisance scores to policy rows
    joined_df = join_nuisance_predictions(policy_df, nuisance_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined AIPW policy rows")
    # Select predictions implied by the target policy
    joined_df = select_policy_predictions(joined_df)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(joined_df)
    # Add support probabilities and adherence flags
    joined_df = add_support_and_adherence(joined_df, CLIP_LOWER, CLIP_UPPER)
    # Collapse row scores to policy-episode scores
    policy_episode_df, row_df = build_policy_episode_scores(joined_df)

    # Build rows for the observed current-practice regime
    current_rows = build_current_practice_rows(nuisance_df, policy_df)
    # Select predictions implied by the target policy
    current_rows = select_policy_predictions(current_rows)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(current_rows)
    # Build episode scores for current practice
    current_episode_df = build_current_practice_episode_scores(current_rows)

    episode_df = pd.concat([policy_episode_df, current_episode_df], ignore_index=True, sort=False)
    return episode_df, row_df, current_episode_df


def run_panel_estimation(panel_path, artefact_root, model_type, bootstrap_mode, panel_name=None):
    if bootstrap_mode not in ("fixed", "refit"):
        raise ValueError(f"Unknown bootstrap mode: {bootstrap_mode!r}")
    policy_manifest_path = artefact_root / "counterfactual_policies/policy_panels.csv"
    nuisance_model_dir = artefact_root / "nuisance_models" / model_type
    nuisance_predictions_path = nuisance_model_dir / "nuisance_predictions.csv"
    outcome_models_path = nuisance_model_dir / "outcome_models.pkl"
    outdir = artefact_root / "policy_eval/aipw" / bootstrap_mode
    output_paths = {key: outdir / name for key, name in OUTPUT_FILENAMES.items()}
    outdir.mkdir(exist_ok=True, parents=True)
    policy_df = read_policy_collection(policy_manifest_path)
    policy_df["subject_id"] = policy_df.subject_id.astype(str)
    refit_panel = None
    if bootstrap_mode == "refit":
        refit_panel = nuisance.load_panel(panel_path)
        subjects = pd.Index(sorted(refit_panel.subject_id.unique()), name="subject_id")
        nuisance_df, _ = bootstrap.refit_nuisance_predictions(
            refit_panel, subjects, np.ones(len(subjects), dtype=int), "aipw",
            n_splits=REFIT_CROSSFIT_FOLDS, model_type=model_type,
        )
    else:
        nuisance_df = load_nuisance_predictions(nuisance_predictions_path)
        nuisance_df = fill_missing_counterfactual_predictions(nuisance_df, outcome_models_path)
    nuisance_df["subject_id"] = nuisance_df.subject_id.astype(str)
    episode_df, row_df, current_episode_df = evaluate_policy_episodes(policy_df, nuisance_df)

    # Build policy-level summary estimates
    summary_df = build_policy_summary(episode_df, RESIDUAL_NORMALISATION)
    # Build support diagnostic output
    support_diagnostics_df = build_support_diagnostics(row_df)
    # Build weight diagnostic output
    weight_diagnostics_df = build_weight_diagnostics(episode_df)
    summary_df = pec.add_overlap_quality_flags(
        summary_df,
        support_diagnostics=support_diagnostics_df,
        weight_diagnostics=weight_diagnostics_df,
        current_practice_label=CURRENT_PRACTICE_LABEL,
    )
    # Build AIPW residual diagnostics
    residual_diagnostics_df = build_residual_diagnostics(
        episode_df,
        RESIDUAL_NORMALISATION,
    )
    # Build clipping-sensitivity output
    clipping_sensitivity_df = build_clipping_sensitivity(
        episode_df,
        RESIDUAL_NORMALISATION,
        CLIP_LOWER,
        CLIP_UPPER,
    )

    # Save rounded policy-level report outputs
    pec.save_report_df(summary_df, output_paths["summary"])
    # Save episode-level scores at full precision for later inference
    episode_df.to_csv(output_paths["episodes"], index=False)
    # Save rounded support diagnostics
    pec.save_report_df(support_diagnostics_df, output_paths["support_diagnostics"])
    # Save rounded weight diagnostics
    pec.save_report_df(weight_diagnostics_df, output_paths["weight_diagnostics"])
    # Save rounded residual diagnostics
    pec.save_report_df(residual_diagnostics_df, output_paths["residual_diagnostics"])
    # Save rounded clipping-sensitivity estimates
    pec.save_report_df(clipping_sensitivity_df, output_paths["clipping_sensitivity"])
    # Save current-practice episode scores at full precision
    current_episode_df.to_csv(output_paths["current_practice"], index=False)
    print_console_summary(summary_df, row_df, episode_df, output_paths, policy_manifest_path, model_type)

    bootstrap.run_bootstrap(
        "aipw", episode_df, policy_df, evaluate_policy_episodes, outdir,
        N_BOOTSTRAP, refit_panel=refit_panel, seed=BOOTSTRAP_SEED,
        refit_n_splits=REFIT_CROSSFIT_FOLDS, model_type=model_type,
        panel_name=panel_name or panel_path.stem,
    )


def main():
    for panel_name, panel_path, nuisance_root in PANEL_RUNS:
        for mode in BOOTSTRAP_MODES:
            print(f"[PANEL] {panel_name}; estimator=aipw; bootstrap={mode}", flush=True)
            run_panel_estimation(panel_path, nuisance_root.parent, NUISANCE_MODEL_TYPE, mode,
                                 panel_name=panel_name)


if __name__ == "__main__":
    main()
