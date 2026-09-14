#!/usr/bin/env python3
# Evaluate catheter-removal policies with AIPW estimates


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import policy_eval_common as pec
from panel_run_config import add_panel_argument, resolve_panel_run


# Paths and constants

REPO_ROOT = Path(__file__).resolve().parent
NUISANCE_MODEL_TYPE = "xgboost"
NUISANCE_MODEL_DIR = (
    REPO_ROOT / "artefacts" / "nuisance_models" / NUISANCE_MODEL_TYPE
)

POLICY_PANEL_PATH = (
    REPO_ROOT
    / "artefacts"
    / "policy_interventions"
    / "policy_intervention_panel_long.csv"
)
NUISANCE_PREDICTIONS_PATH = NUISANCE_MODEL_DIR / "nuisance_predictions.csv"
OUTCOME_MODELS_PATH = NUISANCE_MODEL_DIR / "outcome_models.pkl"
OUTDIR = REPO_ROOT / "artefacts" / "policy_eval" / "aipw"

OUTPUT_PATHS = {
    "summary": OUTDIR / "aipw_policy_outcomes_summary.csv",
    "episodes": OUTDIR / "aipw_policy_episode_scores.csv",
    "support_diagnostics": OUTDIR / "aipw_policy_support_diagnostics.csv",
    "weight_diagnostics": OUTDIR / "aipw_weight_diagnostics.csv",
    "residual_diagnostics": OUTDIR / "aipw_residual_diagnostics.csv",
    "clipping_sensitivity": OUTDIR / "aipw_clipping_sensitivity.csv",
    "current_practice": OUTDIR / "current_practice_aipw_episode_scores.csv",
    "metadata": OUTDIR / "aipw_run_metadata.json",
}


def configure_panel_run(panel_name):
    # Resolve mutually consistent policy, nuisance, model, and evaluator paths
    global NUISANCE_MODEL_DIR, POLICY_PANEL_PATH, NUISANCE_PREDICTIONS_PATH
    global OUTCOME_MODELS_PATH, OUTDIR, OUTPUT_PATHS
    paths = resolve_panel_run(REPO_ROOT, panel_name)
    NUISANCE_MODEL_DIR = (
        paths.artefact_root / "nuisance_models" / NUISANCE_MODEL_TYPE
    )
    POLICY_PANEL_PATH = (
        paths.artefact_root
        / "policy_interventions"
        / "policy_intervention_panel_long.csv"
    )
    NUISANCE_PREDICTIONS_PATH = NUISANCE_MODEL_DIR / "nuisance_predictions.csv"
    OUTCOME_MODELS_PATH = NUISANCE_MODEL_DIR / "outcome_models.pkl"
    OUTDIR = paths.artefact_root / "policy_eval" / "aipw"
    OUTPUT_PATHS = {
        "summary": OUTDIR / "aipw_policy_outcomes_summary.csv",
        "episodes": OUTDIR / "aipw_policy_episode_scores.csv",
        "support_diagnostics": OUTDIR / "aipw_policy_support_diagnostics.csv",
        "weight_diagnostics": OUTDIR / "aipw_weight_diagnostics.csv",
        "residual_diagnostics": OUTDIR / "aipw_residual_diagnostics.csv",
        "clipping_sensitivity": OUTDIR / "aipw_clipping_sensitivity.csv",
        "current_practice": OUTDIR / "current_practice_aipw_episode_scores.csv",
        "metadata": OUTDIR / "aipw_run_metadata.json",
    }
    return paths

CLIP_LOWER = 0.01
CLIP_UPPER = 0.99
RESIDUAL_NORMALISATION = "hajek"
CROSSFIT_FOLD_COL = "_crossfit_fold"

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

PREDICTION_COLUMNS = [
    "p_cauti_if_keep",
    "p_cauti_if_remove",
    "p_cauti_if_out",
    "p_reinsertion_if_out",
    "p_death_if_keep",
    "p_death_if_remove",
    "p_death_if_out",
    "p_icu_exit_alive_if_keep",
    "p_icu_exit_alive_if_remove",
    "p_icu_exit_alive_if_out",
    "p_no_event_if_keep",
    "p_no_event_if_remove",
    "p_no_event_if_out",
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


def first_non_null(series):
    # Return the first non-missing value
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series):
    # Return whether any binary value is present
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def cumulative_event_probability(probabilities):
    # Calculate cumulative event probability
    probs = pd.to_numeric(probabilities, errors="coerce").dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))


def valid_weight_series(weights):
    # Return positive finite weights
    weights = pd.to_numeric(weights, errors="coerce")
    return weights[weights.notna() & np.isfinite(weights) & weights.gt(0)]


def effective_sample_size(weights):
    # Calculate the effective sample size
    # Return positive finite weights
    weights = valid_weight_series(weights)
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


def bound_probability_estimate(value):
    # Bound a final probability-scale estimate while retaining missing values
    if pd.isna(value):
        return np.nan
    return float(np.clip(float(value), 0.0, 1.0))


def load_policy_panel(path):
    # Load and validate the policy panel
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if df["policy_name"].dropna().empty:
        raise ValueError("Policy panel contains no policy_name values.")
    if df["policy_remove_day"].isna().any():
        examples = df.loc[df["policy_remove_day"].isna(), ["policy_name", "decision_row_id"]].head(10)
        raise ValueError(f"Policy panel has missing policy_remove_day values. Examples:\n{examples}")
    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col=EPISODE_ID_COL,
        context=str(path),
    )
    return df


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

def add_episode_day_since_insertion(df):
    # Add episode day since catheter insertion
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def predict_fold_model(fold_model, features):
    # Predict probabilities from one fold model
    if fold_model["fallback"]:
        return np.full(len(features), float(fold_model["fallback_probability"]), dtype=float)

    retained_feature_cols = list(fold_model["retained_feature_cols"])
    missing_features = sorted(set(retained_feature_cols) - set(features.columns))
    if missing_features:
        raise ValueError(
            "Rescoring data are missing features retained by the fitted fold model: "
            f"{missing_features}"
        )
    return fold_model["model"].predict_proba(
        features.loc[:, retained_feature_cols].to_numpy(dtype=float)
    )[:, 1]


def rescore_state_action_predictions(
    df,
    payload,
    state,
    outcome,
    output_col,
    target_mask,
    action_remove=None,
):
    # Rescore missing state-action predictions
    if int(target_mask.sum()) == 0:
        return df

    models_key = "in_models" if state == "in" else "out_models"
    x_cols_key = "x_cols_in" if state == "in" else "x_cols_out"
    feature_cols = list(payload[x_cols_key])

    fold_models = payload[models_key][outcome]["fold_models"]
    # Predict probabilities from one fold model
    for fold_model in fold_models:
        fold = int(fold_model["fold"])
        rows = target_mask & pd.to_numeric(
            df[CROSSFIT_FOLD_COL],
            errors="coerce",
        ).eq(fold)
        if int(rows.sum()) == 0:
            continue
        features = df.loc[rows, feature_cols].copy()
        if state == "in":
            action_col = payload["action_remove_col"]
            features[action_col] = action_remove
        # Predict probabilities from one fold model
        df.loc[rows, output_col] = predict_fold_model(fold_model, features[feature_cols])
        df.loc[rows, f"__rescored_{output_col}"] = True
    return df


def fill_missing_counterfactual_predictions(
    df,
    outcome_models_path,
):
    # Standardise prediction columns and track rescored values
    df[PREDICTION_COLUMNS] = df[PREDICTION_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    rescored_cols = {
        f"__rescored_{col}": False
        for col in PREDICTION_COLUMNS
    }
    df = pd.concat([df, pd.DataFrame(rescored_cols, index=df.index)], axis=1)

    # Define every state-action prediction needed downstream
    needed_specs = [
        ("in", "cauti", "p_cauti_if_keep", 0),
        ("in", "cauti", "p_cauti_if_remove", 1),
        ("out", "cauti", "p_cauti_if_out", None),
        ("out", "reinsertion", "p_reinsertion_if_out", None),
        ("in", "death", "p_death_if_keep", 0),
        ("in", "death", "p_death_if_remove", 1),
        ("out", "death", "p_death_if_out", None),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_keep", 0),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_remove", 1),
        ("out", "icu_exit_alive", "p_icu_exit_alive_if_out", None),
        ("in", "no_event", "p_no_event_if_keep", 0),
        ("in", "no_event", "p_no_event_if_remove", 1),
        ("out", "no_event", "p_no_event_if_out", None),
    ]
    missing_before = {col: int(df[col].isna().sum()) for _, _, col, _ in needed_specs}
    if not any(missing_before.values()):
        return df, {
            "outcome_models_used_for_rescoring": False,
            "missing_prediction_counts_before_rescoring": missing_before,
            "rescored_prediction_counts": {col: 0 for col in PREDICTION_COLUMNS},
        }

    # Load saved outcome model artefacts
    payload = joblib.load(outcome_models_path)

    # Rescore missing state-action predictions
    for state, outcome, col, action_remove in needed_specs:
        missing_mask = df[col].isna()
        if int(missing_mask.sum()) == 0:
            continue
        # Rescore missing state-action predictions
        df = rescore_state_action_predictions(
            df,
            payload,
            state,
            outcome,
            col,
            missing_mask,
            action_remove,
        )

    rescored_counts = {
        col: int(df[f"__rescored_{col}"].sum())
        for col in PREDICTION_COLUMNS
    }
    return df, {
        "outcome_models_used_for_rescoring": any(count > 0 for count in rescored_counts.values()),
        "missing_prediction_counts_before_rescoring": missing_before,
        "rescored_prediction_counts": rescored_counts,
    }


def assign_mu_from_source(df, target_col, source_col, mask):
    # Copy selected prediction values into mean columns
    df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    rescored_col = f"__rescored_{source_col}"
    df.loc[mask & df[rescored_col], "__used_rescored_prediction"] = True


def select_policy_predictions(df):
    # Select predictions implied by the target policy
    for col in MU_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df[
        "policy_action_remove_resolved"
    ].eq(0)
    remove_rows = df["policy_catheter_state"].eq("in") & df[
        "policy_action_remove_resolved"
    ].eq(1)
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_keep", keep_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "mu_recatheterisation_under_policy"] = 0.0

    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_remove", remove_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "mu_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "mu_cauti_under_policy"] = 0.0
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_out", out_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
    # Copy selected prediction values into mean columns
    assign_mu_from_source(df, "mu_no_event_under_policy", "p_no_event_if_out", out_rows)

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
    remove_rows = applicable & df["policy_action_remove"].eq(1)
    keep_rows = applicable & df["policy_action_remove"].eq(0)
    df.loc[remove_rows, "policy_support"] = df.loc[remove_rows, "p_remove_obs"]
    df.loc[keep_rows, "policy_support"] = df.loc[keep_rows, "p_keep_obs"]
    support = pd.to_numeric(df["policy_support"], errors="coerce")

    invalid_support = applicable & (support.isna() | ~np.isfinite(support) | support.lt(0) | support.gt(1))
    if invalid_support.any():
        examples = df.loc[
            invalid_support,
            ["policy_name", "decision_row_id", "policy_action_remove", "policy_support"],
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


def add_observed_outcomes_to_rows(df):
    # Validate and copy the mutually exclusive ICU-exit-alive outcome
    df = pec.add_observed_icu_exit_alive_period(df)
    df["_icu_exit_alive_period"] = df["observed_icu_exit_alive_in_period"]
    return df


def build_policy_episode_scores(df):
    # Collapse row scores to policy-episode scores
    # Add observed outcomes to rows
    df = add_observed_outcomes_to_rows(df)
    df["_applicable_int"] = df["policy_applicable"].astype(int)
    df["_matched_applicable_int"] = (
        df["policy_applicable"] & df["policy_matches_observed_action_today"].eq(1)
    ).astype(int)
    df["_catheter_in_row_int"] = df["catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_policy_catheter_in_row_int"] = df["policy_catheter_state"].eq("in").astype(int)
    df["_policy_remove_row_int"] = df[
        "policy_action_resolved"
    ].eq("remove").astype(int)
    df["_observed_catheter_exposure_days"] = df["_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    df["_policy_catheter_exposure_days"] = df["_policy_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    day = pd.to_numeric(df["episode_day_since_insertion"], errors="coerce")
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
        & pd.to_numeric(df["policy_action_remove"], errors="coerce").eq(1)
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
        plugin_expected_catheter_in_intervals=("_policy_catheter_in_row_int", "sum"),
        plugin_expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
        observed_catheter_in_intervals=("_catheter_in_row_int", "sum"),
        observed_catheter_exposure_days=("_observed_catheter_exposure_days", "sum"),
        max_episode_day_since_insertion=("episode_day_since_insertion", "max"),
        episode_has_terminal_event=("_terminal_period", "max"),
        episode_observed_removed_before_policy_day=("_observed_removed_before_policy_day", "max"),
        episode_failed_to_remove_on_policy_day=("_failed_to_remove_on_policy_day", "max"),
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
    )
    episode_df["plugin_expected_catheter_in_interval_rows"] = episode_df["plugin_expected_catheter_in_intervals"]
    episode_df["observed_catheter_in_interval_rows"] = episode_df["observed_catheter_in_intervals"]
    episode_df["episode_has_more_than_one_policy_remove_row"] = (
        episode_df["n_policy_remove_rows"].gt(1)
    ).astype(int)
    max_day = pd.to_numeric(episode_df["max_episode_day_since_insertion"], errors="coerce")
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
        )[col].agg(first_non_null)
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
        observed_values = df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[period_col].agg(max_binary)
        observed_values = observed_values.rename(
            columns={period_col: spec["observed"]}
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
    df = add_episode_day_since_insertion(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action_resolved"] = "out"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(0),
        "policy_action_resolved",
    ] = "keep"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(1),
        "policy_action_resolved",
    ] = "remove"
    df["policy_action_remove_resolved"] = np.nan
    df.loc[
        df["policy_action_resolved"].eq("keep"),
        "policy_action_remove_resolved",
    ] = 0.0
    df.loc[
        df["policy_action_resolved"].eq("remove"),
        "policy_action_remove_resolved",
    ] = 1.0
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
    df = add_observed_outcomes_to_rows(current_rows)
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
        plugin_expected_catheter_in_intervals=("_policy_catheter_in_row_int", "sum"),
        plugin_expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
        observed_catheter_in_intervals=("_catheter_in_row_int", "sum"),
        observed_catheter_exposure_days=("_observed_catheter_exposure_days", "sum"),
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
    )
    episode_df["plugin_expected_catheter_in_interval_rows"] = episode_df["plugin_expected_catheter_in_intervals"]
    episode_df["observed_catheter_in_interval_rows"] = episode_df["observed_catheter_in_intervals"]
    for col in EPISODE_FIRST_COLS:
        values = df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[col].agg(first_non_null)
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
        observed_values = df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[period_col].agg(max_binary)
        observed_values = observed_values.rename(
            columns={period_col: spec["observed"]}
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
    residual_weight = (
        pd.to_numeric(complete_df[weight_col], errors="coerce").fillna(0.0)
        if weight_col in complete_df.columns
        else pd.Series(dtype=float)
    )
    residual_weight_all = (
        pd.to_numeric(policy_df[weight_col], errors="coerce").fillna(0.0)
        if weight_col in policy_df.columns
        else pd.Series(dtype=float)
    )
    # Return positive finite weights
    positive_residual_weights = valid_weight_series(residual_weight_all)
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
    if fixed_day_policy and "max_episode_day_since_insertion" in policy_df.columns:
        n_episodes_reaching_policy_removal_day = int(
            pd.to_numeric(policy_df["max_episode_day_since_insertion"], errors="coerce")
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
        "n_patients": int(policy_df["subject_id"].nunique()) if "subject_id" in policy_df.columns else np.nan,
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
            ) if "plugin_expected_catheter_in_interval_rows" in complete_df.columns and len(complete_df) else np.nan
            row["aipw_mean_catheter_in_interval_rows"] = row[
                "plugin_expected_mean_catheter_in_interval_rows"
            ]
        else:
            bounded_ht_value = bound_probability_estimate(ht_value)
            bounded_hajek_value = bound_probability_estimate(hajek_value)
            bounded_selected_value = bound_probability_estimate(selected_value)
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
        residual_weights = valid_weight_series(policy_df[RESIDUAL_WEIGHT_COL])
        # Return positive finite weights
        raw_episode_weights = valid_weight_series(policy_df[WEIGHT_COL])
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
        positive_residual_weights = valid_weight_series(residual_weight_all)
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
            plugin = pd.to_numeric(complete_df.get(plugin_col), errors="coerce")
            observed = pd.to_numeric(complete_df.get(observed_col), errors="coerce")
            residual = pd.to_numeric(complete_df.get(residual_col), errors="coerce")
            weighted_residual = pd.to_numeric(complete_df.get(weighted_residual_col), errors="coerce")
            ht_score = pd.to_numeric(complete_df.get(ht_score_col), errors="coerce")
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
            bounded_ht_value = bound_probability_estimate(ht_value) if is_probability else ht_value
            bounded_hajek_value = bound_probability_estimate(hajek_value) if is_probability else hajek_value
            bounded_selected_value = (
                bound_probability_estimate(selected_value) if is_probability else selected_value
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
        valid_default = valid_weight_series(default_weights)
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
            valid_weights = valid_weight_series(weights)
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
                "aipw_cauti_risk": summary_row.get("aipw_cauti_risk"),
                "aipw_recatheterisation_risk": summary_row.get("aipw_recatheterisation_risk"),
                "aipw_death_risk": summary_row.get("aipw_death_risk"),
                "aipw_icu_exit_alive_risk": summary_row.get("aipw_icu_exit_alive_risk"),
                "aipw_mean_catheter_exposure_days": summary_row.get("aipw_mean_catheter_exposure_days"),
                "aipw_mean_catheter_in_interval_rows": summary_row.get("aipw_mean_catheter_in_interval_rows"),
            })
    return pd.DataFrame(rows)


# Output ordering and metadata

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
        "plugin_expected_catheter_in_intervals",
        "observed_catheter_in_intervals",
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


def metadata_payload(
    output_paths,
    row_df,
    episode_df,
    rescore_metadata,
):
    # Build run metadata
    policies_with_zero_adherent = (
        episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)]
        .groupby("policy_name")["episode_adherent_to_policy"]
        .sum()
        .loc[lambda s: s == 0]
        .index.tolist()
    )
    return {
        "estimator": ESTIMATOR_NAME,
        "nuisance_model_type": NUISANCE_MODEL_TYPE,
        "residual_normalisation": RESIDUAL_NORMALISATION,
        "current_practice_comparator_type": "aipw_observed_regime",
        "clipping_bounds": {"clip_lower": CLIP_LOWER, "clip_upper": CLIP_UPPER},
        "probability_estimate_bounding": {
            "method": "clip_final_aggregate_to_unit_interval",
            "bounds": [0.0, 1.0],
            "scope": "probability outcomes only; catheter exposure is not bounded",
            "unbounded_values_retained": True,
            "diagnostic_suffix": "unbounded_estimate",
            "bounding_flag_suffix": "was_bounded",
        },
        "input_paths": {
            "policy_panel": str(POLICY_PANEL_PATH),
            "nuisance_predictions": str(NUISANCE_PREDICTIONS_PATH),
            "outcome_models": str(OUTCOME_MODELS_PATH),
        },
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "required_nuisance_columns": ["p_remove_obs", "p_keep_obs", *PREDICTION_COLUMNS],
        "number_of_policies": int(row_df["policy_name"].nunique()),
        "number_of_patients": int(row_df["subject_id"].nunique()),
        "number_of_episodes": int(row_df[EPISODE_ID_COL].nunique()),
        "number_of_complete_prediction_episodes": int(episode_df["prediction_complete"].astype(bool).sum()),
        "number_of_policies_with_zero_adherent_episodes": int(len(policies_with_zero_adherent)),
        "policies_with_zero_adherent_episodes": policies_with_zero_adherent,
        "target_policy_timing_source": pec.TARGET_POLICY_TIMING_SOURCE,
        "target_policy_timeline_helper": pec.TARGET_POLICY_TIMELINE_HELPER,
        "target_policy_timeline_semantics": pec.TARGET_POLICY_TIMELINE_SEMANTICS,
        "duration_semantics": {
            "period_duration_days": "period_end - period_start in days",
            "catheter_exposure_days": "sum of period_duration_days where policy_catheter_state == in",
            "max_reasonable_period_duration_days": pec.MAX_REASONABLE_PERIOD_DURATION_DAYS,
            "n_long_period_duration_rows": int(
                row_df["period_duration_long_flag"].sum()
            ),
        },
        "catheter_count_semantics": {
            "expected_catheter_in_intervals": (
                "count of policy-implied catheter-in interval rows on the observed grid; "
                "not necessarily one row per patient-day because transition days may be "
                "split into in and out intervals"
            ),
            "expected_catheter_exposure_days": (
                "sum of period_duration_days where policy_catheter_state == in; "
                "preferred exposure measure for interpretation"
            ),
            "plugin_expected_catheter_in_intervals": (
                "AIPW plug-in count using the same interval-row semantics as "
                "expected_catheter_in_intervals"
            ),
        },
        "icu_exit_alive_definition": (
            "max(icu_exit_alive_in_period == 1); death and ICU exit alive are "
            "mutually exclusive terminal events in the source panel"
        ),
        "overlap_flag_thresholds": {
            "low_adherence_threshold": pec.LOW_ADHERENCE_THRESHOLD,
            "low_ess_min": pec.LOW_ESS_MIN,
            "low_ess_fraction": pec.LOW_ESS_FRACTION,
            "low_support_pct_below_0_05_threshold": pec.LOW_SUPPORT_PCT_BELOW_005_THRESHOLD,
            "extreme_weight_p99_threshold": pec.EXTREME_WEIGHT_P99_THRESHOLD,
            "extreme_weight_max_threshold": pec.EXTREME_WEIGHT_MAX_THRESHOLD,
        },
        "rescoring": rescore_metadata,
        "methodological_limitations": [
            "AIPW estimates depend on cross-fitted propensity and outcome nuisance predictions.",
            "The plug-in component uses the observed patient-day covariate grid.",
            "The residual correction is available only through observed policy-adherent trajectories.",
            "This is not pure IPW, pure g-formula, Policy-DML, DR-Learner, TMLE, or LTMLE.",
            "No composite clinical policy score is calculated.",
        ],
    }


# Main

def print_console_summary(summary_df, row_df, episode_df, output_paths):
    # Print a concise run summary
    target_episode_df = episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)]
    print()
    print("--- AIPW POLICY EVALUATION COMPLETE ---")
    print(f"Policy-intervention panel: {POLICY_PANEL_PATH}")
    print(f"Nuisance model type: {NUISANCE_MODEL_TYPE}")
    print(f"Nuisance predictions: {NUISANCE_PREDICTIONS_PATH}")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-day policies with AIPW."
    )
    add_panel_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_panel_run(args.panel)

    # Create the output directory
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Load and validate the policy panel
    policy_df = load_policy_panel(POLICY_PANEL_PATH)
    # Load and validate the nuisance predictions
    nuisance_df = load_nuisance_predictions(NUISANCE_PREDICTIONS_PATH)
    nuisance_df, rescore_metadata = fill_missing_counterfactual_predictions(
        nuisance_df,
        OUTCOME_MODELS_PATH,
    )
    model_feature_cols = [
        "episode_index",
        *pec.baseline_model_feature_columns(nuisance_df.columns),
    ]
    nuisance_df.drop(
        columns=model_feature_cols,
        inplace=True,
    )
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
    pec.save_report_df(summary_df, OUTPUT_PATHS["summary"])
    # Save episode-level scores at full precision for later inference
    episode_df.to_csv(OUTPUT_PATHS["episodes"], index=False)
    # Save rounded support diagnostics
    pec.save_report_df(support_diagnostics_df, OUTPUT_PATHS["support_diagnostics"])
    # Save rounded weight diagnostics
    pec.save_report_df(weight_diagnostics_df, OUTPUT_PATHS["weight_diagnostics"])
    # Save rounded residual diagnostics
    pec.save_report_df(residual_diagnostics_df, OUTPUT_PATHS["residual_diagnostics"])
    # Save rounded clipping-sensitivity estimates
    pec.save_report_df(clipping_sensitivity_df, OUTPUT_PATHS["clipping_sensitivity"])
    # Save current-practice episode scores at full precision
    current_episode_df.to_csv(OUTPUT_PATHS["current_practice"], index=False)
    pec.save_json(
        metadata_payload(OUTPUT_PATHS, row_df, episode_df, rescore_metadata),
        OUTPUT_PATHS["metadata"],
    )
    # Print a concise run summary
    print_console_summary(summary_df, row_df, episode_df, OUTPUT_PATHS)


if __name__ == "__main__":
    main()
