#!/usr/bin/env python3
# Evaluate catheter-removal policies with plug-in g-formula estimates.


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import policy_eval_common as pec


# Paths and constants

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_POLICY_PANEL_PATH = (
    REPO_ROOT
    / "artifacts"
    / "policy_interventions"
    / "policy_intervention_panel_long.csv"
)
DEFAULT_SCORED_PANEL_PATH = (
    REPO_ROOT / "artifacts" / "nuisance_models" / "scored_panel.csv"
)
DEFAULT_OUTCOME_MODELS_PATH = (
    REPO_ROOT / "artifacts" / "nuisance_models" / "outcome_models.pkl"
)
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "policy_eval" / "gformula"

DEFAULT_OUTPUT_SUMMARY = "gformula_policy_outcomes_summary.csv"
DEFAULT_OUTPUT_EPISODES = "gformula_episode_predictions.csv"
DEFAULT_OUTPUT_DIAGNOSTICS = "gformula_diagnostics.csv"
DEFAULT_OUTPUT_CURRENT_PRACTICE = "current_practice_gformula_episode_predictions.csv"
DEFAULT_OUTPUT_METADATA = "gformula_run_metadata.json"

CURRENT_PRACTICE_LABEL = "current_practice"
ESTIMATOR_NAME = "plugin_gformula"
DEFAULT_PREDICTION_MODE = "observed_grid_plugin"
POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS = 2

EPISODE_ID_COL = "catheter_episode_id"
POLICY_TYPE_COL = "policy_type"

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
    "action_remove",
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

OPTIONAL_SCORED_COLS = [
    *PREDICTION_COLUMNS,
    "cauti_in_period",
    "reinsertion_in_period",
    "death_in_period",
    "icu_end_in_period",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "is_last_period_of_episode",
    "episode_end_reason",
    "reinsertion_time",
    "crossfit_fold",
    "_crossfit_fold",
    "fold_id",
]

OPTIONAL_FIRST_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
    "reinsertion_time",
    "crossfit_fold",
    "_crossfit_fold",
    "fold_id",
    "episode_end_reason",
]

UNDER_POLICY_COLUMNS = [
    "p_cauti_under_policy",
    "p_recatheterisation_under_policy",
    "p_death_under_policy",
    "p_icu_exit_alive_under_policy",
    "p_no_event_under_policy",
]

EPISODE_PREDICTION_SPECS = {
    "predicted_any_cauti": "p_cauti_under_policy",
    "predicted_any_recatheterisation": "p_recatheterisation_under_policy",
    "predicted_any_death": "p_death_under_policy",
    "predicted_icu_exit_alive": "p_icu_exit_alive_under_policy",
}

MISSING_COUNTERFACTUAL_MESSAGE = (
    "Missing counterfactual state/action predictions were found. This matters "
    "because plug-in g-formula uses every episode under every target policy, so "
    "a row observed in one catheter state may require predictions for another "
    "policy-implied state. Re-run nuisance scoring with complete state/action "
    "counterfactual predictions, or provide usable outcome_models.pkl for "
    "rescoring, or pass --allow-missing-counterfactual-state-predictions to "
    "continue with incomplete estimates marked as NA."
)


# Argument parsing and generic helpers

def parse_args():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate catheter-removal policies using plug-in g-formula / "
            "g-computation from an estimator-agnostic policy-intervention panel."
        )
    )
    parser.add_argument(
        "--policy-panel",
        type=Path,
        default=DEFAULT_POLICY_PANEL_PATH,
        help=f"Long-format policy-intervention panel. Default: {DEFAULT_POLICY_PANEL_PATH}",
    )
    parser.add_argument(
        "--scored-panel",
        type=Path,
        default=DEFAULT_SCORED_PANEL_PATH,
        help=f"Scored nuisance panel. Default: {DEFAULT_SCORED_PANEL_PATH}",
    )
    parser.add_argument(
        "--outcome-models",
        type=Path,
        default=DEFAULT_OUTCOME_MODELS_PATH,
        help=(
            "Saved outcome model artefact used to rescore missing "
            f"counterfactual predictions. Default: {DEFAULT_OUTCOME_MODELS_PATH}"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Output directory. Default: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--output-summary",
        default=DEFAULT_OUTPUT_SUMMARY,
        help=f"Policy summary output filename. Default: {DEFAULT_OUTPUT_SUMMARY}",
    )
    parser.add_argument(
        "--output-episodes",
        default=DEFAULT_OUTPUT_EPISODES,
        help=f"Episode predictions output filename. Default: {DEFAULT_OUTPUT_EPISODES}",
    )
    parser.add_argument(
        "--output-diagnostics",
        default=DEFAULT_OUTPUT_DIAGNOSTICS,
        help=f"Diagnostics output filename. Default: {DEFAULT_OUTPUT_DIAGNOSTICS}",
    )
    parser.add_argument(
        "--output-current-practice",
        default=DEFAULT_OUTPUT_CURRENT_PRACTICE,
        help=(
            "Current-practice model-based episode predictions output filename. "
            f"Default: {DEFAULT_OUTPUT_CURRENT_PRACTICE}"
        ),
    )
    parser.add_argument(
        "--output-metadata",
        default=DEFAULT_OUTPUT_METADATA,
        help=f"Run metadata output filename. Default: {DEFAULT_OUTPUT_METADATA}",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Optional maximum episode day since insertion to include. Default: no truncation.",
    )
    parser.add_argument(
        "--prediction-mode",
        default=DEFAULT_PREDICTION_MODE,
        help=f"Prediction mode label. Default: {DEFAULT_PREDICTION_MODE}",
    )
    parser.add_argument(
        "--allow-missing-counterfactual-state-predictions",
        action="store_true",
        help=(
            "Continue with incomplete policy predictions if counterfactual "
            "state/action predictions cannot be selected or rescored."
        ),
    )
    return parser.parse_args()


def resolve_output_path(outdir, name_or_path):
    # Resolve an output file path.
    path = Path(name_or_path)
    return path if path.is_absolute() else outdir / path


def save_df(df, path):
    # Save a data frame as CSV.
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def first_non_null(series):
    # Return the first non-missing value.
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series):
    # Return whether any binary value is present.
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def cumulative_event_probability(probabilities):
    # Calculate cumulative event probability.
    probs = pd.to_numeric(probabilities, errors="coerce")
    probs = probs.dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))


def load_policy_panel(path):
    # Load and validate the policy panel.
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    # Validate policy-panel structure.
    validate_policy_panel(df)
    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col=EPISODE_ID_COL,
        context=str(path),
    )
    return df


def load_scored_panel(path):
    # Load and validate the scored nuisance panel.
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def validate_policy_panel(df):
    # Validate policy-panel structure.
    if df["policy_name"].dropna().empty:
        raise ValueError("Policy panel contains no policy_name values.")
    if df["policy_remove_day"].isna().any():
        examples = df.loc[df["policy_remove_day"].isna(), ["policy_name", "decision_row_id"]].head(10)
        raise ValueError(f"Policy panel has missing policy_remove_day values. Examples:\n{examples}")


def join_scored_panel(policy_df, scored_df):
    scored_add_cols = [
        col
        for col in [
            *OPTIONAL_SCORED_COLS,
            *[f"__rescored_{col}" for col in PREDICTION_COLUMNS],
        ]
        if col in scored_df.columns
        and col not in ROW_JOIN_KEY_COLS
        and col not in policy_df.columns
    ]
    merged = policy_df.merge(
        scored_df[[*ROW_JOIN_KEY_COLS, *scored_add_cols]],
        on=ROW_JOIN_KEY_COLS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, ROW_JOIN_KEY_COLS + ["policy_name"]].head(10)
        raise ValueError(
            "Some policy-panel rows did not match the scored nuisance panel on "
            f"the natural keys. Examples:\n{examples}"
        )

    return merged.drop(columns="_merge")


# Target-policy state timeline

def add_episode_day_since_insertion(df):
    # Add episode day since catheter insertion.
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def apply_horizon(df, horizon_days):
    # Restrict rows to the requested time horizon.
    if horizon_days is None:
        return df
    if horizon_days < 1:
        raise ValueError(f"--horizon-days must be a positive integer when supplied; got {horizon_days}")
    return df.loc[df["episode_day_since_insertion"].le(horizon_days)].copy()


# Counterfactual prediction rescoring

def predict_fold_model(fold_model, features):
    # Predict probabilities from one fold model.
    if fold_model.get("fallback"):
        return np.full(len(features), float(fold_model["fallback_probability"]), dtype=float)
    model = fold_model.get("model")
    if model is None:
        raise ValueError("Fold model is missing and no fallback probability is available.")
    return model.predict_proba(features.to_numpy(dtype=float))[:, 1]


def fold_column(df):
    # Find the available cross-fit fold column.
    for col in ["_crossfit_fold", "crossfit_fold", "fold_id"]:
        if col in df.columns:
            return col
    return None


def load_outcome_model_payload(path):
    # Load saved outcome model artefacts.
    if not path.exists():
        return None
    return joblib.load(path)


def rescore_state_action_predictions(
    df,
    payload,
    state,
    outcome,
    output_col,
    target_mask,
    action_remove=None,
):
    # Rescore missing state-action predictions.
    if int(target_mask.sum()) == 0:
        return df

    models_key = "in_models" if state == "in" else "out_models"
    x_cols_key = "x_cols_in" if state == "in" else "x_cols_out"
    feature_cols = list(payload.get(x_cols_key, payload[models_key][outcome]["features"]))

    # Find the available cross-fit fold column.
    fold_col = fold_column(df)

    fold_models = payload[models_key][outcome]["fold_models"]
    # Predict probabilities from one fold model.
    for fold_model in fold_models:
        fold = int(fold_model["fold"])
        rows = target_mask & pd.to_numeric(df[fold_col], errors="coerce").eq(fold)
        if int(rows.sum()) == 0:
            continue
        features = df.loc[rows, feature_cols].copy()
        if state == "in":
            if action_remove is None:
                raise ValueError("IN-state rescoring requires an action_remove value.")
            action_col = payload.get("action_remove_col", "action_remove")
            features[action_col] = action_remove
        # Predict probabilities from one fold model.
        df.loc[rows, output_col] = predict_fold_model(fold_model, features[feature_cols])
        df.loc[rows, f"__rescored_{output_col}"] = True
    return df


def fill_missing_counterfactual_predictions(
    df,
    outcome_models_path,
):
    df = ensure_prediction_columns(df)
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

    # Load saved outcome model artefacts.
    payload = load_outcome_model_payload(outcome_models_path)
    if payload is None:
        return df, {
            "outcome_models_used_for_rescoring": False,
            "outcome_models_missing": True,
            "missing_prediction_counts_before_rescoring": missing_before,
            "rescored_prediction_counts": {col: 0 for col in PREDICTION_COLUMNS},
        }

    # Rescore missing state-action predictions.
    for state, outcome, col, action_remove in needed_specs:
        missing_mask = df[col].isna()
        if int(missing_mask.sum()) == 0:
            continue
        # Rescore missing state-action predictions.
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
        col: int(df.get(f"__rescored_{col}", pd.Series(False, index=df.index)).sum())
        for col in PREDICTION_COLUMNS
    }
    return df, {
        "outcome_models_used_for_rescoring": any(count > 0 for count in rescored_counts.values()),
        "outcome_models_missing": False,
        "missing_prediction_counts_before_rescoring": missing_before,
        "rescored_prediction_counts": rescored_counts,
    }


def ensure_prediction_columns(df):
    # Ensure all prediction columns exist.
    additions = {}
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            additions[col] = np.nan
        rescored_col = f"__rescored_{col}"
        if rescored_col not in df.columns:
            additions[rescored_col] = False
    if additions:
        df = pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)
    df[PREDICTION_COLUMNS] = df[PREDICTION_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return df


def fill_missing_counterfactual_predictions_safely(
    df,
    outcome_models_path,
    allow_missing,
    context,
):
    # Fill predictions and handle allowed failures.
    # Fill required counterfactual prediction columns.
    try:
        # Fill required counterfactual prediction columns.
        return fill_missing_counterfactual_predictions(df, outcome_models_path)
    except Exception as exc:
        if not allow_missing:
            raise
        print(
            "WARNING: counterfactual prediction rescoring failed for "
            f"{context}; continuing with missing predictions because "
            "--allow-missing-counterfactual-state-predictions was supplied. "
            f"Reason: {exc}",
            flush=True,
        )
        # Ensure all prediction columns exist.
        return ensure_prediction_columns(df), {
            "outcome_models_used_for_rescoring": False,
            "rescoring_failed": True,
            "context": context,
            "error": str(exc),
        }


# Prediction selection and validation

def assign_prediction_from_source(
    df,
    target_col,
    source_col,
    mask,
):
    # Copy selected prediction values into target columns.
    df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    rescored_col = f"__rescored_{source_col}"
    if rescored_col in df.columns:
        df.loc[mask & df[rescored_col].fillna(False), "__used_rescored_prediction"] = True


def select_policy_predictions(df):
    # Select predictions implied by the target policy.
    for col in UNDER_POLICY_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(0)
    remove_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(1)
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_keep", keep_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "p_recatheterisation_under_policy"] = 0.0

    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_remove", remove_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "p_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "p_cauti_under_policy"] = 0.0
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_out", out_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
    # Copy selected prediction values into target columns.
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_out", out_rows)

    missing_any = df[UNDER_POLICY_COLUMNS].isna().any(axis=1)
    invalid_any = pd.Series(False, index=df.index)
    for col in UNDER_POLICY_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_any |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))

    df["prediction_source"] = np.where(
        df["__used_rescored_prediction"],
        "scored_panel_plus_outcome_model_rescore",
        "scored_panel",
    )
    df["prediction_status"] = "complete"
    df.loc[missing_any, "prediction_status"] = "missing_prediction"
    df.loc[invalid_any, "prediction_status"] = "invalid_probability"
    return df


def validate_prediction_completeness(
    df,
    allow_missing_counterfactual_state_predictions,
):
    # Check prediction completeness and probability bounds.
    missing_counts = df.groupby("policy_name", dropna=False)[UNDER_POLICY_COLUMNS].apply(
        lambda frame: frame.isna().sum()
    )
    total_missing = int(df[UNDER_POLICY_COLUMNS].isna().sum().sum())
    invalid_rows = pd.Series(False, index=df.index)
    for col in UNDER_POLICY_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_rows |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))

    if invalid_rows.any():
        examples = df.loc[
            invalid_rows,
            ["policy_name", "decision_row_id", *UNDER_POLICY_COLUMNS],
        ].head(10)
        raise ValueError(
            "G-formula predictions must be finite probabilities between 0 and 1 "
            f"where present. Examples:\n{examples}"
        )

    if total_missing and not allow_missing_counterfactual_state_predictions:
        raise ValueError(
            f"{MISSING_COUNTERFACTUAL_MESSAGE}\nMissing prediction counts by policy:\n{missing_counts}"
        )


# Current-practice model-based comparator

def map_episode_ids_to_scored_panel(scored_df, policy_df):
    # Map episode identifiers onto scored rows.
    if EPISODE_ID_COL in scored_df.columns:
        return scored_df.copy()

    episode_map = policy_df[[*EPISODE_KEY_COLS, EPISODE_ID_COL]].drop_duplicates()
    out = scored_df.merge(
        episode_map,
        on=EPISODE_KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    if out[EPISODE_ID_COL].isna().any():
        examples = out.loc[out[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
        raise ValueError(f"Some scored-panel episodes could not be mapped. Examples:\n{examples}")
    return out


def build_current_practice_rows(scored_df, policy_df):
    # Build rows for the observed current-practice regime.
    # Map episode identifiers onto scored rows.
    df = map_episode_ids_to_scored_panel(scored_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice g-formula rows")
    # Add episode day since catheter insertion.
    df = add_episode_day_since_insertion(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action_gformula"] = "out"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["action_remove"], errors="coerce").eq(0),
        "policy_action_gformula",
    ] = "keep"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["action_remove"], errors="coerce").eq(1),
        "policy_action_gformula",
    ] = "remove"
    df["policy_action_remove_gformula"] = np.nan
    df.loc[df["policy_action_gformula"].eq("keep"), "policy_action_remove_gformula"] = 0.0
    df.loc[df["policy_action_gformula"].eq("remove"), "policy_action_remove_gformula"] = 1.0
    df["policy_periods_in"] = np.where(df["policy_catheter_state"].eq("in"), df["periods_in_state"], np.nan)
    df["policy_periods_out"] = np.where(df["policy_catheter_state"].eq("out"), df["periods_in_state"], np.nan)
    return df


# Episode and policy-level aggregation

def add_observed_crude_episode_outcomes(episode_df, row_df):
    # Add observed episode outcomes to predictions.
    row_df = pec.add_observed_icu_exit_alive_period(row_df)
    outcome_cols = [
        ("observed_any_cauti", "cauti_in_period"),
        ("observed_any_recatheterisation", "reinsertion_in_period"),
        ("observed_any_death", "death_in_period"),
        ("observed_icu_exit_alive", "observed_icu_exit_alive_in_period"),
    ]
    available = [(target, source) for target, source in outcome_cols if source in row_df.columns]
    if not available:
        return episode_df
    aggs = {source: max_binary for _, source in available}
    crude = row_df.groupby([EPISODE_ID_COL], as_index=False, dropna=False).agg(aggs)
    crude = crude.rename(columns={source: target for target, source in available})
    return episode_df.merge(crude, on=EPISODE_ID_COL, how="left")


def build_episode_predictions(row_df):
    # Collapse row predictions to episode predictions.
    row_df = row_df.copy()
    row_df["_policy_catheter_in_row_int"] = row_df["policy_catheter_state"].astype("string").str.lower().eq("in").astype(int)
    row_df["_policy_catheter_exposure_days"] = row_df["_policy_catheter_in_row_int"] * pd.to_numeric(
        row_df["period_duration_days"],
        errors="coerce",
    )
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day", EPISODE_ID_COL]
    # pandas named aggregation is clearer here than building custom apply rows.
    base = row_df.groupby(group_cols, as_index=False, dropna=False).agg(
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_policy_rows_used=("prediction_status", "size"),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
        expected_catheter_in_intervals=("_policy_catheter_in_row_int", "sum"),
        expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
    )
    base["expected_catheter_in_interval_rows"] = base["expected_catheter_in_intervals"]

    for col in OPTIONAL_FIRST_COLS:
        if col in row_df.columns and col not in group_cols:
            values = row_df.groupby(group_cols, as_index=False, dropna=False)[col].agg(first_non_null)
            base = base.merge(values, on=group_cols, how="left")

    for episode_col, row_col in EPISODE_PREDICTION_SPECS.items():
        values = row_df.groupby(group_cols, as_index=False, dropna=False)[row_col].agg(cumulative_event_probability)
        values = values.rename(columns={row_col: episode_col})
        base = base.merge(values, on=group_cols, how="left")

    # Order episode-level output columns.
    return order_episode_columns(base)


def order_episode_columns(df):
    # Order episode-level output columns.
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
        "predicted_any_cauti",
        "predicted_any_recatheterisation",
        "predicted_any_death",
        "predicted_icu_exit_alive",
        "expected_catheter_in_intervals",
        "expected_catheter_exposure_days",
        "expected_catheter_in_interval_rows",
        "n_policy_rows_used",
        "n_missing_prediction_rows",
        "prediction_complete",
        "crossfit_fold",
        "_crossfit_fold",
        "fold_id",
        "episode_end_reason",
        "reinsertion_time",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


def build_policy_summary(
    episode_df,
    prediction_mode,
):
    # Build policy-level summary estimates.
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
    for policy_values, policy_df in episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        complete_df = policy_df.loc[policy_df["prediction_complete"].astype(bool)].copy()
        row = {
            "policy_name": policy_name,
            POLICY_TYPE_COL: policy_type,
            "policy_remove_day": policy_remove_day,
            "estimator": ESTIMATOR_NAME,
            "prediction_mode": prediction_mode,
            "n_patients": int(policy_df["subject_id"].nunique()) if "subject_id" in policy_df.columns else np.nan,
            "n_episodes": int(len(policy_df)),
            "n_complete_prediction_episodes": int(len(complete_df)),
            "n_incomplete_prediction_episodes": int(len(policy_df) - len(complete_df)),
            "predicted_cauti_risk": float(complete_df["predicted_any_cauti"].mean()) if len(complete_df) else np.nan,
            "predicted_recatheterisation_risk": float(complete_df["predicted_any_recatheterisation"].mean()) if len(complete_df) else np.nan,
            "predicted_death_risk": float(complete_df["predicted_any_death"].mean()) if len(complete_df) else np.nan,
            "predicted_icu_exit_alive_risk": float(complete_df["predicted_icu_exit_alive"].mean()) if len(complete_df) else np.nan,
            "expected_mean_catheter_in_intervals": float(complete_df["expected_catheter_in_intervals"].mean()) if len(complete_df) else np.nan,
            "expected_mean_catheter_exposure_days": float(complete_df["expected_catheter_exposure_days"].mean()) if len(complete_df) else np.nan,
            "expected_mean_catheter_in_interval_rows": float(complete_df["expected_catheter_in_interval_rows"].mean()) if len(complete_df) else np.nan,
        }
        for col in [
            "predicted_cauti_risk",
            "predicted_recatheterisation_risk",
            "predicted_death_risk",
            "predicted_icu_exit_alive_risk",
        ]:
            row[f"{col}_pct"] = row[col] * 100 if pd.notna(row[col]) else np.nan
        rows.append(row)
    # Add comparisons against current practice.
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary):
    # Add comparisons against current practice.
    return pec.add_standard_comparisons(
        summary,
        baseline_label=CURRENT_PRACTICE_LABEL,
        comparison_map={
            "cauti_risk": "predicted_cauti_risk",
            "recatheterisation_risk": "predicted_recatheterisation_risk",
            "death_risk": "predicted_death_risk",
            "icu_exit_alive_risk": "predicted_icu_exit_alive_risk",
            "catheter_exposure_days": "expected_mean_catheter_exposure_days",
        },
    )


# Diagnostics and metadata

def prediction_bounds(df, col):
    # Summarise prediction bounds for one column.
    values = pd.to_numeric(df[col], errors="coerce")
    present = values.dropna()
    return {
        f"min_{col}": float(present.min()) if len(present) else np.nan,
        f"max_{col}": float(present.max()) if len(present) else np.nan,
        f"n_{col}_below_0": int(present.lt(0).sum()) if len(present) else 0,
        f"n_{col}_above_1": int(present.gt(1).sum()) if len(present) else 0,
    }


def build_diagnostics(row_df, episode_df):
    # Build diagnostic rows for policy outputs.
    rows = []
    # Return the first non-missing value.
    for policy_name, policy_df in row_df.groupby("policy_name", dropna=False, sort=False):
        policy_episode_df = episode_df.loc[episode_df["policy_name"].eq(policy_name)]
        # Return the first non-missing value.
        policy_remove_day = first_non_null(policy_df["policy_remove_day"]) if "policy_remove_day" in policy_df.columns else pd.NA
        numeric_remove_day = pd.to_numeric(pd.Series([policy_remove_day]), errors="coerce").iloc[0]
        if pd.notna(numeric_remove_day) and "expected_catheter_in_intervals" in policy_episode_df.columns:
            too_many_policy_in = policy_episode_df["expected_catheter_in_intervals"].gt(numeric_remove_day)
        else:
            too_many_policy_in = pd.Series(False, index=policy_episode_df.index)
        remove_rows_by_episode = (
            policy_df["policy_action_gformula"].eq("remove")
            .groupby(policy_df[EPISODE_ID_COL], sort=False)
            .sum()
        )
        if pd.notna(numeric_remove_day):
            reaches_policy_removal_day = (
                pd.to_numeric(policy_df["episode_day_since_insertion"], errors="coerce")
                .eq(numeric_remove_day)
                .groupby(policy_df[EPISODE_ID_COL], sort=False)
                .max()
            )
            n_episodes_reaching_policy_removal_day = int(reaches_policy_removal_day.sum())
        else:
            n_episodes_reaching_policy_removal_day = 0
        n_policy_remove_rows = int(policy_df["policy_action_gformula"].eq("remove").sum())
        fixed_day_policy = pd.notna(numeric_remove_day) and policy_name != CURRENT_PRACTICE_LABEL
        policy_remove_row_shortfall = (
            int(n_episodes_reaching_policy_removal_day - n_policy_remove_rows)
            if fixed_day_policy
            else pd.NA
        )
        # Return the first non-missing value.
        row = {
            "policy_name": policy_name,
            POLICY_TYPE_COL: first_non_null(policy_df[POLICY_TYPE_COL]) if POLICY_TYPE_COL in policy_df.columns else pd.NA,
            "policy_remove_day": policy_remove_day,
            "n_rows": int(len(policy_df)),
            "n_episodes": int(policy_df[EPISODE_ID_COL].nunique()),
            "n_patients": int(policy_df["subject_id"].nunique()) if "subject_id" in policy_df.columns else np.nan,
            "n_policy_in_rows": int(policy_df["policy_catheter_state"].eq("in").sum()),
            "n_policy_remove_rows": n_policy_remove_rows,
            "n_episodes_reaching_policy_removal_day": n_episodes_reaching_policy_removal_day,
            "n_policy_remove_row_shortfall_vs_reached_episodes": policy_remove_row_shortfall,
            "n_episodes_with_more_than_one_remove_row": int(remove_rows_by_episode.gt(1).sum()),
            "n_policy_removal_day_extra_rows_treated_as_out": int(
                policy_df.get(
                    "policy_removal_day_extra_row_treated_as_out",
                    pd.Series(0, index=policy_df.index),
                ).sum()
            ),
            "n_policy_out_rows": int(policy_df["policy_catheter_state"].eq("out").sum()),
            "n_duplicate_policy_episode_day_rows": pec.duplicate_episode_day_count(
                policy_df,
                ["policy_name", EPISODE_ID_COL],
            ),
            "n_episodes_with_more_policy_in_intervals_than_expected": int(too_many_policy_in.sum()),
            "n_invalid_period_duration_rows": 0,
            "n_long_period_duration_rows": int(policy_df.get("period_duration_long_flag", pd.Series(0, index=policy_df.index)).sum()),
            "n_missing_cauti_predictions": int(policy_df["p_cauti_under_policy"].isna().sum()),
            "n_missing_recatheterisation_predictions": int(policy_df["p_recatheterisation_under_policy"].isna().sum()),
            "n_missing_death_predictions": int(policy_df["p_death_under_policy"].isna().sum()),
            "n_missing_icu_exit_predictions": int(policy_df["p_icu_exit_alive_under_policy"].isna().sum()),
            "n_missing_no_event_predictions": int(policy_df["p_no_event_under_policy"].isna().sum()),
            "n_complete_prediction_episodes": int(policy_episode_df["prediction_complete"].astype(bool).sum()) if len(policy_episode_df) else 0,
            "n_incomplete_prediction_episodes": int((~policy_episode_df["prediction_complete"].astype(bool)).sum()) if len(policy_episode_df) else 0,
            "mean_expected_catheter_in_intervals": float(policy_episode_df["expected_catheter_in_intervals"].mean()) if len(policy_episode_df) else np.nan,
            "mean_expected_catheter_exposure_days": float(policy_episode_df["expected_catheter_exposure_days"].mean()) if len(policy_episode_df) else np.nan,
            "mean_expected_catheter_in_interval_rows": float(policy_episode_df["expected_catheter_in_interval_rows"].mean()) if len(policy_episode_df) else np.nan,
            "uses_policy_matching": False,
            "uses_ipw_weights": False,
            "uses_observed_grid": True,
            "full_longitudinal_covariate_simulation": False,
            "covariate_propagation_method": DEFAULT_PREDICTION_MODE,
        }
        total_below = 0
        total_above = 0
        # Summarise prediction bounds for one column.
        for col in UNDER_POLICY_COLUMNS:
            # Summarise prediction bounds for one column.
            bounds = prediction_bounds(policy_df, col)
            row.update(bounds)
            total_below += bounds[f"n_{col}_below_0"]
            total_above += bounds[f"n_{col}_above_1"]
        row["n_predictions_below_0"] = total_below
        row["n_predictions_above_1"] = total_above
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_payload(
    args,
    output_paths,
    row_df,
    episode_df,
    rescore_metadata,
):
    # Build run metadata.
    return {
        "estimator": ESTIMATOR_NAME,
        "prediction_mode": args.prediction_mode,
        "current_practice_comparator_type": "model_based_plugin_observed_regime",
        "horizon_days": args.horizon_days,
        "input_paths": {
            "policy_panel": str(args.policy_panel),
            "scored_panel": str(args.scored_panel),
            "outcome_models": str(args.outcome_models),
        },
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "required_prediction_columns": PREDICTION_COLUMNS,
        "allow_missing_counterfactual_state_predictions": bool(
            args.allow_missing_counterfactual_state_predictions
        ),
        "number_of_policies": int(row_df["policy_name"].nunique()),
        "number_of_episodes": int(row_df[EPISODE_ID_COL].nunique()),
        "number_of_complete_prediction_episodes": int(episode_df["prediction_complete"].astype(bool).sum()),
        "target_policy_timing_source": pec.TARGET_POLICY_TIMING_SOURCE,
        "target_policy_timeline_helper": pec.TARGET_POLICY_TIMELINE_HELPER,
        "target_policy_timeline_semantics": pec.TARGET_POLICY_TIMELINE_SEMANTICS,
        "duration_semantics": {
            "period_duration_days": "period_end - period_start in days",
            "catheter_exposure_days": "sum of period_duration_days where policy_catheter_state == in",
            "max_reasonable_period_duration_days": pec.MAX_REASONABLE_PERIOD_DURATION_DAYS,
            "n_long_period_duration_rows": int(row_df.get("period_duration_long_flag", pd.Series(0)).sum()),
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
        },
        "icu_exit_alive_definition": (
            "max(icu_end_in_period == 1 and death_in_period != 1); death takes "
            "precedence when death and ICU exit occur in the same interval"
        ),
        "rescoring": rescore_metadata,
        "methodological_limitations": [
            "Plug-in estimates depend on the fitted nuisance outcome models.",
            "The implementation uses the observed patient-day covariate grid.",
            "The implementation does not simulate future time-varying covariates under each policy.",
            "The implementation does not model absorbing terminal state propagation beyond observed rows.",
            "The implementation is not IPW, AIPW, DML, DR-Learner, TMLE, or LTMLE.",
        ],
    }


# Console summary and main

def print_console_summary(
    args,
    summary_df,
    row_df,
    episode_df,
    output_paths,
):
    # Print a concise run summary.
    print()
    print("--- G-FORMULA POLICY EVALUATION COMPLETE ---")
    print(f"Policy-intervention panel: {args.policy_panel}")
    print(f"Scored nuisance panel: {args.scored_panel}")
    print(f"Number of policies: {row_df['policy_name'].nunique():,}")
    print(f"Number of patients: {row_df['subject_id'].nunique():,}")
    print(f"Number of episodes: {row_df[EPISODE_ID_COL].nunique():,}")
    print(f"Complete prediction episodes: {int(episode_df['prediction_complete'].astype(bool).sum()):,}")
    print()
    display_cols = [
        "policy_name",
        "policy_remove_day",
        "predicted_cauti_risk_pct",
        "predicted_recatheterisation_risk_pct",
        "expected_mean_catheter_exposure_days",
    ]
    available = [col for col in display_cols if col in summary_df.columns]
    print(summary_df[available].to_string(index=False))
    print()
    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")


def main():
    # Run the script workflow.
    # Parse command-line arguments.
    args = parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    # Resolve an output file path.
    output_paths = {
        "summary": resolve_output_path(args.outdir, args.output_summary),
        "episodes": resolve_output_path(args.outdir, args.output_episodes),
        "diagnostics": resolve_output_path(args.outdir, args.output_diagnostics),
        "current_practice": resolve_output_path(args.outdir, args.output_current_practice),
        "metadata": resolve_output_path(args.outdir, args.output_metadata),
    }

    # Load and validate the policy panel.
    policy_df = load_policy_panel(args.policy_panel)
    # Load and validate the scored nuisance panel.
    scored_df = load_scored_panel(args.scored_panel)
    scored_df, rescore_metadata = fill_missing_counterfactual_predictions_safely(
        scored_df,
        args.outcome_models,
        args.allow_missing_counterfactual_state_predictions,
        "scored_panel",
    )
    model_feature_cols = [
        "episode_index",
        *pec.baseline_model_feature_columns(scored_df.columns),
    ]
    scored_df.drop(
        columns=[col for col in model_feature_cols if col in scored_df.columns],
        inplace=True,
    )

    # Join nuisance scores to policy rows.
    joined_df = join_scored_panel(policy_df, scored_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined g-formula policy rows")
    # Restrict rows to the requested time horizon.
    joined_df = apply_horizon(joined_df, args.horizon_days)
    joined_df = pec.attach_resolved_timeline_aliases(
        joined_df,
        action_col="policy_action_gformula",
        action_remove_col="policy_action_remove_gformula",
    )
    # Select predictions implied by the target policy.
    row_df = select_policy_predictions(joined_df)
    # Check prediction completeness and probability bounds.
    validate_prediction_completeness(
        row_df,
        args.allow_missing_counterfactual_state_predictions,
    )

    # Collapse row predictions to episode predictions.
    episode_df = build_episode_predictions(row_df)
    # Build rows for the observed current-practice regime.
    current_rows = build_current_practice_rows(scored_df, policy_df)
    # Restrict rows to the requested time horizon.
    current_rows = apply_horizon(current_rows, args.horizon_days)
    # Select predictions implied by the target policy.
    current_rows = select_policy_predictions(current_rows)
    # Check prediction completeness and probability bounds.
    validate_prediction_completeness(
        current_rows,
        args.allow_missing_counterfactual_state_predictions,
    )
    # Collapse row predictions to episode predictions.
    current_episode_df = build_episode_predictions(current_rows)
    # Add observed episode outcomes to predictions.
    current_episode_df = add_observed_crude_episode_outcomes(current_episode_df, current_rows)

    combined_episode_df = pd.concat([episode_df, current_episode_df], ignore_index=True, sort=False)
    # Build policy-level summary estimates.
    summary_df = build_policy_summary(combined_episode_df, args.prediction_mode)
    # Build diagnostic rows for policy outputs.
    diagnostics_df = build_diagnostics(
        pd.concat([row_df, current_rows], ignore_index=True, sort=False),
        combined_episode_df,
    )

    # Save rounded policy-level report outputs.
    pec.save_report_df(summary_df, output_paths["summary"])
    # Save episode-level data at full precision for later inference.
    save_df(combined_episode_df, output_paths["episodes"])
    # Save rounded diagnostics.
    pec.save_report_df(diagnostics_df, output_paths["diagnostics"])
    # Save current-practice episode data at full precision.
    save_df(current_episode_df, output_paths["current_practice"])
    pec.save_json(
        metadata_payload(args, output_paths, row_df, combined_episode_df, rescore_metadata),
        output_paths["metadata"],
    )

    # Print a concise run summary.
    print_console_summary(args, summary_df, row_df, combined_episode_df, output_paths)


# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
