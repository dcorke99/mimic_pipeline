#!/usr/bin/env python3
# Evaluate catheter-removal policies with plug-in g-formula estimates


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import policy_eval_common as pec


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
SCORED_PANEL_PATH = NUISANCE_MODEL_DIR / "scored_panel.csv"
OUTCOME_MODELS_PATH = NUISANCE_MODEL_DIR / "outcome_models.pkl"
OUTDIR = REPO_ROOT / "artefacts" / "policy_eval" / "gformula"

OUTPUT_PATHS = {
    "summary": OUTDIR / "gformula_policy_outcomes_summary.csv",
    "episodes": OUTDIR / "gformula_episode_predictions.csv",
    "diagnostics": OUTDIR / "gformula_diagnostics.csv",
    "current_practice": OUTDIR / "current_practice_gformula_episode_predictions.csv",
    "metadata": OUTDIR / "gformula_run_metadata.json",
}

CURRENT_PRACTICE_LABEL = "current_practice"
ESTIMATOR_NAME = "plugin_gformula"
PREDICTION_MODE = "observed_grid_plugin"
POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS = 2
CROSSFIT_FOLD_COL = "_crossfit_fold"

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

SCORED_COLS = [
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
    "counterfactual predictions or provide outcome_models.pkl for rescoring."
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
    probs = pd.to_numeric(probabilities, errors="coerce")
    probs = probs.dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))


def load_policy_panel(path):
    # Load and validate the policy panel
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    # Validate policy-panel structure
    validate_policy_panel(df)
    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col=EPISODE_ID_COL,
        context=str(path),
    )
    return df


def load_scored_panel(path):
    # Load and validate the scored nuisance panel
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def validate_policy_panel(df):
    # Validate policy-panel structure
    if df["policy_name"].dropna().empty:
        raise ValueError("Policy panel contains no policy_name values.")
    if df["policy_remove_day"].isna().any():
        examples = df.loc[df["policy_remove_day"].isna(), ["policy_name", "decision_row_id"]].head(10)
        raise ValueError(f"Policy panel has missing policy_remove_day values. Examples:\n{examples}")


def join_scored_panel(policy_df, scored_df):
    scored_add_cols = [
        col
        for col in [
            *SCORED_COLS,
            *[f"__rescored_{col}" for col in PREDICTION_COLUMNS],
        ]
        if col not in ROW_JOIN_KEY_COLS
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
    # Add episode day since catheter insertion
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


# Counterfactual prediction rescoring

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


# Prediction selection and validation

def assign_prediction_from_source(
    df,
    target_col,
    source_col,
    mask,
):
    # Copy selected prediction values into target columns
    df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    rescored_col = f"__rescored_{source_col}"
    df.loc[mask & df[rescored_col], "__used_rescored_prediction"] = True


def select_policy_predictions(df):
    # Select predictions implied by the target policy
    for col in UNDER_POLICY_COLUMNS:
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

    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_keep", keep_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "p_recatheterisation_under_policy"] = 0.0

    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_remove", remove_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "p_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "p_cauti_under_policy"] = 0.0
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_out", out_rows)
    # Copy selected prediction values into target columns
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
    # Copy selected prediction values into target columns
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


def validate_prediction_completeness(df):
    # Check prediction completeness and probability bounds
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

    if total_missing:
        raise ValueError(
            f"{MISSING_COUNTERFACTUAL_MESSAGE}\nMissing prediction counts by policy:\n{missing_counts}"
        )


# Current-practice model-based comparator

def map_episode_ids_to_scored_panel(scored_df, policy_df):
    # Map episode identifiers onto scored rows
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
    # Build rows for the observed current-practice regime
    # Map episode identifiers onto scored rows
    df = map_episode_ids_to_scored_panel(scored_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice g-formula rows")
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
    return df


# Episode and policy-level aggregation

def add_observed_crude_episode_outcomes(episode_df, row_df):
    # Add observed episode outcomes to predictions
    row_df = pec.add_observed_icu_exit_alive_period(row_df)
    outcome_cols = [
        ("observed_any_cauti", "cauti_in_period"),
        ("observed_any_recatheterisation", "reinsertion_in_period"),
        ("observed_any_death", "death_in_period"),
        ("observed_icu_exit_alive", "observed_icu_exit_alive_in_period"),
    ]
    aggs = {source: max_binary for _, source in outcome_cols}
    crude = row_df.groupby([EPISODE_ID_COL], as_index=False, dropna=False).agg(aggs)
    crude = crude.rename(columns={source: target for target, source in outcome_cols})
    return episode_df.merge(crude, on=EPISODE_ID_COL, how="left")


def build_episode_predictions(row_df):
    # Collapse row predictions to episode predictions
    row_df = row_df.copy()
    row_df["_policy_catheter_in_row_int"] = row_df["policy_catheter_state"].astype("string").str.lower().eq("in").astype(int)
    row_df["_policy_catheter_exposure_days"] = row_df["_policy_catheter_in_row_int"] * pd.to_numeric(
        row_df["period_duration_days"],
        errors="coerce",
    )
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day", EPISODE_ID_COL]
    # pandas named aggregation is clearer here than building custom apply rows
    base = row_df.groupby(group_cols, as_index=False, dropna=False).agg(
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_policy_rows_used=("prediction_status", "size"),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
        expected_catheter_in_intervals=("_policy_catheter_in_row_int", "sum"),
        expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
    )
    base["expected_catheter_in_interval_rows"] = base["expected_catheter_in_intervals"]

    for col in EPISODE_FIRST_COLS:
        values = row_df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[col].agg(first_non_null)
        base = base.merge(values, on=group_cols, how="left")

    for episode_col, row_col in EPISODE_PREDICTION_SPECS.items():
        values = row_df.groupby(group_cols, as_index=False, dropna=False)[row_col].agg(cumulative_event_probability)
        values = values.rename(columns={row_col: episode_col})
        base = base.merge(values, on=group_cols, how="left")

    # Order episode-level output columns
    return order_episode_columns(base)


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
        "_crossfit_fold",
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
    # Build policy-level summary estimates
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
            "n_patients": int(policy_df["subject_id"].nunique()),
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
    # Add comparisons against current practice
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary):
    # Add comparisons against current practice
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
    # Summarise prediction bounds for one column
    values = pd.to_numeric(df[col], errors="coerce")
    present = values.dropna()
    return {
        f"min_{col}": float(present.min()) if len(present) else np.nan,
        f"max_{col}": float(present.max()) if len(present) else np.nan,
        f"n_{col}_below_0": int(present.lt(0).sum()) if len(present) else 0,
        f"n_{col}_above_1": int(present.gt(1).sum()) if len(present) else 0,
    }


def build_diagnostics(row_df, episode_df):
    # Build diagnostic rows for policy outputs
    rows = []
    # Return the first non-missing value
    for policy_name, policy_df in row_df.groupby("policy_name", dropna=False, sort=False):
        policy_episode_df = episode_df.loc[episode_df["policy_name"].eq(policy_name)]
        # Return the first non-missing value
        policy_remove_day = first_non_null(policy_df["policy_remove_day"])
        numeric_remove_day = pd.to_numeric(pd.Series([policy_remove_day]), errors="coerce").iloc[0]
        if pd.notna(numeric_remove_day):
            too_many_policy_in = policy_episode_df["expected_catheter_in_intervals"].gt(numeric_remove_day)
        else:
            too_many_policy_in = pd.Series(False, index=policy_episode_df.index)
        remove_rows_by_episode = (
            policy_df["policy_action_resolved"].eq("remove")
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
        n_policy_remove_rows = int(
            policy_df["policy_action_resolved"].eq("remove").sum()
        )
        fixed_day_policy = pd.notna(numeric_remove_day) and policy_name != CURRENT_PRACTICE_LABEL
        policy_remove_row_shortfall = (
            int(n_episodes_reaching_policy_removal_day - n_policy_remove_rows)
            if fixed_day_policy
            else pd.NA
        )
        # Return the first non-missing value
        row = {
            "policy_name": policy_name,
            POLICY_TYPE_COL: first_non_null(policy_df[POLICY_TYPE_COL]),
            "policy_remove_day": policy_remove_day,
            "n_rows": int(len(policy_df)),
            "n_episodes": int(policy_df[EPISODE_ID_COL].nunique()),
            "n_patients": int(policy_df["subject_id"].nunique()),
            "n_policy_in_rows": int(policy_df["policy_catheter_state"].eq("in").sum()),
            "n_policy_remove_rows": n_policy_remove_rows,
            "n_episodes_reaching_policy_removal_day": n_episodes_reaching_policy_removal_day,
            "n_policy_remove_row_shortfall_vs_reached_episodes": policy_remove_row_shortfall,
            "n_episodes_with_more_than_one_remove_row": int(remove_rows_by_episode.gt(1).sum()),
            "n_policy_removal_day_extra_rows_treated_as_out": int(
                policy_df["policy_removal_day_extra_row_treated_as_out"]
                .fillna(0)
                .sum()
            ),
            "n_policy_out_rows": int(policy_df["policy_catheter_state"].eq("out").sum()),
            "n_duplicate_policy_episode_day_rows": pec.duplicate_episode_day_count(
                policy_df,
                ["policy_name", EPISODE_ID_COL],
            ),
            "n_episodes_with_more_policy_in_intervals_than_expected": int(too_many_policy_in.sum()),
            "n_invalid_period_duration_rows": 0,
            "n_long_period_duration_rows": int(
                policy_df["period_duration_long_flag"].sum()
            ),
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
            "covariate_propagation_method": PREDICTION_MODE,
        }
        total_below = 0
        total_above = 0
        # Summarise prediction bounds for one column
        for col in UNDER_POLICY_COLUMNS:
            # Summarise prediction bounds for one column
            bounds = prediction_bounds(policy_df, col)
            row.update(bounds)
            total_below += bounds[f"n_{col}_below_0"]
            total_above += bounds[f"n_{col}_above_1"]
        row["n_predictions_below_0"] = total_below
        row["n_predictions_above_1"] = total_above
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_payload(
    output_paths,
    row_df,
    episode_df,
    rescore_metadata,
):
    # Build run metadata
    return {
        "estimator": ESTIMATOR_NAME,
        "nuisance_model_type": NUISANCE_MODEL_TYPE,
        "prediction_mode": PREDICTION_MODE,
        "current_practice_comparator_type": "model_based_plugin_observed_regime",
        "input_paths": {
            "policy_panel": str(POLICY_PANEL_PATH),
            "scored_panel": str(SCORED_PANEL_PATH),
            "outcome_models": str(OUTCOME_MODELS_PATH),
        },
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "required_prediction_columns": PREDICTION_COLUMNS,
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
        },
        "icu_exit_alive_definition": (
            "max(icu_exit_alive_in_period == 1); death and ICU exit alive are "
            "mutually exclusive terminal events in the source panel"
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
    summary_df,
    row_df,
    episode_df,
    output_paths,
):
    # Print a concise run summary
    print()
    print("--- G-FORMULA POLICY EVALUATION COMPLETE ---")
    print(f"Policy-intervention panel: {POLICY_PANEL_PATH}")
    print(f"Nuisance model type: {NUISANCE_MODEL_TYPE}")
    print(f"Scored nuisance panel: {SCORED_PANEL_PATH}")
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
    print(summary_df[display_cols].to_string(index=False))
    print()
    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")


def main():
    # Create the output directory
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Load and validate the policy panel
    policy_df = load_policy_panel(POLICY_PANEL_PATH)
    # Load and validate the scored nuisance panel
    scored_df = load_scored_panel(SCORED_PANEL_PATH)
    scored_df, rescore_metadata = fill_missing_counterfactual_predictions(
        scored_df,
        OUTCOME_MODELS_PATH,
    )
    model_feature_cols = [
        "episode_index",
        *pec.baseline_model_feature_columns(scored_df.columns),
    ]
    scored_df.drop(
        columns=model_feature_cols,
        inplace=True,
    )

    # Join nuisance scores to policy rows
    joined_df = join_scored_panel(policy_df, scored_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined g-formula policy rows")
    # Select predictions implied by the target policy
    row_df = select_policy_predictions(joined_df)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(row_df)

    # Collapse row predictions to episode predictions
    episode_df = build_episode_predictions(row_df)
    # Build rows for the observed current-practice regime
    current_rows = build_current_practice_rows(scored_df, policy_df)
    # Select predictions implied by the target policy
    current_rows = select_policy_predictions(current_rows)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(current_rows)
    # Collapse row predictions to episode predictions
    current_episode_df = build_episode_predictions(current_rows)
    # Add observed episode outcomes to predictions
    current_episode_df = add_observed_crude_episode_outcomes(current_episode_df, current_rows)

    combined_episode_df = pd.concat([episode_df, current_episode_df], ignore_index=True, sort=False)
    # Build policy-level summary estimates
    summary_df = build_policy_summary(combined_episode_df, PREDICTION_MODE)
    # Build diagnostic rows for policy outputs
    diagnostics_df = build_diagnostics(
        pd.concat([row_df, current_rows], ignore_index=True, sort=False),
        combined_episode_df,
    )

    # Save rounded policy-level report outputs
    pec.save_report_df(summary_df, OUTPUT_PATHS["summary"])
    # Save episode-level data at full precision for later inference
    combined_episode_df.to_csv(OUTPUT_PATHS["episodes"], index=False)
    # Save rounded diagnostics
    pec.save_report_df(diagnostics_df, OUTPUT_PATHS["diagnostics"])
    # Save current-practice episode data at full precision
    current_episode_df.to_csv(OUTPUT_PATHS["current_practice"], index=False)
    pec.save_json(
        metadata_payload(OUTPUT_PATHS, row_df, combined_episode_df, rescore_metadata),
        OUTPUT_PATHS["metadata"],
    )

    # Print a concise run summary
    print_console_summary(summary_df, row_df, combined_episode_df, OUTPUT_PATHS)


if __name__ == "__main__":
    main()
