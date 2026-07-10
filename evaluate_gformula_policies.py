#!/usr/bin/env python3
"""
Evaluate deterministic catheter-removal policies using a plug-in g-formula /
g-computation estimator from an estimator-agnostic policy-intervention panel.

Policy definitions are created upstream by `build_policy_intervention_panels.py`.
This script estimates model-based policy values. It does not mutate or redefine
candidate policies.

This script does not perform IPW, AIPW, DML, DR-Learner, TMLE, or LTMLE. It uses
all eligible episodes rather than only policy-adherent observed episodes. The
implementation is a plug-in g-formula estimator on the observed patient-day
grid unless later extended to full longitudinal Monte Carlo state propagation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

import policy_eval_common as pec


# =============================================================================
# Paths and constants
# =============================================================================

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
DEFAULT_OUTPUT_ROWS = "gformula_row_predictions.csv"
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

REQUIRED_POLICY_PANEL_COLS = [
    *EPISODE_KEY_COLS,
    EPISODE_ID_COL,
    "decision_row_id",
    "period_start",
    "period_end",
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "action_remove",
    "is_decision_row",
    "policy_name",
    POLICY_TYPE_COL,
    "policy_remove_day",
    "policy_action",
    "policy_action_remove",
    "policy_applicable",
    "policy_reason",
    "episode_day_since_insertion",
    *pec.RESOLVED_TIMELINE_COLUMNS,
    "policy_removal_day_extra_row_treated_as_out",
]

REQUIRED_SCORED_PANEL_COLS = [
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
    "split",
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
    "split",
    "crossfit_fold",
    "_crossfit_fold",
    "fold_id",
    "episode_end_reason",
]

DATETIME_KEY_COLS = {"inserted", "removed", "period_start", "period_end"}
LOWER_TEXT_KEY_COLS = {"catheter_state", "observed_action"}
NUMERIC_KEY_COLS = {"subject_id", "hadm_id", "stay_id", "periods_in_state", "action_remove"}

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


# =============================================================================
# Argument parsing and generic helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
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
        "--output-rows",
        default=DEFAULT_OUTPUT_ROWS,
        help=f"Row predictions output filename. Default: {DEFAULT_OUTPUT_ROWS}",
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
    pec.add_bootstrap_args(parser)
    return parser.parse_args()


def resolve_output_path(outdir: Path, name_or_path: str) -> Path:
    path = Path(name_or_path)
    return path if path.is_absolute() else outdir / path


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required {context} columns: {missing}")


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def cumulative_event_probability(probabilities: pd.Series) -> float:
    probs = pd.to_numeric(probabilities, errors="coerce")
    probs = probs.dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
    )
    return mapped.fillna(False).astype(bool)


def canonical_numeric_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if not np.isfinite(numeric):
        return "<NA>"
    if np.isclose(numeric, round(numeric), atol=1e-9):
        return str(int(round(numeric)))
    return f"{numeric:.12g}"


def canonical_key_series(series: pd.Series, column: str) -> pd.Series:
    if column in DATETIME_KEY_COLS:
        raw = series.astype("string").str.strip()
        parsed = pd.to_datetime(series, errors="coerce")
        out = raw.fillna("<NA>").astype("object")
        parsed_mask = parsed.notna()
        out.loc[parsed_mask] = parsed.loc[parsed_mask].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.loc[out.astype("string").str.lower().isin(["", "nan", "nat", "none", "<na>"])] = "<NA>"
        return out.astype("string")

    if column in LOWER_TEXT_KEY_COLS:
        out = series.astype("string").str.strip().str.lower()
        return out.fillna("<NA>")

    if column in NUMERIC_KEY_COLS:
        return series.map(canonical_numeric_value).astype("string")

    out = series.astype("string").str.strip()
    return out.fillna("<NA>")


def add_join_key_columns(df: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    require_columns(df, key_cols, "join-key")
    out = df.copy()
    join_cols = []
    for idx, col in enumerate(key_cols):
        join_col = f"__join_key_{idx}"
        out[join_col] = canonical_key_series(out[col], col)
        join_cols.append(join_col)
    return out, join_cols


def duplicate_key_examples(df: pd.DataFrame, join_cols: list[str], display_cols: list[str]) -> pd.DataFrame:
    duplicated = df.duplicated(join_cols, keep=False)
    if not duplicated.any():
        return pd.DataFrame()
    return df.loc[duplicated, display_cols].head(10)


# =============================================================================
# Loading and robust joining
# =============================================================================

def normalise_row_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "catheter_state" in df.columns:
        df["catheter_state"] = df["catheter_state"].astype("string").str.strip().str.lower()
    if "observed_action" in df.columns:
        df["observed_action"] = df["observed_action"].astype("string").str.strip().str.lower()
    if "periods_in_state" in df.columns:
        df["periods_in_state"] = pd.to_numeric(df["periods_in_state"], errors="coerce")
    if "action_remove" in df.columns:
        df["action_remove"] = pd.to_numeric(df["action_remove"], errors="coerce")
    return df


def load_policy_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Policy-intervention panel not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    require_columns(df, REQUIRED_POLICY_PANEL_COLS, "policy-panel")
    df = normalise_row_key_types(df)
    df["is_decision_row"] = coerce_bool(df["is_decision_row"])
    df["policy_applicable"] = coerce_bool(df["policy_applicable"])
    df["policy_name"] = df["policy_name"].astype("string").str.strip()
    df[POLICY_TYPE_COL] = df[POLICY_TYPE_COL].astype("string").str.strip().str.lower()
    df["policy_action"] = df["policy_action"].astype("string").str.strip().str.lower()
    df["policy_reason"] = df["policy_reason"].astype("string").str.strip().str.lower()
    df["policy_remove_day"] = pd.to_numeric(df["policy_remove_day"], errors="coerce").astype("Int64")
    df["policy_action_remove"] = pd.to_numeric(df["policy_action_remove"], errors="coerce")
    df["policy_action_resolved"] = df["policy_action_resolved"].astype("string").str.strip().str.lower()
    df["policy_action_remove_resolved"] = pd.to_numeric(
        df["policy_action_remove_resolved"],
        errors="coerce",
    )
    df["policy_catheter_state"] = df["policy_catheter_state"].astype("string").str.strip().str.lower()
    df["episode_day_since_insertion"] = pd.to_numeric(
        df["episode_day_since_insertion"],
        errors="coerce",
    )
    validate_policy_panel(df)
    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col=EPISODE_ID_COL,
        context=str(path),
    )
    return df


def load_scored_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scored nuisance panel not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    require_columns(df, REQUIRED_SCORED_PANEL_COLS, "scored-panel")
    df = normalise_row_key_types(df)
    return df


def validate_policy_panel(df: pd.DataFrame) -> None:
    if df["policy_name"].dropna().empty:
        raise ValueError("Policy panel contains no policy_name values.")
    if df["policy_remove_day"].isna().any():
        examples = df.loc[df["policy_remove_day"].isna(), ["policy_name", "decision_row_id"]].head(10)
        raise ValueError(f"Policy panel has missing policy_remove_day values. Examples:\n{examples}")


def join_scored_panel(policy_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    """Join model predictions and observed outcomes onto the policy panel."""
    policy_keyed, join_cols = add_join_key_columns(policy_df, ROW_JOIN_KEY_COLS)
    scored_keyed, _ = add_join_key_columns(scored_df, ROW_JOIN_KEY_COLS)

    scored_add_cols = [
        col
        for col in OPTIONAL_SCORED_COLS
        if col in scored_keyed.columns and col not in ROW_JOIN_KEY_COLS and col not in policy_df.columns
    ]

    duplicates = scored_keyed.duplicated(join_cols, keep=False)
    if duplicates.any():
        examples = duplicate_key_examples(scored_keyed, join_cols, ROW_JOIN_KEY_COLS)
        raise ValueError(
            "Scored panel is not unique on the natural patient-day join keys. "
            "The policy-to-scored join must be many-to-one. Examples:\n"
            f"{examples}"
        )

    before_rows = len(policy_keyed)
    right = scored_keyed[[*join_cols, *scored_add_cols]].copy()
    merged = policy_keyed.merge(
        right,
        on=join_cols,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(merged) != before_rows:
        raise ValueError(
            "Joining scored panel changed the number of policy-panel rows: "
            f"{before_rows} -> {len(merged)}"
        )

    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, ROW_JOIN_KEY_COLS + ["policy_name"]].head(10)
        raise ValueError(
            "Some policy-panel rows did not match the scored nuisance panel on "
            f"the natural keys. Examples:\n{examples}"
        )

    return merged.drop(columns=[*join_cols, "_merge"])


# =============================================================================
# Target-policy state timeline
# =============================================================================

def add_episode_day_since_insertion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    invalid = inserted.isna() | period_start.isna()
    if invalid.any():
        examples = df.loc[invalid, ["policy_name", EPISODE_ID_COL, "inserted", "period_start"]].head(10)
        raise ValueError(
            "Could not parse inserted or period_start for episode-day calculation. "
            f"Examples:\n{examples}"
        )
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def apply_horizon(df: pd.DataFrame, horizon_days: int | None) -> pd.DataFrame:
    if horizon_days is None:
        return df
    if horizon_days < 1:
        raise ValueError(f"--horizon-days must be a positive integer when supplied; got {horizon_days}")
    return df.loc[df["episode_day_since_insertion"].le(horizon_days)].copy()


# =============================================================================
# Counterfactual prediction rescoring
# =============================================================================

def predict_fold_model(fold_model: dict, features: pd.DataFrame) -> np.ndarray:
    if fold_model.get("fallback"):
        return np.full(len(features), float(fold_model["fallback_probability"]), dtype=float)
    model = fold_model.get("model")
    if model is None:
        raise ValueError("Fold model is missing and no fallback probability is available.")
    return model.predict_proba(features.to_numpy(dtype=float))[:, 1]


def fold_column(df: pd.DataFrame) -> str | None:
    for col in ["_crossfit_fold", "crossfit_fold", "fold_id"]:
        if col in df.columns:
            return col
    return None


def load_outcome_model_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def validate_model_feature_columns(df: pd.DataFrame, feature_cols: list[str], model_name: str) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot rescore {model_name}: missing feature columns required by "
            f"outcome_models.pkl: {missing[:20]}"
        )


def rescore_state_action_predictions(
    df: pd.DataFrame,
    payload: dict,
    state: str,
    outcome: str,
    output_col: str,
    target_mask: pd.Series,
    action_remove: int | None = None,
) -> pd.DataFrame:
    """Use saved cross-fitted outcome models to fill missing predictions."""
    if int(target_mask.sum()) == 0:
        return df

    models_key = "in_models" if state == "in" else "out_models"
    x_cols_key = "x_cols_in" if state == "in" else "x_cols_out"
    if models_key not in payload or outcome not in payload[models_key]:
        raise ValueError(f"outcome_models.pkl does not contain a {state} {outcome} model.")
    feature_cols = list(payload.get(x_cols_key, payload[models_key][outcome]["features"]))
    validate_model_feature_columns(df, feature_cols, f"{state}_{outcome}")

    fold_col = fold_column(df)
    if fold_col is None:
        raise ValueError(
            "Cannot rescore missing counterfactual predictions because no "
            "_crossfit_fold, crossfit_fold, or fold_id column is available."
        )

    df = df.copy()
    df[f"__rescored_{output_col}"] = False
    fold_models = payload[models_key][outcome]["fold_models"]
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
        df.loc[rows, output_col] = predict_fold_model(fold_model, features[feature_cols])
        df.loc[rows, f"__rescored_{output_col}"] = True
    return df


def fill_missing_counterfactual_predictions(
    df: pd.DataFrame,
    outcome_models_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing policy-required predictions using saved outcome models where
    possible. Existing scored-panel predictions are left unchanged.
    """
    needed_specs = [
        ("in", "cauti", "p_cauti_if_keep", "keep", 0),
        ("in", "cauti", "p_cauti_if_remove", "remove", 1),
        ("out", "cauti", "p_cauti_if_out", "out", None),
        ("out", "reinsertion", "p_reinsertion_if_out", "out", None),
        ("in", "death", "p_death_if_keep", "keep", 0),
        ("in", "death", "p_death_if_remove", "remove", 1),
        ("out", "death", "p_death_if_out", "out", None),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_keep", "keep", 0),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_remove", "remove", 1),
        ("out", "icu_exit_alive", "p_icu_exit_alive_if_out", "out", None),
        ("in", "no_event", "p_no_event_if_keep", "keep", 0),
        ("in", "no_event", "p_no_event_if_remove", "remove", 1),
        ("out", "no_event", "p_no_event_if_out", "out", None),
    ]

    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"__rescored_{col}"] = False

    masks = {
        "keep": df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(0),
        "remove": df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(1),
        "out": df["policy_catheter_state"].eq("out"),
    }
    # OUT CAUTI is only needed during the policy-implied attribution window.
    out_cauti_needed = masks["out"] & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    missing_before = {}
    for state, outcome, col, action, action_remove in needed_specs:
        needed_mask = out_cauti_needed if col == "p_cauti_if_out" else masks[action]
        missing_before[col] = int((needed_mask & df[col].isna()).sum())

    if not any(missing_before.values()):
        return df, {
            "outcome_models_used_for_rescoring": False,
            "missing_prediction_counts_before_rescoring": missing_before,
            "rescored_prediction_counts": {col: 0 for col in PREDICTION_COLUMNS},
        }

    payload = load_outcome_model_payload(outcome_models_path)
    if payload is None:
        return df, {
            "outcome_models_used_for_rescoring": False,
            "outcome_models_missing": True,
            "missing_prediction_counts_before_rescoring": missing_before,
            "rescored_prediction_counts": {col: 0 for col in PREDICTION_COLUMNS},
        }

    for state, outcome, col, action, action_remove in needed_specs:
        needed_mask = out_cauti_needed if col == "p_cauti_if_out" else masks[action]
        missing_mask = needed_mask & df[col].isna()
        if int(missing_mask.sum()) == 0:
            continue
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


def ensure_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        rescored_col = f"__rescored_{col}"
        if rescored_col not in df.columns:
            df[rescored_col] = False
    return df


def fill_missing_counterfactual_predictions_safely(
    df: pd.DataFrame,
    outcome_models_path: Path,
    allow_missing: bool,
    context: str,
) -> tuple[pd.DataFrame, dict]:
    try:
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
        return ensure_prediction_columns(df), {
            "outcome_models_used_for_rescoring": False,
            "rescoring_failed": True,
            "context": context,
            "error": str(exc),
        }


# =============================================================================
# Prediction selection and validation
# =============================================================================

def assign_prediction_from_source(
    df: pd.DataFrame,
    target_col: str,
    source_col: str,
    mask: pd.Series,
) -> None:
    df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    rescored_col = f"__rescored_{source_col}"
    if rescored_col in df.columns:
        df.loc[mask & df[rescored_col].fillna(False), "__used_rescored_prediction"] = True


def select_policy_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in UNDER_POLICY_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(0)
    remove_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_gformula"].eq(1)
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_keep", keep_rows)
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "p_recatheterisation_under_policy"] = 0.0

    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_remove", remove_rows)
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    assign_prediction_from_source(df, "p_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "p_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "p_cauti_under_policy"] = 0.0
    assign_prediction_from_source(df, "p_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    assign_prediction_from_source(df, "p_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    assign_prediction_from_source(df, "p_death_under_policy", "p_death_if_out", out_rows)
    assign_prediction_from_source(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
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
    df: pd.DataFrame,
    allow_missing_counterfactual_state_predictions: bool,
) -> None:
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


# =============================================================================
# Current-practice model-based comparator
# =============================================================================

def map_episode_ids_to_scored_panel(scored_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    if EPISODE_ID_COL in scored_df.columns:
        return scored_df.copy()

    episode_map = policy_df[[*EPISODE_KEY_COLS, EPISODE_ID_COL]].drop_duplicates()
    scored_keyed, join_cols = add_join_key_columns(scored_df, EPISODE_KEY_COLS)
    map_keyed, _ = add_join_key_columns(episode_map, EPISODE_KEY_COLS)
    duplicates = map_keyed.duplicated(join_cols, keep=False)
    if duplicates.any():
        examples = map_keyed.loc[duplicates, EPISODE_KEY_COLS + [EPISODE_ID_COL]].head(10)
        raise ValueError(
            "Policy panel maps at least one episode key to multiple catheter_episode_id values. "
            f"Examples:\n{examples}"
        )
    out = scored_keyed.merge(
        map_keyed[[*join_cols, EPISODE_ID_COL]],
        on=join_cols,
        how="left",
        validate="many_to_one",
    ).drop(columns=join_cols)
    if out[EPISODE_ID_COL].isna().any():
        examples = out.loc[out[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
        raise ValueError(f"Some scored-panel episodes could not be mapped. Examples:\n{examples}")
    return out


def build_current_practice_rows(scored_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    df = map_episode_ids_to_scored_panel(scored_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice g-formula rows")
    df = add_episode_day_since_insertion(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action_gformula"] = "out"
    df.loc[df["policy_catheter_state"].eq("in") & pd.to_numeric(df["action_remove"], errors="coerce").eq(0), "policy_action_gformula"] = "keep"
    df.loc[df["policy_catheter_state"].eq("in") & pd.to_numeric(df["action_remove"], errors="coerce").eq(1), "policy_action_gformula"] = "remove"
    df["policy_action_remove_gformula"] = np.nan
    df.loc[df["policy_action_gformula"].eq("keep"), "policy_action_remove_gformula"] = 0.0
    df.loc[df["policy_action_gformula"].eq("remove"), "policy_action_remove_gformula"] = 1.0
    df["policy_periods_in"] = np.where(df["policy_catheter_state"].eq("in"), df["periods_in_state"], np.nan)
    df["policy_periods_out"] = np.where(df["policy_catheter_state"].eq("out"), df["periods_in_state"], np.nan)
    return df


# =============================================================================
# Episode and policy-level aggregation
# =============================================================================

def add_observed_crude_episode_outcomes(episode_df: pd.DataFrame, row_df: pd.DataFrame) -> pd.DataFrame:
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


def build_episode_predictions(row_df: pd.DataFrame) -> pd.DataFrame:
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

    return order_episode_columns(base)


def order_episode_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        "split",
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
    episode_df: pd.DataFrame,
    prediction_mode: str,
) -> pd.DataFrame:
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
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
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


# =============================================================================
# Diagnostics and metadata
# =============================================================================

def prediction_bounds(df: pd.DataFrame, col: str) -> dict:
    values = pd.to_numeric(df[col], errors="coerce")
    present = values.dropna()
    return {
        f"min_{col}": float(present.min()) if len(present) else np.nan,
        f"max_{col}": float(present.max()) if len(present) else np.nan,
        f"n_{col}_below_0": int(present.lt(0).sum()) if len(present) else 0,
        f"n_{col}_above_1": int(present.gt(1).sum()) if len(present) else 0,
    }


def build_diagnostics(row_df: pd.DataFrame, episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, policy_df in row_df.groupby("policy_name", dropna=False, sort=False):
        policy_episode_df = episode_df.loc[episode_df["policy_name"].eq(policy_name)]
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
        for col in UNDER_POLICY_COLUMNS:
            bounds = prediction_bounds(policy_df, col)
            row.update(bounds)
            total_below += bounds[f"n_{col}_below_0"]
            total_above += bounds[f"n_{col}_above_1"]
        row["n_predictions_below_0"] = total_below
        row["n_predictions_above_1"] = total_above
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_payload(
    args: argparse.Namespace,
    output_paths: dict,
    row_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    rescore_metadata: dict,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
        "bootstrap": pec.bootstrap_metadata(args, row_df.columns),
        "rescoring": rescore_metadata,
        "methodological_limitations": [
            "Plug-in estimates depend on the fitted nuisance outcome models.",
            "The implementation uses the observed patient-day covariate grid.",
            "The implementation does not simulate future time-varying covariates under each policy.",
            "The implementation does not model absorbing terminal state propagation beyond observed rows.",
            "The implementation is not IPW, AIPW, DML, DR-Learner, TMLE, or LTMLE.",
        ],
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# =============================================================================
# Output column ordering
# =============================================================================

def order_row_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "decision_row_id",
        EPISODE_ID_COL,
        "subject_id",
        "hadm_id",
        "stay_id",
        "period_start",
        "period_end",
        "period_duration_days",
        "episode_day_since_insertion",
        "row_order_within_episode_day",
        "catheter_state",
        "observed_action",
        "action_remove",
        "policy_name",
        POLICY_TYPE_COL,
        "policy_remove_day",
        "policy_catheter_state",
        "policy_action_resolved",
        "policy_action_remove_resolved",
        "policy_action_gformula",
        "policy_action_remove_gformula",
        "policy_periods_in",
        "policy_periods_out",
        "policy_removal_day_extra_row_treated_as_out",
        *UNDER_POLICY_COLUMNS,
        "prediction_source",
        "prediction_status",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [
        col
        for col in df.columns
        if col not in ordered and not col.startswith("__join_key_") and not col.startswith("__rescored_")
    ]
    return df[[*ordered, *remaining]].copy()


# =============================================================================
# Console summary and main
# =============================================================================

def print_console_summary(
    args: argparse.Namespace,
    summary_df: pd.DataFrame,
    row_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    output_paths: dict,
) -> None:
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


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    output_paths = {
        "summary": resolve_output_path(args.outdir, args.output_summary),
        "episodes": resolve_output_path(args.outdir, args.output_episodes),
        "rows": resolve_output_path(args.outdir, args.output_rows),
        "diagnostics": resolve_output_path(args.outdir, args.output_diagnostics),
        "current_practice": resolve_output_path(args.outdir, args.output_current_practice),
        "metadata": resolve_output_path(args.outdir, args.output_metadata),
    }

    policy_df = load_policy_panel(args.policy_panel)
    scored_df = load_scored_panel(args.scored_panel)

    joined_df = join_scored_panel(policy_df, scored_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined g-formula policy rows")
    joined_df = apply_horizon(joined_df, args.horizon_days)
    joined_df = pec.attach_resolved_timeline_aliases(
        joined_df,
        episode_id_col=EPISODE_ID_COL,
        context=str(args.policy_panel),
        action_col="policy_action_gformula",
        action_remove_col="policy_action_remove_gformula",
    )
    joined_df, rescore_metadata = fill_missing_counterfactual_predictions_safely(
        joined_df,
        args.outcome_models,
        args.allow_missing_counterfactual_state_predictions,
        "target_policy_rows",
    )
    row_df = select_policy_predictions(joined_df)
    validate_prediction_completeness(
        row_df,
        args.allow_missing_counterfactual_state_predictions,
    )

    episode_df = build_episode_predictions(row_df)
    current_rows = build_current_practice_rows(scored_df, policy_df)
    current_rows = apply_horizon(current_rows, args.horizon_days)
    current_rows, current_rescore_metadata = fill_missing_counterfactual_predictions_safely(
        current_rows,
        args.outcome_models,
        args.allow_missing_counterfactual_state_predictions,
        "current_practice_rows",
    )
    current_rows = select_policy_predictions(current_rows)
    validate_prediction_completeness(
        current_rows,
        args.allow_missing_counterfactual_state_predictions,
    )
    rescore_metadata["current_practice_rescoring"] = current_rescore_metadata
    current_episode_df = build_episode_predictions(current_rows)
    current_episode_df = add_observed_crude_episode_outcomes(current_episode_df, current_rows)

    combined_episode_df = pd.concat([episode_df, current_episode_df], ignore_index=True, sort=False)
    summary_df = build_policy_summary(combined_episode_df, args.prediction_mode)
    diagnostics_df = build_diagnostics(pd.concat([row_df, current_rows], ignore_index=True, sort=False), combined_episode_df)

    save_df(summary_df, output_paths["summary"])
    save_df(combined_episode_df, output_paths["episodes"])
    save_df(order_row_columns(row_df), output_paths["rows"])
    save_df(diagnostics_df, output_paths["diagnostics"])
    save_df(current_episode_df, output_paths["current_practice"])
    save_json(
        metadata_payload(args, output_paths, row_df, combined_episode_df, rescore_metadata),
        output_paths["metadata"],
    )

    print_console_summary(args, summary_df, row_df, combined_episode_df, output_paths)


if __name__ == "__main__":
    main()
