#!/usr/bin/env python3
"""
Evaluate deterministic catheter-removal policies using an AIPW / doubly robust
estimator from an estimator-agnostic policy-intervention panel.

Policy definitions are created upstream by `build_policy_intervention_panels.py`.
Cross-fitted nuisance predictions are created upstream by `fit_nuisance_models.py`.
This script calculates AIPW-specific plug-in, support, adherence,
residual-correction and policy-value quantities.

It does not perform pure IPW, pure g-formula, Policy-DML, DR-Learner, TMLE, or
LTMLE. It does not mutate or redefine candidate policies. It uses all eligible
episodes for the plug-in component and adds a weighted residual correction for
observed policy-adherent trajectories.
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
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "policy_eval" / "aipw"

DEFAULT_OUTPUT_SUMMARY = "aipw_policy_outcomes_summary.csv"
DEFAULT_OUTPUT_EPISODES = "aipw_policy_episode_scores.csv"
DEFAULT_OUTPUT_ROWS = "aipw_row_scores.csv"
DEFAULT_OUTPUT_SUPPORT_DIAGNOSTICS = "aipw_policy_support_diagnostics.csv"
DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS = "aipw_weight_diagnostics.csv"
DEFAULT_OUTPUT_RESIDUAL_DIAGNOSTICS = "aipw_residual_diagnostics.csv"
DEFAULT_OUTPUT_CLIPPING_SENSITIVITY = "aipw_clipping_sensitivity.csv"
DEFAULT_OUTPUT_CURRENT_PRACTICE = "current_practice_aipw_episode_scores.csv"
DEFAULT_OUTPUT_METADATA = "aipw_run_metadata.json"

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
    "policy_matches_observed_action_today",
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
    "p_remove_obs",
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
    "p_keep_obs",
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
        "period_outcome": "icu_end_in_period",
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
    "provide usable outcome_models.pkl for rescoring, or pass "
    "--allow-missing-counterfactual-state-predictions to continue with affected "
    "policy-outcome estimates marked as incomplete."
)


# =============================================================================
# Argument parsing and generic helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic catheter-removal policies using an AIPW / "
            "doubly robust estimator from an estimator-agnostic policy panel."
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
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-episodes", default=DEFAULT_OUTPUT_EPISODES)
    parser.add_argument("--output-rows", default=DEFAULT_OUTPUT_ROWS)
    parser.add_argument("--output-support-diagnostics", default=DEFAULT_OUTPUT_SUPPORT_DIAGNOSTICS)
    parser.add_argument("--output-weight-diagnostics", default=DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS)
    parser.add_argument("--output-residual-diagnostics", default=DEFAULT_OUTPUT_RESIDUAL_DIAGNOSTICS)
    parser.add_argument("--output-clipping-sensitivity", default=DEFAULT_OUTPUT_CLIPPING_SENSITIVITY)
    parser.add_argument("--output-current-practice", default=DEFAULT_OUTPUT_CURRENT_PRACTICE)
    parser.add_argument("--output-metadata", default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=0.99)
    parser.add_argument(
        "--residual-normalisation",
        choices=["ht", "hajek"],
        default="hajek",
        help="Generic aipw_* columns use HT or Hájek residual correction. Default: hajek.",
    )
    parser.add_argument(
        "--allow-missing-counterfactual-state-predictions",
        action="store_true",
        help="Continue with incomplete estimates if some counterfactual predictions are unavailable.",
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


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def cumulative_event_probability(probabilities: pd.Series) -> float:
    probs = pd.to_numeric(probabilities, errors="coerce").dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))


def valid_weight_series(weights: pd.Series) -> pd.Series:
    weights = pd.to_numeric(weights, errors="coerce")
    return weights[weights.notna() & np.isfinite(weights) & weights.gt(0)]


def effective_sample_size(weights: pd.Series) -> float:
    weights = valid_weight_series(weights)
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


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
        return series.astype("string").str.strip().str.lower().fillna("<NA>")
    if column in NUMERIC_KEY_COLS:
        return series.map(canonical_numeric_value).astype("string")
    return series.astype("string").str.strip().fillna("<NA>")


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
    df["policy_matches_observed_action_today"] = pd.to_numeric(
        df["policy_matches_observed_action_today"],
        errors="coerce",
    )
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


def load_scored_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scored nuisance panel not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    require_columns(df, REQUIRED_SCORED_PANEL_COLS, "scored-panel")
    df = normalise_row_key_types(df)
    df["p_remove_obs"] = pd.to_numeric(df["p_remove_obs"], errors="coerce")
    if "p_keep_obs" not in df.columns:
        df["p_keep_obs"] = 1.0 - df["p_remove_obs"]
    else:
        df["p_keep_obs"] = pd.to_numeric(df["p_keep_obs"], errors="coerce")
    return df


def join_scored_panel(policy_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    policy_keyed, join_cols = add_join_key_columns(policy_df, ROW_JOIN_KEY_COLS)
    scored_keyed, _ = add_join_key_columns(scored_df, ROW_JOIN_KEY_COLS)

    scored_add_cols = [
        col
        for col in OPTIONAL_SCORED_COLS
        if col in scored_keyed.columns and col not in ROW_JOIN_KEY_COLS and col not in policy_df.columns
    ]
    for col in ["p_remove_obs", "p_keep_obs"]:
        if col not in scored_add_cols:
            scored_add_cols.insert(0, col)

    duplicates = scored_keyed.duplicated(join_cols, keep=False)
    if duplicates.any():
        examples = duplicate_key_examples(scored_keyed, join_cols, ROW_JOIN_KEY_COLS)
        raise ValueError(
            "Scored panel is not unique on the natural patient-day join keys. "
            "The policy-to-scored join must be many-to-one. Examples:\n"
            f"{examples}"
        )

    before_rows = len(policy_keyed)
    merged = policy_keyed.merge(
        scored_keyed[[*join_cols, *scored_add_cols]],
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
            "Some policy-panel rows did not match the scored nuisance panel. "
            f"Examples:\n{examples}"
        )
    return merged.drop(columns=[*join_cols, "_merge"])


# =============================================================================
# Policy timeline and nuisance prediction selection
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


def fold_column(df: pd.DataFrame) -> str | None:
    for col in ["_crossfit_fold", "crossfit_fold", "fold_id"]:
        if col in df.columns:
            return col
    return None


def predict_fold_model(fold_model: dict, features: pd.DataFrame) -> np.ndarray:
    if fold_model.get("fallback"):
        return np.full(len(features), float(fold_model["fallback_probability"]), dtype=float)
    model = fold_model.get("model")
    if model is None:
        raise ValueError("Fold model is missing and no fallback probability is available.")
    return model.predict_proba(features.to_numpy(dtype=float))[:, 1]


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


def ensure_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        rescored_col = f"__rescored_{col}"
        if rescored_col not in df.columns:
            df[rescored_col] = False
    return df


def fill_missing_counterfactual_predictions(
    df: pd.DataFrame,
    outcome_models_path: Path,
) -> tuple[pd.DataFrame, dict]:
    df = ensure_prediction_columns(df)
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
    masks = {
        "keep": df["policy_catheter_state"].eq("in") & df["policy_action_remove_aipw"].eq(0),
        "remove": df["policy_catheter_state"].eq("in") & df["policy_action_remove_aipw"].eq(1),
        "out": df["policy_catheter_state"].eq("out"),
    }
    out_cauti_needed = masks["out"] & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )
    missing_before = {}
    for _, _, col, action, _ in needed_specs:
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


def assign_mu_from_source(df: pd.DataFrame, target_col: str, source_col: str, mask: pd.Series) -> None:
    df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    rescored_col = f"__rescored_{source_col}"
    if rescored_col in df.columns:
        df.loc[mask & df[rescored_col].fillna(False), "__used_rescored_prediction"] = True


def select_policy_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MU_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_aipw"].eq(0)
    remove_rows = df["policy_catheter_state"].eq("in") & df["policy_action_remove_aipw"].eq(1)
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_keep", keep_rows)
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    assign_mu_from_source(df, "mu_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "mu_recatheterisation_under_policy"] = 0.0

    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_remove", remove_rows)
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    assign_mu_from_source(df, "mu_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "mu_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "mu_cauti_under_policy"] = 0.0
    assign_mu_from_source(df, "mu_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    assign_mu_from_source(df, "mu_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    assign_mu_from_source(df, "mu_death_under_policy", "p_death_if_out", out_rows)
    assign_mu_from_source(df, "mu_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
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


def validate_prediction_completeness(df: pd.DataFrame, allow_missing: bool) -> None:
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
    if total_missing and not allow_missing:
        missing_counts = df.groupby("policy_name", dropna=False)[MU_COLUMNS].apply(
            lambda frame: frame.isna().sum()
        )
        raise ValueError(
            f"{MISSING_COUNTERFACTUAL_MESSAGE}\nMissing prediction counts by policy:\n{missing_counts}"
        )


# =============================================================================
# AIPW support, adherence and episode-level scores
# =============================================================================

def validate_clip_bounds(clip_lower: float, clip_upper: float) -> None:
    if not (0 < clip_lower < clip_upper <= 1):
        raise ValueError(
            "Support clipping bounds must satisfy 0 < clip_lower < clip_upper <= 1. "
            f"Received clip_lower={clip_lower}, clip_upper={clip_upper}."
        )


def add_support_and_adherence(df: pd.DataFrame, clip_lower: float, clip_upper: float) -> pd.DataFrame:
    validate_clip_bounds(clip_lower, clip_upper)
    df = df.copy()
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


def product_components_by_episode(df: pd.DataFrame, component_col: str) -> pd.DataFrame:
    group_cols = ["policy_name", "policy_remove_day", EPISODE_ID_COL]
    return (
        df.groupby(group_cols, dropna=False, sort=False)[component_col]
        .prod(min_count=1)
        .reset_index()
    )


def add_observed_outcomes_to_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "death_in_period" in df.columns:
        death = pd.to_numeric(df["death_in_period"], errors="coerce").fillna(0).astype(int)
    else:
        death = pd.Series(0, index=df.index, dtype=int)
    if "icu_end_in_period" in df.columns:
        icu_exit = pd.to_numeric(df["icu_end_in_period"], errors="coerce").fillna(0).astype(int)
        df["_icu_exit_alive_period"] = ((icu_exit == 1) & (death == 0)).astype(int)
    return df


def build_policy_episode_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_observed_outcomes_to_rows(df)
    df = df.copy()
    df["_applicable_int"] = df["policy_applicable"].astype(int)
    df["_matched_applicable_int"] = (
        df["policy_applicable"] & df["policy_matches_observed_action_today"].eq(1)
    ).astype(int)
    df["_catheter_in_row_int"] = df["catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_policy_catheter_in_row_int"] = df["policy_catheter_state"].eq("in").astype(int)
    df["_policy_remove_row_int"] = df["policy_action_aipw"].eq("remove").astype(int)
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
    death_period = (
        pd.to_numeric(df["death_in_period"], errors="coerce").fillna(0)
        if "death_in_period" in df.columns
        else pd.Series(0, index=df.index)
    )
    icu_period = (
        pd.to_numeric(df["icu_end_in_period"], errors="coerce").fillna(0)
        if "icu_end_in_period" in df.columns
        else pd.Series(0, index=df.index)
    )
    terminal_period = death_period.eq(1) | icu_period.eq(1)
    df["_terminal_period"] = terminal_period.astype(int)
    df["_observed_removed_before_policy_day"] = (
        day.lt(remove_day)
        & df["is_decision_row"].astype(bool)
        & pd.to_numeric(df["action_remove"], errors="coerce").eq(1)
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

    for col in OPTIONAL_FIRST_COLS:
        if col in df.columns and col not in group_cols:
            values = df.groupby(group_cols, as_index=False, dropna=False)[col].agg(first_non_null)
            episode_df = episode_df.merge(values, on=group_cols, how="left")

    for outcome_name, spec in OUTCOME_SPECS.items():
        if outcome_name == "catheter_exposure_days":
            continue
        values = df.groupby(group_cols, as_index=False, dropna=False)[spec["mu"]].agg(
            cumulative_event_probability
        )
        values = values.rename(columns={spec["mu"]: spec["plugin"]})
        episode_df = episode_df.merge(values, on=group_cols, how="left")

        period_col = spec["period_outcome"]
        if outcome_name == "icu_exit_alive" and "_icu_exit_alive_period" in df.columns:
            observed_values = df.groupby(group_cols, as_index=False, dropna=False)["_icu_exit_alive_period"].agg(max_binary)
            observed_values = observed_values.rename(columns={"_icu_exit_alive_period": spec["observed"]})
            episode_df = episode_df.merge(observed_values, on=group_cols, how="left")
        elif period_col in df.columns:
            observed_values = df.groupby(group_cols, as_index=False, dropna=False)[period_col].agg(max_binary)
            observed_values = observed_values.rename(columns={period_col: spec["observed"]})
            episode_df = episode_df.merge(observed_values, on=group_cols, how="left")
        else:
            episode_df[spec["observed"]] = np.nan

    weight_products = product_components_by_episode(df, "row_ipw_component")
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

    return order_episode_columns(episode_df), df


# =============================================================================
# Current-practice comparator
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
        raise ValueError(f"Episode key maps to multiple catheter_episode_id values. Examples:\n{examples}")
    out = scored_keyed.merge(
        map_keyed[[*join_cols, EPISODE_ID_COL]],
        on=join_cols,
        how="left",
        validate="many_to_one",
    ).drop(columns=join_cols)
    if out[EPISODE_ID_COL].isna().any():
        examples = out.loc[out[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
        raise ValueError(f"Some scored-panel rows could not be mapped to episodes. Examples:\n{examples}")
    return out


def build_current_practice_rows(scored_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    df = map_episode_ids_to_scored_panel(scored_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice AIPW rows")
    df = add_episode_day_since_insertion(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action_aipw"] = "out"
    df.loc[df["policy_catheter_state"].eq("in") & pd.to_numeric(df["action_remove"], errors="coerce").eq(0), "policy_action_aipw"] = "keep"
    df.loc[df["policy_catheter_state"].eq("in") & pd.to_numeric(df["action_remove"], errors="coerce").eq(1), "policy_action_aipw"] = "remove"
    df["policy_action_remove_aipw"] = np.nan
    df.loc[df["policy_action_aipw"].eq("keep"), "policy_action_remove_aipw"] = 0.0
    df.loc[df["policy_action_aipw"].eq("remove"), "policy_action_remove_aipw"] = 1.0
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


def build_current_practice_episode_scores(current_rows: pd.DataFrame) -> pd.DataFrame:
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
    for col in OPTIONAL_FIRST_COLS:
        if col in df.columns and col not in group_cols:
            values = df.groupby(group_cols, as_index=False, dropna=False)[col].agg(first_non_null)
            episode_df = episode_df.merge(values, on=group_cols, how="left")
    for outcome_name, spec in OUTCOME_SPECS.items():
        if outcome_name == "catheter_exposure_days":
            continue
        values = df.groupby(group_cols, as_index=False, dropna=False)[spec["mu"]].agg(
            cumulative_event_probability
        )
        values = values.rename(columns={spec["mu"]: spec["plugin"]})
        episode_df = episode_df.merge(values, on=group_cols, how="left")
        if outcome_name == "icu_exit_alive" and "_icu_exit_alive_period" in df.columns:
            observed_values = df.groupby(group_cols, as_index=False, dropna=False)["_icu_exit_alive_period"].agg(max_binary)
            observed_values = observed_values.rename(columns={"_icu_exit_alive_period": spec["observed"]})
            episode_df = episode_df.merge(observed_values, on=group_cols, how="left")
        elif spec["period_outcome"] in df.columns:
            observed_values = df.groupby(group_cols, as_index=False, dropna=False)[spec["period_outcome"]].agg(max_binary)
            observed_values = observed_values.rename(columns={spec["period_outcome"]: spec["observed"]})
            episode_df = episode_df.merge(observed_values, on=group_cols, how="left")
        else:
            episode_df[spec["observed"]] = np.nan

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
    return order_episode_columns(episode_df)


# =============================================================================
# Summaries, diagnostics and sensitivity
# =============================================================================

def policy_summary_row(
    policy_df: pd.DataFrame,
    policy_name,
    policy_type,
    policy_remove_day,
    residual_normalisation: str,
    weight_col: str = RESIDUAL_WEIGHT_COL,
) -> dict:
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
    positive_residual_weights = valid_weight_series(residual_weight_all)
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
        "effective_sample_size": effective_sample_size(residual_weight_all),
        "residual_correction_effective_sample_size": effective_sample_size(residual_weight_all),
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
            row[f"plugin_predicted_{summary_stub}"] = mean_plugin
            row[f"aipw_ht_{summary_stub}"] = ht_value
            row[f"aipw_hajek_{summary_stub}"] = hajek_value
            row[f"aipw_{summary_stub}"] = selected_value
            row[f"aipw_{summary_stub}_pct"] = selected_value * 100 if pd.notna(selected_value) else np.nan
            row[f"residual_correction_{outcome_name}"] = selected_correction
            row[f"aipw_ht_out_of_bounds_{outcome_name}"] = bool(pd.notna(ht_value) and (ht_value < 0 or ht_value > 1))
            row[f"aipw_hajek_out_of_bounds_{outcome_name}"] = bool(pd.notna(hajek_value) and (hajek_value < 0 or hajek_value > 1))
            row[f"aipw_selected_out_of_bounds_{outcome_name}"] = bool(
                pd.notna(selected_value) and (selected_value < 0 or selected_value > 1)
            )
    return row


def build_policy_summary(episode_df: pd.DataFrame, residual_normalisation: str) -> pd.DataFrame:
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
    for policy_values, policy_df in episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        rows.append(policy_summary_row(policy_df, policy_name, policy_type, policy_remove_day, residual_normalisation))
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
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


def inverse_support_ess(support: pd.Series) -> float:
    valid = pd.to_numeric(support, errors="coerce")
    valid = valid[valid.notna() & np.isfinite(valid) & valid.gt(0)]
    if valid.empty:
        return np.nan
    return effective_sample_size(1.0 / valid)


def build_support_diagnostics(row_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_values, policy_df in row_df.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        for label, group_df in [("all", policy_df)]:
            rows.append(support_diagnostic_row(group_df, label, policy_name, policy_remove_day))
        for split_col in ["split", "crossfit_fold", "_crossfit_fold", "fold_id"]:
            if split_col in policy_df.columns:
                for split_value, split_df in policy_df.groupby(split_col, dropna=False, sort=False):
                    rows.append(
                        support_diagnostic_row(
                            split_df,
                            f"{split_col}={split_value}",
                            policy_name,
                            policy_remove_day,
                        )
                    )
    return pd.DataFrame(rows)


def support_diagnostic_row(df: pd.DataFrame, label: str, policy_name, policy_remove_day) -> dict:
    applicable = df["policy_applicable"]
    support = pd.to_numeric(df.loc[applicable, "policy_support"], errors="coerce")
    finite = support.notna() & np.isfinite(support)
    valid = support.loc[finite]
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


def build_weight_diagnostics(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_values, policy_df in episode_df.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        if policy_name == CURRENT_PRACTICE_LABEL:
            continue
        residual_weights = valid_weight_series(policy_df[RESIDUAL_WEIGHT_COL])
        raw_episode_weights = valid_weight_series(policy_df[WEIGHT_COL])
        n_total = int(len(policy_df))
        n_adherent = int(policy_df["episode_adherent_to_policy"].eq(1).sum())
        pct_adherent = n_adherent / n_total if n_total else np.nan
        ess = effective_sample_size(policy_df[RESIDUAL_WEIGHT_COL])
        p99 = float(residual_weights.quantile(0.99)) if len(residual_weights) else np.nan
        max_weight = float(residual_weights.max()) if len(residual_weights) else np.nan
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
            "raw_episode_effective_sample_size": effective_sample_size(policy_df[WEIGHT_COL]),
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
    episode_df: pd.DataFrame,
    residual_normalisation: str,
) -> pd.DataFrame:
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
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
        positive_residual_weights = valid_weight_series(residual_weight_all)
        valid_residual_weight = residual_weight.notna() & np.isfinite(residual_weight)
        weight_sum = float(residual_weight.loc[valid_residual_weight].sum())

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
                "effective_sample_size": effective_sample_size(residual_weight_all),
                "residual_correction_effective_sample_size": effective_sample_size(residual_weight_all),
                "plugin_mean": mean_plugin,
                "observed_mean_among_adherent": float(observed.loc[adherent.eq(1)].mean()) if len(observed.loc[adherent.eq(1)].dropna()) else np.nan,
                "mean_residual": float(residual.mean()) if len(residual.dropna()) else np.nan,
                "mean_weighted_residual": float(weighted_residual.mean()) if len(weighted_residual.dropna()) else np.nan,
                "p99_abs_weighted_residual": float(weighted_residual.abs().quantile(0.99)) if len(weighted_residual.dropna()) else np.nan,
                "max_abs_weighted_residual": float(weighted_residual.abs().max()) if len(weighted_residual.dropna()) else np.nan,
                "min_episode_aipw_ht_score": float(ht_score.min()) if len(ht_score.dropna()) else np.nan,
                "max_episode_aipw_ht_score": float(ht_score.max()) if len(ht_score.dropna()) else np.nan,
                "aipw_ht_estimate": ht_value,
                "aipw_hajek_estimate": hajek_value,
                "aipw_selected_estimate": selected_value,
                "aipw_ht_out_of_bounds": bool(is_probability and pd.notna(ht_value) and (ht_value < 0 or ht_value > 1)),
                "aipw_hajek_out_of_bounds": bool(is_probability and pd.notna(hajek_value) and (hajek_value < 0 or hajek_value > 1)),
                "aipw_selected_out_of_bounds": bool(is_probability and pd.notna(selected_value) and (selected_value < 0 or selected_value > 1)),
            })
    return pd.DataFrame(rows)


def build_clipping_sensitivity(
    episode_df: pd.DataFrame,
    residual_normalisation: str,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    rows = []
    target_df = episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
    for policy_values, policy_df in target_df.groupby(["policy_name", "policy_type", "policy_remove_day"], dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        default_weights = pd.to_numeric(policy_df[RESIDUAL_WEIGHT_COL], errors="coerce").fillna(0.0)
        unclipped_weights = pd.to_numeric(
            policy_df[UNCLIPPED_RESIDUAL_WEIGHT_COL],
            errors="coerce",
        ).fillna(0.0)
        valid_default = valid_weight_series(default_weights)
        p99 = float(valid_default.quantile(0.99)) if len(valid_default) else np.nan
        for rule, weights, support_low, support_high, upper in [
            ("unclipped_support_where_safe", unclipped_weights, np.nan, np.nan, np.nan),
            ("row_support_clipped", default_weights, clip_lower, clip_upper, np.nan),
            ("upper_episode_weight_clipped_at_p99", default_weights.clip(upper=p99) if pd.notna(p99) else default_weights, clip_lower, clip_upper, p99),
            ("upper_episode_weight_clipped_at_30", default_weights.clip(upper=30.0), clip_lower, clip_upper, 30.0),
            ("upper_episode_weight_clipped_at_20", default_weights.clip(upper=20.0), clip_lower, clip_upper, 20.0),
        ]:
            temp = policy_df.copy()
            temp["__sensitivity_weight"] = weights
            summary_row = policy_summary_row(
                temp,
                policy_name,
                policy_type,
                policy_remove_day,
                residual_normalisation,
                "__sensitivity_weight",
            )
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
                "sum_weights": float(valid_weight_series(weights).sum()),
                "sum_residual_correction_weights": float(valid_weight_series(weights).sum()),
                "effective_sample_size": effective_sample_size(weights),
                "residual_correction_effective_sample_size": effective_sample_size(weights),
                "aipw_cauti_risk": summary_row.get("aipw_cauti_risk"),
                "aipw_recatheterisation_risk": summary_row.get("aipw_recatheterisation_risk"),
                "aipw_death_risk": summary_row.get("aipw_death_risk"),
                "aipw_icu_exit_alive_risk": summary_row.get("aipw_icu_exit_alive_risk"),
                "aipw_mean_catheter_exposure_days": summary_row.get("aipw_mean_catheter_exposure_days"),
                "aipw_mean_catheter_in_interval_rows": summary_row.get("aipw_mean_catheter_in_interval_rows"),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Output ordering and metadata
# =============================================================================

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
        "split",
        "crossfit_fold",
        "_crossfit_fold",
        "fold_id",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


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
        "policy_remove_day",
        "policy_catheter_state",
        "policy_action_resolved",
        "policy_action_remove_resolved",
        "policy_action_aipw",
        "policy_action_remove_aipw",
        "policy_periods_in",
        "policy_periods_out",
        "policy_removal_day_extra_row_treated_as_out",
        "p_remove_obs",
        "p_keep_obs",
        "policy_support",
        "policy_support_clipped",
        "policy_matches_observed_action_today",
        "deviated_from_policy_today",
        "followed_policy_so_far",
        *MU_COLUMNS,
        "prediction_status",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [
        col
        for col in df.columns
        if col not in ordered and not col.startswith("__join_key_") and not col.startswith("__rescored_")
    ]
    return df[[*ordered, *remaining]].copy()


def metadata_payload(
    args: argparse.Namespace,
    output_paths: dict,
    row_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    rescore_metadata: dict,
) -> dict:
    policies_with_zero_adherent = (
        episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)]
        .groupby("policy_name")["episode_adherent_to_policy"]
        .sum()
        .loc[lambda s: s == 0]
        .index.tolist()
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": ESTIMATOR_NAME,
        "residual_normalisation": args.residual_normalisation,
        "current_practice_comparator_type": "aipw_observed_regime",
        "clipping_bounds": {"clip_lower": args.clip_lower, "clip_upper": args.clip_upper},
        "input_paths": {
            "policy_panel": str(args.policy_panel),
            "scored_panel": str(args.scored_panel),
            "outcome_models": str(args.outcome_models),
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
            "plugin_expected_catheter_in_intervals": (
                "AIPW plug-in count using the same interval-row semantics as "
                "expected_catheter_in_intervals"
            ),
        },
        "icu_exit_alive_definition": (
            "max(icu_end_in_period == 1 and death_in_period != 1); death takes "
            "precedence when death and ICU exit occur in the same interval"
        ),
        "overlap_flag_thresholds": {
            "low_adherence_threshold": pec.LOW_ADHERENCE_THRESHOLD,
            "low_ess_min": pec.LOW_ESS_MIN,
            "low_ess_fraction": pec.LOW_ESS_FRACTION,
            "low_support_pct_below_0_05_threshold": pec.LOW_SUPPORT_PCT_BELOW_005_THRESHOLD,
            "extreme_weight_p99_threshold": pec.EXTREME_WEIGHT_P99_THRESHOLD,
            "extreme_weight_max_threshold": pec.EXTREME_WEIGHT_MAX_THRESHOLD,
        },
        "bootstrap": pec.bootstrap_metadata(args, row_df.columns),
        "rescoring": rescore_metadata,
        "methodological_limitations": [
            "AIPW estimates depend on cross-fitted propensity and outcome nuisance predictions.",
            "The plug-in component uses the observed patient-day covariate grid.",
            "The residual correction is available only through observed policy-adherent trajectories.",
            "This is not pure IPW, pure g-formula, Policy-DML, DR-Learner, TMLE, or LTMLE.",
            "No composite clinical policy score is calculated.",
        ],
    }


# =============================================================================
# Main
# =============================================================================

def print_console_summary(args, summary_df, row_df, episode_df, output_paths) -> None:
    target_episode_df = episode_df.loc[~episode_df["policy_name"].eq(CURRENT_PRACTICE_LABEL)]
    print()
    print("--- AIPW POLICY EVALUATION COMPLETE ---")
    print(f"Policy-intervention panel: {args.policy_panel}")
    print(f"Scored nuisance panel: {args.scored_panel}")
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
        "support_diagnostics": resolve_output_path(args.outdir, args.output_support_diagnostics),
        "weight_diagnostics": resolve_output_path(args.outdir, args.output_weight_diagnostics),
        "residual_diagnostics": resolve_output_path(args.outdir, args.output_residual_diagnostics),
        "clipping_sensitivity": resolve_output_path(args.outdir, args.output_clipping_sensitivity),
        "current_practice": resolve_output_path(args.outdir, args.output_current_practice),
        "metadata": resolve_output_path(args.outdir, args.output_metadata),
    }

    policy_df = load_policy_panel(args.policy_panel)
    scored_df = load_scored_panel(args.scored_panel)
    joined_df = join_scored_panel(policy_df, scored_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined AIPW policy rows")
    joined_df = pec.attach_resolved_timeline_aliases(
        joined_df,
        episode_id_col=EPISODE_ID_COL,
        context=str(args.policy_panel),
        action_col="policy_action_aipw",
        action_remove_col="policy_action_remove_aipw",
    )
    joined_df, rescore_metadata = fill_missing_counterfactual_predictions_safely(
        joined_df,
        args.outcome_models,
        args.allow_missing_counterfactual_state_predictions,
        "target_policy_rows",
    )
    joined_df = select_policy_predictions(joined_df)
    validate_prediction_completeness(
        joined_df,
        args.allow_missing_counterfactual_state_predictions,
    )
    joined_df = add_support_and_adherence(joined_df, args.clip_lower, args.clip_upper)
    policy_episode_df, row_df = build_policy_episode_scores(joined_df)

    current_rows = build_current_practice_rows(scored_df, policy_df)
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
    current_episode_df = build_current_practice_episode_scores(current_rows)
    rescore_metadata["current_practice_rescoring"] = current_rescore_metadata

    episode_df = pd.concat([policy_episode_df, current_episode_df], ignore_index=True, sort=False)
    summary_df = build_policy_summary(episode_df, args.residual_normalisation)
    support_diagnostics_df = build_support_diagnostics(row_df)
    weight_diagnostics_df = build_weight_diagnostics(episode_df)
    summary_df = pec.add_overlap_quality_flags(
        summary_df,
        support_diagnostics=support_diagnostics_df,
        weight_diagnostics=weight_diagnostics_df,
        current_practice_label=CURRENT_PRACTICE_LABEL,
    )
    residual_diagnostics_df = build_residual_diagnostics(
        episode_df,
        args.residual_normalisation,
    )
    clipping_sensitivity_df = build_clipping_sensitivity(
        episode_df,
        args.residual_normalisation,
        args.clip_lower,
        args.clip_upper,
    )

    save_df(summary_df, output_paths["summary"])
    save_df(episode_df, output_paths["episodes"])
    save_df(order_row_columns(row_df), output_paths["rows"])
    save_df(support_diagnostics_df, output_paths["support_diagnostics"])
    save_df(weight_diagnostics_df, output_paths["weight_diagnostics"])
    save_df(residual_diagnostics_df, output_paths["residual_diagnostics"])
    save_df(clipping_sensitivity_df, output_paths["clipping_sensitivity"])
    save_df(current_episode_df, output_paths["current_practice"])
    save_json(
        metadata_payload(args, output_paths, row_df, episode_df, rescore_metadata),
        output_paths["metadata"],
    )
    print_console_summary(args, summary_df, row_df, episode_df, output_paths)


if __name__ == "__main__":
    main()
