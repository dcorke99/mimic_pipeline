#!/usr/bin/env python3
"""Evaluate catheter-removal policies with sequential IPW estimates."""


import argparse
from datetime import datetime, timezone
from pathlib import Path

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
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "policy_eval" / "ipw"

DEFAULT_OUTPUT_SUMMARY = "ipw_policy_outcomes_summary.csv"
DEFAULT_OUTPUT_EPISODES = "ipw_policy_episode_outcomes.csv"
DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS = "ipw_weight_diagnostics.csv"
DEFAULT_OUTPUT_SUPPORT_DIAGNOSTICS = "ipw_policy_support_diagnostics.csv"
DEFAULT_OUTPUT_CLIPPING_SENSITIVITY = "ipw_clipping_sensitivity.csv"
DEFAULT_OUTPUT_CURRENT_PRACTICE = "current_practice_episode_outcomes.csv"
DEFAULT_OUTPUT_METADATA = "ipw_run_metadata.json"

CURRENT_PRACTICE_LABEL = "current_practice"
EPISODE_ID_COL = "catheter_episode_id"
POLICY_TYPE_COL = "policy_type"
WEIGHT_COL = "episode_ipw_weight"
UNCLIPPED_WEIGHT_COL = "episode_ipw_weight_unclipped"

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

OPTIONAL_SCORED_COLS = [
    "p_keep_obs",
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

STALE_POLICY_PANEL_ESTIMATOR_COLS = [
    "p_remove_obs",
    "p_keep_obs",
    "p_observed_action",
    "p_observed_action_clipped",
    "policy_support",
    "policy_support_clipped",
    "policy_weight_component",
    "ipw_component",
    "ipw_component_unclipped",
    "episode_ipw_weight",
    "episode_ipw_weight_unclipped",
    "followed_policy_so_far",
    "episode_matches_policy",
    "episode_adherent_to_policy",
]

DATETIME_KEY_COLS = {"inserted", "removed", "period_start", "period_end"}
LOWER_TEXT_KEY_COLS = {"catheter_state", "observed_action"}
NUMERIC_KEY_COLS = {"subject_id", "hadm_id", "stay_id", "periods_in_state", "action_remove"}

OUTCOME_SPECS = {
    "cauti": ("any_cauti", "cauti_in_period"),
    "recatheterisation": ("any_recatheterisation", "reinsertion_in_period"),
    "death": ("any_death", "death_in_period"),
    "icu_exit_alive": ("observed_icu_exit_alive", "observed_icu_exit_alive_in_period"),
}


# Generic helpers

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic catheter-removal policies using sequential "
            "IPW from an estimator-agnostic policy-intervention panel."
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
        help=f"Scored nuisance panel containing p_remove_obs. Default: {DEFAULT_SCORED_PANEL_PATH}",
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
        help=f"Episode-level output filename. Default: {DEFAULT_OUTPUT_EPISODES}",
    )
    parser.add_argument(
        "--output-weight-diagnostics",
        default=DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS,
        help=f"Weight diagnostics output filename. Default: {DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS}",
    )
    parser.add_argument(
        "--output-support-diagnostics",
        default=DEFAULT_OUTPUT_SUPPORT_DIAGNOSTICS,
        help=f"Support diagnostics output filename. Default: {DEFAULT_OUTPUT_SUPPORT_DIAGNOSTICS}",
    )
    parser.add_argument(
        "--output-clipping-sensitivity",
        default=DEFAULT_OUTPUT_CLIPPING_SENSITIVITY,
        help=f"Clipping sensitivity output filename. Default: {DEFAULT_OUTPUT_CLIPPING_SENSITIVITY}",
    )
    parser.add_argument(
        "--output-current-practice",
        default=DEFAULT_OUTPUT_CURRENT_PRACTICE,
        help=f"Current-practice episode output filename. Default: {DEFAULT_OUTPUT_CURRENT_PRACTICE}",
    )
    parser.add_argument(
        "--output-metadata",
        default=DEFAULT_OUTPUT_METADATA,
        help=f"Run metadata output filename. Default: {DEFAULT_OUTPUT_METADATA}",
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=0.01,
        help="Lower bound for row-level behaviour-policy support. Default: 0.01",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=0.99,
        help="Upper bound for row-level behaviour-policy support. Default: 0.99",
    )
    parser.add_argument(
        "--zero-applicable-policy-episodes",
        choices=["weight-one", "exclude", "error"],
        default="weight-one",
        help=(
            "How to handle adherent policy episodes with no applicable decision "
            "rows. Default assigns weight 1 and flags the episode."
        ),
    )
    pec.add_bootstrap_args(parser)
    return parser.parse_args()


def resolve_output_path(outdir, name_or_path):
    """Resolve an output file path."""
    path = Path(name_or_path)
    return path if path.is_absolute() else outdir / path


def save_df(df, path):
    """Save a data frame as CSV."""
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def first_non_null(series):
    """Return the first non-missing value."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series):
    """Return whether any binary value is present."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def weighted_mean(values, weights):
    """Calculate a weighted mean."""
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if int(valid.sum()) == 0:
        return np.nan
    return float(np.sum(values.loc[valid] * weights.loc[valid]) / np.sum(weights.loc[valid]))


def valid_weight_series(weights):
    """Return positive finite weights."""
    weights = pd.to_numeric(weights, errors="coerce")
    return weights[weights.notna() & np.isfinite(weights) & (weights > 0)]


def effective_sample_size(weights):
    """Calculate the effective sample size."""
    # Return positive finite weights.
    weights = valid_weight_series(weights)
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


def safe_ratio(numerator, denominator):
    """Calculate a ratio with missing-value protection."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def coerce_bool(series):
    """Convert common text values to booleans."""
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


def canonical_numeric_value(value):
    """Format a numeric join value consistently."""
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


def canonical_key_series(series, column):
    """Create a stable join-key series."""
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


def add_join_key_columns(df, key_cols):
    """Add stable join-key columns."""
    out = df.copy()
    join_cols = []
    # Create a stable join-key series.
    for idx, col in enumerate(key_cols):
        join_col = f"__join_key_{idx}"
        # Create a stable join-key series.
        out[join_col] = canonical_key_series(out[col], col)
        join_cols.append(join_col)
    return out, join_cols


def duplicate_key_examples(df, join_cols, display_cols):
    """Return examples of duplicate join keys."""
    duplicated = df.duplicated(join_cols, keep=False)
    if not duplicated.any():
        return pd.DataFrame()
    return df.loc[duplicated, display_cols].head(10)


# Loading and joining

def load_policy_panel(path):
    """Load and validate the policy panel."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    stale_cols = [col for col in STALE_POLICY_PANEL_ESTIMATOR_COLS if col in df.columns]
    if stale_cols:
        print(
            "Ignoring pre-existing IPW/nuisance columns in policy panel; "
            f"this script recalculates them: {stale_cols}"
        )
        df = df.drop(columns=stale_cols)

    # Normalise policy panel types.
    df = normalise_policy_panel_types(df)
    # Validate policy-panel structure.
    validate_policy_panel(df)
    return df


def load_scored_panel(path):
    """Load and validate the scored nuisance panel."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Normalise row-key columns for joining.
    df = normalise_row_key_types(df)
    df["p_remove_obs"] = pd.to_numeric(df["p_remove_obs"], errors="coerce")
    if "p_keep_obs" not in df.columns:
        df["p_keep_obs"] = 1.0 - df["p_remove_obs"]
    else:
        df["p_keep_obs"] = pd.to_numeric(df["p_keep_obs"], errors="coerce")
    return df


def normalise_row_key_types(df):
    """Normalise row-key columns for joining."""
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


def normalise_policy_panel_types(df):
    """Normalise policy panel types."""
    # Normalise row-key columns for joining.
    df = normalise_row_key_types(df)
    # Convert common text values to booleans.
    df["is_decision_row"] = coerce_bool(df["is_decision_row"])
    # Convert common text values to booleans.
    df["policy_applicable"] = coerce_bool(df["policy_applicable"])
    df["policy_action"] = df["policy_action"].astype("string").str.strip().str.lower()
    df["policy_reason"] = df["policy_reason"].astype("string").str.strip().str.lower()
    if POLICY_TYPE_COL in df.columns:
        df[POLICY_TYPE_COL] = df[POLICY_TYPE_COL].astype("string").str.strip().str.lower()
    else:
        df[POLICY_TYPE_COL] = pd.NA
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
    return df


def validate_policy_panel(df):
    """Validate policy-panel structure."""
    policies = sorted(df["policy_name"].dropna().unique().tolist())
    if not policies:
        raise ValueError("Policy panel contains no policy_name values.")

    applicable_counts = df.groupby("policy_name")["policy_applicable"].sum()
    empty_policies = applicable_counts[applicable_counts == 0].index.tolist()
    if empty_policies:
        raise ValueError(f"Policy panel has no applicable decision rows for: {empty_policies}")

    applicable = df["policy_applicable"]
    invalid_action = applicable & ~df["policy_action_remove"].isin([0, 1])
    if invalid_action.any():
        examples = df.loc[
            invalid_action,
            ["policy_name", "decision_row_id", "policy_action", "policy_action_remove"],
        ].head(10)
        raise ValueError(
            "Applicable policy rows must have policy_action_remove equal to 0 or 1. "
            f"Examples:\n{examples}"
        )

    missing_match = applicable & df["policy_matches_observed_action_today"].isna()
    if missing_match.any():
        examples = df.loc[
            missing_match,
            ["policy_name", "decision_row_id", "policy_action", "action_remove"],
        ].head(10)
        raise ValueError(
            "Applicable policy rows are missing policy_matches_observed_action_today. "
            f"Examples:\n{examples}"
        )

    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col=EPISODE_ID_COL,
        context="IPW policy panel",
    )


def join_scored_panel(policy_df, scored_df):
    """Join nuisance scores to policy rows."""
    # Add stable join-key columns.
    policy_keyed, join_cols = add_join_key_columns(policy_df, ROW_JOIN_KEY_COLS)
    # Add stable join-key columns.
    scored_keyed, _ = add_join_key_columns(scored_df, ROW_JOIN_KEY_COLS)

    scored_add_cols = [
        "p_remove_obs",
        "p_keep_obs",
        *[
            col
            for col in OPTIONAL_SCORED_COLS
            if col in scored_keyed.columns and col not in {"p_keep_obs"}
        ],
    ]
    scored_add_cols = list(dict.fromkeys(scored_add_cols))
    scored_add_cols = [
        col
        for col in scored_add_cols
        if col not in ROW_JOIN_KEY_COLS and (col not in policy_df.columns or col.startswith("p_"))
    ]

    duplicates = scored_keyed.duplicated(join_cols, keep=False)
    # Return examples of duplicate join keys.
    if duplicates.any():
        # Return examples of duplicate join keys.
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

    merged = merged.drop(columns=[*join_cols, "_merge"])
    return merged


# IPW row-level calculations and adherence

def validate_clip_bounds(clip_lower, clip_upper):
    """Validate support clipping bounds."""
    if not (0 < clip_lower < clip_upper <= 1):
        raise ValueError(
            "Support clipping bounds must satisfy 0 < clip_lower < clip_upper <= 1. "
            f"Received clip_lower={clip_lower}, clip_upper={clip_upper}."
        )


def add_ipw_row_quantities(
    df,
    clip_lower,
    clip_upper,
):
    """Add row-level IPW quantities."""
    # Validate support clipping bounds.
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
            "Applicable IN decision rows are missing p_remove_obs or p_keep_obs "
            f"after joining scored panel. Examples:\n{examples}"
        )

    for col in ["p_remove_obs", "p_keep_obs"]:
        invalid = applicable & (
            ~np.isfinite(df[col])
            | df[col].lt(0)
            | df[col].gt(1)
        )
        if invalid.any():
            examples = df.loc[
                invalid,
                ["policy_name", "decision_row_id", col],
            ].head(10)
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
    invalid_support = applicable & (
        support.isna()
        | ~np.isfinite(support)
        | support.lt(0)
        | support.gt(1)
    )
    if invalid_support.any():
        examples = df.loc[
            invalid_support,
            ["policy_name", "decision_row_id", "policy_action_remove", "policy_support"],
        ].head(10)
        raise ValueError(
            "Policy support must be finite and between 0 and 1 before clipping. "
            f"Examples:\n{examples}"
        )

    df["policy_support_clipped"] = support.clip(lower=clip_lower, upper=clip_upper)
    df["ipw_component"] = np.nan
    df["ipw_component_unclipped"] = np.nan
    df["zero_support_matched_row"] = 0

    match_rows = applicable & df["policy_matches_observed_action_today"].eq(1)
    df.loc[match_rows, "ipw_component"] = 1.0 / df.loc[match_rows, "policy_support_clipped"]

    safe_unclipped = match_rows & support.gt(0)
    df.loc[safe_unclipped, "ipw_component_unclipped"] = 1.0 / support.loc[safe_unclipped]
    df.loc[match_rows & support.eq(0), "zero_support_matched_row"] = 1

    df["deviated_from_policy_today"] = 0
    df.loc[applicable & df["policy_matches_observed_action_today"].eq(0), "deviated_from_policy_today"] = 1

    return df


def add_adherence(df):
    """Add cumulative policy-adherence flags."""
    df = df.copy()
    sort_cols = ["policy_name", EPISODE_ID_COL, "period_start", "period_end", "decision_row_id"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    group_cols = ["policy_name", EPISODE_ID_COL]
    deviation_cummax = df.groupby(group_cols, sort=False)["deviated_from_policy_today"].cummax()
    df["followed_policy_so_far"] = (1 - deviation_cummax).astype(int)
    episode_deviation = df.groupby(group_cols, sort=False)["deviated_from_policy_today"].transform("max")
    df["episode_adherent_to_policy"] = episode_deviation.eq(0).astype(int)
    return df


# Episode-level collapse

def product_components_by_episode(df, component_col):
    """Multiply row components within each episode."""
    group_cols = ["policy_name", "policy_remove_day", EPISODE_ID_COL]
    out = (
        df.groupby(group_cols, dropna=False, sort=False)[component_col]
        .prod(min_count=1)
        .reset_index()
    )
    return out


def add_episode_level_flags(df):
    """Add helper flags for episode aggregation."""
    df = df.copy()
    df["_applicable_int"] = df["policy_applicable"].astype(int)
    df["_matched_applicable_int"] = (
        df["policy_applicable"] & df["policy_matches_observed_action_today"].eq(1)
    ).astype(int)
    df["_remove_assigned_int"] = (
        df["policy_applicable"] & df["policy_action_remove"].eq(1)
    ).astype(int)
    df["_observed_remove_under_policy_int"] = (
        df["policy_applicable"]
        & df["policy_action_remove"].eq(1)
        & df["policy_matches_observed_action_today"].eq(1)
    ).astype(int)
    df["_catheter_in_row_int"] = df["catheter_state"].astype("string").str.lower().eq("in").astype(int)
    df["_catheter_exposure_days"] = df["_catheter_in_row_int"] * pd.to_numeric(
        df["period_duration_days"],
        errors="coerce",
    )
    return df


def build_policy_episode_panel(
    df,
    zero_applicable_policy_episodes,
):
    """Build adherent policy-episode outcomes."""
    # Add helper flags for episode aggregation.
    df = add_episode_level_flags(df)
    group_cols = ["policy_name", "policy_remove_day", EPISODE_ID_COL]
    aggregations = {
        POLICY_TYPE_COL: first_non_null,
        "episode_adherent_to_policy": "max",
        "_applicable_int": "sum",
        "_matched_applicable_int": "sum",
        "deviated_from_policy_today": "sum",
        "_remove_assigned_int": "max",
        "_observed_remove_under_policy_int": "max",
        "_catheter_in_row_int": "sum",
        "_catheter_exposure_days": "sum",
        "zero_support_matched_row": "max",
    }

    for col in OPTIONAL_FIRST_COLS:
        if col in df.columns and col not in group_cols:
            aggregations[col] = first_non_null

    for episode_col, period_col in OUTCOME_SPECS.values():
        if period_col in df.columns:
            aggregations[period_col] = max_binary

    for risk_col in ["at_risk_cauti", "at_risk_reinsertion"]:
        if risk_col in df.columns:
            aggregations[risk_col] = "sum"

    episode_all = df.groupby(group_cols, dropna=False, as_index=False, sort=False).agg(aggregations)
    episode_all = episode_all.rename(
        columns={
            "_applicable_int": "n_applicable_policy_rows",
            "_matched_applicable_int": "n_matched_policy_rows",
            "deviated_from_policy_today": "n_deviation_rows",
            "_remove_assigned_int": "policy_remove_assigned",
            "_observed_remove_under_policy_int": "observed_remove_under_policy",
            "_catheter_in_row_int": "observed_catheter_in_intervals",
            "_catheter_exposure_days": "observed_catheter_exposure_days",
            "at_risk_cauti": "cauti_at_risk_rows",
            "at_risk_reinsertion": "reinsertion_at_risk_rows",
        }
    )
    episode_all["observed_catheter_in_interval_rows"] = episode_all["observed_catheter_in_intervals"]

    for episode_col, period_col in OUTCOME_SPECS.values():
        if period_col in episode_all.columns:
            episode_all = episode_all.rename(columns={period_col: episode_col})
        elif episode_col not in episode_all.columns:
            episode_all[episode_col] = np.nan

    for col in ["cauti_at_risk_rows", "reinsertion_at_risk_rows"]:
        if col not in episode_all.columns:
            episode_all[col] = np.nan

    # Multiply row components within each episode.
    weight_products = product_components_by_episode(df, "ipw_component")
    # Multiply row components within each episode.
    unclipped_weight_products = product_components_by_episode(df, "ipw_component_unclipped")
    episode_all = episode_all.merge(weight_products, on=group_cols, how="left")
    episode_all = episode_all.merge(
        unclipped_weight_products,
        on=group_cols,
        how="left",
        suffixes=("", "_unclipped"),
    )
    episode_all = episode_all.rename(
        columns={
            "ipw_component": WEIGHT_COL,
            "ipw_component_unclipped": UNCLIPPED_WEIGHT_COL,
        }
    )

    episode_all["zero_applicable_rows_weight_assigned"] = 0
    zero_applicable_adherent = (
        episode_all["episode_adherent_to_policy"].eq(1)
        & episode_all["n_applicable_policy_rows"].eq(0)
    )
    if zero_applicable_adherent.any():
        n_zero = int(zero_applicable_adherent.sum())
        if zero_applicable_policy_episodes == "error":
            raise ValueError(
                f"{n_zero} adherent policy episodes have no applicable decision rows."
            )
        if zero_applicable_policy_episodes == "exclude":
            print(
                "WARNING: excluding adherent policy episodes with no applicable "
                f"decision rows: {n_zero}"
            )
            episode_all.loc[zero_applicable_adherent, "episode_adherent_to_policy"] = 0
        else:
            print(
                "WARNING: assigning weight 1 to adherent policy episodes with "
                f"no applicable decision rows: {n_zero}"
            )
            episode_all.loc[zero_applicable_adherent, WEIGHT_COL] = 1.0
            episode_all.loc[zero_applicable_adherent, UNCLIPPED_WEIGHT_COL] = 1.0
            episode_all.loc[zero_applicable_adherent, "zero_applicable_rows_weight_assigned"] = 1

    missing_adherent_weight = (
        episode_all["episode_adherent_to_policy"].eq(1)
        & episode_all["n_applicable_policy_rows"].gt(0)
        & episode_all[WEIGHT_COL].isna()
    )
    if missing_adherent_weight.any():
        examples = episode_all.loc[
            missing_adherent_weight,
            ["policy_name", EPISODE_ID_COL, "n_applicable_policy_rows", "n_matched_policy_rows"],
        ].head(10)
        raise ValueError(
            "Some adherent policy episodes have applicable rows but no IPW weight. "
            f"Examples:\n{examples}"
        )

    policy_episode_df = episode_all.loc[
        episode_all["episode_adherent_to_policy"].eq(1)
    ].copy()
    return episode_all, policy_episode_df


def build_current_practice_episode_panel(scored_df, policy_df):
    """Build observed current-practice episode outcomes."""
    scored_df = scored_df.copy()
    scored_df = pec.add_period_duration_days(scored_df, context="current-practice IPW scored rows")
    scored_df = pec.add_observed_icu_exit_alive_period(scored_df)
    # Add stable join-key columns.
    if EPISODE_ID_COL not in scored_df.columns:
        episode_map = policy_df[[*EPISODE_KEY_COLS, EPISODE_ID_COL]].drop_duplicates()
        # Add stable join-key columns.
        scored_keyed, join_cols = add_join_key_columns(scored_df, EPISODE_KEY_COLS)
        # Add stable join-key columns.
        map_keyed, _ = add_join_key_columns(episode_map, EPISODE_KEY_COLS)

        duplicated_episode_keys = map_keyed.duplicated(join_cols, keep=False)
        if duplicated_episode_keys.any():
            examples = map_keyed.loc[duplicated_episode_keys, EPISODE_KEY_COLS + [EPISODE_ID_COL]].head(10)
            raise ValueError(
                "Policy panel maps at least one natural episode key to multiple "
                f"catheter_episode_id values. Examples:\n{examples}"
            )
        scored_df = scored_keyed.merge(
            map_keyed[[*join_cols, EPISODE_ID_COL]],
            on=join_cols,
            how="left",
            validate="many_to_one",
        ).drop(columns=join_cols)
        if scored_df[EPISODE_ID_COL].isna().any():
            examples = scored_df.loc[scored_df[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
            raise ValueError(
                "Some scored-panel rows could not be mapped to catheter_episode_id "
                f"from the policy panel. Examples:\n{examples}"
            )

    scored_df["catheter_state"] = scored_df["catheter_state"].astype("string").str.strip().str.lower()
    scored_df["_catheter_in_row_int"] = scored_df["catheter_state"].eq("in").astype(int)
    scored_df["_catheter_exposure_days"] = scored_df["_catheter_in_row_int"] * pd.to_numeric(
        scored_df["period_duration_days"],
        errors="coerce",
    )

    group_cols = [EPISODE_ID_COL]
    aggregations = {
        "_catheter_in_row_int": "sum",
        "_catheter_exposure_days": "sum",
    }
    for col in OPTIONAL_FIRST_COLS:
        if col in scored_df.columns:
            aggregations[col] = first_non_null
    for episode_col, period_col in OUTCOME_SPECS.values():
        if period_col in scored_df.columns:
            aggregations[period_col] = max_binary
    for risk_col in ["at_risk_cauti", "at_risk_reinsertion"]:
        if risk_col in scored_df.columns:
            aggregations[risk_col] = "sum"

    episode_df = scored_df.groupby(group_cols, dropna=False, as_index=False, sort=False).agg(aggregations)
    episode_df = episode_df.rename(
        columns={
            "_catheter_in_row_int": "observed_catheter_in_intervals",
            "_catheter_exposure_days": "observed_catheter_exposure_days",
            "at_risk_cauti": "cauti_at_risk_rows",
            "at_risk_reinsertion": "reinsertion_at_risk_rows",
        }
    )
    episode_df["observed_catheter_in_interval_rows"] = episode_df["observed_catheter_in_intervals"]
    for episode_col, period_col in OUTCOME_SPECS.values():
        if period_col in episode_df.columns:
            episode_df = episode_df.rename(columns={period_col: episode_col})
        elif episode_col not in episode_df.columns:
            episode_df[episode_col] = np.nan

    episode_df["policy_name"] = CURRENT_PRACTICE_LABEL
    episode_df[POLICY_TYPE_COL] = "observed"
    episode_df["policy_remove_day"] = pd.NA
    episode_df["n_applicable_policy_rows"] = pd.NA
    episode_df["n_matched_policy_rows"] = pd.NA
    episode_df["n_deviation_rows"] = pd.NA
    episode_df["episode_adherent_to_policy"] = 1
    episode_df["policy_remove_assigned"] = pd.NA
    episode_df["observed_remove_under_policy"] = pd.NA
    episode_df["zero_applicable_rows_weight_assigned"] = 0
    episode_df[WEIGHT_COL] = 1.0
    episode_df[UNCLIPPED_WEIGHT_COL] = 1.0
    # Order episode-level output columns.
    return order_episode_columns(episode_df)


def order_episode_columns(df):
    """Order episode-level output columns."""
    preferred = [
        "policy_name",
        POLICY_TYPE_COL,
        "policy_remove_day",
        EPISODE_ID_COL,
        *[col for col in OPTIONAL_FIRST_COLS if col in df.columns],
        "episode_adherent_to_policy",
        "n_applicable_policy_rows",
        "n_matched_policy_rows",
        "n_deviation_rows",
        "policy_remove_assigned",
        "observed_remove_under_policy",
        "zero_applicable_rows_weight_assigned",
        "zero_support_matched_row",
        WEIGHT_COL,
        UNCLIPPED_WEIGHT_COL,
        "any_cauti",
        "any_recatheterisation",
        "any_death",
        "observed_icu_exit_alive",
        "observed_catheter_in_intervals",
        "observed_catheter_exposure_days",
        "observed_catheter_in_interval_rows",
        "cauti_at_risk_rows",
        "reinsertion_at_risk_rows",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


# Diagnostics and summaries

def inverse_support_ess(support):
    """Calculate inverse-support effective sample size."""
    support = pd.to_numeric(support, errors="coerce")
    valid = support[support.notna() & np.isfinite(support) & support.gt(0)]
    if valid.empty:
        return np.nan
    # Calculate the effective sample size.
    return effective_sample_size(1.0 / valid)


def support_diagnostic_row(df, label, metadata):
    """Build one support diagnostic row."""
    applicable = df["policy_applicable"]
    support = pd.to_numeric(df.loc[applicable, "policy_support"], errors="coerce")
    finite = support.notna() & np.isfinite(support)
    valid = support.loc[finite]
    # Calculate inverse-support effective sample size.
    row = {
        **metadata,
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


def build_support_diagnostics(df):
    """Build support diagnostic output."""
    rows = []
    policy_cols = ["policy_name", "policy_remove_day"]
    # Build one support diagnostic row.
    for policy_values, policy_df in df.groupby(policy_cols, dropna=False, sort=False):
        metadata = {
            "policy_name": policy_values[0],
            "policy_remove_day": policy_values[1],
        }
        # Build one support diagnostic row.
        rows.append(support_diagnostic_row(policy_df, "all", metadata))

        # Build one support diagnostic row.
        for split_col in ["split", "crossfit_fold", "_crossfit_fold", "fold_id"]:
            # Build one support diagnostic row.
            if split_col in policy_df.columns:
                # Build one support diagnostic row.
                for split_value, split_df in policy_df.groupby(split_col, dropna=False, sort=False):
                    # Build one support diagnostic row.
                    rows.append(
                        support_diagnostic_row(
                            split_df,
                            f"{split_col}={split_value}",
                            metadata,
                        )
                    )
    return pd.DataFrame(rows)


def weight_diagnostic_row(
    policy_name,
    policy_remove_day,
    episode_all,
    adherent_df,
):
    """Build one weight diagnostic row."""
    # Return positive finite weights.
    weights = valid_weight_series(adherent_df[WEIGHT_COL]) if WEIGHT_COL in adherent_df.columns else pd.Series(dtype=float)
    n_total = int(len(episode_all))
    n_adherent = int(episode_all["episode_adherent_to_policy"].eq(1).sum())
    n_non_adherent = int(n_total - n_adherent)

    # Calculate the effective sample size.
    ess = effective_sample_size(weights)
    pct_adherent = n_adherent / n_total if n_total else np.nan
    p99 = float(weights.quantile(0.99)) if len(weights) else np.nan
    max_weight = float(weights.max()) if len(weights) else np.nan
    return {
        "policy_name": policy_name,
        "policy_remove_day": policy_remove_day,
        "min_weight": float(weights.min()) if len(weights) else np.nan,
        "median_weight": float(weights.median()) if len(weights) else np.nan,
        "p90_weight": float(weights.quantile(0.90)) if len(weights) else np.nan,
        "p95_weight": float(weights.quantile(0.95)) if len(weights) else np.nan,
        "p99_weight": p99,
        "max_weight": max_weight,
        "effective_sample_size": ess,
        "n_adherent_episodes": n_adherent,
        "n_non_adherent_episodes": n_non_adherent,
        "pct_adherent_episodes": pct_adherent,
        "n_total_policy_episodes": n_total,
        "low_adherence_flag": bool(pd.notna(pct_adherent) and pct_adherent < pec.LOW_ADHERENCE_THRESHOLD),
        "low_ess_flag": bool(
            pd.notna(ess)
            and (ess < pec.LOW_ESS_MIN or (n_total > 0 and ess < pec.LOW_ESS_FRACTION * n_total))
        ),
        "extreme_weight_flag": bool(
            (pd.notna(p99) and p99 > pec.EXTREME_WEIGHT_P99_THRESHOLD)
            or (pd.notna(max_weight) and max_weight > pec.EXTREME_WEIGHT_MAX_THRESHOLD)
        ),
        "n_zero_applicable_rows_weight_assigned": int(
            episode_all.get("zero_applicable_rows_weight_assigned", pd.Series(dtype=int)).sum()
        ),
        "n_zero_support_matched_episodes": int(
            episode_all.get("zero_support_matched_row", pd.Series(dtype=int)).sum()
        ),
    }


def build_weight_diagnostics(episode_all, policy_episode_df):
    """Build weight diagnostic output."""
    rows = []
    group_cols = ["policy_name", "policy_remove_day"]
    # Build one weight diagnostic row.
    for policy_values, all_df in episode_all.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        adherent_df = policy_episode_df.loc[
            policy_episode_df["policy_name"].eq(policy_name)
            & policy_episode_df["policy_remove_day"].eq(policy_remove_day)
        ]
        # Build one weight diagnostic row.
        rows.append(weight_diagnostic_row(policy_name, policy_remove_day, all_df, adherent_df))
    return pd.DataFrame(rows)


def available_outcome_columns(df):
    """List available episode outcome columns."""
    return {
        outcome_name: episode_col
        for outcome_name, (episode_col, _) in OUTCOME_SPECS.items()
        if episode_col in df.columns
    }


def summarise_episode_estimates(
    episode_df,
    policy_name,
    policy_remove_day,
    n_total_policy_episodes,
    n_adherent_episodes,
    weight_col=WEIGHT_COL,
):
    """Summarise episode-level estimates."""
    weights = pd.to_numeric(episode_df[weight_col], errors="coerce") if weight_col in episode_df.columns else pd.Series(dtype=float)
    # Calculate the effective sample size.
    row = {
        "policy_name": policy_name,
        "policy_remove_day": policy_remove_day,
        "n_episodes": int(len(episode_df)),
        "n_total_policy_episodes": int(n_total_policy_episodes),
        "n_adherent_episodes": int(n_adherent_episodes),
        "pct_adherent_episodes": (
            float(n_adherent_episodes / n_total_policy_episodes)
            if n_total_policy_episodes
            else np.nan
        ),
        "sum_weights": float(valid_weight_series(weights).sum()) if len(weights) else np.nan,
        "mean_weight": float(valid_weight_series(weights).mean()) if len(valid_weight_series(weights)) else np.nan,
        "max_weight": float(valid_weight_series(weights).max()) if len(valid_weight_series(weights)) else np.nan,
        "effective_sample_size": effective_sample_size(weights),
    }

    # Calculate a weighted mean.
    for outcome_name, episode_col in available_outcome_columns(episode_df).items():
        unweighted = pd.to_numeric(episode_df[episode_col], errors="coerce").mean()
        # Calculate a weighted mean.
        weighted = weighted_mean(episode_df[episode_col], weights)
        row[f"unweighted_{outcome_name}_risk"] = float(unweighted) if pd.notna(unweighted) else np.nan
        row[f"ipw_weighted_{outcome_name}_risk"] = weighted
        row[f"unweighted_{outcome_name}_risk_pct"] = row[f"unweighted_{outcome_name}_risk"] * 100
        row[f"ipw_weighted_{outcome_name}_risk_pct"] = weighted * 100 if pd.notna(weighted) else np.nan

    # Calculate a weighted mean.
    if "observed_catheter_in_intervals" in episode_df.columns:
        unweighted_rows = pd.to_numeric(
            episode_df["observed_catheter_in_intervals"],
            errors="coerce",
        ).mean()
        # Calculate a weighted mean.
        weighted_rows = weighted_mean(episode_df["observed_catheter_in_intervals"], weights)
        row["unweighted_mean_catheter_in_intervals"] = (
            float(unweighted_rows) if pd.notna(unweighted_rows) else np.nan
        )
        row["ipw_weighted_mean_catheter_in_intervals"] = weighted_rows
        row["unweighted_mean_catheter_in_interval_rows"] = row[
            "unweighted_mean_catheter_in_intervals"
        ]
        row["ipw_weighted_mean_catheter_in_interval_rows"] = row[
            "ipw_weighted_mean_catheter_in_intervals"
        ]

    # Calculate a weighted mean.
    if "observed_catheter_exposure_days" in episode_df.columns:
        unweighted_exposure = pd.to_numeric(
            episode_df["observed_catheter_exposure_days"],
            errors="coerce",
        ).mean()
        # Calculate a weighted mean.
        weighted_exposure = weighted_mean(episode_df["observed_catheter_exposure_days"], weights)
        row["unweighted_mean_catheter_exposure_days"] = (
            float(unweighted_exposure) if pd.notna(unweighted_exposure) else np.nan
        )
        row["ipw_weighted_mean_catheter_exposure_days"] = weighted_exposure

    return row


def build_policy_summary(
    episode_all,
    policy_episode_df,
    current_practice_episode_df,
):
    """Build policy-level summary estimates."""
    rows = []
    # Summarise episode-level estimates.
    rows.append(
        summarise_episode_estimates(
            current_practice_episode_df,
            CURRENT_PRACTICE_LABEL,
            pd.NA,
            len(current_practice_episode_df),
            len(current_practice_episode_df),
        )
    )

    group_cols = ["policy_name", "policy_remove_day"]
    # Summarise episode-level estimates.
    for policy_values, all_df in episode_all.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        adherent_df = policy_episode_df.loc[
            policy_episode_df["policy_name"].eq(policy_name)
            & policy_episode_df["policy_remove_day"].eq(policy_remove_day)
        ]
        # Summarise episode-level estimates.
        rows.append(
            summarise_episode_estimates(
                adherent_df,
                policy_name,
                policy_remove_day,
                len(all_df),
                int(all_df["episode_adherent_to_policy"].eq(1).sum()),
            )
        )

    # Add comparisons against current practice.
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary):
    """Add comparisons against current practice."""
    return pec.add_standard_comparisons(
        summary,
        baseline_label=CURRENT_PRACTICE_LABEL,
        comparison_map={
            "cauti_risk": "ipw_weighted_cauti_risk",
            "recatheterisation_risk": "ipw_weighted_recatheterisation_risk",
            "death_risk": "ipw_weighted_death_risk",
            "icu_exit_alive_risk": "ipw_weighted_icu_exit_alive_risk",
            "catheter_exposure_days": "ipw_weighted_mean_catheter_exposure_days",
        },
    )


def clipping_estimate_row(
    episode_df,
    policy_name,
    policy_remove_day,
    clipping_rule,
    weights,
    clip_upper_weight,
    support_clip_lower=np.nan,
    support_clip_upper=np.nan,
):
    """Build one clipping-sensitivity row."""
    # Return positive finite weights.
    valid_weights = valid_weight_series(weights)
    # Calculate the effective sample size.
    row = {
        "policy_name": policy_name,
        "policy_remove_day": policy_remove_day,
        "clipping_rule": clipping_rule,
        "support_clip_lower": support_clip_lower,
        "support_clip_upper": support_clip_upper,
        "clip_upper_weight": clip_upper_weight,
        "n_episodes": int(len(episode_df)),
        "n_weighted_episodes": int(len(valid_weights)),
        "sum_weights": float(valid_weights.sum()) if len(valid_weights) else np.nan,
        "effective_sample_size": effective_sample_size(weights),
    }
    # Calculate a weighted mean.
    for outcome_name, episode_col in available_outcome_columns(episode_df).items():
        # Calculate a weighted mean.
        weighted = weighted_mean(episode_df[episode_col], weights)
        row[f"ipw_weighted_{outcome_name}_risk"] = weighted
        row[f"ipw_weighted_{outcome_name}_risk_pct"] = weighted * 100 if pd.notna(weighted) else np.nan
    # Calculate a weighted mean.
    if "observed_catheter_in_intervals" in episode_df.columns:
        # Calculate a weighted mean.
        row["ipw_weighted_mean_catheter_in_intervals"] = weighted_mean(
            episode_df["observed_catheter_in_intervals"],
            weights,
        )
        row["ipw_weighted_mean_catheter_in_interval_rows"] = row[
            "ipw_weighted_mean_catheter_in_intervals"
        ]
    # Calculate a weighted mean.
    if "observed_catheter_exposure_days" in episode_df.columns:
        # Calculate a weighted mean.
        row["ipw_weighted_mean_catheter_exposure_days"] = weighted_mean(
            episode_df["observed_catheter_exposure_days"],
            weights,
        )
    return row


def build_clipping_sensitivity(
    policy_episode_df,
    clip_lower,
    clip_upper,
):
    """Build clipping-sensitivity output."""
    rows = []
    group_cols = ["policy_name", "policy_remove_day"]
    # Return positive finite weights.
    for policy_values, episode_df in policy_episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_remove_day = policy_values
        default_weights = pd.to_numeric(episode_df[WEIGHT_COL], errors="coerce")
        unclipped_weights = pd.to_numeric(episode_df[UNCLIPPED_WEIGHT_COL], errors="coerce")
        # Return positive finite weights.
        valid_default = valid_weight_series(default_weights)
        p99 = float(valid_default.quantile(0.99)) if len(valid_default) else np.nan

        # Build one clipping-sensitivity row.
        rows.append(
            clipping_estimate_row(
                episode_df,
                policy_name,
                policy_remove_day,
                "unclipped_support_where_safe",
                unclipped_weights,
                np.nan,
            )
        )
        # Build one clipping-sensitivity row.
        rows.append(
            clipping_estimate_row(
                episode_df,
                policy_name,
                policy_remove_day,
                "row_support_clipped",
                default_weights,
                np.nan,
                clip_lower,
                clip_upper,
            )
        )
        # Build one clipping-sensitivity row.
        for label, upper in [
            ("upper_weight_clipped_at_p99", p99),
            ("upper_weight_clipped_at_30", 30.0),
            ("upper_weight_clipped_at_20", 20.0),
        ]:
            clipped = default_weights.clip(upper=upper) if pd.notna(upper) else default_weights
            # Build one clipping-sensitivity row.
            rows.append(
                clipping_estimate_row(
                    episode_df,
                    policy_name,
                    policy_remove_day,
                    label,
                    clipped,
                    upper,
                    clip_lower,
                    clip_upper,
                )
            )
    return pd.DataFrame(rows)


def metadata_payload(
    args,
    output_paths,
    joined_df,
    episode_all,
    policy_episode_df,
):
    """Build run metadata."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "sequential_ipw",
        "current_practice_comparator_type": "observed_weight_one",
        "clipping_bounds": {"clip_lower": args.clip_lower, "clip_upper": args.clip_upper},
        "input_paths": {
            "policy_panel": str(args.policy_panel),
            "scored_panel": str(args.scored_panel),
        },
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "number_of_policies": int(joined_df["policy_name"].nunique()),
        "number_of_patients": int(joined_df["subject_id"].nunique()),
        "number_of_policy_episode_rows": int(len(episode_all)),
        "number_of_adherent_policy_episode_rows": int(len(policy_episode_df)),
        "target_policy_timeline_reconstructed_in_script": False,
        "target_policy_timing_source": pec.TARGET_POLICY_TIMING_SOURCE,
        "target_policy_timeline_helper": pec.TARGET_POLICY_TIMELINE_HELPER,
        "target_policy_timeline_semantics": pec.TARGET_POLICY_TIMELINE_SEMANTICS,
        "duration_semantics": {
            "period_duration_days": "period_end - period_start in days",
            "catheter_exposure_days": "sum of period_duration_days where catheter_state == in",
            "max_reasonable_period_duration_days": pec.MAX_REASONABLE_PERIOD_DURATION_DAYS,
            "n_long_period_duration_rows": int(joined_df.get("period_duration_long_flag", pd.Series(0)).sum()),
        },
        "catheter_count_semantics": {
            "observed_catheter_in_intervals": (
                "count of observed catheter-in interval rows on the observed grid; "
                "not necessarily one row per patient-day because transition days may be "
                "split into in and out intervals"
            ),
            "observed_catheter_exposure_days": (
                "sum of period_duration_days where catheter_state == in; "
                "preferred exposure measure for interpretation"
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
        "bootstrap": pec.bootstrap_metadata(args, joined_df.columns),
        "methodological_limitations": [
            "This script estimates IPW values only.",
            "It does not run g-formula, AIPW, DML, DR-Learner, TMLE, or LTMLE.",
            "Uncertainty intervals are not calculated; bootstrap arguments are recorded only.",
        ],
    }


# Main

def print_summary(
    policy_panel_path,
    scored_panel_path,
    episode_all,
    policy_episode_df,
    current_practice_episode_df,
    summary_path,
    episodes_path,
    weight_diagnostics_path,
    support_diagnostics_path,
    clipping_sensitivity_path,
    current_practice_path,
    metadata_path,
):
    """Print a concise run summary."""
    n_policies = int(episode_all["policy_name"].nunique()) if "policy_name" in episode_all.columns else 0
    zero_adherent = (
        episode_all.groupby("policy_name")["episode_adherent_to_policy"].sum().loc[lambda s: s == 0].index.tolist()
    )

    print()
    print("--- IPW POLICY EVALUATION COMPLETE ---")
    print(f"Policy-intervention panel: {policy_panel_path}")
    print(f"Scored nuisance panel: {scored_panel_path}")
    print(f"Candidate policies: {n_policies:,}")
    print(f"Current-practice episodes: {len(current_practice_episode_df):,}")
    print(f"Adherent target-policy episode rows: {len(policy_episode_df):,}")
    if zero_adherent:
        print(f"WARNING: policies with no adherent episodes: {zero_adherent}")
    print(f"Saved policy summary: {summary_path}")
    print(f"Saved episode outcomes: {episodes_path}")
    print(f"Saved weight diagnostics: {weight_diagnostics_path}")
    print(f"Saved support diagnostics: {support_diagnostics_path}")
    print(f"Saved clipping sensitivity: {clipping_sensitivity_path}")
    print(f"Saved current-practice episodes: {current_practice_path}")
    print(f"Saved metadata: {metadata_path}")


def main():
    """Run the script workflow."""
    # Parse command-line arguments.
    args = parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    # Resolve an output file path.
    summary_path = resolve_output_path(args.outdir, args.output_summary)
    # Resolve an output file path.
    episodes_path = resolve_output_path(args.outdir, args.output_episodes)
    # Resolve an output file path.
    weight_diagnostics_path = resolve_output_path(args.outdir, args.output_weight_diagnostics)
    # Resolve an output file path.
    support_diagnostics_path = resolve_output_path(args.outdir, args.output_support_diagnostics)
    # Resolve an output file path.
    clipping_sensitivity_path = resolve_output_path(args.outdir, args.output_clipping_sensitivity)
    # Resolve an output file path.
    current_practice_path = resolve_output_path(args.outdir, args.output_current_practice)
    # Resolve an output file path.
    metadata_path = resolve_output_path(args.outdir, args.output_metadata)

    # Load and validate the policy panel.
    policy_df = load_policy_panel(args.policy_panel)
    # Load and validate the scored nuisance panel.
    scored_df = load_scored_panel(args.scored_panel)

    # Join nuisance scores to policy rows.
    joined_df = join_scored_panel(policy_df, scored_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined IPW policy rows")
    joined_df = pec.add_observed_icu_exit_alive_period(joined_df)
    # Add row-level IPW quantities.
    joined_df = add_ipw_row_quantities(joined_df, args.clip_lower, args.clip_upper)
    # Add cumulative policy-adherence flags.
    joined_df = add_adherence(joined_df)

    # Build support diagnostic output.
    support_diagnostics_df = build_support_diagnostics(joined_df)
    # Build adherent policy-episode outcomes.
    episode_all, policy_episode_df = build_policy_episode_panel(
        joined_df,
        args.zero_applicable_policy_episodes,
    )
    # Order episode-level output columns.
    episode_all = order_episode_columns(episode_all)
    # Order episode-level output columns.
    policy_episode_df = order_episode_columns(policy_episode_df)

    if policy_episode_df.empty:
        raise ValueError(
            "No target policies have adherent episodes after IPW adherence "
            "assessment. Support and weight diagnostics cannot produce policy estimates."
        )

    # Build observed current-practice episode outcomes.
    current_practice_episode_df = build_current_practice_episode_panel(scored_df, policy_df)
    output_episode_df = pd.concat(
        [current_practice_episode_df, policy_episode_df],
        ignore_index=True,
        sort=False,
    )

    # Build policy-level summary estimates.
    summary_df = build_policy_summary(
        episode_all,
        policy_episode_df,
        current_practice_episode_df,
    )
    # Build weight diagnostic output.
    weight_diagnostics_df = build_weight_diagnostics(episode_all, policy_episode_df)
    # Build clipping-sensitivity output.
    clipping_sensitivity_df = build_clipping_sensitivity(
        policy_episode_df,
        args.clip_lower,
        args.clip_upper,
    )
    summary_df = pec.add_overlap_quality_flags(
        summary_df,
        support_diagnostics=support_diagnostics_df,
        weight_diagnostics=weight_diagnostics_df,
        current_practice_label=CURRENT_PRACTICE_LABEL,
    )
    output_paths = {
        "summary": summary_path,
        "episodes": episodes_path,
        "weight_diagnostics": weight_diagnostics_path,
        "support_diagnostics": support_diagnostics_path,
        "clipping_sensitivity": clipping_sensitivity_path,
        "current_practice": current_practice_path,
        "metadata": metadata_path,
    }

    # Save a data frame as CSV.
    save_df(output_episode_df, episodes_path)
    # Save a data frame as CSV.
    save_df(summary_df, summary_path)
    # Save a data frame as CSV.
    save_df(weight_diagnostics_df, weight_diagnostics_path)
    # Save a data frame as CSV.
    save_df(support_diagnostics_df, support_diagnostics_path)
    # Save a data frame as CSV.
    save_df(clipping_sensitivity_df, clipping_sensitivity_path)
    # Save a data frame as CSV.
    save_df(current_practice_episode_df, current_practice_path)
    # Build run metadata.
    pec.save_json(
        metadata_payload(args, output_paths, joined_df, episode_all, policy_episode_df),
        metadata_path,
    )

    # Print a concise run summary.
    print_summary(
        args.policy_panel,
        args.scored_panel,
        episode_all,
        policy_episode_df,
        current_practice_episode_df,
        summary_path,
        episodes_path,
        weight_diagnostics_path,
        support_diagnostics_path,
        clipping_sensitivity_path,
        current_practice_path,
        metadata_path,
    )


# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
