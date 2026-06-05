#!/usr/bin/env python3
"""
Calculate IPW-weighted policy outcomes from IPW-ready catheter policy panels.

Expected input: a patient-day panel already restricted/flagged for one or more policies, with:
- catheter_episode_id
- policy_name / policy_remove_day, for multi-policy panels
- episode_ipw_weight
- episode_matches_policy or followed_policy_so_far, if present
- outcome columns such as episode_cauti / episode_reinsertion or cauti_in_period / reinsertion_in_period

Default example:
    python calculate_ipw_policy_outcomes.py

Default paths:
    Input:
        artifacts/policy_eval/ipw_policy_remove_days_1_to_5_panel.csv
        artifacts/nuisance_models/scored_panel.csv
    Outputs:
        artifacts/policy_eval/ipw_policy_remove_days_1_to_5_outcomes_summary.csv
        artifacts/policy_eval/ipw_policy_remove_days_1_to_5_episode_outcomes.csv
        artifacts/policy_eval/ipw_policy_remove_days_1_to_5_weight_diagnostics.csv
        artifacts/policy_eval/ipw_policy_remove_days_1_to_5_clipping_sensitivity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


EPISODE_ID_COL = "catheter_episode_id"
WEIGHT_COL = "episode_ipw_weight"
POLICY_COLS = ["policy_name", "policy_remove_day"]
EPISODE_KEY_COLS = ["subject_id", "hadm_id", "stay_id", "inserted", "removed"]
CURRENT_PRACTICE_LABEL = "current_practice"

REPO_ROOT = Path(__file__).resolve().parent
OUTDIR = REPO_ROOT / "artifacts" / "policy_eval"
NUISANCE_MODEL_DIR = REPO_ROOT / "artifacts" / "nuisance_models"

DEFAULT_INPUT_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_panel.csv"
DEFAULT_SCORED_PANEL_PATH = NUISANCE_MODEL_DIR / "scored_panel.csv"
DEFAULT_OUTPUT_SUMMARY_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_outcomes_summary.csv"
DEFAULT_OUTPUT_EPISODES_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_episode_outcomes.csv"
DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_weight_diagnostics.csv"
DEFAULT_OUTPUT_CLIPPING_SENSITIVITY_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_clipping_sensitivity.csv"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Return sum(w*y) / sum(w), ignoring rows with missing values or weights."""
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def valid_weight_series(weights: pd.Series) -> pd.Series:
    """Return positive finite weights as numeric values."""
    weights = pd.to_numeric(weights, errors="coerce")
    return weights[weights.notna() & np.isfinite(weights) & (weights > 0)]


def effective_sample_size(weights: pd.Series) -> float:
    weights = valid_weight_series(weights)
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


def first_non_null(series: pd.Series):
    """Take the first non-null value in a group."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series: pd.Series) -> int:
    """Collapse repeated patient-day outcome flags to one episode-level any-event flag."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return int(numeric.max() > 0)


def available_policy_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in POLICY_COLS if col in df.columns]


def policy_episode_group_cols(df: pd.DataFrame) -> list[str]:
    return [*available_policy_cols(df), EPISODE_ID_COL]


def policy_metadata(group_values, group_cols: list[str]) -> dict:
    if not isinstance(group_values, tuple):
        group_values = (group_values,)
    return dict(zip(group_cols, group_values))


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required {context} column(s): {missing}")


def choose_outcome_column(df: pd.DataFrame, episode_col: str, period_col: str) -> Optional[str]:
    """
    Prefer precomputed episode-level outcome columns if present.
    Fall back to patient-day period outcome flags if not.
    """
    if episode_col in df.columns:
        return episode_col
    if period_col in df.columns:
        return period_col
    return None


def add_catheter_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    if EPISODE_ID_COL in df.columns:
        return df

    require_columns(df, EPISODE_KEY_COLS, "episode-key")
    df = df.copy()
    df[EPISODE_ID_COL] = df.groupby(EPISODE_KEY_COLS, sort=False).ngroup() + 1
    return df


def normalise_policy_remove_day(df: pd.DataFrame) -> pd.DataFrame:
    if "policy_remove_day" not in df.columns:
        return df
    df = df.copy()
    df["policy_remove_day"] = pd.to_numeric(df["policy_remove_day"], errors="coerce").astype("Int64")
    return df


def build_episode_level_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the IPW patient-day panel to one row per policy and catheter episode."""
    required = {EPISODE_ID_COL, WEIGHT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required column(s): {sorted(missing)}")

    # Keep only episodes that matched the policy, if the column is present.
    # The panel produced by the previous script should already be restricted,
    # but this makes the outcome script safer to reuse.
    if "episode_matches_policy" in df.columns:
        df = df[pd.to_numeric(df["episode_matches_policy"], errors="coerce").fillna(0).astype(int) == 1].copy()

    if df.empty:
        raise ValueError("No policy-matching episodes remain after filtering.")

    cauti_source = choose_outcome_column(df, "episode_cauti", "cauti_in_period")
    reins_source = choose_outcome_column(df, "episode_reinsertion", "reinsertion_in_period")

    if cauti_source is None:
        raise ValueError("Could not find either 'episode_cauti' or 'cauti_in_period'.")
    if reins_source is None:
        raise ValueError("Could not find either 'episode_reinsertion' or 'reinsertion_in_period'.")

    group_cols = policy_episode_group_cols(df)
    aggregations = {
        WEIGHT_COL: first_non_null,
        cauti_source: max_binary,
        reins_source: max_binary,
    }

    optional_first_cols = [
        "policy_name",
        "policy_remove_day",
        "subject_id",
        "hadm_id",
        "stay_id",
        "split",
        "inserted",
        "removed",
        "reinsertion_time",
        "episode_end_reason",
    ]
    group_col_set = set(group_cols)
    for col in optional_first_cols:
        if col in df.columns and col not in group_col_set:
            aggregations[col] = first_non_null

    # Useful descriptive episode-level quantities.
    if "catheter_state" in df.columns:
        in_rows = df[df["catheter_state"].astype(str).str.lower().eq("in")]
        catheter_days = in_rows.groupby(group_cols).size().reset_index(name="observed_catheter_day_rows")
    else:
        catheter_days = None

    if "at_risk_cauti" in df.columns:
        cauti_risk_rows = df.groupby(group_cols)["at_risk_cauti"].sum().reset_index(name="cauti_at_risk_rows")
    else:
        cauti_risk_rows = None

    if "at_risk_reinsertion" in df.columns:
        reins_risk_rows = df.groupby(group_cols)["at_risk_reinsertion"].sum().reset_index(name="reinsertion_at_risk_rows")
    else:
        reins_risk_rows = None

    episode_df = df.groupby(group_cols, as_index=False).agg(aggregations)
    episode_df = episode_df.rename(
        columns={
            cauti_source: "any_cauti",
            reins_source: "any_recatheterisation",
        }
    )

    if catheter_days is not None:
        episode_df = episode_df.merge(catheter_days, on=group_cols, how="left")
    if cauti_risk_rows is not None:
        episode_df = episode_df.merge(cauti_risk_rows, on=group_cols, how="left")
    if reins_risk_rows is not None:
        episode_df = episode_df.merge(reins_risk_rows, on=group_cols, how="left")

    for col in ["observed_catheter_day_rows", "cauti_at_risk_rows", "reinsertion_at_risk_rows"]:
        if col not in episode_df.columns:
            episode_df[col] = 0
        episode_df[col] = episode_df[col].fillna(0).astype(int)

    return episode_df


def build_current_practice_episode_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the original observed scored panel to one row per catheter episode."""
    df = add_catheter_episode_id(df.copy())

    cauti_source = choose_outcome_column(df, "episode_cauti", "cauti_in_period")
    reins_source = choose_outcome_column(df, "episode_reinsertion", "reinsertion_in_period")

    if cauti_source is None:
        raise ValueError("Could not find either 'episode_cauti' or 'cauti_in_period' in scored panel.")
    if reins_source is None:
        raise ValueError("Could not find either 'episode_reinsertion' or 'reinsertion_in_period' in scored panel.")

    group_cols = [EPISODE_ID_COL]
    aggregations = {
        cauti_source: max_binary,
        reins_source: max_binary,
    }

    optional_first_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "split",
        "inserted",
        "removed",
        "reinsertion_time",
        "episode_end_reason",
    ]
    for col in optional_first_cols:
        if col in df.columns:
            aggregations[col] = first_non_null

    if "catheter_state" in df.columns:
        in_rows = df[df["catheter_state"].astype(str).str.lower().eq("in")]
        catheter_days = in_rows.groupby(group_cols).size().reset_index(name="observed_catheter_day_rows")
    else:
        catheter_days = None

    if "at_risk_cauti" in df.columns:
        cauti_risk_rows = df.groupby(group_cols)["at_risk_cauti"].sum().reset_index(name="cauti_at_risk_rows")
    else:
        cauti_risk_rows = None

    if "at_risk_reinsertion" in df.columns:
        reins_risk_rows = df.groupby(group_cols)["at_risk_reinsertion"].sum().reset_index(name="reinsertion_at_risk_rows")
    else:
        reins_risk_rows = None

    episode_df = df.groupby(group_cols, as_index=False).agg(aggregations)
    episode_df = episode_df.rename(
        columns={
            cauti_source: "any_cauti",
            reins_source: "any_recatheterisation",
        }
    )

    if catheter_days is not None:
        episode_df = episode_df.merge(catheter_days, on=group_cols, how="left")
    if cauti_risk_rows is not None:
        episode_df = episode_df.merge(cauti_risk_rows, on=group_cols, how="left")
    if reins_risk_rows is not None:
        episode_df = episode_df.merge(reins_risk_rows, on=group_cols, how="left")

    for col in ["observed_catheter_day_rows", "cauti_at_risk_rows", "reinsertion_at_risk_rows"]:
        if col not in episode_df.columns:
            episode_df[col] = 0
        episode_df[col] = episode_df[col].fillna(0).astype(int)

    episode_df["policy_name"] = CURRENT_PRACTICE_LABEL
    episode_df["policy_remove_day"] = pd.NA
    episode_df[WEIGHT_COL] = 1.0

    ordered_cols = [
        "policy_name",
        "policy_remove_day",
        EPISODE_ID_COL,
        WEIGHT_COL,
        *[col for col in episode_df.columns if col not in {"policy_name", "policy_remove_day", EPISODE_ID_COL, WEIGHT_COL}],
    ]
    return episode_df[ordered_cols]


def summarise_group(episode_df: pd.DataFrame, label: str, metadata: Optional[dict] = None) -> dict:
    weights = episode_df[WEIGHT_COL]
    row = {
        "group": label,
        "n_episodes": int(len(episode_df)),
        "sum_weights": float(pd.to_numeric(weights, errors="coerce").sum()),
        "mean_weight": float(pd.to_numeric(weights, errors="coerce").mean()),
        "max_weight": float(pd.to_numeric(weights, errors="coerce").max()),
        "unweighted_cauti_risk": float(pd.to_numeric(episode_df["any_cauti"], errors="coerce").mean()),
        "ipw_weighted_cauti_risk": weighted_mean(episode_df["any_cauti"], weights),
        "unweighted_recatheterisation_risk": float(pd.to_numeric(episode_df["any_recatheterisation"], errors="coerce").mean()),
        "ipw_weighted_recatheterisation_risk": weighted_mean(episode_df["any_recatheterisation"], weights),
        "unweighted_mean_catheter_day_rows": float(pd.to_numeric(episode_df["observed_catheter_day_rows"], errors="coerce").mean()),
        "ipw_weighted_mean_catheter_day_rows": weighted_mean(episode_df["observed_catheter_day_rows"], weights),
    }
    if metadata:
        row = {**metadata, **row}
    return row


def build_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_cols = available_policy_cols(episode_df)

    if policy_cols:
        for policy_values, policy_df in episode_df.groupby(policy_cols, dropna=False, sort=False):
            metadata = policy_metadata(policy_values, policy_cols)
            rows.append(summarise_group(policy_df, "all", metadata))

            if "split" in episode_df.columns:
                for split_value, split_df in policy_df.groupby("split", dropna=False):
                    rows.append(summarise_group(split_df, f"split={split_value}", metadata))
    else:
        rows.append(summarise_group(episode_df, "all"))

        if "split" in episode_df.columns:
            for split_value, split_df in episode_df.groupby("split", dropna=False):
                rows.append(summarise_group(split_df, f"split={split_value}"))

    summary = pd.DataFrame(rows)

    # Add percentage versions for the two main risks.
    for col in ["ipw_weighted_cauti_risk", "ipw_weighted_recatheterisation_risk"]:
        summary[col + "_pct"] = summary[col] * 100

    return summary


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.where(denominator != 0) / denominator.where(denominator != 0)


def add_current_practice_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    if "policy_name" not in summary.columns:
        return summary

    baseline = summary[summary["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
    if baseline.empty:
        return summary

    baseline_cols = [
        "group",
        "ipw_weighted_cauti_risk",
        "ipw_weighted_recatheterisation_risk",
        "ipw_weighted_mean_catheter_day_rows",
    ]
    baseline = baseline[baseline_cols].rename(
        columns={
            "ipw_weighted_cauti_risk": "current_practice_cauti_risk",
            "ipw_weighted_recatheterisation_risk": "current_practice_recatheterisation_risk",
            "ipw_weighted_mean_catheter_day_rows": "current_practice_mean_catheter_day_rows",
        }
    )

    out = summary.merge(baseline, on="group", how="left")

    out["cauti_risk_difference_vs_current_practice"] = (
        out["ipw_weighted_cauti_risk"] - out["current_practice_cauti_risk"]
    )
    out["cauti_risk_difference_pct_points_vs_current_practice"] = (
        out["cauti_risk_difference_vs_current_practice"] * 100
    )
    out["cauti_risk_ratio_vs_current_practice"] = safe_ratio(
        out["ipw_weighted_cauti_risk"],
        out["current_practice_cauti_risk"],
    )

    out["recatheterisation_risk_difference_vs_current_practice"] = (
        out["ipw_weighted_recatheterisation_risk"] - out["current_practice_recatheterisation_risk"]
    )
    out["recatheterisation_risk_difference_pct_points_vs_current_practice"] = (
        out["recatheterisation_risk_difference_vs_current_practice"] * 100
    )
    out["recatheterisation_risk_ratio_vs_current_practice"] = safe_ratio(
        out["ipw_weighted_recatheterisation_risk"],
        out["current_practice_recatheterisation_risk"],
    )

    out["mean_catheter_day_rows_difference_vs_current_practice"] = (
        out["ipw_weighted_mean_catheter_day_rows"] - out["current_practice_mean_catheter_day_rows"]
    )

    return out


def summarise_weight_group(episode_df: pd.DataFrame, label: str, metadata: Optional[dict] = None) -> dict:
    weights = valid_weight_series(episode_df[WEIGHT_COL])

    if weights.empty:
        row = {
            "group": label,
            "min_weight": np.nan,
            "median_weight": np.nan,
            "p90_weight": np.nan,
            "p95_weight": np.nan,
            "p99_weight": np.nan,
            "max_weight": np.nan,
            "effective_sample_size": np.nan,
        }
        if metadata:
            row = {**metadata, **row}
        return row

    row = {
        "group": label,
        "min_weight": float(weights.min()),
        "median_weight": float(weights.quantile(0.50)),
        "p90_weight": float(weights.quantile(0.90)),
        "p95_weight": float(weights.quantile(0.95)),
        "p99_weight": float(weights.quantile(0.99)),
        "max_weight": float(weights.max()),
        "effective_sample_size": effective_sample_size(weights),
    }
    if metadata:
        row = {**metadata, **row}
    return row


def build_weight_diagnostics(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_cols = available_policy_cols(episode_df)

    if policy_cols:
        for policy_values, policy_df in episode_df.groupby(policy_cols, dropna=False, sort=False):
            metadata = policy_metadata(policy_values, policy_cols)
            rows.append(summarise_weight_group(policy_df, "all", metadata))

            if "split" in episode_df.columns:
                for split_value, split_df in policy_df.groupby("split", dropna=False):
                    rows.append(summarise_weight_group(split_df, f"split={split_value}", metadata))
    else:
        rows.append(summarise_weight_group(episode_df, "all"))

        if "split" in episode_df.columns:
            for split_value, split_df in episode_df.groupby("split", dropna=False):
                rows.append(summarise_weight_group(split_df, f"split={split_value}"))

    return pd.DataFrame(rows)


def clipped_weights(weights: pd.Series, upper: Optional[float]) -> pd.Series:
    weights = pd.to_numeric(weights, errors="coerce")
    if upper is None or not np.isfinite(upper):
        return weights
    return weights.clip(upper=upper)


def summarise_clipping_rule(
    episode_df: pd.DataFrame,
    label: str,
    upper: Optional[float],
    metadata: Optional[dict] = None,
) -> dict:
    weights = clipped_weights(episode_df[WEIGHT_COL], upper)
    cauti_risk = weighted_mean(episode_df["any_cauti"], weights)
    recatheterisation_risk = weighted_mean(episode_df["any_recatheterisation"], weights)

    row = {
        "clipping_rule": label,
        "clip_upper": upper if upper is not None else np.nan,
        "n_episodes": int(len(episode_df)),
        "sum_weights": float(valid_weight_series(weights).sum()),
        "effective_sample_size": effective_sample_size(weights),
        "ipw_weighted_cauti_risk": cauti_risk,
        "ipw_weighted_cauti_risk_pct": cauti_risk * 100,
        "ipw_weighted_recatheterisation_risk": recatheterisation_risk,
        "ipw_weighted_recatheterisation_risk_pct": recatheterisation_risk * 100,
    }
    if metadata:
        row = {**metadata, **row}
    return row


def clipping_sensitivity_rows(episode_df: pd.DataFrame, metadata: Optional[dict] = None) -> list[dict]:
    weights = valid_weight_series(episode_df[WEIGHT_COL])
    p99_upper = float(weights.quantile(0.99)) if not weights.empty else np.nan

    return [
        summarise_clipping_rule(episode_df, "unclipped", None, metadata),
        summarise_clipping_rule(episode_df, "clip_at_p99", p99_upper, metadata),
        summarise_clipping_rule(episode_df, "clip_at_30", 30.0, metadata),
        summarise_clipping_rule(episode_df, "clip_at_20", 20.0, metadata),
    ]


def build_clipping_sensitivity(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_cols = available_policy_cols(episode_df)

    if policy_cols:
        for policy_values, policy_df in episode_df.groupby(policy_cols, dropna=False, sort=False):
            metadata = policy_metadata(policy_values, policy_cols)
            rows.extend(clipping_sensitivity_rows(policy_df, metadata))
    else:
        rows.extend(clipping_sensitivity_rows(episode_df))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate IPW-weighted outcomes for one or more policy panels.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to IPW-ready policy panel CSV.")
    parser.add_argument("--scored-panel", default=DEFAULT_SCORED_PANEL_PATH, help="Path to original scored panel CSV for current-practice baseline.")
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY_PATH, help="Output CSV for policy outcome summary.")
    parser.add_argument("--output-episodes", default=DEFAULT_OUTPUT_EPISODES_PATH, help="Output CSV for current-practice episodes and one-row-per-policy-episode data.")
    parser.add_argument(
        "--output-weight-diagnostics",
        default=DEFAULT_OUTPUT_WEIGHT_DIAGNOSTICS_PATH,
        help="Output CSV for IPW weight diagnostics.",
    )
    parser.add_argument(
        "--output-clipping-sensitivity",
        default=DEFAULT_OUTPUT_CLIPPING_SENSITIVITY_PATH,
        help="Output CSV for clipped-weight sensitivity estimates.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    scored_panel_path = Path(args.scored_panel)
    output_summary_path = Path(args.output_summary)
    output_episodes_path = Path(args.output_episodes)
    output_weight_diagnostics_path = Path(args.output_weight_diagnostics)
    output_clipping_sensitivity_path = Path(args.output_clipping_sensitivity)

    df = pd.read_csv(input_path)
    policy_episode_df = normalise_policy_remove_day(build_episode_level_panel(df))

    scored_df = pd.read_csv(scored_panel_path, low_memory=False)
    current_practice_episode_df = normalise_policy_remove_day(build_current_practice_episode_panel(scored_df))

    episode_df = normalise_policy_remove_day(pd.concat([current_practice_episode_df, policy_episode_df], ignore_index=True))
    summary_df = add_current_practice_comparisons(build_summary(episode_df))
    weight_diagnostics_df = build_weight_diagnostics(policy_episode_df)
    clipping_sensitivity_df = build_clipping_sensitivity(policy_episode_df)

    output_summary_path.parent.mkdir(exist_ok=True, parents=True)
    output_episodes_path.parent.mkdir(exist_ok=True, parents=True)
    output_weight_diagnostics_path.parent.mkdir(exist_ok=True, parents=True)
    output_clipping_sensitivity_path.parent.mkdir(exist_ok=True, parents=True)

    episode_df.to_csv(output_episodes_path, index=False)
    summary_df.to_csv(output_summary_path, index=False)
    weight_diagnostics_df.to_csv(output_weight_diagnostics_path, index=False)
    clipping_sensitivity_df.to_csv(output_clipping_sensitivity_path, index=False)

    print("IPW policy outcome summary")
    print("==========================")
    print(f"Input panel: {input_path}")
    print(f"Current-practice scored panel: {scored_panel_path}")
    print(f"Current-practice episodes analysed: {len(current_practice_episode_df):,}")
    print(f"Policy-episodes analysed: {len(policy_episode_df):,}")
    if "policy_name" in policy_episode_df.columns:
        print(f"IPW policies analysed: {policy_episode_df['policy_name'].nunique():,}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print("IPW weight diagnostics")
    print("======================")
    print(weight_diagnostics_df.to_string(index=False))
    print()
    print("Clipped-weight sensitivity")
    print("==========================")
    print(clipping_sensitivity_df.to_string(index=False))
    print()
    print(f"Saved episode-level data to: {output_episodes_path}")
    print(f"Saved summary to: {output_summary_path}")
    print(f"Saved weight diagnostics to: {output_weight_diagnostics_path}")
    print(f"Saved clipping sensitivity to: {output_clipping_sensitivity_path}")


if __name__ == "__main__":
    main()
