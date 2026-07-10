"""Shared utilities for catheter-removal policy-evaluation scripts.

The three estimator scripts keep separate estimator-specific logic, but share
basic definitions for period duration, ICU-exit-alive outcomes, bootstrap
metadata placeholders, current-practice comparisons, and overlap-quality flags.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAX_REASONABLE_PERIOD_DURATION_DAYS = 7.0
LOW_ADHERENCE_THRESHOLD = 0.05
LOW_ESS_MIN = 100.0
LOW_ESS_FRACTION = 0.10
LOW_SUPPORT_PCT_BELOW_005_THRESHOLD = 0.10
EXTREME_WEIGHT_P99_THRESHOLD = 30.0
EXTREME_WEIGHT_MAX_THRESHOLD = 100.0
RESOLVED_TIMELINE_COLUMNS = [
    "row_order_within_episode_day",
    "policy_catheter_state",
    "policy_action_resolved",
    "policy_action_remove_resolved",
    "policy_periods_in",
    "policy_periods_out",
]
TARGET_POLICY_TIMING_SOURCE = "policy_intervention_panel_long.csv resolved target-policy timeline"
TARGET_POLICY_TIMELINE_HELPER = "policy_eval_common.add_fixed_day_target_policy_timeline"
TARGET_POLICY_TIMELINE_SEMANTICS = (
    "fixed-day removal: before removal day is in/keep; first row on removal "
    "day is in/remove; later rows on the same removal day are out/out with "
    "policy_periods_out = 0; later days are out/out"
)


def add_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    """Add common clustered-bootstrap arguments.

    Bootstrap confidence intervals are intentionally not implemented in these
    scripts yet. The arguments are accepted and recorded in metadata so future
    patient-clustered uncertainty can be added without changing the CLI.
    """
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help="Number of clustered bootstrap resamples. Default: 0 (not run).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260708,
        help="Random seed for future clustered bootstrap. Default: 20260708.",
    )
    parser.add_argument(
        "--cluster-col",
        default="subject_id",
        help="Cluster column for future uncertainty estimation. Default: subject_id.",
    )


def bootstrap_metadata(args: argparse.Namespace, available_columns: Iterable[str]) -> dict:
    cluster_col = getattr(args, "cluster_col", "subject_id")
    n_bootstrap = int(getattr(args, "n_bootstrap", 0))
    return {
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": int(getattr(args, "bootstrap_seed", 20260708)),
        "cluster_col": cluster_col,
        "cluster_col_available": cluster_col in set(available_columns),
        "bootstrap_ci_calculated": False,
        "bootstrap_status": (
            "not_requested"
            if n_bootstrap == 0
            else "requested_but_not_implemented_no_confidence_intervals_written"
        ),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    import json

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def add_period_duration_days(
    df: pd.DataFrame,
    *,
    context: str,
    max_reasonable_days: float = MAX_REASONABLE_PERIOD_DURATION_DAYS,
) -> pd.DataFrame:
    """Add period_duration_days and validate the row interval.

    Non-positive or unparsable durations are fatal because all exposure
    estimands depend on valid time intervals. Very long intervals are flagged
    and warned about rather than silently ignored.
    """
    required = ["period_start", "period_end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot calculate period durations for {context}; missing columns: {missing}")

    out = df.copy()
    start = pd.to_datetime(out["period_start"], errors="coerce")
    end = pd.to_datetime(out["period_end"], errors="coerce")
    duration_days = (end - start).dt.total_seconds() / 86400.0
    invalid = start.isna() | end.isna() | duration_days.isna() | ~np.isfinite(duration_days) | duration_days.le(0)
    if invalid.any():
        examples = out.loc[invalid, ["period_start", "period_end"]].head(10)
        raise ValueError(
            f"Invalid period durations found while preparing {context}. "
            "period_start/period_end must parse and period_end must be after period_start. "
            f"Examples:\n{examples}"
        )

    out["period_duration_days"] = duration_days.astype(float)
    out["period_duration_long_flag"] = out["period_duration_days"].gt(max_reasonable_days).astype(int)
    n_long = int(out["period_duration_long_flag"].sum())
    if n_long:
        print(
            f"WARNING: {context} has {n_long:,} rows with period_duration_days "
            f"> {max_reasonable_days}. They are retained and flagged.",
            flush=True,
        )
    return out


def add_observed_icu_exit_alive_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add observed_icu_exit_alive_in_period with death taking precedence."""
    out = df.copy()
    death = (
        pd.to_numeric(out["death_in_period"], errors="coerce").fillna(0)
        if "death_in_period" in out.columns
        else pd.Series(0, index=out.index)
    )
    icu_exit = (
        pd.to_numeric(out["icu_end_in_period"], errors="coerce").fillna(0)
        if "icu_end_in_period" in out.columns
        else pd.Series(0, index=out.index)
    )
    out["death_and_icu_exit_same_period"] = ((death.eq(1)) & (icu_exit.eq(1))).astype(int)
    out["observed_icu_exit_alive_in_period"] = ((icu_exit.eq(1)) & death.ne(1)).astype(int)
    return out


def add_fixed_day_target_policy_timeline(
    df: pd.DataFrame,
    *,
    episode_id_col: str,
    action_col: str = "policy_action_resolved",
    action_remove_col: str = "policy_action_remove_resolved",
    policy_name_col: str = "policy_name",
    policy_remove_day_col: str = "policy_remove_day",
    episode_day_col: str = "episode_day_since_insertion",
    period_start_col: str = "period_start",
    period_end_col: str = "period_end",
    decision_row_id_col: str = "decision_row_id",
    state_col: str = "policy_catheter_state",
    periods_in_col: str = "policy_periods_in",
    periods_out_col: str = "policy_periods_out",
) -> pd.DataFrame:
    """Apply the shared fixed-day catheter-removal target-policy timeline.

    Transition-day convention:
      * before policy removal day: IN / keep;
      * first row on policy removal day: IN / remove;
      * later rows on policy removal day: OUT / out with periods_out = 0;
      * after policy removal day: OUT / out.

    This handles observed-grid panels where a single calendar/episode day can
    contain split within-day intervals, for example an IN row before removal and
    an OUT row after removal.
    """
    required = [
        policy_name_col,
        episode_id_col,
        episode_day_col,
        policy_remove_day_col,
        period_start_col,
        period_end_col,
        decision_row_id_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build fixed-day target-policy timeline; missing columns: {missing}")

    out = df.copy()
    out = out.sort_values(
        [
            policy_name_col,
            episode_id_col,
            episode_day_col,
            period_start_col,
            period_end_col,
            decision_row_id_col,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    out["row_order_within_episode_day"] = (
        out.groupby(
            [policy_name_col, episode_id_col, episode_day_col],
            dropna=False,
            sort=False,
        ).cumcount()
        + 1
    )

    day = pd.to_numeric(out[episode_day_col], errors="coerce")
    remove_day = pd.to_numeric(out[policy_remove_day_col], errors="coerce")
    before_remove = day.lt(remove_day)
    on_remove = day.eq(remove_day)
    first_row_on_remove_day = on_remove & out["row_order_within_episode_day"].eq(1)
    later_row_on_remove_day = on_remove & out["row_order_within_episode_day"].gt(1)
    after_remove = day.gt(remove_day)

    out[state_col] = "out"
    out.loc[before_remove | first_row_on_remove_day, state_col] = "in"

    out[action_col] = "out"
    out.loc[before_remove, action_col] = "keep"
    out.loc[first_row_on_remove_day, action_col] = "remove"

    out[action_remove_col] = np.nan
    out.loc[before_remove, action_remove_col] = 0.0
    out.loc[first_row_on_remove_day, action_remove_col] = 1.0

    out[periods_in_col] = np.nan
    out.loc[before_remove | first_row_on_remove_day, periods_in_col] = day.loc[
        before_remove | first_row_on_remove_day
    ]

    out[periods_out_col] = np.nan
    out.loc[later_row_on_remove_day, periods_out_col] = 0.0
    out.loc[after_remove, periods_out_col] = day.loc[after_remove] - remove_day.loc[after_remove]
    out["policy_removal_day_extra_row_treated_as_out"] = later_row_on_remove_day.astype(int)
    return out


def resolved_timeline_diagnostics(
    df: pd.DataFrame,
    *,
    episode_id_col: str = "catheter_episode_id",
    policy_name_col: str = "policy_name",
    policy_remove_day_col: str = "policy_remove_day",
    episode_day_col: str = "episode_day_since_insertion",
) -> pd.DataFrame:
    """Summarise fixed-day resolved target-policy timeline safety diagnostics."""
    required = [
        policy_name_col,
        episode_id_col,
        policy_remove_day_col,
        episode_day_col,
        "policy_catheter_state",
        "policy_action_resolved",
        "policy_action_remove_resolved",
        "policy_removal_day_extra_row_treated_as_out",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Resolved target-policy timeline is missing required columns: {missing}")

    rows = []
    for policy_name, policy_df in df.groupby(policy_name_col, dropna=False, sort=False):
        remove_day = pd.to_numeric(policy_df[policy_remove_day_col], errors="coerce").dropna()
        numeric_remove_day = float(remove_day.iloc[0]) if len(remove_day) else np.nan
        action = policy_df["policy_action_resolved"].astype("string").str.strip().str.lower()
        action_remove = pd.to_numeric(policy_df["policy_action_remove_resolved"], errors="coerce")
        remove_rows = action.eq("remove") | action_remove.eq(1)
        remove_rows_by_episode = remove_rows.groupby(policy_df[episode_id_col], sort=False).sum()

        if pd.notna(numeric_remove_day):
            reaches_policy_removal_day = (
                pd.to_numeric(policy_df[episode_day_col], errors="coerce")
                .eq(numeric_remove_day)
                .groupby(policy_df[episode_id_col], sort=False)
                .max()
            )
            n_reached = int(reaches_policy_removal_day.sum())
        else:
            n_reached = 0

        n_remove_rows = int(remove_rows.sum())
        rows.append(
            {
                "policy_name": policy_name,
                "policy_remove_day": numeric_remove_day,
                "n_policy_remove_rows": n_remove_rows,
                "n_episodes_reaching_policy_removal_day": n_reached,
                "n_episodes_with_more_than_one_remove_row": int(remove_rows_by_episode.gt(1).sum()),
                "n_policy_removal_day_extra_rows_treated_as_out": int(
                    pd.to_numeric(
                        policy_df["policy_removal_day_extra_row_treated_as_out"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "n_policy_remove_row_shortfall_vs_reached_episodes": int(n_reached - n_remove_rows),
            }
        )
    return pd.DataFrame(rows)


def validate_resolved_target_policy_timeline(
    df: pd.DataFrame,
    *,
    episode_id_col: str = "catheter_episode_id",
    context: str = "policy_intervention_panel_long.csv",
) -> None:
    """Fail fast if the resolved target-policy timeline is internally unsafe."""
    required = [
        "episode_day_since_insertion",
        *RESOLVED_TIMELINE_COLUMNS,
        "policy_removal_day_extra_row_treated_as_out",
        "policy_name",
        "policy_remove_day",
        episode_id_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing resolved target-policy timeline columns: {missing}. "
            "Rebuild it with build_policy_intervention_panels.py."
        )

    state = df["policy_catheter_state"].astype("string").str.strip().str.lower()
    action = df["policy_action_resolved"].astype("string").str.strip().str.lower()
    action_remove = pd.to_numeric(df["policy_action_remove_resolved"], errors="coerce")

    invalid_state = ~state.isin(["in", "out"])
    if invalid_state.any():
        examples = df.loc[invalid_state, ["policy_name", episode_id_col, "policy_catheter_state"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_catheter_state values. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    invalid_action = ~action.isin(["keep", "remove", "out"])
    if invalid_action.any():
        examples = df.loc[invalid_action, ["policy_name", episode_id_col, "policy_action_resolved"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_action_resolved values. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    out_remove = state.eq("out") & (action.eq("remove") | action_remove.eq(1))
    if out_remove.any():
        examples = df.loc[
            out_remove,
            ["policy_name", episode_id_col, "policy_catheter_state", "policy_action_resolved"],
        ].head(10)
        raise ValueError(
            f"{context} assigns a resolved remove action on OUT-state rows. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    diagnostics = resolved_timeline_diagnostics(df, episode_id_col=episode_id_col)
    too_many = diagnostics["n_episodes_with_more_than_one_remove_row"].gt(0)
    shortfall = diagnostics["n_policy_remove_row_shortfall_vs_reached_episodes"].ne(0)
    if too_many.any() or shortfall.any():
        failing = diagnostics.loc[too_many | shortfall].head(20)
        raise ValueError(
            f"{context} has unsafe fixed-day resolved target-policy timing. "
            "Rebuild it with build_policy_intervention_panels.py. Diagnostics:\n"
            f"{failing}"
        )


def attach_resolved_timeline_aliases(
    df: pd.DataFrame,
    *,
    action_col: str,
    action_remove_col: str,
    episode_id_col: str = "catheter_episode_id",
    context: str = "policy_intervention_panel_long.csv",
) -> pd.DataFrame:
    """Copy authoritative resolved policy timeline columns into estimator aliases."""
    validate_resolved_target_policy_timeline(
        df,
        episode_id_col=episode_id_col,
        context=context,
    )
    out = df.copy()
    out["policy_catheter_state"] = out["policy_catheter_state"].astype("string").str.strip().str.lower()
    out["policy_action_resolved"] = out["policy_action_resolved"].astype("string").str.strip().str.lower()
    out["policy_action_remove_resolved"] = pd.to_numeric(
        out["policy_action_remove_resolved"],
        errors="coerce",
    )
    out[action_col] = out["policy_action_resolved"]
    out[action_remove_col] = out["policy_action_remove_resolved"]
    return out


def catheter_exposure_aggregation(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    state_col: str,
    row_count_col: str,
    exposure_col: str,
) -> pd.DataFrame:
    """Aggregate IN-row counts and duration-based catheter exposure."""
    in_mask = df[state_col].astype("string").str.lower().eq("in")
    temp = df[[*group_cols, "period_duration_days"]].copy()
    temp["__in_row"] = in_mask.astype(int)
    temp["__in_exposure_days"] = np.where(in_mask, df["period_duration_days"], 0.0)
    return temp.groupby(group_cols, as_index=False, dropna=False, sort=False).agg(
        **{
            row_count_col: ("__in_row", "sum"),
            exposure_col: ("__in_exposure_days", "sum"),
        }
    )


def duplicate_episode_day_count(
    df: pd.DataFrame,
    group_cols: list[str],
    day_col: str = "episode_day_since_insertion",
) -> int:
    if day_col not in df.columns:
        return 0
    duplicated = df.duplicated([*group_cols, day_col], keep=False)
    return int(duplicated.sum())


def add_standard_comparisons(
    summary: pd.DataFrame,
    *,
    baseline_label: str,
    comparison_map: dict[str, str],
) -> pd.DataFrame:
    """Add standard difference/percentage-point/ratio columns vs current practice."""
    baseline_rows = summary.loc[summary["policy_name"].eq(baseline_label)]
    if baseline_rows.empty:
        return summary
    baseline = baseline_rows.iloc[0]
    out = summary.copy()
    for standard_name, value_col in comparison_map.items():
        if value_col not in out.columns or value_col not in baseline.index:
            continue
        baseline_value = baseline[value_col]
        diff_col = f"{standard_name}_difference_vs_current_practice"
        out[diff_col] = out[value_col] - baseline_value
        # The pct-points column is meaningful for risk outcomes. For exposure it
        # is retained for schema consistency and equals the raw day difference.
        out[f"{standard_name}_difference_pct_points_vs_current_practice"] = (
            out[diff_col] * 100 if standard_name.endswith("_risk") else out[diff_col]
        )
        out[f"{standard_name}_ratio_vs_current_practice"] = out[value_col].apply(
            lambda value: np.nan
            if pd.isna(value) or pd.isna(baseline_value) or baseline_value == 0
            else float(value / baseline_value)
        )
    return out


def add_overlap_quality_flags(
    summary: pd.DataFrame,
    *,
    support_diagnostics: pd.DataFrame | None = None,
    weight_diagnostics: pd.DataFrame | None = None,
    current_practice_label: str = "current_practice",
) -> pd.DataFrame:
    """Add low-support/low-ESS/extreme-weight warning flags."""
    out = summary.copy()
    if "pct_adherent_episodes" in out.columns:
        out["low_adherence_flag"] = out["pct_adherent_episodes"].lt(LOW_ADHERENCE_THRESHOLD)
    else:
        out["low_adherence_flag"] = False

    n_col = "n_total_policy_episodes" if "n_total_policy_episodes" in out.columns else "n_episodes"
    ess = pd.to_numeric(out.get("effective_sample_size", np.nan), errors="coerce")
    n_total = pd.to_numeric(out.get(n_col, np.nan), errors="coerce")
    out["low_ess_flag"] = ess.lt(LOW_ESS_MIN) | ess.lt(LOW_ESS_FRACTION * n_total)

    out["low_support_flag"] = False
    if support_diagnostics is not None and not support_diagnostics.empty:
        support_all = support_diagnostics.loc[
            support_diagnostics.get("group", "all").eq("all"),
            ["policy_name", "pct_below_0_05"],
        ].drop_duplicates("policy_name")
        support_all["low_support_flag"] = pd.to_numeric(
            support_all["pct_below_0_05"], errors="coerce"
        ).gt(LOW_SUPPORT_PCT_BELOW_005_THRESHOLD)
        out = out.drop(columns=["low_support_flag"], errors="ignore").merge(
            support_all[["policy_name", "low_support_flag"]],
            on="policy_name",
            how="left",
        )
        out["low_support_flag"] = out["low_support_flag"].fillna(False)

    out["extreme_weight_flag"] = False
    if weight_diagnostics is not None and not weight_diagnostics.empty:
        weight_flags = weight_diagnostics[["policy_name"]].copy()
        weight_flags["extreme_weight_flag"] = (
            pd.to_numeric(weight_diagnostics.get("p99_weight", np.nan), errors="coerce").gt(
                EXTREME_WEIGHT_P99_THRESHOLD
            )
            | pd.to_numeric(weight_diagnostics.get("max_weight", np.nan), errors="coerce").gt(
                EXTREME_WEIGHT_MAX_THRESHOLD
            )
        )
        weight_flags = weight_flags.drop_duplicates("policy_name")
        out = out.drop(columns=["extreme_weight_flag"], errors="ignore").merge(
            weight_flags,
            on="policy_name",
            how="left",
        )
        out["extreme_weight_flag"] = out["extreme_weight_flag"].fillna(False)

    current = out["policy_name"].eq(current_practice_label)
    for col in ["low_adherence_flag", "low_ess_flag", "low_support_flag", "extreme_weight_flag"]:
        out.loc[current, col] = False
    return out
