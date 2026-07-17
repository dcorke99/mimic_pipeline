#!/usr/bin/env python3
# Audit duplicate episode-day rows in the catheter panel.

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import policy_eval_common as pec


DEFAULT_INPUT = Path("data/modeling_panel.csv")
DEFAULT_OUTDIR = Path("artifacts/diagnostics/duplicate_episode_days")


def parse_args():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-panel", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def add_episode_day_since_insertion(df):
    # Add episode day since catheter insertion.
    df = df.copy()

    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")

    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"] < 1, "episode_day_since_insertion"] = 1

    return df


def main():
    # Run the script workflow.
    # Parse command-line arguments.
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_panel, low_memory=False)
    df.columns = df.columns.str.strip()

    if "catheter_episode_id" not in df.columns:
        episode_key_cols = ["subject_id", "hadm_id", "stay_id", "inserted"]
        if "removed" in df.columns:
            episode_key_cols.append("removed")
        df["catheter_episode_id"] = pd.factorize(
            df[episode_key_cols].astype(str).agg("|".join, axis=1),
            sort=True,
        )[0] + 1

    # Add episode day since catheter insertion.
    df = add_episode_day_since_insertion(df)

    # Main duplicate check: more than one row per episode-day
    day_key = ["catheter_episode_id", "episode_day_since_insertion"]

    counts = (
        df.groupby(day_key, dropna=False)
        .size()
        .reset_index(name="n_rows_for_episode_day")
        .sort_values("n_rows_for_episode_day", ascending=False)
    )

    duplicate_days = counts[counts["n_rows_for_episode_day"] > 1].copy()

    duplicate_rows = df.merge(
        duplicate_days[day_key],
        on=day_key,
        how="inner",
    ).sort_values(
        ["catheter_episode_id", "episode_day_since_insertion", "period_start", "period_end"]
    )

    # Exact duplicate interval check
    exact_key = [
        "catheter_episode_id",
        "episode_day_since_insertion",
        "period_start",
        "period_end",
    ]

    for optional_col in ["catheter_state", "observed_action", "action_remove", "periods_in_state"]:
        if optional_col in df.columns:
            exact_key.append(optional_col)

    exact_counts = (
        df.groupby(exact_key, dropna=False)
        .size()
        .reset_index(name="n_exact_duplicate_rows")
        .sort_values("n_exact_duplicate_rows", ascending=False)
    )

    exact_duplicates = exact_counts[exact_counts["n_exact_duplicate_rows"] > 1].copy()

    exact_duplicate_rows = df.merge(
        exact_duplicates[exact_key],
        on=exact_key,
        how="inner",
    ).sort_values(exact_key)

    # Useful compact episode-day summary
    summary_cols = [
        "catheter_episode_id",
        "episode_day_since_insertion",
        "n_rows_for_episode_day",
    ]

    if not duplicate_rows.empty:
        interval_summary = (
            duplicate_rows.groupby(day_key, dropna=False)
            .agg(
                subject_id=("subject_id", "first"),
                hadm_id=("hadm_id", "first"),
                stay_id=("stay_id", "first"),
                inserted=("inserted", "first"),
                first_period_start=("period_start", "min"),
                last_period_end=("period_end", "max"),
                n_unique_period_starts=("period_start", "nunique"),
                n_unique_period_ends=("period_end", "nunique"),
            )
            .reset_index()
            .merge(duplicate_days, on=day_key, how="left")
            .sort_values("n_rows_for_episode_day", ascending=False)
        )
    else:
        interval_summary = duplicate_days

    # Save outputs
    pec.save_report_df(counts, args.outdir / "episode_day_row_counts_all.csv")
    pec.save_report_df(duplicate_days, args.outdir / "duplicate_episode_day_summary.csv")
    pec.save_report_df(duplicate_rows, args.outdir / "duplicate_episode_day_rows.csv")
    pec.save_report_df(exact_duplicates, args.outdir / "exact_duplicate_interval_summary.csv")
    pec.save_report_df(exact_duplicate_rows, args.outdir / "exact_duplicate_interval_rows.csv")
    pec.save_report_df(interval_summary, args.outdir / "duplicate_episode_day_trace_summary.csv")

    print()
    print("--- DUPLICATE EPISODE-DAY AUDIT COMPLETE ---")
    print(f"Input panel: {args.input_panel}")
    print(f"Rows in input panel: {len(df):,}")
    print(f"Unique catheter episodes: {df['catheter_episode_id'].nunique():,}")
    print(f"Episode-days with more than one row: {len(duplicate_days):,}")
    print(f"Rows belonging to duplicate episode-days: {len(duplicate_rows):,}")
    print(f"Exact duplicate interval groups: {len(exact_duplicates):,}")
    print(f"Rows belonging to exact duplicate intervals: {len(exact_duplicate_rows):,}")
    print()
    print(f"Saved outputs to: {args.outdir}")
    print("Main file to inspect:")
    print(args.outdir / "duplicate_episode_day_rows.csv")


# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
