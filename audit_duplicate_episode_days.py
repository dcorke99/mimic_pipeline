#!/usr/bin/env python3
# Audit duplicate episode-day rows in the catheter panel

from pathlib import Path
import numpy as np
import pandas as pd

import policy_eval_common as pec


REPO_ROOT = Path(__file__).resolve().parent
INPUT_PATH = REPO_ROOT / "data" / "modelling_panel.csv"
OUTDIR = REPO_ROOT / "artefacts" / "diagnostics" / "duplicate_episode_days"


def add_episode_day_since_insertion(df):
    # Count whole days since catheter insertion
    df = df.copy()

    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")

    # Treat the insertion date as episode day one
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"] < 1, "episode_day_since_insertion"] = 1

    return df


def main():
    # Create the audit directory
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load and standardise the modelling panel
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()

    # Assign one stable number to each catheter episode
    episode_key_cols = ["subject_id", "hadm_id", "stay_id", "inserted", "removed"]
    df["catheter_episode_id"] = pd.factorize(
        df[episode_key_cols].astype(str).agg("|".join, axis=1),
        sort=True,
    )[0] + 1

    # Add episode day since catheter insertion
    df = add_episode_day_since_insertion(df)

    # Count rows within each episode-day
    day_key = ["catheter_episode_id", "episode_day_since_insertion"]

    counts = (
        df.groupby(day_key, dropna=False)
        .size()
        .reset_index(name="n_rows_for_episode_day")
        .sort_values("n_rows_for_episode_day", ascending=False)
    )

    # Retain episode-days with multiple rows
    duplicate_days = counts[counts["n_rows_for_episode_day"] > 1].copy()

    # Recover the full rows behind those counts
    duplicate_rows = df.merge(
        duplicate_days[day_key],
        on=day_key,
        how="inner",
    ).sort_values(
        ["catheter_episode_id", "episode_day_since_insertion", "period_start", "period_end"]
    )

    # Define exact duplicate interval fields
    exact_key = [
        "catheter_episode_id",
        "episode_day_since_insertion",
        "period_start",
        "period_end",
        "catheter_state",
        "observed_action",
        "removed_in_period",
        "periods_in_state",
    ]

    # Count identical interval rows
    exact_counts = (
        df.groupby(exact_key, dropna=False)
        .size()
        .reset_index(name="n_exact_duplicate_rows")
        .sort_values("n_exact_duplicate_rows", ascending=False)
    )

    # Retain groups with exact duplicates
    exact_duplicates = exact_counts[exact_counts["n_exact_duplicate_rows"] > 1].copy()

    # Recover the full exact duplicate rows
    exact_duplicate_rows = df.merge(
        exact_duplicates[exact_key],
        on=exact_key,
        how="inner",
    ).sort_values(exact_key)

    # Summarise the time range of each duplicate episode-day
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

    # Save each audit table at three decimal places
    pec.save_report_df(counts, OUTDIR / "episode_day_row_counts_all.csv")
    pec.save_report_df(duplicate_days, OUTDIR / "duplicate_episode_day_summary.csv")
    pec.save_report_df(duplicate_rows, OUTDIR / "duplicate_episode_day_rows.csv")
    pec.save_report_df(
        exact_duplicates,
        OUTDIR / "exact_duplicate_interval_summary.csv",
    )
    pec.save_report_df(
        exact_duplicate_rows,
        OUTDIR / "exact_duplicate_interval_rows.csv",
    )
    pec.save_report_df(
        interval_summary,
        OUTDIR / "duplicate_episode_day_trace_summary.csv",
    )

    # Print the main audit counts
    print()
    print("--- DUPLICATE EPISODE-DAY AUDIT COMPLETE ---")
    print(f"Input panel: {INPUT_PATH}")
    print(f"Rows in input panel: {len(df):,}")
    print(f"Unique catheter episodes: {df['catheter_episode_id'].nunique():,}")
    print(f"Episode-days with more than one row: {len(duplicate_days):,}")
    print(f"Rows belonging to duplicate episode-days: {len(duplicate_rows):,}")
    print(f"Exact duplicate interval groups: {len(exact_duplicates):,}")
    print(f"Rows belonging to exact duplicate intervals: {len(exact_duplicate_rows):,}")
    print()
    print(f"Saved outputs to: {OUTDIR}")
    print("Main file to inspect:")
    print(OUTDIR / "duplicate_episode_day_rows.csv")


if __name__ == "__main__":
    main()
