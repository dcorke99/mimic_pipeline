"""
Standalone diagnostics for the filtered catheter panel.

Purpose
-------
Runs the panel/risk-set checks that were previously printed inside
01_step1_transition_models.py, but without fitting any models.

This script audits:
- split summary
- raw outcome distributions
- IN/OUT state counts
- CAUTI first-event risk-set construction
- reinsertion OUT-state fitting/evaluation subset construction
- event rates by days_in_state
- event rates by day_index
- CAUTI fit-set composition by state
- episode-level post-CAUTI exclusion behaviour

It reads directly from filtered_panel.csv and does not require
step1_scored_panel.csv or transition_models.pkl.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

INDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\panel_diagnostics")

INFILE = INDIR / "filtered_panel.csv"

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
DAYS_COL = "days_in_state"
INTERVAL_COL = "interval_hours"
ACTION_COL = "removed_today"
SPLIT_COL = "split"

Y_CAUTI = "cauti_today"
Y_REINS = "reinsertion_today"

LAST_DAY_COL = "is_last_day_of_episode"
END_REASON_COL = "episode_end_reason"

EPISODE_KEYS = ["stay_id", "inserted"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if c.startswith("itemid_") or c.startswith("sex_") or c.startswith("ethnicity_")
    ]
    if "age" in df.columns:
        cols.append("age")
    return cols


def _validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()


def _coerce_binaryish_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue

        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0
            })
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()

    required_cols = [
        ID_COL, TIME_COL, STATE_COL, DAYS_COL, SPLIT_COL,
        Y_CAUTI, Y_REINS, LAST_DAY_COL, END_REASON_COL,
        "day_end", *EPISODE_KEYS
    ]
    _check_required_columns(df, required_cols)

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()
    _validate_split(df)

    feat = _feature_cols(df)

    numeric_like_cols = [
        TIME_COL, DAYS_COL, INTERVAL_COL, ACTION_COL,
        Y_CAUTI, Y_REINS, LAST_DAY_COL
    ] + feat
    _coerce_binaryish_numeric(df, numeric_like_cols)

    # ----------------------------------------------------------------
    # Split + raw panel diagnostics
    # ----------------------------------------------------------------
    print("\n--- Split summary ---", flush=True)
    print(df[SPLIT_COL].value_counts(dropna=False), flush=True)
    print(
        f"Unique patients: train={df.loc[df[SPLIT_COL] == 'train', ID_COL].nunique()}, "
        f"test={df.loc[df[SPLIT_COL] == 'test', ID_COL].nunique()}",
        flush=True
    )

    print("\n--- Outcome distributions ---", flush=True)

    print("\nCAUTI across all rows", flush=True)
    print(df[Y_CAUTI].value_counts(dropna=False), flush=True)

    print("\nCAUTI by state", flush=True)
    print(df.groupby(STATE_COL)[Y_CAUTI].value_counts(dropna=False), flush=True)

    print("\nReinsertion in OUT state", flush=True)
    print(df.loc[df[STATE_COL] == "out", Y_REINS].value_counts(dropna=False), flush=True)

    print("\n--- Initialization ---", flush=True)
    print(f"Total rows loaded: {len(df)}", flush=True)
    print(f"Features identified: {len(feat)}", flush=True)

    print("\nRows by state", flush=True)
    print(df[STATE_COL].value_counts(dropna=False), flush=True)

    # Extra explicit consistency check for reinsertion labels by state
    print("\nReinsertion positives by state", flush=True)
    print(
        df.groupby(STATE_COL)[Y_REINS]
        .agg(["count", "sum", "mean"]),
        flush=True
    )

    impossible_reins = df[(df[STATE_COL] != "out") & (df[Y_REINS] == 1)]
    print(f"\nReinsertion positives outside OUT state: {len(impossible_reins)}", flush=True)

    # ----------------------------------------------------------------
    # Order within episode, then build CAUTI first-event fit set
    # ----------------------------------------------------------------
    df = (
        df.sort_values(EPISODE_KEYS + ["day_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
    )

    df["prior_cauti_count"] = (
        df.groupby(EPISODE_KEYS)[Y_CAUTI]
        .cumsum()
        .shift(fill_value=0)
    )

    df_cauti = df[df["prior_cauti_count"] == 0].copy()

    print("\n--- CAUTI risk-set construction ---", flush=True)
    print(f"Rows identified for CAUTI modelling: {len(df_cauti)}", flush=True)
    print(f"Excluded post-CAUTI rows: {len(df) - len(df_cauti)}", flush=True)

    # ----------------------------------------------------------------
    # Reinsertion OUT-state fit subset
    # ----------------------------------------------------------------
    df_in = df[df[STATE_COL] == "in"].copy()
    df_out = df[df[STATE_COL] == "out"].copy()

    df_out_fit = df_out[
        ~(
            (df_out[LAST_DAY_COL] == 1) &
            (df_out[END_REASON_COL] == "icu_end") &
            (df_out[Y_REINS] == 0)
        )
    ].copy()

    print("\n--- Reinsertion fit-set construction ---", flush=True)
    print(f"Rows identified as OUT state: {len(df_out)}", flush=True)
    print(f"Rows used for reinsertion fitting/evaluation: {len(df_out_fit)}", flush=True)
    print(f"Excluded censored terminal OUT rows: {len(df_out) - len(df_out_fit)}", flush=True)

    # ----------------------------------------------------------------
    # Event rates by days_in_state
    # ----------------------------------------------------------------
    print("\n--- Event rates by days_in_state ---", flush=True)

    if not df_cauti.empty:
        print("\nCAUTI rate by days_in_state:", flush=True)
        print(
            df_cauti.groupby(DAYS_COL)[Y_CAUTI]
            .agg(["count", "sum", "mean"])
            .head(20),
            flush=True
        )

    if not df_out_fit.empty:
        print("\nReinsertion rate by days_in_state:", flush=True)
        print(
            df_out_fit.groupby(DAYS_COL)[Y_REINS]
            .agg(["count", "sum", "mean"])
            .head(20),
            flush=True
        )

    # More interpretable split by state for CAUTI
    if not df_cauti.empty:
        print("\nCAUTI rate by state and days_in_state:", flush=True)
        print(
            df_cauti.groupby([STATE_COL, DAYS_COL])[Y_CAUTI]
            .agg(["count", "sum", "mean"])
            .head(40),
            flush=True
        )

    # ----------------------------------------------------------------
    # Event rates by day_index
    # ----------------------------------------------------------------
    print("\n--- Event rates by day_index ---", flush=True)

    if not df_cauti.empty:
        print("\nCAUTI rate by day_index:", flush=True)
        print(
            df_cauti.groupby(TIME_COL)[Y_CAUTI]
            .agg(["count", "sum", "mean"])
            .head(20),
            flush=True
        )

    if not df_out_fit.empty:
        print("\nReinsertion rate by day_index:", flush=True)
        print(
            df_out_fit.groupby(TIME_COL)[Y_REINS]
            .agg(["count", "sum", "mean"])
            .head(20),
            flush=True
        )

    # ----------------------------------------------------------------
    # CAUTI fit-set composition by state
    # ----------------------------------------------------------------
    print("\n--- CAUTI fit set by state ---", flush=True)

    print("\nCAUTI fit rows by state:", flush=True)
    print(
        df_cauti[STATE_COL].value_counts(dropna=False),
        flush=True
    )

    print("\nCAUTI positives by state in fit set:", flush=True)
    print(
        df_cauti.groupby(STATE_COL)[Y_CAUTI]
        .agg(["count", "sum", "mean"]),
        flush=True
    )

    # ----------------------------------------------------------------
    # Post-CAUTI exclusion audit at episode level
    # ----------------------------------------------------------------
    print("\n--- Post-CAUTI exclusion check ---", flush=True)

    episode_summary = (
        df.groupby(EPISODE_KEYS)
        .agg(
            total_rows=(Y_CAUTI, "size"),
            cauti_events=(Y_CAUTI, "sum")
        )
        .reset_index()
    )

    cauti_episode_keys = episode_summary.loc[
        episode_summary["cauti_events"] > 0, EPISODE_KEYS
    ]

    df_cauti_episode_counts = (
        df_cauti.groupby(EPISODE_KEYS)
        .size()
        .reset_index(name="rows_in_cauti_fit")
    )

    df_all_episode_counts = (
        df.groupby(EPISODE_KEYS)
        .size()
        .reset_index(name="rows_in_full_df")
    )

    merged_episode_counts = (
        cauti_episode_keys
        .merge(df_all_episode_counts, on=EPISODE_KEYS, how="left")
        .merge(df_cauti_episode_counts, on=EPISODE_KEYS, how="left")
    )

    merged_episode_counts["rows_in_cauti_fit"] = (
        merged_episode_counts["rows_in_cauti_fit"]
        .fillna(0)
        .astype(int)
    )

    merged_episode_counts["excluded_post_cauti_rows"] = (
        merged_episode_counts["rows_in_full_df"] - merged_episode_counts["rows_in_cauti_fit"]
    )

    print("\nEpisodes with at least one CAUTI:", flush=True)
    print(len(merged_episode_counts), flush=True)

    print("\nDistribution of excluded post-CAUTI rows per CAUTI episode:", flush=True)
    print(
        merged_episode_counts["excluded_post_cauti_rows"]
        .value_counts()
        .sort_index()
        .head(20),
        flush=True
    )

    # ----------------------------------------------------------------
    # Save summary outputs
    # ----------------------------------------------------------------
    summary = {
        "n_rows": {
            "all": int(len(df)),
            "in": int(len(df_in)),
            "out": int(len(df_out)),
            "cauti_fit": int(len(df_cauti)),
            "out_fit": int(len(df_out_fit)),
        },
        "split": {
            "train_rows": int((df[SPLIT_COL] == "train").sum()),
            "test_rows": int((df[SPLIT_COL] == "test").sum()),
            "train_patients": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique()),
            "test_patients": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique()),
        },
        "events": {
            "cauti_all": int(df[Y_CAUTI].sum()),
            "reins_all": int(df[Y_REINS].sum()),
            "reins_out": int(df.loc[df[STATE_COL] == "out", Y_REINS].sum()),
            "reins_outside_out_state": int(len(impossible_reins)),
        },
        "filters": {
            "excluded_post_cauti_rows": int(len(df) - len(df_cauti)),
            "excluded_terminal_out_rows": int(len(df_out) - len(df_out_fit)),
            "episodes_with_cauti": int(len(merged_episode_counts)),
        },
    }

    (OUTDIR / "panel_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Restore original order before optional export
    df_export = (
        df.sort_values("_orig_index")
        .drop(columns=["_orig_index", "prior_cauti_count"])
        .copy()
    )
    df_export.to_csv(OUTDIR / "panel_with_diagnostic_flags.csv", index=False, float_format="%.6f")

    print("\n--- SUCCESS ---", flush=True)
    print(f"Summary saved: {OUTDIR / 'panel_diagnostics_summary.json'}", flush=True)
    print(f"Flagged panel saved: {OUTDIR / 'panel_with_diagnostic_flags.csv'}", flush=True)


if __name__ == "__main__":
    main()