"""
Build the master panel by aggregating itemid covariates onto the saved base panel.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

BASE_PANEL_FILE = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data\base_panel.csv")
CLEANED_CHART_FILE = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data\cleaned_chart_covariates.csv")
CHUNK_ROWS_CHARTEVENTS = 1_000_000
AGG_STATS = [
    "count",
    "mean",
    # "min",
    # "max",
    "std",
    # "first",
    "last",
    # "delta",
    # "range",
    "slope_per_hour",
]


def aggregate_itemid_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    # Reduce the panel down to the row-level covariate windows we need to score.
    windows = panel[["row_id", "stay_id", "cov_start", "cov_end"]].copy()
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())

    chart_cols = ["stay_id", "itemid", "charttime", "valuenum"]
    partial_stats = []
    first_obs_parts = []
    last_obs_parts = []

    kept_rows_total = 0
    chunk_idx = 0
    t0 = time.time()

    print("[Base panel rows]", len(panel))
    print("[EHR] Aggregating chartevents covariates...")

    for chunk in pd.read_csv(
        CLEANED_CHART_FILE,
        usecols=chart_cols,
        chunksize=CHUNK_ROWS_CHARTEVENTS,
        low_memory=False,
    ):
        chunk_idx += 1
        t_chunk0 = time.time()

        # Keep only chart rows from stays that appear in the panel.
        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        # Parse the measurement fields needed for interval matching and aggregation.
        chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
        chunk_filtered["valuenum"] = pd.to_numeric(chunk_filtered["valuenum"], errors="coerce")
        chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"]).copy()

        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} numeric_keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        # Join each chart row to the panel windows for the same stay and keep interval matches.
        merged = chunk_filtered.merge(windows, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["cov_start"]) &
            (merged["charttime"] < merged["cov_end"])
        ].copy()

        kept = len(merged)
        kept_rows_total += kept

        if kept > 0:
            # Precompute the sums needed for chunk-wise descriptive stats and slopes.
            merged["charttime_seconds"] = merged["charttime"].astype("int64") / 1_000_000_000.0
            merged["valuenum_sq"] = merged["valuenum"] * merged["valuenum"]
            merged["charttime_sq"] = merged["charttime_seconds"] * merged["charttime_seconds"]
            merged["charttime_value"] = merged["charttime_seconds"] * merged["valuenum"]

            # Aggregate chunk-level partial sums that can be combined across chunks.
            item_summary = (
                merged.groupby(["row_id", "itemid"])
                .agg(
                    count=("valuenum", "count"),
                    total=("valuenum", "sum"),
                    sum_sq=("valuenum_sq", "sum"),
                    min=("valuenum", "min"),
                    max=("valuenum", "max"),
                    sum_t=("charttime_seconds", "sum"),
                    sum_tt=("charttime_sq", "sum"),
                    sum_ty=("charttime_value", "sum"),
                )
                .reset_index()
            )
            partial_stats.append(item_summary)

            # Preserve the first observed value per row_id/itemid across chunks.
            first_obs = (
                merged.sort_values(["row_id", "itemid", "charttime"])
                .drop_duplicates(["row_id", "itemid"], keep="first")
                [["row_id", "itemid", "charttime", "valuenum"]]
                .rename(columns={"charttime": "first_time", "valuenum": "first"})
            )
            first_obs_parts.append(first_obs)

            # Preserve the last observed value per row_id/itemid across chunks.
            last_obs = (
                merged.sort_values(["row_id", "itemid", "charttime"])
                .drop_duplicates(["row_id", "itemid"], keep="last")
                [["row_id", "itemid", "charttime", "valuenum"]]
                .rename(columns={"charttime": "last_time", "valuenum": "last"})
            )
            last_obs_parts.append(last_obs)

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} stay_filtered={len(chunk_filtered):,} "
            f"matched={kept:,} cum_matched={kept_rows_total:,} dt={time.time()-t_chunk0:.1f}s"
        )

    if not partial_stats:
        print(f"[EHR] No chartevents matched windows. Total time: {time.time()-t0:.1f}s")
        return panel

    # Combine the chunk-level summaries into one row_id/itemid summary table.
    agg = pd.concat(partial_stats, ignore_index=True)
    agg = (
        agg.groupby(["row_id", "itemid"])
        .agg(
            count=("count", "sum"),
            total=("total", "sum"),
            sum_sq=("sum_sq", "sum"),
            min=("min", "min"),
            max=("max", "max"),
            sum_t=("sum_t", "sum"),
            sum_tt=("sum_tt", "sum"),
            sum_ty=("sum_ty", "sum"),
        )
        .reset_index()
    )

    # Derive final summary statistics from the saved sums.
    agg["mean"] = agg["total"] / agg["count"]
    agg["range"] = agg["max"] - agg["min"]

    variance_num = agg["sum_sq"] - (agg["total"] * agg["total"] / agg["count"])
    agg["std"] = np.where(
        agg["count"] > 1,
        np.sqrt((variance_num / (agg["count"] - 1)).clip(lower=0)),
        np.nan,
    )

    if first_obs_parts:
        # Keep the earliest observation seen across all chunks.
        first_obs = (
            pd.concat(first_obs_parts, ignore_index=True)
            .sort_values(["row_id", "itemid", "first_time"])
            .drop_duplicates(["row_id", "itemid"], keep="first")
        )
        agg = agg.merge(first_obs[["row_id", "itemid", "first"]], on=["row_id", "itemid"], how="left")

    if last_obs_parts:
        # Keep the latest observation seen across all chunks.
        last_obs = (
            pd.concat(last_obs_parts, ignore_index=True)
            .sort_values(["row_id", "itemid", "last_time"])
            .drop_duplicates(["row_id", "itemid"], keep="last")
        )
        agg = agg.merge(last_obs[["row_id", "itemid", "last"]], on=["row_id", "itemid"], how="left")

    agg["delta"] = agg["last"] - agg["first"]

    slope_denom = agg["count"] * agg["sum_tt"] - agg["sum_t"] * agg["sum_t"]
    slope_num = agg["count"] * agg["sum_ty"] - agg["sum_t"] * agg["total"]
    agg["slope_per_hour"] = np.where(
        (agg["count"] > 1) & (slope_denom != 0),
        (slope_num / slope_denom) * 3600.0,
        np.nan,
    )

    # Pivot the long item summaries back to one row per panel window.
    value_cols = [c for c in AGG_STATS if c in agg.columns]
    wide = agg.pivot_table(
        index="row_id",
        columns="itemid",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"itemid_{itemid}__{stat}" for stat, itemid in wide.columns]
    wide = wide.reset_index()

    panel = panel.merge(wide, on="row_id", how="left")
    print(f"[EHR] Covariates aggregated. Total time: {time.time()-t0:.1f}s")
    return panel


def main():
    # Load the base panel and parse the interval boundaries used for aggregation.
    panel = pd.read_csv(BASE_PANEL_FILE, low_memory=False)
    for col in ["inserted", "removed", "reinsertion_time", "period_start", "period_end", "cov_start", "cov_end"]:
        panel[col] = pd.to_datetime(panel[col], errors="coerce")

    # Join the chart-derived covariates onto the base panel.
    panel = aggregate_itemid_covariates(panel)
    panel = panel.drop(columns=["cov_start", "cov_end", "row_id"])

    # Keep base columns first and itemid covariates after them.
    covariate_cols = sorted([c for c in panel.columns if c.startswith("itemid_")])
    base_cols = [c for c in panel.columns if not c.startswith("itemid_")]
    panel = panel[base_cols + covariate_cols]

    # Save the master panel.
    outdir = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / "master_panel.csv"
    panel.to_csv(outfile, index=False)

    print("[SAVE]", outfile)
    print("Removals:", panel["removed_in_period"].sum())
    print("Reinsertions:", panel["reinsertion_in_period"].sum())
    print("CAUTI:", panel["cauti_in_period"].sum())
    print("ICU end rows:", panel["icu_end_in_period"].sum())
    print("Last episode periods:", panel["is_last_period_of_episode"].sum())
    print("Rows:", len(panel))
    print("[DONE]")


if __name__ == "__main__":
    main()
