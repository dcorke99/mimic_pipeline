"""
Build the master panel by aggregating itemid covariates onto the saved base panel.
"""

from pathlib import Path
import time

import pandas as pd

BASE_PANEL_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\base_panel.csv")
CLEANED_CHART_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\cleaned_chart_covariates.csv")
CHUNK_ROWS_CHARTEVENTS = 1_000_000


def main():
    panel = pd.read_csv(BASE_PANEL_FILE, low_memory=False)
    for col in ["inserted", "removed", "reinsertion_time", "day_start", "day_end", "cov_start", "cov_end"]:
        panel[col] = pd.to_datetime(panel[col], errors="coerce")

    windows = panel[["row_id", "stay_id", "cov_start", "cov_end"]].copy()
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())

    print("[Base panel rows]", len(panel))
    print("[EHR] Aggregating chartevents covariates...")

    usecols_ce = ["stay_id", "itemid", "charttime", "valuenum"]
    partial_stats = []

    kept_rows_total = 0
    chunk_idx = 0
    t0 = time.time()

    for chunk in pd.read_csv(
        CLEANED_CHART_FILE,
        usecols=usecols_ce,
        chunksize=CHUNK_ROWS_CHARTEVENTS,
        low_memory=False,
    ):
        chunk_idx += 1
        t_chunk0 = time.time()

        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
        chunk_filtered["valuenum"] = pd.to_numeric(chunk_filtered["valuenum"], errors="coerce")
        chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])

        if len(chunk_filtered) == 0:
            print(f"[EHR][{chunk_idx}] read={len(chunk):,} numeric_keep=0 dt={time.time()-t_chunk0:.1f}s")
            continue

        merged = chunk_filtered.merge(windows, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["cov_start"]) &
            (merged["charttime"] < merged["cov_end"])
        ].copy()

        kept = len(merged)
        kept_rows_total += kept

        if kept > 0:
            item_summary = (
                merged.groupby(["row_id", "itemid"])["valuenum"]
                .agg(count="count", total="sum", min="min", max="max")
                .reset_index()
            )
            partial_stats.append(item_summary)

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} stay_filtered={len(chunk_filtered):,} "
            f"matched={kept:,} cum_matched={kept_rows_total:,} dt={time.time()-t_chunk0:.1f}s"
        )

    if partial_stats:
        agg = pd.concat(partial_stats, ignore_index=True)
        agg = (
            agg.groupby(["row_id", "itemid"])
            .agg(
                count=("count", "sum"),
                total=("total", "sum"),
                min=("min", "min"),
                max=("max", "max"),
            )
            .reset_index()
        )
        agg["mean"] = agg["total"] / agg["count"]

        wide = agg.pivot_table(
            index="row_id",
            columns="itemid",
            values=["mean", "min", "max"],
            aggfunc="first",
        )
        wide.columns = [f"itemid_{itemid}__{stat}" for stat, itemid in wide.columns]
        wide = wide.reset_index()
        panel = panel.merge(wide, on="row_id", how="left")
        print(f"[EHR] Covariates aggregated. Total time: {time.time()-t0:.1f}s")
    else:
        print(f"[EHR] No chartevents matched windows. Total time: {time.time()-t0:.1f}s")

    panel = panel.drop(columns=["cov_start", "cov_end", "row_id"])
    covariate_cols = sorted([c for c in panel.columns if c.startswith("itemid_")])
    base_cols = [c for c in panel.columns if not c.startswith("itemid_")]
    panel = panel[base_cols + covariate_cols]

    outdir = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / "master_panel.csv"
    panel.to_csv(outfile, index=False)

    print("[SAVE]", outfile)
    print("Removals:", panel["removed_today"].sum())
    print("Reinsertions:", panel["reinsertion_today"].sum())
    print("CAUTI:", panel["cauti_today"].sum())
    print("ICU end rows:", panel["icu_end_today"].sum())
    print("Last episode days:", panel["is_last_day_of_episode"].sum())
    print("Rows:", len(panel))
    print("[DONE]")


if __name__ == "__main__":
    main()
