"""
Extract raw chart-event covariates for the CAUTI analysis cohort.

Creates a raw long-format chart-events file restricted to the catheter episodes
selected for the downstream panel by 00_identify_required_catheter_episodes.py.
"""

from pathlib import Path
import time

import pandas as pd

MIMIC_DIR = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1")
EPISODE_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\required_catheter_episodes.csv")
CHUNK_ROWS_CHARTEVENTS = 1_000_000
LOOKBACK_HOURS = 24
SAMPLE_ROWS = 1000


def build_chart_extraction_windows(
    episodes: pd.DataFrame,
    lookback_hours: int = LOOKBACK_HOURS,
) -> pd.DataFrame:
    # Start from the saved catheter-episode cohort.
    windows = episodes.copy()
    # End each extraction window at reinsertion if present, otherwise ICU discharge.
    window_end = windows["reinsertion_time"].where(windows["reinsertion_time"].notna(), windows["ICU_out"])
    # Begin each extraction window lookback_hours before insertion.
    window_start = windows["inserted"] - pd.Timedelta(hours=lookback_hours)
    # Cap the lookback at ICU admission time.
    window_start = windows[["ICU_in"]].assign(window_start=window_start).max(axis=1)

    # Store the computed bounds and drop rows without a valid interval.
    windows = windows.assign(window_start=window_start, window_end=window_end)
    windows = windows.dropna(subset=["window_start", "window_end"]).copy()
    windows = windows[windows["window_end"] > windows["window_start"]].copy()

    merged_windows = []

    # Merge overlapping windows within each stay before scanning chartevents.
    for stay_id, stay_windows in windows.groupby("stay_id"):
        # Sort windows by start time within the stay.
        stay_windows = stay_windows.sort_values("window_start")
        current_start = None
        current_end = None

        for row in stay_windows.itertuples():
            if current_start is None:
                # Initialise the first merged window.
                current_start = row.window_start
                current_end = row.window_end
                continue

            if row.window_start <= current_end:
                # Extend the current merged window when windows overlap.
                current_end = max(current_end, row.window_end)
            else:
                # Save the completed merged window and start a new one.
                merged_windows.append((stay_id, current_start, current_end))
                current_start = row.window_start
                current_end = row.window_end

        if current_start is not None:
            # Save the final merged window for the stay.
            merged_windows.append((stay_id, current_start, current_end))

    return pd.DataFrame(merged_windows, columns=["stay_id", "window_start", "window_end"])


def main() -> None:
    print("[CONFIG]", MIMIC_DIR)

    # Read the saved catheter-episode cohort.
    episodes = pd.read_csv(EPISODE_FILE, low_memory=False)
    # Parse the episode timing columns used for window construction.
    for col in ["inserted", "removed", "reinsertion_time", "ICU_in", "ICU_out"]:
        episodes[col] = pd.to_datetime(episodes[col], errors="coerce")

    # Build the per-stay chart extraction windows from the episode table.
    windows = build_chart_extraction_windows(episodes)

    print("[Catheter episodes]", len(episodes))
    print("[Chart windows]", len(windows))
    print("[EHR] Extracting raw chartevents covariates...")

    # Read the chart-event columns needed for downstream cleaning and aggregation.
    chart_cols = [
        "subject_id", "hadm_id", "stay_id", "itemid", "charttime",
        "storetime", "valuenum", "value", "valueuom",
    ]

    # Create the output directory and reset any previous output file.
    outdir = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / "raw_chart_covariates.csv"
    sample_outfile = outdir / "raw_chart_covariates__first_1000_rows.csv"
    if outfile.exists():
        outfile.unlink()
    if sample_outfile.exists():
        sample_outfile.unlink()

    # Track the ICU stays that appear in the extraction windows.
    stay_ids = set(windows["stay_id"].dropna().astype(int).unique())
    p_chartevents = MIMIC_DIR / "icu/chartevents.csv"
    kept_rows_total = 0
    chunk_idx = 0
    t0 = time.time()
    first_write = True
    sample_rows_written = 0

    # Stream chartevents in chunks to avoid loading the full file into memory.
    for chunk in pd.read_csv(
        p_chartevents,
        usecols=chart_cols,
        chunksize=CHUNK_ROWS_CHARTEVENTS,
        low_memory=False,
    ):
        chunk_idx += 1
        t_chunk0 = time.time()

        # Restrict the chunk to stays that appear in the episode windows.
        chunk_filtered = chunk[chunk["stay_id"].isin(stay_ids)].copy()
        if len(chunk_filtered) == 0:
            kept = 0
        else:
            # Parse chart timestamps before time-window filtering.
            chunk_filtered["charttime"] = pd.to_datetime(chunk_filtered["charttime"], errors="coerce")
            chunk_filtered["storetime"] = pd.to_datetime(chunk_filtered["storetime"], errors="coerce")

            # Drop rows without the stay/time fields needed for matching.
            chunk_filtered = chunk_filtered.dropna(subset=["stay_id", "charttime"]).copy()

            if len(chunk_filtered) == 0:
                kept = 0
            else:
                # Join the chunk to the extraction windows by stay.
                matched = chunk_filtered.merge(windows, on="stay_id", how="inner")

                # Keep only chart rows that fall inside a valid extraction window.
                matched = matched[
                    (matched["charttime"] >= matched["window_start"]) &
                    (matched["charttime"] < matched["window_end"])
                ].copy()

                # Restore the original output columns and drop duplicate matches.
                chunk_filtered = matched[chart_cols].drop_duplicates()
                kept = len(chunk_filtered)

        # Track the number of extracted rows across chunks.
        kept_rows_total += kept

        if kept > 0:
            # Write the first chunk with a header and append subsequent chunks.
            chunk_filtered.to_csv(
                outfile,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

            # Write the first SAMPLE_ROWS extracted rows to a separate sample file.
            if sample_rows_written < SAMPLE_ROWS:
                sample_chunk = chunk_filtered.head(SAMPLE_ROWS - sample_rows_written).copy()
                sample_chunk.to_csv(
                    sample_outfile,
                    mode="w" if sample_rows_written == 0 else "a",
                    header=sample_rows_written == 0,
                    index=False,
                )
                sample_rows_written += len(sample_chunk)

        print(
            f"[EHR][{chunk_idx}] read={len(chunk):,} keep={kept:,} "
            f"cum_keep={kept_rows_total:,} dt={time.time()-t_chunk0:.1f}s"
        )

    # Print the final extraction summary.
    print(f"[EHR] Raw chart covariates extracted. Total time: {time.time()-t0:.1f}s")
    print("[SAVE]", outfile)
    print("[SAVE SAMPLE]", sample_outfile)
    print("Rows:", kept_rows_total)
    print("[DONE]")


if __name__ == "__main__":
    main()
