"""
Filter preprocessed chart-event covariates to the configured item allowlist.

This runs after temperature/unit preprocessing so remapped itemids, such as
Fahrenheit temperature rows reassigned to Celsius, are filtered by their
canonical itemid.
"""

from pathlib import Path

import pandas as pd

DATADIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
CONFIGDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\config")
INFILE = DATADIR / "preprocessed_raw_chart_covariates.csv"
OUTFILE = DATADIR / "preprocessed_raw_chart_covariates_kept.csv"
SAMPLE_OUTFILE = DATADIR / "preprocessed_raw_chart_covariates_kept__first_1000_rows.csv"
TMP_OUTFILE = OUTFILE.with_suffix(OUTFILE.suffix + ".writing")
TMP_SAMPLE_OUTFILE = SAMPLE_OUTFILE.with_suffix(SAMPLE_OUTFILE.suffix + ".writing")
D_ITEMS_KEEP_PATH = CONFIGDIR / "d_items_keep.csv"

CHUNK_ROWS = 1_000_000
SAMPLE_ROWS = 1000


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot remove existing file: {path}\n"
            "Close any program using it, including Excel/preview panes, and pause or "
            "let OneDrive finish syncing this folder before rerunning the pipeline."
        ) from exc


def replace_output(tmp_path: Path, final_path: Path) -> None:
    try:
        tmp_path.replace(final_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot replace output file: {final_path}\n"
            "Close any program using it, including Excel/preview panes, and pause or "
            "let OneDrive finish syncing this folder before rerunning the pipeline.\n"
            f"The completed replacement file is still available at: {tmp_path}"
        ) from exc


def load_keep_itemids() -> set[int]:
    # d_items_keep.csv has the same structure as MIMIC d_items.csv.
    keep_df = pd.read_csv(
        D_ITEMS_KEEP_PATH,
        usecols=["itemid"],
        low_memory=False,
    )
    keep_df["itemid"] = pd.to_numeric(keep_df["itemid"], errors="coerce")
    keep_df = keep_df.dropna(subset=["itemid"]).copy()
    return set(keep_df["itemid"].astype(int))


def main() -> None:
    DATADIR.mkdir(exist_ok=True, parents=True)

    keep_itemids = load_keep_itemids()

    remove_if_exists(TMP_OUTFILE)
    remove_if_exists(TMP_SAMPLE_OUTFILE)

    first_write = True
    sample_rows_written = 0
    chunk_idx = 0
    read_rows_total = 0
    kept_rows_total = 0

    print(f"[LOAD] keep itemids={len(keep_itemids):,} from {D_ITEMS_KEEP_PATH}")

    for chunk in pd.read_csv(INFILE, chunksize=CHUNK_ROWS, low_memory=False):
        chunk_idx += 1
        read_rows_total += len(chunk)

        itemids = pd.to_numeric(chunk["itemid"], errors="coerce")
        keep_mask = itemids.isin(keep_itemids)
        kept = chunk.loc[keep_mask].copy()
        kept_rows_total += len(kept)

        if len(kept) > 0:
            try:
                kept.to_csv(
                    TMP_OUTFILE,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                )
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot write temporary output file: {TMP_OUTFILE}\n"
                    "Close any program using it, including Excel/preview panes, and pause "
                    "or let OneDrive finish syncing this folder before rerunning the pipeline."
                ) from exc
            first_write = False

            if sample_rows_written < SAMPLE_ROWS:
                sample_chunk = kept.head(SAMPLE_ROWS - sample_rows_written).copy()
                try:
                    sample_chunk.to_csv(
                        TMP_SAMPLE_OUTFILE,
                        mode="w" if sample_rows_written == 0 else "a",
                        header=sample_rows_written == 0,
                        index=False,
                    )
                except PermissionError as exc:
                    raise PermissionError(
                        f"Cannot write temporary sample file: {TMP_SAMPLE_OUTFILE}\n"
                        "Close any program using it, including Excel/preview panes, and pause "
                        "or let OneDrive finish syncing this folder before rerunning the pipeline."
                    ) from exc
                sample_rows_written += len(sample_chunk)

        print(
            f"[CHUNK {chunk_idx}] rows={len(chunk):,} "
            f"keep={len(kept):,} cum_keep={kept_rows_total:,}"
        )

    if first_write:
        header = pd.read_csv(INFILE, nrows=0)
        header.to_csv(TMP_OUTFILE, index=False)
        print(f"[WARN] no rows matched d_items_keep.csv; wrote empty file with headers: {OUTFILE}")

    replace_output(TMP_OUTFILE, OUTFILE)
    print(f"[SAVE] {OUTFILE}")
    if TMP_SAMPLE_OUTFILE.exists():
        replace_output(TMP_SAMPLE_OUTFILE, SAMPLE_OUTFILE)
        print(f"[SAVE SAMPLE] {SAMPLE_OUTFILE}")
    print(f"[INFO] rows read: {read_rows_total:,}")
    print(f"[INFO] rows kept: {kept_rows_total:,}")
    print(f"[INFO] rows dropped: {read_rows_total - kept_rows_total:,}")


if __name__ == "__main__":
    main()
