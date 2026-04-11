from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Config
DATA_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\cleaned_chart_covariates.csv")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

# Optional bounds file.
# Expected columns can include:
#   itemid,lower_bound,upper_bound
# or
#   itemid,stat,lower_bound,upper_bound
# If stat is present it is ignored here because this script audits raw values.
BOUNDS_FILE = OUTDIR / "panel_covariate_bounds.csv"

DP = 3
CHUNK_ROWS = 1_000_000


# Load d_items labels for readability.
def load_item_labels(d_items_path: Path) -> dict[int, str]:
    # Require the label table so item-level audit output is interpretable.
    if not d_items_path.exists():
        raise FileNotFoundError(f"d_items file not found: {d_items_path}")

    d_items_df = pd.read_csv(d_items_path, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items_df["itemid"] = pd.to_numeric(d_items_df["itemid"], errors="coerce")
    d_items_df = d_items_df.dropna(subset=["itemid"]).copy()
    d_items_df["itemid"] = d_items_df["itemid"].astype(int)
    d_items_df["label"] = d_items_df["label"].astype(str)
    return d_items_df.set_index("itemid")["label"].to_dict()


# Load optional bounds file, resolving bounds at the itemid level.
def load_bounds(bounds_file: Path) -> dict[int, tuple[float, float]]:
    by_itemid: dict[int, tuple[float, float]] = {}

    if not bounds_file.exists():
        return by_itemid

    bounds_df = pd.read_csv(bounds_file)

    required_cols = {"itemid", "lower_bound", "upper_bound"}
    if not required_cols.issubset(bounds_df.columns):
        raise ValueError(f"Bounds file is missing required columns: {sorted(required_cols)}")

    bounds_rows = bounds_df[["itemid", "lower_bound", "upper_bound"]].copy()
    bounds_rows["itemid"] = pd.to_numeric(bounds_rows["itemid"], errors="coerce")
    bounds_rows["lower_bound"] = pd.to_numeric(bounds_rows["lower_bound"], errors="coerce")
    bounds_rows["upper_bound"] = pd.to_numeric(bounds_rows["upper_bound"], errors="coerce")
    bounds_rows = bounds_rows.dropna(subset=["itemid"]).copy()
    bounds_rows["itemid"] = bounds_rows["itemid"].astype(int)

    # Keep the first bounds row for each itemid.
    bounds_rows = bounds_rows.drop_duplicates(subset=["itemid"], keep="first")

    for _, row in bounds_rows.iterrows():
        by_itemid[int(row["itemid"])] = (row["lower_bound"], row["upper_bound"])

    return by_itemid


# Build the raw long-format audit table.
def build_raw_value_audit_from_parts(
    value_parts: dict[int, list[pd.Series]],
    row_counts: dict[int, int],
    unit_counts_by_itemid: dict[int, dict[str, int]],
    itemid_to_label: dict[int, str],
    bounds_by_itemid: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for itemid in sorted(row_counts):
        non_missing = (
            pd.concat(value_parts[itemid], ignore_index=True)
            if value_parts[itemid]
            else pd.Series(dtype=float)
        )

        rows_for_itemid = int(row_counts[itemid])
        non_missing_n = int(non_missing.shape[0])
        pct_missing = float((1.0 - (non_missing_n / rows_for_itemid)) * 100.0) if rows_for_itemid > 0 else np.nan

        lower_bound, upper_bound = bounds_by_itemid.get(itemid, (np.nan, np.nan))

        if non_missing_n > 0:
            min_val = float(non_missing.min())
            p01 = float(non_missing.quantile(0.01))
            median_val = float(non_missing.median())
            p99 = float(non_missing.quantile(0.99))
            max_val = float(non_missing.max())
        else:
            min_val = np.nan
            p01 = np.nan
            median_val = np.nan
            p99 = np.nan
            max_val = np.nan

        if pd.notna(lower_bound):
            n_below = int((non_missing < lower_bound).sum())
        else:
            n_below = np.nan

        if pd.notna(upper_bound):
            n_above = int((non_missing > upper_bound).sum())
        else:
            n_above = np.nan

        if pd.notna(n_below) and pd.notna(n_above) and non_missing_n > 0:
            n_out_of_range = int(n_below + n_above)
            pct_out_of_range_non_missing = float((n_out_of_range / non_missing_n) * 100.0)
        else:
            n_out_of_range = np.nan
            pct_out_of_range_non_missing = np.nan

        unit_counts = pd.Series(unit_counts_by_itemid[itemid]).sort_values(ascending=False)

        if len(unit_counts) > 0:
            n_unique_units = int(unit_counts.shape[0])
            top_unit = str(unit_counts.index[0])
            top_unit_n = int(unit_counts.iloc[0])
            units_seen = " | ".join([f"{u} ({n})" for u, n in unit_counts.items()])
        else:
            n_unique_units = 0
            top_unit = ""
            top_unit_n = 0
            units_seen = ""

        rows.append({
            "itemid": itemid,
            "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
            "rows_for_itemid": rows_for_itemid,
            "non_missing_n": non_missing_n,
            "pct_missing": pct_missing,
            "min": min_val,
            "p01": p01,
            "median": median_val,
            "p99": p99,
            "max": max_val,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_below_lower_bound": n_below,
            "n_above_upper_bound": n_above,
            "n_out_of_range": n_out_of_range,
            "pct_out_of_range_non_missing": pct_out_of_range_non_missing,
            "n_unique_units": n_unique_units,
            "top_unit": top_unit,
            "top_unit_n": top_unit_n,
            "units_seen": units_seen,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(DP)

    return out


# Build an explicit itemid x unit table for unit-mix review.
def build_unit_audit_from_parts(
    unit_counts_by_itemid: dict[int, dict[str, int]],
    itemid_to_label: dict[int, str],
) -> pd.DataFrame:
    rows = []

    for itemid in sorted(unit_counts_by_itemid):
        for unit, n_rows in unit_counts_by_itemid[itemid].items():
            rows.append({
                "itemid": itemid,
                "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
                "valueuom": unit,
                "n_rows": n_rows,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out.reindex(columns=["itemid", "label", "valueuom", "n_rows"])

    return out.sort_values(
        ["itemid", "n_rows", "valueuom"],
        ascending=[True, False, True]
    ).reset_index(drop=True)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Raw chart file not found: {DATA_FILE}")

    header = pd.read_csv(DATA_FILE, nrows=0)
    header.columns = header.columns.str.strip()
    available_cols = set(header.columns)

    desired_cols = ["stay_id", "itemid", "charttime", "valuenum", "value", "valueuom"]
    usecols = [c for c in desired_cols if c in available_cols]

    print(f"[RAW FILE] {DATA_FILE}")
    print(f"[AVAILABLE COLS] {sorted(available_cols)}")
    print(f"[READING COLS] {usecols}")

    if "itemid" not in available_cols:
        raise ValueError("Raw chart file must contain 'itemid'.")

    if "valuenum" not in available_cols and "value" not in available_cols:
        raise ValueError("Raw chart file must contain at least one of 'valuenum' or 'value'.")

    itemid_to_label = load_item_labels(D_ITEMS_PATH)
    bounds_by_itemid = load_bounds(BOUNDS_FILE)
    row_counts: dict[int, int] = defaultdict(int)
    value_parts: dict[int, list[pd.Series]] = defaultdict(list)
    unit_counts_by_itemid: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Stream the cleaned file so the audit can handle large chart tables.
    for chunk in pd.read_csv(DATA_FILE, usecols=usecols, chunksize=CHUNK_ROWS, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk["valueuom"] = chunk["valueuom"].fillna("").astype(str).str.strip()

        chunk = chunk.dropna(subset=["itemid"]).copy()
        chunk["itemid"] = chunk["itemid"].astype(int)

        row_count_chunk = chunk.groupby("itemid").size()
        for itemid, count in row_count_chunk.items():
            row_counts[int(itemid)] += int(count)

        numeric_chunk = chunk.dropna(subset=["valuenum"])
        for itemid, g in numeric_chunk.groupby("itemid", sort=False):
            value_parts[int(itemid)].append(g["valuenum"].reset_index(drop=True))

        unit_chunk = chunk.loc[chunk["valueuom"] != "", ["itemid", "valueuom"]].copy()
        if len(unit_chunk) > 0:
            unit_count_chunk = unit_chunk.groupby(["itemid", "valueuom"]).size()
            for (itemid, unit), count in unit_count_chunk.items():
                unit_counts_by_itemid[int(itemid)][str(unit)] += int(count)

    audit = build_raw_value_audit_from_parts(
        value_parts=value_parts,
        row_counts=row_counts,
        unit_counts_by_itemid=unit_counts_by_itemid,
        itemid_to_label=itemid_to_label,
        bounds_by_itemid=bounds_by_itemid,
    )
    audit.to_csv(OUTDIR / "cleaned_chart_integrity_audit.csv", index=False)

    # Save a second copy sorted to surface the most suspicious variables first.
    audit_problem = audit.copy()
    audit_problem["abs_max_minus_p99"] = (
        pd.to_numeric(audit_problem["max"], errors="coerce")
        - pd.to_numeric(audit_problem["p99"], errors="coerce")
    ).abs()
    audit_problem["abs_p01_minus_min"] = (
        pd.to_numeric(audit_problem["p01"], errors="coerce")
        - pd.to_numeric(audit_problem["min"], errors="coerce")
    ).abs()

    sort_cols = [
        "n_out_of_range",
        "pct_out_of_range_non_missing",
        "n_unique_units",
        "abs_max_minus_p99",
        "abs_p01_minus_min",
        "pct_missing",
    ]
    audit_problem = audit_problem.sort_values(sort_cols, ascending=[False, False, False, False, False, False])
    audit_problem.to_csv(OUTDIR / "cleaned_chart_integrity_audit__sorted_problem_first.csv", index=False)

    unit_audit = build_unit_audit_from_parts(unit_counts_by_itemid, itemid_to_label)
    unit_audit.to_csv(OUTDIR / "cleaned_chart_units_by_itemid.csv", index=False)

    print(f"Saved: {OUTDIR / 'cleaned_chart_integrity_audit.csv'}")
    print(f"Saved: {OUTDIR / 'cleaned_chart_integrity_audit__sorted_problem_first.csv'}")
    print(f"Saved: {OUTDIR / 'cleaned_chart_units_by_itemid.csv'}")


if __name__ == "__main__":
    main()
