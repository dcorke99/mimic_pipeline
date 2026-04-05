from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Config
DATA_FILE = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data\raw_chart_covariates.csv")
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


# Load d_items labels for readability.
def load_item_labels(d_items_path: Path) -> dict[int, str]:
    if not d_items_path.exists():
        return {}

    d_items = pd.read_csv(d_items_path, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
    d_items = d_items.dropna(subset=["itemid"]).copy()
    d_items["itemid"] = d_items["itemid"].astype(int)
    d_items["label"] = d_items["label"].astype(str)
    itemid_label_map = d_items.set_index("itemid")["label"].to_dict()
    return {k: v for k, v in itemid_label_map.items()}


# Load optional bounds file, resolving bounds at the itemid level.
def load_bounds(bounds_file: Path) -> dict[int, tuple[float, float]]:
    by_itemid: dict[int, tuple[float, float]] = {}

    if not bounds_file.exists():
        return by_itemid

    b = pd.read_csv(bounds_file)

    if not {"itemid", "lower_bound", "upper_bound"}.issubset(b.columns):
        return by_itemid

    tmp = b[["itemid", "lower_bound", "upper_bound"]].copy()
    tmp["itemid"] = pd.to_numeric(tmp["itemid"], errors="coerce")
    tmp["lower_bound"] = pd.to_numeric(tmp["lower_bound"], errors="coerce")
    tmp["upper_bound"] = pd.to_numeric(tmp["upper_bound"], errors="coerce")
    tmp = tmp.dropna(subset=["itemid"]).copy()
    tmp["itemid"] = tmp["itemid"].astype(int)

    # Keep the first bounds row for each itemid.
    tmp = tmp.drop_duplicates(subset=["itemid"], keep="first")

    for _, row in tmp.iterrows():
        by_itemid[int(row["itemid"])] = (row["lower_bound"], row["upper_bound"])

    return by_itemid


# Build the raw long-format audit table.
def build_raw_value_audit(
    df: pd.DataFrame,
    itemid_to_label: dict[int, str],
    bounds_by_itemid: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = df.groupby("itemid", dropna=False)

    for itemid, g in grouped:
        if pd.isna(itemid):
            continue

        itemid_num = pd.to_numeric(pd.Series([itemid]), errors="coerce").iloc[0]
        if pd.isna(itemid_num):
            continue
        itemid = int(itemid_num)
        s = pd.to_numeric(g["valuenum"], errors="coerce")
        non_missing = s.dropna()

        rows_for_itemid = int(len(g))
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

        unit_non_missing = g["valueuom"].fillna("").astype(str).str.strip()
        unit_non_missing = unit_non_missing[unit_non_missing != ""]
        unit_counts = unit_non_missing.value_counts()

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
def build_unit_audit(df: pd.DataFrame, itemid_to_label: dict[int, str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["itemid"] = pd.to_numeric(tmp["itemid"], errors="coerce")
    tmp["valueuom"] = tmp["valueuom"].fillna("").astype(str).str.strip()
    tmp = tmp.dropna(subset=["itemid"]).copy()
    tmp["itemid"] = tmp["itemid"].astype(int)

    out = (
        tmp.groupby(["itemid", "valueuom"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["itemid", "n_rows", "valueuom"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    out["label"] = out["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    return out[["itemid", "label", "valueuom", "n_rows"]]


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

    df = pd.read_csv(DATA_FILE, usecols=usecols, low_memory=False)
    df.columns = df.columns.str.strip()

    if "valuenum" not in df.columns:
        df["valuenum"] = np.nan
    if "value" not in df.columns:
        df["value"] = np.nan
    if "valueuom" not in df.columns:
        df["valueuom"] = ""
    if "stay_id" not in df.columns:
        df["stay_id"] = np.nan
    if "charttime" not in df.columns:
        df["charttime"] = pd.NaT

    df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")

    itemid_to_label = load_item_labels(D_ITEMS_PATH)
    bounds_by_itemid = load_bounds(BOUNDS_FILE)

    audit = build_raw_value_audit(
        df=df,
        itemid_to_label=itemid_to_label,
        bounds_by_itemid=bounds_by_itemid,
    )
    audit.to_csv(OUTDIR / "raw_chart_integrity_audit.csv", index=False)

    # Convenience file: likely problematic variables first.
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
    audit_problem.to_csv(OUTDIR / "raw_chart_integrity_audit__sorted_problem_first.csv", index=False)

    unit_audit = build_unit_audit(df, itemid_to_label)
    unit_audit.to_csv(OUTDIR / "raw_chart_units_by_itemid.csv", index=False)

    print(f"Saved: {OUTDIR / 'raw_chart_integrity.csv'}")
    print(f"Saved: {OUTDIR / 'raw_chart_integrity_sorted.csv'}")
    print(f"Saved: {OUTDIR / 'raw_chart_units_by_itemid.csv'}")


if __name__ == "__main__":
    main()
