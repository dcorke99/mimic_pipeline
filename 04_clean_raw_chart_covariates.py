"""
Clean raw chart-event covariates extracted for the CAUTI analysis cohort.

Expected input:
    data/preprocessed_raw_chart_covariates_kept.csv

Compatible with the output of the raw extraction script that writes:
    subject_id, hadm_id, stay_id, itemid, charttime, storetime,
    valuenum, value, valueuom

Outputs:
    data/raw_chart_covariates_cleaned.csv
    data/chart_covariate_cleaning_rules.csv
    data/chart_covariate_cleaning_audit.csv

Cleaning logic (per itemid):
1. Likely placeholder zeros -> set valuenum to missing
2. Mild tail values outside p1/p99 -> clip valuenum to p1/p99
3. Wildly extreme values far beyond the tail -> set valuenum to missing

The cleaned output preserves the same column structure as the input file.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DATADIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")

INFILE = DATADIR / "preprocessed_raw_chart_covariates_kept.csv"
OUTFILE = DATADIR / "cleaned_chart_covariates.csv"
RULES_OUTFILE = DATADIR / "chart_covariate_cleaning_rules.csv"
AUDIT_OUTFILE = DATADIR / "chart_covariate_cleaning_audit.csv"

ITEM_COL = "itemid"
VALUE_COL = "valuenum"

CHUNK_ROWS = 1_000_000
MIN_N_FOR_RULES = 100
ZERO_MAX_FRAC = 0.10
FAR_OUT_SPREAD_MULT = 3.0

# Optional manual overrides for specific itemids
ALWAYS_ZERO_TO_MISSING = set()
NEVER_ZERO_TO_MISSING = set()

# ============================================================
# FIT RULES
# ============================================================

def _load_numeric_values_by_itemid(infile: Path) -> dict[int, pd.Series]:
    # Stream item/value pairs and keep one numeric series per itemid.
    value_parts: dict[int, list[pd.Series]] = defaultdict(list)

    for chunk in pd.read_csv(
        infile,
        usecols=[ITEM_COL, VALUE_COL],
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL], errors="coerce")
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[ITEM_COL, VALUE_COL]).copy()
        if chunk.empty:
            continue

        chunk[ITEM_COL] = chunk[ITEM_COL].astype(int)
        for itemid, item_rows in chunk.groupby(ITEM_COL, sort=False):
            value_parts[int(itemid)].append(item_rows[VALUE_COL].reset_index(drop=True))

    return {
        itemid: pd.concat(parts, ignore_index=True)
        for itemid, parts in value_parts.items()
        if parts
    }


def _count_numeric_values_by_itemid(infile: Path) -> pd.DataFrame:
    # Stream the item/value pairs and count total and non-missing numeric rows per itemid.
    count_parts = []

    for chunk in pd.read_csv(
        infile,
        usecols=[ITEM_COL, VALUE_COL],
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        chunk[ITEM_COL] = pd.to_numeric(chunk[ITEM_COL], errors="coerce")
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[ITEM_COL]).copy()
        if chunk.empty:
            continue

        count_parts.append(
            chunk.groupby(ITEM_COL, sort=False).agg(
                n_rows_total=(VALUE_COL, "size"),
                n_non_missing=(VALUE_COL, lambda values: values.notna().sum()),
            )
        )

    if not count_parts:
        return pd.DataFrame(columns=[ITEM_COL, "n_rows_total", "n_non_missing"])

    counts = pd.concat(count_parts).groupby(level=0, sort=False).sum().reset_index()
    counts[ITEM_COL] = counts[ITEM_COL].astype(int)
    return counts


def fit_cleaning_rules(infile: Path) -> pd.DataFrame:
    # Stream only the item/value columns needed to fit item-level cleaning rules.
    values_by_itemid = _load_numeric_values_by_itemid(infile)
    if not values_by_itemid:
        raise ValueError("No numeric valuenum rows found in input file.")

    # Measure zero frequency for each itemid.
    rule_rows = []
    for itemid, values in values_by_itemid.items():
        nonzero_values = values[values != 0]
        rule_rows.append(
            {
                ITEM_COL: itemid,
                "n_non_missing": int(values.shape[0]),
                "zero_fraction": float(values.eq(0).mean()),
                "p5_nonzero": float(nonzero_values.quantile(0.05)) if not nonzero_values.empty else np.nan,
            }
        )

    rules = pd.DataFrame(rule_rows).set_index(ITEM_COL)
    rules["zero_to_missing"] = False

    # Flag low-frequency zeros as missing when the observed scale is otherwise positive.
    auto_zero_mask = (
        (rules["n_non_missing"] >= MIN_N_FOR_RULES)
        & (rules["p5_nonzero"] > 0)
        & (rules["zero_fraction"] > 0)
        & (rules["zero_fraction"] <= ZERO_MAX_FRAC)
    )
    rules.loc[auto_zero_mask, "zero_to_missing"] = True

    if ALWAYS_ZERO_TO_MISSING:
        rules.loc[rules.index.isin(ALWAYS_ZERO_TO_MISSING), "zero_to_missing"] = True
    if NEVER_ZERO_TO_MISSING:
        rules.loc[rules.index.isin(NEVER_ZERO_TO_MISSING), "zero_to_missing"] = False

    # Refit tail thresholds after removing itemids whose zeros should be ignored.
    threshold_rows = []
    for itemid, values in values_by_itemid.items():
        filtered_values = values.copy()
        if bool(rules.loc[itemid, "zero_to_missing"]):
            filtered_values = filtered_values[filtered_values != 0]

        if filtered_values.empty:
            threshold_rows.append(
                {
                    ITEM_COL: itemid,
                    "n_for_thresholds": 0,
                    "p1": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "p99": np.nan,
                }
            )
            continue

        threshold_rows.append(
            {
                ITEM_COL: itemid,
                "n_for_thresholds": int(filtered_values.shape[0]),
                "p1": float(filtered_values.quantile(0.01)),
                "q1": float(filtered_values.quantile(0.25)),
                "q3": float(filtered_values.quantile(0.75)),
                "p99": float(filtered_values.quantile(0.99)),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows).set_index(ITEM_COL)
    rules = rules.join(threshold_df, how="left")

    rules["status"] = "ok"
    too_few_mask = rules["n_for_thresholds"].fillna(0) < MIN_N_FOR_RULES
    rules.loc[too_few_mask, "status"] = "too_few_values_for_thresholds"

    # Store both clip thresholds and wider delete thresholds for each itemid.
    iqr = rules["q3"] - rules["q1"]
    tail_span = rules["p99"] - rules["p1"]
    spread = pd.concat([iqr, tail_span], axis=1).max(axis=1)
    spread = spread.fillna(0.0).clip(lower=1e-8)

    rules["lower_clip"] = rules["p1"]
    rules["upper_clip"] = rules["p99"]
    rules["lower_delete"] = rules["p1"] - FAR_OUT_SPREAD_MULT * spread
    rules["upper_delete"] = rules["p99"] + FAR_OUT_SPREAD_MULT * spread

    rules = rules.reset_index()
    return rules


# ============================================================
# APPLY RULES IN CHUNKS
# ============================================================

def apply_cleaning_rules(
    infile: Path,
    outfile: Path,
    rules: pd.DataFrame,
) -> pd.DataFrame:
    if outfile.exists():
        outfile.unlink()

    rules_small = rules[
        [
            ITEM_COL,
            "status",
            "zero_to_missing",
            "lower_clip",
            "upper_clip",
            "lower_delete",
            "upper_delete",
        ]
    ].copy()

    audit_parts = []
    first_write = True

    # Apply the fitted rules chunk-by-chunk to the full long-format file.
    for chunk in pd.read_csv(
        infile,
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        original_columns = list(chunk.columns)

        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")

        chunk = chunk.merge(
            rules_small,
            on=ITEM_COL,
            how="left",
        )

        # Track which cleaning action each row receives.
        action = pd.Series("unchanged", index=chunk.index, dtype="object")
        action.loc[chunk[VALUE_COL].isna()] = "original_missing_or_non_numeric"

        no_rule_mask = chunk["status"].isna() & chunk[VALUE_COL].notna()
        action.loc[no_rule_mask] = "no_rule"

        bad_rule_mask = (
            chunk["status"].notna()
            & chunk["status"].ne("ok")
            & chunk[VALUE_COL].notna()
        )
        action.loc[bad_rule_mask] = "rule_not_applied"

        # 1) Zero -> missing
        zero_mask = (
            chunk["status"].eq("ok")
            & chunk["zero_to_missing"].fillna(False)
            & chunk[VALUE_COL].eq(0)
        )
        chunk.loc[zero_mask, VALUE_COL] = np.nan
        action.loc[zero_mask] = "zero_to_missing"

        # 2) Far out -> missing
        far_low_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] < chunk["lower_delete"])
        )
        far_high_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] > chunk["upper_delete"])
        )
        far_mask = far_low_mask | far_high_mask
        chunk.loc[far_mask, VALUE_COL] = np.nan
        action.loc[far_mask] = "far_out_to_missing"

        # 3) Mild tails -> clip
        clip_low_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] < chunk["lower_clip"])
        )
        clip_high_mask = (
            chunk["status"].eq("ok")
            & chunk[VALUE_COL].notna()
            & (chunk[VALUE_COL] > chunk["upper_clip"])
        )

        chunk.loc[clip_low_mask, VALUE_COL] = chunk.loc[clip_low_mask, "lower_clip"]
        chunk.loc[clip_high_mask, VALUE_COL] = chunk.loc[clip_high_mask, "upper_clip"]

        clip_mask = (clip_low_mask | clip_high_mask) & action.eq("unchanged")
        action.loc[clip_mask] = "tail_clipped"

        # Summarise the cleaning actions for this chunk.
        audit_chunk = (
            pd.DataFrame({
                ITEM_COL: chunk[ITEM_COL],
                "action": action
            })
            .groupby([ITEM_COL, "action"], dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
        )
        audit_parts.append(audit_chunk)

        # Write back only the original columns so the output schema stays unchanged.
        chunk = chunk[original_columns]

        chunk.to_csv(
            outfile,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

    audit_long = pd.concat(audit_parts, ignore_index=True)
    audit_long = (
        audit_long.groupby([ITEM_COL, "action"], dropna=False)["n_rows"]
        .sum()
        .reset_index()
    )

    # Pivot the action summary wide so it can be merged with the rule table.
    audit_wide = audit_long.pivot(
        index=ITEM_COL,
        columns="action",
        values="n_rows",
    ).fillna(0)
    audit_wide.columns.name = None
    audit_wide = audit_wide.reset_index()

    before_counts = _count_numeric_values_by_itemid(infile).rename(
        columns={"n_non_missing": "n_non_missing_before"}
    )
    after_counts = _count_numeric_values_by_itemid(outfile).rename(
        columns={"n_non_missing": "n_non_missing_after"}
    )[[ITEM_COL, "n_non_missing_after"]]

    audit = (
        rules.merge(audit_wide, on=ITEM_COL, how="left")
        .merge(before_counts, on=ITEM_COL, how="left")
        .merge(after_counts, on=ITEM_COL, how="left")
    )

    # Fill missing action counts with zero while leaving rule parameters untouched.
    for col in audit.columns:
        if col not in {
            ITEM_COL,
            "n_non_missing",
            "zero_fraction",
            "p5_nonzero",
            "zero_to_missing",
            "n_for_thresholds",
            "p1",
            "q1",
            "q3",
            "p99",
            "status",
            "lower_clip",
            "upper_clip",
            "lower_delete",
            "upper_delete",
        }:
            audit[col] = audit[col].fillna(0)

    return audit


# ============================================================
# MAIN
# ============================================================

def main():
    # Fit the item-level rules and save them for inspection.
    print("[FIT RULES]", INFILE)
    rules = fit_cleaning_rules(INFILE)
    rules.to_csv(RULES_OUTFILE, index=False)
    print("[SAVE RULES]", RULES_OUTFILE)

    # Apply the rules to the full file and save the cleaning audit.
    print("[APPLY RULES]", INFILE)
    audit = apply_cleaning_rules(INFILE, OUTFILE, rules)
    print("[SAVE CLEANED]", OUTFILE)

    audit.to_csv(AUDIT_OUTFILE, index=False)
    print("[SAVE AUDIT]", AUDIT_OUTFILE)

    print("[DONE]")


if __name__ == "__main__":
    main()
