"""
Clean raw chart-event covariates extracted for the CAUTI analysis cohort.

Expected input:
    data/raw_chart_covariates.csv

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
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DATADIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")

INFILE = DATADIR / "raw_chart_covariates.csv"
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

def fit_cleaning_rules(infile: Path) -> pd.DataFrame:
    slim = pd.read_csv(
        infile,
        usecols=[ITEM_COL, VALUE_COL],
        low_memory=False,
    )

    slim[VALUE_COL] = pd.to_numeric(slim[VALUE_COL], errors="coerce")
    slim = slim.dropna(subset=[ITEM_COL])
    slim = slim.dropna(subset=[VALUE_COL]).copy()

    if slim.empty:
        raise ValueError("No numeric valuenum rows found in input file.")

    slim["is_zero"] = slim[VALUE_COL].eq(0)

    base_stats = (
        slim.groupby(ITEM_COL, sort=False)
        .agg(
            n_non_missing=(VALUE_COL, "size"),
            zero_fraction=("is_zero", "mean"),
        )
    )

    nonzero = slim[slim[VALUE_COL] != 0].copy()
    if nonzero.empty:
        p5_nonzero = pd.Series(dtype=float, name="p5_nonzero")
    else:
        p5_nonzero = (
            nonzero.groupby(ITEM_COL, sort=False)[VALUE_COL]
            .quantile(0.05)
            .rename("p5_nonzero")
        )

    rules = base_stats.join(p5_nonzero, how="left")
    rules["zero_to_missing"] = False

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

    fit_df = slim[[ITEM_COL, VALUE_COL]].merge(
        rules[["zero_to_missing"]],
        left_on=ITEM_COL,
        right_index=True,
        how="left",
    )

    remove_zero_mask = fit_df["zero_to_missing"].fillna(False) & fit_df[VALUE_COL].eq(0)
    fit_df = fit_df.loc[~remove_zero_mask, [ITEM_COL, VALUE_COL]].copy()

    n_for_thresholds = (
        fit_df.groupby(ITEM_COL, sort=False)
        .size()
        .rename("n_for_thresholds")
    )

    q = (
        fit_df.groupby(ITEM_COL, sort=False)[VALUE_COL]
        .quantile([0.01, 0.25, 0.75, 0.99])
        .unstack()
    )
    q = q.rename(columns={0.01: "p1", 0.25: "q1", 0.75: "q3", 0.99: "p99"})

    rules = rules.join(n_for_thresholds, how="left").join(q, how="left")

    rules["status"] = "ok"
    too_few_mask = rules["n_for_thresholds"].fillna(0) < MIN_N_FOR_RULES
    rules.loc[too_few_mask, "status"] = "too_few_values_for_thresholds"

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

        # audit
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

        # write only original columns back out
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

    audit_wide = audit_long.pivot(
        index=ITEM_COL,
        columns="action",
        values="n_rows",
    ).fillna(0)
    audit_wide.columns.name = None
    audit_wide = audit_wide.reset_index()

    cleaned_slim = pd.read_csv(
        outfile,
        usecols=[ITEM_COL, VALUE_COL],
        low_memory=False,
    )
    cleaned_slim[VALUE_COL] = pd.to_numeric(cleaned_slim[VALUE_COL], errors="coerce")

    before_slim = pd.read_csv(
        infile,
        usecols=[ITEM_COL, VALUE_COL],
        low_memory=False,
    )
    before_slim[VALUE_COL] = pd.to_numeric(before_slim[VALUE_COL], errors="coerce")

    before_counts = (
        before_slim.groupby(ITEM_COL, sort=False)
        .agg(
            n_rows_total=(VALUE_COL, "size"),
            n_non_missing_before=(VALUE_COL, lambda s: s.notna().sum()),
        )
        .reset_index()
    )

    after_counts = (
        cleaned_slim.groupby(ITEM_COL, sort=False)
        .agg(
            n_non_missing_after=(VALUE_COL, lambda s: s.notna().sum()),
        )
        .reset_index()
    )

    audit = (
        rules.merge(audit_wide, on=ITEM_COL, how="left")
        .merge(before_counts, on=ITEM_COL, how="left")
        .merge(after_counts, on=ITEM_COL, how="left")
    )

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
    print("[FIT RULES]", INFILE)
    rules = fit_cleaning_rules(INFILE)
    rules.to_csv(RULES_OUTFILE, index=False)
    print("[SAVE RULES]", RULES_OUTFILE)

    print("[APPLY RULES]", INFILE)
    audit = apply_cleaning_rules(INFILE, OUTFILE, rules)
    print("[SAVE CLEANED]", OUTFILE)

    audit.to_csv(AUDIT_OUTFILE, index=False)
    print("[SAVE AUDIT]", AUDIT_OUTFILE)

    print("[DONE]")


if __name__ == "__main__":
    main()