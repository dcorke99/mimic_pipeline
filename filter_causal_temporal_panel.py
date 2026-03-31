from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration

# Data folder
DATA_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")

# Input and output files
MASTER_DATASET = DATA_DIR / "master_panel.csv"
SUMMARY_CSV = DATA_DIR / "covariate_retention_log.csv"
OUT_DATASET = DATA_DIR / "filtered_panel.csv"
SPLIT_CSV = DATA_DIR / "train_test_split.csv"

# Covariate statistics to keep
KEEP_STATS = ["mean"]

# Decimal places for retained covariates
ROUND_DP = 3

# Patient-level split settings
SUBJECT_ID_COL = "subject_id"
TEST_SIZE = 0.20
SEED = 42


# Extract itemid and statistic name from a covariate column
def parse_itemid_and_stat(col: str):
    if not col.startswith("itemid_") or "__" not in col:
        return None, None
    left, stat = col.split("__", 1)
    try:
        itemid = int(left.split("_", 1)[1])
        return itemid, stat.lower()
    except (IndexError, ValueError):
        return None, None


# Create a patient-level train/test split
def create_patient_split(df: pd.DataFrame) -> pd.DataFrame:
    patient_ids = df[SUBJECT_ID_COL].dropna().unique()

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    split_df = pd.DataFrame({
        SUBJECT_ID_COL: list(train_ids) + list(test_ids),
        "split": ["train"] * len(train_ids) + ["test"] * len(test_ids)
    })

    return split_df

# Run the retention filter, round retained covariates, and assign train/test splits.
def main():
    keep_stats = [s.lower() for s in KEEP_STATS]

    # Load retain and drop decisions
    summary = pd.read_csv(SUMMARY_CSV, low_memory=False)

    # Collect itemids marked for retention
    # Build the retained itemid set from the selection log generated upstream.
    retain_itemids = set(
        summary.loc[
            summary["decision"].astype(str).str.lower() == "retain", "itemid"
        ].astype(int)
    )

    # Load the master dataset
    df = pd.read_csv(MASTER_DATASET, low_memory=False)

    # Convert subject ids to strings before splitting and merging
    df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

    # Separate covariate columns from non-covariate columns
    cov_cols = [c for c in df.columns if c.startswith("itemid_") and "__" in c]
    base_cols = [c for c in df.columns if c not in cov_cols]

    # Keep retained covariates matching the requested statistics
    kept_cov_cols = []
    for c in cov_cols:
        itemid, stat = parse_itemid_and_stat(c)
        if itemid in retain_itemids and stat in keep_stats:
            kept_cov_cols.append(c)

    out_df = df[base_cols + kept_cov_cols].copy()

    # Convert retained covariates to numeric and round values
    for c in kept_cov_cols:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce")
    out_df[kept_cov_cols] = out_df[kept_cov_cols].round(ROUND_DP)

    # Create a persistent patient-level split
    split_df = create_patient_split(out_df)
    split_df[SUBJECT_ID_COL] = split_df[SUBJECT_ID_COL].astype(str)

    # Attach the patient-level split back onto every panel row.
    out_df = out_df.merge(split_df, on=SUBJECT_ID_COL, how="left")

    # Write output datasets
    out_df.to_csv(OUT_DATASET, index=False)
    split_df.to_csv(SPLIT_CSV, index=False)

    # Print summary information
    n_patients = split_df[SUBJECT_ID_COL].nunique()
    n_train_patients = split_df.loc[split_df["split"] == "train", SUBJECT_ID_COL].nunique()
    n_test_patients = split_df.loc[split_df["split"] == "test", SUBJECT_ID_COL].nunique()

    train_rows = int((out_df["split"] == "train").sum())
    test_rows = int((out_df["split"] == "test").sum())

    print(f"[READ]  {MASTER_DATASET} rows={len(df):,} cols={len(df.columns):,}")
    print(f"[KEEP]  retained itemids={len(retain_itemids):,}")
    print(f"[KEEP]  stats={keep_stats} covariate columns={len(kept_cov_cols):,}")
    print(f"[SPLIT] patients total={n_patients:,} train={n_train_patients:,} test={n_test_patients:,}")
    print(f"[SPLIT] rows train={train_rows:,} test={test_rows:,}")
    print(f"[WRITE] {OUT_DATASET} rows={len(out_df):,} cols={len(out_df.columns):,}")
    print(f"[WRITE] {SPLIT_CSV} rows={len(split_df):,} cols={len(split_df.columns):,}")


if __name__ == "__main__":
    main()
