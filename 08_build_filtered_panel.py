from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
MASTER_DATASET = DATA_DIR / "master_panel.csv"
SUMMARY_CSV = DATA_DIR / "covariate_retention_log.csv"
OUT_DATASET = DATA_DIR / "filtered_panel.csv"
SPLIT_CSV = DATA_DIR / "train_test_split.csv"

ROUND_DP = 3
SUBJECT_ID_COL = "subject_id"
TEST_SIZE = 0.20
SEED = 42


def create_patient_split(panel: pd.DataFrame) -> pd.DataFrame:
    # Split unique patients once so every row for a patient stays together.
    subject_ids = panel[SUBJECT_ID_COL].dropna().astype(str).unique()
    train_ids, test_ids = train_test_split(
        subject_ids,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    split_df = pd.DataFrame(
        {
            SUBJECT_ID_COL: list(train_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        }
    )
    return split_df


def main() -> None:
    # Load the retain/drop decisions and keep only retained covariates.
    summary_df = pd.read_csv(SUMMARY_CSV, low_memory=False)
    retained_cov_cols = summary_df.loc[
        summary_df["decision"].astype(str).str.lower() == "retain",
        "column_name",
    ].astype(str).tolist()

    # Load the full panel and standardise the subject identifier.
    panel = pd.read_csv(MASTER_DATASET, low_memory=False)
    panel[SUBJECT_ID_COL] = panel[SUBJECT_ID_COL].astype(str)

    # Separate base columns from itemid-derived covariates.
    covariate_cols = [column_name for column_name in panel.columns if column_name.startswith("itemid_") and "__" in column_name]
    base_cols = [column_name for column_name in panel.columns if column_name not in covariate_cols]

    retained_cov_set = set(retained_cov_cols)
    kept_cov_cols = [column_name for column_name in covariate_cols if column_name in retained_cov_set]
    filtered_panel = panel[base_cols + kept_cov_cols].copy()

    # Coerce retained covariates to numeric once before rounding.
    for column_name in kept_cov_cols:
        filtered_panel[column_name] = pd.to_numeric(filtered_panel[column_name], errors="coerce")
    filtered_panel[kept_cov_cols] = filtered_panel[kept_cov_cols].round(ROUND_DP)

    # Build the subject-level split and join it back to every row.
    split_df = create_patient_split(filtered_panel)
    split_df[SUBJECT_ID_COL] = split_df[SUBJECT_ID_COL].astype(str)
    filtered_panel = filtered_panel.merge(split_df, on=SUBJECT_ID_COL, how="left")

    # Save the filtered panel and the persistent split table.
    filtered_panel.to_csv(OUT_DATASET, index=False)
    split_df.to_csv(SPLIT_CSV, index=False)

    patient_total = split_df[SUBJECT_ID_COL].nunique()
    train_patients = split_df.loc[split_df["split"] == "train", SUBJECT_ID_COL].nunique()
    test_patients = split_df.loc[split_df["split"] == "test", SUBJECT_ID_COL].nunique()
    train_rows = int((filtered_panel["split"] == "train").sum())
    test_rows = int((filtered_panel["split"] == "test").sum())

    print(f"[READ]  {MASTER_DATASET} rows={len(panel):,} cols={len(panel.columns):,}")
    print(f"[KEEP]  retained covariate columns={len(kept_cov_cols):,}")
    print(f"[SPLIT] patients total={patient_total:,} train={train_patients:,} test={test_patients:,}")
    print(f"[SPLIT] rows train={train_rows:,} test={test_rows:,}")
    print(f"[WRITE] {OUT_DATASET} rows={len(filtered_panel):,} cols={len(filtered_panel.columns):,}")
    print(f"[WRITE] {SPLIT_CSV} rows={len(split_df):,} cols={len(split_df.columns):,}")


if __name__ == "__main__":
    main()
