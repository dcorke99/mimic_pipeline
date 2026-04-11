from pathlib import Path
import re

import pandas as pd

# Configuration
DATA_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
DATASET_PATH = DATA_DIR / "master_panel.csv"
OUT_PATH = DATA_DIR / "covariate_retention_log.csv"
D_ITEMS_PATH = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

SELECTION_MODE = "threshold"
MIN_ROW_COVERAGE = 0.05
MIN_STAY_COVERAGE = 0.10
REQUIRED_ITEMIDS = [
    227444,
    229355,
    220045,
    220277,
    220210,
    220546,
    227457,
    223762,
    225668,
    225643,
]
MEAN_ONLY = False


def detect_covariate_cols(columns: list[str]) -> list[dict[str, object]]:
    # Match either all supported summary stats or mean-only columns.
    if MEAN_ONLY:
        col_pattern = re.compile(r"^itemid_(\d+)__mean$")
    else:
        col_pattern = re.compile(r"^itemid_(\d+)__([a-z0-9_]+)$", flags=re.IGNORECASE)

    covariates: list[dict[str, object]] = []
    for column_name in columns:
        match = col_pattern.match(column_name)
        if not match:
            continue

        covariates.append(
            {
                "column_name": column_name,
                "itemid": int(match.group(1)),
                "stat": "mean" if MEAN_ONLY else match.group(2),
            }
        )

    return covariates


def load_item_labels() -> dict[int, str]:
    # Load item labels once so the output log is readable.
    label_df = pd.read_csv(
        D_ITEMS_PATH,
        usecols=["itemid", "label"],
        low_memory=False,
    ).drop_duplicates("itemid")
    label_df["itemid"] = pd.to_numeric(label_df["itemid"], errors="coerce")
    label_df = label_df.dropna(subset=["itemid"]).copy()
    label_df["itemid"] = label_df["itemid"].astype(int)
    return label_df.set_index("itemid")["label"].to_dict()


def decide_retention(row_cov: float, stay_cov: float, itemid: int, required_ids: set[int]) -> tuple[str, str]:
    # Apply the configured retention rule to one covariate.
    if SELECTION_MODE == "threshold":
        keep_col = row_cov >= MIN_ROW_COVERAGE and stay_cov >= MIN_STAY_COVERAGE
        return ("retain" if keep_col else "drop", "coverage")

    if SELECTION_MODE == "required":
        keep_col = itemid in required_ids
        return ("retain" if keep_col else "drop", "required_itemid")

    raise ValueError(f"Unsupported SELECTION_MODE: {SELECTION_MODE}")


def build_retention_log(panel: pd.DataFrame, covariates: list[dict[str, object]]) -> pd.DataFrame:
    # Summarise row coverage, stay coverage, and the keep/drop decision.
    total_rows = len(panel)
    total_stays = panel["stay_id"].nunique()
    stay_ids = panel["stay_id"]
    required_ids = set(REQUIRED_ITEMIDS)
    itemid_to_label = load_item_labels()

    rows: list[dict[str, object]] = []
    for covariate in covariates:
        column_name = str(covariate["column_name"])
        itemid = int(covariate["itemid"])
        stat = str(covariate["stat"])
        values = panel[column_name]

        # Count how often this covariate is observed across rows and stays.
        n_rows = int(values.notna().sum())
        row_cov = n_rows / total_rows if total_rows else 0.0

        has_value_by_stay = values.notna().groupby(stay_ids, sort=False).any()
        n_stays = int(has_value_by_stay.sum())
        stay_cov = n_stays / total_stays if total_stays else 0.0

        decision, reason = decide_retention(row_cov, stay_cov, itemid, required_ids)
        rows.append(
            {
                "itemid": itemid,
                "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
                "stat": stat,
                "column_name": column_name,
                "n_rows": n_rows,
                "row_coverage": round(row_cov, 4),
                "n_stays": n_stays,
                "stay_coverage": round(stay_cov, 4),
                "decision": decision,
                "selection_reason": reason,
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(
        ["decision", "row_coverage", "stay_coverage", "n_rows", "itemid", "stat"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)


def main() -> None:
    # Read the header first so covariate detection is cheap.
    all_columns = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
    covariates = detect_covariate_cols(all_columns)
    usecols = ["stay_id"] + [str(covariate["column_name"]) for covariate in covariates]

    print(f"[LOAD] {DATASET_PATH}")
    print(f"[INFO] loading stay_id + {len(covariates):,} covariate columns")

    # Load only the columns needed for coverage calculations.
    panel = pd.read_csv(DATASET_PATH, usecols=usecols, low_memory=False)
    summary_df = build_retention_log(panel, covariates)
    summary_df.to_csv(OUT_PATH, index=False)

    retained_total = int((summary_df["decision"] == "retain").sum())
    dropped_total = int((summary_df["decision"] == "drop").sum())

    print(f"[INFO] panel rows: {len(panel):,}")
    print(f"[INFO] catheterised stays: {panel['stay_id'].nunique():,}")
    print(f"[INFO] selection mode: {SELECTION_MODE}")
    print(f"[WRITE] {OUT_PATH}")
    print(f"[INFO] retained columns: {retained_total:,}")
    print(f"[INFO] dropped columns: {dropped_total:,}")


if __name__ == "__main__":
    main()
