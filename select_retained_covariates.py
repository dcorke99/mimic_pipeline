from pathlib import Path
import pandas as pd
import re

# Configuration

DATA_DIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")

# Input panel and output log paths
DATASET_PATH = DATA_DIR / "master_panel.csv"
OUT_PATH = DATA_DIR / "covariate_retention_log.csv"

# Path to MIMIC-IV item labels
D_ITEMS_PATH = Path(
    r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv"
)

# Select covariates by coverage "threshold" or by a fixed "required" list
SELECTION_MODE = "threshold"

# Minimum row and stay coverage for threshold-based selection

MIN_ROW_COVERAGE = 0.05
MIN_STAY_COVERAGE = 0.10

# Itemids to retain when using required-itemid selection
REQUIRED_ITEMIDS = [
    227444,
    229355,
    220045,
    220277,
    220210,
    220546,
    227457,
    223761,
    225668,
    225643,
]

# Limit the summary to __mean columns only
MEAN_ONLY = True

# Load the column names first so covariate columns can be selected cheaply.
cols = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()

# Identify covariate columns

# Choose the naming pattern that identifies eligible covariate columns.
if MEAN_ONLY:
    pat = re.compile(r"^itemid_(\d+)__mean$")
else:
    pat = re.compile(r"^itemid_(\d+)__(mean|min|max)$")

cov_cols = []
itemids = []
stats = []

# Parse each matching covariate column into its itemid and summary statistic.
for c in cols:
    m = pat.match(c)
    if not m:
        continue

    itemid = int(m.group(1))
    stat = m.group(2) if not MEAN_ONLY else "mean"

    cov_cols.append(c)
    itemids.append(itemid)
    stats.append(stat)

usecols = ["stay_id"] + cov_cols

print(f"[LOAD] {DATASET_PATH}")
print(f"[INFO] loading stay_id + {len(cov_cols):,} covariate columns")

df = pd.read_csv(DATASET_PATH, usecols=usecols, low_memory=False)

# Load item labels

d_items = pd.read_csv(
    D_ITEMS_PATH,
    usecols=["itemid", "label"],
    low_memory=False
).drop_duplicates("itemid")

d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
d_items = d_items.dropna(subset=["itemid"])
d_items["itemid"] = d_items["itemid"].astype(int)

itemid_to_label = d_items.set_index("itemid")["label"].to_dict()

# Compute row and stay coverage for each covariate

total_rows = len(df)
total_stays = df["stay_id"].nunique()

print(f"[INFO] panel rows: {total_rows:,}")
print(f"[INFO] catheterised stays: {total_stays:,}")
print(f"[INFO] selection mode: {SELECTION_MODE}")

rows = []
stay_id_series = df["stay_id"]

# Convert the required itemid list to a set for fast membership checks.
required_set = set(REQUIRED_ITEMIDS)

# Summarize coverage and retention decisions for each candidate covariate.
for col, itemid, stat in zip(cov_cols, itemids, stats):
    s = df[col]

    # Count rows where the covariate is observed
    n_rows = int(s.notna().sum())
    row_cov = n_rows / total_rows if total_rows else 0.0

    # Count stays with at least one observed value
    has_any_by_stay = s.notna().groupby(stay_id_series, sort=False).any()
    n_stays = int(has_any_by_stay.sum())
    stay_cov = n_stays / total_stays if total_stays else 0.0

    # Apply the configured retention rule
    if SELECTION_MODE == "threshold":
        decision = (
            "retain"
            if (row_cov >= MIN_ROW_COVERAGE and stay_cov >= MIN_STAY_COVERAGE)
            else "drop"
        )
        reason = "coverage"
    else:
        decision = "retain" if itemid in required_set else "drop"
        reason = "required_itemid"

    rows.append({
        "itemid": itemid,
        "label": itemid_to_label.get(itemid, "UNKNOWN ITEMID"),
        "stat": stat,
        "column_name": col,
        "n_rows": n_rows,
        "row_coverage": round(row_cov, 4),
        "n_stays": n_stays,
        "stay_coverage": round(stay_cov, 4),
        "decision": decision,
        "selection_reason": reason,
    })

summary = pd.DataFrame(rows)

# Sort and write the retention summary

summary = summary.sort_values(
    ["decision", "row_coverage", "stay_coverage", "n_rows", "itemid", "stat"],
    ascending=[True, False, False, False, True, True]
).reset_index(drop=True)

summary.to_csv(OUT_PATH, index=False)

print(f"[WRITE] {OUT_PATH}")
print(f"[INFO] retained columns: {(summary['decision'] == 'retain').sum():,}")
print(f"[INFO] dropped columns:  {(summary['decision'] == 'drop').sum():,}")
