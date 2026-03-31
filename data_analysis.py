import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

DataFile = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\data\catheter_two_process_daily_filtered.csv"
ResultsDir = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\data\DataAnalysis"

# Optional: for mapping itemids -> labels in outputs
D_ITEMS_PATH = r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv"

dp = 2

# Which stats to analyse for each covariate
KEEP_STATS = {"mean"}  # {"mean"} or {"mean","min","max"}

# ==========================================================
# Helpers
# ==========================================================

def detect_covariate_cols(columns, keep_stats):
    """
    Detect columns like itemid_<ID>__mean/min/max and return:
      - cov_cols: list of matching column names
      - cov_meta: DataFrame with columns [col, itemid, stat]
    """
    pat = re.compile(r"^itemid_(\d+)__(mean|min|max)$", flags=re.IGNORECASE)
    cov_cols = []
    meta = []
    for c in columns:
        m = pat.match(str(c))
        if not m:
            continue
        itemid = int(m.group(1))
        stat = m.group(2).lower()
        if stat in keep_stats:
            cov_cols.append(c)
            meta.append({"col": c, "itemid": itemid, "stat": stat})
    cov_meta = pd.DataFrame(meta)
    return cov_cols, cov_meta


def load_item_labels(d_items_path):
    if not d_items_path or not os.path.exists(d_items_path):
        return {}
    d_items = pd.read_csv(d_items_path, usecols=["itemid", "label"], low_memory=False).drop_duplicates("itemid")
    d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
    d_items = d_items.dropna(subset=["itemid"])
    d_items["itemid"] = d_items["itemid"].astype(int)
    return d_items.set_index("itemid")["label"].to_dict()


def cliffs_delta(x1, x0):
    """
    Cliff's delta using rank-based U (handles ties).
    """
    xy = np.concatenate([x1, x0])
    ranks = stats.rankdata(xy)
    rx = ranks[: x1.size].sum()
    u = rx - x1.size * (x1.size + 1) / 2
    delta = (2 * u) / (x1.size * x0.size) - 1
    return float(delta)


def make_pretty_name(label, stat):
    """
    Show actual covariate name and include stat.
    Example:
      Heart Rate [mean]
      WBC [min]
    """
    return f"{label} [{stat}]"


def safe_plot_heatmap(df_corr, title, filename, results_dir):
    """
    Plot heatmap only if matrix is non-empty and not all NaN.
    """
    if df_corr.empty or df_corr.isnull().all().all():
        print(f"[SKIP] {title} is empty or all NaN. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    arr = df_corr.fillna(0).to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="seismic", vmin=-1, vmax=1, interpolation="nearest")

    labels = list(df_corr.columns)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax)

    save_path = os.path.join(results_dir, filename)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =========================
# Load data
# =========================

os.makedirs(ResultsDir, exist_ok=True)

# Load + defragment early
df = pd.read_csv(DataFile, low_memory=False).copy()

# Detect covariates (itemid_*__stat)
cov_cols, cov_meta = detect_covariate_cols(df.columns.tolist(), KEEP_STATS)
if len(cov_cols) == 0:
    raise ValueError(
        f"No covariate columns found matching itemid_<ID>__<stat> with stats {sorted(KEEP_STATS)} "
        f"in {DataFile}"
    )

# Make covariates numeric (vectorised; avoids per-column inserts)
df[cov_cols] = df[cov_cols].apply(pd.to_numeric, errors="coerce")

# Add covariate count columns (single assignment; avoids fragmentation)
cov_count = df[cov_cols].notna().sum(axis=1).astype("int32")
df = df.assign(
    covariate_count=cov_count,
    any_covariate=(cov_count > 0).astype("int8"),
)

# Outcomes numeric (in case saved as strings)
for n in ["removal_next_day", "reinsertion_next_day", "cauti_next_day"]:
    if n in df.columns:
        df[n] = pd.to_numeric(df[n], errors="coerce")
    else:
        raise ValueError(f"Expected outcome column '{n}' not found in dataset")

# Optional: label map for nicer reporting
itemid_to_label = load_item_labels(D_ITEMS_PATH)
if not cov_meta.empty:
    cov_meta["label"] = cov_meta["itemid"].map(itemid_to_label).fillna("UNKNOWN ITEMID")
    cov_meta["pretty_name"] = cov_meta.apply(
        lambda r: make_pretty_name(r["label"], r["stat"]),
        axis=1
    )
else:
    cov_meta["label"] = cov_meta["col"]
    cov_meta["pretty_name"] = cov_meta["col"]

# Numeric covariate columns in dataset
covariates = cov_cols

# Mapping from raw column name to readable covariate name
covariate_name_map = dict(zip(cov_meta["col"], cov_meta["pretty_name"]))

# Save covariate dictionary
cov_meta.sort_values(["label", "itemid", "stat"]).to_csv(
    os.path.join(ResultsDir, "00_covariate_dictionary.csv"), index=False
)

# =====================
# 01) Cohort counts
# =====================

rows = []
rows.append({
    "process": "overall",
    "rows": int(len(df)),
    "unique_subject_id": int(df["subject_id"].nunique()),
    "unique_hadm_id": int(df["hadm_id"].nunique()),
    "unique_stay_id": int(df["stay_id"].nunique()),
})

for proc, g in df.groupby("process", dropna=False):
    rows.append({
        "process": str(proc),
        "rows": int(len(g)),
        "unique_subject_id": int(g["subject_id"].nunique()),
        "unique_hadm_id": int(g["hadm_id"].nunique()),
        "unique_stay_id": int(g["stay_id"].nunique()),
    })

out_01 = pd.DataFrame(rows)
out_01.to_csv(os.path.join(ResultsDir, "01_cohort_counts.csv"), index=False)

# ==========================================================
# 02) Covariate counts
# ==========================================================

rows = []

s = df["covariate_count"].astype(float)
rows.append({
    "process": "overall",
    "rows": int(len(df)),
    "mean_covariate_count": float(s.mean()),
    "median_covariate_count": float(s.median()),
    "min_covariate_count": float(s.min()),
    "max_covariate_count": float(s.max()),
})

for proc, g in df.groupby("process", dropna=False):
    s = g["covariate_count"].astype(float)
    rows.append({
        "process": str(proc),
        "rows": int(len(g)),
        "mean_covariate_count": float(s.mean()),
        "median_covariate_count": float(s.median()),
        "min_covariate_count": float(s.min()),
        "max_covariate_count": float(s.max()),
    })

out_02 = pd.DataFrame(rows)
numeric_cols = out_02.select_dtypes(include=[np.number]).columns
out_02[numeric_cols] = out_02[numeric_cols].round(dp)
out_02.to_csv(os.path.join(ResultsDir, "02_covariate_count.csv"), index=False)

# ==========================================================
# 03) Outcome counts and occurrence rates
# ==========================================================

rows = []

g = df
row = {"process": "overall", "rows": int(len(g))}
for outcome in ["removal_next_day", "reinsertion_next_day", "cauti_next_day"]:
    y = g[outcome]
    row[f"occurrence_rate_{outcome}"] = float(y.mean())
    row[f"count_{outcome}_1"] = int((y == 1).sum())
    row[f"count_{outcome}_0"] = int((y == 0).sum())
rows.append(row)

for proc, g in df.groupby("process", dropna=False):
    row = {"process": str(proc), "rows": int(len(g))}
    for outcome in ["removal_next_day", "reinsertion_next_day", "cauti_next_day"]:
        y = g[outcome]
        row[f"occurrence_rate_{outcome}"] = float(y.mean())
        row[f"count_{outcome}_1"] = int((y == 1).sum())
        row[f"count_{outcome}_0"] = int((y == 0).sum())
    rows.append(row)

out_03 = pd.DataFrame(rows)
numeric_cols = out_03.select_dtypes(include=[np.number]).columns
out_03[numeric_cols] = out_03[numeric_cols].round(dp)
out_03.to_csv(os.path.join(ResultsDir, "03_outcome_rates.csv"), index=False)

# ==========================================================
# 04) Descriptive statistics (overall)
# ==========================================================

rows = []
for col in covariates:
    s = df[col].dropna()
    n = int(s.shape[0])
    pretty = covariate_name_map.get(col, col)

    rows.append({
        "covariate": pretty,
        "col": col,
        "N": n,
        "mean": float(s.mean()) if n else np.nan,
        "sd": float(s.std(ddof=1)) if n > 1 else 0.0,
        "min": float(s.min()) if n else np.nan,
        "q1": float(s.quantile(0.25)) if n else np.nan,
        "median": float(s.median()) if n else np.nan,
        "q3": float(s.quantile(0.75)) if n else np.nan,
        "max": float(s.max()) if n else np.nan,
    })

out_04 = pd.DataFrame(rows)
numeric_cols = out_04.select_dtypes(include=[np.number]).columns
out_04[numeric_cols] = out_04[numeric_cols].round(dp)
out_04.to_csv(os.path.join(ResultsDir, "04_describe_overall.csv"), index=False)

# ==========================================================
# 05) Descriptive statistics by process
# ==========================================================

rows = []
for proc, g in df.groupby("process", dropna=False):
    for col in covariates:
        s = g[col].dropna()
        n = int(s.shape[0])
        pretty = covariate_name_map.get(col, col)

        rows.append({
            "process": str(proc),
            "covariate": pretty,
            "col": col,
            "N": n,
            "mean": float(s.mean()) if n else np.nan,
            "sd": float(s.std(ddof=1)) if n > 1 else 0.0,
            "min": float(s.min()) if n else np.nan,
            "q1": float(s.quantile(0.25)) if n else np.nan,
            "median": float(s.median()) if n else np.nan,
            "q3": float(s.quantile(0.75)) if n else np.nan,
            "max": float(s.max()) if n else np.nan,
        })

out_05 = pd.DataFrame(rows)
numeric_cols = out_05.select_dtypes(include=[np.number]).columns
out_05[numeric_cols] = out_05[numeric_cols].round(dp)
out_05.to_csv(os.path.join(ResultsDir, "05_describe_by_process.csv"), index=False)

# ==========================================================
# 06) Describe catheter_in by removal_next_day
# ==========================================================

tmp = df[df["process"] == "catheter_in"].copy()

rows = []
for outcome_value, g in tmp.groupby("removal_next_day", dropna=False):
    for col in covariates:
        s = g[col].dropna()
        n = int(s.shape[0])
        pretty = covariate_name_map.get(col, col)

        rows.append({
            "removal_next_day": float(outcome_value) if pd.notna(outcome_value) else np.nan,
            "covariate": pretty,
            "col": col,
            "N": n,
            "mean": float(s.mean()) if n else np.nan,
            "sd": float(s.std(ddof=1)) if n > 1 else 0.0,
            "min": float(s.min()) if n else np.nan,
            "q1": float(s.quantile(0.25)) if n else np.nan,
            "median": float(s.median()) if n else np.nan,
            "q3": float(s.quantile(0.75)) if n else np.nan,
            "max": float(s.max()) if n else np.nan,
        })

out_06 = pd.DataFrame(rows)
numeric_cols = out_06.select_dtypes(include=[np.number]).columns
out_06[numeric_cols] = out_06[numeric_cols].round(dp)
out_06.to_csv(os.path.join(ResultsDir, "06_describe_catheter_in_by_removal_next_day.csv"), index=False)

# ==========================================================
# 07) Describe catheter_out by reinsertion_next_day
# ==========================================================

tmp = df[df["process"] == "catheter_out"].copy()

rows = []
for outcome_value, g in tmp.groupby("reinsertion_next_day", dropna=False):
    for col in covariates:
        s = g[col].dropna()
        n = int(s.shape[0])
        pretty = covariate_name_map.get(col, col)

        rows.append({
            "reinsertion_next_day": float(outcome_value) if pd.notna(outcome_value) else np.nan,
            "covariate": pretty,
            "col": col,
            "N": n,
            "mean": float(s.mean()) if n else np.nan,
            "sd": float(s.std(ddof=1)) if n > 1 else 0.0,
            "min": float(s.min()) if n else np.nan,
            "q1": float(s.quantile(0.25)) if n else np.nan,
            "median": float(s.median()) if n else np.nan,
            "q3": float(s.quantile(0.75)) if n else np.nan,
            "max": float(s.max()) if n else np.nan,
        })

out_07 = pd.DataFrame(rows)
numeric_cols = out_07.select_dtypes(include=[np.number]).columns
out_07[numeric_cols] = out_07[numeric_cols].round(dp)
out_07.to_csv(os.path.join(ResultsDir, "07_describe_catheter_out_by_reinsertion_next_day.csv"), index=False)

# ==========================================================
# 08) GROUP COMPARISON TESTS
# ==========================================================

MIN_N_PER_GROUP = 20
VAR_EPS = 1e-12
rows = []

def cov_name(col: str) -> str:
    return covariate_name_map.get(col, col)

def welch_pvalue(x1: np.ndarray, x0: np.ndarray) -> float:
    v1 = float(np.nanvar(x1, ddof=1)) if x1.size > 1 else 0.0
    v0 = float(np.nanvar(x0, ddof=1)) if x0.size > 1 else 0.0
    if v1 < VAR_EPS and v0 < VAR_EPS:
        return np.nan
    return float(stats.ttest_ind(x1, x0, equal_var=False).pvalue)

def mannwhitney_pvalue(x1: np.ndarray, x0: np.ndarray) -> float:
    if x1.size == 0 or x0.size == 0:
        return np.nan
    if np.all(x1 == x1[0]) and np.all(x0 == x0[0]) and x1[0] == x0[0]:
        return np.nan
    return float(stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue)

# a) catheter_in vs removal_next_day
d = df[df["process"] == "catheter_in"].copy()
d = d[d["removal_next_day"].isin([0, 1])]

for col in covariates:
    x1 = d.loc[d["removal_next_day"] == 1, col].dropna().to_numpy()
    x0 = d.loc[d["removal_next_day"] == 0, col].dropna().to_numpy()
    n1, n0 = int(x1.size), int(x0.size)

    if n1 < MIN_N_PER_GROUP or n0 < MIN_N_PER_GROUP:
        rows.append({
            "outcome": "removal_next_day",
            "process": "catheter_in",
            "covariate": col,
            "covariate_name": cov_name(col),
            "n_outcome_1": n1,
            "n_outcome_0": n0,
            "welch_t_p": np.nan,
            "mannwhitney_p": np.nan,
            "cliffs_delta": np.nan,
            "low_variance": np.nan,
        })
        continue

    p_welch = welch_pvalue(x1, x0)
    p_mw = mannwhitney_pvalue(x1, x0)
    delta = cliffs_delta(x1, x0)

    rows.append({
        "outcome": "removal_next_day",
        "process": "catheter_in",
        "covariate": col,
        "covariate_name": cov_name(col),
        "n_outcome_1": n1,
        "n_outcome_0": n0,
        "welch_t_p": p_welch,
        "mannwhitney_p": p_mw,
        "cliffs_delta": float(delta),
        "low_variance": int(pd.isna(p_welch)),
    })

# b) catheter_out vs reinsertion_next_day
d = df[df["process"] == "catheter_out"].copy()
d = d[d["reinsertion_next_day"].isin([0, 1])]

for col in covariates:
    x1 = d.loc[d["reinsertion_next_day"] == 1, col].dropna().to_numpy()
    x0 = d.loc[d["reinsertion_next_day"] == 0, col].dropna().to_numpy()
    n1, n0 = int(x1.size), int(x0.size)

    if n1 < MIN_N_PER_GROUP or n0 < MIN_N_PER_GROUP:
        rows.append({
            "outcome": "reinsertion_next_day",
            "process": "catheter_out",
            "covariate": col,
            "covariate_name": cov_name(col),
            "n_outcome_1": n1,
            "n_outcome_0": n0,
            "welch_t_p": np.nan,
            "mannwhitney_p": np.nan,
            "cliffs_delta": np.nan,
            "low_variance": np.nan,
        })
        continue

    p_welch = welch_pvalue(x1, x0)
    p_mw = mannwhitney_pvalue(x1, x0)
    delta = cliffs_delta(x1, x0)

    rows.append({
        "outcome": "reinsertion_next_day",
        "process": "catheter_out",
        "covariate": col,
        "covariate_name": cov_name(col),
        "n_outcome_1": n1,
        "n_outcome_0": n0,
        "welch_t_p": p_welch,
        "mannwhitney_p": p_mw,
        "cliffs_delta": float(delta),
        "low_variance": int(pd.isna(p_welch)),
    })

# c) overall CAUTI
d = df[df["cauti_next_day"].isin([0, 1])].copy()

for col in covariates:
    x1 = d.loc[d["cauti_next_day"] == 1, col].dropna().to_numpy()
    x0 = d.loc[d["cauti_next_day"] == 0, col].dropna().to_numpy()
    n1, n0 = int(x1.size), int(x0.size)

    if n1 < MIN_N_PER_GROUP or n0 < MIN_N_PER_GROUP:
        rows.append({
            "outcome": "cauti_next_day",
            "process": "overall",
            "covariate": col,
            "covariate_name": cov_name(col),
            "n_outcome_1": n1,
            "n_outcome_0": n0,
            "welch_t_p": np.nan,
            "mannwhitney_p": np.nan,
            "cliffs_delta": np.nan,
            "low_variance": np.nan,
        })
        continue

    p_welch = welch_pvalue(x1, x0)
    p_mw = mannwhitney_pvalue(x1, x0)
    delta = cliffs_delta(x1, x0)

    rows.append({
        "outcome": "cauti_next_day",
        "process": "overall",
        "covariate": col,
        "covariate_name": cov_name(col),
        "n_outcome_1": n1,
        "n_outcome_0": n0,
        "welch_t_p": p_welch,
        "mannwhitney_p": p_mw,
        "cliffs_delta": float(delta),
        "low_variance": int(pd.isna(p_welch)),
    })

out_08 = pd.DataFrame(rows)
numeric_cols = out_08.select_dtypes(include=[np.number]).columns
out_08[numeric_cols] = out_08[numeric_cols].round(dp)
out_08.to_csv(os.path.join(ResultsDir, "08_outcome_comparison_tests.csv"), index=False)

# ==========================================================
# 09) Spearman correlation (overall)
# ==========================================================

out_09 = df[covariates].corr(method="spearman").round(dp)
out_09 = out_09.rename(index=covariate_name_map, columns=covariate_name_map)
out_09.to_csv(os.path.join(ResultsDir, "09_spearman_overall.csv"))

# ==========================================================
# 10) Spearman correlation (by process)
# ==========================================================

for proc, g in df.groupby("process", dropna=False):
    corr_p = g[covariates].corr(method="spearman").round(dp)
    corr_p = corr_p.rename(index=covariate_name_map, columns=covariate_name_map)
    corr_p.to_csv(os.path.join(ResultsDir, f"10_spearman_{proc}.csv"))

# ==========================================================
# 11) Plots
# ==========================================================

# 0. Histogram: distribution of observed covariate count per patient-day
hist_path = os.path.join(ResultsDir, "plot_covariate_count_hist.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df["covariate_count"].astype(float).to_numpy(), bins=30)
ax.set_title("Distribution of observed covariate count per patient-day")
ax.set_xlabel("covariate_count")
ax.set_ylabel("patient-days")
fig.savefig(hist_path, dpi=120)
plt.close(fig)

print(f"[OK] Saved: {hist_path}")

# 1. Overall Heatmap
safe_plot_heatmap(out_09, "Spearman correlation (overall)", "spearman_heatmap_overall.png", ResultsDir)

# 2. Heatmaps by process
for proc, g in df.groupby("process", dropna=False):
    proc_str = str(proc) if pd.notna(proc) else "missing_process"
    corr_p = g[covariates].corr(method="spearman").round(dp)
    corr_p = corr_p.rename(index=covariate_name_map, columns=covariate_name_map)

    safe_plot_heatmap(
        corr_p,
        f"Spearman correlation ({proc_str})",
        f"spearman_heatmap_{proc_str}.png",
        ResultsDir
    )

print(f"[DONE] Outputs saved to: {ResultsDir}")
print("[INFO] Covariate columns analysed:")
print(f"[INFO] N covariates: {len(covariates)} (stats kept: {sorted(KEEP_STATS)})")
print("[INFO] Covariate dictionary saved as 00_covariate_dictionary.csv")