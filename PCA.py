"""
PCA pipeline for Makic-style daily dataset

What this script does:
1) Loads makic_tpm_daily_dataset.csv
2) Selects numeric covariate columns (excludes IDs, timestamps, process, and outcome flags)
3) Removes outliers by clipping each covariate to percentile bounds (default: 1st–99th)
4) Imputes missing values (median per covariate)
5) Standardises (z-score) all covariates
6) Runs PCA (default: keep enough components to explain 95% variance, capped)
7) Writes outputs:
   - 01_covariate_columns_used.csv
   - 02_explained_variance.csv
   - 03_pca_loadings.csv
   - 04_pca_scores.csv  (PC scores per row, with identifiers/outcomes kept)
   - 05_pc_scatter_pc1_pc2.png (optional quick diagnostic plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# CONFIG
# =========================

INPUT_CSV = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\output\makic_tpm_daily_dataset.csv"
OUTPUT_DIR = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\output\pca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Outlier handling: clip each covariate to these percentiles
CLIP_LO = 0.01   # 1st percentile
CLIP_HI = 0.99   # 99th percentile

# PCA settings
VARIANCE_TO_EXPLAIN = 0.95     # keep enough PCs to explain this proportion of variance
MAX_COMPONENTS = 30            # safety cap (useful if you have many covariates)

# Plot
SAVE_PLOT = True


# =========================
# LOAD
# =========================

df = pd.read_csv(INPUT_CSV, low_memory=False)

# Columns we do NOT want in PCA (non-covariates)
EXCLUDE_COLS = {
    "subject_id", "hadm_id", "stay_id",
    "process",
    "inserted", "removed",
    "day_index", "day_start", "day_end",
    "removal_next_day", "reinsertion_next_day", "cauti_next_day",
}

# Keep identifiers/outcomes to join back onto PC scores later
KEEP_META_COLS = [c for c in [
    "subject_id", "hadm_id", "stay_id", "process",
    "inserted", "removed", "day_index", "day_start", "day_end",
    "removal_next_day", "reinsertion_next_day", "cauti_next_day"
] if c in df.columns]

meta = df[KEEP_META_COLS].copy()

# =========================
# SELECT COVARIATES
# =========================

# Candidate covariates = numeric columns not in EXCLUDE_COLS
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
covariate_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

if not covariate_cols:
    raise RuntimeError(
        "No numeric covariate columns were found for PCA. "
        "Check the input file and that covariates are numeric."
    )

# Save which covariates were used
pd.DataFrame({"covariate": covariate_cols}).to_csv(
    os.path.join(OUTPUT_DIR, "01_covariate_columns_used.csv"),
    index=False
)

X = df[covariate_cols].copy()

# Ensure numeric (in case any numeric-looking columns were read as objects)
for c in covariate_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# =========================
# OUTLIER REMOVAL (CLIPPING)
# =========================
# We clip rather than dropping rows, so we do not destroy the patient-day structure.

lo = X.quantile(CLIP_LO, numeric_only=True)
hi = X.quantile(CLIP_HI, numeric_only=True)

# Align indices and clip per-column
X_clipped = X.clip(lower=lo, upper=hi, axis=1)

# =========================
# IMPUTE + STANDARDISE
# =========================

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler(with_mean=True, with_std=True)

X_imp = imputer.fit_transform(X_clipped)      # numpy array
X_std = scaler.fit_transform(X_imp)           # numpy array

# =========================
# PCA
# =========================

# First fit a full PCA to determine how many components needed for variance threshold
pca_full = PCA(n_components=min(len(covariate_cols), MAX_COMPONENTS), random_state=0)
pca_full.fit(X_std)

explained = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained)

# Determine number of components to reach VARIANCE_TO_EXPLAIN
k = int(np.searchsorted(cum_explained, VARIANCE_TO_EXPLAIN) + 1)
k = min(k, pca_full.n_components_)

# Refit PCA with exactly k components (cleaner outputs)
pca = PCA(n_components=k, random_state=0)
scores = pca.fit_transform(X_std)

# =========================
# OUTPUTS
# =========================

# 02) Explained variance table
out_ev = pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(k)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
})
out_ev.to_csv(os.path.join(OUTPUT_DIR, "02_explained_variance.csv"), index=False)

# 03) Loadings (how each covariate contributes to each PC)
# Rows = covariates; Columns = PCs
loadings = pd.DataFrame(
    pca.components_.T,
    index=covariate_cols,
    columns=[f"PC{i+1}" for i in range(k)]
)
loadings.to_csv(os.path.join(OUTPUT_DIR, "03_pca_loadings.csv"))

# 04) PC scores per patient-day (plus meta columns so you can relate to event flags later)
score_cols = [f"PC{i+1}" for i in range(k)]
out_scores = pd.DataFrame(scores, columns=score_cols)
out_scores = pd.concat([meta.reset_index(drop=True), out_scores], axis=1)
out_scores.to_csv(os.path.join(OUTPUT_DIR, "04_pca_scores.csv"), index=False)

# 05) Quick diagnostic plot (PC1 vs PC2), coloured by process if available
if SAVE_PLOT and k >= 2:
    plt.figure()
    x = out_scores["PC1"].to_numpy()
    y = out_scores["PC2"].to_numpy()

    if "process" in out_scores.columns:
        # Simple colouring by process without fancy palettes
        procs = out_scores["process"].astype(str).fillna("NA")
        for proc in sorted(procs.unique()):
            m = procs == proc
            plt.scatter(x[m], y[m], s=8, alpha=0.4, label=proc)
        plt.legend(markerscale=2)
    else:
        plt.scatter(x, y, s=8, alpha=0.4)

    plt.title("PCA: PC1 vs PC2 (standardised covariates)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_pc_scatter_pc1_pc2.png"), dpi=200)
    plt.close()

print(f"[DONE] PCA complete.")
print(f"Input:  {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}")
print(f"Covariates used: {len(covariate_cols)}")
print(f"Components kept: {k} (explains {out_ev['cumulative_explained_variance_ratio'].iloc[-1]:.3f} variance)")
