import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

DataFile = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\data\catheter_two_process_daily_makic_filtered.csv"
ResultsDir = r"C:\Users\DavidUni\Repos\CatheterDataExtractor\data\DataAnalysis_2"

os.makedirs(ResultsDir, exist_ok=True)

covariates = [
    ("Heart Rate", "heart_rate"),
    ("Respiratory Rate", "resp_rate"),
    ("O2 saturation pulseoxymetry", "spo2"),
    ("Arterial Blood Pressure mean", "abp_mean"),
    ("Non Invasive Blood Pressure mean", "nibp_mean"),
    ("Temperature Fahrenheit", "temp_f"),
    ("Temperature Celsius", "temp_c"),
    ("Richmond-RAS Scale", "rass"),
    ("GCS - Eye Opening", "gcs_eye"),
    ("GCS - Verbal Response", "gcs_verbal"),
    ("GCS - Motor Response", "gcs_motor"),
    ("Activity / Mobility (JH-HLM)", "mobility"),
]

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DataFile, low_memory=False)

for col, _ in covariates:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# HISTOGRAMS
# ==========================================================

for col, fname in covariates:
    x = df[col].dropna().to_numpy()
    if x.size == 0:
        continue

    plt.figure()
    plt.hist(x, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("patient-days")
    plt.tight_layout()
    plt.savefig(os.path.join(ResultsDir, f"hist_{fname}.png"), dpi=200)
    plt.close()

# ==========================================================
# BOXPLOTS
# ==========================================================

for col, fname in covariates:
    x = df[col].dropna().to_numpy()
    if x.size == 0:
        continue

    plt.figure()
    plt.boxplot(x, vert=True, showfliers=True)
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(ResultsDir, f"box_{fname}.png"), dpi=200)
    plt.close()

print(f"[DONE] Distribution and box plots saved to:\n{ResultsDir}")
