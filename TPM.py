"""
TrainTPMLogReg.py

Reads your prepared Makic-style daily dataset CSV (e.g. output/makic_daily_dataset.csv),
does basic preprocessing, and fits a Makic-like discrete-time competing risks model
using logistic regression:

Makic-style factorisation:
  1) Binary logistic regression for P(any event next day)
  2) Multinomial logistic regression for P(event type | an event happens next day)

Where "event type" is derived from the label columns already in your dataset:
  - cauti_next_day
  - removal_next_day
  - reinsert_next_day

Notes (important):
- This script assumes each row is a catheter-day (your current dataset design).
- Your current reinsert_next_day is a proxy label on the day BEFORE removal, based on
  whether reinsertion later occurs. That is workable for a first replication.
- We split by subject_id to reduce leakage (patient-level split).
- This is a baseline replication scaffold, not a full TPM with two separate processes.

Outputs:
- Prints validation metrics
- Optionally saves fitted preprocessors + models to ./models/

Dependencies:
  pip install pandas numpy scikit-learn joblib
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

import joblib

# =========================
# CONFIG
# =========================

DATASET_PATH = Path(r"C:\Users\DavidUni\Repos\CatheterDataExtractor\output\makic_daily_dataset.csv")

# How much data to keep for validation
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42

# Where to save fitted objects
SAVE_MODELS = True
MODELS_DIR = Path(r"C:\Users\DavidUni\Repos\CatheterDataExtractor\models")

# If True, drop rows where multiple events are labelled simultaneously (rare, but can happen)
DROP_MULTILABEL_ROWS = True

# If True, exclude "terminal" catheter-day rows from training (often done to avoid edge effects)
# Terminal = last row of an episode (day_end == episode_end), which is a partial day.
EXCLUDE_TERMINAL_DAY = False


# =========================
# HELPERS
# =========================

def load_dataset(path: Path) -> pd.DataFrame:
    print(f"[LOAD] {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] rows={len(df):,} cols={len(df.columns):,}")
    return df


def coerce_datetime(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def find_feature_columns(df: pd.DataFrame) -> list[str]:
    # Makic-like daily covariates generated from chartevents aggregation
    feat_cols = [c for c in df.columns if c.startswith("itemid_")]
    return feat_cols


def build_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two target views:

    1) any_event_next_day ∈ {0,1}
       any_event = 1 if any of cauti/removal/reinsert is 1.

    2) event_type_next_day ∈ {0,1,2,3}
       0 = no event
       1 = removal
       2 = CAUTI
       3 = reinsert

    Priority rule (if labels overlap):
      CAUTI > removal > reinsert
    You can change this, but you must have a deterministic mapping.
    """
    d = df.copy()

    for col in ["cauti_next_day", "removal_next_day", "reinsert_next_day"]:
        if col not in d.columns:
            raise ValueError(f"Missing required label column: {col}")

    # Ensure integer 0/1
    for col in ["cauti_next_day", "removal_next_day", "reinsert_next_day"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

    # Optional sanity: remove rows with multiple labels = 1 (if you want strict single-event days)
    if DROP_MULTILABEL_ROWS:
        multi = (d[["cauti_next_day", "removal_next_day", "reinsert_next_day"]].sum(axis=1) > 1)
        n_multi = int(multi.sum())
        if n_multi > 0:
            print(f"[CLEAN] Dropping multilabel rows (sum(labels)>1): {n_multi:,}")
            d = d.loc[~multi].copy()

    d["any_event_next_day"] = (
        (d["cauti_next_day"] == 1) |
        (d["removal_next_day"] == 1) |
        (d["reinsert_next_day"] == 1)
    ).astype(int)

    # event_type mapping with priority
    d["event_type_next_day"] = 0
    d.loc[d["reinsert_next_day"] == 1, "event_type_next_day"] = 3
    d.loc[d["removal_next_day"] == 1, "event_type_next_day"] = 1
    d.loc[d["cauti_next_day"] == 1, "event_type_next_day"] = 2

    return d


def exclude_terminal_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Terminal day = the last row of each catheterisation interval.
    In your dataset this is typically where day_end == episode_end.
    Excluding it is optional; Makic-style implementations sometimes exclude terminal
    rows depending on how hazards are defined and how censoring is handled.
    """
    if "day_end" not in df.columns or "episode_end" not in df.columns:
        print("[WARN] Cannot exclude terminal day: missing day_end or episode_end columns.")
        return df

    d = df.copy()
    d = coerce_datetime(d, ["day_end", "episode_end"])
    mask_terminal = (d["day_end"].notna()) & (d["episode_end"].notna()) & (d["day_end"] == d["episode_end"])
    n_term = int(mask_terminal.sum())
    print(f"[CLEAN] Terminal-day rows detected: {n_term:,}")
    d = d.loc[~mask_terminal].copy()
    print(f"[CLEAN] After excluding terminal days: rows={len(d):,}")
    return d


def patient_split(df: pd.DataFrame, valid_frac=0.2, seed=42):
    """
    Patient-level split (subject_id).
    """
    if "subject_id" not in df.columns:
        raise ValueError("Dataset must contain subject_id for patient-level splitting.")

    subjects = df["subject_id"].dropna().unique()
    train_subj, valid_subj = train_test_split(subjects, test_size=valid_frac, random_state=seed)

    train_df = df[df["subject_id"].isin(train_subj)].copy()
    valid_df = df[df["subject_id"].isin(valid_subj)].copy()

    print(f"[SPLIT] train subjects={len(train_subj):,} rows={len(train_df):,}")
    print(f"[SPLIT] valid subjects={len(valid_subj):,} rows={len(valid_df):,}")
    return train_df, valid_df


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Basic numeric preprocessing:
      - median imputation
      - standard scaling
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, feature_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# =========================
# TRAIN / EVAL
# =========================

def main():
    df = load_dataset(DATASET_PATH)

    # Optional: ensure datetime columns are parsed (helps if you later do filtering)
    df = coerce_datetime(df, ["episode_start", "episode_end", "day_start", "day_end"])

    # Optional: exclude terminal day rows
    if EXCLUDE_TERMINAL_DAY:
        df = exclude_terminal_day(df)

    # Build targets
    df = build_event_labels(df)

    # Identify covariates
    feature_cols = find_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns starting with 'itemid_').")

    print(f"[FEATURES] Using {len(feature_cols):,} covariate columns (itemid_*)")

    # Basic label distribution
    print("\n[LABEL RATES]")
    print(df[["any_event_next_day", "removal_next_day", "cauti_next_day", "reinsert_next_day"]].mean(numeric_only=True))

    # Patient-level split
    train_df, valid_df = patient_split(df, valid_frac=VALIDATION_FRACTION, seed=RANDOM_SEED)

    X_train = train_df[feature_cols]
    y_any_train = train_df["any_event_next_day"].astype(int)

    X_valid = valid_df[feature_cols]
    y_any_valid = valid_df["any_event_next_day"].astype(int)

    # -------------------------
    # 1) Binary model: any event
    # -------------------------
    print("\n[MODEL 1] Binary logistic regression: P(any event next day)")

    pre = make_preprocessor(feature_cols)

    # class_weight to handle imbalance (events can be rare)
    any_event_model = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        class_weight="balanced",
        n_jobs=None,
    )

    any_event_pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", any_event_model),
    ])

    any_event_pipe.fit(X_train, y_any_train)

    # Predict probabilities for validation
    p_any_valid = any_event_pipe.predict_proba(X_valid)[:, 1]

    # Metrics
    try:
        auc = roc_auc_score(y_any_valid, p_any_valid)
    except ValueError:
        auc = np.nan
    ap = average_precision_score(y_any_valid, p_any_valid)

    print(f"[VALID] AUROC(any-event) = {auc:.4f}")
    print(f"[VALID] AUPRC(any-event) = {ap:.4f}")

    # ----------------------------------------
    # 2) Multinomial model: event type | event
    # ----------------------------------------
    print("\n[MODEL 2] Multinomial logistic regression: P(type | event happens next day)")

    # Only train this on rows where an event happens
    train_events = train_df[train_df["any_event_next_day"] == 1].copy()
    valid_events = valid_df[valid_df["any_event_next_day"] == 1].copy()

    # event_type_next_day values should be in {1,2,3} here
    y_type_train = train_events["event_type_next_day"].astype(int)
    y_type_valid = valid_events["event_type_next_day"].astype(int)

    # If there are too few event rows, warn early
    print(f"[INFO] train event-rows={len(train_events):,} valid event-rows={len(valid_events):,}")
    if len(train_events) < 1000:
        print("[WARN] Very few event rows in training; multinomial estimates may be unstable.")

    type_pre = make_preprocessor(feature_cols)

    type_model = LogisticRegression(
        solver="lbfgs",
        max_iter=400,
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=None,
    )

    type_pipe = Pipeline(steps=[
        ("pre", type_pre),
        ("clf", type_model),
    ])

    # Fit only if we have at least 2 classes present
    unique_types = sorted(y_type_train.unique().tolist())
    if len(unique_types) < 2:
        print(f"[SKIP] Multinomial model skipped: only one event class present in training: {unique_types}")
        type_pipe = None
    else:
        type_pipe.fit(train_events[feature_cols], y_type_train)

        y_type_pred = type_pipe.predict(valid_events[feature_cols])

        print("\n[VALID] Event-type classification report (only among event-days)")
        print(classification_report(y_type_valid, y_type_pred, digits=4))

        print("[VALID] Confusion matrix (rows=true, cols=pred)")
        print(confusion_matrix(y_type_valid, y_type_pred))

    # -------------------------
    # Save fitted objects
    # -------------------------
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        any_path = MODELS_DIR / "logreg_any_event_pipeline.joblib"
        joblib.dump(any_event_pipe, any_path)
        print(f"\n[SAVE] any-event pipeline -> {any_path}")

        if type_pipe is not None:
            type_path = MODELS_DIR / "logreg_event_type_pipeline.joblib"
            joblib.dump(type_pipe, type_path)
            print(f"[SAVE] event-type pipeline -> {type_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
