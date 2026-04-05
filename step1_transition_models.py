"""
01_step1_transition_models.py
Step 1 — Transition modelling using the original temporal panel.

This version keeps the original panel builder unchanged and fixes Step 1 by:

1. Fitting a separate removal model on IN-state rows:
   P(remove_today = 1 | H_t, state=in)

2. Fitting a CAUTI transition model on the CAUTI risk set:
   - all IN-state rows
   - OUT-state rows with days_in_state <= 2
   This encodes the rule that CAUTI may still occur during the two days following removal.

3. Fitting a reinsertion model on OUT-state rows:
   P(reinsertion_today = 1 | H_t, state=out)

Key modelling choice
--------------------
We do NOT use removed_today as a predictor inside the CAUTI model.
Instead, "remove today" changes CAUTI risk by moving the patient to a
counterfactual OUT-state day-1 row.

This avoids mixing same-interval action and outcome in the CAUTI model and
keeps interval_hours out of the predictors.

Scored outputs
--------------
- p_remove_obs
- p_cauti_if_keep
- p_cauti_if_remove
- p_cauti_if_out
- p_reins_if_remove
- p_reins_if_out
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier


# Global configuration
SEED = 42
MODEL_TYPE = "xgb"  # "rf" or "xgb"

# Input / output locations
INDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step1")
MODEL_DIR = OUTDIR

INFILE = INDIR / "filtered_panel.csv"
D_ITEMS_FILE = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

# Core column names used throughout the script
ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
DAYS_COL = "days_in_state"
SPLIT_COL = "split"

# Targets / actions
ACTION_COL = "removed_today"
Y_CAUTI = "cauti_today"
Y_REINS = "reinsertion_today"

# Episode-ending information used when defining fit/evaluation sets
LAST_DAY_COL = "is_last_day_of_episode"
END_REASON_COL = "episode_end_reason"

# Keys that define a catheter episode in the panel
EPISODE_KEYS = ["stay_id", "inserted"]

# Number of OUT-state days after removal during which CAUTI is still considered possible
POST_REMOVE_RISK_DAYS = 2

TOP_FEATURES_TO_SAVE = 15

# SHAP output controls
SAVE_SHAP = True
SHAP_SAMPLE_N = 2000

# Evaluation settings for saved diagnostics
CALIBRATION_BINS = 10
MIN_ROWS_BY_DAY = 30
MIN_EVENTS_BY_DAY = 5


# Return the base structured feature columns used by the models.
def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if c.startswith("itemid_") or c.startswith("sex_") or c.startswith("ethnicity_")
    ]
    cols.append("age")
    return cols


# Fit the chosen binary classifier pipeline after imputation and missingness indicators.
def _fit_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if MODEL_TYPE == "rf":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=SEED,
            ))
        ])
    elif MODEL_TYPE == "xgb":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=SEED,
                n_jobs=1,
            ))
        ])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    pipe.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=int))
    return pipe


# Predict class-1 probabilities from a fitted sklearn pipeline.
def _predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X.to_numpy(dtype=float))[:, 1]


# Normalise split labels in-place so train/test comparisons are reliable.
def _validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()


# Load item labels from d_items so printed feature names are readable.
def _load_item_labels() -> dict[int, str]:
    d_items = pd.read_csv(
        D_ITEMS_FILE,
        usecols=["itemid", "label"],
        low_memory=False,
    ).drop_duplicates("itemid")
    d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")
    d_items = d_items.dropna(subset=["itemid"])
    d_items["itemid"] = d_items["itemid"].astype("int64")
    return {
        itemid: label
        for itemid, label in zip(
            d_items["itemid"].to_list(),
            d_items["label"].astype(str).to_list(),
        )
    }


# Convert raw itemid feature names into readable labels plus statistic names.
def _format_feature_name(feature_name: str, item_labels: dict[int, str]) -> str:
    match = re.match(r"^itemid_(\d+)__(.+)$", feature_name)
    if not match:
        return feature_name

    itemid = int(match.group(1))
    stat = match.group(2)
    label = item_labels.get(itemid, f"itemid_{itemid}")
    return f"{label} [{stat}]"


# Recover the full feature list after imputation, including missingness indicators.
def _model_feature_names(pipe: Pipeline, X_cols: list[str]) -> list[str]:
    imputer = pipe.named_steps["imputer"]
    feature_names = list(X_cols)

    if getattr(imputer, "indicator_", None) is not None:
        for idx in imputer.indicator_.features_:
            feature_names.append(f"{X_cols[idx]} [missing]")

    return feature_names


# Return fitted-estimator feature importances as a sorted Series.
def _feature_importance_series(pipe: Pipeline, X_cols: list[str]) -> pd.Series:
    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    return pd.Series(
        estimator.feature_importances_,
        index=_model_feature_names(pipe, X_cols)
    ).sort_values(ascending=False)


def _shap_importance_series(
    pipe: Pipeline,
    X: pd.DataFrame,
    item_labels: dict[int, str],
    sample_n: int = 2000,
) -> pd.Series:
    X_plot = X.sample(n=min(sample_n, len(X)), random_state=SEED).copy()

    imputer = pipe.named_steps["imputer"]
    X_imp = imputer.transform(X_plot.to_numpy(dtype=float))

    # Build readable names first.
    feature_names = _model_feature_names(pipe, list(X.columns))
    feature_names = [_format_feature_name(name, item_labels) for name in feature_names]

    # Build SHAP/XGBoost-safe names for the dataframe passed into TreeExplainer.
    safe_feature_names = [_safe_shap_feature_name(name) for name in feature_names]

    X_imp_df = pd.DataFrame(X_imp, columns=safe_feature_names)

    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X_imp_df)

    vals = np.asarray(explanation.values)

    if vals.ndim == 3:
        if vals.shape[2] == 2:
            vals = vals[:, :, 1]
        else:
            vals = vals.mean(axis=2)

    shap_mean_abs = np.abs(vals).mean(axis=0)

    # Return the series indexed by the original readable names, not the safe ones.
    return pd.Series(shap_mean_abs, index=feature_names).sort_values(ascending=False)

# Convert a ranked importance series into a tidy top-features table.
def _top_series_df(
    model_name: str,
    importance_name: str,
    s: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    out = s.head(top_n).reset_index()
    out.columns = ["feature", importance_name]
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.insert(0, "model", model_name)
    return out

# Clean feature names slightly so SHAP plotting is less brittle.
def _safe_shap_feature_name(name: str) -> str:
    return (
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "lt_")
    )


# Save SHAP beeswarm and bar plots for a fitted model on a test sample.
def _save_shap_plots(
    pipe: Pipeline,
    X: pd.DataFrame,
    out_prefix: Path,
    item_labels: dict[int, str],
    sample_n: int = 2000,
) -> None:
    X_plot = X.sample(n=min(sample_n, len(X)), random_state=SEED).copy()

    imputer = pipe.named_steps["imputer"]
    X_imp = imputer.transform(X_plot.to_numpy(dtype=float))

    feature_names = _model_feature_names(pipe, list(X.columns))
    feature_names = [_format_feature_name(name, item_labels) for name in feature_names]
    feature_names = [_safe_shap_feature_name(name) for name in feature_names]

    X_imp_df = pd.DataFrame(X_imp, columns=feature_names)

    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X_imp_df)

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    shap.plots.bar(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Build IN-state rows for CAUTI scoring under the "keep catheter in" action.
def _make_cauti_keep_rows(df_in: pd.DataFrame, x_cols_cauti: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_cauti].copy()
    X["state_is_out"] = 0
    return X


# Build counterfactual CAUTI rows for scoring under the "remove now" action.
def _make_cauti_remove_rows(df_in: pd.DataFrame, x_cols_cauti: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_cauti].copy()
    X[DAYS_COL] = 1
    X["state_is_out"] = 1
    return X


# Build counterfactual reinsertion rows for scoring under the "remove now" action.
def _make_reins_remove_rows(df_in: pd.DataFrame, x_cols_reins: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_reins].copy()
    X[DAYS_COL] = 1
    return X


# Prepare a clean observed-vs-predicted frame for evaluation metrics.
def _clean_eval_frame(df: pd.DataFrame, y_col: str, p_col: str) -> pd.DataFrame:
    out = df[[y_col, p_col]].copy()
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce").clip(0.0, 1.0)
    out = out.dropna(subset=[y_col, p_col]).copy()
    out[y_col] = out[y_col].astype(int)
    return out


# Compute headline binary-performance summaries for one target/probability pair.
def _scalar_binary_metrics(df: pd.DataFrame, y_col: str, p_col: str) -> dict:
    x = _clean_eval_frame(df, y_col, p_col)

    n = int(len(x))
    events = int(x[y_col].sum()) if n > 0 else 0
    prevalence = (events / n) if n > 0 else np.nan

    auc = np.nan
    ap = np.nan
    brier = np.nan

    if n > 0:
        brier = float(brier_score_loss(x[y_col], x[p_col]))

        if x[y_col].nunique() > 1:
            auc = float(roc_auc_score(x[y_col], x[p_col]))

        if events > 0:
            ap = float(average_precision_score(x[y_col], x[p_col]))

    return {
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "auc": auc,
        "average_precision": ap,
        "brier": brier,
    }


# Build a simple calibration table by probability bins.
def _calibration_table(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    bins: int = 10,
) -> pd.DataFrame:
    x = _clean_eval_frame(df, y_col, p_col)

    if x.empty:
        return pd.DataFrame(columns=[
            "bin", "n", "events", "prevalence",
            "pred_min", "pred_mean", "pred_max", "obs_rate"
        ])

    if x[p_col].nunique() <= 1:
        x["bin"] = 0
    else:
        q = min(bins, int(x[p_col].nunique()))
        x["bin"] = pd.qcut(x[p_col], q=q, labels=False, duplicates="drop")

    out = x.groupby("bin", observed=False).agg(
        n=(y_col, "size"),
        events=(y_col, "sum"),
        prevalence=(y_col, "mean"),
        pred_min=(p_col, "min"),
        pred_mean=(p_col, "mean"),
        pred_max=(p_col, "max"),
        obs_rate=(y_col, "mean"),
    ).reset_index()

    out["bin"] = out["bin"].astype(int)
    return out


# Compute by-day evaluation summaries, optionally stratified further by group columns.
def _metrics_by_day(
    df: pd.DataFrame,
    day_col: str,
    y_col: str,
    p_col: str,
    group_cols: list[str] | None = None,
    min_rows: int = 30,
    min_events: int = 5,
) -> pd.DataFrame:
    x = df.copy()
    x[y_col] = pd.to_numeric(x[y_col], errors="coerce")
    x[p_col] = pd.to_numeric(x[p_col], errors="coerce").clip(0.0, 1.0)
    x = x.dropna(subset=[day_col, y_col, p_col]).copy()
    x[y_col] = x[y_col].astype(int)

    groupers = list(group_cols or []) + [day_col]
    rows = []

    for keys, g in x.groupby(groupers, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(groupers, keys))

        n = int(len(g))
        events = int(g[y_col].sum())
        prevalence = (events / n) if n > 0 else np.nan

        auc = np.nan
        ap = np.nan

        if n >= min_rows and events >= min_events and events < n:
            auc = float(roc_auc_score(g[y_col], g[p_col]))

        if n >= min_rows and events > 0:
            ap = float(average_precision_score(g[y_col], g[p_col]))

        rows.append({
            **key_dict,
            "n": n,
            "events": events,
            "prevalence": prevalence,
            "auc": auc,
            "average_precision": ap,
            "pred_mean": float(g[p_col].mean()),
        })

    if not rows:
        return pd.DataFrame(columns=groupers + [
            "n", "events", "prevalence", "auc", "average_precision", "pred_mean"
        ])

    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True)


# Save a dataframe in a consistent CSV format.
def _save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


# Convert numpy / pandas scalar types into JSON-safe python types.
def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# Run the full Step 1 workflow from loading the panel through saving models and metrics.
def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # Load the panel and trim any stray whitespace in column names.
    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()

    # Load lookup labels so model outputs are easier to interpret.
    item_labels = _load_item_labels()

    # Standardise core identifier / state columns.
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    # Normalise train/test split labels.
    _validate_split(df)

    # Build the base structured feature set.
    feat = _feature_cols(df)

    # Internal helper feature only; derived in Step 1, not required in the panel file.
    df["state_is_out"] = (df[STATE_COL] == "out").astype(int)

    # Predictor sets for each model.
    # interval_hours is deliberately excluded from the predictive models.
    X_cols_remove = [TIME_COL, DAYS_COL, *feat]
    X_cols_cauti = [TIME_COL, DAYS_COL, "state_is_out", *feat]
    X_cols_reins = [DAYS_COL, *feat]

    # Coerce feature columns to numeric but KEEP missing values as NaN
    # so the imputer + missingness indicators can work properly.
    feature_cols_all = list(dict.fromkeys(X_cols_remove + X_cols_cauti + X_cols_reins))

    for col in feature_cols_all:
        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0,
            })
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce true binary targets / flags to numeric and fill missing with 0.
    target_flag_cols = [ACTION_COL, Y_CAUTI, Y_REINS, LAST_DAY_COL]

    for col in target_flag_cols:
        if df[col].dtype == object:
            df[col] = df[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0,
            })
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    print("Feature missingness preserved:", int(df[feature_cols_all].isna().sum().sum()))
    # Create empty score columns that will later be filled by the fitted models.
    score_cols = pd.DataFrame(
        {
            "p_remove_obs": np.zeros(len(df), dtype=float),
            "p_cauti_if_keep": np.zeros(len(df), dtype=float),
            "p_cauti_if_remove": np.zeros(len(df), dtype=float),
            "p_cauti_if_out": np.zeros(len(df), dtype=float),
            "p_reins_if_remove": np.zeros(len(df), dtype=float),
            "p_reins_if_out": np.zeros(len(df), dtype=float),
        },
        index=df.index,
    )
    df = pd.concat([df, score_cols], axis=1).copy()

    # Sort within episode and preserve the original row order so it can be restored later.
    df = (
        df.sort_values(EPISODE_KEYS + ["day_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
        .copy()
    )

    # First-event CAUTI risk set only:
    # after a CAUTI has already happened, later rows are not used for CAUTI-risk modelling.
    df["prior_cauti_count"] = (
        df.groupby(EPISODE_KEYS)[Y_CAUTI]
        .cumsum()
        .shift(fill_value=0)
    )

    # CAUTI risk set:
    # - all IN rows
    # - OUT rows where days_in_state <= POST_REMOVE_RISK_DAYS
    df["cauti_risk_row"] = (
        (df[STATE_COL] == "in") |
        ((df[STATE_COL] == "out") & (df[DAYS_COL] <= POST_REMOVE_RISK_DAYS))
    ).astype(int)

    # Rows used to fit/evaluate the CAUTI model.
    df_cauti = df[
        (df["prior_cauti_count"] == 0) &
        (df["cauti_risk_row"] == 1)
    ].copy()

    # IN-state rows used for:
    # - fitting the removal model
    # - building counterfactual keep/remove CAUTI rows
    df_in = df[
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    # All OUT rows, before any further exclusions.
    df_out = df[df[STATE_COL] == "out"].copy()

    # Reinsertion fit set excludes terminal OUT rows that only end because ICU follow-up stops.
    df_out_fit = df_out[
        ~(
            (df_out[LAST_DAY_COL] == 1) &
            (df_out[END_REASON_COL] == "icu_end") &
            (df_out[Y_REINS] == 0)
        )
    ].copy()

    # Initialise fitted-model placeholders and headline AUC outputs.
    remove_model = None
    cauti_model = None
    reins_model = None

    auc_remove = float("nan")
    auc_cauti = float("nan")
    auc_reins = float("nan")

    # Store test design matrices for optional SHAP plotting.
    X_test_remove = None
    X_test_cauti = None
    X_test_out = None

    # Removal model on IN rows
    X_train_remove = df_in.loc[df_in[SPLIT_COL] == "train", X_cols_remove]
    y_train_remove = df_in.loc[df_in[SPLIT_COL] == "train", ACTION_COL].astype(int)

    X_test_remove = df_in.loc[df_in[SPLIT_COL] == "test", X_cols_remove]
    y_test_remove = df_in.loc[df_in[SPLIT_COL] == "test", ACTION_COL].astype(int)

    print(f"Fitting removal model with {MODEL_TYPE}...", flush=True)

    # Fit on TRAIN IN-state rows only.
    remove_model = _fit_model(X_train_remove, y_train_remove)

    # Score all eligible IN rows in the full dataset.
    df.loc[df_in.index, "p_remove_obs"] = _predict_proba(remove_model, df_in[X_cols_remove])

    # Compute headline ROC AUC on TEST rows.
    p_test_remove = _predict_proba(remove_model, X_test_remove)
    auc_remove = roc_auc_score(y_test_remove, p_test_remove) if y_test_remove.nunique() > 1 else float("nan")

    # CAUTI transition model on risk rows
    X_train_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "train", X_cols_cauti]
    y_train_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "train", Y_CAUTI].astype(int)

    X_test_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "test", X_cols_cauti]
    y_test_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "test", Y_CAUTI].astype(int)

    print(f"Fitting CAUTI transition model with {MODEL_TYPE}...", flush=True)

    # Fit the CAUTI transition model on the CAUTI risk set.
    cauti_model = _fit_model(X_train_cauti, y_train_cauti)

    # Score actual observed OUT rows, but only within the post-removal risk window.
    df_out_cauti = df_out[df_out[DAYS_COL] <= POST_REMOVE_RISK_DAYS].copy()
    df.loc[df_out_cauti.index, "p_cauti_if_out"] = _predict_proba(
        cauti_model,
        df_out_cauti[X_cols_cauti]
    )

    # Score counterfactual CAUTI probabilities on IN rows under:
    # - keep catheter in
    # - remove now
    X_keep = _make_cauti_keep_rows(df_in, X_cols_cauti)
    X_remove = _make_cauti_remove_rows(df_in, X_cols_cauti)

    df.loc[df_in.index, "p_cauti_if_keep"] = _predict_proba(cauti_model, X_keep)
    df.loc[df_in.index, "p_cauti_if_remove"] = _predict_proba(cauti_model, X_remove)

    # Compute headline ROC AUC on TEST risk rows.
    p_test_cauti = _predict_proba(cauti_model, X_test_cauti)
    auc_cauti = roc_auc_score(y_test_cauti, p_test_cauti) if y_test_cauti.nunique() > 1 else float("nan")

    # Reinsertion model on OUT rows
    X_train_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", X_cols_reins]
    y_train_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", Y_REINS].astype(int)

    X_test_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", X_cols_reins]
    y_test_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", Y_REINS].astype(int)

    print(f"Fitting reinsertion model with {MODEL_TYPE}...", flush=True)

    # Fit on TRAIN OUT-state rows only.
    reins_model = _fit_model(X_train_out, y_train_out)

    # Score all observed OUT rows in the full panel.
    df.loc[df_out.index, "p_reins_if_out"] = _predict_proba(reins_model, df_out[X_cols_reins])

    # Compute headline ROC AUC on TEST rows.
    p_test_reins = _predict_proba(reins_model, X_test_out)
    auc_reins = roc_auc_score(y_test_out, p_test_reins) if y_test_out.nunique() > 1 else float("nan")

    # Score counterfactual reinsertion probability on IN rows if removed now.
    X_cf = _make_reins_remove_rows(df_in, X_cols_reins)
    df.loc[df_in.index, "p_reins_if_remove"] = _predict_proba(reins_model, X_cf)

    # Test-set evaluation outputs

    # Removal evaluation:
    # test rows, IN state, and still pre-CAUTI
    remove_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    # CAUTI evaluation:
    # test rows in the CAUTI risk set, still pre-first-CAUTI
    cauti_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df["prior_cauti_count"] == 0) &
        (df["cauti_risk_row"] == 1)
    ].copy()

    # For observed-state CAUTI evaluation:
    # - IN rows use p_cauti_if_keep
    # - OUT residual-risk rows use p_cauti_if_out
    cauti_eval_test["p_cauti_obs_eval"] = np.where(
        cauti_eval_test[STATE_COL] == "in",
        cauti_eval_test["p_cauti_if_keep"],
        cauti_eval_test["p_cauti_if_out"],
    )

    # Reinsertion evaluation:
    # OUT-state TEST rows, excluding terminal ICU-end rows with no reinsertion.
    reins_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[STATE_COL] == "out")
    ].copy()

    reins_eval_test = reins_eval_test[
        ~(
            (reins_eval_test[LAST_DAY_COL] == 1) &
            (reins_eval_test[END_REASON_COL] == "icu_end") &
            (reins_eval_test[Y_REINS] == 0)
        )
    ].copy()

    # Headline scalar summaries
    remove_summary = _scalar_binary_metrics(remove_eval_test, ACTION_COL, "p_remove_obs")
    cauti_summary = _scalar_binary_metrics(cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval")
    reins_summary = _scalar_binary_metrics(reins_eval_test, Y_REINS, "p_reins_if_out")

    # Calibration tables
    remove_cal = _calibration_table(
        remove_eval_test, ACTION_COL, "p_remove_obs", bins=CALIBRATION_BINS
    )
    cauti_cal = _calibration_table(
        cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval", bins=CALIBRATION_BINS
    )
    reins_cal = _calibration_table(
        reins_eval_test, Y_REINS, "p_reins_if_out", bins=CALIBRATION_BINS
    )

    # By-day evaluation summaries
    remove_by_day = _metrics_by_day(
        remove_eval_test,
        DAYS_COL,
        ACTION_COL,
        "p_remove_obs",
        min_rows=MIN_ROWS_BY_DAY,
        min_events=MIN_EVENTS_BY_DAY,
    )

    # CAUTI is stratified by state because IN day 2 and OUT day 2 are different clinical settings.
    cauti_by_state_day = _metrics_by_day(
        cauti_eval_test,
        DAYS_COL,
        Y_CAUTI,
        "p_cauti_obs_eval",
        group_cols=[STATE_COL],
        min_rows=MIN_ROWS_BY_DAY,
        min_events=MIN_EVENTS_BY_DAY,
    )

    reins_by_day = _metrics_by_day(
        reins_eval_test,
        DAYS_COL,
        Y_REINS,
        "p_reins_if_out",
        min_rows=MIN_ROWS_BY_DAY,
        min_events=MIN_EVENTS_BY_DAY,
    )

    # Save test-set evaluation artifacts to disk.
    _save_df(remove_cal, OUTDIR / "remove_calibration_test.csv")
    _save_df(cauti_cal, OUTDIR / "cauti_calibration_test.csv")
    _save_df(reins_cal, OUTDIR / "reinsertion_calibration_test.csv")

    _save_df(remove_by_day, OUTDIR / "remove_by_day_test.csv")
    _save_df(cauti_by_state_day, OUTDIR / "cauti_by_state_day_test.csv")
    _save_df(reins_by_day, OUTDIR / "reinsertion_by_day_test.csv")

    # Restore original row order and drop temporary helper columns.
    df = (
        df.sort_values("_orig_index")
        .drop(columns=["_orig_index", "prior_cauti_count", "cauti_risk_row", "state_is_out"])
    )

    # Build and save separate top-feature tables for model importance and SHAP importance.
    model_feature_parts = []
    shap_feature_parts = []

    if remove_model is not None:
        remove_importance = _feature_importance_series(remove_model, X_cols_remove)
        remove_importance.index = [
            _format_feature_name(feature_name, item_labels)
            for feature_name in remove_importance.index
        ]
        model_feature_parts.append(
            _top_series_df(
                model_name="removal",
                importance_name="model_importance",
                s=remove_importance,
                top_n=TOP_FEATURES_TO_SAVE,
            )
        )

        if X_test_remove is not None and len(X_test_remove) > 0:
            remove_shap = _shap_importance_series(
                pipe=remove_model,
                X=X_test_remove,
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )
            shap_feature_parts.append(
                _top_series_df(
                    model_name="removal",
                    importance_name="shap_mean_abs",
                    s=remove_shap,
                    top_n=TOP_FEATURES_TO_SAVE,
                )
            )

    if cauti_model is not None:
        cauti_importance = _feature_importance_series(cauti_model, X_cols_cauti)
        cauti_importance.index = [
            _format_feature_name(feature_name, item_labels)
            for feature_name in cauti_importance.index
        ]
        model_feature_parts.append(
            _top_series_df(
                model_name="cauti",
                importance_name="model_importance",
                s=cauti_importance,
                top_n=TOP_FEATURES_TO_SAVE,
            )
        )

        if X_test_cauti is not None and len(X_test_cauti) > 0:
            cauti_shap = _shap_importance_series(
                pipe=cauti_model,
                X=X_test_cauti,
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )
            shap_feature_parts.append(
                _top_series_df(
                    model_name="cauti",
                    importance_name="shap_mean_abs",
                    s=cauti_shap,
                    top_n=TOP_FEATURES_TO_SAVE,
                )
            )

    if reins_model is not None:
        reins_importance = _feature_importance_series(reins_model, X_cols_reins)
        reins_importance.index = [
            _format_feature_name(feature_name, item_labels)
            for feature_name in reins_importance.index
        ]
        model_feature_parts.append(
            _top_series_df(
                model_name="reinsertion",
                importance_name="model_importance",
                s=reins_importance,
                top_n=TOP_FEATURES_TO_SAVE,
            )
        )

        if X_test_out is not None and len(X_test_out) > 0:
            reins_shap = _shap_importance_series(
                pipe=reins_model,
                X=X_test_out,
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )
            shap_feature_parts.append(
                _top_series_df(
                    model_name="reinsertion",
                    importance_name="shap_mean_abs",
                    s=reins_shap,
                    top_n=TOP_FEATURES_TO_SAVE,
                )
            )

    if model_feature_parts:
        model_feature_df = pd.concat(model_feature_parts, ignore_index=True)
        _save_df(model_feature_df, OUTDIR / "step1_top_model_features.csv")

    if shap_feature_parts:
        shap_feature_df = pd.concat(shap_feature_parts, ignore_index=True)
        _save_df(shap_feature_df, OUTDIR / "step1_top_shap_features.csv")

    # SHAP plots
    if SAVE_SHAP:
        if X_test_remove is not None and len(X_test_remove) > 0:
            _save_shap_plots(
                pipe=remove_model,
                X=X_test_remove,
                out_prefix=OUTDIR / f"remove_shap_{MODEL_TYPE}",
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )

        if X_test_cauti is not None and len(X_test_cauti) > 0:
            _save_shap_plots(
                pipe=cauti_model,
                X=X_test_cauti,
                out_prefix=OUTDIR / f"cauti_shap_{MODEL_TYPE}",
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )

        if X_test_out is not None and len(X_test_out) > 0:
            _save_shap_plots(
                pipe=reins_model,
                X=X_test_out,
                out_prefix=OUTDIR / f"reinsertion_shap_{MODEL_TYPE}",
                item_labels=item_labels,
                sample_n=SHAP_SAMPLE_N,
            )

    # Save scored panel
    out_scored = OUTDIR / "step1_scored_panel.csv"
    df.to_csv(out_scored, index=False, float_format="%.6f")

    # Save the fitted models and the feature specifications needed later by Step 2/3.
    joblib.dump(
        {
            "model_type": MODEL_TYPE,
            "remove_model": remove_model,
            "cauti_model": cauti_model,
            "reins_model": reins_model,
            "features": feat,
            "x_cols_remove": X_cols_remove,
            "x_cols_cauti": X_cols_cauti,
            "x_cols_reins": X_cols_reins,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "split_col": SPLIT_COL,
            "post_remove_risk_days": POST_REMOVE_RISK_DAYS,
        },
        MODEL_DIR / "transition_models.pkl"
    )

    # Columns that downstream steps expect to find in the scored panel.
    required_scored_cols = [
        ID_COL,
        "stay_id",
        "hadm_id",
        STATE_COL,
        DAYS_COL,
        TIME_COL,
        SPLIT_COL,
        ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        "p_remove_obs",
        "p_cauti_if_keep",
        "p_cauti_if_remove",
        "p_cauti_if_out",
        "p_reins_if_remove",
        "p_reins_if_out",
    ]

    # Collect machine-readable metadata and evaluation summaries.
    metrics = {
        "seed": SEED,
        "model_type": MODEL_TYPE,
        "post_remove_risk_days": POST_REMOVE_RISK_DAYS,
        "split": {
            "method": "precomputed patient-level split from filtered_panel.csv",
            "train_rows": int((df[SPLIT_COL] == "train").sum()),
            "test_rows": int((df[SPLIT_COL] == "test").sum()),
            "train_patients": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique()),
            "test_patients": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique()),
        },
        "auc": {
            "remove_in": auc_remove,
            "cauti_transition": auc_cauti,
            "reins_out": auc_reins,
        },
        "test_performance": {
            "remove_in": remove_summary,
            "cauti_transition": cauti_summary,
            "reins_out": reins_summary,
        },
        "n_rows": {
            "all": int(len(df)),
            "in": int((df[STATE_COL] == "in").sum()),
            "out": int((df[STATE_COL] == "out").sum()),
            "cauti_fit": int(len(df_cauti)),
            "out_fit": int(len(df_out_fit)),
        },
        "artifacts": {
            "scored_panel": str(out_scored),
            "remove_calibration_test_csv": str(OUTDIR / "remove_calibration_test.csv"),
            "cauti_calibration_test_csv": str(OUTDIR / "cauti_calibration_test.csv"),
            "reinsertion_calibration_test_csv": str(OUTDIR / "reinsertion_calibration_test.csv"),
            "remove_by_day_test_csv": str(OUTDIR / "remove_by_day_test.csv"),
            "cauti_by_state_day_test_csv": str(OUTDIR / "cauti_by_state_day_test.csv"),
            "reinsertion_by_day_test_csv": str(OUTDIR / "reinsertion_by_day_test.csv"),
            "top_model_features_csv": str(OUTDIR / "step1_top_model_features.csv"),
            "top_shap_features_csv": str(OUTDIR / "step1_top_shap_features.csv"),
        },
        "scored_panel_schema": {
            "required_columns_present": all(col in df.columns for col in required_scored_cols),
            "missing_columns": [col for col in required_scored_cols if col not in df.columns],
        },
    }

    # Save JSON metrics in a format safe for non-python consumers.
    (OUTDIR / "step1_metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2),
        encoding="utf-8"
    )

    # Final console summary
    print("\n--- SUCCESS ---", flush=True)
    print(f"Model type: {MODEL_TYPE}", flush=True)
    print(f"Scored panel saved: {out_scored}", flush=True)
    print(f"AUC removal: {auc_remove}", flush=True)
    print(f"AUC CAUTI: {auc_cauti}", flush=True)
    print(f"AUC Reinsertion: {auc_reins}", flush=True)

    print("\n--- Test-set summary ---", flush=True)
    print(
        f"Removal prevalence / AP / Brier: "
        f"{remove_summary['prevalence']:.6f} / "
        f"{remove_summary['average_precision']:.6f} / "
        f"{remove_summary['brier']:.6f}",
        flush=True
    )
    print(
        f"CAUTI prevalence / AP / Brier: "
        f"{cauti_summary['prevalence']:.6f} / "
        f"{cauti_summary['average_precision']:.6f} / "
        f"{cauti_summary['brier']:.6f}",
        flush=True
    )
    print(
        f"Reinsertion prevalence / AP / Brier: "
        f"{reins_summary['prevalence']:.6f} / "
        f"{reins_summary['average_precision']:.6f} / "
        f"{reins_summary['brier']:.6f}",
        flush=True
    )


if __name__ == "__main__":
    main()