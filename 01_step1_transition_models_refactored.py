
"""
01_step1_transition_models.py
Step 1 — Transition modelling using a pre-built Step 1 feature panel.

This version expects preprocessing to have already been run by:
    00_build_step1_feature_panel.py

It fits three transition models:
1. Removal on IN-state rows
2. CAUTI on the CAUTI risk set
3. Reinsertion on OUT-state rows

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


SEED = 42
MODEL_TYPE = "xgb"  # "rf" or "xgb"

INDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step1")
MODEL_DIR = OUTDIR

INFILE = INDIR / "step1_feature_panel.csv"
FEATURE_SPEC_FILE = OUTDIR / "step1_feature_spec.json"
D_ITEMS_FILE = Path(r"C:\Users\DavidUni\Repos\Data\MIMIC-IV\mimic-iv-3.1\icu\d_items.csv")

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
DAYS_COL = "days_in_state"
SPLIT_COL = "split"

ACTION_COL = "removed_today"
Y_CAUTI = "cauti_today"
Y_REINS = "reinsertion_today"

LAST_DAY_COL = "is_last_day_of_episode"
END_REASON_COL = "episode_end_reason"

EPISODE_KEYS = ["stay_id", "inserted"]
POST_REMOVE_RISK_DAYS = 2

TOP_FEATURES_TO_SAVE = 15

SAVE_SHAP = True
SHAP_SAMPLE_N = 2000

CALIBRATION_BINS = 10
MIN_ROWS_BY_DAY = 30
MIN_EVENTS_BY_DAY = 5


def _fit_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    if MODEL_TYPE == "rf":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
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
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
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


def _predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X.to_numpy(dtype=float))[:, 1]


def _validate_split(df: pd.DataFrame) -> None:
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()


def _load_feature_spec(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required feature specification not found: {path}. "
            f"Run 00_build_step1_feature_panel.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def _format_feature_name(feature_name: str, item_labels: dict[int, str]) -> str:
    match_missing = re.match(r"^itemid_(\d+)__(.+)__missing$", feature_name)
    if match_missing:
        itemid = int(match_missing.group(1))
        stat = match_missing.group(2)
        label = item_labels.get(itemid, f"itemid_{itemid}")
        return f"{label} [{stat}] [missing]"

    match = re.match(r"^itemid_(\d+)__(.+)$", feature_name)
    if match:
        itemid = int(match.group(1))
        stat = match.group(2)
        label = item_labels.get(itemid, f"itemid_{itemid}")
        return f"{label} [{stat}]"

    if feature_name.endswith("__missing"):
        base = feature_name[:-10]
        return f"{base} [missing]"

    return feature_name


def _model_feature_names(pipe: Pipeline, X_cols: list[str]) -> list[str]:
    return list(X_cols)


def _feature_importance_series(pipe: Pipeline, X_cols: list[str]) -> pd.Series:
    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    return pd.Series(
        estimator.feature_importances_,
        index=_model_feature_names(pipe, X_cols)
    ).sort_values(ascending=False)


def _safe_shap_feature_name(name: str) -> str:
    return (
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "lt_")
    )


def _shap_importance_series(
    pipe: Pipeline,
    X: pd.DataFrame,
    item_labels: dict[int, str],
    sample_n: int = 2000,
) -> pd.Series:
    X_plot = X.sample(n=min(sample_n, len(X)), random_state=SEED).copy()

    imputer = pipe.named_steps["imputer"]
    X_imp = imputer.transform(X_plot.to_numpy(dtype=float))

    feature_names = _model_feature_names(pipe, list(X.columns))
    feature_names = [_format_feature_name(name, item_labels) for name in feature_names]
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
    return pd.Series(shap_mean_abs, index=feature_names).sort_values(ascending=False)


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


def _make_cauti_keep_rows(df_in: pd.DataFrame, x_cols_cauti: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_cauti].copy()
    X["state_is_out"] = 0
    return X


def _make_cauti_remove_rows(df_in: pd.DataFrame, x_cols_cauti: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_cauti].copy()
    X[DAYS_COL] = 1
    X["state_is_out"] = 1
    return X


def _make_reins_remove_rows(df_in: pd.DataFrame, x_cols_reins: list[str]) -> pd.DataFrame:
    X = df_in[x_cols_reins].copy()
    X[DAYS_COL] = 1
    return X


def _clean_eval_frame(df: pd.DataFrame, y_col: str, p_col: str) -> pd.DataFrame:
    out = df[[y_col, p_col]].copy()
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce").clip(0.0, 1.0)
    out = out.dropna(subset=[y_col, p_col]).copy()
    out[y_col] = out[y_col].astype(int)
    return out


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


def _save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


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


def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    feature_spec = _load_feature_spec(FEATURE_SPEC_FILE)

    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()

    item_labels = _load_item_labels()

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()
    _validate_split(df)

    feat = feature_spec["features"]
    x_cols_remove = feature_spec["x_cols_remove"]
    x_cols_cauti = feature_spec["x_cols_cauti"]
    x_cols_reins = feature_spec["x_cols_reins"]

    required_cols = list(dict.fromkeys(
        x_cols_remove + x_cols_cauti + x_cols_reins +
        [ID_COL, "stay_id", "hadm_id", STATE_COL, DAYS_COL, TIME_COL, SPLIT_COL,
         ACTION_COL, Y_CAUTI, Y_REINS, LAST_DAY_COL, END_REASON_COL, "day_end", *EPISODE_KEYS]
    ))
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "The Step 1 feature panel is missing required columns. "
            f"Run 00_build_step1_feature_panel.py again. Missing: {missing_required}"
        )

    print("Step 1 uses precomputed explicit features only.")
    print("Hidden imputer indicator columns created inside Step 1: 0")
    print(f"Loaded Step 1 features: {len(feat)}")

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

    df = (
        df.sort_values(EPISODE_KEYS + ["day_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
        .copy()
    )

    df["prior_cauti_count"] = (
        df.groupby(EPISODE_KEYS)[Y_CAUTI]
        .cumsum()
        .shift(fill_value=0)
    )

    df["cauti_risk_row"] = (
        (df[STATE_COL] == "in") |
        ((df[STATE_COL] == "out") & (df[DAYS_COL] <= POST_REMOVE_RISK_DAYS))
    ).astype(int)

    df_cauti = df[
        (df["prior_cauti_count"] == 0) &
        (df["cauti_risk_row"] == 1)
    ].copy()

    df_in = df[
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    df_out = df[df[STATE_COL] == "out"].copy()

    df_out_fit = df_out[
        ~(
            (df_out[LAST_DAY_COL] == 1) &
            (df_out[END_REASON_COL] == "icu_end") &
            (df_out[Y_REINS] == 0)
        )
    ].copy()

    remove_model = None
    cauti_model = None
    reins_model = None

    auc_remove = float("nan")
    auc_cauti = float("nan")
    auc_reins = float("nan")

    X_test_remove = None
    X_test_cauti = None
    X_test_out = None

    X_train_remove = df_in.loc[df_in[SPLIT_COL] == "train", x_cols_remove]
    y_train_remove = df_in.loc[df_in[SPLIT_COL] == "train", ACTION_COL].astype(int)

    X_test_remove = df_in.loc[df_in[SPLIT_COL] == "test", x_cols_remove]
    y_test_remove = df_in.loc[df_in[SPLIT_COL] == "test", ACTION_COL].astype(int)

    print(f"Fitting removal model with {MODEL_TYPE}...", flush=True)
    remove_model = _fit_model(X_train_remove, y_train_remove)

    df.loc[df_in.index, "p_remove_obs"] = _predict_proba(remove_model, df_in[x_cols_remove])

    p_test_remove = _predict_proba(remove_model, X_test_remove)
    auc_remove = roc_auc_score(y_test_remove, p_test_remove) if y_test_remove.nunique() > 1 else float("nan")

    X_train_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "train", x_cols_cauti]
    y_train_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "train", Y_CAUTI].astype(int)

    X_test_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "test", x_cols_cauti]
    y_test_cauti = df_cauti.loc[df_cauti[SPLIT_COL] == "test", Y_CAUTI].astype(int)

    print(f"Fitting CAUTI transition model with {MODEL_TYPE}...", flush=True)
    cauti_model = _fit_model(X_train_cauti, y_train_cauti)

    df_out_cauti = df_out[df_out[DAYS_COL] <= POST_REMOVE_RISK_DAYS].copy()
    df.loc[df_out_cauti.index, "p_cauti_if_out"] = _predict_proba(
        cauti_model,
        df_out_cauti[x_cols_cauti]
    )

    X_keep = _make_cauti_keep_rows(df_in, x_cols_cauti)
    X_remove = _make_cauti_remove_rows(df_in, x_cols_cauti)

    df.loc[df_in.index, "p_cauti_if_keep"] = _predict_proba(cauti_model, X_keep)
    df.loc[df_in.index, "p_cauti_if_remove"] = _predict_proba(cauti_model, X_remove)

    p_test_cauti = _predict_proba(cauti_model, X_test_cauti)
    auc_cauti = roc_auc_score(y_test_cauti, p_test_cauti) if y_test_cauti.nunique() > 1 else float("nan")

    X_train_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", x_cols_reins]
    y_train_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", Y_REINS].astype(int)

    X_test_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", x_cols_reins]
    y_test_out = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", Y_REINS].astype(int)

    print(f"Fitting reinsertion model with {MODEL_TYPE}...", flush=True)
    reins_model = _fit_model(X_train_out, y_train_out)

    df.loc[df_out.index, "p_reins_if_out"] = _predict_proba(reins_model, df_out[x_cols_reins])

    p_test_reins = _predict_proba(reins_model, X_test_out)
    auc_reins = roc_auc_score(y_test_out, p_test_reins) if y_test_out.nunique() > 1 else float("nan")

    X_cf = _make_reins_remove_rows(df_in, x_cols_reins)
    df.loc[df_in.index, "p_reins_if_remove"] = _predict_proba(reins_model, X_cf)

    remove_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    cauti_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df["prior_cauti_count"] == 0) &
        (df["cauti_risk_row"] == 1)
    ].copy()

    cauti_eval_test["p_cauti_obs_eval"] = np.where(
        cauti_eval_test[STATE_COL] == "in",
        cauti_eval_test["p_cauti_if_keep"],
        cauti_eval_test["p_cauti_if_out"],
    )

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

    remove_summary = _scalar_binary_metrics(remove_eval_test, ACTION_COL, "p_remove_obs")
    cauti_summary = _scalar_binary_metrics(cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval")
    reins_summary = _scalar_binary_metrics(reins_eval_test, Y_REINS, "p_reins_if_out")

    remove_cal = _calibration_table(
        remove_eval_test, ACTION_COL, "p_remove_obs", bins=CALIBRATION_BINS
    )
    cauti_cal = _calibration_table(
        cauti_eval_test, Y_CAUTI, "p_cauti_obs_eval", bins=CALIBRATION_BINS
    )
    reins_cal = _calibration_table(
        reins_eval_test, Y_REINS, "p_reins_if_out", bins=CALIBRATION_BINS
    )

    remove_by_day = _metrics_by_day(
        remove_eval_test,
        DAYS_COL,
        ACTION_COL,
        "p_remove_obs",
        min_rows=MIN_ROWS_BY_DAY,
        min_events=MIN_EVENTS_BY_DAY,
    )

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

    _save_df(remove_cal, OUTDIR / "remove_calibration_test.csv")
    _save_df(cauti_cal, OUTDIR / "cauti_calibration_test.csv")
    _save_df(reins_cal, OUTDIR / "reinsertion_calibration_test.csv")

    _save_df(remove_by_day, OUTDIR / "remove_by_day_test.csv")
    _save_df(cauti_by_state_day, OUTDIR / "cauti_by_state_day_test.csv")
    _save_df(reins_by_day, OUTDIR / "reinsertion_by_day_test.csv")

    df = (
        df.sort_values("_orig_index")
        .drop(columns=["_orig_index", "prior_cauti_count", "cauti_risk_row"])
    )

    model_feature_parts = []
    shap_feature_parts = []

    if remove_model is not None:
        remove_importance = _feature_importance_series(remove_model, x_cols_remove)
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
        cauti_importance = _feature_importance_series(cauti_model, x_cols_cauti)
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
        reins_importance = _feature_importance_series(reins_model, x_cols_reins)
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

    out_scored = OUTDIR / "step1_scored_panel.csv"
    df.to_csv(out_scored, index=False, float_format="%.6f")

    joblib.dump(
        {
            "model_type": MODEL_TYPE,
            "remove_model": remove_model,
            "cauti_model": cauti_model,
            "reins_model": reins_model,
            "features": feat,
            "x_cols_remove": x_cols_remove,
            "x_cols_cauti": x_cols_cauti,
            "x_cols_reins": x_cols_reins,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "split_col": SPLIT_COL,
            "post_remove_risk_days": POST_REMOVE_RISK_DAYS,
            "feature_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
        },
        MODEL_DIR / "transition_models.pkl"
    )

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

    metrics = {
        "seed": SEED,
        "model_type": MODEL_TYPE,
        "post_remove_risk_days": POST_REMOVE_RISK_DAYS,
        "split": {
            "method": "precomputed patient-level split from step1_feature_panel.csv",
            "train_rows": int((df[SPLIT_COL] == "train").sum()),
            "test_rows": int((df[SPLIT_COL] == "test").sum()),
            "train_patients": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique()),
            "test_patients": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique()),
        },
        "preprocessing": {
            "feature_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
            "hidden_indicator_columns_created_inside_step1": 0,
            "explicit_feature_count": int(len(feat)),
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

    (OUTDIR / "step1_metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2),
        encoding="utf-8"
    )

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
