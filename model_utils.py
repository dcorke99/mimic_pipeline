from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


SEED = 42
MODEL_TYPE = "xgb"  # "rf" or "xgb"

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\catheter_models")
MODEL_DIR = OUTDIR

INFILE = INDIR / "modeling_panel.csv"
FEATURE_SPEC_FILE = INDIR / "feature_spec.json"
COVARIATE_DICT_FILE = INDIR / "covariate_dictionary.csv"

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
SPLIT_COL = "split"

ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"

LAST_PERIOD_COL = "is_last_period_of_episode"
END_REASON_COL = "episode_end_reason"
AT_RISK_CAUTI = "at_risk_cauti"
AT_RISK_REINS = "at_risk_reinsertion"

EPISODE_KEYS = ["stay_id", "inserted"]
POST_REMOVE_RISK_PERIODS = 2

TOP_FEATURES_TO_SAVE = 15
SAVE_SHAP = True
SHAP_SAMPLE_N = 2000
CALIBRATION_BINS = 10


def fit_model(features, target):
    if MODEL_TYPE == "rf":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=SEED,
            )),
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
            )),
        ])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    pipe.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=int))
    return pipe


def predict_proba(pipe, features):
    return pipe.predict_proba(features.to_numpy(dtype=float))[:, 1]


def feature_importance_series(pipe, feature_cols):
    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    return pd.Series(estimator.feature_importances_, index=list(feature_cols)).sort_values(ascending=False)


def shap_importance_series(pipe, features, sample_n=2000):
    sample_df = features.sample(n=min(sample_n, len(features)), random_state=SEED).copy()
    imputer = pipe.named_steps["imputer"]
    features_imp = imputer.transform(sample_df.to_numpy(dtype=float))
    feature_names = list(features.columns)
    features_imp_df = pd.DataFrame(features_imp, columns=feature_names)

    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(features_imp_df)
    shap_values = np.asarray(explanation.values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values.mean(axis=2)

    return pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)


def top_series_df(model_name, importance_name, values, top_n):
    top_df = values.head(top_n).reset_index()
    top_df.columns = ["feature", importance_name]
    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
    top_df.insert(0, "model", model_name)
    return top_df


def add_feature_descriptions(feature_df, covariate_dict):
    itemid_to_label = {}
    for row in covariate_dict.itertuples(index=False):
        if hasattr(row, "itemid") and pd.notna(row.itemid):
            itemid_to_label[int(row.itemid)] = str(row.label)

    pattern = re.compile(r"^itemid_(\d+)__(.+?)(?:__missing)?$")

    def describe_feature(feature):
        match = pattern.match(str(feature))
        if not match:
            return pd.NA
        itemid = int(match.group(1))
        stat = match.group(2)
        description = f"{itemid_to_label.get(itemid, 'UNKNOWN ITEMID')} [{stat}]"
        if str(feature).endswith("__missing"):
            description = f"{description} [missing]"
        return description

    feature_df = feature_df.copy()
    feature_df.insert(3, "description", feature_df["feature"].apply(describe_feature))
    return feature_df


def save_shap_plots(pipe, features, out_prefix, sample_n=2000):
    sample_df = features.sample(n=min(sample_n, len(features)), random_state=SEED).copy()
    imputer = pipe.named_steps["imputer"]
    features_imp = imputer.transform(sample_df.to_numpy(dtype=float))
    feature_names = list(features.columns)
    features_imp_df = pd.DataFrame(features_imp, columns=feature_names)

    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(features_imp_df)

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


def clean_eval_frame(df, outcome_col, pred_col):
    eval_df = df[[outcome_col, pred_col]].copy()
    eval_df[outcome_col] = pd.to_numeric(eval_df[outcome_col], errors="coerce")
    eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce").clip(0.0, 1.0)
    eval_df = eval_df.dropna(subset=[outcome_col, pred_col]).copy()
    eval_df[outcome_col] = eval_df[outcome_col].astype(int)
    return eval_df


def scalar_binary_metrics(df, outcome_col, pred_col):
    eval_df = clean_eval_frame(df, outcome_col, pred_col)
    n = int(len(eval_df))
    events = int(eval_df[outcome_col].sum())
    prevalence = events / n if n else np.nan
    auc = float(roc_auc_score(eval_df[outcome_col], eval_df[pred_col])) if n and eval_df[outcome_col].nunique() > 1 else np.nan
    avg_precision = float(average_precision_score(eval_df[outcome_col], eval_df[pred_col])) if n and events > 0 else np.nan
    brier = float(brier_score_loss(eval_df[outcome_col], eval_df[pred_col])) if n else np.nan
    return {
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "auc": auc,
        "average_precision": avg_precision,
        "brier": brier,
    }


def calibration_table(df, outcome_col, pred_col, bins=10):
    eval_df = clean_eval_frame(df, outcome_col, pred_col)
    if eval_df.empty or eval_df[pred_col].nunique() == 0:
        return pd.DataFrame(columns=["bin", "n", "events", "prevalence", "pred_min", "pred_mean", "pred_max", "obs_rate"])

    q = min(bins, int(eval_df[pred_col].nunique()))
    eval_df["bin"] = pd.qcut(eval_df[pred_col], q=q, labels=False, duplicates="drop")
    calib_df = eval_df.groupby("bin", observed=False).agg(
        n=(outcome_col, "size"),
        events=(outcome_col, "sum"),
        prevalence=(outcome_col, "mean"),
        pred_min=(pred_col, "min"),
        pred_mean=(pred_col, "mean"),
        pred_max=(pred_col, "max"),
        obs_rate=(outcome_col, "mean"),
    ).reset_index()
    calib_df["bin"] = calib_df["bin"].astype(int)
    return calib_df


def metrics_by_period(df, periods_col, outcome_col, pred_col, group_cols=None):
    eval_df = df.copy()
    eval_df[outcome_col] = pd.to_numeric(eval_df[outcome_col], errors="coerce")
    eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce").clip(0.0, 1.0)
    eval_df = eval_df.dropna(subset=[periods_col, outcome_col, pred_col]).copy()
    eval_df[outcome_col] = eval_df[outcome_col].astype(int)

    groupers = list(group_cols or []) + [periods_col]
    rows = []
    for keys, group_df in eval_df.groupby(groupers, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(groupers, keys))
        n = int(len(group_df))
        events = int(group_df[outcome_col].sum())
        rows.append({
            **key_dict,
            "n": n,
            "events": events,
            "prevalence": events / n if n else np.nan,
            "auc": float(roc_auc_score(group_df[outcome_col], group_df[pred_col])) if n and group_df[outcome_col].nunique() > 1 else np.nan,
            "average_precision": float(average_precision_score(group_df[outcome_col], group_df[pred_col])) if n and events > 0 else np.nan,
            "pred_mean": float(group_df[pred_col].mean()),
        })

    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True) if rows else pd.DataFrame()


def save_df(df, path):
    df.to_csv(path, index=False, float_format="%.6f")


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if pd.isna(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def load_feature_spec():
    return json.loads(FEATURE_SPEC_FILE.read_text(encoding="utf-8"))


def load_panel():
    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()

    missing_risk_cols = [c for c in [AT_RISK_CAUTI, AT_RISK_REINS] if c not in df.columns]
    if missing_risk_cols:
        raise ValueError(f"Missing required risk-set columns: {missing_risk_cols}")

    df[AT_RISK_CAUTI] = pd.to_numeric(df[AT_RISK_CAUTI], errors="coerce").fillna(0).astype(int)
    df[AT_RISK_REINS] = pd.to_numeric(df[AT_RISK_REINS], errors="coerce").fillna(0).astype(int)
    return df


def add_prior_cauti_count(df):
    df = (
        df.sort_values(EPISODE_KEYS + ["period_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
        .copy()
    )
    # Count prior CAUTI events within the same episode only.
    cauti_events = pd.to_numeric(df[Y_CAUTI], errors="coerce").fillna(0).astype(int)
    df["prior_cauti_count"] = cauti_events.groupby([df[key] for key in EPISODE_KEYS]).cumsum() - cauti_events
    return df


def restore_original_order(df):
    drop_cols = [c for c in ["_orig_index", "prior_cauti_count"] if c in df.columns]
    if "_orig_index" in df.columns:
        df = df.sort_values("_orig_index")
    return df.drop(columns=drop_cols)


def insert_score_columns_before_age(df, score_col_names):
    ordered_cols = [col for col in df.columns if col not in score_col_names]
    insert_at = ordered_cols.index("age") if "age" in ordered_cols else len(ordered_cols)
    ordered_cols[insert_at:insert_at] = [col for col in score_col_names if col in df.columns]
    return df[ordered_cols].copy()


def dump_joblib(payload, path):
    joblib.dump(payload, path)
