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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_TYPE = "xgb"  # "rf" or "xgb" for the binary propensity model.

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\nuisance_models")
MODEL_DIR = OUTDIR

INFILE = INDIR / "modeling_panel.csv"
FEATURE_SPEC_FILE = INDIR / "feature_spec.json"
COVARIATE_DICT_FILE = INDIR / "covariate_dictionary.csv"
FINAL_PANEL = OUTDIR / "scored_panel.csv"

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
SPLIT_COL = "split"

ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
Y_DEATH = "death_in_period"
Y_ICU_EXIT = "icu_end_in_period"
TRANSITION_LABEL_COL = "next_state"
OBSERVED_ACTION_COL = "observed_action"
ACTION_REMOVE_COL = "action_remove"
ACTION_OUT_COL = "action_out"

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

PROPENSITY_SCORE_COL = "p_remove_obs"
TRANSITION_CLASSES = [
    "cauti",
    "reinsertion",
    "removal",
    "death",
    "icu_exit_alive",
    "no_event_continue",
]
ACTION_VALUES = ["keep", "remove", "out"]
OUTCOME_SCORE_COLS = [
    f"p_next_{transition_class}_if_{action}"
    for action in ACTION_VALUES
    for transition_class in TRANSITION_CLASSES
]
OBSERVED_TRANSITION_SCORE_COLS = [
    f"p_next_{transition_class}_obs"
    for transition_class in TRANSITION_CLASSES
]
ALL_SCORE_COLS = [PROPENSITY_SCORE_COL, *OUTCOME_SCORE_COLS, *OBSERVED_TRANSITION_SCORE_COLS]
LOW_COUNT_WARNING_THRESHOLD = 20


# =============================================================================
# Support functions
# =============================================================================

def require_columns(df, cols, context):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required {context} columns: {missing}")


def save_df(df, path):
    df.to_csv(path, index=False, float_format="%.6f")


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


def dump_joblib(payload, path):
    joblib.dump(payload, path)


def insert_score_columns_before_age(df, score_col_names):
    ordered_cols = [col for col in df.columns if col not in score_col_names]
    insert_at = ordered_cols.index("age") if "age" in ordered_cols else len(ordered_cols)
    ordered_cols[insert_at:insert_at] = [col for col in score_col_names if col in df.columns]
    return df[ordered_cols].copy()


def fit_binary_model(features, target):
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


def predict_binary_proba(pipe, features):
    return pipe.predict_proba(features.to_numpy(dtype=float))[:, 1]


def feature_importance_series(pipe, feature_cols):
    estimator = pipe.named_steps["rf"] if "rf" in pipe.named_steps else pipe.named_steps["xgb"]
    return pd.Series(estimator.feature_importances_, index=list(feature_cols)).sort_values(ascending=False)


def shap_importance_series(pipe, features, sample_n=2000):
    sample_df = features.sample(n=min(sample_n, len(features)), random_state=SEED).copy()
    imputer = pipe.named_steps["imputer"]
    features_imp = imputer.transform(sample_df.to_numpy(dtype=float))
    feature_names = list(features.columns)
    features_imp_df = pd.DataFrame(features_imp, columns=feature_names)

    estimator = pipe.named_steps["rf"] if "rf" in pipe.named_steps else pipe.named_steps["xgb"]
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

    estimator = pipe.named_steps["rf"] if "rf" in pipe.named_steps else pipe.named_steps["xgb"]
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


def derive_transition_label(df):
    require_columns(df, [ACTION_COL, Y_CAUTI, Y_REINS, Y_DEATH, Y_ICU_EXIT], "transition-label")
    out = pd.Series("no_event_continue", index=df.index, dtype="object")
    out.loc[pd.to_numeric(df[ACTION_COL], errors="coerce").fillna(0).astype(int) == 1] = "removal"
    out.loc[pd.to_numeric(df[Y_REINS], errors="coerce").fillna(0).astype(int) == 1] = "reinsertion"
    out.loc[pd.to_numeric(df[Y_CAUTI], errors="coerce").fillna(0).astype(int) == 1] = "cauti"
    icu_exit = pd.to_numeric(df[Y_ICU_EXIT], errors="coerce").fillna(0).astype(int)
    death = pd.to_numeric(df[Y_DEATH], errors="coerce").fillna(0).astype(int)
    out.loc[(icu_exit == 1) & (death == 0)] = "icu_exit_alive"
    out.loc[death == 1] = "death"
    return out


def derive_observed_action(df):
    require_columns(df, [STATE_COL, ACTION_COL], "observed-action")
    action = pd.Series("keep", index=df.index, dtype="object")
    removed = pd.to_numeric(df[ACTION_COL], errors="coerce").fillna(0).astype(int)
    action.loc[(df[STATE_COL] == "in") & (removed == 1)] = "remove"
    action.loc[df[STATE_COL] == "out"] = "out"
    return action


def prepare_transition_frame(df):
    df = df.copy()
    # Re-derive these labels so the model target and action features stay consistent.
    df[TRANSITION_LABEL_COL] = derive_transition_label(df)
    df[OBSERVED_ACTION_COL] = derive_observed_action(df)
    df[ACTION_REMOVE_COL] = (df[OBSERVED_ACTION_COL] == "remove").astype(int)
    df[ACTION_OUT_COL] = (df[OBSERVED_ACTION_COL] == "out").astype(int)
    if "state_is_out" not in df.columns:
        df["state_is_out"] = (df[STATE_COL] == "out").astype(int)
    return df


def class_codes(labels):
    class_to_code = {label: idx for idx, label in enumerate(TRANSITION_CLASSES)}
    unknown = sorted(set(labels.dropna()) - set(class_to_code))
    if unknown:
        raise ValueError(f"Unexpected transition labels: {unknown}")
    return labels.map(class_to_code).astype(int)


def fit_transition_model(features, target_codes, n_classes):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("xgb", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
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
    pipe.fit(features.to_numpy(dtype=float), target_codes)
    return pipe


def predict_transition_proba(pipe, features):
    return pipe.predict_proba(features.to_numpy(dtype=float))


def score_action(df, row_mask, action, feature_cols, model):
    if int(row_mask.sum()) == 0:
        return np.empty((0, len(TRANSITION_CLASSES)))

    if action == "keep":
        state_is_out, action_remove, action_out = 0, 0, 0
    elif action == "remove":
        state_is_out, action_remove, action_out = 0, 1, 0
    elif action == "out":
        state_is_out, action_remove, action_out = 1, 0, 1
    else:
        raise ValueError(f"Unknown action for scoring: {action}")

    features = df.loc[row_mask, feature_cols].copy()
    features["state_is_out"] = state_is_out
    features[ACTION_REMOVE_COL] = action_remove
    features[ACTION_OUT_COL] = action_out
    return predict_transition_proba(model, features[feature_cols])


def assign_action_scores(df, row_mask, action, proba):
    target_index = df.index[row_mask]
    for class_idx, transition_class in enumerate(TRANSITION_CLASSES):
        df.loc[target_index, f"p_next_{transition_class}_if_{action}"] = proba[:, class_idx]


def assign_observed_scores(df):
    for transition_class in TRANSITION_CLASSES:
        df[f"p_next_{transition_class}_obs"] = np.nan

    for action in ACTION_VALUES:
        action_rows = df[OBSERVED_ACTION_COL] == action
        for transition_class in TRANSITION_CLASSES:
            source_col = f"p_next_{transition_class}_if_{action}"
            target_col = f"p_next_{transition_class}_obs"
            df.loc[action_rows, target_col] = df.loc[action_rows, source_col]


def prob_sum_summary(df):
    rows = []
    for action in ACTION_VALUES:
        cols = [f"p_next_{transition_class}_if_{action}" for transition_class in TRANSITION_CLASSES]
        sums = df[cols].sum(axis=1, skipna=False).dropna()
        rows.append({
            "action": action,
            "n": int(len(sums)),
            "min_sum": float(sums.min()) if len(sums) else np.nan,
            "mean_sum": float(sums.mean()) if len(sums) else np.nan,
            "max_sum": float(sums.max()) if len(sums) else np.nan,
            "max_abs_error_from_1": float((sums - 1.0).abs().max()) if len(sums) else np.nan,
        })
    return pd.DataFrame(rows)


def test_transition_metrics(eval_df, proba, y_codes):
    metrics = {
        "n": int(len(eval_df)),
        "multiclass_log_loss": None,
        "one_vs_rest_auc": {},
        "risk_set_auc": {},
    }
    if len(eval_df):
        labels = list(range(len(TRANSITION_CLASSES)))
        metrics["multiclass_log_loss"] = float(log_loss(y_codes, proba, labels=labels))
        for class_idx, transition_class in enumerate(TRANSITION_CLASSES):
            binary_target = (y_codes == class_idx).astype(int)
            if binary_target.nunique() > 1:
                metrics["one_vs_rest_auc"][transition_class] = float(
                    roc_auc_score(binary_target, proba[:, class_idx])
                )
            else:
                metrics["one_vs_rest_auc"][transition_class] = None

            if transition_class == "cauti":
                risk_mask = (
                    pd.to_numeric(eval_df[AT_RISK_CAUTI], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .to_numpy() == 1
                )
            elif transition_class == "reinsertion":
                risk_mask = (eval_df[OBSERVED_ACTION_COL] == "out").to_numpy()
            elif transition_class == "removal":
                risk_mask = (eval_df[STATE_COL] == "in").to_numpy()
            else:
                risk_mask = np.ones(len(eval_df), dtype=bool)

            risk_target = np.asarray(binary_target)[risk_mask]
            if len(risk_target) and len(np.unique(risk_target)) > 1:
                metrics["risk_set_auc"][transition_class] = float(roc_auc_score(risk_target, proba[risk_mask, class_idx]))
            else:
                metrics["risk_set_auc"][transition_class] = None
    return metrics


def predicted_vs_observed(eval_df):
    rows = []
    n = len(eval_df)
    for transition_class in TRANSITION_CLASSES:
        rows.append({
            "next_state": transition_class,
            "n": int(n),
            "observed_count": int((eval_df[TRANSITION_LABEL_COL] == transition_class).sum()),
            "observed_rate": float((eval_df[TRANSITION_LABEL_COL] == transition_class).mean()) if n else np.nan,
            "predicted_mean": float(eval_df[f"p_next_{transition_class}_obs"].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def transition_diagnostic_rows(transition_summary, class_distribution, pred_vs_obs, probability_summary):
    rows = [
        {
            "section": "model",
            "name": "multiclass_log_loss",
            "next_state": pd.NA,
            "action": pd.NA,
            "value": transition_summary["multiclass_log_loss"],
        }
    ]
    for transition_class, auc in transition_summary["one_vs_rest_auc"].items():
        rows.append({
            "section": "model",
            "name": "one_vs_rest_auc",
            "next_state": transition_class,
            "action": pd.NA,
            "value": auc,
        })
    for transition_class, auc in transition_summary["risk_set_auc"].items():
        rows.append({
            "section": "model",
            "name": "risk_set_auc",
            "next_state": transition_class,
            "action": pd.NA,
            "value": auc,
        })

    for row in class_distribution.itertuples(index=False):
        rows.extend([
            {"section": "class_distribution", "name": "n", "next_state": row.next_state, "action": pd.NA, "value": row.n},
            {"section": "class_distribution", "name": "rate", "next_state": row.next_state, "action": pd.NA, "value": row.rate},
        ])

    for row in pred_vs_obs.itertuples(index=False):
        rows.extend([
            {"section": "predicted_vs_observed_test", "name": "observed_count", "next_state": row.next_state, "action": pd.NA, "value": row.observed_count},
            {"section": "predicted_vs_observed_test", "name": "observed_rate", "next_state": row.next_state, "action": pd.NA, "value": row.observed_rate},
            {"section": "predicted_vs_observed_test", "name": "predicted_mean", "next_state": row.next_state, "action": pd.NA, "value": row.predicted_mean},
        ])

    for row in probability_summary.itertuples(index=False):
        rows.extend([
            {"section": "probability_sum", "name": "n", "next_state": pd.NA, "action": row.action, "value": row.n},
            {"section": "probability_sum", "name": "mean_sum", "next_state": pd.NA, "action": row.action, "value": row.mean_sum},
            {"section": "probability_sum", "name": "max_abs_error_from_1", "next_state": pd.NA, "action": row.action, "value": row.max_abs_error_from_1},
        ])
    return pd.DataFrame(rows)


def propensity_summary_rows(df, eligible_df, feature_list, feature_spec, summary):
    return pd.DataFrame([
        {"metric": "model_type", "value": MODEL_TYPE},
        {"metric": "period_hours", "value": feature_spec.get("period_hours")},
        {"metric": "train_rows", "value": int((df[SPLIT_COL] == "train").sum())},
        {"metric": "test_rows", "value": int((df[SPLIT_COL] == "test").sum())},
        {"metric": "train_patients", "value": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique())},
        {"metric": "test_patients", "value": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique())},
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "decision_eligible_rows", "value": int(len(eligible_df))},
        {"metric": "decision_eligible_train", "value": int((eligible_df[SPLIT_COL] == "train").sum())},
        {"metric": "decision_eligible_test", "value": int((eligible_df[SPLIT_COL] == "test").sum())},
        {"metric": "explicit_feature_count", "value": int(len(feature_list))},
        {"metric": "test_n", "value": summary["n"]},
        {"metric": "test_events", "value": summary["events"]},
        {"metric": "test_prevalence", "value": summary["prevalence"]},
        {"metric": "test_auc", "value": summary["auc"]},
        {"metric": "test_average_precision", "value": summary["average_precision"]},
        {"metric": "test_brier", "value": summary["brier"]},
    ])


def outcome_summary_rows(
    df,
    feature_list,
    transition_feature_cols,
    train_df,
    test_df,
    in_rows,
    out_rows,
    low_count_classes,
    period_hours,
):
    return pd.DataFrame([
        {"metric": "model_type", "value": "xgb_multi_softprob"},
        {"metric": "period_hours", "value": period_hours},
        {"metric": "train_rows", "value": int(len(train_df))},
        {"metric": "test_rows", "value": int(len(test_df))},
        {"metric": "train_patients", "value": int(df.loc[df[SPLIT_COL] == "train", ID_COL].nunique())},
        {"metric": "test_patients", "value": int(df.loc[df[SPLIT_COL] == "test", ID_COL].nunique())},
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "in_rows_scored_keep_remove", "value": int(in_rows.sum())},
        {"metric": "out_rows_scored_out", "value": int(out_rows.sum())},
        {"metric": "explicit_feature_count", "value": int(len(feature_list))},
        {"metric": "transition_feature_count", "value": int(len(transition_feature_cols))},
        {"metric": "transition_classes", "value": "|".join(TRANSITION_CLASSES)},
        {"metric": "action_values", "value": "|".join(ACTION_VALUES)},
        {"metric": "low_count_classes", "value": "|".join(low_count_classes)},
    ])


# =============================================================================
# Propensity model
# =============================================================================

def fit_propensity_scores(df, feature_spec):
    feature_list = feature_spec["features"]
    remove_feature_cols = feature_spec["x_cols_remove"]
    require_columns(df, [STATE_COL, SPLIT_COL, ACTION_COL, *remove_feature_cols], "propensity")

    print("Propensity model uses precomputed explicit features only.")
    print("Hidden imputer indicator columns created inside propensity fit: 0")
    print(f"Loaded features: {len(feature_list)}")

    df = df.copy()
    df[PROPENSITY_SCORE_COL] = np.zeros(len(df), dtype=float)

    eligible_df = df[df[STATE_COL] == "in"].copy()
    train_features = eligible_df.loc[eligible_df[SPLIT_COL] == "train", remove_feature_cols]
    train_target = eligible_df.loc[eligible_df[SPLIT_COL] == "train", ACTION_COL].astype(int)

    print(f"Fitting removal propensity model with {MODEL_TYPE}...", flush=True)
    remove_model = fit_binary_model(train_features, train_target)
    df.loc[eligible_df.index, PROPENSITY_SCORE_COL] = predict_binary_proba(remove_model, eligible_df[remove_feature_cols])

    eval_test = df[(df[SPLIT_COL] == "test") & (df[STATE_COL] == "in")].copy()
    summary = scalar_binary_metrics(eval_test, ACTION_COL, PROPENSITY_SCORE_COL)

    print(f"AUC removal: {summary['auc']}", flush=True)
    return df, remove_model, summary


# =============================================================================
# Outcome model
# =============================================================================

def fit_outcome_scores(df, feature_spec):
    df = prepare_transition_frame(df)

    feature_list = feature_spec["features"]
    transition_feature_cols = feature_spec.get("x_cols_transition")
    if not transition_feature_cols:
        transition_feature_cols = [
            TIME_COL,
            PERIODS_COL,
            "state_is_out",
            ACTION_REMOVE_COL,
            ACTION_OUT_COL,
            *feature_list,
        ]
    require_columns(df, transition_feature_cols, "transition-feature")

    print("Transition model uses precomputed explicit features plus observed/counterfactual action indicators.")
    print("Hidden imputer indicator columns created inside transition fit: 0")
    print(f"Loaded features: {len(feature_list)}")
    print(f"Transition features: {len(transition_feature_cols)}")

    for col in OUTCOME_SCORE_COLS + OBSERVED_TRANSITION_SCORE_COLS:
        df[col] = np.nan

    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()
    train_y = class_codes(train_df[TRANSITION_LABEL_COL])
    test_y = class_codes(test_df[TRANSITION_LABEL_COL])

    class_distribution = (
        df[TRANSITION_LABEL_COL]
        .value_counts()
        .reindex(TRANSITION_CLASSES, fill_value=0)
        .rename_axis("next_state")
        .reset_index(name="n")
    )
    class_distribution["rate"] = class_distribution["n"] / len(df)

    low_count_classes = class_distribution.loc[
        class_distribution["n"] < LOW_COUNT_WARNING_THRESHOLD,
        "next_state",
    ].tolist()
    if low_count_classes:
        print(
            f"WARNING: transition classes with fewer than {LOW_COUNT_WARNING_THRESHOLD} rows: "
            f"{low_count_classes}",
            flush=True,
        )

    print("Fitting multiclass transition model with xgb multi:softprob...", flush=True)
    transition_model = fit_transition_model(
        train_df[transition_feature_cols],
        train_y.to_numpy(dtype=int),
        len(TRANSITION_CLASSES),
    )

    # Score only clinically possible actions for each catheter state.
    in_rows = df[STATE_COL] == "in"
    out_rows = df[STATE_COL] == "out"
    assign_action_scores(df, in_rows, "keep", score_action(df, in_rows, "keep", transition_feature_cols, transition_model))
    assign_action_scores(df, in_rows, "remove", score_action(df, in_rows, "remove", transition_feature_cols, transition_model))
    assign_action_scores(df, out_rows, "out", score_action(df, out_rows, "out", transition_feature_cols, transition_model))
    assign_observed_scores(df)

    eval_test = df[df[SPLIT_COL] == "test"].copy()
    test_proba = eval_test[OBSERVED_TRANSITION_SCORE_COLS].to_numpy(dtype=float)
    transition_summary = test_transition_metrics(eval_test, test_proba, test_y.reset_index(drop=True))

    print(f"Transition log loss: {transition_summary['multiclass_log_loss']}", flush=True)
    for transition_class, auc in transition_summary["risk_set_auc"].items():
        print(f"Risk-set AUC {transition_class}: {auc}", flush=True)

    return df, transition_model, transition_summary


# =============================================================================
# Final panel assembly
# =============================================================================

def save_scored_panel(df):
    # The combined run writes one scored panel and avoids intermediate artifacts.
    final_df = insert_score_columns_before_age(df, ALL_SCORE_COLS)
    final_df.to_csv(FINAL_PANEL, index=False, float_format="%.6f")
    return final_df


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    feature_spec = load_feature_spec()
    df = load_panel()

    df, _, _ = fit_propensity_scores(df, feature_spec)
    df, _, _ = fit_outcome_scores(df, feature_spec)
    save_scored_panel(df)

    print("\n--- SUCCESS NUISANCE MODEL FIT ---", flush=True)
    print(f"Final scored panel saved: {FINAL_PANEL}", flush=True)


if __name__ == "__main__":
    main()
