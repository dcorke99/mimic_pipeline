from pathlib import Path
import json
import joblib
import re
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

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step1")
MODEL_DIR = OUTDIR

INFILE = INDIR / "feature_panel.csv"
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
POST_REMOVE_RISK_PERIODS = 2  # retained for metadata/backwards compatibility; risk sets come from panel columns

TOP_FEATURES_TO_SAVE = 15

SAVE_SHAP = True
SHAP_SAMPLE_N = 2000

CALIBRATION_BINS = 10


def _fit_model(features, target):
    # Fit the configured binary classifier behind a simple pipeline.
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

    pipe.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=int))
    return pipe


def _predict_proba(pipe, features):
    return pipe.predict_proba(features.to_numpy(dtype=float))[:, 1]


def _feature_importance_series(pipe, feature_cols):
    estimator = pipe.named_steps["rf"] if MODEL_TYPE == "rf" else pipe.named_steps["xgb"]
    return pd.Series(
        estimator.feature_importances_,
        index=list(feature_cols)
    ).sort_values(ascending=False)


def _shap_importance_series(
    pipe,
    features,
    sample_n=2000,
):
    # Rank features by mean absolute SHAP value on a sample.
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
        if shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values.mean(axis=2)

    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(shap_mean_abs, index=feature_names).sort_values(ascending=False)


def _top_series_df(
    model_name,
    importance_name,
    values,
    top_n,
):
    top_df = values.head(top_n).reset_index()
    top_df.columns = ["feature", importance_name]
    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
    top_df.insert(0, "model", model_name)
    return top_df


def _add_feature_descriptions(feature_df, covariate_dict):
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
        label = itemid_to_label.get(itemid, "UNKNOWN ITEMID")
        description = f"{label} [{stat}]"
        if str(feature).endswith("__missing"):
            description = f"{description} [missing]"
        return description

    feature_df = feature_df.copy()
    feature_df.insert(3, "description", feature_df["feature"].apply(describe_feature))
    return feature_df


def _save_shap_plots(
    pipe,
    features,
    out_prefix,
    sample_n=2000,
):
    # Save the standard SHAP beeswarm and bar plots.
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


def _clean_eval_frame(df, outcome_col, pred_col):
    # Keep only valid binary labels and bounded predictions.
    eval_df = df[[outcome_col, pred_col]].copy()
    eval_df[outcome_col] = pd.to_numeric(eval_df[outcome_col], errors="coerce")
    eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce").clip(0.0, 1.0)
    eval_df = eval_df.dropna(subset=[outcome_col, pred_col]).copy()
    eval_df[outcome_col] = eval_df[outcome_col].astype(int)
    return eval_df


def _scalar_binary_metrics(df, outcome_col, pred_col):
    # Summarise held-out performance with scalar metrics.
    eval_df = _clean_eval_frame(df, outcome_col, pred_col)

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


def _calibration_table(
    df,
    outcome_col,
    pred_col,
    bins=10,
):
    # Bin predictions into quantiles for a calibration table.
    eval_df = _clean_eval_frame(df, outcome_col, pred_col)
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


def _metrics_by_period(
    df,
    periods_col,
    outcome_col,
    pred_col,
    group_cols=None,
):
    # Track predictive performance across periods in state.
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
        prevalence = events / n if n else np.nan
        auc = float(roc_auc_score(group_df[outcome_col], group_df[pred_col])) if n and group_df[outcome_col].nunique() > 1 else np.nan
        avg_precision = float(average_precision_score(group_df[outcome_col], group_df[pred_col])) if n and events > 0 else np.nan

        rows.append({
            **key_dict,
            "n": n,
            "events": events,
            "prevalence": prevalence,
            "auc": auc,
            "average_precision": avg_precision,
            "pred_mean": float(group_df[pred_col].mean()),
        })

    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True)


def _save_df(df, path):
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


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # Load the saved Step 1 feature specification.
    feature_spec = json.loads(FEATURE_SPEC_FILE.read_text(encoding="utf-8"))
    covariate_dict = pd.read_csv(COVARIATE_DICT_FILE)

    # Load and standardise the feature panel.
    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()

    required_risk_cols = [AT_RISK_CAUTI, AT_RISK_REINS]
    missing_risk_cols = [c for c in required_risk_cols if c not in df.columns]
    if missing_risk_cols:
        raise ValueError(f"Missing required risk-set columns: {missing_risk_cols}")

    df[AT_RISK_CAUTI] = pd.to_numeric(df[AT_RISK_CAUTI], errors="coerce").fillna(0).astype(int)
    df[AT_RISK_REINS] = pd.to_numeric(df[AT_RISK_REINS], errors="coerce").fillna(0).astype(int)

    # Pull the precomputed feature sets for each model.
    feature_list = feature_spec["features"]
    remove_feature_cols = feature_spec["x_cols_remove"]
    cauti_feature_cols = feature_spec["x_cols_cauti"]
    reins_feature_cols = feature_spec["x_cols_reins"]

    print("Step 1 uses precomputed explicit features only.")
    print("Hidden imputer indicator columns created inside Step 1: 0")
    print(f"Loaded Step 1 features: {len(feature_list)}")

    # Add the score columns that will be filled by the fitted models.
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

    # Sort into episode-time order for transition logic.
    df = (
        df.sort_values(EPISODE_KEYS + ["period_end"])
        .reset_index(drop=False)
        .rename(columns={"index": "_orig_index"})
        .copy()
    )

    # Mark which rows belong to each fitting set.
    df["prior_cauti_count"] = (
        df.groupby(EPISODE_KEYS)[Y_CAUTI]
        .cumsum()
        .shift(fill_value=0)
    )

    df_cauti = df[
        (df["prior_cauti_count"] == 0) &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()

    df_in = df[
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    df_out = df[df[AT_RISK_REINS] == 1].copy()

    df_out_fit = df_out[
        ~(
            (df_out[LAST_PERIOD_COL] == 1) &
            (df_out[END_REASON_COL] == "icu_end") &
            (df_out[Y_REINS] == 0)
        )
    ].copy()

    # Fit the removal model and score the observed IN rows.
    remove_train_features = df_in.loc[df_in[SPLIT_COL] == "train", remove_feature_cols]
    remove_train_target = df_in.loc[df_in[SPLIT_COL] == "train", ACTION_COL].astype(int)

    remove_test_features = df_in.loc[df_in[SPLIT_COL] == "test", remove_feature_cols]
    remove_test_target = df_in.loc[df_in[SPLIT_COL] == "test", ACTION_COL].astype(int)

    print(f"Fitting removal model with {MODEL_TYPE}...", flush=True)
    remove_model = _fit_model(remove_train_features, remove_train_target)

    df.loc[df_in.index, "p_remove_obs"] = _predict_proba(remove_model, df_in[remove_feature_cols])

    remove_test_pred = _predict_proba(remove_model, remove_test_features)
    auc_remove = roc_auc_score(remove_test_target, remove_test_pred) if remove_test_target.nunique() > 1 else np.nan

    # Fit the CAUTI model and score observed and counterfactual rows.
    cauti_train_features = df_cauti.loc[df_cauti[SPLIT_COL] == "train", cauti_feature_cols]
    cauti_train_target = df_cauti.loc[df_cauti[SPLIT_COL] == "train", Y_CAUTI].astype(int)

    cauti_test_features = df_cauti.loc[df_cauti[SPLIT_COL] == "test", cauti_feature_cols]
    cauti_test_target = df_cauti.loc[df_cauti[SPLIT_COL] == "test", Y_CAUTI].astype(int)

    print(f"Fitting CAUTI transition model with {MODEL_TYPE}...", flush=True)
    cauti_model = _fit_model(cauti_train_features, cauti_train_target)

    df_out_cauti = df[
        (df[STATE_COL] == "out") &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()
    df.loc[df_out_cauti.index, "p_cauti_if_out"] = _predict_proba(
        cauti_model,
        df_out_cauti[cauti_feature_cols]
    )

    cauti_keep_features = df_in[cauti_feature_cols].copy()
    cauti_keep_features["state_is_out"] = 0

    cauti_remove_features = df_in[cauti_feature_cols].copy()
    cauti_remove_features[PERIODS_COL] = 1
    cauti_remove_features["state_is_out"] = 1

    df.loc[df_in.index, "p_cauti_if_keep"] = _predict_proba(cauti_model, cauti_keep_features)
    df.loc[df_in.index, "p_cauti_if_remove"] = _predict_proba(cauti_model, cauti_remove_features)

    cauti_test_pred = _predict_proba(cauti_model, cauti_test_features)
    auc_cauti = roc_auc_score(cauti_test_target, cauti_test_pred) if cauti_test_target.nunique() > 1 else np.nan

    # Fit the reinsertion model and score observed and counterfactual rows.
    reins_train_features = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", reins_feature_cols]
    reins_train_target = df_out_fit.loc[df_out_fit[SPLIT_COL] == "train", Y_REINS].astype(int)

    reins_test_features = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", reins_feature_cols]
    reins_test_target = df_out_fit.loc[df_out_fit[SPLIT_COL] == "test", Y_REINS].astype(int)

    print(f"Fitting reinsertion model with {MODEL_TYPE}...", flush=True)
    reins_model = _fit_model(reins_train_features, reins_train_target)

    df.loc[df_out.index, "p_reins_if_out"] = _predict_proba(reins_model, df_out[reins_feature_cols])

    reins_test_pred = _predict_proba(reins_model, reins_test_features)
    auc_reins = roc_auc_score(reins_test_target, reins_test_pred) if reins_test_target.nunique() > 1 else np.nan

    reins_remove_features = df_in[reins_feature_cols].copy()
    reins_remove_features[PERIODS_COL] = 1
    df.loc[df_in.index, "p_reins_if_remove"] = _predict_proba(reins_model, reins_remove_features)

    # Build the held-out evaluation frames.
    remove_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[STATE_COL] == "in") &
        (df["prior_cauti_count"] == 0)
    ].copy()

    cauti_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df["prior_cauti_count"] == 0) &
        (df[AT_RISK_CAUTI] == 1)
    ].copy()

    cauti_eval_test["p_cauti_obs_eval"] = np.where(
        cauti_eval_test[STATE_COL] == "in",
        cauti_eval_test["p_cauti_if_keep"],
        cauti_eval_test["p_cauti_if_out"],
    )

    reins_eval_test = df[
        (df[SPLIT_COL] == "test") &
        (df[AT_RISK_REINS] == 1)
    ].copy()

    reins_eval_test = reins_eval_test[
        ~(
            (reins_eval_test[LAST_PERIOD_COL] == 1) &
            (reins_eval_test[END_REASON_COL] == "icu_end") &
            (reins_eval_test[Y_REINS] == 0)
        )
    ].copy()

    # Summarise performance on the held-out rows.
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

    remove_by_period = _metrics_by_period(
        remove_eval_test,
        PERIODS_COL,
        ACTION_COL,
        "p_remove_obs",
    )

    cauti_by_state_period = _metrics_by_period(
        cauti_eval_test,
        PERIODS_COL,
        Y_CAUTI,
        "p_cauti_obs_eval",
        group_cols=[STATE_COL],
    )

    reins_by_period = _metrics_by_period(
        reins_eval_test,
        PERIODS_COL,
        Y_REINS,
        "p_reins_if_out",
    )

    _save_df(remove_cal, OUTDIR / "remove_calibration_test.csv")
    _save_df(cauti_cal, OUTDIR / "cauti_calibration_test.csv")
    _save_df(reins_cal, OUTDIR / "reinsertion_calibration_test.csv")

    _save_df(remove_by_period, OUTDIR / "remove_by_period_test.csv")
    _save_df(cauti_by_state_period, OUTDIR / "cauti_by_state_period_test.csv")
    _save_df(reins_by_period, OUTDIR / "reinsertion_by_period_test.csv")

    # Restore the original row order before saving outputs.
    df = (
        df.sort_values("_orig_index")
        .drop(columns=["_orig_index", "prior_cauti_count"])
    )

    # Save the top raw features for each fitted model.
    model_feature_df = pd.concat(
        [
            _top_series_df("removal", "model_importance", _feature_importance_series(remove_model, remove_feature_cols), TOP_FEATURES_TO_SAVE),
            _top_series_df("cauti", "model_importance", _feature_importance_series(cauti_model, cauti_feature_cols), TOP_FEATURES_TO_SAVE),
            _top_series_df("reinsertion", "model_importance", _feature_importance_series(reins_model, reins_feature_cols), TOP_FEATURES_TO_SAVE),
        ],
        ignore_index=True,
    )
    model_feature_df = _add_feature_descriptions(model_feature_df, covariate_dict)
    _save_df(model_feature_df, OUTDIR / "top_model_features.csv")

    shap_feature_df = pd.concat(
        [
            _top_series_df(
                "removal",
                "shap_mean_abs",
                _shap_importance_series(remove_model, remove_test_features, SHAP_SAMPLE_N),
                TOP_FEATURES_TO_SAVE,
            ),
            _top_series_df(
                "cauti",
                "shap_mean_abs",
                _shap_importance_series(cauti_model, cauti_test_features, SHAP_SAMPLE_N),
                TOP_FEATURES_TO_SAVE,
            ),
            _top_series_df(
                "reinsertion",
                "shap_mean_abs",
                _shap_importance_series(reins_model, reins_test_features, SHAP_SAMPLE_N),
                TOP_FEATURES_TO_SAVE,
            ),
        ],
        ignore_index=True,
    )
    shap_feature_df = _add_feature_descriptions(shap_feature_df, covariate_dict)
    _save_df(shap_feature_df, OUTDIR / "top_shap_features.csv")

    # Save SHAP plots for each model if requested.
    if SAVE_SHAP:
        _save_shap_plots(
            pipe=remove_model,
            features=remove_test_features,
            out_prefix=OUTDIR / f"remove_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )
        _save_shap_plots(
            pipe=cauti_model,
            features=cauti_test_features,
            out_prefix=OUTDIR / f"cauti_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )
        _save_shap_plots(
            pipe=reins_model,
            features=reins_test_features,
            out_prefix=OUTDIR / f"reinsertion_shap_{MODEL_TYPE}",
            sample_n=SHAP_SAMPLE_N,
        )

    # Move score columns just before age in the scored panel export.
    score_col_names = [
        "p_remove_obs",
        "p_cauti_if_keep",
        "p_cauti_if_remove",
        "p_cauti_if_out",
        "p_reins_if_remove",
        "p_reins_if_out",
    ]
    age_col_idx = df.columns.get_loc("age")
    ordered_cols = [
        col for col in df.columns
        if col not in score_col_names
    ]
    ordered_cols[age_col_idx:age_col_idx] = score_col_names
    df = df[ordered_cols].copy()

    # Save the scored panel and fitted model bundle.
    out_scored = OUTDIR / "step1_scored_panel.csv"
    df.to_csv(out_scored, index=False, float_format="%.6f")

    joblib.dump(
        {
            "model_type": MODEL_TYPE,
            "remove_model": remove_model,
            "cauti_model": cauti_model,
            "reins_model": reins_model,
            "features": feature_list,
            "x_cols_remove": remove_feature_cols,
            "x_cols_cauti": cauti_feature_cols,
            "x_cols_reins": reins_feature_cols,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "split_col": SPLIT_COL,
            "period_hours": feature_spec.get("period_hours"),
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
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
        PERIODS_COL,
        TIME_COL,
        SPLIT_COL,
        ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        AT_RISK_CAUTI,
        AT_RISK_REINS,
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
        "period_hours": feature_spec.get("period_hours"),
        "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
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
            "explicit_feature_count": int(len(feature_list)),
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
            "at_risk_cauti": int((df[AT_RISK_CAUTI] == 1).sum()),
            "at_risk_reinsertion": int((df[AT_RISK_REINS] == 1).sum()),
            "cauti_fit": int(len(df_cauti)),
            "out_fit": int(len(df_out_fit)),
        },
        "artifacts": {
            "scored_panel": str(out_scored),
            "remove_calibration_test_csv": str(OUTDIR / "remove_calibration_test.csv"),
            "cauti_calibration_test_csv": str(OUTDIR / "cauti_calibration_test.csv"),
            "reinsertion_calibration_test_csv": str(OUTDIR / "reinsertion_calibration_test.csv"),
            "remove_by_period_test_csv": str(OUTDIR / "remove_by_period_test.csv"),
            "cauti_by_state_period_test_csv": str(OUTDIR / "cauti_by_state_period_test.csv"),
            "reinsertion_by_period_test_csv": str(OUTDIR / "reinsertion_by_period_test.csv"),
            "top_model_features_csv": str(OUTDIR / "top_model_features.csv"),
            "top_shap_features_csv": str(OUTDIR / "top_shap_features.csv"),
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
