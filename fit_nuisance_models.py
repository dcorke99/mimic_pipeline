# Fit propensity and outcome nuisance models for policy evaluation
import argparse
from pathlib import Path
import re

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import policy_eval_common as pec
from panel_run_config import add_panel_argument, resolve_panel_run


# Configuration

SEED = 42
MODEL_TYPES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
)
MODEL_TYPE = "xgboost"
MODEL_ALIASES = {
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "mlp": "mlp",
}

REPO_ROOT = Path(__file__).resolve().parent
INDIR = REPO_ROOT / "data"
NUISANCE_ROOT = REPO_ROOT / "artefacts" / "nuisance_models"
MODEL_OUTPUT_NAMES = {
    model_type: model_type for model_type in MODEL_TYPES
}
MODEL_OUTPUT_NAME = MODEL_OUTPUT_NAMES.get(
    MODEL_TYPE,
    re.sub(r"[^a-z0-9]+", "_", MODEL_TYPE.lower()).strip("_") or "model",
)
OUTDIR = NUISANCE_ROOT / MODEL_OUTPUT_NAME

INFILE = INDIR / "modelling_panel.csv"
COVARIATE_DICT_FILE = INDIR / "covariate_dictionary.csv"
NUISANCE_PREDICTIONS_FILE = OUTDIR / "nuisance_predictions.csv"
PERFORMANCE_METRICS_FILE = OUTDIR / "performance_metrics.csv"
CROSSFIT_ROW_ASSIGNMENTS_FILE = OUTDIR / "crossfit_row_assignments.csv"
MODEL_COMPARISON_FILE = NUISANCE_ROOT / "nuisance_model_comparison.csv"
CONSTANT_FEATURES_FILE = OUTDIR / "constant_features_by_fold.csv"

ID_COL = "subject_id"
TIME_COL = "episode_index"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"

ACTION_COL = "removed_in_period"
Y_CAUTI = "cauti_in_period"
Y_REINS = "reinsertion_in_period"
Y_DEATH = "death_in_period"
Y_ICU_EXIT_ALIVE = "icu_exit_alive_in_period"
OBSERVED_ACTION_COL = "observed_action"
END_REASON_COL = "episode_end_reason"
AT_RISK_CAUTI = "at_risk_cauti"
AT_RISK_REINS = "at_risk_reinsertion"

Y_NO_EVENT_IN = "_target_no_event_in"
Y_NO_EVENT_OUT = "_target_no_event_out"

NO_EVENT_DEFINITION = {
    "in": (
        "No CAUTI, death, or ICU exit alive in the next outcome window; this does not "
        "imply continued catheter-in state when removed_in_period=1."
    ),
    "out": (
        "No reinsertion, CAUTI, death, or ICU exit alive in the next outcome window."
    ),
}

POST_REMOVE_RISK_PERIODS = 2
PERIOD_HOURS = 24
TOP_FEATURES_TO_SAVE = 15
CALIBRATION_BINS = 10
N_CROSSFIT_FOLDS = 5
CROSSFIT_FOLD_COL = "_crossfit_fold"
LEARNING_CURVE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
FALLBACK_PRIOR_EVENTS = 1.0
FALLBACK_PRIOR_NON_EVENTS = 1.0

LEARNER_CONFIGURATIONS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": SEED,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
        "n_jobs": 1,
        "random_state": SEED,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "n_jobs": 1,
    },
    "lightgbm": {
        "objective": "binary",
        "n_estimators": 300,
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": 1,
    },
    "mlp": {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate_init": 0.001,
        "max_iter": 200,
        "shuffle": True,
        "early_stopping": False,
        "random_state": SEED,
    },
}

LEARNER_PREPROCESSING = {
    "logistic_regression": {
        "preprocessing_type": "median_imputation_and_standardisation",
        "imputation_used": True,
        "scaling_used": True,
        "native_missing_handling": False,
    },
    "random_forest": {
        "preprocessing_type": "median_imputation",
        "imputation_used": True,
        "scaling_used": False,
        "native_missing_handling": False,
    },
    "xgboost": {
        "preprocessing_type": "native_missing_values",
        "imputation_used": False,
        "scaling_used": False,
        "native_missing_handling": True,
    },
    "lightgbm": {
        "preprocessing_type": "native_missing_values",
        "imputation_used": False,
        "scaling_used": False,
        "native_missing_handling": True,
    },
    "mlp": {
        "preprocessing_type": "median_imputation_and_standardisation",
        "imputation_used": True,
        "scaling_used": True,
        "native_missing_handling": False,
    },
}

PROPENSITY_SCORE_COL = "p_remove_obs"
KEEP_PROPENSITY_SCORE_COL = "p_keep_obs"

IN_OUTCOMES = {
    "cauti": Y_CAUTI,
    "death": Y_DEATH,
    "icu_exit_alive": Y_ICU_EXIT_ALIVE,
    "no_event": Y_NO_EVENT_IN,
}
OUT_OUTCOMES = {
    "reinsertion": Y_REINS,
    "cauti": Y_CAUTI,
    "death": Y_DEATH,
    "icu_exit_alive": Y_ICU_EXIT_ALIVE,
    "no_event": Y_NO_EVENT_OUT,
}

ALL_SCORE_COLS = [
    PROPENSITY_SCORE_COL,
    KEEP_PROPENSITY_SCORE_COL,
    "p_cauti_if_keep",
    "p_cauti_if_remove",
    "p_cauti_if_out",
    "p_reinsertion_if_out",
    "p_death_if_keep",
    "p_death_if_remove",
    "p_death_if_out",
    "p_icu_exit_alive_if_keep",
    "p_icu_exit_alive_if_remove",
    "p_icu_exit_alive_if_out",
    "p_no_event_if_keep",
    "p_no_event_if_remove",
    "p_no_event_if_out",
]

# Support functions


def normalise_model_type(model_type):
    # Resolve supported learner aliases to their canonical names
    key = re.sub(r"[^a-z0-9]+", "_", str(model_type).lower()).strip("_")
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown MODEL_TYPE: {model_type}. Expected one of {MODEL_TYPES}"
        )
    return MODEL_ALIASES[key]


def learner_provenance():
    # Describe the active learner and its fold-fitted preprocessing
    return {
        "model_type": MODEL_TYPE,
        "model_output_name": MODEL_OUTPUT_NAME,
        **LEARNER_PREPROCESSING[MODEL_TYPE],
        "random_seed": SEED,
        "learner_configuration": LEARNER_CONFIGURATIONS[MODEL_TYPE].copy(),
    }


def configure_model_run(model_type):
    # Set the active learner and its model-specific output paths
    global MODEL_TYPE, MODEL_OUTPUT_NAME, OUTDIR
    global NUISANCE_PREDICTIONS_FILE, PERFORMANCE_METRICS_FILE
    global CROSSFIT_ROW_ASSIGNMENTS_FILE
    global CONSTANT_FEATURES_FILE
    MODEL_TYPE = normalise_model_type(model_type)
    MODEL_OUTPUT_NAME = MODEL_OUTPUT_NAMES.get(
        MODEL_TYPE,
        re.sub(r"[^a-z0-9]+", "_", MODEL_TYPE.lower()).strip("_") or "model",
    )
    OUTDIR = NUISANCE_ROOT / MODEL_OUTPUT_NAME
    NUISANCE_PREDICTIONS_FILE = OUTDIR / "nuisance_predictions.csv"
    PERFORMANCE_METRICS_FILE = OUTDIR / "performance_metrics.csv"
    CROSSFIT_ROW_ASSIGNMENTS_FILE = OUTDIR / "crossfit_row_assignments.csv"
    CONSTANT_FEATURES_FILE = OUTDIR / "constant_features_by_fold.csv"


def configure_panel_run(panel_name):
    # Route the source panel and every fitted-model artefact together
    global INFILE, NUISANCE_ROOT, MODEL_COMPARISON_FILE
    paths = resolve_panel_run(REPO_ROOT, panel_name)
    INFILE = paths.panel_path
    NUISANCE_ROOT = paths.artefact_root / "nuisance_models"
    MODEL_COMPARISON_FILE = NUISANCE_ROOT / "nuisance_model_comparison.csv"
    configure_model_run(MODEL_TYPE)
    return paths


def binary_values(series):
    # Convert values
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def load_panel():
    # Read the full modelling panel
    df = pd.read_csv(INFILE, low_memory=False)

    # Standardise identifiers and categorical values
    df.columns = df.columns.str.strip()

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    df[END_REASON_COL] = (
        df[END_REASON_COL].astype("string").str.strip().str.lower()
    )

    unknown_states = sorted(set(df[STATE_COL].dropna()) - {"in", "out"})
    if unknown_states:
        raise ValueError(f"Unexpected catheter states: {unknown_states}")

    # Convert values
    for col in [
        ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        Y_DEATH,
        Y_ICU_EXIT_ALIVE,
        AT_RISK_CAUTI,
        AT_RISK_REINS,
    ]:
        # Convert values
        df[col] = binary_values(df[col])

    # Consolidate internal column blocks
    df = df.copy()
    return df


def add_grouped_crossfit_folds(df, n_splits=N_CROSSFIT_FOLDS):
    # Add grouped crossfit folds
    n_groups = int(df[ID_COL].nunique())
    if n_groups < n_splits:
        raise ValueError(
            f"Grouped cross-fitting needs at least {n_splits} patients; found {n_groups}"
        )

    df[CROSSFIT_FOLD_COL] = -1
    splitter = GroupKFold(n_splits=n_splits)
    groups = df[ID_COL].to_numpy()
    placeholder = np.zeros(len(df), dtype=np.uint8)
    for fold, (_, held_out_index) in enumerate(
        splitter.split(placeholder, groups=groups)
    ):
        df.iloc[
            held_out_index,
            df.columns.get_loc(CROSSFIT_FOLD_COL),
        ] = fold

    if df[CROSSFIT_FOLD_COL].lt(0).any():
        raise ValueError("Cross-fitting failed to assign every row to a fold")
    folds_per_patient = df.groupby(ID_COL)[CROSSFIT_FOLD_COL].nunique()
    if folds_per_patient.ne(1).any():
        raise ValueError("A patient was assigned to more than one cross-fit fold")
    return df


def crossfit_fold_summary(df):
    # Summarise cross-fit folds
    return (
        df.groupby(CROSSFIT_FOLD_COL, observed=False)
        .agg(rows=(ID_COL, "size"), patients=(ID_COL, "nunique"))
        .reset_index()
        .rename(columns={CROSSFIT_FOLD_COL: "fold"})
    )


def crossfit_row_assignments(df):
    # Build cross-fit row assignments
    columns = [
        ID_COL,
        "stay_id",
        "inserted",
        TIME_COL,
        STATE_COL,
        CROSSFIT_FOLD_COL,
    ]
    return df[columns].copy()


def fit_binary_model(features, target, model_name):
    # Fit binary model
    # Convert values
    target = binary_values(target)
    if features.empty:
        raise ValueError(f"Cannot fit {model_name}: training risk set is empty")
    if target.nunique() < 2:
        raise ValueError(
            f"Cannot fit {model_name}: training target contains only class "
            f"{int(target.iloc[0])}"
        )

    learner_configuration = LEARNER_CONFIGURATIONS[MODEL_TYPE]
    if MODEL_TYPE == "logistic_regression":
        estimator = LogisticRegression(**learner_configuration)
    elif MODEL_TYPE == "random_forest":
        estimator = RandomForestClassifier(**learner_configuration)
    elif MODEL_TYPE == "xgboost":
        estimator = XGBClassifier(**learner_configuration)
    elif MODEL_TYPE == "lightgbm":
        estimator = LGBMClassifier(**learner_configuration)
    elif MODEL_TYPE == "mlp":
        estimator = MLPClassifier(**learner_configuration)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    steps = []
    if LEARNER_PREPROCESSING[MODEL_TYPE]["imputation_used"]:
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=False)))
    if LEARNER_PREPROCESSING[MODEL_TYPE]["scaling_used"]:
        steps.append(("standardiser", StandardScaler()))
    steps.append((MODEL_TYPE, estimator))
    pipe = Pipeline(steps)
    pipe.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=int))
    # Training uses CUDA, but downstream predictions use NumPy arrays in CPU
    # memory. Match the fitted booster to those arrays to avoid XGBoost's
    # cross-device DMatrix fallback during prediction.
    if MODEL_TYPE == "xgboost":
        pipe.named_steps["xgboost"].set_params(device="cpu")
    return pipe


def predict_binary_proba(pipe, features):
    # Predict binary probabilities
    feature_values = features.to_numpy(dtype=float)
    if MODEL_TYPE == "lightgbm":
        feature_values = pd.DataFrame(
            feature_values,
            columns=pipe.named_steps["lightgbm"].feature_names_in_,
        )
    return pipe.predict_proba(feature_values)[:, 1]


def split_constant_features(features):
    # Identify unusable features from training rows only
    retained_feature_cols = []
    constant_features = []
    for feature in features.columns:
        non_missing = features[feature].dropna()
        if non_missing.empty:
            reason = "all_missing"
        elif non_missing.nunique(dropna=True) == 1:
            value = non_missing.iloc[0]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = np.nan
            if numeric_value == 0:
                reason = "constant_zero"
            elif numeric_value == 1:
                reason = "constant_one"
            else:
                reason = "single_unique_value"
        else:
            retained_feature_cols.append(feature)
            continue
        constant_features.append({"feature": feature, "reason": reason})
    return retained_feature_cols, constant_features


def fit_crossfit_fold_model(
    features,
    target,
    model_name,
    fold,
):
    # Fit crossfit fold model
    if len(features) == 0:
        raise ValueError(f"Cannot fit {model_name} fold {fold}: training risk set is empty")
    # Convert values
    target = binary_values(target)
    events = int(target.sum())
    non_events = int(len(target) - events)
    retained_feature_cols, constant_features = split_constant_features(features)
    fold_metadata = {
        "fold": int(fold),
        "training_n": int(len(target)),
        "training_events": events,
        "training_non_events": non_events,
        "candidate_feature_cols": list(features.columns),
        "retained_feature_cols": retained_feature_cols,
        "constant_features": constant_features,
    }
    fallback_reason = None
    if target.nunique() < 2:
        fallback_reason = "single_target_class"
    elif not retained_feature_cols:
        fallback_reason = "no_usable_features"
    if fallback_reason is not None:
        # Beta(1, 1) / Laplace smoothing avoids exact zero or one while using
        # only this fold's training rows. Held-out outcomes are never used
        fallback_probability = float(
            (events + FALLBACK_PRIOR_EVENTS)
            / (
                len(target)
                + FALLBACK_PRIOR_EVENTS
                + FALLBACK_PRIOR_NON_EVENTS
            )
        )
        if not np.isfinite(fallback_probability) or not 0.0 <= fallback_probability <= 1.0:
            raise ValueError(
                f"Invalid fallback probability for {model_name} fold {fold}: "
                f"{fallback_probability}"
            )
        return {
            **fold_metadata,
            "model": None,
            "fallback": True,
            "fallback_reason": fallback_reason,
            "fallback_probability": fallback_probability,
            "fallback_probability_source": "fold_training_rows_only",
            "fallback_smoothing": "beta_binomial",
            "fallback_prior_events": FALLBACK_PRIOR_EVENTS,
            "fallback_prior_non_events": FALLBACK_PRIOR_NON_EVENTS,
        }

    # Fit binary model
    model = fit_binary_model(
        features.loc[:, retained_feature_cols],
        target,
        f"{model_name}_fold_{fold}",
    )
    return {
        **fold_metadata,
        "model": model,
        "fallback": False,
        "fallback_reason": None,
        "fallback_probability": None,
        "fallback_probability_source": None,
        "fallback_smoothing": None,
        "fallback_prior_events": None,
        "fallback_prior_non_events": None,
    }


def predict_crossfit_fold(fold_model, features):
    # Predict crossfit fold
    if fold_model["fallback"]:
        return np.full(
            len(features),
            fold_model["fallback_probability"],
            dtype=float,
        )
    # Predict binary probabilities
    retained_feature_cols = fold_model["retained_feature_cols"]
    missing_features = set(retained_feature_cols) - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Held-out data are missing retained features: {sorted(missing_features)}"
        )
    return predict_binary_proba(
        fold_model["model"],
        features.loc[:, retained_feature_cols],
    )


def feature_importance_series(pipe, feature_cols):
    # Build importance series
    estimator = pipe.named_steps[MODEL_TYPE]
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        return pd.Series(dtype=float)
    return pd.Series(
        values,
        index=list(feature_cols),
    ).sort_values(ascending=False)


def mean_feature_importance_series(fold_models):
    # Calculate feature importance series
    # Build importance series
    importances = [
        feature_importance_series(
            fold_model["model"], fold_model["retained_feature_cols"]
        )
        for fold_model in fold_models
        if not fold_model["fallback"] and fold_model["model"] is not None
    ]
    importances = [importance for importance in importances if not importance.empty]
    if not importances:
        return pd.Series(dtype=float)
    return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def outcome_fallback_counts(outcome_models):
    # Summarise constant-probability fold counts
    counts = {}
    for state, models in outcome_models.items():
        for outcome, payload in models.items():
            counts[f"{state}_{outcome}"] = sum(
                int(fold_model["fallback"])
                for fold_model in payload["fold_models"]
            )
    return counts


def constant_features_by_fold(propensity_fold_models, in_models, out_models):
    # Build the training-fold-only constant-feature audit
    columns = [
        "model_type",
        "model_group",
        "outcome",
        "risk_set",
        "fold",
        "feature",
        "reason",
    ]
    rows = []

    def add_rows(model_group, outcome, risk_set, fold_models):
        for fold_model in fold_models:
            for constant_feature in fold_model["constant_features"]:
                rows.append({
                    "model_type": MODEL_TYPE,
                    "model_group": model_group,
                    "outcome": outcome,
                    "risk_set": risk_set,
                    "fold": fold_model["fold"],
                    **constant_feature,
                })

    add_rows("propensity", "removal", "all IN rows", propensity_fold_models)
    for outcome, payload in in_models.items():
        add_rows("in_outcome", outcome, payload["risk_set"], payload["fold_models"])
    for outcome, payload in out_models.items():
        add_rows("out_outcome", outcome, payload["risk_set"], payload["fold_models"])
    return pd.DataFrame(rows, columns=columns)


def top_series_df(model_name, values, top_n):
    # Build series data frame
    if values.empty:
        return pd.DataFrame([{
            "model": model_name,
            "rank": pd.NA,
            "feature": pd.NA,
            "model_importance": np.nan,
            "importance_available": False,
            "importance_reason": (
                "native_feature_importance_unavailable"
                if MODEL_TYPE == "mlp"
                else "no_fitted_fold_importance_available"
            ),
        }])
    top_df = values.head(top_n).reset_index()
    top_df.columns = ["feature", "model_importance"]
    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
    top_df.insert(0, "model", model_name)
    top_df["importance_available"] = True
    top_df["importance_reason"] = pd.NA
    return top_df


def add_feature_descriptions(feature_df, covariate_dict):
    # Add feature descriptions
    itemid_to_label = {}
    for row in covariate_dict.itertuples(index=False):
        if hasattr(row, "itemid") and pd.notna(row.itemid):
            itemid_to_label[int(row.itemid)] = str(row.label)

    pattern = re.compile(r"^itemid_(\d+)__(.+?)(?:__missing)?$")

    def describe_feature(feature):
        # Describe feature
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


def clean_eval_frame(df, outcome_col, pred_col):
    # Clean eval frame
    eval_df = df[[outcome_col, pred_col]].copy()
    eval_df[outcome_col] = pd.to_numeric(eval_df[outcome_col], errors="coerce")
    eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce")
    eval_df = eval_df.dropna(subset=[outcome_col, pred_col]).copy()
    eval_df[outcome_col] = eval_df[outcome_col].astype(int)
    return eval_df


def scalar_binary_metrics(df, outcome_col, pred_col):
    # Calculate binary metrics
    # Clean eval frame
    eval_df = clean_eval_frame(df, outcome_col, pred_col)
    n = int(len(eval_df))
    events = int(eval_df[outcome_col].sum())
    prevalence = events / n if n else np.nan
    has_both_classes = n > 0 and eval_df[outcome_col].nunique() > 1
    auc = (
        float(roc_auc_score(eval_df[outcome_col], eval_df[pred_col]))
        if has_both_classes
        else np.nan
    )
    average_precision = (
        float(average_precision_score(eval_df[outcome_col], eval_df[pred_col]))
        if events > 0
        else np.nan
    )
    brier = (
        float(brier_score_loss(eval_df[outcome_col], eval_df[pred_col]))
        if n
        else np.nan
    )
    calibration_intercept = np.nan
    calibration_slope = np.nan
    if has_both_classes:
        clipped_predictions = np.clip(
            eval_df[pred_col].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
        )
        prediction_logit = np.log(clipped_predictions / (1.0 - clipped_predictions))
        if not np.isclose(np.ptp(prediction_logit), 0.0, atol=1e-12, rtol=0.0):
            try:
                calibration_model = LogisticRegression(
                    C=np.inf,
                    solver="lbfgs",
                    max_iter=1000,
                )
                calibration_model.fit(
                    prediction_logit.reshape(-1, 1),
                    eval_df[outcome_col],
                )
                calibration_intercept = float(calibration_model.intercept_[0])
                calibration_slope = float(calibration_model.coef_[0, 0])
            except (TypeError, ValueError):
                calibration_intercept = np.nan
                calibration_slope = np.nan
    return {
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "auc": auc,
        "average_precision": average_precision,
        "brier": brier,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def calibration_table(df, outcome_col, pred_col, bins=10):
    # Build table
    # Clean eval frame
    eval_df = clean_eval_frame(df, outcome_col, pred_col)
    columns = [
        "bin",
        "n",
        "events",
        "prevalence",
        "pred_min",
        "pred_mean",
        "pred_max",
        "obs_rate",
    ]
    if eval_df.empty:
        return pd.DataFrame(columns=columns)

    unique_predictions = int(eval_df[pred_col].nunique())
    if unique_predictions <= 1:
        eval_df["bin"] = 0
    else:
        q = min(bins, unique_predictions)
        eval_df["bin"] = pd.qcut(
            eval_df[pred_col], q=q, labels=False, duplicates="drop"
        )

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
    return calib_df[columns]


def prepare_outcome_targets(df):
    # Prepare outcome targets
    df[OBSERVED_ACTION_COL] = np.where(
        df[STATE_COL].eq("out"),
        "out",
        np.where(df[ACTION_COL].eq(1), "remove", "keep"),
    )
    # These are interval outcomes, not transition-state labels. In particular,
    # IN no_event under removed_in_period=1 does not mean the catheter stayed IN
    # Removal is deliberately absent because it is the action
    df[Y_NO_EVENT_IN] = (
        df[Y_CAUTI].eq(0)
        & df[Y_DEATH].eq(0)
        & df[Y_ICU_EXIT_ALIVE].eq(0)
    ).astype(int)
    df[Y_NO_EVENT_OUT] = (
        df[Y_REINS].eq(0)
        & df[Y_CAUTI].eq(0)
        & df[Y_DEATH].eq(0)
        & df[Y_ICU_EXIT_ALIVE].eq(0)
    ).astype(int)
    return df


def outcome_risk_mask(df, state, outcome):
    # Summarise risk mask
    mask = df[STATE_COL].eq(state)
    if outcome == "cauti":
        mask &= df[AT_RISK_CAUTI].eq(1)
    elif outcome == "reinsertion":
        mask &= df[AT_RISK_REINS].eq(1)
    return mask


def assert_valid_predictions(df, row_mask, columns, context):
    # Check valid predictions
    values = df.loc[row_mask, columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        missing = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Missing required predictions for {context}: {missing}")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite predictions found for {context}")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"Predictions outside [0, 1] found for {context}")


def validate_exported_probabilities(df):
    # Validate exported probabilities
    numeric_scores = df[ALL_SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    invalid_coercions = df[ALL_SCORE_COLS].notna() & numeric_scores.isna()
    if invalid_coercions.any().any():
        invalid = invalid_coercions.columns[invalid_coercions.any()].tolist()
        raise ValueError(f"Non-numeric exported probabilities found: {invalid}")

    present_values = numeric_scores.to_numpy(dtype=float)
    present_values = present_values[~np.isnan(present_values)]
    if not np.isfinite(present_values).all():
        raise ValueError("Non-finite exported probabilities found")
    if ((present_values < 0.0) | (present_values > 1.0)).any():
        raise ValueError("Exported probabilities outside [0, 1] found")

    in_rows = df[STATE_COL].eq("in")
    out_rows = df[STATE_COL].eq("out")
    # Check valid predictions
    assert_valid_predictions(
        df,
        in_rows,
        ["p_cauti_if_keep", "p_cauti_if_remove"],
        "eligible IN CAUTI rows",
    )
    # Check valid predictions
    assert_valid_predictions(
        df,
        out_rows,
        ["p_reinsertion_if_out"],
        "eligible OUT reinsertion rows",
    )

    out_cauti_risk = out_rows & df[AT_RISK_CAUTI].eq(1)
    out_cauti_non_risk = out_rows & df[AT_RISK_CAUTI].eq(0)
    # Check valid predictions
    assert_valid_predictions(
        df,
        out_cauti_risk,
        ["p_cauti_if_out"],
        "at-risk OUT CAUTI rows",
    )
    non_risk_cauti = pd.to_numeric(
        df.loc[out_cauti_non_risk, "p_cauti_if_out"], errors="coerce"
    )
    if non_risk_cauti.isna().any() or not np.allclose(
        non_risk_cauti.to_numpy(dtype=float), 0.0, atol=0.0
    ):
        raise ValueError("OUT rows outside the CAUTI risk set must have p_cauti_if_out = 0")


def model_summary_row(
    model_group,
    outcome,
    target_col,
    feature_cols,
    modelling_df,
    eval_df,
    pred_col,
    risk_set,
):
    # Build summary row
    # Calculate binary metrics
    metrics = scalar_binary_metrics(eval_df, target_col, pred_col)
    # Convert values
    modelling_target = binary_values(modelling_df[target_col])
    return {
        "model_group": model_group,
        "model_type": MODEL_TYPE,
        "outcome": outcome,
        "target_col": target_col,
        "risk_set": risk_set,
        "evaluation": "grouped_cross_fit",
        "crossfit_folds": N_CROSSFIT_FOLDS,
        "crossfit_group_col": ID_COL,
        "feature_count": int(len(feature_cols)),
        "modelling_n": int(len(modelling_df)),
        "modelling_events": int(modelling_target.sum()),
        "modelling_prevalence": float(modelling_target.mean()) if len(modelling_target) else np.nan,
        **metrics,
        # Primary metrics already use the outcome-specific risk set. These named
        # fields are retained for policy-evaluation diagnostics
        "risk_set_auc": metrics["auc"],
        "risk_set_brier": metrics["brier"],
    }


def labelled_calibration(eval_df, target_col, pred_col, model_group, outcome, risk_set):
    # Build calibration
    # Build table
    table = calibration_table(eval_df, target_col, pred_col, bins=CALIBRATION_BINS)
    table.insert(0, "risk_set", risk_set)
    table.insert(0, "outcome", outcome)
    table.insert(0, "model_group", model_group)
    return table


def propensity_summary_rows(df, eligible_mask, feature_cols, summary):
    # Build propensity summary rows
    eligible_patients = df.loc[eligible_mask, ID_COL].nunique()
    p_remove = pd.to_numeric(
        df.loc[eligible_mask, PROPENSITY_SCORE_COL], errors="coerce"
    ).dropna()
    return pd.DataFrame([
        {"metric": "model_type", "value": MODEL_TYPE},
        {"metric": "evaluation", "value": "grouped_cross_fit"},
        {"metric": "crossfit_folds", "value": N_CROSSFIT_FOLDS},
        {"metric": "crossfit_group_col", "value": ID_COL},
        {"metric": "period_hours", "value": PERIOD_HOURS},
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "all_patients", "value": int(df[ID_COL].nunique())},
        {"metric": "decision_eligible_rows", "value": int(eligible_mask.sum())},
        {"metric": "decision_eligible_patients", "value": int(eligible_patients)},
        {"metric": "explicit_feature_count", "value": int(len(feature_cols))},
        {"metric": "n", "value": summary["n"]},
        {"metric": "events", "value": summary["events"]},
        {"metric": "prevalence", "value": summary["prevalence"]},
        {"metric": "auc", "value": summary["auc"]},
        {"metric": "average_precision", "value": summary["average_precision"]},
        {"metric": "brier", "value": summary["brier"]},
        {"metric": "calibration_intercept", "value": summary["calibration_intercept"]},
        {"metric": "calibration_slope", "value": summary["calibration_slope"]},
        {"metric": "p_remove_min", "value": p_remove.min()},
        {"metric": "p_remove_p01", "value": p_remove.quantile(0.01)},
        {"metric": "p_remove_p05", "value": p_remove.quantile(0.05)},
        {"metric": "p_remove_p50", "value": p_remove.quantile(0.50)},
        {"metric": "p_remove_p95", "value": p_remove.quantile(0.95)},
        {"metric": "p_remove_p99", "value": p_remove.quantile(0.99)},
        {"metric": "p_remove_max", "value": p_remove.max()},
        {"metric": "pct_p_remove_below_0_01", "value": 100.0 * p_remove.lt(0.01).mean()},
        {"metric": "pct_p_remove_below_0_05", "value": 100.0 * p_remove.lt(0.05).mean()},
        {"metric": "pct_p_remove_above_0_95", "value": 100.0 * p_remove.gt(0.95).mean()},
        {"metric": "pct_p_remove_above_0_99", "value": 100.0 * p_remove.gt(0.99).mean()},
    ])


def performance_metrics_rows(propensity_summary, in_summary, out_summary):
    # Build performance metrics rows
    rows = []
    for metric, value in propensity_summary.items():
        rows.append({
            "model": "propensity",
            "model_type": MODEL_TYPE,
            "split": "grouped_cross_fit",
            "outcome": "removal",
            "metric": metric,
            "value": value,
        })

    metric_names = [
        "n",
        "events",
        "prevalence",
        "auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "risk_set_auc",
        "risk_set_brier",
    ]
    for summary_df in [in_summary, out_summary]:
        for row in summary_df.itertuples(index=False):
            for metric in metric_names:
                rows.append({
                    "model": row.model_group,
                    "model_type": row.model_type,
                    "split": "grouped_cross_fit",
                    "outcome": row.outcome,
                    "metric": metric,
                    "value": getattr(row, metric),
                })
    return pd.DataFrame(rows)


def fold_performance_metrics(df):
    # Build held-out performance metrics for each cross-fit fold
    rows = []

    def add_rows(model_group, outcome, risk_set, eval_df, target_col, pred_col):
        for fold in range(N_CROSSFIT_FOLDS):
            fold_df = eval_df.loc[eval_df[CROSSFIT_FOLD_COL].eq(fold)]
            rows.append({
                "model_group": model_group,
                "model_type": MODEL_TYPE,
                "outcome": outcome,
                "risk_set": risk_set,
                "fold": fold,
                **scalar_binary_metrics(fold_df, target_col, pred_col),
            })

    propensity_mask = df[STATE_COL].eq("in")
    add_rows(
        "propensity",
        "removal",
        "all IN rows",
        df.loc[propensity_mask],
        ACTION_COL,
        PROPENSITY_SCORE_COL,
    )
    for outcome, target_col in IN_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "in", outcome)
        keep_col = f"p_{outcome}_if_keep"
        remove_col = f"p_{outcome}_if_remove"
        pred_col = f"_p_{outcome}_observed_in"
        eval_df = df.loc[
            risk_mask,
            [CROSSFIT_FOLD_COL, target_col, ACTION_COL, keep_col, remove_col],
        ].copy()
        eval_df[pred_col] = np.where(
            eval_df[ACTION_COL].eq(1), eval_df[remove_col], eval_df[keep_col]
        )
        risk_set = "IN and at_risk_cauti == 1" if outcome == "cauti" else "all IN rows"
        add_rows("in_outcome", outcome, risk_set, eval_df, target_col, pred_col)

    for outcome, target_col in OUT_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "out", outcome)
        pred_col = f"p_{outcome}_if_out"
        if outcome == "cauti":
            risk_set = "OUT and at_risk_cauti == 1 (48-hour attribution window)"
        elif outcome == "reinsertion":
            risk_set = "OUT and at_risk_reinsertion == 1"
        else:
            risk_set = "all OUT rows"
        add_rows(
            "out_outcome",
            outcome,
            risk_set,
            df.loc[risk_mask],
            target_col,
            pred_col,
        )
    return pd.DataFrame(rows)


def nuisance_subgroup_diagnostics(df):
    # Build subgroup performance and calibration from production predictions
    performance_rows = []
    calibration_tables = []

    def add_task(
        model_group,
        outcome,
        risk_set,
        risk_mask,
        target_col,
        prediction,
        period_subgroup,
    ):
        columns = [ID_COL, target_col, "age", PERIODS_COL]
        if "sex_M" in df.columns:
            columns.append("sex_M")
        eval_df = df.loc[risk_mask, columns].copy()
        pred_col = "_subgroup_prediction"
        eval_df[pred_col] = prediction.loc[risk_mask]

        subgroups = []
        if "sex_M" in eval_df.columns:
            sex = pd.to_numeric(eval_df["sex_M"], errors="coerce")
            sex_levels = pd.Series(
                np.select(
                    [sex.eq(1), sex.eq(0)],
                    ["Male", "Female"],
                    default="Unknown",
                ),
                index=eval_df.index,
            )
            subgroups.append(("sex", sex_levels, ["Male", "Female", "Unknown"]))

        age = pd.to_numeric(eval_df["age"], errors="coerce")
        age_levels = pd.Series(
            np.select(
                [
                    age.lt(50),
                    age.ge(50) & age.lt(65),
                    age.ge(65) & age.lt(80),
                    age.ge(80),
                ],
                ["<50", "50-64", "65-79", "80+"],
                default="Missing",
            ),
            index=eval_df.index,
        )
        subgroups.append(
            ("age_band", age_levels, ["<50", "50-64", "65-79", "80+", "Missing"])
        )

        period = pd.to_numeric(eval_df[PERIODS_COL], errors="coerce")
        period_levels = pd.Series(
            np.select(
                [
                    period.eq(0),
                    period.eq(1),
                    period.eq(2),
                    period.eq(3),
                    period.eq(4),
                    period.ge(5),
                    period.isna(),
                ],
                ["0", "1", "2", "3", "4", "5+", "Missing"],
                default="Unknown",
            ),
            index=eval_df.index,
        )
        subgroups.append(
            (
                period_subgroup,
                period_levels,
                ["0", "1", "2", "3", "4", "5+", "Missing", "Unknown"],
            )
        )

        for subgroup_variable, levels, level_order in subgroups:
            for subgroup_level in level_order:
                subgroup_df = eval_df.loc[levels.eq(subgroup_level)]
                if subgroup_df.empty:
                    continue
                metrics = scalar_binary_metrics(subgroup_df, target_col, pred_col)
                performance_rows.append({
                    "model_group": model_group,
                    "model_type": MODEL_TYPE,
                    "outcome": outcome,
                    "risk_set": risk_set,
                    "subgroup_variable": subgroup_variable,
                    "subgroup_level": subgroup_level,
                    "n": metrics["n"],
                    "patients": int(subgroup_df[ID_COL].nunique()),
                    "events": metrics["events"],
                    "prevalence": metrics["prevalence"],
                    "auc": metrics["auc"],
                    "average_precision": metrics["average_precision"],
                    "brier": metrics["brier"],
                    "calibration_intercept": metrics["calibration_intercept"],
                    "calibration_slope": metrics["calibration_slope"],
                })
                table = calibration_table(
                    subgroup_df,
                    target_col,
                    pred_col,
                    bins=CALIBRATION_BINS,
                )
                metadata = {
                    "model_group": model_group,
                    "model_type": MODEL_TYPE,
                    "outcome": outcome,
                    "risk_set": risk_set,
                    "subgroup_variable": subgroup_variable,
                    "subgroup_level": subgroup_level,
                }
                for column, value in reversed(list(metadata.items())):
                    table.insert(0, column, value)
                calibration_tables.append(table)

    propensity_mask = df[STATE_COL].eq("in")
    add_task(
        "propensity",
        "removal",
        "all IN rows",
        propensity_mask,
        ACTION_COL,
        df[PROPENSITY_SCORE_COL],
        "catheter_duration_period",
    )
    for outcome, target_col in IN_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "in", outcome)
        factual_prediction = pd.Series(
            np.where(
                df[ACTION_COL].eq(1),
                df[f"p_{outcome}_if_remove"],
                df[f"p_{outcome}_if_keep"],
            ),
            index=df.index,
        )
        risk_set = "IN and at_risk_cauti == 1" if outcome == "cauti" else "all IN rows"
        add_task(
            "in_outcome",
            outcome,
            risk_set,
            risk_mask,
            target_col,
            factual_prediction,
            "catheter_duration_period",
        )

    for outcome, target_col in OUT_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "out", outcome)
        if outcome == "cauti":
            risk_set = "OUT and at_risk_cauti == 1 (48-hour attribution window)"
        elif outcome == "reinsertion":
            risk_set = "OUT and at_risk_reinsertion == 1"
        else:
            risk_set = "all OUT rows"
        add_task(
            "out_outcome",
            outcome,
            risk_set,
            risk_mask,
            target_col,
            df[f"p_{outcome}_if_out"],
            "post_removal_period",
        )

    return (
        pd.DataFrame(performance_rows),
        pd.concat(calibration_tables, ignore_index=True),
    )


def nuisance_learning_curves(
    df,
    remove_feature_cols,
    in_feature_cols,
    out_feature_cols,
    propensity_fold_models,
    in_models,
    out_models,
):
    # Build deterministic patient-grouped diagnostic learning curves
    tasks = [{
        "model_group": "propensity",
        "outcome": "removal",
        "risk_set": "all IN rows",
        "risk_mask": df[STATE_COL].eq("in"),
        "target_col": ACTION_COL,
        "feature_cols": remove_feature_cols,
        "fold_models": propensity_fold_models,
        "prediction": df[PROPENSITY_SCORE_COL],
        "model_name": "propensity_removal",
    }]
    for outcome, target_col in IN_OUTCOMES.items():
        risk_set = "IN and at_risk_cauti == 1" if outcome == "cauti" else "all IN rows"
        tasks.append({
            "model_group": "in_outcome",
            "outcome": outcome,
            "risk_set": risk_set,
            "risk_mask": outcome_risk_mask(df, "in", outcome),
            "target_col": target_col,
            "feature_cols": in_feature_cols,
            "fold_models": in_models[outcome]["fold_models"],
            "prediction": pd.Series(
                np.where(
                    df[ACTION_COL].eq(1),
                    df[f"p_{outcome}_if_remove"],
                    df[f"p_{outcome}_if_keep"],
                ),
                index=df.index,
            ),
            "model_name": f"in_{outcome}",
        })
    for outcome, target_col in OUT_OUTCOMES.items():
        if outcome == "cauti":
            risk_set = "OUT and at_risk_cauti == 1 (48-hour attribution window)"
        elif outcome == "reinsertion":
            risk_set = "OUT and at_risk_reinsertion == 1"
        else:
            risk_set = "all OUT rows"
        tasks.append({
            "model_group": "out_outcome",
            "outcome": outcome,
            "risk_set": risk_set,
            "risk_mask": outcome_risk_mask(df, "out", outcome),
            "target_col": target_col,
            "feature_cols": out_feature_cols,
            "fold_models": out_models[outcome]["fold_models"],
            "prediction": df[f"p_{outcome}_if_out"],
            "model_name": f"out_{outcome}",
        })

    rows = []
    for task in tasks:
        pooled_predictions = {fraction: [] for fraction in LEARNING_CURVE_FRACTIONS}
        training_summaries = {fraction: [] for fraction in LEARNING_CURVE_FRACTIONS}
        fallback_flags = {fraction: [] for fraction in LEARNING_CURVE_FRACTIONS}
        for fold in range(N_CROSSFIT_FOLDS):
            training_mask = task["risk_mask"] & df[CROSSFIT_FOLD_COL].ne(fold)
            validation_mask = task["risk_mask"] & df[CROSSFIT_FOLD_COL].eq(fold)
            training_patients = np.array(
                sorted(df.loc[training_mask, ID_COL].unique()),
                dtype=object,
            )
            held_out_patients = set(
                df.loc[df[CROSSFIT_FOLD_COL].eq(fold), ID_COL].unique()
            )
            rng = np.random.default_rng(SEED + fold)
            training_patients = training_patients[
                rng.permutation(len(training_patients))
            ]
            previous_patients = set()

            for training_fraction in LEARNING_CURVE_FRACTIONS:
                selected_n = (
                    len(training_patients)
                    if training_fraction == 1.0
                    else max(1, int(np.ceil(training_fraction * len(training_patients))))
                )
                selected_patients = set(training_patients[:selected_n])
                if not previous_patients.issubset(selected_patients):
                    raise ValueError("Learning-curve patient subsets are not nested")
                if selected_patients & held_out_patients:
                    raise ValueError("Learning-curve patient leakage detected")
                previous_patients = selected_patients
                subset_mask = training_mask & df[ID_COL].isin(selected_patients)
                training_target = binary_values(df.loc[subset_mask, task["target_col"]])

                reuse_model = training_fraction == 1.0
                print(
                    f"[LEARNING CURVE] {task['model_name']} "
                    f"fold {fold + 1}/{N_CROSSFIT_FOLDS} "
                    f"fraction={training_fraction:.2f} "
                    f"source={'production model' if reuse_model else 'diagnostic fit'}",
                    flush=True,
                )
                fold_model = (
                    task["fold_models"][fold]
                    if reuse_model
                    else fit_crossfit_fold_model(
                        df.loc[subset_mask, task["feature_cols"]],
                        df.loc[subset_mask, task["target_col"]],
                        f"learning_curve_{task['model_name']}_{training_fraction:.2f}",
                        fold,
                    )
                )
                predictions = predict_crossfit_fold(
                    fold_model,
                    df.loc[validation_mask, task["feature_cols"]],
                )
                if reuse_model and not np.allclose(
                    predictions,
                    task["prediction"].loc[validation_mask].to_numpy(dtype=float),
                    atol=1e-10,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Full learning-curve predictions differ from production "
                        f"predictions for {task['model_name']} fold {fold}"
                    )

                validation_df = df.loc[
                    validation_mask, [ID_COL, task["target_col"]]
                ].copy()
                pred_col = "_learning_curve_prediction"
                validation_df[pred_col] = predictions
                metrics = scalar_binary_metrics(
                    validation_df, task["target_col"], pred_col
                )
                fallback_used = bool(fold_model["fallback"])
                rows.append({
                    "aggregation": "fold",
                    "model_group": task["model_group"],
                    "model_type": MODEL_TYPE,
                    "outcome": task["outcome"],
                    "risk_set": task["risk_set"],
                    "fold": fold,
                    "training_fraction": training_fraction,
                    "training_patients": len(selected_patients),
                    "training_n": int(len(training_target)),
                    "training_events": int(training_target.sum()),
                    "validation_patients": int(validation_df[ID_COL].nunique()),
                    "validation_n": metrics["n"],
                    "validation_events": metrics["events"],
                    "prevalence": metrics["prevalence"],
                    "auc": metrics["auc"],
                    "average_precision": metrics["average_precision"],
                    "brier": metrics["brier"],
                    "calibration_intercept": metrics["calibration_intercept"],
                    "calibration_slope": metrics["calibration_slope"],
                    "fallback_used": fallback_used,
                })
                pooled_predictions[training_fraction].append(validation_df)
                training_summaries[training_fraction].append(
                    (len(selected_patients), len(training_target), int(training_target.sum()))
                )
                fallback_flags[training_fraction].append(fallback_used)

        for training_fraction in LEARNING_CURVE_FRACTIONS:
            pooled_df = pd.concat(
                pooled_predictions[training_fraction], ignore_index=True
            )
            pred_col = "_learning_curve_prediction"
            metrics = scalar_binary_metrics(pooled_df, task["target_col"], pred_col)
            training_summary = training_summaries[training_fraction]
            rows.append({
                "aggregation": "pooled_predictions",
                "model_group": task["model_group"],
                "model_type": MODEL_TYPE,
                "outcome": task["outcome"],
                "risk_set": task["risk_set"],
                "fold": np.nan,
                "training_fraction": training_fraction,
                "training_patients": sum(value[0] for value in training_summary),
                "training_n": sum(value[1] for value in training_summary),
                "training_events": sum(value[2] for value in training_summary),
                "validation_patients": int(pooled_df[ID_COL].nunique()),
                "validation_n": metrics["n"],
                "validation_events": metrics["events"],
                "prevalence": metrics["prevalence"],
                "auc": metrics["auc"],
                "average_precision": metrics["average_precision"],
                "brier": metrics["brier"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "fallback_used": any(fallback_flags[training_fraction]),
            })

    return pd.DataFrame(rows)


# Propensity model: observed clinician removal behaviour among IN rows

def fit_propensity_scores(df, feature_cols, remove_feature_cols):
    # Fit propensity scores
    df[PROPENSITY_SCORE_COL] = np.nan
    df[KEEP_PROPENSITY_SCORE_COL] = np.nan

    eligible_mask = df[STATE_COL].eq("in")
    fold_models = []

    # Fit crossfit fold model
    for fold in range(N_CROSSFIT_FOLDS):
        train_mask = eligible_mask & df[CROSSFIT_FOLD_COL].ne(fold)
        held_out_mask = eligible_mask & df[CROSSFIT_FOLD_COL].eq(fold)
        print(
            f"[FIT] propensity_removal fold {fold + 1}/{N_CROSSFIT_FOLDS} "
            f"train={int(train_mask.sum()):,} "
            f"held_out={int(held_out_mask.sum()):,}",
            flush=True,
        )
        # Fit crossfit fold model
        fold_model = fit_crossfit_fold_model(
            df.loc[train_mask, remove_feature_cols],
            df.loc[train_mask, ACTION_COL],
            "propensity_removal",
            fold,
        )
        held_out_index = df.index[held_out_mask]
        # Predict crossfit fold
        p_remove = predict_crossfit_fold(
            fold_model, df.loc[held_out_mask, remove_feature_cols]
        )
        df.loc[held_out_index, PROPENSITY_SCORE_COL] = p_remove
        df.loc[held_out_index, KEEP_PROPENSITY_SCORE_COL] = 1.0 - p_remove
        fold_models.append(fold_model)

    # Check valid predictions
    assert_valid_predictions(
        df,
        eligible_mask,
        [PROPENSITY_SCORE_COL, KEEP_PROPENSITY_SCORE_COL],
        "IN propensity rows",
    )
    propensity_sums = df.loc[
        eligible_mask, [PROPENSITY_SCORE_COL, KEEP_PROPENSITY_SCORE_COL]
    ].sum(axis=1)
    if not np.allclose(propensity_sums.to_numpy(), 1.0, atol=1e-10):
        raise ValueError("p_remove_obs + p_keep_obs does not equal 1 for all IN rows")

    evaluation_df = df.loc[
        eligible_mask,
        [ACTION_COL, PROPENSITY_SCORE_COL],
    ]
    # Calculate binary metrics
    summary = scalar_binary_metrics(evaluation_df, ACTION_COL, PROPENSITY_SCORE_COL)
    # Save a data frame as CSV
    pec.save_report_df(
        propensity_summary_rows(df, eligible_mask, feature_cols, summary),
        OUTDIR / "propensity_summary.csv",
    )
    # Save a data frame as CSV
    pec.save_report_df(
        calibration_table(
            evaluation_df,
            ACTION_COL,
            PROPENSITY_SCORE_COL,
            bins=CALIBRATION_BINS,
        ),
        OUTDIR / "propensity_calibration.csv",
    )
    # Write joblib
    joblib.dump(
        {
            **learner_provenance(),
            "evaluation": "grouped_cross_fit",
            "crossfit_folds": N_CROSSFIT_FOLDS,
            "crossfit_group_col": ID_COL,
            "fallback_probability_source": "fold_training_rows_only",
            "fallback_smoothing": {
                "method": "beta_binomial",
                "prior_events": FALLBACK_PRIOR_EVENTS,
                "prior_non_events": FALLBACK_PRIOR_NON_EVENTS,
            },
            "remove_fold_models": fold_models,
            "features": feature_cols,
            "x_cols_remove": remove_feature_cols,
            "target_col": ACTION_COL,
            "eligible_state": "in",
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "period_hours": PERIOD_HOURS,
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "modelling_panel_file": str(INFILE),
        },
        OUTDIR / "propensity_model.pkl",
    )

    return df, summary, fold_models


# State-specific binary outcome nuisance models

def fit_outcome_scores(df, feature_cols, in_feature_cols, out_feature_cols):
    # Fit outcome scores
    covariate_dict = pd.read_csv(COVARIATE_DICT_FILE)
    # Prepare outcome targets
    df = prepare_outcome_targets(df)

    for col in ALL_SCORE_COLS[2:]:
        df[col] = np.nan

    in_models = {}
    out_models = {}
    in_summary_rows = []
    out_summary_rows = []
    in_calibration_tables = []
    out_calibration_tables = []
    in_action_summary_rows = []
    in_action_calibration_tables = []
    importance_tables = []

    in_rows = df[STATE_COL].eq("in")
    out_rows = df[STATE_COL].eq("out")

    # IN models are fitted on observed keep/remove actions, then every eligible
    # held-out row is scored twice with all pre-decision features held fixed
    for outcome, target_col in IN_OUTCOMES.items():
        # Summarise risk mask
        risk_mask = outcome_risk_mask(df, "in", outcome)

        model_name = f"in_{outcome}"
        keep_col = f"p_{outcome}_if_keep"
        remove_col = f"p_{outcome}_if_remove"
        if outcome == "cauti":
            # Any unexpected non-risk IN row contributes zero CAUTI risk
            df.loc[in_rows & ~risk_mask, [keep_col, remove_col]] = 0.0

        fold_models = []
        # Fit crossfit fold model
        for fold in range(N_CROSSFIT_FOLDS):
            train_mask = risk_mask & df[CROSSFIT_FOLD_COL].ne(fold)
            held_out_mask = risk_mask & df[CROSSFIT_FOLD_COL].eq(fold)
            print(
                f"[FIT] {model_name} fold {fold + 1}/{N_CROSSFIT_FOLDS} "
                f"train={int(train_mask.sum()):,} "
                f"held_out={int(held_out_mask.sum()):,}",
                flush=True,
            )
            # Fit crossfit fold model
            fold_model = fit_crossfit_fold_model(
                df.loc[train_mask, in_feature_cols],
                df.loc[train_mask, target_col],
                model_name,
                fold,
            )
            counterfactual_features = df.loc[
                held_out_mask, in_feature_cols
            ].copy()
            counterfactual_features[ACTION_COL] = 0
            # Predict crossfit fold
            df.loc[held_out_mask, keep_col] = predict_crossfit_fold(
                fold_model, counterfactual_features
            )
            counterfactual_features[ACTION_COL] = 1
            # Predict crossfit fold
            df.loc[held_out_mask, remove_col] = predict_crossfit_fold(
                fold_model, counterfactual_features
            )
            fold_models.append(fold_model)

        observed_pred_col = f"_p_{outcome}_observed_in"
        eval_df = df.loc[
            risk_mask,
            [target_col, ACTION_COL, keep_col, remove_col],
        ].copy()
        eval_df[observed_pred_col] = np.where(
            eval_df[ACTION_COL].eq(1),
            eval_df[remove_col],
            eval_df[keep_col],
        )
        modelling_df = eval_df[[target_col]]
        risk_set = "IN and at_risk_cauti == 1" if outcome == "cauti" else "all IN rows"
        # Build summary row
        in_summary_rows.append(
            model_summary_row(
                "in_outcome",
                outcome,
                target_col,
                in_feature_cols,
                modelling_df,
                eval_df,
                observed_pred_col,
                risk_set,
            )
        )
        # Build calibration
        in_calibration_tables.append(
            labelled_calibration(
                eval_df,
                target_col,
                observed_pred_col,
                "in_outcome",
                outcome,
                risk_set,
            )
        )
        for action_value, observed_action, pred_col in [
            (0, "keep", keep_col),
            (1, "remove", remove_col),
        ]:
            action_eval_df = eval_df.loc[eval_df[ACTION_COL].eq(action_value)]
            action_metrics = scalar_binary_metrics(
                action_eval_df, target_col, pred_col
            )
            in_action_summary_rows.append({
                "model_group": "in_outcome",
                "model_type": MODEL_TYPE,
                "outcome": outcome,
                "risk_set": risk_set,
                "observed_action": observed_action,
                **action_metrics,
            })
            action_calibration = calibration_table(
                action_eval_df,
                target_col,
                pred_col,
                bins=CALIBRATION_BINS,
            )
            action_calibration.insert(0, "observed_action", observed_action)
            action_calibration.insert(0, "risk_set", risk_set)
            action_calibration.insert(0, "outcome", outcome)
            action_calibration.insert(0, "model_group", "in_outcome")
            in_action_calibration_tables.append(action_calibration)
        # Build series data frame
        importance_tables.append(
            top_series_df(
                model_name,
                mean_feature_importance_series(fold_models),
                TOP_FEATURES_TO_SAVE,
            )
        )
        in_models[outcome] = {
            "fold_models": fold_models,
            "features": in_feature_cols,
            "target_col": target_col,
            "risk_set": risk_set,
            "no_event_definition": NO_EVENT_DEFINITION["in"] if outcome == "no_event" else None,
            "no_event_is_transition_state": False if outcome == "no_event" else None,
        }

    # OUT models contain no action or state indicator. CAUTI is trained/scored
    # only during the 48-hour attribution window; non-risk OUT rows are zero
    for outcome, target_col in OUT_OUTCOMES.items():
        # Summarise risk mask
        risk_mask = outcome_risk_mask(df, "out", outcome)

        model_name = f"out_{outcome}"
        score_col = f"p_{outcome}_if_out"
        if outcome in {"cauti", "reinsertion"}:
            df.loc[out_rows & ~risk_mask, score_col] = 0.0

        fold_models = []
        # Fit crossfit fold model
        for fold in range(N_CROSSFIT_FOLDS):
            train_mask = risk_mask & df[CROSSFIT_FOLD_COL].ne(fold)
            held_out_mask = risk_mask & df[CROSSFIT_FOLD_COL].eq(fold)
            print(
                f"[FIT] {model_name} fold {fold + 1}/{N_CROSSFIT_FOLDS} "
                f"train={int(train_mask.sum()):,} "
                f"held_out={int(held_out_mask.sum()):,}",
                flush=True,
            )
            # Fit crossfit fold model
            fold_model = fit_crossfit_fold_model(
                df.loc[train_mask, out_feature_cols],
                df.loc[train_mask, target_col],
                model_name,
                fold,
            )
            # Predict crossfit fold
            df.loc[held_out_mask, score_col] = predict_crossfit_fold(
                fold_model, df.loc[held_out_mask, out_feature_cols]
            )
            fold_models.append(fold_model)

        eval_df = df.loc[risk_mask, [target_col, score_col]]
        modelling_df = eval_df[[target_col]]
        if outcome == "cauti":
            risk_set = "OUT and at_risk_cauti == 1 (48-hour attribution window)"
        elif outcome == "reinsertion":
            risk_set = "OUT and at_risk_reinsertion == 1"
        else:
            risk_set = "all OUT rows"
        # Build summary row
        out_summary_rows.append(
            model_summary_row(
                "out_outcome",
                outcome,
                target_col,
                out_feature_cols,
                modelling_df,
                eval_df,
                score_col,
                risk_set,
            )
        )
        # Build calibration
        out_calibration_tables.append(
            labelled_calibration(
                eval_df,
                target_col,
                score_col,
                "out_outcome",
                outcome,
                risk_set,
            )
        )
        # Build series data frame
        importance_tables.append(
            top_series_df(
                model_name,
                mean_feature_importance_series(fold_models),
                TOP_FEATURES_TO_SAVE,
            )
        )
        out_models[outcome] = {
            "fold_models": fold_models,
            "features": out_feature_cols,
            "target_col": target_col,
            "risk_set": risk_set,
            "non_risk_prediction": 0.0 if outcome in {"cauti", "reinsertion"} else None,
            "no_event_definition": NO_EVENT_DEFINITION["out"] if outcome == "no_event" else None,
            "no_event_is_transition_state": False if outcome == "no_event" else None,
        }

    in_required = [
        f"p_{outcome}_if_{action}"
        for outcome in IN_OUTCOMES
        for action in ["keep", "remove"]
    ]
    out_required = [f"p_{outcome}_if_out" for outcome in OUT_OUTCOMES]
    # Check valid predictions
    assert_valid_predictions(df, in_rows, in_required, "IN outcome rows")
    # Check valid predictions
    assert_valid_predictions(df, out_rows, out_required, "OUT outcome rows")

    in_summary = pd.DataFrame(in_summary_rows)
    out_summary = pd.DataFrame(out_summary_rows)
    # Save a data frame as CSV
    pec.save_report_df(in_summary, OUTDIR / "in_outcome_summary.csv")
    # Save a data frame as CSV
    pec.save_report_df(out_summary, OUTDIR / "out_outcome_summary.csv")
    # Save a data frame as CSV
    pec.save_report_df(
        pd.concat(in_calibration_tables, ignore_index=True),
        OUTDIR / "in_outcome_calibration.csv",
    )
    # Save a data frame as CSV
    pec.save_report_df(
        pd.concat(out_calibration_tables, ignore_index=True),
        OUTDIR / "out_outcome_calibration.csv",
    )
    # Save factual IN outcome diagnostics by observed action
    pec.save_report_df(
        pd.DataFrame(in_action_summary_rows),
        OUTDIR / "in_outcome_action_summary.csv",
    )
    pec.save_report_df(
        pd.concat(in_action_calibration_tables, ignore_index=True),
        OUTDIR / "in_outcome_action_calibration.csv",
    )

    # Add feature descriptions
    importance_df = add_feature_descriptions(
        pd.concat(importance_tables, ignore_index=True),
        covariate_dict,
    )
    # Save a data frame as CSV
    pec.save_report_df(importance_df, OUTDIR / "outcome_top_model_features.csv")

    fallback_counts = outcome_fallback_counts(
        {"in": in_models, "out": out_models}
    )
    # Write joblib
    joblib.dump(
        {
            **learner_provenance(),
            "model_group": "state_specific_binary",
            "evaluation": "grouped_cross_fit",
            "crossfit_folds": N_CROSSFIT_FOLDS,
            "crossfit_group_col": ID_COL,
            "fallback_probability_source": "fold_training_rows_only",
            "fallback_smoothing": {
                "method": "beta_binomial",
                "prior_events": FALLBACK_PRIOR_EVENTS,
                "prior_non_events": FALLBACK_PRIOR_NON_EVENTS,
            },
            "in_models": in_models,
            "out_models": out_models,
            "x_cols_in": in_feature_cols,
            "x_cols_out": out_feature_cols,
            "action_remove_col": ACTION_COL,
            "features": feature_cols,
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "period_hours": PERIOD_HOURS,
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "out_cauti_non_risk_prediction": 0.0,
            "fallback_fold_counts": fallback_counts,
            "no_event_definition": NO_EVENT_DEFINITION,
            "no_event_is_transition_state": False,
            "modelling_panel_file": str(INFILE),
        },
        OUTDIR / "outcome_models.pkl",
    )

    return df, fallback_counts, in_summary, out_summary, in_models, out_models


def build_nuisance_model_comparison():
    # Consolidate saved pooled prediction diagnostics without fitting any models
    propensity_tail_metrics = [
        "p_remove_min",
        "p_remove_p01",
        "p_remove_p05",
        "p_remove_p50",
        "p_remove_p95",
        "p_remove_p99",
        "p_remove_max",
        "pct_p_remove_below_0_01",
        "pct_p_remove_below_0_05",
        "pct_p_remove_above_0_95",
        "pct_p_remove_above_0_99",
    ]
    stability_columns = [
        "fold_auc_mean",
        "fold_auc_sd",
        "fold_brier_mean",
        "fold_brier_sd",
        "fold_calibration_intercept_mean",
        "fold_calibration_intercept_sd",
        "fold_calibration_slope_mean",
        "fold_calibration_slope_sd",
    ]
    columns = [
        "model_type",
        "model_folder",
        "model_group",
        "outcome",
        "risk_set",
        "n",
        "events",
        "prevalence",
        "auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        *propensity_tail_metrics,
        *stability_columns,
    ]
    required_files = [
        "propensity_summary.csv",
        "in_outcome_summary.csv",
        "out_outcome_summary.csv",
        "fold_performance_metrics.csv",
    ]
    required_outcome_columns = {
        "model_group",
        "model_type",
        "outcome",
        "risk_set",
        "n",
        "events",
        "prevalence",
        "auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    }
    required_fold_columns = {
        "model_group",
        "model_type",
        "outcome",
        "auc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    }
    required_propensity_metrics = {
        "n",
        "events",
        "prevalence",
        "auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    }

    rows = []
    ignored_folders = []
    for model_folder in sorted(path for path in NUISANCE_ROOT.iterdir() if path.is_dir()):
        missing_files = [
            name for name in required_files if not (model_folder / name).is_file()
        ]
        if missing_files:
            print(
                f"[WARNING] Ignoring incomplete model folder {model_folder.name}: "
                f"missing {', '.join(missing_files)}",
                flush=True,
            )
            ignored_folders.append(model_folder.name)
            continue

        try:
            propensity_summary = pd.read_csv(
                model_folder / "propensity_summary.csv"
            )
            in_summary = pd.read_csv(model_folder / "in_outcome_summary.csv")
            out_summary = pd.read_csv(model_folder / "out_outcome_summary.csv")
            fold_metrics = pd.read_csv(
                model_folder / "fold_performance_metrics.csv"
            )
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            print(
                f"[WARNING] Ignoring invalid model folder {model_folder.name}: {error}",
                flush=True,
            )
            ignored_folders.append(model_folder.name)
            continue

        propensity_columns_valid = {"metric", "value"}.issubset(
            propensity_summary.columns
        )
        propensity_metrics = (
            set(propensity_summary["metric"])
            if propensity_columns_valid
            else set()
        )
        if (
            not propensity_columns_valid
            or not required_propensity_metrics.issubset(propensity_metrics)
            or not required_outcome_columns.issubset(in_summary.columns)
            or not required_outcome_columns.issubset(out_summary.columns)
            or not required_fold_columns.issubset(fold_metrics.columns)
        ):
            print(
                f"[WARNING] Ignoring invalid model folder {model_folder.name}: "
                "required comparison columns are missing",
                flush=True,
            )
            ignored_folders.append(model_folder.name)
            continue

        identities = set()
        model_type_values = propensity_summary.loc[
            propensity_summary["metric"].eq("model_type"), "value"
        ].dropna()
        identities.update(model_type_values.astype(str))
        for summary_df in [in_summary, out_summary, fold_metrics]:
            identities.update(summary_df["model_type"].dropna().astype(str).unique())
        if len(identities) > 1:
            print(
                f"[WARNING] Ignoring invalid model folder {model_folder.name}: "
                "inconsistent model identifiers",
                flush=True,
            )
            ignored_folders.append(model_folder.name)
            continue
        model_type = next(iter(identities), model_folder.name)

        def propensity_value(metric):
            values = propensity_summary.loc[
                propensity_summary["metric"].eq(metric), "value"
            ]
            if values.empty:
                return np.nan
            return pd.to_numeric(values.iloc[0], errors="coerce")

        folder_rows = [{
            "model_type": model_type,
            "model_folder": model_folder.name,
            "model_group": "propensity",
            "outcome": "removal",
            "risk_set": "all IN rows",
            "n": propensity_value("n"),
            "events": propensity_value("events"),
            "prevalence": propensity_value("prevalence"),
            "auc": propensity_value("auc"),
            "average_precision": propensity_value("average_precision"),
            "brier": propensity_value("brier"),
            "calibration_intercept": propensity_value("calibration_intercept"),
            "calibration_slope": propensity_value("calibration_slope"),
            **{
                metric: propensity_value(metric)
                for metric in propensity_tail_metrics
            },
        }]
        for outcome_row in pd.concat(
            [in_summary, out_summary], ignore_index=True
        ).to_dict("records"):
            folder_rows.append({
                "model_type": model_type,
                "model_folder": model_folder.name,
                "model_group": outcome_row["model_group"],
                "outcome": outcome_row["outcome"],
                "risk_set": outcome_row["risk_set"],
                "n": outcome_row["n"],
                "events": outcome_row["events"],
                "prevalence": outcome_row["prevalence"],
                "auc": outcome_row["auc"],
                "average_precision": outcome_row["average_precision"],
                "brier": outcome_row["brier"],
                "calibration_intercept": outcome_row["calibration_intercept"],
                "calibration_slope": outcome_row["calibration_slope"],
                **{metric: np.nan for metric in propensity_tail_metrics},
            })

        fold_groups = {
            key: group
            for key, group in fold_metrics.groupby(
                ["model_group", "outcome"], observed=False
            )
        }
        for row in folder_rows:
            fold_group = fold_groups.get((row["model_group"], row["outcome"]))
            stability = {}
            for metric in [
                "auc",
                "brier",
                "calibration_intercept",
                "calibration_slope",
            ]:
                values = (
                    pd.to_numeric(fold_group[metric], errors="coerce")
                    if fold_group is not None
                    else pd.Series(dtype=float)
                )
                stability[f"fold_{metric}_mean"] = (
                    float(values.mean()) if values.notna().any() else np.nan
                )
                stability[f"fold_{metric}_sd"] = (
                    float(values.std()) if values.notna().sum() > 1 else np.nan
                )
            row.update(stability)
        rows.extend(folder_rows)

    comparison = pd.DataFrame(rows, columns=columns)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["model_group", "outcome", "model_type", "model_folder"]
        ).reset_index(drop=True)
    pec.save_report_df(comparison, MODEL_COMPARISON_FILE)
    return comparison, ignored_folders


# Final panel assembly

def save_nuisance_predictions(df):
    # Save the modelling panel together with its nuisance predictions
    excluded_cols = {Y_NO_EVENT_IN, Y_NO_EVENT_OUT}
    output_cols = [
        col
        for col in df.columns
        if col not in excluded_cols and col not in ALL_SCORE_COLS
    ]
    insert_at = output_cols.index("age")
    output_cols[insert_at:insert_at] = ALL_SCORE_COLS
    df.to_csv(
        NUISANCE_PREDICTIONS_FILE,
        columns=output_cols,
        index=False,
        float_format="%.6f",
    )


def run_nuisance_model(model_type):
    configure_model_run(model_type)
    print(
        f"[MODEL] Starting {MODEL_OUTPUT_NAME} ({MODEL_TYPE})",
        flush=True,
    )
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Load and prepare the modelling panel
    df = load_panel()

    # Derive model features
    feature_cols = [
        col for col in df.columns
        if col.startswith(("itemid_", "sex_", "ethnicity_"))
    ]
    feature_cols.append("age")
    remove_feature_cols = [TIME_COL, PERIODS_COL, *feature_cols]
    out_feature_cols = [TIME_COL, PERIODS_COL, *feature_cols]
    in_feature_cols = [*out_feature_cols, ACTION_COL]

    # Add grouped crossfit folds
    df = add_grouped_crossfit_folds(df)

    # Save fold diagnostics
    pec.save_report_df(
        crossfit_fold_summary(df),
        OUTDIR / "crossfit_fold_summary.csv",
    )
    # Save a data frame as CSV
    crossfit_row_assignments(df).to_csv(
        CROSSFIT_ROW_ASSIGNMENTS_FILE,
        index=False,
        float_format="%.6f",
    )

    # Fit propensity scores
    df, propensity_summary, propensity_fold_models = fit_propensity_scores(
        df,
        feature_cols,
        remove_feature_cols,
    )

    # Fit outcome scores
    df, _, in_summary, out_summary, in_models, out_models = fit_outcome_scores(
        df,
        feature_cols,
        in_feature_cols,
        out_feature_cols,
    )
    pec.save_report_df(
        constant_features_by_fold(
            propensity_fold_models,
            in_models,
            out_models,
        ),
        CONSTANT_FEATURES_FILE,
    )

    # Validate exported probabilities
    validate_exported_probabilities(df)

    # Save combined performance metrics
    pec.save_report_df(
        performance_metrics_rows(propensity_summary, in_summary, out_summary),
        PERFORMANCE_METRICS_FILE,
    )
    pec.save_report_df(
        fold_performance_metrics(df),
        OUTDIR / "fold_performance_metrics.csv",
    )

    production_predictions = df[ALL_SCORE_COLS].copy()
    subgroup_performance, subgroup_calibration = nuisance_subgroup_diagnostics(df)
    pec.save_report_df(
        subgroup_performance,
        OUTDIR / "nuisance_subgroup_performance.csv",
    )
    pec.save_report_df(
        subgroup_calibration,
        OUTDIR / "nuisance_subgroup_calibration.csv",
    )
    pec.save_report_df(
        nuisance_learning_curves(
            df,
            remove_feature_cols,
            in_feature_cols,
            out_feature_cols,
            propensity_fold_models,
            in_models,
            out_models,
        ),
        OUTDIR / "nuisance_learning_curves.csv",
    )
    if not df[ALL_SCORE_COLS].equals(production_predictions):
        raise ValueError("Diagnostics altered production nuisance predictions")

    # Save nuisance predictions
    save_nuisance_predictions(df)
    print(
        f"[MODEL] Completed {MODEL_OUTPUT_NAME} ({MODEL_TYPE})",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit cross-fitted nuisance models for one source panel."
    )
    add_panel_argument(parser)
    parser.add_argument(
        "--model-type",
        choices=("all", *MODEL_TYPES),
        default="xgboost",
        help=(
            "Nuisance learner to fit. The default is the selected production "
            "learner, xgboost; use 'all' only for a model-comparison run."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = configure_panel_run(args.panel)
    print(f"[PANEL] {paths.panel_name}: {paths.panel_path}", flush=True)

    model_types = MODEL_TYPES if args.model_type == "all" else (args.model_type,)
    for model_type in model_types:
        run_nuisance_model(model_type)

    # Consolidate completed nuisance-model runs
    build_nuisance_model_comparison()


if __name__ == "__main__":
    main()
