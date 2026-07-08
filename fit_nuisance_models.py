from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_TYPE = "xgb"

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\data")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\nuisance_models")
MODEL_DIR = OUTDIR

INFILE = INDIR / "modeling_panel.csv"
FEATURE_SPEC_FILE = INDIR / "feature_spec.json"
COVARIATE_DICT_FILE = INDIR / "covariate_dictionary.csv"
FINAL_PANEL = OUTDIR / "scored_panel.csv"
PERFORMANCE_METRICS_FILE = OUTDIR / "performance_metrics.csv"
CROSSFIT_ROW_ASSIGNMENTS_FILE = OUTDIR / "crossfit_row_assignments.csv"

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
OBSERVED_ACTION_COL = "observed_action"
ACTION_REMOVE_COL = "action_remove"
END_REASON_COL = "episode_end_reason"
AT_RISK_CAUTI = "at_risk_cauti"
AT_RISK_REINS = "at_risk_reinsertion"

Y_ICU_EXIT_ALIVE = "_target_icu_exit_alive"
Y_NO_EVENT_IN = "_target_no_event_in"
Y_NO_EVENT_OUT = "_target_no_event_out"

NO_EVENT_DEFINITION = {
    "in": (
        "No CAUTI, death, or ICU exit in the next outcome window; this does not "
        "imply continued catheter-in state when action_remove=1."
    ),
    "out": (
        "No reinsertion, CAUTI, death, or ICU exit in the next outcome window."
    ),
}

POST_REMOVE_RISK_PERIODS = 2
TOP_FEATURES_TO_SAVE = 15
CALIBRATION_BINS = 10
LOW_COUNT_WARNING_THRESHOLD = 20
N_CROSSFIT_FOLDS = 5
CROSSFIT_FOLD_COL = "_crossfit_fold"
FALLBACK_PRIOR_EVENTS = 1.0
FALLBACK_PRIOR_NON_EVENTS = 1.0

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

# These belonged to the old pooled multiclass model. Removal is an action, not
# an outcome class, and observed transition probabilities are no longer needed.
OLD_TRANSITION_SCORE_COLS = [
    *(f"p_removal_if_{action}" for action in ["keep", "remove", "out"]),
    *(f"p_reinsertion_if_{action}" for action in ["keep", "remove"]),
    *(f"p_no_event_continue_if_{action}" for action in ["keep", "remove", "out"]),
    *(
        f"p_{outcome}_obs"
        for outcome in [
            "cauti",
            "reinsertion",
            "removal",
            "death",
            "icu_exit_alive",
            "no_event_continue",
        ]
    ),
]


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


def binary_values(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def load_panel():
    df = pd.read_csv(INFILE, low_memory=False)
    df.columns = df.columns.str.strip()
    require_columns(
        df,
        [
            ID_COL,
            STATE_COL,
            ACTION_COL,
            Y_CAUTI,
            Y_REINS,
            Y_DEATH,
            Y_ICU_EXIT,
            AT_RISK_CAUTI,
            AT_RISK_REINS,
        ],
        "modeling-panel",
    )

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()
    if SPLIT_COL in df.columns:
        # Retained as source metadata only; it is not used for model fitting.
        df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    if END_REASON_COL in df.columns:
        df[END_REASON_COL] = df[END_REASON_COL].astype(str).str.strip().str.lower()

    unknown_states = sorted(set(df[STATE_COL].dropna()) - {"in", "out"})
    if unknown_states:
        raise ValueError(f"Unexpected catheter states: {unknown_states}")

    for col in [
        ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        Y_DEATH,
        Y_ICU_EXIT,
        AT_RISK_CAUTI,
        AT_RISK_REINS,
    ]:
        df[col] = binary_values(df[col])
    return df


def add_grouped_crossfit_folds(df, n_splits=N_CROSSFIT_FOLDS):
    """Assign one shared fold per patient for every nuisance model."""
    n_groups = int(df[ID_COL].nunique())
    if n_groups < n_splits:
        raise ValueError(
            f"Grouped cross-fitting needs at least {n_splits} patients; found {n_groups}"
        )

    df = df.copy()
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
    print(
        f"Confirmed {n_groups} patients are each assigned to exactly one "
        f"of {n_splits} cross-fit folds.",
        flush=True,
    )
    return df


def crossfit_fold_summary(df):
    return (
        df.groupby(CROSSFIT_FOLD_COL, observed=False)
        .agg(rows=(ID_COL, "size"), patients=(ID_COL, "nunique"))
        .reset_index()
        .rename(columns={CROSSFIT_FOLD_COL: "fold"})
    )


def crossfit_row_assignments(df):
    required = [ID_COL, TIME_COL, STATE_COL, CROSSFIT_FOLD_COL]
    require_columns(df, required, "cross-fit row-assignment")
    optional_identifiers = [
        col for col in ["stay_id", "inserted"] if col in df.columns
    ]
    columns = [
        ID_COL,
        *optional_identifiers,
        TIME_COL,
        STATE_COL,
        CROSSFIT_FOLD_COL,
    ]
    return df[columns].copy()


def dump_joblib(payload, path):
    joblib.dump(payload, path)


def insert_score_columns_before_age(df, score_col_names):
    ordered_cols = [col for col in df.columns if col not in score_col_names]
    insert_at = ordered_cols.index("age") if "age" in ordered_cols else len(ordered_cols)
    ordered_cols[insert_at:insert_at] = [col for col in score_col_names if col in df.columns]
    return df[ordered_cols].copy()


def fit_binary_model(features, target, model_name):
    target = binary_values(target)
    if features.empty:
        raise ValueError(f"Cannot fit {model_name}: training risk set is empty")
    if target.nunique() < 2:
        raise ValueError(
            f"Cannot fit {model_name}: training target contains only class "
            f"{int(target.iloc[0])}"
        )

    events = int(target.sum())
    non_events = int(len(target) - events)
    if min(events, non_events) < LOW_COUNT_WARNING_THRESHOLD:
        print(
            f"WARNING: {model_name} training data have {events} events and "
            f"{non_events} non-events.",
            flush=True,
        )

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
    pipe.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=int))
    return pipe


def predict_binary_proba(pipe, features):
    return pipe.predict_proba(features.to_numpy(dtype=float))[:, 1]


def fit_crossfit_fold_model(
    features,
    target,
    model_name,
    fold,
):
    """Fit one fold model, or use a training-only smoothed constant."""
    if features.empty:
        raise ValueError(f"Cannot fit {model_name} fold {fold}: training risk set is empty")
    target = binary_values(target)
    events = int(target.sum())
    non_events = int(len(target) - events)
    fold_metadata = {
        "fold": int(fold),
        "training_n": int(len(target)),
        "training_events": events,
        "training_non_events": non_events,
    }
    if target.nunique() < 2:
        # Beta(1, 1) / Laplace smoothing avoids exact zero or one while using
        # only this fold's training rows. Held-out outcomes are never used.
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
        print(
            f"WARNING: {model_name} fold {fold} has a single-class training "
            f"target (events={events}, n={len(target)}); using training-only "
            f"Laplace-smoothed fallback p={fallback_probability:.6f}.",
            flush=True,
        )
        return {
            **fold_metadata,
            "model": None,
            "fallback": True,
            "fallback_probability": fallback_probability,
            "fallback_probability_source": "fold_training_rows_only",
            "fallback_smoothing": "beta_binomial",
            "fallback_prior_events": FALLBACK_PRIOR_EVENTS,
            "fallback_prior_non_events": FALLBACK_PRIOR_NON_EVENTS,
        }

    model = fit_binary_model(features, target, f"{model_name}_fold_{fold}")
    return {
        **fold_metadata,
        "model": model,
        "fallback": False,
        "fallback_probability": None,
        "fallback_probability_source": None,
        "fallback_smoothing": None,
        "fallback_prior_events": None,
        "fallback_prior_non_events": None,
    }


def predict_crossfit_fold(fold_model, features):
    if fold_model["fallback"]:
        return np.full(
            len(features),
            fold_model["fallback_probability"],
            dtype=float,
        )
    return predict_binary_proba(fold_model["model"], features)


def feature_importance_series(pipe, feature_cols):
    estimator = pipe.named_steps["xgb"]
    return pd.Series(
        estimator.feature_importances_,
        index=list(feature_cols),
    ).sort_values(ascending=False)


def mean_feature_importance_series(fold_models, feature_cols):
    importances = [
        feature_importance_series(fold_model["model"], feature_cols)
        for fold_model in fold_models
        if not fold_model["fallback"] and fold_model["model"] is not None
    ]
    if not importances:
        return pd.Series(dtype=float)
    return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def outcome_fallback_counts(outcome_models):
    counts = {}
    for state, models in outcome_models.items():
        for outcome, payload in models.items():
            counts[f"{state}_{outcome}"] = sum(
                int(fold_model["fallback"])
                for fold_model in payload["fold_models"]
            )
    return counts


def top_series_df(model_name, values, top_n):
    top_df = values.head(top_n).reset_index()
    top_df.columns = ["feature", "model_importance"]
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


def clean_eval_frame(df, outcome_col, pred_col):
    eval_df = df[[outcome_col, pred_col]].copy()
    eval_df[outcome_col] = pd.to_numeric(eval_df[outcome_col], errors="coerce")
    eval_df[pred_col] = pd.to_numeric(eval_df[pred_col], errors="coerce")
    eval_df = eval_df.dropna(subset=[outcome_col, pred_col]).copy()
    eval_df[outcome_col] = eval_df[outcome_col].astype(int)
    return eval_df


def scalar_binary_metrics(df, outcome_col, pred_col):
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
    return {
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "auc": auc,
        "average_precision": average_precision,
        "brier": brier,
    }


def calibration_table(df, outcome_col, pred_col, bins=10):
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
    df = df.copy()
    df[OBSERVED_ACTION_COL] = np.where(
        df[STATE_COL].eq("out"),
        "out",
        np.where(df[ACTION_COL].eq(1), "remove", "keep"),
    )
    df[ACTION_REMOVE_COL] = (
        df[STATE_COL].eq("in") & df[ACTION_COL].eq(1)
    ).astype(int)

    # ICU exit alive excludes deaths occurring in the same interval.
    df[Y_ICU_EXIT_ALIVE] = (
        df[Y_ICU_EXIT].eq(1) & df[Y_DEATH].eq(0)
    ).astype(int)

    # These are interval outcomes, not transition-state labels. In particular,
    # IN no_event under action_remove=1 does not mean the catheter stayed IN.
    # Removal is deliberately absent because it is the action.
    df[Y_NO_EVENT_IN] = (
        df[Y_CAUTI].eq(0)
        & df[Y_DEATH].eq(0)
        & df[Y_ICU_EXIT].eq(0)
    ).astype(int)
    df[Y_NO_EVENT_OUT] = (
        df[Y_REINS].eq(0)
        & df[Y_CAUTI].eq(0)
        & df[Y_DEATH].eq(0)
        & df[Y_ICU_EXIT].eq(0)
    ).astype(int)
    return df


def state_feature_lists(feature_spec):
    configured = feature_spec.get("x_cols_transition")
    if not configured:
        configured = [TIME_COL, PERIODS_COL, ACTION_REMOVE_COL, *feature_spec["features"]]

    excluded = {"state_is_out", ACTION_REMOVE_COL}
    base_cols = [col for col in configured if col not in excluded]
    in_feature_cols = [*base_cols, ACTION_REMOVE_COL]
    out_feature_cols = base_cols

    leakage_cols = {
        ACTION_COL,
        OBSERVED_ACTION_COL,
        Y_CAUTI,
        Y_REINS,
        Y_DEATH,
        Y_ICU_EXIT,
        AT_RISK_CAUTI,
        AT_RISK_REINS,
    }
    leaked = sorted(set(base_cols) & leakage_cols)
    if leaked:
        raise ValueError(f"Outcome feature specification contains post-decision columns: {leaked}")
    return in_feature_cols, out_feature_cols


def outcome_risk_mask(df, state, outcome):
    mask = df[STATE_COL].eq(state)
    if outcome == "cauti":
        mask &= df[AT_RISK_CAUTI].eq(1)
    elif outcome == "reinsertion":
        mask &= df[AT_RISK_REINS].eq(1)
    return mask


def assert_valid_predictions(df, row_mask, columns, context):
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
    require_columns(df, ALL_SCORE_COLS, "exported probability")
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
    assert_valid_predictions(
        df,
        in_rows,
        ["p_cauti_if_keep", "p_cauti_if_remove"],
        "eligible IN CAUTI rows",
    )
    assert_valid_predictions(
        df,
        out_rows,
        ["p_reinsertion_if_out"],
        "eligible OUT reinsertion rows",
    )

    out_cauti_risk = out_rows & df[AT_RISK_CAUTI].eq(1)
    out_cauti_non_risk = out_rows & df[AT_RISK_CAUTI].eq(0)
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
    modeling_df,
    eval_df,
    pred_col,
    risk_set,
):
    metrics = scalar_binary_metrics(eval_df, target_col, pred_col)
    modeling_target = binary_values(modeling_df[target_col])
    if metrics["events"] < LOW_COUNT_WARNING_THRESHOLD:
        print(
            f"WARNING: {model_group} {outcome} out-of-fold risk set has only "
            f"{metrics['events']} events.",
            flush=True,
        )
    return {
        "model_group": model_group,
        "model_type": MODEL_TYPE,
        "outcome": outcome,
        "target_col": target_col,
        "risk_set": risk_set,
        "evaluation": "grouped_cross_fit_oof",
        "crossfit_folds": N_CROSSFIT_FOLDS,
        "crossfit_group_col": ID_COL,
        "feature_count": int(len(feature_cols)),
        "modeling_n": int(len(modeling_df)),
        "modeling_events": int(modeling_target.sum()),
        "modeling_prevalence": float(modeling_target.mean()) if len(modeling_target) else np.nan,
        **metrics,
        # Primary metrics already use the outcome-specific risk set. These named
        # fields are retained for policy-evaluation diagnostics.
        "risk_set_auc": metrics["auc"],
        "risk_set_brier": metrics["brier"],
    }


def labelled_calibration(eval_df, target_col, pred_col, model_group, outcome, risk_set):
    table = calibration_table(eval_df, target_col, pred_col, bins=CALIBRATION_BINS)
    table.insert(0, "risk_set", risk_set)
    table.insert(0, "outcome", outcome)
    table.insert(0, "model_group", model_group)
    return table


def propensity_summary_rows(df, eligible_df, feature_list, feature_spec, summary):
    return pd.DataFrame([
        {"metric": "model_type", "value": MODEL_TYPE},
        {"metric": "evaluation", "value": "grouped_cross_fit_oof"},
        {"metric": "crossfit_folds", "value": N_CROSSFIT_FOLDS},
        {"metric": "crossfit_group_col", "value": ID_COL},
        {"metric": "period_hours", "value": feature_spec.get("period_hours")},
        {"metric": "all_rows", "value": int(len(df))},
        {"metric": "all_patients", "value": int(df[ID_COL].nunique())},
        {"metric": "decision_eligible_rows", "value": int(len(eligible_df))},
        {"metric": "decision_eligible_patients", "value": int(eligible_df[ID_COL].nunique())},
        {"metric": "explicit_feature_count", "value": int(len(feature_list))},
        {"metric": "oof_n", "value": summary["n"]},
        {"metric": "oof_events", "value": summary["events"]},
        {"metric": "oof_prevalence", "value": summary["prevalence"]},
        {"metric": "oof_auc", "value": summary["auc"]},
        {"metric": "oof_average_precision", "value": summary["average_precision"]},
        {"metric": "oof_brier", "value": summary["brier"]},
    ])


def performance_metrics_rows(propensity_summary, in_summary, out_summary):
    rows = []
    for metric, value in propensity_summary.items():
        rows.append({
            "model": "propensity",
            "model_type": MODEL_TYPE,
            "split": "grouped_cross_fit_oof",
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
        "risk_set_auc",
        "risk_set_brier",
    ]
    for summary_df in [in_summary, out_summary]:
        for row in summary_df.itertuples(index=False):
            for metric in metric_names:
                rows.append({
                    "model": row.model_group,
                    "model_type": row.model_type,
                    "split": "grouped_cross_fit_oof",
                    "outcome": row.outcome,
                    "metric": metric,
                    "value": getattr(row, metric),
                })
    return pd.DataFrame(rows)


# =============================================================================
# Propensity model: observed clinician removal behaviour among IN rows
# =============================================================================

def fit_propensity_scores(df, feature_spec):
    feature_list = feature_spec["features"]
    remove_feature_cols = feature_spec["x_cols_remove"]
    forbidden_action_features = {
        ACTION_COL,
        ACTION_REMOVE_COL,
        OBSERVED_ACTION_COL,
        "policy_action",
        "policy_action_remove",
    }
    leaked = sorted(set(remove_feature_cols) & forbidden_action_features)
    if leaked:
        raise ValueError(f"Propensity feature specification contains action columns: {leaked}")
    require_columns(
        df,
        [STATE_COL, CROSSFIT_FOLD_COL, ACTION_COL, *remove_feature_cols],
        "propensity",
    )

    print("Propensity model estimates observed clinician removal behaviour among IN rows.")
    print("Candidate policy actions are not propensity-model inputs.")
    print(f"Loaded propensity features: {len(remove_feature_cols)}")

    df = df.copy()
    df[PROPENSITY_SCORE_COL] = np.nan
    df[KEEP_PROPENSITY_SCORE_COL] = np.nan

    eligible_mask = df[STATE_COL].eq("in")
    eligible_df = df.loc[eligible_mask].copy()
    fold_models = []

    print(
        f"Cross-fitting removal propensity model across {N_CROSSFIT_FOLDS} "
        f"patient-grouped folds...",
        flush=True,
    )
    for fold in range(N_CROSSFIT_FOLDS):
        train_mask = eligible_mask & df[CROSSFIT_FOLD_COL].ne(fold)
        held_out_mask = eligible_mask & df[CROSSFIT_FOLD_COL].eq(fold)
        fold_model = fit_crossfit_fold_model(
            df.loc[train_mask, remove_feature_cols],
            df.loc[train_mask, ACTION_COL],
            "propensity_removal",
            fold,
        )
        held_out_index = df.index[held_out_mask]
        p_remove = predict_crossfit_fold(
            fold_model, df.loc[held_out_mask, remove_feature_cols]
        )
        df.loc[held_out_index, PROPENSITY_SCORE_COL] = p_remove
        df.loc[held_out_index, KEEP_PROPENSITY_SCORE_COL] = 1.0 - p_remove
        fold_models.append(fold_model)

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

    eval_oof = df.loc[eligible_mask].copy()
    summary = scalar_binary_metrics(eval_oof, ACTION_COL, PROPENSITY_SCORE_COL)
    save_df(
        propensity_summary_rows(df, eligible_df, feature_list, feature_spec, summary),
        OUTDIR / "propensity_summary.csv",
    )
    save_df(
        calibration_table(
            eval_oof,
            ACTION_COL,
            PROPENSITY_SCORE_COL,
            bins=CALIBRATION_BINS,
        ),
        OUTDIR / "propensity_calibration.csv",
    )
    dump_joblib(
        {
            "model_type": MODEL_TYPE,
            "evaluation": "grouped_cross_fit_oof",
            "crossfit_folds": N_CROSSFIT_FOLDS,
            "crossfit_group_col": ID_COL,
            "fallback_probability_source": "fold_training_rows_only",
            "fallback_smoothing": {
                "method": "beta_binomial",
                "prior_events": FALLBACK_PRIOR_EVENTS,
                "prior_non_events": FALLBACK_PRIOR_NON_EVENTS,
            },
            "remove_fold_models": fold_models,
            "features": feature_list,
            "x_cols_remove": remove_feature_cols,
            "target_col": ACTION_COL,
            "eligible_state": "in",
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "source_split_col_ignored": SPLIT_COL if SPLIT_COL in df.columns else None,
            "period_hours": feature_spec.get("period_hours"),
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "modeling_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
        },
        MODEL_DIR / "propensity_model.pkl",
    )

    print(f"Out-of-fold AUC removal: {summary['auc']}", flush=True)
    return df, fold_models, summary


# =============================================================================
# State-specific binary outcome nuisance models
# =============================================================================

def fit_outcome_scores(df, feature_spec):
    covariate_dict = pd.read_csv(COVARIATE_DICT_FILE)
    df = prepare_outcome_targets(df)
    in_feature_cols, out_feature_cols = state_feature_lists(feature_spec)
    require_columns(df, [CROSSFIT_FOLD_COL], "cross-fit")
    require_columns(df, in_feature_cols, "IN outcome-feature")
    require_columns(df, out_feature_cols, "OUT outcome-feature")

    print("IN outcome models estimate risks under keep/remove using action_remove.")
    print("OUT outcome models estimate post-removal risks, including reinsertion and 48-hour CAUTI attribution.")
    print("Removal is an action, not an outcome class.")
    print(f"IN outcome features: {len(in_feature_cols)}")
    print(f"OUT outcome features: {len(out_feature_cols)}")

    for col in ALL_SCORE_COLS[2:]:
        df[col] = np.nan

    in_models = {}
    out_models = {}
    in_summary_rows = []
    out_summary_rows = []
    in_calibration_tables = []
    out_calibration_tables = []
    importance_tables = []

    in_rows = df[STATE_COL].eq("in")
    out_rows = df[STATE_COL].eq("out")

    # IN models are fitted on observed keep/remove actions, then every eligible
    # held-out row is scored twice with all pre-decision features held fixed.
    for outcome, target_col in IN_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "in", outcome)
        modeling_df = df.loc[risk_mask].copy()

        model_name = f"in_{outcome}"
        keep_col = f"p_{outcome}_if_keep"
        remove_col = f"p_{outcome}_if_remove"
        if outcome == "cauti":
            # Any unexpected non-risk IN row contributes zero CAUTI risk.
            df.loc[in_rows & ~risk_mask, [keep_col, remove_col]] = 0.0

        print(f"Cross-fitting {model_name} binary outcome model...", flush=True)
        fold_models = []
        for fold in range(N_CROSSFIT_FOLDS):
            train_mask = risk_mask & df[CROSSFIT_FOLD_COL].ne(fold)
            held_out_mask = risk_mask & df[CROSSFIT_FOLD_COL].eq(fold)
            fold_model = fit_crossfit_fold_model(
                df.loc[train_mask, in_feature_cols],
                df.loc[train_mask, target_col],
                model_name,
                fold,
            )
            counterfactual_features = df.loc[
                held_out_mask, in_feature_cols
            ].copy()
            counterfactual_features[ACTION_REMOVE_COL] = 0
            df.loc[held_out_mask, keep_col] = predict_crossfit_fold(
                fold_model, counterfactual_features
            )
            counterfactual_features[ACTION_REMOVE_COL] = 1
            df.loc[held_out_mask, remove_col] = predict_crossfit_fold(
                fold_model, counterfactual_features
            )
            fold_models.append(fold_model)

        observed_pred_col = f"_p_{outcome}_observed_in"
        df[observed_pred_col] = np.nan
        df.loc[in_rows, observed_pred_col] = np.where(
            df.loc[in_rows, ACTION_REMOVE_COL].eq(1),
            df.loc[in_rows, remove_col],
            df.loc[in_rows, keep_col],
        )
        eval_df = df.loc[risk_mask].copy()
        risk_set = "IN and at_risk_cauti == 1" if outcome == "cauti" else "all IN rows"
        in_summary_rows.append(
            model_summary_row(
                "in_outcome",
                outcome,
                target_col,
                in_feature_cols,
                modeling_df,
                eval_df,
                observed_pred_col,
                risk_set,
            )
        )
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
        importance_tables.append(
            top_series_df(
                model_name,
                mean_feature_importance_series(fold_models, in_feature_cols),
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
        df.drop(columns=[observed_pred_col], inplace=True)

    # OUT models contain no action or state indicator. CAUTI is trained/scored
    # only during the 48-hour attribution window; non-risk OUT rows are zero.
    for outcome, target_col in OUT_OUTCOMES.items():
        risk_mask = outcome_risk_mask(df, "out", outcome)
        modeling_df = df.loc[risk_mask].copy()

        model_name = f"out_{outcome}"
        score_col = f"p_{outcome}_if_out"
        if outcome in {"cauti", "reinsertion"}:
            df.loc[out_rows & ~risk_mask, score_col] = 0.0

        print(f"Cross-fitting {model_name} binary outcome model...", flush=True)
        fold_models = []
        for fold in range(N_CROSSFIT_FOLDS):
            train_mask = risk_mask & df[CROSSFIT_FOLD_COL].ne(fold)
            held_out_mask = risk_mask & df[CROSSFIT_FOLD_COL].eq(fold)
            fold_model = fit_crossfit_fold_model(
                df.loc[train_mask, out_feature_cols],
                df.loc[train_mask, target_col],
                model_name,
                fold,
            )
            df.loc[held_out_mask, score_col] = predict_crossfit_fold(
                fold_model, df.loc[held_out_mask, out_feature_cols]
            )
            fold_models.append(fold_model)

        eval_df = df.loc[risk_mask].copy()
        if outcome == "cauti":
            risk_set = "OUT and at_risk_cauti == 1 (48-hour attribution window)"
        elif outcome == "reinsertion":
            risk_set = "OUT and at_risk_reinsertion == 1"
        else:
            risk_set = "all OUT rows"
        out_summary_rows.append(
            model_summary_row(
                "out_outcome",
                outcome,
                target_col,
                out_feature_cols,
                modeling_df,
                eval_df,
                score_col,
                risk_set,
            )
        )
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
        importance_tables.append(
            top_series_df(
                model_name,
                mean_feature_importance_series(fold_models, out_feature_cols),
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
    assert_valid_predictions(df, in_rows, in_required, "IN outcome rows")
    assert_valid_predictions(df, out_rows, out_required, "OUT outcome rows")

    in_summary = pd.DataFrame(in_summary_rows)
    out_summary = pd.DataFrame(out_summary_rows)
    save_df(in_summary, OUTDIR / "in_outcome_summary.csv")
    save_df(out_summary, OUTDIR / "out_outcome_summary.csv")
    # Retain the existing combined filename for callers that consume it.
    save_df(pd.concat([in_summary, out_summary], ignore_index=True), OUTDIR / "outcome_summary.csv")
    save_df(
        pd.concat(in_calibration_tables, ignore_index=True),
        OUTDIR / "in_outcome_calibration.csv",
    )
    save_df(
        pd.concat(out_calibration_tables, ignore_index=True),
        OUTDIR / "out_outcome_calibration.csv",
    )

    importance_df = add_feature_descriptions(
        pd.concat(importance_tables, ignore_index=True),
        covariate_dict,
    )
    save_df(importance_df, OUTDIR / "outcome_top_model_features.csv")

    dump_joblib(
        {
            "model_type": "state_specific_binary_xgb",
            "evaluation": "grouped_cross_fit_oof",
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
            "action_remove_col": ACTION_REMOVE_COL,
            "features": feature_spec["features"],
            "id_col": ID_COL,
            "time_col": TIME_COL,
            "source_split_col_ignored": SPLIT_COL if SPLIT_COL in df.columns else None,
            "period_hours": feature_spec.get("period_hours"),
            "post_remove_risk_periods": POST_REMOVE_RISK_PERIODS,
            "risk_set_columns": {"cauti": AT_RISK_CAUTI, "reinsertion": AT_RISK_REINS},
            "out_cauti_non_risk_prediction": 0.0,
            "fallback_fold_counts": outcome_fallback_counts(
                {"in": in_models, "out": out_models}
            ),
            "no_event_definition": NO_EVENT_DEFINITION,
            "no_event_is_transition_state": False,
            "modeling_panel_file": str(INFILE),
            "feature_spec_file": str(FEATURE_SPEC_FILE),
        },
        MODEL_DIR / "outcome_models.pkl",
    )

    for summary_row in [*in_summary_rows, *out_summary_rows]:
        print(
            f"{summary_row['model_group']} {summary_row['outcome']}: "
            f"n={summary_row['n']}, events={summary_row['events']}, "
            f"AUC={summary_row['auc']}, Brier={summary_row['brier']}",
            flush=True,
        )
    return df, {"in": in_models, "out": out_models}, in_summary, out_summary


# =============================================================================
# Final panel assembly
# =============================================================================

def save_scored_panel(df):
    drop_cols = [
        col
        for col in [
            *OLD_TRANSITION_SCORE_COLS,
            Y_ICU_EXIT_ALIVE,
            Y_NO_EVENT_IN,
            Y_NO_EVENT_OUT,
        ]
        if col in df.columns
    ]
    final_df = df.drop(columns=drop_cols)
    final_df = insert_score_columns_before_age(final_df, ALL_SCORE_COLS)
    final_df.to_csv(FINAL_PANEL, index=False, float_format="%.6f")
    return final_df


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    feature_spec = load_feature_spec()
    df = load_panel()
    df = add_grouped_crossfit_folds(df)
    save_df(crossfit_fold_summary(df), OUTDIR / "crossfit_fold_summary.csv")
    save_df(crossfit_row_assignments(df), CROSSFIT_ROW_ASSIGNMENTS_FILE)

    df, _, propensity_summary = fit_propensity_scores(df, feature_spec)
    df, outcome_models, in_summary, out_summary = fit_outcome_scores(df, feature_spec)
    validate_exported_probabilities(df)
    save_df(
        performance_metrics_rows(propensity_summary, in_summary, out_summary),
        PERFORMANCE_METRICS_FILE,
    )
    save_scored_panel(df)

    print("\n--- SUCCESS NUISANCE MODEL FIT ---", flush=True)
    print(f"Final scored panel saved: {FINAL_PANEL}", flush=True)
    print(f"Propensity model saved: {MODEL_DIR / 'propensity_model.pkl'}", flush=True)
    print(f"Outcome models saved: {MODEL_DIR / 'outcome_models.pkl'}", flush=True)
    print(f"Propensity summary saved: {OUTDIR / 'propensity_summary.csv'}", flush=True)
    print(f"IN outcome summary saved: {OUTDIR / 'in_outcome_summary.csv'}", flush=True)
    print(f"OUT outcome summary saved: {OUTDIR / 'out_outcome_summary.csv'}", flush=True)
    print(f"Performance metrics saved: {PERFORMANCE_METRICS_FILE}", flush=True)
    print(f"Outcome top features saved: {OUTDIR / 'outcome_top_model_features.csv'}", flush=True)
    print(f"Cross-fit fold summary saved: {OUTDIR / 'crossfit_fold_summary.csv'}", flush=True)
    print("\n--- CROSS-FIT SUMMARY ---", flush=True)
    print(f"Folds: {N_CROSSFIT_FOLDS}", flush=True)
    print(f"Patients: {df[ID_COL].nunique()}", flush=True)
    fallback_counts = outcome_fallback_counts(outcome_models)
    print(
        "Fallback folds by outcome: "
        + ", ".join(f"{name}={count}" for name, count in fallback_counts.items()),
        flush=True,
    )
    print(f"Final scored panel: {FINAL_PANEL}", flush=True)
    print(f"Cross-fit row assignments: {CROSSFIT_ROW_ASSIGNMENTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
