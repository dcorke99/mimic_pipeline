from pathlib import Path

import numpy as np
import pandas as pd

import policy_eval_common as pec
import policy_bootstrap_common as bootstrap
import fit_nuisance_models as nuisance
from build_policy_panels import read_policy_collection
from policy_eval_common import (
    cumulative_event_probability,
    assign_policy_prediction,
    PREDICTION_COLUMNS,
    add_episode_day,
    fill_missing_counterfactual_predictions,
)


# Configuration

REPO_ROOT = Path(__file__).resolve().parent
NUISANCE_MODEL_TYPE = "xgboost"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260923
REFIT_CROSSFIT_FOLDS = 5
BOOTSTRAP_MODES = ("fixed",)  # Use ("fixed", "refit") to run both modes.


# Panels to process in this script only; comment out entries to skip them.
VALIDATION_DIR = REPO_ROOT / "artefacts/semi-synthetic_validation/semi_synthetic_measured_confounding"
PANEL_RUNS = (
    ("real", REPO_ROOT / "data/modelling_panel.csv", REPO_ROOT / "artefacts/nuisance_models"),
    ("semi_synthetic_with_confounding", VALIDATION_DIR / "semi_synthetic_panel.csv",
     VALIDATION_DIR / "pipeline_runs/semi_synthetic_with_confounding/nuisance_models"),
    ("confounder_omitted", VALIDATION_DIR / "semi_synthetic_panel_confounder_omitted.csv",
     VALIDATION_DIR / "pipeline_runs/confounder_omitted/nuisance_models"),
    ("randomised_action", VALIDATION_DIR / "semi_synthetic_panel_randomised_action.csv",
     VALIDATION_DIR / "pipeline_runs/randomised_action/nuisance_models"),
)

OUTPUT_FILENAMES = {
    "summary": "gformula_policy_outcomes_summary.csv",
    "episodes": "gformula_episode_predictions.csv",
    "diagnostics": "gformula_diagnostics.csv",
    "current_practice": "current_practice_gformula_episode_predictions.csv",
}

CURRENT_PRACTICE_LABEL = "current_practice"
ESTIMATOR_NAME = "plugin_gformula"
PREDICTION_MODE = "observed_grid_plugin"
POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS = 2

EPISODE_ID_COL = "catheter_episode_id"
POLICY_TYPE_COL = "policy_type"

EPISODE_KEY_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
]

ROW_JOIN_KEY_COLS = [
    *EPISODE_KEY_COLS,
    "period_start",
    "period_end",
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "removed_in_period",
]


NUISANCE_COLUMNS = [
    *PREDICTION_COLUMNS,
    "cauti_in_period",
    "reinsertion_in_period",
    "death_in_period",
    "icu_exit_alive_in_period",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "episode_end_reason",
    "reinsertion_time",
    "_crossfit_fold",
]

EPISODE_FIRST_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
    "reinsertion_time",
    "_crossfit_fold",
    "episode_end_reason",
]

UNDER_POLICY_COLUMNS = [
    "p_cauti_under_policy",
    "p_recatheterisation_under_policy",
    "p_death_under_policy",
    "p_icu_exit_alive_under_policy",
    "p_no_event_under_policy",
]

EPISODE_PREDICTION_SPECS = {
    "predicted_any_cauti": "p_cauti_under_policy",
    "predicted_any_recatheterisation": "p_recatheterisation_under_policy",
    "predicted_any_death": "p_death_under_policy",
    "predicted_icu_exit_alive": "p_icu_exit_alive_under_policy",
}

MISSING_COUNTERFACTUAL_MESSAGE = (
    "Missing counterfactual state/action predictions were found. This matters "
    "because plug-in g-formula uses every episode under every target policy, so "
    "a row observed in one catheter state may require predictions for another "
    "policy-implied state. Re-run nuisance scoring with complete state/action "
    "counterfactual predictions or provide outcome_models.pkl for rescoring."
)


# Generic helpers


def join_nuisance_predictions(policy_df, nuisance_df):
    nuisance_add_cols = [
        col
        for col in [
            *NUISANCE_COLUMNS,
            *[f"__rescored_{col}" for col in PREDICTION_COLUMNS],
        ]
        if col not in ROW_JOIN_KEY_COLS
    ]
    merged = policy_df.merge(
        nuisance_df[[*ROW_JOIN_KEY_COLS, *nuisance_add_cols]],
        on=ROW_JOIN_KEY_COLS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, ROW_JOIN_KEY_COLS + ["policy_name"]].head(10)
        raise ValueError(
            "Some policy-panel rows did not match the nuisance predictions on "
            f"the natural keys. Examples:\n{examples}"
        )

    return merged.drop(columns="_merge")


# Prediction selection and validation


def select_policy_predictions(df):
    # Select predictions implied by the target policy
    for col in UNDER_POLICY_COLUMNS:
        df[col] = np.nan
    df["__used_rescored_prediction"] = False

    keep_rows = df["policy_catheter_state"].eq("in") & df["policy_action"].eq("keep")
    remove_rows = df["policy_catheter_state"].eq("in") & df["policy_action"].eq("remove")
    out_rows = df["policy_catheter_state"].eq("out")
    out_cauti_rows = out_rows & pd.to_numeric(df["policy_periods_out"], errors="coerce").le(
        POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
    )

    assign_policy_prediction(df, "p_cauti_under_policy", "p_cauti_if_keep", keep_rows)
    assign_policy_prediction(df, "p_death_under_policy", "p_death_if_keep", keep_rows)
    assign_policy_prediction(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_keep", keep_rows)
    assign_policy_prediction(df, "p_no_event_under_policy", "p_no_event_if_keep", keep_rows)
    df.loc[keep_rows, "p_recatheterisation_under_policy"] = 0.0

    assign_policy_prediction(df, "p_cauti_under_policy", "p_cauti_if_remove", remove_rows)
    assign_policy_prediction(df, "p_death_under_policy", "p_death_if_remove", remove_rows)
    assign_policy_prediction(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_remove", remove_rows)
    assign_policy_prediction(df, "p_no_event_under_policy", "p_no_event_if_remove", remove_rows)
    df.loc[remove_rows, "p_recatheterisation_under_policy"] = 0.0

    df.loc[out_rows, "p_cauti_under_policy"] = 0.0
    assign_policy_prediction(df, "p_cauti_under_policy", "p_cauti_if_out", out_cauti_rows)
    assign_policy_prediction(df, "p_recatheterisation_under_policy", "p_reinsertion_if_out", out_rows)
    assign_policy_prediction(df, "p_death_under_policy", "p_death_if_out", out_rows)
    assign_policy_prediction(df, "p_icu_exit_alive_under_policy", "p_icu_exit_alive_if_out", out_rows)
    assign_policy_prediction(df, "p_no_event_under_policy", "p_no_event_if_out", out_rows)

    missing_any = df[UNDER_POLICY_COLUMNS].isna().any(axis=1)
    invalid_any = pd.Series(False, index=df.index)
    for col in UNDER_POLICY_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_any |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))

    df["prediction_source"] = np.where(
        df["__used_rescored_prediction"],
        "nuisance_predictions_plus_outcome_model_rescore",
        "nuisance_predictions",
    )
    df["prediction_status"] = "complete"
    df.loc[missing_any, "prediction_status"] = "missing_prediction"
    df.loc[invalid_any, "prediction_status"] = "invalid_probability"
    return df


def validate_prediction_completeness(df):
    # Check prediction completeness and probability bounds
    missing_counts = df.groupby("policy_name", dropna=False)[UNDER_POLICY_COLUMNS].apply(
        lambda frame: frame.isna().sum()
    )
    total_missing = int(df[UNDER_POLICY_COLUMNS].isna().sum().sum())
    invalid_rows = pd.Series(False, index=df.index)
    for col in UNDER_POLICY_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_rows |= numeric.notna() & (~np.isfinite(numeric) | numeric.lt(0) | numeric.gt(1))

    if invalid_rows.any():
        examples = df.loc[
            invalid_rows,
            ["policy_name", "decision_row_id", *UNDER_POLICY_COLUMNS],
        ].head(10)
        raise ValueError(
            "G-formula predictions must be finite probabilities between 0 and 1 "
            f"where present. Examples:\n{examples}"
        )

    if total_missing:
        raise ValueError(
            f"{MISSING_COUNTERFACTUAL_MESSAGE}\nMissing prediction counts by policy:\n{missing_counts}"
        )


# Current-practice model-based comparator

def map_episode_ids_to_nuisance_predictions(nuisance_df, policy_df):
    # Map episode identifiers onto nuisance-prediction rows
    episode_map = policy_df[[*EPISODE_KEY_COLS, EPISODE_ID_COL]].drop_duplicates()
    out = nuisance_df.merge(
        episode_map,
        on=EPISODE_KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    if out[EPISODE_ID_COL].isna().any():
        examples = out.loc[out[EPISODE_ID_COL].isna(), EPISODE_KEY_COLS].head(10)
        raise ValueError(
            "Some nuisance-prediction episodes could not be mapped. "
            f"Examples:\n{examples}"
        )
    return out


def build_current_practice_rows(nuisance_df, policy_df):
    # Build rows for the observed current-practice regime
    # Map episode identifiers onto nuisance-prediction rows
    df = map_episode_ids_to_nuisance_predictions(nuisance_df, policy_df)
    df = pec.add_period_duration_days(df, context="current-practice g-formula rows")
    # Add episode day since catheter insertion
    df = add_episode_day(df)
    df["policy_name"] = CURRENT_PRACTICE_LABEL
    df[POLICY_TYPE_COL] = "observed"
    df["policy_remove_day"] = pd.NA
    df["policy_catheter_state"] = df["catheter_state"].astype("string").str.lower()
    df["policy_action"] = "out"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(0),
        "policy_action",
    ] = "keep"
    df.loc[
        df["policy_catheter_state"].eq("in")
        & pd.to_numeric(df["removed_in_period"], errors="coerce").eq(1),
        "policy_action",
    ] = "remove"
    df["policy_periods_in"] = np.where(df["policy_catheter_state"].eq("in"), df["periods_in_state"], np.nan)
    df["policy_periods_out"] = np.where(df["policy_catheter_state"].eq("out"), df["periods_in_state"], np.nan)
    return df


# Episode and policy-level aggregation

def add_observed_crude_episode_outcomes(episode_df, row_df):
    # Add observed episode outcomes to predictions
    row_df = pec.add_observed_icu_exit_alive_period(row_df)
    outcome_cols = [
        ("observed_any_cauti", "cauti_in_period"),
        ("observed_any_recatheterisation", "reinsertion_in_period"),
        ("observed_any_death", "death_in_period"),
        ("observed_icu_exit_alive", "observed_icu_exit_alive_in_period"),
    ]
    for _, source in outcome_cols:
        row_df[source] = pd.to_numeric(row_df[source], errors="coerce").gt(0).astype(int)
    aggs = {source: "max" for _, source in outcome_cols}
    crude = row_df.groupby([EPISODE_ID_COL], as_index=False, dropna=False).agg(aggs)
    crude = crude.rename(columns={source: target for target, source in outcome_cols})
    return episode_df.merge(crude, on=EPISODE_ID_COL, how="left")


def build_episode_predictions(row_df):
    # Collapse row predictions to episode predictions
    row_df = row_df.copy()
    row_df["_policy_catheter_in_row_int"] = row_df["policy_catheter_state"].astype("string").str.lower().eq("in").astype(int)
    row_df["_policy_catheter_exposure_days"] = row_df["_policy_catheter_in_row_int"] * pd.to_numeric(
        row_df["period_duration_days"],
        errors="coerce",
    )
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day", EPISODE_ID_COL]
    # pandas named aggregation is clearer here than building custom apply rows
    base = row_df.groupby(group_cols, as_index=False, dropna=False).agg(
        prediction_complete=("prediction_status", lambda s: bool(s.eq("complete").all())),
        n_policy_rows_used=("prediction_status", "size"),
        n_missing_prediction_rows=("prediction_status", lambda s: int(s.ne("complete").sum())),
        expected_catheter_in_interval_rows=("_policy_catheter_in_row_int", "sum"),
        expected_catheter_exposure_days=("_policy_catheter_exposure_days", "sum"),
    )

    for col in EPISODE_FIRST_COLS:
        values = row_df.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[col].first()
        base = base.merge(values, on=group_cols, how="left")

    for episode_col, row_col in EPISODE_PREDICTION_SPECS.items():
        values = row_df.groupby(group_cols, as_index=False, dropna=False)[row_col].agg(cumulative_event_probability)
        values = values.rename(columns={row_col: episode_col})
        base = base.merge(values, on=group_cols, how="left")

    # Order episode-level output columns
    return order_episode_columns(base)


def order_episode_columns(df):
    # Order episode-level output columns
    preferred = [
        "subject_id",
        "hadm_id",
        "stay_id",
        EPISODE_ID_COL,
        "inserted",
        "removed",
        "policy_name",
        POLICY_TYPE_COL,
        "policy_remove_day",
        "predicted_any_cauti",
        "predicted_any_recatheterisation",
        "predicted_any_death",
        "predicted_icu_exit_alive",
        "expected_catheter_in_interval_rows",
        "expected_catheter_exposure_days",
        "n_policy_rows_used",
        "n_missing_prediction_rows",
        "prediction_complete",
        "_crossfit_fold",
        "episode_end_reason",
        "reinsertion_time",
    ]
    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


def build_policy_summary(
    episode_df,
    prediction_mode,
):
    # Build policy-level summary estimates
    rows = []
    group_cols = ["policy_name", POLICY_TYPE_COL, "policy_remove_day"]
    for policy_values, policy_df in episode_df.groupby(group_cols, dropna=False, sort=False):
        policy_name, policy_type, policy_remove_day = policy_values
        complete_df = policy_df.loc[policy_df["prediction_complete"].astype(bool)].copy()
        row = {
            "policy_name": policy_name,
            POLICY_TYPE_COL: policy_type,
            "policy_remove_day": policy_remove_day,
            "estimator": ESTIMATOR_NAME,
            "prediction_mode": prediction_mode,
            "n_patients": int(policy_df["subject_id"].nunique()),
            "n_episodes": int(len(policy_df)),
            "n_complete_prediction_episodes": int(len(complete_df)),
            "n_incomplete_prediction_episodes": int(len(policy_df) - len(complete_df)),
            "predicted_cauti_risk": float(complete_df["predicted_any_cauti"].mean()) if len(complete_df) else np.nan,
            "predicted_recatheterisation_risk": float(complete_df["predicted_any_recatheterisation"].mean()) if len(complete_df) else np.nan,
            "predicted_death_risk": float(complete_df["predicted_any_death"].mean()) if len(complete_df) else np.nan,
            "predicted_icu_exit_alive_risk": float(complete_df["predicted_icu_exit_alive"].mean()) if len(complete_df) else np.nan,
            "expected_mean_catheter_in_interval_rows": float(complete_df["expected_catheter_in_interval_rows"].mean()) if len(complete_df) else np.nan,
            "expected_mean_catheter_exposure_days": float(complete_df["expected_catheter_exposure_days"].mean()) if len(complete_df) else np.nan,
        }
        for col in [
            "predicted_cauti_risk",
            "predicted_recatheterisation_risk",
            "predicted_death_risk",
            "predicted_icu_exit_alive_risk",
        ]:
            row[f"{col}_pct"] = row[col] * 100 if pd.notna(row[col]) else np.nan
        rows.append(row)
    # Add comparisons against current practice
    return add_current_practice_comparisons(pd.DataFrame(rows))


def add_current_practice_comparisons(summary):
    # Add comparisons against current practice
    return pec.add_standard_comparisons(
        summary,
        baseline_label=CURRENT_PRACTICE_LABEL,
        comparison_map={
            "cauti_risk": "predicted_cauti_risk",
            "recatheterisation_risk": "predicted_recatheterisation_risk",
            "death_risk": "predicted_death_risk",
            "icu_exit_alive_risk": "predicted_icu_exit_alive_risk",
            "catheter_exposure_days": "expected_mean_catheter_exposure_days",
        },
    )


# Diagnostics

def prediction_bounds(df, col):
    # Summarise prediction bounds for one column
    values = pd.to_numeric(df[col], errors="coerce")
    present = values.dropna()
    return {
        f"min_{col}": float(present.min()) if len(present) else np.nan,
        f"max_{col}": float(present.max()) if len(present) else np.nan,
        f"n_{col}_below_0": int(present.lt(0).sum()) if len(present) else 0,
        f"n_{col}_above_1": int(present.gt(1).sum()) if len(present) else 0,
    }


def build_diagnostics(row_df, episode_df):
    # Build diagnostic rows for policy outputs
    rows = []
    for policy_name, policy_df in row_df.groupby("policy_name", dropna=False, sort=False):
        policy_episode_df = episode_df.loc[episode_df["policy_name"].eq(policy_name)]
        removal_days = policy_df["policy_remove_day"].dropna()
        policy_remove_day = removal_days.iloc[0] if len(removal_days) else np.nan
        numeric_remove_day = pd.to_numeric(pd.Series([policy_remove_day]), errors="coerce").iloc[0]
        if pd.notna(numeric_remove_day):
            too_many_policy_in = policy_episode_df["expected_catheter_in_interval_rows"].gt(numeric_remove_day)
        else:
            too_many_policy_in = pd.Series(False, index=policy_episode_df.index)
        remove_rows_by_episode = (
            policy_df["policy_action"].eq("remove")
            .groupby(policy_df[EPISODE_ID_COL], sort=False)
            .sum()
        )
        if pd.notna(numeric_remove_day):
            reaches_policy_removal_day = (
                pd.to_numeric(policy_df["episode_day"], errors="coerce")
                .eq(numeric_remove_day)
                .groupby(policy_df[EPISODE_ID_COL], sort=False)
                .max()
            )
            n_episodes_reaching_policy_removal_day = int(reaches_policy_removal_day.sum())
        else:
            n_episodes_reaching_policy_removal_day = 0
        n_policy_remove_rows = int(
            policy_df["policy_action"].eq("remove").sum()
        )
        fixed_day_policy = pd.notna(numeric_remove_day) and policy_name != CURRENT_PRACTICE_LABEL
        policy_remove_row_shortfall = (
            int(n_episodes_reaching_policy_removal_day - n_policy_remove_rows)
            if fixed_day_policy
            else pd.NA
        )
        policy_types = policy_df[POLICY_TYPE_COL].dropna()
        row = {
            "policy_name": policy_name,
            POLICY_TYPE_COL: policy_types.iloc[0] if len(policy_types) else np.nan,
            "policy_remove_day": policy_remove_day,
            "n_rows": int(len(policy_df)),
            "n_episodes": int(policy_df[EPISODE_ID_COL].nunique()),
            "n_patients": int(policy_df["subject_id"].nunique()),
            "n_policy_in_rows": int(policy_df["policy_catheter_state"].eq("in").sum()),
            "n_policy_remove_rows": n_policy_remove_rows,
            "n_episodes_reaching_policy_removal_day": n_episodes_reaching_policy_removal_day,
            "n_policy_remove_row_shortfall_vs_reached_episodes": policy_remove_row_shortfall,
            "n_episodes_with_more_than_one_remove_row": int(remove_rows_by_episode.gt(1).sum()),
            "n_policy_removal_day_extra_rows_treated_as_out": int(
                policy_df["policy_removal_day_extra_row_treated_as_out"]
                .fillna(0)
                .sum()
            ),
            "n_policy_out_rows": int(policy_df["policy_catheter_state"].eq("out").sum()),
            "n_duplicate_policy_episode_day_rows": pec.duplicate_episode_day_count(
                policy_df,
                ["policy_name", EPISODE_ID_COL],
            ),
            "n_episodes_with_more_policy_in_intervals_than_expected": int(too_many_policy_in.sum()),
            "n_invalid_period_duration_rows": 0,
            "n_long_period_duration_rows": int(
                policy_df["period_duration_long_flag"].sum()
            ),
            "n_missing_cauti_predictions": int(policy_df["p_cauti_under_policy"].isna().sum()),
            "n_missing_recatheterisation_predictions": int(policy_df["p_recatheterisation_under_policy"].isna().sum()),
            "n_missing_death_predictions": int(policy_df["p_death_under_policy"].isna().sum()),
            "n_missing_icu_exit_predictions": int(policy_df["p_icu_exit_alive_under_policy"].isna().sum()),
            "n_missing_no_event_predictions": int(policy_df["p_no_event_under_policy"].isna().sum()),
            "n_complete_prediction_episodes": int(policy_episode_df["prediction_complete"].astype(bool).sum()) if len(policy_episode_df) else 0,
            "n_incomplete_prediction_episodes": int((~policy_episode_df["prediction_complete"].astype(bool)).sum()) if len(policy_episode_df) else 0,
            "mean_expected_catheter_in_interval_rows": float(policy_episode_df["expected_catheter_in_interval_rows"].mean()) if len(policy_episode_df) else np.nan,
            "mean_expected_catheter_exposure_days": float(policy_episode_df["expected_catheter_exposure_days"].mean()) if len(policy_episode_df) else np.nan,
            "uses_policy_matching": False,
            "uses_ipw_weights": False,
            "uses_observed_grid": True,
            "full_longitudinal_covariate_simulation": False,
            "covariate_propagation_method": PREDICTION_MODE,
        }
        total_below = 0
        total_above = 0
        # Summarise prediction bounds for one column
        for col in UNDER_POLICY_COLUMNS:
            # Summarise prediction bounds for one column
            bounds = prediction_bounds(policy_df, col)
            row.update(bounds)
            total_below += bounds[f"n_{col}_below_0"]
            total_above += bounds[f"n_{col}_above_1"]
        row["n_predictions_below_0"] = total_below
        row["n_predictions_above_1"] = total_above
        rows.append(row)
    return pd.DataFrame(rows)


# Console summary and main

def print_console_summary(
    summary_df,
    row_df,
    episode_df,
    output_paths,
    policy_manifest_path,
    model_type,
):
    print()
    print("--- G-FORMULA POLICY EVALUATION COMPLETE ---")
    print(f"Policy panel index: {policy_manifest_path}")
    print(f"Nuisance model type: {model_type}")
    print(f"Number of policies: {row_df['policy_name'].nunique():,}")
    print(f"Number of patients: {row_df['subject_id'].nunique():,}")
    print(f"Number of episodes: {row_df[EPISODE_ID_COL].nunique():,}")
    print(f"Complete prediction episodes: {int(episode_df['prediction_complete'].astype(bool).sum()):,}")
    print()
    display_cols = [
        "policy_name",
        "policy_remove_day",
        "predicted_cauti_risk_pct",
        "predicted_recatheterisation_risk_pct",
        "expected_mean_catheter_exposure_days",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print()
    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")


def evaluate_policy_episodes(policy_df, nuisance_df):
    model_feature_cols = [
        "episode_index",
        *[col for col in nuisance_df if col == "age" or col.startswith(("itemid_", "sex_", "ethnicity_"))],
    ]
    nuisance_df = nuisance_df.drop(columns=model_feature_cols)

    # Join nuisance scores to policy rows
    joined_df = join_nuisance_predictions(policy_df, nuisance_df)
    joined_df = pec.add_period_duration_days(joined_df, context="joined g-formula policy rows")
    # Select predictions implied by the target policy
    row_df = select_policy_predictions(joined_df)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(row_df)

    # Collapse row predictions to episode predictions
    episode_df = build_episode_predictions(row_df)
    # Build rows for the observed current-practice regime
    current_rows = build_current_practice_rows(nuisance_df, policy_df)
    # Select predictions implied by the target policy
    current_rows = select_policy_predictions(current_rows)
    # Check prediction completeness and probability bounds
    validate_prediction_completeness(current_rows)
    # Collapse row predictions to episode predictions
    current_episode_df = build_episode_predictions(current_rows)
    # Add observed episode outcomes to predictions
    current_episode_df = add_observed_crude_episode_outcomes(current_episode_df, current_rows)

    combined_episode_df = pd.concat([episode_df, current_episode_df], ignore_index=True, sort=False)
    return combined_episode_df, row_df, current_rows, current_episode_df


def run_panel_estimation(panel_path, artefact_root, model_type, bootstrap_mode, panel_name=None):
    if bootstrap_mode not in ("fixed", "refit"):
        raise ValueError(f"Unknown bootstrap mode: {bootstrap_mode!r}")
    policy_manifest_path = artefact_root / "counterfactual_policies/policy_panels.csv"
    nuisance_model_dir = artefact_root / "nuisance_models" / model_type
    nuisance_predictions_path = nuisance_model_dir / "nuisance_predictions.csv"
    outcome_models_path = nuisance_model_dir / "outcome_models.pkl"
    outdir = artefact_root / "policy_eval/gformula" / bootstrap_mode
    output_paths = {key: outdir / name for key, name in OUTPUT_FILENAMES.items()}
    outdir.mkdir(exist_ok=True, parents=True)
    policy_df = read_policy_collection(policy_manifest_path)
    policy_df["subject_id"] = policy_df.subject_id.astype(str)
    refit_panel = None
    if bootstrap_mode == "refit":
        refit_panel = nuisance.load_panel(panel_path)
        subjects = pd.Index(sorted(refit_panel.subject_id.unique()), name="subject_id")
        nuisance_df, _ = bootstrap.refit_nuisance_predictions(
            refit_panel, subjects, np.ones(len(subjects), dtype=int), "gformula",
            n_splits=REFIT_CROSSFIT_FOLDS, model_type=model_type,
        )
    else:
        nuisance_df = pd.read_csv(nuisance_predictions_path, low_memory=False)
        nuisance_df.columns = nuisance_df.columns.str.strip()
        nuisance_df = fill_missing_counterfactual_predictions(nuisance_df, outcome_models_path)
    nuisance_df["subject_id"] = nuisance_df.subject_id.astype(str)
    combined_episode_df, row_df, current_rows, current_episode_df = evaluate_policy_episodes(
        policy_df, nuisance_df,
    )

    # Build policy-level summary estimates
    summary_df = build_policy_summary(combined_episode_df, PREDICTION_MODE)
    # Build diagnostic rows for policy outputs
    diagnostics_df = build_diagnostics(
        pd.concat([row_df, current_rows], ignore_index=True, sort=False),
        combined_episode_df,
    )

    # Save rounded policy-level report outputs
    pec.save_report_df(summary_df, output_paths["summary"])
    # Save episode-level data at full precision for later inference
    combined_episode_df.to_csv(output_paths["episodes"], index=False)
    # Save rounded diagnostics
    pec.save_report_df(diagnostics_df, output_paths["diagnostics"])
    # Save current-practice episode data at full precision
    current_episode_df.to_csv(output_paths["current_practice"], index=False)

    print_console_summary(summary_df, row_df, combined_episode_df, output_paths, policy_manifest_path, model_type)

    bootstrap.run_bootstrap(
        "gformula", combined_episode_df, policy_df, evaluate_policy_episodes, outdir,
        N_BOOTSTRAP, refit_panel=refit_panel, seed=BOOTSTRAP_SEED,
        refit_n_splits=REFIT_CROSSFIT_FOLDS, model_type=model_type,
        panel_name=panel_name or panel_path.stem,
    )


def main():
    for panel_name, panel_path, nuisance_root in PANEL_RUNS:
        for mode in BOOTSTRAP_MODES:
            print(f"[PANEL] {panel_name}; estimator=gformula; bootstrap={mode}", flush=True)
            run_panel_estimation(panel_path, nuisance_root.parent, NUISANCE_MODEL_TYPE, mode,
                                 panel_name=panel_name)


if __name__ == "__main__":
    main()
