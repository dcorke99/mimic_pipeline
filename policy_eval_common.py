# Share estimator-agnostic helpers for policy-evaluation scripts


import joblib
import numpy as np
import pandas as pd


MAX_REASONABLE_PERIOD_DURATION_DAYS = 7.0
LOW_ADHERENCE_THRESHOLD = 0.05
LOW_ESS_MIN = 100.0
LOW_ESS_FRACTION = 0.10
LOW_SUPPORT_PCT_BELOW_005_THRESHOLD = 0.10
EXTREME_WEIGHT_P99_THRESHOLD = 30.0
EXTREME_WEIGHT_MAX_THRESHOLD = 100.0


def save_report_df(df, path, decimals=3):
    # Round report values without changing source data
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


def baseline_model_feature_columns(columns):
    # Return baseline and chart covariates in their existing order
    return [
        column
        for column in columns
        if column == "age"
        or str(column).startswith(("itemid_", "sex_", "ethnicity_"))
    ]


def add_period_duration_days(
    df,
    *,
    context,
    max_reasonable_days=MAX_REASONABLE_PERIOD_DURATION_DAYS,
):
    # Add period duration in days
    out = df.copy()
    start = pd.to_datetime(out["period_start"], errors="coerce")
    end = pd.to_datetime(out["period_end"], errors="coerce")
    duration_days = (end - start).dt.total_seconds() / 86400.0
    invalid = start.isna() | end.isna() | duration_days.isna() | ~np.isfinite(duration_days) | duration_days.le(0)
    if invalid.any():
        examples = out.loc[invalid, ["period_start", "period_end"]].head(10)
        raise ValueError(
            f"Invalid period durations found while preparing {context}. "
            "period_start/period_end must parse and period_end must be after period_start. "
            f"Examples:\n{examples}"
        )

    out["period_duration_days"] = duration_days.astype(float)
    out["period_duration_long_flag"] = out["period_duration_days"].gt(max_reasonable_days).astype(int)
    n_long = int(out["period_duration_long_flag"].sum())
    if n_long:
        print(
            f"WARNING: {context} has {n_long:,} rows with period_duration_days "
            f"> {max_reasonable_days}. They are retained and flagged.",
            flush=True,
        )
    return out


def add_observed_icu_exit_alive_period(df):
    # Validate and expose the mutually exclusive ICU-exit-alive outcome
    out = df.copy()
    death = pd.to_numeric(out["death_in_period"], errors="coerce").fillna(0)
    icu_exit_alive = pd.to_numeric(
        out["icu_exit_alive_in_period"],
        errors="coerce",
    ).fillna(0)
    overlap = death.eq(1) & icu_exit_alive.eq(1)
    if overlap.any():
        raise ValueError(
            "death_in_period and icu_exit_alive_in_period must be mutually exclusive"
        )
    out["observed_icu_exit_alive_in_period"] = icu_exit_alive.eq(1).astype(int)
    return out


def add_fixed_day_target_policy_timeline(
    df,
    *,
    episode_id_col,
    action_col="policy_action_resolved",
    action_remove_col="policy_action_remove_resolved",
    policy_name_col="policy_name",
    policy_remove_day_col="policy_remove_day",
    episode_day_col="episode_day_since_insertion",
    period_start_col="period_start",
    period_end_col="period_end",
    decision_row_id_col="decision_row_id",
    state_col="policy_catheter_state",
    periods_in_col="policy_periods_in",
    periods_out_col="policy_periods_out",
):
    # Add the resolved fixed-day policy timeline
    out = df.copy()
    out = out.sort_values(
        [
            policy_name_col,
            episode_id_col,
            episode_day_col,
            period_start_col,
            period_end_col,
            decision_row_id_col,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    out["row_order_within_episode_day"] = (
        out.groupby(
            [policy_name_col, episode_id_col, episode_day_col],
            dropna=False,
            sort=False,
        ).cumcount()
        + 1
    )

    day = pd.to_numeric(out[episode_day_col], errors="coerce")
    remove_day = pd.to_numeric(out[policy_remove_day_col], errors="coerce")
    before_remove = day.lt(remove_day)
    on_remove = day.eq(remove_day)
    first_row_on_remove_day = on_remove & out["row_order_within_episode_day"].eq(1)
    later_row_on_remove_day = on_remove & out["row_order_within_episode_day"].gt(1)
    after_remove = day.gt(remove_day)

    out[state_col] = "out"
    out.loc[before_remove | first_row_on_remove_day, state_col] = "in"

    out[action_col] = "out"
    out.loc[before_remove, action_col] = "keep"
    out.loc[first_row_on_remove_day, action_col] = "remove"

    out[action_remove_col] = np.nan
    out.loc[before_remove, action_remove_col] = 0.0
    out.loc[first_row_on_remove_day, action_remove_col] = 1.0

    out[periods_in_col] = np.nan
    out.loc[before_remove | first_row_on_remove_day, periods_in_col] = day.loc[
        before_remove | first_row_on_remove_day
    ]

    out[periods_out_col] = np.nan
    out.loc[later_row_on_remove_day, periods_out_col] = 0.0
    out.loc[after_remove, periods_out_col] = day.loc[after_remove] - remove_day.loc[after_remove]
    out["policy_removal_day_extra_row_treated_as_out"] = later_row_on_remove_day.astype(int)
    return out


def resolved_timeline_diagnostics(
    df,
    *,
    episode_id_col="catheter_episode_id",
    policy_name_col="policy_name",
    policy_remove_day_col="policy_remove_day",
    episode_day_col="episode_day_since_insertion",
):
    # Summarise resolved timeline diagnostics
    rows = []
    for policy_name, policy_df in df.groupby(policy_name_col, dropna=False, sort=False):
        remove_day = pd.to_numeric(policy_df[policy_remove_day_col], errors="coerce").dropna()
        numeric_remove_day = float(remove_day.iloc[0]) if len(remove_day) else np.nan
        action = policy_df["policy_action_resolved"].astype("string").str.strip().str.lower()
        action_remove = pd.to_numeric(policy_df["policy_action_remove_resolved"], errors="coerce")
        remove_rows = action.eq("remove") | action_remove.eq(1)
        remove_rows_by_episode = remove_rows.groupby(policy_df[episode_id_col], sort=False).sum()

        if pd.notna(numeric_remove_day):
            reaches_policy_removal_day = (
                pd.to_numeric(policy_df[episode_day_col], errors="coerce")
                .eq(numeric_remove_day)
                .groupby(policy_df[episode_id_col], sort=False)
                .max()
            )
            n_reached = int(reaches_policy_removal_day.sum())
        else:
            n_reached = 0

        n_remove_rows = int(remove_rows.sum())
        rows.append(
            {
                "policy_name": policy_name,
                "policy_remove_day": numeric_remove_day,
                "n_policy_remove_rows": n_remove_rows,
                "n_episodes_reaching_policy_removal_day": n_reached,
                "n_episodes_with_more_than_one_remove_row": int(remove_rows_by_episode.gt(1).sum()),
                "n_policy_removal_day_extra_rows_treated_as_out": int(
                    pd.to_numeric(
                        policy_df["policy_removal_day_extra_row_treated_as_out"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "n_policy_remove_row_shortfall_vs_reached_episodes": int(n_reached - n_remove_rows),
            }
        )
    return pd.DataFrame(rows)


def validate_resolved_target_policy_timeline(
    df,
    *,
    episode_id_col="catheter_episode_id",
    context="policy_intervention_panel_long.csv",
):
    # Validate the resolved policy timeline
    state = df["policy_catheter_state"].astype("string").str.strip().str.lower()
    action = df["policy_action_resolved"].astype("string").str.strip().str.lower()
    action_remove = pd.to_numeric(df["policy_action_remove_resolved"], errors="coerce")

    invalid_state = ~state.isin(["in", "out"])
    if invalid_state.any():
        examples = df.loc[invalid_state, ["policy_name", episode_id_col, "policy_catheter_state"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_catheter_state values. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    invalid_action = ~action.isin(["keep", "remove", "out"])
    if invalid_action.any():
        examples = df.loc[invalid_action, ["policy_name", episode_id_col, "policy_action_resolved"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_action_resolved values. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    out_remove = state.eq("out") & (action.eq("remove") | action_remove.eq(1))
    if out_remove.any():
        examples = df.loc[
            out_remove,
            ["policy_name", episode_id_col, "policy_catheter_state", "policy_action_resolved"],
        ].head(10)
        raise ValueError(
            f"{context} assigns a resolved remove action on OUT-state rows. "
            f"Rebuild with build_policy_intervention_panels.py. Examples:\n{examples}"
        )

    # Summarise resolved timeline diagnostics
    diagnostics = resolved_timeline_diagnostics(df, episode_id_col=episode_id_col)
    too_many = diagnostics["n_episodes_with_more_than_one_remove_row"].gt(0)
    shortfall = diagnostics["n_policy_remove_row_shortfall_vs_reached_episodes"].ne(0)
    if too_many.any() or shortfall.any():
        failing = diagnostics.loc[too_many | shortfall].head(20)
        raise ValueError(
            f"{context} has unsafe fixed-day resolved target-policy timing. "
            "Rebuild it with build_policy_intervention_panels.py. Diagnostics:\n"
            f"{failing}"
        )


def duplicate_episode_day_count(
    df,
    group_cols,
    day_col="episode_day_since_insertion",
):
    # Count duplicate episode-day rows
    duplicated = df.duplicated([*group_cols, day_col], keep=False)
    return int(duplicated.sum())


def add_standard_comparisons(
    summary,
    *,
    baseline_label,
    comparison_map,
):
    # Add standard comparisons against current practice
    baseline = summary.loc[
        summary["policy_name"].eq(baseline_label)
    ].iloc[0]
    out = summary.copy()
    for standard_name, value_col in comparison_map.items():
        baseline_value = baseline[value_col]
        diff_col = f"{standard_name}_difference_vs_current_practice"
        out[diff_col] = out[value_col] - baseline_value
        # The pct-points column is meaningful for risk outcomes. For exposure it
        # is retained for schema consistency and equals the raw day difference
        out[f"{standard_name}_difference_pct_points_vs_current_practice"] = (
            out[diff_col] * 100 if standard_name.endswith("_risk") else out[diff_col]
        )
        out[f"{standard_name}_ratio_vs_current_practice"] = out[value_col].apply(
            lambda value: np.nan
            if pd.isna(value) or pd.isna(baseline_value) or baseline_value == 0
            else float(value / baseline_value)
        )
    return out


def add_overlap_quality_flags(
    summary,
    *,
    support_diagnostics,
    weight_diagnostics,
    current_practice_label="current_practice",
):
    # Add overlap quality flags
    out = summary.copy()
    out["low_adherence_flag"] = out["pct_adherent_episodes"].lt(
        LOW_ADHERENCE_THRESHOLD
    )

    n_col = "n_total_policy_episodes" if "n_total_policy_episodes" in out.columns else "n_episodes"
    ess = pd.to_numeric(out["effective_sample_size"], errors="coerce")
    n_total = pd.to_numeric(out[n_col], errors="coerce")
    out["low_ess_flag"] = ess.lt(LOW_ESS_MIN) | ess.lt(LOW_ESS_FRACTION * n_total)

    support_all = support_diagnostics.loc[
        support_diagnostics["group"].eq("all"),
        ["policy_name", "pct_below_0_05"],
    ].drop_duplicates("policy_name")
    support_all["low_support_flag"] = pd.to_numeric(
        support_all["pct_below_0_05"],
        errors="coerce",
    ).gt(LOW_SUPPORT_PCT_BELOW_005_THRESHOLD)
    out = out.merge(
        support_all[["policy_name", "low_support_flag"]],
        on="policy_name",
        how="left",
    )
    out["low_support_flag"] = out["low_support_flag"].fillna(False)

    weight_flags = weight_diagnostics[["policy_name"]].copy()
    weight_flags["extreme_weight_flag"] = (
        pd.to_numeric(
            weight_diagnostics["p99_weight"],
            errors="coerce",
        ).gt(EXTREME_WEIGHT_P99_THRESHOLD)
        | pd.to_numeric(
            weight_diagnostics["max_weight"],
            errors="coerce",
        ).gt(EXTREME_WEIGHT_MAX_THRESHOLD)
    )
    weight_flags = weight_flags.drop_duplicates("policy_name")
    out = out.merge(weight_flags, on="policy_name", how="left")
    out["extreme_weight_flag"] = out["extreme_weight_flag"].fillna(False)

    current = out["policy_name"].eq(current_practice_label)
    for col in ["low_adherence_flag", "low_ess_flag", "low_support_flag", "extreme_weight_flag"]:
        out.loc[current, col] = False
    return out


CROSSFIT_FOLD_COL = "_crossfit_fold"


PREDICTION_COLUMNS = [
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


def first_non_null(series):
    # Return the first non-missing value
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def max_binary(series):
    # Return whether any binary value is present
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return np.nan
    return int(numeric.max() > 0)


def add_episode_day_since_insertion(df):
    # Add episode day since catheter insertion
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def predict_fold_model(fold_model, features):
    if fold_model["fallback"]:
        return np.full(len(features), float(fold_model["fallback_probability"]), dtype=float)

    retained_feature_cols = list(fold_model["retained_feature_cols"])
    return fold_model["model"].predict_proba(
        features.loc[:, retained_feature_cols].to_numpy(dtype=float)
    )[:, 1]


def rescore_state_action_predictions(
    df,
    payload,
    state,
    outcome,
    output_col,
    target_mask,
    action_remove=None,
):
    if int(target_mask.sum()) == 0:
        return df

    models_key = "in_models" if state == "in" else "out_models"
    x_cols_key = "x_cols_in" if state == "in" else "x_cols_out"
    feature_cols = list(payload[x_cols_key])

    fold_models = payload[models_key][outcome]["fold_models"]
    for fold_model in fold_models:
        fold = int(fold_model["fold"])
        rows = target_mask & pd.to_numeric(
            df[CROSSFIT_FOLD_COL],
            errors="coerce",
        ).eq(fold)
        if int(rows.sum()) == 0:
            continue
        features = df.loc[rows, feature_cols].copy()
        if state == "in":
            action_col = payload["action_remove_col"]
            features[action_col] = action_remove
        df.loc[rows, output_col] = predict_fold_model(fold_model, features[feature_cols])
        df.loc[rows, f"__rescored_{output_col}"] = True
    return df


def fill_missing_counterfactual_predictions(
    df,
    outcome_models_path,
):
    # Standardise prediction columns and track rescored values
    df[PREDICTION_COLUMNS] = df[PREDICTION_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    rescored_cols = {
        f"__rescored_{col}": False
        for col in PREDICTION_COLUMNS
    }
    df = pd.concat([df, pd.DataFrame(rescored_cols, index=df.index)], axis=1)

    # Define every state-action prediction needed downstream
    needed_specs = [
        ("in", "cauti", "p_cauti_if_keep", 0),
        ("in", "cauti", "p_cauti_if_remove", 1),
        ("out", "cauti", "p_cauti_if_out", None),
        ("out", "reinsertion", "p_reinsertion_if_out", None),
        ("in", "death", "p_death_if_keep", 0),
        ("in", "death", "p_death_if_remove", 1),
        ("out", "death", "p_death_if_out", None),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_keep", 0),
        ("in", "icu_exit_alive", "p_icu_exit_alive_if_remove", 1),
        ("out", "icu_exit_alive", "p_icu_exit_alive_if_out", None),
        ("in", "no_event", "p_no_event_if_keep", 0),
        ("in", "no_event", "p_no_event_if_remove", 1),
        ("out", "no_event", "p_no_event_if_out", None),
    ]


    if not df[PREDICTION_COLUMNS].isna().any().any():
        return df

    # Load saved outcome model artefacts
    payload = joblib.load(outcome_models_path)

    for state, outcome, col, action_remove in needed_specs:
        missing_mask = df[col].isna()
        if int(missing_mask.sum()) == 0:
            continue
        df = rescore_state_action_predictions(
            df,
            payload,
            state,
            outcome,
            col,
            missing_mask,
            action_remove,
        )

    return df
