# Share estimator-agnostic helpers for policy-evaluation scripts


import joblib
import numpy as np
import pandas as pd
import pipeline_config as run_config


REINSERTION_CONTEXT_COL = "reinsertion_remove_period"
REINSERTION_PERIOD_COL = "reinsertion_periods_out"
REINSERTION_POLICY_PREFIX = "p_reinsertion_policy_day_"
EXPOSURE_MEASURE = "nominal_catheter_in_decision_period_days"
POLICY_OUTCOMES = ("cauti", "reinsertion", "death", "icu_exit_alive", "no_event")


def policy_prediction_columns(days=None):
    days = run_config.POLICY_DAYS if days is None else days
    return [f"p_{outcome}_policy_day_{day}" for day in days for outcome in POLICY_OUTCOMES]


def is_policy_prediction_column(column):
    return any(column.startswith(f"p_{outcome}_policy_day_") for outcome in POLICY_OUTCOMES)


def policy_feature_overrides(df, removal_day):
    """Candidate inputs on the preserved grid; clinical history stays factual."""
    day = ((pd.to_datetime(df.period_start, format="mixed")
            - pd.to_datetime(df.inserted, format="mixed")) // pd.Timedelta(days=1) + 1).astype(int)
    # On legacy split-day rows only the first interval receives REMOVE, as in
    # the policy builder. The insertion-anchored panel has one row per day.
    keys = ["subject_id", "hadm_id", "stay_id", "inserted", "removed"]
    first = df.loc[day.eq(removal_day)].sort_values(["period_start", "period_end"]).groupby(
        keys, dropna=False, sort=False,
    ).head(1).index
    remove = pd.Series(df.index.isin(first), index=df.index)
    in_state = day.lt(removal_day) | remove
    periods_out = (day - removal_day).clip(lower=1)
    return {
        "catheter_state": pd.Series(np.where(in_state, "in", "out"), index=df.index),
        "removed_in_period": remove.astype(int),
        "periods_in_state": day.where(in_state, periods_out),
        REINSERTION_CONTEXT_COL: remove.astype(int),
        REINSERTION_PERIOD_COL: periods_out.where(~in_state, 0),
    }


def add_policy_predictions(df, payload, *, predictor=None, days=None):
    """Unmasked hazards from each patient's held-out models and policy inputs.

    Factual eligibility defines training only. No factual at_risk_* flags or
    masked factual predictions enter these candidate-policy hazards.
    """
    predictor = predict_fold_model if predictor is None else predictor
    days = run_config.POLICY_DAYS if days is None else days
    if df.groupby("subject_id", dropna=False)[CROSSFIT_FOLD_COL].nunique().gt(1).any():
        raise ValueError("A patient appears in multiple held-out folds")
    scores = pd.DataFrame(np.nan, index=df.index, columns=policy_prediction_columns(days))
    for day in days:
        overrides = policy_feature_overrides(df, day)
        keep = overrides["catheter_state"].eq("in") & overrides["removed_in_period"].eq(0)
        scores.loc[keep, f"p_reinsertion_policy_day_{day}"] = 0.0
        for state in ("in", "out"):
            for outcome, model in payload[f"{state}_models"].items():
                features = model["features"]
                eligible_state = (overrides["catheter_state"].eq("out") | overrides["removed_in_period"].eq(1)
                                  if outcome == "reinsertion" else overrides["catheter_state"].eq(state))
                column = f"p_{outcome}_policy_day_{day}"
                for fold_model in model["fold_models"]:
                    rows = eligible_state & df[CROSSFIT_FOLD_COL].eq(fold_model["fold"])
                    if not rows.any():
                        continue
                    inputs = df.loc[rows, features].copy()
                    for feature, values in overrides.items():
                        if feature in inputs:
                            inputs[feature] = values.loc[rows]
                    scores.loc[rows, column] = predictor(fold_model, inputs)
    if scores.isna().any().any():
        raise ValueError("Policy hazards lack a corresponding held-out outcome model")
    return pd.concat([df.drop(columns=scores.columns, errors="ignore"), scores], axis=1)


def select_policy_outcomes(df, prefix, attribution_periods=2):
    """Select raw policy hazards, then apply policy eligibility to contributions."""
    current = df.policy_name.eq("current_practice")
    suffix = df.policy_action.map({"keep": "keep", "remove": "remove", "out": "out"})
    for outcome in POLICY_OUTCOMES:
        target = f"{prefix}_{'recatheterisation' if outcome == 'reinsertion' else outcome}_under_policy"
        df[target] = np.nan
        for action in ("keep", "remove", "out"):
            rows = current & suffix.eq(action)
            if outcome == "reinsertion" and action == "keep":
                df.loc[rows, target] = 0.0
            else:
                df.loc[rows, target] = df.loc[rows, f"p_{outcome}_if_{action}"]
        for day in pd.to_numeric(df.loc[~current, "policy_remove_day"]).unique():
            rows = ~current & df.policy_remove_day.eq(day)
            df.loc[rows, target] = df.loc[rows, f"p_{outcome}_policy_day_{int(day)}"]
    in_state = df.policy_catheter_state.eq("in")
    window = in_state | pd.to_numeric(df.policy_periods_out).between(0, attribution_periods)
    window.loc[current] = df.loc[current, "at_risk_cauti"].eq(1)
    df["policy_at_risk_cauti"] = window.astype(int)
    df["policy_cauti_hazard"] = df[f"{prefix}_cauti_under_policy"]
    df.loc[~window, f"{prefix}_cauti_under_policy"] = 0.0
    df.loc[current & df.at_risk_reinsertion.eq(0), f"{prefix}_recatheterisation_under_policy"] = 0.0
    df["policy_periods_in_state"] = df.policy_periods_in.where(in_state, df.policy_periods_out.clip(lower=1))
    df["__used_rescored_prediction"] = ~current
    for column in (REINSERTION_CONTEXT_COL, REINSERTION_PERIOD_COL):
        if current.any():
            df.loc[current, f"policy_{column}"] = df.loc[current, column]


def reinsertion_policy_columns():
    return [f"{REINSERTION_POLICY_PREFIX}{day}" for day in run_config.POLICY_DAYS]


def add_reinsertion_context(df):
    """Pre-period timing; realised post-action exposure is never a predictor."""
    out = df.copy()
    in_rows = out["catheter_state"].eq("in")
    out[REINSERTION_CONTEXT_COL] = (in_rows & out["removed_in_period"].eq(1)).astype(int)
    # Zero denotes the REMOVE period. Subsequent periods use the one-based
    # elapsed 24-hour bin from known removal, consistently across start states.
    elapsed = pd.to_datetime(out["period_start"], format="mixed") - pd.to_datetime(out["removed"], format="mixed")
    out[REINSERTION_PERIOD_COL] = np.where(in_rows, 0, elapsed // pd.Timedelta(days=1) + 1).astype(int)
    return out


def reinsertion_feature_columns(out_feature_cols):
    return [col for col in out_feature_cols if col != "periods_in_state"] + [
        REINSERTION_CONTEXT_COL, REINSERTION_PERIOD_COL,
    ]


def fixed_day_reinsertion_context(day, removal_day):
    """Model inputs also exported as policy_reinsertion_* panel columns."""
    return {
        REINSERTION_CONTEXT_COL: day.eq(removal_day).astype(int),
        REINSERTION_PERIOD_COL: (day - removal_day).clip(lower=0).astype(int),
    }


def predict_reinsertion_columns(fold_model, df, feature_cols, predictor=None):
    """Score REMOVE, factual OUT, and each day-policy context with one model."""
    predictor = predict_fold_model if predictor is None else predictor
    if not {REINSERTION_CONTEXT_COL, REINSERTION_PERIOD_COL}.issubset(feature_cols):
        raise ValueError("Reinsertion model predates the rebuilt grid; refit nuisance models")
    base = df.reindex(columns=feature_cols).copy()
    base[REINSERTION_CONTEXT_COL] = 0
    base[REINSERTION_PERIOD_COL] = df[REINSERTION_PERIOD_COL].clip(lower=1)
    scores = {"p_reinsertion_if_out": predictor(fold_model, base)}
    base[REINSERTION_CONTEXT_COL] = 1
    base[REINSERTION_PERIOD_COL] = 0
    scores["p_reinsertion_if_remove"] = predictor(fold_model, base)
    day = add_episode_day(df)["episode_day"]
    for removal_day in run_config.POLICY_DAYS:
        inputs = base.copy()
        for column, values in fixed_day_reinsertion_context(day, removal_day).items():
            inputs[column] = values
        prediction = predictor(fold_model, inputs)
        scores[f"{REINSERTION_POLICY_PREFIX}{removal_day}"] = np.where(day.lt(removal_day), 0.0, prediction)
    return scores


def factual_reinsertion_prediction(df):
    return pd.Series(np.where(
        df["catheter_state"].eq("in"), df["p_reinsertion_if_remove"], df["p_reinsertion_if_out"],
    ), index=df.index)


MAX_REASONABLE_PERIOD_DURATION_DAYS = 7.0


def save_report_df(df, path, decimals=3):
    # Round report values without changing source data
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


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


def observed_catheter_exposure_days(df):
    """Actual catheter-IN time, including the partial final REMOVE period."""
    start = pd.to_datetime(df["period_start"], format="mixed")
    end = pd.to_datetime(df["period_end"], format="mixed")
    removed = pd.to_datetime(df["removed"], format="mixed")
    catheter_end = end.clip(upper=removed)
    return (catheter_end - start).dt.total_seconds().clip(lower=0) / 86400.0


def label_nominal_exposure(df):
    """Legacy comparison columns retain their nominal decision-period scale."""
    aliases = {
        f"nominal_{col}": df[col] for col in df
        if "catheter_exposure_days" in col and not col.startswith(("nominal_", "actual_"))
    }
    out = df.drop(columns=list(aliases), errors="ignore")
    out = pd.concat([out, pd.DataFrame(aliases, index=df.index)], axis=1)
    out["catheter_exposure_measure"] = EXPOSURE_MEASURE
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
    action_col="policy_action",
    policy_name_col="policy_name",
    policy_remove_day_col="policy_remove_day",
    episode_day_col="episode_day",
    period_start_col="period_start",
    period_end_col="period_end",
    decision_row_id_col="decision_row_id",
    state_col="policy_catheter_state",
    periods_in_col="policy_periods_in",
    periods_out_col="policy_periods_out",
):
    # Add the fixed-day policy timeline
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

    out[periods_in_col] = np.nan
    out.loc[before_remove | first_row_on_remove_day, periods_in_col] = day.loc[
        before_remove | first_row_on_remove_day
    ]

    out[periods_out_col] = np.nan
    out.loc[later_row_on_remove_day, periods_out_col] = 0.0
    out.loc[after_remove, periods_out_col] = day.loc[after_remove] - remove_day.loc[after_remove]
    out["policy_removal_day_extra_row_treated_as_out"] = later_row_on_remove_day.astype(int)
    for column, values in fixed_day_reinsertion_context(day, remove_day).items():
        out[f"policy_{column}"] = values
    return out


def policy_timeline_checks(
    df,
    *,
    episode_id_col="catheter_episode_id",
    policy_name_col="policy_name",
    policy_remove_day_col="policy_remove_day",
    episode_day_col="episode_day",
):
    # Summarise policy timeline checks
    rows = []
    for policy_name, policy_df in df.groupby(policy_name_col, dropna=False, sort=False):
        remove_day = pd.to_numeric(policy_df.get(policy_remove_day_col, pd.Series(dtype=float)), errors="coerce").dropna()
        numeric_remove_day = float(remove_day.iloc[0]) if len(remove_day) else np.nan
        action = policy_df["policy_action"].astype("string").str.strip().str.lower()
        remove_rows = action.eq("remove")
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
                "n_policy_remove_row_shortfall_vs_reached_episodes": int(n_reached - n_remove_rows) if pd.notna(numeric_remove_day) else 0,
            }
        )
    return pd.DataFrame(rows)


def check_policy_timeline(
    df,
    *,
    episode_id_col="catheter_episode_id",
    context="policy panel",
):
    # Validate the policy timeline
    state = df["policy_catheter_state"].astype("string").str.strip().str.lower()
    action = df["policy_action"].astype("string").str.strip().str.lower()

    invalid_state = ~state.isin(["in", "out"])
    if invalid_state.any():
        examples = df.loc[invalid_state, ["policy_name", episode_id_col, "policy_catheter_state"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_catheter_state values. "
            f"Rebuild with build_policy_panels.py. Examples:\n{examples}"
        )

    invalid_action = ~action.isin(["keep", "remove", "out"])
    if invalid_action.any():
        examples = df.loc[invalid_action, ["policy_name", episode_id_col, "policy_action"]].head(10)
        raise ValueError(
            f"{context} has invalid policy_action values. "
            f"Rebuild with build_policy_panels.py. Examples:\n{examples}"
        )

    out_remove = state.eq("out") & action.eq("remove")
    if out_remove.any():
        examples = df.loc[
            out_remove,
            ["policy_name", episode_id_col, "policy_catheter_state", "policy_action"],
        ].head(10)
        raise ValueError(
            f"{context} assigns a remove action on OUT-state rows. "
            f"Rebuild with build_policy_panels.py. Examples:\n{examples}"
        )

    # Summarise policy timeline checks
    diagnostics = policy_timeline_checks(df, episode_id_col=episode_id_col)
    too_many = diagnostics["n_episodes_with_more_than_one_remove_row"].gt(0)
    shortfall = diagnostics["n_policy_remove_row_shortfall_vs_reached_episodes"].ne(0)
    if too_many.any() or shortfall.any():
        failing = diagnostics.loc[too_many | shortfall].head(20)
        raise ValueError(
            f"{context} has unsafe fixed-day policy timing. "
            "Rebuild it with build_policy_panels.py. Diagnostics:\n"
            f"{failing}"
        )


def duplicate_episode_day_count(
    df,
    group_cols,
    day_col="episode_day",
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
    return label_nominal_exposure(out)


CROSSFIT_FOLD_COL = "_crossfit_fold"


PREDICTION_COLUMNS = [
    "p_cauti_if_keep",
    "p_cauti_if_remove",
    "p_cauti_if_out",
    "p_reinsertion_if_out",
    "p_reinsertion_if_remove",
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


def add_episode_day(df):
    # Add episode day since catheter insertion
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")
    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day"].lt(1), "episode_day"] = 1
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
    feature_cols = list(payload[models_key][outcome].get("features", payload[x_cols_key]))

    fold_models = payload[models_key][outcome]["fold_models"]
    for fold_model in fold_models:
        fold = int(fold_model["fold"])
        rows = target_mask & pd.to_numeric(
            df[CROSSFIT_FOLD_COL],
            errors="coerce",
        ).eq(fold)
        if int(rows.sum()) == 0:
            continue
        if outcome == "reinsertion":
            scores = predict_reinsertion_columns(fold_model, df.loc[rows], feature_cols)
            for column, values in scores.items():
                df.loc[rows, column] = values
            df.loc[rows, f"__rescored_{output_col}"] = True
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
    for column in [*PREDICTION_COLUMNS, *reinsertion_policy_columns()]:
        if column not in df:
            df[column] = np.nan
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
        ("out", "reinsertion", "p_reinsertion_if_remove", None),
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


    for column in reinsertion_policy_columns():
        needed_specs.append(("out", "reinsertion", column, None))
    policy_complete = df.reindex(columns=policy_prediction_columns()).notna().all().all()
    if not df[[*PREDICTION_COLUMNS, *reinsertion_policy_columns()]].isna().any().any() and policy_complete:
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

    return df if policy_complete else add_policy_predictions(df, payload)


def effective_sample_size(weights):
    # Calculate the effective sample size
    # Return positive finite weights
    weights = pd.to_numeric(weights, errors="coerce")
    weights = weights[np.isfinite(weights) & weights.gt(0)]
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


def cumulative_event_probability(probabilities):
    # Calculate cumulative event probability
    probs = pd.to_numeric(probabilities, errors="coerce").dropna()
    if probs.empty:
        return np.nan
    probs = probs.clip(0.0, 1.0)
    return float(1.0 - np.prod(1.0 - probs.to_numpy(dtype=float)))
