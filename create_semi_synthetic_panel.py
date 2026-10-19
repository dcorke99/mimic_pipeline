from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from policy_eval_common import cumulative_event_probability, add_episode_day
import policy_eval_common as pec

from build_policy_panels import (
    POLICY_INPUT_COLS,
    add_stable_ids_and_decision_flag,
    apply_fixed_day_policy,
)

# Run choices come from pipeline_config.py; simulation settings stay here.
from pipeline_config import POLICY_DAYS, POLICY_DML_HORIZON_HOURS


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_PANEL = REPO_ROOT / "data/modelling_panel.csv"
OUTPUT_DIR = REPO_ROOT / "artefacts/validation"
RANDOM_SEED = 20260811
# Running this stage generates all three validation datasets.

# Base-panel columns replaced in validation copies

STATE_COL = "catheter_state"
ACTION_COL = "removed_in_period"
ACTION_REMOVE_COL = "action_remove"
OBSERVED_ACTION_COL = "observed_action"
OUTCOME_COL = "cauti_in_period"
CAUTI_RISK_COL = "at_risk_cauti"
REINSERTION_RISK_COL = "at_risk_reinsertion"

EPISODE_IDENTITY_COLS = ["subject_id", "hadm_id", "stay_id", "inserted"]
TERMINAL_EVENT_COLS = [
    "reinsertion_in_period",
    "death_in_period",
    "icu_exit_alive_in_period",
]
OPTIONAL_DECISION_INDICATOR_COLS = ["is_decision_row", "decision_row"]

# Deliberately measured pre-decision confounders

AGE_COL = "age"
PERIODS_COL = "periods_in_state"
HEART_RATE_MEAN_COL = "itemid_220045__mean"
HEART_RATE_ITEM_PREFIX = "itemid_220045__"

CONFOUNDER_COLS = [AGE_COL, PERIODS_COL, HEART_RATE_MEAN_COL]
Z_COLS = {
    AGE_COL: "z_age",
    PERIODS_COL: "z_periods_in_state",
    HEART_RATE_MEAN_COL: "z_heart_rate_mean",
}

# Data-generating coefficients

ACTION_COEFFICIENTS = {
    "z_age": 0.45,
    "z_periods_in_state": 0.70,
    "z_heart_rate_mean": 0.80,
}
ACTION_PROBABILITY_LOWER = 0.05
ACTION_PROBABILITY_UPPER = 0.95

OUTCOME_INTERCEPT = -2.40
OUTCOME_COEFFICIENTS = {
    "z_age": 0.40,
    "z_periods_in_state": 0.55,
    "z_heart_rate_mean": 0.75,
}
TREATMENT_LOG_ODDS_EFFECT = -0.90

POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS = 2
VALIDATION_ROW_ID_COL = "validation_row_id"

PRIMARY_PANEL_FILENAME = "with_confounding.csv"
OMITTED_PANEL_FILENAME = "confounder_omitted.csv"
RANDOMISED_PANEL_FILENAME = "randomised_action.csv"
TRUTH_FILENAME = "row_truth.csv"
KNOWN_TRUTH_POLICY_FILENAME = "known_truth_gformula.csv"
KNOWN_TRUTH_DML_FILENAME = "known_truth_policy_dml.csv"
COEFFICIENTS_FILENAME = "simulation_coefficients.csv"

TRUTH_ONLY_PREFIXES = (
    "true_",
    "known_truth_",
    "dgp_",
    "z_dgp_",
)

def expit(linear_predictor):
    values = np.asarray(linear_predictor, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def normalise_binary(series, column_name):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        examples = series.loc[numeric.isna() | ~numeric.isin([0, 1])].head(10)
        raise ValueError(f"{column_name} must contain only 0/1 values. Examples:\n{examples}")
    return numeric.astype(np.int8)

def load_source_panel(path):

    panel = pd.read_csv(path, low_memory=False)
    panel.columns = panel.columns.str.strip()
    if panel.columns.duplicated().any():
        duplicates = panel.columns[panel.columns.duplicated()].tolist()
        raise ValueError(f"Source panel contains duplicate columns: {duplicates}")

    states = panel[STATE_COL].astype("string").str.strip().str.lower()
    unknown_states = sorted(set(states.dropna()) - {"in", "out"})
    if unknown_states:
        raise ValueError(f"Unexpected catheter states: {unknown_states}")
    if states.isna().any():
        raise ValueError("catheter_state must be present on every source row")

    for column in [ACTION_COL, OUTCOME_COL, CAUTI_RISK_COL]:
        normalise_binary(panel[column], column)

    for column in CONFOUNDER_COLS:
        numeric = pd.to_numeric(panel[column], errors="coerce")
        if numeric.notna().sum() < 2:
            raise ValueError(f"Selected confounder has fewer than two observed values: {column}")

    return panel

def build_standardised_confounders(panel, reference_mask):
    z_values = pd.DataFrame(index=panel.index)
    statistics = {}

    for source_col, z_col in Z_COLS.items():
        numeric = pd.to_numeric(panel[source_col], errors="coerce")
        reference = numeric.loc[reference_mask].dropna()
        if len(reference) < 2:
            raise ValueError(
                f"Cannot standardise {source_col}: fewer than two observed reference values"
            )

        mean = float(reference.mean())
        sd = float(reference.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid standardisation values for {source_col}: mean={mean}, sd={sd}")

        # Mean imputation is used only inside the DGP. The source panel values
        # and their empirical missingness patterns remain untouched.
        z_values[z_col] = (numeric.fillna(mean) - mean) / sd
        statistics[source_col] = {
            "mean": mean,
            "sd_population_ddof_0": sd,
            "reference_n_observed": int(len(reference)),
            "panel_n_missing": int(numeric.isna().sum()),
            "dgp_missing_value_handling": "mean imputation, giving z=0",
        }

    return z_values, statistics

def apply_standardisation(panel, statistics):
    """Apply saved DGP standardisation after rebuilding periods_in_state."""
    z_values = pd.DataFrame(index=panel.index)
    for source_col, z_col in Z_COLS.items():
        numeric = pd.to_numeric(panel[source_col], errors="coerce")
        mean = float(statistics[source_col]["mean"])
        sd = float(statistics[source_col]["sd_population_ddof_0"])
        z_values[z_col] = (numeric.fillna(mean) - mean) / sd
    return z_values

def episode_codes(panel):
    """Identify source catheter episodes without using mutable removal time."""
    key_frame = panel.loc[:, EPISODE_IDENTITY_COLS].astype("string").fillna("<NA>")
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(key_frame), sort=True)
    if (codes < 0).any():
        raise ValueError("Unable to construct stable catheter episode identifiers")
    return pd.Series(codes, index=panel.index, dtype=np.int64)

def chronological_episode_positions(panel, codes):
    order = pd.DataFrame({
        "_position": np.arange(len(panel), dtype=np.int64),
        "_episode_code": codes.to_numpy(dtype=np.int64),
        "_period_start": pd.to_datetime(panel["period_start"], errors="coerce"),
        "_period_end": pd.to_datetime(panel["period_end"], errors="coerce"),
    })
    if order[["_period_start", "_period_end"]].isna().any().any():
        raise ValueError("period_start and period_end must be valid timestamps")
    order = order.sort_values(
        ["_episode_code", "_period_start", "_period_end", "_position"],
        kind="mergesort",
    )
    return order.groupby("_episode_code", sort=False)["_position"].apply(list)

def calibrate_action_intercept(linear_without_intercept, target_prevalence):
    if not ACTION_PROBABILITY_LOWER <= target_prevalence <= ACTION_PROBABILITY_UPPER:
        raise ValueError(
            "The target action prevalence must lie within the configured "
            f"probability bounds [{ACTION_PROBABILITY_LOWER}, "
            f"{ACTION_PROBABILITY_UPPER}]; received {target_prevalence}"
        )

    lower_intercept = -30.0
    upper_intercept = 30.0
    for _ in range(200):
        midpoint = (lower_intercept + upper_intercept) / 2.0
        probabilities = np.clip(
            expit(midpoint + linear_without_intercept),
            ACTION_PROBABILITY_LOWER,
            ACTION_PROBABILITY_UPPER,
        )
        if float(probabilities.mean()) < target_prevalence:
            lower_intercept = midpoint
        else:
            upper_intercept = midpoint
    return float((lower_intercept + upper_intercept) / 2.0)

def rebuild_observed_action(panel):
    states = panel[STATE_COL].astype("string").str.strip().str.lower()
    actions = normalise_binary(panel[ACTION_COL], ACTION_COL)
    observed_action = pd.Series("out", index=panel.index, dtype="string")
    observed_action.loc[states.eq("in") & actions.eq(0)] = "keep"
    observed_action.loc[states.eq("in") & actions.eq(1)] = "remove"
    panel[OBSERVED_ACTION_COL] = observed_action
    panel[ACTION_REMOVE_COL] = (
        states.eq("in") & actions.eq(1)
    ).astype(np.int8)
    return panel

def generate_sequential_trajectory(
    source_panel,
    source_z_values,
    standardisation,
    rng,
    *,
    randomised,
    target_prevalence,
):
    """Draw actions in episode order and rebuild all catheter-state fields."""
    source_states = source_panel[STATE_COL].astype("string").str.strip().str.lower()
    source_decision_mask = source_states.eq("in")

    if randomised:
        action_intercept = float(np.log(target_prevalence / (1.0 - target_prevalence)))
    else:
        # Form the confounded action-model linear predictor.
        reference_z = source_z_values.loc[source_decision_mask]
        reference_linear = np.zeros(len(reference_z), dtype=float)
        for z_col, coefficient in ACTION_COEFFICIENTS.items():
            reference_linear += float(coefficient) * pd.to_numeric(
                reference_z[z_col], errors="raise"
            ).to_numpy(dtype=float)
        action_intercept = calibrate_action_intercept(
            reference_linear,
            target_prevalence,
        )

    panel = source_panel.copy(deep=True)
    codes = episode_codes(source_panel)
    positions_by_episode = chronological_episode_positions(source_panel, codes)
    period_starts = pd.to_datetime(source_panel["period_start"], errors="coerce")
    period_ends = pd.to_datetime(source_panel["period_end"], errors="coerce")
    episode_ends = pd.to_datetime(source_panel["episode_end_time"], errors="coerce")
    if episode_ends.isna().any():
        raise ValueError("episode_end_time must be a valid timestamp on every row")

    synthetic_action = np.zeros(len(panel), dtype=np.int8)
    true_propensity = np.full(len(panel), np.nan, dtype=float)
    synthetic_state = np.empty(len(panel), dtype=object)
    periods_in_state = np.zeros(len(panel), dtype=np.int64)
    episode_index = np.zeros(len(panel), dtype=np.int64)
    synthetic_removed_time = np.empty(len(panel), dtype="datetime64[ns]")

    age_z = source_z_values[Z_COLS[AGE_COL]].to_numpy(dtype=float)
    heart_rate_z = source_z_values[Z_COLS[HEART_RATE_MEAN_COL]].to_numpy(dtype=float)
    periods_mean = float(standardisation[PERIODS_COL]["mean"])
    periods_sd = float(standardisation[PERIODS_COL]["sd_population_ddof_0"])

    for positions in positions_by_episode:
        already_removed = False
        in_period = 0
        out_period = 0
        removal_time = None

        for row_number, position in enumerate(positions):
            episode_index[position] = row_number
            if already_removed:
                out_period += 1
                synthetic_state[position] = "out"
                periods_in_state[position] = out_period
                continue

            in_period += 1
            synthetic_state[position] = "in"
            periods_in_state[position] = in_period

            if randomised:
                propensity = target_prevalence
            else:
                z_periods = (in_period - periods_mean) / periods_sd
                action_linear = (
                    ACTION_COEFFICIENTS[Z_COLS[AGE_COL]] * age_z[position]
                    + ACTION_COEFFICIENTS[Z_COLS[PERIODS_COL]] * z_periods
                    + ACTION_COEFFICIENTS[Z_COLS[HEART_RATE_MEAN_COL]]
                    * heart_rate_z[position]
                )
                propensity = float(np.clip(
                    expit(action_intercept + action_linear),
                    ACTION_PROBABILITY_LOWER,
                    ACTION_PROBABILITY_UPPER,
                ))

            true_propensity[position] = propensity
            action = int(rng.binomial(1, propensity))
            synthetic_action[position] = action
            if action == 1:
                already_removed = True
                removal_time = period_ends.iloc[position] - pd.Timedelta(microseconds=1)
                if removal_time <= period_starts.iloc[position]:
                    raise ValueError("A panel period is too short to locate removal within it")

        final_position = positions[-1]
        if removal_time is None:
            # A terminal-time value represents no removal before the preserved
            # endpoint, matching the base panel's terminal-tie convention.
            removal_time = episode_ends.iloc[final_position]
        synthetic_removed_time[np.asarray(positions, dtype=int)] = removal_time.to_datetime64()

    panel["removed"] = pd.to_datetime(synthetic_removed_time)
    panel[STATE_COL] = pd.Series(synthetic_state, index=panel.index, dtype="string")
    panel["episode_index"] = episode_index
    panel[PERIODS_COL] = periods_in_state
    panel[ACTION_COL] = synthetic_action
    # CAUTI remains attributable through 48 hours after removal.
    panel[CAUTI_RISK_COL] = pd.to_datetime(panel["period_start"]).lt(
        pd.to_datetime(panel["removed"]) + pd.Timedelta(hours=48)
    ).astype(np.int8)
    duration = (period_ends - period_starts).dt.total_seconds() / 3600.0
    removed = panel["removed"]
    in_hours = (period_ends.clip(upper=removed) - period_starts).dt.total_seconds().clip(lower=0) / 3600.0
    # Timestamp subtraction preserves exact zero when removal is at the
    # terminal boundary; subtracting rounded hour totals can invent OUT time.
    out_time = (period_ends - removed.clip(lower=period_starts)).clip(lower=pd.Timedelta(0))
    out_hours = out_time.dt.total_seconds() / 3600.0
    panel[REINSERTION_RISK_COL] = out_time.gt(pd.Timedelta(0)).astype(np.int8)
    # The source row grid supplies a fixed follow-up schedule. A source
    # reinsertion endpoint is a simulated event only after simulated removal;
    # otherwise it is explicitly a scheduled simulation cutoff.
    panel["simulation_end_time"] = episode_ends
    impossible_reinsertion = panel["reinsertion_in_period"].eq(1) & panel[REINSERTION_RISK_COL].eq(0)
    panel.loc[impossible_reinsertion, "reinsertion_in_period"] = 0
    panel.loc[impossible_reinsertion, "episode_end_reason"] = "simulation_horizon"
    for column, values in (("interval_hours", duration), ("catheter_in_hours", in_hours),
                           ("catheter_out_hours", out_hours)):
        if column in panel:
            panel[column] = values
    if "episode_day" in panel:
        panel["episode_day"] = add_episode_day(panel)["episode_day"]
    panel = rebuild_observed_action(panel)
    panel = pec.add_reinsertion_context(panel)

    decision_rows = panel[STATE_COL].eq("in") & panel[OBSERVED_ACTION_COL].isin(
        ["keep", "remove"]
    )
    for column in OPTIONAL_DECISION_INDICATOR_COLS:
        if column in panel.columns:
            panel[column] = decision_rows.astype(bool)

    return panel, true_propensity, action_intercept

def outcome_potential_probabilities(z_values):
    # Form the outcome-model linear predictor.
    baseline_linear = np.full(len(z_values), OUTCOME_INTERCEPT, dtype=float)
    for z_col, coefficient in OUTCOME_COEFFICIENTS.items():
        baseline_linear += float(coefficient) * pd.to_numeric(
            z_values[z_col], errors="raise"
        ).to_numpy(dtype=float)
    true_mu_keep = expit(baseline_linear)
    true_mu_remove = expit(baseline_linear + TREATMENT_LOG_ODDS_EFFECT)
    # On the sequential trajectory, OUT means removal has already occurred.
    true_mu_out = true_mu_remove.copy()
    return true_mu_keep, true_mu_remove, true_mu_out

def generate_outcome(
    panel,
    synthetic_action,
    true_mu_keep,
    true_mu_remove,
    true_mu_out,
    outcome_uniform,
):
    """Draw the first CAUTI per episode and end its risk set after that event."""
    states = panel[STATE_COL].astype("string").str.strip().str.lower()
    # CAUTI remains attributable through 48 hours after removal.
    outcome_risk = pd.to_datetime(panel["period_start"]).lt(
        pd.to_datetime(panel["removed"]) + pd.Timedelta(hours=48)
    )
    in_rows = states.eq("in")
    out_rows = states.eq("out")

    observed_probability = np.zeros(len(panel), dtype=float)
    keep_rows = outcome_risk & in_rows & pd.Series(synthetic_action, index=panel.index).eq(0)
    remove_rows = outcome_risk & in_rows & pd.Series(synthetic_action, index=panel.index).eq(1)
    at_risk_out_rows = outcome_risk & out_rows
    observed_probability[keep_rows.to_numpy()] = true_mu_keep[keep_rows.to_numpy()]
    observed_probability[remove_rows.to_numpy()] = true_mu_remove[remove_rows.to_numpy()]
    observed_probability[at_risk_out_rows.to_numpy()] = true_mu_out[at_risk_out_rows.to_numpy()]

    synthetic_outcome = np.zeros(len(panel), dtype=np.int8)
    risk_positions = np.flatnonzero(outcome_risk.to_numpy())
    synthetic_outcome[risk_positions] = (
        outcome_uniform[risk_positions] < observed_probability[risk_positions]
    ).astype(np.int8)
    starts = pd.to_datetime(panel["period_start"])
    first_event = starts.where(synthetic_outcome == 1).groupby(episode_codes(panel)).transform("min")
    outcome_risk &= first_event.isna() | starts.le(first_event)
    synthetic_outcome[~outcome_risk.to_numpy()] = 0
    observed_probability[~outcome_risk.to_numpy()] = 0.0
    return synthetic_outcome, observed_probability, outcome_risk.to_numpy(dtype=np.int8)


def make_omitted_confounder_panel(primary_panel):
    omitted_columns = sorted(
        column
        for column in primary_panel.columns
        if str(column).startswith(HEART_RATE_ITEM_PREFIX)
    )
    if HEART_RATE_MEAN_COL not in omitted_columns:
        raise ValueError(
            f"The omitted-confounder block does not include {HEART_RATE_MEAN_COL}"
        )

    omitted_panel = primary_panel.copy(deep=True)
    # Replace each complete column rather than assigning through ``.loc``.
    # Recent pandas versions reject inserting NaN into the integer-typed
    # missingness indicators with LossySetitemError. Whole-column replacement
    # safely promotes both numeric aggregates and indicators to all-missing
    # float columns, which the normal fold-specific feature check classifies as
    # ``all_missing``.
    for column in omitted_columns:
        omitted_panel[column] = np.nan
    return omitted_panel


def build_truth_table(
    primary_panel,
    z_values,
    true_propensity,
    true_mu_keep,
    true_mu_remove,
    true_mu_out,
    observed_probability,
    randomised_truth,
):
    preferred_identifiers = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "inserted",
        "removed",
        "period_start",
        "period_end",
        STATE_COL,
        "episode_index",
        PERIODS_COL,
        CAUTI_RISK_COL,
        "simulation_end_time",
        "episode_end_reason",
    ]
    identifier_columns = [
        column for column in preferred_identifiers if column in primary_panel.columns
    ]

    truth = primary_panel.loc[:, identifier_columns].copy()
    truth.insert(
        0,
        VALIDATION_ROW_ID_COL,
        np.arange(1, len(primary_panel) + 1, dtype=np.int64),
    )
    truth["dgp_action_eligible"] = (
        primary_panel[STATE_COL].astype("string").str.strip().str.lower().eq("in")
    ).astype(np.int8)
    truth["dgp_outcome_at_risk"] = normalise_binary(
        primary_panel[CAUTI_RISK_COL],
        CAUTI_RISK_COL,
    )
    truth["synthetic_action"] = normalise_binary(
        primary_panel[ACTION_COL], ACTION_COL
    ).to_numpy(dtype=np.int8)
    truth["synthetic_outcome"] = normalise_binary(
        primary_panel[OUTCOME_COL], OUTCOME_COL
    ).to_numpy(dtype=np.int8)

    for z_col in z_values.columns:
        truth[f"z_dgp_{z_col.removeprefix('z_')}"] = z_values[z_col].to_numpy(dtype=float)

    truth["true_propensity_remove"] = true_propensity
    truth["true_mu_keep"] = true_mu_keep
    truth["true_mu_remove"] = true_mu_remove
    truth["true_mu_out"] = true_mu_out
    truth["true_mu_observed_action"] = observed_probability

    randomised_panel = randomised_truth["panel"]
    randomised_states = (
        randomised_panel[STATE_COL].astype("string").str.strip().str.lower()
    )
    for column in [
        "removed",
        STATE_COL,
        "episode_index",
        PERIODS_COL,
        CAUTI_RISK_COL,
        REINSERTION_RISK_COL,
        "simulation_end_time",
        "episode_end_reason",
    ]:
        truth[f"{column}_randomised"] = randomised_panel[column].to_numpy()
    truth["dgp_action_eligible_randomised"] = randomised_states.eq("in").astype(
        np.int8
    )
    truth["dgp_outcome_at_risk_randomised"] = normalise_binary(
        randomised_panel[CAUTI_RISK_COL], CAUTI_RISK_COL
    )
    for z_col in randomised_truth["z_values"].columns:
        truth[f"z_dgp_{z_col.removeprefix('z_')}_randomised"] = randomised_truth[
            "z_values"
        ][z_col].to_numpy(dtype=float)
    truth["true_propensity_remove_randomised"] = randomised_truth["propensity"]
    truth["synthetic_action_randomised"] = normalise_binary(
        randomised_panel[ACTION_COL], ACTION_COL
    ).to_numpy(dtype=np.int8)
    truth["synthetic_outcome_randomised"] = normalise_binary(
        randomised_panel[OUTCOME_COL], OUTCOME_COL
    ).to_numpy(dtype=np.int8)
    truth["true_mu_keep_randomised"] = randomised_truth["mu_keep"]
    truth["true_mu_remove_randomised"] = randomised_truth["mu_remove"]
    truth["true_mu_out_randomised"] = randomised_truth["mu_out"]
    truth["true_mu_observed_action_randomised"] = randomised_truth[
        "observed_probability"
    ]
    return truth


def build_known_truth_policy_values(primary_panel, truth, standardisation, horizon_hours=None):
    # Reuse the shared policy builder so names, days, episode IDs, and
    # fixed-day timeline semantics cannot drift from normal evaluation.
    base_columns = list(dict.fromkeys([
        *POLICY_INPUT_COLS,
        CAUTI_RISK_COL,
        AGE_COL,
        HEART_RATE_MEAN_COL,
    ]))
    policy_base = primary_panel.loc[:, base_columns].copy()
    policy_base[VALIDATION_ROW_ID_COL] = truth[VALIDATION_ROW_ID_COL].to_numpy()
    if horizon_hours is not None:
        cutoff = pd.to_datetime(policy_base.inserted) + pd.Timedelta(hours=horizon_hours)
        start, end = pd.to_datetime(policy_base.period_start), pd.to_datetime(policy_base.period_end)
        if ((start < cutoff) & (end > cutoff)).any():
            raise ValueError("Known-truth horizon cuts a period")
        policy_base = policy_base.loc[end.le(cutoff)].copy()
    policy_base = add_stable_ids_and_decision_flag(policy_base)
    policy_base = add_episode_day(policy_base)

    current_truth_columns = [
        VALIDATION_ROW_ID_COL,
        "true_mu_keep",
        "true_mu_remove",
        "true_mu_out",
        "true_propensity_remove",
    ]
    current_probability_lookup = truth.loc[:, current_truth_columns]

    # Build one full panel copy per fixed-day policy.
    policy_frames = [
        apply_fixed_day_policy(policy_base, removal_day)
        for removal_day in POLICY_DAYS
    ]
    policy_long = pd.concat(policy_frames, ignore_index=True)
    policy_state = policy_long["policy_catheter_state"].astype("string").str.lower()
    policy_periods_in = pd.to_numeric(
        policy_long["policy_periods_in"], errors="coerce"
    )
    policy_periods_out = pd.to_numeric(
        policy_long["policy_periods_out"], errors="coerce"
    )
    # The first OUT grid period is bin 1, matching the generated trajectory.
    # A legacy extra OUT row on the removal day also belongs to bin 1.
    policy_dgp_periods = pd.Series(
        np.where(policy_state.eq("in"), policy_periods_in, policy_periods_out.clip(lower=1.0)),
        index=policy_long.index,
        dtype=float,
    )
    if policy_dgp_periods.isna().any():
        raise AssertionError("Fixed-day policy rows are missing a state-duration value")

    policy_z = pd.DataFrame(index=policy_long.index)
    for source_col in [AGE_COL, HEART_RATE_MEAN_COL]:
        mean = float(standardisation[source_col]["mean"])
        sd = float(standardisation[source_col]["sd_population_ddof_0"])
        numeric = pd.to_numeric(policy_long[source_col], errors="coerce")
        policy_z[Z_COLS[source_col]] = (numeric.fillna(mean) - mean) / sd
    periods_mean = float(standardisation[PERIODS_COL]["mean"])
    periods_sd = float(standardisation[PERIODS_COL]["sd_population_ddof_0"])
    policy_z[Z_COLS[PERIODS_COL]] = (policy_dgp_periods - periods_mean) / periods_sd
    policy_mu_keep, policy_mu_remove, policy_mu_out = outcome_potential_probabilities(
        policy_z
    )

    policy_long["true_mu_under_policy"] = 0.0

    keep_rows = policy_long["policy_action"].eq("keep")
    remove_rows = policy_long["policy_action"].eq("remove")
    out_risk_rows = (
        policy_long["policy_action"].eq("out")
        & pd.to_numeric(policy_long["policy_periods_out"], errors="coerce").le(
            POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
        )
    )
    policy_long.loc[keep_rows, "true_mu_under_policy"] = policy_mu_keep[
        keep_rows.to_numpy()
    ]
    policy_long.loc[remove_rows, "true_mu_under_policy"] = policy_mu_remove[
        remove_rows.to_numpy()
    ]
    policy_long.loc[out_risk_rows, "true_mu_under_policy"] = policy_mu_out[
        out_risk_rows.to_numpy()
    ]

    fixed_group_cols = [
        "policy_name",
        "policy_type",
        "policy_remove_day",
        "catheter_episode_id",
    ]
    fixed_episode = (
        policy_long.groupby(fixed_group_cols, as_index=False, dropna=False)
        .agg(
            subject_id=("subject_id", "first"),
            n_policy_rows=(VALIDATION_ROW_ID_COL, "size"),
            known_truth_episode_any_cauti_risk=(
                "true_mu_under_policy",
                cumulative_event_probability,
            ),
        )
    )
    fixed_policy = (
        fixed_episode.groupby(
            ["policy_name", "policy_type", "policy_remove_day"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_patients=("subject_id", "nunique"),
            n_episodes=("catheter_episode_id", "size"),
            known_truth_mean_episode_any_cauti_risk=(
                "known_truth_episode_any_cauti_risk",
                "mean",
            ),
        )
    )

    # Match the g-formula current-practice calculation: use the
    # realised synthetic IN action, the OUT hazard within the existing 48-hour
    # attribution window, and zero CAUTI hazard outside that window.
    current = policy_base.merge(
        current_probability_lookup,
        on=VALIDATION_ROW_ID_COL,
        how="left",
        validate="one_to_one",
    )
    states = current[STATE_COL].astype("string").str.strip().str.lower()
    actions = normalise_binary(current[ACTION_COL], ACTION_COL)
    # Integrate all time-eligible hazards, independently of realised CAUTI.
    # Match the 48-hour CAUTI attribution window used by the evaluator.
    risk = pd.to_datetime(current["period_start"]).lt(
        pd.to_datetime(current["removed"]) + pd.Timedelta(hours=48)
    )
    current["true_mu_under_policy"] = 0.0
    current_keep = risk & states.eq("in") & actions.eq(0)
    current_remove = risk & states.eq("in") & actions.eq(1)
    current_out = risk & states.eq("out")
    current.loc[current_keep, "true_mu_under_policy"] = current.loc[
        current_keep, "true_mu_keep"
    ]
    current.loc[current_remove, "true_mu_under_policy"] = current.loc[
        current_remove, "true_mu_remove"
    ]
    current.loc[current_out, "true_mu_under_policy"] = current.loc[
        current_out, "true_mu_out"
    ]

    current_episode = (
        current.groupby("catheter_episode_id", as_index=False, dropna=False)
        .agg(
            subject_id=("subject_id", "first"),
            known_truth_episode_any_cauti_risk=(
                "true_mu_under_policy",
                cumulative_event_probability,
            ),
        )
    )
    current_policy = pd.DataFrame([
        {
            "policy_name": "current_practice",
            "policy_type": "observed",
            "policy_remove_day": np.nan,
            "n_patients": int(current_episode["subject_id"].nunique()),
            "n_episodes": int(len(current_episode)),
            "known_truth_mean_episode_any_cauti_risk": float(
                current_episode["known_truth_episode_any_cauti_risk"].mean()
            ),
        }
    ])

    known_truth = pd.concat([fixed_policy, current_policy], ignore_index=True, sort=False)
    known_truth["known_truth_mean_episode_any_cauti_risk_pct"] = (
        100.0 * known_truth["known_truth_mean_episode_any_cauti_risk"]
    )
    known_truth["target_population"] = (
        "all catheter episodes on the preserved row grid with policy-specific IN/OUT states"
    )
    known_truth["episode_aggregation"] = "1 - product(1 - row hazard)"
    known_truth["policy_aggregation"] = "unweighted mean of episode risks"
    known_truth["evaluation_horizon_hours"] = horizon_hours
    return known_truth


def build_policy_dml_known_truth(primary_panel, randomised_panel, truth, standardisation):
    columns = ["true_mu_keep", "true_mu_remove", "true_mu_out", "true_propensity_remove"]
    randomised_truth = truth[[VALIDATION_ROW_ID_COL, *[f"{col}_randomised" for col in columns]]].rename(
        columns={f"{col}_randomised": col for col in columns}
    )
    tables = []
    for name in ("semi_synthetic_with_confounding", "confounder_omitted", "randomised_action"):
        panel, lookup = (randomised_panel, randomised_truth) if name == "randomised_action" else (primary_panel, truth)
        table = build_known_truth_policy_values(panel, lookup, standardisation,
                                               horizon_hours=POLICY_DML_HORIZON_HOURS[name])
        table.insert(0, "panel_name", name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def build_coefficients_table(action_intercept, randomised_action_intercept):
    rows = [
        {
            "mechanism": "observational_action",
            "term": "intercept",
            "coefficient": action_intercept,
            "notes": "calibrated to observed action prevalence on catheter-IN rows",
        }
    ]
    rows.extend(
        {
            "mechanism": "observational_action",
            "term": term,
            "coefficient": coefficient,
            "notes": "configured log-odds coefficient before propensity clipping",
        }
        for term, coefficient in ACTION_COEFFICIENTS.items()
    )
    rows.append({
        "mechanism": "outcome",
        "term": "intercept",
        "coefficient": OUTCOME_INTERCEPT,
        "notes": "configured row-level synthetic CAUTI log-odds intercept",
    })
    rows.extend(
        {
            "mechanism": "outcome",
            "term": term,
            "coefficient": coefficient,
            "notes": "configured shared-confounder log-odds coefficient",
        }
        for term, coefficient in OUTCOME_COEFFICIENTS.items()
    )
    rows.append({
        "mechanism": "outcome",
        "term": "synthetic_removal_or_out",
        "coefficient": TREATMENT_LOG_ODDS_EFFECT,
        "notes": "known conditional causal log-odds effect; negative is protective",
    })
    rows.append({
        "mechanism": "randomised_action",
        "term": "intercept",
        "coefficient": randomised_action_intercept,
        "notes": "constant randomised assignment probability; all confounder coefficients are zero",
    })
    return pd.DataFrame(rows)


def assert_no_truth_columns(panel, panel_name):
    leaked_columns = [
        column
        for column in panel.columns
        if str(column).startswith(TRUTH_ONLY_PREFIXES)
        or column == VALIDATION_ROW_ID_COL
    ]
    if leaked_columns:
        raise AssertionError(
            f"Truth-only columns leaked into {panel_name}: {leaked_columns}"
        )


def assert_output_safety(source_path, paths):
    source_resolved = source_path.resolve()
    for label, path in paths.items():
        if path.resolve() == source_resolved:
            raise AssertionError(f"Output {label} would overwrite the observed source panel")


def main():
    source_path = SOURCE_PANEL.resolve()
    output_dir = OUTPUT_DIR.resolve()
    paths = {
        "primary_panel": output_dir / PRIMARY_PANEL_FILENAME,
        "omitted_panel": output_dir / OMITTED_PANEL_FILENAME,
        "randomised_panel": output_dir / RANDOMISED_PANEL_FILENAME,
        "truth": output_dir / TRUTH_FILENAME,
        "known_truth_policy_values": output_dir / KNOWN_TRUTH_POLICY_FILENAME,
        "policy_dml_known_truth": output_dir / KNOWN_TRUTH_DML_FILENAME,
        "coefficients": output_dir / COEFFICIENTS_FILENAME,
    }
    assert_output_safety(
        source_path,
        paths,
    )

    source_panel = load_source_panel(source_path)
    states = source_panel[STATE_COL].astype("string").str.strip().str.lower()
    decision_mask = states.eq("in")
    outcome_reference_mask = normalise_binary(
        source_panel[CAUTI_RISK_COL],
        CAUTI_RISK_COL,
    ).eq(1)
    observed_action = normalise_binary(source_panel[ACTION_COL], ACTION_COL)
    target_action_prevalence = float(observed_action.loc[decision_mask].mean())

    z_values, standardisation = build_standardised_confounders(
        source_panel,
        outcome_reference_mask,
    )

    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    action_seed, randomised_action_seed, outcome_seed = seed_sequence.spawn(3)
    action_rng = np.random.default_rng(action_seed)
    randomised_action_rng = np.random.default_rng(randomised_action_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    outcome_uniform = outcome_rng.random(len(source_panel))

    primary_trajectory, true_propensity, action_intercept = generate_sequential_trajectory(
        source_panel,
        z_values,
        standardisation,
        action_rng,
        randomised=False,
        target_prevalence=target_action_prevalence,
    )
    primary_z_values = apply_standardisation(primary_trajectory, standardisation)
    true_mu_keep, true_mu_remove, true_mu_out = outcome_potential_probabilities(
        primary_z_values
    )
    synthetic_outcome, observed_probability, synthetic_risk = generate_outcome(
        primary_trajectory,
        primary_trajectory[ACTION_COL].to_numpy(dtype=np.int8),
        true_mu_keep,
        true_mu_remove,
        true_mu_out,
        outcome_uniform,
    )
    primary_panel = primary_trajectory.copy()
    primary_panel[OUTCOME_COL] = np.asarray(synthetic_outcome, dtype=np.int8)
    primary_panel[CAUTI_RISK_COL] = synthetic_risk
    omitted_panel = make_omitted_confounder_panel(primary_panel)

    (
        randomised_trajectory,
        randomised_propensity,
        randomised_action_intercept,
    ) = generate_sequential_trajectory(
        source_panel,
        z_values,
        standardisation,
        randomised_action_rng,
        randomised=True,
        target_prevalence=target_action_prevalence,
    )
    randomised_z_values = apply_standardisation(
        randomised_trajectory, standardisation
    )
    (
        randomised_mu_keep,
        randomised_mu_remove,
        randomised_mu_out,
    ) = outcome_potential_probabilities(randomised_z_values)
    randomised_outcome, randomised_observed_probability, randomised_risk = generate_outcome(
        randomised_trajectory,
        randomised_trajectory[ACTION_COL].to_numpy(dtype=np.int8),
        randomised_mu_keep,
        randomised_mu_remove,
        randomised_mu_out,
        outcome_uniform,
    )
    randomised_panel = randomised_trajectory.copy()
    randomised_panel[OUTCOME_COL] = np.asarray(randomised_outcome, dtype=np.int8)
    randomised_panel[CAUTI_RISK_COL] = randomised_risk
    randomised_truth = {
        "panel": randomised_panel,
        "propensity": randomised_propensity,
        "z_values": randomised_z_values,
        "mu_keep": randomised_mu_keep,
        "mu_remove": randomised_mu_remove,
        "mu_out": randomised_mu_out,
        "observed_probability": randomised_observed_probability,
    }

    truth = build_truth_table(
        primary_panel,
        primary_z_values,
        true_propensity,
        true_mu_keep,
        true_mu_remove,
        true_mu_out,
        observed_probability,
        randomised_truth=randomised_truth,
    )
    known_truth_policy_values = build_known_truth_policy_values(
        primary_panel, truth, standardisation
    )
    policy_dml_known_truth = build_policy_dml_known_truth(primary_panel, randomised_panel, truth, standardisation)
    coefficients = build_coefficients_table(
        action_intercept,
        randomised_action_intercept,
    )

    assert_no_truth_columns(omitted_panel, "omitted-confounder estimator-input panel")
    assert_no_truth_columns(randomised_panel, "randomised estimator-input panel")

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_panel.to_csv(paths["primary_panel"], index=False)
    omitted_panel.to_csv(paths["omitted_panel"], index=False)
    truth.to_csv(paths["truth"], index=False)
    known_truth_policy_values.to_csv(paths["known_truth_policy_values"], index=False)
    policy_dml_known_truth.to_csv(paths["policy_dml_known_truth"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    randomised_panel.to_csv(paths["randomised_panel"], index=False)

    print("Semi-synthetic validation files created:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
