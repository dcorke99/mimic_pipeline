"""Create estimator-agnostic semi-synthetic validation panels and oracle truth.

This script uses the observed longitudinal panel as a row and covariate scaffold,
but rebuilds the catheter trajectory sequentially. Each episode remains IN until
its first synthetic removal and is OUT thereafter. Oracle quantities are written
to separate files and never added to an estimator input panel.

The script is intentionally standalone and is not called by the production
pipeline. See README_validation.md for the estimand, DGP, and staging caveats.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_policy_intervention_panels import (
    EPISODE_KEY_COLS,
    POLICY_DAYS,
    POLICY_INPUT_COLS,
    add_policy_episode_day,
    add_stable_ids_and_decision_flag,
    build_long_policy_panel,
)


# Paths and reproducibility

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_PANEL = REPO_ROOT / "data" / "modelling_panel.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artefacts"
    / "validation"
    / "semi_synthetic_measured_confounding"
)
RANDOM_SEED = 20260811


# Production columns replaced in validation copies

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
#
# The action intercept is calibrated at run time so that the mean synthetic
# propensity on catheter-IN decision rows equals the real action prevalence.
# Positive shared coefficients deliberately make higher-risk rows more likely
# to receive removal, while the negative treatment coefficient makes removal
# genuinely protective conditional on the measured confounders.

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

PRIMARY_PANEL_FILENAME = "semi_synthetic_panel.csv"
OMITTED_PANEL_FILENAME = "semi_synthetic_panel_confounder_omitted.csv"
RANDOMISED_PANEL_FILENAME = "semi_synthetic_panel_randomised_action.csv"
TRUTH_FILENAME = "semi_synthetic_truth.csv"
ORACLE_POLICY_FILENAME = "oracle_policy_values.csv"
COEFFICIENTS_FILENAME = "simulation_coefficients.csv"
METADATA_FILENAME = "simulation_metadata.json"

ORACLE_ONLY_PREFIXES = (
    "true_",
    "oracle_",
    "dgp_",
    "z_dgp_",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create sequential-trajectory semi-synthetic measured-confounding panels "
            "and separate oracle truth files."
        )
    )
    parser.add_argument(
        "--source-panel",
        type=Path,
        default=DEFAULT_SOURCE_PANEL,
        help="Existing real modelling panel to copy without modifying.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Separate directory for validation panels and oracle files.",
    )
    parser.add_argument(
        "--write-randomised-action",
        action="store_true",
        help=(
            "Also write a negative-control panel whose action probability is "
            "constant and independent of the selected confounders."
        ),
    )
    parser.add_argument(
        "--overwrite-validation-outputs",
        action="store_true",
        help="Allow replacement of existing files inside the validation output directory.",
    )
    return parser.parse_args()


def expit(linear_predictor):
    values = np.asarray(linear_predictor, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def cumulative_event_probability(probabilities):
    probs = pd.to_numeric(probabilities, errors="coerce").dropna()
    if probs.empty:
        return np.nan
    values = probs.clip(0.0, 1.0).to_numpy(dtype=float)
    return float(1.0 - np.prod(1.0 - values))


def normalise_binary(series, column_name):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        examples = series.loc[numeric.isna() | ~numeric.isin([0, 1])].head(10)
        raise ValueError(f"{column_name} must contain only 0/1 values. Examples:\n{examples}")
    return numeric.astype(np.int8)


def required_source_columns():
    return {
        *EPISODE_KEY_COLS,
        *POLICY_INPUT_COLS,
        "episode_index",
        REINSERTION_RISK_COL,
        "reinsertion_in_period",
        "death_in_period",
        "icu_exit_alive_in_period",
        "episode_end_time",
        "episode_end_reason",
        AGE_COL,
        HEART_RATE_MEAN_COL,
        CAUTI_RISK_COL,
    }


def load_source_panel(path):
    if not path.is_file():
        raise FileNotFoundError(f"Source modelling panel not found: {path}")

    panel = pd.read_csv(path, low_memory=False)
    panel.columns = panel.columns.str.strip()
    if panel.columns.duplicated().any():
        duplicates = panel.columns[panel.columns.duplicated()].tolist()
        raise ValueError(f"Source panel contains duplicate columns: {duplicates}")

    missing = sorted(required_source_columns() - set(panel.columns))
    if missing:
        raise ValueError(f"Source panel is missing required columns: {missing}")

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


def linear_component(z_values, coefficients):
    result = np.zeros(len(z_values), dtype=float)
    for z_col, coefficient in coefficients.items():
        result += float(coefficient) * pd.to_numeric(
            z_values[z_col], errors="raise"
        ).to_numpy(dtype=float)
    return result


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
        reference_linear = linear_component(
            source_z_values.loc[source_decision_mask],
            ACTION_COEFFICIENTS,
        )
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
            # endpoint, matching the production panel's terminal-tie convention.
            removal_time = episode_ends.iloc[final_position]
        synthetic_removed_time[np.asarray(positions, dtype=int)] = removal_time.to_datetime64()

    panel["removed"] = pd.to_datetime(synthetic_removed_time)
    panel[STATE_COL] = pd.Series(synthetic_state, index=panel.index, dtype="string")
    panel["episode_index"] = episode_index
    panel[PERIODS_COL] = periods_in_state
    panel[ACTION_COL] = synthetic_action
    panel[CAUTI_RISK_COL] = (
        panel[STATE_COL].eq("in")
        | (panel[STATE_COL].eq("out") & panel[PERIODS_COL].le(
            POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS
        ))
    ).astype(np.int8)
    panel[REINSERTION_RISK_COL] = panel[STATE_COL].eq("out").astype(np.int8)
    panel = rebuild_observed_action(panel)

    decision_rows = panel[STATE_COL].eq("in") & panel[OBSERVED_ACTION_COL].isin(
        ["keep", "remove"]
    )
    for column in OPTIONAL_DECISION_INDICATOR_COLS:
        if column in panel.columns:
            panel[column] = decision_rows.astype(bool)

    return panel, true_propensity, action_intercept


def outcome_potential_probabilities(z_values):
    baseline_linear = OUTCOME_INTERCEPT + linear_component(
        z_values,
        OUTCOME_COEFFICIENTS,
    )
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
    states = panel[STATE_COL].astype("string").str.strip().str.lower()
    outcome_risk = normalise_binary(panel[CAUTI_RISK_COL], CAUTI_RISK_COL).eq(1)
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
    return synthetic_outcome, observed_probability


def attach_synthetic_outcome(panel, synthetic_outcome):
    panel = panel.copy(deep=True)
    panel[OUTCOME_COL] = np.asarray(synthetic_outcome, dtype=np.int8)
    return panel


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
    return omitted_panel, omitted_columns


def build_truth_table(
    primary_panel,
    z_values,
    true_propensity,
    true_mu_keep,
    true_mu_remove,
    true_mu_out,
    observed_probability,
    randomised_truth=None,
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

    if randomised_truth is not None:
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


def assert_oracle_policy_trajectory(policy_long):
    for _, episode in policy_long.groupby(
        ["policy_name", "catheter_episode_id"], sort=False, dropna=False
    ):
        actions = (
            episode["policy_action_resolved"].astype("string").str.strip().str.lower()
            .to_numpy(dtype=object)
        )
        states = (
            episode["policy_catheter_state"].astype("string").str.strip().str.lower()
            .to_numpy(dtype=object)
        )
        remove_offsets = np.flatnonzero(actions == "remove")
        if len(remove_offsets) > 1:
            raise AssertionError("An oracle fixed-day path has multiple removal rows")
        if len(remove_offsets) == 1:
            removal_offset = int(remove_offsets[0])
            if states[removal_offset] != "in":
                raise AssertionError("An oracle removal does not occur while catheter-IN")
            if not np.all(actions[:removal_offset] == "keep"):
                raise AssertionError("An oracle path is not keep before first removal")
            if not np.all(states[: removal_offset + 1] == "in"):
                raise AssertionError("An oracle path leaves IN before its removal")
            if not np.all(actions[removal_offset + 1 :] == "out"):
                raise AssertionError("An oracle path has decisions after removal")
            if not np.all(states[removal_offset + 1 :] == "out"):
                raise AssertionError("An oracle path returns IN after removal")
        elif not (np.all(actions == "keep") and np.all(states == "in")):
            raise AssertionError(
                "An oracle episode not reaching its removal day must remain IN/keep"
            )


def build_oracle_policy_values(primary_panel, truth, standardisation):
    # Reuse the production policy builder so names, days, episode IDs, and
    # fixed-day timeline semantics cannot drift from normal evaluation.
    base_columns = list(dict.fromkeys([
        *POLICY_INPUT_COLS,
        CAUTI_RISK_COL,
        AGE_COL,
        HEART_RATE_MEAN_COL,
    ]))
    policy_base = primary_panel.loc[:, base_columns].copy()
    policy_base[VALIDATION_ROW_ID_COL] = truth[VALIDATION_ROW_ID_COL].to_numpy()
    policy_base = add_stable_ids_and_decision_flag(policy_base)
    policy_base = add_policy_episode_day(policy_base)

    current_truth_columns = [
        VALIDATION_ROW_ID_COL,
        "true_mu_keep",
        "true_mu_remove",
        "true_mu_out",
        "true_propensity_remove",
    ]
    current_probability_lookup = truth.loc[:, current_truth_columns]

    policy_long = build_long_policy_panel(policy_base, POLICY_DAYS)
    assert_oracle_policy_trajectory(policy_long)
    policy_state = policy_long["policy_catheter_state"].astype("string").str.lower()
    policy_periods_in = pd.to_numeric(
        policy_long["policy_periods_in"], errors="coerce"
    )
    policy_periods_out = pd.to_numeric(
        policy_long["policy_periods_out"], errors="coerce"
    )
    # Production periods_in_state is one-based in either state. The policy
    # helper uses zero for an extra OUT row on the removal day, so translate it
    # to one-based values before applying the outcome DGP.
    policy_dgp_periods = pd.Series(
        np.where(policy_state.eq("in"), policy_periods_in, policy_periods_out + 1.0),
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

    keep_rows = policy_long["policy_action_resolved"].eq("keep")
    remove_rows = policy_long["policy_action_resolved"].eq("remove")
    out_risk_rows = (
        policy_long["policy_action_resolved"].eq("out")
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
            oracle_episode_any_cauti_risk=(
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
            oracle_mean_episode_any_cauti_risk=(
                "oracle_episode_any_cauti_risk",
                "mean",
            ),
        )
    )

    # Match the production current-practice plug-in convention: use the
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
    risk = normalise_binary(current[CAUTI_RISK_COL], CAUTI_RISK_COL).eq(1)
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
            oracle_episode_any_cauti_risk=(
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
            "oracle_mean_episode_any_cauti_risk": float(
                current_episode["oracle_episode_any_cauti_risk"].mean()
            ),
        }
    ])

    oracle = pd.concat([fixed_policy, current_policy], ignore_index=True, sort=False)
    oracle["oracle_mean_episode_any_cauti_risk_pct"] = (
        100.0 * oracle["oracle_mean_episode_any_cauti_risk"]
    )
    oracle["target_population"] = (
        "all catheter episodes on the preserved row grid with policy-specific IN/OUT states"
    )
    oracle["episode_aggregation"] = "1 - product(1 - row hazard)"
    oracle["policy_aggregation"] = "unweighted mean of episode risks"
    return oracle


def build_coefficients_table(action_intercept, randomised_action_intercept=None):
    rows = [
        {
            "mechanism": "observational_action",
            "term": "intercept",
            "coefficient": action_intercept,
            "notes": "calibrated to real action prevalence on catheter-IN rows",
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
    if randomised_action_intercept is not None:
        rows.append({
            "mechanism": "randomised_action",
            "term": "intercept",
            "coefficient": randomised_action_intercept,
            "notes": "constant randomised assignment probability; all confounder coefficients are zero",
        })
    return pd.DataFrame(rows)


def state_derived_columns(panel):
    columns = {
        "removed",
        "episode_index",
        STATE_COL,
        PERIODS_COL,
        ACTION_COL,
        ACTION_REMOVE_COL,
        OBSERVED_ACTION_COL,
        OUTCOME_COL,
        CAUTI_RISK_COL,
        REINSERTION_RISK_COL,
        *OPTIONAL_DECISION_INDICATOR_COLS,
    }
    return [column for column in panel.columns if column in columns]


def assert_trajectory_integrity(panel):
    states = panel[STATE_COL].astype("string").str.strip().str.lower()
    actions = normalise_binary(panel[ACTION_COL], ACTION_COL)
    action_remove = normalise_binary(panel[ACTION_REMOVE_COL], ACTION_REMOVE_COL)
    outcomes = normalise_binary(panel[OUTCOME_COL], OUTCOME_COL)
    periods = pd.to_numeric(panel[PERIODS_COL], errors="coerce")
    actual_cauti_risk = normalise_binary(panel[CAUTI_RISK_COL], CAUTI_RISK_COL)
    actual_reinsertion_risk = normalise_binary(
        panel[REINSERTION_RISK_COL], REINSERTION_RISK_COL
    )

    if not states.isin(["in", "out"]).all():
        raise AssertionError("Synthetic catheter_state contains values other than IN/OUT")
    if actions.loc[states.eq("out")].ne(0).any():
        raise AssertionError("removed_in_period must be zero on every OUT row")
    if periods.isna().any() or periods.lt(1).any():
        raise AssertionError("periods_in_state must be a positive integer on every row")
    if outcomes.loc[actual_cauti_risk.eq(0)].ne(0).any():
        raise AssertionError("Synthetic CAUTI must be zero outside the recalculated risk set")

    expected_cauti_risk = (
        states.eq("in")
        | (states.eq("out") & periods.le(POST_REMOVAL_CAUTI_ATTRIBUTION_PERIODS))
    ).astype(np.int8)
    expected_reinsertion_risk = states.eq("out").astype(np.int8)
    if not actual_cauti_risk.equals(expected_cauti_risk):
        raise AssertionError("at_risk_cauti disagrees with the synthetic catheter path")
    if not actual_reinsertion_risk.equals(expected_reinsertion_risk):
        raise AssertionError("at_risk_reinsertion disagrees with the synthetic catheter path")

    expected_observed_action = pd.Series("out", index=panel.index, dtype="string")
    expected_observed_action.loc[states.eq("in") & actions.eq(0)] = "keep"
    expected_observed_action.loc[states.eq("in") & actions.eq(1)] = "remove"
    actual_observed_action = panel[OBSERVED_ACTION_COL].astype("string")
    if not actual_observed_action.equals(expected_observed_action):
        raise AssertionError("observed_action is inconsistent with state and removal")
    expected_action_remove = (
        states.eq("in")
        & actual_observed_action.eq("remove")
        & actions.eq(1)
    ).astype(np.int8)
    if not action_remove.equals(expected_action_remove):
        raise AssertionError(
            "action_remove is inconsistent with catheter_state, observed_action, "
            "or removed_in_period"
        )

    expected_decision_rows = states.eq("in")
    for column in OPTIONAL_DECISION_INDICATOR_COLS:
        if column not in panel.columns:
            continue
        actual = panel[column].fillna(False).astype(bool)
        if not actual.equals(expected_decision_rows):
            raise AssertionError(f"{column} disagrees with the synthetic catheter state")

    codes = episode_codes(panel)
    positions_by_episode = chronological_episode_positions(panel, codes)
    period_start = pd.to_datetime(panel["period_start"], errors="coerce")
    period_end = pd.to_datetime(panel["period_end"], errors="coerce")
    episode_end = pd.to_datetime(panel["episode_end_time"], errors="coerce")
    removed_time = pd.to_datetime(panel["removed"], errors="coerce")
    if episode_end.isna().any() or removed_time.isna().any():
        raise AssertionError("Synthetic episode/removal timestamps must be valid")

    terminal_events = panel.loc[:, TERMINAL_EVENT_COLS].apply(
        pd.to_numeric, errors="coerce"
    )
    if terminal_events.isna().any().any() or not terminal_events.isin([0, 1]).all().all():
        raise AssertionError("Terminal-event fields must be binary")

    for positions_list in positions_by_episode:
        positions = np.asarray(positions_list, dtype=int)
        episode_actions = actions.iloc[positions].to_numpy(dtype=np.int8)
        episode_states = states.iloc[positions].to_numpy(dtype=object)
        episode_periods = periods.iloc[positions].to_numpy(dtype=int)
        removal_offsets = np.flatnonzero(episode_actions == 1)
        if len(removal_offsets) > 1:
            raise AssertionError("A synthetic catheter episode has more than one removal")

        if len(removal_offsets) == 1:
            removal_offset = int(removal_offsets[0])
            expected_states = np.array(
                ["in"] * (removal_offset + 1)
                + ["out"] * (len(positions) - removal_offset - 1),
                dtype=object,
            )
            expected_periods = np.concatenate([
                np.arange(1, removal_offset + 2, dtype=int),
                np.arange(1, len(positions) - removal_offset, dtype=int),
            ])
            post_positions = positions[removal_offset + 1 :]
            if episode_actions[removal_offset + 1 :].any():
                raise AssertionError("A removal decision occurs after the first removal")
            if len(post_positions) and not panel.iloc[post_positions][
                OBSERVED_ACTION_COL
            ].astype("string").eq("out").all():
                raise AssertionError("A post-removal row has a keep/remove observed_action")

            removal_position = positions[removal_offset]
            timestamp = removed_time.iloc[removal_position]
            if not (
                timestamp > period_start.iloc[removal_position]
                and timestamp <= period_end.iloc[removal_position]
                and timestamp < episode_end.iloc[removal_position]
            ):
                raise AssertionError("Synthetic removed time is outside its removal period")
        else:
            expected_states = np.array(["in"] * len(positions), dtype=object)
            expected_periods = np.arange(1, len(positions) + 1, dtype=int)
            final_position = positions[-1]
            if not removed_time.iloc[final_position] == episode_end.iloc[final_position]:
                raise AssertionError(
                    "An episode without synthetic removal must remain IN to its endpoint"
                )

        if not np.array_equal(episode_states, expected_states):
            raise AssertionError("An episode does not remain OUT after its first removal")
        if not np.array_equal(episode_periods, expected_periods):
            raise AssertionError("periods_in_state is incoherent within an episode")
        if not np.array_equal(
            pd.to_numeric(panel.iloc[positions]["episode_index"], errors="coerce").to_numpy(),
            np.arange(len(positions)),
        ):
            raise AssertionError("episode_index is not chronological and zero-based")
        if removed_time.iloc[positions].nunique(dropna=False) != 1:
            raise AssertionError("removed is not constant within a catheter episode")
        if period_end.iloc[positions].gt(episode_end.iloc[positions]).any():
            raise AssertionError("A row occurs after the preserved terminal endpoint")
        if period_end.iloc[positions[-1]] != episode_end.iloc[positions[-1]]:
            raise AssertionError("The final row does not end at the preserved endpoint")

        terminal_count = terminal_events.iloc[positions].sum(axis=1).to_numpy(dtype=int)
        expected_terminal = np.zeros(len(positions), dtype=int)
        expected_terminal[-1] = 1
        if not np.array_equal(terminal_count, expected_terminal):
            raise AssertionError("Terminal events must occur exactly once on the final row")


def assert_primary_integrity(source_panel, primary_panel):
    expected_columns = list(source_panel.columns)
    if ACTION_REMOVE_COL not in expected_columns:
        expected_columns.append(ACTION_REMOVE_COL)
    if list(primary_panel.columns) != expected_columns:
        raise AssertionError(
            "Primary semi-synthetic panel changed source columns or column order "
            "beyond the derived action_remove field"
        )
    source_positions = source_panel.index.get_indexer(primary_panel.index)
    if (source_positions < 0).any() or not np.all(np.diff(source_positions) > 0):
        raise AssertionError("Retained rows are not an order-preserving source-panel subset")

    retained_source = source_panel.loc[primary_panel.index]
    for identifier in ["subject_id", "hadm_id", "stay_id", *EPISODE_IDENTITY_COLS]:
        if identifier not in source_panel.columns:
            continue
        if set(source_panel[identifier].dropna()) != set(primary_panel[identifier].dropna()):
            raise AssertionError(f"The retained panel changed the set of {identifier} values")
    if "catheter_episode_id" in source_panel.columns:
        if set(source_panel["catheter_episode_id"].dropna()) != set(
            primary_panel["catheter_episode_id"].dropna()
        ):
            raise AssertionError("The retained panel changed catheter episode IDs")

    source_episode_keys = {
        tuple(row)
        for row in source_panel.loc[:, EPISODE_IDENTITY_COLS].astype("string").itertuples(
            index=False, name=None
        )
    }
    primary_episode_keys = {
        tuple(row)
        for row in primary_panel.loc[:, EPISODE_IDENTITY_COLS].astype("string").itertuples(
            index=False, name=None
        )
    }
    if source_episode_keys != primary_episode_keys:
        raise AssertionError("The retained panel changed catheter episode identities")

    intended_changes = set(state_derived_columns(source_panel))
    ordinary_columns = [
        column for column in source_panel.columns if column not in intended_changes
    ]
    pd.testing.assert_frame_equal(
        retained_source.loc[:, ordinary_columns],
        primary_panel.loc[:, ordinary_columns],
        check_dtype=True,
        check_exact=True,
    )
    if not retained_source.loc[:, ordinary_columns].isna().equals(
        primary_panel.loc[:, ordinary_columns].isna()
    ):
        raise AssertionError("Ordinary covariate missingness patterns changed")
    pd.testing.assert_frame_equal(
        retained_source.loc[:, TERMINAL_EVENT_COLS],
        primary_panel.loc[:, TERMINAL_EVENT_COLS],
        check_dtype=True,
        check_exact=True,
    )
    assert_trajectory_integrity(primary_panel)


def assert_no_oracle_columns(panel, panel_name):
    leaked_columns = [
        column
        for column in panel.columns
        if str(column).startswith(ORACLE_ONLY_PREFIXES)
        or column == VALIDATION_ROW_ID_COL
    ]
    if leaked_columns:
        raise AssertionError(
            f"Oracle-only columns leaked into {panel_name}: {leaked_columns}"
        )


def assert_truth_safety(primary_panel, truth, probability_arrays):
    if len(truth) != len(primary_panel):
        raise AssertionError("Truth table row count does not match the validation panel")
    expected_ids = np.arange(1, len(primary_panel) + 1, dtype=np.int64)
    if not np.array_equal(truth[VALIDATION_ROW_ID_COL].to_numpy(), expected_ids):
        raise AssertionError("Validation truth row IDs are not stable and sequential")

    assert_no_oracle_columns(primary_panel, "primary estimator-input panel")

    eligible = normalise_binary(truth["dgp_action_eligible"], "dgp_action_eligible").eq(1)
    propensity = pd.to_numeric(truth["true_propensity_remove"], errors="coerce")
    if propensity.loc[eligible].isna().any() or propensity.loc[~eligible].notna().any():
        raise AssertionError("Primary truth propensities do not match sequential decision rows")
    if not np.array_equal(
        truth["synthetic_action"].to_numpy(dtype=np.int8),
        normalise_binary(primary_panel[ACTION_COL], ACTION_COL).to_numpy(dtype=np.int8),
    ):
        raise AssertionError("Primary truth actions do not match the estimator-input panel")
    if not np.array_equal(
        truth["synthetic_outcome"].to_numpy(dtype=np.int8),
        normalise_binary(primary_panel[OUTCOME_COL], OUTCOME_COL).to_numpy(dtype=np.int8),
    ):
        raise AssertionError("Primary truth outcomes do not match the estimator-input panel")

    if "true_propensity_remove_randomised" in truth.columns:
        randomised_eligible = normalise_binary(
            truth["dgp_action_eligible_randomised"],
            "dgp_action_eligible_randomised",
        ).eq(1)
        randomised_propensity = pd.to_numeric(
            truth["true_propensity_remove_randomised"], errors="coerce"
        )
        if (
            randomised_propensity.loc[randomised_eligible].isna().any()
            or randomised_propensity.loc[~randomised_eligible].notna().any()
        ):
            raise AssertionError(
                "Randomised truth propensities do not match sequential decision rows"
            )

    for label, values in probability_arrays.items():
        numeric = np.asarray(values, dtype=float)
        present = numeric[np.isfinite(numeric)]
        if present.size and ((present < 0.0) | (present > 1.0)).any():
            raise AssertionError(f"{label} contains probabilities outside [0, 1]")


def assert_omitted_panel(primary_panel, omitted_panel, omitted_columns):
    if len(omitted_panel) != len(primary_panel):
        raise AssertionError("Omitted-confounder panel changed the row count")
    if list(omitted_panel.columns) != list(primary_panel.columns):
        raise AssertionError("Omitted-confounder panel changed the panel schema")
    if omitted_panel.loc[:, omitted_columns].notna().any().any():
        raise AssertionError("The selected heart-rate confounder block remains available")

    retained_columns = [
        column for column in primary_panel.columns if column not in omitted_columns
    ]
    pd.testing.assert_frame_equal(
        primary_panel.loc[:, retained_columns],
        omitted_panel.loc[:, retained_columns],
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        primary_panel[ACTION_COL],
        omitted_panel[ACTION_COL],
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        primary_panel[OUTCOME_COL],
        omitted_panel[OUTCOME_COL],
        check_dtype=True,
        check_exact=True,
    )


def output_paths(output_dir, include_randomised):
    paths = {
        "primary_panel": output_dir / PRIMARY_PANEL_FILENAME,
        "omitted_panel": output_dir / OMITTED_PANEL_FILENAME,
        "truth": output_dir / TRUTH_FILENAME,
        "oracle_policy_values": output_dir / ORACLE_POLICY_FILENAME,
        "coefficients": output_dir / COEFFICIENTS_FILENAME,
        "metadata": output_dir / METADATA_FILENAME,
    }
    if include_randomised:
        paths["randomised_panel"] = output_dir / RANDOMISED_PANEL_FILENAME
    return paths


def assert_output_safety(source_path, paths, overwrite):
    source_resolved = source_path.resolve()
    for label, path in paths.items():
        if path.resolve() == source_resolved:
            raise AssertionError(f"Output {label} would overwrite the real source panel")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Validation output already exists: {path}. Use "
                "--overwrite-validation-outputs only after reviewing it."
            )


def build_metadata(
    source_path,
    output_dir,
    source_panel,
    primary_panel,
    standardisation,
    action_intercept,
    true_propensity,
    observed_probability,
    omitted_columns,
    write_randomised,
):
    source_states = source_panel[STATE_COL].astype("string").str.strip().str.lower()
    source_decision_mask = source_states.eq("in")
    primary_states = primary_panel[STATE_COL].astype("string").str.strip().str.lower()
    decision_mask = primary_states.eq("in")
    outcome_risk = normalise_binary(
        primary_panel[CAUTI_RISK_COL], CAUTI_RISK_COL
    ).eq(1)
    real_action = normalise_binary(source_panel[ACTION_COL], ACTION_COL)
    synthetic_action = normalise_binary(primary_panel[ACTION_COL], ACTION_COL)
    synthetic_outcome = normalise_binary(primary_panel[OUTCOME_COL], OUTCOME_COL)

    return {
        "source_panel": str(source_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "generation_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "scenario": "semi_synthetic_measured_confounding_sequential_catheter_trajectory",
        "selected_confounders": [
            {
                "column": AGE_COL,
                "description": "age",
                "timing": "baseline/pre-decision",
            },
            {
                "column": PERIODS_COL,
                "description": "number of periods already spent in the current catheter state",
                "timing": "available at the current decision",
            },
            {
                "column": HEART_RATE_MEAN_COL,
                "description": "mean heart rate in the production 24-hour lookback ending at period_start",
                "timing": "pre-decision chart covariate",
            },
        ],
        "standardisation_reference": (
            "all source rows with at_risk_cauti == 1; missing continuous values "
            "are mean-imputed only inside the DGP so their z-score is zero"
        ),
        "standardisation": standardisation,
        "action_dgp": {
            "eligible_rows": (
                "sequential synthetic catheter_state == 'in'; eligibility ends "
                "permanently at the first synthetic removal"
            ),
            "formula": (
                "clip(expit(alpha_0 + 0.45*z_age + "
                "0.70*z_periods_in_state + 0.80*z_heart_rate_mean), 0.05, 0.95)"
            ),
            "alpha_0_calibrated": action_intercept,
            "coefficients": ACTION_COEFFICIENTS,
            "probability_bounds": [
                ACTION_PROBABILITY_LOWER,
                ACTION_PROBABILITY_UPPER,
            ],
            "intercept_calibration_target": (
                "real action prevalence on source catheter-IN rows; the realised "
                "sequential prevalence can differ because later decisions cease after removal"
            ),
        },
        "outcome_dgp": {
            "outcome": "row-level synthetic CAUTI indicator",
            "risk_rows": (
                "all synthetic IN rows and the first two synthetic OUT periods; "
                "zero outside this recalculated risk set"
            ),
            "formula_keep": (
                "expit(-2.40 + 0.40*z_age + 0.55*z_periods_in_state + "
                "0.75*z_heart_rate_mean)"
            ),
            "formula_remove": (
                "expit(-2.40 + 0.40*z_age + 0.55*z_periods_in_state + "
                "0.75*z_heart_rate_mean - 0.90)"
            ),
            "formula_out": "same as formula_remove on eligible synthetic OUT follow-up rows",
            "intercept": OUTCOME_INTERCEPT,
            "confounder_coefficients": OUTCOME_COEFFICIENTS,
            "treatment_log_odds_effect_tau": TREATMENT_LOG_ODDS_EFFECT,
            "causal_interpretation": (
                "Removal is conditionally protective on the row-level outcome odds. "
                "Positive shared confounder coefficients make higher-risk rows more "
                "likely to be removed, deliberately biasing crude comparisons toward "
                "less protection, the null, or harm."
            ),
            "row_draws": (
                "independent Bernoulli draws conditional on action and measured "
                "pre-decision covariates; repeated interval events within an episode "
                "are permitted for this code-validation endpoint"
            ),
        },
        "prevalence": {
            "real_action_prevalence_on_decision_rows": float(
                real_action.loc[source_decision_mask].mean()
            ),
            "intended_synthetic_action_prevalence": float(
                real_action.loc[source_decision_mask].mean()
            ),
            "mean_true_synthetic_propensity_on_decision_rows": float(
                np.nanmean(true_propensity)
            ),
            "realised_synthetic_action_prevalence_on_decision_rows": float(
                synthetic_action.loc[decision_mask].mean()
            ),
            "intended_synthetic_event_prevalence_on_risk_rows": float(
                np.asarray(observed_probability)[outcome_risk.to_numpy()].mean()
            ),
            "realised_synthetic_event_prevalence_on_risk_rows": float(
                synthetic_outcome.loc[outcome_risk].mean()
            ),
        },
        "replaced_columns": {
            "synthetic_removal_time": "removed",
            "catheter_state": STATE_COL,
            "episode_index": "episode_index",
            "periods_in_state": PERIODS_COL,
            "action": ACTION_COL,
            "binary_action": ACTION_REMOVE_COL,
            "derived_action_label": OBSERVED_ACTION_COL,
            "cauti_risk_set": CAUTI_RISK_COL,
            "reinsertion_risk_set": REINSERTION_RISK_COL,
            "outcome": OUTCOME_COL,
        },
        "unchanged_observed_outcomes": [
            "reinsertion_in_period",
            "death_in_period",
            "icu_exit_alive_in_period",
        ],
        "recalculated_state_dependent_fields": [
            "removed",
            STATE_COL,
            "episode_index",
            PERIODS_COL,
            ACTION_COL,
            ACTION_REMOVE_COL,
            OBSERVED_ACTION_COL,
            CAUTI_RISK_COL,
            REINSERTION_RISK_COL,
            *[
                column
                for column in OPTIONAL_DECISION_INDICATOR_COLS
                if column in primary_panel.columns
            ],
        ],
        "omitted_confounder": {
            "selected_information": "Heart Rate (MIMIC itemid 220045)",
            "method": (
                "set every itemid_220045__* aggregation and missingness indicator "
                "to all-missing; normal fold-specific feature handling removes them"
            ),
            "columns": omitted_columns,
        },
        "optional_randomised_action_panel_written": bool(write_randomised),
        "oracle": {
            "row_truth_file": TRUTH_FILENAME,
            "policy_truth_file": ORACLE_POLICY_FILENAME,
            "policies_source": "build_policy_intervention_panels.POLICY_DAYS and production helper functions",
            "policy_days": list(POLICY_DAYS),
            "policy_names": [f"remove_on_day_{day}" for day in POLICY_DAYS],
            "target_population": (
                "all catheter episodes on the preserved row grid with policy-specific IN/OUT states"
            ),
            "trajectory_rule": (
                "IN/keep before the fixed removal day, IN/remove on its first row, "
                "then OUT with no further keep/remove decisions"
            ),
            "episode_aggregation": "1 - product(1 - row hazard)",
            "policy_aggregation": "equally weighted mean across catheter episodes",
        },
        "trajectory_scaffold": {
            "source_row_count": int(len(source_panel)),
            "retained_row_count": int(len(primary_panel)),
            "dropped_row_count": int(len(source_panel) - len(primary_panel)),
            "patients": int(source_panel["subject_id"].nunique()),
            "stays": int(source_panel["stay_id"].nunique()),
            "stable_episode_identity_columns": EPISODE_IDENTITY_COLS,
            "downstream_episode_key": EPISODE_KEY_COLS,
            "statement": (
                "Synthetic removal determines the subsequent catheter IN/OUT path. "
                "The first removal ends keep/remove decisions; retained later rows are "
                "OUT follow-up rows. Clinical covariates and terminal events remain fixed."
            ),
            "post_removal_rows": (
                "retained through the existing terminal endpoint; the first two OUT "
                "periods remain in the CAUTI attribution risk set"
            ),
            "episodes_without_synthetic_removal": (
                "remain IN through the existing terminal endpoint; removed is set equal "
                "to episode_end_time and no removal is forced"
            ),
        },
        "limitations": [
            "Tests measured-confounding adjustment while preserving the observed longitudinal EHR scaffold.",
            "Does not validate treatment-confounder feedback in clinical covariates.",
            "Does not simulate action-induced changes in later clinical covariate values.",
            "Does not simulate policy-induced changes in episode duration.",
            "Does not simulate terminal-event evolution.",
            "Post-removal catheter state and deterministic risk sets are rebuilt, but future clinical state variables are not simulated.",
            "Does not test hidden confounding.",
            "Does not establish full longitudinal identification.",
            "Does not test severe positivity failure because action probabilities are bounded to [0.05, 0.95].",
            "Repeated row-level synthetic CAUTI events within an episode are permitted; the target endpoint is any event.",
        ],
    }


def main():
    args = parse_args()
    source_path = args.source_panel.resolve()
    output_dir = args.output_dir.resolve()
    paths = output_paths(output_dir, args.write_randomised_action)
    assert_output_safety(
        source_path,
        paths,
        args.overwrite_validation_outputs,
    )

    source_panel = load_source_panel(source_path)
    states = source_panel[STATE_COL].astype("string").str.strip().str.lower()
    decision_mask = states.eq("in")
    outcome_reference_mask = normalise_binary(
        source_panel[CAUTI_RISK_COL],
        CAUTI_RISK_COL,
    ).eq(1)
    real_action = normalise_binary(source_panel[ACTION_COL], ACTION_COL)
    target_action_prevalence = float(real_action.loc[decision_mask].mean())

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
    synthetic_outcome, observed_probability = generate_outcome(
        primary_trajectory,
        primary_trajectory[ACTION_COL].to_numpy(dtype=np.int8),
        true_mu_keep,
        true_mu_remove,
        true_mu_out,
        outcome_uniform,
    )
    primary_panel = attach_synthetic_outcome(primary_trajectory, synthetic_outcome)
    omitted_panel, omitted_columns = make_omitted_confounder_panel(primary_panel)

    randomised_panel = None
    randomised_propensity = None
    randomised_action_intercept = None
    randomised_truth = None
    if args.write_randomised_action:
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
        randomised_outcome, randomised_observed_probability = generate_outcome(
            randomised_trajectory,
            randomised_trajectory[ACTION_COL].to_numpy(dtype=np.int8),
            randomised_mu_keep,
            randomised_mu_remove,
            randomised_mu_out,
            outcome_uniform,
        )
        randomised_panel = attach_synthetic_outcome(
            randomised_trajectory, randomised_outcome
        )
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
    oracle_policy_values = build_oracle_policy_values(
        primary_panel, truth, standardisation
    )
    coefficients = build_coefficients_table(
        action_intercept,
        randomised_action_intercept,
    )
    metadata = build_metadata(
        source_path,
        output_dir,
        source_panel,
        primary_panel,
        standardisation,
        action_intercept,
        true_propensity,
        observed_probability,
        omitted_columns,
        args.write_randomised_action,
    )

    # All safety checks run before any validation output is written.
    assert_primary_integrity(source_panel, primary_panel)
    assert_omitted_panel(primary_panel, omitted_panel, omitted_columns)
    assert_no_oracle_columns(omitted_panel, "omitted-confounder estimator-input panel")
    probability_arrays = {
        "true_propensity_remove": true_propensity,
        "true_mu_keep": true_mu_keep,
        "true_mu_remove": true_mu_remove,
        "true_mu_out": true_mu_out,
        "true_mu_observed_action": observed_probability,
    }
    if randomised_propensity is not None:
        probability_arrays["true_propensity_remove_randomised"] = randomised_propensity
        probability_arrays["true_mu_keep_randomised"] = randomised_truth["mu_keep"]
        probability_arrays["true_mu_remove_randomised"] = randomised_truth["mu_remove"]
        probability_arrays["true_mu_out_randomised"] = randomised_truth["mu_out"]
        probability_arrays["true_mu_observed_action_randomised"] = randomised_truth[
            "observed_probability"
        ]
    assert_truth_safety(primary_panel, truth, probability_arrays)
    if randomised_panel is not None:
        assert_primary_integrity(source_panel, randomised_panel)
        assert_no_oracle_columns(randomised_panel, "randomised estimator-input panel")

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_panel.to_csv(paths["primary_panel"], index=False)
    omitted_panel.to_csv(paths["omitted_panel"], index=False)
    truth.to_csv(paths["truth"], index=False)
    oracle_policy_values.to_csv(paths["oracle_policy_values"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    if randomised_panel is not None:
        randomised_panel.to_csv(paths["randomised_panel"], index=False)

    print("Semi-synthetic validation files created:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
