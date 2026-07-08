#!/usr/bin/env python3
"""
Create IPW-ready panels for deterministic catheter-removal policies.

Proof-of-concept policies:
    Remove catheter on days 1, 2, 3, 4, and 5.

Input:
    artifacts/nuisance_models/scored_panel.csv
    This should already contain the nuisance-model output, especially:
        - p_remove_obs
        - observed_action
        - action_remove
        - catheter_state
        - periods_in_state
        - outcome flags

Output:
    artifacts/policy_eval/ipw_policy_remove_day1_panel.csv
    artifacts/policy_eval/ipw_policy_remove_day2_panel.csv
    artifacts/policy_eval/ipw_policy_remove_day3_panel.csv
    artifacts/policy_eval/ipw_policy_remove_day4_panel.csv
    artifacts/policy_eval/ipw_policy_remove_day5_panel.csv
    artifacts/policy_eval/ipw_policy_remove_days_1_to_5_panel.csv
    artifacts/policy_eval/ipw_policy_remove_days_1_to_5_overlap_diagnostics.csv

Each policy-specific output contains only episodes that actually followed that
trial policy. For example, remove-on-day-3 means:
    day 1: keep
    day 2: keep
    day 3: remove

Non-matching episodes are excluded from the output for this policy evaluation.
This is equivalent to giving non-matching episodes zero contribution for this
specific policy estimate. The combined output stacks all policy-specific panels.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
NUISANCE_MODEL_DIR = REPO_ROOT / "artifacts" / "nuisance_models"
OUTDIR = REPO_ROOT / "artifacts" / "policy_eval"

INPUT_PATH = NUISANCE_MODEL_DIR / "scored_panel.csv"
COMBINED_OUTPUT_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_panel.csv"
OVERLAP_DIAGNOSTICS_PATH = OUTDIR / "ipw_policy_remove_days_1_to_5_overlap_diagnostics.csv"

POLICY_REMOVE_DAYS = [1, 2, 3, 4, 5]

# Clipping avoids infinite/unstable weights from very small probabilities.
# You can change this later, but 0.01 is a sensible proof-of-concept value.
PROPENSITY_CLIP_LOWER = 0.01
PROPENSITY_CLIP_UPPER = 0.99


# ---------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------

ID_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
]

ROW_ORDER_COLS = [
    "period_start",
    "period_end",
]

ACTION_COLS = [
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "action_remove",
    "p_remove_obs",
]

OUTCOME_COLS = [
    "at_risk_cauti",
    "at_risk_reinsertion",
    "cauti_in_period",
    "reinsertion_in_period",
    "death_in_period",
    "icu_end_in_period",
    "is_last_period_of_episode",
    "episode_end_reason",
]

OPTIONAL_COLS = [
    "reinsertion_time",
    "episode_index",
    "split",
]

KEEP_COLS = ID_COLS + OPTIONAL_COLS + ROW_ORDER_COLS + ACTION_COLS + OUTCOME_COLS

OUTPUT_COLS = [
    "catheter_episode_id",
    "policy_name",
    "policy_remove_day",
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
    "reinsertion_time",
    "episode_index",
    "period_start",
    "period_end",
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "action_remove",
    "p_remove_obs",
    "p_keep_obs",
    "p_observed_action",
    "p_observed_action_clipped",
    "policy_action",
    "policy_action_remove",
    "policy_support",
    "policy_support_clipped",
    "policy_weight_component",
    "matched_policy_today",
    "followed_policy_so_far",
    "episode_matches_policy",
    "ipw_component",
    "episode_ipw_weight",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "cauti_in_period",
    "reinsertion_in_period",
    "episode_cauti",
    "episode_reinsertion",
    "death_in_period",
    "icu_end_in_period",
    "is_last_period_of_episode",
    "episode_end_reason",
    "split",
]


def check_required_columns(df: pd.DataFrame) -> None:
    required = ID_COLS + ROW_ORDER_COLS + ACTION_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def policy_name_for_day(policy_remove_day: int) -> str:
    return f"remove_on_day_{policy_remove_day}"


def output_path_for_policy(policy_remove_day: int) -> Path:
    return OUTDIR / f"ipw_policy_remove_day{policy_remove_day}_panel.csv"


def valid_weight_series(weights: pd.Series) -> pd.Series:
    weights = pd.to_numeric(weights, errors="coerce")
    return weights[weights.notna() & np.isfinite(weights) & (weights > 0)]


def effective_sample_size(weights: pd.Series) -> float:
    weights = valid_weight_series(weights)
    if weights.empty:
        return np.nan
    sum_weights = float(weights.sum())
    sum_squared_weights = float(np.square(weights).sum())
    return float((sum_weights ** 2) / sum_squared_weights) if sum_squared_weights > 0 else np.nan


def support_weights(support: pd.Series) -> pd.Series:
    support = pd.to_numeric(support, errors="coerce")
    support = support[support.notna() & np.isfinite(support) & (support > 0)]
    if support.empty:
        return pd.Series(dtype=float)
    return 1.0 / support


def support_summary_fields(support: pd.Series, prefix: str = "") -> dict:
    support = pd.to_numeric(support, errors="coerce")
    valid_support = support[support.notna() & np.isfinite(support)]
    prefix = f"{prefix}_" if prefix else ""

    if valid_support.empty:
        return {
            f"{prefix}n": 0,
            f"{prefix}mean_support": np.nan,
            f"{prefix}median_support": np.nan,
            f"{prefix}min_support": np.nan,
            f"{prefix}pct_below_0_10": np.nan,
            f"{prefix}pct_below_0_05": np.nan,
            f"{prefix}pct_below_0_01": np.nan,
            f"{prefix}effective_sample_size": np.nan,
        }

    return {
        f"{prefix}n": int(len(valid_support)),
        f"{prefix}mean_support": float(valid_support.mean()),
        f"{prefix}median_support": float(valid_support.median()),
        f"{prefix}min_support": float(valid_support.min()),
        f"{prefix}pct_below_0_10": float((valid_support < 0.10).mean()),
        f"{prefix}pct_below_0_05": float((valid_support < 0.05).mean()),
        f"{prefix}pct_below_0_01": float((valid_support < 0.01).mean()),
        f"{prefix}effective_sample_size": effective_sample_size(support_weights(valid_support)),
    }


def add_policy_support(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_support"] = np.nan

    support_rows = df["is_decision_row"] & df["policy_action_remove"].isin([0, 1])
    p_remove = pd.to_numeric(df.loc[support_rows, "p_remove_obs"], errors="coerce").clip(0.0, 1.0)
    policy_remove = df.loc[support_rows, "policy_action_remove"].eq(1)

    df.loc[support_rows, "policy_support"] = np.where(
        policy_remove,
        p_remove,
        1.0 - p_remove,
    )
    df["policy_support_clipped"] = df["policy_support"].clip(
        lower=PROPENSITY_CLIP_LOWER,
        upper=PROPENSITY_CLIP_UPPER,
    )
    df["policy_weight_component"] = np.nan
    df.loc[support_rows, "policy_weight_component"] = 1.0 / df.loc[
        support_rows,
        "policy_support",
    ]
    return df


def episode_policy_support(policy_df: pd.DataFrame) -> pd.Series:
    support_rows = policy_df["policy_support"].notna()
    support_df = policy_df.loc[support_rows, ["catheter_episode_id", "policy_support"]].copy()
    support_df["policy_support"] = pd.to_numeric(support_df["policy_support"], errors="coerce")
    support_df = support_df.dropna(subset=["policy_support"])
    if support_df.empty:
        return pd.Series(dtype=float)
    return support_df.groupby("catheter_episode_id")["policy_support"].prod(min_count=1)


def overlap_summary_row(policy_df: pd.DataFrame, label: str) -> dict:
    support_rows = policy_df["policy_support"].notna()
    row_support = policy_df.loc[support_rows, "policy_support"]
    episode_support = episode_policy_support(policy_df)

    row = {
        "group": label,
        "n_policy_decision_rows": int(support_rows.sum()),
        "n_policy_decision_episodes": int(policy_df.loc[support_rows, "catheter_episode_id"].nunique()),
    }
    row.update(support_summary_fields(row_support))
    row.update(support_summary_fields(episode_support, prefix="episode"))
    return row


def overlap_summary_rows(policy_df: pd.DataFrame, policy_name: str, policy_remove_day: int) -> list[dict]:
    rows = []
    base = {"policy_name": policy_name, "policy_remove_day": policy_remove_day}
    rows.append({**base, **overlap_summary_row(policy_df, "all")})

    if "split" in policy_df.columns:
        for split_value, split_df in policy_df.groupby("split", dropna=False, sort=False):
            rows.append({**base, **overlap_summary_row(split_df, f"split={split_value}")})

    return rows


def build_policy_panel(base_df: pd.DataFrame, policy_remove_day: int):
    df = base_df.copy()
    policy_name = policy_name_for_day(policy_remove_day)

    # -----------------------------------------------------------------
    # Apply deterministic policy: keep before policy day, remove on policy day.
    # -----------------------------------------------------------------

    df["policy_name"] = policy_name
    df["policy_remove_day"] = policy_remove_day

    df["policy_action"] = "not_applicable"
    df.loc[df["is_decision_row"] & (df["periods_in_state"] < policy_remove_day), "policy_action"] = "keep"
    df.loc[df["is_decision_row"] & (df["periods_in_state"] == policy_remove_day), "policy_action"] = "remove"

    # If the row is still an IN decision row after the policy day, the patient has
    # already deviated from the corresponding remove-on-day-N policy.
    df.loc[df["is_decision_row"] & (df["periods_in_state"] > policy_remove_day), "policy_action"] = (
        "already_deviated_should_have_been_removed"
    )

    df["policy_action_remove"] = np.nan
    df.loc[df["policy_action"].eq("keep"), "policy_action_remove"] = 0
    df.loc[df["policy_action"].eq("remove"), "policy_action_remove"] = 1

    # Daily match is only meaningful for decision rows up to and including the
    # policy removal day.
    df["matched_policy_today"] = np.nan
    in_policy_decision_window = df["is_decision_row"] & (df["periods_in_state"] <= policy_remove_day)

    df.loc[in_policy_decision_window, "matched_policy_today"] = (
        df.loc[in_policy_decision_window, "action_remove"]
        == df.loc[in_policy_decision_window, "policy_action_remove"]
    ).astype(int)

    # A deviation occurs if:
    #   - the decision row is within the policy window and does not match, or
    #   - the catheter is still in after the policy removal day.
    df["deviated_from_policy_today"] = 0
    df.loc[in_policy_decision_window & df["matched_policy_today"].eq(0), "deviated_from_policy_today"] = 1
    df.loc[df["is_decision_row"] & (df["periods_in_state"] > policy_remove_day), "deviated_from_policy_today"] = 1

    df["followed_policy_so_far"] = (
        1
        - df.groupby("catheter_episode_id")["deviated_from_policy_today"].cummax()
    ).astype(int)

    # Episode fully matches only if it actually has a remove action on the policy
    # day and has no earlier/later deviation.
    removed_on_policy_day = (
        df["is_decision_row"]
        & df["periods_in_state"].eq(policy_remove_day)
        & df["action_remove"].eq(1)
    )

    episode_removed_on_policy_day = removed_on_policy_day.groupby(df["catheter_episode_id"]).transform("max").astype(int)
    episode_ever_deviated = df.groupby("catheter_episode_id")["deviated_from_policy_today"].transform("max").astype(int)

    df["episode_matches_policy"] = (
        episode_removed_on_policy_day.eq(1) & episode_ever_deviated.eq(0)
    ).astype(int)

    df = add_policy_support(df)
    overlap_rows = overlap_summary_rows(df, policy_name, policy_remove_day)

    # -----------------------------------------------------------------
    # IPW components
    # -----------------------------------------------------------------

    # Only the decision rows needed to follow the policy contribute to the
    # episode weight: keep before the policy day, then remove on that day.
    df["ipw_component"] = np.nan
    weight_rows = (
        df["episode_matches_policy"].eq(1)
        & df["is_decision_row"]
        & (df["periods_in_state"] <= policy_remove_day)
    )

    df.loc[weight_rows, "ipw_component"] = 1.0 / df.loc[weight_rows, "p_observed_action_clipped"]

    episode_weight = (
        df.loc[weight_rows, ["catheter_episode_id", "ipw_component"]]
        .groupby("catheter_episode_id")["ipw_component"]
        .prod()
        .rename("episode_ipw_weight")
    )

    df = df.merge(episode_weight, on="catheter_episode_id", how="left")

    # Keep only matching episodes for this policy-evaluation panel.
    ipw_panel = df[df["episode_matches_policy"].eq(1)].copy()
    output_cols = [c for c in OUTPUT_COLS if c in ipw_panel.columns]
    return ipw_panel[output_cols], overlap_rows


def main() -> None:
    OUTDIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(INPUT_PATH)
    check_required_columns(df)

    available_keep_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[available_keep_cols].copy()

    # Create a stable catheter-episode identifier.
    # Do not use episode_index for this: in this panel it behaves like a row index
    # within the catheter episode, not a unique episode ID.
    episode_key = ["subject_id", "hadm_id", "stay_id", "inserted", "removed"]
    df["catheter_episode_id"] = df.groupby(episode_key, sort=False).ngroup() + 1

    # Sort rows so cumulative policy-following logic is correct.
    df = df.sort_values(["catheter_episode_id", "period_start", "period_end"]).reset_index(drop=True)

    # Decision rows are the rows where the catheter is still in and the action is keep/remove.
    # OUT rows are kept later for outcome calculation but do not generate a removal propensity weight.
    df["is_decision_row"] = df["observed_action"].isin(["keep", "remove"]) & df["catheter_state"].eq("in")

    # Observed probability of the action that actually happened.
    # If actual action = remove, use p_remove_obs.
    # If actual action = keep, use 1 - p_remove_obs.
    df["p_keep_obs"] = 1.0 - df["p_remove_obs"]

    df["p_observed_action"] = np.nan
    df.loc[df["is_decision_row"] & df["action_remove"].eq(1), "p_observed_action"] = df.loc[
        df["is_decision_row"] & df["action_remove"].eq(1), "p_remove_obs"
    ]
    df.loc[df["is_decision_row"] & df["action_remove"].eq(0), "p_observed_action"] = df.loc[
        df["is_decision_row"] & df["action_remove"].eq(0), "p_keep_obs"
    ]

    df["p_observed_action_clipped"] = df["p_observed_action"].clip(
        lower=PROPENSITY_CLIP_LOWER,
        upper=PROPENSITY_CLIP_UPPER,
    )

    # Episode-level outcomes, repeated on each row so the next script can estimate
    # weighted policy outcome risks directly from this panel.
    if "cauti_in_period" in df.columns:
        df["episode_cauti"] = df.groupby("catheter_episode_id")["cauti_in_period"].transform("max").astype(int)

    if "reinsertion_in_period" in df.columns:
        df["episode_reinsertion"] = (
            df.groupby("catheter_episode_id")["reinsertion_in_period"].transform("max").astype(int)
        )

    n_total_episodes = df["catheter_episode_id"].nunique()
    policy_panels = []
    overlap_rows = []

    print(f"Input rows: {len(df):,}")
    print(f"Total catheter episodes: {n_total_episodes:,}")
    print(f"Policies: remove on days {POLICY_REMOVE_DAYS}")

    for policy_remove_day in POLICY_REMOVE_DAYS:
        ipw_panel, policy_overlap_rows = build_policy_panel(df, policy_remove_day)
        output_path = output_path_for_policy(policy_remove_day)
        ipw_panel.to_csv(output_path, index=False)
        policy_panels.append(ipw_panel)
        overlap_rows.extend(policy_overlap_rows)

        n_matching_episodes = ipw_panel["catheter_episode_id"].nunique()
        print()
        print(f"Policy: {policy_name_for_day(policy_remove_day)}")
        print(f"Matching episodes retained: {n_matching_episodes:,}")
        print(f"Mean row-level policy support: {policy_overlap_rows[0]['mean_support']:.4f}")
        print(f"Episode-level overlap ESS: {policy_overlap_rows[0]['episode_effective_sample_size']:.1f}")
        print(f"Output rows: {len(ipw_panel):,}")
        print(f"Saved: {output_path}")

    combined_panel = pd.concat(policy_panels, ignore_index=True) if policy_panels else pd.DataFrame()
    combined_panel.to_csv(COMBINED_OUTPUT_PATH, index=False)
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OVERLAP_DIAGNOSTICS_PATH, index=False)

    print()
    print(f"Combined output rows: {len(combined_panel):,}")
    print(f"Saved combined panel: {COMBINED_OUTPUT_PATH}")
    print(f"Saved overlap diagnostics: {OVERLAP_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
