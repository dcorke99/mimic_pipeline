#!/usr/bin/env python3
# Build estimator-agnostic target-counterfactual policy panels


from pathlib import Path
import re

import numpy as np
import pandas as pd

import policy_eval_common as pec
from policy_eval_common import add_episode_day
import pipeline_config as run_config
from pipeline_config import POLICY_DAYS


# Configuration

REPO_ROOT = Path(__file__).resolve().parent

# Run selections and panel paths come from pipeline_config.py.
PANEL_RUNS = tuple((name, *run_config.PANEL_PATHS[name]) for name in run_config.POLICY_PANELS)

# Initial paths for direct single-panel calls use the first selected panel.
_PANEL_NAME, INPUT_PATH, _NUISANCE_ROOT = (
    PANEL_RUNS[0] if PANEL_RUNS else ("observed", *run_config.PANEL_PATHS["observed"])
)
OUTDIR = run_config.POLICY_DIRS[_PANEL_NAME]


# Input and output column definitions

EPISODE_KEY_COLS = [
    "subject_id",
    "stay_id",
    "inserted",
    "removed",
]

POLICY_INPUT_COLS = [
    *EPISODE_KEY_COLS,
    "period_start",
    "period_end",
    "catheter_state",
    "periods_in_state",
    "observed_action",
    "removed_in_period",
]

POLICY_OUTPUT_COLS = [
    "catheter_episode_id",
    "decision_row_id",
    *POLICY_INPUT_COLS,
    "episode_day",
    "is_decision_row",
    "row_order_within_episode_day",
    "policy_catheter_state",
    "policy_periods_in",
    "policy_periods_out",
    "policy_removed_in_period",
    f"policy_{pec.REINSERTION_PERIOD_COL}",
    "policy_removal_day_extra_row_treated_as_out",
    "policy_action",
    "policy_dml_action",
    "policy_matches_observed",
    "policy_applicable",
    "policy_matches_observed_action_today",
]

POLICY_TYPE = "fixed_day_removal"


def add_stable_ids_and_decision_flag(df):
    # Standardise the state and action fields
    df["catheter_state"] = df["catheter_state"].astype(str).str.strip().str.lower()
    df["observed_action"] = df["observed_action"].astype(str).str.strip().str.lower()
    df["periods_in_state"] = pd.to_numeric(df["periods_in_state"], errors="coerce")
    df["removed_in_period"] = pd.to_numeric(df["removed_in_period"], errors="coerce")

    # Assign one stable number to each catheter episode
    key_frame = df[EPISODE_KEY_COLS].astype("string").fillna("<NA>")
    episode_codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(key_frame),
        sort=True,
    )
    df["catheter_episode_id"] = episode_codes + 1

    # Put rows in episode-time order
    df = df.sort_values(
        ["catheter_episode_id", "period_start", "period_end"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["decision_row_id"] = np.arange(1, len(df) + 1, dtype=np.int64)

    # Mark rows on which removal is possible
    df["is_decision_row"] = (
        df["observed_action"].isin(["keep", "remove"])
        & df["catheter_state"].eq("in")
    )
    return df


def policy_removal_day_matches_observed(df):
    """Flag every row of episodes with observed and policy removal on the same day."""
    groups = [df["catheter_episode_id"]]
    if "policy_name" in df:
        groups.insert(0, df["policy_name"])
    observed_day = df["episode_day"].where(df["removed_in_period"].eq(1)).groupby(groups).transform("min")
    policy_day = df["episode_day"].where(df["policy_action"].eq("remove")).groupby(groups).transform("min")
    return observed_day.notna() & policy_day.notna() & observed_day.eq(policy_day)


def apply_fixed_day_policy(base_df, policy_remove_day):
    # Copy the observed panel for one target policy
    df = base_df.copy()
    policy_name = f"remove_on_day_{policy_remove_day}"

    # Resolve the policy state and action on every row
    df["policy_name"] = policy_name
    df["policy_type"] = POLICY_TYPE
    df["policy_remove_day"] = policy_remove_day
    df = pec.add_fixed_day_target_policy_timeline(
        df,
        episode_id_col="catheter_episode_id",
        episode_day_col="episode_day",
    )

    # On observed IN rows after the target day, Policy-DML prescribes removal.
    df["policy_dml_action"] = np.where(
        df["is_decision_row"],
        np.where(df["episode_day"].ge(policy_remove_day), "remove", "keep"),
        "out",
    )

    # The policy action is the complete hypothetical instruction on every row.
    df["policy_applicable"] = (
        df["is_decision_row"] & df["policy_action"].isin(["keep", "remove"])
    )
    numeric_action = df["policy_action"].map({"keep": 0.0, "remove": 1.0})
    df["policy_matches_observed_action_today"] = (
        df["removed_in_period"].eq(numeric_action).astype(float)
        .where(df["policy_applicable"])
    )
    df["policy_matches_observed"] = policy_removal_day_matches_observed(df)

    return df


def read_policy_panel(path):
    """Read one saved policy panel."""
    path = Path(path)
    df = pd.read_csv(
        path, low_memory=False,
        dtype={"is_decision_row": "boolean", "policy_applicable": "boolean",
               "policy_matches_observed": "boolean"},
    )
    df = df.loc[:, POLICY_OUTPUT_COLS]
    for column in ("is_decision_row", "policy_applicable", "policy_matches_observed"):
        df[column] = df[column].astype(bool)
    return df


def discover_policy_panels(panel_dir):
    """Discover saved fixed-day policies from their existing filenames."""
    panel_dir = Path(panel_dir)
    policies = []
    for path in sorted(panel_dir.glob("*.csv")):
        match = re.fullmatch(r"remove_on_day_([1-9][0-9]*)", path.stem)
        if match is None:
            raise ValueError(f"Unsupported policy panel filename: {path.name}")
        policies.append({"name": path.stem, "kind": POLICY_TYPE,
                         "remove_day": int(match.group(1)), "path": path})
    if not policies:
        raise FileNotFoundError(f"No policy panels found in {panel_dir}")
    return sorted(policies, key=lambda policy: policy["remove_day"])


def read_policy_collection(panel_dir):
    """Load all existing policy panels, without a separate index file."""
    frames = []
    for policy in discover_policy_panels(panel_dir):
        panel = read_policy_panel(policy["path"])
        panel["policy_name"] = policy["name"]
        panel["policy_type"] = policy["kind"]
        panel["policy_remove_day"] = policy["remove_day"]
        frames.append(panel)
    return pd.concat(frames, ignore_index=True)


def save_policy_panels(long_df, panel_dir):
    """Export one file per policy; evaluators discover the files directly."""
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    for policy_name, panel in long_df.groupby("policy_name", sort=False):
        panel_path = panel_dir / f"{policy_name}.csv"
        panel.to_csv(panel_path, columns=POLICY_OUTPUT_COLS, index=False)


def print_console_summary(
    input_path,
    base_df,
    long_df,
    panel_dir,
):
    # Print the main row counts and output paths
    print()
    print("--- POLICY PANEL BUILD COMPLETE ---")
    print(f"Input panel: {input_path}")
    print(f"Panel interval rows retained: {len(base_df):,}")
    print(f"Catheter episodes: {base_df['catheter_episode_id'].nunique():,}")
    print(f"Decision rows: {int(base_df['is_decision_row'].sum()):,}")
    print(f"Candidate policies: {long_df['policy_name'].nunique():,}")
    print(f"Total rows across policy panels: {len(long_df):,}")
    print()
    print(f"Saved policy panels: {panel_dir}")


def run_panel():
    # Create the output directory
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # Load only the columns used to construct policies
    input_df = pd.read_csv(
        INPUT_PATH,
        usecols=POLICY_INPUT_COLS,
        low_memory=False,
    )

    # Add stable episode and decision identifiers
    base_df = add_stable_ids_and_decision_flag(input_df)
    base_df = add_episode_day(base_df)

    # Construct and export the policy panels.
    # Build one full panel copy per policy.
    policy_frames = [
        apply_fixed_day_policy(base_df, policy_remove_day)
        for policy_remove_day in POLICY_DAYS
    ]
    long_df = pd.concat(policy_frames, ignore_index=True)
    save_policy_panels(long_df, OUTDIR / "panels")

    print_console_summary(
        INPUT_PATH,
        base_df,
        long_df,
        OUTDIR / "panels",
    )


def main():
    global INPUT_PATH, OUTDIR
    for panel_name, input_path, nuisance_root in PANEL_RUNS:
        INPUT_PATH = input_path
        OUTDIR = run_config.POLICY_DIRS[panel_name]
        print(f"[PANEL] {panel_name}: {INPUT_PATH}", flush=True)
        run_panel()


if __name__ == "__main__":
    main()
