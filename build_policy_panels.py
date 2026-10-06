#!/usr/bin/env python3
# Build estimator-agnostic target-counterfactual policy panels


from pathlib import Path

import numpy as np
import pandas as pd

import policy_eval_common as pec
from policy_eval_common import add_episode_day


# Configuration

REPO_ROOT = Path(__file__).resolve().parent

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

# Initial paths for direct single-panel calls use the first selected panel.
_, INPUT_PATH, _NUISANCE_ROOT = PANEL_RUNS[0]
OUTDIR = _NUISANCE_ROOT.parent / "counterfactual_policies"

POLICY_DAYS = [1, 2, 3, 4, 5]
POLICY_MANIFEST_PATH = OUTDIR / "policy_panels.csv"
CHECK_OUTPUT_PATH = OUTDIR / "policy_panel_checks.csv"


# Input and output column definitions

EPISODE_KEY_COLS = [
    "subject_id",
    "hadm_id",
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
    "policy_removal_day_extra_row_treated_as_out",
    "policy_action",
    "policy_matches_observed",
    "policy_applicable",
    "policy_matches_observed_action_today",
]

POLICY_TYPE = "fixed_day_removal"
POLICY_MANIFEST_COLS = ["policy_name", "policy_type", "policy_remove_day", "panel_file"]


def add_stable_ids_and_decision_flag(df):
    # Standardise the state and action fields
    df["catheter_state"] = df["catheter_state"].astype(str).str.strip().str.lower()
    df["observed_action"] = df["observed_action"].astype(str).str.strip().str.lower()
    df["periods_in_state"] = pd.to_numeric(df["periods_in_state"], errors="coerce")
    df["removed_in_period"] = pd.to_numeric(df["removed_in_period"], errors="coerce")

    # Reject values outside the panel schema
    unknown_states = sorted(set(df["catheter_state"].dropna()) - {"in", "out"})
    if unknown_states:
        raise ValueError(f"Unexpected catheter_state values: {unknown_states}")

    unknown_actions = sorted(
        set(df["observed_action"].dropna()) - {"keep", "remove", "out"}
    )
    if unknown_actions:
        raise ValueError(f"Unexpected observed_action values: {unknown_actions}")

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


def build_long_policy_panel(
    base_df,
    policy_days,
):
    # Build one full panel copy per policy
    policy_frames = [
        apply_fixed_day_policy(base_df, policy_remove_day)
        for policy_remove_day in policy_days
    ]
    long_df = pd.concat(policy_frames, ignore_index=True)

    # Confirm that each policy has a coherent timeline
    pec.check_policy_timeline(
        long_df,
        episode_id_col="catheter_episode_id",
        context="counterfactual policy panel",
    )
    return long_df


def read_policy_panel(path):
    """Read one policy file and check its saved row fields."""
    path = Path(path)
    df = pd.read_csv(
        path, low_memory=False,
        dtype={"is_decision_row": "boolean", "policy_applicable": "boolean",
               "policy_matches_observed": "boolean"},
    )
    missing = sorted(set(POLICY_OUTPUT_COLS) - set(df.columns))
    if missing:
        raise ValueError(
            f"Policy panel is missing explicit policy fields: {missing}. "
            "Rebuild it with build_policy_panels.py."
        )
    df = df.loc[:, POLICY_OUTPUT_COLS]
    if df.empty:
        raise ValueError("Policy panel must contain time rows")
    for column in ("is_decision_row", "policy_applicable", "policy_matches_observed"):
        if df[column].isna().any():
            raise ValueError(f"Policy panel has missing {column} values")
        df[column] = df[column].astype(bool)
    # The observed-decision fields must agree with the saved target timeline.
    decision = df["catheter_state"].eq("in") & df["observed_action"].isin(["keep", "remove"])
    action = df["policy_action"]
    applicable = decision & action.isin(["keep", "remove"])
    numeric_action = action.map({"keep": 0.0, "remove": 1.0})
    expected_match = df["removed_in_period"].eq(numeric_action).astype(float).where(applicable)
    invalid = (
        df["is_decision_row"].ne(decision)
        | df["policy_matches_observed"].ne(policy_removal_day_matches_observed(df))
        | df["policy_applicable"].ne(applicable)
        | ~df["policy_matches_observed_action_today"].fillna(-1).eq(expected_match.fillna(-1))
        | df["policy_catheter_state"].ne(np.where(action.eq("out"), "out", "in"))
    )
    if invalid.any():
        raise ValueError("Saved policy actions, states, or observed-action comparisons are inconsistent")
    return df


def read_policy_collection(path):
    """Load only the separate panels explicitly listed in the policy index."""
    path = Path(path)
    manifest = pd.read_csv(path)
    missing = sorted(set(POLICY_MANIFEST_COLS) - set(manifest.columns))
    if missing or manifest.empty:
        raise ValueError(f"Policy index is empty or missing columns: {missing}")
    if manifest[POLICY_MANIFEST_COLS].isna().any().any():
        raise ValueError("Policy index has missing names, types, parameters, or file paths")
    if manifest.policy_name.duplicated().any() or manifest.panel_file.duplicated().any():
        raise ValueError("Policy index must list each policy and panel file only once")
    frames = []
    for policy in manifest.itertuples(index=False):
        panel_path = (path.parent / policy.panel_file).resolve()
        if not panel_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Policy panel files must be inside the policy index directory")
        panel = read_policy_panel(panel_path)
        if panel_path.stem != policy.policy_name:
            raise ValueError(f"Panel metadata does not match index entry {policy.policy_name}")
        # Metadata belongs to the policy index, not every exported time row.
        panel["policy_name"] = policy.policy_name
        panel["policy_type"] = policy.policy_type
        panel["policy_remove_day"] = policy.policy_remove_day
        frames.append(panel)
    collection = pd.concat(frames, ignore_index=True)
    pec.check_policy_timeline(collection, context=str(path))
    return collection


def save_policy_panels(long_df, manifest_path):
    """Export one file per policy and publish the index after validation."""
    manifest_path = Path(manifest_path)
    panel_dir = manifest_path.parent / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for policy_name, panel in long_df.groupby("policy_name", sort=False):
        # Policy names also identify files; reject path separators and unsafe names.
        if not policy_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in policy_name):
            raise ValueError(f"Policy name cannot be used as a filename: {policy_name}")
        panel_path = panel_dir / f"{policy_name}.csv"
        panel.to_csv(panel_path, columns=POLICY_OUTPUT_COLS, index=False)
        entries.append({
            "policy_name": policy_name,
            "policy_type": panel.policy_type.iloc[0],
            "policy_remove_day": panel.policy_remove_day.iloc[0],
            "panel_file": panel_path.relative_to(manifest_path.parent).as_posix(),
        })
    pd.DataFrame(entries, columns=POLICY_MANIFEST_COLS).to_csv(manifest_path, index=False)


def check_row(policy_df):
    # Count decisions and observed-policy agreement
    applicable = policy_df["policy_applicable"]
    matches = policy_df["policy_matches_observed_action_today"]
    n_applicable = int(applicable.sum())
    n_matches = int(matches.eq(1).sum())
    n_disagrees = int(matches.eq(0).sum())
    timeline_diagnostics = pec.policy_timeline_checks(
        policy_df,
        episode_id_col="catheter_episode_id",
    ).iloc[0].to_dict()

    return {
        "policy_name": policy_df["policy_name"].iloc[0],
        "policy_type": policy_df["policy_type"].iloc[0],
        "policy_remove_day": int(policy_df["policy_remove_day"].iloc[0]),
        "n_rows": int(len(policy_df)),
        "n_decision_rows": int(policy_df["is_decision_row"].sum()),
        "n_policy_applicable_rows": n_applicable,
        "n_applicable_policy_keep_rows": int((applicable & policy_df["policy_action"].eq("keep")).sum()),
        "n_applicable_policy_remove_rows": int(
            (applicable & policy_df["policy_action"].eq("remove")).sum()
        ),
        "n_policy_remove_rows": int(
            policy_df["policy_action"].eq("remove").sum()
        ),
        "n_episodes_with_more_than_one_remove_row": int(
            timeline_diagnostics["n_episodes_with_more_than_one_remove_row"]
        ),
        "n_policy_removal_day_extra_rows_treated_as_out": int(
            timeline_diagnostics["n_policy_removal_day_extra_rows_treated_as_out"]
        ),
        "n_policy_remove_row_shortfall_vs_reached_episodes": int(
            timeline_diagnostics["n_policy_remove_row_shortfall_vs_reached_episodes"]
        ),
        "n_not_applicable_rows": int(
            (
                ~policy_df["policy_applicable"]
                & ~policy_df["policy_action"].eq("out")
            ).sum()
        ),
        "n_already_removed_under_policy_rows": int(
            policy_df["policy_action"].eq("out").sum()
        ),
        "n_policy_matches_observed_action_today": n_matches,
        "n_policy_disagrees_with_observed_action_today": n_disagrees,
        "proportion_policy_matches_observed_action_today": (
            n_matches / n_applicable if n_applicable else np.nan
        ),
    }


def build_check_report(long_df, policy_days):
    # Summarise each target policy
    rows = []
    for policy_remove_day in policy_days:
        policy_name = f"remove_on_day_{policy_remove_day}"
        policy_df = long_df.loc[long_df["policy_name"].eq(policy_name)]
        rows.append(check_row(policy_df))
    return pd.DataFrame(rows)


def print_console_summary(
    input_path,
    base_df,
    long_df,
    check_df,
    manifest_path,
    check_output_path,
):
    # Print the main row counts and output paths
    print()
    print("--- POLICY PANEL BUILD COMPLETE ---")
    print(f"Input panel: {input_path}")
    print(f"Panel interval rows retained: {len(base_df):,}")
    print(f"Catheter episodes: {base_df['catheter_episode_id'].nunique():,}")
    print(f"Decision rows: {int(base_df['is_decision_row'].sum()):,}")
    print(f"Candidate policies: {check_df['policy_name'].nunique():,}")
    print(f"Total rows across policy panels: {len(long_df):,}")
    print()
    print("Policy checks:")
    for row in check_df.itertuples(index=False):
        pct = row.proportion_policy_matches_observed_action_today
        pct_text = "NA" if pd.isna(pct) else f"{100 * pct:.1f}%"
        print(
            f"  {row.policy_name}: applicable={row.n_policy_applicable_rows:,}, "
            f"keep={row.n_applicable_policy_keep_rows:,}, "
            f"remove={row.n_applicable_policy_remove_rows:,}, "
            f"today-match={pct_text}"
        )
    print()
    print(f"Saved policy panel index: {manifest_path}")
    print(f"Saved policy check report: {check_output_path}")


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

    # Construct the policy panel and its check report
    long_df = build_long_policy_panel(base_df, POLICY_DAYS)
    # Export one explicit definition per policy, indexed for all evaluators.
    save_policy_panels(long_df, POLICY_MANIFEST_PATH)

    # Audit the saved fields rather than a separate reconstructed policy.
    saved_df = read_policy_collection(POLICY_MANIFEST_PATH)
    check_df = build_check_report(saved_df, POLICY_DAYS)

    # Save report data rounded to three decimal places
    pec.save_report_df(check_df, CHECK_OUTPUT_PATH)

    print_console_summary(
        INPUT_PATH,
        base_df,
        long_df,
        check_df,
        POLICY_MANIFEST_PATH,
        CHECK_OUTPUT_PATH,
    )


def main():
    global INPUT_PATH, OUTDIR, POLICY_MANIFEST_PATH, CHECK_OUTPUT_PATH
    for panel_name, input_path, nuisance_root in PANEL_RUNS:
        INPUT_PATH = input_path
        OUTDIR = nuisance_root.parent / "counterfactual_policies"
        POLICY_MANIFEST_PATH = OUTDIR / "policy_panels.csv"
        CHECK_OUTPUT_PATH = OUTDIR / "policy_panel_checks.csv"
        print(f"[PANEL] {panel_name}: {INPUT_PATH}", flush=True)
        run_panel()


if __name__ == "__main__":
    main()
