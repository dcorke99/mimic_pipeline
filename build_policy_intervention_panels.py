#!/usr/bin/env python3
# Build estimator-agnostic target-policy intervention panels.


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import policy_eval_common as pec


# Defaults

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "data"
    / "modelling_panel.csv"
)
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "policy_interventions"
DEFAULT_POLICY_DAYS = [1, 2, 3, 4, 5]

DEFAULT_COMBINED_OUTPUT_NAME = "policy_intervention_panel_long.csv"
DEFAULT_QA_OUTPUT_NAME = "policy_intervention_panel_qa.csv"


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

POLICY_COLS = [
    "policy_name",
    "policy_type",
    "policy_remove_day",
    "policy_action",
    "policy_action_remove",
    "policy_applicable",
    "row_order_within_episode_day",
    "policy_catheter_state",
    "policy_action_resolved",
    "policy_action_remove_resolved",
    "policy_periods_in",
    "policy_periods_out",
    "policy_removal_day_extra_row_treated_as_out",
    "policy_matches_observed_action_today",
]

POLICY_TYPE = "fixed_day_removal"


def parse_args():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description=(
            "Build estimator-agnostic catheter-removal policy-intervention "
            "panel in long format."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input clean patient-day decision panel. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Output directory. Default: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--policy-days",
        nargs="+",
        type=int,
        default=DEFAULT_POLICY_DAYS,
        help="Fixed catheter-removal days to encode. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--combined-output-name",
        default=DEFAULT_COMBINED_OUTPUT_NAME,
        help=f"Long-format output filename. Default: {DEFAULT_COMBINED_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--qa-output-name",
        default=DEFAULT_QA_OUTPUT_NAME,
        help=f"QA report filename. Default: {DEFAULT_QA_OUTPUT_NAME}",
    )
    return parser.parse_args()


def resolve_output_path(outdir, name_or_path):
    # Resolve an output file path.
    path = Path(name_or_path)
    return path if path.is_absolute() else outdir / path


def normalise_policy_days(policy_days):
    # Normalise policy days.
    if not policy_days:
        raise ValueError("At least one policy day must be supplied.")

    invalid = [day for day in policy_days if day < 1]
    if invalid:
        raise ValueError(f"Policy days must be positive integers. Invalid: {invalid}")

    duplicates = sorted({day for day in policy_days if policy_days.count(day) > 1})
    if duplicates:
        raise ValueError(f"Policy days must be unique. Duplicates: {duplicates}")

    return list(policy_days)


def load_patient_day_panel(input_path):
    return pd.read_csv(input_path, usecols=POLICY_INPUT_COLS, low_memory=False)


def add_stable_ids_and_decision_flag(df):
    df["catheter_state"] = df["catheter_state"].astype(str).str.strip().str.lower()
    df["observed_action"] = df["observed_action"].astype(str).str.strip().str.lower()
    df["periods_in_state"] = pd.to_numeric(df["periods_in_state"], errors="coerce")
    df["removed_in_period"] = pd.to_numeric(df["removed_in_period"], errors="coerce")

    unknown_states = sorted(set(df["catheter_state"].dropna()) - {"in", "out"})
    if unknown_states:
        raise ValueError(f"Unexpected catheter_state values: {unknown_states}")

    unknown_actions = sorted(
        set(df["observed_action"].dropna()) - {"keep", "remove", "out"}
    )
    if unknown_actions:
        raise ValueError(f"Unexpected observed_action values: {unknown_actions}")

    key_frame = df[EPISODE_KEY_COLS].astype("string").fillna("<NA>")
    episode_codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(key_frame),
        sort=True,
    )
    df["catheter_episode_id"] = episode_codes + 1

    df = df.sort_values(
        ["catheter_episode_id", "period_start", "period_end"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["decision_row_id"] = np.arange(1, len(df) + 1, dtype=np.int64)

    df["is_decision_row"] = (
        df["observed_action"].isin(["keep", "remove"])
        & df["catheter_state"].eq("in")
    )
    return df


def policy_name_for_day(policy_remove_day):
    return f"remove_on_day_{policy_remove_day}"


def add_policy_episode_day(df):
    # Add policy episode day.
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")

    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def apply_fixed_day_policy(base_df, policy_remove_day):
    df = base_df.copy()
    policy_name = policy_name_for_day(policy_remove_day)

    df["policy_name"] = policy_name
    df["policy_type"] = POLICY_TYPE
    df["policy_remove_day"] = policy_remove_day
    df = pec.add_fixed_day_target_policy_timeline(
        df,
        episode_id_col="catheter_episode_id",
        episode_day_col="episode_day_since_insertion",
    )

    df["policy_action"] = "not_applicable"
    df["policy_action_remove"] = np.nan
    df["policy_applicable"] = False

    decision_rows = df["is_decision_row"]
    keep_rows = decision_rows & df["policy_action_resolved"].eq("keep")
    remove_rows = decision_rows & df["policy_action_resolved"].eq("remove")
    already_removed_rows = df["policy_action_resolved"].eq("out")
    df.loc[keep_rows, "policy_action"] = "keep"
    df.loc[keep_rows, "policy_action_remove"] = 0.0
    df.loc[keep_rows, "policy_applicable"] = True

    df.loc[remove_rows, "policy_action"] = "remove"
    df.loc[remove_rows, "policy_action_remove"] = 1.0
    df.loc[remove_rows, "policy_applicable"] = True

    df.loc[already_removed_rows, "policy_action"] = "already_removed_under_policy"

    df["policy_matches_observed_action_today"] = np.nan
    applicable_rows = df["policy_applicable"]
    df.loc[applicable_rows, "policy_matches_observed_action_today"] = (
        df.loc[applicable_rows, "removed_in_period"]
        .eq(df.loc[applicable_rows, "policy_action_remove"])
        .astype(float)
    )

    return df


def order_long_columns(df):
    base_order = [
        "catheter_episode_id",
        "decision_row_id",
        "episode_day_since_insertion",
        *POLICY_INPUT_COLS,
        "is_decision_row",
        *POLICY_COLS,
    ]
    ordered = []
    for col in base_order:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]]


def build_long_policy_panel(
    base_df,
    policy_days,
):
    policy_frames = [
        apply_fixed_day_policy(base_df, policy_remove_day)
        for policy_remove_day in policy_days
    ]
    long_df = pd.concat(policy_frames, ignore_index=True)
    pec.validate_resolved_target_policy_timeline(
        long_df,
        episode_id_col="catheter_episode_id",
        context="policy intervention panel",
    )
    return order_long_columns(long_df)


def qa_row(policy_df):
    # Build one QA row.
    applicable = policy_df["policy_applicable"]
    matches = policy_df["policy_matches_observed_action_today"]
    n_applicable = int(applicable.sum())
    n_matches = int(matches.eq(1).sum())
    n_disagrees = int(matches.eq(0).sum())
    timeline_diagnostics = pec.resolved_timeline_diagnostics(
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
        "n_policy_keep_assignments": int(policy_df["policy_action"].eq("keep").sum()),
        "n_policy_remove_assignments": int(
            policy_df["policy_action"].eq("remove").sum()
        ),
        "n_resolved_policy_remove_rows": int(
            policy_df["policy_action_resolved"].eq("remove").sum()
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
                & ~policy_df["policy_action_resolved"].eq("out")
            ).sum()
        ),
        "n_already_removed_under_policy_rows": int(
            policy_df["policy_action_resolved"].eq("out").sum()
        ),
        "n_policy_matches_observed_action_today": n_matches,
        "n_policy_disagrees_with_observed_action_today": n_disagrees,
        "pct_policy_matches_observed_action_today": (
            n_matches / n_applicable if n_applicable else np.nan
        ),
    }


def build_qa_report(long_df, policy_days):
    rows = []
    for policy_remove_day in policy_days:
        policy_name = policy_name_for_day(policy_remove_day)
        policy_df = long_df.loc[long_df["policy_name"].eq(policy_name)]
        rows.append(qa_row(policy_df))
    return pd.DataFrame(rows)


def save_df(df, path):
    # Save a data frame as CSV.
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def print_console_summary(
    input_path,
    base_df,
    long_df,
    qa_df,
    long_output_path,
    qa_output_path,
):
    # Print a concise run summary.
    print()
    print("--- POLICY-INTERVENTION PANEL BUILD COMPLETE ---")
    print(f"Input panel: {input_path}")
    print(f"Patient-day rows retained: {len(base_df):,}")
    print(f"Catheter episodes: {base_df['catheter_episode_id'].nunique():,}")
    print(f"Decision rows: {int(base_df['is_decision_row'].sum()):,}")
    print(f"Candidate policies: {qa_df['policy_name'].nunique():,}")
    print(f"Long panel rows: {len(long_df):,}")
    print()
    print("Policy QA:")
    for row in qa_df.itertuples(index=False):
        pct = row.pct_policy_matches_observed_action_today
        pct_text = "NA" if pd.isna(pct) else f"{100 * pct:.1f}%"
        print(
            f"  {row.policy_name}: applicable={row.n_policy_applicable_rows:,}, "
            f"keep={row.n_policy_keep_assignments:,}, "
            f"remove={row.n_policy_remove_assignments:,}, "
            f"today-match={pct_text}"
        )
    print()
    print(f"Saved long policy-intervention panel: {long_output_path}")
    print(f"Saved policy QA report: {qa_output_path}")


def main():
    # Run the script workflow.
    # Parse command-line arguments.
    args = parse_args()
    # Normalise policy days.
    policy_days = normalise_policy_days(args.policy_days)
    args.outdir.mkdir(exist_ok=True, parents=True)

    long_output_path = resolve_output_path(args.outdir, args.combined_output_name)
    qa_output_path = resolve_output_path(args.outdir, args.qa_output_name)

    input_df = load_patient_day_panel(args.input)
    base_df = add_stable_ids_and_decision_flag(input_df)
    base_df = add_policy_episode_day(base_df)

    long_df = build_long_policy_panel(base_df, policy_days)
    qa_df = build_qa_report(long_df, policy_days)

    save_df(long_df, long_output_path)
    pec.save_report_df(qa_df, qa_output_path)

    print_console_summary(
        args.input,
        base_df,
        long_df,
        qa_df,
        long_output_path,
        qa_output_path,
    )


# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
