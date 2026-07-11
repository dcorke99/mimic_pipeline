#!/usr/bin/env python3
"""Build estimator-agnostic target-policy intervention panels."""


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
    / "modeling_panel.csv"
)
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "policy_interventions"
DEFAULT_POLICY_DAYS = [1, 2, 3, 4, 5]

DEFAULT_COMBINED_OUTPUT_NAME = "policy_intervention_panel_long.csv"
DEFAULT_WIDE_OUTPUT_NAME = "policy_action_matrix_wide.csv"
DEFAULT_QA_OUTPUT_NAME = "policy_intervention_panel_qa.csv"
DEFAULT_POLICY_LIBRARY_OUTPUT_NAME = "policy_library.csv"


# Input and output column definitions

EPISODE_KEY_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "inserted",
    "removed",
]

OPTIONAL_INPUT_COLS = [
    "reinsertion_time",
    "episode_index",
    "split",
    "fold_id",
    "crossfit_fold",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "cauti_in_period",
    "reinsertion_in_period",
    "death_in_period",
    "icu_end_in_period",
    "is_last_period_of_episode",
    "episode_end_reason",
]

DERIVED_BASE_COLS = [
    "catheter_episode_id",
    "decision_row_id",
    "is_decision_row",
    "episode_day_since_insertion",
]

POLICY_COLS = [
    "policy_name",
    "policy_type",
    "policy_remove_day",
    "policy_action",
    "policy_action_remove",
    "policy_applicable",
    "policy_reason",
    "row_order_within_episode_day",
    "policy_catheter_state",
    "policy_action_resolved",
    "policy_action_remove_resolved",
    "policy_periods_in",
    "policy_periods_out",
    "policy_removal_day_extra_row_treated_as_out",
    "policy_matches_observed_action_today",
    "policy_match_status",
]

# Generated estimator columns are excluded from the policy-intervention input.
EXPLICIT_ESTIMATOR_COLS = {
    "p_remove_obs",
    "p_keep_obs",
    "p_observed_action",
    "p_observed_action_clipped",
    "policy_support",
    "policy_support_clipped",
    "policy_weight_component",
    "matched_policy_today",
    "followed_policy_so_far",
    "episode_matches_policy",
    "ipw_component",
    "episode_ipw_weight",
    "mu_hat_0",
    "mu_hat_1",
    "mu_hat_keep",
    "mu_hat_remove",
    "aipw_score",
    "dr_score",
    "tmle_score",
    "clever_covariate",
}

ESTIMATOR_PREFIXES = (
    "p_",
    "mu_hat",
    "ipw_",
    "aipw_",
    "dr_",
    "tmle_",
    "ltmle_",
    "gformula_",
    "clever_",
    "pseudo_outcome",
)

POLICY_TYPE = "fixed_day_removal"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build estimator-agnostic catheter-removal policy-intervention "
            "panels in long and wide formats."
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
        "--wide-output-name",
        default=DEFAULT_WIDE_OUTPUT_NAME,
        help=f"Wide policy-action matrix filename. Default: {DEFAULT_WIDE_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--qa-output-name",
        default=DEFAULT_QA_OUTPUT_NAME,
        help=f"QA report filename. Default: {DEFAULT_QA_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--policy-library-output-name",
        default=DEFAULT_POLICY_LIBRARY_OUTPUT_NAME,
        help=(
            "Policy library output filename. "
            f"Default: {DEFAULT_POLICY_LIBRARY_OUTPUT_NAME}"
        ),
    )
    return parser.parse_args()


def resolve_output_path(outdir, name_or_path):
    """Resolve an output file path."""
    path = Path(name_or_path)
    return path if path.is_absolute() else outdir / path


def normalise_policy_days(policy_days):
    """Normalise policy days."""
    if not policy_days:
        raise ValueError("At least one policy day must be supplied.")

    invalid = [day for day in policy_days if day < 1]
    if invalid:
        raise ValueError(f"Policy days must be positive integers. Invalid: {invalid}")

    duplicates = sorted({day for day in policy_days if policy_days.count(day) > 1})
    if duplicates:
        raise ValueError(f"Policy days must be unique. Duplicates: {duplicates}")

    return list(policy_days)


def find_estimator_columns(columns):
    """Find estimator columns."""
    estimator_cols = []
    for col in columns:
        col_lower = col.lower()
        if col in EXPLICIT_ESTIMATOR_COLS:
            estimator_cols.append(col)
        elif col_lower.startswith(ESTIMATOR_PREFIXES):
            estimator_cols.append(col)
    return estimator_cols


def drop_generated_or_estimator_columns(df):
    """Drop generated or estimator columns."""
    generated_cols = [
        col for col in [*DERIVED_BASE_COLS, *POLICY_COLS] if col in df.columns
    ]
    # Find estimator columns.
    estimator_cols = find_estimator_columns(list(df.columns))
    drop_cols = sorted(set(generated_cols + estimator_cols))

    if drop_cols:
        print(
            "Excluding pre-existing generated/estimator columns from input: "
            f"{drop_cols}"
        )
        df = df.drop(columns=drop_cols)

    return df


def load_patient_day_panel(input_path):
    """Load patient day panel."""
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = df.columns.str.strip()
    # Drop generated or estimator columns.
    df = drop_generated_or_estimator_columns(df)
    return df


def normalise_text_column(df, column):
    """Normalise text column."""
    df[column] = df[column].astype(str).str.strip().str.lower()


def add_stable_ids_and_decision_flag(df):
    """Add stable IDs and decision flag."""
    df = df.copy()

    # Normalise text column.
    normalise_text_column(df, "catheter_state")
    # Normalise text column.
    normalise_text_column(df, "observed_action")
    df["periods_in_state"] = pd.to_numeric(df["periods_in_state"], errors="coerce")
    df["action_remove"] = pd.to_numeric(df["action_remove"], errors="coerce")

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
    """Build name for day."""
    return f"remove_on_day_{policy_remove_day}"


def policy_library_row(policy_remove_day):
    """Build library row."""
    # Build name for day.
    policy_name = policy_name_for_day(policy_remove_day)
    return {
        "policy_name": policy_name,
        "policy_type": POLICY_TYPE,
        "policy_remove_day": policy_remove_day,
        "description": (
            "Proof-of-concept fixed-day catheter-removal target policy: keep "
            f"the catheter before day {policy_remove_day} and assign removal "
            f"on day {policy_remove_day}."
        ),
        "rule_summary": (
            f"Resolved target-policy timeline uses episode day {policy_remove_day}: "
            "before removal day is in/keep; first row on removal day is "
            "in/remove; later rows on the same removal day and later days are "
            "out/out."
        ),
        "is_proof_of_concept": True,
        "clinical_rationale": (
            "Candidate deterministic timing rule for catheter-removal policy "
            "evaluation. It is a proof-of-concept policy, not a claim of "
            "clinical optimality; the implementation is suitable for the "
            "final estimator-agnostic policy-intervention framework."
        ),
    }


def build_policy_library(policy_days):
    """Build policy library."""
    # Build library row.
    return pd.DataFrame([policy_library_row(day) for day in policy_days])


def add_policy_episode_day(df):
    """Add policy episode day."""
    df = df.copy()
    inserted = pd.to_datetime(df["inserted"], errors="coerce")
    period_start = pd.to_datetime(df["period_start"], errors="coerce")

    elapsed_days = (period_start - inserted).dt.total_seconds() / 86400.0
    df["episode_day_since_insertion"] = np.floor(elapsed_days).astype(int) + 1
    df.loc[df["episode_day_since_insertion"].lt(1), "episode_day_since_insertion"] = 1
    return df


def apply_fixed_day_policy(base_df, policy_remove_day):
    """Apply fixed day policy."""
    # Add policy episode day.
    df = add_policy_episode_day(base_df)
    # Build name for day.
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
    df["policy_reason"] = "non_decision_row"

    decision_rows = df["is_decision_row"]
    keep_rows = decision_rows & df["policy_action_resolved"].eq("keep")
    remove_rows = decision_rows & df["policy_action_resolved"].eq("remove")
    already_removed_rows = df["policy_action_resolved"].eq("out")
    missing_policy_day = decision_rows & df["episode_day_since_insertion"].isna()

    df.loc[keep_rows, "policy_action"] = "keep"
    df.loc[keep_rows, "policy_action_remove"] = 0.0
    df.loc[keep_rows, "policy_applicable"] = True
    df.loc[keep_rows, "policy_reason"] = "before_policy_removal_day"

    df.loc[remove_rows, "policy_action"] = "remove"
    df.loc[remove_rows, "policy_action_remove"] = 1.0
    df.loc[remove_rows, "policy_applicable"] = True
    df.loc[remove_rows, "policy_reason"] = "policy_removal_day"

    df.loc[already_removed_rows, "policy_action"] = "already_removed_under_policy"
    df.loc[already_removed_rows, "policy_reason"] = "already_removed_under_policy"

    df.loc[missing_policy_day, "policy_reason"] = "missing_policy_episode_day"

    df["policy_matches_observed_action_today"] = np.nan
    applicable_rows = df["policy_applicable"]
    df.loc[applicable_rows, "policy_matches_observed_action_today"] = (
        df.loc[applicable_rows, "action_remove"]
        .eq(df.loc[applicable_rows, "policy_action_remove"])
        .astype(float)
    )

    df["policy_match_status"] = df["policy_reason"]
    df.loc[
        applicable_rows & df["policy_matches_observed_action_today"].eq(1),
        "policy_match_status",
    ] = "matches_observed_action"
    df.loc[
        applicable_rows & df["policy_matches_observed_action_today"].eq(0),
        "policy_match_status",
    ] = "disagrees_with_observed_action"

    pec.validate_resolved_target_policy_timeline(
        df,
        episode_id_col="catheter_episode_id",
        context=f"policy {policy_name}",
    )
    return df


def order_long_columns(df, original_columns):
    """Order long columns."""
    base_order = [
        "catheter_episode_id",
        "decision_row_id",
        "episode_day_since_insertion",
        *[col for col in original_columns if col in df.columns],
        "is_decision_row",
        *POLICY_COLS,
    ]
    ordered = []
    for col in base_order:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    remaining = [col for col in df.columns if col not in ordered]
    return df[[*ordered, *remaining]].copy()


def build_long_policy_panel(
    base_df,
    policy_days,
    original_columns,
):
    """Build long policy panel."""
    # Apply fixed day policy.
    policy_frames = [
        apply_fixed_day_policy(base_df, policy_remove_day)
        for policy_remove_day in policy_days
    ]
    long_df = pd.concat(policy_frames, ignore_index=True)
    # Order long columns.
    return order_long_columns(long_df, original_columns)


def build_wide_policy_action_matrix(
    long_df,
    base_df,
    policy_library,
    original_columns,
):
    """Build wide policy action matrix."""
    id_columns = [
        "decision_row_id",
        "catheter_episode_id",
        "episode_day_since_insertion",
        *[col for col in original_columns if col in base_df.columns],
        "is_decision_row",
    ]
    id_columns = list(dict.fromkeys(id_columns))
    wide_df = base_df[id_columns].copy()

    matrix_fields = [
        "policy_action_remove",
        "policy_applicable",
        "policy_action",
        "policy_reason",
        "row_order_within_episode_day",
        "policy_catheter_state",
        "policy_action_resolved",
        "policy_action_remove_resolved",
        "policy_periods_in",
        "policy_periods_out",
        "policy_removal_day_extra_row_treated_as_out",
        "policy_matches_observed_action_today",
        "policy_match_status",
    ]
    policy_names = policy_library["policy_name"].tolist()

    for field in matrix_fields:
        pivot = long_df.pivot(
            index="decision_row_id",
            columns="policy_name",
            values=field,
        )
        pivot = pivot.reindex(columns=policy_names)
        pivot.columns = [f"{field}__{policy_name}" for policy_name in pivot.columns]
        pivot = pivot.reset_index()
        wide_df = wide_df.merge(pivot, on="decision_row_id", how="left")

    return wide_df


def qa_row(policy_df):
    """Build one QA row."""
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
            policy_df["policy_reason"].eq("non_decision_row").sum()
        ),
        "n_already_removed_under_policy_rows": int(
            policy_df["policy_reason"].eq("already_removed_under_policy").sum()
        ),
        "n_policy_matches_observed_action_today": n_matches,
        "n_policy_disagrees_with_observed_action_today": n_disagrees,
        "pct_policy_matches_observed_action_today": (
            n_matches / n_applicable if n_applicable else np.nan
        ),
    }


def build_qa_report(long_df, policy_days):
    """Build QA report."""
    rows = []
    # Build name for day.
    for policy_remove_day in policy_days:
        # Build name for day.
        policy_name = policy_name_for_day(policy_remove_day)
        policy_df = long_df.loc[long_df["policy_name"].eq(policy_name)]
        # Build one QA row.
        rows.append(qa_row(policy_df))
    return pd.DataFrame(rows)


def save_df(df, path):
    """Save a data frame as CSV."""
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def print_console_summary(
    input_path,
    base_df,
    long_df,
    qa_df,
    long_output_path,
    wide_output_path,
    qa_output_path,
    policy_library_output_path,
):
    """Print a concise run summary."""
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
    print(f"Saved wide policy-action matrix: {wide_output_path}")
    print(f"Saved policy QA report: {qa_output_path}")
    print(f"Saved policy library: {policy_library_output_path}")


def main():
    """Run the script workflow."""
    # Parse command-line arguments.
    args = parse_args()
    # Normalise policy days.
    policy_days = normalise_policy_days(args.policy_days)
    args.outdir.mkdir(exist_ok=True, parents=True)

    # Resolve an output file path.
    long_output_path = resolve_output_path(args.outdir, args.combined_output_name)
    # Resolve an output file path.
    wide_output_path = resolve_output_path(args.outdir, args.wide_output_name)
    # Resolve an output file path.
    qa_output_path = resolve_output_path(args.outdir, args.qa_output_name)
    # Resolve an output file path.
    policy_library_output_path = resolve_output_path(
        args.outdir,
        args.policy_library_output_name,
    )

    # Load patient day panel.
    input_df = load_patient_day_panel(args.input)
    original_columns = list(input_df.columns)
    # Add stable IDs and decision flag.
    base_df = add_stable_ids_and_decision_flag(input_df)
    # Add policy episode day.
    base_df = add_policy_episode_day(base_df)

    # Build policy library.
    policy_library = build_policy_library(policy_days)
    # Build long policy panel.
    long_df = build_long_policy_panel(base_df, policy_days, original_columns)
    # Build wide policy action matrix.
    wide_df = build_wide_policy_action_matrix(
        long_df,
        base_df,
        policy_library,
        original_columns,
    )
    # Build QA report.
    qa_df = build_qa_report(long_df, policy_days)

    # Save a data frame as CSV.
    save_df(long_df, long_output_path)
    # Save a data frame as CSV.
    save_df(wide_df, wide_output_path)
    # Save a data frame as CSV.
    save_df(qa_df, qa_output_path)
    # Save a data frame as CSV.
    save_df(policy_library, policy_library_output_path)

    # Print a concise run summary.
    print_console_summary(
        args.input,
        base_df,
        long_df,
        qa_df,
        long_output_path,
        wide_output_path,
        qa_output_path,
        policy_library_output_path,
    )


# Run the script workflow.
if __name__ == "__main__":
    # Run the script workflow.
    main()
