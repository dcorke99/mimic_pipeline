#!/usr/bin/env python3
"""
Audit baseline consistency across policy-evaluation outputs.

This script does not estimate any new causal quantities. It reads the existing
policy-intervention, g-formula, AIPW and IPW artefacts and checks that their
structural assumptions agree.

On success it prints "BASELINE QA PASSED" and writes a compact audit report.
On failure it writes the same report and exits non-zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_POLICY_QA = REPO_ROOT / "artifacts" / "policy_interventions" / "policy_intervention_panel_qa.csv"
DEFAULT_GFORMULA_SUMMARY = (
    REPO_ROOT / "artifacts" / "policy_eval" / "gformula" / "gformula_policy_outcomes_summary.csv"
)
DEFAULT_GFORMULA_DIAGNOSTICS = (
    REPO_ROOT / "artifacts" / "policy_eval" / "gformula" / "gformula_diagnostics.csv"
)
DEFAULT_AIPW_SUMMARY = (
    REPO_ROOT / "artifacts" / "policy_eval" / "aipw" / "aipw_policy_outcomes_summary.csv"
)
DEFAULT_AIPW_EPISODES = (
    REPO_ROOT / "artifacts" / "policy_eval" / "aipw" / "aipw_policy_episode_scores.csv"
)
DEFAULT_AIPW_WEIGHT_DIAGNOSTICS = (
    REPO_ROOT / "artifacts" / "policy_eval" / "aipw" / "aipw_weight_diagnostics.csv"
)
DEFAULT_IPW_SUMMARY = (
    REPO_ROOT / "artifacts" / "policy_eval" / "ipw" / "ipw_policy_outcomes_summary.csv"
)
DEFAULT_IPW_WEIGHT_DIAGNOSTICS = (
    REPO_ROOT / "artifacts" / "policy_eval" / "ipw" / "ipw_weight_diagnostics.csv"
)
DEFAULT_IPW_SUPPORT_DIAGNOSTICS = (
    REPO_ROOT / "artifacts" / "policy_eval" / "ipw" / "ipw_policy_support_diagnostics.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "policy_eval" / "baseline_policy_evaluation_audit.csv"

CURRENT_PRACTICE_LABEL = "current_practice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit structural consistency across existing policy-evaluation outputs."
    )
    parser.add_argument("--policy-qa", type=Path, default=DEFAULT_POLICY_QA)
    parser.add_argument("--gformula-summary", type=Path, default=DEFAULT_GFORMULA_SUMMARY)
    parser.add_argument("--gformula-diagnostics", type=Path, default=DEFAULT_GFORMULA_DIAGNOSTICS)
    parser.add_argument("--aipw-summary", type=Path, default=DEFAULT_AIPW_SUMMARY)
    parser.add_argument("--aipw-episodes", type=Path, default=DEFAULT_AIPW_EPISODES)
    parser.add_argument("--aipw-weight-diagnostics", type=Path, default=DEFAULT_AIPW_WEIGHT_DIAGNOSTICS)
    parser.add_argument("--ipw-summary", type=Path, default=DEFAULT_IPW_SUMMARY)
    parser.add_argument("--ipw-weight-diagnostics", type=Path, default=DEFAULT_IPW_WEIGHT_DIAGNOSTICS)
    parser.add_argument("--ipw-support-diagnostics", type=Path, default=DEFAULT_IPW_SUPPORT_DIAGNOSTICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Absolute tolerance for g-formula vs AIPW plug-in equality checks.",
    )
    return parser.parse_args()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def policy_day_key(value) -> str:
    if pd.isna(value):
        return "<NA>"
    numeric = float(value)
    if np.isclose(numeric, round(numeric), atol=1e-9):
        return str(int(round(numeric)))
    return f"{numeric:.12g}"


def add_policy_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__policy_day_key"] = out["policy_remove_day"].map(policy_day_key)
    out["__policy_key"] = out["policy_name"].astype(str) + "|" + out["__policy_day_key"].astype(str)
    return out


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes", "y"])


def is_true(value) -> bool:
    return bool_series(pd.Series([value])).iloc[0]


def add_result(results: list[dict], check: str, passed: bool, detail: str) -> None:
    results.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def policy_key_set(df: pd.DataFrame) -> set[str]:
    return set(add_policy_key(df)["__policy_key"])


def target_policy_names(df: pd.DataFrame) -> set[str]:
    return set(df.loc[~df["policy_name"].eq(CURRENT_PRACTICE_LABEL), "policy_name"].astype(str))


def check_same_policies(
    results: list[dict],
    g_summary: pd.DataFrame,
    aipw_summary: pd.DataFrame,
    ipw_summary: pd.DataFrame,
) -> None:
    sets = {
        "gformula": policy_key_set(g_summary),
        "aipw": policy_key_set(aipw_summary),
        "ipw": policy_key_set(ipw_summary),
    }
    reference = sets["gformula"]
    passed = all(values == reference for values in sets.values())
    detail = "; ".join(f"{name}={len(values)} policies" for name, values in sets.items())
    if not passed:
        detail += "; differences=" + str({name: sorted(values ^ reference) for name, values in sets.items()})
    add_result(results, "same_policies_across_methods", passed, detail)


def structural_episode_counts(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    df = add_policy_key(summary)
    if method == "ipw":
        total_col = np.where(
            df["policy_name"].eq(CURRENT_PRACTICE_LABEL),
            pd.to_numeric(df["n_episodes"], errors="coerce"),
            pd.to_numeric(df["n_total_policy_episodes"], errors="coerce"),
        )
        df["structural_n_episodes"] = total_col
    else:
        df["structural_n_episodes"] = pd.to_numeric(df["n_episodes"], errors="coerce")
    return df[["__policy_key", "policy_name", "policy_remove_day", "structural_n_episodes"]].copy()


def check_same_episode_counts(
    results: list[dict],
    g_summary: pd.DataFrame,
    aipw_summary: pd.DataFrame,
    ipw_summary: pd.DataFrame,
) -> None:
    g = structural_episode_counts(g_summary, "gformula").rename(
        columns={"structural_n_episodes": "n_gformula"}
    )
    a = structural_episode_counts(aipw_summary, "aipw").rename(
        columns={"structural_n_episodes": "n_aipw"}
    )
    i = structural_episode_counts(ipw_summary, "ipw").rename(
        columns={"structural_n_episodes": "n_ipw_total"}
    )
    merged = g.merge(a[["__policy_key", "n_aipw"]], on="__policy_key", how="outer").merge(
        i[["__policy_key", "n_ipw_total"]],
        on="__policy_key",
        how="outer",
    )
    counts = merged[["n_gformula", "n_aipw", "n_ipw_total"]]
    passed = counts.notna().all(axis=1).all() and counts.nunique(axis=1).eq(1).all()
    if passed:
        detail = f"all {len(merged)} policy rows have matching structural episode counts"
    else:
        detail = merged.loc[
            ~(counts.notna().all(axis=1) & counts.nunique(axis=1).eq(1)),
            ["policy_name", "policy_remove_day", "n_gformula", "n_aipw", "n_ipw_total"],
        ].to_string(index=False)
    add_result(results, "same_structural_episode_counts", bool(passed), detail)


def check_gformula_matches_aipw_plugin(
    results: list[dict],
    g_summary: pd.DataFrame,
    aipw_summary: pd.DataFrame,
    tolerance: float,
) -> None:
    mappings = [
        ("predicted_cauti_risk", "plugin_predicted_cauti_risk"),
        ("predicted_recatheterisation_risk", "plugin_predicted_recatheterisation_risk"),
        ("predicted_death_risk", "plugin_predicted_death_risk"),
        ("predicted_icu_exit_alive_risk", "plugin_predicted_icu_exit_alive_risk"),
        ("expected_mean_catheter_exposure_days", "plugin_expected_mean_catheter_exposure_days"),
        ("expected_mean_catheter_in_interval_rows", "plugin_expected_mean_catheter_in_interval_rows"),
    ]
    g = add_policy_key(g_summary)
    a = add_policy_key(aipw_summary)
    merged = g.merge(a, on="__policy_key", suffixes=("_g", "_a"), how="inner")

    failures = []
    checked = 0
    max_diff = 0.0
    for g_col, a_col in mappings:
        if g_col not in merged.columns or a_col not in merged.columns:
            continue
        checked += 1
        diff = (
            pd.to_numeric(merged[g_col], errors="coerce")
            - pd.to_numeric(merged[a_col], errors="coerce")
        ).abs()
        max_diff = max(max_diff, float(diff.max(skipna=True)))
        bad = diff.gt(tolerance) | diff.isna()
        if bad.any():
            failures.append(
                merged.loc[bad, ["policy_name_g", "policy_remove_day_g", g_col, a_col]].assign(
                    compared_columns=f"{g_col} vs {a_col}",
                    abs_diff=diff.loc[bad].to_numpy(),
                )
            )

    passed = checked > 0 and not failures
    detail = f"checked {checked} column pairs; max_abs_diff={max_diff:.3g}; tolerance={tolerance}"
    if failures:
        detail = pd.concat(failures, ignore_index=True).head(20).to_string(index=False)
    add_result(results, "gformula_values_match_aipw_plugin_values", passed, detail)


def check_aipw_ess_leq_adherent(
    results: list[dict],
    aipw_summary: pd.DataFrame,
    aipw_weight_diagnostics: pd.DataFrame,
    tolerance: float,
) -> None:
    failures = []
    for label, df in [
        ("aipw_summary", aipw_summary),
        ("aipw_weight_diagnostics", aipw_weight_diagnostics),
    ]:
        if "residual_correction_effective_sample_size" in df.columns:
            ess_col = "residual_correction_effective_sample_size"
        else:
            ess_col = "effective_sample_size"
        require_columns(df, ["policy_name", "n_adherent_episodes", ess_col], label)
        target = df.loc[~df["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
        bad = pd.to_numeric(target[ess_col], errors="coerce").gt(
            pd.to_numeric(target["n_adherent_episodes"], errors="coerce") + tolerance
        )
        if bad.any():
            failures.append(
                target.loc[bad, ["policy_name", "policy_remove_day", ess_col, "n_adherent_episodes"]].assign(
                    source=label
                )
            )
    passed = not failures
    detail = "AIPW residual-correction ESS is <= adherent episodes in summary and diagnostics"
    if failures:
        detail = pd.concat(failures, ignore_index=True).to_string(index=False)
    add_result(results, "aipw_ess_lte_adherent_episodes", passed, detail)


def check_ipw_summary_flags_match_diagnostics(
    results: list[dict],
    ipw_summary: pd.DataFrame,
    ipw_weight_diagnostics: pd.DataFrame,
    ipw_support_diagnostics: pd.DataFrame,
) -> None:
    summary = add_policy_key(ipw_summary)
    weight = add_policy_key(ipw_weight_diagnostics)
    failures = []

    flag_cols = ["low_adherence_flag", "low_ess_flag", "extreme_weight_flag"]
    merged = summary.merge(
        weight[["__policy_key", *flag_cols]],
        on="__policy_key",
        how="left",
        suffixes=("_summary", "_weight_diagnostics"),
    )
    for flag_col in flag_cols:
        summary_col = f"{flag_col}_summary"
        diag_col = f"{flag_col}_weight_diagnostics"
        target = merged.loc[~merged["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
        mismatch = bool_series(target[summary_col]).ne(bool_series(target[diag_col]))
        if mismatch.any():
            failures.append(
                target.loc[mismatch, ["policy_name", "policy_remove_day", summary_col, diag_col]].assign(
                    flag=flag_col
                )
            )

    support_all = ipw_support_diagnostics.loc[
        ipw_support_diagnostics["group"].astype("string").eq("all")
    ].copy()
    support_all = add_policy_key(support_all)
    support_merged = summary.merge(
        support_all[["__policy_key", "low_support_flag"]],
        on="__policy_key",
        how="left",
        suffixes=("_summary", "_support_diagnostics"),
    )
    target = support_merged.loc[~support_merged["policy_name"].eq(CURRENT_PRACTICE_LABEL)].copy()
    mismatch = bool_series(target["low_support_flag_summary"]).ne(
        bool_series(target["low_support_flag_support_diagnostics"])
    )
    if mismatch.any():
        failures.append(
            target.loc[
                mismatch,
                [
                    "policy_name",
                    "policy_remove_day",
                    "low_support_flag_summary",
                    "low_support_flag_support_diagnostics",
                ],
            ].assign(flag="low_support_flag")
        )

    passed = not failures
    detail = "IPW summary flags match weight/support diagnostics for target policies"
    if failures:
        detail = pd.concat(failures, ignore_index=True).to_string(index=False)
    add_result(results, "ipw_summary_flags_match_diagnostics", passed, detail)


def check_remove_rows(results: list[dict], policy_qa: pd.DataFrame, g_diagnostics: pd.DataFrame, aipw_summary: pd.DataFrame) -> None:
    failures = []
    for label, df in [
        ("policy_intervention_qa", policy_qa),
        ("gformula_diagnostics", g_diagnostics),
        ("aipw_summary", aipw_summary),
    ]:
        if "n_episodes_with_more_than_one_remove_row" not in df.columns:
            continue
        bad = pd.to_numeric(df["n_episodes_with_more_than_one_remove_row"], errors="coerce").fillna(0).gt(0)
        if bad.any():
            failures.append(
                df.loc[bad, ["policy_name", "policy_remove_day", "n_episodes_with_more_than_one_remove_row"]].assign(
                    source=label
                )
            )
    passed = not failures
    detail = "no method reports more than one remove row per episode-policy"
    if failures:
        detail = pd.concat(failures, ignore_index=True).to_string(index=False)
    add_result(results, "no_more_than_one_remove_row_per_episode_policy", passed, detail)


def check_no_missing_predictions(
    results: list[dict],
    g_summary: pd.DataFrame,
    g_diagnostics: pd.DataFrame,
    aipw_summary: pd.DataFrame,
    aipw_episodes: pd.DataFrame,
) -> None:
    checks = []
    if "n_incomplete_prediction_episodes" in g_summary.columns:
        checks.append(("gformula_summary_incomplete", int(pd.to_numeric(g_summary["n_incomplete_prediction_episodes"], errors="coerce").fillna(0).sum())))
    g_missing_cols = [col for col in g_diagnostics.columns if col.startswith("n_missing_")]
    for col in g_missing_cols:
        checks.append((f"gformula_diagnostics_{col}", int(pd.to_numeric(g_diagnostics[col], errors="coerce").fillna(0).sum())))
    if "n_incomplete_prediction_episodes" in aipw_summary.columns:
        checks.append(("aipw_summary_incomplete", int(pd.to_numeric(aipw_summary["n_incomplete_prediction_episodes"], errors="coerce").fillna(0).sum())))
    if "n_missing_prediction_rows" in aipw_episodes.columns:
        checks.append(("aipw_episode_missing_rows", int(pd.to_numeric(aipw_episodes["n_missing_prediction_rows"], errors="coerce").fillna(0).sum())))

    failures = [(name, value) for name, value in checks if value != 0]
    passed = bool(checks) and not failures
    detail = "all prediction-missing counters are zero"
    if failures:
        detail = str(failures)
    add_result(results, "no_missing_predictions", passed, detail)


def check_no_probabilities_outside_unit_interval(
    results: list[dict],
    g_summary: pd.DataFrame,
    g_diagnostics: pd.DataFrame,
    aipw_summary: pd.DataFrame,
    ipw_summary: pd.DataFrame,
) -> None:
    failures = []
    for col in ["n_predictions_below_0", "n_predictions_above_1"]:
        if col in g_diagnostics.columns:
            count = int(pd.to_numeric(g_diagnostics[col], errors="coerce").fillna(0).sum())
            if count:
                failures.append(f"gformula_diagnostics {col}={count}")

    out_of_bounds_cols = [col for col in aipw_summary.columns if col.endswith("_out_of_bounds") or "_out_of_bounds_" in col]
    for col in out_of_bounds_cols:
        if bool_series(aipw_summary[col]).any():
            failures.append(f"aipw_summary {col} has true values")

    for label, df in [
        ("gformula_summary", g_summary),
        ("aipw_summary", aipw_summary),
        ("ipw_summary", ipw_summary),
    ]:
        probability_cols = [
            col
            for col in df.columns
            if col.endswith("_risk")
            and "difference" not in col
            and "ratio" not in col
            and not col.endswith("_risk_pct")
        ]
        for col in probability_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            bad = values.lt(0).sum() + values.gt(1).sum()
            if bad:
                failures.append(f"{label} {col} has {int(bad)} values outside [0, 1]")

    passed = not failures
    detail = "diagnostic counters and summary probability columns are within [0, 1]"
    if failures:
        detail = "; ".join(failures)
    add_result(results, "no_probabilities_outside_0_1", passed, detail)


def check_remove_day_1_poor_support(
    results: list[dict],
    ipw_summary: pd.DataFrame,
    ipw_support_diagnostics: pd.DataFrame,
    aipw_summary: pd.DataFrame,
) -> None:
    failures = []
    for label, df in [("ipw_summary", ipw_summary), ("aipw_summary", aipw_summary)]:
        row = df.loc[df["policy_name"].eq("remove_on_day_1")]
        if row.empty or "low_support_flag" not in row.columns or not is_true(row["low_support_flag"].iloc[0]):
            failures.append(f"{label} remove_on_day_1 low_support_flag is not true")

    support_all = ipw_support_diagnostics.loc[
        ipw_support_diagnostics["policy_name"].eq("remove_on_day_1")
        & ipw_support_diagnostics["group"].astype("string").eq("all")
    ]
    if support_all.empty or not is_true(support_all["low_support_flag"].iloc[0]):
        failures.append("ipw_support_diagnostics remove_on_day_1/all low_support_flag is not true")

    passed = not failures
    detail = "remove_on_day_1 is flagged as poor support in IPW/AIPW outputs"
    if failures:
        detail = "; ".join(failures)
    add_result(results, "remove_day_1_flagged_as_poor_support", passed, detail)


def save_report(report: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    results: list[dict] = []

    policy_qa = load_csv(args.policy_qa, "policy-intervention QA")
    g_summary = load_csv(args.gformula_summary, "g-formula summary")
    g_diagnostics = load_csv(args.gformula_diagnostics, "g-formula diagnostics")
    aipw_summary = load_csv(args.aipw_summary, "AIPW summary")
    aipw_episodes = load_csv(args.aipw_episodes, "AIPW episode scores")
    aipw_weight_diagnostics = load_csv(args.aipw_weight_diagnostics, "AIPW weight diagnostics")
    ipw_summary = load_csv(args.ipw_summary, "IPW summary")
    ipw_weight_diagnostics = load_csv(args.ipw_weight_diagnostics, "IPW weight diagnostics")
    ipw_support_diagnostics = load_csv(args.ipw_support_diagnostics, "IPW support diagnostics")

    for label, df in [
        ("policy QA", policy_qa),
        ("g-formula summary", g_summary),
        ("AIPW summary", aipw_summary),
        ("IPW summary", ipw_summary),
    ]:
        require_columns(df, ["policy_name", "policy_remove_day"], label)

    check_same_policies(results, g_summary, aipw_summary, ipw_summary)
    check_same_episode_counts(results, g_summary, aipw_summary, ipw_summary)
    check_gformula_matches_aipw_plugin(results, g_summary, aipw_summary, args.tolerance)
    check_aipw_ess_leq_adherent(results, aipw_summary, aipw_weight_diagnostics, args.tolerance)
    check_ipw_summary_flags_match_diagnostics(results, ipw_summary, ipw_weight_diagnostics, ipw_support_diagnostics)
    check_remove_rows(results, policy_qa, g_diagnostics, aipw_summary)
    check_no_missing_predictions(results, g_summary, g_diagnostics, aipw_summary, aipw_episodes)
    check_no_probabilities_outside_unit_interval(results, g_summary, g_diagnostics, aipw_summary, ipw_summary)
    check_remove_day_1_poor_support(results, ipw_summary, ipw_support_diagnostics, aipw_summary)

    report = pd.DataFrame(results)
    save_report(report, args.output)

    failed = report.loc[report["status"].eq("FAIL")]
    if failed.empty:
        print("BASELINE QA PASSED")
        print(f"Checks passed: {len(report)}")
        print(f"Saved audit report: {args.output}")
        return

    print("BASELINE QA FAILED")
    print(failed.to_string(index=False))
    print(f"Saved audit report: {args.output}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
