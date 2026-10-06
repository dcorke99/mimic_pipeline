#!/usr/bin/env python3
# Audit structural consistency across policy-evaluation outputs


from pathlib import Path

import numpy as np
import pandas as pd

import policy_eval_common as pec


# Configuration

REPO_ROOT = Path(__file__).resolve().parent
BOOTSTRAP_MODE = "fixed"  # Select "fixed" or "refit" results for this audit.
if BOOTSTRAP_MODE not in ("fixed", "refit"):
    raise ValueError(f"Unknown bootstrap mode: {BOOTSTRAP_MODE!r}")

POLICY_CHECK_PATH = REPO_ROOT / "artefacts" / "counterfactual_policies" / "policy_panel_checks.csv"
GFORMULA_SUMMARY_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "gformula" / BOOTSTRAP_MODE / "gformula_policy_outcomes_summary.csv"
)
GFORMULA_DIAGNOSTICS_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "gformula" / BOOTSTRAP_MODE / "gformula_diagnostics.csv"
)
AIPW_SUMMARY_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "aipw" / BOOTSTRAP_MODE / "aipw_policy_outcomes_summary.csv"
)
AIPW_WEIGHT_DIAGNOSTICS_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "aipw" / BOOTSTRAP_MODE / "aipw_weight_diagnostics.csv"
)
IPW_SUMMARY_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "ipw" / BOOTSTRAP_MODE / "ipw_policy_outcomes_summary.csv"
)
IPW_WEIGHT_DIAGNOSTICS_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "ipw" / BOOTSTRAP_MODE / "ipw_weight_diagnostics.csv"
)
IPW_SUPPORT_DIAGNOSTICS_PATH = (
    REPO_ROOT / "artefacts" / "policy_eval" / "ipw" / BOOTSTRAP_MODE / "ipw_policy_support_diagnostics.csv"
)
OUTPUT_PATH = REPO_ROOT / "artefacts" / "policy_eval" / f"policy_results_checks_{BOOTSTRAP_MODE}.csv"

CURRENT_PRACTICE_LABEL = "current_practice"
TOLERANCE = 1e-10


def add_policy_key(df):
    out = df.copy()
    day = pd.to_numeric(out["policy_remove_day"], errors="raise").fillna(-1)
    out["__policy_key"] = list(zip(out["policy_name"], day))
    return out


def bool_series(series):
    values = series.astype("string").str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    })
    if values.isna().any():
        raise ValueError(f"{series.name} contains missing or invalid boolean flags")
    return values.astype(bool)


def add_result(results, check, passed, detail):
    results.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def policy_key_set(df):
    # Build key set
    return set(add_policy_key(df)["__policy_key"])


def check_same_policies(
    results,
    g_summary,
    aipw_summary,
    ipw_summary,
):
    # Check same policies
    # Build key set
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


def structural_episode_counts(summary, method):
    # Build structural episode counts
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
    results,
    g_summary,
    aipw_summary,
    ipw_summary,
):
    # Check same episode counts
    # Build structural episode counts
    g = structural_episode_counts(g_summary, "gformula").rename(
        columns={"structural_n_episodes": "n_gformula"}
    )
    # Build structural episode counts
    a = structural_episode_counts(aipw_summary, "aipw").rename(
        columns={"structural_n_episodes": "n_aipw"}
    )
    # Build structural episode counts
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
    results,
    g_summary,
    aipw_summary,
    tolerance,
):
    # Check gformula matches AIPW plugin
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
    results,
    aipw_summary,
    aipw_weight_diagnostics,
    tolerance,
):
    # Check AIPW ESS leq adherent
    failures = []
    for label, df in [
        ("aipw_summary", aipw_summary),
        ("aipw_weight_diagnostics", aipw_weight_diagnostics),
    ]:
        ess_col = "residual_correction_effective_sample_size"
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
    results,
    ipw_summary,
    ipw_weight_diagnostics,
    ipw_support_diagnostics,
):
    # Check IPW summary flags match diagnostics
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


def check_remove_rows(results, policy_check, g_diagnostics, aipw_summary):
    # Check remove rows
    failures = []
    for label, df in [
        ("policy_panel_checks", policy_check),
        ("gformula_diagnostics", g_diagnostics),
        ("aipw_summary", aipw_summary),
    ]:
        bad = pd.to_numeric(df["n_episodes_with_more_than_one_remove_row"], errors="coerce").ne(0)
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


def check_no_missing_predictions(results, g_summary, g_diagnostics, aipw_summary):
    counts = [
        ("gformula_summary_incomplete", g_summary["n_incomplete_prediction_episodes"]),
        ("aipw_summary_incomplete", aipw_summary["n_incomplete_prediction_episodes"]),
    ]
    missing_cols = [col for col in g_diagnostics if col.startswith("n_missing_")]
    counts.extend((f"gformula_diagnostics_{col}", g_diagnostics[col]) for col in missing_cols)
    failures = []
    if not missing_cols:
        failures.append("G-formula diagnostics have no prediction-missing counters")
    for name, series in counts:
        values = pd.to_numeric(series, errors="coerce")
        if values.empty or values.ne(0).any():
            failures.append(f"{name}: missing, invalid or nonzero counts")
    add_result(results, "no_missing_predictions", not failures,
               "; ".join(failures) if failures else "all prediction-missing counters are present and zero")


def check_no_probabilities_outside_unit_interval(
    results,
    g_summary,
    g_diagnostics,
    aipw_summary,
    ipw_summary,
):
    # Check no probabilities outside unit interval
    failures = []
    for col in ["n_predictions_below_0", "n_predictions_above_1"]:
        values = pd.to_numeric(g_diagnostics[col], errors="coerce")
        if values.empty or values.ne(0).any():
            failures.append(f"gformula_diagnostics {col} contains missing, invalid or nonzero counts")

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


def main():
    # Collect each result check
    results = []

    # Load the policy and estimator reports
    policy_check = pd.read_csv(POLICY_CHECK_PATH, low_memory=False)
    g_summary = pd.read_csv(GFORMULA_SUMMARY_PATH, low_memory=False)
    g_diagnostics = pd.read_csv(GFORMULA_DIAGNOSTICS_PATH, low_memory=False)
    aipw_summary = pd.read_csv(AIPW_SUMMARY_PATH, low_memory=False)
    aipw_weight_diagnostics = pd.read_csv(
        AIPW_WEIGHT_DIAGNOSTICS_PATH,
        low_memory=False,
    )
    ipw_summary = pd.read_csv(IPW_SUMMARY_PATH, low_memory=False)
    ipw_weight_diagnostics = pd.read_csv(
        IPW_WEIGHT_DIAGNOSTICS_PATH,
        low_memory=False,
    )
    ipw_support_diagnostics = pd.read_csv(
        IPW_SUPPORT_DIAGNOSTICS_PATH,
        low_memory=False,
    )

    # Check same policies
    check_same_policies(results, g_summary, aipw_summary, ipw_summary)
    # Check same episode counts
    check_same_episode_counts(results, g_summary, aipw_summary, ipw_summary)
    # Check gformula matches AIPW plugin
    check_gformula_matches_aipw_plugin(results, g_summary, aipw_summary, TOLERANCE)
    # Check AIPW ESS leq adherent
    check_aipw_ess_leq_adherent(
        results,
        aipw_summary,
        aipw_weight_diagnostics,
        TOLERANCE,
    )
    # Check IPW summary flags match diagnostics
    check_ipw_summary_flags_match_diagnostics(results, ipw_summary, ipw_weight_diagnostics, ipw_support_diagnostics)
    # Check remove rows
    check_remove_rows(results, policy_check, g_diagnostics, aipw_summary)
    # Check no missing predictions
    check_no_missing_predictions(results, g_summary, g_diagnostics, aipw_summary)
    # Check no probabilities outside unit interval
    check_no_probabilities_outside_unit_interval(results, g_summary, g_diagnostics, aipw_summary, ipw_summary)

    # Save the result checks
    report = pd.DataFrame(results)
    pec.save_report_df(report, OUTPUT_PATH)

    # Stop the pipeline when any check fails
    failed = report.loc[report["status"].eq("FAIL")]
    if failed.empty:
        print("POLICY RESULTS CHECKS PASSED")
        print(f"Checks passed: {len(report)}")
        print(f"Saved result checks: {OUTPUT_PATH}")
        return

    print("POLICY RESULTS CHECKS FAILED")
    print(failed.to_string(index=False))
    print(f"Saved result checks: {OUTPUT_PATH}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
