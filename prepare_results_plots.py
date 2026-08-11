#!/usr/bin/env python3
# Create baseline OPE progress plots

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import policy_eval_common as pec


# Paths

REPO_ROOT = Path(__file__).resolve().parent
POLICY_EVAL_DIR = REPO_ROOT / "artefacts" / "policy_eval"
OUTDIR = REPO_ROOT / "artefacts" / "results_plots"


def load_outputs():
    # Load the current policy-evaluation outputs from artefacts
    paths = {
        "g_summary": POLICY_EVAL_DIR / "gformula" / "gformula_policy_outcomes_summary.csv",
        "ipw_summary": POLICY_EVAL_DIR / "ipw" / "ipw_policy_outcomes_summary.csv",
        "ipw_weights": POLICY_EVAL_DIR / "ipw" / "ipw_weight_diagnostics.csv",
        "ipw_support": POLICY_EVAL_DIR / "ipw" / "ipw_policy_support_diagnostics.csv",
        "aipw_summary": POLICY_EVAL_DIR / "aipw" / "aipw_policy_outcomes_summary.csv",
        "current_practice": POLICY_EVAL_DIR / "ipw" / "current_practice_episode_outcomes.csv",
        "baseline_audit": POLICY_EVAL_DIR / "baseline_policy_evaluation_audit.csv",
    }

    print("Using files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}


def pretty_policy(name):
    # Convert policy names into short display labels
    if pd.isna(name):
        return "Missing"

    name = str(name)
    if name == "current_practice":
        return "Current practice"

    match = re.search(r"remove_on_day_(\d+)", name)
    if match:
        return f"Remove day {match.group(1)}"

    return name.replace("_", " ").title()


def policy_sort_key(name):
    # Sort current practice first, then fixed-day policies by day
    if name == "current_practice":
        return (0, 0)
    match = re.search(r"remove_on_day_(\d+)", str(name))
    if match:
        return (1, int(match.group(1)))
    return (2, 999)


def add_policy_labels(df):
    # Add readable labels and stable policy ordering
    out = df.copy()
    out["policy_label"] = out["policy_name"].map(pretty_policy)
    out["_policy_sort"] = out["policy_name"].map(policy_sort_key)
    return out.sort_values("_policy_sort").drop(columns="_policy_sort")


def savefig(path):
    # Save and close the current figure
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_observed_exposure_distribution(current, outdir):
    # Plot observed catheter exposure duration under current practice
    duration = pd.to_numeric(current["observed_catheter_exposure_days"], errors="coerce").dropna()
    duration = duration[duration >= 0]

    end_day = np.ceil(duration).astype(int)
    end_day[end_day < 1] = 1

    max_show_day = 10
    labels = np.where(end_day > max_show_day, f"{max_show_day + 1}+", end_day.astype(str))
    counts = pd.Series(labels).value_counts().sort_index(
        key=lambda s: s.map(lambda x: int(str(x).replace("+", "")))
    )
    percents = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(percents.index.astype(str), percents.values)
    ax.set_title("Observed catheter exposure duration")
    ax.set_xlabel("Catheter exposure duration, days")
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0, max(percents.values) * 1.2)

    for i, value in enumerate(percents.values):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.text(
        0.5,
        -0.22,
        "Source: observed_catheter_exposure_days in current-practice episode outputs.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    savefig(outdir / "01_observed_catheter_exposure_duration.png")


def plot_policy_tradeoff(aipw, outdir):
    # Plot the AIPW CAUTI versus recatheterisation policy trade-off
    df = add_policy_labels(aipw)

    x = pd.to_numeric(df["aipw_recatheterisation_risk_pct"], errors="coerce")
    y = pd.to_numeric(df["aipw_cauti_risk_pct"], errors="coerce")
    exposure = pd.to_numeric(df["aipw_mean_catheter_exposure_days"], errors="coerce")

    sizes = 80 + (exposure.fillna(exposure.median()) * 45)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=sizes, alpha=0.75)

    for _, row in df.iterrows():
        ax.annotate(
            row["policy_label"],
            (
                pd.to_numeric(row["aipw_recatheterisation_risk_pct"], errors="coerce"),
                pd.to_numeric(row["aipw_cauti_risk_pct"], errors="coerce"),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Baseline AIPW policy trade-off")
    ax.set_xlabel("Estimated recatheterisation risk (%)")
    ax.set_ylabel("Estimated CAUTI risk (%)")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.5,
        -0.18,
        "Point size represents estimated catheter exposure days.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    savefig(outdir / "02_policy_tradeoff_aipw_cauti_vs_recatheterisation.png")


def build_estimator_comparison(g, ipw, aipw, outcome):
    # Build the long table used for estimator comparison plots
    if outcome == "cauti":
        cols = {
            "gformula": "predicted_cauti_risk_pct",
            "ipw": "ipw_weighted_cauti_risk_pct",
            "aipw": "aipw_cauti_risk_pct",
        }
    elif outcome == "recatheterisation":
        cols = {
            "gformula": "predicted_recatheterisation_risk_pct",
            "ipw": "ipw_weighted_recatheterisation_risk_pct",
            "aipw": "aipw_recatheterisation_risk_pct",
        }
    else:
        raise ValueError("outcome must be 'cauti' or 'recatheterisation'")

    parts = []
    for method, df, col in [
        ("G-formula", g, cols["gformula"]),
        ("IPW", ipw, cols["ipw"]),
        ("AIPW", aipw, cols["aipw"]),
    ]:
        tmp = df[["policy_name", col]].copy()
        tmp = tmp.rename(columns={col: "risk_pct"})
        tmp["method"] = method
        parts.append(tmp)

    return add_policy_labels(pd.concat(parts, ignore_index=True))


def plot_estimator_comparison(g, ipw, aipw, outcome, outdir):
    # Plot estimator comparisons as policy-indexed points
    df = build_estimator_comparison(g, ipw, aipw, outcome)
    pivot = df.pivot_table(index="policy_label", columns="method", values="risk_pct", aggfunc="first").reset_index()

    policy_order = add_policy_labels(pd.DataFrame({"policy_name": df["policy_name"].unique()}))["policy_label"].tolist()
    pivot["policy_label"] = pd.Categorical(pivot["policy_label"], categories=policy_order, ordered=True)
    pivot = pivot.sort_values("policy_label")

    x = np.arange(len(pivot))
    methods = ["G-formula", "IPW", "AIPW"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        ax.scatter(x, pivot[method], s=55, label=method)

    title_outcome = "CAUTI" if outcome == "cauti" else "Recatheterisation"
    ax.set_title(f"Baseline estimator comparison: {title_outcome}")
    ax.set_xlabel("Policy")
    ax.set_ylabel(f"Estimated {title_outcome} risk (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["policy_label"], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    savefig(outdir / f"03_estimator_comparison_{outcome}.png")


def plot_ess_support(ipw_weights, ipw_support, outdir):
    # Plot IPW support through ESS and low-support proportions
    support = ipw_support.loc[ipw_support["group"].eq("all"), ["policy_name", "pct_below_0_05"]]
    df = add_policy_labels(ipw_weights.merge(support, on="policy_name", how="left"))

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(df))
    ess = pd.to_numeric(df["effective_sample_size"], errors="coerce")
    adherent = pd.to_numeric(df["n_adherent_episodes"], errors="coerce")
    pct_adherent = pd.to_numeric(df["pct_adherent_episodes"], errors="coerce") * 100
    pct_low_support = pd.to_numeric(df["pct_below_0_05"], errors="coerce") * 100

    ax.barh(y, ess)
    ax.set_yticks(y)
    ax.set_yticklabels(df["policy_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Effective sample size")
    ax.set_title("IPW policy support diagnostics")
    ax.grid(axis="x", alpha=0.3)

    for i, (e, n, pct, low_support) in enumerate(zip(ess, adherent, pct_adherent, pct_low_support)):
        ax.text(
            e,
            i,
            f"  ESS {e:.0f}; adherent {int(n):,} ({pct:.1f}%); support <0.05: {low_support:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.text(
        0.5,
        -0.16,
        "Low ESS indicates weak empirical support for the target policy under observed practice.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    savefig(outdir / "04_effective_sample_size_by_policy.png")


def format_baseline_audit_table(audit):
    # Format baseline audit checks for the QA table
    out = audit[["check", "status", "detail"]].copy()
    out = out.rename(columns={"check": "Check", "status": "Result", "detail": "Detail"})
    out["Result"] = out["Result"].map({"PASS": "Passed", "FAIL": "Check"}).fillna(out["Result"])
    return out


def render_qa_table(qa, outdir):
    # Save the baseline audit checks as a PNG
    outdir.mkdir(parents=True, exist_ok=True)

    fig_height = max(4, 0.45 * len(qa) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=qa.values,
        colLabels=qa.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    ax.set_title("Baseline OPE validation checks", pad=20, fontsize=14)

    savefig(outdir / "05_baseline_ope_qa_checks.png")


def build_results_summary(outputs, outdir):
    # Save a compact policy summary table
    g = outputs["g_summary"][
        [
            "policy_name",
            "predicted_cauti_risk_pct",
            "predicted_recatheterisation_risk_pct",
            "expected_mean_catheter_exposure_days",
        ]
    ].copy()

    ipw = outputs["ipw_summary"][
        [
            "policy_name",
            "ipw_weighted_cauti_risk_pct",
            "ipw_weighted_recatheterisation_risk_pct",
            "effective_sample_size",
            "pct_adherent_episodes",
            "low_adherence_flag",
            "low_ess_flag",
            "low_support_flag",
            "extreme_weight_flag",
        ]
    ].copy()

    aipw = outputs["aipw_summary"][
        [
            "policy_name",
            "aipw_cauti_risk_pct",
            "aipw_recatheterisation_risk_pct",
            "aipw_mean_catheter_exposure_days",
        ]
    ].copy()

    summary = g.merge(ipw, on="policy_name", how="outer").merge(aipw, on="policy_name", how="outer")
    summary = add_policy_labels(summary)

    path = outdir / "policy_summary_table.csv"
    pec.save_report_df(summary, path)
    print(f"Saved: {path}")


def main():
    # Create the plot directory
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load all estimator and audit outputs
    outputs = load_outputs()

    # Plot current-practice exposure duration
    plot_observed_exposure_distribution(outputs["current_practice"], OUTDIR)

    # Plot the AIPW trade-off across target policies
    plot_policy_tradeoff(outputs["aipw_summary"], OUTDIR)

    # Plot estimator agreement for CAUTI
    plot_estimator_comparison(
        outputs["g_summary"],
        outputs["ipw_summary"],
        outputs["aipw_summary"],
        "cauti",
        OUTDIR,
    )

    # Plot estimator agreement for recatheterisation
    plot_estimator_comparison(
        outputs["g_summary"],
        outputs["ipw_summary"],
        outputs["aipw_summary"],
        "recatheterisation",
        OUTDIR,
    )

    # Plot IPW support and adherence diagnostics
    plot_ess_support(outputs["ipw_weights"], outputs["ipw_support"], OUTDIR)

    # Render the baseline QA checks
    qa = format_baseline_audit_table(outputs["baseline_audit"])
    render_qa_table(qa, OUTDIR)

    # Save the compact summary table
    build_results_summary(outputs, OUTDIR)

    print()
    print("--- DONE ---")
    print(f"Plots saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
