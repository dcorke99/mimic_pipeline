from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
SEED = 42
N_ROLLOUTS = 50
PROGRESS_EVERY = 100

INDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step1")
OUTDIR = Path(r"C:\Users\DavidUni\Repos\mimic_pipeline\artifacts\step3")
MODEL_DIR = OUTDIR

OUTDIR.mkdir(exist_ok=True, parents=True)

PANEL_FILE = INDIR / "step1_scored_panel.csv"

TRAJ_KEYS = ["subject_id", "hadm_id", "stay_id", "inserted"]

TIME_COL = "episode_index"
START_COL = "period_start"
INSERTED_COL = "inserted"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
INTERVAL_COL = "interval_hours"
SPLIT_COL = "split"

# Utility weights
W_CAUTI = 10.0
W_REINS = 3.0
W_CATH_DAYS = 1.0

rng = np.random.default_rng(SEED)

# -----------------------------
# Same policies as Step-2
# -----------------------------
# RISK_TAUS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
RISK_TAUS = [0.01, 0.02]

def policy_fixed_period_remove(periods_in_state: int, remove_period: int) -> int:
    return 1 if periods_in_state >= remove_period else 0


def policy_risk_threshold(p_cauti_keep: float, tau: float) -> int:
    return 1 if p_cauti_keep >= tau else 0


def policy_hybrid(
    p_cauti_keep: float,
    p_reins_if_remove: float,
    tau_cauti: float = 0.15,
    ratio: float = 1.0,
) -> int:
    if p_cauti_keep < tau_cauti:
        return 0
    return 1 if p_cauti_keep > ratio * p_reins_if_remove else 0


POLICIES = [
    ("fixed_period3", lambda r: policy_fixed_period_remove(int(r[PERIODS_COL]), 3)),
    ("fixed_period5", lambda r: policy_fixed_period_remove(int(r[PERIODS_COL]), 5)),
    # ("fixed_period7", lambda r: policy_fixed_period_remove(int(r[PERIODS_COL]), 7)),
    *[
        (
            f"risk_tau_{tau:.2f}".replace(".", "_"),
            lambda r, tau=tau: policy_risk_threshold(float(r["p_cauti_if_keep"]), tau),
        )
        for tau in RISK_TAUS
    ],
    # (
    #    "hybrid_tau0_15_ratio1_0",
    #    lambda r: policy_hybrid(
    #        float(r["p_cauti_if_keep"]),
    #        float(r["p_reins_if_remove"]),
    #        tau_cauti=0.15,
    #        ratio=1.0,
    #    ),
    # ),
]

# -----------------------------
# Validation
# -----------------------------
def validate_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()

    n_train = int((df[SPLIT_COL] == "train").sum())
    if n_train == 0:
        print("[STEP3] Warning: no train rows found in Step-1 scored panel.", flush=True)

    return df


# -----------------------------
# Probability helpers
# -----------------------------
def _clip01(x: float) -> float:
    if pd.isna(x):
        return 0.0
    return float(np.clip(float(x), 0.0, 1.0))


def _resolve_competing(p_cauti: float, p_other: float) -> tuple[float, float]:
    """
    Turn two separately predicted interval probabilities into a valid competing pair.
    If they sum to >1, shrink them proportionally so total event probability is 1.
    """
    p_cauti = _clip01(p_cauti)
    p_other = _clip01(p_other)
    total = p_cauti + p_other

    if total <= 1.0:
        return p_cauti, p_other
    if total <= 1e-12:
        return 0.0, 0.0

    scale = 1.0 / total
    return p_cauti * scale, p_other * scale


def _sample_competing_event(
    p_cauti: float,
    p_other: float,
    rng: np.random.Generator,
) -> str:
    """
    Sample one of: 'cauti', 'other', 'none'
    """
    p_cauti, p_other = _resolve_competing(p_cauti, p_other)
    u = rng.uniform()

    if u < p_cauti:
        return "cauti"
    if u < p_cauti + p_other:
        return "other"
    return "none"


# -----------------------------
# Monte-Carlo rollout
# -----------------------------
def simulate_rollout(g: pd.DataFrame, policy_fn):
    """
    Simulate one stochastic trajectory under a candidate policy using Step-1
    predicted event probabilities at each observed interval.

    This is the agreed lighter Step-3 implementation:
    - it uses the observed row scaffold
    - it includes CAUTI risk while OUT
    - it does not fully rebuild counterfactual state-age/covariate histories
    """
    g = g.sort_values([START_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    first_row = g.iloc[0]
    state = str(first_row[STATE_COL]).strip().lower()

    cauti = 0
    reins = 0
    removals = 0
    in_days = 0.0
    out_days = 0.0

    for _, row in g.iterrows():
        if cauti == 1:
            break

        observed_state = str(row[STATE_COL]).strip().lower()
        interval_days = float(row[INTERVAL_COL]) / 24.0

        p_keep = _clip01(row.get("p_cauti_if_keep", 0.0))
        p_remove = _clip01(row.get("p_cauti_if_remove", 0.0))
        p_cauti_out = _clip01(row.get("p_cauti_if_out", 0.0))
        p_reins_remove = _clip01(row.get("p_reins_if_remove", 0.0))
        p_reins_out = _clip01(row.get("p_reins_if_out", 0.0))

        # -------------------------------------------------
        # Counterfactual IN state
        # -------------------------------------------------
        if state == "in":
            in_days += interval_days

            # Agreed lighter approximation:
            # only actively reapply the removal policy on observed IN rows.
            action = int(policy_fn(row)) if observed_state == "in" else 0

            if action == 1:
                removals += 1

                outcome = _sample_competing_event(
                    p_cauti=p_remove,
                    p_other=p_reins_remove,
                    rng=rng,
                )

                if outcome == "cauti":
                    cauti = 1
                    break
                elif outcome == "other":
                    reins += 1
                    state = "in"
                else:
                    state = "out"

            else:
                if rng.uniform() < p_keep:
                    cauti = 1
                    break
                state = "in"

        # -------------------------------------------------
        # Counterfactual OUT state
        # -------------------------------------------------
        else:
            out_days += interval_days

            outcome = _sample_competing_event(
                p_cauti=p_cauti_out,
                p_other=p_reins_out,
                rng=rng,
            )

            if outcome == "cauti":
                cauti = 1
                break
            elif outcome == "other":
                reins += 1
                state = "in"
            else:
                state = "out"

    utility = W_CAUTI * cauti + W_REINS * reins + W_CATH_DAYS * in_days
    return cauti, reins, removals, in_days, out_days, utility


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t_all0 = time.time()

    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df = validate_split(df)

    df[START_COL] = pd.to_datetime(df[START_COL], errors="coerce")
    df[INSERTED_COL] = pd.to_datetime(df[INSERTED_COL], errors="coerce")

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[PERIODS_COL] = pd.to_numeric(df[PERIODS_COL], errors="coerce")
    df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")

    df = df.dropna(
        subset=[
            START_COL,
            INSERTED_COL,
            TIME_COL,
            PERIODS_COL,
            INTERVAL_COL,
            "hadm_id",
            "stay_id",
        ]
    ).copy()

    df[TIME_COL] = df[TIME_COL].astype("int64")
    df[PERIODS_COL] = df[PERIODS_COL].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["stay_id"] = df["stay_id"].astype("int64")

    print("\n[STEP3] Split summary:", flush=True)
    print(df[SPLIT_COL].value_counts(dropna=False), flush=True)
    print(
        f"[STEP3] Unique patients: "
        f"train={df.loc[df[SPLIT_COL] == 'train', 'subject_id'].nunique()} "
        f"test={df.loc[df[SPLIT_COL] == 'test', 'subject_id'].nunique()}",
        flush=True
    )

    test_df = df[df[SPLIT_COL] == "test"].copy()

    test_df = test_df.sort_values(
        TRAJ_KEYS + [START_COL, TIME_COL],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = [g.copy() for _, g in test_df.groupby(TRAJ_KEYS, sort=False)]
    n_traj = len(grouped)

    print(f"[STEP3] Test rows: {len(test_df):,}", flush=True)
    print(f"[STEP3] Test trajectories: {n_traj}", flush=True)
    print(f"[STEP3] Policies: {len(POLICIES)}", flush=True)
    print(f"[STEP3] Rollouts per trajectory: {N_ROLLOUTS}", flush=True)

    policy_summary = []

    for policy_idx, (policy_name, policy_fn) in enumerate(POLICIES, start=1):
        t_policy0 = time.time()
        print(
            f"\n[STEP3] Starting policy {policy_idx}/{len(POLICIES)}: {policy_name}",
            flush=True,
        )

        patient_rows = []

        for i, g in enumerate(grouped, start=1):
            subj = str(g["subject_id"].iloc[0])
            hadm = int(g["hadm_id"].iloc[0])
            stay = int(g["stay_id"].iloc[0])
            inserted = g[INSERTED_COL].iloc[0]

            acc = np.zeros(6, dtype=float)

            for _ in range(N_ROLLOUTS):
                out = simulate_rollout(g, policy_fn)
                acc += np.array(out, dtype=float)

            acc /= N_ROLLOUTS

            patient_rows.append({
                "subject_id": subj,
                "hadm_id": hadm,
                "stay_id": stay,
                "inserted": inserted,
                "split": "test",
                "policy": policy_name,
                "cauti_mean": acc[0],
                "reinsertions_mean": acc[1],
                "removals_mean": acc[2],
                "catheter_in_days_mean": acc[3],
                "catheter_out_days_mean": acc[4],
                "utility_mean": acc[5],
            })

            if i % PROGRESS_EVERY == 0 or i == n_traj:
                elapsed = time.time() - t_policy0
                per_traj = elapsed / i if i > 0 else float("nan")
                remaining = per_traj * (n_traj - i) if i > 0 else float("nan")
                print(
                    f"[STEP3] {policy_name}: {i}/{n_traj} trajectories "
                    f"| elapsed {elapsed:.1f}s "
                    f"| est remaining {remaining:.1f}s",
                    flush=True,
                )

        patient_df = pd.DataFrame(patient_rows)

        patient_df.to_csv(
            OUTDIR / f"step3_patient_summary__{policy_name}.csv",
            index=False,
        )

        if len(patient_df) > 0:
            policy_summary.append({
                "policy": policy_name,
                "split_used": "test",
                "n_trajectories": len(patient_df),
                "cauti_mean": patient_df["cauti_mean"].mean(),
                "reinsertions_mean": patient_df["reinsertions_mean"].mean(),
                "removals_mean": patient_df["removals_mean"].mean(),
                "catheter_in_days_mean": patient_df["catheter_in_days_mean"].mean(),
                "catheter_out_days_mean": patient_df["catheter_out_days_mean"].mean(),
                "utility_mean": patient_df["utility_mean"].mean(),
            })
        else:
            policy_summary.append({
                "policy": policy_name,
                "split_used": "test",
                "n_trajectories": 0,
                "cauti_mean": np.nan,
                "reinsertions_mean": np.nan,
                "removals_mean": np.nan,
                "catheter_in_days_mean": np.nan,
                "catheter_out_days_mean": np.nan,
                "utility_mean": np.nan,
            })

        print(
            f"[STEP3] Finished {policy_name} in {time.time() - t_policy0:.1f}s",
            flush=True,
        )

    pd.DataFrame(policy_summary).to_csv(
        OUTDIR / "step3_policy_summary.csv",
        index=False,
    )

    print(f"\n[STEP3] Complete in {time.time() - t_all0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
