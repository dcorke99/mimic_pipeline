"""
04_step4_causal_policy_evaluation.py
Step 4 — Standard causal policy evaluation using stabilised IPW for dynamic regimes.

Uses the Step-1 scored panel and evaluates target policies using stabilised
inverse-probability weighting on decision-eligible IN-state intervals.

Outputs:
- artifacts/step4/step4_policy_value_estimates.csv
- artifacts/step4/step4_patient_summary__<policy>.csv
- artifacts/step4/step4_clipping_sensitivity__<policy>.csv   (selected policies)
- artifacts/step4/step4_config.json

Run:
  python 04_step4_causal_policy_evaluation.py
"""

from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Config
# -----------------------------
SEED = 42
EPS = 1e-6
BOOTSTRAP = 300
CLIP_CUM_W = 50.0
BOOT_PROGRESS_EVERY = 25

# Clipping sensitivity settings
SENSITIVITY_POLICIES = {
    "fixed_period5",
    "fixed_period7",
    "risk_tau_0_05",
    "risk_tau_0_10",
    "risk_tau_0_15",
}
SENSITIVITY_CLIPS = [None, 100.0, 50.0, 25.0, 10.0]

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step1")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step4")
MODEL_DIR = OUTDIR

OUTDIR.mkdir(exist_ok=True, parents=True)

INFILE = INDIR / "step1_scored_panel.csv"

ID_COL = "subject_id"
TRAJ_KEYS = ["subject_id", "hadm_id", "stay_id", "inserted"]

TIME_COL = "episode_index"
START_COL = "period_start"
INSERTED_COL = "inserted"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
INTERVAL_COL = "interval_hours"
SPLIT_COL = "split"

A_COL = "removed_in_period"
CAUTI_TODAY = "cauti_in_period"
REINS_TODAY = "reinsertion_in_period"

# Utility weights
W_CAUTI = 10.0
W_REINS = 3.0
W_CATH_DAYS = 1.0

# -----------------------------
# Same policies as Step 2 / 3
# -----------------------------
RISK_TAUS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


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
    ("fixed_period7", lambda r: policy_fixed_period_remove(int(r[PERIODS_COL]), 7)),
    *[
        (
            f"risk_tau_{tau:.2f}".replace(".", "_"),
            lambda r, tau=tau: policy_risk_threshold(float(r["p_cauti_if_keep"]), tau),
        )
        for tau in RISK_TAUS
    ],
    (
        "hybrid_tau0_15_ratio1_0",
        lambda r: policy_hybrid(
            float(r["p_cauti_if_keep"]),
            float(r["p_reins_if_remove"]),
            tau_cauti=0.15,
            ratio=1.0,
        ),
    ),
]

# -----------------------------
# Validation / feature handling
# -----------------------------
def validate_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [TIME_COL, PERIODS_COL, INTERVAL_COL]
    cols += [c for c in df.columns if c.startswith("itemid_")]
    cols += [c for c in df.columns if c.startswith("sex_")]
    cols += [c for c in df.columns if c.startswith("ethnicity_")]
    cols.append("age")
    return list(dict.fromkeys(cols))


def numerator_cols() -> list[str]:
    # Stabilisation model: time-only terms
    return [TIME_COL, PERIODS_COL]


# -----------------------------
# Utilities
# -----------------------------
def build_decision_eligibility(df: pd.DataFrame, traj_keys: list[str]) -> pd.DataFrame:
    """
    Decision-eligible rows:
    - current row is IN state
    - no CAUTI has already occurred earlier in this trajectory
    """
    g = df.sort_values(traj_keys + [START_COL, TIME_COL], kind="mergesort").copy()

    g["cauti_cum"] = g.groupby(traj_keys, sort=False)[CAUTI_TODAY].cumsum()
    g["prior_cauti"] = g.groupby(traj_keys, sort=False)["cauti_cum"].shift(fill_value=0)

    g["decision_eligible"] = (
        (g[STATE_COL].astype(str).str.strip().str.lower() == "in")
        & (g["prior_cauti"] == 0)
    ).astype(int)

    return g


def fit_propensity_models(
    df_dec_train: pd.DataFrame,
    X_cols_den: list[str],
    X_cols_num: list[str],
) -> tuple[Pipeline, Pipeline]:
    """
    Denominator: P(A_t | rich covariates)
    Numerator:   P(A_t | time only) for stabilisation

    Fit on TRAIN decision-eligible rows only.
    """
    if len(df_dec_train) == 0:
        raise ValueError("No TRAIN decision-eligible rows available for propensity fitting.")

    y = df_dec_train[A_COL].astype(int).to_numpy()
    if pd.Series(y).nunique() < 2:
        raise ValueError("TRAIN decision-eligible action labels have fewer than 2 classes.")

    pipe_den = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=SEED,
        )),
    ])
    pipe_den.fit(df_dec_train[X_cols_den].to_numpy(dtype=float), y)

    pipe_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=SEED,
        )),
    ])
    pipe_num.fit(df_dec_train[X_cols_num].to_numpy(dtype=float), y)

    return pipe_den, pipe_num


def predict_pA(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    p = pipe.predict_proba(X)[:, 1]
    return np.clip(p, EPS, 1.0 - EPS)


def build_trajectory_utility(df: pd.DataFrame, traj_keys: list[str]) -> pd.DataFrame:
    """
    Observed realised utility per trajectory.
    Expected input here is the TEST evaluation panel.
    """
    g = df.sort_values(traj_keys + [START_COL, TIME_COL], kind="mergesort").copy()

    g["catheter_in_days_interval"] = np.where(
        g[STATE_COL].astype(str).str.strip().str.lower() == "in",
        g[INTERVAL_COL] / 24.0,
        0.0,
    )

    out = (
        g.groupby(traj_keys, sort=False, as_index=False)
        .agg(
            cauti=(CAUTI_TODAY, "max"),
            reinsertions=(REINS_TODAY, "sum"),
            catheter_in_days=("catheter_in_days_interval", "sum"),
        )
        .copy()
    )

    out["cauti"] = out["cauti"].astype(int)
    out["reinsertions"] = out["reinsertions"].astype(int)
    out["utility"] = (
        W_CAUTI * out["cauti"]
        + W_REINS * out["reinsertions"]
        + W_CATH_DAYS * out["catheter_in_days"]
    )
    return out


def summarise_weights(w: pd.Series) -> dict:
    """
    Summarise trajectory-level final regime weights.
    """
    w = pd.to_numeric(w, errors="coerce")
    w = w[np.isfinite(w)].astype(float)

    if len(w) == 0:
        return {
            "n_trajectories": 0,
            "n_positive_weight": 0,
            "zero_weight_prop": float("nan"),
            "ess": float("nan"),
            "weight_min": float("nan"),
            "weight_p25": float("nan"),
            "weight_p50": float("nan"),
            "weight_p75": float("nan"),
            "weight_p90": float("nan"),
            "weight_p95": float("nan"),
            "weight_p99": float("nan"),
            "weight_max": float("nan"),
            "weight_mean": float("nan"),
        }

    n = len(w)
    n_positive = int((w > 0).sum())
    zero_prop = float((w == 0).mean())

    w_sum = float(w.sum())
    w_sq_sum = float((w ** 2).sum())
    ess = (w_sum ** 2) / w_sq_sum if w_sq_sum > 0 else float("nan")

    qs = np.percentile(w, [0, 25, 50, 75, 90, 95, 99, 100])

    return {
        "n_trajectories": int(n),
        "n_positive_weight": n_positive,
        "zero_weight_prop": zero_prop,
        "ess": ess,
        "weight_min": float(qs[0]),
        "weight_p25": float(qs[1]),
        "weight_p50": float(qs[2]),
        "weight_p75": float(qs[3]),
        "weight_p90": float(qs[4]),
        "weight_p95": float(qs[5]),
        "weight_p99": float(qs[6]),
        "weight_max": float(qs[7]),
        "weight_mean": float(w.mean()),
    }


def _group_cum_regime_weight(g: pd.DataFrame) -> pd.Series:
    """
    Per-trajectory cumulative regime weight:
    product of stabilised weights until first deviation from policy,
    then zero thereafter.
    """
    swt = g["sw_t"].to_numpy(dtype=float)
    adher = g["adherent"].to_numpy(dtype=int)

    out = np.zeros(len(g), dtype=float)
    cum = 1.0
    active = True

    for i, (s, a) in enumerate(zip(swt, adher)):
        if not active:
            out[i] = 0.0
            continue
        if a == 0:
            active = False
            out[i] = 0.0
            continue
        cum *= float(s)
        out[i] = cum

    return pd.Series(out, index=g.index, dtype=float)


def evaluate_policy_value_ipw(
    df: pd.DataFrame,
    policy_name: str,
    policy_fn,
    X_cols_den: list[str],
    X_cols_num: list[str],
    traj_keys: list[str] | None = None,
    clip_cum_w: float | None = None,
    verbose: bool = True,
) -> tuple[float, float, pd.DataFrame, dict]:
    """
    Uses the predefined split:
    - fit propensity models on TRAIN decision-eligible rows
    - evaluate regime on TEST trajectories only

    traj_keys:
      grouping keys defining a trajectory.
      In standard evaluation this is TRAJ_KEYS.
      In bootstrap evaluation this should be TRAJ_KEYS + ['_boot_rep'].

    Returns:
    - policy_value_ipw
    - observed_practice_value
    - trajectory-level weighted summary dataframe
    - weight_summary dict
    """
    if traj_keys is None:
        traj_keys = TRAJ_KEYS

    if clip_cum_w is None:
        clip_cum_w = CLIP_CUM_W

    if verbose:
        print(f"[STEP4] {policy_name}: building decision subset", flush=True)

    dec = df[df["decision_eligible"] == 1].copy()
    dec = dec.sort_values(traj_keys + [START_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    if verbose:
        print(f"[STEP4] {policy_name}: decision rows = {len(dec)}", flush=True)

    dec_train = dec[dec[SPLIT_COL] == "train"].copy()
    dec_test = dec[dec[SPLIT_COL] == "test"].copy()

    if verbose:
        print(f"[STEP4] {policy_name}: train decision rows = {len(dec_train)}", flush=True)
        print(f"[STEP4] {policy_name}: test decision rows = {len(dec_test)}", flush=True)
        print(f"[STEP4] {policy_name}: fitting propensity models on TRAIN split...", flush=True)

    pipe_den, pipe_num = fit_propensity_models(dec_train, X_cols_den, X_cols_num)

    if verbose:
        print(f"[STEP4] {policy_name}: propensity models fitted", flush=True)
        print(f"[STEP4] {policy_name}: predicting probabilities on TEST split...", flush=True)

    p_den = predict_pA(pipe_den, dec_test[X_cols_den].to_numpy(dtype=float))
    p_num = predict_pA(pipe_num, dec_test[X_cols_num].to_numpy(dtype=float))

    A = dec_test[A_COL].astype(int).to_numpy()
    pA_den = np.where(A == 1, p_den, 1.0 - p_den)
    pA_num = np.where(A == 1, p_num, 1.0 - p_num)

    dec_test["sw_t"] = np.clip(pA_num / pA_den, EPS, np.inf)
    dec_test["pi_action"] = dec_test.apply(policy_fn, axis=1).astype(int)
    dec_test["adherent"] = (dec_test[A_COL].astype(int) == dec_test["pi_action"]).astype(int)

    if verbose:
        print(f"[STEP4] {policy_name}: computing cumulative regime weights...", flush=True)

    w_parts = []
    for _, g in dec_test.groupby(traj_keys, sort=False):
        w_parts.append(_group_cum_regime_weight(g))

    dec_test["W_cum"] = pd.concat(w_parts).sort_index()

    if clip_cum_w is not None:
        dec_test["W_cum"] = dec_test["W_cum"].clip(upper=float(clip_cum_w))

    w_end = (
        dec_test.groupby(traj_keys, sort=False)["W_cum"]
        .last()
        .rename("W_end")
        .reset_index()
    )

    if verbose:
        print(f"[STEP4] {policy_name}: building TEST trajectory utility...", flush=True)

    df_test = df[df[SPLIT_COL] == "test"].copy()
    util = build_trajectory_utility(df_test, traj_keys=traj_keys)
    util = util.merge(w_end, on=traj_keys, how="left").fillna({"W_end": 0.0})

    weight_summary = summarise_weights(util["W_end"])

    num = float((util["utility"] * util["W_end"]).sum())
    den = float(util["W_end"].sum())
    policy_value = num / den if den > 0 else float("nan")

    observed_value = float(util["utility"].mean())

    util["policy"] = policy_name
    util["split"] = "test"

    if verbose:
        print(f"[STEP4] {policy_name}: IPW evaluation complete", flush=True)

    return policy_value, observed_value, util, weight_summary


# -----------------------------
# Bootstrap helpers
# -----------------------------
def _resample_test_trajectories_with_rep_ids(
    df_test: pd.DataFrame,
    traj_keys: list[str],
    rng_local: np.random.Generator,
) -> pd.DataFrame:
    groups = [g.copy() for _, g in df_test.groupby(traj_keys, sort=False)]
    n = len(groups)

    sample_indices = rng_local.choice(n, size=n, replace=True)

    parts = []
    for rep, idx in enumerate(sample_indices):
        part = groups[idx].copy().reset_index(drop=True)
        part["_boot_rep"] = rep
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def _attach_train_boot_rep(df_train: pd.DataFrame) -> pd.DataFrame:
    out = df_train.copy()
    out["_boot_rep"] = 0
    return out


def bootstrap_ci(
    df: pd.DataFrame,
    policy_name: str,
    policy_fn,
    X_cols_den: list[str],
    X_cols_num: list[str],
    n_boot: int = 300,
) -> tuple[float, float]:
    """
    Trajectory-level bootstrap over TEST trajectories.
    TRAIN trajectories remain fixed because they define propensity fitting.
    Duplicated sampled TEST trajectories are kept distinct via _boot_rep.
    """
    df_train = df[df[SPLIT_COL] == "train"].copy()
    df_test = df[df[SPLIT_COL] == "test"].copy()

    vals = []

    print(f"[STEP4] {policy_name}: bootstrap started (n={n_boot})", flush=True)
    t0 = time.time()

    for b in range(n_boot):
        rng_local = np.random.default_rng(SEED + 10000 + b)

        df_train_b = _attach_train_boot_rep(df_train)
        df_test_b = _resample_test_trajectories_with_rep_ids(
            df_test=df_test,
            traj_keys=TRAJ_KEYS,
            rng_local=rng_local,
        )

        df_b = pd.concat([df_train_b, df_test_b], ignore_index=True)

        boot_traj_keys = TRAJ_KEYS + ["_boot_rep"]
        df_b = build_decision_eligibility(df_b, traj_keys=boot_traj_keys)

        try:
            v, _, _, _ = evaluate_policy_value_ipw(
                df=df_b,
                policy_name=policy_name,
                policy_fn=policy_fn,
                X_cols_den=X_cols_den,
                X_cols_num=X_cols_num,
                traj_keys=boot_traj_keys,
                clip_cum_w=CLIP_CUM_W,
                verbose=False,
            )
        except Exception:
            v = float("nan")

        vals.append(v)

        if (b + 1) % BOOT_PROGRESS_EVERY == 0 or (b + 1) == n_boot:
            elapsed = time.time() - t0
            per_iter = elapsed / (b + 1)
            remaining = per_iter * (n_boot - (b + 1))
            print(
                f"[STEP4] {policy_name}: bootstrap {b+1}/{n_boot} "
                f"| elapsed {elapsed:.1f}s "
                f"| est remaining {remaining:.1f}s",
                flush=True,
            )

    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) < max(30, n_boot // 5):
        return float("nan"), float("nan")

    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"[STEP4] {policy_name}: bootstrap complete", flush=True)
    return float(lo), float(hi)


def clipping_sensitivity(
    df: pd.DataFrame,
    policy_name: str,
    policy_fn,
    X_cols_den: list[str],
    X_cols_num: list[str],
    traj_keys: list[str],
    clip_values: list[float | None],
) -> pd.DataFrame:
    """
    Evaluate one policy under a range of cumulative-weight clipping thresholds.
    """
    rows = []

    for clip_val in clip_values:
        val, obs, _, weight_summary = evaluate_policy_value_ipw(
            df=df,
            policy_name=policy_name,
            policy_fn=policy_fn,
            X_cols_den=X_cols_den,
            X_cols_num=X_cols_num,
            traj_keys=traj_keys,
            clip_cum_w=clip_val,
            verbose=False,
        )

        rows.append({
            "policy": policy_name,
            "clip_cum_w": "none" if clip_val is None else float(clip_val),
            "policy_value_ipw": val,
            "observed_practice_value": obs,
            "delta_policy_minus_observed": (
                val - obs if np.isfinite(val) and np.isfinite(obs) else float("nan")
            ),
            "zero_weight_prop": weight_summary["zero_weight_prop"],
            "ess": weight_summary["ess"],
            "weight_p95": weight_summary["weight_p95"],
            "weight_p99": weight_summary["weight_p99"],
            "weight_max": weight_summary["weight_max"],
        })

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t_all0 = time.time()

    print("[STEP4] starting", flush=True)
    print("[STEP4] reading file", flush=True)

    df = pd.read_csv(INFILE, low_memory=False)
    df = validate_split(df)

    print("[STEP4] file loaded", flush=True)

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[START_COL] = pd.to_datetime(df[START_COL], errors="coerce")
    df[INSERTED_COL] = pd.to_datetime(df[INSERTED_COL], errors="coerce")
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[PERIODS_COL] = pd.to_numeric(df[PERIODS_COL], errors="coerce")
    df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")
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

    cols_to_numeric = feature_cols(df) + numerator_cols() + [A_COL, CAUTI_TODAY, REINS_TODAY]
    cols_to_numeric = list(dict.fromkeys(cols_to_numeric))

    for c in cols_to_numeric:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            df[c] = df[c].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                "true": 1, "false": 0,
            })
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    print("[STEP4] split summary", flush=True)
    print(df[SPLIT_COL].value_counts(dropna=False), flush=True)
    print(
        f"[STEP4] unique patients: "
        f"train={df.loc[df[SPLIT_COL] == 'train', ID_COL].nunique()} "
        f"test={df.loc[df[SPLIT_COL] == 'test', ID_COL].nunique()}",
        flush=True,
    )

    df = build_decision_eligibility(df, traj_keys=TRAJ_KEYS)

    print("[STEP4] decision eligibility built", flush=True)
    print("[STEP4] decision eligible rows:", int(df["decision_eligible"].sum()), flush=True)
    print(
        f"[STEP4] decision eligible train rows: "
        f"{int(((df['decision_eligible'] == 1) & (df[SPLIT_COL] == 'train')).sum())}",
        flush=True,
    )
    print(
        f"[STEP4] decision eligible test rows: "
        f"{int(((df['decision_eligible'] == 1) & (df[SPLIT_COL] == 'test')).sum())}",
        flush=True,
    )

    X_cols_den = feature_cols(df)
    X_cols_num = numerator_cols()

    print(f"[STEP4] denominator feature cols: {len(X_cols_den)}", flush=True)
    print(f"[STEP4] numerator feature cols: {X_cols_num}", flush=True)
    print(f"[STEP4] evaluating {len(POLICIES)} policies", flush=True)

    results = []

    for policy_idx, (policy_name, policy_fn) in enumerate(POLICIES, start=1):
        t_policy0 = time.time()
        print(f"\n[STEP4] Starting policy {policy_idx}/{len(POLICIES)}: {policy_name}", flush=True)

        policy_value, observed_value, util_df, weight_summary = evaluate_policy_value_ipw(
            df=df,
            policy_name=policy_name,
            policy_fn=policy_fn,
            X_cols_den=X_cols_den,
            X_cols_num=X_cols_num,
            traj_keys=TRAJ_KEYS,
            clip_cum_w=CLIP_CUM_W,
            verbose=True,
        )

        lo, hi = bootstrap_ci(
            df=df,
            policy_name=policy_name,
            policy_fn=policy_fn,
            X_cols_den=X_cols_den,
            X_cols_num=X_cols_num,
            n_boot=BOOTSTRAP,
        )

        results.append({
            "policy": policy_name,
            "split_used": "test",
            "propensity_fit_split": "train",
            "method": "stabilised_ipw",
            "policy_value_ipw": policy_value,
            "policy_value_ipw_CI_low": lo,
            "policy_value_ipw_CI_high": hi,
            "observed_practice_value": observed_value,
            "delta_policy_minus_observed": (
                policy_value - observed_value if np.isfinite(policy_value) else float("nan")
            ),
            "weights": json.dumps({
                "W_CAUTI": W_CAUTI,
                "W_REINS": W_REINS,
                "W_CATH_DAYS": W_CATH_DAYS,
            }),
            "n_trajectories": weight_summary["n_trajectories"],
            "n_positive_weight": weight_summary["n_positive_weight"],
            "zero_weight_prop": weight_summary["zero_weight_prop"],
            "ess": weight_summary["ess"],
            "weight_min": weight_summary["weight_min"],
            "weight_p25": weight_summary["weight_p25"],
            "weight_p50": weight_summary["weight_p50"],
            "weight_p75": weight_summary["weight_p75"],
            "weight_p90": weight_summary["weight_p90"],
            "weight_p95": weight_summary["weight_p95"],
            "weight_p99": weight_summary["weight_p99"],
            "weight_max": weight_summary["weight_max"],
            "weight_mean": weight_summary["weight_mean"],
        })

        out_patient = OUTDIR / f"step4_patient_summary__{policy_name}.csv"
        util_df.to_csv(out_patient, index=False, float_format="%.6f")

        if policy_name in SENSITIVITY_POLICIES:
            sens_df = clipping_sensitivity(
                df=df,
                policy_name=policy_name,
                policy_fn=policy_fn,
                X_cols_den=X_cols_den,
                X_cols_num=X_cols_num,
                traj_keys=TRAJ_KEYS,
                clip_values=SENSITIVITY_CLIPS,
            )
            out_sens = OUTDIR / f"step4_clipping_sensitivity__{policy_name}.csv"
            sens_df.to_csv(out_sens, index=False, float_format="%.6f")
            print(f"[STEP4] {policy_name}: clipping sensitivity -> {out_sens.resolve()}", flush=True)

        print(
            f"[STEP4] {policy_name}: "
            f"IPW policy value={policy_value:.4f} "
            f"(observed test mean={observed_value:.4f}) "
            f"| ESS={weight_summary['ess']:.2f} "
            f"| zero_w={weight_summary['zero_weight_prop']:.3f}",
            flush=True,
        )
        print(f"[STEP4] {policy_name}: patient summary -> {out_patient.resolve()}", flush=True)
        print(f"[STEP4] Finished {policy_name} in {time.time() - t_policy0:.1f}s", flush=True)

    out_res = OUTDIR / "step4_policy_value_estimates.csv"
    pd.DataFrame(results).to_csv(out_res, index=False, float_format="%.6f")

    meta = {
        "seed": SEED,
        "bootstrap": BOOTSTRAP,
        "clip_cum_weight": CLIP_CUM_W,
        "sensitivity_policies": sorted(SENSITIVITY_POLICIES),
        "sensitivity_clips": [("none" if x is None else x) for x in SENSITIVITY_CLIPS],
        "split_usage": {
            "source": "predefined split from Step-0 carried via Step-1 scored panel",
            "propensity_fit_split": "train",
            "policy_evaluation_split": "test",
        },
        "trajectory_keys": TRAJ_KEYS,
        "utility_weights": {
            "W_CAUTI": W_CAUTI,
            "W_REINS": W_REINS,
            "W_CATH_DAYS": W_CATH_DAYS,
        },
        "panel_file": str(INFILE),
        "method_note": (
            "Step 4 is the standard causal layer only: stabilised inverse-probability weighting "
            "for dynamic regime evaluation. Doubly robust evaluation is handled separately in Step 5."
        ),
        "note": (
            "IPW regime evaluation aligned to the current interval-based catheter panel. "
            "Propensity models are trained on the predefined TRAIN split and policy value is evaluated "
            "on the predefined TEST split. Trajectories are defined at the catheter-episode level "
            "using subject_id, hadm_id, stay_id, and inserted. This version also writes trajectory-weight "
            "diagnostics (ESS, zero-weight proportion, quantiles) and clipping-sensitivity outputs "
            "for selected policies."
        ),
    }
    (OUTDIR / "step4_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[STEP4] complete", flush=True)
    print(f"[STEP4] policy value estimates: {out_res.resolve()}", flush=True)
    print(f"[STEP4] config: {(OUTDIR / 'step4_config.json').resolve()}", flush=True)
    print(f"[STEP4] total elapsed: {time.time() - t_all0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
