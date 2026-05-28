from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Config
# -----------------------------
SEED = 42
N_ROLLOUTS = 200
PROGRESS_EVERY = 100
EPS = 1e-12

INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step1")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step3")

OUTDIR.mkdir(exist_ok=True, parents=True)

PANEL_FILE = INDIR / "step1_scored_panel.csv"
MODEL_FILE = INDIR / "transition_models.pkl"

TRAJ_KEYS = ["subject_id", "hadm_id", "stay_id", "inserted"]

TIME_COL = "episode_index"
START_COL = "period_start"
END_COL = "period_end"
INSERTED_COL = "inserted"
STATE_COL = "catheter_state"
PERIODS_COL = "periods_in_state"
INTERVAL_COL = "interval_hours"
SPLIT_COL = "split"

# Utility weights
W_CAUTI = 10.0
W_REINS = 3.0
W_CATH_DAYS = 1.0

# -----------------------------
# Same policies as Step-2
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


def _predict_proba(pipe, feature_array: np.ndarray) -> float:
    return float(pipe.predict_proba(feature_array)[:, 1][0])


# -----------------------------
# Step-1 rescoring helpers
# -----------------------------
def _build_feature_array(
    base_vec: np.ndarray,
    col_idx: dict[str, int],
    time_value: int,
    periods_value: int,
    state_is_out: int | None = None,
) -> np.ndarray:
    feature_vec = base_vec.copy()

    if TIME_COL in col_idx:
        feature_vec[col_idx[TIME_COL]] = float(time_value)
    if PERIODS_COL in col_idx:
        feature_vec[col_idx[PERIODS_COL]] = float(periods_value)
    if "state_is_out" in col_idx:
        if state_is_out is None:
            raise ValueError("state_is_out was requested but not supplied.")
        feature_vec[col_idx["state_is_out"]] = float(state_is_out)

    return feature_vec.reshape(1, -1)


def score_in_state(
    base_cauti_vec: np.ndarray,
    base_reins_vec: np.ndarray,
    time_value: int,
    in_period: int,
    cauti_model,
    reins_model,
    cauti_col_idx: dict[str, int],
    reins_col_idx: dict[str, int],
) -> tuple[float, float, float]:
    """
    Returns:
    - p_cauti_if_keep
    - p_cauti_if_remove
    - p_reins_if_remove
    """
    X_keep = _build_feature_array(
        base_vec=base_cauti_vec,
        col_idx=cauti_col_idx,
        time_value=time_value,
        periods_value=in_period,
        state_is_out=0,
    )
    p_cauti_if_keep = _clip01(_predict_proba(cauti_model, X_keep))

    X_remove_cauti = _build_feature_array(
        base_vec=base_cauti_vec,
        col_idx=cauti_col_idx,
        time_value=time_value,
        periods_value=1,
        state_is_out=1,
    )
    p_cauti_if_remove = _clip01(_predict_proba(cauti_model, X_remove_cauti))

    X_remove_reins = _build_feature_array(
        base_vec=base_reins_vec,
        col_idx=reins_col_idx,
        time_value=time_value,
        periods_value=1,
    )
    p_reins_if_remove = _clip01(_predict_proba(reins_model, X_remove_reins))

    return p_cauti_if_keep, p_cauti_if_remove, p_reins_if_remove


def score_out_state(
    base_cauti_vec: np.ndarray,
    base_reins_vec: np.ndarray,
    time_value: int,
    out_period: int,
    cauti_model,
    reins_model,
    cauti_col_idx: dict[str, int],
    reins_col_idx: dict[str, int],
) -> tuple[float, float]:
    """
    Returns:
    - p_cauti_if_out
    - p_reins_if_out
    """
    X_out_cauti = _build_feature_array(
        base_vec=base_cauti_vec,
        col_idx=cauti_col_idx,
        time_value=time_value,
        periods_value=out_period,
        state_is_out=1,
    )
    p_cauti_if_out = _clip01(_predict_proba(cauti_model, X_out_cauti))

    X_out_reins = _build_feature_array(
        base_vec=base_reins_vec,
        col_idx=reins_col_idx,
        time_value=time_value,
        periods_value=out_period,
    )
    p_reins_if_out = _clip01(_predict_proba(reins_model, X_out_reins))

    return p_cauti_if_out, p_reins_if_out


# -----------------------------
# Counterfactual interval helpers
# -----------------------------
def _get_horizon(g_sorted: pd.DataFrame) -> pd.Timestamp:
    if END_COL in g_sorted.columns:
        return pd.Timestamp(g_sorted[END_COL].iloc[-1])
    last_start = pd.Timestamp(g_sorted[START_COL].iloc[-1])
    last_hours = float(g_sorted[INTERVAL_COL].iloc[-1])
    return last_start + pd.Timedelta(hours=last_hours)


def prepare_trajectory_sim_data(
    g: pd.DataFrame,
    x_cols_cauti: list[str],
    x_cols_reins: list[str],
) -> dict[str, object]:
    # Sort once and precompute the numeric feature vectors reused across rollouts.
    g_sorted = g.sort_values([START_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    start_times = [pd.Timestamp(ts) for ts in g_sorted[START_COL]]
    cauti_col_idx = {col: idx for idx, col in enumerate(x_cols_cauti)}
    reins_col_idx = {col: idx for idx, col in enumerate(x_cols_reins)}

    base_cauti: list[np.ndarray] = []
    base_reins: list[np.ndarray] = []

    for _, row in g_sorted.iterrows():
        base_cauti.append(np.array([float(row.get(col, 0.0)) for col in x_cols_cauti], dtype=float))
        base_reins.append(np.array([float(row.get(col, 0.0)) for col in x_cols_reins], dtype=float))

    return {
        "g_sorted": g_sorted,
        "start_times": start_times,
        "horizon": _get_horizon(g_sorted),
        "initial_state": str(g_sorted[STATE_COL].iloc[0]).strip().lower(),
        "initial_periods_in_state": int(g_sorted[PERIODS_COL].iloc[0]),
        "base_cauti": base_cauti,
        "base_reins": base_reins,
        "cauti_col_idx": cauti_col_idx,
        "reins_col_idx": reins_col_idx,
    }


# -----------------------------
# Proper counterfactual rollout
# -----------------------------
def simulate_rollout_counterfactual(
    traj_data: dict[str, object],
    policy_fn,
    cauti_model,
    reins_model,
    period_hours: int,
    score_cache: dict[tuple[object, ...], tuple[float, ...]],
    rng: np.random.Generator,
) -> tuple[int, int, int, float, float, float]:
    """
    Proper stochastic rollout within the current Step-1-only setup.

    Key differences from the old lighter version:
    - does NOT walk the observed state row-by-row
    - rebuilds counterfactual state over time
    - rescores each counterfactual interval using the saved Step-1 models

    Remaining approximation:
    - time-varying covariates are carried forward from the latest observed row
      available at the start of each counterfactual interval
    """
    g_sorted = traj_data["g_sorted"]
    start_times = traj_data["start_times"]
    horizon = pd.Timestamp(traj_data["horizon"])
    state = str(traj_data["initial_state"])
    periods_in_state = int(traj_data["initial_periods_in_state"])
    base_cauti = traj_data["base_cauti"]
    base_reins = traj_data["base_reins"]
    cauti_col_idx = traj_data["cauti_col_idx"]
    reins_col_idx = traj_data["reins_col_idx"]

    t = pd.Timestamp(start_times[0])
    cf_time_index = 0
    anchor_idx = 0

    cauti = 0
    reins = 0
    removals = 0
    in_days = 0.0
    out_days = 0.0

    while t < horizon:
        next_t = min(t + pd.Timedelta(hours=period_hours), horizon)
        interval_hours = (next_t - t).total_seconds() / 3600.0
        interval_days = interval_hours / 24.0

        while anchor_idx + 1 < len(start_times) and start_times[anchor_idx + 1] <= t:
            anchor_idx += 1

        # -------------------------
        # IN-state rollout
        # -------------------------
        if state == "in":
            cache_key = ("in", anchor_idx, cf_time_index, periods_in_state)
            if cache_key in score_cache:
                p_keep, p_cauti_remove, p_reins_remove = score_cache[cache_key]
            else:
                p_keep, p_cauti_remove, p_reins_remove = score_in_state(
                    base_cauti_vec=base_cauti[anchor_idx],
                    base_reins_vec=base_reins[anchor_idx],
                    time_value=cf_time_index,
                    in_period=periods_in_state,
                    cauti_model=cauti_model,
                    reins_model=reins_model,
                    cauti_col_idx=cauti_col_idx,
                    reins_col_idx=reins_col_idx,
                )
                score_cache[cache_key] = (p_keep, p_cauti_remove, p_reins_remove)

            policy_ctx = {
                PERIODS_COL: periods_in_state,
                "p_cauti_if_keep": p_keep,
                "p_cauti_if_remove": p_cauti_remove,
                "p_reins_if_remove": p_reins_remove,
            }
            action = int(policy_fn(policy_ctx))

            if action == 0:
                in_days += interval_days

                if rng.uniform() < p_keep:
                    cauti = 1
                    break

                state = "in"
                periods_in_state += 1

            else:
                removals += 1
                out_days += interval_days

                outcome = _sample_competing_event(
                    p_cauti=p_cauti_remove,
                    p_other=p_reins_remove,
                    rng=rng,
                )

                if outcome == "cauti":
                    cauti = 1
                    break
                elif outcome == "other":
                    reins += 1
                    state = "in"
                    periods_in_state = 1
                else:
                    state = "out"
                    periods_in_state = 2

        # -------------------------
        # OUT-state rollout
        # -------------------------
        else:
            cache_key = ("out", anchor_idx, cf_time_index, periods_in_state)
            if cache_key in score_cache:
                p_cauti_out, p_reins_out = score_cache[cache_key]
            else:
                p_cauti_out, p_reins_out = score_out_state(
                    base_cauti_vec=base_cauti[anchor_idx],
                    base_reins_vec=base_reins[anchor_idx],
                    time_value=cf_time_index,
                    out_period=periods_in_state,
                    cauti_model=cauti_model,
                    reins_model=reins_model,
                    cauti_col_idx=cauti_col_idx,
                    reins_col_idx=reins_col_idx,
                )
                score_cache[cache_key] = (p_cauti_out, p_reins_out)

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
                periods_in_state = 1
            else:
                state = "out"
                periods_in_state += 1

        t = next_t
        cf_time_index += 1

    utility = W_CAUTI * cauti + W_REINS * reins + W_CATH_DAYS * in_days
    return cauti, reins, removals, in_days, out_days, utility


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t_all0 = time.time()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(PANEL_FILE, low_memory=False)
    df = validate_split(df)

    bundle = joblib.load(MODEL_FILE)
    cauti_model = bundle["cauti_model"]
    reins_model = bundle["reins_model"]
    x_cols_cauti = bundle["x_cols_cauti"]
    x_cols_reins = bundle["x_cols_reins"]
    feature_cols = bundle["features"]
    period_hours = int(bundle["period_hours"])

    if cauti_model is None or reins_model is None:
        raise ValueError("transition_models.pkl does not contain the required fitted CAUTI/reinsertion models.")

    df[START_COL] = pd.to_datetime(df[START_COL], errors="coerce")
    df[END_COL] = pd.to_datetime(df[END_COL], errors="coerce")
    df[INSERTED_COL] = pd.to_datetime(df[INSERTED_COL], errors="coerce")

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[PERIODS_COL] = pd.to_numeric(df[PERIODS_COL], errors="coerce")
    df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()

    cols_to_numeric = list(dict.fromkeys(feature_cols + [TIME_COL, PERIODS_COL, INTERVAL_COL]))
    for c in cols_to_numeric:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            df[c] = df[c].replace(
                {
                    "TRUE": 1, "FALSE": 0,
                    "True": 1, "False": 0,
                    "true": 1, "false": 0,
                }
            )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.dropna(
        subset=[
            START_COL,
            END_COL,
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
            subj = str(g["subject_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            hadm = int(g["hadm_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            stay = int(g["stay_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            inserted = g[INSERTED_COL].iloc[0]
            traj_data = prepare_trajectory_sim_data(
                g=g,
                x_cols_cauti=x_cols_cauti,
                x_cols_reins=x_cols_reins,
            )
            score_cache: dict[tuple[object, ...], tuple[float, ...]] = {}

            acc = np.zeros(6, dtype=float)

            for _ in range(N_ROLLOUTS):
                out = simulate_rollout_counterfactual(
                    traj_data=traj_data,
                    policy_fn=policy_fn,
                    cauti_model=cauti_model,
                    reins_model=reins_model,
                    period_hours=period_hours,
                    score_cache=score_cache,
                    rng=rng,
                )
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
            float_format="%.6f",
        )

        if len(patient_df) > 0:
            policy_summary.append({
                "policy": policy_name,
                "split_used": "test",
                "n_trajectories": len(patient_df),
                "cauti_mean": float(patient_df["cauti_mean"].mean()),
                "reinsertions_mean": float(patient_df["reinsertions_mean"].mean()),
                "removals_mean": float(patient_df["removals_mean"].mean()),
                "catheter_in_days_mean": float(patient_df["catheter_in_days_mean"].mean()),
                "catheter_out_days_mean": float(patient_df["catheter_out_days_mean"].mean()),
                "utility_mean": float(patient_df["utility_mean"].mean()),
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
        float_format="%.6f",
    )

    print(f"\n[STEP3] Complete in {time.time() - t_all0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
