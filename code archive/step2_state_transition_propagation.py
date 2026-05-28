from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Config
# -----------------------------
INDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step1")
OUTDIR = Path(r"C:\Users\DavidUni\OneDrive - University of Reading\repos\mimic_pipeline\artifacts\step2")
MODEL_DIR = INDIR

OUTDIR.mkdir(exist_ok=True, parents=True)

INFILE = INDIR / "step1_scored_panel.csv"
MODEL_FILE = MODEL_DIR / "transition_models.pkl"

TRAJ_KEYS = ["subject_id", "hadm_id", "stay_id", "inserted"]
STATE_COL = "catheter_state"
TIME_COL = "episode_index"
START_COL = "period_start"
END_COL = "period_end"
INSERTED_COL = "inserted"
PERIODS_COL = "periods_in_state"
INTERVAL_HOURS_COL = "interval_hours"
SPLIT_COL = "split"
PERIOD_HOURS = 24

EPS = 1e-12

# Utility weights
W_CAUTI = 10.0
W_REINS = 3.0
W_CATH_DAYS = 1.0

# Threshold sweep for risk policy frontier
RISK_TAUS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


# -----------------------------
# Policies
# -----------------------------
def policy_fixed_period_remove(periods_in_state: int, remove_period: int = 3) -> int:
    return 1 if periods_in_state >= remove_period else 0


def policy_risk_threshold(p_cauti_keep: float, tau: float = 0.20) -> int:
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
    ("fixed_period3", lambda row: policy_fixed_period_remove(int(row[PERIODS_COL]), remove_period=3)),
    ("fixed_period5", lambda row: policy_fixed_period_remove(int(row[PERIODS_COL]), remove_period=5)),
    ("fixed_period7", lambda row: policy_fixed_period_remove(int(row[PERIODS_COL]), remove_period=7)),
    *[
        (
            f"risk_tau_{tau:.2f}".replace(".", "_"),
            lambda row, tau=tau: policy_risk_threshold(float(row["p_cauti_if_keep"]), tau=tau),
        )
        for tau in RISK_TAUS
    ],
    (
        "hybrid_tau0_15_ratio1_0",
        lambda row: policy_hybrid(
            float(row["p_cauti_if_keep"]),
            float(row["p_reins_if_remove"]),
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
        print("Warning: no train rows found in Step-1 scored panel.", flush=True)

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


def _predict_proba(pipe, X: pd.DataFrame) -> float:
    return float(pipe.predict_proba(X.to_numpy(dtype=float))[:, 1][0])


# -----------------------------
# Step-1 rescoring helpers
# -----------------------------
def _build_feature_frame(
    anchor_row: pd.Series,
    x_cols: list[str],
    time_value: int,
    periods_value: int,
    state_is_out: int | None = None,
) -> pd.DataFrame:
    data: dict[str, float] = {}

    for c in x_cols:
        if c == TIME_COL:
            data[c] = float(time_value)
        elif c == PERIODS_COL:
            data[c] = float(periods_value)
        elif c == "state_is_out":
            if state_is_out is None:
                raise ValueError("state_is_out was requested but not supplied.")
            data[c] = float(state_is_out)
        else:
            data[c] = float(anchor_row.get(c, 0.0))

    return pd.DataFrame([data], columns=x_cols)


def score_in_state(
    anchor_row: pd.Series,
    time_value: int,
    in_period: int,
    cauti_model,
    reins_model,
    x_cols_cauti: list[str],
    x_cols_reins: list[str],
) -> tuple[float, float, float]:
    """
    Returns:
    - p_cauti_if_keep
    - p_cauti_if_remove   (first OUT interval after removal)
    - p_reins_if_remove   (first OUT interval after removal)
    """
    X_keep = _build_feature_frame(
        anchor_row=anchor_row,
        x_cols=x_cols_cauti,
        time_value=time_value,
        periods_value=in_period,
        state_is_out=0,
    )
    p_cauti_if_keep = _clip01(_predict_proba(cauti_model, X_keep))

    X_remove_cauti = _build_feature_frame(
        anchor_row=anchor_row,
        x_cols=x_cols_cauti,
        time_value=time_value,
        periods_value=1,
        state_is_out=1,
    )
    p_cauti_if_remove = _clip01(_predict_proba(cauti_model, X_remove_cauti))

    X_remove_reins = _build_feature_frame(
        anchor_row=anchor_row,
        x_cols=x_cols_reins,
        time_value=time_value,
        periods_value=1,
        state_is_out=None,
    )
    p_reins_if_remove = _clip01(_predict_proba(reins_model, X_remove_reins))

    return p_cauti_if_keep, p_cauti_if_remove, p_reins_if_remove


def score_out_state(
    anchor_row: pd.Series,
    time_value: int,
    out_period: int,
    cauti_model,
    reins_model,
    x_cols_cauti: list[str],
    x_cols_reins: list[str],
) -> tuple[float, float]:
    """
    Returns:
    - p_cauti_if_out
    - p_reins_if_out
    """
    X_out_cauti = _build_feature_frame(
        anchor_row=anchor_row,
        x_cols=x_cols_cauti,
        time_value=time_value,
        periods_value=out_period,
        state_is_out=1,
    )
    p_cauti_if_out = _clip01(_predict_proba(cauti_model, X_out_cauti))

    X_out_reins = _build_feature_frame(
        anchor_row=anchor_row,
        x_cols=x_cols_reins,
        time_value=time_value,
        periods_value=out_period,
        state_is_out=None,
    )
    p_reins_if_out = _clip01(_predict_proba(reins_model, X_out_reins))

    return p_cauti_if_out, p_reins_if_out


# -----------------------------
# Counterfactual interval helpers
# -----------------------------
def _get_horizon(g: pd.DataFrame) -> pd.Timestamp:
    if END_COL in g.columns:
        return pd.Timestamp(g[END_COL].iloc[-1])
    last_start = pd.Timestamp(g[START_COL].iloc[-1])
    last_hours = float(g[INTERVAL_HOURS_COL].iloc[-1])
    return last_start + pd.Timedelta(hours=last_hours)


def _get_anchor_row(g: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """
    Last observed row whose start time is <= t.
    This provides the latest observed covariate snapshot available at time t.
    """
    mask = g[START_COL] <= t
    if mask.any():
        return g.loc[mask].iloc[-1]
    return g.iloc[0]


def _combine_nodes(nodes: list[dict]) -> list[dict]:
    """
    Combine nodes with the same state and period-in-state.
    """
    agg: dict[tuple[str, int], float] = {}

    for node in nodes:
        mass = float(node["mass"])
        if mass <= EPS:
            continue
        key = (str(node["state"]), int(node["periods_in_state"]))
        agg[key] = agg.get(key, 0.0) + mass

    out = [
        {"state": state, "periods_in_state": period, "mass": mass}
        for (state, period), mass in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1]))
        if mass > EPS
    ]
    return out


# -----------------------------
# Proper counterfactual propagation
# -----------------------------
def propagate_trajectory_counterfactual(
    g: pd.DataFrame,
    policy_fn,
    cauti_model,
    reins_model,
    x_cols_cauti: list[str],
    x_cols_reins: list[str],
    period_hours: int,
) -> pd.DataFrame:
    """
    Proper state/timing propagation within the current Step-1-only setup.

    Key differences from the old lighter version:
    - does NOT walk the observed state row-by-row
    - rebuilds counterfactual state over time
    - rescores each counterfactual interval using the saved Step-1 models

    Remaining approximation:
    - time-varying covariates are carried forward from the latest observed row
      available at the start of each counterfactual interval
    """
    g = g.sort_values([START_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    t = pd.Timestamp(g[START_COL].iloc[0])
    horizon = _get_horizon(g)

    first_state = str(g[STATE_COL].iloc[0]).strip().lower()
    first_period = int(g[PERIODS_COL].iloc[0])

    active_nodes = [
        {
            "state": first_state,
            "periods_in_state": first_period,
            "mass": 1.0,
        }
    ]

    rows = []
    cf_time_index = 0

    cif_cauti = 0.0
    cum_reins = 0.0
    cum_removals = 0.0
    exp_in_days = 0.0
    exp_out_days = 0.0

    while (t < horizon) and active_nodes:
        next_t = min(t + pd.Timedelta(hours=period_hours), horizon)
        interval_hours = (next_t - t).total_seconds() / 3600.0
        interval_days = interval_hours / 24.0

        anchor_row = _get_anchor_row(g, t)
        anchor_state = str(anchor_row[STATE_COL]).strip().lower()

        p_surv_start = float(sum(node["mass"] for node in active_nodes))
        p_in_start = float(sum(node["mass"] for node in active_nodes if node["state"] == "in"))
        p_out_start = float(sum(node["mass"] for node in active_nodes if node["state"] == "out"))

        p_cauti_interval = 0.0
        p_reins_interval = 0.0
        removals_interval = 0.0

        next_nodes: list[dict] = []

        for node in active_nodes:
            state = str(node["state"])
            periods_in_state = int(node["periods_in_state"])
            mass = float(node["mass"])

            if mass <= EPS:
                continue

            # -------------------------
            # IN-state mass
            # -------------------------
            if state == "in":
                p_keep, p_cauti_remove, p_reins_remove = score_in_state(
                    anchor_row=anchor_row,
                    time_value=cf_time_index,
                    in_period=periods_in_state,
                    cauti_model=cauti_model,
                    reins_model=reins_model,
                    x_cols_cauti=x_cols_cauti,
                    x_cols_reins=x_cols_reins,
                )

                policy_ctx = pd.Series(
                    {
                        PERIODS_COL: periods_in_state,
                        "p_cauti_if_keep": p_keep,
                        "p_cauti_if_remove": p_cauti_remove,
                        "p_reins_if_remove": p_reins_remove,
                    }
                )
                a_remove = int(policy_fn(policy_ctx))

                if a_remove == 0:
                    # Stay IN for this interval
                    cauti_mass = mass * p_keep
                    survive_mass = mass - cauti_mass

                    p_cauti_interval += cauti_mass
                    exp_in_days += mass * interval_days

                    if survive_mass > EPS:
                        next_nodes.append(
                            {
                                "state": "in",
                                "periods_in_state": periods_in_state + 1,
                                "mass": survive_mass,
                            }
                        )

                else:
                    # Remove at the start of the interval.
                    # This interval is now an OUT day-1 interval.
                    removals_interval += mass
                    cum_removals += mass
                    exp_out_days += mass * interval_days

                    p_cauti_r, p_reins_r = _resolve_competing(
                        p_cauti=p_cauti_remove,
                        p_other=p_reins_remove,
                    )

                    cauti_mass = mass * p_cauti_r
                    reins_mass = mass * p_reins_r
                    out_survive_mass = mass - cauti_mass - reins_mass

                    p_cauti_interval += cauti_mass
                    p_reins_interval += reins_mass

                    if reins_mass > EPS:
                        next_nodes.append(
                            {
                                "state": "in",
                                "periods_in_state": 1,
                                "mass": reins_mass,
                            }
                        )

                    if out_survive_mass > EPS:
                        next_nodes.append(
                            {
                                "state": "out",
                                "periods_in_state": 2,
                                "mass": out_survive_mass,
                            }
                        )

            # -------------------------
            # OUT-state mass
            # -------------------------
            else:
                p_cauti_out, p_reins_out = score_out_state(
                    anchor_row=anchor_row,
                    time_value=cf_time_index,
                    out_period=periods_in_state,
                    cauti_model=cauti_model,
                    reins_model=reins_model,
                    x_cols_cauti=x_cols_cauti,
                    x_cols_reins=x_cols_reins,
                )

                p_cauti_o, p_reins_o = _resolve_competing(
                    p_cauti=p_cauti_out,
                    p_other=p_reins_out,
                )

                cauti_mass = mass * p_cauti_o
                reins_mass = mass * p_reins_o
                out_survive_mass = mass - cauti_mass - reins_mass

                p_cauti_interval += cauti_mass
                p_reins_interval += reins_mass
                exp_out_days += mass * interval_days

                if reins_mass > EPS:
                    next_nodes.append(
                        {
                            "state": "in",
                            "periods_in_state": 1,
                            "mass": reins_mass,
                        }
                    )

                if out_survive_mass > EPS:
                    next_nodes.append(
                        {
                            "state": "out",
                            "periods_in_state": periods_in_state + 1,
                            "mass": out_survive_mass,
                        }
                    )

        cif_cauti += p_cauti_interval
        cum_reins += p_reins_interval

        p_surv_end = float(sum(node["mass"] for node in next_nodes))

        rows.append(
            {
                "period_index": cf_time_index,
                "interval_start": t,
                "interval_end": next_t,
                "interval_hours": float(interval_hours),
                "anchor_row_start": anchor_row[START_COL],
                "anchor_observed_state": anchor_state,
                "p_survival_start": p_surv_start,
                "p_in_start": p_in_start,
                "p_out_start": p_out_start,
                "removals_interval": removals_interval,
                "p_cauti_interval": p_cauti_interval,
                "p_reins_interval": p_reins_interval,
                "cif_cauti": cif_cauti,
                "cum_reinsertions": cum_reins,
                "cum_removals": cum_removals,
                "exp_catheter_in_days": exp_in_days,
                "exp_catheter_out_days": exp_out_days,
                "p_survival_end": p_surv_end,
            }
        )

        active_nodes = _combine_nodes(next_nodes)
        t = next_t
        cf_time_index += 1

        if p_surv_end <= EPS:
            break

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    df = pd.read_csv(INFILE, low_memory=False)
    df = validate_split(df)

    bundle = joblib.load(MODEL_FILE)
    cauti_model = bundle["cauti_model"]
    reins_model = bundle["reins_model"]
    x_cols_cauti = bundle["x_cols_cauti"]
    x_cols_reins = bundle["x_cols_reins"]
    feature_cols = bundle["features"]
    period_hours = int(bundle.get("period_hours", PERIOD_HOURS))

    if cauti_model is None or reins_model is None:
        raise ValueError("transition_models.pkl does not contain the required fitted CAUTI/reinsertion models.")

    df[START_COL] = pd.to_datetime(df[START_COL], errors="coerce")
    df[END_COL] = pd.to_datetime(df[END_COL], errors="coerce")
    df[INSERTED_COL] = pd.to_datetime(df[INSERTED_COL], errors="coerce")

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[PERIODS_COL] = pd.to_numeric(df[PERIODS_COL], errors="coerce")
    df[INTERVAL_HOURS_COL] = pd.to_numeric(df[INTERVAL_HOURS_COL], errors="coerce")

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.lower()

    # Coerce the Step-1 feature columns to numeric.
    cols_to_numeric = list(dict.fromkeys(feature_cols + [TIME_COL, PERIODS_COL, INTERVAL_HOURS_COL]))
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
            INTERVAL_HOURS_COL,
            "hadm_id",
            "stay_id",
        ]
    ).copy()

    df[TIME_COL] = df[TIME_COL].astype("int64")
    df[PERIODS_COL] = df[PERIODS_COL].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["stay_id"] = df["stay_id"].astype("int64")

    print("\n--- Step-2 split summary ---", flush=True)
    print(df[SPLIT_COL].value_counts(dropna=False), flush=True)
    print(
        f"Unique patients: train={df.loc[df[SPLIT_COL] == 'train', 'subject_id'].nunique()}, "
        f"test={df.loc[df[SPLIT_COL] == 'test', 'subject_id'].nunique()}",
        flush=True
    )

    # Use only test rows / test trajectories for Step-2 propagation summary
    test_df = df[df[SPLIT_COL] == "test"].copy()

    print(f"Rows entering Step-2 propagation: {len(test_df):,}", flush=True)
    print(
        f"Unique test trajectories entering Step-2: "
        f"{test_df[TRAJ_KEYS].drop_duplicates().shape[0]:,}",
        flush=True
    )

    out_summaries = []

    for policy_name, policy_fn in POLICIES:
        trajectory_rows = []
        trajectory_summ = []

        for _, g in test_df.groupby(TRAJ_KEYS, sort=False):
            traj = propagate_trajectory_counterfactual(
                g=g,
                policy_fn=policy_fn,
                cauti_model=cauti_model,
                reins_model=reins_model,
                x_cols_cauti=x_cols_cauti,
                x_cols_reins=x_cols_reins,
                period_hours=period_hours,
            )

            subj = str(g["subject_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            hadm = int(g["hadm_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            stay = int(g["stay_id"].iloc[0])  # pyright: ignore[reportArgumentType]
            inserted = g[INSERTED_COL].iloc[0]

            if len(traj) == 0:
                continue

            traj.insert(0, "subject_id", subj)
            traj.insert(1, "hadm_id", hadm)
            traj.insert(2, "stay_id", stay)
            traj.insert(3, "inserted", inserted)
            traj.insert(4, "split", "test")
            traj.insert(5, "policy", policy_name)
            trajectory_rows.append(traj)

            last = traj.iloc[-1]
            trajectory_summ.append({
                "subject_id": subj,
                "hadm_id": hadm,
                "stay_id": stay,
                "inserted": inserted,
                "split": "test",
                "policy": policy_name,
                "cif_cauti_end": float(last["cif_cauti"]),
                "cum_reinsertions_end": float(last["cum_reinsertions"]),
                "cum_removals_end": float(last["cum_removals"]),
                "exp_in_days_end": float(last["exp_catheter_in_days"]),
                "exp_out_days_end": float(last["exp_catheter_out_days"]),
                "expected_utility": float(
                    W_CAUTI * last["cif_cauti"]
                    + W_REINS * last["cum_reinsertions"]
                    + W_CATH_DAYS * last["exp_catheter_in_days"]
                ),
            })

        rows_df = pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame()
        summ_df = pd.DataFrame(trajectory_summ)

        out_rows = OUTDIR / f"step2_propagation__{policy_name}.csv"
        out_summ = OUTDIR / f"step2_patient_summary__{policy_name}.csv"

        rows_df.to_csv(out_rows, index=False, float_format="%.6f")
        summ_df.to_csv(out_summ, index=False, float_format="%.6f")

        if len(summ_df) > 0:
            out_summaries.append({
                "policy": policy_name,
                "n_trajectories": int(len(summ_df)),
                "split_used": "test",
                "mean_expected_utility": float(summ_df["expected_utility"].mean()),
                "mean_cif_cauti": float(summ_df["cif_cauti_end"].mean()),
                "mean_reinsertions": float(summ_df["cum_reinsertions_end"].mean()),
                "mean_in_days": float(summ_df["exp_in_days_end"].mean()),
                "mean_out_days": float(summ_df["exp_out_days_end"].mean()),
            })

        print(f"Wrote Step-2 outputs for policy={policy_name}", flush=True)

    pd.DataFrame(out_summaries).to_csv(
        OUTDIR / "step2_policy_level_summary.csv",
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    main()
