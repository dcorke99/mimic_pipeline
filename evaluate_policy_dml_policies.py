"""Recursive Policy-DML evaluation on existing fixed-period panels.

CoR Section 3.5 and Appendix B.1--B.2: fit continuation models backwards,
calculate held-out recursive scores, then average the first score per episode.
OUT rows supply linked outcomes to the last IN decision. Recursive scores
remain unbounded.
"""

import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import build_policy_panels as policy_inputs
import fit_nuisance_models as nuisance
import pipeline_config as run_config
from pipeline_config import BOOTSTRAP_MODES, NUISANCE_MODEL_TYPE
import policy_eval_common as pec
import policy_inference as inference

FOLD = "_crossfit_fold"
EPISODE = "catheter_episode_id"
ROW = "decision_row_id"
KEYS = ["subject_id", "stay_id", "inserted", "episode_index"]
OUTCOME_COLUMNS = {
    "cauti": "cauti_in_period",
    "recatheterisation": "reinsertion_in_period",
    "death": "death_in_period",
    "icu_exit_alive": "icu_exit_alive_in_period",
    "catheter_exposure_days": "nominal_exposure_days",
}
OUTCOMES = tuple(OUTCOME_COLUMNS)


# Technical estimator settings; routine run choices come from pipeline_config.py.
SEED = 20260923
SETTINGS = {
    "n_folds": 5,
    "inner_folds": 2,
    "seed": SEED,
    "continuation_learner": run_config.POLICY_DML_CONTINUATION_LEARNER,
    "propensity_clip": None,
    "horizon_hours": None,
}

# Continuation models predict real-valued recursive targets for every outcome.
LEARNER_CONFIGURATIONS = {
    "ridge": {
        "alpha": 1.0,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": SEED,
    },
    "extra_trees": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": SEED,
    },
    "xgboost": {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "n_jobs": -1,
    },
}

LEARNER_PREPROCESSING = {
    "ridge": {
        "preprocessing_type": "median_imputation_and_standardisation",
        "imputation_used": True,
        "scaling_used": True,
        "native_missing_handling": False,
    },
    "random_forest": {
        "preprocessing_type": "median_imputation",
        "imputation_used": True,
        "scaling_used": False,
        "native_missing_handling": False,
    },
    "extra_trees": {
        "preprocessing_type": "median_imputation",
        "imputation_used": True,
        "scaling_used": False,
        "native_missing_handling": False,
    },
    "xgboost": {
        "preprocessing_type": "native_missing_values",
        "imputation_used": False,
        "scaling_used": False,
        "native_missing_handling": True,
    },
}


N_BOOTSTRAP = run_config.POLICY_DML_N_BOOTSTRAP
BOOTSTRAP_SEED = SETTINGS['seed']
CLIPPING_SENSITIVITY = run_config.POLICY_DML_CLIPPING_SENSITIVITY
PANEL_RUNS = tuple((name, *run_config.PANEL_PATHS[name]) for name in run_config.EVALUATION_PANELS)
SUMMARY_COLUMNS = {
    "cauti": "predicted_cauti_risk", "recatheterisation": "predicted_recatheterisation_risk",
    "death": "predicted_death_risk", "icu_exit_alive": "predicted_icu_exit_alive_risk",
    "catheter_exposure_days": "expected_mean_catheter_exposure_days",
}


# Prepare observed decisions and link their outcome contributions.
def require_columns(frame, columns, context):
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def probabilities(values, name):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}: probabilities must be finite and in [0,1]")
    return values


def validate_folds(frame):
    require_columns(frame, ["subject_id", FOLD], "fold ownership")
    folds = pd.to_numeric(frame[FOLD], errors="raise")
    if folds.isna().any() or (folds < 0).any() or (folds % 1 != 0).any():
        raise ValueError("Missing or invalid fold IDs")
    if frame.groupby("subject_id")[FOLD].nunique().ne(1).any():
        raise ValueError("Patient leakage: patient belongs to multiple folds")
    if folds.nunique() < 2:
        raise ValueError("At least two populated patient folds are required")


def prepare_decisions(panel, settings=None):
    """Validate the existing fixed grid, attach OUT rewards, and link IN rows.

    No new panel or time grid is constructed. IDs use the existing policy builder.
    A configured horizon must coincide with an existing boundary (or follow the
    structural episode endpoint); partially observed intervals are not split.
    """
    settings = SETTINGS if settings is None else settings
    required = [*policy_inputs.POLICY_INPUT_COLS, "episode_index", "episode_end_time",
                "episode_end_reason", "at_risk_cauti", "at_risk_reinsertion",
                *list(OUTCOME_COLUMNS.values())[:-1]]
    require_columns(panel, required, "fixed-period panel")
    if panel.empty:
        raise ValueError("Empty fixed-period panel")
    df = panel.copy()
    if df[policy_inputs.EPISODE_KEY_COLS].isna().any().any():
        raise ValueError("Missing patient/episode identity")
    df["subject_id"] = df.subject_id.astype(str)
    for col in ["inserted", "removed", "period_start", "period_end", "episode_end_time"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="raise")
        if df[col].isna().any():
            raise ValueError(f"Missing {col}")
    if "simulation_end_time" in df:
        df["simulation_end_time"] = pd.to_datetime(df.simulation_end_time, format="mixed", errors="raise")
    for col in [*list(OUTCOME_COLUMNS.values())[:-1], "removed_in_period", "at_risk_cauti", "at_risk_reinsertion"]:
        df[col] = pd.to_numeric(df[col]).astype(int)
    # No censoring probability exists in this repository. Do not invent one or
    # silently interpret nonterminal loss as a structural endpoint.
    for col in ("censored", "censored_in_period", "administrative_censoring", "loss_to_followup"):
        if col in df and pd.to_numeric(df[col]).astype(int).any():
            raise NotImplementedError("Nonterminal censoring requires OOF observation probabilities and an explicit attribution model")
    original_decision = df.get("is_decision_row")
    expected_decision = df.catheter_state.eq("in") & df.observed_action.isin(["keep", "remove"])
    if original_decision is not None:
        parsed = original_decision.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if parsed.isna().any() or parsed.ne(expected_decision).any():
            raise ValueError("Inconsistent decision eligibility")
    df = policy_inputs.add_stable_ids_and_decision_flag(df)
    if df.duplicated([EPISODE, "episode_index"]).any():
        raise ValueError("Duplicate decision/period indices")
    terminal_cols = ["reinsertion_in_period", "death_in_period", "icu_exit_alive_in_period"]
    links = []
    grid_seconds = set()
    trajectory_columns = list(dict.fromkeys([*required, EPISODE, ROW, "is_decision_row"]))
    if "simulation_end_time" in df:
        trajectory_columns.append("simulation_end_time")
    for episode_id, group in df[trajectory_columns].groupby(EPISODE, sort=False):
        if not np.array_equal(group.episode_index, np.arange(len(group))):
            raise ValueError(f"Broken episode ordering in episode {episode_id}")
        if group.episode_end_time.nunique() != 1 or group.iloc[0].period_start != group.iloc[0].inserted:
            raise ValueError("Episode start/end metadata disagree")
        if not (group.period_end > group.period_start).all():
            raise ValueError("Nonpositive period length")
        if not np.array_equal(group.period_start.iloc[1:].to_numpy(), group.period_end.iloc[:-1].to_numpy()):
            raise ValueError("Missing next-period linkage or overlapping periods")
        duration = (group.period_end - group.period_start).dt.total_seconds().to_numpy()
        if len(group) > 1:
            grid_seconds.add(float(duration[0]))
        if len(group) > 1 and (not np.allclose(duration[:-1], duration[0]) or duration[-1] > duration[0] + 1e-6):
            raise ValueError("Panel is not an insertion-anchored fixed grid")
        last = group.iloc[-1]
        if last.period_end != last.episode_end_time:
            raise ValueError("Incomplete follow-up before episode endpoint")
        reasons = {"death": "death_in_period", "reinsertion": "reinsertion_in_period", "icu_exit_alive": "icu_exit_alive_in_period"}
        simulation_horizon = last.episode_end_reason == "simulation_horizon"
        if simulation_horizon:
            if ("simulation_end_time" not in group or group.simulation_end_time.isna().any()
                    or group.simulation_end_time.nunique() != 1 or last.simulation_end_time != last.period_end):
                raise ValueError("Simulation horizon must match its explicit fixed episode cutoff")
        elif last.episode_end_reason not in reasons:
            raise NotImplementedError("Unresolved/nonterminal episode ending: censoring model required")
        if (group.iloc[:-1][terminal_cols].to_numpy().sum() != 0
                or last[terminal_cols].sum() != (0 if simulation_horizon else 1)
                or (not simulation_horizon and last[reasons[last.episode_end_reason]] != 1)):
            raise ValueError("Decision rows after termination or inconsistent terminal precedence")
        if group.iloc[:-1].episode_end_reason.notna().any():
            raise ValueError("Terminal reason before final period")
        decisions = group.loc[group.is_decision_row]
        if decisions.empty or not np.array_equal(decisions.episode_index, np.arange(len(decisions))):
            raise ValueError("Decisions must be a nonempty contiguous episode prefix")
        if not group.observed_action.eq(np.where(group.catheter_state.eq("out"), "out", np.where(group.removed_in_period.eq(1), "remove", "keep"))).all():
            raise ValueError("Observed actions disagree with state/action indicator")
        remove = decisions.removed_in_period.eq(1)
        if remove.sum() > 1 or (remove.any() and not remove.iloc[-1]):
            raise ValueError("Decision rows after removal")
        if len(decisions) < len(group) and not remove.iloc[-1]:
            raise ValueError("OUT follow-up without linked removal decision")
        expected_remove = (group.removed.lt(group.episode_end_time) & group.removed.gt(group.period_start)
                           & group.removed.le(group.period_end) & group.catheter_state.eq("in"))
        if not group.removed_in_period.eq(expected_remove.astype(int)).all():
            raise ValueError("Removal timestamps disagree with decision ordering")
        if not group.catheter_state.eq(np.where(group.period_start.lt(group.removed), "in", "out")).all():
            raise ValueError("Catheter state disagrees with recorded removal")
        if group.loc[~group.is_decision_row, "removed_in_period"].any():
            raise ValueError("Removal on OUT row")
        if (group.cauti_in_period > group.at_risk_cauti).any() or group.cauti_in_period.sum() > 1:
            raise ValueError("CAUTI event outside risk set or repeated first event")
        prior = group.cauti_in_period.cumsum() - group.cauti_in_period
        if group.loc[prior.gt(0), "at_risk_cauti"].any():
            raise ValueError("CAUTI risk set continues after first event")
        # CoR section 3.2 specifies 48 elapsed hours, independently of grid size.
        # Reject conflicting input masks; never silently rewrite source outcomes.
        expected_cauti_risk = group.period_start.lt(group.removed + pd.Timedelta(hours=48)) & prior.eq(0)
        if not group.at_risk_cauti.eq(expected_cauti_risk.astype(int)).all():
            raise ValueError("CAUTI risk mask conflicts with CoR's 48-hour post-removal first-event window")
        if (group.reinsertion_in_period > group.at_risk_reinsertion).any():
            raise ValueError("Reinsertion outside its risk set")
        # Attribution is inherited from the validated panel, including partial
        # REMOVE periods and terminal-event precedence.
        keep = group
        if settings['horizon_hours'] is not None:
            cutoff = group.inserted.iloc[0] + pd.Timedelta(hours=settings['horizon_hours'])
            if ((group.period_start < cutoff) & (group.period_end > cutoff)).any():
                raise ValueError("Evaluation horizon cuts a period; choose an existing grid boundary")
            keep = group.loc[group.period_end.le(cutoff)]
            if keep.empty:
                raise ValueError("Horizon contains no complete eligible decision")
        dest = keep.loc[keep.is_decision_row, ROW].iloc[-1]
        for row in keep.itertuples():
            links.append({ROW: row.decision_row_id, "contribution_decision_row_id": row.decision_row_id if row.is_decision_row else dest})
    if len(grid_seconds) > 1:
        raise ValueError("Panel mixes multiple fixed-period configurations")
    attribution = pd.DataFrame(links)
    df = df.merge(attribution, on=ROW, validate="one_to_one")
    df["nominal_exposure_days"] = ((df.period_end - df.period_start).dt.total_seconds() / 86400).where(df.is_decision_row, 0.0)
    df["actual_exposure_days"] = pec.observed_catheter_exposure_days(df)
    df["prior_cauti"] = df.groupby(EPISODE).cauti_in_period.cumsum() - df.cauti_in_period
    df["elapsed_hours"] = (df.period_start - df.inserted).dt.total_seconds() / 3600
    decisions = df.loc[df.is_decision_row].copy().reset_index(drop=True)
    for outcome, col in OUTCOME_COLUMNS.items():
        reward = df.groupby("contribution_decision_row_id")[col].sum()
        decisions[f"reward_{outcome}"] = decisions[ROW].map(reward)
    decisions["actual_observed_catheter_exposure_days"] = decisions[ROW].map(df.groupby("contribution_decision_row_id").actual_exposure_days.sum())
    decisions["next_decision_row_id"] = decisions.groupby(EPISODE)[ROW].shift(-1).astype("Int64")
    decisions["decision_period"] = decisions.episode_index + 1
    decisions["n_linked_out_rows"] = decisions[ROW].map(df.loc[~df.is_decision_row].groupby("contribution_decision_row_id").size()).fillna(0).astype(int)
    for col in [f"reward_{k}" for k in OUTCOMES]:
        if not np.isfinite(decisions[col]).all():
            raise ValueError("Unexplained missing policy contribution")
    return decisions, df[[ROW, EPISODE, "subject_id", "period_start", "period_end", "is_decision_row", "contribution_decision_row_id", *OUTCOME_COLUMNS.values()]]


# Define target policies and the recursive Policy-DML calculation.
def policy_probabilities(policy, frame, behaviour):
    """Removal probabilities from current practice or a saved policy instruction."""
    if policy["kind"] == "current_practice":
        return probabilities(behaviour, "current-practice policy")
    if policy["kind"] == "fixed_day_removal":
        day = policy.get("remove_day", np.nan)
        if not np.isfinite(day) or day < 1 or day % 1:
            raise ValueError("Fixed-day policy requires a positive integer removal day")
        return frame[f"policy_dml_action_{int(day)}"].eq("remove").to_numpy(dtype=float)
    if policy["kind"] == "stochastic":
        return probabilities(np.full(len(frame), policy["remove_probability"]), "stochastic policy")
    raise ValueError(f"Unknown policy kind {policy['kind']}")


def configured_policies(days):
    policies = [{"name": "current_practice", "kind": "current_practice", "remove_day": np.nan}]
    for day in days:
        policies.append({"name": f"remove_on_day_{day}", "kind": "fixed_day_removal", "remove_day": day})
    return policies


def ratio_quantities(action, pi_remove, g_remove, clip=None):
    action = np.asarray(action)
    if not np.isin(action, [0, 1]).all():
        raise ValueError("Invalid observed action")
    pi_remove = probabilities(pi_remove, "policy")
    g_remove = probabilities(g_remove, "behaviour")
    pi = np.where(action == 1, pi_remove, 1 - pi_remove)
    raw = np.where(action == 1, g_remove, 1 - g_remove)
    if clip is not None and not 0 < clip < .5:
        raise ValueError("Invalid propensity clipping rule")
    used_remove = g_remove if clip is None else np.clip(g_remove, clip, 1 - clip)
    used = np.where(action == 1, used_remove, 1 - used_remove)
    if (used == 0).any():
        raise ValueError("Zero observed-action probability: undefined ratio; specify a sensitivity rule explicitly")
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = pi / raw
    return {"policy_observed_probability": pi, "behaviour_observed_probability_raw": raw,
            "behaviour_observed_probability_used": used, "ratio_raw": raw_ratio, "ratio": pi / used,
            "propensity_clipped": raw != used,
            "behaviour_policy_action_probability": np.where(pi_remove == 1, g_remove, np.where(pi_remove == 0, 1 - g_remove, np.nan)),
            "policy_behaviour_probability_overlap": pi_remove * g_remove + (1 - pi_remove) * (1 - g_remove),
            "policy_observed_action_agreement": np.where(np.isin(pi_remove, [0, 1]), action == pi_remove, np.nan)}


def recursive_step(reward, next_score, q_keep, q_remove, action, pi_remove, ratio):
    """CoR Qbar and psi; arrays may contain signed/unbounded pseudo-outcomes."""
    qbar = (1 - pi_remove) * q_keep + pi_remove * q_remove
    q_observed = np.where(np.asarray(action) == 1, q_remove, q_keep)
    score = qbar + ratio * (reward + next_score - q_observed)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite recursive score")
    return score, qbar, q_observed


def state_features(frame):
    # Explicit pre-decision allowlist; never include endpoints, true DGP fields,
    # actions, current outcomes, post-removal timing or actual interval length.
    cols = ["episode_index", "periods_in_state", "elapsed_hours", "prior_cauti", "age"]
    cols += sorted(c for c in frame if c.startswith(("itemid_", "sex_", "ethnicity_")))
    require_columns(frame, cols, "pre-decision state")
    if np.isinf(frame[cols].to_numpy(dtype=float)).any():
        raise ValueError("Infinite state feature")
    return cols


def make_regressor(settings):
    """Build the selected continuation regressor and its preprocessing steps."""
    model_type = settings["continuation_learner"]
    estimator_types = {
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "xgboost": XGBRegressor,
    }
    learner_configuration = LEARNER_CONFIGURATIONS[model_type]
    estimator = estimator_types[model_type](**learner_configuration)

    steps = []
    if LEARNER_PREPROCESSING[model_type]["imputation_used"]:
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if LEARNER_PREPROCESSING[model_type]["scaling_used"]:
        steps.append(("standardiser", StandardScaler()))
    steps.append((model_type, estimator))
    return Pipeline(steps)


def record_fit(audit, train, predict, *, component, outer_fold, inner_fold=None,
               policy="all", outcome="all", period="all", action_counts=None):
    """Record a model fit and periodically print progress."""
    if set(train.subject_id) & set(predict.subject_id):
        raise ValueError("Patient leakage in model training/prediction membership")
    if train.empty:
        raise ValueError(f"Empty training set for {component}, period={period}, fold={outer_fold}")
    model_id = f"model_{len(audit['records']):07d}"
    audit["records"].append({"model_id": model_id, "component": component, "outer_fold": int(outer_fold),
                             "inner_fold": inner_fold, "policy": policy, "outcome": outcome,
                             "decision_period": period, "training_folds": sorted(train[FOLD].unique().tolist()),
                             "held_out_fold": int(outer_fold), "training_rows": len(train),
                             "held_out_rows": len(predict), "action_counts": action_counts})
    now = monotonic()
    if audit["progress"] and now - audit["last_message"] >= 30:
        print(f"[POLICY-DML FIT] {audit['context']}; models={len(audit['records'])}; "
              f"elapsed_minutes={(now - audit['started']) / 60:.1f}; outer_fold={outer_fold}; "
              f"inner_fold={inner_fold}; policy={policy}; outcome={outcome}; period={period}", flush=True)
        audit["last_message"] = now
    return model_id


def _next_scores(frame, values):
    lookup = pd.Series(values, index=frame[ROW])
    result = frame.next_decision_row_id.map(lookup)
    if result.loc[frame.next_decision_row_id.notna()].isna().any():
        raise ValueError("Missing next-period score/linkage")
    return result.fillna(0).to_numpy(dtype=float)


def fit_backward_sequence(train, predict, g_train, g_predict, policy, features, settings, audit,
                          outer_fold, inner_fold, regressor_factory):
    """Auxiliary nested nuisance fit; the entire prediction set is excluded.

    Training-only fitted recursive targets initialise the nuisance sequence.
    Its *held-out* recursive scores become outer continuation regression targets.
    No auxiliary in-sample prediction is exported as an OOF causal score.
    """
    train, predict = train.copy(), predict.copy()
    train_scores = np.zeros((len(train), len(OUTCOMES)))
    held_scores = np.zeros((len(predict), len(OUTCOMES)))
    pi_train = policy_probabilities(policy, train, g_train)
    pi_predict = policy_probabilities(policy, predict, g_predict)
    wt = ratio_quantities(train.removed_in_period, pi_train, g_train, settings['propensity_clip'])["ratio"]
    wp = ratio_quantities(predict.removed_in_period, pi_predict, g_predict, settings['propensity_clip'])["ratio"]
    stages = sorted(set(train.decision_period) | set(predict.decision_period), reverse=True)
    for stage in stages:
        tr = np.flatnonzero(train.decision_period.eq(stage))
        ho = np.flatnonzero(predict.decision_period.eq(stage))
        if len(tr) == 0:
            raise ValueError(f"No training trajectories at decision period {stage}; shorten prespecified horizon or supply more support")
        for k, outcome in enumerate(OUTCOMES):
            target = train[f"reward_{outcome}"].to_numpy() + _next_scores(train, train_scores[:, k])
            model, model_id = fit_continuation(train.iloc[tr], predict.iloc[ho], target[tr], features,
                                              outcome, policy, stage, settings, audit, outer_fold,
                                              inner_fold, regressor_factory, "auxiliary_continuation")
            for frame, idx, scores, pi, w in ((train, tr, train_scores, pi_train, wt), (predict, ho, held_scores, pi_predict, wp)):
                if len(idx) == 0:
                    continue
                q0, q1 = predict_actions(model, frame.iloc[idx], features, outcome)
                nxt = _next_scores(frame, scores[:, k])[idx]
                scores[idx, k] = recursive_step(frame[f"reward_{outcome}"].to_numpy()[idx], nxt, q0, q1,
                                               frame.removed_in_period.to_numpy()[idx], pi[idx], w[idx])[0]
    return held_scores


def fit_continuation(train, heldout, target, features, outcome, policy, stage, settings, audit,
                     outer, inner, factory, component="continuation"):
    # Only future first-CAUTI contributions vanish after a previous CAUTI;
    # other outcome streams retain their trajectories.
    at_risk = train.prior_cauti.eq(0).to_numpy() if outcome == "cauti" else np.ones(len(train), dtype=bool)
    eligible = train.loc[at_risk]
    counts = {str(a): int(eligible.removed_in_period.eq(a).sum()) for a in (0, 1)}
    model_id = record_fit(
        audit,
        train,
        heldout,
        component=component,
        outer_fold=outer,
        inner_fold=inner,
        policy=policy['name'],
        outcome=outcome,
        period=int(stage),
        action_counts=counts,
    )
    audit["records"][-1]["risk_set_training_rows"] = len(eligible)
    if eligible.empty:
        if outcome == "cauti" and heldout.prior_cauti.eq(0).any():
            raise ValueError("No CAUTI-risk training trajectories at this stage")
        return None, model_id
    x = eligible[[*features, "removed_in_period"]]
    if factory is None:
        model = make_regressor(settings)
    else:
        model = clone(factory())
    model.fit(x, np.asarray(target)[at_risk])
    # Match CPU prediction inputs after fitting on the configured training device.
    if factory is None and settings["continuation_learner"] == "xgboost":
        model.named_steps["xgboost"].set_params(device="cpu")
    return model, model_id


def predict_actions(model, frame, features, outcome):
    if model is None:
        return np.zeros(len(frame)), np.zeros(len(frame))
    x = frame[[*features, "removed_in_period"]].copy()
    answers = []
    for action in (0, 1):
        x["removed_in_period"] = action
        answer = np.asarray(model.predict(x), dtype=float)
        if outcome == "cauti":
            answer[frame.prior_cauti.gt(0).to_numpy()] = 0
        if not np.isfinite(answer).all():
            raise ValueError("Missing/nonfinite continuation prediction")
        answers.append(answer)
    return tuple(answers)


# Fit nested patient folds and calculate each held-out decision score.
def crossfit_policy_scores(decisions, policies, *, settings=None, behaviour_learner="logistic_regression",
                          saved_behaviour=None, patient_counts=None, regressor_factory=None,
                          progress=False, progress_detail=False, progress_context="panel"):
    """Outer OOF score, with nested OOF targets isolated from the outer fold.

    saved_behaviour is a validated decision-row keyed table including g and
    folds. Refit bootstrap represents multiplicity only in training and episode
    averaging; duplicated patients never cross training/held-out boundaries.
    """
    settings = SETTINGS if settings is None else settings
    df = decisions.copy().reset_index(drop=True)
    if saved_behaviour is not None:
        require_columns(saved_behaviour, [ROW, "p_remove_obs", FOLD], "OOF behaviour")
        if FOLD in df:
            existing = df.set_index(ROW)[FOLD]
            supplied = saved_behaviour.set_index(ROW)[FOLD].reindex(existing.index)
            if not existing.eq(supplied).all():
                raise ValueError("Inconsistent nuisance/continuation fold IDs")
            df = df.drop(columns=FOLD)
        df = df.merge(saved_behaviour[[ROW, "p_remove_obs", FOLD]], on=ROW, how="left", validate="one_to_one")
        probabilities(df.p_remove_obs, "saved OOF behaviour")
    elif FOLD not in df:
        df = nuisance.add_grouped_crossfit_folds(df, n_splits=settings['n_folds'])
    validate_folds(df)
    features = state_features(df)
    behaviour_features = [c for c in features if c not in ("prior_cauti", "elapsed_hours")]
    if patient_counts is None:
        patient_counts = pd.Series(1, index=pd.Index(df.subject_id.unique()))
    count = df.subject_id.map(patient_counts)
    if count.isna().any() or (count < 0).any() or (count % 1 != 0).any():
        raise ValueError("Invalid patient bootstrap multiplicity")
    df = df.loc[count.gt(0)].reset_index(drop=True)
    validate_folds(df)
    outer_folds = sorted(df[FOLD].unique())
    inner_assignments = {}
    for outer in outer_folds:
        train = df.loc[df[FOLD].ne(outer)].reset_index(drop=True)
        if train.subject_id.nunique() < 2:
            raise ValueError("Nested continuation cross-fitting requires two training patients outside each outer fold")
        inner = nuisance.add_grouped_crossfit_folds(
            train.drop(columns=FOLD).copy(),
            n_splits=min(settings['inner_folds'], train.subject_id.nunique()),
        )[FOLD].to_numpy()
        for inner_fold in (None, *np.unique(inner)):
            training_periods = train.decision_period if inner_fold is None else train.loc[inner != inner_fold, "decision_period"]
            required_periods = df.decision_period if inner_fold is None else train.decision_period
            missing = sorted(set(required_periods) - set(training_periods))
            if missing:
                raise ValueError(f"No training trajectories at decision periods {missing} in outer fold {outer}, "
                                 f"inner fold {inner_fold}. Choose an explicit evaluation horizon or a model "
                                 "that pools information across periods; no continuation models have been fitted.")
        inner_assignments[outer] = inner
    # One record per fit, with a timer for progress messages.
    started = monotonic()
    audit = {"records": [], "progress": progress, "context": progress_context,
             "started": started, "last_message": started}
    results = []
    for outer_number, outer in enumerate(outer_folds, start=1):
        if progress:
            print(f"[POLICY-DML FIT] {progress_context}; outer={outer_number}/{len(outer_folds)}; phase=targets", flush=True)
        train = df.loc[df[FOLD].ne(outer)].copy().reset_index(drop=True)
        held = df.loc[df[FOLD].eq(outer)].copy().reset_index(drop=True)
        expanded = train.loc[train.index.repeat(train.subject_id.map(patient_counts))].reset_index(drop=True)
        if saved_behaviour is None:
            behaviour_id = record_fit(
                audit,
                expanded,
                held,
                component="behaviour",
                outer_fold=outer,
                inner_fold=None,
            )
            fitted_behaviour = nuisance.fit_crossfit_fold_model(
                expanded[behaviour_features],
                expanded.removed_in_period,
                "policy_dml_behaviour",
                outer,
                model_type=behaviour_learner,
                groups=expanded.subject_id,
            )
            g_held = nuisance.predict_crossfit_fold(
                fitted_behaviour,
                held[behaviour_features],
            )
        else:
            g_held = held.p_remove_obs.to_numpy()
            behaviour_id = record_fit(audit, train, held, component="reused_behaviour", outer_fold=outer)
        # Inner assignment uses source patient identity, before multiplicity.
        inner = inner_assignments[outer]
        inner_targets = {p['name']: np.full((len(train), len(OUTCOMES)), np.nan) for p in policies}
        for inner_fold in np.unique(inner):
            if progress_detail:
                print(f"[POLICY-DML FIT] {progress_context}; outer={outer_number}/{len(outer_folds)}; "
                      f"inner={int(inner_fold) + 1}/{len(np.unique(inner))}; phase=auxiliary", flush=True)
            training = train.loc[inner != inner_fold].copy().reset_index(drop=True)
            inner_held = train.loc[inner == inner_fold].copy().reset_index(drop=True)
            training = training.loc[training.index.repeat(training.subject_id.map(patient_counts))].reset_index(drop=True)
            # Duplicate rows need a unique internal trajectory identity while
            # retaining original patient ownership in model training.
            training = expand_training_trajectories(training)
            record_fit(
                audit,
                training,
                inner_held,
                component="behaviour",
                outer_fold=outer,
                inner_fold=int(inner_fold),
            )
            g_model = nuisance.fit_crossfit_fold_model(
                training[behaviour_features],
                training.removed_in_period,
                "policy_dml_behaviour",
                outer,
                model_type=behaviour_learner,
                groups=training.subject_id,
            )
            g_inner = nuisance.predict_crossfit_fold(
                g_model,
                inner_held[behaviour_features],
            )
            g_training = nuisance.predict_crossfit_fold(g_model, training[behaviour_features])
            for policy in policies:
                inner_targets[policy['name']][inner == inner_fold] = fit_backward_sequence(
                    training, inner_held, g_training, g_inner, policy, features, settings, audit,
                    outer, int(inner_fold), regressor_factory)
                if progress_detail:
                    print(f"[POLICY-DML FIT] {progress_context}; outer={outer_number}/{len(outer_folds)}; "
                          f"inner={int(inner_fold) + 1}/{len(np.unique(inner))}; policy={policy['name']}; completed", flush=True)
        for policy in policies:
            if progress_detail:
                print(f"[POLICY-DML FIT] {progress_context}; outer={outer_number}/{len(outer_folds)}; "
                      f"policy={policy['name']}; phase=outer_scores", flush=True)
            pi = policy_probabilities(policy, held, g_held)
            quantities = ratio_quantities(held.removed_in_period, pi, g_held, settings['propensity_clip'])
            scores = np.zeros((len(held), len(OUTCOMES)))
            training_next = np.column_stack([_next_scores(train, inner_targets[policy['name']][:, k]) for k in range(len(OUTCOMES))])
            for stage in sorted(held.decision_period.unique(), reverse=True):
                tr = train.decision_period.eq(stage)
                ho = np.flatnonzero(held.decision_period.eq(stage))
                repeat = np.repeat(np.flatnonzero(tr), train.loc[tr, "subject_id"].map(patient_counts))
                expanded_stage = train.iloc[repeat]
                for k, outcome in enumerate(OUTCOMES):
                    target = train[f"reward_{outcome}"].to_numpy() + training_next[:, k]
                    model, model_id = fit_continuation(expanded_stage, held.iloc[ho], target[repeat], features,
                                                      outcome, policy, stage, settings, audit, outer, None, regressor_factory)
                    q0, q1 = predict_actions(model, held.iloc[ho], features, outcome)
                    nxt = _next_scores(held, scores[:, k])[ho]
                    score, qbar, qobs = recursive_step(held[f"reward_{outcome}"].to_numpy()[ho], nxt, q0, q1,
                                                      held.removed_in_period.to_numpy()[ho], pi[ho], quantities["ratio"][ho])
                    scores[ho, k] = score
                    part = held.iloc[ho][["subject_id", "hadm_id", "stay_id", "inserted", "removed", EPISODE, ROW, FOLD,
                                         "decision_period", "period_start", "period_end", "next_decision_row_id", "removed_in_period", "n_linked_out_rows"]].copy()
                    part["policy_name"], part["policy_type"], part["policy_remove_day"] = policy['name'], policy['kind'], policy['remove_day']
                    part["outcome"] = outcome
                    part["reward"] = held[f"reward_{outcome}"].to_numpy()[ho]
                    part["next_score"], part["q_keep"], part["q_remove"] = nxt, q0, q1
                    part["qbar"], part["q_observed"], part["recursive_score"] = qbar, qobs, score
                    part["pi_remove"], part["g_remove"] = pi[ho], g_held[ho]
                    for key, value in quantities.items():
                        part[key] = value[ho]
                    part["continuation_model_id"], part["behaviour_model_id"] = model_id, behaviour_id
                    part["clipping_epsilon"] = settings['propensity_clip']
                    results.append(part)
        if progress:
            print(f"[POLICY-DML FIT] {progress_context}; outer={outer_number}/{len(outer_folds)}; completed", flush=True)
    rows = pd.concat(results, ignore_index=True).sort_values(["policy_name", "outcome", EPISODE, "decision_period"]).reset_index(drop=True)
    groups = ["policy_name", "outcome", EPISODE]
    rows["cumulative_ratio"] = rows.groupby(groups).ratio.cumprod()
    rows["cumulative_ratio_raw"] = rows.groupby(groups).ratio_raw.cumprod()
    if not np.isfinite(rows.cumulative_ratio).all():
        raise ValueError("Cumulative ratio overflow; review support")
    return rows, audit["records"], df


def expand_training_trajectories(frame):
    """Internal copies for bootstrap fitting; patient IDs/folds stay unchanged."""
    frame = frame.copy()
    occurrence = frame.groupby(ROW).cumcount()
    original_next = frame.next_decision_row_id.copy()
    key = list(zip(frame[ROW], occurrence))
    lookup = {item: i + 1 for i, item in enumerate(key)}
    frame[ROW] = [lookup[item] for item in key]
    frame["next_decision_row_id"] = pd.array([
        lookup[(nxt, copy)] if pd.notna(nxt) else pd.NA for nxt, copy in zip(original_next, occurrence)
    ], dtype="Int64")
    return frame


# Summarise episode values, support and bootstrap uncertainty.
def episode_scores(rows):
    keys = ["policy_name", "policy_type", "policy_remove_day", "subject_id", EPISODE, FOLD]
    keys += [c for c in ("hadm_id", "stay_id", "inserted", "removed") if c in rows]
    first = rows.loc[rows.decision_period.eq(1)]
    if first.duplicated(["policy_name", EPISODE, "outcome"]).any():
        raise ValueError("More than one first score per episode")
    wide = first.set_index([*keys, "outcome"]).recursive_score.unstack("outcome").reset_index()
    if wide[list(OUTCOMES)].isna().any().any():
        raise ValueError("Missing outcome score")
    wide = wide.rename(columns={k: f"score_{k}" for k in OUTCOMES})
    wide["prediction_complete"] = True
    wide["catheter_exposure_measure"] = pec.EXPOSURE_MEASURE
    return wide


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("ESS requires finite nonnegative weights")
    if not w.size or not w.max():
        return 0.0
    w = w / w.max()
    return float(w.sum() ** 2 / np.square(w).sum())


def support_diagnostics(rows):
    # Weight quantities are shared across outcomes; avoid counting them five times.
    unique = rows.loc[rows.outcome.eq(OUTCOMES[0])]
    results = []
    for (policy, fold, stage), group in unique.groupby(["policy_name", FOLD, "decision_period"]):
        results.append(_support_row(group, policy, "fold_period", fold, stage))
    for (policy, stage), group in unique.groupby(["policy_name", "decision_period"]):
        results.append(_support_row(group, policy, "period", np.nan, stage))
    for policy, group in unique.groupby("policy_name"):
        final = group.sort_values("decision_period").groupby(EPISODE).tail(1)
        results.append(_support_row(final, policy, "episode_final", np.nan, np.nan))
    return pd.DataFrame(results)


def _support_row(group, policy, scope, fold, stage):
    weights = group.cumulative_ratio
    patient = group.groupby("subject_id").cumulative_ratio.sum()
    return {"policy_name": policy, "scope": scope, FOLD: fold, "decision_period": stage,
            "n_episodes": group[EPISODE].nunique(), "n_patients": group.subject_id.nunique(),
            "n_contributing_episodes": int(weights.gt(0).sum()), "n_contributing_patients": int(patient.gt(0).sum()),
            "episode_ess": effective_sample_size(weights), "patient_ess": effective_sample_size(patient),
            "ratio_max": group.ratio.max(), "cumulative_ratio_max": weights.max(),
            "cumulative_ratio_p50": weights.quantile(.5), "cumulative_ratio_p95": weights.quantile(.95),
            "cumulative_ratio_p99": weights.quantile(.99), "n_clipped": int(group.propensity_clipped.sum()),
            "min_observed_behaviour_probability": group.behaviour_observed_probability_raw.min(),
            "mean_policy_action_agreement": group.policy_observed_action_agreement.mean()}


def load_saved_behaviour(decisions, model_dir):
    """Load the existing patient-level OOF behaviour probabilities."""
    directory = Path(model_dir)
    score_path = directory / "nuisance_predictions.csv"
    # Match the saved identifier types to the prepared decision rows.
    scores = pd.read_csv(score_path, low_memory=False)
    scores["subject_id"] = scores.subject_id.astype(str)
    scores["inserted"] = pd.to_datetime(scores.inserted, format="mixed")
    output = decisions[[ROW, *KEYS]].merge(
        scores[KEYS + [FOLD, "p_remove_obs"]],
        on=KEYS,
        how="left",
    )
    return output, {"source": str(score_path), "held_out_folds": sorted(output[FOLD].unique().tolist())}


def policy_summary(episodes, decisions):
    rows = []
    for (name, kind, day), group in episodes.groupby(["policy_name", "policy_type", "policy_remove_day"], dropna=False, sort=False):
        row = {"policy_name": name, "policy_type": kind, "policy_remove_day": day,
               "estimator": "policy_dml", "prediction_mode": "recursive_patient_crossfit",
               "n_patients": group.subject_id.nunique(), "n_episodes": len(group),
               "n_complete_prediction_episodes": len(group), "n_incomplete_prediction_episodes": 0,
               "catheter_exposure_measure": pec.EXPOSURE_MEASURE}
        for outcome, col in SUMMARY_COLUMNS.items():
            row[col] = float(group[f"score_{outcome}"].mean())
            row[f"observed_event_count_{outcome}"] = float(decisions[f"reward_{outcome}"].sum())
            if outcome != "catheter_exposure_days":
                row[col + "_pct"] = row[col] * 100
        rows.append(row)
    summary = pd.DataFrame(rows)
    comparison = {k + "_risk" if k != "catheter_exposure_days" else k: v for k, v in SUMMARY_COLUMNS.items()}
    return pec.add_standard_comparisons(summary, baseline_label="current_practice", comparison_map=comparison)


def patient_statistics(episodes):
    """Sum unnormalised recursive scores and episode counts within patients."""
    subjects = pd.Index(sorted(episodes.subject_id.unique()))
    entries, sums = [], []
    for (policy, day), frame in episodes.groupby(["policy_name", "policy_remove_day"], dropna=False, sort=False):
        codes = subjects.get_indexer(frame.subject_id)
        for outcome in OUTCOMES:
            entries.append({"estimator": "policy_dml", "policy_name": policy, "policy_remove_day": day, "outcome": outcome})
            sums.append(np.column_stack((np.bincount(codes, weights=frame[f"score_{outcome}"], minlength=len(subjects)),
                                         np.bincount(codes, minlength=len(subjects)))))
    return subjects, pd.DataFrame(entries), np.stack(sums, axis=1)


def run_bootstrap(episodes, decisions, policies, output, *, mode, settings, learner, n_bootstrap, seed, regressor_factory=None):
    subjects, entries, statistics = patient_statistics(episodes)
    # Average episode scores after applying each patient's bootstrap count.
    totals = np.einsum("i,ijk->jk", np.ones(len(subjects)), statistics)
    point = totals[:, 0] / totals[:, 1]
    values = np.empty((n_bootstrap, len(entries)))
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    for replicate in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        multiplicity = pd.Series(counts, index=subjects)
        if mode == "fixed":
            totals = np.einsum("i,ijk->jk", counts, statistics)
            values[replicate] = totals[:, 0] / totals[:, 1]
        else:
            # Entire patient ownership, including duplicate copies, stays fixed.
            scored, _, _ = crossfit_policy_scores(decisions, policies, settings=settings,
                behaviour_learner=learner, patient_counts=multiplicity, regressor_factory=regressor_factory,
                progress=True, progress_context=f"bootstrap={replicate + 1}")
            sampled = episode_scores(scored)
            sampled_subjects, _, sampled_stats = patient_statistics(sampled)
            sampled_counts = multiplicity.reindex(sampled_subjects).to_numpy()
            totals = np.einsum("i,ijk->jk", sampled_counts, sampled_stats)
            values[replicate] = totals[:, 0] / totals[:, 1]
        if (replicate + 1) % 5 == 0 or replicate + 1 == n_bootstrap:
            print(f"[BOOTSTRAP] estimator=policy_dml; bootstrap={mode}; completed={replicate + 1}/{n_bootstrap}", flush=True)
    if n_bootstrap:
        interval = inference.interval_table(entries, point, values, len(subjects))
        interval["bootstrap_mode"] = mode
        interval["nuisance_refitting_uncertainty"] = mode == "refit"
        interval.to_csv(output / "policy_values.csv", index=False)
        differences = inference.interval_table(entries, inference.paired_differences(point, entries), inference.paired_differences(values, entries), len(subjects))
        differences["bootstrap_mode"] = mode
        differences.to_csv(output / "policy_differences.csv", index=False)
    return values


# Read the configured inputs and run estimation for each selected panel.
def run_panel_estimation(panel_path, artefact_root, model_type, bootstrap_mode, panel_name=None, *,
                         settings=None, n_bootstrap=None, regressor_factory=None):
    if settings is None:
        settings = dict(SETTINGS)
        settings["horizon_hours"] = run_config.POLICY_DML_HORIZON_HOURS.get(panel_name, SETTINGS["horizon_hours"])
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}; bootstrap={bootstrap_mode}; phase=load_panel", flush=True)
    panel = nuisance.load_panel(panel_path)
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}; phase=prepare_decisions", flush=True)
    decisions, _ = prepare_decisions(panel, settings)
    policies = configured_policies(run_config.POLICY_DAYS)
    # Attach each policy's instruction to the observed decision rows.
    policy_dir = run_config.POLICY_DIRS[panel_name]
    manifest = pd.read_csv(policy_dir / "policy_panels.csv")
    for day in run_config.POLICY_DAYS:
        policy = manifest.loc[manifest.policy_remove_day.eq(day)].iloc[0]
        actions = pd.read_csv(
            policy_dir / policy.panel_file,
            usecols=[ROW, "policy_dml_action"],
        ).rename(columns={"policy_dml_action": f"policy_dml_action_{day}"})
        decisions = decisions.merge(actions, on=ROW, how="left", validate="one_to_one")
        if decisions[f"policy_dml_action_{day}"].isna().any():
            raise ValueError(f"Missing day-{day} policy instructions for decision rows")
    saved, provenance = None, None
    model_dir = Path(artefact_root) / model_type
    # Reuse suitable primary OOF action predictions in either mode. Full refit
    # draws always refit g and Q. Fixed mode explicitly requires saved g.
    if (model_dir / "nuisance_predictions.csv").exists() or bootstrap_mode == "fixed":
        saved, provenance = load_saved_behaviour(decisions, model_dir)
    stages = int(decisions.decision_period.nunique())
    outer_folds = int(saved[FOLD].nunique()) if saved is not None else settings['n_folds']
    approximate_fits = outer_folds * (settings['inner_folds'] + 1) * len(policies) * len(OUTCOMES) * stages
    learner_configuration = LEARNER_CONFIGURATIONS[settings["continuation_learner"]]
    learner_preprocessing = LEARNER_PREPROCESSING[settings["continuation_learner"]]
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}; phase=point_estimate; "
          f"patients={decisions.subject_id.nunique()}; decisions={len(decisions)}; stages={stages}; "
          f"outer_folds={outer_folds}; inner_folds={settings['inner_folds']}; policies={len(policies)}; "
          f"continuation={settings['continuation_learner']}; trees={learner_configuration.get('n_estimators', 'n/a')}; "
          f"jobs={learner_configuration.get('n_jobs', 'n/a')}; approximate_fits={approximate_fits}", flush=True)
    rows, _, folded = crossfit_policy_scores(decisions, policies, settings=settings, behaviour_learner=model_type,
                                            saved_behaviour=saved, regressor_factory=regressor_factory,
                                            progress=True, progress_detail=True,
                                            progress_context=panel_name or Path(panel_path).stem)
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}; phase=point_estimate; completed", flush=True)
    episodes = episode_scores(rows)
    summary = policy_summary(episodes, decisions)
    output = run_config.RESULT_DIRS[panel_name] / "policy_dml" / bootstrap_mode
    output.mkdir(parents=True, exist_ok=True)
    support_diagnostics(rows).to_csv(output / "diagnostics.csv", index=False)
    metadata = {"estimator": "CoR Appendix B.2 recursive Policy-DML", "settings": settings,
                "continuation_configuration": learner_configuration,
                "continuation_preprocessing": learner_preprocessing,
                "panel": str(panel_path), "panel_name": panel_name, "behaviour_learner": model_type,
                "saved_behaviour_provenance": provenance, "bootstrap_mode": bootstrap_mode,
                "n_bootstrap": n_bootstrap, "bootstrap_seed": BOOTSTRAP_SEED,
                "evaluation_period": "complete panel episode" if settings['horizon_hours'] is None else f"{settings['horizon_hours']} hours from insertion",
                "current_practice": "cross-fitted stochastic behaviour policy pi_current(a|S)=g_hat(a|S)",
                "censoring": "structural endpoints only; nonterminal censoring rejected",
                "exposure": pec.EXPOSURE_MEASURE, "scores_clipped": False,
                "fixed_day_off_path_extension": "remove on first available IN decision on/after the target day",
                "continuation_targets": "nested patient OOF recursive scores; auxiliary nuisance fits exclude outer and inner held-out patients"}
    (output / "run_settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_bootstrap(episodes, folded.drop(columns="p_remove_obs", errors="ignore"), policies, output,
                  mode=bootstrap_mode, settings=settings, learner=model_type, n_bootstrap=n_bootstrap,
                  seed=BOOTSTRAP_SEED, regressor_factory=regressor_factory)
    sensitivity = []
    for clip in CLIPPING_SENSITIVITY:
        alternative, _, _ = crossfit_policy_scores(decisions, policies, settings=dict(settings, propensity_clip=clip),
                                                  behaviour_learner=model_type, saved_behaviour=saved, regressor_factory=regressor_factory)
        result = policy_summary(episode_scores(alternative), decisions)
        result["clipping_epsilon"] = clip
        sensitivity.append(result)
    if sensitivity:
        pd.concat(sensitivity, ignore_index=True).to_csv(output / "clipping_sensitivity.csv", index=False)
    print(f"[POLICY-DML] {panel_name or Path(panel_path).stem}: {len(episodes) // len(policies)} episodes; saved {output}", flush=True)
    return summary, episodes, rows


def main():
    for name, panel, nuisance_root in PANEL_RUNS:
        panel_settings = dict(SETTINGS, horizon_hours=run_config.POLICY_DML_HORIZON_HOURS[name])
        for mode in BOOTSTRAP_MODES:
            run_panel_estimation(panel, nuisance_root, NUISANCE_MODEL_TYPE, mode, name,
                                 settings=panel_settings, n_bootstrap=N_BOOTSTRAP)


if __name__ == "__main__":
    main()
