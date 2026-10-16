"""Recursive episode Policy-DML, CoR Appendix B.1--B.2 (pp. 34--36).

Only this module's first recursive score per episode is a value contribution.
OUT rows are outcome linkage, never additional decisions. No score clipping,
Hajek normalisation, simulated trajectories, or immediate-risk substitution.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_policy_panels as policy_inputs
import fit_nuisance_models as nuisance
import policy_eval_common as pec

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


@dataclass(frozen=True)
class DMLSettings:
    n_folds: int = 5
    inner_folds: int = 3
    seed: int = 20260923
    continuation_learner: str = "extra_trees"
    continuation_trees: int = 100
    continuation_min_leaf: int = 5
    propensity_clip: float | None = None
    horizon_hours: float | None = None

    def __post_init__(self):
        if self.n_folds < 2 or self.inner_folds < 2:
            raise ValueError("Patient cross-fitting requires at least two folds")
        if self.propensity_clip is not None and not 0 < self.propensity_clip < .5:
            raise ValueError("propensity_clip must be None or strictly between 0 and 0.5")
        if self.horizon_hours is not None and (not np.isfinite(self.horizon_hours) or self.horizon_hours <= 0):
            raise ValueError("horizon_hours must be positive and finite")


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


def prepare_decisions(panel, settings=DMLSettings()):
    """Validate the existing fixed grid, attach OUT rewards, and link IN rows.

    No new panel or time grid is constructed. IDs use the existing policy builder.
    A configured horizon must coincide with an existing boundary (or follow the
    structural episode endpoint); partially observed intervals are not split.
    """
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
    for col in [*list(OUTCOME_COLUMNS.values())[:-1], "removed_in_period", "at_risk_cauti", "at_risk_reinsertion"]:
        df[col] = nuisance.binary_values(df[col])
    # No censoring probability exists in this repository. Do not invent one or
    # silently interpret nonterminal loss as a structural endpoint.
    for col in ("censored", "censored_in_period", "administrative_censoring", "loss_to_followup"):
        if col in df and nuisance.binary_values(df[col]).any():
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
    for episode_id, group in df.groupby(EPISODE, sort=False):
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
        if last.episode_end_reason not in reasons:
            raise NotImplementedError("Unresolved/nonterminal episode ending: censoring model required")
        if group.iloc[:-1][terminal_cols].to_numpy().sum() != 0 or last[terminal_cols].sum() != 1 or last[reasons[last.episode_end_reason]] != 1:
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
        if settings.horizon_hours is not None:
            cutoff = group.inserted.iloc[0] + pd.Timedelta(hours=settings.horizon_hours)
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


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    remove_day: float = np.nan
    remove_probability: float | None = None

    def __post_init__(self):
        if not self.name or self.kind not in ("current_practice", "fixed_day_removal", "stochastic"):
            raise ValueError("Invalid policy identity/type")
        if self.kind == "fixed_day_removal" and (not np.isfinite(self.remove_day) or self.remove_day < 1 or self.remove_day % 1):
            raise ValueError("Fixed-day policy requires a positive integer removal day")
        if self.kind == "stochastic":
            probabilities([self.remove_probability], "stochastic policy")

    def probabilities(self, frame, behaviour):
        if self.kind == "current_practice":
            # Explicit learned stochastic representation of clinical practice.
            return probabilities(behaviour, "current-practice policy")
        if self.kind == "fixed_day_removal":
            # Same on-policy path as legacy panels. On off-policy IN histories
            # after the target day, remove at the next available decision.
            return (frame.elapsed_hours.to_numpy() >= (self.remove_day - 1) * 24).astype(float)
        if self.kind == "stochastic":
            return probabilities(np.full(len(frame), self.remove_probability), "stochastic policy")
        raise ValueError(f"Unknown policy kind {self.kind}")


def configured_policies(days):
    policies = [Policy("current_practice", "current_practice")]
    policies.extend(Policy(f"remove_on_day_{day}", "fixed_day_removal", day) for day in days)
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
    """Real-valued regression is necessary even for binary outcome streams."""
    if settings.continuation_learner == "ridge":
        learner = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif settings.continuation_learner in ("extra_trees", "random_forest"):
        cls = ExtraTreesRegressor if settings.continuation_learner == "extra_trees" else RandomForestRegressor
        learner = cls(n_estimators=settings.continuation_trees, min_samples_leaf=settings.continuation_min_leaf,
                      random_state=settings.seed, n_jobs=1)
    else:
        raise ValueError("Choose ridge, extra_trees, random_forest, or pass a regressor factory")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), learner)


class FitAudit:
    """Compact fold-level provenance for models entering exported scores."""
    def __init__(self):
        self.records = []

    def add(self, train, predict, *, component, outer_fold, inner_fold=None,
            policy="all", outcome="all", period="all", action_counts=None, features=()):
        if set(train.subject_id) & set(predict.subject_id):
            raise ValueError("Patient leakage in model training/prediction membership")
        if train.empty:
            raise ValueError(f"Empty training set for {component}, period={period}, fold={outer_fold}")
        model_id = f"model_{len(self.records):07d}"
        self.records.append({"model_id": model_id, "component": component, "outer_fold": int(outer_fold),
                             "inner_fold": inner_fold, "policy": policy, "outcome": outcome,
                             "decision_period": period, "training_folds": sorted(train[FOLD].unique().tolist()),
                             "held_out_fold": int(outer_fold), "training_rows": len(train),
                             "held_out_rows": len(predict), "action_counts": action_counts,
                             "feature_columns": list(features)})
        return model_id


def fit_behaviour(train, predict, features, learner, audit, outer, inner=None):
    model_id = audit.add(train, predict, component="behaviour", outer_fold=outer, inner_fold=inner, features=features)
    fitted = nuisance.fit_crossfit_fold_model(train[features], train.removed_in_period,
                                            "policy_dml_behaviour", outer, model_type=learner, groups=train.subject_id)
    return nuisance.predict_crossfit_fold(fitted, predict[features]), model_id, fitted


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
    pi_train = policy.probabilities(train, g_train)
    pi_predict = policy.probabilities(predict, g_predict)
    wt = ratio_quantities(train.removed_in_period, pi_train, g_train, settings.propensity_clip)["ratio"]
    wp = ratio_quantities(predict.removed_in_period, pi_predict, g_predict, settings.propensity_clip)["ratio"]
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
    model_id = audit.add(train, heldout, component=component, outer_fold=outer, inner_fold=inner,
                         policy=policy.name, outcome=outcome, period=int(stage), action_counts=counts,
                         features=[*features, "removed_in_period"])
    audit.records[-1]["risk_set_training_rows"] = len(eligible)
    if eligible.empty:
        if outcome == "cauti" and heldout.prior_cauti.eq(0).any():
            raise ValueError("No CAUTI-risk training trajectories at this stage")
        return None, model_id
    x = eligible[[*features, "removed_in_period"]]
    model = clone(factory())
    model.fit(x, np.asarray(target)[at_risk])
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


def crossfit_policy_scores(decisions, policies, *, settings=DMLSettings(), behaviour_learner="logistic_regression",
                          saved_behaviour=None, patient_counts=None, regressor_factory=None):
    """Outer OOF score, with nested OOF targets isolated from the outer fold.

    saved_behaviour is a validated decision-row keyed table including g and
    folds. Refit bootstrap represents multiplicity only in training and episode
    averaging; duplicated patients never cross training/held-out boundaries.
    """
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
        df = nuisance.add_grouped_crossfit_folds(df, n_splits=settings.n_folds)
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
    audit = FitAudit()
    factory = regressor_factory or (lambda: make_regressor(settings))
    results = []
    for outer in sorted(df[FOLD].unique()):
        train = df.loc[df[FOLD].ne(outer)].copy().reset_index(drop=True)
        held = df.loc[df[FOLD].eq(outer)].copy().reset_index(drop=True)
        if train.subject_id.nunique() < 2:
            raise ValueError("Nested continuation cross-fitting requires two training patients outside each outer fold")
        expanded = train.loc[train.index.repeat(train.subject_id.map(patient_counts))].reset_index(drop=True)
        if saved_behaviour is None:
            g_held, behaviour_id, _ = fit_behaviour(expanded, held, behaviour_features, behaviour_learner, audit, outer)
        else:
            g_held = held.p_remove_obs.to_numpy()
            behaviour_id = audit.add(train, held, component="reused_behaviour", outer_fold=outer, features=behaviour_features)
        # Inner assignment uses source patient identity, before multiplicity.
        inner = nuisance.add_grouped_crossfit_folds(train.drop(columns=FOLD).copy(),
                                                  n_splits=min(settings.inner_folds, train.subject_id.nunique()))[FOLD].to_numpy()
        inner_targets = {p.name: np.full((len(train), len(OUTCOMES)), np.nan) for p in policies}
        for inner_fold in np.unique(inner):
            training = train.loc[inner != inner_fold].copy().reset_index(drop=True)
            inner_held = train.loc[inner == inner_fold].copy().reset_index(drop=True)
            training = training.loc[training.index.repeat(training.subject_id.map(patient_counts))].reset_index(drop=True)
            # Duplicate rows need a unique internal trajectory identity while
            # retaining original patient ownership in model training.
            training = expand_training_trajectories(training)
            g_inner, _, g_model = fit_behaviour(training, inner_held, behaviour_features, behaviour_learner, audit, outer, int(inner_fold))
            g_training = nuisance.predict_crossfit_fold(g_model, training[behaviour_features])
            for policy in policies:
                inner_targets[policy.name][inner == inner_fold] = fit_backward_sequence(
                    training, inner_held, g_training, g_inner, policy, features, settings, audit,
                    outer, int(inner_fold), factory)
        for policy in policies:
            pi = policy.probabilities(held, g_held)
            quantities = ratio_quantities(held.removed_in_period, pi, g_held, settings.propensity_clip)
            scores = np.zeros((len(held), len(OUTCOMES)))
            training_next = np.column_stack([_next_scores(train, inner_targets[policy.name][:, k]) for k in range(len(OUTCOMES))])
            for stage in sorted(held.decision_period.unique(), reverse=True):
                tr = train.decision_period.eq(stage)
                ho = np.flatnonzero(held.decision_period.eq(stage))
                repeat = np.repeat(np.flatnonzero(tr), train.loc[tr, "subject_id"].map(patient_counts))
                expanded_stage = train.iloc[repeat]
                for k, outcome in enumerate(OUTCOMES):
                    target = train[f"reward_{outcome}"].to_numpy() + training_next[:, k]
                    model, model_id = fit_continuation(expanded_stage, held.iloc[ho], target[repeat], features,
                                                      outcome, policy, stage, settings, audit, outer, None, factory)
                    q0, q1 = predict_actions(model, held.iloc[ho], features, outcome)
                    nxt = _next_scores(held, scores[:, k])[ho]
                    score, qbar, qobs = recursive_step(held[f"reward_{outcome}"].to_numpy()[ho], nxt, q0, q1,
                                                      held.removed_in_period.to_numpy()[ho], pi[ho], quantities["ratio"][ho])
                    scores[ho, k] = score
                    part = held.iloc[ho][["subject_id", "hadm_id", "stay_id", "inserted", "removed", EPISODE, ROW, FOLD,
                                         "decision_period", "period_start", "period_end", "next_decision_row_id", "removed_in_period", "n_linked_out_rows"]].copy()
                    part["policy_name"], part["policy_type"], part["policy_remove_day"] = policy.name, policy.kind, policy.remove_day
                    part["outcome"] = outcome
                    part["reward"] = held[f"reward_{outcome}"].to_numpy()[ho]
                    part["next_score"], part["q_keep"], part["q_remove"] = nxt, q0, q1
                    part["qbar"], part["q_observed"], part["recursive_score"] = qbar, qobs, score
                    part["pi_remove"], part["g_remove"] = pi[ho], g_held[ho]
                    for key, value in quantities.items():
                        part[key] = value[ho]
                    part["continuation_model_id"], part["behaviour_model_id"] = model_id, behaviour_id
                    part["clipping_epsilon"] = settings.propensity_clip
                    results.append(part)
    rows = pd.concat(results, ignore_index=True).sort_values(["policy_name", "outcome", EPISODE, "decision_period"]).reset_index(drop=True)
    groups = ["policy_name", "outcome", EPISODE]
    rows["cumulative_ratio"] = rows.groupby(groups).ratio.cumprod()
    rows["cumulative_ratio_raw"] = rows.groupby(groups).ratio_raw.cumprod()
    if not np.isfinite(rows.cumulative_ratio).all():
        raise ValueError("Cumulative ratio overflow; review support")
    return rows, audit.records, df


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
