"""Independent arithmetic, leakage, trajectory and integration tests for CoR B.2."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import create_data_panel as builder
import evaluate_policy_dml_policies as evaluator
import fit_nuisance_models as nuisance
import policy_dml_common as dml
import test_terminal_panel


def example_panel(n=48, hours=24, stages=3):
    episodes = []
    for patient in range(n):
        for episode in range(2 if patient % 4 == 0 else 1):
            t0 = pd.Timestamp("2020-01-01") + pd.Timedelta(days=episode * 10)
            end = t0 + pd.Timedelta(hours=hours * stages)
            remove_stage = patient % stages + 1
            removed = t0 + pd.Timedelta(hours=hours * remove_stage - 1)
            ending = patient % 3
            record = test_terminal_panel.TerminalPanelTests.episode(patient * 10 + episode, inserted=t0, removed=removed,
                reinsertion_time=end if ending == 0 else pd.NaT,
                death_time=end if ending == 1 else pd.NaT,
                icu_out=end + pd.Timedelta(hours=1) if ending != 2 else end,
                cauti_time=min(removed + pd.Timedelta(hours=hours), end) if patient % 5 == 0 else pd.NaT)
            record["subject_id"] = str(patient)
            record["age"] = 40 + patient % 30
            episodes.append(record)
    with patch.object(builder, "PERIOD_DURATION", pd.Timedelta(hours=hours)):
        panel = builder.build_base_panel(builder.add_episode_endpoints(pd.DataFrame(episodes)))
    panel["observed_action"] = np.where(panel.catheter_state.eq("out"), "out", np.where(panel.removed_in_period.eq(1), "remove", "keep"))
    panel["itemid_220045__mean"] = 65 + panel.age % 17
    return panel


def save_behaviour(panel, directory, folds=3):
    """Produce inexpensive saved OOF action predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    frame = nuisance.add_grouped_crossfit_folds(panel.copy(), n_splits=folds)
    features = ["age", "episode_index", "periods_in_state"]
    frame["p_remove_obs"] = np.nan
    frame["p_keep_obs"] = np.nan
    for fold in range(folds):
        training = frame.catheter_state.eq("in") & frame[dml.FOLD].ne(fold)
        held = frame.catheter_state.eq("in") & frame[dml.FOLD].eq(fold)
        model = nuisance.fit_crossfit_fold_model(frame.loc[training, features], frame.loc[training, "removed_in_period"],
                                                "test_behaviour", fold, model_type="logistic_regression")
        p = nuisance.predict_crossfit_fold(model, frame.loc[held, features])
        frame.loc[held, "p_remove_obs"] = p
        frame.loc[held, "p_keep_obs"] = 1 - p
    frame.to_csv(directory / "nuisance_predictions.csv", index=False)
    return frame


class ArithmeticTests(unittest.TestCase):
    def test_two_stage_double_robust_expectation_by_path_enumeration(self):
        # Enumerate every behaviour path independently of fitting. Removal at
        # stage 1 ends decisions and includes its linked reward (.9).
        truth = .3 * .9 + .7 * (.1 + .6 * .8 + .4 * .2)
        for correct_q in (False, True):
            g1, g2 = ((.2, .4) if correct_q else (.4, .7))
            q2 = (.2, .8) if correct_q else (.7, -.2)
            q1 = (.66, .9) if correct_q else (-.3, .5)
            total = 0.
            for a1, a2, mass in [(1, None, .4), (0, 0, .6 * .3), (0, 1, .6 * .7)]:
                if a1:
                    nxt, reward = 0., .9
                else:
                    w2 = (.6 if a2 else .4) / (g2 if a2 else 1 - g2)
                    nxt = dml.recursive_step(.8 if a2 else .2, 0., *q2, a2, .6, w2)[0]
                    reward = .1
                w1 = (.3 if a1 else .7) / (g1 if a1 else 1 - g1)
                total += mass * dml.recursive_step(reward, nxt, *q1, a1, .3, w1)[0]
            self.assertAlmostEqual(total, truth)

    def test_three_stage_hand_recursion(self):
        # Observed KEEP, KEEP, REMOVE; target KEEP, REMOVE, REMOVE.
        # terminal=0; psi3=.2+1.5*(1-.2)=1.4; psi2=.4 (w=0);
        # psi1=.3+2*(.1+.4-.3)=.7. No episode-level residual formula.
        reward = [.1, .2, 1.]
        q0, q1 = [.3, .5, .1], [.2, .4, .2]
        action, pi, weight = [0, 0, 1], [0, 1, 1], [2, 0, 1.5]
        result = []
        nxt = 0.
        for i in (2, 1, 0):
            nxt = dml.recursive_step(reward[i], nxt, q0[i], q1[i], action[i], pi[i], weight[i])[0]
            result.append(float(nxt))
        np.testing.assert_allclose(result, [1.4, .4, .7])

    def test_two_stage_coherent_hand_recursion(self):
        # Observed KEEP then REMOVE, policy KEEP then REMOVE, g=(.5,.25).
        q0, q1 = [.4, .2], [.3, .6]
        last = dml.recursive_step(1., 0., q0[1], q1[1], 1, 1., 4.)[0]
        first = dml.recursive_step(.1, last, q0[0], q1[0], 0, 0., 2.)[0]
        self.assertAlmostEqual(last, 2.2)
        self.assertAlmostEqual(first, 4.2)

    def test_single_decision_and_stochastic_mixture(self):
        g = .4
        pi = .3
        ratio = dml.ratio_quantities([1], [pi], [g])["ratio"][0]
        score, qbar, qobs = dml.recursive_step(1., 0., .2, .8, 1, pi, ratio)
        self.assertAlmostEqual(qbar, .38)
        self.assertAlmostEqual(qobs, .8)
        self.assertAlmostEqual(score, .53)

    def test_deterministic_ratios_clipping_and_invalid_probabilities(self):
        q = dml.ratio_quantities([0, 1, 1], [0, 0, 1], [.25, .25, .01], .05)
        np.testing.assert_allclose(q["ratio"], [1 / .75, 0, 20])
        np.testing.assert_allclose(q["ratio_raw"], [1 / .75, 0, 100])
        self.assertEqual(q["propensity_clipped"].tolist(), [False, False, True])
        for pi, g in [([1.1], [.4]), ([.2], [-.1]), ([np.nan], [.4]), ([1], [0])]:
            with self.assertRaises(ValueError):
                dml.ratio_quantities([1], pi, g)
        self.assertAlmostEqual(dml.effective_sample_size([0, 1, 3]), 1.6)
        self.assertEqual(dml.effective_sample_size([0, 0]), 0)
        with self.assertRaises(ValueError):
            dml.Policy("missing_day", "fixed_day_removal")
        with self.assertRaises(ValueError):
            dml.Policy("bad_probability", "stochastic", remove_probability=1.1)


class PanelTests(unittest.TestCase):
    def test_linked_outcomes_terminal_removal_and_exposure(self):
        panel = example_panel()
        decisions, links = dml.prepare_decisions(panel)
        for outcome, col in dml.OUTCOME_COLUMNS.items():
            if outcome == "catheter_exposure_days":
                expected = ((panel.period_end - panel.period_start).dt.total_seconds() / 86400).where(panel.catheter_state.eq("in"), 0).sum()
            else:
                expected = panel[col].sum()
            self.assertAlmostEqual(decisions[f"reward_{outcome}"].sum(), expected)
        remove = decisions.removed_in_period.eq(1)
        self.assertTrue(decisions.loc[remove, "next_decision_row_id"].isna().all())
        self.assertGreater(decisions.n_linked_out_rows.sum(), 0)
        self.assertTrue(links.loc[~links.is_decision_row, "contribution_decision_row_id"].isin(decisions.loc[remove, dml.ROW]).all())
        self.assertTrue(decisions.loc[remove, "reward_cauti"].gt(0).any())

    def test_non_24_hour_grid_and_explicit_horizon(self):
        panel = example_panel(hours=12)
        decisions, _ = dml.prepare_decisions(panel)
        self.assertAlmostEqual(decisions.reward_catheter_exposure_days.iloc[0], .5)
        selected, _ = dml.prepare_decisions(panel, dml.DMLSettings(horizon_hours=24))
        self.assertTrue(selected.decision_period.le(2).all())
        self.assertTrue(selected.loc[selected.decision_period.eq(2), "next_decision_row_id"].isna().all())
        with self.assertRaisesRegex(ValueError, "cuts a period"):
            dml.prepare_decisions(panel, dml.DMLSettings(horizon_hours=13))
        self.assertEqual(len(dml.configured_policies([1, 2])), 3)

    def test_cor_cauti_window_is_hours_not_number_of_grid_periods(self):
        panel = example_panel(hours=12, stages=5)
        with self.assertRaisesRegex(ValueError, "48-hour"):
            dml.prepare_decisions(panel)
        keys = ["subject_id", "stay_id", "inserted"]
        prior = panel.groupby(keys).cauti_in_period.cumsum() - panel.cauti_in_period
        panel["at_risk_cauti"] = (panel.period_start.lt(panel.removed + pd.Timedelta(hours=48)) & prior.eq(0)).astype(int)
        decisions, _ = dml.prepare_decisions(panel)
        self.assertTrue(decisions.reward_catheter_exposure_days.eq(.5).all())

    def test_bad_order_missing_link_duplicate_and_after_removal(self):
        panel = example_panel()
        mutations = []
        broken = panel.copy(); broken.loc[1, "episode_index"] = 7; mutations.append(broken)
        mutations.append(panel.drop(index=1))
        mutations.append(pd.concat([panel, panel.iloc[[0]]]))
        broken = panel.copy(); broken.loc[1, "catheter_state"] = "in"; broken.loc[1, "observed_action"] = "keep"; broken.loc[1, "is_decision_row"] = True; mutations.append(broken)
        for broken in mutations:
            with self.assertRaises(ValueError):
                dml.prepare_decisions(broken)

    def test_censoring_and_risk_sets_fail_loudly(self):
        panel = example_panel()
        panel["censored_in_period"] = 0
        panel.loc[0, "censored_in_period"] = 1
        with self.assertRaises(NotImplementedError):
            dml.prepare_decisions(panel)
        panel = example_panel()
        panel.loc[panel.cauti_in_period.eq(1), "at_risk_cauti"] = 0
        with self.assertRaises(ValueError):
            dml.prepare_decisions(panel)

    def test_inconsistent_folds(self):
        panel = example_panel()
        decisions, _ = dml.prepare_decisions(panel)
        folded = nuisance.add_grouped_crossfit_folds(decisions.copy(), n_splits=3)
        patient = folded.subject_id.iloc[0]
        folded.loc[folded.subject_id.eq(patient).idxmax(), dml.FOLD] = (folded.loc[0, dml.FOLD] + 1) % 3
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            dml.validate_folds(folded)


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = example_panel()
        cls.decisions, _ = dml.prepare_decisions(cls.panel)
        cls.policies = dml.configured_policies([1, 2])
        cls.settings = dml.DMLSettings(n_folds=3, inner_folds=2, continuation_learner="ridge")
        cls.rows, cls.audit, cls.folded = dml.crossfit_policy_scores(cls.decisions, cls.policies, settings=cls.settings)

    def test_all_model_membership_is_patient_disjoint(self):
        for model in self.audit:
            self.assertNotIn(model["outer_fold"], model["training_folds"])
            self.assertEqual(model["outer_fold"], model["held_out_fold"])
        self.assertTrue(self.folded.groupby("subject_id")[dml.FOLD].nunique().eq(1).all())
        self.assertTrue(self.rows.continuation_model_id.isin([m["model_id"] for m in self.audit]).all())

    def test_unsupported_late_period_fails_before_fitting(self):
        sparse = self.decisions.copy()
        sparse.loc[sparse.index[-1], "decision_period"] = 99
        with patch.object(dml, "fit_behaviour") as fit:
            with self.assertRaisesRegex(ValueError, "No training trajectories.*99"):
                dml.crossfit_policy_scores(sparse, self.policies, settings=self.settings)
        fit.assert_not_called()

    def test_heldout_rewards_cannot_affect_own_q_even_indirectly(self):
        mutated = self.folded.copy()
        fold = mutated[dml.FOLD].iloc[0]
        mutated.loc[mutated[dml.FOLD].eq(fold), "reward_death"] += 1000
        scores, _, _ = dml.crossfit_policy_scores(mutated, self.policies, settings=self.settings)
        columns = ["q_keep", "q_remove", "g_remove"]
        pd.testing.assert_frame_equal(self.rows.loc[self.rows[dml.FOLD].eq(fold), columns].reset_index(drop=True),
                                      scores.loc[scores[dml.FOLD].eq(fold), columns].reset_index(drop=True))

    def test_continuation_target_contains_next_recursive_score(self):
        seen = []
        original = dml.fit_continuation
        def inspect(train, held, target, *args, **kwargs):
            component = args[-1] if isinstance(args[-1], str) else kwargs.get("component", "continuation")
            if component == "continuation":
                outcome = args[1]
                expected = train[f"reward_{outcome}"].to_numpy() + np.where(train.next_decision_row_id.notna(), 10., 0.)
                np.testing.assert_allclose(target, expected)
                seen.append(True)
            return original(train, held, target, *args, **kwargs)
        def inner_scores(train, held, *args, **kwargs):
            return np.full((len(held), len(dml.OUTCOMES)), 10.)
        with patch.object(dml, "fit_backward_sequence", side_effect=inner_scores), patch.object(dml, "fit_continuation", side_effect=inspect):
            dml.crossfit_policy_scores(self.decisions, self.policies, settings=self.settings)
        self.assertTrue(seen)

    def test_reproducibility_recursion_and_episode_averaging(self):
        again, _, _ = dml.crossfit_policy_scores(self.decisions, self.policies, settings=self.settings)
        pd.testing.assert_frame_equal(self.rows, again)
        episodes = dml.episode_scores(self.rows)
        summary = evaluator.policy_summary(episodes, self.decisions)
        baseline = summary.loc[summary.policy_name.eq("current_practice"), "predicted_cauti_risk"].iloc[0]
        np.testing.assert_allclose(summary.cauti_risk_difference_vs_current_practice, summary.predicted_cauti_risk - baseline)
        self.assertEqual(len(episodes), self.decisions[dml.EPISODE].nunique() * len(self.policies))
        self.assertTrue((self.rows.recursive_score > 1).any())
        # Cumulative weights are products, not the decision ratios used by psi.
        for _, group in self.rows.groupby(["policy_name", "outcome", dml.EPISODE]):
            np.testing.assert_allclose(group.cumulative_ratio, np.cumprod(group.ratio))

    def test_fixed_cluster_bootstrap_matches_physical_patient_expansion(self):
        episodes = dml.episode_scores(self.rows)
        subjects, entries, stats = evaluator.patient_statistics(episodes)
        counts = np.arange(len(subjects)) % 3
        estimate = evaluator.estimate_from_counts(stats, counts)
        expanded = episodes.loc[episodes.index.repeat(episodes.subject_id.map(pd.Series(counts, index=subjects)))].copy()
        actual = [expanded.loc[expanded.policy_name.eq(e.policy_name), f"score_{e.outcome}"].mean() for e in entries.itertuples()]
        np.testing.assert_allclose(estimate, actual)

    def test_refit_bootstrap_retains_all_patient_episodes(self):
        episodes = dml.episode_scores(self.rows)
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            output = Path(temporary)
            estimates = evaluator.run_bootstrap(episodes, self.folded, self.policies, output, mode="refit", settings=self.settings,
                                                learner="logistic_regression", n_bootstrap=1, seed=14)
            self.assertTrue(np.isfinite(estimates).all())
            intervals = pd.read_csv(output / "policy_dml_policy_value_confidence_intervals.csv")
            self.assertTrue(intervals.nuisance_refitting_uncertainty.all())

    def test_stochastic_policy_runs_same_equation(self):
        scores, _, _ = dml.crossfit_policy_scores(self.decisions, [dml.Policy("coin", "stochastic", remove_probability=.3)], settings=self.settings)
        self.assertTrue(scores.pi_remove.eq(.3).all())
        np.testing.assert_allclose(scores.qbar, .7 * scores.q_keep + .3 * scores.q_remove)

    def test_saved_behaviour_reuse_and_probability_validation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_behaviour(self.panel, root)
            cached, provenance = evaluator.load_saved_behaviour(self.decisions, root)
            scores, audit, _ = dml.crossfit_policy_scores(self.decisions, self.policies, settings=self.settings, saved_behaviour=cached)
            self.assertEqual(sum(m["component"] == "reused_behaviour" for m in audit), 3)
            self.assertEqual(sum(m["component"] == "behaviour" and m["inner_fold"] is None for m in audit), 0)
            self.assertEqual(provenance["held_out_folds"], [0, 1, 2])
            cached.loc[0, "p_remove_obs"] = np.nan
            with self.assertRaises(ValueError):
                dml.crossfit_policy_scores(self.decisions, self.policies, settings=self.settings, saved_behaviour=cached)

    def test_end_to_end_output_schema(self):
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            source = root / "panel.csv"
            self.panel.to_csv(source, index=False)
            save_behaviour(self.panel, root / "nuisance_models/logistic_regression")
            for mode in ("fixed", "refit"):
                summary, episodes, rows = evaluator.run_panel_estimation(source, root, "logistic_regression", mode,
                    settings=self.settings, n_bootstrap=1)
                output = root / "policy_eval/policy_dml" / mode
                self.assertTrue(set(evaluator.SUMMARY_COLUMNS.values()).issubset(summary.columns))
                self.assertTrue((output / "confidence_intervals/policy_dml_policy_difference_confidence_intervals.csv").exists())
                self.assertFalse((root / "policy_eval/gformula").exists())


if __name__ == "__main__":
    unittest.main()
