"""Expected first-CAUTI risk must not stop at the realised first event."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import build_policy_panels as policies
import create_observed_panel as observed
import create_semi_synthetic_panel as synthetic
import evaluate_gformula_policies as gformula
import fit_nuisance_models as nuisance
import policy_bootstrap_common as bootstrap
import policy_eval_common as pec
from test_base_panel import episode


def example():
    # One patient has two episodes, including an early event followed by both
    # IN and OUT time. Other episodes end before removal / during attribution.
    cases = [episode(1, removal=38, terminal=120, cauti=12),
             episode(2, removal=48, terminal=120, cauti=60),
             episode(3, removal=96, terminal=60, reason="death", cauti=12),
             episode(4, removal=38, terminal=42, reason="reinsertion")]
    cases[2]["subject_id"] = 1
    panel = observed.build_base_panel(observed.add_episode_endpoints(pd.DataFrame(cases)))
    panel["subject_id"] = panel.subject_id.astype(str)
    panel["age"] = 40 + panel.stay_id
    panel[synthetic.HEART_RATE_MEAN_COL] = 60 + panel.episode_index
    panel = nuisance.prepare_outcome_targets(panel)
    panel = nuisance.add_grouped_crossfit_folds(panel, n_splits=2)
    base = policies.add_stable_ids_and_decision_flag(pec.add_episode_day(panel.copy()))
    policy_rows = pd.concat([policies.apply_fixed_day_policy(base, day) for day in (1, 3)], ignore_index=True)
    policy_rows = policy_rows[[*policies.POLICY_OUTPUT_COLS, "policy_name", "policy_type", "policy_remove_day"]]
    return panel, policy_rows


def payload():
    features = ["episode_index", "periods_in_state", "age"]
    models = {"action_remove_col": nuisance.ACTION_COL,
              "x_cols_in": [*features, nuisance.ACTION_COL], "x_cols_out": features}
    for state, outcomes in (("in", nuisance.IN_OUTCOMES), ("out", nuisance.OUT_OUTCOMES)):
        models[f"{state}_models"] = {}
        for outcome in outcomes:
            probability = (.2 if state == "in" else .1) if outcome == "cauti" else .03
            columns = ([*features, nuisance.ACTION_COL] if state == "in" else
                       pec.reinsertion_feature_columns(features) if outcome == "reinsertion" else features)
            models[f"{state}_models"][outcome] = {
                "features": columns,
                "fold_models": [{"fold": fold, "fallback": True,
                                 "fallback_probability": probability + .01 * fold}
                                for fold in range(2)],
            }
    return models


def saved_scores(panel):
    scored = panel.copy()
    for column in [*pec.PREDICTION_COLUMNS, *pec.reinsertion_policy_columns()]:
        scored[column] = .03
    for suffix in ("keep", "remove", "out"):
        scored[f"p_cauti_if_{suffix}"] = np.where(scored.at_risk_cauti.eq(1), .2, 0.)
    return pec.add_policy_predictions(scored, payload())


class GformulaCurrentPracticeTests(unittest.TestCase):
    def test_held_out_scoring_keeps_factual_features_and_replaces_masked_zeros(self):
        panel, _ = example()
        before = saved_scores(panel)
        seen = []

        def predict(model, inputs):
            pd.testing.assert_frame_equal(inputs, panel.loc[inputs.index, inputs.columns])
            self.assertTrue(panel.loc[inputs.index, "_crossfit_fold"].eq(model["fold"]).all())
            seen.extend(inputs.index)
            # Action and timing affect the score, so silently overriding them
            # with a candidate policy would change the expected result.
            return .01 * inputs.age + .01 * inputs.episode_index

        actual = pec.add_current_practice_cauti_predictions(before, payload(), predictor=predict)
        eligible = pec.current_practice_cauti_window(panel)
        self.assertCountEqual(seen, panel.index[eligible])
        self.assertEqual(len(seen), int(eligible.sum()))
        post_event = eligible & panel.at_risk_cauti.eq(0)
        self.assertTrue(post_event.any())
        self.assertTrue(actual.loc[post_event, pec.CURRENT_PRACTICE_CAUTI_COL].gt(0).all())
        pd.testing.assert_frame_equal(actual[before.columns], before)

    def test_current_expected_risk_matches_unchanged_known_truth(self):
        panel, policy_rows = example()
        scored = pec.add_current_practice_cauti_predictions(saved_scores(panel), payload())
        for column in pec.PREDICTION_COLUMNS:
            scored[f"__rescored_{column}"] = False
        episodes, fixed_rows, current_rows, current = gformula.evaluate_policy_episodes(policy_rows, scored)
        _, standardisation = synthetic.build_standardised_confounders(panel, panel.catheter_state.eq("in"))
        truth = pd.DataFrame({"validation_row_id": np.arange(1, len(panel) + 1),
                              "true_mu_keep": .2 + .01 * panel._crossfit_fold,
                              "true_mu_remove": .2 + .01 * panel._crossfit_fold,
                              "true_mu_out": .1 + .01 * panel._crossfit_fold,
                              "true_propensity_remove": .3})
        known = synthetic.build_known_truth_policy_values(panel, truth, standardisation)
        reference = known.loc[known.policy_name.eq("current_practice"),
                              "known_truth_mean_episode_any_cauti_risk"].iloc[0]
        self.assertAlmostEqual(current.predicted_any_cauti.mean(), reference)
        self.assertTrue(episodes.prediction_complete.all())
        expected = pec.current_practice_cauti_window(current_rows)
        self.assertEqual(current_rows.policy_at_risk_cauti.tolist(), expected.astype(int).tolist())
        self.assertTrue(current_rows.loc[~expected, "p_cauti_under_policy"].eq(0).all())
        # Observed first events, their stopped risk sets, and crude outcomes survive.
        self.assertEqual(int(current.observed_any_cauti.sum()), 3)
        pd.testing.assert_series_equal(current_rows.at_risk_cauti, panel.at_risk_cauti)
        self.assertTrue(panel.groupby("stay_id").cauti_in_period.sum().le(1).all())
        # The opt-in changes neither fixed-day rows nor legacy helper semantics.
        legacy = fixed_rows.copy()
        pec.select_policy_outcomes(legacy, "p")
        pd.testing.assert_frame_equal(legacy[gformula.UNDER_POLICY_COLUMNS],
                                      fixed_rows[gformula.UNDER_POLICY_COLUMNS])
        legacy_current = gformula.build_current_practice_rows(scored, policy_rows)
        pec.select_policy_outcomes(legacy_current, "p")
        self.assertLess(pec.cumulative_event_probability(legacy_current.p_cauti_under_policy),
                        pec.cumulative_event_probability(current_rows.p_cauti_under_policy))

    def test_window_uses_elapsed_hours_not_out_period_bins_or_infection(self):
        start = pd.Timestamp("2024-01-01")
        panel = pd.DataFrame({"period_start": [start + pd.Timedelta(hours=h) for h in (0, 24, 48, 60, 72)],
                              "removed": start + pd.Timedelta(hours=12), "at_risk_cauti": 0,
                              "periods_in_state": 1})
        self.assertEqual(pec.current_practice_cauti_window(panel).tolist(), [True, True, True, False, False])

    def test_complete_legacy_saved_predictions_still_rescore_current_practice(self):
        panel, _ = example()
        before = saved_scores(panel)
        self.assertFalse(before[pec.policy_prediction_columns()].isna().any().any())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "outcome_models.pkl"
            joblib.dump(payload(), path)
            actual = pec.fill_missing_counterfactual_predictions(
                before.copy(), path, current_practice_expected_risk=True,
            )
        expected = pec.add_current_practice_cauti_predictions(before, payload())
        pd.testing.assert_series_equal(actual[pec.CURRENT_PRACTICE_CAUTI_COL],
                                       expected[pec.CURRENT_PRACTICE_CAUTI_COL])
        pd.testing.assert_frame_equal(actual[before.columns], before)

    def test_missing_fold_or_patient_in_multiple_folds_fails(self):
        panel, _ = example()
        panel.loc[0, "_crossfit_fold"] = 99
        with self.assertRaisesRegex(ValueError, "one fold per patient"):
            pec.add_current_practice_cauti_predictions(panel, payload())
        panel["_crossfit_fold"] = 99
        with self.assertRaisesRegex(ValueError, "Missing or invalid held-out"):
            pec.add_current_practice_cauti_predictions(panel, payload())

    def test_refit_bootstrap_preserves_training_risk_and_patient_resampling(self):
        panel, policy_rows = example()
        subjects = pd.Index(sorted(panel.subject_id.unique()))
        fits = []

        def fit(features, target, name, fold, **kwargs):
            groups = kwargs["groups"]
            fits.append((name, fold, list(groups)))
            # Recover source rows using factual features, including the target
            # patient's second episode. Every CAUTI training row is pre-event.
            if name.endswith("cauti"):
                source = panel.merge(features.drop_duplicates(), on=list(features.columns))
                self.assertTrue(source.at_risk_cauti.eq(1).all())
            return {"fold": fold, "fallback": True,
                    "fallback_probability": (.2 if name.startswith("in") else .1)}

        with patch.object(nuisance, "fit_crossfit_fold_model", side_effect=fit):
            scored, _ = bootstrap.refit_nuisance_predictions(
                panel, subjects, np.array([2, 1, 1]), n_splits=2,
            )
            episodes, _, current_rows, current = gformula.evaluate_policy_episodes(policy_rows, scored)
            eligible = pec.current_practice_cauti_window(scored)
            post_event = eligible & scored.at_risk_cauti.eq(0)
            self.assertTrue(scored.loc[post_event, pec.CURRENT_PRACTICE_CAUTI_COL].gt(0).all())
            self.assertTrue(scored.groupby("subject_id")._crossfit_fold.nunique().eq(1).all())
            for _, fold, groups in fits:
                held_subjects = scored.loc[scored._crossfit_fold.eq(fold), "subject_id"]
                self.assertFalse(set(groups) & set(held_subjects))
            for _, group in current_rows.groupby("catheter_episode_id"):
                hazards = np.where(pec.current_practice_cauti_window(group),
                                   np.where(group.catheter_state.eq("in"), .2, .1), 0.)
                risk = 1 - np.prod(1 - hazards)
                actual = current.loc[current.catheter_episode_id.eq(group.catheter_episode_id.iloc[0])]
                self.assertAlmostEqual(actual.predicted_any_cauti.iloc[0], risk)
            # Fixed mode resamples patient sums of corrected episodes; refit
            # mode calls the same corrected evaluator for every nuisance draw.
            with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
                for mode in ("fixed", "refit"):
                    outdir = Path(directory) / mode
                    bootstrap.run_bootstrap(
                        episodes, policy_rows, gformula.evaluate_policy_episodes, outdir,
                        2, refit_panel=panel if mode == "refit" else None,
                        refit_n_splits=2, seed=42,
                    )
                    values = pd.read_csv(outdir / "policy_values.csv")
                    baseline = values.loc[values.policy_name.eq("current_practice") & values.outcome.eq("cauti")]
                    self.assertAlmostEqual(baseline.point_estimate.iloc[0], current.predicted_any_cauti.mean(), places=4)


if __name__ == "__main__":
    unittest.main()
