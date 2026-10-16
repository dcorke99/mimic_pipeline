"""Tests for panel semantics shared by nuisance fitting and policy estimators."""

from unittest.mock import Mock, patch
import unittest

import numpy as np
import pandas as pd

import create_data_panel as builder
import create_semi_synthetic_panel as synthetic
import fit_nuisance_models as nuisance
import panel_analysis as analysis
import policy_eval_common as pec
from test_base_panel import episode


def example_panel():
    episodes = builder.add_episode_endpoints(pd.DataFrame([
        episode(1, removal=38, terminal=42, reason="reinsertion"),
        episode(2, removal=38, terminal=110, reason="reinsertion"),
        episode(3, removal=48, terminal=60, reason="reinsertion"),
        episode(4, removal=96, terminal=60, reason="death"),
        episode(5, removal=38, terminal=40, reason="reinsertion"),
        episode(6, removal=50, terminal=90, reason="icu_exit_alive"),
    ]))
    panel = builder.build_base_panel(episodes)
    panel["subject_id"] = panel.subject_id.astype(str)
    panel["age"] = 30 + pd.to_numeric(panel.subject_id) * 5
    panel["itemid_220045__mean"] = 70 + panel.episode_index * 2
    panel["observed_action"] = np.where(
        panel.catheter_state.eq("out"), "out",
        np.where(panel.removed_in_period.eq(1), "remove", "keep"),
    )
    return panel


class RestructuredPanelDownstreamTests(unittest.TestCase):
    def test_reinsertion_risk_and_no_event_include_same_period_events(self):
        panel = nuisance.prepare_outcome_targets(example_panel())
        risk = nuisance.outcome_risk_mask(panel, "out", "reinsertion")
        self.assertTrue(risk.equals(panel.at_risk_reinsertion.eq(1)))
        self.assertEqual(int(panel.loc[risk, "reinsertion_in_period"].sum()), 4)
        removal = panel.catheter_state.eq("in") & panel.reinsertion_in_period.eq(1)
        self.assertEqual(int(removal.sum()), 2)
        self.assertTrue(panel.loc[removal, nuisance.Y_NO_EVENT_IN].eq(0).all())
        self.assertTrue(panel.loc[removal, pec.REINSERTION_CONTEXT_COL].eq(1).all())
        self.assertTrue(panel.loc[removal, pec.REINSERTION_PERIOD_COL].eq(0).all())
        analysed = analysis.build_risk_sets(panel)
        self.assertEqual(int(analysed.loc[analysed.reinsertion_fit_row.eq(1), "reinsertion_in_period"].sum()), 4)

    def test_policy_reinsertion_features_use_the_candidate_timeline(self):
        panel = nuisance.prepare_outcome_targets(example_panel())
        columns = pec.reinsertion_feature_columns(["episode_index", "periods_in_state", "age"])

        def predict(model, features):
            self.assertNotIn("catheter_out_hours", features)
            self.assertNotIn("periods_in_state", features)
            return np.where(features[pec.REINSERTION_CONTEXT_COL].eq(1), 0.4,
                            0.1 + features[pec.REINSERTION_PERIOD_COL] * 0.05)

        scores = pec.predict_reinsertion_columns({}, panel, columns, predictor=predict)
        for column, values in scores.items():
            panel[column] = values
        first = panel.loc[panel.subject_id.eq("1") & panel.episode_day.eq(1)].iloc[0]
        second = panel.loc[panel.subject_id.eq("1") & panel.episode_day.eq(2)].iloc[0]
        self.assertEqual(first.at_risk_reinsertion, 0)
        self.assertEqual(first[f"{pec.REINSERTION_POLICY_PREFIX}1"], 0.4)
        self.assertAlmostEqual(second[f"{pec.REINSERTION_POLICY_PREFIX}1"], 0.15)

    def test_actual_exposure_uses_catheter_hours(self):
        panel = example_panel()
        actual = pec.observed_catheter_exposure_days(panel)
        self.assertAlmostEqual(actual.loc[panel.subject_id.eq("1")].sum(), 38 / 24)
        self.assertTrue(actual.loc[panel.catheter_state.eq("out")].eq(0).all())

    def test_synthetic_state_metadata_follows_generated_removal(self):
        source = example_panel()
        z_values, standardisation = synthetic.build_standardised_confounders(
            source, source.catheter_state.eq("in"),
        )
        panel, _, _ = synthetic.generate_sequential_trajectory(
            source, z_values, standardisation, np.random.default_rng(123),
            randomised=True, target_prevalence=0.9,
        )
        self.assertTrue(panel.removed_in_period.eq(1).any())
        self.assertTrue(panel.loc[panel.removed_in_period.eq(1), "at_risk_reinsertion"].eq(1).all())
        np.testing.assert_allclose(panel.catheter_in_hours, pec.observed_catheter_exposure_days(panel) * 24)
        np.testing.assert_allclose(panel.catheter_in_hours + panel.catheter_out_hours, panel.interval_hours)

    def test_model_fitting_reads_saved_reinsertion_features(self):
        panel = example_panel()
        panel[pec.REINSERTION_PERIOD_COL] = 7
        with patch.object(pec, "add_reinsertion_context", side_effect=AssertionError("Must read saved features")):
            prepared = nuisance.prepare_outcome_targets(panel.copy())
            seen = []

            def predict(model, features):
                seen.append(features.copy())
                return np.full(len(features), 0.1)

            pec.predict_reinsertion_columns({}, prepared, pec.reinsertion_feature_columns(["age"]), predictor=predict)
            self.assertTrue(seen[0][pec.REINSERTION_PERIOD_COL].eq(7).all())

    def test_clipped_period_without_removal_has_zero_out_time(self):
        source = example_panel()
        for subject, seconds in (("1", 847 * 60 + 48), ("2", 13 * 60 + 31)):
            last = source.loc[source.subject_id.eq(subject)].index[-1]
            end = source.loc[last, "period_start"] + pd.Timedelta(seconds=seconds)
            source.loc[last, "period_end"] = end
            source.loc[source.subject_id.eq(subject), "episode_end_time"] = end
        z_values, standardisation = synthetic.build_standardised_confounders(
            source, source.catheter_state.eq("in"),
        )
        panel, _, _ = synthetic.generate_sequential_trajectory(
            source, z_values, standardisation, Mock(binomial=Mock(return_value=0)),
            randomised=True, target_prevalence=0.5,
        )
        self.assertTrue(panel.removed_in_period.eq(0).all())
        self.assertTrue(panel.catheter_out_hours.eq(0).all())
        self.assertTrue(panel.at_risk_reinsertion.eq(0).all())


if __name__ == "__main__":
    unittest.main()
