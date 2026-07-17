import unittest

import numpy as np
import pandas as pd

import build_policy_intervention_panels as policy_builder
import evaluate_aipw_policies as aipw
import evaluate_gformula_policies as gformula
import fit_nuisance_models as nuisance
import panel_analysis
import policy_eval_common as policy_common


class DownstreamRemovalAtOutTests(unittest.TestCase):
    def test_nuisance_targets_keep_out_removal_action(self):
        df = pd.DataFrame({
            "catheter_state": ["in", "out", "out"],
            "removed_in_period": [0, 1, 0],
            "cauti_in_period": [0, 0, 0],
            "reinsertion_in_period": [0, 1, 0],
            "death_in_period": [0, 0, 0],
            "icu_end_in_period": [0, 0, 0],
        })
        result = nuisance.prepare_outcome_targets(df)

        self.assertEqual(result["observed_action"].tolist(), ["keep", "remove", "out"])
        self.assertEqual(result.loc[0, "action_remove"], 0)
        self.assertEqual(result.loc[1, "action_remove"], 1)
        self.assertTrue(pd.isna(result.loc[2, "action_remove"]))
        self.assertEqual(result.loc[1, nuisance.Y_NO_EVENT_IN], 0)

    def test_decision_models_do_not_use_reset_out_period_counter(self):
        feature_spec = {
            "x_cols_transition": [
                "episode_index",
                "periods_in_state",
                "state_is_out",
                "action_remove",
                "age",
            ]
        }
        decision_features, out_features = nuisance.state_feature_lists(feature_spec)
        self.assertEqual(decision_features, ["episode_index", "age", "action_remove"])
        self.assertEqual(out_features, ["episode_index", "periods_in_state", "age"])

    def test_policy_timeline_uses_out_remove_then_out(self):
        df = pd.DataFrame({
            "policy_name": ["remove_on_day_2"] * 3,
            "catheter_episode_id": [1] * 3,
            "policy_remove_day": [2] * 3,
            "episode_day_since_insertion": [1, 2, 2],
            "period_start": pd.to_datetime([
                "2162-06-21 06:45", "2162-06-22 06:45", "2162-06-22 11:02"
            ]),
            "period_end": pd.to_datetime([
                "2162-06-22 06:45", "2162-06-22 11:02", "2162-06-22 20:52"
            ]),
            "decision_row_id": [1, 2, 3],
        })
        result = policy_common.add_fixed_day_target_policy_timeline(
            df,
            episode_id_col="catheter_episode_id",
        )

        self.assertEqual(result["policy_catheter_state"].tolist(), ["in", "out", "out"])
        self.assertEqual(result["policy_action_resolved"].tolist(), ["keep", "remove", "out"])
        self.assertEqual(result["policy_action_remove_resolved"].iloc[:2].tolist(), [0.0, 1.0])
        self.assertTrue(pd.isna(result.loc[2, "policy_action_remove_resolved"]))
        self.assertEqual(result["policy_periods_out"].iloc[1:].tolist(), [1.0, 2.0])
        policy_common.validate_resolved_target_policy_timeline(result)

        invalid = result.copy()
        invalid.loc[1, "policy_catheter_state"] = "in"
        with self.assertRaisesRegex(ValueError, "remove action outside an OUT-state row"):
            policy_common.validate_resolved_target_policy_timeline(invalid)

    def test_policy_builder_preserves_exported_decision_rows(self):
        df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "hadm_id": [2, 2, 2],
            "stay_id": [3, 3, 3],
            "inserted": ["2162-06-21 06:45"] * 3,
            "removed": ["2162-06-22 11:02"] * 3,
            "period_start": pd.to_datetime([
                "2162-06-21 06:45", "2162-06-22 06:45", "2162-06-22 11:02"
            ]),
            "period_end": pd.to_datetime([
                "2162-06-22 06:45", "2162-06-22 11:02", "2162-06-22 20:52"
            ]),
            "catheter_state": ["in", "in", "out"],
            "observed_action": ["keep", "keep", "remove"],
            "action_remove": [0, 0, 1],
            "periods_in_state": [1, 2, 1],
            "is_decision_row": [1, 1, 1],
        })
        retained = policy_builder.drop_generated_or_estimator_columns(df)
        self.assertIn("is_decision_row", retained.columns)
        result = policy_builder.add_stable_ids_and_decision_flag(retained)
        self.assertTrue(result["is_decision_row"].all())

        policy = policy_builder.apply_fixed_day_policy(result, 2)
        self.assertEqual(policy["policy_catheter_state"].tolist(), ["in", "out", "out"])
        self.assertEqual(policy["policy_action_resolved"].tolist(), ["keep", "remove", "out"])
        self.assertTrue(policy.loc[1, "policy_applicable"])
        self.assertEqual(policy.loc[1, "policy_matches_observed_action_today"], 0.0)
        self.assertFalse(policy.loc[2, "policy_applicable"])

    def test_evaluators_select_remove_before_generic_out(self):
        base = {
            "policy_catheter_state": ["in", "out", "out"],
            "p_cauti_if_keep": [0.11] * 3,
            "p_cauti_if_remove": [0.22] * 3,
            "p_cauti_if_out": [0.33] * 3,
            "p_reinsertion_if_out": [0.44] * 3,
            "p_death_if_keep": [0.12] * 3,
            "p_death_if_remove": [0.23] * 3,
            "p_death_if_out": [0.34] * 3,
            "p_icu_exit_alive_if_keep": [0.13] * 3,
            "p_icu_exit_alive_if_remove": [0.24] * 3,
            "p_icu_exit_alive_if_out": [0.35] * 3,
            "p_no_event_if_keep": [0.64] * 3,
            "p_no_event_if_remove": [0.31] * 3,
            "p_no_event_if_out": [0.30] * 3,
            "policy_periods_out": [np.nan, 1, 2],
        }

        aipw_df = pd.DataFrame({**base, "policy_action_remove_aipw": [0, 1, np.nan]})
        aipw_result = aipw.select_policy_predictions(aipw_df)
        self.assertEqual(aipw_result.loc[1, "mu_cauti_under_policy"], 0.22)
        self.assertEqual(aipw_result.loc[1, "mu_recatheterisation_under_policy"], 0.44)
        self.assertEqual(aipw_result.loc[2, "mu_cauti_under_policy"], 0.33)

        g_df = pd.DataFrame({**base, "policy_action_remove_gformula": [0, 1, np.nan]})
        g_result = gformula.select_policy_predictions(g_df)
        self.assertEqual(g_result.loc[1, "p_cauti_under_policy"], 0.22)
        self.assertEqual(g_result.loc[1, "p_recatheterisation_under_policy"], 0.44)
        self.assertEqual(g_result.loc[2, "p_cauti_under_policy"], 0.33)

    def test_aipw_bounds_published_risks_and_retains_unbounded_estimates(self):
        policy_df = pd.DataFrame({
            "subject_id": [1],
            "prediction_complete": [True],
            "episode_adherent_to_policy": [1],
            "n_policy_remove_rows": [1],
            "max_episode_day_since_insertion": [3],
            aipw.RESIDUAL_WEIGHT_COL: [1.0],
            "plugin_expected_catheter_in_interval_rows": [2.0],
            "plugin_predicted_any_cauti": [0.8],
            "plugin_predicted_any_recatheterisation": [0.8],
            "plugin_predicted_any_death": [0.8],
            "plugin_predicted_icu_exit_alive": [0.8],
            "plugin_expected_catheter_exposure_days": [2.0],
            "residual_cauti": [0.5],
            "residual_recatheterisation": [0.5],
            "residual_death": [0.5],
            "residual_icu_exit_alive": [0.5],
            "residual_catheter_exposure_days": [1.0],
        })
        summary = aipw.policy_summary_row(
            policy_df,
            "remove_on_day_3",
            "fixed_day_removal",
            3,
            "hajek",
        )

        self.assertEqual(summary["aipw_icu_exit_alive_risk"], 1.0)
        self.assertEqual(summary["aipw_hajek_unbounded_estimate_icu_exit_alive"], 1.3)
        self.assertTrue(summary["aipw_hajek_was_bounded_icu_exit_alive"])
        self.assertEqual(summary["aipw_hajek_mean_catheter_exposure_days"], 3.0)

    def test_panel_analysis_uses_decision_rows_and_pre_removal_period_count(self):
        df = pd.DataFrame({
            "stay_id": [30000213] * 3,
            "inserted": pd.to_datetime(["2162-06-21 06:45"] * 3),
            "period_end": pd.to_datetime([
                "2162-06-22 06:45", "2162-06-22 11:02", "2162-06-22 20:52"
            ]),
            "catheter_state": ["in", "in", "out"],
            "periods_in_state": [1, 2, 1],
            "is_decision_row": [1, 1, 1],
            "cauti_in_period": [0, 0, 0],
            "reinsertion_in_period": [0, 0, 0],
            "is_last_period_of_episode": [0, 0, 1],
            "episode_end_reason": [pd.NA, pd.NA, "icu_end"],
        })
        result = panel_analysis.build_risk_sets(df)
        self.assertEqual(result[panel_analysis.DECISION_PERIOD_COL].tolist(), [1, 2, 2])
        self.assertEqual(result["removal_fit_row"].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
