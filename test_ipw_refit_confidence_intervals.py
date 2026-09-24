"""Regression checks for nuisance refitting and whole-patient IPW resampling."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import calculate_ipw_refit_confidence_intervals as bootstrap


def example_panel():
    rows, policies = [], []
    for patient in range(20):
        # Unequal episode counts distinguish patient sampling from row sampling.
        for episode in range(2 if patient % 3 == 0 else 1):
            episode_id = patient * 10 + episode
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=episode * 5)
            removal_day = 1 if patient % 3 == 0 else 2
            for day in range(1, 4):
                state = "in" if day <= removal_day else "out"
                removal = int(day == removal_day)
                row = {
                    "subject_id": str(patient), "hadm_id": patient, "stay_id": patient,
                    "inserted": str(start), "removed": str(start + pd.Timedelta(days=removal_day)),
                    "period_start": str(start + pd.Timedelta(days=day - 1)),
                    "period_end": str(start + pd.Timedelta(days=day)),
                    "catheter_state": state, "periods_in_state": day if state == "in" else day - removal_day,
                    "episode_index": episode, "removed_in_period": removal,
                    "observed_action": "remove" if removal else "keep" if state == "in" else "out",
                    "cauti_in_period": int(day == 3 and patient % 4 == 0),
                    "reinsertion_in_period": int(day == 3 and patient % 5 == 0),
                    "death_in_period": int(day == 3 and patient % 6 == 0),
                    "icu_exit_alive_in_period": int(day == 3 and patient % 6 != 0),
                    "at_risk_cauti": 1, "at_risk_reinsertion": int(state == "out"),
                    "episode_end_reason": "icu_exit", "reinsertion_time": np.nan,
                    "age": 30 + patient * 2, "sex_M": patient % 2,
                    "_source_row": len(rows),
                }
                rows.append(row)
                for target_day in (1, 2):
                    applicable = state == "in" and day <= target_day
                    action_remove = int(day == target_day)
                    policies.append({
                        **{column: row[column] for column in bootstrap.ipw.ROW_JOIN_KEY_COLS},
                        "catheter_episode_id": episode_id, "decision_row_id": len(rows) - 1,
                        "episode_day_since_insertion": day,
                        "policy_name": f"remove_on_day_{target_day}",
                        "policy_type": "fixed_day_removal", "policy_remove_day": target_day,
                        "policy_applicable": applicable, "policy_action_remove": action_remove,
                        "policy_matches_observed_action_today": int(action_remove == removal),
                    })
    panel = pd.DataFrame(rows)
    return panel, pd.DataFrame(policies), pd.Index(sorted(panel.subject_id.unique()))


class RefitBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.policy_rows, self.subjects = example_panel()
        self.features = ["episode_index", "periods_in_state", "age", "sex_M"]
        # Exercise actual model fitting cheaply; production uses existing XGBoost.
        self.learner = patch.object(bootstrap.nuisance, "MODEL_TYPE", "logistic_regression")
        self.learner.start()
        self.addCleanup(self.learner.stop)

    def test_each_fold_refits_with_patient_multiplicity_and_no_leakage(self):
        counts = np.ones(len(self.subjects), dtype=int)
        counts[:4] = [3, 0, 0, 1]
        original_fit = bootstrap.nuisance.fit_crossfit_fold_model
        with patch.object(bootstrap.nuisance, "fit_crossfit_fold_model", wraps=original_fit) as fit:
            scores, _ = bootstrap.refit_propensities(self.panel, self.subjects, counts, self.features)
        self.assertEqual(fit.call_count, 5)
        sample = self.panel.loc[self.panel.index.repeat(
            counts[self.subjects.get_indexer(self.panel.subject_id)]
        )].reset_index(drop=True)
        sample = bootstrap.nuisance.add_grouped_crossfit_folds(sample)
        self.assertTrue(sample.groupby("subject_id")._crossfit_fold.nunique().eq(1).all())
        for fold, call in enumerate(fit.call_args_list):
            training = sample.loc[sample.catheter_state.eq("in") & sample._crossfit_fold.ne(fold)]
            held_out = sample.loc[sample._crossfit_fold.eq(fold)]
            self.assertFalse(set(training.subject_id) & set(held_out.subject_id))
            pd.testing.assert_frame_equal(call.args[0], training[self.features])
            pd.testing.assert_series_equal(call.args[1], training.removed_in_period)
        unsampled = counts[self.subjects.get_indexer(self.panel.subject_id)] == 0
        self.assertTrue(scores.loc[unsampled].isna().all().all())
        scored = scores.p_remove_obs.notna()
        np.testing.assert_allclose(scores.loc[scored, ["p_remove_obs", "p_keep_obs"]].sum(axis=1), 1)
        repeated, _ = bootstrap.refit_propensities(self.panel, self.subjects, counts, self.features)
        pd.testing.assert_frame_equal(scores, repeated)

    def test_refitted_weights_match_original_evaluator_on_physical_clusters(self):
        full_scores, _ = bootstrap.refit_propensities(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), self.features,
        )
        policy_rows, episodes, summary = bootstrap.prepare_policy_evaluation(
            self.panel, self.policy_rows, full_scores,
        )
        policies = summary[["policy_name", "policy_remove_day"]]
        counts = np.ones(len(self.subjects), dtype=int)
        counts[:4] = [3, 0, 0, 1]
        scores, _ = bootstrap.refit_propensities(self.panel, self.subjects, counts, self.features)
        shared = full_scores.p_remove_obs.notna() & scores.p_remove_obs.notna()
        self.assertFalse(np.allclose(full_scores.loc[shared, "p_remove_obs"], scores.loc[shared, "p_remove_obs"]))
        result = bootstrap.evaluate_ipw_sample(policy_rows, episodes, self.subjects, counts, scores, policies)

        # Physically copy all rows/episodes of each draw, assigning copy IDs only
        # after fitting so duplicates could not leak across cross-fit folds.
        scored = self.panel.assign(**{column: scores[column] for column in scores}).drop(columns="_source_row")
        copied_panel, copied_policy = [], []
        for patient, count in zip(self.subjects, counts):
            for copy in range(count):
                panel_part = scored.loc[scored.subject_id.eq(patient)].copy()
                policy_part = self.policy_rows.loc[self.policy_rows.subject_id.eq(patient)].copy()
                panel_part["subject_id"] = policy_part["subject_id"] = f"{patient}_copy_{copy}"
                policy_part["catheter_episode_id"] = policy_part.catheter_episode_id.astype(str) + f"_copy_{copy}"
                copied_panel.append(panel_part)
                copied_policy.append(policy_part)
        expanded_panel = pd.concat(copied_panel, ignore_index=True)
        expanded_policy = pd.concat(copied_policy, ignore_index=True)
        joined = bootstrap.ipw.join_nuisance_predictions(expanded_policy, expanded_panel)
        joined = bootstrap.pec.add_period_duration_days(joined, context="test")
        joined = bootstrap.pec.add_observed_icu_exit_alive_period(joined)
        joined = bootstrap.ipw.add_ipw_row_quantities(joined, bootstrap.ipw.CLIP_LOWER, bootstrap.ipw.CLIP_UPPER)
        joined = bootstrap.ipw.add_adherence(joined)
        all_episodes, adherent = bootstrap.ipw.build_policy_episode_panel(joined)
        current = bootstrap.ipw.build_current_practice_episode_panel(expanded_panel, expanded_policy)
        reference = bootstrap.ipw.build_policy_summary(all_episodes, adherent, current).set_index("policy_name")
        columns = [f"ipw_weighted_{name}_risk" for name in list(bootstrap.IPW_COLUMNS)[:-1]]
        columns.append("ipw_weighted_mean_catheter_exposure_days")
        expected = reference.loc[policies.policy_name, columns].to_numpy().ravel()
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(current), sum(counts[self.subjects.get_indexer(
            episodes.loc[episodes.policy_name.eq("current_practice"), "subject_id"]
        )]))

    def test_single_class_resample_uses_training_fallback_and_keeps_zero_support_missing(self):
        scores, _ = bootstrap.refit_propensities(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), self.features,
        )
        rows, episodes, summary = bootstrap.prepare_policy_evaluation(self.panel, self.policy_rows, scores)
        policies = summary[["policy_name", "policy_remove_day"]]
        # All sampled patients remove on day 1: day 2 has no adherent episodes.
        counts = np.array([int(int(patient) % 3 == 0) for patient in self.subjects])
        counts[0] += len(self.subjects) - counts.sum()
        scores, fallback_folds = bootstrap.refit_propensities(self.panel, self.subjects, counts, self.features)
        self.assertEqual(fallback_folds, 5)
        probability = scores.p_remove_obs.dropna()
        self.assertTrue(probability.between(0, 1, inclusive="neither").all())
        values = bootstrap.evaluate_ipw_sample(rows, episodes, self.subjects, counts, scores, policies)
        per_policy = values.reshape(len(policies), len(bootstrap.IPW_COLUMNS))
        self.assertTrue(np.isnan(per_policy[policies.policy_name.eq("remove_on_day_2")]).all())
        self.assertTrue(np.isfinite(per_policy[policies.policy_name.eq("current_practice")]).all())


if __name__ == "__main__":
    unittest.main()
