"""Check refitting, patient clustering, and both evaluator entry-point modes."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import policy_bootstrap as bootstrap
import policy_eval_common as pec
import fit_nuisance_models as nuisance
import evaluate_ipw_policies as ipw
import evaluate_aipw_policies as aipw
import evaluate_gformula_policies as gformula
from panel_run_config import resolve_panel_run
from test_policy_confidence_intervals import SUMMARY_COLUMNS

ESTIMATORS = {"ipw": ipw, "gformula": gformula, "aipw": aipw}


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
                        **{column: row[column] for column in ipw.ROW_JOIN_KEY_COLS},
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
        self.policy_rows["is_decision_row"] = self.policy_rows.catheter_state.eq("in")
        self.policy_rows["policy_action"] = np.where(self.policy_rows.policy_action_remove.eq(1), "remove", "keep")
        self.policy_rows = pec.add_fixed_day_target_policy_timeline(
            self.policy_rows, episode_id_col=ipw.EPISODE_ID_COL,
        )
        self.counts = np.ones(len(self.subjects), dtype=int)
        self.counts[:4] = [3, 0, 0, 1]
        learner = patch.object(nuisance, "MODEL_TYPE", "logistic_regression")
        learner.start()
        self.addCleanup(learner.stop)

    def test_each_needed_model_refits_without_patient_leakage(self):
        sample = nuisance.prepare_outcome_targets(self.panel.copy())
        sample = sample.loc[sample.index.repeat(
            self.counts[self.subjects.get_indexer(sample.subject_id)]
        )].reset_index(drop=True)
        sample = nuisance.add_grouped_crossfit_folds(sample)
        original_fit = nuisance.fit_crossfit_fold_model
        for name, expected_fits in (("ipw", 5), ("gformula", 45), ("aipw", 50)):
            with self.subTest(estimator=name), patch.object(nuisance, "fit_crossfit_fold_model", wraps=original_fit) as fit:
                scores, _ = bootstrap.refit_nuisance_predictions(self.panel, self.subjects, self.counts, name)
                self.assertEqual(fit.call_count, expected_fits)
                self.assertEqual(set(scores.subject_id), set(self.subjects[self.counts > 0]))
                self.assertTrue(scores.groupby("subject_id")._crossfit_fold.nunique().eq(1).all())
                for call in fit.call_args_list:
                    features, target, model_name, fold = call.args
                    state, outcome = model_name.split("_", 1)
                    risk = sample.catheter_state.eq("in") if outcome == "removal" else nuisance.outcome_risk_mask(sample, state, outcome)
                    training = sample.loc[risk & sample._crossfit_fold.ne(fold)]
                    held_out = sample.loc[sample._crossfit_fold.eq(fold)]
                    self.assertFalse(set(training.subject_id) & set(held_out.subject_id))
                    pd.testing.assert_frame_equal(features, training[features.columns])
                    pd.testing.assert_series_equal(target, training[target.name])
                if name != "ipw":
                    self.assertTrue(scores[pec.PREDICTION_COLUMNS].notna().all().all())

    def test_refits_are_reproducible_and_respond_to_patient_multiplicity(self):
        full, _ = bootstrap.refit_nuisance_predictions(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), "aipw",
        )
        sampled, _ = bootstrap.refit_nuisance_predictions(self.panel, self.subjects, self.counts, "aipw")
        repeated, _ = bootstrap.refit_nuisance_predictions(self.panel, self.subjects, self.counts, "aipw")
        pd.testing.assert_frame_equal(sampled, repeated)
        full = full.set_index(ipw.ROW_JOIN_KEY_COLS)
        sampled = sampled.set_index(ipw.ROW_JOIN_KEY_COLS)
        full = full.loc[sampled.index]
        in_rows = sampled.p_remove_obs.notna()
        self.assertFalse(np.allclose(full.loc[in_rows, "p_remove_obs"], sampled.loc[in_rows, "p_remove_obs"]))
        self.assertFalse(np.allclose(full[pec.PREDICTION_COLUMNS], sampled[pec.PREDICTION_COLUMNS]))

    def test_refitted_estimates_match_physical_patient_copies(self):
        for name, module in ESTIMATORS.items():
            with self.subTest(estimator=name):
                scores, _ = bootstrap.refit_nuisance_predictions(self.panel, self.subjects, self.counts, name)
                rows = self.policy_rows.loc[self.policy_rows.subject_id.isin(scores.subject_id)]
                episodes = module.evaluate_policy_episodes(rows, scores)[0]
                policies = episodes[["policy_name", "policy_remove_day"]].drop_duplicates()
                design = bootstrap.build_patient_statistics({name: episodes}, policies, self.subjects)
                actual = design.estimate(self.counts)

                copied_scores, copied_rows = [], []
                for patient, count in zip(self.subjects, self.counts):
                    for copy in range(count):
                        panel_part = scores.loc[scores.subject_id.eq(patient)].copy()
                        rows_part = rows.loc[rows.subject_id.eq(patient)].copy()
                        panel_part["subject_id"] = rows_part["subject_id"] = f"{patient}_copy_{copy}"
                        rows_part["catheter_episode_id"] = rows_part.catheter_episode_id.astype(str) + f"_copy_{copy}"
                        copied_scores.append(panel_part)
                        copied_rows.append(rows_part)
                expanded = module.evaluate_policy_episodes(
                    pd.concat(copied_rows, ignore_index=True), pd.concat(copied_scores, ignore_index=True),
                )
                if name == "ipw":
                    all_episodes, current = expanded[2:]
                    adherent = expanded[0].loc[~expanded[0].policy_name.eq("current_practice")]
                    summary = module.build_policy_summary(all_episodes, adherent, current)
                else:
                    summary = module.build_policy_summary(
                        expanded[0], "hajek" if name == "aipw" else module.PREDICTION_MODE,
                    )
                expected = summary.set_index("policy_name").loc[
                    policies.policy_name, list(SUMMARY_COLUMNS[name].values()),
                ].to_numpy().ravel()
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_single_class_fallback_and_zero_policy_support(self):
        counts = np.array([int(int(patient) % 3 == 0) for patient in self.subjects])
        counts[0] += len(self.subjects) - counts.sum()
        scores, fallbacks = bootstrap.refit_nuisance_predictions(self.panel, self.subjects, counts, "ipw")
        self.assertEqual(fallbacks, 5)
        self.assertTrue(scores.p_remove_obs.dropna().between(0, 1, inclusive="neither").all())
        rows = self.policy_rows.loc[self.policy_rows.subject_id.isin(scores.subject_id)]
        episodes = ipw.evaluate_policy_episodes(rows, scores)[0]
        policies = pd.DataFrame({"policy_name": ["current_practice", "remove_on_day_2"],
                                 "policy_remove_day": [np.nan, 2.0]})
        values = bootstrap.build_patient_statistics({"ipw": episodes}, policies, self.subjects).estimate(counts)
        self.assertTrue(np.isfinite(values[:5]).all())
        self.assertTrue(np.isnan(values[5:]).all())

    def test_counterfactual_cauti_scoring_preserves_factual_nonrisk_zeros(self):
        panel = self.panel.copy()
        panel.loc[panel.catheter_state.eq("out"), "at_risk_cauti"] = 0
        # Keep a training OUT risk set in each fold.
        panel.loc[panel.subject_id.astype(int).mod(2).eq(0), "at_risk_cauti"] = 1
        scores, _ = bootstrap.refit_nuisance_predictions(
            panel, self.subjects, np.ones(len(self.subjects), dtype=int), "gformula",
        )
        nonrisk = scores.catheter_state.eq("out") & scores.at_risk_cauti.eq(0)
        self.assertTrue(scores.loc[nonrisk, "p_cauti_if_out"].eq(0).all())
        self.assertTrue(scores.loc[nonrisk, "p_cauti_if_keep"].notna().all())

    def test_evaluator_entry_points_support_both_modes_and_validation(self):
        expected = {"bootstrap_replicate_estimates.csv", "policy_value_confidence_intervals.csv",
                    "policy_difference_confidence_intervals.csv"}
        full_scores, _ = bootstrap.refit_nuisance_predictions(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), "aipw",
        )
        full_scores = full_scores.drop(columns=[c for c in full_scores if c.startswith("__rescored_")])
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            for panel_name in ("real", "validation", "validation-omitted", "validation-randomised"):
                paths = resolve_panel_run(root, panel_name)
                paths.panel_path.parent.mkdir(parents=True, exist_ok=True)
                self.panel.to_csv(paths.panel_path, index=False)
                policy_path = paths.artefact_root / "policy_interventions" / "policy_intervention_panel_long.csv"
                policy_path.parent.mkdir(parents=True, exist_ok=True)
                self.policy_rows.to_csv(policy_path, index=False)
                prediction_path = paths.artefact_root / "nuisance_models" / "logistic_regression" / "nuisance_predictions.csv"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                for name, module in ESTIMATORS.items():
                    with self.subTest(panel=panel_name, estimator=name), \
                         patch.object(module, "REPO_ROOT", root), \
                         patch.object(module, "NUISANCE_MODEL_TYPE", "logistic_regression"), \
                         patch.object(module, "N_BOOTSTRAP", 2):
                        full_scores.to_csv(prediction_path, index=False)
                        with patch.object(nuisance, "fit_crossfit_fold_model", side_effect=AssertionError("fixed mode must not refit")):
                            module.main(["--panel", panel_name])
                        prediction_path.unlink()
                        # Refit mode must work without any saved predictions/models.
                        module.main(["--panel", panel_name, "--refit-nuisance"])
                        for mode in ("fixed", "refit"):
                            outdir = module.OUTDIR / "confidence_intervals" / mode
                            self.assertEqual({p.name for p in outdir.iterdir()}, expected)
                            draws = pd.read_csv(outdir / "bootstrap_replicate_estimates.csv")
                            self.assertEqual(draws.bootstrap_replicate.tolist(), [1, 2])
                            intervals = pd.read_csv(outdir / "policy_value_confidence_intervals.csv")
                            self.assertEqual(set(intervals.estimator), {name})
                            self.assertEqual(len(intervals), 15)
                            differences = pd.read_csv(outdir / "policy_difference_confidence_intervals.csv")
                            current = differences.loc[differences.policy_name.eq("current_practice")]
                            np.testing.assert_allclose(current[["point_estimate", "ci_lower", "ci_upper"]], 0)
                    module.configure_panel_run("real")
            self.assertFalse(list(root.rglob("*.json")))
            self.assertFalse(list(root.rglob("*.pkl")))


if __name__ == "__main__":
    unittest.main()
