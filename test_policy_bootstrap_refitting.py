"""Check refitting, patient clustering, and both evaluator entry-point modes."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import policy_bootstrap_common as bootstrap
import policy_eval_common as pec
import fit_nuisance_models as nuisance
import build_policy_panels as policies
import evaluate_ipw_policies as ipw
import evaluate_aipw_policies as aipw
import evaluate_gformula_policies as gformula
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
                        "episode_day": day,
                        "policy_name": f"remove_on_day_{target_day}",
                        "policy_type": "fixed_day_removal", "policy_remove_day": target_day,
                        "policy_applicable": applicable, "policy_action": "remove" if action_remove else "keep",
                        "policy_matches_observed_action_today": int(action_remove == removal),
                    })
    panel = pd.DataFrame(rows)
    return panel, pd.DataFrame(policies), pd.Index(sorted(panel.subject_id.unique()))


class RefitBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.policy_rows, self.subjects = example_panel()
        self.policy_rows["is_decision_row"] = self.policy_rows.catheter_state.eq("in")
        self.policy_rows = pec.add_fixed_day_target_policy_timeline(
            self.policy_rows, episode_id_col=ipw.EPISODE_ID_COL,
        )
        self.counts = np.ones(len(self.subjects), dtype=int)
        self.counts[:4] = [3, 0, 0, 1]
        # Direct helper tests use the logistic-regression configuration.
        names = ("MODEL_TYPE", "MODEL_OUTPUT_NAME", "INFILE", "NUISANCE_ROOT", "OUTDIR",
                 "NUISANCE_PREDICTIONS_FILE", "PERFORMANCE_METRICS_FILE",
                 "CROSSFIT_ROW_ASSIGNMENTS_FILE", "CONSTANT_FEATURES_FILE", "MODEL_COMPARISON_FILE")
        settings = {name: getattr(nuisance, name) for name in names}
        settings["MODEL_TYPE"] = "logistic_regression"
        learner = patch.multiple(nuisance, **settings)
        learner.start()
        self.addCleanup(learner.stop)
        policy_names = ("INPUT_PATH", "OUTDIR", "POLICY_MANIFEST_PATH", "CHECK_OUTPUT_PATH")
        policy_paths = patch.multiple(policies, **{name: getattr(policies, name) for name in policy_names})
        policy_paths.start()
        self.addCleanup(policy_paths.stop)

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

    def test_explicit_policy_csv_preserves_timing_and_estimates(self):
        panel = self.panel.copy()
        # Split a removal day into two intervals: only the first can receive
        # the target removal action, even when observed removal is later.
        extra = panel.iloc[[0]].copy()
        midpoint = str(pd.Timestamp(extra.period_start.iloc[0]) + pd.Timedelta(hours=12))
        panel.loc[0, "period_end"] = midpoint
        panel.loc[0, "removed_in_period"] = 0
        panel.loc[0, "observed_action"] = "keep"
        extra["period_start"] = midpoint
        panel = pd.concat([panel, extra], ignore_index=True)
        base = policies.add_stable_ids_and_decision_flag(panel[policies.POLICY_INPUT_COLS].copy())
        base = pec.add_episode_day(base)
        original = policies.build_long_policy_panel(base, [1, 2, 5])
        scores, _ = bootstrap.refit_nuisance_predictions(
            panel, self.subjects, np.ones(len(self.subjects), dtype=int), "aipw",
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "policies.csv"
            single_policy = original.loc[original.policy_name.eq("remove_on_day_1")].reset_index(drop=True)
            single_policy.to_csv(path, columns=policies.POLICY_OUTPUT_COLS, index=False)
            self.assertEqual(pd.read_csv(path, nrows=0).columns.tolist(), policies.POLICY_OUTPUT_COLS)
            with patch.object(policies, "apply_fixed_day_policy",
                              side_effect=AssertionError("Saved policies must not be rebuilt")):
                restored = policies.read_policy_panel(path)
            restored["subject_id"] = restored.subject_id.astype(str)
            pd.testing.assert_frame_equal(restored[policies.POLICY_OUTPUT_COLS],
                                          single_policy[policies.POLICY_OUTPUT_COLS], check_dtype=False)
            self.assertNotIn("policy_name", restored)
            self.assertNotIn("policy_type", restored)
            self.assertNotIn("policy_remove_day", restored)

            manifest_path = Path(temporary) / "policy_panels.csv"
            policies.save_policy_panels(original, manifest_path)
            manifest = pd.read_csv(manifest_path)
            self.assertEqual(manifest.policy_name.tolist(),
                             ["remove_on_day_1", "remove_on_day_2", "remove_on_day_5"])
            for entry in manifest.itertuples(index=False):
                saved = pd.read_csv(manifest_path.parent / entry.panel_file)
                self.assertNotIn("policy_name", saved)
                self.assertEqual(Path(entry.panel_file).stem, entry.policy_name)
                self.assertEqual(len(saved), len(base))
            # Unlisted files must not add policies to an evaluation.
            original.to_csv(Path(temporary) / "panels/obsolete.csv", index=False)
            with patch.object(policies, "apply_fixed_day_policy",
                              side_effect=AssertionError("Saved policies must not be rebuilt")):
                for name, module in ESTIMATORS.items():
                    loaded = policies.read_policy_collection(manifest_path)
                    loaded["subject_id"] = loaded.subject_id.astype(str)
                    after = module.evaluate_policy_episodes(loaded, scores)[0]
                    before = module.evaluate_policy_episodes(original, scores)[0]
                    pd.testing.assert_frame_equal(after, before, check_dtype=False)
            manifest.loc[0, "policy_name"] = "wrong_policy"
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                policies.read_policy_collection(manifest_path)

    def test_policy_reader_rejects_missing_or_inconsistent_saved_actions(self):
        base = pec.add_episode_day(policies.add_stable_ids_and_decision_flag(
            self.panel[policies.POLICY_INPUT_COLS].copy()))
        policy = policies.build_long_policy_panel(base, [1])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "policies.csv"
            policy.to_csv(path, columns=[column for column in policies.POLICY_OUTPUT_COLS
                                        if column != "policy_action"], index=False)
            with self.assertRaisesRegex(ValueError, "Rebuild it"):
                policies.read_policy_panel(path)
            corrupted = policy.copy()
            remove_row = corrupted.index[corrupted.policy_action.eq("remove")][0]
            corrupted.loc[remove_row, "policy_matches_observed_action_today"] = 0.5
            corrupted.to_csv(path, columns=policies.POLICY_OUTPUT_COLS, index=False)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                policies.read_policy_panel(path)

    def test_policy_matches_observed_compares_removal_days_for_whole_episode(self):
        # Same-day removal can occur on a later interval or without an extra row.
        rows = pd.DataFrame({
            "catheter_episode_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "episode_day": [1, 2, 2, 1, 2, 1, 2, 1, 2],
            "policy_action": ["keep", "remove", "out", "keep", "remove",
                              "remove", "out", "keep", "remove"],
            "removed_in_period": [0, 0, 1, 0, 1, 0, 1, 0, 0],
        })
        self.assertEqual(policies.policy_removal_day_matches_observed(rows).tolist(),
                         [True, True, True, True, True, False, False, False, False])

    def test_all_panel_entry_points_generate_both_modes_without_overwrites(self):
        full_scores, _ = bootstrap.refit_nuisance_predictions(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), "aipw",
        )
        full_scores = full_scores.drop(columns=[c for c in full_scores if c.startswith("__rescored_")])
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            panels = tuple((name, root / f"{name}.csv", root / name / "nuisance_models")
                           for name, _, _ in nuisance.PANEL_RUNS)
            for _, panel_path, nuisance_root in panels:
                self.panel.to_csv(panel_path, index=False)
                predictions = nuisance_root / "logistic_regression/nuisance_predictions.csv"
                predictions.parent.mkdir(parents=True)
                full_scores.to_csv(predictions, index=False)
            with patch.object(nuisance, "PANEL_RUNS", ()), \
                 patch.object(policies, "PANEL_RUNS", panels), \
                 patch.object(policies, "POLICY_DAYS", [1, 2]):
                policies.main()
                for name, module in ESTIMATORS.items():
                    with self.subTest(estimator=name), \
                         patch.object(module, "NUISANCE_MODEL_TYPE", "logistic_regression"), \
                         patch.object(module, "N_BOOTSTRAP", 1), \
                         patch.object(module, "BOOTSTRAP_MODES", ("fixed", "refit")), \
                         patch.object(module, "PANEL_RUNS", panels), \
                         patch.object(module, "run_panel_estimation", wraps=module.run_panel_estimation) as run:
                        module.main()
                        self.assertEqual(run.call_count, 8)
                    self.assertEqual(run.call_args.args, (panels[-1][1], panels[-1][2].parent, "logistic_regression", "refit"))
                    for _, _, nuisance_root in panels:
                        output = nuisance_root.parent / "policy_eval" / name
                        for refit in (False, True):
                            mode = "refit" if refit else "fixed"
                            mode_output = output / mode
                            for filename in module.OUTPUT_FILENAMES.values():
                                self.assertTrue((mode_output / filename).is_file())
                            intervals = mode_output / "confidence_intervals"
                            draws = pd.read_csv(intervals / f"{name}_bootstrap_replicate_estimates.csv")
                            self.assertEqual(draws.bootstrap_replicate.tolist(), [1])
                            values = pd.read_csv(intervals / f"{name}_policy_value_confidence_intervals.csv")
                            self.assertEqual(len(values), 15)
                            summary = pd.read_csv(mode_output / module.OUTPUT_FILENAMES["summary"])
                            pd.testing.assert_series_equal(
                                summary.policy_name.sort_values().reset_index(drop=True),
                                pd.Series(["current_practice", "remove_on_day_1", "remove_on_day_2"], name="policy_name"),
                            )

    def test_batch_missing_input_raises_when_read(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent.csv"
            with patch.object(nuisance, "PANEL_RUNS", ()):
                for module in ESTIMATORS.values():
                    with patch.object(module, "PANEL_RUNS", (("missing", missing, missing.parent / "nuisance_models"),)), \
                         patch.object(module, "run_panel_estimation", wraps=module.run_panel_estimation) as run:
                        with self.assertRaisesRegex(FileNotFoundError, "policy_panels.csv"):
                            module.main()
                        run.assert_called_once()

    def test_script_selections_are_independent(self):
        # An evaluator uses only its own selected panel and mode, even if the
        # fitting script and builder select different panels.
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            selected = (("selected", root / "selected.csv", root / "selected/nuisance_models"),)
            other = (("other", root / "other.csv", root / "other/nuisance_models"),)
            with patch.object(nuisance, "PANEL_RUNS", other), \
                 patch.object(policies, "PANEL_RUNS", other):
                for name, module in ESTIMATORS.items():
                    with self.subTest(estimator=name), \
                         patch.object(module, "PANEL_RUNS", selected), \
                         patch.object(module, "BOOTSTRAP_MODES", ("refit",)), \
                         patch.object(module, "NUISANCE_MODEL_TYPE", "random_forest"), \
                         patch.object(nuisance, "configure_panel_run", side_effect=AssertionError("must not use fitting configuration")), \
                         patch.object(module, "run_panel_estimation") as run:
                        module.main()
                        run.assert_called_once()
                        run.assert_called_once_with(selected[0][1], selected[0][2].parent, "random_forest", "refit",
                                                    panel_name=selected[0][0])
                self.assertEqual(nuisance.PANEL_RUNS, other)
                self.assertEqual(policies.PANEL_RUNS, other)

    def test_single_panel_entry_points_support_both_modes_and_validation(self):
        expected = {"bootstrap_replicate_estimates.csv", "policy_value_confidence_intervals.csv",
                    "policy_difference_confidence_intervals.csv"}
        full_scores, _ = bootstrap.refit_nuisance_predictions(
            self.panel, self.subjects, np.ones(len(self.subjects), dtype=int), "aipw",
        )
        full_scores = full_scores.drop(columns=[c for c in full_scores if c.startswith("__rescored_")])
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            validation = root / "artefacts/semi-synthetic_validation/semi_synthetic_measured_confounding"
            datasets = [
                ("real", root / "data/modelling_panel.csv", root / "artefacts"),
                ("validation", validation / "semi_synthetic_panel.csv", validation / "pipeline_runs/semi_synthetic_with_confounding"),
                ("validation-omitted", validation / "semi_synthetic_panel_confounder_omitted.csv", validation / "pipeline_runs/confounder_omitted"),
                ("validation-randomised", validation / "semi_synthetic_panel_randomised_action.csv", validation / "pipeline_runs/randomised_action"),
            ]
            for panel_name, panel_path, artefact_root in datasets:
                panel_path.parent.mkdir(parents=True, exist_ok=True)
                self.panel.to_csv(panel_path, index=False)
                policy_path = artefact_root / "counterfactual_policies" / "policy_panels.csv"
                policy_path.parent.mkdir(parents=True, exist_ok=True)
                with patch.multiple(policies, INPUT_PATH=panel_path, OUTDIR=policy_path.parent,
                                    POLICY_MANIFEST_PATH=policy_path,
                                    CHECK_OUTPUT_PATH=policy_path.parent / "policy_panel_checks.csv",
                                    POLICY_DAYS=[1, 2]):
                    policies.run_panel()
                self.assertEqual(pd.read_csv(policy_path, nrows=0).columns.tolist(), policies.POLICY_MANIFEST_COLS)
                self.assertEqual(len(list((policy_path.parent / "panels").glob("*.csv"))), 2)
                prediction_path = artefact_root / "nuisance_models" / "logistic_regression" / "nuisance_predictions.csv"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                for name, module in ESTIMATORS.items():
                    output_root = artefact_root / "policy_eval" / name
                    with self.subTest(panel=panel_name, estimator=name), patch.object(module, "N_BOOTSTRAP", 2):
                        full_scores.to_csv(prediction_path, index=False)
                        with patch.object(nuisance, "fit_crossfit_fold_model", side_effect=AssertionError("fixed mode must not refit")):
                            module.run_panel_estimation(panel_path, artefact_root, "logistic_regression", "fixed")
                        prediction_path.unlink()
                        # Refit mode requires no saved predictions and does not
                        # change the fitting script's selected learner or paths.
                        original_model_type = nuisance.MODEL_TYPE
                        with patch.object(nuisance, "configure_model_run", side_effect=AssertionError("must not mutate fitting configuration")):
                            module.run_panel_estimation(panel_path, artefact_root, "logistic_regression", "refit")
                        self.assertEqual(nuisance.MODEL_TYPE, original_model_type)
                        for mode in ("fixed", "refit"):
                            outdir = output_root / mode / "confidence_intervals"
                            self.assertEqual({p.name for p in outdir.iterdir()}, {f"{name}_{filename}" for filename in expected})
                            draws = pd.read_csv(outdir / f"{name}_bootstrap_replicate_estimates.csv")
                            self.assertEqual(draws.bootstrap_replicate.tolist(), [1, 2])
                            self.assertTrue(draws.columns.is_unique)
                            self.assertFalse(any(column.startswith(f"{name}__") for column in draws))
                            intervals = pd.read_csv(outdir / f"{name}_policy_value_confidence_intervals.csv")
                            self.assertEqual(set(intervals.estimator), {name})
                            self.assertEqual(len(intervals), 15)
                            differences = pd.read_csv(outdir / f"{name}_policy_difference_confidence_intervals.csv")
                            current = differences.loc[differences.policy_name.eq("current_practice")]
                            np.testing.assert_allclose(current[["point_estimate", "ci_lower", "ci_upper"]], 0)
            self.assertFalse(list(root.rglob("*.json")))
            self.assertFalse(list(root.rglob("*.pkl")))


if __name__ == "__main__":
    unittest.main()
