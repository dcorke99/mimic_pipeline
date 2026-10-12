"""Run-choice wiring and orchestration tests; no real data or full model fits."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline_config as config
import run_pipeline as runner

REPO_ROOT = Path(__file__).resolve().parent


class PipelineConfigTests(unittest.TestCase):
    def load_stage(self, name):
        return runpy.run_path(str(REPO_ROOT / name), run_name="config_wiring_test")

    def test_config_contains_run_choices_and_shared_paths(self):
        names = {name for name in vars(config) if not name.startswith("_")}
        self.assertEqual(names, {
            "Path", "REPO_ROOT", "DATA_DIR", "ARTEFACT_ROOT", "VALIDATION_DIR", "PANEL_PATHS",
            "PIPELINE_STAGES", "MODEL_TYPES", "RUN_LEARNING_CURVES",
            "SUPERLEARNER_BASE_MODELS", "NUISANCE_PANELS", "POLICY_PANELS",
            "EVALUATION_PANELS", "NUISANCE_MODEL_TYPE", "BOOTSTRAP_MODES",
            "POLICY_DAYS", "ANALYSIS_PANEL",
            "ANALYSIS_INCLUDE_AGE_SPLIT", "RESULT_CHECK_PANEL", "RESULT_CHECK_BOOTSTRAP_MODE",
        })

    def test_changed_choices_reach_all_stage_modules_and_matching_paths(self):
        overrides = {
            "MODEL_TYPES": ["lightgbm", "superlearner"], "RUN_LEARNING_CURVES": True,
            "SUPERLEARNER_BASE_MODELS": ["logistic_regression", "extra_trees"],
            "NUISANCE_PANELS": ["confounder_omitted"],
            "POLICY_PANELS": ["real", "randomised_action"],
            "EVALUATION_PANELS": ["semi_synthetic_with_confounding"],
            "NUISANCE_MODEL_TYPE": "catboost", "BOOTSTRAP_MODES": ["refit"],
            "POLICY_DAYS": [2, 4],
            "ANALYSIS_PANEL": "randomised_action", "ANALYSIS_INCLUDE_AGE_SPLIT": False,
            "RESULT_CHECK_PANEL": "confounder_omitted", "RESULT_CHECK_BOOTSTRAP_MODE": "refit",
        }
        with patch.multiple(config, **overrides):
            builder = self.load_stage("create_data_panel.py")
            self.assertEqual(builder["DATA_DIR"], (REPO_ROOT / "data"))
            synthetic = self.load_stage("create_semi_synthetic_panel.py")
            self.assertEqual(synthetic["SOURCE_PANEL"], builder["DATA_DIR"] / "modelling_panel.csv")
            self.assertEqual(synthetic["OUTPUT_DIR"], (REPO_ROOT / "artefacts/semi-synthetic_validation/semi_synthetic_measured_confounding"))
            self.assertEqual(synthetic["POLICY_DAYS"], [2, 4])
            nuisance = self.load_stage("fit_nuisance_models.py")
            self.assertEqual(nuisance["MODEL_TYPES"], ["lightgbm", "superlearner"])
            self.assertTrue(nuisance["RUN_LEARNING_CURVES"])
            self.assertEqual(nuisance["SUPERLEARNER_BASE_MODELS"], overrides["SUPERLEARNER_BASE_MODELS"])
            self.assertEqual(nuisance["PANEL_RUNS"], (("confounder_omitted", config.VALIDATION_DIR / "semi_synthetic_panel_confounder_omitted.csv", config.VALIDATION_DIR / "pipeline_runs/confounder_omitted/nuisance_models"),))
            self.assertEqual(nuisance["COVARIATE_DICT_FILE"], builder["DATA_DIR"] / "covariate_dictionary.csv")
            policies = self.load_stage("build_policy_panels.py")
            self.assertEqual(policies["POLICY_DAYS"], synthetic["POLICY_DAYS"])
            self.assertEqual(policies["PANEL_RUNS"], (("real", config.DATA_DIR / "modelling_panel.csv", config.ARTEFACT_ROOT / "nuisance_models"), ("randomised_action", config.VALIDATION_DIR / "semi_synthetic_panel_randomised_action.csv", config.VALIDATION_DIR / "pipeline_runs/randomised_action/nuisance_models")))
            for estimator in ("gformula", "ipw", "aipw"):
                evaluation = self.load_stage(f"evaluate_{estimator}_policies.py")
                self.assertEqual(evaluation["NUISANCE_MODEL_TYPE"], "catboost")
                self.assertEqual(evaluation["BOOTSTRAP_MODES"], ["refit"])
                self.assertEqual(evaluation["PANEL_RUNS"], (("semi_synthetic_with_confounding", config.VALIDATION_DIR / "semi_synthetic_panel.csv", config.VALIDATION_DIR / "pipeline_runs/semi_synthetic_with_confounding/nuisance_models"),))
            analysis = self.load_stage("panel_analysis.py")
            path = config.VALIDATION_DIR / "semi_synthetic_panel_randomised_action.csv"
            nuisance_root = config.VALIDATION_DIR / "pipeline_runs/randomised_action/nuisance_models"
            self.assertEqual(analysis["DATA_FILE"], path)
            self.assertEqual(analysis["NUISANCE_DIR"], nuisance_root / "catboost")
            self.assertEqual(analysis["RESULTS_DIR"], nuisance_root.parent / "panel_analysis")
            audit = self.load_stage("check_policy_results.py")
            audit_nuisance = config.VALIDATION_DIR / "pipeline_runs/confounder_omitted/nuisance_models"
            self.assertEqual(audit["BOOTSTRAP_MODE"], "refit")
            self.assertEqual(audit["POLICY_CHECK_PATH"], audit_nuisance.parent / "counterfactual_policies/policy_panel_checks.csv")
            self.assertEqual(audit["OUTPUT_PATH"], audit_nuisance.parent / "policy_eval/policy_results_checks_refit.csv")

    def test_downstream_entry_points_consume_matching_paths_for_all_panels(self):
        filenames = {
            "real": "modelling_panel.csv",
            "semi_synthetic_with_confounding": "semi_synthetic_panel.csv",
            "confounder_omitted": "semi_synthetic_panel_confounder_omitted.csv",
            "randomised_action": "semi_synthetic_panel_randomised_action.csv",
        }
        # These expected locations come from the documented layout,
        # independently of the shared path table.
        data = REPO_ROOT / "data"
        artefacts = REPO_ROOT / "artefacts"
        validation = artefacts / "semi-synthetic_validation/semi_synthetic_measured_confounding"
        expected = []
        for panel, filename in filenames.items():
            panel_path = data / filename if panel == "real" else validation / filename
            panel_root = artefacts if panel == "real" else validation / "pipeline_runs" / panel
            expected.append((panel, panel_path, panel_root))
        with patch.multiple(
            config, NUISANCE_PANELS=list(filenames),
            POLICY_PANELS=list(filenames), EVALUATION_PANELS=list(filenames),
            MODEL_TYPES=["logistic_regression"], BOOTSTRAP_MODES=["fixed", "refit"],
            NUISANCE_MODEL_TYPE="logistic_regression",
        ), redirect_stdout(StringIO()):
            nuisance = self.load_stage("fit_nuisance_models.py")
            globals_ = nuisance["main"].__globals__
            fitted = []
            def record_fit(model_type, panel_name=None):
                globals_["configure_model_run"](model_type)
                fitted.append((panel_name, globals_["INFILE"], globals_["OUTDIR"],
                               globals_["COVARIATE_DICT_FILE"]))
            with patch.dict(globals_, run_nuisance_model=record_fit,
                            build_nuisance_model_comparison=Mock()):
                nuisance["main"]()
            self.assertEqual(fitted, [(panel, path, artefact / "nuisance_models/logistic_regression",
                                      data / "covariate_dictionary.csv")
                                     for panel, path, artefact in expected])
            policies = self.load_stage("build_policy_panels.py")
            globals_ = policies["main"].__globals__
            policy_runs = []
            def record_policy():
                policy_runs.append((globals_["INPUT_PATH"], globals_["OUTDIR"],
                                    globals_["POLICY_MANIFEST_PATH"], globals_["CHECK_OUTPUT_PATH"]))
            with patch.dict(globals_, run_panel=record_policy):
                policies["main"]()
            self.assertEqual(policy_runs, [(path, artefact / "counterfactual_policies",
                                            artefact / "counterfactual_policies/policy_panels.csv",
                                            artefact / "counterfactual_policies/policy_panel_checks.csv")
                                           for _, path, artefact in expected])
            for estimator in ("gformula", "ipw", "aipw"):
                evaluation = self.load_stage(f"evaluate_{estimator}_policies.py")
                execute = Mock()
                with patch.dict(evaluation["main"].__globals__, run_panel_estimation=execute):
                    evaluation["main"]()
                self.assertEqual([call.args for call in execute.call_args_list],
                                 [(path, artefact, "logistic_regression", mode)
                                  for _, path, artefact in expected for mode in ("fixed", "refit")])
                self.assertEqual([call.kwargs["panel_name"] for call in execute.call_args_list],
                                 [panel for panel, _, _ in expected for _ in ("fixed", "refit")])
                # Exercise the evaluator's real path setup; stop before
                # reading data, fitting models or writing any output.
                class StopBeforeRead(Exception):
                    pass
                for panel, path, artefact in expected:
                    for mode in ("fixed", "refit"):
                        with self.subTest(estimator=estimator, panel=panel, mode=mode), \
                             patch.object(Path, "mkdir", autospec=True) as mkdir, \
                             patch.dict(evaluation["run_panel_estimation"].__globals__,
                                        read_policy_collection=Mock(side_effect=StopBeforeRead)):
                            read_policy = evaluation["run_panel_estimation"].__globals__["read_policy_collection"]
                            with self.assertRaises(StopBeforeRead):
                                evaluation["run_panel_estimation"](
                                    path, artefact, "logistic_regression", mode, panel_name=panel)
                            read_policy.assert_called_once_with(artefact / "counterfactual_policies/policy_panels.csv")
                            output_dir = artefact / "policy_eval" / estimator / mode
                            mkdir.assert_called_once_with(output_dir, exist_ok=True, parents=True)
            for panel, path, artefact in expected:
                with patch.multiple(config, ANALYSIS_PANEL=panel, RESULT_CHECK_PANEL=panel):
                    analysis = self.load_stage("panel_analysis.py")
                    self.assertEqual(analysis["DATA_FILE"], path)
                    self.assertEqual(analysis["RESULTS_DIR"], artefact / "panel_analysis")
                    self.assertEqual(analysis["NUISANCE_DIR"], artefact / "nuisance_models/logistic_regression")
                    self.assertEqual(analysis["COVARIATE_DICT_FILE"], data / "covariate_dictionary.csv")
                    for mode in ("fixed", "refit"):
                        with patch.object(config, "RESULT_CHECK_BOOTSTRAP_MODE", mode):
                            audit = self.load_stage("check_policy_results.py")
                            self.assertEqual(audit["POLICY_CHECK_PATH"], artefact / "counterfactual_policies/policy_panel_checks.csv")
                            for name, value in audit.items():
                                if name.endswith("_PATH") and name not in ("_PANEL_PATH", "POLICY_CHECK_PATH"):
                                    self.assertTrue(value.is_relative_to(artefact / "policy_eval"), (name, value))
                                    if name != "OUTPUT_PATH":
                                        self.assertTrue(value.is_relative_to(artefact / "policy_eval" / name.split("_")[0].lower() / mode))
                            self.assertEqual(audit["OUTPUT_PATH"], artefact / "policy_eval" / f"policy_results_checks_{mode}.csv")


    def test_empty_panel_and_model_lists_allow_import_and_skip_work(self):
        with patch.multiple(config, MODEL_TYPES=[], NUISANCE_PANELS=[], POLICY_PANELS=[], EVALUATION_PANELS=[]):
            nuisance = self.load_stage("fit_nuisance_models.py")
            policies = self.load_stage("build_policy_panels.py")
            with redirect_stdout(StringIO()):
                nuisance["main"]()
                policies["main"]()
                for estimator in ("gformula", "ipw", "aipw"):
                    self.load_stage(f"evaluate_{estimator}_policies.py")["main"]()
            self.assertEqual(nuisance["PANEL_RUNS"], ())
            self.assertEqual(policies["PANEL_RUNS"], ())

    def test_runner_uses_active_python_and_only_selected_stages_in_order(self):
        stages = ["build_policy_panels.py", "evaluate_ipw_policies.py"]
        with patch.object(config, "PIPELINE_STAGES", stages), \
             patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute, \
             redirect_stdout(StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual([call.args[0] for call in execute.call_args_list],
                         [[runner.sys.executable, str(REPO_ROOT / stage)] for stage in stages])
        self.assertTrue(all(call.kwargs["cwd"] == REPO_ROOT for call in execute.call_args_list))

    def test_runner_stops_after_failed_stage(self):
        stages = ["create_data_panel.py", "fit_nuisance_models.py", "panel_analysis.py"]
        with patch.object(config, "PIPELINE_STAGES", stages), \
             patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)]) as execute, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(runner.main(), 7)
        self.assertEqual(execute.call_count, 2)

    def test_empty_stage_list_executes_nothing(self):
        with patch.object(config, "PIPELINE_STAGES", []), \
             patch.object(runner.subprocess, "run") as execute:
            self.assertEqual(runner.main(), 0)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
