"""Configuration and pipeline-runner wiring tests."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline_config as config
import run_pipeline as runner

REPO_ROOT = Path(__file__).resolve().parent


class PipelineConfigTests(unittest.TestCase):
    def load_stage(self, name):
        return runpy.run_path(str(REPO_ROOT / name), run_name="config_wiring_test")

    def test_config_contains_only_supported_run_choices(self):
        names = {name for name in vars(config) if not name.startswith("_")}
        self.assertEqual(names, {
            "Path", "REPO_ROOT", "DATA_DIR", "ARTEFACT_ROOT", "VALIDATION_DIR",
            "PANEL_PATHS", "PIPELINE_STAGES", "MODEL_TYPES",
            "SUPERLEARNER_BASE_MODELS", "NUISANCE_PANELS", "POLICY_PANELS",
            "EVALUATION_PANELS", "NUISANCE_MODEL_TYPE", "BOOTSTRAP_MODES",
            "POLICY_DAYS", "ANALYSIS_PANEL", "ANALYSIS_INCLUDE_AGE_SPLIT",
        })

    def test_choices_reach_current_stage_modules(self):
        overrides = {
            "MODEL_TYPES": ["lightgbm", "superlearner"],
            "SUPERLEARNER_BASE_MODELS": ["logistic_regression", "extra_trees"],
            "NUISANCE_PANELS": ["confounder_omitted"],
            "POLICY_PANELS": ["real", "randomised_action"],
            "EVALUATION_PANELS": ["semi_synthetic_with_confounding"],
            "NUISANCE_MODEL_TYPE": "catboost", "BOOTSTRAP_MODES": ["refit"],
            "POLICY_DAYS": [2, 4], "ANALYSIS_PANEL": "randomised_action",
            "ANALYSIS_INCLUDE_AGE_SPLIT": False,
        }
        with patch.multiple(config, **overrides):
            synthetic = self.load_stage("create_semi_synthetic_panel.py")
            self.assertEqual(synthetic["POLICY_DAYS"], [2, 4])
            nuisance = self.load_stage("fit_nuisance_models.py")
            self.assertEqual(nuisance["MODEL_TYPES"], overrides["MODEL_TYPES"])
            self.assertEqual(nuisance["SUPERLEARNER_BASE_MODELS"], overrides["SUPERLEARNER_BASE_MODELS"])
            self.assertEqual(nuisance["PANEL_RUNS"][0][0], "confounder_omitted")
            policies = self.load_stage("build_policy_panels.py")
            self.assertEqual([run[0] for run in policies["PANEL_RUNS"]], ["real", "randomised_action"])
            for filename in ("evaluate_gformula_policies.py", "evaluate_policy_dml_policies.py"):
                evaluation = self.load_stage(filename)
                self.assertEqual(evaluation["NUISANCE_MODEL_TYPE"], "catboost")
                self.assertEqual(evaluation["BOOTSTRAP_MODES"], ["refit"])
                self.assertEqual(evaluation["PANEL_RUNS"][0][0], "semi_synthetic_with_confounding")
            analysis = self.load_stage("panel_analysis.py")
            self.assertEqual(analysis["DATA_FILE"], config.PANEL_PATHS["randomised_action"][0])

    def test_empty_selections_skip_work(self):
        with patch.multiple(config, MODEL_TYPES=[], NUISANCE_PANELS=[],
                            POLICY_PANELS=[], EVALUATION_PANELS=[]):
            nuisance = self.load_stage("fit_nuisance_models.py")
            policies = self.load_stage("build_policy_panels.py")
            gformula = self.load_stage("evaluate_gformula_policies.py")
            dml = self.load_stage("evaluate_policy_dml_policies.py")
            with redirect_stdout(StringIO()):
                nuisance["main"]()
                policies["main"]()
                gformula["main"]()
                with patch.object(sys, "argv", ["evaluate_policy_dml_policies.py"]):
                    dml["main"]()
            self.assertEqual(nuisance["PANEL_RUNS"], ())
            self.assertEqual(policies["PANEL_RUNS"], ())

    def test_runner_uses_active_python_and_selected_order(self):
        stages = ["fit_nuisance_models.py", "evaluate_policy_dml_policies.py"]
        with patch.object(config, "PIPELINE_STAGES", stages), \
             patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute, \
             redirect_stdout(StringIO()):
            self.assertIsNone(runner.main())
        self.assertEqual([call.args[0] for call in execute.call_args_list],
                         [[runner.sys.executable, str(REPO_ROOT / stage)] for stage in stages])
        self.assertTrue(all(call.kwargs["cwd"] == REPO_ROOT for call in execute.call_args_list))

    def test_runner_stops_after_failed_stage(self):
        stages = ["create_data_panel.py", "fit_nuisance_models.py", "panel_analysis.py"]
        with patch.object(config, "PIPELINE_STAGES", stages), \
             patch.object(runner.subprocess, "run",
                          side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)]) as execute, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(runner.main(), 7)
        self.assertEqual(execute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
