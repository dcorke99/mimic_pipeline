"""Check multi-panel fitting orchestration without training full models."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fit_nuisance_models as nuisance


class NuisancePanelRunsTests(unittest.TestCase):
    def setUp(self):
        names = ("MODEL_TYPE", "MODEL_OUTPUT_NAME", "INFILE", "NUISANCE_ROOT", "OUTDIR",
                 "NUISANCE_PREDICTIONS_FILE", "PERFORMANCE_METRICS_FILE",
                 "CROSSFIT_ROW_ASSIGNMENTS_FILE", "CONSTANT_FEATURES_FILE", "MODEL_COMPARISON_FILE")
        settings = patch.multiple(nuisance, **{name: getattr(nuisance, name) for name in names})
        settings.start()
        self.addCleanup(settings.stop)

    def test_each_panel_gets_selected_models_and_separate_reports(self):
        for selection in ("xgboost", "all"):
            with self.subTest(selection=selection), TemporaryDirectory() as temporary:
                root = Path(temporary)
                panels = tuple(
                    (name, root / f"{name}.csv", root / name / "nuisance_models")
                    for name, _, _ in nuisance.PANEL_RUNS
                )
                for _, path, _ in panels:
                    path.touch()
                runs, reports = [], []

                def fit(model_type, panel_name=None):
                    nuisance.configure_model_run(model_type)
                    runs.append((nuisance.INFILE, nuisance.OUTDIR, model_type))
                    self.assertEqual(nuisance.NUISANCE_PREDICTIONS_FILE,
                                     nuisance.OUTDIR / "nuisance_predictions.csv")
                    self.assertEqual(nuisance.CROSSFIT_ROW_ASSIGNMENTS_FILE,
                                     nuisance.OUTDIR / "crossfit_row_assignments.csv")

                with patch.object(nuisance, "PANEL_RUNS", panels), \
                     patch.object(nuisance, "MODEL_TYPE", selection), \
                     patch.object(nuisance, "run_nuisance_model", side_effect=fit), \
                     patch.object(nuisance, "build_nuisance_model_comparison",
                                  side_effect=lambda: reports.append(nuisance.MODEL_COMPARISON_FILE)), \
                     redirect_stdout(StringIO()):
                    nuisance.main()
                    self.assertEqual(nuisance.MODEL_TYPE,
                                     nuisance.MODEL_TYPES[-1] if selection == "all" else selection)
                models = nuisance.MODEL_TYPES if selection == "all" else (selection,)
                self.assertEqual(runs, [(path, output / model, model)
                                        for _, path, output in panels for model in models])
                self.assertEqual(reports, [output / "nuisance_model_comparison.csv"
                                           for _, _, output in panels])
                # The direct loop leaves paths configured for its final run.
                self.assertEqual((nuisance.INFILE, nuisance.NUISANCE_ROOT,
                                  nuisance.MODEL_COMPARISON_FILE, nuisance.OUTDIR),
                                 (panels[-1][1], panels[-1][2],
                                  panels[-1][2] / "nuisance_model_comparison.csv",
                                  panels[-1][2] / models[-1]))

    def test_missing_panel_raises_when_read(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "real.csv"
            existing.touch()
            missing = root / "missing.csv"
            panels = (("real", existing, root / "real"),
                      ("missing", missing, root / "missing"))
            with patch.object(nuisance, "PANEL_RUNS", panels), \
                 patch.object(nuisance, "MODEL_TYPE", "xgboost"), \
                 patch.object(nuisance, "build_nuisance_model_comparison"), \
                 patch.object(nuisance, "run_nuisance_model",
                              side_effect=lambda model, panel_name: nuisance.INFILE.read_text()) as fit:
                with self.assertRaisesRegex(FileNotFoundError, "missing.csv"):
                    nuisance.main()
                self.assertEqual(fit.call_count, 2)


if __name__ == "__main__":
    unittest.main()
