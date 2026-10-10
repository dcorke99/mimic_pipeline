import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

import fit_nuisance_models as nuisance
import policy_bootstrap_common as bootstrap
from superlearner import GroupedSuperLearner
from test_policy_bootstrap_refitting import example_panel


class PatientIsolationClassifier(ClassifierMixin, BaseEstimator):
    """Fail if an inner held-out patient was used to fit its predictor."""

    def __init__(self, inverse=False):
        self.inverse = inverse

    def fit(self, X, y):
        self.training_patients_ = set(X[:, 0])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if self.training_patients_ & set(X[:, 0]):
            raise AssertionError("Patient leaked into inner validation")
        positive = np.where(X[:, 1] > 0, 0.9, 0.1)
        if self.inverse:
            positive = 1.0 - positive
        return np.column_stack([1.0 - positive, positive])


class SuperLearnerTests(unittest.TestCase):
    def small_library(self):
        configuration = {name: dict(values) for name, values in nuisance.LEARNER_CONFIGURATIONS.items()}
        for name in ("random_forest", "extra_trees", "xgboost", "lightgbm"):
            configuration[name]["n_estimators"] = 8
        configuration["xgboost"]["device"] = "cpu"
        configuration["catboost"]["iterations"] = 8
        configuration["catboost"]["depth"] = 2
        for name in ("random_forest", "extra_trees"):
            configuration[name]["min_samples_leaf"] = 1
        return patch.multiple(nuisance, LEARNER_CONFIGURATIONS=configuration,
                              SUPERLEARNER_INNER_FOLDS=2)

    def test_patient_grouped_oof_weights_favour_better_learner(self):
        groups = np.repeat(np.arange(12), 2)
        target = np.tile([0, 1], 12)
        features = np.column_stack([groups, target * 2.0 - 1.0])
        library = [("good", Pipeline([("probe", PatientIsolationClassifier())])),
                   ("bad", Pipeline([("probe", PatientIsolationClassifier(inverse=True))]))]
        ensemble = GroupedSuperLearner(library, n_splits=5).fit(features, target, groups=groups)
        for patient in np.unique(groups):
            self.assertEqual(np.unique(ensemble.inner_fold_assignments_[groups == patient]).size, 1)
        np.testing.assert_allclose(ensemble.weights_, [1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(ensemble.inner_brier_, 0.01)
        prediction = ensemble.predict_proba(np.array([[100, -1], [101, 1]]))
        np.testing.assert_allclose(prediction[:, 1], [0.1, 0.9], atol=1e-6)
        np.testing.assert_allclose(prediction.sum(axis=1), 1.0)

    def test_inner_single_class_and_constant_features_have_smoothed_predictions(self):
        library = [("probe", Pipeline([("probe", PatientIsolationClassifier())]))]
        for features, target in (
            (np.array([[0, 1], [0, 2], [1, 3], [1, 4]]), np.array([0, 0, 1, 1])),
            (np.array([[0, 1], [0, 1], [1, 2], [1, 2]]), np.array([0, 1, 0, 1])),
        ):
            with self.subTest(target=target.tolist()):
                ensemble = GroupedSuperLearner(library, n_splits=5).fit(
                    features, target, groups=[0, 0, 1, 1]
                )
                self.assertEqual(ensemble.n_inner_splits_, 2)
                self.assertTrue(np.isfinite(ensemble.inner_brier_))

    def test_six_learners_missing_features_and_saved_fold_round_trip(self):
        rng = np.random.default_rng(42)
        training = pd.DataFrame({"signal": rng.normal(size=80),
                                 "missing": [np.nan, 0.0, 1.0, 2.0] * 20,
                                 "constant": np.ones(80)})
        target = pd.Series((training.signal > 0).astype(int))
        with self.small_library():
            fold = nuisance.fit_crossfit_fold_model(
                training, target, "removal", 0, model_type="superlearner",
                groups=np.repeat(np.arange(40), 2),
            )
        estimator = fold["model"].named_steps["superlearner"]
        self.assertEqual(estimator.learner_names_, list(nuisance.SUPERLEARNER_BASE_MODELS))
        self.assertEqual(len(estimator.estimators_), 6)
        self.assertAlmostEqual(estimator.weights_.sum(), 1.0)
        self.assertTrue((estimator.weights_ >= 0).all())
        self.assertLessEqual(estimator.inner_brier_, estimator.base_inner_brier_.min() + 1e-7)
        held_out = training.iloc[:5].loc[:, ["constant", "missing", "signal"]]
        prediction = nuisance.predict_crossfit_fold(fold, held_out)
        self.assertTrue(np.isfinite(prediction).all())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold.pkl"
            joblib.dump(fold, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(nuisance.predict_crossfit_fold(restored, held_out), prediction)
        report = nuisance.superlearner_weights_by_fold([fold], {}, {})
        self.assertEqual(report.learner.tolist(), list(nuisance.SUPERLEARNER_BASE_MODELS))
        self.assertAlmostEqual(report.weight.sum(), 1.0)

    def test_groups_required_and_one_patient_falls_back(self):
        features = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0]})
        target = pd.Series([0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "patient groups"):
            nuisance.fit_crossfit_fold_model(features, target, "removal", 0, model_type="superlearner")
        fold = nuisance.fit_crossfit_fold_model(
            features, target, "removal", 0, model_type="superlearner", groups=[1] * 4
        )
        self.assertEqual(fold["fallback_reason"], "insufficient_inner_patient_groups")
        np.testing.assert_allclose(nuisance.predict_crossfit_fold(fold, features), 0.5)

    def test_standalone_run_exports_and_bootstrap_refits(self):
        panel, _, subjects = example_panel()
        settings = ("MODEL_TYPE", "MODEL_OUTPUT_NAME", "INFILE", "NUISANCE_ROOT", "OUTDIR",
                    "NUISANCE_PREDICTIONS_FILE", "PERFORMANCE_METRICS_FILE",
                    "CROSSFIT_ROW_ASSIGNMENTS_FILE", "CONSTANT_FEATURES_FILE", "MODEL_COMPARISON_FILE")
        with TemporaryDirectory() as temporary, self.small_library(), redirect_stdout(StringIO()), \
             patch.multiple(nuisance, **{name: getattr(nuisance, name) for name in settings}):
            root = Path(temporary)
            panel_path = root / "panel.csv"
            dictionary = root / "dictionary.csv"
            panel.to_csv(panel_path, index=False)
            pd.DataFrame(columns=["itemid", "label"]).to_csv(dictionary, index=False)
            with patch.multiple(nuisance, N_CROSSFIT_FOLDS=2, RUN_LEARNING_CURVES=True,
                                LEARNING_CURVE_FRACTIONS=(0.5, 1.0), COVARIATE_DICT_FILE=dictionary), \
                 patch.object(nuisance, "add_grouped_crossfit_folds",
                              side_effect=lambda df: nuisance_group_folds(df, n_splits=2)):
                nuisance.configure_panel_run(panel_path, root / "nuisance_models")
                nuisance.run_nuisance_model("superlearner", panel_name="test")
                output = root / "nuisance_models/superlearner"
                for filename in ("nuisance_predictions.csv", "propensity_model.pkl", "outcome_models.pkl",
                                 "performance_metrics.csv", "superlearner_weights_by_fold.csv",
                                 "nuisance_learning_curves.csv"):
                    self.assertTrue((output / filename).is_file(), filename)
                comparison = nuisance.build_nuisance_model_comparison()
                self.assertEqual(set(comparison.model_type), {"superlearner"})
            # Exercise the evaluator's explicit selection independently of the
            # nuisance fitting script's global model setting.
            nuisance.MODEL_TYPE = "logistic_regression"
            counts = np.ones(len(subjects), dtype=int)
            counts[0] = 2
            scores, _ = bootstrap.refit_nuisance_predictions(
                panel, subjects, counts, "aipw", n_splits=2, model_type="superlearner"
            )
            nuisance.validate_exported_probabilities(scores)
            self.assertTrue(scores.groupby("subject_id")[nuisance.CROSSFIT_FOLD_COL].nunique().eq(1).all())


nuisance_group_folds = nuisance.add_grouped_crossfit_folds


if __name__ == "__main__":
    unittest.main()
