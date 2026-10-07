import unittest
import warnings

import numpy as np
import pandas as pd

import fit_nuisance_models as nuisance


class LightGBMPredictionTests(unittest.TestCase):
    def test_numpy_trained_fold_predicts_with_retained_feature_order(self):
        training = pd.DataFrame({
            "signal": np.arange(80, dtype=float),
            "with_missing": [np.nan, 0.0, 1.0, 2.0] * 20,
            "constant": np.ones(80),
        })
        target = pd.Series((training["signal"] >= 40).astype(int))
        fold = nuisance.fit_crossfit_fold_model(
            training, target, "removal", 0, model_type="lightgbm"
        )
        self.assertFalse(fold["fallback"])
        self.assertEqual(fold["retained_feature_cols"], ["signal", "with_missing"])
        held_out = pd.DataFrame({
            "constant": [1.0, 1.0, 1.0],
            "with_missing": [np.nan, 1.0, 2.0],
            "signal": [5.0, 45.0, 75.0],
        }, index=[101, 102, 103])
        estimator = fold["model"].named_steps["lightgbm"]
        expected = estimator.booster_.predict(
            held_out.loc[:, fold["retained_feature_cols"]].to_numpy(dtype=float)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = nuisance.predict_crossfit_fold(fold, held_out)
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        self.assertFalse(any("feature names" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
