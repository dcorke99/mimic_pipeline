"""Patient-grouped Super Learner with a convex Brier-loss combination."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted


class GroupedSuperLearner(ClassifierMixin, BaseEstimator):
    """Learn weights on inner out-of-fold probabilities, then refit each learner.

    Each supplied estimator is a pipeline containing its own preprocessing.
    Inner folds use patient IDs supplied by the outer nuisance-model training
    fold. No outer held-out rows participate in fitting the ensemble weights.
    """

    def __init__(self, estimators, n_splits=5):
        self.estimators = estimators
        self.n_splits = n_splits

    @staticmethod
    def _fit_library(estimators, features, target):
        retained = np.array([
            col for col in range(features.shape[1])
            if np.unique(features[~np.isnan(features[:, col]), col]).size > 1
        ], dtype=int)
        # Rare outcomes can leave an inner training fold with just one class.
        # All-missing / constant features are also assessed on training only.
        if np.unique(target).size < 2 or retained.size == 0:
            probability = float((target.sum() + 1.0) / (len(target) + 2.0))
            return [(name, None, retained, probability) for name, _ in estimators]
        fitted = []
        for name, template in estimators:
            model = clone(template)
            model.fit(features[:, retained], target)
            if "xgboost" in model.named_steps:
                model.named_steps["xgboost"].set_params(device="cpu")
            fitted.append((name, model, retained, None))
        return fitted

    @staticmethod
    def _library_predictions(library, features):
        predictions = []
        for _, model, retained, probability in library:
            if model is None:
                predictions.append(np.full(len(features), probability))
                continue
            values = features[:, retained]
            if "lightgbm" in model.named_steps:
                values = pd.DataFrame(
                    values, columns=model.named_steps["lightgbm"].feature_name_
                )
            predictions.append(model.predict_proba(values)[:, 1])
        return np.column_stack(predictions)

    def fit(self, X, y, groups=None):
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        if features.ndim != 2 or target.shape != (len(features),) or not len(features):
            raise ValueError("Super Learner requires nonempty aligned X and y")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Super Learner requires binary 0/1 targets")
        if groups is None:
            raise ValueError("Super Learner requires patient groups for inner cross-validation")
        groups = np.asarray(groups)
        if groups.shape != target.shape or pd.isna(groups).any():
            raise ValueError("Super Learner requires one nonmissing patient ID per row")
        self.n_inner_splits_ = min(self.n_splits, len(pd.unique(groups)))
        if self.n_inner_splits_ < 2:
            raise ValueError("Super Learner requires at least two patient groups")
        self.learner_names_ = [name for name, _ in self.estimators]
        if not self.learner_names_ or len(set(self.learner_names_)) != len(self.learner_names_):
            raise ValueError("Super Learner requires distinct nonempty learner names")
        self.n_features_in_ = features.shape[1]
        self.classes_ = np.array([0, 1])
        out_of_fold = np.full((len(features), len(self.estimators)), np.nan)
        self.inner_fold_assignments_ = np.full(len(features), -1, dtype=int)
        splitter = GroupKFold(n_splits=self.n_inner_splits_)
        for fold, (training, validation) in enumerate(
            splitter.split(features, target, groups=groups)
        ):
            library = self._fit_library(
                self.estimators, features[training], target[training]
            )
            out_of_fold[validation] = self._library_predictions(library, features[validation])
            self.inner_fold_assignments_[validation] = fold
        if not np.isfinite(out_of_fold).all() or ((out_of_fold < 0) | (out_of_fold > 1)).any():
            raise ValueError("Invalid Super Learner inner out-of-fold probabilities")

        # Minimise mean squared probability error (Brier loss) on the simplex.
        # Precompute the quadratic terms to avoid scanning the panel at every
        # optimiser step. A convex combination keeps probabilities in [0, 1].
        gram = out_of_fold.T @ out_of_fold / len(target)
        cross = out_of_fold.T @ target / len(target)
        initial = np.full(len(self.estimators), 1.0 / len(self.estimators))
        result = minimize(
            lambda weights: float(weights @ gram @ weights - 2.0 * weights @ cross),
            initial,
            jac=lambda weights: 2.0 * (gram @ weights - cross),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(initial),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                         "jac": lambda weights: np.ones_like(weights)},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f"Super Learner weight optimisation failed: {result.message}")
        self.weights_ = np.clip(result.x, 0.0, 1.0)
        self.weights_ /= self.weights_.sum()
        self.inner_brier_ = float(np.mean((out_of_fold @ self.weights_ - target) ** 2))
        self.base_inner_brier_ = np.mean((out_of_fold - target[:, None]) ** 2, axis=0)
        self.estimators_ = self._fit_library(self.estimators, features, target)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["weights_", "estimators_"])
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("Super Learner prediction features do not match training")
        if not len(features):
            return np.empty((0, 2), dtype=float)
        probability = np.clip(self._library_predictions(self.estimators_, features) @ self.weights_, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
