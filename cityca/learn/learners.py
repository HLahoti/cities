"""Learned transition models (requires scikit-learn).

``LearnedModel`` = a FeatureSet + a fitted classifier giving, for every
developable cell, the probability that it becomes urban over the training
interval. Inside a simulation it is used as a factor (``{"type": "learned",
"model_file": ...}``): demand still sets *how much* grows, the learned
surface decides *where*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import Registry
from .features import FeatureSet

LEARNERS = Registry("learner")


@LEARNERS.register("logistic")
def logistic(C: float = 1.0, **_):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


@LEARNERS.register("gbm")
def gbm(max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40, l2_regularization: float = 1.0, **_):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        early_stopping=False, random_state=0)


class LearnedModel:
    def __init__(self, features: FeatureSet, learner: str = "gbm", params: dict | None = None):
        self.features = features
        self.learner_name = learner
        self.params = dict(params or {})
        self.estimator = LEARNERS.get(learner)(**self.params)
        self.meta: dict = {}

    def fit(self, X, y, w=None):
        self.estimator.fit(X, y, **({} if w is None else self._weight_kw(w)))
        return self

    def _weight_kw(self, w):
        name = type(self.estimator).__name__
        if name == "Pipeline":
            return {f"{self.estimator.steps[-1][0]}__sample_weight": w}
        return {"sample_weight": w}

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict_grid(self, grid, frozen: tuple | None = None) -> np.ndarray:
        """Probability surface over developable cells (0 elsewhere).

        ``frozen`` = (column indices, (H, W, k) values) replaces those feature
        columns, e.g. to keep density-based inputs at their start-of-forecast values.
        """
        mask = grid["developable"]
        out = np.zeros(grid.shape)
        if mask.any():
            X = self.features.matrix(grid)
            if frozen is not None:
                X = X.copy()
                X[..., frozen[0]] = frozen[1]
            out[mask] = self.predict_proba(X[mask])
        return out

    # --- persistence ---------------------------------------------------------------------
    def save(self, path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": self.features.specs, "learner": self.learner_name,
                     "params": self.params, "estimator": self.estimator, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path) -> "LearnedModel":
        import joblib
        d = joblib.load(path)
        m = cls.__new__(cls)
        m.features = FeatureSet(d["features"])
        m.learner_name, m.params = d["learner"], d["params"]
        m.estimator, m.meta = d["estimator"], d.get("meta", {})
        return m


def evaluate(model: LearnedModel, X, y, w=None) -> dict:
    """Discrimination on held-out transitions (rank quality is what allocation uses)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = model.predict_proba(X)
    return {"roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "avg_precision": float(average_precision_score(y, p, sample_weight=w)),
            "base_rate": float(np.average(y, weights=w))}


def permutation_importance(model: LearnedModel, X, y, w=None, repeats: int = 3,
                           seed: int = 0) -> list[dict]:
    """Drop in ROC AUC when each feature is shuffled (model-agnostic importance)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X), sample_weight=w)
    out = []
    for j, name in enumerate(model.features.names):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp), sample_weight=w))
        out.append({"feature": name, "auc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return sorted(out, key=lambda r: -r["auc_drop"])
