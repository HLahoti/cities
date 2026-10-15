"""Learned densification: *which* cells thicken, and by how much.

A two-part (hurdle) model trained on observed density snapshots:

* a classifier gives P(density rises by ≥ ``event_threshold`` within one
  ``period``), from the same kind of per-cell features as growth;
* a regressor gives the size of the rise, given that it happens.

Real infill is lumpy — a few cells jump while most stay put — so a model that
can tell *which* cells will densify beats one that spreads the average
increase over every cell. Used in simulations through the
``learned_densify`` process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import Registry
from .features import FeatureSet
from .learners import LEARNERS

REGRESSORS = Registry("regressor")


@REGRESSORS.register("linear")
def linear(alpha: float = 1.0, **_):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


@REGRESSORS.register("gbm")
def gbm(max_iter: int = 200, learning_rate: float = 0.05, max_leaf_nodes: int = 15,
        min_samples_leaf: int = 100, **_):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                         max_leaf_nodes=max_leaf_nodes,
                                         min_samples_leaf=min_samples_leaf, random_state=0)


def density_domain(grid) -> np.ndarray:
    """Cells that can densify: land that is urban or already partly built, not saturated."""
    d = grid["density"]
    land = ~grid["blocked"]
    return land & ((grid["urban"] > 0) | (d > 0)) & (d < 0.999)


def density_transitions(states: dict, features: FeatureSet, pairs, event_threshold: float):
    Xs, events, amounts = [], [], []
    for t0, t1 in pairs:
        g0, g1 = states[t0], states[t1]
        dom = density_domain(g0)
        dd = (g1["density"] - g0["density"])[dom]
        Xs.append(features.matrix(g0)[dom])
        events.append(dd >= event_threshold)
        amounts.append(np.clip(dd, 0, 1))
    return np.concatenate(Xs), np.concatenate(events), np.concatenate(amounts)


class DensityModel:
    def __init__(self, features: FeatureSet, classifier: str = "logistic", regressor: str = "gbm",
                 classifier_params: dict | None = None, regressor_params: dict | None = None,
                 period: float = 2.0, event_threshold: float = 0.04, new_density: float = 0.35):
        self.features = features
        self.classifier_name, self.regressor_name = classifier, regressor
        self.clf = LEARNERS.get(classifier)(**(classifier_params or {}))
        self.reg = REGRESSORS.get(regressor)(**(regressor_params or {}))
        self.period, self.event_threshold, self.new_density = period, event_threshold, new_density
        self.meta: dict = {}

    def fit(self, X, event, amount):
        self.clf.fit(X, event)
        self.reg.fit(X[event], amount[event])
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        p = self.clf.predict_proba(X)[:, 1]
        m = np.clip(self.reg.predict(X), self.event_threshold, 1.0)
        return p, m

    def predict_grid(self, grid):
        """(P(event within one period), size of the rise) on the density domain."""
        dom = density_domain(grid)
        p = np.zeros(grid.shape)
        m = np.zeros(grid.shape)
        if dom.any():
            p[dom], m[dom] = self.predict(self.features.matrix(grid)[dom])
        return p, m, dom

    def save(self, path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "DensityModel":
        import joblib
        return joblib.load(path)


def train_density_model(spec: dict, base_grid, until: float, states: dict | None = None,
                        features: list | None = None, classifier: str = "logistic",
                        regressor: str = "gbm", classifier_params: dict | None = None,
                        regressor_params: dict | None = None, event_threshold: float = 0.04,
                        holdout: bool = True):
    """Fit on observed density intervals ending ≤ ``until`` (the most common
    interval length becomes the model's ``period``). Returns (model, report)."""
    from sklearn.metrics import roc_auc_score
    from .train import build_states
    from ..metrics.compare import density_errors

    states = states or build_states(spec, base_grid)
    years = sorted(y for y in states if y == base_grid.year
                   or "density" in base_grid.observed.get(y, {}))
    pairs = [(a, b) for a, b in zip(years[:-1], years[1:]) if b <= until]
    if not pairs:
        raise ValueError(f"no observed density intervals end by {until}")
    lengths = [b - a for a, b in pairs]
    period = max(set(lengths), key=lengths.count)
    pairs = [p for p in pairs if p[1] - p[0] == period]
    fs = FeatureSet(features)
    X, ev, amt = density_transitions(states, fs, pairs, event_threshold)
    # Typical density of newly urban cells at their first observation.
    new_d = [states[b]["density"][(states[b]["urban"] > 0) & (states[a]["urban"] == 0)]
             for a, b in pairs]
    new_d = np.concatenate(new_d)
    model = DensityModel(fs, classifier, regressor, classifier_params, regressor_params,
                         period=period, event_threshold=event_threshold,
                         new_density=float(np.median(new_d)) if new_d.size else 0.35)
    model.fit(X, ev, amt)
    report = {"pairs": pairs, "period": period, "event_rate": float(ev.mean()),
              "new_density": model.new_density,
              "train_auc": float(roc_auc_score(ev, model.clf.predict_proba(X)[:, 1]))}
    later = [(a, b) for a, b in zip(years[:-1], years[1:]) if a >= until]
    if holdout and later:
        a, b = later[0]
        Xh, evh, _ = density_transitions(states, fs, [(a, b)], event_threshold)
        report["holdout_pair"] = (a, b)
        report["holdout_auc"] = float(roc_auc_score(evh, model.clf.predict_proba(Xh)[:, 1]))
        g0, g1 = states[a], states[b]
        p, m, dom = model.predict_grid(g0)
        steps = (b - a) / period
        expected = np.clip(g0["density"] + (1 - (1 - p) ** steps) * m, 0, 1)
        e = density_errors(g1["density"], expected, dom, reference=g0["density"])
        report["holdout_expected"] = {k: round(v, 4) for k, v in e.items()}
    model.meta = report
    return model, report
