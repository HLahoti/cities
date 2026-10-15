"""Train a learned transition model from a scenario's observed history."""

from __future__ import annotations

from .dataset import observed_state, observed_years, transitions
from .features import FeatureSet
from .learners import LearnedModel, evaluate, permutation_importance


def state_years(base_grid) -> list[float]:
    return [float(base_grid.year)] + [y for y in observed_years(base_grid) if y > base_grid.year]


def build_states(spec: dict, base_grid, years=None) -> dict:
    years = state_years(base_grid) if years is None else years
    return {y: observed_state(spec, base_grid, y) for y in years}


def train_from_scenario(spec: dict, base_grid, train_until: float, learner: str = "gbm",
                        params: dict | None = None, features: list | None = None,
                        states: dict | None = None, holdout: bool = True,
                        importance: bool = True, neg_ratio: float = 6.0, seed: int = 0):
    """Fit on consecutive observed intervals ending ≤ ``train_until``.

    If ``holdout``, the next interval after ``train_until`` is used to report
    held-out discrimination and permutation importance. Returns (model, report).
    """
    fs = FeatureSet(features)
    states = states or build_states(spec, base_grid)
    years = sorted(states)
    pairs = [(a, b) for a, b in zip(years[:-1], years[1:]) if b <= train_until]
    if not pairs:
        raise ValueError(f"no observed intervals end by {train_until}")
    X, y, w, _ = transitions(states, fs, pairs, neg_ratio=neg_ratio, seed=seed)
    model = LearnedModel(fs, learner, params).fit(X, y, w)
    model.meta = {"train_pairs": pairs, "n_samples": int(len(y)), "positives": int(y.sum())}
    report = {"train_pairs": pairs, "train": evaluate(model, X, y, w)}
    later = [(a, b) for a, b in zip(years[:-1], years[1:]) if a >= train_until]
    if holdout and later:
        Xh, yh, wh, _ = transitions(states, fs, later[:1], neg_ratio=neg_ratio, seed=seed + 1)
        report["holdout_pair"] = later[0]
        report["holdout"] = evaluate(model, Xh, yh, wh)
        if importance:
            report["importance"] = permutation_importance(model, Xh, yh, wh, seed=seed)
    model.meta["report"] = report
    return model, report
