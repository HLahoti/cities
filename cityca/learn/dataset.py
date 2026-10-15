"""Observed states and transition datasets.

``observed_state`` rebuilds the city as it was at an observed year: the
scenario's events and anchors are replayed up to that year (with no
simulated growth), then the urban and density layers are replaced by the
observations. These states are both the training examples for learned
transition models and the starting points for rolling-origin forecasts.
"""

from __future__ import annotations

import copy

import numpy as np

from ..engine import MODELS, Simulation, build_events
from ..anchors.field import AnchorField


def observed_years(grid, layer: str = "urban") -> list[float]:
    return sorted(y for y, o in grid.observed.items() if layer in o)


def observed_state(spec: dict, base_grid, year: float):
    """Grid at ``year``: infrastructure replayed, urban/density set to observations.

    ``base_grid`` is the scenario's start grid (city + scenario anchors).
    ``year`` may be the start year itself (no observation needed).
    """
    grid = base_grid.copy()
    if year > grid.year:
        events = build_events(copy.deepcopy(spec.get("events", [])))
        sim = Simulation(grid, MODELS.build("none"), field=AnchorField(**spec.get("field", {})),
                         events=events, record={"every": 10 ** 9, "layers": [], "metrics": []})
        sim.run(until=year)
        grid = sim.grid
    if year != base_grid.year:
        obs = base_grid.observed[float(year)]
        u = obs["urban"] > 0
        new = u & (grid["urban"] == 0)
        grid["urban"] = u
        ua = grid["urbanized_at"]
        ua[new] = year
        ua[~u] = np.nan
        grid.touch("urbanized_at")
        if "density" in obs:
            grid["density"] = obs["density"]
        elif "density" in grid.layers:
            d = grid["density"]
            d[u & (d < 0.25)] = 0.25  # newly urban cells without a density observation
            grid.touch("density")
    grid["anchor_potential"] = AnchorField(**spec.get("field", {})).compute(grid)
    return grid


def transitions(states: dict, features, pairs, neg_ratio: float = 6.0, seed: int = 0):
    """Training samples for each (t0, t1) pair of observed states.

    Candidates are developable cells at t0; label = urban at t1. Negatives are
    subsampled to ``neg_ratio`` per positive and re-weighted (weight = 1/rate)
    so probabilities stay unbiased. Returns X, y, w, interval index.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, ws, gs = [], [], [], []
    for k, (t0, t1) in enumerate(pairs):
        g0, g1 = states[t0], states[t1]
        cand = g0["developable"]
        y = (g1["urban"] > 0)[cand]
        X = features.matrix(g0)[cand]
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        keep_neg = min(neg.size, int(neg_ratio * max(pos.size, 1)))
        rate = keep_neg / max(neg.size, 1)
        neg = rng.choice(neg, size=keep_neg, replace=False) if keep_neg < neg.size else neg
        idx = np.concatenate([pos, neg])
        w = np.concatenate([np.ones(pos.size), np.full(neg.size, 1.0 / max(rate, 1e-9))])
        Xs.append(X[idx])
        ys.append(y[idx])
        ws.append(w / (t1 - t0))  # per-year weighting across unequal intervals
        gs.append(np.full(idx.size, k))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(gs)
