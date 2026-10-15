"""Fitting demand (how much the city grows) to an observed area history.

Two families:

* **Growth laws** — linear, exponential, logistic and West–Brown–Enquist
  allometric. Each is fitted with the same yearly Euler steps the simulation
  uses, so a fitted law reproduces itself inside a run. ``law="auto"`` picks
  the law that best predicts the last training point from the earlier ones
  (a one-step temporal hold-out), then refits it on everything.
* **Population scaling** — built area A = c · P^β fitted on (population, area)
  pairs; growth then follows a population path (observed + projected). β is
  the urban allometric exponent: β < 1 means each new resident needs less new
  land than the last (densifying), β > 1 means sprawl.

All functions return plain DEMANDS specs, so the result drops straight into a
scenario.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

LAWS = ("linear", "exponential", "logistic", "allometric")


def _rate(law: str, p: np.ndarray, area: float) -> float:
    if law == "linear":
        return p[0]
    if law == "exponential":
        return p[0] * area
    if law == "logistic":
        return p[0] * area * (1.0 - area / p[1])
    if law == "allometric":
        return p[0] * area ** 0.75 * (1.0 - (area / p[1]) ** 0.25)
    raise ValueError(law)


def simulate_law(law: str, params, a0: float, t0: float, years) -> np.ndarray:
    """Area at each of ``years`` (≥ t0) under yearly Euler steps of the law."""
    p = np.asarray(params, float)
    out, a, t = [], float(a0), float(t0)
    for y in sorted(years):
        while t < y - 1e-9:
            a = max(a + max(_rate(law, p, a), 0.0), a)
            t += 1.0
        out.append(a)
    return np.array(out)


def fit_law(law: str, years, areas) -> dict:
    """Least-squares fit (relative error) of one law; first point is the anchor."""
    years, areas = np.asarray(years, float), np.asarray(areas, float)
    t0, a0 = years[0], areas[0]
    span = max(years[-1] - t0, 1.0)
    slope = (areas[-1] - a0) / span
    amax = areas.max()
    init = {"linear": ([slope], ([0.0], [np.inf])),
            "exponential": ([max(slope / a0, 1e-4)], ([0.0], [1.0])),
            "logistic": ([max(2 * slope / a0, 1e-3), 2.0 * amax], ([0.0, amax * 1.001], [2.0, amax * 50])),
            "allometric": ([max(slope / a0 ** 0.75, 1e-3), 2.0 * amax], ([0.0, amax * 1.001], [1e3, amax * 50]))}
    x0, (lo, hi) = init[law]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    def resid(p):
        return (simulate_law(law, p, a0, t0, years[1:]) - areas[1:]) / areas[1:]

    if len(years) < 2:
        raise ValueError("need at least two observations")
    res = least_squares(resid, x0, bounds=(lo, hi))
    return {"law": law, "params": res.x.tolist(), "rel_rmse": float(np.sqrt(np.mean(res.fun ** 2))),
            "t0": t0, "a0": a0}


def law_to_demand(fit: dict) -> dict:
    """DEMANDS spec reproducing a fitted law inside a simulation."""
    law, p = fit["law"], fit["params"]
    if law == "linear":
        return {"type": "constant", "cells": p[0]}
    if law == "exponential":
        return {"type": "exponential", "rate": p[0]}
    if law == "logistic":
        return {"type": "logistic", "rate": p[0], "capacity": p[1], "anchor_capacity": False}
    if law == "allometric":
        return {"type": "allometric", "a": p[0], "capacity": p[1], "exponent": 0.75,
                "anchor_capacity": False}
    raise ValueError(law)


def select_law(years, areas, laws=LAWS) -> dict:
    """Pick the law that best predicts the last point from the earlier ones.

    With fewer than three points only one-parameter laws are considered and
    the best in-sample fit wins.
    """
    years, areas = list(map(float, years)), list(map(float, areas))
    nparams = {"linear": 1, "exponential": 1, "logistic": 2, "allometric": 2}
    if len(years) < 3:
        cands = [fit_law(l, years, areas) for l in laws if nparams[l] == 1]
        best = min(cands, key=lambda f: f["rel_rmse"])
        best["selection"] = "in-sample (too few points for hold-out)"
        return best
    scores = {}
    for law in laws:
        if nparams[law] >= len(years) - 1:
            continue
        f = fit_law(law, years[:-1], areas[:-1])
        pred = simulate_law(law, f["params"], areas[0], years[0], [years[-1]])[0]
        scores[law] = abs(pred - areas[-1]) / areas[-1]
    law = min(scores, key=scores.get)
    best = fit_law(law, years, areas)
    best["selection"] = {"holdout_rel_err": {k: round(v, 4) for k, v in scores.items()}}
    return best


def fit_population_scaling(pop_years, pop, years, areas) -> dict:
    """Fit A = c · P^β on observed (P(t), A(t)) pairs (log-log least squares)."""
    p = np.interp(years, pop_years, pop)
    lp, la = np.log(p), np.log(np.asarray(areas, float))
    if len(years) >= 2 and np.ptp(lp) > 0:
        beta, logc = np.polyfit(lp, la, 1)
    else:
        beta, logc = 1.0, la[0] - lp[0]
    return {"c": float(np.exp(logc)), "beta": float(beta)}


def population_demand(pop_years, pop, years, areas, horizon_years) -> dict:
    """A ``series`` demand (cells/year) following the fitted scaling law along
    the population path. The level is anchored to the last observed area."""
    fit = fit_population_scaling(pop_years, pop, years, areas)
    t_last = float(years[-1])
    ys = np.arange(t_last, max(horizon_years) + 1.0)
    target = fit["c"] * np.interp(ys, pop_years, pop) ** fit["beta"]
    cells = np.diff(target)
    return {"type": "series", "years": (ys[1:]).tolist(), "cells": np.maximum(cells, 0).tolist(),
            "_fit": {**fit, "law": "population"}}
