"""Comparing simulated and observed urban maps (calibration / validation)."""

from __future__ import annotations

import numpy as np


def lee_sallee(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / |A ∪ B| — SLEUTH's shape-match index (1 = identical)."""
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def kappa(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Cohen's kappa for binary maps (agreement beyond chance)."""
    a, b = a > 0, b > 0
    if mask is not None:
        a, b = a[mask], b[mask]
    n = a.size
    po = (a == b).sum() / n
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def figure_of_merit(initial: np.ndarray, observed: np.ndarray, simulated: np.ndarray) -> float:
    """Pontius FoM on *change*: hits / (hits + misses + false alarms).

    Stricter than map agreement because unchanged (persistent) cells — usually
    the majority — don't count as success.
    """
    i, o, s = initial > 0, observed > 0, simulated > 0
    obs_change, sim_change = o & ~i, s & ~i
    hits = (obs_change & sim_change).sum()
    misses = (obs_change & ~sim_change).sum()
    false_alarms = (~obs_change & sim_change).sum()
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom else 1.0


def compare_maps(initial, observed, simulated) -> dict[str, float]:
    return {"lee_sallee": lee_sallee(observed, simulated),
            "kappa": kappa(observed, simulated),
            "fom": figure_of_merit(initial, observed, simulated)}


def r_squared(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def density_errors(observed: np.ndarray, simulated: np.ndarray, mask: np.ndarray,
                   reference: np.ndarray | None = None) -> dict[str, float]:
    """Errors of a simulated density map on ``mask`` cells.

    With ``reference`` (e.g. the start-year density, i.e. "nothing changes"),
    also returns ``skill`` = 1 − MAE_sim / MAE_ref: > 0 means the model beats
    persistence, 1 is perfect.
    """
    o, s = observed[mask].astype(float), simulated[mask].astype(float)
    if o.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r": float("nan")}
    err = s - o
    out = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "bias": float(err.mean()),
           "r": float(np.corrcoef(o, s)[0, 1]) if o.std() > 0 and s.std() > 0 else float("nan")}
    if reference is not None:
        mae_ref = float(np.abs(reference[mask].astype(float) - o).mean())
        out["mae_persistence"] = mae_ref
        out["skill"] = 1.0 - out["mae"] / mae_ref if mae_ref > 0 else float("nan")
    return out
