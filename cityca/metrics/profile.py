"""Radial density profiles (Batty 2008, Fig. 1C: density decays with distance)."""

from __future__ import annotations

import numpy as np


def radial_profile(values: np.ndarray, center: tuple[float, float], bin_width: float = 2.0,
                   mask: np.ndarray | None = None, max_radius: float | None = None):
    """Mean of ``values`` in rings around ``center`` (x, y).

    Returns (ring mid-radii, mean value per ring). With ``values = urban`` this
    is the urban density gradient; ``mask`` excludes e.g. water from rings.
    """
    rr, cc = np.indices(values.shape)
    d = np.hypot(cc - center[0], rr - center[1])
    keep = np.ones(values.shape, bool) if mask is None else mask
    rmax = max_radius or d[keep].max()
    edges = np.arange(0, rmax + bin_width, bin_width)
    idx = np.digitize(d[keep], edges) - 1
    v = values[keep].astype(float)
    sums = np.bincount(idx, weights=v, minlength=len(edges))
    cnts = np.bincount(idx, minlength=len(edges))
    n = len(edges) - 1
    with np.errstate(invalid="ignore"):
        prof = sums[:n] / cnts[:n]
    return 0.5 * (edges[:-1] + edges[1:]), prof


def density_decay_fit(r: np.ndarray, density: np.ndarray):
    """Fit density ≈ D0·exp(-r/λ) (Clark's law). Returns (D0, λ)."""
    ok = np.isfinite(density) & (density > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(r[ok], np.log(density[ok]), 1)
    return float(np.exp(intercept)), float(-1 / slope) if slope < 0 else float("inf")
