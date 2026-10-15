"""Charts: metric time series, growth composition, ensembles, profiles."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..metrics.profile import radial_profile
from . import style

LABELS = {"urban_area": "Urban area (cells)", "edges": "Edge cells", "clusters": "Clusters",
          "largest_cluster_share": "Largest cluster share", "fractal_dimension": "Fractal dimension",
          "radius_of_gyration": "Radius of gyration (cells)", "new": "New cells per step",
          "compactness": "Compactness", "growth_rate": "Growth rate (%)"}


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=style.INK_2)


def plot_series(runs, keys=("urban_area", "new", "clusters", "fractal_dimension"),
                observed: dict | None = None, cols: int = 2):
    """Small multiples, one metric per panel (no dual axes).

    ``runs``: a History, or {label: History}. ``observed``: {metric: (years, values)}.
    """
    style.apply()
    if not isinstance(runs, dict):
        runs = {"simulated": runs}
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.9 * rows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        for i, (label, h) in enumerate(runs.items()):
            y, v = h.years, h.series(key)
            ax.plot(y, v, color=style.SERIES[i], label=label)
            if 1 < len(runs) <= 4:
                _end_label(ax, y, v, label, style.SERIES[i])
        if observed and key in observed:
            oy, ov = observed[key]
            ax.plot(oy, ov, "o", ms=8, mfc=style.SURFACE, mec=style.INK, mew=1.8, label="observed")
        ax.set_title(LABELS.get(key, key), loc="left")
        ax.set_xlabel("Year")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    if len(runs) > 1 or observed:
        axes.ravel()[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_growth_types(history, kinds=("edge", "road", "spontaneous", "spreading", "anchor")):
    """Stacked contribution of each SLEUTH growth rule per step."""
    style.apply()
    present = [k for k in kinds if np.isfinite(history.series(k)).any()]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if not present:
        ax.text(0.5, 0.5, "No per-rule growth stats for this model", ha="center", transform=ax.transAxes)
        return fig
    y = history.years[1:]
    vals = [np.nan_to_num(history.series(k)[1:]) for k in present]
    ax.stackplot(y, *vals, labels=present, colors=style.SERIES[:len(present)],
                 edgecolor=style.SURFACE, linewidth=1.0)
    ax.set_title("New cells per year by growth rule", loc="left")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", ncol=len(present))
    fig.tight_layout()
    return fig


def plot_ensemble(ensemble, keys=("urban_area", "new"), labels=None, others: dict | None = None):
    """Mean ± 1 s.d. across Monte Carlo runs. ``others`` adds more ensembles for comparison."""
    style.apply()
    ens = {(labels or "scenario"): ensemble, **(others or {})}
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.1), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        for i, (label, e) in enumerate(ens.items()):
            yrs, mu, sd = e.metric(key)
            c = style.SERIES[i]
            ax.fill_between(yrs, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)
            ax.plot(yrs, mu, color=c, label=label)
            if 1 < len(ens) <= 4:
                _end_label(ax, yrs, mu, label, c)
        ax.set_title(LABELS.get(key, key) + " (mean ± 1 s.d.)", loc="left")
        ax.set_xlabel("Year")
    if len(ens) > 1:
        axes.ravel()[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_radial_profiles(grids: dict, center, bin_width: float = 3.0, max_radius=None):
    """Urban density vs distance from a centre for several states (Batty 2008)."""
    style.apply()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, (label, g) in enumerate(grids.items()):
        r, p = radial_profile(g["urban"], center, bin_width, mask=g["water"] == 0,
                              max_radius=max_radius)
        ax.plot(r, p, color=style.SERIES[i], label=label)
        if 1 < len(grids) <= 4:
            ok = np.isfinite(p)
            _end_label(ax, r[ok], p[ok], label, style.SERIES[i])
    ax.set_xlabel("Distance from centre (cells)")
    ax.set_title("Urban density by distance from centre", loc="left")
    ax.set_ylim(0, 1.02)
    if len(grids) > 1:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_anchor_strength(anchors, years):
    """Temporal envelope m(t) of each anchor (announcement → opening → maturity)."""
    style.apply()
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, a in enumerate(anchors[:8]):
        ax.plot(years, [a.strength(y) for y in years], color=style.SERIES[i], label=a.name)
    ax.set_title("Anchor strength over time", loc="left")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
