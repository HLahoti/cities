"""Scalar descriptors of urban form, computed from a grid.

These are what get tracked over time and compared against history:
SLEUTH's calibration statistics (area, edges, clusters), cluster structure,
fractal dimension (Batty & Longley's 'fractal cities'), and centrality.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..registry import Registry

METRICS = Registry("metric")
_EIGHT = np.ones((3, 3), dtype=int)


def _urban(g) -> np.ndarray:
    return (g["urban"] if hasattr(g, "layers") else g) > 0


def labels(urban: np.ndarray):
    return ndimage.label(urban, structure=_EIGHT)


@METRICS.register("urban_area")
def urban_area(g) -> float:
    return float(_urban(g).sum())


@METRICS.register("urban_fraction")
def urban_fraction(g) -> float:
    u = _urban(g)
    land = ~g["blocked"] if hasattr(g, "layers") else np.ones_like(u)
    return float(u.sum() / max(land.sum(), 1))


@METRICS.register("edges")
def edges(g) -> float:
    """Urban cells with at least one non-urban 4-neighbour (SLEUTH 'edges')."""
    u = _urban(g)
    inner = ndimage.binary_erosion(u, structure=ndimage.generate_binary_structure(2, 1),
                                   border_value=1)
    return float((u & ~inner).sum())


@METRICS.register("clusters")
def clusters(g) -> float:
    return float(labels(_urban(g))[1])


@METRICS.register("largest_cluster_share")
def largest_cluster_share(g) -> float:
    lab, n = labels(_urban(g))
    if n == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / sizes.sum())


@METRICS.register("mean_cluster_size")
def mean_cluster_size(g) -> float:
    lab, n = labels(_urban(g))
    return float(_urban(g).sum() / n) if n else 0.0


@METRICS.register("fractal_dimension")
def fractal_dimension(g) -> float:
    """Box-counting dimension of the urban pattern (2 = solid, ~1.7 = DLA-like)."""
    u = _urban(g)
    if u.sum() < 4:
        return float("nan")
    size = 2 ** int(np.floor(np.log2(min(u.shape))))
    u = u[:size, :size]
    sizes, counts = [], []
    s = size // 2
    while s >= 1:
        boxes = u.reshape(size // s, s, size // s, s).any(axis=(1, 3)).sum()
        sizes.append(s)
        counts.append(boxes)
        s //= 2
    sizes, counts = np.array(sizes, float), np.array(counts, float)
    ok = counts > 0
    slope, _ = np.polyfit(np.log(1 / sizes[ok]), np.log(counts[ok]), 1)
    return float(slope)


@METRICS.register("radius_of_gyration")
def radius_of_gyration(g) -> float:
    rr, cc = np.nonzero(_urban(g))
    if rr.size == 0:
        return 0.0
    return float(np.sqrt(((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2).mean()))


@METRICS.register("compactness")
def compactness(g) -> float:
    """Area / area of circle with the radius of gyration (1 ≈ compact disc)."""
    a = urban_area(g)
    rg = radius_of_gyration(g)
    return float(a / (2 * np.pi * rg ** 2)) if rg > 0 else 0.0


@METRICS.register("mean_center_distance")
def mean_center_distance(g) -> float:
    d = g.derive("center_distance")
    u = _urban(g)
    vals = d[u & np.isfinite(d)]
    return float(vals.mean()) if vals.size else float("nan")


def compute(grid, names) -> dict[str, float]:
    return {n: METRICS.get(n)(grid) for n in names}


DEFAULT_METRICS = ["urban_area", "edges", "clusters", "largest_cluster_share",
                   "fractal_dimension", "radius_of_gyration"]
