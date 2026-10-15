"""Derived layers: properties computed from base layers and cached.

Register a new one with::

    @derived("my_property", deps=["urban", "roads"])
    def my_property(grid, radius=3):
        ...
        return array

``deps`` lists base layers (or the pseudo-keys ``__anchors__`` / ``__year__``)
whose changes invalidate the cache. Results may be parameterised; each
parameter combination is cached separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from .. import kernels
from ..registry import Registry

DERIVED = Registry("derived layer")


@dataclass(frozen=True)
class Derived:
    name: str
    fn: Callable
    deps: tuple[str, ...]
    description: str = ""


def derived(name: str, deps: list[str], description: str = ""):
    def deco(fn):
        DERIVED.register(name)(Derived(name, fn, tuple(deps), description or (fn.__doc__ or "").strip()))
        return fn
    return deco


def _edt(mask: np.ndarray, return_indices: bool = False):
    """Distance (cells) from every cell to the nearest True cell of ``mask``."""
    if not mask.any():
        d = np.full(mask.shape, np.inf)
        if return_indices:
            return d, None
        return d
    return ndimage.distance_transform_edt(~mask, return_indices=return_indices)


# --- terrain ------------------------------------------------------------------
@derived("slope", ["elevation"])
def slope(grid):
    """Terrain slope in degrees."""
    gy, gx = np.gradient(grid["elevation"].astype(float), grid.cell_size)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


@derived("hillshade", ["elevation"])
def hillshade(grid, azimuth: float = 315.0, altitude: float = 45.0):
    """0..1 shaded relief, for rendering."""
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    return ls.hillshade(grid["elevation"].astype(float), vert_exag=2.0,
                        dx=grid.cell_size, dy=grid.cell_size)


# --- availability -------------------------------------------------------------
@derived("blocked", ["water", "excluded"])
def blocked(grid):
    """Cells that can never be developed (water or fully excluded)."""
    return (grid["water"] > 0) | (grid["excluded"] >= 1.0)


@derived("developable", ["urban", "water", "excluded"])
def developable(grid):
    """Non-urban cells that could in principle be developed."""
    return (grid["urban"] == 0) & ~grid.derive("blocked")


# --- infrastructure -------------------------------------------------------------
@derived("road_mask", ["roads"])
def road_mask(grid, min_class: int = 1):
    return grid["roads"] >= min_class


@derived("road_distance", ["roads"])
def road_distance(grid, min_class: int = 1):
    """Euclidean distance (cells) to the nearest road of at least ``min_class``."""
    return _edt(grid.derive("road_mask", min_class=min_class)).astype(np.float32)


@derived("road_nearest", ["roads"])
def road_nearest(grid, min_class: int = 1):
    """(2, H, W) row/col indices of the nearest road cell (None if no roads)."""
    _, idx = _edt(grid.derive("road_mask", min_class=min_class), return_indices=True)
    return idx


@derived("rail_distance", ["rail"])
def rail_distance(grid):
    """Euclidean distance (cells) to the nearest rail line."""
    return _edt(grid["rail"] > 0).astype(np.float32)


# --- urban morphology -----------------------------------------------------------
@derived("urban_neighbors", ["urban"])
def urban_neighbors(grid):
    """Number of urban cells in the 3x3 Moore neighbourhood."""
    return kernels.convolve(grid["urban"], kernels.moore(1)).round().astype(np.int16)


@derived("urban_density", ["urban"])
def urban_density(grid, radius: float = 3.0, decay: str | dict | None = None):
    """Distance-weighted share of urban cells within ``radius``."""
    k = kernels.disk(radius, decay)
    return kernels.convolve(grid["urban"], k, normalize=True).astype(np.float32)


@derived("cluster_size_map", ["urban"])
def cluster_size_map(grid):
    """Each urban cell labelled with the size (cells) of its 8-connected cluster."""
    lab, n = ndimage.label(grid["urban"] > 0, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros(grid.shape, np.float32)
    sizes = np.bincount(lab.ravel()).astype(np.float32)
    sizes[0] = 0
    return sizes[lab]


@derived("urban_distance", ["urban"])
def urban_distance(grid):
    """Distance (cells) to the nearest urban cell."""
    return _edt(grid["urban"] > 0).astype(np.float32)


@derived("edge", ["urban"])
def edge(grid):
    """Urban cells with at least one non-urban Moore neighbour."""
    u = grid["urban"] > 0
    return u & (grid.derive("urban_neighbors") < 8)


@derived("water_distance", ["water"])
def water_distance(grid):
    return _edt(grid["water"] > 0).astype(np.float32)


@derived("center_distance", ["__anchors__"])
def center_distance(grid, kinds: tuple = ("cbd",)):
    """Euclidean distance (cells) to the nearest anchor of the given kinds."""
    pts = [a.rc for a in grid.anchors if a.kind in kinds]
    if not pts:
        return np.full(grid.shape, np.inf, dtype=np.float32)
    rr, cc = grid.coords()
    d = np.full(grid.shape, np.inf)
    for r, c in pts:
        d = np.minimum(d, np.hypot(rr - r, cc - c))
    return d.astype(np.float32)


@derived("age", ["urbanized_at", "__year__"])
def age(grid):
    """Years since each cell was urbanised (NaN if not urban)."""
    return (grid.year - grid["urbanized_at"]).astype(np.float32)
