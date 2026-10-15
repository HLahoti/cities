"""Feature sets: per-cell predictors built from a grid's layers.

A feature is any base or derived layer, optionally transformed and smoothed:

    {"layer": "road_distance", "params": {"min_class": 2}, "transform": "log1p"}
    {"layer": "density", "smooth": 3}

``FeatureSet.matrix(grid)`` returns an (H, W, F) stack, recomputed from the
grid's current state, so a learned model can be applied at every simulation
step as urban land (and hence neighbourhood features) changes.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

_MATRIX_CACHE: dict = {}

TRANSFORMS = {
    "none": lambda a: a,
    "log1p": lambda a: np.log1p(np.maximum(a, 0)),
    "sqrt": lambda a: np.sqrt(np.maximum(a, 0)),
}

# A broad default: neighbourhood structure at several scales, access,
# terrain, constraints, anchors and the cell's own partial built-up share.
DEFAULT_FEATURES = [
    {"layer": "urban_density", "params": {"radius": 1.5}, "name": "urban_share_r1.5"},
    {"layer": "urban_density", "params": {"radius": 4, "decay": {"type": "gaussian", "scale": 2}},
     "name": "urban_share_r4"},
    {"layer": "urban_density", "params": {"radius": 12, "decay": {"type": "gaussian", "scale": 6}},
     "name": "urban_share_r12"},
    {"layer": "urban_density", "params": {"radius": 30, "decay": {"type": "gaussian", "scale": 15}},
     "name": "urban_share_r30"},
    {"layer": "urban_distance", "transform": "log1p"},
    {"layer": "density", "name": "own_built_share"},
    {"layer": "density", "smooth": 2, "name": "built_share_s2"},
    {"layer": "road_distance", "params": {"min_class": 1}, "transform": "log1p", "name": "road_dist_any"},
    {"layer": "road_distance", "params": {"min_class": 2}, "transform": "log1p", "name": "road_dist_arterial"},
    {"layer": "road_distance", "params": {"min_class": 3}, "transform": "log1p", "name": "road_dist_highway"},
    {"layer": "rail_distance", "transform": "log1p"},
    {"layer": "water_distance", "transform": "log1p"},
    {"layer": "center_distance", "transform": "log1p"},
    {"layer": "anchor_potential"},
    {"layer": "slope"},
    {"layer": "elevation"},
    {"layer": "excluded"},
    # Momentum: what changed over the last lag period (zeros if the city has no lag layers)
    {"layer": "recent_growth", "params": {"sigma": 4}, "name": "recent_growth_s4"},
    {"layer": "recent_growth", "params": {"sigma": 12}, "name": "recent_growth_s12"},
    {"layer": "density_change", "params": {"sigma": 0}, "name": "own_density_change"},
    {"layer": "density_change", "params": {"sigma": 2}, "name": "density_change_s2"},
]


def feature_name(spec: dict) -> str:
    if "name" in spec:
        return spec["name"]
    n = spec["layer"]
    if spec.get("params"):
        n += "(" + ",".join(f"{k}={v}" for k, v in sorted(spec["params"].items())) + ")"
    if spec.get("transform", "none") != "none":
        n = f"{spec['transform']}({n})"
    if spec.get("smooth"):
        n += f"~s{spec['smooth']}"
    return n


class FeatureSet:
    def __init__(self, specs: list | None = None):
        self.specs = [dict(s) if isinstance(s, dict) else {"layer": s}
                      for s in (specs if specs is not None else DEFAULT_FEATURES)]
        self.names = [feature_name(s) for s in self.specs]

    def layer(self, grid, spec: dict) -> np.ndarray:
        name = spec["layer"]
        if name not in grid.layers and name in ("density", "rail", "anchor_potential"):
            a = np.zeros(grid.shape)
        elif spec.get("params"):
            a = grid.derive(name, **spec["params"])
        else:
            a = grid[name]
        a = np.asarray(a, dtype=np.float64)
        finite = np.isfinite(a)
        if not finite.all():  # e.g. distance to a feature that doesn't exist yet
            a = np.where(finite, a, a[finite].max() if finite.any() else 0.0)
        a = TRANSFORMS[spec.get("transform", "none")](a)
        if spec.get("smooth"):
            a = ndimage.gaussian_filter(a, spec["smooth"])
        return a

    def matrix(self, grid) -> np.ndarray:
        """(H, W, F) feature stack for the grid's current state.

        Cached per grid state, so a learned growth model and a learned density
        model with the same features compute them once per step.
        """
        key = (self.to_json(), id(grid), grid.year,
               tuple(grid.version(k) for k in ("urban", "density", "roads", "water", "excluded",
                                               "anchor_potential", "rail", "__anchors__")))
        hit = _MATRIX_CACHE.get(key[:2])
        if hit is not None and hit[0] == key:
            return hit[1]
        m = np.stack([self.layer(grid, s) for s in self.specs], axis=-1).astype(np.float32)
        _MATRIX_CACHE[key[:2]] = (key, m)
        if len(_MATRIX_CACHE) > 8:
            _MATRIX_CACHE.pop(next(iter(_MATRIX_CACHE)))
        return m

    def to_json(self) -> str:
        return json.dumps(self.specs)

    def __len__(self):
        return len(self.specs)
