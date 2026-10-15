"""Factors: per-cell fields that score how attractive a cell is for growth.

Factors are the "properties" side of the experiment: each one turns some layer
(or derived layer) into a non-negative field, and models combine whichever
factors a scenario lists. Adding a property to the model is a matter of
adding a layer to the city file and a factor here (or using the generic
``layer`` factor).

Every factor has a ``weight``. In a ``product`` combination it is an exponent
(P = Π f_i^w_i, the White–Engelen form); in a ``sum`` combination it is a
coefficient (P = Σ w_i f_i, the logistic-regression form).
"""

from __future__ import annotations

import numpy as np

from .. import kernels
from ..registry import Registry

FACTORS = Registry("factor")


class Factor:
    name = "factor"

    def __init__(self, weight: float = 1.0, enabled: bool = True, label: str | None = None):
        self.weight = weight
        self.enabled = enabled
        self.label = label or self.name

    def __call__(self, grid, ctx) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label}, w={self.weight})"


def _floor(f: np.ndarray, floor: float) -> np.ndarray:
    return floor + (1.0 - floor) * f


@FACTORS.register("neighborhood")
class NeighborhoodFactor(Factor):
    """Share of developed land around the cell — the CA 'contagion' term."""
    name = "neighborhood"

    def __init__(self, radius: float = 2.0, decay: str | dict | None = None,
                 floor: float = 0.0, power: float = 1.0, **kw):
        super().__init__(**kw)
        self.radius, self.decay, self.floor, self.power = radius, decay, floor, power

    def __call__(self, grid, ctx):
        d = grid.derive("urban_density", radius=self.radius, decay=self.decay)
        return _floor(np.clip(d, 0, 1) ** self.power, self.floor)


@FACTORS.register("urban_field")
class UrbanFieldFactor(Factor):
    """How much city surrounds a cell, at the scale of kilometres.

    Gaussian-weighted urban share within ``radius`` cells, mapped through
    min(1, (share / ref)^power). On the city fringe the field is high; around
    an isolated village it is tiny, and it rises as the city approaches. Used
    to modulate edge growth, so villages grow because the city reaches them,
    not on their own.
    """
    name = "urban_field"

    def __init__(self, radius: float = 20.0, ref: float = 0.3, power: float = 1.0,
                 floor: float = 0.0, **kw):
        super().__init__(**kw)
        self.radius, self.ref, self.power, self.floor = radius, ref, power, floor

    def __call__(self, grid, ctx):
        d = grid.derive("urban_density", radius=self.radius,
                        decay={"type": "gaussian", "scale": self.radius / 2.0})
        return _floor(np.clip(d / self.ref, 0.0, 1.0) ** self.power, self.floor)


@FACTORS.register("cluster_size")
class ClusterSizeFactor(Factor):
    """Size of the urban cluster a cell belongs to (or touches): min(1, (S/ref)^power).

    An allometric alternative to ``urban_field``: big organisms grow, small
    ones barely do. Non-urban cells get the size of the largest adjacent cluster.
    """
    name = "cluster_size"

    def __init__(self, ref: float = 2000.0, power: float = 0.5, floor: float = 0.0, **kw):
        super().__init__(**kw)
        self.ref, self.power, self.floor = ref, power, floor

    def __call__(self, grid, ctx):
        from scipy import ndimage
        size = grid.derive("cluster_size_map")
        size = ndimage.grey_dilation(size, size=(3, 3))
        return _floor(np.clip(size / self.ref, 0.0, 1.0) ** self.power, self.floor)


@FACTORS.register("slope")
class SlopeFactor(Factor):
    """SLEUTH slope acceptance: ((crit - s)/crit)^(resistance/50), 0 above crit."""
    name = "slope"

    def __init__(self, critical: float = 21.0, resistance: float = 50.0, **kw):
        super().__init__(**kw)
        self.critical, self.resistance = critical, resistance

    def __call__(self, grid, ctx):
        return slope_acceptance(grid["slope"], self.critical, self.resistance)


def slope_acceptance(slope: np.ndarray, critical: float, resistance: float) -> np.ndarray:
    v = np.clip((critical - slope) / critical, 0.0, 1.0)
    return v ** (resistance / 50.0)


@FACTORS.register("road_access")
class RoadAccessFactor(Factor):
    """Decay with distance to the road network (accessibility)."""
    name = "road_access"

    def __init__(self, scale: float = 5.0, min_class: int = 1, decay: str = "exponential",
                 floor: float = 0.1, **kw):
        super().__init__(**kw)
        self.scale, self.min_class, self.decay, self.floor = scale, min_class, decay, floor

    def __call__(self, grid, ctx):
        d = grid.derive("road_distance", min_class=self.min_class)
        if not np.isfinite(d).any():
            return np.ones(grid.shape)
        return _floor(kernels.decay({"type": self.decay, "scale": self.scale}, d), self.floor)


@FACTORS.register("center_access")
class CenterAccessFactor(Factor):
    """Decay with distance to the nearest CBD-type anchor (monocentric rent gradient)."""
    name = "center_access"

    def __init__(self, scale: float = 40.0, kinds=("cbd",), decay: str = "exponential",
                 floor: float = 0.05, **kw):
        super().__init__(**kw)
        self.scale, self.kinds, self.decay, self.floor = scale, tuple(kinds), decay, floor

    def __call__(self, grid, ctx):
        d = grid.derive("center_distance", kinds=self.kinds)
        if not np.isfinite(d).any():
            return np.ones(grid.shape)
        return _floor(kernels.decay({"type": self.decay, "scale": self.scale}, d), self.floor)


@FACTORS.register("anchor")
class AnchorFactor(Factor):
    """Coupling to the anchor potential Φ: max(1 + gain·Φ, floor).

    Φ = 0 leaves the cell unchanged; positive Φ boosts it; the negative core of
    a 'dog' kernel (airport noise zone) suppresses it.
    """
    name = "anchor"

    def __init__(self, gain: float = 2.0, floor: float = 0.0, **kw):
        super().__init__(**kw)
        self.gain, self.floor = gain, floor

    def __call__(self, grid, ctx):
        phi = grid["anchor_potential"]
        return np.maximum(1.0 + self.gain * phi, self.floor)


@FACTORS.register("excluded")
class ExcludedFactor(Factor):
    """1 - excluded (protected land, parks, ...)."""
    name = "excluded"

    def __call__(self, grid, ctx):
        return np.clip(1.0 - grid["excluded"], 0.0, 1.0)


@FACTORS.register("water_amenity")
class WaterAmenityFactor(Factor):
    name = "water_amenity"

    def __init__(self, scale: float = 5.0, floor: float = 0.7, **kw):
        super().__init__(**kw)
        self.scale, self.floor = scale, floor

    def __call__(self, grid, ctx):
        d = grid["water_distance"]
        return _floor(kernels.decay({"type": "exponential", "scale": self.scale}, d), self.floor)


@FACTORS.register("noise")
class NoiseFactor(Factor):
    """White–Engelen stochastic perturbation v = 1 + (-ln r)^alpha."""
    name = "noise"

    def __init__(self, alpha: float = 0.5, **kw):
        super().__init__(**kw)
        self.alpha = alpha

    def __call__(self, grid, ctx):
        r = ctx.rng.random(grid.shape)
        return 1.0 + (-np.log(np.clip(r, 1e-12, 1.0))) ** self.alpha


@FACTORS.register("layer")
class LayerFactor(Factor):
    """Generic factor from any (base or derived) layer.

    transform: ``linear`` (min-max normalised), ``inverse`` (1 - linear),
    ``decay`` (treat the layer as a distance; uses ``decay``/``scale``),
    ``threshold`` (layer >= ``threshold``), ``raw``.
    """
    name = "layer"

    def __init__(self, layer: str, transform: str = "linear", scale: float = 10.0,
                 decay: str = "exponential", threshold: float = 0.5,
                 floor: float = 0.0, params: dict | None = None, **kw):
        kw.setdefault("label", layer)
        super().__init__(**kw)
        self.layer, self.transform, self.scale = layer, transform, scale
        self.decay, self.threshold, self.floor = decay, threshold, floor
        self.layer_params = params or {}

    def __call__(self, grid, ctx):
        a = (grid.derive(self.layer, **self.layer_params) if self.layer_params
             else grid[self.layer]).astype(float)
        if self.transform == "raw":
            f = a
        elif self.transform in ("linear", "inverse"):
            finite = np.isfinite(a)
            lo, hi = (a[finite].min(), a[finite].max()) if finite.any() else (0.0, 1.0)
            f = np.where(finite, (a - lo) / (hi - lo or 1.0), 0.0)
            if self.transform == "inverse":
                f = 1.0 - f
        elif self.transform == "decay":
            f = kernels.decay({"type": self.decay, "scale": self.scale}, a)
        elif self.transform == "threshold":
            f = (a >= self.threshold).astype(float)
        else:
            raise ValueError(f"Unknown transform '{self.transform}'")
        return _floor(f, self.floor)


@FACTORS.register("learned")
class LearnedFactor(Factor):
    """Transition probability from a trained ``cityca.learn.LearnedModel``.

    ``model_file`` is a saved model (relative paths in scenarios are resolved
    against the scenario file). ``power`` sharpens (>1) or flattens (<1) the
    surface before allocation.
    """
    name = "learned"
    _cache: dict = {}

    def __init__(self, model_file: str, power: float = 1.0, freeze: list | None = None, **kw):
        super().__init__(**kw)
        # Layers whose features stay at their start-of-forecast values (e.g. ["density",
        # "density_change"]): the model was trained on features measured at the start
        # of each interval, so letting a densification process rewrite them shifts the
        # inputs away from what the model learned.
        self.freeze = list(freeze or [])
        self._frozen = None
        self.model_file, self.power = str(model_file), power
        import os
        key = (self.model_file, os.path.getmtime(self.model_file))  # reload if the file changes
        if key not in self._cache:
            from ..learn.learners import LearnedModel
            self._cache[key] = LearnedModel.load(self.model_file)
        self.model = self._cache[key]

    def __call__(self, grid, ctx):
        frozen = None
        if self.freeze:
            if self._frozen is None or self._frozen[0] != id(grid):
                cols = [i for i, sp in enumerate(self.model.features.specs) if sp["layer"] in self.freeze]
                self._frozen = (id(grid), cols, self.model.features.matrix(grid)[..., cols].copy())
            frozen = (self._frozen[1], self._frozen[2])
        p = self.model.predict_grid(grid, frozen)
        return p ** self.power if self.power != 1 else p


def build_factors(specs) -> list[Factor]:
    return [FACTORS.build(s) for s in (specs or [])]


def combine(factors: list[Factor], grid, ctx, mode: str = "product",
            out: dict | None = None) -> np.ndarray:
    """Combine enabled factors into one field. ``out`` collects each factor's field."""
    active = [f for f in factors if f.enabled]
    if mode == "product":
        p = np.ones(grid.shape)
        for f in active:
            v = np.maximum(f(grid, ctx), 0.0)
            if out is not None:
                out[f.label] = v
            p *= v ** f.weight if f.weight != 1 else v
        return p
    if mode == "sum":
        p = np.zeros(grid.shape)
        for f in active:
            v = f(grid, ctx)
            if out is not None:
                out[f.label] = v
            p += f.weight * v
        return np.maximum(p, 0.0)
    if mode == "logistic":
        z = np.zeros(grid.shape)
        for f in active:
            v = f(grid, ctx)
            if out is not None:
                out[f.label] = v
            z += f.weight * v
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown combine mode '{mode}'")
