"""Processes: extra per-step dynamics that run after the growth model.

The growth model decides *which* cells become urban; processes evolve other
state on top of that. The first one is densification: urban tissue thickens
over time, like an organism's tissue maturing after it first forms.

Processes are listed in a scenario under ``"processes"`` and are calibratable
via dotted paths (``processes.0.rate``).
"""

from __future__ import annotations

import numpy as np

from ..registry import Registry
from .factors import build_factors, combine

PROCESSES = Registry("process")


class Process:
    name = "process"

    def reset(self, grid) -> None:
        """Called once before the first step."""

    def step(self, grid, ctx) -> dict:
        raise NotImplementedError

    def set_param(self, key, value):
        setattr(self, key, value)


@PROCESSES.register("densify")
class Densify(Process):
    """Logistic densification of built-up density ``d`` on urban cells.

        d ← d + rate · d · (1 − d / K(x)) · dt
        K(x) = k_min + (k_max − k_min) · clip(combine(capacity_factors), 0, 1)

    Newly urbanised cells start at ``new_density`` (or keep a higher observed
    value). With no capacity factors K = k_max everywhere. Typical factors:
    ``urban_field`` (cells surrounded by city can pack more), the anchor
    potential (stations, CBD), slope.

    If the grid has no ``density`` layer, one is initialised from the urban
    mask and the local urban share (``init_radius``), so any city works.
    """
    name = "densify"

    def __init__(self, rate: float = 0.15, new_density: float = 0.35, k_min: float = 0.6,
                 k_max: float = 1.0, capacity_factors: list | None = None,
                 combine: str = "sum", init_radius: float = 1.5):
        self.rate, self.new_density = rate, new_density
        self.k_min, self.k_max = k_min, k_max
        self.capacity_factors = build_factors(
            capacity_factors if capacity_factors is not None
            else [{"type": "urban_field", "radius": 10, "ref": 0.5}])
        self.combine_mode = combine
        self.init_radius = init_radius

    def reset(self, grid) -> None:
        if "density" in grid.layers and float(grid["density"].max()) > 0:
            return
        u = grid["urban"] > 0
        local = grid.derive("urban_density", radius=self.init_radius)
        grid["density"] = np.where(u, np.clip(0.3 + 0.7 * local, 0, 1), 0.0)

    def capacity(self, grid, ctx) -> np.ndarray:
        if not self.capacity_factors:
            return np.full(grid.shape, self.k_max)
        c = np.clip(combine(self.capacity_factors, grid, ctx, self.combine_mode), 0.0, 1.0)
        return self.k_min + (self.k_max - self.k_min) * c

    def step(self, grid, ctx) -> dict:
        d = grid.ensure("density")
        u = grid["urban"] > 0
        new = u & (grid["urbanized_at"] == grid.year)
        d[new] = np.maximum(d[new], self.new_density)
        k = self.capacity(grid, ctx)
        grow = u & (d < k) & (d > 0)
        dd = d[grow]
        d[grow] = np.clip(dd + self.rate * ctx.dt * dd * (1.0 - dd / k[grow]), 0.0, 1.0)
        grid.touch("density")
        return {"mean_density": float(d[u].mean()) if u.any() else 0.0}


@PROCESSES.register("learned_densify")
class LearnedDensify(Process):
    """Densification from a trained ``cityca.learn.densify.DensityModel``.

    Each step every cell in the density domain densifies with probability
    1 − (1 − p)^(dt/period) by the predicted amount (``mode="stochastic"``,
    lumpy like real infill), or by that probability × amount
    (``mode="expected"``, smooth). Newly urban cells start at the model's
    ``new_density`` (overridable). Non-urban cells may thicken (``nonurban``)
    but are capped at ``nonurban_cap``: becoming urban is the growth model's job.
    """
    name = "learned_densify"
    _cache: dict = {}

    def __init__(self, model_file: str, mode: str = "stochastic", new_density: float | None = None,
                 nonurban_cap: float = 0.24, init_radius: float = 1.5, nonurban: bool = True):
        self.model_file, self.mode = str(model_file), mode
        import os
        key = (self.model_file, os.path.getmtime(self.model_file))  # reload if the file changes
        if key not in self._cache:
            from ..learn.densify import DensityModel
            self._cache[key] = DensityModel.load(self.model_file)
        self.model = self._cache[key]
        self.new_density = self.model.new_density if new_density is None else new_density
        self.nonurban_cap, self.init_radius = nonurban_cap, init_radius
        self.nonurban = nonurban  # False: only urban cells densify

    def reset(self, grid) -> None:
        Densify.reset(self, grid)

    def step(self, grid, ctx) -> dict:
        d = grid.ensure("density")
        u = grid["urban"] > 0
        new = u & (grid["urbanized_at"] == grid.year)
        d[new] = np.maximum(d[new], self.new_density)
        grid.touch("density")
        p, m, dom = self.model.predict_grid(grid)
        p_step = 1.0 - (1.0 - p) ** (ctx.dt / self.model.period)
        if self.mode == "stochastic":
            inc = np.where(ctx.rng.random(grid.shape) < p_step, m, 0.0)
        elif self.mode == "expected":
            inc = p_step * m
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
        inc[~dom] = 0.0
        if not self.nonurban:
            inc[~u] = 0.0
        d += inc
        np.minimum(d, np.where(u, 1.0, np.maximum(self.nonurban_cap, d - inc)), out=d)
        grid.touch("density")
        return {"mean_density": float(d[u].mean()) if u.any() else 0.0}


@PROCESSES.register("track_lags")
class TrackLags(Process):
    """Keeps ``urban_lag`` / ``density_lag`` equal to the state ``lag_years`` ago.

    Added automatically by ``Simulation`` when the grid carries lag layers, so
    momentum features stay current during a forecast.
    """
    name = "track_lags"

    def __init__(self, lag_years: float | None = None):
        self.lag_years = lag_years
        self.snapshots: list = []

    def reset(self, grid) -> None:
        self.lag = float(self.lag_years or grid.meta.get("lag_years", 2))
        self.snapshots = []
        if "urban_lag" in grid.layers:
            dl = grid["density_lag"] if "density_lag" in grid.layers else None
            self.snapshots.append((grid.year - self.lag, grid["urban_lag"].copy(),
                                   None if dl is None else dl.copy()))
        self._push(grid)

    def _push(self, grid):
        d = grid["density"].copy() if "density" in grid.layers else None
        self.snapshots.append((grid.year, grid["urban"].copy(), d))

    def step(self, grid, ctx) -> dict:
        self._push(grid)
        target = grid.year - self.lag
        old = [s for s in self.snapshots if s[0] <= target + 1e-9]
        if old:
            y, u, d = old[-1]
            grid["urban_lag"] = u
            if d is not None:
                grid["density_lag"] = d
            self.snapshots = [s for s in self.snapshots if s[0] >= y]
        return {}


def build_processes(specs) -> list[Process]:
    return [PROCESSES.build(s) for s in (specs or [])]
