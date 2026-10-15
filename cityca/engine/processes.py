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


def build_processes(specs) -> list[Process]:
    return [PROCESSES.build(s) for s in (specs or [])]
