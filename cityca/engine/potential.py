"""Constrained transition-potential CA (White & Engelen style).

    P(x) = combine(f_1(x), ..., f_n(x))        # factors (any properties)
    n    = demand(t)                           # how much growth this step
    new  = allocate(P restricted to developable cells, n)

This is the most modular model: the scenario chooses which factors (cell
properties) enter P, how they combine, which growth law sets the quantity,
and how cells are drawn. It is the natural test bed for "which properties
explain this city's growth?".
"""

from __future__ import annotations

import numpy as np

from .allocation import ALLOCATORS
from .base import MODELS, GrowthModel, StepContext
from .demand import DEMANDS
from .factors import build_factors, combine

DEFAULT_FACTORS = [
    {"type": "neighborhood", "radius": 2, "floor": 0.02},
    {"type": "slope", "critical": 21, "resistance": 40},
    {"type": "road_access", "scale": 6, "floor": 0.2},
    {"type": "anchor", "gain": 2.0},
    {"type": "excluded"},
    {"type": "noise", "alpha": 0.5},
]


@MODELS.register("potential")
class PotentialModel(GrowthModel):
    name = "potential"

    def __init__(self, factors: list | None = None, combine: str = "product",
                 demand: dict | str | None = None, allocator: dict | str | None = None,
                 store_factors: bool = False):
        self.factors = build_factors(factors if factors is not None else DEFAULT_FACTORS)
        self.combine_mode = combine
        self.demand = DEMANDS.build(demand or {"type": "logistic", "rate": 0.04, "capacity": 8000})
        self.allocator = ALLOCATORS.build(allocator or {"type": "roulette", "gamma": 3.0})
        self.store_factors = store_factors
        self.last_factors: dict[str, np.ndarray] = {}

    def reset(self, grid) -> None:
        self.demand.reset()

    def potential(self, grid, ctx) -> np.ndarray:
        out = {} if self.store_factors else None
        p = combine(self.factors, grid, ctx, self.combine_mode, out)
        p[~grid["developable"]] = 0.0
        if out is not None:
            self.last_factors = out
        return p

    def step(self, grid, ctx: StepContext) -> dict:
        p = self.potential(grid, ctx)
        n_req = self.demand.amount(grid, ctx)
        mask = self.allocator(p, n_req, ctx.rng)
        grid["transition_potential"] = p
        n = grid.urbanize(mask)
        return {"new": n, "demand": n_req}

    def params(self) -> dict:
        return {"factors": [f.label for f in self.factors if f.enabled],
                "combine": self.combine_mode, "demand": type(self.demand).__name__}


@MODELS.register("neighborhood")
class NeighborhoodModel(GrowthModel):
    """Batty's (1997) probabilistic neighbourhood rule.

    IF an empty cell has between ``min_neighbors`` and ``max_neighbors``
    developed cells in its Moore neighbourhood THEN it develops with
    probability p. Each failed attempt lowers that cell's probability to
    p^k (k = attempts), which leaves a realistic share of vacant land.
    Optional ``modifiers`` (factors) scale p, e.g. slope or anchors.
    """
    name = "neighborhood"

    def __init__(self, p: float = 0.3, min_neighbors: int = 1, max_neighbors: int = 8,
                 vacancy_decay: bool = True, modifiers: list | None = None):
        self.p, self.min_neighbors, self.max_neighbors = p, min_neighbors, max_neighbors
        self.vacancy_decay = vacancy_decay
        self.modifiers = build_factors(modifiers)
        self._attempts = None

    def reset(self, grid) -> None:
        self._attempts = None

    def step(self, grid, ctx: StepContext) -> dict:
        if self._attempts is None or self._attempts.shape != grid.shape:
            self._attempts = np.zeros(grid.shape, dtype=np.int32)
        nb = grid["urban_neighbors"]
        cand = grid["developable"] & (nb >= self.min_neighbors) & (nb <= self.max_neighbors)
        p = np.full(grid.shape, self.p)
        if self.vacancy_decay:
            p = p ** (1 + self._attempts)
        if self.modifiers:
            p = p * combine(self.modifiers, grid, ctx, "product")
        hit = cand & (ctx.rng.random(grid.shape) < np.clip(p, 0, 1))
        self._attempts[cand & ~hit] += 1
        return {"new": grid.urbanize(hit), "candidates": int(cand.sum())}

    def params(self) -> dict:
        return {"p": self.p, "min_neighbors": self.min_neighbors, "max_neighbors": self.max_neighbors}
