"""Demand: *how much* the city grows per step (the CA decides *where*).

Separating quantity from location is the standard 'constrained CA' design
(White & Engelen) and is where the organism analogy lives: total urban area A
follows a growth law like a body's mass, e.g. West–Brown–Enquist's ontogenetic
growth equation

    dA/dt = a · A^(3/4) · [1 - (A/K)^(1/4)]

whose 3/4 exponent comes from the branching supply network (the 1997 Science
paper in research/). Anchors can raise the carrying capacity K: a new airport
creates jobs, so the 'adult size' of the organism grows.
"""

from __future__ import annotations

import numpy as np

from ..registry import Registry

DEMANDS = Registry("demand law")


class Demand:
    name = "demand"

    def __init__(self, anchor_capacity: bool = True):
        self.anchor_capacity = anchor_capacity
        self._carry = 0.0

    def reset(self):
        self._carry = 0.0

    def rate(self, area: float, grid, ctx) -> float:
        raise NotImplementedError

    def capacity_bonus(self, grid, year: float) -> float:
        """Extra capacity from anchors: Σ capacity_k · m_k(t)/mass_k."""
        if not self.anchor_capacity:
            return 0.0
        bonus = 0.0
        for a in grid.anchors:
            cap = a.meta.get("capacity", 0.0)
            if cap and a.mass:
                bonus += cap * a.strength(year) / a.mass
        return bonus

    def amount(self, grid, ctx) -> int:
        """Whole cells to allocate this step (fractions carried forward)."""
        area = float(grid["urban"].sum())
        x = max(self.rate(area, grid, ctx), 0.0) * ctx.dt + self._carry
        n = int(np.floor(x))
        self._carry = x - n
        return n

    def __repr__(self):
        return f"{type(self).__name__}({ {k: v for k, v in vars(self).items() if not k.startswith('_')} })"


@DEMANDS.register("constant")
class ConstantDemand(Demand):
    def __init__(self, cells: float = 50.0, **kw):
        super().__init__(**kw)
        self.cells = cells

    def rate(self, area, grid, ctx):
        return self.cells


@DEMANDS.register("exponential")
class ExponentialDemand(Demand):
    def __init__(self, rate: float = 0.03, **kw):
        super().__init__(**kw)
        self.r = rate

    def rate(self, area, grid, ctx):
        return self.r * area


@DEMANDS.register("logistic")
class LogisticDemand(Demand):
    def __init__(self, rate: float = 0.05, capacity: float = 10000.0, **kw):
        super().__init__(**kw)
        self.r, self.capacity = rate, capacity

    def rate(self, area, grid, ctx):
        k = self.capacity + self.capacity_bonus(grid, ctx.year)
        return self.r * area * (1.0 - area / k)


@DEMANDS.register("allometric")
class AllometricDemand(Demand):
    """West–Brown–Enquist ontogenetic growth: a·A^b·(1 - (A/K)^(1-b))."""

    def __init__(self, a: float = 0.5, capacity: float = 10000.0, exponent: float = 0.75, **kw):
        super().__init__(**kw)
        self.a, self.capacity, self.exponent = a, capacity, exponent

    def rate(self, area, grid, ctx):
        k = self.capacity + self.capacity_bonus(grid, ctx.year)
        b = self.exponent
        return self.a * area ** b * (1.0 - (area / k) ** (1.0 - b))


@DEMANDS.register("series")
class SeriesDemand(Demand):
    """Explicit cells-per-year from data, linearly interpolated by year."""

    def __init__(self, years: list, cells: list, _fit: dict | None = None, **kw):
        super().__init__(**kw)
        self.years, self.cells = list(years), list(cells)
        self.fit_info = _fit  # provenance when produced by demand_fit

    def rate(self, area, grid, ctx):
        return float(np.interp(ctx.year, self.years, self.cells))
