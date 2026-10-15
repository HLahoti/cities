"""SLEUTH-style self-modifying urban growth CA (Clarke, Hoppen & Gaydos 1997).

Each step applies four growth rules in order, controlled by five coefficients
(0–100):

1. **Spontaneous** (diffusion): ``diffusion·0.005·diag`` random cells are
   tested and urbanised if suitable — scattered, leapfrog development.
2. **New spreading centres** (breed): each spontaneous cell, with probability
   breed/100, urbanises 2 of its neighbours and becomes a seed.
3. **Edge / organic** (spread): every urban cell with ≥2 urban neighbours,
   with probability spread/100, urbanises one random neighbour.
4. **Road-influenced** (road_gravity, diffusion, breed): ``1+breed`` cells
   from this step's growth look for a road within a road-gravity radius,
   random-walk along the network ``diffusion`` steps and found a new centre
   beside the road.

Every candidate cell must pass a suitability test: slope acceptance
(slope_resistance, critical_slope), exclusion, and any extra ``modifiers``
(factor specs, e.g. ``{"type": "anchor", "gain": 2}`` couples growth to the
anchor potential field). Modifiers also bias where spontaneous growth is
sampled and scale the edge-growth rate of each source cell, so an announced
airport attracts leapfrog development and speeds up nearby expansion.
``anchor_seeding`` (extension to SLEUTH) adds Poisson(anchor_seeding·Σm_k(t))
extra seeds per step placed ∝ Φ⁺; they can breed into new spreading centres.

Self-modification: if the growth rate exceeds ``critical_high`` the
diffusion/breed/spread coefficients are multiplied by ``boom``; below
``critical_low`` by ``bust`` — producing the S-shaped growth curve.
"""

from __future__ import annotations

import random

import numpy as np

from ..kernels import MOORE_OFFSETS
from .base import MODELS, GrowthModel, StepContext
from .factors import build_factors, combine, slope_acceptance

_OFFS = [tuple(o) for o in MOORE_OFFSETS.tolist()]


@MODELS.register("sleuth")
class SleuthModel(GrowthModel):
    name = "sleuth"
    COEFFS = ("diffusion", "breed", "spread", "slope_resistance", "road_gravity")

    def __init__(self, diffusion: float = 20, breed: float = 20, spread: float = 30,
                 slope_resistance: float = 30, road_gravity: float = 30,
                 critical_slope: float = 21.0, modifiers: list | None = None,
                 spontaneous_bias: bool = True, anchor_seeding: float = 0.0,
                 self_modification: dict | bool | None = None):
        self.initial = dict(diffusion=diffusion, breed=breed, spread=spread,
                            slope_resistance=slope_resistance, road_gravity=road_gravity)
        self.critical_slope = critical_slope
        self.modifiers = build_factors(modifiers)
        self.spontaneous_bias = spontaneous_bias
        self.anchor_seeding = anchor_seeding
        sm = {"enabled": True} if self_modification is True else (self_modification or {})
        self.sm = {"enabled": sm.get("enabled", bool(sm)),
                   "critical_high": sm.get("critical_high", 1.3),
                   "critical_low": sm.get("critical_low", 0.97),
                   "boom": sm.get("boom", 1.01),
                   "bust": sm.get("bust", 0.9),
                   "slope_sensitivity": sm.get("slope_sensitivity", 0.1),
                   "road_grav_sensitivity": sm.get("road_grav_sensitivity", 0.01)}
        self.reset(None)

    # --- parameters ------------------------------------------------------------
    def reset(self, grid) -> None:
        for k, v in self.initial.items():
            setattr(self, k, float(v))

    def params(self) -> dict:
        return {k: round(getattr(self, k), 3) for k in self.COEFFS}

    def set_param(self, key, value):
        super().set_param(key, value)
        if key in self.initial:
            self.initial[key] = value

    # --- suitability -------------------------------------------------------------
    def acceptance(self, grid, ctx, modifier: np.ndarray | None = None) -> np.ndarray:
        """Probability that a candidate cell is accepted for development."""
        p = slope_acceptance(grid["slope"], self.critical_slope, self.slope_resistance)
        p = p * np.clip(1.0 - grid["excluded"], 0.0, 1.0)
        p[grid["water"] > 0] = 0.0
        if modifier is None and self.modifiers:
            modifier = combine(self.modifiers, grid, ctx, "product")
        if modifier is not None:
            p = p * modifier
        return np.clip(p, 0.0, 1.0)

    # --- step --------------------------------------------------------------------
    def step(self, grid, ctx: StepContext) -> dict:
        rng = ctx.rng
        h, w = grid.shape
        urban = grid["urban"]
        mod = combine(self.modifiers, grid, ctx, "product") if self.modifiers else None
        accept = self.acceptance(grid, ctx, mod)
        bias = mod if self.spontaneous_bias else None
        new = np.zeros(grid.shape, dtype=bool)

        def try_cells(r, c) -> np.ndarray:
            """Vectorised suitability test; returns mask of accepted candidates."""
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            r, c = r[ok], c[ok]
            good = (urban[r, c] == 0) & ~new[r, c] & (rng.random(r.size) < accept[r, c])
            r, c = r[good], c[good]
            new[r, c] = True
            return r, c

        def random_neighbours(r, c, k):
            """k random Moore neighbours of each (r, c)."""
            if r.size == 0:
                return r, c
            pick = np.argsort(rng.random((r.size, 8)), axis=1)[:, :k]
            off = MOORE_OFFSETS[pick]  # (n, k, 2)
            return (r[:, None] + off[..., 0]).ravel(), (c[:, None] + off[..., 1]).ravel()

        # 1. spontaneous growth
        n_try = int(self.diffusion * 0.005 * np.hypot(h, w))
        if bias is not None:
            wgt = np.maximum(bias, 0).ravel()
            s = wgt.sum()
            idx = (rng.choice(h * w, size=n_try, p=wgt / s) if s > 0
                   else rng.integers(0, h * w, n_try))
        else:
            idx = rng.integers(0, h * w, n_try)
        sr, sc = try_cells(idx // w, idx % w)
        n_spont = sr.size

        # 1b. anchor seeding: anchors generate extra spontaneous seeds,
        #     Poisson(anchor_seeding · Σ m_k(t)), placed ∝ max(Φ, 0).
        n_anchor = 0
        if self.anchor_seeding > 0 and grid.anchors:
            total = sum(a.strength(ctx.year) for a in grid.anchors)
            phi = np.maximum(grid["anchor_potential"], 0).ravel()
            k = rng.poisson(self.anchor_seeding * total) if total > 0 else 0
            if k and phi.sum() > 0:
                idx = rng.choice(h * w, size=k, p=phi / phi.sum())
                ar, ac = try_cells(idx // w, idx % w)
                n_anchor = ar.size
                sr, sc = np.concatenate([sr, ar]), np.concatenate([sc, ac])

        # 2. new spreading centres
        breeders = rng.random(sr.size) < self.breed / 100.0
        nr, nc = random_neighbours(sr[breeders], sc[breeders], 2)
        n_breed = try_cells(nr, nc)[0].size

        # 3. edge (organic) growth from established urban cells
        nb = grid["urban_neighbors"]
        er, ec = np.nonzero((urban > 0) & (nb >= 2) & (nb < 8))
        p_spread = np.full(er.size, self.spread / 100.0)
        if mod is not None:  # modifiers (e.g. anchors) speed up or slow down edge growth
            p_spread = np.clip(p_spread * mod[er, ec], 0.0, 1.0)
        go = rng.random(er.size) < p_spread
        tr, tc = random_neighbours(er[go], ec[go], 1)
        n_edge = try_cells(tr, tc)[0].size

        # 4. road-influenced growth
        n_road = self._road_growth(grid, ctx, new, try_cells, random_neighbours)

        n = grid.urbanize(new)
        stats = {"new": n, "spontaneous": n_spont, "anchor": n_anchor, "spreading": n_breed,
                 "edge": n_edge, "road": n_road}
        if self.sm["enabled"]:
            self._self_modify(grid, n, stats)
        stats.update(self.params())
        return stats

    def _road_growth(self, grid, ctx, new, try_cells, random_neighbours) -> int:
        rng = ctx.rng
        roads = grid["roads"]
        nearest = grid.derive("road_nearest")
        if nearest is None or self.road_gravity <= 0:
            return 0
        dist = grid.derive("road_distance")
        gr, gc = np.nonzero(new)
        if gr.size == 0:
            return 0
        h, w = grid.shape
        max_search = 4.0 * self.road_gravity * (1 + max(h, w) / 2.0) / 100.0
        walk_len = int(self.diffusion)
        py_rng = random.Random(int(rng.integers(2**31)))
        count = 0
        for _ in range(1 + int(self.breed)):
            k = rng.integers(gr.size)
            r, c = gr[k], gc[k]
            if dist[r, c] > max_search:
                continue
            r, c = int(nearest[0, r, c]), int(nearest[1, r, c])
            # random walk along the road network (scalar loop: tiny, hot)
            prev = None
            for _ in range(walk_len):
                opts = [(r + dr, c + dc) for dr, dc in _OFFS
                        if 0 <= r + dr < h and 0 <= c + dc < w and roads[r + dr, c + dc] > 0]
                if len(opts) > 1 and prev in opts:
                    opts.remove(prev)  # don't immediately reverse
                if not opts:
                    break
                prev = (r, c)
                r, c = opts[py_rng.randrange(len(opts))]
            # found a new centre beside the road, then let it breed
            ar, ac = random_neighbours(np.array([r]), np.array([c]), 8)
            for i in range(ar.size):
                cr, cc_ = try_cells(ar[i:i + 1], ac[i:i + 1])
                if cr.size:
                    count += 1
                    br, bc = random_neighbours(cr, cc_, 2)
                    count += try_cells(br, bc)[0].size
                    break
        return count

    def _self_modify(self, grid, n_new: int, stats: dict) -> None:
        sm = self.sm
        n_urban = float(grid["urban"].sum())
        growth_rate = 100.0 * n_new / max(n_urban, 1.0)
        land = float((~grid["blocked"]).sum())
        pct_urban = 100.0 * n_urban / max(land, 1.0)
        stats["growth_rate"] = growth_rate
        if growth_rate > sm["critical_high"]:
            for k in ("diffusion", "breed", "spread"):
                setattr(self, k, min(getattr(self, k) * sm["boom"], 100.0))
            self.slope_resistance = max(self.slope_resistance - pct_urban * sm["slope_sensitivity"], 1.0)
            self.road_gravity = min(self.road_gravity + pct_urban * sm["road_grav_sensitivity"], 100.0)
            stats["regime"] = "boom"
        elif growth_rate < sm["critical_low"]:
            for k in ("diffusion", "breed", "spread"):
                setattr(self, k, max(getattr(self, k) * sm["bust"], 1.0))
            self.slope_resistance = min(self.slope_resistance + pct_urban * sm["slope_sensitivity"], 100.0)
            self.road_gravity = max(self.road_gravity - pct_urban * sm["road_grav_sensitivity"], 1.0)
            stats["regime"] = "bust"
        else:
            stats["regime"] = "steady"
