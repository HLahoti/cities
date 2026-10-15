"""Anchor potential field.

    Φ(x, t) = Σ_k  m_k(t) · K_k( d_k(x) )

* ``m_k(t)``  — the anchor's time-varying strength (``Anchor.strength``)
* ``K_k``     — its decay kernel (see ``cityca.kernels``)
* ``d_k(x)``  — distance from the anchor, either straight-line or *travel*
  distance through a cost surface where roads are fast and water is slow.
  Travel distance is what makes an airport's influence stretch along the
  highway that serves it instead of spreading as a circle.

Distances are cached per anchor and invalidated when roads/water change, so
recomputing Φ each step only costs a weighted sum.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .. import kernels
from .anchor import Anchor

DEFAULT_SPEEDS = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}  # road class -> relative speed


class AnchorField:
    def __init__(self, combine: str = "sum", speeds: dict | None = None,
                 water_cost: float = 25.0, clip_min: float | None = None):
        self.combine = combine
        self.speeds = {int(k): float(v) for k, v in (speeds or DEFAULT_SPEEDS).items()}
        self.water_cost = water_cost
        self.clip_min = clip_min
        self._graph_key = None
        self._graph = None
        self._dist_cache: dict[tuple, np.ndarray] = {}

    # --- distances ------------------------------------------------------------
    def cost_surface(self, grid) -> np.ndarray:
        """Per-cell traversal cost (1 = off-road walking/driving speed)."""
        roads = grid["roads"].astype(int)
        speed = np.full(grid.shape, self.speeds.get(0, 1.0))
        for cls, s in self.speeds.items():
            if cls > 0:
                speed[roads == cls] = s
        cost = 1.0 / speed
        cost[(grid["water"] > 0) & (roads == 0)] = self.water_cost  # roads on water = bridges
        return cost

    def _travel_graph(self, grid):
        key = (grid.version("roads"), grid.version("water"), grid.shape)
        if key != self._graph_key:
            cost = self.cost_surface(grid).ravel()
            h, w = grid.shape
            idx = np.arange(h * w).reshape(h, w)
            rows, cols, vals = [], [], []
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                a = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
                b = idx[max(0, dr):h + min(0, dr) or None, max(0, dc):w + min(0, dc) or None]
                a, b = a.ravel(), b.ravel()
                step = np.hypot(dr, dc)
                wgt = step * 0.5 * (cost[a] + cost[b])
                rows += [a, b]
                cols += [b, a]
                vals += [wgt, wgt]
            self._graph = sparse.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(h * w, h * w))
            self._graph_key = key
            self._dist_cache = {k: v for k, v in self._dist_cache.items() if k[0] == "euclidean"}
        return self._graph

    def distance(self, grid, anchor: Anchor) -> np.ndarray:
        r, c = anchor.rc
        r = int(np.clip(r, 0, grid.height - 1))
        c = int(np.clip(c, 0, grid.width - 1))
        key = (anchor.metric, r, c, grid.shape)
        if key in self._dist_cache:
            return self._dist_cache[key]
        if anchor.metric == "euclidean":
            rr, cc = grid.coords()
            d = np.hypot(rr - anchor.position[1], cc - anchor.position[0])
        elif anchor.metric == "travel":
            g = self._travel_graph(grid)
            d = dijkstra(g, directed=False, indices=r * grid.width + c).reshape(grid.shape)
        else:
            raise ValueError(f"Unknown distance metric '{anchor.metric}'")
        self._dist_cache[key] = d
        return d

    # --- field ----------------------------------------------------------------
    def contribution(self, grid, anchor: Anchor, year: float | None = None) -> np.ndarray:
        year = grid.year if year is None else year
        m = anchor.strength(year)
        if m == 0:
            return np.zeros(grid.shape)
        return m * kernels.decay(anchor.decay, self.distance(grid, anchor))

    def compute(self, grid, year: float | None = None) -> np.ndarray:
        phi = np.zeros(grid.shape)
        for a in grid.anchors:
            contrib = self.contribution(grid, a, year)
            phi = phi + contrib if self.combine == "sum" else np.maximum(phi, contrib)
        if self.clip_min is not None:
            phi = np.maximum(phi, self.clip_min)
        return phi.astype(np.float32)
