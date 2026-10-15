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

import hashlib
from collections import OrderedDict

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .. import kernels
from .anchor import Anchor

DEFAULT_SPEEDS = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}  # road class -> relative speed

# Travel-distance fields are expensive (one Dijkstra per anchor) and identical
# across Monte Carlo runs on the same network, so they are shared process-wide,
# keyed by a digest of the cost surface.
_TRAVEL_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_GRAPH_CACHE: "OrderedDict[str, sparse.csr_matrix]" = OrderedDict()
_TRAVEL_CACHE_SIZE = 256


class AnchorField:
    def __init__(self, combine: str = "sum", speeds: dict | None = None,
                 water_cost: float = 25.0, clip_min: float | None = None,
                 kind_weights: dict | None = None):
        self.combine = combine
        # Multiplier per anchor kind (e.g. {"station": 0.5}) — lets a whole class of
        # anchors be scaled or calibrated at once.
        self.kind_weights = dict(kind_weights or {})
        self.speeds = {int(k): float(v) for k, v in (speeds or DEFAULT_SPEEDS).items()}
        self.water_cost = water_cost
        self.clip_min = clip_min
        self._graph_key = None
        self._graph = None
        self._cost_digest = None
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
            cost = self.cost_surface(grid)
            digest = hashlib.blake2b(cost.tobytes() + str(cost.shape).encode(),
                                     digest_size=16).hexdigest()
            if digest not in _GRAPH_CACHE:
                _GRAPH_CACHE[digest] = self._build_graph(cost)
                while len(_GRAPH_CACHE) > 8:
                    _GRAPH_CACHE.popitem(last=False)
            self._graph = _GRAPH_CACHE[digest]
            self._cost_digest = digest
            self._graph_key = key
        return self._graph

    @staticmethod
    def _build_graph(cost2d: np.ndarray) -> sparse.csr_matrix:
        h, w = cost2d.shape
        cost = cost2d.ravel()
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, vals = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
            b = idx[max(0, dr):h + min(0, dr) or None, max(0, dc):w + min(0, dc) or None]
            a, b = a.ravel(), b.ravel()
            wgt = np.hypot(dr, dc) * 0.5 * (cost[a] + cost[b])
            rows += [a, b]
            cols += [b, a]
            vals += [wgt, wgt]
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(h * w, h * w))

    def distance(self, grid, anchor: Anchor) -> np.ndarray:
        r, c = anchor.rc
        r = int(np.clip(r, 0, grid.height - 1))
        c = int(np.clip(c, 0, grid.width - 1))
        if anchor.metric == "euclidean":
            key = ("euclidean", float(anchor.position[0]), float(anchor.position[1]), grid.shape)
            if key not in self._dist_cache:
                rr, cc = grid.coords()
                self._dist_cache[key] = np.hypot(rr - anchor.position[1], cc - anchor.position[0])
            return self._dist_cache[key]
        if anchor.metric == "travel":
            g = self._travel_graph(grid)
            key = (self._cost_digest, r, c)
            d = _TRAVEL_CACHE.get(key)
            if d is None:
                d = dijkstra(g, directed=False, indices=r * grid.width + c).reshape(grid.shape)
                d = d.astype(np.float32)
                _TRAVEL_CACHE[key] = d
                while len(_TRAVEL_CACHE) > _TRAVEL_CACHE_SIZE:
                    _TRAVEL_CACHE.popitem(last=False)
            else:
                _TRAVEL_CACHE.move_to_end(key)
            return d
        raise ValueError(f"Unknown distance metric '{anchor.metric}'")

    # --- field ----------------------------------------------------------------
    def contribution(self, grid, anchor: Anchor, year: float | None = None) -> np.ndarray:
        year = grid.year if year is None else year
        m = anchor.strength(year) * self.kind_weights.get(anchor.kind, 1.0)
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
