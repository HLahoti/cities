"""The city state: a stack of named dense layers on a regular grid.

``CityGrid`` is deliberately dumb storage plus caching:

* base layers are set with ``grid[name] = array`` (or mutated in place and then
  ``grid.touch(name)``); each assignment bumps a version counter;
* derived layers (slope, distance-to-road, urban density, ...) are computed on
  demand by functions in ``grid.derived`` and cached until one of their
  dependencies changes version.

Reading ``grid[name]`` works for both kinds, so models never need to know
whether a property was supplied in the city file or computed.
"""

from __future__ import annotations

import copy as _copy
import json
from typing import Any, Iterable

import numpy as np

from ..anchors.anchor import Anchor
from .derived import DERIVED
from .layers import spec_for

ANCHORS_KEY = "__anchors__"
YEAR_KEY = "__year__"


class CityGrid:
    def __init__(self, shape: tuple[int, int], cell_size: float = 100.0,
                 year: float = 0.0, name: str = "city", meta: dict | None = None):
        self.shape = (int(shape[0]), int(shape[1]))
        self.cell_size = float(cell_size)  # metres per cell edge
        self.year = float(year)
        self.name = name
        self.meta: dict[str, Any] = dict(meta or {})
        self.layers: dict[str, np.ndarray] = {}
        self.anchors: list[Anchor] = []
        # Observed historical snapshots for calibration: {year: {layer: array}}
        self.observed: dict[float, dict[str, np.ndarray]] = {}
        self._versions: dict[str, int] = {}
        self._cache: dict[tuple, tuple[tuple, Any]] = {}
        for core in ("urban", "urbanized_at", "elevation", "water", "excluded", "roads"):
            self.ensure(core)

    # --- layer access ---------------------------------------------------------
    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.layers:
            return self.layers[name]
        if name in DERIVED:
            return self.derive(name)
        raise KeyError(f"No layer '{name}'. Base: {sorted(self.layers)}; "
                       f"derived: {DERIVED.names()}")

    def __setitem__(self, name: str, arr: Any) -> None:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = np.full(self.shape, arr)
        if arr.shape != self.shape:
            raise ValueError(f"Layer '{name}' has shape {arr.shape}, grid is {self.shape}")
        self.layers[name] = spec_for(name).coerce(arr).copy()
        self.touch(name)

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def has(self, name: str) -> bool:
        return name in self.layers or name in DERIVED

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def ensure(self, name: str) -> np.ndarray:
        """Return a base layer, creating it with its default fill if missing."""
        if name not in self.layers:
            self.layers[name] = spec_for(name).empty(self.shape)
            self.touch(name)
        return self.layers[name]

    def touch(self, name: str) -> None:
        """Mark a layer (or pseudo-key like anchors/year) as changed."""
        self._versions[name] = self._versions.get(name, 0) + 1

    def version(self, name: str) -> int:
        return self._versions.get(name, 0)

    # --- derived layers -------------------------------------------------------
    def derive(self, name: str, **params) -> Any:
        d = DERIVED.get(name)
        key = (name, json.dumps(params, sort_keys=True, default=str))
        stamp = tuple(self.version(dep) for dep in d.deps)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        value = d.fn(self, **params)
        self._cache[key] = (stamp, value)
        return value

    # --- mutation helpers used by models --------------------------------------
    def urbanize(self, mask: np.ndarray, year: float | None = None) -> int:
        """Convert cells in ``mask`` to urban; returns number of new cells."""
        urban = self.layers["urban"]
        new = mask.astype(bool) & (urban == 0)
        n = int(new.sum())
        if n:
            urban[new] = 1
            self.layers["urbanized_at"][new] = self.year if year is None else year
            self.touch("urban")
            self.touch("urbanized_at")
        return n

    def set_year(self, year: float) -> None:
        self.year = float(year)
        self.touch(YEAR_KEY)

    def add_anchor(self, anchor: Anchor | dict) -> Anchor:
        if isinstance(anchor, dict):
            anchor = Anchor.from_dict(anchor)
        self.anchors.append(anchor)
        self.touch(ANCHORS_KEY)
        return anchor

    def remove_anchor(self, name: str) -> None:
        self.anchors = [a for a in self.anchors if a.name != name]
        self.touch(ANCHORS_KEY)

    # --- coordinates ----------------------------------------------------------
    def in_bounds(self, r, c):
        return (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Row and column index arrays of shape (H, W)."""
        return np.indices(self.shape)

    # --- misc -----------------------------------------------------------------
    def copy(self) -> "CityGrid":
        g = CityGrid.__new__(CityGrid)
        g.shape, g.cell_size, g.year, g.name = self.shape, self.cell_size, self.year, self.name
        g.meta = _copy.deepcopy(self.meta)
        g.layers = {k: v.copy() for k, v in self.layers.items()}
        g.anchors = [_copy.deepcopy(a) for a in self.anchors]
        g.observed = {y: {k: v.copy() for k, v in d.items()} for y, d in self.observed.items()}
        g._versions = dict(self._versions)
        g._cache = {}
        return g

    def layer_names(self, include_derived: bool = False) -> list[str]:
        names = sorted(self.layers)
        return names + DERIVED.names() if include_derived else names

    def summary(self) -> str:
        u = self.layers["urban"]
        lines = [f"CityGrid '{self.name}' {self.height}x{self.width} cells "
                 f"@ {self.cell_size:g} m, year {self.year:g}",
                 f"  urban cells: {int(u.sum())} ({100 * u.mean():.1f}%)",
                 f"  base layers: {', '.join(sorted(self.layers))}",
                 f"  anchors: {', '.join(f'{a.name}<{a.kind}>' for a in self.anchors) or '-'}"]
        if self.observed:
            lines.append(f"  observed years: {sorted(self.observed)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<CityGrid {self.name} {self.shape} year={self.year:g}>"

    def iter_layers(self, names: Iterable[str]):
        for n in names:
            yield n, self[n]
