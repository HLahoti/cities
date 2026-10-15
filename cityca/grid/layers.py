"""Layer specifications.

A layer is a dense (H, W) array holding one property of every cell. Known
layers get a ``LayerSpec`` describing dtype, default fill and how to draw them;
any other name is still allowed (it is treated as a float continuous layer),
so new properties can be added to a city file without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..registry import Registry

LAYERS = Registry("layer spec")


@dataclass(frozen=True)
class LayerSpec:
    name: str
    dtype: str = "float32"
    default: float = 0.0
    kind: str = "continuous"  # continuous | binary | categorical | count
    description: str = ""
    cmap: str = "viridis"

    def empty(self, shape: tuple[int, int]) -> np.ndarray:
        return np.full(shape, self.default, dtype=self.dtype)

    def coerce(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if self.kind == "binary":
            return (arr > 0).astype(self.dtype)
        return arr.astype(self.dtype, copy=False)


def _spec(**kw) -> LayerSpec:
    s = LayerSpec(**kw)
    LAYERS.register(s.name)(s)
    return s


# --- Core state -------------------------------------------------------------
_spec(name="urban", dtype="uint8", kind="binary", cmap="Greys",
      description="1 if the cell is developed/urbanised")
_spec(name="density", dtype="float32", cmap="cityca_density",
      description="Built-up density: share of the cell covered by buildings (0..1)")
_spec(name="urbanized_at", dtype="float32", default=np.nan, cmap="plasma",
      description="Year the cell became urban (NaN if never)")

# --- Physical environment ---------------------------------------------------
_spec(name="elevation", description="Terrain height in metres", cmap="terrain")
_spec(name="water", dtype="uint8", kind="binary", cmap="Blues",
      description="1 for sea/lakes/rivers (never developable)")
_spec(name="excluded", dtype="float32", cmap="Greens",
      description="0..1 probability that development is refused "
                  "(parks, protected land, military). Water is always excluded.")

# --- Infrastructure ---------------------------------------------------------
_spec(name="roads", dtype="uint8", kind="categorical", cmap="Greys",
      description="Road class: 0 none, 1 local, 2 arterial, 3 highway")
_spec(name="rail", dtype="uint8", kind="binary", cmap="Greys",
      description="1 on rail lines")

# --- Socio-economic (optional, used if present) -----------------------------
_spec(name="landuse", dtype="uint8", kind="categorical", cmap="tab10",
      description="Land-use class code (see city metadata for the legend)")
_spec(name="population", description="Residents per cell", cmap="magma")
_spec(name="jobs", description="Jobs per cell", cmap="magma")
_spec(name="land_value", description="Relative land value", cmap="cividis")
_spec(name="zoning", dtype="float32", default=1.0, cmap="RdYlGn",
      description="0..1 planning permissiveness multiplier")

# --- Fields written by the engine ---------------------------------------------
_spec(name="anchor_potential", description="Summed anchor potential field Φ",
      cmap="inferno")
_spec(name="transition_potential", description="Last-step transition potential",
      cmap="inferno")


def spec_for(name: str) -> LayerSpec:
    if name in LAYERS:
        return LAYERS.get(name)
    return LayerSpec(name=name)
