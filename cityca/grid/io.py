"""City file format (``*.city.json``).

A city file describes the initial state as raster *layers* plus vector
*features* burned on top, *anchors*, and optional *observed* historical
snapshots used for calibration::

    {
      "format": "cityca/1",
      "name": "Riverton",
      "year": 1950,
      "grid": {"height": 200, "width": 200, "cell_size": 100, "units": "cells"},
      "layers": {
        "elevation": {"source": "generator", "generator": "fractal_noise",
                      "params": {"amplitude": 120, "seed": 3}},
        "excluded":  {"source": "constant", "value": 0},
        "urban":     {"source": "npz", "key": "urban"}          # sidecar file
      },
      "features": [
        {"geometry": "polyline", "layer": "roads", "value": 3, "width": 1,
         "coords": [[0, 100], [199, 120]]},
        {"geometry": "polygon",  "layer": "water", "value": 1,
         "coords": [[0, 0], [40, 0], [30, 60], [0, 50]]},
        {"geometry": "circle",   "layer": "urban", "value": 1,
         "center": [100, 100], "radius": 6}
      ],
      "anchors": [{"name": "CBD", "kind": "cbd", "position": [100, 100]}],
      "observed": [{"year": 1970, "layers": {"urban": {"source": "npz", "key": "obs_1970_urban"}}}]
    }

Layer sources are pluggable (``SOURCES`` registry): constant, inline, npy,
npz, image, generator. Coordinates in features/anchors are (x, y) in cells
unless ``grid.units == "m"``, in which case they are metres.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..anchors.anchor import Anchor
from ..registry import Registry
from .generators import GENERATORS
from .raster import burn, feature_mask
from .state import CityGrid

FORMAT = "cityca/1"
# Layers the engine rewrites every step; not part of a city's saved state.
TRANSIENT = {"anchor_potential", "transition_potential"}
SOURCES = Registry("layer source")


class _Ctx:
    """Resolution context for layer sources (base dir, shape, lazy npz)."""

    def __init__(self, base: Path, shape: tuple[int, int], npz_path: Path | None):
        self.base, self.shape, self._npz_path, self._npz = base, shape, npz_path, None

    def npz(self, path: str | None = None):
        p = self.base / path if path else self._npz_path
        if p is None:
            raise ValueError("npz source used but no sidecar file given")
        if path is not None:
            return np.load(p)
        if self._npz is None:
            self._npz = np.load(p)
        return self._npz


@SOURCES.register("constant")
def _src_constant(spec, ctx):
    return np.full(ctx.shape, spec["value"])


@SOURCES.register("inline")
def _src_inline(spec, ctx):
    return np.asarray(spec["data"])


@SOURCES.register("npy")
def _src_npy(spec, ctx):
    return np.load(ctx.base / spec["path"])


@SOURCES.register("npz")
def _src_npz(spec, ctx):
    return ctx.npz(spec.get("path"))[spec["key"]]


@SOURCES.register("generator")
def _src_generator(spec, ctx):
    return GENERATORS.get(spec["generator"])(ctx.shape, **spec.get("params", {}))


@SOURCES.register("image")
def _src_image(spec, ctx):
    """Read a PNG/JPG. Options:

    * ``colors``: {"#rrggbb": value, ...} exact-colour legend (nearest colour wins)
    * otherwise greyscale 0..1 * ``scale`` + ``offset``; ``threshold`` binarises.
    The image is resampled (nearest) to the grid shape if needed.
    """
    from PIL import Image

    img = Image.open(ctx.base / spec["path"]).convert("RGB")
    if img.size != (ctx.shape[1], ctx.shape[0]):
        img = img.resize((ctx.shape[1], ctx.shape[0]), Image.NEAREST)
    rgb = np.asarray(img, dtype=float)
    if "colors" in spec:
        legend = [(np.array([int(k[i:i + 2], 16) for i in (1, 3, 5)], float), v)
                  for k, v in spec["colors"].items()]
        dists = np.stack([np.linalg.norm(rgb - c, axis=-1) for c, _ in legend])
        values = np.array([v for _, v in legend])
        return values[dists.argmin(0)]
    grey = rgb.mean(-1) / 255.0
    if spec.get("invert"):
        grey = 1.0 - grey
    if "threshold" in spec:
        return (grey >= spec["threshold"]).astype(float)
    return grey * spec.get("scale", 1.0) + spec.get("offset", 0.0)


def resolve_layer(spec: dict | float | list, ctx: _Ctx) -> np.ndarray:
    if isinstance(spec, (int, float)):
        spec = {"source": "constant", "value": spec}
    elif isinstance(spec, list):
        spec = {"source": "inline", "data": spec}
    return np.asarray(SOURCES.get(spec["source"])(spec, ctx))


def _sidecar_for(path: Path) -> Path:
    name = path.name
    stem = name[: -len(".city.json")] if name.endswith(".city.json") else path.stem
    return path.with_name(stem + ".npz")


def city_from_dict(doc: dict, base: Path | str = ".", npz_path: Path | None = None) -> CityGrid:
    base = Path(base)
    g = doc["grid"]
    shape = (int(g["height"]), int(g["width"]))
    grid = CityGrid(shape, cell_size=g.get("cell_size", 100.0), year=doc.get("year", 0),
                    name=doc.get("name", "city"), meta=doc.get("meta", {}))
    scale = grid.cell_size if g.get("units", "cells") in ("m", "meters", "metres") else 1.0
    ctx = _Ctx(base, shape, npz_path)

    for name, spec in doc.get("layers", {}).items():
        grid[name] = resolve_layer(spec, ctx)

    for feat in doc.get("features", []):
        layer = grid.ensure(feat["layer"])
        burn(layer, feature_mask(shape, feat, scale), feat.get("value", 1), feat.get("op", "set"))
        grid.touch(feat["layer"])

    # Cells urban at load time get the start year unless a history was supplied.
    u = grid["urban"] > 0
    ua = grid["urbanized_at"]
    missing = u & np.isnan(ua)
    if missing.any():
        ua[missing] = grid.year
        grid.touch("urbanized_at")

    for a in doc.get("anchors", []):
        a = dict(a)
        a["position"] = tuple(np.asarray(a["position"], float) / scale)
        grid.add_anchor(Anchor.from_dict(a))

    for obs in doc.get("observed", []):
        grid.observed[float(obs["year"])] = {
            k: resolve_layer(s, ctx) for k, s in obs["layers"].items()}
    return grid


def load_city(path: str | Path) -> CityGrid:
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Unsupported city format {doc.get('format')!r}")
    sidecar = _sidecar_for(path)
    return city_from_dict(doc, base=path.parent, npz_path=sidecar if sidecar.exists() else None)


def save_city(grid: CityGrid, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write ``<name>.city.json`` plus a ``<name>.npz`` sidecar with all arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_for(path)
    arrays: dict[str, np.ndarray] = {}
    layers = {}
    for name, arr in grid.layers.items():
        if name in TRANSIENT:
            continue
        if np.all(arr == arr.flat[0]) and not np.isnan(arr.flat[0]):
            layers[name] = {"source": "constant", "value": arr.flat[0].item()}
        else:
            arrays[name] = arr
            layers[name] = {"source": "npz", "key": name}
    observed = []
    for year, obs in sorted(grid.observed.items()):
        entry = {}
        for name, arr in obs.items():
            key = f"obs_{year:g}_{name}"
            arrays[key] = arr
            entry[name] = {"source": "npz", "key": key}
        observed.append({"year": year, "layers": entry})
    doc = {
        "format": FORMAT,
        "name": grid.name,
        "year": grid.year,
        "grid": {"height": grid.height, "width": grid.width,
                 "cell_size": grid.cell_size, "units": "cells"},
        "meta": grid.meta,
        "layers": layers,
        "features": [],
        "anchors": [a.to_dict() for a in grid.anchors],
        "observed": observed,
        **(extra or {}),
    }
    np.savez_compressed(sidecar, **arrays)
    path.write_text(json.dumps(doc, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
