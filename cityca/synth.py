"""Synthetic test city ("Riverton") for development without real data.

Builds a coastal river-mouth city: terrain rising inland with a hill ridge,
a meandering river, coastal and riverside highways, radial arterials, a CBD
and a few satellite towns. A hidden 'truth' SLEUTH run then grows it from
1950 to 1990, producing observed snapshots — a twin experiment, so
calibration can be checked against known parameters.

Writes two city files:
* ``<name>_1950.city.json`` — start state with observed 1960–1990 snapshots
  (for replication / calibration);
* ``<name>.city.json`` — the 1990 'present day' (for forecasts / scenarios).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .grid.generators import fractal_noise
from .grid.io import city_from_dict, save_city

TRUTH = {"type": "sleuth", "diffusion": 4, "breed": 15, "spread": 9,
         "slope_resistance": 50, "road_gravity": 35, "critical_slope": 18,
         "self_modification": {"enabled": True, "critical_high": 4.0, "critical_low": 2.0,
                               "boom": 1.02, "bust": 0.95}}


def meander(p0, p1, n: int = 60, amplitude: float = 8.0, smooth: float = 4.0, seed: int = 0):
    """Wiggly polyline from p0 to p1 (low-pass filtered random offsets)."""
    rng = np.random.default_rng(seed)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)[:, None]
    line = p0 + (p1 - p0) * t
    normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
    normal /= np.linalg.norm(normal)
    off = ndimage.gaussian_filter1d(rng.normal(size=n), smooth)
    off = off / (np.abs(off).max() or 1) * amplitude
    off *= np.sin(np.pi * t[:, 0]) ** 0.5  # pin the endpoints
    return (line + off[:, None] * normal).round(2).tolist()


def riverton_doc(size: int = 200, seed: int = 7) -> dict:
    h = w = size
    s = size / 200.0  # scale geometry with grid size
    rng = np.random.default_rng(seed)

    # Coastline on the west: x = 22 + wiggle(y)
    ys = np.arange(h)
    coast = 22 * s + ndimage.gaussian_filter1d(rng.normal(size=h), 6 * s) * 25 * s
    sea = np.indices((h, w))[1] < coast[:, None]

    # Terrain: rises inland, fractal roughness, a hill ridge to the north-east.
    d_coast = ndimage.distance_transform_edt(~sea)
    rough = fractal_noise((h, w), beta=3.3, amplitude=60, seed=seed)
    ridge = np.zeros((h, w))
    rr, cc = np.indices((h, w))
    for x, y, hh, rad in [(130, 35, 420, 11), (150, 50, 380, 10), (168, 30, 450, 12),
                          (110, 22, 300, 9), (175, 165, 350, 12), (155, 180, 300, 10)]:
        ridge += hh * np.exp(-((cc - x * s) ** 2 + (rr - y * s) ** 2) / (2 * (rad * s) ** 2))
    elevation = 1.2 * d_coast / s + rough * np.clip(d_coast / (10 * s), 0, 1) + ridge
    elevation[sea] = -5.0

    river = meander((w - 1, 120 * s), (coast[int(100 * s)] + 2, 100 * s),
                    amplitude=10 * s, seed=seed + 1)
    cbd = [44 * s, 92 * s]
    P = lambda *pts: [[x * s, y * s] for x, y in pts]  # noqa: E731

    features = [
        {"geometry": "polyline", "layer": "water", "value": 1, "width": 3, "coords": river},
        # coastal highway N-S and riverside highway to the east
        {"geometry": "polyline", "layer": "roads", "value": 3,
         "coords": meander((34 * s, 0), (36 * s, h - 1), amplitude=4 * s, seed=seed + 2)},
        {"geometry": "polyline", "layer": "roads", "value": 3,
         "coords": P((40, 92), (80, 104), (120, 108), (160, 106), (199, 112))},
        # radial arterials from the CBD
        {"geometry": "polyline", "layer": "roads", "value": 2, "coords": P((44, 92), (90, 60), (140, 0))},
        {"geometry": "polyline", "layer": "roads", "value": 2, "coords": P((44, 92), (90, 150), (130, 199))},
        {"geometry": "polyline", "layer": "roads", "value": 2, "coords": P((44, 92), (70, 40), (80, 0))},
        # a bridge/arterial south across the river
        {"geometry": "polyline", "layer": "roads", "value": 2, "coords": P((44, 92), (48, 140), (40, 199))},
        # local grid in the core
        *[{"geometry": "polyline", "layer": "roads", "value": 1, "coords": P((32, y), (62, y))}
          for y in (82, 87, 97, 102)],
        *[{"geometry": "polyline", "layer": "roads", "value": 1, "coords": P((x, 78), (x, 106))}
          for x in (38, 50, 56)],
        # initial settlements (1950)
        {"geometry": "circle", "layer": "urban", "value": 1, "center": cbd, "radius": 9 * s},
        {"geometry": "circle", "layer": "urban", "value": 1, "center": [36 * s, 28 * s], "radius": 3.5 * s},
        {"geometry": "circle", "layer": "urban", "value": 1, "center": [86 * s, 63 * s], "radius": 3 * s},
        {"geometry": "circle", "layer": "urban", "value": 1, "center": [125 * s, 108 * s], "radius": 3 * s},
        {"geometry": "circle", "layer": "urban", "value": 1, "center": [38 * s, 160 * s], "radius": 3 * s},
        # protected land: hill park (fully) and coastal wetland (partially)
        {"geometry": "polygon", "layer": "excluded", "value": 1.0,
         "coords": P((150, 20), (185, 18), (190, 55), (160, 62), (140, 45))},
        {"geometry": "polygon", "layer": "excluded", "value": 0.7,
         "coords": P((30, 108), (60, 112), (62, 125), (32, 128))},
    ]
    return {
        "format": "cityca/1",
        "name": "Riverton",
        "year": 1950,
        "grid": {"height": h, "width": w, "cell_size": 100.0 / s, "units": "cells"},
        "meta": {"description": "Synthetic coastal river-mouth city (twin-experiment data)",
                 "truth_model": TRUTH},
        "layers": {"elevation": {"source": "inline", "data": elevation.round(1).tolist()},
                   "water": {"source": "inline", "data": sea.astype(int).tolist()}},
        "features": features,
        "anchors": [
            {"name": "CBD", "kind": "cbd", "position": cbd},
            {"name": "Port", "kind": "port", "position": [30 * s, 104 * s]},
        ],
    }


def make_riverton(out_dir: str | Path = "examples", size: int = 200, seed: int = 7,
                  start: int = 1950, end: int = 1990, every: int = 10) -> tuple[Path, Path]:
    from .engine import MODELS, Simulation

    grid = city_from_dict(riverton_doc(size, seed))
    grid.set_year(start)
    truth = Simulation(grid.copy(), MODELS.build(TRUTH), seed=seed,
                       record={"every": 1, "layers": ["urban", "urbanized_at"], "metrics": []})
    hist = truth.run(until=end)

    out = Path(out_dir)
    hist_grid = grid.copy()
    for y in range(start + every, end + 1, every):
        hist_grid.observed[float(y)] = {"urban": hist.frame_at(y).layers["urban"].copy()}
    p_hist = save_city(hist_grid, out / "riverton_1950.city.json")

    present = truth.grid
    present.meta["history_from"] = start
    p_now = save_city(present, out / "riverton.city.json")
    return p_hist, p_now

