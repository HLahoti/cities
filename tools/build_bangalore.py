"""Build a real Bangalore city file (2004 start) from open data.

Sources (downloaded into data/raw/ by the commands in examples/bangalore/README.md):

* DLR World Settlement Footprint Evolution v1, tile 76_12 (CC BY 4.0) —
  year each ~30 m pixel was first detected as settlement, 1985–2015.
* DLR World Settlement Footprint 2019 v1, tile 76_12 (CC BY 4.0) — ~10 m.
* OpenStreetMap via Overpass (ODbL) — roads, rail, water, protected land.
* AWS Terrain Tiles (Terrarium encoding; SRTM-derived) — elevation.
* Nominatim (OSM) — anchor locations.

Run (isolated mode, so nothing from data/ can shadow imports):

    python3 -I tools/build_bangalore.py

Outputs examples/bangalore/bangalore_2004.city.json (+ .npz) and the
scenario files under examples/bangalore/scenarios/.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityca.grid.raster import polygon_mask, polyline_mask  # noqa: E402
from cityca.grid.state import CityGrid  # noqa: E402
from cityca.grid.io import save_city  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "examples" / "bangalore"

# --- grid: aligned to the WSF-Evolution pixel grid, 5x5 pixels per cell (~150 m)
SOUTH, NORTH, WEST, EAST = 12.75, 13.30, 77.35, 77.90
EVO_PX = 0.00026949458523585647
EVO_LON0, EVO_LAT0 = 75.98992718812492, 14.010215002656471
W19_PX = 8.983152841195215e-05
W19_LON0, W19_LAT0 = 75.98992718812491, 14.010035339599645
CELL_PX = 5
CELL_DEG = CELL_PX * EVO_PX

START_YEAR = 2004
EVO_OBSERVED = [2006, 2008, 2010, 2012, 2015]
URBAN_THRESHOLD = 0.25  # share of built 30 m pixels for a cell to count as urban

# Anchors (lat, lon from Nominatim/OSM; see data/raw/nominatim)
CITY_ANCHORS = [
    {"name": "CBD", "kind": "cbd", "lat": 12.9772, "lon": 77.5900},
    {"name": "HAL Airport", "kind": "airport", "lat": 12.9515, "lon": 77.6706,
     "footprint_radius": 0, "access_road": 0},
    {"name": "Electronic City", "kind": "subcenter", "lat": 12.8497, "lon": 77.6650, "mass": 0.6},
    {"name": "ITPL Whitefield", "kind": "subcenter", "lat": 12.9877, "lon": 77.7369, "mass": 0.5},
    {"name": "Peenya", "kind": "industrial", "lat": 13.0160, "lon": 77.5045},
]
KIA = {"name": "KIA", "kind": "airport", "lat": 13.1976, "lon": 77.7075,
       "announce_year": 2004, "open_year": 2008, "footprint_radius": 7, "access_road": 3}

# Roads that did not exist in 2004 (dropped from the start state).
POST_2004 = re.compile(r"NICE|Mysuru Expressway|Chennai Expressway|Satellite Town Ring Road|"
                       r"Dabaspete|Devanahalli Bypass|Electronic City Flyover|Peripheral Ring Road",
                       re.I)
ROAD_CLASS = {"motorway": 3, "motorway_link": 3, "trunk": 3, "trunk_link": 3,
              "primary": 2, "primary_link": 2, "secondary": 1}


# --------------------------------------------------------------------------------
class Geo:
    """Cell <-> lon/lat for the model grid (cell (r, c) centre at integer (c, r))."""

    def __init__(self):
        self.c0 = int(round((WEST - EVO_LON0) / EVO_PX))
        self.r0 = int(round((EVO_LAT0 - NORTH) / EVO_PX))
        self.ncols = int(round((EAST - WEST) / EVO_PX)) // CELL_PX
        self.nrows = int(round((NORTH - SOUTH) / EVO_PX)) // CELL_PX
        self.west = EVO_LON0 + self.c0 * EVO_PX
        self.north = EVO_LAT0 - self.r0 * EVO_PX

    @property
    def shape(self):
        return self.nrows, self.ncols

    def xy(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon - self.west) / CELL_DEG - 0.5, (self.north - lat) / CELL_DEG - 0.5

    def lonlat(self, x, y):
        return self.west + (np.asarray(x) + 0.5) * CELL_DEG, self.north - (np.asarray(y) + 0.5) * CELL_DEG

    def cell_size_m(self):
        lat = math.radians((self.north + SOUTH) / 2)
        dy = CELL_DEG * 110_574
        dx = CELL_DEG * 111_320 * math.cos(lat)
        return math.sqrt(dx * dy), dx, dy

    def as_meta(self):
        return {"crs": "EPSG:4326", "west": self.west, "north": self.north,
                "cell_deg": CELL_DEG, "rows": self.nrows, "cols": self.ncols,
                "note": "cell (r, c) spans lon west+c*cell_deg.., lat north-r*cell_deg.."}


def _read_tif(path: Path) -> np.ndarray:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path))


def built_fraction(geo: Geo):
    """WSF-Evolution: per-cell share of built pixels by year, plus WSF2019."""
    evo = _read_tif(RAW / "wsf_evo" / "WSFevolution_v1_76_12.tif")
    h, w = geo.shape
    win = evo[geo.r0:geo.r0 + h * CELL_PX, geo.c0:geo.c0 + w * CELL_PX].astype(np.int32)
    blocks = win.reshape(h, CELL_PX, w, CELL_PX)
    years = list(range(1985, 2016))
    frac = {y: ((blocks > 0) & (blocks <= y)).mean(axis=(1, 3)) for y in years}
    del evo

    w19 = _read_tif(RAW / "wsf2019" / "WSF2019_v1_76_12.tif")
    k = CELL_PX * 3  # WSF2019 pixels are exactly a third of an Evo pixel
    r19 = int(round((W19_LAT0 - geo.north) / W19_PX))
    c19 = int(round((geo.west - W19_LON0) / W19_PX))
    win19 = w19[r19:r19 + h * k, c19:c19 + w * k]
    frac[2019] = (win19 > 0).reshape(h, k, w, k).mean(axis=(1, 3))
    return frac


def elevation(geo: Geo, z: int = 11) -> np.ndarray:
    """Sample Terrarium terrain tiles at cell centres (bilinear)."""
    from PIL import Image
    from scipy import ndimage

    n = 2 ** z

    def tile_xy(lon, lat):
        x = (lon + 180) / 360 * n
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
        return x, y

    x0, y0 = tile_xy(geo.west, geo.north)
    x1, y1 = tile_xy(EAST, SOUTH)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    tdir = RAW / "terrain"
    tdir.mkdir(parents=True, exist_ok=True)
    mosaic = np.zeros((len(ty) * 256, len(tx) * 256))
    for j, y in enumerate(ty):
        for i, x in enumerate(tx):
            f = tdir / f"terrarium_{z}_{x}_{y}.png"
            if not f.exists():
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
                urllib.request.urlretrieve(url, f)
            rgb = np.asarray(Image.open(f).convert("RGB"), float)
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = (
                rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768)
    rr, cc = np.indices(geo.shape)
    lon, lat = geo.lonlat(cc, rr)
    px = (lon + 180) / 360 * n
    py = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
    return ndimage.map_coordinates(mosaic, [(py - ty[0]) * 256 - 0.5, (px - tx[0]) * 256 - 0.5],
                                   order=1, mode="nearest")


# --- OSM ---------------------------------------------------------------------------
def osm(name):
    return json.loads((RAW / "osm" / f"{name}.json").read_text())["elements"]


def way_xy(geo, geom):
    return np.c_[geo.xy([p["lon"] for p in geom], [p["lat"] for p in geom])]


def stitch(parts: list[list[tuple]]) -> list[list[tuple]]:
    """Join open way segments into closed rings by matching endpoints."""
    parts = [list(p) for p in parts if len(p) > 1]
    rings = []
    while parts:
        ring = parts.pop()
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, p in enumerate(parts):
                if p[0] == ring[-1]:
                    ring += p[1:]
                elif p[-1] == ring[-1]:
                    ring += p[::-1][1:]
                elif p[-1] == ring[0]:
                    ring = p[:-1] + ring
                elif p[0] == ring[0]:
                    ring = p[::-1][:-1] + ring
                else:
                    continue
                parts.pop(i)
                changed = True
                break
        if len(ring) > 3:
            rings.append(ring)
    return rings


def polygons(geo, el):
    """Outer rings of a way or multipolygon relation, in cell coordinates."""
    if el["type"] == "way":
        g = el.get("geometry") or []
        if len(g) > 3 and (g[0]["lat"], g[0]["lon"]) == (g[-1]["lat"], g[-1]["lon"]):
            return [way_xy(geo, g)]
        return []
    parts = [[(p["lon"], p["lat"]) for p in m.get("geometry") or []]
             for m in el.get("members", []) if m.get("role") in ("outer", "") and m.get("geometry")]
    out = []
    for ring in stitch(parts):
        lon, lat = zip(*ring)
        out.append(np.c_[geo.xy(lon, lat)])
    return out


def area_m2(geo, ring):
    _, dx, dy = geo.cell_size_m()
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * dx * dy


def build_roads(geo):
    roads = np.zeros(geo.shape, np.uint8)
    later = {"nice": [], "mysuru": [], "ballari": []}
    for el in osm("roads"):
        t = el.get("tags", {})
        cls = ROAD_CLASS.get(t.get("highway"))
        if not cls or not el.get("geometry"):
            continue
        name, ref = t.get("name", ""), t.get("ref", "")
        xy = way_xy(geo, el["geometry"])
        line = xy.round(2).tolist()
        if "NICE" in name:
            later["nice"].append(line)
            continue
        if "Mysuru Expressway" in name:
            later["mysuru"].append(line)
            continue
        if POST_2004.search(name) or "NE7" in ref:
            continue
        lat_max = max(p["lat"] for p in el["geometry"])
        if ("Ballari" in name or "Bellary" in name) and "NH44" in ref and lat_max > 13.04:
            later["ballari"].append(line)  # NH7 north of Hebbal: arterial until 2014
            cls = 2
        if "NH648" in ref:
            cls = 1  # state highways in 2004; upgraded to NH648 much later
        m = polyline_mask(geo.shape, xy)
        roads[m] = np.maximum(roads[m], cls)
    return roads, later


def build_rail(geo):
    rail = np.zeros(geo.shape, bool)
    for el in osm("rail"):
        t = el.get("tags", {})
        if el.get("geometry") and t.get("service") not in ("yard", "siding", "spur"):
            rail |= polyline_mask(geo.shape, way_xy(geo, el["geometry"]))
    return rail


def build_water(geo):
    water = np.zeros(geo.shape, bool)
    for el in osm("water"):
        if el.get("tags", {}).get("water") in ("swimming_pool", "pond_small", "fountain"):
            continue
        for ring in polygons(geo, el):
            water |= polygon_mask(geo.shape, ring)
    return water


def build_protected(geo):
    excluded = np.zeros(geo.shape, np.float32)
    kia = np.zeros(geo.shape, bool)
    names = []
    for el in osm("protected"):
        t = el.get("tags", {})
        name = t.get("name", "")
        if t.get("aeroway") == "aerodrome" and "Kempegowda" in name:
            for ring in polygons(geo, el):
                kia |= polygon_mask(geo.shape, ring)
            continue
        if t.get("landuse") == "military" or t.get("aeroway") == "aerodrome" or \
                t.get("boundary") in ("national_park", "protected_area") or \
                t.get("leisure") == "nature_reserve":
            value = 1.0
        elif t.get("leisure") == "park":
            value = 1.0
        elif t.get("landuse") == "forest":
            value = 0.8
        else:
            continue
        for ring in polygons(geo, el):
            if t.get("leisure") == "park" and area_m2(geo, ring) < 50_000:
                continue  # only large parks (Lalbagh, Cubbon, ...) block growth
            m = polygon_mask(geo.shape, ring)
            if m.any():
                excluded[m] = np.maximum(excluded[m], value)
                if value >= 1 and m.sum() > 200:
                    names.append(name or t.get("landuse") or t.get("leisure"))
    return excluded, kia, names


# --------------------------------------------------------------------------------
def main():
    geo = Geo()
    size, dx, dy = geo.cell_size_m()
    print(f"grid {geo.shape}, cell ≈ {dx:.0f} m (E-W) × {dy:.0f} m (N-S)")

    frac = built_fraction(geo)
    urban_by = {y: frac[y] >= URBAN_THRESHOLD for y in frac if y <= 2015}
    # Harmonise WSF2019 (10 m, finer building outlines) with WSF-Evolution (30 m):
    # scale the threshold by the ratio of mean built share on 2015-urban cells,
    # and assume built land does not revert (2019 ⊇ 2015).
    u15 = urban_by[2015]
    t19 = URBAN_THRESHOLD * frac[2019][u15].mean() / frac[2015][u15].mean()
    urban_by[2019] = (frac[2019] >= t19) | u15
    print(f"  WSF2019 harmonised threshold {t19:.3f}")
    for y in (1985, 1995, 2004, 2010, 2015, 2019):
        print(f"  urban {y}: {int(urban_by[y].sum()):6d} cells")

    roads, later = build_roads(geo)
    water = build_water(geo)
    excluded, kia_area, big_protected = build_protected(geo)
    roads[kia_area] = 0  # airport-internal roads did not exist in 2004
    rail = build_rail(geo)
    elev = elevation(geo)
    print(f"  roads {int((roads > 0).sum())} cells, water {int(water.sum())}, "
          f"protected {int((excluded >= 1).sum())}, rail {int(rail.sum())}, "
          f"elevation {elev.min():.0f}–{elev.max():.0f} m")

    grid = CityGrid(geo.shape, cell_size=round(size, 1), year=START_YEAR, name="Bangalore",
                    meta={"description": "Bengaluru, 2004, from WSF-Evolution / OSM / SRTM",
                          "georef": geo.as_meta(), "urban_threshold": URBAN_THRESHOLD,
                          "urban_threshold_2019": round(float(t19), 4),
                          "cell_size_xy_m": [round(dx, 1), round(dy, 1)],
                          "sources": {
                              "urban": "DLR WSF Evolution v1 (1985-2015) & WSF 2019 v1, CC BY 4.0",
                              "roads/rail/water/protected": "OpenStreetMap contributors, ODbL "
                                                            "(present-day, filtered to 2004)",
                              "elevation": "AWS Terrain Tiles (SRTM-derived)"},
                          "caveats": [
                              "OSM gives today's network; known post-2004 highways are removed "
                              "and re-added as scenario events, but newer secondary roads remain.",
                              "Observed 2019 comes from WSF2019 (Sentinel/Landsat, 10 m) while "
                              "earlier years come from WSF-Evolution (Landsat, 30 m); 2019 uses a "
                              "ratio-harmonised threshold and 2019 ⊇ 2015 — lowest confidence.",
                              "Lakes and protected land are present-day extents."]})
    grid["elevation"] = elev
    grid["water"] = water
    grid["excluded"] = excluded
    grid["roads"] = roads
    grid["rail"] = rail
    u = urban_by[START_YEAR] & ~water
    grid["urban"] = u
    ua = np.full(geo.shape, np.nan, np.float32)
    for y in range(START_YEAR, 1984, -1):  # first year each cell crossed the threshold
        ua[urban_by[y] & u] = y
    grid["urbanized_at"] = ua
    for y in EVO_OBSERVED + [2019]:
        grid.observed[float(y)] = {"urban": (urban_by[y] & ~water).astype(np.uint8)}
    for a in CITY_ANCHORS:
        a = dict(a)
        x, y = geo.xy(a.pop("lon"), a.pop("lat"))
        grid.add_anchor({**a, "position": [round(float(x), 2), round(float(y), 2)]})

    OUT.mkdir(parents=True, exist_ok=True)
    city = save_city(grid, OUT / "bangalore_2004.city.json")
    print("wrote", city)

    # --- scenarios -------------------------------------------------------------
    kx, ky = geo.xy(KIA["lon"], KIA["lat"])
    kia = {k: v for k, v in KIA.items() if k not in ("lat", "lon")}
    kia["position"] = [round(float(kx), 2), round(float(ky), 2)]
    kia_poly = np.argwhere(kia_area)
    hal_x, hal_y = geo.xy(77.6706, 12.9515)
    events = [
        {"type": "burn", "year": 2005, "layer": "excluded", "value": 1,
         "label": "KIA land acquired (construction starts Jul 2005)",
         "geometry": "rect", "bounds": [int(kia_poly[:, 1].min()), int(kia_poly[:, 0].min()),
                                         int(kia_poly[:, 1].max()), int(kia_poly[:, 0].max())]}
        if kia_poly.size else None,
        {"type": "add_road", "year": 2007, "value": 3, "label": "NICE Road (phased, ~2007)",
         "lines": later["nice"]},
        {"type": "remove_anchor", "year": 2008, "anchor": "HAL Airport",
         "label": "HAL civil enclave closes (24 May 2008)"},
        {"type": "add_anchor", "year": 2008, "label": "HAL aerospace works remain",
         "anchor": {"name": "HAL works", "kind": "industrial", "mass": 0.4,
                    "position": [round(float(hal_x), 2), round(float(hal_y), 2)]}},
        {"type": "add_road", "year": 2014, "value": 3, "label": "Hebbal–KIA expressway (Jan 2014)",
         "lines": later["ballari"]},
        {"type": "add_road", "year": 2023, "value": 3, "label": "Bengaluru–Mysuru Expressway (2023)",
         "lines": later["mysuru"]},
    ]
    events = [e for e in events if e]
    sdir = OUT / "scenarios"
    sdir.mkdir(exist_ok=True)
    sleuth = {"type": "sleuth", "diffusion": 5, "breed": 15, "spread": 12,
              "slope_resistance": 40, "road_gravity": 35, "critical_slope": 15,
              "modifiers": [{"type": "anchor", "gain": 1.5}], "anchor_seeding": 10,
              "self_modification": {"enabled": True, "critical_high": 4.0, "critical_low": 2.0,
                                    "boom": 1.02, "bust": 0.95}}
    base = {"city": "../bangalore_2004.city.json", "seed": 2004, "until": 2026,
            "record": {"every": 1}, "monte_carlo": {"runs": 10},
            "output": {"root": "../../../runs", "animation": "growth.mp4", "fps": 6}}
    scenarios = {
        "real_sleuth": {"name": "real-sleuth", "title": "Bangalore — SLEUTH with KIA",
                        "model": sleuth, "anchors": [kia], "events": events,
                        "output": {**base["output"], "show_field": True}},
        "no_kia_sleuth": {"name": "no-kia-sleuth", "title": "Bangalore — SLEUTH, no KIA (counterfactual)",
                          "model": sleuth, "anchors": [],
                          "events": [e for e in events if "KIA" not in e.get("label", "")]},
        "real_potential": {"name": "real-potential", "title": "Bangalore — constrained CA with KIA",
                           "model": {"type": "potential", "combine": "product",
                                     "factors": [{"type": "neighborhood", "radius": 2, "floor": 0.03},
                                                 {"type": "slope", "critical": 15, "resistance": 40},
                                                 {"type": "road_access", "scale": 4, "floor": 0.15},
                                                 {"type": "anchor", "gain": 2.0},
                                                 {"type": "excluded"},
                                                 {"type": "noise", "alpha": 0.6}],
                                     "demand": {"type": "series", "years": [2004, 2015, 2026],
                                                "cells": [1050, 1050, 1050]},
                                     "allocator": {"type": "roulette", "gamma": 3.0}},
                           "anchors": [kia], "events": events},
    }
    for fname, sc in scenarios.items():
        doc = {**base, **sc}
        (sdir / f"{fname}.json").write_text(json.dumps(doc, indent=1))
        print("wrote", sdir / f"{fname}.json")
    print("protected (large):", sorted(set(big_protected))[:15])


if __name__ == "__main__":
    main()
