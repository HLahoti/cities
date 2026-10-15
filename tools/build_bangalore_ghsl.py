"""Bangalore on GHSL R2023A: one consistent data source for 2000 → 2020.

Same grid, roads, water, protected land, terrain and anchors as
``bangalore_2004.city.json``; urban extent and density come from the JRC
Global Human Settlement Layer instead of WSF:

* GHS-BUILT-S (built-up surface, m² per ~90 m pixel) → ``density`` (0..1)
* GHS-BUILT-V (built-up volume, m³)                  → ``built_volume`` (m³ per m² of cell)
* GHS-POP 1 km (residents)                            → ``population`` and a city-total series

Epochs 2000, 2005, 2010, 2015, 2020 come from one processing chain, so there
is no sensor jump at the end. 1995 provides momentum history (lag layers).
GHS-POP 2025/2030 are projections and are stored as such.

Inputs: data/raw/ghsl/<tile dirs> and data/derived/ghs_pop (see
tools/convert_ghs_pop.py). Run: python3 -P -E tools/build_bangalore_ghsl.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityca.grid.io import load_city, save_city  # noqa: E402

RAW = ROOT / "data" / "raw" / "ghsl"
POP = ROOT / "data" / "derived" / "ghs_pop"
BDIR = ROOT / "examples" / "bangalore"
EPOCHS = [2000, 2005, 2010, 2015, 2020]
START, LAG = 2000, 5
POP_EPOCHS = [2000, 2005, 2010, 2015, 2020, 2025, 2030]


def overlap_matrix(cell_edges: np.ndarray, px_edges: np.ndarray) -> np.ndarray:
    """W[i, k] = length of overlap between cell i and pixel k (1-D, ascending edges)."""
    lo = np.maximum(cell_edges[:-1, None], px_edges[None, :-1])
    hi = np.minimum(cell_edges[1:, None], px_edges[None, 1:])
    return np.clip(hi - lo, 0, None)


def read_tile(product: str, epoch: int):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    path = next((RAW / f"GHS_BUILT_{product}_E{epoch}_GLOBE_R2023A_4326_3ss_V1_0_R8_C26").glob("*.tif"))
    im = Image.open(path)
    sx, sy, _ = im.tag_v2[33550]
    _, _, _, lon0, lat0, _ = im.tag_v2[33922]
    return np.asarray(im), lon0, lat0, sx, sy


class Resampler:
    """Exact area-weighted resampling from a lat/lon pixel grid to the model grid."""

    def __init__(self, geo: dict, lon0, lat0, sx, sy, pad: int = 3):
        h, w, cd = geo["rows"], geo["cols"], geo["cell_deg"]
        west, north = geo["west"], geo["north"]
        self.c0 = int((west - lon0) / sx) - pad
        self.c1 = int(math.ceil((west + w * cd - lon0) / sx)) + pad
        self.r0 = int((lat0 - north) / sy) - pad
        self.r1 = int(math.ceil((lat0 - (north - h * cd)) / sy)) + pad
        px_x = lon0 + np.arange(self.c0, self.c1 + 1) * sx
        px_y = -(lat0 - np.arange(self.r0, self.r1 + 1) * sy)  # ascending (negated latitude)
        cell_x = west + np.arange(w + 1) * cd
        cell_y = -(north - np.arange(h + 1) * cd)
        self.wx = overlap_matrix(cell_x, px_x)
        self.wy = overlap_matrix(cell_y, px_y)
        self.cell_area = cd * cd
        self.px_lat = lat0 - (np.arange(self.r0, self.r1) + 0.5) * sy
        self.sx, self.sy = sx, sy

    def crop(self, a):
        return a[self.r0:self.r1, self.c0:self.c1].astype(np.float64)

    def mean_density(self, per_area: np.ndarray) -> np.ndarray:
        """Average of a per-unit-area quantity over each cell."""
        return self.wy @ per_area @ self.wx.T / self.cell_area

    def px_area_m2(self) -> np.ndarray:
        dy = self.sy * 110_574.0
        dx = self.sx * 111_320.0 * np.cos(np.radians(self.px_lat))
        return (dx * dy)[:, None]


def main():
    wsf = load_city(BDIR / "bangalore_2004.city.json")
    geo = wsf.meta["georef"]
    water = wsf["water"] > 0

    _, lon0, lat0, sx, sy = read_tile("S", 2020)
    rs = Resampler(geo, lon0, lat0, sx, sy)
    area = rs.px_area_m2()
    dens, vol = {}, {}
    for e in [1995] + EPOCHS:
        s = rs.crop(read_tile("S", e)[0])
        dens[e] = np.where(water, 0.0, np.clip(rs.mean_density(np.clip(s / area, 0, 1)), 0, 1))
        if e >= 2000:
            v = rs.crop(read_tile("V", e)[0])
            vol[e] = np.where(water, 0.0, rs.mean_density(v / area))  # m³ per m² of cell

    # Population: 1 km people counts -> people per model cell (area-weighted share).
    plon0, plat0, psx, psy = np.load(POP / "pop_georef.npy")
    prs = Resampler(geo, plon0, plat0, psx, psy, pad=0)
    pop, pop_total = {}, {}
    for e in POP_EPOCHS:
        p = np.load(POP / f"pop_{e}.npy")
        pc = p[prs.r0:prs.r1, prs.c0:prs.c1] / (psx * psy)  # people per deg²
        pop[e] = prs.mean_density(pc) * prs.cell_area
        pop_total[e] = float(pop[e].sum())

    # Urban threshold: match the WSF-based city's 2015 urban extent, so "urban"
    # means the same amount of land in both versions of Bangalore.
    target = int((wsf.observed[2015.0]["urban"] > 0).sum())
    thr = float(np.quantile(dens[2015][~water], 1 - target / (~water).sum()))
    urban = {e: (dens[e] >= thr) & ~water for e in dens}
    for e in [1995] + EPOCHS:
        print(f"  {e}: urban {int(urban[e].sum()):6d} cells, mean density (urban) "
              f"{dens[e][urban[e]].mean():.3f}, built volume {vol.get(e, np.zeros(1)).sum() * 1e-6:.1f}")
    print(f"  urban threshold {thr:.3f} (matches WSF 2015 extent of {target} cells)")
    print("  population in grid:", {k: f"{v / 1e6:.2f}M" for k, v in pop_total.items()})

    g = wsf.copy()
    g.name = "Bangalore (GHSL)"
    g.set_year(START)
    g.observed = {}
    g["urban"] = urban[START]
    ua = np.full(g.shape, np.nan, np.float32)
    ua[urban[START]] = START
    ua[urban[1995] & urban[START]] = 1995
    g["urbanized_at"] = ua
    g["density"] = dens[START]
    g["urban_lag"] = urban[1995]
    g["density_lag"] = dens[1995]
    g["built_volume"] = vol[START]
    g["population"] = pop[START]
    for e in EPOCHS[1:]:
        g.observed[float(e)] = {"urban": urban[e].astype(np.uint8),
                                "density": dens[e].astype(np.float32),
                                "built_volume": vol[e].astype(np.float32),
                                "population": pop[e].astype(np.float32)}
    g.meta.update({
        "description": "Bengaluru 2000-2020 on GHSL R2023A (consistent built-up time series)",
        "lag_years": LAG,
        "urban_threshold": thr,
        "population_series": {"years": POP_EPOCHS, "population": [pop_total[e] for e in POP_EPOCHS],
                              "projected_from": 2025,
                              "source": "GHS-POP R2023A 30ss, summed over the model grid"},
        "sources": {**wsf.meta.get("sources", {}),
                    "urban/density/volume": "JRC GHS-BUILT-S / GHS-BUILT-V R2023A, 3 arcsec "
                                            "(CC BY 4.0), area-weighted to the model grid",
                    "population": "JRC GHS-POP R2023A 30 arcsec (CC BY 4.0)"},
        "caveats": [
            "OSM-derived roads are the filtered 2004 network; a few roads built 2000-2004 are "
            "present from the 2000 start.",
            "GHSL density = built-up surface share (horizontal); built_volume adds height.",
            "GHS-POP 2025/2030 are projections, not observations.",
            "urban threshold chosen to match the WSF city's 2015 urban extent."]})
    g.meta.pop("urban_threshold_2019", None)
    out = save_city(g, BDIR / "bangalore_2000_ghsl.city.json")
    print("wrote", out)

    # Population series for the WSF city too (same grid), for population-driven demand.
    wsf_doc = json.loads((BDIR / "bangalore_2004.city.json").read_text())
    wsf_doc["meta"]["population_series"] = g.meta["population_series"]
    (BDIR / "bangalore_2004.city.json").write_text(json.dumps(wsf_doc, indent=2))
    print("added population series to bangalore_2004.city.json")


if __name__ == "__main__":
    main()
