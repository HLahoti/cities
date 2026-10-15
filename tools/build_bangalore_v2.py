"""Bangalore v2 scenarios: the three model fixes, as a cumulative ablation.

* B ``v2_kia_ramp``  — KIA's pull builds up slowly (logistic, ~20 y), and
  its mass / ramp become calibration targets.
* C ``v2_villages``  — B + edge growth scaled by the ``urban_field``
  (villages spread only once the city reaches them).
* D ``v2_full``      — C + dated tech-park and metro-station anchors.

Anchor sources: examples/bangalore/sources/{metro_openings,tech_parks}.tsv
and OSM station nodes (data/raw/osm/stations.json).

    python3 -P -E tools/build_bangalore_v2.py
"""

from __future__ import annotations

import copy
import csv
import difflib
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BDIR = ROOT / "examples" / "bangalore"
SRC = BDIR / "sources"


def _rows(path: Path):
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def _year(iso: str) -> float:
    d = date.fromisoformat(iso)
    return round(d.year + (d.timetuple().tm_yday - 1) / 365.25, 2)


def _norm(name: str) -> str:
    n = name.lower()
    for a, b in (("mysuru", "mysore"), ("indira nagar", "indiranagar"), ("yeshwanthpur", "yeshwantpur"),
                 ("visveshwaraya", "visvesvaraya"), ("seetharamapalya", "seetharampalya"),
                 ("nallurhalli", "nallurahalli"), ("southend", "south end"), (" stn.", " station")):
        n = n.replace(a, b)
    return re.sub(r"[^a-z0-9]+", " ", n).strip()


def station_coords():
    els = json.loads((ROOT / "data" / "raw" / "osm" / "stations.json").read_text())["elements"]
    metro = [e for e in els if "name" in e.get("tags", {}) and (
        e["tags"].get("station") == "subway" or e["tags"].get("subway") == "yes"
        or "metro" in json.dumps(e["tags"]).lower())]
    return {_norm(e["tags"]["name"]): (e["lat"], e["lon"]) for e in metro}


def match(name: str, coords: dict):
    key = _norm(name)
    if key in coords:
        return coords[key]
    best, score = None, 0.0
    for cand in coords:
        parts = [cand] + [p.strip() for p in cand.split("  ")]
        s = max(difflib.SequenceMatcher(None, key, p).ratio() for p in parts)
        if key in cand or cand.endswith(key.split()[-1]) and key.split()[0] in cand:
            s = max(s, 0.9)
        if s > score:
            best, score = cand, s
    return coords[best] if score >= 0.8 else None


def main():
    city = json.loads((BDIR / "bangalore_2004.city.json").read_text())
    geo = city["meta"]["georef"]

    def xy(lat, lon):
        return [round((lon - geo["west"]) / geo["cell_deg"] - 0.5, 2),
                round((geo["north"] - lat) / geo["cell_deg"] - 0.5, 2)]

    def inside(p):
        return 0 <= p[0] < geo["cols"] and 0 <= p[1] < geo["rows"]

    # Tech parks: pre-2004 parks are already open (ramping), later ones open on schedule.
    tech = []
    for r in _rows(SRC / "tech_parks.tsv"):
        y = int(r["opened"])
        tech.append({"name": r["name"], "kind": "tech_park", "position": xy(float(r["lat"]), float(r["lon"])),
                     "announce_year": y - 2, "open_year": y, "meta": {"note": r["note"]}})

    # Metro stations: announced ~5 years before opening (construction period).
    coords = station_coords()
    stations, unmatched = [], []
    for r in _rows(SRC / "metro_openings.tsv"):
        c = match(r["station"], coords)
        if c is None:
            unmatched.append(r["station"])
            continue
        p = xy(*c)
        if not inside(p):
            continue
        y = _year(r["opened"])
        stations.append({"name": f"Metro: {r['station']}", "kind": "station", "position": p,
                         "announce_year": round(y - 5, 2), "open_year": y,
                         "meta": {"line": r["line"]}})
    print(f"tech parks: {len(tech)}; metro stations matched: {len(stations)}; unmatched: {unmatched}")
    (SRC / "dynamic_anchors.json").write_text(json.dumps({"tech_parks": tech, "stations": stations}, indent=1))

    base = json.loads((BDIR / "scenarios" / "calibrated_sleuth.json").read_text())
    kia = copy.deepcopy(base["anchors"][0])
    assert kia["name"] == "KIA"
    for k in ("anticipation", "ramp_years", "ramp_shape"):
        kia.pop(k, None)  # take the new slow-ramp airport preset

    def scenario(name, title, anchors, edge_modifiers=None):
        d = copy.deepcopy(base)
        d.pop("calibration", None)
        d["name"], d["title"], d["anchors"] = name, title, anchors
        if edge_modifiers:
            d["model"]["edge_modifiers"] = edge_modifiers
        return d

    field = [{"type": "urban_field", "radius": 20, "ref": 0.3}]
    out = {
        "v2_kia_ramp": scenario("v2-kia-ramp", "Bangalore v2 — slow KIA ramp", [kia]),
        "v2_villages": scenario("v2-villages", "Bangalore v2 — + village fix", [kia], field),
        "v2_full": scenario("v2-full", "Bangalore v2 — + tech parks & metro",
                            [kia] + tech + stations, field),
    }
    out["v2_full"]["field"] = {"kind_weights": {"station": 1.0, "tech_park": 1.0}}
    for fname, doc in out.items():
        (BDIR / "scenarios" / f"{fname}.json").write_text(json.dumps(doc, indent=1))
        print("wrote", BDIR / "scenarios" / f"{fname}.json")


if __name__ == "__main__":
    sys.exit(main())
