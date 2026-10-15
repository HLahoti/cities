import numpy as np
import pytest

from cityca.anchors import Anchor, AnchorField
from cityca.engine import ALLOCATORS, DEMANDS, MODELS, Simulation, StepContext, build_events
from cityca.grid import CityGrid, city_from_dict, load_city, save_city
from cityca.metrics import METRICS, lee_sallee, figure_of_merit
from cityca.scenario import run_scenario


def small_city(**extra):
    doc = {
        "name": "t", "year": 2000, "grid": {"height": 60, "width": 60},
        "layers": {"elevation": {"source": "generator", "generator": "fractal_noise",
                                 "params": {"amplitude": 40, "seed": 1}}},
        "features": [
            {"geometry": "polyline", "layer": "roads", "value": 3, "coords": [[0, 30], [59, 30]]},
            {"geometry": "circle", "layer": "urban", "center": [30, 30], "radius": 4},
            {"geometry": "rect", "layer": "water", "bounds": [0, 0, 8, 59]},
            {"geometry": "rect", "layer": "excluded", "value": 1, "bounds": [45, 0, 59, 12]},
        ],
        "anchors": [{"name": "CBD", "kind": "cbd", "position": [30, 30]}],
        **extra,
    }
    return city_from_dict(doc)


def test_load_save_roundtrip(tmp_path):
    g = small_city()
    g.observed[2010.0] = {"urban": g["urban"].copy()}
    p = save_city(g, tmp_path / "x.city.json")
    h = load_city(p)
    for name in ("urban", "elevation", "water", "roads", "excluded"):
        np.testing.assert_allclose(g[name], h[name])
    assert [a.name for a in h.anchors] == ["CBD"]
    assert 2010.0 in h.observed


def test_derived_cache_invalidates():
    g = small_city()
    d1 = g["urban_distance"].copy()
    g.urbanize(np.pad(np.ones((1, 1), bool), ((50, 9), (50, 9))))
    d2 = g["urban_distance"]
    assert d2[50, 50] == 0 and d1[50, 50] > 0


def test_anchor_strength_envelope():
    a = Anchor.from_dict({"name": "A", "kind": "airport", "position": [0, 0],
                          "announce_year": 2000, "open_year": 2010})
    assert a.strength(1999) == 0
    assert 0 < a.strength(2005) < a.strength(2010) <= a.mass
    assert a.strength(2030) == pytest.approx(a.mass)
    assert a.meta["capacity"] == 1500  # unknown preset keys land in meta


def test_travel_distance_follows_roads():
    g = small_city()
    f = AnchorField()
    a = Anchor("x", (30, 30), metric="travel")
    d_travel = f.distance(g, a)
    d_euclid = f.distance(g, Anchor("y", (30, 30)))
    assert d_travel[30, 59] < d_euclid[30, 59]  # along the highway: faster
    assert d_travel[0, 30] >= d_euclid[0, 30] * 0.95  # off-road: no real shortcut


@pytest.mark.parametrize("spec", [
    {"type": "sleuth", "self_modification": True, "anchor_seeding": 5,
     "modifiers": [{"type": "anchor", "gain": 2}]},
    {"type": "potential", "demand": {"type": "constant", "cells": 30}},
    {"type": "neighborhood", "p": 0.3},
])
def test_models_grow_only_on_developable_land(spec):
    g = small_city()
    start = g["urban"].copy()
    Simulation(g, MODELS.build(spec), seed=0).run(steps=10)
    grown = (g["urban"] > 0) & (start == 0)
    assert grown.sum() > 0
    assert not (grown & (g["water"] > 0)).any()
    assert not (grown & (g["excluded"] >= 1)).any()
    assert np.all(g["urbanized_at"][grown] > 2000)


def test_potential_model_respects_demand():
    g = small_city()
    sim = Simulation(g, MODELS.build({"type": "potential", "allocator": "top_n",
                                      "demand": {"type": "constant", "cells": 25}}), seed=0)
    h = sim.run(steps=4)
    assert [r["new"] for r in h.records[1:]] == [25] * 4


def test_events_and_anchor_opening():
    g = small_city(anchors=[{"name": "Air", "kind": "airport", "position": [40, 50],
                             "announce_year": 2001, "open_year": 2003}])
    events = build_events([{"type": "add_road", "year": 2002, "coords": [[30, 0], [30, 59]], "value": 2}])
    sim = Simulation(g, MODELS.build({"type": "neighborhood", "p": 0.0}), events=events, seed=0)
    h = sim.run(steps=5)
    kinds = [e["type"] for e in h.events]
    assert kinds == ["add_road", "anchor_open"]
    assert g["roads"][10, 30] == 2
    assert g["urban"][50, 40] == 1  # airport footprint built


def test_demand_laws_saturate():
    g = small_city()
    ctx = StepContext(1, 2000, 1.0, np.random.default_rng(0))
    for spec in ({"type": "logistic", "rate": 0.1, "capacity": 49},
                 {"type": "allometric", "a": 1.0, "capacity": 49}):
        d = DEMANDS.build(spec)
        assert d.rate(49.0, g, ctx) == pytest.approx(0.0, abs=1e-9)
        assert d.rate(20.0, g, ctx) > 0


def test_allocators():
    rng = np.random.default_rng(0)
    p = rng.random((20, 20))
    assert ALLOCATORS.build("top_n")(p, 17, rng).sum() == 17
    assert ALLOCATORS.build({"type": "roulette", "gamma": 2})(p, 17, rng).sum() == 17


def test_metrics_and_comparison():
    u = np.zeros((32, 32), np.uint8)
    u[2:6, 2:6] = 1
    u[20:24, 20:24] = 1
    assert METRICS.get("clusters")(u) == 2
    assert METRICS.get("urban_area")(u) == 32
    assert METRICS.get("edges")(u) == 24
    assert lee_sallee(u, u) == 1.0
    assert figure_of_merit(u * 0, u, u) == 1.0


def test_run_scenario_dict(tmp_path):
    city = small_city()
    save_city(city, tmp_path / "c.city.json")
    res = run_scenario({"name": "s", "city": str(tmp_path / "c.city.json"), "steps": 5,
                        "model": {"type": "sleuth"}}, out_dir=tmp_path / "out",
                       animation=False, figures=False, progress=False)
    assert res["summary"]["urban_end"] >= res["summary"]["urban_start"]
    assert (tmp_path / "out" / "history" / "history.npz").exists()


def test_run_dir_naming(tmp_path):
    from datetime import datetime

    from cityca.naming import run_dir
    when = datetime(2026, 10, 15, 14, 35, 2)
    a = run_dir("Riverton", "airport", tmp_path, when=when)
    b = run_dir("Riverton", "airport", tmp_path, when=when)  # same second: no clobbering
    assert a.name == "riverton_20261015-143502_airport"
    assert b.name == "riverton_20261015-143502_airport-2"
    assert run_dir("New York", "Compare A/B", tmp_path, when=when).name == \
        "new-york_20261015-143502_compare-a-b"


def test_run_scenario_auto_named_dir(tmp_path):
    save_city(small_city(), tmp_path / "c.city.json")
    res = run_scenario({"name": "smoke", "city": str(tmp_path / "c.city.json"), "steps": 2,
                        "model": {"type": "sleuth"}}, root=tmp_path / "runs",
                       animation=False, figures=False, progress=False)
    assert res["out"].parent == tmp_path / "runs"
    assert res["out"].name.startswith("t_") and res["out"].name.endswith("_smoke")
