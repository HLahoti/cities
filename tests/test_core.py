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
    assert a.strength(2060) == pytest.approx(a.mass)
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


@pytest.mark.parametrize("shape", ["linear", "saturating", "logistic"])
def test_ramp_shapes_monotone(shape):
    a = Anchor("a", (0, 0), mass=2.0, announce_year=2000, open_year=2005, anticipation=0.1,
               ramp_years=10, ramp_shape=shape)
    s = [a.strength(y) for y in np.arange(1999, 2030, 0.5)]
    assert all(b >= a_ - 1e-12 for a_, b in zip(s, s[1:]))
    assert a.strength(2005) == pytest.approx(0.2)  # continuous at opening
    assert a.strength(2015) > 1.8 if shape != "saturating" else a.strength(2015) > 1.85
    if shape == "logistic":  # slow start
        assert a.strength(2007) < a.strength(2005) + 0.25 * (a.mass - a.strength(2005))


def test_urban_field_suppresses_isolated_villages():
    from cityca.engine import FACTORS
    g = small_city()
    village = np.zeros(g.shape, bool)
    village[52:54, 52:54] = True
    g.urbanize(village)
    f = FACTORS.build({"type": "urban_field", "radius": 6, "ref": 0.3})(g, None)
    assert f[30, 35] > 3 * f[52, 51]  # city edge ≫ isolated village edge


def test_cluster_size_map():
    g = small_city()
    sizes = g.derive("cluster_size_map")
    assert sizes[30, 30] == (g["urban"] > 0).sum()
    assert sizes[0, 59] == 0


def test_station_anchor_does_not_seed():
    a = Anchor.from_dict({"name": "s", "kind": "station", "position": [0, 0]})
    assert a.seeds < 0.1 and a.metric == "euclidean"


def test_densify_process():
    from cityca.engine import build_processes
    g = small_city()
    procs = build_processes([{"type": "densify", "rate": 0.3, "new_density": 0.4,
                              "capacity_factors": []}])
    sim = Simulation(g, MODELS.build({"type": "potential", "allocator": "top_n",
                                      "demand": {"type": "constant", "cells": 20}}),
                     seed=0, processes=procs)
    d_start = g["density"].copy()  # initialised from the urban mask
    assert d_start[30, 30] > 0 and d_start[0, 59] == 0
    sim.run(steps=5)
    u0 = d_start > 0
    assert (g["density"][u0] >= d_start[u0] - 1e-6).all()  # never thins
    assert g["density"][u0].mean() > d_start[u0].mean()  # thickens
    new = (g["urban"] > 0) & ~u0
    assert (g["density"][new] >= 0.4 - 1e-6).all()  # new cells start at new_density
    assert g["density"].max() <= 1.0
    assert "density" in sim.history.frames[-1].layers


def test_density_errors_skill():
    from cityca.metrics import density_errors
    obs = np.array([[0.5, 0.9], [0.8, 1.0]])
    ref = np.array([[0.3, 0.7], [0.6, 0.8]])
    mask = np.ones_like(obs, bool)
    assert density_errors(obs, obs, mask, reference=ref)["skill"] == pytest.approx(1.0)
    assert density_errors(obs, ref, mask, reference=ref)["skill"] == pytest.approx(0.0)


def test_density_heatmap_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cityca.viz import render_map
    from cityca.viz.render import density_rgba
    rgba = density_rgba(np.array([[0.0, 0.5, 1.0]]), smooth=0)
    assert rgba[0, 0, 3] == 0 and rgba[0, 2, 3] == 1  # bare land transparent, full opaque
    g = small_city()
    g["density"] = g["urban"] * 0.8
    fig, ax = plt.subplots()
    render_map(ax, g)  # auto -> density mode with colour bar
    assert len(fig.axes) == 2
    plt.close(fig)
