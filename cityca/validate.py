"""Rolling-origin (time-series) validation.

For each origin year o, every method is fitted using only information up to
o, then forecasts from the *observed* state at o to every later observed year.
Scores are reported against forecast horizon, so you see how accuracy decays
the further ahead a method predicts — and methods are compared on equal terms.

Config (JSON)::

    {
      "scenario": "scenarios/calibrated_v3_density.json",   # city, anchors, events, processes
      "origins": [2006, 2008, 2010, 2012, 2015],
      "runs": 10,
      "methods": {
        "proximity":  {"type": "potential", "factors": [...], "allocator": {...}},
        "learned-gbm": {"type": "learned", "learner": "gbm", "params": {...},
                        "power": 1.0, "allocator": {"type": "roulette", "gamma": 2}},
        "sleuth-cem": {"type": "calibrated", "search": "cem", "space": {...},
                       "iterations": 5, "population": 24, "runs": 3, "elites": 5}
      }
    }

Method types:

* ``potential`` — the scenario city with a ``potential`` model using the given
  factors; demand = mean observed growth (cells/year) from start to origin.
* ``learned`` — trains a LearnedModel on observed transitions ending ≤ origin,
  then forecasts with it as the transition surface (same trend demand).
* ``calibrated`` — calibrates the scenario's own model on observed years ≤
  origin (simulating from the start year), then forecasts from the origin
  state with the top ``elites`` parameter sets.
* ``fixed`` — the scenario's model as-is (beware: may use information after o).

Per-method options (``potential`` / ``learned``):

* ``"demand"`` — how much grows each year, fitted on observed areas ≤ origin:
  ``{"type": "trend"}`` (default, mean growth), ``{"type": "fit", "law":
  "auto" | "linear" | "exponential" | "logistic" | "allometric"}``, or
  ``{"type": "population"}`` (A = c·P^β along the city's population series).
* ``"processes"`` — replaces the scenario's processes. ``{"type":
  "learned_densify", "train": {...}}`` trains a DensityModel on density
  intervals ending ≤ origin; ``[]`` means no densification.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .engine.calibrate import ParallelEvaluator, calibrate, elites
from .learn.dataset import observed_state
from .learn.train import build_states, train_from_scenario
from .metrics.compare import density_errors, figure_of_merit
from .scenario import _resolve, load_scenario, load_scenario_city, run_scenario_ensemble


def _trend_demand(base, origin) -> float:
    a0 = float((base["urban"] > 0).sum())
    a1 = float((base.observed[float(origin)]["urban"] > 0).sum())
    return max(a1 - a0, 0.0) / max(origin - base.year, 1e-9)


def _area_history(base, origin):
    ys = [float(base.year)] + sorted(y for y in base.observed if y <= origin and "urban" in base.observed[y])
    areas = [float((base["urban"] > 0).sum())] + [float((base.observed[y]["urban"] > 0).sum()) for y in ys[1:]]
    return ys, areas


def make_demand(cfg, base, origin, last_year) -> tuple[dict, dict]:
    """Demand spec (and fit info) using only observations ≤ origin."""
    from .engine.demand_fit import law_to_demand, population_demand, select_law, fit_law
    cfg = cfg or {"type": "trend"}
    if cfg["type"] == "trend":
        rate = _trend_demand(base, origin)
        return {"type": "constant", "cells": rate}, {"demand": "trend", "rate": rate}
    ys, areas = _area_history(base, origin)
    if cfg["type"] == "fit":
        law = cfg.get("law", "auto")
        fit = select_law(ys, areas) if law == "auto" else fit_law(law, ys, areas)
        return law_to_demand(fit), {"demand": f"fit:{fit['law']}", "params": fit["params"]}
    if cfg["type"] == "population":
        ps = base.meta["population_series"]
        spec = population_demand(ps["years"], ps["population"], ys, areas, [last_year])
        return spec, {"demand": "population", **spec["_fit"]}
    raise ValueError(f"Unknown demand type '{cfg['type']}'")


def make_processes(cfg, spec, base, origin, states, out, name, log):
    """Processes for a forecast; trains learned densification when asked."""
    if cfg is None:
        return copy.deepcopy(spec.get("processes", []))
    from .learn.densify import train_density_model
    procs = []
    for p in cfg:
        p = copy.deepcopy(p)
        train = p.pop("train", None)
        if p["type"] == "learned_densify" and train is not None:
            dm, rep = train_density_model(spec, base, origin, states=states, **train)
            path = out / "models" / f"{name}_density_{origin:g}.joblib"
            dm.save(path)
            p["model_file"] = str(path.resolve())
            log({"density_model": {k: rep[k] for k in ("pairs", "period", "train_auc") if k in rep},
                 "density_holdout": rep.get("holdout_expected")})
        procs.append(p)
    return procs


def _forecast_spec(spec, model, origin, last_year):
    f = copy.deepcopy(spec)
    f["model"] = model
    f.pop("parameter_sets", None)
    f["until"] = last_year
    f["events"] = [e for e in spec.get("events", []) if e["year"] > origin]
    return f


def method_forecast_spec(name, mcfg, spec, base, origin, last_year, states, out, workers, log):
    """Fit a method on information ≤ origin; return the scenario spec to forecast with."""
    f, info = _method_forecast_spec(name, mcfg, spec, base, origin, last_year, states, out,
                                    workers, log)
    f["processes"] = make_processes(mcfg.get("processes"), spec, base, origin, states, out, name, log)
    return f, info


def _method_forecast_spec(name, mcfg, spec, base, origin, last_year, states, out, workers, log):
    kind = mcfg["type"]
    demand, dinfo = make_demand(mcfg.get("demand"), base, origin, last_year)
    log({"demand": dinfo})
    alloc = mcfg.get("allocator", {"type": "roulette", "gamma": 2.0})
    if kind == "potential":
        model = {"type": "potential", "factors": mcfg["factors"], "demand": demand,
                 "allocator": alloc, "combine": mcfg.get("combine", "product")}
        return _forecast_spec(spec, model, origin, last_year), {}
    if kind == "learned":
        lm, rep = train_from_scenario(spec, base, origin, learner=mcfg.get("learner", "gbm"),
                                      params=mcfg.get("params"), features=mcfg.get("features"),
                                      states=states, importance=False)
        path = out / "models" / f"{name}_{origin:g}.joblib"
        lm.save(path)
        factors = [{"type": "learned", "model_file": str(path.resolve()),
                    "power": mcfg.get("power", 1.0), "freeze": mcfg.get("freeze")}] + \
            mcfg.get("extra_factors", [])
        model = {"type": "potential", "factors": factors, "demand": demand, "allocator": alloc}
        log({"train": rep.get("train"), "holdout": rep.get("holdout")})
        return _forecast_spec(spec, model, origin, last_year), {"train_auc": rep["train"]["roc_auc"]}
    if kind == "calibrated":
        ev = ParallelEvaluator(_resolve(spec, spec["city"]), workers)
        try:
            trials = calibrate(ev, base, spec, mcfg["space"], runs=mcfg.get("runs", 3),
                               seed=int(origin), until=origin, objective="fom_area",
                               search=mcfg.get("search", "cem"), iterations=mcfg.get("iterations", 5),
                               population=mcfg.get("population"), samples=mcfg.get("samples", 40),
                               progress=False)
        finally:
            ev.close()
        f = _forecast_spec(spec, spec["model"], origin, last_year)
        f["parameter_sets"] = elites(trials, mcfg.get("elites", 5))
        log({"best": trials[0]["params"], "best_score": trials[0]["score"],
             "n_trials": len(trials)})
        return f, {"calib_score": trials[0]["score"]}
    if kind == "fixed":
        return _forecast_spec(spec, spec["model"], origin, last_year), {}
    raise ValueError(f"Unknown method type '{kind}'")


def score_forecast(ens, state0, base, origin) -> list[dict]:
    rows = []
    u0 = state0["urban"] > 0
    d0 = state0["density"] if "density" in state0.layers else None
    for y in sorted(t for t in base.observed if t > origin):
        obs = base.observed[y]
        i = int(np.argmin(np.abs(ens.years - y)))
        p = ens.probability[i]
        o = obs["urban"] > 0
        row = {"target": y, "horizon": y - origin,
               "fom": figure_of_merit(u0, o, p >= 0.5),
               "area_err": float((p.sum() - o.sum()) / o.sum()),
               "observed_new": int((o & ~u0).sum()), "expected_new": float(np.where(u0, 0, p).sum())}
        if "density" in obs and ens.density is not None and d0 is not None:
            both = o & (p >= 0.5)
            e = density_errors(obs["density"], ens.density[i], both, reference=d0)
            row["density_skill"] = e.get("skill")
            row["density_mae"] = e["mae"]
            # All built-up land, including partly built non-urban cells.
            land = (state0["water"] == 0) & ((obs["density"] > 0) | (ens.density[i] > 0) | (d0 > 0))
            row["density_skill_all"] = density_errors(obs["density"], ens.density[i], land,
                                                      reference=d0).get("skill")
        rows.append(row)
    return rows


def rolling_origin(config: dict, out: Path, workers: int | None = None, progress: bool = True) -> dict:
    spec = load_scenario(config["scenario_path"] if "scenario_path" in config else config["scenario"])
    base = load_scenario_city(spec)
    origins = [float(o) for o in config["origins"]]
    last_year = max(base.observed)
    runs = config.get("runs", 10)
    states = build_states(spec, base)
    results, fits = [], {}
    (out / "models").mkdir(parents=True, exist_ok=True)
    for origin in origins:
        state0 = states.get(origin) or observed_state(spec, base, origin)
        for name, mcfg in config["methods"].items():
            log_entry = {}
            fspec, info = method_forecast_spec(name, mcfg, spec, base, origin, last_year, states, out,
                                               workers, log_entry.update)
            fits[f"{name}@{origin:g}"] = log_entry
            ens = run_scenario_ensemble(fspec, runs, workers=workers, start_grid=state0, until=last_year)
            rows = score_forecast(ens, state0, base, origin)
            for r in rows:
                results.append({"method": name, "origin": origin, **info, **r})
            if progress:
                summ = "  ".join(f"{r['target']:g}:FoM {r['fom']:.3f}/area {100 * r['area_err']:+.0f}%"
                                 + (f"/dens {r['density_skill']:+.2f}" if r.get("density_skill") is not None else "")
                                 for r in rows)
                print(f"  origin {origin:g}  {name:18s} {summ}", flush=True)
    report = {"config": {k: v for k, v in config.items() if k != "_base"}, "results": results, "fits": fits}
    (out / "validation.json").write_text(json.dumps(report, indent=1, default=float))
    return report
