"""Scenarios: a JSON file that wires a city, a model and interventions together.

    {
      "name": "airport",
      "city": "../examples/riverton.city.json",
      "seed": 1, "until": 2030, "dt": 1,
      "model": {"type": "sleuth", "diffusion": 4, ..., "modifiers": [{"type": "anchor", "gain": 3}]},
      "field": {"combine": "sum"},
      "anchors": [{"name": "Airport", "kind": "airport", "position": [160, 150],
                   "announce_year": 1995, "open_year": 2002}],
      "events": [{"type": "add_road", "year": 2000, "coords": [[...]], "value": 3}],
      "record": {"every": 1},
      "monte_carlo": {"runs": 10},
      "output": {"root": "../runs", "animation": "growth.mp4", "fps": 8}
    }

Relative paths are resolved against the scenario file's directory. Each run
writes to a fresh ``<root>/<city_name>_<datetime>_<run_type>`` directory
(see ``cityca.naming``); ``root`` defaults to ``runs`` in the working directory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .anchors.field import AnchorField
from .engine import MODELS, Simulation, build_events, run_ensemble
from .engine.calibrate import score_history
from .grid.io import load_city, save_city
from .naming import run_dir


def load_scenario(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        spec = copy.deepcopy(path_or_dict)
        spec.setdefault("_base", ".")
        return spec
    p = Path(path_or_dict)
    spec = json.loads(p.read_text())
    spec["_base"] = str(p.parent)
    spec.setdefault("name", p.stem)
    return spec


def _resolve(spec: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(spec.get("_base", ".")) / p


def load_scenario_city(spec: dict):
    grid = load_city(_resolve(spec, spec["city"]))
    for a in spec.get("anchors", []):
        grid.add_anchor(dict(a))
    if "start_year" in spec:
        grid.set_year(spec["start_year"])
    return grid


def build_simulation(spec: dict, grid, seed: int | None = None, model_spec: dict | None = None) -> Simulation:
    return Simulation(
        grid,
        MODELS.build(copy.deepcopy(model_spec or spec["model"])),
        field=AnchorField(**spec.get("field", {})),
        events=build_events(copy.deepcopy(spec.get("events", []))),
        dt=spec.get("dt", 1.0),
        seed=spec.get("seed", 0) if seed is None else seed,
        record=copy.deepcopy(spec.get("record")),
    )


def output_dir(spec: dict, city_name: str, run_type: str, out_dir=None, root=None) -> Path:
    """Explicit ``out_dir`` wins; otherwise a fresh named directory under the root."""
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out
    if root is None:
        cfg_root = spec.get("output", {}).get("root")
        root = _resolve(spec, cfg_root) if cfg_root else Path("runs")
    return run_dir(city_name, run_type, root)


def _run_kwargs(spec):
    if "until" in spec:
        return {"until": spec["until"]}
    return {"steps": spec.get("steps", 20)}


def run_scenario(path_or_dict, out_dir: str | Path | None = None, animation: bool = True,
                 runs: int | None = None, figures: bool = True, progress: bool = True,
                 root: str | Path | None = None) -> dict:
    """Run one simulation (plus optional Monte Carlo) and write all outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import viz

    spec = load_scenario(path_or_dict)
    out_cfg = spec.get("output", {})
    grid0 = load_scenario_city(spec)
    out = output_dir(spec, grid0.name, spec.get("name", "scenario"), out_dir, root)
    sim = build_simulation(spec, grid0.copy())
    if progress:
        print(f"[{spec.get('name')}] {type(sim.model).__name__} on {grid0.name}, "
              f"from {grid0.year:g}")
    hist = sim.run(**_run_kwargs(spec), progress=progress)
    hist.save(out / "history")
    save_city(sim.grid, out / "final.city.json")

    summary = {"name": spec.get("name"), "start_year": grid0.year, "end_year": sim.grid.year,
               "urban_start": int(grid0["urban"].sum()), "urban_end": int(sim.grid["urban"].sum()),
               "events": hist.events, "final_params": sim.model.params()}

    # Validation against observed snapshots that fall inside the run.
    obs_years = [y for y in grid0.observed if grid0.year < y <= sim.grid.year]
    if obs_years:
        summary["validation"] = score_history(hist, grid0)

    if figures:
        def save(fig, name):
            fig.savefig(out / name, dpi=130, bbox_inches="tight")
            plt.close(fig)

        viz.style.apply()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        viz.render_map(axes[0], grid0, title=f"Start — {grid0.year:g}")
        ua = viz.animate.first_urban_year(grid0, hist)
        viz.render_map(axes[1], sim.grid, urbanized_at=ua, start_year=grid0.year,
                       title=f"Simulated — {sim.grid.year:g}")
        fig.tight_layout()
        save(fig, "map_start_end.png")

        if sim.grid.anchors:
            fig, ax = plt.subplots(figsize=(7.5, 7))
            viz.render_field(ax, sim.grid)
            save(fig, "anchor_field.png")
            if any(a.open_year is not None for a in sim.grid.anchors):
                save(viz.plot_anchor_strength(sim.grid.anchors, hist.years), "anchor_strength.png")

        observed = None
        if obs_years:
            from .metrics.morphology import METRICS
            ys = sorted(grid0.observed)
            observed = {k: (ys, [METRICS.get(k)(grid0.observed[y]["urban"]) for y in ys])
                        for k in ("urban_area", "clusters", "edges")}
            y_last = max(obs_years)
            fig, ax = plt.subplots(figsize=(7.5, 7))
            viz.render_comparison(ax, grid0, grid0["urban"], grid0.observed[y_last]["urban"],
                                  hist.frame_at(y_last).layers["urban"],
                                  title=f"Simulated vs observed change, {grid0.year:g}–{y_last:g}")
            save(fig, "validation_map.png")
        keys = [k for k in ("urban_area", "new", "clusters", "fractal_dimension", "edges",
                            "radius_of_gyration") if np.isfinite(hist.series(k)).any()]
        save(viz.plot_series(hist, keys=keys, observed=observed), "metrics.png")
        if "edge" in hist.records[-1]:
            save(viz.plot_growth_types(hist), "growth_types.png")
        cbd = next((a for a in grid0.anchors if a.kind == "cbd"), None)
        if cbd is not None:
            save(viz.plot_radial_profiles({f"{grid0.year:g}": grid0, f"{sim.grid.year:g}": sim.grid},
                                          cbd.position), "radial_profile.png")

    anim_name = out_cfg.get("animation", "growth.mp4")
    if animation and anim_name:
        if progress:
            print("  rendering animation…")
        viz.animate_history(grid0, hist, out / anim_name, fps=out_cfg.get("fps", 8),
                            show_field=out_cfg.get("show_field", False), grid_final=sim.grid,
                            title=spec.get("title"))

    n_runs = runs if runs is not None else spec.get("monte_carlo", {}).get("runs", 0)
    if n_runs and n_runs > 1:
        ens = run_scenario_ensemble(spec, n_runs, progress=progress)
        ens.save(out / "ensemble")
        summary["monte_carlo"] = {"runs": n_runs,
                                  "expected_urban_end": float(ens.final.sum()),
                                  "cells_p_over_50": int((ens.final > 0.5).sum())}
        if figures:
            fig, ax = plt.subplots(figsize=(7.5, 7))
            viz.render_probability(ax, grid0, ens.final, year=ens.years[-1],
                                   title=f"P(urban) by {ens.years[-1]:g} — {n_runs} runs")
            save(fig, "probability.png")
            save(viz.plot_ensemble(ens, labels=spec.get("name")), "ensemble.png")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if progress:
        print(f"  urban {summary['urban_start']} → {summary['urban_end']} cells; outputs in {out}")
    return {"summary": summary, "history": hist, "grid": sim.grid, "grid0": grid0, "out": out}


def run_scenario_ensemble(spec, runs: int, progress: bool = False):
    spec = load_scenario(spec)
    grid0 = load_scenario_city(spec)
    base = spec.get("seed", 0) * 10007
    rec = {"every": spec.get("record", {}).get("every", 1), "layers": ["urban"],
           "metrics": ["urban_area", "clusters"]}
    spec_mc = {**spec, "record": rec}
    return run_ensemble(lambda s: build_simulation(spec_mc, grid0.copy(), seed=s),
                        runs=runs, base_seed=base, progress=progress, **_run_kwargs(spec))


def compare_scenarios(paths: list, out_dir: str | Path | None = None, runs: int = 10,
                      progress: bool = True, root: str | Path | None = None) -> dict:
    """Monte Carlo several scenarios on the same city and map their differences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from . import viz

    specs = [load_scenario(p) for p in paths]
    names = [s.get("name", f"s{i}") for i, s in enumerate(specs)]
    grids = {n: load_scenario_city(s) for n, s in zip(names, specs)}
    out = output_dir(specs[0], grids[names[0]].name, "compare", out_dir, root)
    ens = {}
    for name, s in zip(names, specs):
        if progress:
            print(f"[{name}] {runs} runs")
        ens[name] = run_scenario_ensemble(s, runs, progress=progress)

    viz.style.apply()
    k = len(names)
    fig, axes = plt.subplots(1, k + (1 if k == 2 else 0), figsize=(7 * (k + (k == 2)), 7))
    for ax, n in zip(np.atleast_1d(axes), names):
        viz.render_probability(ax, grids[n], ens[n].final, year=ens[n].years[-1],
                               title=f"{n}: P(urban) by {ens[n].years[-1]:g}")
    if k == 2:
        a, b = names
        diff = ens[b].final - ens[a].final
        ax = axes[-1]
        ax.imshow(viz.render.base_rgb(grids[b]), interpolation="nearest")
        vmax = max(float(np.abs(diff).max()), 1e-6)
        im = ax.imshow(np.ma.masked_where(np.abs(diff) < 0.02, diff), cmap=viz.style.DIVERGING,
                       norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
        viz.render.draw_anchors(ax, grids[b], ens[b].years[-1])
        viz.style.map_axes(ax)
        ax.set_title(f"Change in P(urban): {b} − {a}", loc="left")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("ΔP", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_probability.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    first, *rest = names
    others = {n: ens[n] for n in rest}
    fig = viz.plot_ensemble(ens[first], keys=("urban_area", "clusters"), labels=first, others=others)
    fig.savefig(out / "compare_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    for n, e in ens.items():
        e.save(out / n)
    summary = {n: {"expected_urban_end": float(e.final.sum())} for n, e in ens.items()}
    summary["scenarios"] = [str(p) if not isinstance(p, dict) else p.get("name") for p in paths]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"ensembles": ens, "summary": summary, "out": out}
