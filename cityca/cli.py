"""Command line interface: ``python -m cityca <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_synth(a):
    from .synth import make_riverton
    for p in make_riverton(a.out, size=a.size, seed=a.seed):
        print("wrote", p)


def _cmd_info(a):
    from .grid import load_city
    print(load_city(a.city).summary())


def _cmd_show(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import viz
    from .anchors import AnchorField
    from .grid import load_city

    from .naming import run_dir

    g = load_city(a.city)
    if g.anchors:
        g["anchor_potential"] = AnchorField().compute(g)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = run_dir(g.name, "show", a.root)
    fig = viz.render_layers(g, a.layers.split(",") if a.layers else None)
    fig.savefig(out / f"{g.name}_layers.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    viz.style.apply()
    fig, ax = plt.subplots(figsize=(8, 8))
    viz.render_map(ax, g)
    fig.savefig(out / f"{g.name}_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{g.name}_layers.png", "and", out / f"{g.name}_map.png")


def _cmd_run(a):
    from .scenario import run_scenario
    res = run_scenario(a.scenario, out_dir=a.out, animation=not a.no_anim, runs=a.runs,
                       root=a.root)
    if "validation" in res["summary"]:
        print("  validation:", json.dumps(res["summary"]["validation"], indent=None, default=float))


def _cmd_compare(a):
    from .scenario import compare_scenarios
    res = compare_scenarios(a.scenarios, a.out, runs=a.runs, root=a.root)
    print(json.dumps(res["summary"], indent=2))
    print("outputs in", res["out"])


def _cmd_calibrate(a):
    from .engine.calibrate import calibrate
    from .scenario import build_simulation, load_scenario, load_scenario_city, output_dir

    spec = load_scenario(a.scenario)
    grid0 = load_scenario_city(spec)
    space = {}
    for item in a.space:
        k, rng = item.split("=")
        lo, hi = rng.split(":")
        space[k] = (float(lo), float(hi))
    trials = calibrate(lambda m, s: build_simulation(spec, grid0.copy(), seed=s, model_spec=m),
                       grid0, spec["model"], space, samples=a.samples, runs=a.runs, seed=a.seed,
                       until=a.until)
    for t in trials[:5]:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}))
    out = output_dir(spec, grid0.name, "calibrate", a.out, a.root)
    (out / "trials.json").write_text(json.dumps(trials, indent=1))
    (out / "calibration.json").write_text(json.dumps(
        {"scenario": a.scenario, "space": space, "samples": a.samples, "runs": a.runs,
         "until": a.until,
         "seed": a.seed, "best": trials[0]}, indent=1))
    print("outputs in", out)


def _cmd_list(a):
    from .anchors.anchor import ANCHOR_KINDS
    from .engine import ALLOCATORS, DEMANDS, EVENTS, FACTORS, MODELS
    from .grid import DERIVED, GENERATORS, LAYERS, SOURCES
    from .kernels import DECAY
    from .metrics import METRICS
    for reg in (MODELS, FACTORS, DEMANDS, ALLOCATORS, EVENTS, DECAY, LAYERS, DERIVED,
                SOURCES, GENERATORS, METRICS):
        print(f"{reg.kind:>16}: {', '.join(reg.names())}")
    print(f"{'anchor kind':>16}: {', '.join(sorted(ANCHOR_KINDS))}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cityca", description="Cellular-automata urban growth")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic Riverton city files")
    s.add_argument("--out", default="examples")
    s.add_argument("--size", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=_cmd_synth)

    s = sub.add_parser("info", help="summarise a city file")
    s.add_argument("city")
    s.set_defaults(fn=_cmd_info)

    s = sub.add_parser("show", help="render a city's layers and map")
    s.add_argument("city")
    s.add_argument("--out", help="explicit output directory (skips auto-naming)")
    s.add_argument("--root", default="runs", help="parent of auto-named run directories")
    s.add_argument("--layers", help="comma-separated layer names (base or derived)")
    s.set_defaults(fn=_cmd_show)

    s = sub.add_parser("run", help="run a scenario file")
    s.add_argument("scenario")
    s.add_argument("--out", help="explicit output directory (skips auto-naming)")
    s.add_argument("--root", help="parent of auto-named run directories "
                                  "(default: scenario output.root, else runs)")
    s.add_argument("--runs", type=int, help="Monte Carlo runs (overrides scenario)")
    s.add_argument("--no-anim", action="store_true")
    s.set_defaults(fn=_cmd_run)

    s = sub.add_parser("compare", help="Monte Carlo several scenarios and map differences")
    s.add_argument("scenarios", nargs="+")
    s.add_argument("--out", help="explicit output directory (skips auto-naming)")
    s.add_argument("--root", help="parent of auto-named run directories")
    s.add_argument("--runs", type=int, default=10)
    s.set_defaults(fn=_cmd_compare)

    s = sub.add_parser("calibrate", help="random-search model parameters against observed maps")
    s.add_argument("scenario")
    s.add_argument("--space", nargs="+", required=True, help="param=low:high (dotted paths ok)")
    s.add_argument("--until", type=float, help="calibrate on observed years <= this (hold out later ones)")
    s.add_argument("--samples", type=int, default=20)
    s.add_argument("--runs", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="explicit output directory (skips auto-naming)")
    s.add_argument("--root", help="parent of auto-named run directories")
    s.set_defaults(fn=_cmd_calibrate)

    s = sub.add_parser("list", help="list all registered components")
    s.set_defaults(fn=_cmd_list)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
