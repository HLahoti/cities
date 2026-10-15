"""Basic calibration: random search of model parameters against observed maps.

A deliberately simple baseline in SLEUTH's spirit: simulate from the city's
start year through every observed year, score the match, keep the best.
Swap in a smarter optimiser later; the scoring function is the contract.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from ..metrics import compare as cmp
from ..metrics.morphology import METRICS


OBJECTIVES = ("fom_area", "density")


def score_history(history, grid0, max_year: float | None = None,
                  objective: str = "fom_area") -> dict:
    """Compare a run with ``grid0.observed`` snapshots.

    Only observed years inside the run (and ≤ ``max_year``) are scored.
    ``objective`` picks what ``score`` means:

    * ``fom_area`` — figure of merit on change × (1 − area error);
    * ``density``  — density skill vs persistence, on cells urban in both the
      observed and simulated maps (densification, separate from placement).
      Needs observed ``density`` snapshots.
    """
    if not grid0.observed:
        raise ValueError("city has no observed snapshots to calibrate against")
    initial = grid0["urban"]
    out = {"lee_sallee": [], "fom": [], "area_err": []}
    dens = {"density_mae": [], "density_skill": [], "density_bias": []}
    d0 = grid0["density"] if "density" in grid0.layers else None
    sim_m = {k: [] for k in ("urban_area", "edges", "clusters")}
    obs_m = {k: [] for k in sim_m}
    last = history.frames[-1].year if max_year is None else min(max_year, history.frames[-1].year)
    for year, obs in sorted(grid0.observed.items()):
        if "urban" not in obs or year <= grid0.year or year > last:
            continue
        frame = history.frame_at(year)
        sim = frame.layers["urban"]
        o = obs["urban"]
        out["lee_sallee"].append(cmp.lee_sallee(o, sim))
        out["fom"].append(cmp.figure_of_merit(initial, o, sim))
        out["area_err"].append(abs(sim.sum() - (o > 0).sum()) / max((o > 0).sum(), 1))
        for k in sim_m:
            sim_m[k].append(METRICS.get(k)(sim))
            obs_m[k].append(METRICS.get(k)(o))
        if "density" in obs and "density" in frame.layers and d0 is not None:
            both = (o > 0) & (sim > 0)
            e = cmp.density_errors(obs["density"], frame.layers["density"], both, reference=d0)
            dens["density_mae"].append(e["mae"])
            dens["density_skill"].append(e["skill"])
            dens["density_bias"].append(e["bias"])
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res.update({k: float(np.mean(v)) for k, v in dens.items() if v})
    for k in sim_m:
        res[f"r2_{k}"] = cmp.r_squared(sim_m[k], obs_m[k])
    # Composite: *change* placement (figure of merit) × area accuracy. FoM ignores
    # persistent cells, which otherwise dominate agreement for an established city.
    if objective == "fom_area":
        res["score"] = res["fom"] * max(0.0, 1.0 - res["area_err"])
    elif objective == "density":
        if "density_skill" not in res:
            raise ValueError("objective 'density' needs observed density snapshots")
        res["score"] = res["density_skill"]
    else:
        raise ValueError(f"Unknown objective '{objective}'; use one of {OBJECTIVES}")
    return res


def set_path(spec: dict, path: str, value) -> None:
    """Set ``spec["a"]["b"][0]["c"]`` via ``"a.b.0.c"``."""
    parts = path.split(".")
    obj = spec
    for p in parts[:-1]:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


def calibrate(make_sim: Callable[[dict, int], "Simulation"], grid0, base_spec: dict,
              space: dict[str, tuple[float, float]], samples: int = 30, runs: int = 2,
              seed: int = 0, integer: bool = False, progress: bool = True,
              until: float | None = None, objective: str = "fom_area") -> list[dict]:
    """Random search. ``make_sim(spec, seed)`` builds a fresh simulation from a
    full scenario spec.

    ``space`` maps dotted paths inside ``base_spec`` to (low, high) ranges, e.g.
    ``model.spread`` or ``anchors.0.mass``; a bare name whose first part is not
    a top-level key is taken as a model parameter (``spread`` → ``model.spread``).
    ``until`` limits calibration to observed years ≤ until (hold the rest out).
    Returns trials sorted best-first: ``{"params": {...}, "score": ..., ...}``.
    """
    rng = np.random.default_rng(seed)
    until = max(grid0.observed) if until is None else until
    space = {(k if k.split(".")[0] in base_spec else f"model.{k}"): v for k, v in space.items()}
    trials = []
    for i in range(samples):
        params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}
        if integer:
            params = {k: round(v) for k, v in params.items()}
        spec = copy.deepcopy(base_spec)
        for k, v in params.items():
            set_path(spec, k, v)
        scores = []
        for r in range(runs):
            sim = make_sim(spec, seed * 1000 + i * runs + r)
            sim.record.every = 1
            sim.record.layers = ["urban", "density"]
            sim.record.metrics = []
            scores.append(score_history(sim.run(until=until), grid0, max_year=until,
                                        objective=objective))
        agg = {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]
               if isinstance(scores[0][k], (int, float))}
        trials.append({"params": params, **agg})
        if progress:
            best = max(t["score"] for t in trials)
            print(f"\r  trial {i + 1}/{samples}  score {agg['score']:.3f}  best {best:.3f}",
                  end="", flush=True)
    if progress:
        print()
    return sorted(trials, key=lambda t: -t["score"])
