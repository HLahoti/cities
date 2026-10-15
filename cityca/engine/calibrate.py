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


def apply_params(spec: dict, params: dict) -> dict:
    """Copy of ``spec`` with dotted-path parameters set."""
    spec = copy.deepcopy(spec)
    for k, v in params.items():
        set_path(spec, k, v)
    return spec


# --- evaluators ----------------------------------------------------------------------------
class SerialEvaluator:
    """Runs jobs in-process. ``make_sim(spec, seed)`` builds a fresh simulation."""

    def __init__(self, make_sim: Callable[[dict, int], "Simulation"], grid0):
        self.make_sim, self.grid0 = make_sim, grid0

    def __call__(self, jobs, until, objective):
        out = []
        for spec, seed in jobs:
            sim = self.make_sim(spec, seed)
            sim.record.every, sim.record.layers, sim.record.metrics = 1, ["urban", "density"], []
            out.append(score_history(sim.run(until=until), self.grid0, max_year=until,
                                     objective=objective))
        return out

    def close(self):
        pass


_WORKER: dict = {}


def _init_worker(city_path: str):
    from ..grid.io import load_city
    _WORKER["base"] = load_city(city_path)


def _eval_job(spec, seed, until, objective):
    from ..scenario import build_simulation, load_scenario_city
    grid0 = load_scenario_city(spec, _WORKER["base"])
    sim = build_simulation(spec, grid0.copy(), seed=seed)
    sim.record.every, sim.record.layers, sim.record.metrics = 1, ["urban", "density"], []
    return score_history(sim.run(until=until), grid0, max_year=until, objective=objective)


class ParallelEvaluator:
    """Runs jobs across worker processes; each worker loads the city once."""

    def __init__(self, city_path, workers: int | None = None):
        import os
        from concurrent.futures import ProcessPoolExecutor
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.pool = ProcessPoolExecutor(self.workers, initializer=_init_worker,
                                        initargs=(str(city_path),))

    def __call__(self, jobs, until, objective):
        futs = [self.pool.submit(_eval_job, spec, seed, until, objective) for spec, seed in jobs]
        return [f.result() for f in futs]

    def close(self):
        self.pool.shutdown()


# --- search strategies -------------------------------------------------------------------------
def _clip(params, space):
    return {k: float(np.clip(v, *space[k])) for k, v in params.items()}


def calibrate(evaluator, grid0, base_spec: dict, space: dict[str, tuple[float, float]],
              samples: int = 30, runs: int = 2, seed: int = 0, integer: bool = False,
              progress: bool = True, until: float | None = None, objective: str = "fom_area",
              search: str = "random", iterations: int = 5, population: int | None = None,
              elite_frac: float = 0.25, smoothing: float = 0.7) -> list[dict]:
    """Search parameters to maximise ``score_history`` against observed years.

    ``evaluator`` is a ``SerialEvaluator``/``ParallelEvaluator`` (or, for
    backwards compatibility, a ``make_sim(spec, seed)`` callable). ``space``
    maps dotted paths in ``base_spec`` to (low, high); a bare name whose first
    part is not a top-level key means a model parameter (``spread`` →
    ``model.spread``). ``until`` limits scoring to observed years ≤ until.

    ``search``:
    * ``random`` — ``samples`` uniform draws;
    * ``cem`` — cross-entropy method: ``iterations`` rounds of ``population``
      draws from a Gaussian (truncated to the bounds) re-fitted each round to
      the top ``elite_frac`` of all trials so far, with ``smoothing``.

    Every trial averages ``runs`` stochastic runs. Returns trials best-first:
    ``{"params": {...}, "score": ..., "round": r, ...}``.
    """
    if not hasattr(evaluator, "close"):
        evaluator = SerialEvaluator(evaluator, grid0)
    rng = np.random.default_rng(seed)
    until = max(grid0.observed) if until is None else until
    space = {(k if k.split(".")[0] in base_spec else f"model.{k}"): tuple(v) for k, v in space.items()}
    keys = list(space)
    lo = np.array([space[k][0] for k in keys])
    hi = np.array([space[k][1] for k in keys])
    trials: list[dict] = []

    def run_round(param_list, rnd):
        jobs, meta = [], []
        for i, params in enumerate(param_list):
            if integer:
                params = {k: round(v) for k, v in params.items()}
            spec = apply_params(base_spec, params)
            for r in range(runs):
                jobs.append((spec, int(seed * 100_000 + len(trials) * 100 + i * runs + r)))
                meta.append(i)
        results = evaluator(jobs, until, objective)
        for i, params in enumerate(param_list):
            scores = [res for res, m in zip(results, meta) if m == i]
            agg = {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]
                   if isinstance(scores[0][k], (int, float))}
            agg["score_sd"] = float(np.nanstd([s["score"] for s in scores]))
            trials.append({"params": params, "round": rnd, **agg})
        if progress:
            best = max(t["score"] for t in trials)
            print(f"\r  round {rnd + 1}: {len(trials)} trials, best {best:.4f}", end="", flush=True)

    if search == "random":
        draws = rng.uniform(lo, hi, size=(samples, len(keys)))
        run_round([dict(zip(keys, map(float, d))) for d in draws], 0)
    elif search == "cem":
        pop = population or max(8, 2 * len(keys))
        mu, sd = (lo + hi) / 2, (hi - lo) / 2
        floor = 0.02 * (hi - lo)
        for rnd in range(iterations):
            if rnd == 0:
                draws = rng.uniform(lo, hi, size=(pop, len(keys)))
            else:
                draws = np.clip(rng.normal(mu, sd, size=(pop, len(keys))), lo, hi)
            run_round([dict(zip(keys, map(float, d))) for d in draws], rnd)
            ranked = sorted(trials, key=lambda t: -t["score"])
            elite = ranked[:max(2, int(elite_frac * len(ranked)))]
            E = np.array([[t["params"][k] for k in keys] for t in elite])
            mu = smoothing * E.mean(0) + (1 - smoothing) * mu
            sd = np.maximum(smoothing * E.std(0) + (1 - smoothing) * sd, floor)
    else:
        raise ValueError(f"Unknown search '{search}' (random | cem)")
    if progress:
        print()
    return sorted(trials, key=lambda t: -t["score"])


def elites(trials: list[dict], k: int = 5) -> list[dict]:
    """Top-k parameter sets — a cheap stand-in for a posterior over parameters."""
    return [t["params"] for t in trials[:k]]
