"""Monte Carlo ensembles: many stochastic runs → urbanisation probability maps.

As in SLEUTH, a single CA run is one possible future; averaging N runs gives
P(cell is urban at year t), which is the actual forecast product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Ensemble:
    years: np.ndarray
    probability: np.ndarray  # (T, H, W) share of runs in which each cell is urban
    density: np.ndarray | None = None  # (T, H, W) mean built-up density across runs
    records: list[list[dict]] = field(default_factory=list)  # per run, per step
    histories: list = field(default_factory=list)

    def metric(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(years, mean, std) of a recorded quantity across runs."""
        vals = np.array([[r.get(key, np.nan) for r in run] for run in self.records], float)
        yrs = np.array([r["year"] for r in self.records[0]])
        return yrs, np.nanmean(vals, 0), np.nanstd(vals, 0)

    @property
    def final(self) -> np.ndarray:
        return self.probability[-1]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"years": self.years, "probability": self.probability.astype(np.float32)}
        if self.density is not None:
            arrays["density"] = self.density.astype(np.float32)
        np.savez_compressed(d / "ensemble.npz", **arrays)
        (d / "ensemble_records.json").write_text(json.dumps(self.records, default=float))
        return d


def run_ensemble(factory: Callable[[int], "Simulation"], runs: int = 10,
                 steps: int | None = None, until: float | None = None,
                 base_seed: int = 0, keep_histories: bool = False,
                 progress: bool = False) -> Ensemble:
    """``factory(seed)`` must return a fresh Simulation on a fresh grid copy."""
    acc = dens = None
    years = None
    records, histories = [], []
    for i in range(runs):
        sim = factory(base_seed + i)
        hist = sim.run(steps=steps, until=until)
        stack = hist.stack("urban").astype(np.float32)
        acc = stack if acc is None else acc + stack
        if "density" in hist.frames[0].layers:
            ds = hist.stack("density").astype(np.float32)
            dens = ds if dens is None else dens + ds
        years = np.array([f.year for f in hist.frames])
        records.append(hist.records)
        if keep_histories:
            histories.append(hist)
        if progress:
            print(f"\r  run {i + 1}/{runs}", end="", flush=True)
    if progress:
        print()
    return Ensemble(years=years, probability=acc / runs, records=records, histories=histories,
                    density=None if dens is None else dens / runs)
