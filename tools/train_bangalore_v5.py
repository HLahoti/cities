"""Train the v5 models on all GHSL intervals (2000-2020) and write the forecast scenario.

Best configuration from rolling-origin validation (validation_v5_final_*.json):
learned growth (logistic, momentum features, density inputs frozen at the
forecast start) + learned densification (gbm, expected mode, urban cells) +
trend demand.

    python3 -P -E tools/train_bangalore_v5.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cityca.learn import train_density_model, train_from_scenario  # noqa: E402
from cityca.learn.train import build_states  # noqa: E402
from cityca.scenario import load_scenario, load_scenario_city  # noqa: E402

BDIR = ROOT / "examples" / "bangalore"


def main():
    spec = load_scenario(BDIR / "scenarios" / "ghsl_base.json")
    base = load_scenario_city(spec)
    states = build_states(spec, base)
    out = BDIR / "models" / "ghsl_2020"
    growth, grep = train_from_scenario(spec, base, 2020, learner="logistic", params={"C": 1.0},
                                       states=states, holdout=False)
    growth.save(out / "growth_logistic.joblib")
    dens, drep = train_density_model(spec, base, 2020, states=states, classifier="gbm",
                                     regressor="gbm", holdout=False)
    dens.save(out / "density_gbm.joblib")
    (out / "report.json").write_text(json.dumps({"growth": grep, "density": drep}, indent=1, default=float))
    a0, a1 = float((base["urban"] > 0).sum()), float((base.observed[2020.0]["urban"] > 0).sum())
    trend = (a1 - a0) / (2020 - base.year)
    print(f"growth train AUC {grep['train']['roc_auc']:.3f}; density period {drep['period']:g}y, "
          f"train AUC {drep['train_auc']:.3f}; trend demand {trend:.0f} cells/yr")
    sc = json.loads((BDIR / "scenarios" / "ghsl_base.json").read_text())
    sc.update({
        "name": "v5-forecast-2030", "title": "Bangalore v5 — forecast 2020→2030 (GHSL)",
        "start_from_observed": 2020, "until": 2030,
        "model": {"type": "potential",
                  "factors": [{"type": "learned", "model_file": "../models/ghsl_2020/growth_logistic.joblib",
                               "freeze": ["density", "density_change"]}],
                  "demand": {"type": "constant", "cells": round(trend, 1)},
                  "allocator": {"type": "roulette", "gamma": 2.0}},
        "processes": [{"type": "learned_densify", "model_file": "../models/ghsl_2020/density_gbm.joblib",
                       "mode": "expected", "nonurban": False}],
        "monte_carlo": {"runs": 20},
        "notes": "Best configuration from rolling-origin validation v5 (see README). Trained on GHSL "
                 "2000-2020; demand = mean 2000-2020 growth.",
    })
    (BDIR / "scenarios" / "v5_forecast_2030.json").write_text(json.dumps(sc, indent=1))
    print("wrote", BDIR / "scenarios" / "v5_forecast_2030.json")


if __name__ == "__main__":
    main()
