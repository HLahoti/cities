"""How close are the simulations to Bangalore's real growth?

Compares Monte Carlo ensembles of several scenario runs with the observed
(WSF) urban maps: area through time, change-placement accuracy, and growth in
distance rings around KIA (to isolate the airport effect).

    python3 -P -E tools/assess_bangalore.py runs/<calibrated-sleuth> runs/<calibrated-no-kia> ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cityca.grid import load_city  # noqa: E402
from cityca.metrics import figure_of_merit, lee_sallee  # noqa: E402
from cityca.naming import run_dir  # noqa: E402
from cityca.viz import style  # noqa: E402
from cityca.viz.render import base_rgb, draw_anchors, render_comparison  # noqa: E402

CITY = ROOT / "examples" / "bangalore" / "bangalore_2004.city.json"
KIA_LONLAT = (77.7075, 13.1976)
RINGS_KM = [0, 5, 10, 15, 20, 30]


def load_run(d: Path):
    s = json.loads((d / "summary.json").read_text())
    z = np.load(d / "ensemble" / "ensemble.npz")
    return s.get("name") or d.name, z["years"], z["probability"]


def main(dirs):
    style.apply()
    city = load_city(CITY)
    geo = city.meta["georef"]
    obs_years = sorted(y for y in city.observed if y > city.year)
    u0 = city["urban"] > 0
    land = city["water"] == 0

    kx = (KIA_LONLAT[0] - geo["west"]) / geo["cell_deg"] - 0.5
    ky = (geo["north"] - KIA_LONLAT[1]) / geo["cell_deg"] - 0.5
    rr, cc = np.indices(city.shape)
    dx, dy = city.meta["cell_size_xy_m"]
    dist_km = np.hypot((cc - kx) * dx, (rr - ky) * dy) / 1000

    runs = [load_run(Path(d)) for d in dirs]
    out = run_dir(city.name, "assessment", ROOT / "runs")
    report = {"runs": [str(d) for d in dirs], "per_year": {}, "kia_rings": {}}

    # --- per observed year ---------------------------------------------------------
    for name, years, prob in runs:
        rows = []
        for y in obs_years:
            i = int(np.argmin(np.abs(years - y)))
            p = prob[i]
            obs = city.observed[y]["urban"] > 0
            sim = p >= 0.5
            rows.append({"year": y, "observed_area": int(obs.sum()),
                         "expected_area": float(p.sum()),
                         "area_err_pct": float(100 * (p.sum() - obs.sum()) / obs.sum()),
                         "fom_p50": figure_of_merit(u0, obs, sim),
                         "lee_sallee_p50": lee_sallee(obs, sim)})
        report["per_year"][name] = rows

    # --- growth by distance from KIA, 2004 -> 2019 ------------------------------------------
    y_last = max(obs_years)
    obs_new = (city.observed[y_last]["urban"] > 0) & ~u0
    ring_obs = []
    for a, b in zip(RINGS_KM[:-1], RINGS_KM[1:]):
        ring_obs.append(int((obs_new & (dist_km >= a) & (dist_km < b)).sum()))
    report["kia_rings"]["observed"] = ring_obs
    for name, years, prob in runs:
        i = int(np.argmin(np.abs(years - y_last)))
        exp_new = np.where(u0, 0.0, prob[i])
        report["kia_rings"][name] = [float(exp_new[(dist_km >= a) & (dist_km < b)].sum())
                                     for a, b in zip(RINGS_KM[:-1], RINGS_KM[1:])]

    (out / "assessment.json").write_text(json.dumps(report, indent=1))

    # --- figures ----------------------------------------------------------------------
    # 1. Area through time: observed points vs ensemble means.
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    oy = [city.year] + obs_years
    oa = [u0.sum()] + [int((city.observed[y]["urban"] > 0).sum()) for y in obs_years]
    for k, (name, years, prob) in enumerate(runs):
        area = prob.reshape(len(years), -1).sum(1)
        ax.plot(years, area, color=style.SERIES[k], label=name)
        ax.annotate(name, (years[-1], area[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=style.INK_2)
    ax.plot(oy, oa, "o", ms=8, mfc=style.SURFACE, mec=style.INK, mew=1.8, label="observed (WSF)")
    ax.axvline(2015.5, color=style.INK_MUTED, lw=1, ls="--")
    ax.text(2015.6, ax.get_ylim()[1] * 0.97, "held out →", fontsize=8, color=style.INK_2, va="top")
    ax.set_title("Bangalore urban area: simulated (ensemble mean) vs observed", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Urban cells (~0.022 km² each)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "area_vs_observed.png", dpi=130)
    plt.close(fig)

    # 2. New growth around KIA by ring.
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = [f"{a}–{b} km" for a, b in zip(RINGS_KM[:-1], RINGS_KM[1:])]
    series = [("observed (WSF)", ring_obs)] + [(n, report["kia_rings"][n]) for n, _, _ in runs]
    width = 0.8 / len(series)
    x = np.arange(len(labels))
    for k, (name, vals) in enumerate(series):
        color = style.INK_2 if k == 0 else style.SERIES[k - 1]
        ax.bar(x + (k - (len(series) - 1) / 2) * width, vals, width * 0.92, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.set_title(f"New urban cells {city.year:g}–{y_last:g} by distance from KIA", loc="left")
    ax.set_ylabel("Cells")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "kia_rings.png", dpi=130)
    plt.close(fig)

    # 3. Maps: observed change and each scenario's change-agreement at the last year.
    n = len(runs) + 1
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.6))
    ax = axes[0]
    img = base_rgb(city)
    img[u0] = matplotlib.colors.to_rgb(style.URBAN_OLD)
    img[obs_new] = matplotlib.colors.to_rgb(style.SERIES[1])
    ax.imshow(img, interpolation="nearest")
    draw_anchors(ax, city, y_last)
    ax.plot(kx, ky, "^", ms=11, mfc=style.INK, mec=style.SURFACE, mew=1.5)
    ax.annotate("KIA", (kx, ky), xytext=(7, 5), textcoords="offset points", fontsize=8)
    style.map_axes(ax)
    ax.set_title(f"Observed: urban {city.year:g} (grey) + new by {y_last:g} (orange)", loc="left")
    for ax, (name, years, prob) in zip(axes[1:], runs):
        i = int(np.argmin(np.abs(years - y_last)))
        render_comparison(ax, city, u0, city.observed[y_last]["urban"], prob[i] >= 0.5,
                          title=f"{name}: P≥0.5 vs observed, {y_last:g}")
    fig.tight_layout()
    fig.savefig(out / "change_maps.png", dpi=110)
    plt.close(fig)

    for name, rows in report["per_year"].items():
        print(name)
        for r in rows:
            print(f"  {r['year']:.0f}: observed {r['observed_area']:6d}  expected {r['expected_area']:8.0f} "
                  f"({r['area_err_pct']:+5.1f}%)  FoM {r['fom_p50']:.3f}")
    print("KIA rings", labels)
    for k, v in report["kia_rings"].items():
        print(f"  {k:20s}", [round(x) for x in v])
    print("outputs in", out)


if __name__ == "__main__":
    main(sys.argv[1:])
