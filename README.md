# cityca — cellular-automata urban growth

Treat a city as a cellular organism: represent it as a dense grid of cell
properties, grow it with CA rules, and see how growth responds to new
*anchors* (an airport, a station) that set up a potential field across the
grid.

```
python3 -m cityca synth                                  # build the test city (examples/)
python3 -m cityca show examples/riverton.city.json       # render all layers + map
python3 -m cityca run examples/scenarios/airport_2030.json
python3 -m cityca compare examples/scenarios/baseline_2030.json \
                          examples/scenarios/airport_2030.json --runs 20
python3 -m cityca calibrate examples/scenarios/replicate_1950_1990.json \
    --space diffusion=1:20 breed=1:40 spread=2:30 slope_resistance=10:90 road_gravity=5:80
python3 -m cityca learn examples/bangalore/scenarios/calibrated_v3_density.json \
    --until 2015 --learner logistic                      # learn growth rules from history
python3 -m cityca validate examples/bangalore/validation_v4.json   # rolling-origin comparison
python3 -m cityca list                                   # every pluggable component
python3 -m pytest
```

Every command writes to a fresh directory named
`<city_name>_<datetime>_<run_type>` under `runs/`, for example
`runs/riverton_20261015-143502_airport`. For `run`, the run type is the
scenario's `name`. For the other commands it is `compare`, `calibrate` or
`show`. Use `--root DIR` to change the parent folder (scenarios can also set
`"output": {"root": ...}`), or `--out DIR` to give an exact path and skip the
automatic naming.

Dependencies: numpy, scipy, matplotlib, Pillow; scikit-learn for `cityca.learn`
(`pip install -e .[learn]`) (and ffmpeg for `.mp4`; use
`"animation": "growth.gif"` without it).

## Architecture

Each box is a **registry** of swappable components. Scenarios name components
by `type`, so trying a different combination means editing JSON, not code.

| Package | Role | Registries |
|---|---|---|
| `cityca/grid` | City state: named dense `(H, W)` layers, derived properties (cached, auto-invalidated), city-file I/O, rasterising vector features | `LAYERS`, `DERIVED`, `SOURCES`, `GENERATORS` |
| `cityca/anchors` | Anchors (mass, decay kernel, announce→open→mature timeline) and the potential field Φ | `ANCHOR_KINDS`, `DECAY` (in `kernels.py`) |
| `cityca/engine` | Growth models, factors, demand laws, allocators, events, the simulation loop, Monte Carlo, calibration | `MODELS`, `FACTORS`, `DEMANDS`, `ALLOCATORS`, `EVENTS` |
| `cityca/metrics` | Urban-form metrics (area, edges, clusters, fractal dimension, …), map comparison (Lee–Sallee, kappa, figure of merit), radial density profiles | `METRICS` |
| `cityca/learn` | Learned transition rules: feature sets from any layers, observed-state replay, transition datasets, logistic / gradient-boosted learners, permutation importance | `LEARNERS` |
| `cityca/validate.py` | Rolling-origin validation: fit each method on data up to an origin year, forecast from the observed state, score by horizon | – |
| `cityca/viz` | Composite maps, layer panels, Φ fields, probability maps, charts, MP4/GIF animation | – |
| `cityca/scenario.py` | Wires a city, a model, anchors and events into one run and writes every output | – |

### Phase 1 — grid and state (`cityca/grid`)

* `CityGrid` holds **base layers** (`urban`, `urbanized_at`, `elevation`,
  `water`, `excluded`, `roads`, plus anything else such as `population`,
  `land_value` or `zoning`). Any layer name is allowed, and known names get dtype
  and colour specs from `layers.py`.
* **Derived layers** are computed on demand and cached by dependency version:
  `slope`, `road_distance`, `road_nearest`, `urban_density(radius, decay)`,
  `urban_neighbors`, `center_distance`, `water_distance`, `developable`, `age`, …
  To add one:

  ```python
  from cityca.grid import derived
  @derived("rail_distance", deps=["rail"])
  def rail_distance(grid): ...
  ```

  `grid["rail_distance"]` then works everywhere, including in the generic
  `layer` factor and in `show`.

### City file format (`*.city.json` + optional `*.npz` sidecar)

```jsonc
{
  "format": "cityca/1", "name": "Riverton", "year": 1990,
  "grid": {"height": 200, "width": 200, "cell_size": 100, "units": "cells"},  // or "m"
  "layers": {                       // raster sources (pluggable: SOURCES)
    "elevation": {"source": "generator", "generator": "fractal_noise", "params": {"seed": 3}},
    "urban":     {"source": "image", "path": "urban.png", "threshold": 0.5},
    "landuse":   {"source": "image", "path": "landuse.png", "colors": {"#ff0000": 1, "#00ff00": 2}},
    "roads":     {"source": "npz", "key": "roads"},
    "zoning":    {"source": "constant", "value": 1}
  },
  "features": [                     // vector geometry burned on top (x = column, y = row)
    {"geometry": "polyline", "layer": "roads", "value": 3, "width": 1, "coords": [[0, 100], [199, 120]]},
    {"geometry": "polygon",  "layer": "excluded", "value": 1, "coords": [[150, 20], [185, 18], [190, 55]]},
    {"geometry": "circle",   "layer": "urban", "center": [44, 92], "radius": 9}
  ],
  "anchors": [{"name": "CBD", "kind": "cbd", "position": [44, 92]}],
  "observed": [{"year": 2000, "layers": {"urban": {"source": "npz", "key": "obs_2000_urban"}}}]
}
```

You can sketch a city design by hand: paint PNG masks or write polygons and
polylines in the JSON. `observed` snapshots are what calibration and
validation compare against.

### Phase 2 — growth engines (`cityca/engine`)

| Model | Source | Idea |
|---|---|---|
| `sleuth` | Clarke, Hoppen & Gaydos 1997 (`research/sfbay.pdf`) | 4 rules (spontaneous, new spreading centre, edge, road-influenced), 5 coefficients, boom/bust self-modification → S-curve |
| `potential` | White–Engelen constrained CA | `P = combine(factors)`; **demand** sets how many cells grow; an **allocator** picks where |
| `neighborhood` | Batty 1997 primer | IF *n* neighbours developed THEN develop with probability *p*, decaying to *pᵏ* after failed attempts |

**Factors** are the cell properties that enter the models, and they are the
main lever for finding out which properties explain growth: `urban_field`
(how much city surrounds a cell, at the km scale), `cluster_size`,
`neighborhood`,
`slope`, `road_access`, `center_access`, `anchor`, `excluded`, `water_amenity`,
`noise`, and the generic `layer` factor, which turns any layer into a factor
(`linear` / `inverse` / `decay` / `threshold`). Combine them as a `product`
(weights as exponents), a `sum`, or a `logistic`. SLEUTH accepts the same
factors as `modifiers`, and as `edge_modifiers`, which scale only edge growth.
For example, `urban_field` stops isolated villages spreading like the city
edge.

**Demand laws** set how many cells grow each step. This is where the organism
analogy lives: `allometric` is West–Brown–Enquist's ontogenetic growth law
`dA/dt = a·A^¾·(1 − (A/K)^¼)` (`research/Science_276_122_1997-1.pdf`). The
other laws are `logistic`, `exponential`, `constant` and `series`. An anchor's
`capacity` raises K: a new airport increases the city's adult size.

**Processes** run after the growth model each step, and are listed under
`"processes"` in a scenario. `densify` evolves the `density` layer (built-up
share 0–1) by logistic thickening toward a capacity that depends on factors.
If a city has no density data, it initialises density from the urban mask.
Calibrate it with `--objective density`.

**Events** fire at a given year: `add_road`, `add_anchor`, `remove_anchor`,
`burn` (any geometry into any layer, e.g. a greenbelt), `urbanize`, and
`set_param` (a policy or economic shock).

### Learning growth rules from data (`cityca/learn`)

Instead of hand-picking factors and their weights, `cityca learn` trains a
classifier to predict which cells become urban.
- **Inputs:** a `FeatureSet` that turns any base or derived layer into a
  predictor. Transforms and smoothing are optional.
- **Examples:** transitions between observed snapshots. The training states
  are rebuilt by `observed_state`, which replays events and anchors up to that
  year and then takes urban and density from the observations.
- **Output:** a held-out score and a ranked list of which inputs matter
  (permutation importance).

The saved model plugs into any simulation as a factor:

```json
{"type": "potential",
 "factors": [{"type": "learned", "model_file": "../models/logistic_2015/model.joblib"}],
 "demand": {"type": "constant", "cells": 1030},
 "allocator": {"type": "roulette", "gamma": 2}}
```

Set `"start_from_observed": 2015` in a scenario to forecast from the real 2015
map instead of the city file's start year.

### Calibration and validation

* `cityca calibrate` defaults to `--search cem`, an iterative cross-entropy
  search: each round re-fits a Gaussian to the best trials so far.
  `--search random` does uniform draws instead.
  - **Parallel:** trials are evaluated across worker processes (`--workers`)
    and average `--runs` stochastic runs each.
  - **Parameter sets:** the top `--elites` combinations are saved as
    `parameter_sets`. When a scenario carries `parameter_sets`, its Monte Carlo
    runs cycle through them, so forecasts include parameter uncertainty and not
    just chance.
* `cityca validate config.json` runs a **rolling-origin** comparison. For
  each origin year, every method:
  1. is fitted using only data up to that year;
  2. forecasts from the observed state at that year;
  3. is scored at every later observed year.

  The results show how accuracy holds up with forecast horizon
  (`validation.png`, `validation.json`). Method types are `potential`,
  `learned`, `calibrated` (re-calibrated at each origin with CEM) and `fixed`.
  See `cityca/validate.py`.

### Phase 3 — anchors and the potential field (`cityca/anchors`)

```
Φ(x, t) = Σ_k m_k(t) · K_k(d_k(x))
```

* `m_k(t)`: the anchor's strength over time. It is 0 before announcement,
  ramps linearly to `anticipation·mass` until opening (speculation), then rises
  to full mass over `ramp_years`.
* `K_k`: the decay kernel, one of `exponential`, `power`, `gaussian`,
  `linear`, `step`, or `dog`. `dog` is a Mexican hat: airports and ports
  *repel* housing from their noise zone and *attract* it beyond.
* Ramp shape: `ramp_shape` sets how the strength builds after opening, as
  `linear`, `saturating`, or `logistic` (slow-start, e.g. an airport city).
  `seeds` weights an anchor in leapfrog seeding; stations default to almost 0.
  `field.kind_weights` scales whole anchor kinds at once.
* `d_k`: either `euclidean` or `travel` distance. Travel distance is
  Dijkstra over a cost surface where highways are 5× faster and water is slow
  except on bridges, so influence stretches along the roads that serve the
  anchor.
* When an anchor opens, its **footprint** is built and an **access road** links
  it to the network.

The field is written to `grid["anchor_potential"]` every step, so any model or
factor can read it. In SLEUTH, the `anchor` modifier affects three things: it
biases where spontaneous seeds land, scales edge-growth rates, and scales
acceptance. `anchor_seeding` adds Poisson(rate·Σm) extra seeds, placed in
proportion to Φ⁺.

### Phase 4 — visualisation (`cityca/viz`)

A scenario run writes the following to its output directory:

* `map_start_end.png`: the start and simulated end maps.
* `anchor_field.png` and `anchor_strength.png`: the anchor field and each
  anchor's strength over time.
* `metrics.png`: metric time series, with observed points overlaid when
  history exists.
* `growth_types.png`: growth per SLEUTH rule.
* `radial_profile.png`: density decay with distance (Batty 2008).
* `validation_map.png`: hits, misses and false alarms against observed change.
* `growth.mp4`: an animated map plus a timeline marking events.
* `density_vs_observed.png` and `density_expected.png`: observed vs
  simulated built-up density with a difference map, and the Monte Carlo
  expected-density map (when density is modelled).

When a city has a `density` layer, maps draw it as a **heatmap**: a single
warm hue that fades in from bare land (transparent) to fully built (deep), with
a colour bar. Smoothing is display-only (`smooth=` in cells). Without
density, or with `mode="age"`, urban cells are coloured by when they were
urbanised.
* `probability.png` and `ensemble.png`: the Monte Carlo P(urban) map and
  the mean ± s.d. of the metrics.
* `history/`: every frame plus per-step records.
* `final.city.json`: the end state, reloadable as a new start.

`compare` produces side-by-side probability maps and a ΔP map.

## The test city

`cityca synth` builds **Riverton**, a synthetic coastal river-mouth city. A
hidden "truth" SLEUTH run (`synth.TRUTH`) grows it from 1950 to 1990, which
makes it a twin experiment: calibration should recover known parameters.

* `examples/riverton_1950.city.json`: the 1950 start, with observed snapshots
  for 1960–1990.
* `examples/riverton.city.json`: the 1990 "present day", used for forecasts.

## Real city: Bangalore 2004 → today

`examples/bangalore/` holds a city built from open data, starting in 2004.
Urban extent comes from DLR's World Settlement Footprint, roads, water and
protected land from OpenStreetMap filtered back to 2004, and elevation from
SRTM. It also has dated events: KIA announced in 2004 and opened in 2008, and
HAL's civil enclave closed in 2008. `tools/build_bangalore.py` rebuilds the
city, and `tools/assess_bangalore.py` scores runs against the observed maps
for 2006–2019. See `examples/bangalore/README.md` for sources, caveats and
findings. In short: area and fringe growth are close, but the airport's
effect on growth around it is strongly overestimated.

## Roadmap / known limits

* **Real data import.** Add GeoTIFF and shapefile sources (rasterio,
  geopandas) as new `SOURCES`, and possibly OSM roads.
* **Urban intensity.** Built-up density (footprint share) is modelled. Height
  and floor area are not, so vertical densification, such as near metro
  stations, is invisible. Several land-use states with competing transitions
  would make the organism analogy richer.
* **Co-evolving networks.** Roads are static except for scheduled events and
  anchor access roads. Growing the network with the city (WBE-style supply
  networks) is the natural next step.
* **Calibration.** Calibration uses CEM or random search over any scenario
  path, with parallel evaluation and kept parameter sets. Bayesian
  optimisation (e.g. Optuna) could be added as another search strategy.
* **Learning.** Learning only predicts *where* cells become urban. Two
  extensions would help:
  - learning *how much* densification happens per cell, since the current
    densify process gets the average right but not the location;
  - learning the amount of growth, instead of using a trend.
* **Speed.** Travel-distance fields cost one Dijkstra run per anchor. They are
  cached process-wide by road network, so Monte Carlo runs reuse them.
