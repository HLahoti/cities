# Bangalore, 2004 → today

This folder holds a real city built from open data. The start state is 2004,
when HAL was the city's airport and Kempegowda International Airport (KIA) had
been contracted but not yet built. The simulations run forward and are scored
against observed growth.

## The city file

`bangalore_2004.city.json` (+ `.npz`) covers 12.75–13.30°N, 77.35–77.90°E. The
grid is 408 × 408 cells of about 146 m × 149 m. Its `meta.georef` field maps
cells back to longitude and latitude.

| Layer | Source | Notes |
|---|---|---|
| `density` (2004) and observed `density` 2006–2015 | DLR WSF Evolution v1 | Share of each cell's 30 m pixels that are built (0–1). This measures horizontal built-up surface, not floor area or height. 2019 has no density because the 10 m sensor isn't comparable. |
| `urban`, `urbanized_at`, `observed` 2006–2015 | DLR **World Settlement Footprint Evolution** v1, tile 76_12 (CC BY 4.0) | Gives the year each 30 m pixel was first built. A cell counts as urban when ≥ 25 % of its pixels are built. |
| `observed` 2019 | DLR **WSF 2019** v1 (CC BY 4.0) | 10 m pixels. The threshold is scaled to 0.139 to match WSF-Evo, and 2019 is assumed to include everything urban in 2015. **This is the lowest-confidence year.** |
| `roads`, `rail`, `water`, `excluded` | **OpenStreetMap** via Overpass (ODbL) | Present-day data, filtered back to 2004 (see below). Protected land covers military areas, aerodromes, reserves and parks ≥ 5 ha; forest is set to 0.8. |
| `elevation` | AWS Terrain Tiles (SRTM-derived) | 679–1041 m. |
| anchors | Nominatim / OSM | CBD (Vidhana Soudha), HAL Airport, Electronic City, ITPL Whitefield, Peenya. KIA is added by the scenarios. |

**Filtering OSM back to 2004.**
- **Removed:** NICE Road, the Bengaluru–Mysuru and Bengaluru–Chennai
  expressways, the Satellite Town Ring Road, the Devanahalli bypass, the
  Electronic City flyover, and the roads inside the KIA site.
- **Downgraded:** NH648 (to a local road) and Ballari Road north of Hebbal (to
  an arterial).
- **Still present:** newer *secondary* roads and the metro. Removing those
  would need historical OSM data.

### Timeline (scenario events)

| Year | Event | Source |
|---|---|---|
| 2004 | KIA announced (anchor starts ramping) | Concession agreement 2004 |
| 2005 | KIA site acquired and excluded from private development | Construction began 2 Jul 2005 |
| 2007 | NICE Road opens | **Approximate.** The road opened in phases; the Mysore Rd–Tumkur Rd stretch was complete by 2007. |
| 2008 | KIA opens with its footprint and access road; the HAL civil enclave closes and HAL becomes an industrial anchor | KIA opened 24 May 2008 |
| 2014 | Hebbal–KIA elevated expressway | Opened Jan 2014 |
| 2023 | Bengaluru–Mysuru Expressway | Opened 2023 |

## Rebuilding from scratch

```bash
mkdir -p data/raw/{wsf_evo,wsf2019,osm,nominatim}
curl -o data/raw/wsf_evo/WSFevolution_v1_76_12.tif https://download.geoservice.dlr.de/WSF_EVO/files/WSFevolution_v1_76_12.tif
curl -o data/raw/wsf2019/WSF2019_v1_76_12.tif     https://download.geoservice.dlr.de/WSF2019/files/WSF2019_v1_76_12.tif
# OSM: roads / rail / water / protected Overpass queries (see tools/build_bangalore.py docstring)
python3 -P -E tools/build_bangalore.py      # terrain tiles are fetched automatically
```

## Scenarios

| File | What it runs |
|---|---|
| `real_sleuth.json` | SLEUTH with KIA and all events; parameters are not calibrated |
| `calibrated_sleuth.json` | SLEUTH calibrated on 2006–2015 with FoM × (1 − area error), 60 trials × 2 runs; 2019 held out |
| `calibrated_no_kia.json` | The same parameters with no KIA: a counterfactual |
| `real_potential.json` | Constrained CA. Its demand is a flat 1,050 cells a year taken from the observed 2004–2015 trend, so **its area match is partly by construction**. |

```bash
python3 -m cityca run examples/bangalore/scenarios/calibrated_sleuth.json
python3 -P -E tools/assess_bangalore.py runs/<run-a> runs/<run-b> ...   # ring + per-year report
```

## Findings (first pass, 2026-10-15)

City-wide, the models land close to observed growth:

* **Area.** Calibrated SLEUTH is +1 % (2006) to +17 % (2015) high, and +15 %
  on the held-out 2019. The constrained CA stays within ±5 %.
* **Placing new growth.** On P ≥ 0.5 consensus maps, FoM is 0.28–0.35 by
  2015–2019.

**The airport effect is wrong.** Between 2004 and 2019, the observed new
urban area is:
- 1,219 cells within 10 km of KIA;
- 3,797 cells 10–20 km from KIA.

With the calibrated parameters, a sweep of KIA's strength (`mass`) gives:

| KIA mass | New cells 0–10 km | New cells 10–20 km | FoM 2019 |
|---|---|---|---|
| 0 (no airport) | 1,445 | 4,956 | **0.341** |
| 0.2 | 2,092 | 5,168 | 0.340 |
| 0.4 | 2,605 | 5,405 | 0.335 |
| 1.2 (default airport) | 4,602 | 6,153 | 0.318 |

Every bit of KIA strength makes the forecast worse. By 2019 the airport had
produced almost no built-up growth around it. Bangalore's real growth was
fringe infill along the ORR, Whitefield, Sarjapur and Electronic City, which
all the models under-predict.

**Fixes this pointed to** (implemented in v2, below):

1. **Anchor effects should take longer.** A 4–5 year anticipation-and-ramp is
   far too fast for a greenfield airport 30 km out.
2. **Rural villages should not grow like the city edge.** SLEUTH edge growth
   grew a ring around every village.
3. **Add jobs-based anchors.** Tech-park corridors and the metro had no
   anchors.
4. **Better observed data.** The 2015→2019 jump mixes two sensors. This one
   is still open.

## v2: the three fixes (2026-10-15)

| Fix | Implementation |
|---|---|
| Slow KIA build-up | Anchors gain a `ramp_shape` option; `logistic` is a slow-start S-curve. The airport preset now has 5 % anticipation and a ~20-year ramp. KIA's `mass` and `ramp_years` are calibrated (`anchors.0.*`). |
| City-influenced villages | A new `urban_field` factor measures the Gaussian-weighted urban share within ~3 km, saturating at `ref`. SLEUTH `edge_modifiers` scale edge growth at each source cell by it, so a village spreads only once the city reaches it. `ref` is calibrated. |
| Tech parks and metro | New `tech_park` and `station` anchor kinds with dated openings from `sources/*.tsv`: 5 parks and 82 stations, matched to OSM. A per-anchor `seeds` weight keeps stations from leapfrog-seeding. `field.kind_weights` scales each anchor kind and is calibrated. |

`tools/build_bangalore_v2.py` writes the cumulative ablation scenarios:

* **B:** A + slow KIA ramp.
* **C:** B + the village fix.
* **D:** C + tech parks and metro.

The `calibrated_v2_*.json` files hold each variant's calibrated parameters.
Every variant used the same calibration budget (60 trials × 2 runs, fit on
2006–2015). Results below are 20-run ensembles; 2019 is held out.

| Variant | Calibration score (2006–15) | 2019 area error | 2019 FoM (P ≥ 0.5) | KIA ≤ 10 km new cells (obs 1,219) | Rural > 3 km new cells (obs 6,897) |
|---|---|---|---|---|---|
| A baseline | 0.193 | +15.0 % | 0.322 | 4,610 (3.8×) | 12,314 (1.8×) |
| B + KIA ramp | 0.203 | +8.0 % | 0.334 | 3,031 (2.5×) | 9,774 (1.4×) |
| C + villages | **0.245** | +3.7 % | **0.366** | **1,296 (1.06×)** | 4,743 (0.69×) |
| D + tech/metro | 0.235 | **+0.8 %** | 0.363 | 1,414 (1.16×) | 4,960 (0.72×) |

**What improved**

* **Slow KIA ramp:** helps on its own, but most of the airport error
  disappears only once the village fix is added. Much of the old blob around
  KIA was villages near the airport growing rings, boosted by the anchor.
* **Village fix:** the biggest single gain. FoM on held-out 2019 rose from
  0.33 to 0.37, the rings are gone, and growth within 10 km of KIA is now
  within 6–16 % of observed.
* **Tech parks and metro:** no measurable gain. The total area in 2019 is
  closest to observed (+0.8 %), but placement accuracy is the same as C.
  Calibration does favour a strong station weight (≈ 2.8), but:
  - Within 1 km of the 29 stations open by 2017, only 38 cells became newly
    urban, because those areas were already built in 2004. The metro's effect
    there is densification, which a binary built/not-built layer cannot show.
  - The tech-park catchments are over-predicted by all variants (1.2–1.4×),
    because most parks sit inside or beside land that was already urban.

**Still wrong**

* **Rural growth is now *under*-predicted** (0.7×), and rural placement is no
  better (FoM ≈ 0.16). Scattered rural building is hard to place in any
  variant.
* **The fringe is over-filled** (1.2–1.3×). The model grows a compact ring
  where reality left gaps.
* **Better data would help:** a continuous density or intensity layer so
  metro densification shows up, and a single-sensor "now" year.

## v3: built-up density (2026-10-15)

`urban` used to be a yes/no mask. The city now also carries `density`, the
built-up share of each cell (0–1), and maps draw it as a heatmap that runs
from bare land to fully built. Simulations evolve density with a pluggable
**process**, `densify`. Urban cells thicken logistically,
`d ← d + r·d·(1 − d/K(x))`, and new cells start at `new_density`. The local
capacity `K(x)` rises with how much city surrounds the cell (`urban_field`)
and with the anchor potential.

The densification parameters were calibrated on observed density for
2006–2012 (`--objective density`), with the v2 growth parameters held fixed.
2015 is held out. Scoring uses only cells urban in both maps; skill is
1 − MAE / MAE of "2004 density unchanged".

| Run (20-run ensembles) | 2015 density MAE | 2015 skill | Δdensity in 2004 core, obs +0.139 | Δdensity on fringe, obs +0.217 | Δdensity ≤1 km of metro, obs +0.049 |
|---|---|---|---|---|---|
| **calibrated v3** | **0.106** | **+0.51** | +0.137 | +0.195 | +0.067 |
| control: no densification | 0.185 | +0.14 | +0.000 | +0.034 | +0.002 |
| control: no anchor term in capacity | 0.107 | +0.50 | +0.132 | +0.191 | +0.067 |
| control: no metro | 0.106 | +0.51 | +0.137 | +0.194 | +0.067 |

* **Densification is real and the model captures it.** Cells already urban in
  2004 went from 0.75 to 0.89 built-up by 2015. The calibrated process
  matches the core and fringe gains closely and roughly halves the density
  error on the held-out year.
* **Anchors and metro add almost nothing to the density fit.** Within 1 km of
  stations the land was already ~92 % built-up in 2004, so there is little
  footprint left to fill. The metro's effect is vertical (more floor space),
  which a footprint dataset cannot see. Testing it needs building-height or
  floor-area data, such as GHSL-BUILT-H or the 3D building footprints that
  have appeared since 2020.
* **Remaining density errors.** The north and east fringe come out too dense
  (the simulated fringe is more solid than reality). The south and west
  fringe come out not dense enough, as the difference map in the assessment
  output shows.

## v4: learned growth rules and rolling-origin validation (2026-10-15)

**Learned growth rules.** The transition surface comes from a classifier
trained on observed transitions (`cityca learn`), not from hand-picked
factors.

**Validation.** `validation_v4.json` runs a rolling-origin test:
- **Origins:** 2006, 2008, 2010, 2012 and 2015.
- **Procedure:** each method is fitted on data up to the origin, forecasts from
  the observed map at that origin, and is scored at every later observed year
  (12 runs per forecast, 25 forecasts per method).
- **Matching:** the proximity baseline and both learned models use the same
  trend demand, so they differ only in *where* they place growth.

| Method | Mean FoM (all) | 2-year horizon | 9–13-year horizon | FoM (single-sensor years only) |
|---|---|---|---|---|
| **learned-logistic** | **0.308** | 0.271 | 0.343 | **0.325** |
| **learned-gbm** | **0.310** | 0.274 | 0.337 | **0.328** |
| sleuth-cem (re-calibrated at each origin, 5 elite parameter sets) | 0.201 | 0.135 | 0.272 | 0.186 |
| sleuth-fixed (v3 parameters, fitted on 2006–15, so it has seen the future) | 0.197 | 0.093 | 0.285 | 0.179 |
| proximity (grow next to existing urban land) | 0.182 | 0.085 | 0.271 | 0.171 |

* **The learned models place new growth about 55 % better than calibrated
  SLEUTH, and about twice as well at short horizons.** Logistic regression
  matches gradient boosting, so the signal is mostly simple.
* **The new search works.** Re-calibrating SLEUTH honestly at each origin with
  CEM matches the earlier calibration that had seen the future.
* **The advantage disappears for 2015→2019.** All methods score FoM 0.12–0.14
  there, and the learned model's held-out ROC AUC falls from about 0.98 to
  0.78. The 2019 sensor marks about 4× more cells as newly urban, so this
  interval measures something different.
* **Short-horizon density.** Over 2–3 years the densify process is slightly
  worse than "no change". It gets average infill right (+0.034 vs +0.032 over
  2012→2015) but spreads it evenly, while real infill is lumpy. Over longer
  horizons its skill rises to +0.2 to +0.4.

**What predicts new urban land.** Permutation importance from
`models/logistic_2015/importance.png`, held-out 2015→2019:

| Rank | Input | Drop in held-out AUC when shuffled |
|---|---|---|
| 1 | Urban share within about 600 m | 0.110 |
| 2 | The cell's own existing built share | 0.063 |
| 3 | Distance to urban land | 0.062 |
| 4 | Urban share within about 1.8 km | 0.050 |
| – | Distance to the CBD | 0.012 |
| – | Anchors, roads, protected land, terrain | ≤ 0.004 each |

Bangalore grew by filling in and thickening around existing settlement.
Infrastructure and anchors add little once that is known.

**Forecast.** `scenarios/v4_learned_forecast.json` starts from the observed
2015 map and runs to 2026 with the logistic model, demand of 1,030 cells a
year (the 2004–2015 trend) and densification.

## v5: learned densification, growth amount and consistent data (2026-10-15)

**#3 Consistent data: GHSL R2023A.** `bangalore_2000_ghsl.city.json` uses
the same grid, roads and anchors as the WSF city, with its observations taken
from the JRC Global Human Settlement Layer instead:
- built-up surface share (`density`) and built volume for 2000, 2005, 2010,
  2015 and 2020, from one processing chain;
- 1995 as momentum history;
- GHS-POP population, plus 2025/2030 projections stored as such.

The urban threshold (0.096) is set so 2015 urban land matches the WSF city.
Built with `tools/build_bangalore_ghsl.py`; GHS-POP is converted by
`tools/convert_ghs_pop.py`, run in an isolated `uv` environment because it
needs `imagecodecs`.

**#1 Learned densification and momentum.**
- `DensityModel` is a two-part model: will a cell's built share rise in a
  period, and by how much.
- *Momentum* features (recent nearby growth, recent density change) come from
  lag layers.
- `freeze` keeps the growth model's density inputs at their
  start-of-forecast values.

**#2 Growth amount.** Fitted growth laws (`demand_fit.select_law`) and
population scaling (A = c·P^β along GHS-POP) were tested against the
mean-growth trend.

All results are rolling-origin forecasts (12 runs each), fitted only on data
up to each origin. Configs are in `validation_v5_*.json`; summarise them with
`tools/summarize_validation.py`.

### What improved

| Change | Data | Before → after |
|---|---|---|
| Momentum features | WSF, all origins, single-sensor targets | Placement FoM 0.325 → **0.339** (short horizon 0.295 → 0.314) |
| Learned densification (expected, urban cells) | WSF, fair origins 2010/12/15 | Density skill on urban cells +0.085 → **+0.16** |
| Frozen density inputs with learned densification | GHSL, origins 2005/10/15 | Placement FoM 0.400 (old densify) → **0.482** |
| Consistent data for the latest interval | 2015 origin | FoM 0.13 (WSF → 2019) → **0.37** (GHSL → 2020) |

Head-to-head on GHSL (2005/2010/2015 → up to 2020):

| Method | Placement FoM | Area error |
|---|---|---|
| **learned growth (frozen) + learned density** | **0.482** | 2.6 % |
| learned growth, no densification | 0.443 | 2.6 % |
| learned growth + old densify | 0.400 | 2.6 % |
| SLEUTH re-calibrated with CEM | 0.277 | 12.0 % |
| proximity | 0.256 | 2.6 % |

### What didn't improve

* **Growth amount.** Neither fitted laws nor population scaling beat the
  simple mean-growth trend.
  - *Fitted laws:* with 3–6 noisy points, saturating laws read short-term
    wiggles as a slowdown. The WSF area error went from 7.5 % to 8.5 %.
  - *Population scaling:* the fitted β was unstable (about 1.6 on early GHSL,
    because area grew faster than population). Extrapolated along the
    population projection, it overshot area by 11–17 % on average and up to
    +33 %.
  - *Keep the trend* until there are longer and cleaner area series.
* **GHSL density.** GHSL's built-up share hardly changes between epochs, so
  "no change" is hard to beat. Learned densification still scores below it
  (−0.46), though it is far better than the old WSF-tuned process (−3.3).
* **Metro.** Volume growth within 1 km of stations open by 2017 was no faster
  than 1–3 km away at the same distance from the CBD (2010–2020). But GHSL
  volume changes track footprint changes closely; its yearly volumes appear
  to come from footprint × a fixed height model, so they can't show vertical
  growth. Testing the metro properly needs yearly building heights, such as
  Open Buildings 2.5D Temporal (2016–2023) via Earth Engine.

### Forecast

`scenarios/v5_forecast_2030.json` uses the best configuration:
- models trained on all GHSL intervals by `tools/train_bangalore_v5.py`;
- starts from the observed 2020 map and runs to 2030;
- demand of 1,327 cells a year, the 2000–2020 mean.

It projects 47,721 → about 61,000 urban cells by 2030, mostly fringe infill
to the north, east and south-east, with densification of existing areas.
