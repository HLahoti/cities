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
