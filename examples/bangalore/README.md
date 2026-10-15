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

**Fixes this points to:**

1. **Anchor effects should take longer.** A 4–5 year anticipation-and-ramp is
   far too fast for a greenfield airport 30 km out. Calibrate anchor masses and
   ramps (and per-anchor gains) rather than only model coefficients.
2. **Rural villages should not grow like the city edge.** SLEUTH edge growth
   grows a ring around every village. Make spread depend on cluster size or
   local density.
3. **Add jobs-based anchors.** The tech-park corridors (ORR, Sarjapur,
   Manyata) pulled growth, and the model has no anchors for them after 2004.
   Dated metro stations from 2011 onward would help too.
4. **Better observed data.** The 2015→2019 jump mixes two sensors. GHSL
   R2023A epochs, or a single consistent product, would make the "now"
   comparison cleaner.
