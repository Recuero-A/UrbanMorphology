# Machu Picchu terrain benchmark: every street generator (EXPLORATORY)

Protocol, terrain adaptiveness score and flags: [docs/SITE_BENCHMARKS.md](../../../docs/SITE_BENCHMARKS.md) (declared before scoring). Code: `scripts/site_benchmark.py`.

**City limits** 1.43 km² on the Machu Picchu ridge (1967-2650 m; median slope 73 %; 87 % of the land steeper than 45 %), jungle outside; GLO-30 DSM, no supplied roads; seeds 1-3 (means ± seed SD).

* **Realism** = typicality vs 156 real neighbourhoods (1 = as typical as a real one; **lower = more realistic**).
* **TAS** = terrain adaptiveness 0-100 (**higher = better**) = mean of G (street length ≤ 15 % grade), C (contour following: 1 along contours, 0.36 random, 0 straight downhill) and S (share of street length on the gentler half of the land; 0.5 = unrelated to slope).
* **SPARSE** = under 5 km of street per km²: no town was built there.
* **(ext)** = external baselines added in M18, same inputs (DEM, water, city limits, seeds 1-3) and the same scoring: *leastcostpath (ext)* = leastcostpath (Lewis) routed between seeded anchors, merge by us; *citygen P&M (ext)* = t-mw/citygen run headless with only Parish & Müller 2001's local constraints added. Setup, parameters and caveats: [outputs/m18/external/README.md](../../m18/external/README.md).

Files: `fig_maps_seed1.png` (all maps), `maps/<rank>_<method>_s<seed>.png` (one map per method and seed), `fig_realism_vs_terrain.png`, `site_terrain.png`, `summary.csv`, `per_seed.csv`, `summary.json`.

| # | Generator | Realism | TAS | G gentle | C contour | S siting | km/km² | Coverage 60 m | Junctions/km² | 4-way | Dead ends/km | Largest piece | > 15 % | > 20 % | Water m | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M* + organic · F1+F4 | 0.90 ± 0.08 | 39 ± 2 | 0.22 | 0.43 | 0.51 | 31.2 | 96 % | 410 | 30 % | 1.0 | 100 % | 78 % | 71 % | 13 | 355 |
| 2 | M* + organic | 0.98 ± 0.02 | 39 ± 1 | 0.22 | 0.43 | 0.51 | 35.5 | 99 % | 505 | 31 % | 0.8 | 98 % | 78 % | 71 % | 23 | 361 |
| 3 | Tracer · organic | 1.11 ± 0.04 | 34 ± 2 | 0.16 | 0.37 | 0.49 | 22.4 | 96 % | 196 | 42 % | 2.2 | 19 % | 84 % | 78 % | 2 | 1 |
| 4 | M* + radial | 1.26 ± 0.16 | 39 ± 2 | 0.22 | 0.43 | 0.52 | 36.9 | 98 % | 499 | 27 % | 0.5 | 100 % | 78 % | 72 % | 11 | 371 |
| 5 | M* + radial · F1+F4 | 1.28 ± 0.06 | 38 ± 1 | 0.21 | 0.43 | 0.50 | 39.5 | 99 % | 531 | 30 % | 0.5 | 100 % | 79 % | 73 % | 33 | 483 |
| 6 | M* + grid | 1.38 ± 0.10 | 41 ± 0 | 0.22 | 0.42 | 0.60 | 12.4 | 81 % | 78 | 20 % | 2.7 | 70 % | 78 % | 72 % | 8 | 35 |
| 7 | Tracer · grid | 1.43 ± 0.13 | 34 ± 0 | 0.15 | 0.36 | 0.51 | 21.1 | 99 % | 123 | 71 % | 2.6 | 19 % | 85 % | 79 % | 0 | 1 |
| 8 | NA-SMC | 1.45 ± 0.28 | 56 ± 1 | 0.45 | 0.64 | 0.61 | 19.1 | 90 % | 102 | 16 % | 1.0 | 56 % | 55 % | 43 % | 0 | 267 |
| 9 | NA-SMC · F1+F4 | 1.51 ± 0.15 | 59 ± 2 | 0.48 | 0.66 | 0.63 | 17.9 | 87 % | 82 | 17 % | 1.3 | 67 % | 52 % | 39 % | 0 | 256 |
| 10 | NA-myopic | 1.65 ± 0.09 | 55 ± 3 | 0.44 | 0.63 | 0.58 | 12.7 | 84 % | 46 | 8 % | 1.8 | 46 % | 56 % | 43 % | 9 | 44 |
|  | **REAL roads (Hiram Bingham switchbacks)** | 1.73 ± – | 73 ± – | 0.71 | 0.83 | 0.67 | 3.8 | 23 % | 0 | 0 % | 0.2 | 64 % | 29 % | 19 % | 45 | – |
| 11 | NA-random | 1.80 ± 0.09 | 59 ± 4 | 0.48 | 0.65 | 0.65 | 10.2 | 76 % | 33 | 9 % | 3.0 | 23 % | 52 % | 40 % | 10 | 128 |
| 12 | leastcostpath (ext) | 1.86 ± 0.09 | 73 ± 0 | 0.73 | 0.79 | 0.67 | 17.8 | 66 % | 52 | 4 % | 0.1 | 97 % | 27 % | 10 % | 9 | 11 |
|  | **REAL Inca trails / citadel paths** | 1.95 ± – | 57 ± – | 0.37 | 0.52 | 0.81 | 4.8 | 35 % | 26 | 16 % | 2.0 | 95 % | 63 % | 53 % | 177 | – |
| 13 | M* TNse (sparse) | 1.98 ± 0.26 | 80 ± 4 | 0.88 | 0.83 | 0.69 | 4.4 | 42 % | 3 | 0 % | 2.0 | 29 % | 12 % | 5 % | 0 | 6 |
| 14 | NA-fixed | 2.04 ± 0.26 | 64 ± 0 | 0.54 | 0.67 | 0.71 | 7.6 | 66 % | 19 | 9 % | 3.5 | 26 % | 46 % | 31 % | 0 | 149 |
| 15 | NA-fixed · F1+F4 | 2.16 ± 0.10 | 65 ± 2 | 0.56 | 0.68 | 0.71 | 8.9 | 71 % | 23 | 12 % | 3.4 | 23 % | 44 % | 30 % | 9 | 189 |
| 16 | NA-adaptive | 2.29 ± 0.26 | 65 ± 2 | 0.56 | 0.69 | 0.71 | 7.8 | 64 % | 20 | 10 % | 4.0 | 24 % | 44 % | 29 % | 0 | 78 |
| 17 | A · F1+F4 (sparse) | 2.78 ± 0.38 | 57 ± 7 | 0.44 | 0.54 | 0.74 | 4.0 | 45 % | 12 | 7 % | 7.1 | 16 % | 56 % | 43 % | 1332 | 6 |
| 18 | A (sparse) | 2.84 ± 0.35 | 57 ± 7 | 0.44 | 0.53 | 0.75 | 3.8 | 44 % | 11 | 12 % | 7.3 | 17 % | 56 % | 44 % | 1340 | 5 |
| 19 | M13-SMC · F1+F4 | 2.84 ± 0.20 | 71 ± 1 | 0.61 | 0.67 | 0.85 | 5.3 | 51 % | 11 | 11 % | 6.3 | 17 % | 39 % | 24 % | 0 | 32 |
| 20 | M13-SMC | 2.93 ± 0.14 | 71 ± 1 | 0.62 | 0.69 | 0.82 | 5.7 | 54 % | 11 | 4 % | 6.2 | 11 % | 38 % | 24 % | 19 | 35 |
| 21 | M* + culdesac (sparse) | 2.98 ± 0.17 | 59 ± 1 | 0.48 | 0.60 | 0.70 | 4.7 | 50 % | 14 | 11 % | 7.3 | 20 % | 52 % | 38 % | 8 | 17 |
| 22 | M* · F1+F4 (sparse) | 3.12 ± 0.18 | 67 ± 1 | 0.58 | 0.66 | 0.77 | 3.7 | 42 % | 7 | 16 % | 7.4 | 14 % | 42 % | 28 % | 0 | 9 |
| 23 | M* (sparse) | 3.12 ± 0.10 | 65 ± 4 | 0.56 | 0.66 | 0.74 | 4.2 | 47 % | 7 | 0 % | 7.9 | 12 % | 44 % | 29 % | 0 | 10 |
| 24 | M12-HR | 3.17 ± 0.26 | 82 ± 2 | 0.90 | 0.84 | 0.73 | 6.1 | 59 % | 10 | 15 % | 6.4 | 7 % | 10 % | 2 % | 9 | 6 |
| 25 | P (sparse) | 4.27 ± 1.42 | 83 ± 3 | 0.81 | 0.77 | 0.92 | 0.8 | 13 % | 1 | 0 % | 9.4 | 28 % | 19 % | 13 % | 146 | 2 |
| 26 | citygen P&M (ext) (sparse) | 4.32 ± 1.03 | 75 ± 4 | 0.92 | 0.88 | 0.45 | 2.9 | 11 % | 14 | 2 % | 5.1 | 100 % | 8 % | 0 % | 0 | 1 |
