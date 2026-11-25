# Petra terrain benchmark: every street generator (EXPLORATORY)

Protocol, terrain adaptiveness score and flags: [docs/SITE_BENCHMARKS.md](../../../docs/SITE_BENCHMARKS.md) (declared before scoring). Code: `scripts/site_benchmark.py`.

**City limits** 1.47 km² in the Petra basin (845-1161 m; median slope 26 %; 26 % of the land steeper than 45 %), desert outside; GLO-30 DSM, no supplied roads; seeds 1-3 (means ± seed SD).

* **Realism** = typicality vs 156 real neighbourhoods (1 = as typical as a real one; **lower = more realistic**).
* **TAS** = terrain adaptiveness 0-100 (**higher = better**) = mean of G (street length ≤ 15 % grade), C (contour following: 1 along contours, 0.36 random, 0 straight downhill) and S (share of street length on the gentler half of the land; 0.5 = unrelated to slope).
* **SPARSE** = under 5 km of street per km²: no town was built there.

Files: `fig_maps_seed1.png` (all maps), `maps/<rank>_<method>_s<seed>.png` (one map per method and seed), `fig_realism_vs_terrain.png`, `site_terrain.png`, `summary.csv`, `per_seed.csv`, `summary.json`.

| # | Generator | Realism | TAS | G gentle | C contour | S siting | km/km² | Coverage 60 m | Junctions/km² | 4-way | Dead ends/km | Largest piece | > 15 % | > 20 % | Water m | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M* + grid | 1.02 ± 0.15 | 56 ± 1 | 0.63 | 0.38 | 0.67 | 17.9 | 86 % | 160 | 28 % | 0.8 | 94 % | 37 % | 24 % | 0 | 36 |
| 2 | NA-SMC | 1.10 ± 0.05 | 58 ± 1 | 0.67 | 0.41 | 0.66 | 25.1 | 96 % | 242 | 20 % | 0.5 | 87 % | 33 % | 20 % | 0 | 710 |
| 3 | NA-SMC · F1+F4 | 1.13 ± 0.03 | 60 ± 3 | 0.70 | 0.42 | 0.69 | 21.6 | 90 % | 182 | 20 % | 0.6 | 94 % | 30 % | 18 % | 0 | 347 |
| 4 | NA-myopic | 1.16 ± 0.06 | 62 ± 3 | 0.72 | 0.41 | 0.73 | 18.9 | 88 % | 158 | 17 % | 0.8 | 95 % | 28 % | 16 % | 0 | 65 |
| 5 | Tracer · organic | 1.22 ± 0.12 | 45 ± 1 | 0.48 | 0.36 | 0.50 | 22.5 | 97 % | 177 | 44 % | 2.4 | 18 % | 52 % | 40 % | 0 | 1 |
| 6 | M* + radial · F1+F4 | 1.31 ± 0.10 | 53 ± 1 | 0.58 | 0.39 | 0.61 | 41.7 | 92 % | 691 | 37 % | 0.7 | 100 % | 42 % | 29 % | 0 | 482 |
| 7 | NA-random | 1.33 ± 0.14 | 66 ± 3 | 0.76 | 0.41 | 0.80 | 13.3 | 79 % | 80 | 15 % | 1.0 | 87 % | 24 % | 11 % | 0 | 152 |
| 8 | M* + organic · F1+F4 | 1.34 ± 0.23 | 52 ± 1 | 0.57 | 0.38 | 0.60 | 47.4 | 95 % | 877 | 40 % | 0.4 | 100 % | 43 % | 29 % | 0 | 754 |
| 9 | NA-adaptive | 1.37 ± 0.09 | 64 ± 4 | 0.75 | 0.39 | 0.78 | 15.0 | 82 % | 110 | 16 % | 1.2 | 89 % | 25 % | 13 % | 0 | 196 |
| 10 | NA-fixed · F1+F4 | 1.39 ± 0.17 | 68 ± 2 | 0.80 | 0.38 | 0.86 | 10.7 | 74 % | 56 | 15 % | 1.5 | 84 % | 20 % | 9 % | 0 | 212 |
| 11 | NA-fixed | 1.41 ± 0.19 | 68 ± 3 | 0.79 | 0.40 | 0.85 | 11.5 | 71 % | 63 | 14 % | 1.0 | 89 % | 21 % | 9 % | 0 | 288 |
| 12 | M* + organic | 1.54 ± 0.30 | 52 ± 3 | 0.58 | 0.38 | 0.60 | 49.8 | 93 % | 1054 | 39 % | 0.4 | 100 % | 42 % | 29 % | 0 | 551 |
| 13 | M* + radial | 1.57 ± 0.14 | 52 ± 4 | 0.58 | 0.38 | 0.61 | 48.3 | 93 % | 872 | 35 % | 0.6 | 99 % | 42 % | 29 % | 0 | 307 |
| 14 | Tracer · grid | 1.59 ± 0.10 | 46 ± 0 | 0.51 | 0.37 | 0.51 | 21.2 | 100 % | 122 | 67 % | 2.5 | 18 % | 49 % | 38 % | 0 | 1 |
|  | **REAL roads / tracks (Petra)** | 1.77 ± – | 82 ± – | 0.96 | 0.63 | 0.89 | 3.1 | 29 % | 7 | 9 % | 2.2 | 89 % | 4 % | 0 % | 0 | – |
| 15 | M13-SMC · F1+F4 | 1.79 ± 0.08 | 72 ± 1 | 0.84 | 0.39 | 0.94 | 7.3 | 59 % | 29 | 16 % | 3.1 | 71 % | 16 % | 6 % | 0 | 33 |
| 16 | P | 1.79 ± 0.03 | 72 ± 0 | 0.90 | 0.56 | 0.71 | 7.7 | 63 % | 21 | 17 % | 1.6 | 33 % | 10 % | 4 % | 0 | 2 |
| 17 | A | 1.88 ± 0.06 | 67 ± 3 | 0.77 | 0.36 | 0.87 | 9.5 | 68 % | 53 | 12 % | 3.8 | 68 % | 23 % | 11 % | 0 | 7 |
| 18 | M13-SMC | 1.93 ± 0.49 | 72 ± 1 | 0.86 | 0.37 | 0.92 | 7.3 | 57 % | 35 | 14 % | 3.2 | 59 % | 14 % | 5 % | 0 | 37 |
| 19 | M* | 2.03 ± 0.21 | 69 ± 3 | 0.83 | 0.36 | 0.89 | 7.8 | 61 % | 41 | 14 % | 3.9 | 66 % | 17 % | 7 % | 0 | 12 |
| 20 | A · F1+F4 | 2.06 ± 0.20 | 67 ± 4 | 0.77 | 0.36 | 0.87 | 8.5 | 67 % | 40 | 7 % | 3.9 | 53 % | 23 % | 12 % | 0 | 7 |
| 21 | M12-HR | 2.08 ± 0.20 | 78 ± 1 | 0.96 | 0.48 | 0.90 | 7.4 | 59 % | 39 | 15 % | 3.5 | 44 % | 4 % | 1 % | 0 | 7 |
|  | **REAL Petra trails / steps** | 2.11 ± – | 67 ± – | 0.80 | 0.53 | 0.69 | 6.2 | 53 % | 17 | 12 % | 4.3 | 23 % | 20 % | 10 % | 0 | – |
| 22 | M* · F1+F4 | 2.12 ± 0.11 | 69 ± 1 | 0.83 | 0.37 | 0.86 | 6.9 | 58 % | 28 | 14 % | 3.7 | 55 % | 17 % | 8 % | 0 | 9 |
| 23 | M* TNse | 2.28 ± 0.35 | 72 ± 1 | 0.89 | 0.62 | 0.66 | 8.1 | 56 % | 27 | 12 % | 2.7 | 43 % | 11 % | 4 % | 0 | 14 |
| 24 | M* + culdesac | 2.47 ± 0.45 | 62 ± 4 | 0.73 | 0.41 | 0.73 | 6.0 | 58 % | 17 | 11 % | 5.5 | 26 % | 27 % | 14 % | 0 | 17 |
