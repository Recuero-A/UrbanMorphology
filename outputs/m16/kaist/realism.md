# Realism of the KAIST campus cities against 156 real neighbourhoods (EXPLORATORY)

Real pool: 120 M15 pilot windows (Spain, Korea, Peru) + 36 style references. 13 features, none optimised by any method. Typicality = mean distance to the 5 nearest real windows / the same for real windows (1 = as typical as a real place; lower is better; ranking key). Precision / density (k = 5) and in-range (all features within the real 2.5–97.5 % range) against the same pool. 27 windows per method (3 seeds x 9).

| Rank | Method | Typicality (mean) | Typicality (worst 10 %) | Precision | Density | In range | KAIST distance |
|---|---|---|---|---|---|---|---|
| 1 | M* + grid | 0.81 | 1.04 | 0.96 | 1.04 | 0.78 | 1.32 |
| 2 | NA-fixed | 0.99 | 1.23 | 1.00 | 1.21 | 0.93 | 1.22 |
| 3 | NA-random | 0.99 | 1.08 | 0.96 | 1.07 | 0.78 | 1.28 |
| 4 | NA-myopic | 1.05 | 1.24 | 0.81 | 0.63 | 0.41 | 1.36 |
| 5 | NA-adaptive | 1.06 | 1.35 | 0.81 | 0.93 | 0.52 | 1.36 |
| 6 | NA-SMC | 1.11 | 1.39 | 0.48 | 0.37 | 0.22 | 1.45 |
| 7 | M12-HR | 1.20 | 1.57 | 0.96 | 1.04 | 0.67 | 1.14 |
| 8 | M* | 1.23 | 1.41 | 1.00 | 0.91 | 0.63 | 1.13 |
| 9 | A | 1.25 | 1.51 | 1.00 | 0.95 | 0.67 | 1.13 |
| 10 | P | 1.29 | 1.40 | 0.70 | 0.19 | 0.22 | 1.51 |
| 11 | M13-SMC | 1.40 | 1.94 | 0.93 | 1.01 | 0.56 | 1.09 |
| 12 | M* TNse | 1.50 | 2.00 | 1.00 | 0.65 | 0.22 | 1.16 |
| 13 | M* + culdesac | 1.74 | 2.36 | 1.00 | 0.72 | 0.37 | 1.18 |
| 14 | M* + radial | 2.47 | 3.81 | 0.81 | 0.21 | 0.07 | 2.14 |
| 15 | M* + organic | 2.52 | 3.65 | 0.85 | 0.19 | 0.04 | 2.12 |
| ref | real KAIST (vehicular) | 1.59 | 2.13 | 1.00 | 0.89 | 0.33 | 0.00 |

Real pool windows score typicality 1.00 by construction (leave one out). The real KAIST campus is scored against the same pool: a university campus is not a typical urban neighbourhood, so its typicality is high.
