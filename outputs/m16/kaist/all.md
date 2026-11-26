# Every milestone's method on the whole KAIST campus (EXPLORATORY; GLO-30 DSM; 1536 m; seeds [1, 2, 3])

Sorted by **KAIST distance** (mean scaled difference from the real campus roads, window by window, on the independent outcomes; lower = closer). **Precision** = share of generated windows that fall inside the manifold of 120 real M15 windows (real-vs-real benchmark 0.9451845556481908).

| Method | Milestone | kaist_distance | precision | street_km | coverage_60 | xwin_unreach | share_4way | deadend_per_km | loops_per_km2 | block_rect_share | jangle_right_share | stroke_xjunc_lw | tree_len_share | share_gt15 | seconds | classifier says |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M13-SMC | M13 | 1.09 | 0.96 | 29.32 | 0.72 | 0.39 | 0.17 | 3.28 | 29.10 | 0.04 | 0.31 | 1.13 | 0.27 | 0.03 | 47 | organic 70%, culdesac 30% |
| A | M13 | 1.13 | 1.00 | 39.39 | 0.83 | 0.11 | 0.20 | 2.83 | 53.26 | 0.01 | 0.26 | 1.32 | 0.20 | 0.05 | 6 | organic 89%, culdesac 11% |
| M* | M13 | 1.13 | 0.96 | 35.24 | 0.75 | 0.12 | 0.20 | 2.69 | 48.18 | 0.01 | 0.28 | 1.28 | 0.20 | 0.05 | 22 | organic 81%, culdesac 19% |
| M12-HR | M12 | 1.14 | 0.93 | 44.40 | 0.85 | 0.14 | 0.21 | 2.82 | 65.98 | 0.02 | 0.28 | 1.39 | 0.21 | 0.01 | 5 | organic 85%, culdesac 15% |
| M* TNse | M13 (TNse) | 1.16 | 1.00 | 32.44 | 0.73 | 0.67 | 0.20 | 3.46 | 36.31 | 0.03 | 0.28 | 0.89 | 0.27 | 0.05 | 19 | organic 63%, culdesac 37% |
| M* + culdesac | M16 | 1.18 | 0.92 | 22.61 | 0.66 | 0.72 | 0.17 | 4.64 | 8.19 | 0.01 | 0.29 | 1.05 | 0.39 | 0.05 | 11 | culdesac 56%, organic 44% |
| NA-fixed | M14 | 1.22 | 0.96 | 45.45 | 0.88 | 0.04 | 0.18 | 1.01 | 77.85 | 0.01 | 0.26 | 1.56 | 0.08 | 0.05 | 211 | organic 96%, culdesac 4% |
| NA-random | M14 | 1.28 | 1.00 | 55.55 | 0.91 | 0.10 | 0.21 | 0.90 | 115.71 | 0.01 | 0.26 | 1.58 | 0.07 | 0.06 | 168 | organic 96%, culdesac 4% |
| M* + grid | M16 | 1.32 | 1.00 | 48.77 | 0.87 | 0.01 | 0.31 | 0.85 | 100.31 | 0.27 | 0.55 | 2.94 | 0.05 | 0.09 | 31 | organic 96%, radial 4% |
| NA-myopic | M14 | 1.36 | 1.00 | 67.77 | 0.92 | 0.13 | 0.23 | 0.71 | 183.67 | 0.02 | 0.26 | 1.81 | 0.04 | 0.06 | 84 | organic 100% |
| NA-adaptive | M14 | 1.36 | 1.00 | 63.73 | 0.92 | 0.05 | 0.21 | 0.86 | 174.49 | 0.02 | 0.26 | 1.83 | 0.06 | 0.05 | 190 | organic 100% |
| NA-SMC | M14 | 1.45 | 1.00 | 76.21 | 0.91 | 0.04 | 0.24 | 0.44 | 267.03 | 0.03 | 0.26 | 2.02 | 0.02 | 0.06 | 371 | organic 100% |
| P | M12 | 1.51 | 0.89 | 34.08 | 0.90 | 1.00 | 0.20 | 0.45 | 39.42 | 0.00 | 0.15 | 0.48 | 0.06 | 0.03 | 1 | organic 100% |
| M* + organic | M16 | 2.12 | 1.00 | 166.53 | 0.91 | 0.00 | 0.41 | 1.07 | 1376.96 | 0.14 | 0.46 | 3.93 | 0.03 | 0.14 | 539 | radial 74%, organic 26% |
| M* + radial | M16 | 2.14 | 1.00 | 157.18 | 0.90 | 0.00 | 0.39 | 1.04 | 1066.98 | 0.34 | 0.57 | 5.30 | 0.03 | 0.13 | 561 | radial 78%, organic 22% |
| real KAIST (vehicular) | reference | 0.00 | 1.00 | 41.68 | 0.85 | 0.48 | 0.15 | 6.06 | 27.55 | 0.15 | 0.44 | 1.62 | 0.30 | 0.04 | – | culdesac 56%, organic 44% |
