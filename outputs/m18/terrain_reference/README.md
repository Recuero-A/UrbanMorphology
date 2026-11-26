# M18 D11 step 1: real-city terrain reference curves and the terrain-realism score

Reference: 473 training tiles with LiDAR ({"ESP_BCN": 39, "ESP_MAD": 70, "ESP_VLC": 31, "USA_CHI": 71, "USA_LAX": 191, "USA_NYC": 71}). Script and definitions: `scripts/m18_terrain_reference.py`. Terrain-realism: mean |set − reference| / between-city SD over 5 slope classes × (density, coverage, contour index, grade p90, share > 15 %); lower = closer.

Real training cities scored against the pooled reference (yardstick): ESP_BCN 0.84, ESP_MAD 0.68, ESP_VLC 1.11, USA_CHI 1.55, USA_LAX 0.24, USA_NYC 0.69

| Set (GEN2 city tiles; "natural:" = leakage-free natural windows) | terrain realism | TAS | G | C | S |
|---|---|---|---|---|---|
| real GEN2 | 0.47 | 66.2 | 0.98 | 0.51 | 0.49 |
| NA-SMC [TN] | 0.62 | 67.6 | 0.98 | 0.52 | 0.53 |
| NA-fixed [TN] | 0.75 | 67.3 | 0.97 | 0.52 | 0.53 |
| M* [TN] | 0.76 | 67.3 | 0.97 | 0.50 | 0.55 |
| natural: M* [TNlp] | 1.02 | 67.2 | 0.88 | 0.49 | 0.64 |
| M*+repair [TN] | 1.12 | 66.8 | 0.97 | 0.50 | 0.54 |
| A [TN] | 1.22 | 66.4 | 0.97 | 0.49 | 0.54 |
| natural: P | 1.36 | 69.0 | 0.90 | 0.50 | 0.67 |
| P | 1.49 | 66.3 | 0.96 | 0.47 | 0.55 |
| natural: NA-fixed [TNg] | 1.57 | 64.8 | 0.84 | 0.51 | 0.59 |
| natural: M13-SMC [TNg] | 1.58 | 67.6 | 0.87 | 0.50 | 0.65 |
| natural: NA-SMC [TNg] | 1.58 | 65.7 | 0.85 | 0.53 | 0.60 |
| natural: NA-SMC [TN] | 1.82 | 67.0 | 0.87 | 0.42 | 0.73 |
| natural: NA-fixed [TN] | 1.87 | 67.5 | 0.87 | 0.41 | 0.74 |
| natural: M13-SMC [TN] | 2.02 | 67.7 | 0.88 | 0.40 | 0.76 |
| natural: M* [TNg] | 2.05 | 64.4 | 0.83 | 0.49 | 0.61 |
| natural: NA-myopic [TNg] | 2.12 | 62.4 | 0.80 | 0.50 | 0.57 |
| natural: M* [TN] | 2.13 | 66.5 | 0.86 | 0.39 | 0.74 |
| NA-SMC [TNg] | 2.29 | 65.1 | 0.94 | 0.48 | 0.53 |
| natural: NA-myopic [TN] | 2.48 | 64.7 | 0.84 | 0.40 | 0.70 |
| natural: M* [TNse] | 2.71 | 63.9 | 0.86 | 0.60 | 0.46 |
| natural: A [TN] | 2.74 | 66.1 | 0.85 | 0.38 | 0.75 |
| M* [TNg] | 2.81 | 63.4 | 0.93 | 0.46 | 0.51 |
| natural: A [TNg] | 2.82 | 62.3 | 0.80 | 0.47 | 0.60 |
| M*+repair [TNg] | 2.85 | 63.2 | 0.93 | 0.46 | 0.51 |
| natural: M* [N0] | 3.26 | 56.9 | 0.73 | 0.45 | 0.52 |
