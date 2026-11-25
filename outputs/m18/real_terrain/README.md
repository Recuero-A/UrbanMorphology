# M18: how real street networks adapt to terrain

Script `scripts/m18_real_terrain_study.py`; figure `fig_real_terrain.png`. Tiles: {"Barcelona": 41, "Madrid": 72, "Valencia": 33, "Chicago": 73, "Los Angeles": 193, "New York": 73, "Alcoi": 20, "Tarragona": 25, "Sacramento": 25, "Santa Barbara": 25}. Test split never loaded.

| Terrain slope (%) | street km | land km² | street km/km² | grade p50 / p90 / p99 (%) | > 10 / 15 / 20 % (share of length) | grade ÷ slope (median) | contour index (mean) |
|---|---|---|---|---|---|---|---|
| 0–5 | 1494 | 96.8 | 15.4 | 0.6 / 3.4 / 7.4 | 0.001 / 0.000 / 0.000 | 0.60 | 0.62 |
| 5–10 | 348 | 22.2 | 15.7 | 3.8 / 8.2 / 12.1 | 0.031 / 0.004 / 0.002 | 0.55 | 0.57 |
| 10–20 | 215 | 16.2 | 13.3 | 6.1 / 14.1 / 21.8 | 0.289 / 0.072 / 0.015 | 0.44 | 0.50 |
| 20–30 | 57 | 7.2 | 7.9 | 7.5 / 16.9 / 27.6 | 0.364 / 0.159 / 0.053 | 0.31 | 0.37 |
| > 30 | 34 | 8.4 | 4.0 | 7.9 / 16.9 / 31.4 | 0.357 / 0.147 / 0.055 | 0.21 | 0.24 |

All streets: grade p50 1.0 %, p90 7.3 %, p99 16.2 %; share > 10 / 15 / 20 %: 0.050 / 0.015 / 0.004. Land slope p50 2.8 %, p90 20.6 %.
