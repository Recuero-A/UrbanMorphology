# M18 D11: leastcostpath on a routing task with shared anchors

Anchors: the 12 farthest-point anchors of each leastcostpath site run (seeds 1-3), 66 unordered pairs per seed. Network routes: shortest path by length after joining each anchor to the nearest street point (<= 60 m). All routes smoothed alike (30 m resampling, 10 m pieces) and measured on the bare-earth proxy. Script `scripts/m18_routing.py`.

Unsmoothed = the same measures on the native polylines (sensitivity: 30 m resampling turns switchbacks into chords up the slope).

| Site | Route source | served | detour p50 | climb m per km p50 | route grade p90 (median) | share > 15 % | unsmoothed grade p90 | unsmoothed share > 15 % | longest climb p50 (m) | detour / LCP | climb / LCP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| machu_picchu | leastcostpath route (ext) | 0.74 | 1.60 | 178 | 49.3 | 0.483 | 19.0 | 0.259 | 455 | nan | nan |
| machu_picchu | real (OSM vehicular) | 0.01 | 1.39 | 96 | 34.1 | 0.231 | 48.6 | 0.284 | 93 | 1.17 | 1.09 |
| machu_picchu | NA-SMC [TN] | 0.24 | 2.03 | 247 | 48.4 | 0.537 | 52.0 | 0.548 | 238 | 1.38 | 1.88 |
| machu_picchu | NA-SMC [TN] + terrain potentials | 0.43 | 2.08 | 276 | 59.9 | 0.599 | 62.7 | 0.619 | 296 | 1.43 | 2.04 |
| machu_picchu | NA-SMC [TNg] | 0.32 | 1.32 | 249 | 72.0 | 0.700 | 73.1 | 0.699 | 313 | 0.82 | 1.42 |
| machu_picchu | NA-SMC [TNg] + terrain potentials | 0.04 | 1.16 | 85 | 43.1 | 0.255 | 50.6 | 0.230 | 104 | 1.01 | 1.17 |
| machu_picchu | M* | 0.00 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| machu_picchu | P | 0.00 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| machu_picchu | citygen P&M (ext) | 0.00 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| machu_picchu | leastcostpath merged network (ext) | 0.74 | 1.61 | 180 | 47.9 | 0.466 | 20.4 | 0.290 | 407 | 1.00 | 1.00 |
| petra | leastcostpath route (ext) | 0.83 | 1.25 | 45 | 13.5 | 0.118 | 13.5 | 0.085 | 196 | nan | nan |
| petra | real (OSM vehicular) | 0.05 | 1.31 | 41 | 11.6 | 0.052 | 12.3 | 0.051 | 266 | 1.03 | 1.27 |
| petra | NA-SMC [TN] | 0.66 | 1.44 | 88 | 22.7 | 0.265 | 22.7 | 0.262 | 269 | 1.17 | 1.75 |
| petra | NA-SMC [TN] + terrain potentials | 0.74 | 1.50 | 86 | 22.2 | 0.308 | 21.9 | 0.306 | 231 | 1.18 | 1.92 |
| petra | NA-SMC [TNg] | 0.51 | 1.38 | 79 | 20.8 | 0.259 | 21.0 | 0.268 | 182 | 1.10 | 1.69 |
| petra | NA-SMC [TNg] + terrain potentials | 0.32 | 1.80 | 66 | 14.5 | 0.087 | 14.5 | 0.090 | 161 | 1.36 | 1.62 |
| petra | M* | 0.10 | 1.47 | 66 | 14.0 | 0.146 | 14.8 | 0.148 | 211 | 1.20 | 1.89 |
| petra | P | 0.01 | 1.57 | 65 | 13.8 | 0.068 | 14.2 | 0.072 | 213 | 1.23 | 1.90 |
| petra | citygen P&M (ext) | 0.16 | 1.32 | 57 | 16.0 | 0.143 | 15.8 | 0.139 | 165 | 1.10 | 1.56 |
| petra | leastcostpath merged network (ext) | 0.83 | 1.25 | 49 | 14.0 | 0.127 | 14.1 | 0.104 | 238 | 1.00 | 1.02 |
