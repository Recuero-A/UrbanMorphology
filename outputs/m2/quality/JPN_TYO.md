# Data-quality report: JPN_TYO

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 4 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| JPN_TYO_e729_n7723 | medium|by1975 | True | train | 0.31 | 0.33 |  |  |
| JPN_TYO_e731_n7714 | medium|by1975 | True | train | 0.23 | 0.34 |  |  |
| JPN_TYO_e734_n7718 | high|by1975 | True | val | 0.38 | 0.45 |  |  |
| JPN_TYO_e743_n7711 | high|by1975 | True | test | 0.46 | 0.45 |  |  |
| JPN_TYO_e751_n7688 | high|by1975 | True | train | 0.40 | 0.44 |  |  |
| JPN_TYO_e756_n7731 | low|by1975 | True | train | 0.20 | 0.23 |  |  |
| JPN_TYO_e757_n7704 | low|1975_2000 | True | train | 0.05 | 0.18 | buildings_incomplete_suspected |  |
| JPN_TYO_e757_n7718 | medium|by1975 | True | train | 0.44 | 0.38 |  |  |
| JPN_TYO_e758_n7692 | low|2000_2020 | True | train | 0.08 | 0.16 |  |  |
| JPN_TYO_e761_n7707 | low|by1975 | True | train | 0.15 | 0.21 |  |  |
| JPN_TYO_e762_n7720 | high|by1975 | True | train | 0.45 | 0.46 |  |  |
| JPN_TYO_e763_n7695 | medium|1975_2000 | True | test | 0.16 | 0.36 | buildings_incomplete_suspected |  |
| JPN_TYO_e764_n7739 | high|by1975 | True | val | 0.28 | 0.43 |  |  |
| JPN_TYO_e776_n7712 | high|by1975 | True | train | 0.05 | 0.44 | buildings_incomplete_suspected |  |
| JPN_TYO_e782_n7714 | high|by1975 | True | train | 0.06 | 0.43 | buildings_incomplete_suspected |  |
