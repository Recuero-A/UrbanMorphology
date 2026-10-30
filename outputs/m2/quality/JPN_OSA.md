# Data-quality report: JPN_OSA

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 3 |
| green_unmapped_or_absent | info | 4 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| JPN_OSA_e1047_n7488 | high|1975_2000 | True | train | 0.00 | 0.47 | buildings_incomplete_suspected | green_unmapped_or_absent |
| JPN_OSA_e1048_n7492 | medium|by1975 | True | train | 0.38 | 0.31 |  | green_unmapped_or_absent |
| JPN_OSA_e1050_n7483 | low|2000_2020 | True | train | 0.07 | 0.15 | buildings_incomplete_suspected | green_unmapped_or_absent |
| JPN_OSA_e1052_n7487 | low|1975_2000 | True | train | 0.04 | 0.19 | buildings_incomplete_suspected |  |
| JPN_OSA_e1055_n7488 | low|by1975 | True | val | 0.15 | 0.23 |  |  |
| JPN_OSA_e1057_n7492 | high|by1975 | True | train | 0.38 | 0.42 |  |  |
| JPN_OSA_e1062_n7485 | medium|by1975 | True | val | 0.23 | 0.32 |  | green_unmapped_or_absent |
| JPN_OSA_e1063_n7501 | medium|by1975 | True | train | 0.28 | 0.37 |  |  |
| JPN_OSA_e1064_n7490 | medium|by1975 | True | test | 0.23 | 0.33 |  |  |
| JPN_OSA_e1065_n7482 | high|by1975 | True | val | 0.39 | 0.42 |  |  |
| JPN_OSA_e1066_n7479 | high|by1975 | True | train | 0.36 | 0.42 |  |  |
| JPN_OSA_e1066_n7498 | low|by1975 | True | train | 0.26 | 0.31 |  |  |
| JPN_OSA_e1071_n7488 | high|by1975 | True | train | 0.48 | 0.44 |  |  |
| JPN_OSA_e1072_n7511 | high|by1975 | True | test | 0.34 | 0.41 |  |  |
| JPN_OSA_e1073_n7485 | high|by1975 | True | train | 0.32 | 0.43 |  |  |
