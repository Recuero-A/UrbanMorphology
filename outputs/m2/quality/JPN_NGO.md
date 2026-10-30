# Data-quality report: JPN_NGO

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 1 |
| green_unmapped_or_absent | info | 1 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| JPN_NGO_e1296_n7591 | low|by1975 | True | train | 0.11 | 0.20 |  |  |
| JPN_NGO_e1299_n7593 | low|by1975 | True | train | 0.14 | 0.22 |  |  |
| JPN_NGO_e1301_n7588 | low|2000_2020 | True | train | 0.08 | 0.16 | buildings_incomplete_suspected |  |
| JPN_NGO_e1304_n7592 | high|by1975 | True | test | 0.30 | 0.46 |  |  |
| JPN_NGO_e1307_n7604 | high|by1975 | True | train | 0.37 | 0.47 |  |  |
| JPN_NGO_e1312_n7589 | high|by1975 | True | train | 0.28 | 0.43 |  |  |
| JPN_NGO_e1319_n7612 | high|by1975 | True | val | 0.32 | 0.41 |  | green_unmapped_or_absent |
| JPN_NGO_e1320_n7608 | high|by1975 | True | train | 0.31 | 0.41 |  |  |
| JPN_NGO_e1321_n7604 | medium|by1975 | True | train | 0.30 | 0.39 |  |  |
| JPN_NGO_e1323_n7579 | high|1975_2000 | True | test | 0.32 | 0.46 |  |  |
| JPN_NGO_e1330_n7609 | medium|by1975 | True | train | 0.29 | 0.39 |  |  |
| JPN_NGO_e1333_n7590 | medium|1975_2000 | True | train | 0.21 | 0.35 |  |  |
| JPN_NGO_e1333_n7604 | high|by1975 | True | val | 0.27 | 0.40 |  |  |
| JPN_NGO_e1334_n7609 | medium|by1975 | True | train | 0.21 | 0.34 |  |  |
| JPN_NGO_e1334_n7621 | low|1975_2000 | True | train | 0.21 | 0.29 |  |  |
