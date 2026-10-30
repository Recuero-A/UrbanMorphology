# Data-quality report: PER_AQP

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 13 |
| green_unmapped_or_absent | info | 2 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| PER_AQP_e434_n15996 | low|1975_2000 | True | train | 0.07 | 0.19 | buildings_incomplete_suspected |  |
| PER_AQP_e435_n16001 | low|2000_2020 | True | train | 0.00 | 0.22 | buildings_incomplete_suspected |  |
| PER_AQP_e435_n16004 | medium|2000_2020 | True | val | 0.01 | 0.33 | buildings_incomplete_suspected |  |
| PER_AQP_e437_n15987 | low|2000_2020 | True | val | 0.05 | 0.20 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_AQP_e437_n15998 | medium|2000_2020 | True | train | 0.07 | 0.30 | buildings_incomplete_suspected |  |
| PER_AQP_e440_n15988 | high|2000_2020 | True | train | 0.04 | 0.39 | buildings_incomplete_suspected |  |
| PER_AQP_e443_n15984 | low|1975_2000 | True | train | 0.15 | 0.26 |  |  |
| PER_AQP_e444_n15996 | low|2000_2020 | True | test | 0.07 | 0.23 | buildings_incomplete_suspected |  |
| PER_AQP_e446_n15987 | medium|1975_2000 | True | train | 0.24 | 0.30 |  |  |
| PER_AQP_e449_n15977 | high|by1975 | True | test | 0.03 | 0.38 | buildings_incomplete_suspected |  |
| PER_AQP_e450_n15985 | medium|1975_2000 | True | train | 0.14 | 0.29 | buildings_incomplete_suspected |  |
| PER_AQP_e450_n15991 | high|by1975 | True | train | 0.04 | 0.45 | buildings_incomplete_suspected |  |
| PER_AQP_e450_n15997 | medium|by1975 | True | train | 0.13 | 0.28 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_AQP_e452_n15975 | low|2000_2020 | True | train | 0.01 | 0.20 | buildings_incomplete_suspected |  |
| PER_AQP_e455_n15981 | high|1975_2000 | True | train | 0.00 | 0.42 | buildings_incomplete_suspected |  |
