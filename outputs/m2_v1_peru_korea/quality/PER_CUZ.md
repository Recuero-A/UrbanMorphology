# Data-quality report: PER_CUZ

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 11 |
| green_unmapped_or_absent | info | 4 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| PER_CUZ_e338_n16611 | low|2000_2020 | True | train | 0.11 | 0.24 | buildings_incomplete_suspected |  |
| PER_CUZ_e341_n16610 | medium|1975_2000 | True | train | 0.00 | 0.32 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_CUZ_e343_n16605 | low|by1975 | True | train | 0.02 | 0.25 | buildings_incomplete_suspected |  |
| PER_CUZ_e346_n16603 | low|1975_2000 | True | train | 0.02 | 0.21 | buildings_incomplete_suspected |  |
| PER_CUZ_e346_n16607 | high|by1975 | True | train | 0.22 | 0.47 | buildings_incomplete_suspected |  |
| PER_CUZ_e349_n16606 | high|by1975 | True | test | 0.09 | 0.45 | buildings_incomplete_suspected |  |
| PER_CUZ_e351_n16609 | medium|by1975 | True | train | 0.02 | 0.34 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_CUZ_e352_n16605 | low|by1975 | True | test | 0.05 | 0.20 | buildings_incomplete_suspected |  |
| PER_CUZ_e357_n16604 | low|1975_2000 | True | train | 0.07 | 0.16 | buildings_incomplete_suspected |  |
| PER_CUZ_e357_n16607 | medium|2000_2020 | True | train | 0.15 | 0.37 | buildings_incomplete_suspected |  |
| PER_CUZ_e362_n16604 | high|by1975 | True | train | 0.28 | 0.39 |  |  |
| PER_CUZ_e364_n16601 | medium|by1975 | True | val | 0.24 | 0.31 |  | green_unmapped_or_absent |
| PER_CUZ_e367_n16601 | medium|by1975 | True | val | 0.19 | 0.35 |  | green_unmapped_or_absent |
| PER_CUZ_e367_n16604 | low|2000_2020 | True | train | 0.05 | 0.17 | buildings_incomplete_suspected |  |
| PER_CUZ_e370_n16601 | low|2000_2020 | True | val | 0.10 | 0.20 |  |  |
