# Data-quality report: PER_LIM

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 14 |
| no_buildings | warning | 3 |
| green_unmapped_or_absent | info | 6 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| PER_LIM_e515_n16980 | low|2000_2020 | True | train | 0.00 | 0.17 | no_buildings;buildings_incomplete_suspected |  |
| PER_LIM_e527_n16976 | medium|1975_2000 | True | train | 0.00 | 0.33 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_LIM_e531_n16966 | low|2000_2020 | True | train | 0.00 | 0.16 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_LIM_e534_n16972 | high|2000_2020 | True | train | 0.00 | 0.38 | no_buildings;buildings_incomplete_suspected |  |
| PER_LIM_e537_n16943 | medium|by1975 | True | train | 0.09 | 0.30 | buildings_incomplete_suspected |  |
| PER_LIM_e543_n16936 | low|1975_2000 | True | train | 0.00 | 0.22 | buildings_incomplete_suspected |  |
| PER_LIM_e546_n16925 | high|by1975 | True | train | 0.19 | 0.50 | buildings_incomplete_suspected |  |
| PER_LIM_e547_n16958 | medium|by1975 | True | train | 0.00 | 0.32 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_LIM_e548_n16912 | high|by1975 | True | train | 0.12 | 0.40 | buildings_incomplete_suspected |  |
| PER_LIM_e551_n16925 | high|by1975 | True | train | 0.02 | 0.44 | buildings_incomplete_suspected |  |
| PER_LIM_e559_n16956 | low|1975_2000 | True | test | 0.00 | 0.21 | no_buildings;buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_LIM_e561_n16895 | high|by1975 | True | train | 0.06 | 0.38 | buildings_incomplete_suspected |  |
| PER_LIM_e573_n16939 | low|by1975 | True | val | 0.01 | 0.18 | buildings_incomplete_suspected | green_unmapped_or_absent |
| PER_LIM_e573_n16946 | medium|2000_2020 | True | test | 0.19 | 0.32 |  | green_unmapped_or_absent |
| PER_LIM_e585_n16872 | high|1975_2000 | True | val | 0.01 | 0.39 | buildings_incomplete_suspected |  |
