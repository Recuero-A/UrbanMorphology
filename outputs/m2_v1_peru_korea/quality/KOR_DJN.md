# Data-quality report: KOR_DJN

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 13 |
| no_buildings | warning | 4 |
| green_unmapped_or_absent | info | 4 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| KOR_DJN_e682_n7861 | low|1975_2000 | True | val | 0.01 | 0.21 | buildings_incomplete_suspected |  |
| KOR_DJN_e685_n7847 | medium|2000_2020 | True | test | 0.10 | 0.32 | buildings_incomplete_suspected |  |
| KOR_DJN_e685_n7865 | high|2000_2020 | True | train | 0.15 | 0.37 | buildings_incomplete_suspected |  |
| KOR_DJN_e690_n7853 | medium|1975_2000 | True | train | 0.15 | 0.28 |  |  |
| KOR_DJN_e692_n7858 | medium|by1975 | True | test | 0.10 | 0.35 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_DJN_e693_n7848 | low|1975_2000 | True | train | 0.05 | 0.27 | buildings_incomplete_suspected |  |
| KOR_DJN_e695_n7853 | high|by1975 | True | train | 0.09 | 0.39 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_DJN_e696_n7857 | low|by1975 | True | train | 0.00 | 0.25 | no_buildings;buildings_incomplete_suspected |  |
| KOR_DJN_e697_n7864 | medium|by1975 | True | train | 0.00 | 0.35 | no_buildings;buildings_incomplete_suspected |  |
| KOR_DJN_e697_n7869 | medium|1975_2000 | True | val | 0.13 | 0.31 | buildings_incomplete_suspected |  |
| KOR_DJN_e697_n7873 | low|2000_2020 | True | train | 0.14 | 0.19 |  |  |
| KOR_DJN_e700_n7875 | high|by1975 | True | train | 0.08 | 0.38 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_DJN_e702_n7862 | high|1975_2000 | True | train | 0.01 | 0.40 | buildings_incomplete_suspected |  |
| KOR_DJN_e704_n7855 | high|by1975 | True | train | 0.00 | 0.50 | no_buildings;buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_DJN_e706_n7844 | low|2000_2020 | True | train | 0.00 | 0.18 | no_buildings;buildings_incomplete_suspected |  |
