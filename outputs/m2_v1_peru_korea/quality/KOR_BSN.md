# Data-quality report: KOR_BSN

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 13 |
| no_buildings | warning | 2 |
| green_unmapped_or_absent | info | 6 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| KOR_BSN_e1008_n7603 | low|2000_2020 | True | test | 0.07 | 0.19 | buildings_incomplete_suspected |  |
| KOR_BSN_e946_n7582 | high|2000_2020 | True | train | 0.18 | 0.38 | buildings_incomplete_suspected |  |
| KOR_BSN_e953_n7584 | high|1975_2000 | True | train | 0.27 | 0.51 |  | green_unmapped_or_absent |
| KOR_BSN_e954_n7588 | medium|2000_2020 | True | train | 0.15 | 0.34 | buildings_incomplete_suspected |  |
| KOR_BSN_e957_n7590 | low|1975_2000 | True | train | 0.00 | 0.22 | buildings_incomplete_suspected |  |
| KOR_BSN_e965_n7591 | low|2000_2020 | True | train | 0.00 | 0.16 | no_buildings;buildings_incomplete_suspected |  |
| KOR_BSN_e967_n7610 | low|1975_2000 | True | train | 0.00 | 0.25 | no_buildings;buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_BSN_e971_n7596 | low|by1975 | True | train | 0.32 | 0.26 |  | green_unmapped_or_absent |
| KOR_BSN_e974_n7602 | high|by1975 | True | train | 0.04 | 0.39 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_BSN_e975_n7582 | medium|by1975 | True | train | 0.04 | 0.30 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_BSN_e982_n7585 | low|by1975 | True | train | 0.08 | 0.26 | buildings_incomplete_suspected |  |
| KOR_BSN_e982_n7597 | high|by1975 | True | train | 0.05 | 0.42 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_BSN_e982_n7609 | medium|1975_2000 | True | val | 0.14 | 0.32 | buildings_incomplete_suspected |  |
| KOR_BSN_e983_n7582 | medium|by1975 | True | val | 0.03 | 0.35 | buildings_incomplete_suspected |  |
| KOR_BSN_e998_n7605 | high|by1975 | True | test | 0.06 | 0.40 | buildings_incomplete_suspected |  |
