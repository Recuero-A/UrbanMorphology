# Data-quality report: KOR_SEL

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 10 |
| green_unmapped_or_absent | info | 1 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| KOR_SEL_e599_n8127 | medium|1975_2000 | True | train | 0.14 | 0.34 | buildings_incomplete_suspected |  |
| KOR_SEL_e602_n8123 | high|2000_2020 | True | train | 0.38 | 0.42 |  |  |
| KOR_SEL_e610_n8118 | medium|by1975 | True | val | 0.21 | 0.30 |  |  |
| KOR_SEL_e612_n8127 | high|1975_2000 | True | train | 0.27 | 0.40 |  |  |
| KOR_SEL_e614_n8122 | high|by1975 | True | train | 0.08 | 0.42 | buildings_incomplete_suspected |  |
| KOR_SEL_e615_n8107 | medium|by1975 | True | train | 0.05 | 0.37 | buildings_incomplete_suspected | green_unmapped_or_absent |
| KOR_SEL_e623_n8127 | low|1975_2000 | True | train | 0.08 | 0.22 | buildings_incomplete_suspected |  |
| KOR_SEL_e637_n8137 | low|1975_2000 | True | val | 0.01 | 0.20 | buildings_incomplete_suspected |  |
| KOR_SEL_e638_n8145 | high|by1975 | True | train | 0.04 | 0.50 | buildings_incomplete_suspected |  |
| KOR_SEL_e643_n8106 | low|by1975 | True | train | 0.16 | 0.24 |  |  |
| KOR_SEL_e644_n8124 | medium|by1975 | True | test | 0.08 | 0.31 | buildings_incomplete_suspected |  |
| KOR_SEL_e650_n8105 | low|2000_2020 | True | test | 0.03 | 0.17 | buildings_incomplete_suspected |  |
| KOR_SEL_e651_n8116 | high|by1975 | True | train | 0.11 | 0.40 | buildings_incomplete_suspected |  |
| KOR_SEL_e654_n8113 | low|by1975 | True | train | 0.11 | 0.24 | buildings_incomplete_suspected |  |
| KOR_SEL_e655_n8104 | medium|2000_2020 | True | train | 0.19 | 0.31 |  |  |
