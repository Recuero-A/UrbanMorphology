# Data-quality report: ESP_MAD

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 2 |
| no_roads | warning | 1 |
| green_unmapped_or_absent | info | 2 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| ESP_MAD_e844_n8730 | low|by1975 | True | train | 0.13 | 0.23 |  |  |
| ESP_MAD_e846_n8750 | low|1975_2000 | True | train | 0.13 | 0.26 | buildings_incomplete_suspected |  |
| ESP_MAD_e849_n8730 | medium|by1975 | True | train | 0.19 | 0.31 |  |  |
| ESP_MAD_e852_n8727 | medium|2000_2020 | True | train | 0.22 | 0.29 |  |  |
| ESP_MAD_e856_n8732 | medium|by1975 | True | train | 0.29 | 0.35 |  |  |
| ESP_MAD_e858_n8750 | low|by1975 | True | test | 0.15 | 0.19 |  |  |
| ESP_MAD_e861_n8756 | low|2000_2020 | True | val | 0.06 | 0.17 | buildings_incomplete_suspected |  |
| ESP_MAD_e865_n8756 | low|2000_2020 | True | train | 0.16 | 0.27 |  |  |
| ESP_MAD_e866_n8735 | high|by1975 | True | train | 0.42 | 0.39 |  |  |
| ESP_MAD_e869_n8741 | high|by1975 | True | val | 0.38 | 0.40 |  |  |
| ESP_MAD_e871_n8747 | medium|by1975 | True | train | 0.23 | 0.28 |  | green_unmapped_or_absent |
| ESP_MAD_e873_n8744 | low|1975_2000 | True | train | 0.11 | 0.18 |  |  |
| ESP_MAD_e877_n8728 | high|2000_2020 | True | train | 0.32 | 0.38 |  |  |
| ESP_MAD_e878_n8754 | high|1975_2000 | True | test | 0.52 | 0.55 |  | green_unmapped_or_absent |
| ESP_MAD_e882_n8755 | medium|1975_2000 | True | train | 0.17 | 0.34 | no_roads |  |
