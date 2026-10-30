# Data-quality report: ESP_BCN

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 2 |
| green_unmapped_or_absent | info | 2 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| ESP_BCN_e832_n8951 | low|by1975 | True | train | 0.21 | 0.27 |  |  |
| ESP_BCN_e836_n8939 | high|by1975 | True | val | 0.24 | 0.49 | buildings_incomplete_suspected |  |
| ESP_BCN_e836_n8943 | medium|by1975 | True | test | 0.27 | 0.32 |  |  |
| ESP_BCN_e836_n8948 | high|by1975 | True | train | 0.49 | 0.40 |  |  |
| ESP_BCN_e836_n8959 | low|by1975 | True | train | 0.14 | 0.20 |  |  |
| ESP_BCN_e837_n8952 | high|by1975 | True | val | 0.57 | 0.44 |  |  |
| ESP_BCN_e839_n8945 | low|1975_2000 | True | test | 0.14 | 0.22 |  |  |
| ESP_BCN_e839_n8948 | high|by1975 | True | train | 0.56 | 0.43 |  |  |
| ESP_BCN_e839_n8959 | high|by1975 | True | train | 0.37 | 0.40 |  | green_unmapped_or_absent |
| ESP_BCN_e840_n8942 | high|by1975 | True | train | 0.04 | 0.51 | buildings_incomplete_suspected | green_unmapped_or_absent |
| ESP_BCN_e840_n8954 | medium|by1975 | True | train | 0.42 | 0.36 |  |  |
| ESP_BCN_e842_n8964 | low|2000_2020 | True | train | 0.08 | 0.15 |  |  |
| ESP_BCN_e843_n8956 | medium|by1975 | True | train | 0.42 | 0.38 |  |  |
| ESP_BCN_e844_n8953 | medium|by1975 | True | train | 0.34 | 0.34 |  |  |
| ESP_BCN_e847_n8952 | low|by1975 | True | train | 0.28 | 0.30 |  |  |
