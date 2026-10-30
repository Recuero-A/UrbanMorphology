# Data-quality report: ESP_BCN

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 3 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| ESP_BCN_e833_n8937 | high|by1975 | True | train | 0.28 | 0.53 |  |  |
| ESP_BCN_e837_n8944 | low|by1975 | True | val | 0.09 | 0.21 | buildings_incomplete_suspected |  |
| ESP_BCN_e837_n8950 | high|by1975 | True | val | 0.60 | 0.42 |  |  |
| ESP_BCN_e837_n8955 | low|by1975 | True | test | 0.25 | 0.25 |  |  |
| ESP_BCN_e837_n8960 | low|1975_2000 | True | train | 0.07 | 0.17 | buildings_incomplete_suspected |  |
| ESP_BCN_e838_n8941 | high|by1975 | True | val | 0.07 | 0.39 | buildings_incomplete_suspected |  |
| ESP_BCN_e840_n8957 | medium|by1975 | True | train | 0.25 | 0.34 |  |  |
| ESP_BCN_e841_n8951 | high|by1975 | True | train | 0.51 | 0.42 |  |  |
| ESP_BCN_e841_n8961 | medium|by1975 | True | train | 0.25 | 0.31 |  |  |
| ESP_BCN_e842_n8964 | low|2000_2020 | True | test | 0.08 | 0.15 |  |  |
| ESP_BCN_e843_n8948 | high|by1975 | True | train | 0.38 | 0.38 |  |  |
| ESP_BCN_e844_n8952 | high|by1975 | True | train | 0.38 | 0.46 |  |  |
| ESP_BCN_e845_n8955 | medium|by1975 | True | train | 0.30 | 0.34 |  |  |
| ESP_BCN_e845_n8959 | high|by1975 | True | train | 0.30 | 0.49 |  |  |
| ESP_BCN_e848_n8954 | medium|by1975 | True | train | 0.31 | 0.33 |  |  |
