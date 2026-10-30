# Data-quality report: ESP_MAD

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
| ESP_MAD_e847_n8745 | low|2000_2020 | True | train | 0.14 | 0.15 |  |  |
| ESP_MAD_e850_n8727 | medium|1975_2000 | True | train | 0.21 | 0.32 |  |  |
| ESP_MAD_e858_n8741 | high|by1975 | True | train | 0.45 | 0.41 |  |  |
| ESP_MAD_e858_n8754 | low|by1975 | True | test | 0.20 | 0.30 |  |  |
| ESP_MAD_e860_n8751 | low|by1975 | True | train | 0.14 | 0.21 |  |  |
| ESP_MAD_e862_n8734 | low|by1975 | True | train | 0.24 | 0.27 |  |  |
| ESP_MAD_e862_n8737 | low|1975_2000 | True | train | 0.12 | 0.18 |  |  |
| ESP_MAD_e862_n8745 | medium|by1975 | True | train | 0.26 | 0.33 |  |  |
| ESP_MAD_e863_n8758 | low|1975_2000 | True | test | 0.06 | 0.19 | buildings_incomplete_suspected |  |
| ESP_MAD_e868_n8740 | medium|by1975 | True | val | 0.31 | 0.37 |  |  |
| ESP_MAD_e869_n8743 | medium|by1975 | True | test | 0.36 | 0.39 |  |  |
| ESP_MAD_e877_n8728 | medium|2000_2020 | True | val | 0.32 | 0.38 |  |  |
| ESP_MAD_e878_n8751 | low|2000_2020 | True | train | 0.09 | 0.18 |  |  |
| ESP_MAD_e878_n8754 | high|1975_2000 | True | train | 0.52 | 0.55 |  | green_unmapped_or_absent |
| ESP_MAD_e881_n8746 | high|by1975 | True | train | 0.24 | 0.43 |  |  |
