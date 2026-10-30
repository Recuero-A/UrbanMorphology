# Data-quality report: ESP_VLC

Planned primaries: 15  
Processed tiles (incl. reserves): 16  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 3 |
| green_unmapped_or_absent | info | 1 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| ESP_VLC_e1410_n8545 | low|by1975 | True | test | 0.26 | 0.30 |  |  |
| ESP_VLC_e1412_n8536 | high|by1975 | True | train | 0.40 | 0.52 |  |  |
| ESP_VLC_e1413_n8540 | low|by1975 | True | val | 0.27 | 0.24 |  |  |
| ESP_VLC_e1413_n8544 | medium|by1975 | True | test | 0.31 | 0.34 |  |  |
| ESP_VLC_e1415_n8534 | low|by1975 | True | train | 0.32 | 0.26 |  |  |
| ESP_VLC_e1416_n8543 | medium|by1975 | True | train | 0.28 | 0.32 |  |  |
| ESP_VLC_e1418_n8536 | medium|by1975 | True | train | 0.39 | 0.36 |  |  |
| ESP_VLC_e1418_n8539 | high|by1975 | True | train | 0.59 | 0.42 |  |  |
| ESP_VLC_e1419_n8530 | low|by1975 | True | train | 0.25 | 0.25 |  |  |
| ESP_VLC_e1419_n8542 | low|by1975 | True | train | 0.37 | 0.30 |  |  |
| ESP_VLC_e1422_n8532 | low|2000_2020 | True | train | 0.07 | 0.17 | buildings_incomplete_suspected |  |
| ESP_VLC_e1422_n8541 | low|1975_2000 | True | train | 0.24 | 0.26 |  |  |
| ESP_VLC_e1425_n8534 | low|1975_2000 | True | train | 0.04 | 0.19 | buildings_incomplete_suspected |  |
| ESP_VLC_e1425_n8538 | medium|by1975 | True | train | 0.33 | 0.33 |  |  |
| ESP_VLC_e1425_n8541 | medium|by1975 | True | not_in_dataset | 0.31 | 0.33 |  |  |
| ESP_VLC_e1427_n8530 | high|1975_2000 | True | val | 0.00 | 0.77 | buildings_incomplete_suspected | green_unmapped_or_absent |
