# Data-quality report: ESP_VLC

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 3 |
| excessive_channel_overlap | warning | 1 |
| green_unmapped_or_absent | info | 1 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| ESP_VLC_e1410_n8545 | medium|by1975 | True | test | 0.26 | 0.30 |  |  |
| ESP_VLC_e1412_n8542 | low|2000_2020 | True | test | 0.15 | 0.17 |  |  |
| ESP_VLC_e1413_n8534 | medium|by1975 | True | train | 0.24 | 0.34 |  |  |
| ESP_VLC_e1413_n8537 | medium|by1975 | True | train | 0.38 | 0.31 |  |  |
| ESP_VLC_e1416_n8544 | high|by1975 | True | train | 0.45 | 0.37 |  |  |
| ESP_VLC_e1417_n8533 | low|1975_2000 | True | train | 0.19 | 0.26 |  |  |
| ESP_VLC_e1417_n8536 | low|by1975 | True | train | 0.32 | 0.27 |  |  |
| ESP_VLC_e1417_n8540 | high|by1975 | True | val | 0.62 | 0.46 | excessive_channel_overlap |  |
| ESP_VLC_e1420_n8532 | medium|1975_2000 | True | train | 0.00 | 0.28 | buildings_incomplete_suspected |  |
| ESP_VLC_e1420_n8537 | low|by1975 | True | train | 0.15 | 0.19 |  |  |
| ESP_VLC_e1421_n8541 | low|by1975 | True | val | 0.24 | 0.26 |  |  |
| ESP_VLC_e1423_n8535 | low|1975_2000 | True | train | 0.05 | 0.17 | buildings_incomplete_suspected |  |
| ESP_VLC_e1425_n8538 | medium|by1975 | True | train | 0.33 | 0.33 |  |  |
| ESP_VLC_e1425_n8541 | medium|by1975 | True | train | 0.31 | 0.33 |  |  |
| ESP_VLC_e1427_n8530 | high|1975_2000 | True | val | 0.00 | 0.77 | buildings_incomplete_suspected | green_unmapped_or_absent |
