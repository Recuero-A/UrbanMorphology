# Data-quality report: USA_CHI

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 1 |
| green_unmapped_or_absent | info | 3 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| USA_CHI_e852_n9055 | low|by1975 | True | train | 0.19 | 0.25 |  |  |
| USA_CHI_e852_n9070 | medium|by1975 | True | train | 0.32 | 0.35 |  |  |
| USA_CHI_e858_n9075 | medium|by1975 | True | train | 0.31 | 0.37 |  | green_unmapped_or_absent |
| USA_CHI_e861_n9063 | medium|by1975 | True | train | 0.28 | 0.39 |  |  |
| USA_CHI_e863_n9046 | medium|by1975 | True | test | 0.24 | 0.39 |  |  |
| USA_CHI_e866_n9027 | low|by1975 | True | train | 0.17 | 0.26 |  |  |
| USA_CHI_e869_n9015 | low|by1975 | True | val | 0.12 | 0.22 |  |  |
| USA_CHI_e869_n9067 | high|by1975 | True | train | 0.38 | 0.39 |  |  |
| USA_CHI_e870_n9077 | medium|by1975 | True | train | 0.33 | 0.32 |  |  |
| USA_CHI_e872_n9068 | high|by1975 | True | train | 0.47 | 0.45 |  |  |
| USA_CHI_e874_n9047 | low|2000_2020 | True | train | 0.09 | 0.16 |  |  |
| USA_CHI_e875_n9027 | low|by1975 | True | val | 0.21 | 0.29 |  |  |
| USA_CHI_e884_n9023 | medium|by1975 | True | train | 0.28 | 0.38 |  | green_unmapped_or_absent |
| USA_CHI_e888_n9014 | low|1975_2000 | True | test | 0.12 | 0.16 |  |  |
| USA_CHI_e889_n9022 | low|by1975 | True | train | 0.04 | 0.18 | buildings_incomplete_suspected | green_unmapped_or_absent |
