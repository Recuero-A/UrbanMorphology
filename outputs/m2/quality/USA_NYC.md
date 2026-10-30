# Data-quality report: USA_NYC

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
| USA_NYC_e1106_n8766 | medium|1975_2000 | True | train | 0.18 | 0.31 |  |  |
| USA_NYC_e1110_n8763 | low|by1975 | True | train | 0.15 | 0.26 |  |  |
| USA_NYC_e1117_n8782 | medium|by1975 | True | train | 0.21 | 0.37 |  |  |
| USA_NYC_e1120_n8782 | medium|by1975 | True | train | 0.21 | 0.36 |  |  |
| USA_NYC_e1123_n8779 | low|1975_2000 | True | train | 0.11 | 0.23 | buildings_incomplete_suspected |  |
| USA_NYC_e1124_n8775 | low|by1975 | True | test | 0.18 | 0.30 |  |  |
| USA_NYC_e1143_n8779 | high|by1975 | True | train | 0.35 | 0.45 |  | green_unmapped_or_absent |
| USA_NYC_e1147_n8798 | medium|by1975 | True | train | 0.28 | 0.31 |  |  |
| USA_NYC_e1149_n8784 | high|by1975 | True | val | 0.31 | 0.39 |  | green_unmapped_or_absent |
| USA_NYC_e1150_n8801 | medium|by1975 | True | train | 0.31 | 0.34 |  |  |
| USA_NYC_e1155_n8796 | high|by1975 | True | train | 0.32 | 0.41 |  |  |
| USA_NYC_e1156_n8844 | low|2000_2020 | True | train | 0.11 | 0.18 |  |  |
| USA_NYC_e1164_n8803 | low|by1975 | True | val | 0.20 | 0.27 |  |  |
| USA_NYC_e1171_n8804 | low|by1975 | True | test | 0.14 | 0.26 |  |  |
| USA_NYC_e1182_n8800 | medium|by1975 | True | train | 0.25 | 0.35 |  | green_unmapped_or_absent |
