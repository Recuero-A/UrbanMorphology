# Data-quality report: USA_LAX

Planned primaries: 15  
Processed tiles (incl. reserves): 15  
Usable in dataset: 15

Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; **info** = not a defect (an empty green channel may be real or unmapped).

| flag | severity | tiles |
|---|---|---|
| buildings_incomplete_suspected | warning | 2 |
| green_unmapped_or_absent | info | 7 |

## Tiles

| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |
|---|---|---|---|---|---|---|---|
| USA_LAX_e680_n7395 | medium|1975_2000 | True | train | 0.23 | 0.33 |  |  |
| USA_LAX_e691_n7410 | medium|2000_2020 | True | train | 0.25 | 0.33 |  |  |
| USA_LAX_e696_n7382 | low|1975_2000 | True | test | 0.16 | 0.25 |  | green_unmapped_or_absent |
| USA_LAX_e703_n7412 | low|1975_2000 | True | train | 0.09 | 0.19 | buildings_incomplete_suspected |  |
| USA_LAX_e712_n7379 | medium|by1975 | True | train | 0.22 | 0.34 |  |  |
| USA_LAX_e715_n7386 | medium|by1975 | True | train | 0.26 | 0.38 |  |  |
| USA_LAX_e716_n7380 | low|2000_2020 | True | train | 0.13 | 0.16 |  |  |
| USA_LAX_e718_n7392 | high|by1975 | True | train | 0.29 | 0.45 |  | green_unmapped_or_absent |
| USA_LAX_e721_n7381 | medium|by1975 | True | train | 0.28 | 0.34 |  |  |
| USA_LAX_e727_n7358 | high|by1975 | True | val | 0.40 | 0.42 |  | green_unmapped_or_absent |
| USA_LAX_e729_n7398 | medium|by1975 | True | train | 0.25 | 0.37 |  | green_unmapped_or_absent |
| USA_LAX_e738_n7366 | high|by1975 | True | train | 0.29 | 0.42 |  | green_unmapped_or_absent |
| USA_LAX_e745_n7345 | low|by1975 | True | test | 0.22 | 0.26 |  | green_unmapped_or_absent |
| USA_LAX_e750_n7290 | high|1975_2000 | True | train | 0.00 | 0.43 | buildings_incomplete_suspected | green_unmapped_or_absent |
| USA_LAX_e758_n7370 | low|by1975 | True | val | 0.16 | 0.25 |  |  |
