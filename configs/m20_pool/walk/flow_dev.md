# dev: flow per stratum (D1)

| stratum | listed | terrain_evaluated | no_dem | eligible | eligible_B | osm_tested | failed_km | failed_coverage | failed_both | block_cap | band_A_full | band_B_third | accepted | accepted_B | accepted_as_fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1:city | 90 | 90 | 0 | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 |
| R1:town | 592 | 592 | 0 | 38 | 5 | 5 | 1 | 0 | 0 | 0 | 19 | 0 | 4 | 2 | 1 |
| R2:city | 215 | 215 | 0 | 8 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 5 | 3 | 0 | 0 |
| R2:town | 1342 | 384 | 0 | 29 | 6 | 8 | 6 | 0 | 0 | 0 | 21 | 21 | 2 | 1 | 0 |
| R3:city | 488 | 488 | 0 | 8 | 0 | 6 | 0 | 3 | 0 | 0 | 0 | 2 | 3 | 0 | 0 |
| R3:town | 3308 | 450 | 0 | 27 | 6 | 26 | 15 | 1 | 9 | 1 | 0 | 0 | 1 | 1 | 0 |
| R4:city | 224 | 224 | 3 | 8 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| R4:town | 1223 | 342 | 4 | 26 | 6 | 4 | 1 | 0 | 0 | 0 | 10 | 0 | 3 | 2 | 0 |
| village | 25441 | 144 | 1 | 18 | 2 | 16 | 6 | 0 | 4 | 1 | 0 | 0 | 6 | 1 | 0 |

Gate failures: km = under the vehicular minimum (15 km; villages 4 km); coverage = under 90 % of built-up cells within 100 m of any way; both = both.

# dev: balance, terrain-eligible settlements

| group | n | slope_med | band_B_share | area_km2_med | pop_med | vehicular_km_med |
|---|---|---|---|---|---|---|
| accepted | 28 | 16.81 | 0.29 | 3.50 | 38127.00 | 35.35 |
| eligible, not accepted | 158 | 14.98 | 0.11 | 3.00 | 8451.00 | 7.51 |

# dev: OSM gate pass rate by country (selection on mapping quality)

| iso3 | tested | passed | pass_rate |
|---|---|---|---|
| CHN | 41 | 4 | 0.10 |
| ITA | 7 | 6 | 0.86 |
| GTM | 5 | 1 | 0.20 |
| MEX | 4 | 3 | 0.75 |
| COL | 2 | 1 | 0.50 |
| TUR | 2 | 1 | 0.50 |
| BRA | 2 | 2 | 1.00 |
| ALB | 1 | 1 | 1.00 |
| TUN | 1 | 1 | 1.00 |
| SYR | 1 | 1 | 1.00 |
| SRB | 1 | 1 | 1.00 |
| JOR | 1 | 1 | 1.00 |
| MCO | 1 | 1 | 1.00 |
| LBN | 1 | 1 | 1.00 |
| ISR | 1 | 1 | 1.00 |
| FRA | 1 | 1 | 1.00 |
| VEN | 1 | 0 | 0.00 |
