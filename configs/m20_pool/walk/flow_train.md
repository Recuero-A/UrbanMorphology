# train: flow per stratum (D1)

| stratum | listed | terrain_evaluated | no_dem | eligible | eligible_B | osm_tested | failed_km | failed_coverage | failed_both | block_cap | band_A_full | band_B_third | accepted | accepted_B | accepted_as_fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1:city | 108 | 108 | 0 | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 |
| R1:town | 720 | 60 | 0 | 12 | 5 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 2 | 2 |
| R2:city | 291 | 291 | 0 | 9 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| R2:town | 1771 | 210 | 0 | 12 | 1 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| R3:city | 740 | 740 | 0 | 7 | 2 | 7 | 2 | 2 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| R3:town | 4975 | 216 | 0 | 13 | 4 | 13 | 9 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| R4:city | 438 | 426 | 9 | 12 | 2 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 1 | 0 |
| R4:town | 2067 | 162 | 3 | 12 | 1 | 7 | 1 | 0 | 2 | 1 | 0 | 0 | 4 | 0 | 0 |

Gate failures: km = under the vehicular minimum (15 km; villages 4 km); coverage = under 90 % of built-up cells within 100 m of any way; both = both.

# train: balance, terrain-eligible settlements

| group | n | slope_med | band_B_share | area_km2_med | pop_med | vehicular_km_med |
|---|---|---|---|---|---|---|
| accepted | 26 | 14.96 | 0.15 | 7.00 | 41790.00 | 54.74 |
| eligible, not accepted | 53 | 15.60 | 0.25 | 2.00 | 15151.00 | 3.61 |

# train: OSM gate pass rate by country (selection on mapping quality)

| iso3 | tested | passed | pass_rate |
|---|---|---|---|
| CHN | 19 | 2 | 0.11 |
| BRA | 6 | 6 | 1.00 |
| YEM | 4 | 2 | 0.50 |
| DZA | 3 | 2 | 0.67 |
| ITA | 3 | 3 | 1.00 |
| TUR | 3 | 3 | 1.00 |
| GRC | 2 | 2 | 1.00 |
| PRT | 2 | 2 | 1.00 |
| BOL | 1 | 0 | 0.00 |
| GTM | 1 | 0 | 0.00 |
| MEX | 1 | 1 | 1.00 |
| PRK | 1 | 0 | 0.00 |
| PSX | 1 | 1 | 1.00 |
| SRB | 1 | 1 | 1.00 |
| VEN | 1 | 1 | 1.00 |
