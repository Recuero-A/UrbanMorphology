# Dataset audit before Milestone 3 training

## 1. Dimensions and values
- 135 tiles loaded; shape/dtype/binary problems: none

## 2. Separation between splits (edge-to-edge, same city)

| pair | min distance (m) | tiles closer than 1000 m |
|---|---|---|
| train–val | 1024 | 0 |
| train–test | 1024 | 0 |
| val–test | 1024 | 0 |

- Minimum edge distance between ANY two tiles: 1024 m (selection spacing 1536 m between centres).

## 3. Duplicates and near-duplicates
- Exact duplicate rasters: 0
- Highest pairwise IoU (building channel, then road channel) between any two tiles:

| building IoU | road IoU | tile A | tile B |
|---|---|---|---|
| 0.415 | 0.131 | ESP_BCN_e837_n8952 | ESP_BCN_e839_n8948 |
| 0.406 | 0.096 | ESP_BCN_e837_n8952 | ESP_VLC_e1418_n8539 |
| 0.396 | 0.045 | ESP_BCN_e837_n8952 | ESP_MAD_e878_n8754 |
| 0.389 | 0.086 | ESP_BCN_e839_n8948 | ESP_VLC_e1418_n8539 |
| 0.389 | 0.113 | ESP_BCN_e837_n8952 | ESP_BCN_e836_n8948 |

Tiles are disjoint grid cells, so high IoU could only mean very regular repeated urban patterns (e.g. identical grid blocks), not shared territory.

## 4. Balance

| country | city | train | val | test |
|---|---|---|---|---|
| japan | nagoya | 11 | 2 | 2 |
| japan | osaka | 10 | 3 | 2 |
| japan | tokyo | 11 | 2 | 2 |
| spain | barcelona | 11 | 2 | 2 |
| spain | madrid | 10 | 2 | 3 |
| spain | valencia | 11 | 2 | 2 |
| united_states | chicago | 11 | 2 | 2 |
| united_states | los_angeles | 11 | 2 | 2 |
| united_states | new_york | 11 | 2 | 2 |

Density strata by split:

| density | test | train | val |
|---|---|---|---|
| high | 4 | 28 | 10 |
| low | 9 | 35 | 6 |
| medium | 6 | 34 | 3 |

## 5. Quality flags by split

- train (97): warnings {'buildings_incomplete_suspected': 14}, info {'green_unmapped_or_absent': 15}
- val (19): warnings {'buildings_incomplete_suspected': 2}, info {'green_unmapped_or_absent': 5}
- test (19): warnings {'buildings_incomplete_suspected': 2}, info {'green_unmapped_or_absent': 2}

## 6. Positive-pixel fraction per channel (class imbalance)

| split | roads | buildings | green |
|---|---|---|---|
| train | 0.134 | 0.250 | 0.095 |
| val | 0.138 | 0.271 | 0.040 |
| test | 0.132 | 0.235 | 0.073 |

## 7. Road-width limitation

| country | share of road segments with class-default width | mean road coverage |
|---|---|---|
| japan | 0.77 | 0.159 |
| spain | 0.42 | 0.128 |
| united_states | 0.66 | 0.115 |

Road coverage is partly an artefact of default widths (largest effect in Japan). The model will learn these widths as if they were real; road-coverage errors are reported but not interpreted as cross-country differences.

