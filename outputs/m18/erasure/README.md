# M18: validation of the street-erased DEM (TNse leakage control)

Script `scripts/m18_erasure_validation.py` (docstring = method); figure `fig_erasure.png`. Development data only.

## 1. Random-strip control (TN backbones s0-s2, inference seeds 1-2, 52 development tiles)

| Sampler | Model input | S_T mean (sd) | km/km² |
|---|---|---|---|
| A | raw | 0.391 (0.058) | 15.2 |
| A | street-erased | 0.691 (0.057) | 14.6 |
| A | random-erased | 0.569 (0.047) | 15.9 |
| M* | raw | 0.350 (0.036) | 15.0 |
| M* | street-erased | 0.594 (0.090) | 13.5 |
| M* | random-erased | 0.466 (0.024) | 14.9 |

Mask area share, own street corridors vs the borrowed ones: 0.35 vs 0.35.

## 2. Leakage probe

| Input | AUC (whole tile) | AP (base rate) | AUC inside erased corridors | AP inside (base rate) |
|---|---|---|---|---|
| raw | 0.779 | 0.342 (0.116) | 0.745 | 0.553 (0.302) |
| lowpass16 | 0.656 | 0.197 (0.116) | 0.559 | 0.355 (0.302) |
| street-erased | 0.736 | 0.240 (0.116) | 0.605 | 0.384 (0.302) |
| random-erased | 0.714 | 0.250 (0.116) | 0.653 | 0.441 (0.302) |
| landform64 | 0.510 | 0.123 (0.116) | 0.516 | 0.307 (0.302) |

## 3. Synthetic imprint test (8 natural IGN windows × 4 road layouts)

* Elevation RMSE in the corridors: stamped 1.82 m, erased 1.72 m, low-pass 16 m 1.98 m; recovery (1 − erased / stamped) median 0.12, range -0.60 to 0.39.
* Slope RMSE in the corridors: stamped 16.2 %, erased 13.4 %; low-pass 16 m off the corridors 9.9 % (it distorts natural terrain everywhere).
* Probe trained on raw real tiles, AUC for the synthetic roads: stamped 0.737, erased 0.530.
