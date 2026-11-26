# M18 audit 1: the M14 connectivity result, reproduced from stored runs, and its trade-offs (GEN2)

Source: `outputs/m14/runs` (M14 confirmatory specs), real `data/processed/m14/gen2/graphs3d_gen.pkl`. Script: `scripts/m18_audit_m14.py`.

## 1. Reproduction of the primary outcome

* Graphs re-scored (M\* and NA-fixed): 1710; empty-tile pattern agrees on 1710.
* Largest |stored − recomputed| OD unreachability: 0.0099, on 3 of 1710 graphs.
* Means stored → recomputed: M\* 0.0363 → 0.0363; NA-fixed 0.0109 → 0.0109.
* Precise re-estimate (10 streams per tile): M\* 0.0389, NA-fixed 0.0116; paired tile mean difference -0.0275.
* Monte Carlo noise of one 200-pair estimate (SD across streams, tiles with any unreachable pair): median 0.0276, p90 0.0461.
* Definition as implemented: pairs of the network's own nodes (30 sources x up to 7 targets, 150-500 m apart), unreachable on the closed graph and on the same dry-land region, over all pairs drawn; an empty network scores 1.
* H1 contrast NA-fixed − M\* (block sign-flip, as M14): {"key": "od_unreach_land", "delta": -0.025892352404185255, "lo": -0.0363346326738472, "hi": -0.015402370885715153, "p_signflip": 0.0, "n_blocks": 34, "per_seed": [-0.018953571381991892, -0.03260266654096055, -0.02612081928960332], "min_p": 1.1641532182693481e-10}

## 2. Trade-offs (mean over tiles per run, then over runs)

| Method | runs | OD unreach. | km/km² | junctions/km² | close15 (RAW) | nn p10 m | link p10 m | link median m | links < 15 m | links < 25 m | blocks/km² | block median m² | blocks < 2,000 m² | compactness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 1 | 0.047 | 13.9 | 77 | 0.145 | 31.7 | 29.5 | 80 | 0.067 | 0.165 | 27 | 8311 | 0.37 | 0.59 |
| A | 9 | 0.036 | 16.8 | 112 | 0.134 | 23.8 | 22.4 | 66 | 0.071 | 0.205 | 43 | 5071 | 0.46 | 0.54 |
| M* | 9 | 0.036 | 15.5 | 100 | 0.121 | 25.5 | 25.1 | 71 | 0.074 | 0.201 | 35 | 5319 | 0.45 | 0.54 |
| M*+repair | 9 | 0.006 | 16.0 | 105 | 0.122 | 23.5 | 25.4 | 71 | 0.072 | 0.196 | 35 | 5178 | 0.46 | 0.54 |
| M13-SMC | 9 | 0.048 | 12.6 | 71 | 0.077 | 33.0 | 29.4 | 79 | 0.060 | 0.168 | 22 | 6935 | 0.40 | 0.57 |
| NA-myopic | 9 | 0.010 | 18.7 | 128 | 0.149 | 20.8 | 21.7 | 67 | 0.079 | 0.203 | 53 | 4866 | 0.43 | 0.54 |
| NA-fixed | 9 | 0.011 | 16.8 | 107 | 0.129 | 23.5 | 24.1 | 72 | 0.074 | 0.189 | 42 | 5454 | 0.42 | 0.54 |
| NA-SMC | 9 | 0.011 | 15.5 | 99 | 0.124 | 26.7 | 29.1 | 79 | 0.066 | 0.178 | 39 | 5972 | 0.38 | 0.55 |
| P | 3 | 0.055 | 15.3 | 66 | 0.002 | 62.7 | 112.8 | 186 | 0.001 | 0.019 | 60 | 13698 | 0.06 | 0.55 |

## 3. Paired differences from M\* (matched backbone, seed and tile; block bootstrap 95 % interval)

| Contrast | od_unreach_land | street_km_per_km2 | junctions_per_km2 | close15 | link_lt15_share | blocks_per_km2 | block_area_med_m2 | block_small_share | block_pp_med |
|---|---|---|---|---|---|---|---|---|---|
| M*+repair - M* | -0.0247 [-0.0337, -0.0172] | +0.337 [+0.261, +0.422] | +4.9 [+3.96, +5.89] | +0.00145 [-0.0013, +0.00483] | -0.00142 [-0.0023, -0.000542] | +0.312 [+0.179, +0.469] | -86.5 [-170, -23.8] | +0.00658 [+0.00289, +0.0109] | -0.002 [-0.00475, +0.000578] |
| NA-myopic - M* | -0.0201 [-0.0274, -0.0133] | +2.7 [+2.04, +3.32] | +23.2 [+17.4, +29.4] | +0.0358 [+0.0217, +0.0507] | +0.00966 [+0.00478, +0.0149] | +15.3 [+11.6, +19.2] | -728 [-1.53e+03, -9.56] | -0.0165 [-0.0399, +0.00593] | +0.00263 [-0.0127, +0.0194] |
| NA-fixed - M* | -0.0208 [-0.0278, -0.0138] | +0.978 [+0.394, +1.57] | +4.07 [-1.33, +9.87] | +0.00732 [-0.00492, +0.0191] | +0.00245 [-0.00242, +0.0072] | +5.22 [+1.96, +8.68] | +334 [-469, +1.07e+03] | -0.0454 [-0.0963, +0.0107] | +0.00267 [-0.0152, +0.0197] |
| NA-SMC - M* | -0.0187 [-0.0293, -0.00693] | -0.39 [-0.883, +0.101] | -4.71 [-9.46, +0.147] | +0.00191 [-0.00838, +0.0121] | -0.00497 [-0.0104, +0.000541] | +1.99 [-0.86, +5] | +383 [-757, +1.34e+03] | -0.0521 [-0.101, +0.00895] | -0.00569 [-0.0451, +0.0227] |

close15 is RAW for generated graphs and built (8 m snap) for real ones; blocks exclude faces touching the tile edge. Means are of tile values; empty tiles give NaN block and link statistics and are skipped.
