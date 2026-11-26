# M17 step 4: fresh geographic holdout (CONFIRMATORY; 9 metros x 6 windows; seeds 101-103)

Metros: Gwangju, Ulsan, Jeonju, Cheongju (KR), Piura, Chiclayo, Huancayo (PE), Castellón, Figueres (ES); GLO-30 DSM. Real close15 of these windows (mean of metros): 0.098. Excess = REBUILT close15 − the same window's real close15. CIs: bootstrap metros → windows → seeds. Margins frozen in amendment 1.

| Arm | Variant | excess [95% CI] | Δ excess vs original [95% CI] | windows overcorrected (< −0.05) | close15 raw / rebuilt | nn p10 rebuilt | consol10 raw / rebuilt | junctions / km² | Δ OD | Δ coverage | Δ street km | Δ > 15 % | Δ block log-area | margins | reuse / tile | evals / tile | s / tile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M* | original | +0.086 [-0.022, +0.169] | +0.000 [+0.000, +0.000] | 22% | 0.198 / 0.185 | 15.2 | 1.037 / 1.025 | 135 | +0.000 | +0.000 | +0.00 | +0.000 | +0.00 | ok | 0.0 | 2814 | 1.0 |
| M* | F1 reuse r=15 | +0.055 [-0.045, +0.136] | -0.033 [-0.073, +0.009] | 25% | 0.169 / 0.154 | 17.9 | 1.031 / 1.020 | 125 | -0.001 | -0.008 | -0.14 | +0.001 | +0.01 | ok | 10.2 | 2711 | 1.0 |
| M* | F0 post-hoc tau=15 | -0.076 [-0.170, -0.008] | -0.162 [-0.197, -0.127] | 38% | 0.012 / 0.022 | 20.3 | 1.000 / 1.001 | 117 | +0.000 | -0.000 | -0.04 | +0.000 | +0.25 | FAIL: block_logA | 0.0 | 2814 | 1.0 |
| M* | F0 post-hoc tau=9 (secondary) | +0.063 [-0.041, +0.143] | -0.023 [-0.033, -0.014] | 24% | 0.159 / 0.161 | 16.0 | 1.010 / 1.011 | 132 | -0.001 | -0.000 | -0.01 | +0.000 | +0.09 | ok | 0.0 | 2814 | 1.0 |
| M* | spacing r=15 (ablation) | +0.006 [-0.096, +0.077] | -0.081 [-0.122, -0.038] | 26% | 0.104 / 0.104 | 19.5 | 1.010 / 1.006 | 120 | +0.008 | +0.008 | -0.10 | +0.002 | +0.25 | ok | 0.0 | 2740 | 1.0 |
| NA-fixed | original | +0.097 [+0.005, +0.167] | +0.000 [+0.000, +0.000] | 18% | 0.216 / 0.193 | 17.3 | 1.044 / 1.030 | 133 | +0.000 | +0.000 | +0.00 | +0.000 | +0.00 | ok | 0.0 | 17566 | 10.5 |
| NA-fixed | F1 reuse r=15 | +0.061 [-0.053, +0.142] | -0.038 [-0.090, +0.012] | 24% | 0.176 / 0.157 | 19.0 | 1.027 / 1.015 | 126 | +0.004 | -0.020 | -0.18 | -0.001 | -0.04 | ok | 66.1 | 16875 | 10.1 |
| NA-fixed | F0 post-hoc tau=15 | -0.087 [-0.183, -0.022] | -0.184 [-0.221, -0.150] | 40% | 0.007 / 0.010 | 23.5 | 1.000 / 1.000 | 113 | +0.000 | -0.000 | -0.04 | +0.000 | +0.41 | FAIL: block_logA | 0.0 | 17566 | 10.5 |
| NA-fixed | F0 post-hoc tau=9 (secondary) | +0.075 [-0.012, +0.144] | -0.021 [-0.037, -0.010] | 21% | 0.171 / 0.171 | 18.4 | 1.015 / 1.016 | 131 | -0.000 | -0.000 | -0.01 | -0.000 | +0.09 | ok | 0.0 | 17566 | 10.5 |
| NA-fixed | spacing r=15 (ablation) | -0.020 [-0.117, +0.052] | -0.116 [-0.159, -0.074] | 30% | 0.081 / 0.079 | 20.1 | 1.010 / 1.006 | 119 | +0.002 | +0.006 | -0.12 | +0.000 | +0.47 | FAIL: block_logA | 0.0 | 17394 | 10.4 |
| NA-SMC | original | +0.111 [+0.016, +0.185] | +0.000 [+0.000, +0.000] | 16% | 0.239 / 0.210 | 16.5 | 1.052 / 1.032 | 159 | +0.000 | +0.000 | +0.00 | +0.000 | +0.00 | ok | 0.0 | 12925 | 5.2 |
| NA-SMC | F1 reuse r=15 | +0.071 [-0.029, +0.144] | -0.040 [-0.084, +0.004] | 22% | 0.188 / 0.168 | 15.5 | 1.039 / 1.024 | 138 | +0.000 | -0.014 | -0.26 | -0.000 | -0.01 | ok | 53.7 | 12354 | 4.8 |
| NA-SMC | F0 post-hoc tau=15 | -0.082 [-0.179, -0.016] | -0.193 [-0.227, -0.160] | 38% | 0.010 / 0.016 | 22.6 | 1.000 / 1.001 | 132 | +0.000 | -0.000 | -0.06 | +0.000 | +0.39 | FAIL: block_logA | 0.0 | 12925 | 5.2 |
| NA-SMC | F0 post-hoc tau=9 (secondary) | +0.090 [-0.007, +0.165] | -0.021 [-0.035, -0.007] | 18% | 0.190 / 0.189 | 17.3 | 1.016 / 1.016 | 155 | +0.000 | -0.000 | -0.02 | +0.000 | +0.12 | ok | 0.0 | 12925 | 5.2 |
| NA-SMC | spacing r=15 (ablation) | +0.000 [-0.106, +0.071] | -0.110 [-0.155, -0.063] | 27% | 0.099 / 0.099 | 20.7 | 1.013 / 1.008 | 124 | +0.007 | -0.012 | -0.31 | +0.001 | +0.47 | FAIL: block_logA | 0.0 | 12062 | 4.4 |
| M* + organic | original | +0.518 [+0.410, +0.602] | +0.000 [+0.000, +0.000] | 2% | 0.655 / 0.616 | 9.0 | 1.237 / 1.149 | 1202 | +0.000 | +0.000 | +0.00 | +0.000 | +0.00 | ok | 0.0 | 10340 | 7.8 |
| M* + organic | F1 reuse r=15 | +0.365 [+0.263, +0.439] | -0.153 [-0.187, -0.117] | 6% | 0.499 / 0.463 | 9.6 | 1.136 / 1.084 | 958 | +0.000 | +0.001 | -1.26 | +0.002 | +0.23 | ok | 408.5 | 8951 | 7.5 |
| M* + organic | F0 post-hoc tau=15 | -0.037 [-0.137, +0.031] | -0.555 [-0.588, -0.522] | 34% | 0.025 / 0.061 | 15.6 | 1.000 / 1.002 | 642 | -0.000 | -0.000 | -1.16 | -0.001 | +0.64 | FAIL: block_logA | 0.0 | 10340 | 7.8 |
| M* + organic | F0 post-hoc tau=9 (secondary) | +0.444 [+0.336, +0.528] | -0.074 [-0.081, -0.067] | 4% | 0.546 / 0.542 | 10.1 | 1.056 / 1.056 | 1093 | -0.000 | -0.000 | -0.40 | -0.000 | +0.19 | ok | 0.0 | 10340 | 7.8 |
| M* + organic | spacing r=15 (ablation) | +0.175 [+0.077, +0.245] | -0.343 [-0.381, -0.305] | 12% | 0.277 / 0.273 | 11.5 | 1.038 / 1.029 | 750 | +0.001 | -0.006 | -2.98 | +0.001 | +0.63 | FAIL: street_km, block_logA | 0.0 | 7581 | 6.0 |
| M* + radial | original | +0.378 [+0.281, +0.453] | +0.000 [+0.000, +0.000] | 4% | 0.512 / 0.476 | 9.4 | 1.149 / 1.100 | 627 | +0.000 | +0.000 | +0.00 | +0.000 | +0.00 | ok | 0.0 | 6913 | 5.1 |
| M* + radial | F1 reuse r=15 | +0.278 [+0.169, +0.360] | -0.100 [-0.137, -0.062] | 8% | 0.407 / 0.376 | 9.9 | 1.105 / 1.071 | 597 | -0.000 | +0.001 | -0.05 | +0.002 | +0.07 | ok | 148.0 | 6789 | 5.2 |
| M* + radial | F0 post-hoc tau=15 | -0.053 [-0.151, +0.012] | -0.431 [-0.454, -0.407] | 36% | 0.024 / 0.045 | 16.3 | 1.000 / 1.001 | 429 | -0.000 | -0.000 | -0.24 | -0.000 | +0.32 | FAIL: block_logA | 0.0 | 6913 | 5.1 |
| M* + radial | F0 post-hoc tau=9 (secondary) | +0.308 [+0.211, +0.384] | -0.070 [-0.079, -0.062] | 7% | 0.409 / 0.406 | 10.6 | 1.042 / 1.042 | 588 | +0.000 | -0.000 | -0.13 | -0.000 | +0.14 | ok | 0.0 | 6913 | 5.1 |
| M* + radial | spacing r=15 (ablation) | +0.110 [+0.013, +0.177] | -0.269 [-0.301, -0.235] | 15% | 0.205 / 0.207 | 12.0 | 1.032 / 1.027 | 500 | -0.001 | -0.007 | -0.82 | +0.001 | +0.38 | FAIL: block_logA | 0.0 | 6173 | 4.2 |

## Mechanism: edit classes of the close pairs (RAW final graphs), original vs F1

* **M* — original** (615 pairs): new_node/u 32%, t_junction/split 30%, crossing_split/split 21%, merge_node/v 5%, merge_node/u 4%
* **M* — F1** (474 pairs): new_node/u 42%, t_junction/split 27%, crossing_split/split 15%, crossing_endpoint/u 4%, crossing_endpoint/v 4%
* **NA-fixed — original** (652 pairs): new_node/u 32%, t_junction/split 29%, crossing_split/split 22%, merge_node/v 4%, crossing_endpoint/v 4%
* **NA-fixed — F1** (481 pairs): new_node/u 37%, t_junction/split 30%, crossing_split/split 18%, merge_node/u 7%, merge_node/v 3%
* **NA-SMC — original** (891 pairs): t_junction/split 31%, new_node/u 29%, crossing_split/split 23%, merge_node/u 6%, crossing_endpoint/v 5%
* **NA-SMC — F1** (591 pairs): new_node/u 35%, t_junction/split 27%, crossing_split/split 20%, merge_node/u 5%, merge_node/v 5%
* **M* + organic — original** (17740 pairs): crossing_split/split [style] 35%, t_junction/split [style] 15%, new_node/u [style] 11%, crossing_endpoint/v [style] 9%, t_junction/split 7%
* **M* + organic — F1** (10166 pairs): crossing_split/split [style] 28%, new_node/u [style] 17%, t_junction/split [style] 9%, crossing_endpoint/v [style] 8%, t_junction/split 7%
* **M* + radial — original** (6780 pairs): t_junction/split [style] 21%, crossing_split/split [style] 16%, t_junction/split 14%, crossing_split/split 10%, new_node/u [style] 10%
* **M* + radial — F1** (4940 pairs): t_junction/split [style] 15%, new_node/u [style] 14%, t_junction/split 13%, crossing_split/split [style] 12%, new_node/u 10%
