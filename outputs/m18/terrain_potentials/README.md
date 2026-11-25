# M18 D12: terrain moment potentials, on vs off (development)

NA-SMC with the frozen potentials (`configs/m18_terrain_potentials_<family>.json`, fitted on 48 training tiles) vs without, paired by backbone (s0-s2) x seed (1, 2) x tile. Δ = on − off, paired block bootstrap 95 % CI, block-swap permutation p. Design and decision rule: docs/M18_LOG.md D12; results F24, F25. Script `scripts/m18_terrain_potentials.py`. Figure `fig_on_off.png`.

## TN · GEN2 (95 city tiles)

| Outcome | real | off | on | Δ [95 % CI] | p |
|---|---|---|---|---|---|
| terrain realism (targeted; lower = closer) | 0.467 | 0.611 | 0.671 | +0.060 [-0.102, +0.227] | 0.414 |
| climb gap to the real streets of the same tiles (untargeted) | 0.000 | 0.191 | 0.155 | -0.037 [-0.104, +0.061] | 0.025 |
| KS distance of climb lengths to the real streets | 0.000 | 0.092 | 0.076 | -0.016 [-0.032, +0.002] | 0.037 |
| coverage within 60 m | 0.802 | 0.813 | 0.808 | -0.005 [-0.025, +0.016] | 0.586 |
| coverage, 10-20 % land | 0.859 | 0.780 | 0.786 | +0.006 [-0.024, +0.045] | 0.700 |
| coverage, 20-30 % land | 0.776 | 0.723 | 0.720 | -0.003 [-0.046, +0.077] | 0.943 |
| geographic access (fixed points within 60 m) | 0.810 | 0.817 | 0.809 | -0.008 [-0.026, +0.011] | 0.376 |
| geographic connectivity (same-land pairs joined and linked) | 0.703 | 0.718 | 0.709 | -0.009 [-0.030, +0.012] | 0.398 |
| detour p50 of linked pairs | 1.517 | 1.527 | 1.533 | +0.006 [-0.005, +0.018] | 0.291 |

## TN · natural windows (16, leakage-free)

| Outcome | real | off | on | Δ [95 % CI] | p |
|---|---|---|---|---|---|
| terrain realism (targeted; lower = closer) | — | 1.823 | 2.007 | +0.185 [-0.006, +0.476] | 0.147 |
| coverage within 60 m | — | 0.717 | 0.702 | -0.015 [-0.056, +0.029] | 0.540 |
| coverage, 10-20 % land | — | 0.760 | 0.728 | -0.032 [-0.077, +0.011] | 0.202 |
| coverage, 20-30 % land | — | 0.626 | 0.649 | +0.023 [-0.030, +0.073] | 0.425 |
| geographic access (fixed points within 60 m) | — | 0.743 | 0.719 | -0.024 [-0.063, +0.015] | 0.311 |
| geographic connectivity (same-land pairs joined and linked) | — | 0.582 | 0.565 | -0.018 [-0.073, +0.041] | 0.568 |
| detour p50 of linked pairs | — | 1.547 | 1.614 | +0.067 [+0.019, +0.135] | 0.018 |

## TNg · GEN2 (95 city tiles)

| Outcome | real | off | on | Δ [95 % CI] | p |
|---|---|---|---|---|---|
| terrain realism (targeted; lower = closer) | 0.467 | 2.217 | 1.417 | -0.800 [-1.202, -0.412] | 0.001 |
| climb gap to the real streets of the same tiles (untargeted) | 0.000 | 0.391 | 0.375 | -0.016 [-0.155, +0.190] | 0.617 |
| KS distance of climb lengths to the real streets | 0.000 | 0.159 | 0.174 | +0.015 [+0.004, +0.028] | 0.015 |
| coverage within 60 m | 0.802 | 0.777 | 0.778 | +0.001 [-0.021, +0.022] | 0.926 |
| coverage, 10-20 % land | 0.859 | 0.762 | 0.710 | -0.053 [-0.107, -0.014] | 0.026 |
| coverage, 20-30 % land | 0.776 | 0.682 | 0.630 | -0.051 [-0.100, -0.018] | 0.031 |
| geographic access (fixed points within 60 m) | 0.810 | 0.778 | 0.775 | -0.003 [-0.025, +0.018] | 0.780 |
| geographic connectivity (same-land pairs joined and linked) | 0.703 | 0.664 | 0.656 | -0.008 [-0.035, +0.021] | 0.621 |
| detour p50 of linked pairs | 1.517 | 1.498 | 1.491 | -0.007 [-0.022, +0.006] | 0.341 |

## TNg · natural windows (16, leakage-free)

| Outcome | real | off | on | Δ [95 % CI] | p |
|---|---|---|---|---|---|
| terrain realism (targeted; lower = closer) | — | 1.579 | 0.897 | -0.682 [-0.953, -0.204] | 0.001 |
| coverage within 60 m | — | 0.690 | 0.707 | +0.017 [-0.057, +0.091] | 0.667 |
| coverage, 10-20 % land | — | 0.750 | 0.736 | -0.014 [-0.087, +0.052] | 0.727 |
| coverage, 20-30 % land | — | 0.662 | 0.615 | -0.047 [-0.128, +0.046] | 0.332 |
| geographic access (fixed points within 60 m) | — | 0.704 | 0.702 | -0.002 [-0.076, +0.077] | 0.976 |
| geographic connectivity (same-land pairs joined and linked) | — | 0.546 | 0.541 | -0.004 [-0.099, +0.092] | 0.935 |
| detour p50 of linked pairs | — | 1.566 | 1.535 | -0.031 [-0.127, +0.060] | 0.509 |

## Sites (whole towns, mean of seeds 1-3; real = OSM vehicular, sparse at both sites)

| Site | NA-SMC version | terrain realism | TAS | typicality | km/km² | coverage | largest piece | geo. connectivity | climb gap to real |
|---|---|---|---|---|---|---|---|---|---|
| machu_picchu | real (OSM vehicular) | 4.35 | — | — | — | — | — | 0.00 | 0.00 |
| machu_picchu | TN, off | 3.64 | 56.38 | 1.45 | 19.14 | 0.90 | 0.56 | 0.73 | 2.36 |
| machu_picchu | TN, on | 3.89 | 53.12 | 1.50 | 24.18 | 0.98 | 0.89 | 0.83 | 2.39 |
| machu_picchu | TNg, off | 5.29 | 40.91 | 1.88 | 45.12 | 0.77 | 0.94 | 0.76 | 1.98 |
| machu_picchu | TNg, on | 1.63 | 74.69 | 1.62 | 14.47 | 0.68 | 0.34 | 0.27 | 0.77 |
| petra | real (OSM vehicular) | 2.65 | — | — | — | — | — | 0.17 | 0.00 |
| petra | TN, off | 4.76 | 57.83 | 1.10 | 25.10 | 0.96 | 0.87 | 0.91 | 2.18 |
| petra | TN, on | 4.87 | 53.51 | 1.19 | 26.80 | 0.98 | 0.95 | 0.93 | 2.37 |
| petra | TNg, off | 4.76 | 53.82 | 1.40 | 36.47 | 0.91 | 0.98 | 0.88 | 3.24 |
| petra | TNg, on | 2.05 | 73.38 | 1.39 | 26.22 | 0.81 | 0.89 | 0.74 | 2.72 |

