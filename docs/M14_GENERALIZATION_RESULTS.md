# Milestone 14 generalization results

**Source and conventions.**
* All numbers are from `outputs/m14/REPORT.md`, generated from stored runs.
* Values are the mean over training seeds TN_s0–s2 and inference seeds 1–3 unless stated. The full tables carry [min, max] over training seeds.
* Tests and CIs are in `M14_STATISTICAL_ANALYSIS.md`.
* The sealed test splits were never read.

**Method key.**

| Label | Sampler |
|---|---|
| A | M12 ancestral |
| M\* | M13 energy sampler with hard terrain constraints and terrain potentials, K = 8 |
| M\*+repair | M\* followed by post‑hoc A\* reconnection |
| P | procedural baseline |
| M13‑SMC | M13 SMC, 4 particles |
| NA‑myopic | network weight on STOP only |
| NA‑fixed | network roll‑outs at every decision (H = 2) |
| NA‑adaptive | roll‑outs where the frozen trigger `deg1` fires |
| NA‑random | roll‑outs at random decisions, same rate |
| NA‑SMC | network potentials with 4‑particle properly weighted SMC |

All NA methods use the frozen λ_D = 1.8143, λ_I = 0, moment‑matched on training tiles.

## 1. GEN2: four fresh cities (primary)

GEN2 has 95 tiles in 34 blocks from Alcoi, Tarragona, Santa Barbara and Sacramento. It was built after the protocol was frozen and was never inspected before the confirmatory runs.

| Method | OD unreach. (land) | OD non‑empty | Islands / tile | Dead‑end ratio | S10 | S_T | Density / real | Unbridged / tile | s / tile |
|---|---|---|---|---|---|---|---|---|---|
| real | 0.047 | 0.047 | 0.13 | 0.023 | — | — | 1.00 | 0.13 | — |
| A | 0.035 | 0.032 | 0.28 | 0.052 | 0.433 | 0.265 | 1.20 | 0.49 | 0.37 |
| M\* | 0.037 | 0.033 | 0.28 | 0.054 | 0.369 | 0.235 | 1.12 | 0.00 | 0.64 |
| M\*+repair | 0.006 | 0.002 | 0.02 | 0.044 | 0.424 | 0.403 | 1.15 | 0.04 | 0.64 |
| P | 0.055 | 0.028 | 0.11 | 0.015 | 0.865 | 0.358 | 1.10 | 0.19 | 0.02 |
| M13‑SMC | 0.048 | 0.040 | 0.23 | 0.051 | 0.309 | 0.198 | 0.90 | 0.00 | 2.19 |
| NA‑myopic | 0.010 | 0.006 | 0.08 | 0.022 | 0.501 | 0.279 | 1.34 | 0.00 | 0.86 |
| NA‑adaptive | 0.012 | 0.008 | 0.09 | 0.022 | 0.486 | 0.263 | 1.32 | 0.00 | 5.35 |
| NA‑random | 0.012 | 0.008 | 0.08 | 0.022 | 0.454 | 0.268 | 1.29 | 0.00 | 4.91 |
| NA‑fixed | 0.011 | 0.007 | 0.07 | 0.019 | 0.370 | 0.248 | 1.21 | 0.00 | 9.75 |
| **NA‑SMC** | 0.012 | **0.003** | **0.02** | **0.014** | **0.332** | 0.217 | **1.11** | 0.00 | 2.74 |

**Findings.**
* **Every network‑aware sampler cuts same‑land unreachability** by about 70 % relative to M\* (H1, p < 10⁻⁴). Islands fall from 0.28 to 0.02–0.09 per tile, and the dead‑end ratio falls from 0.054 to the real level (0.014–0.022 vs 0.023).
* **They differ in realism and density.**
  * The single‑chain variants (myopic, adaptive, random) raise S10 and density.
  * NA‑fixed keeps S10 at the M\* level (+0.000).
  * NA‑SMC lowers S10 (−0.037, p = 0.026) at M\* density (1.11).
* **On GEN2 the generators over‑connect relative to real streets** (real OD 0.047; Tarragona alone 0.128). See `M14_FAILURE_ANALYSIS.md` §2.3.

## 2. DEV‑B (secondary; 27 tiles, 13 blocks)

| Method | OD (land) | Islands | Dead‑end | S10 | Density | s / tile |
|---|---|---|---|---|---|---|
| real | 0.011 | 0.04 | 0.025 | — | 1.00 | — |
| A | 0.021 | 0.20 | 0.047 | 0.435 | 1.14 | 0.43 |
| M\* | 0.028 | 0.23 | 0.047 | 0.442 | 1.14 | 0.77 |
| M\*+repair | 0.001 | 0.00 | 0.040 | 0.510 | 1.17 | 0.77 |
| M13‑SMC | 0.025 | 0.15 | 0.040 | 0.311 | 0.95 | 2.51 |
| NA‑myopic | 0.008 | 0.09 | 0.020 | 0.593 | 1.33 | 1.14 |
| NA‑adaptive | 0.003 | 0.05 | 0.019 | 0.582 | 1.27 | 5.89 |
| NA‑random | 0.004 | 0.04 | 0.018 | 0.542 | 1.27 | 5.37 |
| NA‑fixed | 0.007 | 0.05 | 0.017 | 0.488 | 1.22 | 10.45 |
| NA‑SMC | 0.015 (non‑empty 0.003) | 0.02 | 0.012 | 0.426 | 1.12 | 2.99 |

* **H1 replicates** (−0.021, p = 0.0005).
* **Real DEV‑B tiles are well connected** (0.011). The NA methods move to the real level rather than past it.
* **NA‑SMC's empty‑tile rate (1.2 %) inflates its primary value.** The NA‑SMC − M\* contrast is not significant on DEV‑B (p = 0.42).

## 3. GEN (secondary; Girona, Manresa, San Francisco; 100 tiles, 24 blocks)

GEN was used in M13 for A and M\* summaries, but not for any M14 choice. It was re‑run after the contamination incident (`M14_FAILURE_ANALYSIS.md` §4.1).

| Method | OD (land) | OD non‑empty | Islands | Dead‑end | S10 | Density | s / tile |
|---|---|---|---|---|---|---|---|
| real | 0.003 | 0.003 | 0.05 | 0.017 | — | 1.00 | — |
| A | 0.036 | 0.022 | 0.26 | 0.050 | 0.407 | 1.10 | 0.38 |
| M\* | 0.039 | 0.023 | 0.24 | 0.049 | 0.371 | 1.04 | 0.79 |
| M\*+repair | 0.018 | 0.002 | 0.01 | 0.042 | 0.396 | 1.06 | 0.72 |
| M13‑SMC | 0.040 | 0.027 | 0.19 | 0.046 | 0.383 | 0.85 | 2.69 |
| NA‑myopic | 0.019 | 0.005 | 0.07 | 0.019 | 0.424 | 1.23 | 1.00 |
| NA‑adaptive | 0.021 | 0.006 | 0.06 | 0.018 | 0.361 | 1.16 | 6.24 |
| NA‑random | 0.022 | 0.007 | 0.07 | 0.018 | 0.359 | 1.16 | 5.96 |
| NA‑fixed | 0.021 | 0.006 | 0.07 | 0.017 | **0.277** | 1.06 | 11.39 |
| NA‑SMC | 0.020 | **0.002** | **0.03** | 0.013 | 0.283 | 1.05 | 3.51 |

* **H1 replicates** (−0.017, p < 10⁻⁴). On GEN, NA‑fixed also *improves* S10 (−0.095, p = 0.002), and so does NA‑SMC (−0.089, p = 0.002).
* **On non‑empty tiles the NA methods reach the real level** (0.002–0.007 vs 0.003), from M\* 0.023. The dead‑end ratio matches real (0.017–0.019 vs 0.017).
* **About 0.015 of every method's primary OD comes from empty samples.** These are on five tiles where the frozen M12 start head proposes no start at all (failure analysis §3.4). No inference‑time method can fix this.

## 4. Per city (OD on non‑empty tiles; real → M\* → NA‑fixed → NA‑SMC)

| City (tiles) | Real | M\* | NA‑fixed | NA‑SMC | Density / real (M\* / NA‑fixed) |
|---|---|---|---|---|---|
| Alcoi (20) | 0.021 | 0.062 | 0.011 | 0.009 | 0.84 / 1.00 |
| Tarragona (25) | 0.128 | 0.017 | 0.002 | 0.000 | 1.43 / 1.53 |
| Sacramento (25) | 0.010 | 0.016 | 0.003 | 0.000 | 1.03 / 1.08 |
| Santa Barbara (25) | 0.027 | 0.042 | 0.013 | 0.004 | 1.17 / 1.21 |
| Girona (30) | 0.007 | 0.039 | 0.006 | 0.003 | 1.00 / 1.00 |
| Manresa (30) | 0.000 | 0.022 | 0.006 | 0.003 | 1.14 / 1.19 |
| San Francisco (40) | 0.002 | 0.013 | 0.005 | 0.001 | 1.00 / 1.01 |

* **The connectivity gain holds in all seven cities.** It is largest in the mountain city (Alcoi, ×5.6 reduction).
* **Tarragona is the outlier.** Its real network is much less connected than any generator's, and every generator over‑produces density there.

## 5. Terrain strata (GEN2 + DEV‑B + GEN pooled; TN; strata from the real terrain; descriptive)

| Stratum | Tiles | M\* OD non‑empty | NA‑fixed | NA‑SMC | Dead‑end real / M\* / NA‑fixed | Real unbridged / tile |
|---|---|---|---|---|---|---|
| flat | 95 | 0.015 | 0.004 | 0.001 | 0.015 / 0.041 / 0.015 | 0.05 |
| moderate | 64 | 0.016 | 0.005 | 0.002 | 0.020 / 0.050 / 0.016 | 0.00 |
| steep | 30 | 0.040 | 0.009 | 0.002 | 0.028 / 0.059 / 0.022 | 0.30 |
| mountainous | 33 | 0.078 | 0.017 | 0.010 | 0.030 / 0.074 / 0.027 | 0.12 |
| river | 21 | 0.059 | 0.011 | 0.008 | 0.026 / 0.068 / 0.026 | 0.33 |
| coastal | 8 | 0.007 | 0.002 | 0.002 | 0.028 / 0.046 / 0.020 | 0.00 |
| mixed (water + slope) | 13 | 0.089 | 0.018 | 0.012 | 0.030 / 0.081 / 0.032 | 0.31 |

* **The M\* deficit grows with terrain difficulty** (flat 0.015 → mountainous 0.078, mixed 0.089). The network potentials remove most of it in every stratum. The absolute gain is largest where terrain is hardest.
* **On steep and mountainous tiles M\* builds less than real** (density 15.5 vs 16.6 and 12.3 vs 13.7 km/km²); the NA methods build more. In the flat stratum all methods over‑build.
* **Real "unbridged crossings" are not zero** on steep, river and mixed tiles (0.12–0.33 per tile). They are presumably OSM ways crossing mapped water without a bridge tag (fords, culverts, tag gaps; not verified tile by tile), so the generators' hard constraint is stricter than the data. This is noted as a measurement issue, not a generator failure.
* **Strata overlap, and coastal has only 8 tiles.** No tests are run.

## 6. Leakage controls (H5; GEN2, inference seeds 1–2)

The backbones TNlp (trained on a low‑passed DEM) and TNse (trained on a DEM with street imprints erased by the M13 f4 procedure) show the same effect:

| Family | NA‑adaptive − M\* OD per seed | Verdict |
|---|---|---|
| TNlp | −0.017, −0.025, −0.013 (CI < 0 in 3/3) | supported |
| TNse | −0.039, −0.041, −0.029 (CI < 0 in 3/3) | supported |

* **The effect is larger with TNse.** Its M\* baseline is less connected: non‑empty OD 0.047–0.054 vs 0.023–0.037 for TNlp.
* **The connectivity effect does not rely on street imprints in the DEM**, as far as the erasure procedure removes them. The street‑erased DEM is a reconstruction, not ground truth.

## 7. Synthetic terrain shifts (TN_s0, inference seeds 1–2; robustness only)

| Set | A | M\* | NA‑myopic | NA‑adaptive |
|---|---|---|---|---|
| DEV‑B@x2 (relief doubled): OD | 0.032 | 0.035 | 0.011 | 0.006 |
| DEV‑B@rough (fractal noise): OD | 0.053 | **0.265** | 0.017 | 0.080 |
| DEV‑B@rough: density / real | 1.00 | **0.56** | 1.05 | 0.68 |
| GEN2@x2: OD | 0.090 | 0.111 | 0.039 | 0.052 |

* **The network advantage persists under both shifts.**
* **Rough terrain breaks M\*.** Its premature‑termination share is 0.41 (A: 0.07) and it builds about half the real density; NA‑adaptive is partly affected.
* **NA‑myopic is the most robust.**
* **Real streets are a mismatched reference here by construction.** These sets do **not** establish real‑world generalization.

## 8. Transfer of λ across backbones (training tiles, NA‑fixed)

λ was calibrated with TN_s0 (dead ends 2.15 per tile vs target 2.27). Applied unchanged, it gives:
* TN_s1: 2.89 per tile;
* TN_s2: 2.71 per tile.

That is 19–27 % above target. Islands stay within the target range. The H1 effect holds for all three backbones, but a deployment would recalibrate λ per backbone.

## 9. What generalizes and what does not

| Generalizes (fresh cities, three backbones, two leakage controls) | Does not, or not shown |
|---|---|
| Connectivity gain from network potentials (all 7 cities, all strata, 9/9 set × seed cells) | Adaptive allocation of look‑ahead (no better than random anywhere) |
| Dead‑end ratio brought to the real level | Density calibration (DEV‑A stop biases do not transfer to GEN2) |
| Diversity preserved (H4) | Agreement with real *unreachability* where real networks are fragmented (Tarragona) |
| NA‑SMC realism at M\* level or better (GEN2, GEN) | Real‑world generalization beyond 7 cities in 2 countries; any claim about the sealed test |
