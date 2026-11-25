# Milestone 14 realism, connectivity and diversity

Every number is from `outputs/m14/REPORT.md`. Values are means over TN_s0–s2 and inference seeds 1–3 on GEN2 (primary), unless another set is named.

**Metrics.**
* **S10_cond:** M13 realism gap over 10 network statistics, as the mean W1/IQR to the real tiles of the same set. Lower is more realistic.
* **S_T:** M13 terrain‑response gap.
* **Energy distance:** over the M13 joint keys.

The real‑vs‑real noise floors are given in `M14_STATISTICAL_ANALYSIS.md` §6.

## 1. Realism

| GEN2 | S10 | S_T | Energy dist. | Coverage 60 m | Detour | Bridges / tile | Premature | Density / real |
|---|---|---|---|---|---|---|---|---|
| real | — | — | — | 0.818 | 1.40 | 2.13 | — | 1.00 |
| A | 0.433 | 0.265 | 0.71 | 0.868 | 1.45 | 3.33 | 0.02 | 1.20 |
| M\* | 0.369 | 0.235 | 0.65 | 0.843 | 1.44 | 0.35 | 0.05 | 1.12 |
| M\*+repair | 0.424 | 0.403 | 0.70 | 0.845 | 1.59 | 0.35 | 0.03 | 1.15 |
| P | 0.865 | 0.358 | 0.79 | 0.827 | 1.35 | 0.56 | 0.05 | 1.10 |
| M13‑SMC | **0.309** | **0.198** | 0.52 | 0.772 | 1.40 | 0.06 | **0.12** | 0.90 |
| NA‑myopic | 0.501 | 0.279 | 0.48 | 0.895 | 1.48 | 0.59 | 0.01 | 1.34 |
| NA‑adaptive | 0.486 | 0.263 | 0.42 | 0.895 | 1.47 | 0.38 | 0.01 | 1.32 |
| NA‑random | 0.454 | 0.268 | 0.37 | 0.890 | 1.47 | 0.47 | 0.01 | 1.29 |
| NA‑fixed | 0.370 | 0.248 | **0.26** | 0.877 | 1.46 | 0.22 | 0.02 | 1.21 |
| NA‑SMC | 0.332 | 0.217 | 0.27 | 0.822 | 1.42 | 0.19 | 0.03 | 1.11 |

**Findings.**

1. **Network‑level realism and distribution‑level realism move together only for the corrected samplers.**
   * NA‑fixed and NA‑SMC have the lowest energy distance of all methods (0.26–0.27 vs M\* 0.65). Their S10 is at (NA‑fixed, +0.000) or below (NA‑SMC, −0.037, p = 0.026) the M\* level.
   * The single‑chain NA samplers lower energy distance (0.37–0.48), but raise S10 by 0.09–0.13. Their density is 1.29–1.34.
2. **The S10 increase of the single‑chain samplers tracks density.**
   * Under DEV‑A density matching the S10 ordering reverses: NA‑myopic − M\* is −0.097. See `M14_STATISTICAL_ANALYSIS.md` §5.
   * Between‑method S10 differences are therefore partly density effects. They are also smaller than the GEN2 real‑vs‑real median (0.293).
3. **M13‑SMC has the best S10 and S_T, but it is the only method flagged** for premature termination (0.12 vs A 0.02). It is also the least connected method (OD 0.048) and the sparsest (0.90).
4. **Post‑hoc repair costs terrain realism.** S_T rises 0.235 → 0.403 on GEN2 and 0.394 → 0.793 on DEV‑B. Detours rise to 1.59, and repair links cross rivers without bridges (0.035 per tile). Repair reaches the best OD (0.006) by adding straight connectors that ignore terrain. NA‑SMC reaches 0.012 without these costs.
5. **Bridges.** Real tiles have 2.1 bridges per tile on GEN2 and A builds 3.3. Every constrained sampler builds far fewer (0.06–0.59). This was inherited from M13's bridge rule and was not addressed in M14.
6. **On GEN** the pattern is stronger: NA‑fixed S10 0.277 and NA‑SMC 0.283 vs M\* 0.371 (both p = 0.002). **On DEV‑B** no NA method is below M\* on S10 (NA‑SMC 0.426 vs 0.442, n.s.).

## 2. Connectivity

| GEN2 | OD unreach. (land) | Non‑empty | Islands / tile | Closed giant | Dead‑end ratio | Gap to real: OD / dead‑end / islands |
|---|---|---|---|---|---|---|
| real | 0.047 | 0.047 | 0.13 | 0.962 | 0.023 | — |
| A | 0.035 | 0.032 | 0.28 | 0.983 | 0.052 | 0.012 / 0.029 / 0.15 |
| M\* | 0.037 | 0.033 | 0.28 | 0.980 | 0.054 | 0.011 / 0.031 / 0.15 |
| M\*+repair | 0.006 | 0.002 | 0.02 | 0.995 | 0.044 | 0.041 / 0.021 / 0.11 |
| NA‑fixed | 0.011 | 0.007 | 0.07 | 0.993 | 0.019 | 0.037 / 0.004 / 0.05 |
| NA‑SMC | 0.012 | 0.003 | 0.02 | 0.990 | 0.014 | 0.035 / 0.008 / 0.11 |

* **Network‑level statistics.** The network potentials bring the dead‑end ratio to the real level and cut islands by 3–14×.
* **Distance to real.** On GEN2 the OD gap to real *grows* (0.011 → 0.035–0.037), because real GEN2 networks are less connected than any generator's, mostly in Tarragona. On GEN and DEV‑B the OD gap shrinks: GEN 0.036 → 0.016–0.019; DEV‑B 0.017 → 0.003–0.008.
* **Reading.** "More connected" is an improvement against the M13 deficit and against two of three real sets, not a universal move toward real (`M14_FAILURE_ANALYSIS.md` §2.3).
* **Coverage.** Buildable land within 60 m of a street rises with the NA methods (0.877–0.895 vs M\* 0.843; real 0.818), in line with their higher density. NA‑SMC (0.822) is closest to real.

## 3. Diversity

| GEN2 | Jaccard distance | Orientation L1 | CV density | CV closed giant | CV dead‑end ratio |
|---|---|---|---|---|---|
| A | 0.772 | 0.508 | 0.228 | 0.025 | 0.282 |
| M\* | 0.772 | 0.521 | 0.245 | 0.028 | 0.311 |
| P | 0.928 | 0.501 | 0.405 | 0.074 | 0.660 |
| M13‑SMC | 0.771 | 0.583 | 0.283 | 0.041 | 0.368 |
| NA‑myopic | 0.764 | 0.482 | 0.201 | 0.012 | 0.426 |
| NA‑adaptive | 0.752 | 0.466 | 0.191 | 0.012 | 0.420 |
| NA‑random | 0.764 | 0.483 | 0.205 | 0.012 | 0.417 |
| NA‑fixed | 0.758 | 0.490 | 0.211 | 0.012 | 0.477 |
| NA‑SMC | 0.758 | 0.530 | 0.264 | 0.020 | 0.636 |

* **H4 is met on every set.** The NA‑adaptive / A Jaccard ratio is 0.974 on GEN2, 0.994 on DEV‑B and 0.987 on GEN. The samples of one tile stay as different from each other as under A (mean pairwise Jaccard distance of 8 m occupancy grids).
* **Variation collapses only where intended.** The across‑sample CV of the closed giant share drops from 0.025–0.028 to 0.012–0.020: the network potentials remove the *fragmented* samples, and that is the variation they target. The CV of density changes little (0.19–0.26 vs M\* 0.25).
* **The CV of the dead‑end ratio rises** (0.31 → 0.42–0.64). This is a small‑denominator effect: the ratio is near 0.02, so a single dead end changes it by a large relative amount.
* **The KL trust region is not needed for diversity** at the calibrated λ (ablations §6).
* **P has the highest diversity** because it is a noise‑driven procedural generator; it is also the least realistic (S10 0.865).

## 4. Qualitative inspection

**Galleries.** `outputs/m14/figures/fig_gallery_{gen2,devB,gen}.png` show TN_s0, first sample, for real, A, M\*, NA‑myopic, NA‑fixed and NA‑adaptive.

**Tile selection rule.** Rows are chosen by a fixed rule before looking:
* flattest tile;
* steepest tile;
* largest river share;
* largest sea share;
* the NA‑adaptive failure case.

**Observations on GEN2.**
* **Sacramento grid (flat).** Every generator reproduces the orthogonal grid. Differences are confined to the window edge, where NA‑myopic leaves one short isolated stub (OD 0.08).
* **Alcoi, steepest.** M\* builds a sparse, partly disconnected network on the lower slope. The network‑aware samplers fill the slope with contour‑following roads that are connected to each other. The real network is denser in the lower right of the tile.
* **Sacramento river.** All constrained samplers keep roads off the water. A (no constraints) crosses the river channel repeatedly; its bridges are drawn red.
* **Santa Barbara coast.** NA‑myopic over‑builds a dense rectilinear mesh, which shows the density inflation. NA‑fixed is closer to the real sparse pattern.
* **Failure row (Alcoi, steep valley).** NA‑adaptive leaves a fragment on the slope (OD 0.36), while M\* and NA‑fixed are connected in this sample.
* **GEN failure row.** It is a start‑head failure: every generator is empty on a tile with a dense real grid (`M14_FAILURE_ANALYSIS.md` §3.4).

**Limits of these figures.** One sample per method on six tiles per set illustrates the tables; it is not evidence.

## 5. Summary

| Question | Answer (GEN2 unless stated) |
|---|---|
| Does network‑aware inference improve connectivity? | Yes, by about 70 % on same‑land OD. Islands fall 3–14× and dead ends reach the real level. This holds on all sets, cities, strata and backbones. |
| At what realism cost? | None for NA‑fixed (S10 +0.000) and NA‑SMC (−0.037). +0.09 to +0.13 S10 for the single‑chain variants, which also inflate density. |
| Does it reduce diversity? | No: ≤ 3 % lower Jaccard distance (H4 met). |
| Is it better than post‑hoc repair? | On realism, yes: repair costs +0.17 S_T, 1.59 detour and unbridged crossings. Repair is slightly better on raw OD. |
| Is it closer to real? | For dead ends and islands, yes. For OD unreachability it is closer on DEV‑B and GEN, and further on GEN2, where real networks are fragmented. |
