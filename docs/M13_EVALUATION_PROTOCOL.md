# Milestone 13 evaluation protocol (pre-registered)

**Pre-registered 2026-11-06, before any selection or confirmatory experiment was run.**

What existed before registration:
* the 8-tile sampler pilot of `docs/M13_PROGRESS_LOG.md` (a bug and runtime check);
* a 2-seed smoke test of the driver on DEV-A.

No result of either informed the choices below beyond runtime. Later changes go to §10, with date and reason.
Criteria are not changed after results are seen.

Method: `docs/M13_METHOD.md`. Code: `scripts/m13_experiments.py` (generation and metrics), `scripts/m13_select.py`
(staged selection), `scripts/m13_analyze.py` (statistics, tables, figures).

## 1. Research questions

| | Question |
|---|---|
| RQ1 | Does feasibility-aware probabilistic sampling improve terrain realism over unconstrained sampling and deterministic rejection? |
| RQ2 | Does multi-step look-ahead reduce premature termination and fragmentation compared with one-step filtering? |
| RQ3 | Are realism and diversity preserved? |
| RQ4 | Do improvements persist on unseen cities? |
| RQ5 | Can infrastructure be made more plausible? This is secondary and exploratory. |

## 2. Data and units

* **Unit:** the 512 m tile. Per tile, metrics are averaged over inference seeds first.
* **Independent unit for inference:** the geographic block (§6).

| Tile set | Contents | Use |
|---|---|---|
| **DEV-A** | 25 tiles in 13 blocks | selection and calibration only |
| **DEV-B** | 27 tiles in 14 blocks | confirmatory comparisons on development data |

* Both come from the 52 development tiles with a DEM: the M2 `val` tiles with a DEM, plus San Diego `val_t`.
* Blocks: each `val` city is one block; San Diego is cut into 3 km cells. Blocks are shuffled with seed 20261106
  and dealt alternately (`outputs/m13/splits.json`).

**GEN (generalisation; confirmatory, evaluated once with frozen configurations):**
* Cities: Girona and Manresa (Spain; Catalonia OSM extract; IGN MDT05 DEM, CC BY 4.0), and San Francisco (USA;
  NorCal OSM extract, ODbL; USGS 3DEP DEM, public domain). None is in any training or development split.
* Tile choice (fixed now, terrain-blind):
  * all 512 m cells of the M2 grid inside the city boundary with GHSL 2020 built-up ≥ 0.15;
  * at most 30 tiles per Spanish city and 40 in San Francisco, drawn at random (seed 20261106);
  * blocks: 3 km cells.
* If a city cannot be built (licence, access), it is dropped and the reason is documented. No replacement is chosen
  after results.

**Untouched:** `test` and `test_t`.

## 3. Backbones and inference

**Backbones (trained weights; never retrained for a sampler):**

| Backbone | Terrain input | Training seeds |
|---|---|---|
| TN | elev, slope, water | 0, 1, 2 |
| TN-lp | low-pass σ = 16 m (M12 control B) | 0, 1, 2 |
| TN-se | street-erased DEM, factor 4 (control C; `outputs/m13/audits/street_erasure.json`) | 0, 1, 2 |
| N0 | none | 0, 1, 2 |

Seed 2 of TN, TN-lp and N0 and all TN-se seeds are trained in M13 with the unchanged M12 recipe.

**Inference:**
* selection: 2 inference seeds (1, 2);
* confirmation: 4 inference seeds (1–4) per tile per backbone seed;
* T = 0.5 (the displacement temperature), except in method C.

**Potentials** (`configs/m13_potentials/<backbone>.json`) are calibrated per backbone family on training tiles with
that family's seed-0 checkpoint.

## 4. Metrics

**Primary (fixed definitions):**

| | Metric |
|---|---|
| P1 | **S_T** (M12 definition, unchanged): terrain realism |
| P2 | **RC3**: the response-curve distance, the mean of `response_distance` for street length density, median grade and contour alignment. Per landform-slope bin: abs(mean generated − mean real) / IQR of all real values, with **every bin weighted equally** so flat land cannot dominate |
| P3 | **Premature share** (M12: generated street length < 50 % of the real tile's) and **giant share** (length share of the largest connected component) |
| P4 | **S10_cond** (M12 conditional S10, unchanged): street-network realism |

**Secondary:**
* `circuity_all` (unreachable pairs at a detour of 3.0, never excluded), pair accessibility, unreachable share,
  dead-end ratio, intersection density;
* joint realism: energy distance and MMD (RBF, bandwidth = median pairwise distance of the normalised real training
  features) on `m13_metrics.JOINT_KEYS`, normalised by **real training-tile** means and standard deviations;
* infrastructure: bridges per km of street and per km of river, river crossings bridged and unbridged by width class,
  crossing spacing, spans, unsupported structures;
* earthwork proxy: m³/km, W1/IQR against real;
* diversity: pairwise Jaccard distance of 8 m occupancy, orientation L1, across the inference seeds of one tile;
* density: street km/km² against real;
* efficiency: seconds per tile-sample, forward passes, candidate evaluations, hard-rejection rate, ESS, peak memory,
  successful street km per second.

**No composite score is used.**

## 5. Methods compared

Every method uses the same backbone weights:
* **A0** legacy (M12);
* **A** standard;
* **B** top-k;
* **C** full-temperature;
* **D** filter (hard only);
* **E** energy (hard + soft);
* **softonly** (soft only);
* **E+LA**;
* **F** beam;
* **G** SMC;
* **G+LA**.

**Repair:** M* with the procedural reconnection repair (reported separately).

**Baselines:**
* **P**: the M12 procedural least-cost baseline (Poisson junctions, Gabriel graph, slope- and water-costed Dijkstra
  paths; Galin-inspired, without a curvature term);
* **N0** with M*'s sampler: terrain-blind.

**External learned baselines:** none reproducible (literature review §6).

## 6. Statistics

* **Contrasts:** paired (same tiles, same inference seeds). The tile value is the mean over inference seeds; the
  contrast is method minus control. For S_T, S10_cond and RC3 the statistic is recomputed on each bootstrap
  resample of blocks.
* **Bootstrap:** paired **block** bootstrap over geographic blocks, 2,000 resamples, 95 % percentile intervals.
* **Training seeds:** analysed separately; the criteria below say how many must agree. Inference seeds are never
  treated as independent training replications.
* **p-values:** bootstrap two-sided, p = 2·min(P(Δ > 0), P(Δ < 0)).
* **Multiple comparisons:** **Holm–Bonferroni** across the confirmatory family {H1, H1-lp, H1-se, H2, H4}, at
  α = 0.05, using for each the median p over training seeds.
* **Reporting:** effect sizes (absolute and relative), and per-city, per-block and per-terrain-class results
  (flat / moderate / steep by tile landform slope).

## 7. Selection (DEV-A, backbone TN_s0, inference seeds 1–2), staged and automated by `scripts/m13_select.py`

**Stages:**

| Stage | Configurations |
|---|---|
| S1 one-step | A, A0; B with K = 8 and k ∈ {1, 2, 4}; C with T_full ∈ {0.5, 0.7, 1.0}; D with K ∈ {4, 8, 16, 32}; E with K ∈ {4, 8, 16, 32} × λ_ratio ∈ {0.5, 1, 2}; softonly (K = 8, λ = 1) |
| S2 look-ahead | on the best E of S1: H ∈ {1, 2, 4, 8}, M = 4, β = 1 |
| S3 particles | on the best of S1 ∪ S2: SMC N ∈ {4, 8}; beam B = 4; SMC + LA (best H) with N = 4 |
| S4 scheduler | on the best so far: fifo, class, confidence, connect, stochastic (L = 4) |

**Selection rule (applied by script):**
* Eligible configurations: mean DEV-A street density within ±15 % of real, and premature share ≤ that of A + 0.05.
* Among eligible ones, choose the lowest **S_T**. Ties within 0.02 go to the faster configuration (seconds per
  tile-sample).
* If none is eligible, the rule picks the lowest S_T among configurations with premature share ≤ A + 0.05, and this
  is reported as a **failure of the density criterion**.

**Outcome:**
* **M*** = the winner over S1–S4 (the primary method);
* **E*** = the best one-step E of S1 (the control for H2);
* **H*** = the best look-ahead depth.

**Density-matched variants:** for A and M*, the stop bias b ∈ [−2, 2] is calibrated by bisection on DEV-A, at most
6 runs each, so that mean street density is within ±5 % of real.

## 8. Hypotheses and criteria (confirmatory, DEV-B unless stated)

| | Hypothesis | Criterion |
|---|---|---|
| **H1** (RQ1) | M* improves S_T over A on the TN backbone | ΔS_T 95 % CI < 0 for ≥ 2 of 3 training seeds, and a mean relative improvement ≥ 10 %; also not a failure (below) |
| **H1-lp / H1-se** (leakage) | the same with the TN-lp and TN-se backbones | each tested like H1; real-terrain claims require H1-lp **or** H1-se |
| **H2** (RQ2) | E+LA(H*) vs E* | premature share Δ CI < 0 **and** giant share Δ CI > 0, with ΔS_T ≤ + 0.05, for ≥ 2 of 3 TN seeds |
| **H3** (RQ3) | realism and diversity | S10_cond(M*) ≤ S10_cond(A) + 0.10, **and** mean Jaccard diversity(M*) ≥ 0.8 × diversity(A), for ≥ 2 of 3 TN seeds |
| **H4** (RQ4) | H1 on GEN | ΔS_T CI < 0 for ≥ 2 of 3 TN seeds (and reported for TN-lp and TN-se) |
| **H5** (RQ5, exploratory) | infrastructure | unbridged river crossings and abs(bridges − real) per tile, M* vs A; described, no pass / fail |

**Density-matched check:** H1 is re-tested with the density-matched A and M*. It is reported, but it does not
change the H1 verdict.

**Failure criteria (any method):** a method is **not successful**, whatever its S_T, if on the evaluation set:
* its street density deficit against real exceeds 25 %; or
* its premature share exceeds twice that of A.

## 9. Ablations (DEV-B, TN_s0 unless stated; one hypothesis each)

| Ablation | Tests |
|---|---|
| no terrain conditioning (N0 + M*) | does terrain *input* matter beyond constrained sampling? |
| original vs low-pass vs street-erased (TN / TN-lp / TN-se + M*, 3 seeds) | leakage sensitivity |
| A (none) / D (hard only) / softonly / E (hard + soft) | the role of each constraint type |
| E (H = 0) / E + LA (H = 1) / E + LA (H*) | the role of the look-ahead horizon |
| λ_infra = 0 | the role of the structure potentials |
| fifo vs stochastic scheduler | fixed vs stochastic frontier |
| M* vs M* + repair | what procedural repair adds (length, intersections, bridges, connectivity, realism) |

**Global vs windowed (X5, secondary):**
* Terrain: 3 × 3 procedural landscapes (rolling_hills, high_hills, steep_hillside, narrow_river, mountain_pass;
  3 seeds each).
* Compared: M12 windowed (legacy), global-standard and global-M*.
* Metrics: fragmentation (components, giant share), street density, and seam metrics (density and dead-end ratio as
  a function of distance to the 384 m window boundaries).
* Synthetic terrain, so it is robustness only.

## 10. Compute budget

* Measured pilot rates: standard about 0.3 s per tile-sample; E (K = 8) about 0.7 s; E + LA (H = 2) about 6 s;
  SMC (N = 4) about 2.4 s.
* Training: about 50 min per run, sequential, 9 runs.
* Budget cap: 36 h of active compute (runner). Stages are ordered by decisiveness: selection → confirmation (TN) →
  leakage backbones → GEN → ablations → X5.

## 11. Changes after pre-registration

None yet.
