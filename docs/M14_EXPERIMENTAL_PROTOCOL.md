# Milestone 14 experimental protocol (pre‑registration)

* **Registered:** 2026‑11‑17, before any confirmatory M14 run. It is committed on branch `milestone-14`, and the commit timestamp is the registration record. Amendments are appended as dated sections and never edit the text above them.
* **Scope:** development data only. The sealed test splits (`test`, `test_t`) are never read. Running them needs explicit authorisation from the project owner.
* **Context:** the method is defined in `docs/M14_MATHEMATICAL_FORMULATION.md`.

## 0. Disclosure: what was seen before this protocol

The M14 problem statement comes from a phase‑1 audit (`outputs/m14/audit/`). That audit was first run on the **stored M13 outputs of DEV‑B and GEN**, and then replicated on DEV‑A:

* closed‑window islands per tile: real 0.04–0.05 vs generated 0.18–0.31;
* interior dead‑end ratio: real 0.017–0.025 vs generated 0.037–0.050;
* same‑land unreachable OD pairs: about 3–8× real.

DEV‑B and GEN were therefore seen at the level of A and M\* summaries. No M14 sampler has been run on them.

To keep the confirmation clean, the **primary confirmatory set is GEN2**, four new cities built after the audit and never inspected (§2). DEV‑B and GEN results are reported as secondary.

The pilot (`configs/m14_plans/smoke.json`: DEV‑A, TN_s0, one seed) was run before this protocol, to time the samplers.

## 1. Questions and hypotheses

Primary research question: does adaptive network‑aware planning improve the trade‑off between connectivity, realism, terrain feasibility, diversity and cost, compared with fixed planning and with the M13 constrained samplers?

| ID | Hypothesis (as in the M14 brief) | Operational test (GEN2 primary; DEV‑B, GEN secondary) |
|---|---|---|
| H1 | Local constraints are insufficient; limited network‑aware planning improves connectivity | `na_fixed` − M\* on `od_unreach_land` < 0 (superiority) |
| H2 | Strong planning damages realism | Dose–response on DEV‑B (TN_s0, seeds 1–2): S10_cond at λ multipliers m ∈ {0, 0.5, 1, 2, 4}. **Supported** if S10(m = 4) − S10(m = 0) exceeds the real‑vs‑real S10 noise floor (§5.4) and the trend over m is non‑decreasing in ≥ 3 of 4 steps |
| H3 | Adaptive planning improves the trade‑off | `na_adaptive` vs `na_fixed`. (a) Non‑inferior on `od_unreach_land`, margin +0.010 absolute. (b) Non‑inferior on S10_cond, margin +0.02. (c) Mean seconds per tile‑sample ≤ 0.7 × fixed. **And** `na_adaptive` vs `na_random` (random trigger at the same firing rate): `od_unreach_land` difference < 0, i.e. the *choice* of where to plan matters, not only how much |
| H4 | Diversity matters | Jaccard diversity of `na_adaptive` ≥ 0.9 × that of A (non‑inferiority) |
| H5 | Generalisation requires disentangling terrain from artefacts | The H1 effect (`na_adaptive` − M\* on `od_unreach_land`) is negative with its 95 % CI excluding 0 in ≥ 2 of 3 training seeds for **each** of TNlp and TNse on GEN2. The synthetic shifts (§2) are descriptive only |

**Realism guard.** H1 and H3 count as supported only if realism does not degrade: S10_cond of the method ≤ that of M\* + 0.05, and `density_ratio` within ±25 % of real. Otherwise the result is reported as "connectivity bought with realism".

**Feasibility guard.** Unbridged river crossings per tile ≤ M\* + 0.05, and hard violations 0.

## 2. Data

### Geography roles

| Role | Set | Tiles / blocks | Use |
|---|---|---|---|
| Training | M12 `train` | — | Network training (unchanged checkpoints). `train_cal`: 48 training tiles with a DEM, drawn with seed 20261117, used **only** for λ calibration |
| Development | DEV‑A | 25 / 13 | Pilot, trigger validation, stop‑bias calibration for density matching |
| Confirmation (secondary) | DEV‑B | 27 / 14 | Seen in M13 (A, M\* summaries) |
| Generalisation (secondary) | GEN | 100 / 24 (Girona, Manresa, San Francisco) | Seen in M13 (A, M\* summaries) |
| **Generalisation (primary)** | **GEN2** | 95 / blocks of 3 km | Alcoi, Tarragona, Santa Barbara, Sacramento. Built by `scripts/m14_build_gen2.py`, with the M13 rule (GHSL ≥ 0.15, random cap 25 per city, seed 20261106, no terrain screening). Never inspected before the confirmatory runs |
| Sealed test | `test`, `test_t` | — | **Not used** |

### GEN2 city choice

The four cities were chosen *before* any of their data was seen. Each lies outside every earlier split, has a cached OSM extract and an open DEM (IGN MDT05 CC BY 4.0, USGS 3DEP public domain), and was picked for contrasting terrain by known geography: a mountain gorge town, coastal moderate relief, coast with foothills, and a flat river city.

### Distribution shift (descriptive)

DEV‑B and GEN2 with transformed terrain, applied to both the model input and the rules and metrics:
* `@x2`: relief doubled about the tile mean;
* `@rough`: 4 m‑rms fractal noise (spectrum ∝ k^−1.5).

Real streets are kept as the reference, which is a *mismatch by construction*. These sets test robustness only and **do not establish real‑world generalisation**.

### Terrain strata (per tile, from the original terrain, defined before evaluation)

* **Slope classes,** from the mean 64 m‑cell landform slope `slope_mean_ctx`: flat < 0.05; moderate 0.05–0.10; steep 0.10–0.15; mountainous ≥ 0.15.
* **River:** river pixels ≥ 1 % of the tile.
* **Coastal:** any sea pixel in the tile.
* **Mixed:** any water ≥ 1 % and slope ≥ 0.10.

Strata overlap (river, coastal and mixed cut across the slope classes). Results are reported per stratum with tile counts, and are descriptive.

## 3. Methods compared

All methods share three things:
* the same frozen checkpoints (TN_s0–s2; TNlp and TNse for H5);
* T = 0.5 and K = 8;
* the M13 terrain potentials at λ_r = 0.5.

| Label | Preset | Definition |
|---|---|---|
| A | `standard` | Unconstrained ancestral sampling (M13 control A) |
| M\* | `energy` {K 8, λ_r 0.5} | M13 frozen selection (single chain, terrain only) |
| M13‑SMC | `smc` {K 8, λ_r 0.5, N 4} | Strongest correctly characterised M13 sampler (properly weighted) |
| P | `P` | M13 procedural baseline (terrain‑costed Dijkstra on a Gabriel graph) |
| M\*+repair | M\* + `repair` | M13 procedural connectivity repair (post‑hoc shortest paths between components) |
| NA‑myopic | `na_myopic` | M\* + network potentials (λ_D, λ_I), no look‑ahead |
| NA‑fixed | `na_fixed` | + roll‑out look‑ahead at **every** decision (H = 2, M = 4) |
| NA‑adaptive | `na_adaptive` | + look‑ahead only where the frozen trigger fires (§4) |
| NA‑random | `na_adaptive` with `random:p` | Look‑ahead at a random share p of decisions, with p = the frozen trigger's DEV‑A firing rate |
| NA‑SMC | `na_smc` | N = 4 particles, network and terrain potentials, properly weighted (asymptotically π_λ) |
| NA‑TR | `na_myopic` + `kl_eps` | Per‑decision KL trust region (exploratory; ablation only) |

**λ_D, λ_I.** Calibrated once by generalized iterative scaling on `train_cal`, with `na_fixed`, TN_s0, seeds 1–2 (`scripts/m14_calibrate.py`), then frozen in `configs/m14_lambda.json`. Every NA method uses the same λ, so fixed vs adaptive differ **only** in where roll‑outs run. Transfer check: the calibrated λ is evaluated on `train_cal` with TN_s1 and TN_s2 and the moment mismatch is reported (no re‑calibration).

**Equal generation opportunity.** Every method uses the same start head, node budget, MAX_T, max_rounds and K. Planning changes selection only.

## 4. Trigger selection (DEV‑A only, frozen before confirmation)

1. **Log the VoC.** Run `na_fixed` on DEV‑A (TN_s0, seeds 1–2) with VoC logging. Each decision records its context features (`island`, `deg1`, `water`, `steep`, `feas_low`, `w_spread`, `p_stop`) and VoC = TV(selection with twist, selection without twist).
2. **Candidate triggers.** Each single binary feature; `w_spread > q` for q at its 50th and 75th percentiles; and all pairwise ORs of these.
3. **Selection rule.** Among candidates that capture ≥ 70 % of the total VoC, choose the one with the smallest firing share. If none captures 70 %, choose the one with the largest VoC captured per firing share, and flag it.
4. **Validation (DEV‑A, decides whether `na_adaptive` is run at all).**
   * The VoC captured by the chosen trigger must exceed that of a random trigger at the same firing share, using the share × total expected VoC.
   * In addition, `na_adaptive` must reach `od_unreach_land` ≤ that of `na_random` on DEV‑A.
   * If either check fails, adaptive planning is reported as **not justified**, `na_adaptive` is still run on the confirmation sets (H3 is pre‑registered), and the failure is documented.
5. **Freeze.** The chosen trigger and the firing share p used for `na_random` are written to `configs/m14_frozen.json` before any confirmatory run.

## 5. Outcomes

### 5.1 Primary outcomes (fixed now)

| Dimension | Metric | Direction |
|---|---|---|
| Connectivity | `od_unreach_land`: share of street‑node pairs 150–500 m apart on the same dry‑land region with no path on the window‑closed graph (empty network = 1) | Lower |
| Realism | S10_cond (M13 definition: mean W1/IQR to the real tiles of the same set over 10 network statistics) | Lower; non‑inferiority |
| Feasibility | Unbridged river crossings per tile | Lower; non‑inferiority |
| Diversity | Mean pairwise Jaccard distance (8 m occupancy) between inference samples of a tile | Ratio to A |
| Cost | Seconds per tile‑sample (CPU, 2 threads) | Lower |

### 5.2 Secondary outcomes (reported, no decisions)

* **Network:** islands, island share, closed giant share, interior dead‑end ratio (vs real), `od_unreach_water`, OD detour, coverage of buildable land within 60 m (buildable: dry and landform slope < 0.30; the threshold is an assumption), street metres per connectivity.
* **Realism and terrain:** each S10 component separately; S_T and its components; RC3; energy distance and MMD (M13 JOINT_KEYS); street density and density ratio; intersections per km²; class shares.
* **Infrastructure:** bridges.
* **Premature termination:** M13 definition.
* **Diversity:** orientation diversity (L1 of length‑weighted orientation histograms); the across‑seed coefficient of variation of density, giant share and dead‑end ratio per tile.
* **Cost:** forward passes, candidate evaluations, roll‑out steps, roll‑out firing share, peak RSS, ESS (SMC), and km of street per second.

### 5.3 Real‑vs‑real calibration

For each set, the real tiles are split into two halves by geographic block (all 2^B / 2 block bipartitions when B ≤ 16, otherwise 2,000 random ones). The S10_cond between the halves gives the **noise floor**: a generated set within this distance of real cannot be distinguished from another real set of the same size. The same split is used for each S10 component.

### 5.4 H2 noise floor

H2 uses the median real‑vs‑real S10 distance on DEV‑B.

## 6. Statistics

### Units

The geographic block is the independent unit. Training seeds share training data and are **not** independent replicates.

### Primary analysis (per set)

1. For each method, tile and training seed, average the per‑tile metric over inference seeds (1–3).
2. Sum the paired method difference over the tiles of each block.
3. Average over the 3 training seeds. The result is one difference per block, S_b.
4. The estimate is Σ_b S_b / N_tiles.
5. **Test:** exact sign‑flip randomisation over blocks (all 2^B sign patterns if B ≤ 16, otherwise 20,000 random), two‑sided. This follows Canay, Romano & Shaikh's sign‑change randomisation for few clusters. With B blocks the smallest attainable p is 2^(1−B).
6. **CI:** 95 % block‑bootstrap percentile interval (2,000 resamples).

### Set‑level metrics

For S10_cond and the other distribution‑level metrics: the paired block bootstrap (M13 `contrast`) gives the CI, and a within‑block method‑label randomisation test (2,000 permutations) gives the p‑value.

### Non‑inferiority

A non‑inferiority claim requires the upper 95 % CI limit of (method − reference) to be ≤ the margin.

### Per seed

Every contrast is also reported per training seed, and per city.

### Multiplicity

Holm's procedure at α = 0.05 is applied over the GEN2 primary tests {H1, H3‑vs‑random, H5‑TNlp, H5‑TNse}. The non‑inferiority components of H3 and H4 are decided by their CIs. DEV‑B and GEN are secondary and not corrected; they are labelled as such.

### Sample sizes

* Inference seeds 1–3, so 3 samples per tile and 3 diversity pairs per tile.
* Training seeds 0–2.
* H5 runs and density‑matched runs use inference seeds 1–2.

## 7. Density control

### (1) Intervention

A stop‑logit bias per method is calibrated on DEV‑A (TN_s0, seeds 1–2) by bisection to a density ratio of 1.00 ± 0.05 (M13 procedure). Re‑runs on GEN2 (TN_s0–s2, seeds 1–2) are compared on the primary outcomes.

### (2) Adjustment

On the unmatched runs, the paired per‑tile difference of each primary per‑tile metric is regressed on the paired difference in log density. The intercept (the difference at equal density) is reported with a block bootstrap.

### Interpretation

Matching on a post‑treatment variable can introduce selection bias. Both (1) and (2) are reported next to the unmatched results, and the conclusion requires agreement in sign.

## 8. Ablations

Ablations run on DEV‑A and DEV‑B with TN_s0, seeds 1–2. They are descriptive and are never used to change the frozen configuration.

| Group | Variants |
|---|---|
| Network potentials | No network potential (= M\*); dead‑end only; island only |
| Terrain objective | Terrain potentials off (`soft` = False; hard constraints kept) |
| Look‑ahead | H ∈ {1, 2, 4} |
| Candidates | K ∈ {4, 8, 16} |
| Planning strength | λ multiplier m ∈ {0, 0.5, 1, 2, 4} (also H2) |
| Triggers | Each single feature, and random at the same rate |
| Diversity | Trust region ε ∈ {0.05, 0.2} |

## 9. Failure rule

A method is flagged as **failed** on a set if any of the following hold:
* premature share > A's + 0.10;
* density ratio outside [0.6, 1.6];
* any hard‑constraint violation.

Flagged results are reported, never dropped.

## 10. Compute budget

* **Hardware:** Apple M5 Pro, 15 CPU cores, 48 GB RAM.
* **Execution:** inference runs on CPU workers with 2 threads each, at most 5 in parallel, and no MPS training. The parallel workers are inference‑only and small (< 2 GB each).
* **Budget:** ≈ 10 CPU‑worker‑hours for confirmation, ablations, H5, shifts and density matching. If exceeded, the priority order is:
  1. GEN2 primary (TN);
  2. H5 on GEN2;
  3. DEV‑B;
  4. GEN;
  5. ablations;
  6. shifts.

  Anything not run is listed as not run.

## Amendment 1 (2026‑11‑17, after λ calibration, before any trigger selection or confirmatory M14 run)

**Calibration result** (`configs/m14_lambda.json`, `outputs/m14/calibration/log.json`): the pre‑registered generalized‑iterative‑scaling procedure converged after 4 updates, at **λ_D = 1.814** and **λ_I = 0**.

| Rate per tile | Generated | Real |
|---|---|---|
| Dead ends | 2.15 | 2.27 |
| Islands | 0.06 | 0.10 |

The island weight reached its lower bound. With the dead‑end potential active, generated islands already fall below the real rate, so matching E[I] exactly would require a *negative* weight, which rewards fragments and is excluded by the clip to [0, 6].

**Consequences:**
1. The frozen network objective is the dead‑end potential alone. The island potential stays implemented and tested, but its calibrated weight is 0.
2. **§8 ablation change.** "Dead‑end only" *is* the frozen method. The island potential is instead ablated at a fixed, **heuristic** λ_I = 1, in two variants: island only (λ_D = 0, λ_I = 1) and dead end + island (λ_D = 1.814, λ_I = 1). Both are labelled as not calibrated.

Nothing else changes.

## Amendment 2 (2026‑11‑17, after the DEV‑A pilot, before any confirmatory M14 run)

### Trigger and validation

* **Frozen trigger:** `deg1`, i.e. plan only when the node being expanded is an interior road end, so that STOP would create a dead end (`configs/m14_frozen.json`). On DEV‑A it fires on 43 % of decisions and captures 78 % of the value of computation, 1.8× random allocation.
* **Pilot validation (protocol §4, step 4): FAILED.** On DEV‑A, `na_adaptive` was not better than `na_random` at the same firing rate:

  | | `na_adaptive` | `na_random` |
  |---|---|---|
  | `od_unreach_land` | 0.102 | 0.095 |
  | Dead‑end ratio | 0.022 | 0.017 |

  Adaptive planning is therefore recorded as **not justified by the pilot**. As pre‑registered, it is still run on the confirmation sets and H3 is still tested.

### New secondary outcome

On DEV‑A the same 2 of 25 tiles are empty for every method, including P. Empty tiles count as `od_unreach_land` = 1 and dominate that primary outcome.

* **Added (secondary only):** `od_unreach_land_nonempty`, the mean `od_unreach_land` over tile‑samples with a non‑empty network, together with `empty_share`.
* **Unchanged:** the primary outcome and every decision rule.

### Execution order and budget

Running every planned run would exceed the 10 worker‑hour budget (estimate ≈ 18 worker‑hours, because `na_fixed` costs about 8.5 s per tile‑sample under load). Runs execute in the pre‑registered priority order:
1. GEN2 confirm;
2. H5;
3. DEV‑B confirm;
4. ablations;
5. GEN confirm;
6. shifts;
7. density‑matched runs.

Anything not completed will be listed as not run.
