# Milestone 14 ablations

**Scope.** Descriptive only (protocol §8).
* **Setup:** TN_s0, inference seeds 1–2, on DEV‑A (25 tiles) and DEV‑B (27 tiles).
* **Status:** no ablation result was used to change the frozen configuration (λ_D = 1.8143, λ_I = 0, trigger `deg1`, H = 2, K = 8).
* **Source:** every number is from `outputs/m14/REPORT.md` (section "Ablations"), which is generated from the stored runs. Where a DEV‑B variant is the same configuration as a 3‑seed confirm run (frozen, M\*, NA‑myopic, NA‑fixed, random trigger), that run is cut to seeds 1–2 so all rows use the same samples (`M14_FAILURE_ANALYSIS.md` §4.6).

**How to read.** With 25–27 tiles and 2 seeds, differences of about 0.01 in `od_unreach_land` or 0.05 in S10 are within run‑to‑run noise:
* the DEV‑B real‑vs‑real S10 median is 0.42;
* the across‑seed spread of the confirm runs on DEV‑B is up to 0.015 in OD.

So only large or consistent patterns are interpreted.

## 1. Planning strength (λ multiplier; NA‑myopic; also H2)

**DEV‑B**

| m | 0 (= M\*) | 0.5 | 1 (calibrated) | 2 | 4 | real |
|---|---|---|---|---|---|---|
| Dead‑end ratio | 0.048 | 0.025 | 0.013 | 0.007 | 0.004 | 0.025 |
| OD unreach. (land) | 0.014 | 0.005 | 0.007 | 0.001 | 0.000 | 0.011 |
| Islands / tile | 0.13 | 0.11 | 0.04 | 0.02 | 0.00 | 0.04 |
| S10_cond | 0.411 | 0.351 | 0.532 | 0.619 | 0.732 | (noise floor 0.42) |
| Density / real | 1.14 | 1.08 | 1.21 | 1.29 | 1.38 | 1 |

**DEV‑A**

| m | 0 (= M\*) | 0.5 | 1 (calibrated) | 2 | 4 | real |
|---|---|---|---|---|---|---|
| Dead‑end ratio | 0.042 | 0.032 | 0.018 | 0.010 | 0.006 | 0.017 |
| S10_cond | 0.422 | 0.505 | 0.606 | 0.595 | 0.713 | (noise floor 0.40) |

**Reading:**
* **Calibration roughly works.** The calibrated weight (m = 1) puts the dead‑end ratio near the real level:
  * DEV‑A: 0.018 vs 0.017.
  * DEV‑B: 0.013 vs 0.025 (below real; NA‑myopic, not the `na_fixed` sampler the weight was fitted with, see `M14_FAILURE_ANALYSIS.md` §3.3).

  The calibration used training tiles only, and neither set was used for it.
* **Beyond m = 1 the potential over‑regularises.** Dead ends fall far below real, S10 rises, and density grows. This is the predicted cost of strong planning (H2).
* **H2 is still not supported under the pre‑registered rule** (see `M14_STATISTICAL_ANALYSIS.md`). The DEV‑B rise from m = 0 to m = 4 (+0.320) is below the real‑vs‑real median (0.422), although the trend is non‑decreasing in 3 of 4 steps.

## 2. Network potentials

| Variant | DEV‑A OD | DEV‑A dead‑end | DEV‑A islands | DEV‑A S10 | DEV‑B OD | DEV‑B dead‑end | DEV‑B islands |
|---|---|---|---|---|---|---|---|
| Frozen (dead end only, λ_D = 1.81) | 0.102 | 0.022 | 0.14 | 0.504 | 0.000 | 0.016 | 0.00 |
| Island only (λ_I = 1, heuristic) | 0.096 | 0.039 | 0.06 | 0.398 | 0.011 | 0.037 | 0.09 |
| Dead end + island (λ_I = 1) | 0.098 | 0.024 | 0.10 | 0.534 | 0.000 | 0.016 | 0.00 |
| None (= M\*) | 0.105 | 0.042 | 0.24 | 0.422 | 0.014 | 0.048 | 0.13 |

**Reading:**
* **The dead‑end potential carries the effect.** The island potential alone removes islands but leaves the dead‑end ratio at the M\* level.
* **Adding the island potential to the dead‑end potential changes little.** On DEV‑B it changed **no sample at all**: the stored graphs and per‑tile metrics of the two runs are identical; only the sampler's log‑weights differ. This is consistent with Amendment 1: with the dead‑end potential active, islands are already below the real rate, so the calibrated island weight is 0.
* **The DEV‑A OD values (≈ 0.08–0.10) are dominated by two empty tiles.** They are empty for every method, including P (Amendment 2). The non‑empty variant is in the report.

## 3. Terrain objective

* **Setup:** the density‑ratio terrain potentials (M13 `soft`) are switched off; the hard constraints are kept.
* **Connectivity improves slightly.** DEV‑A OD goes 0.102 → 0.087 and DEV‑B 0.000 → 0.001.
* **Realism worsens:**
  * DEV‑A S10 0.504 → 0.612 and S_T 0.411 → 0.582;
  * DEV‑B S10 0.474 → 0.584 and density 1.11 → 1.25.

The terrain potentials and the network potential therefore pull in partly different directions. The terrain potentials are what keeps terrain realism (S_T) at the M13 level.

## 4. Look‑ahead horizon and candidate count (NA‑adaptive)

| Variant | DEV‑A OD (non‑empty) | DEV‑A S10 | DEV‑B OD | DEV‑B S10 | s / tile (DEV‑A) |
|---|---|---|---|---|---|
| H = 1 | 0.0149 | 0.523 | 0.003 | 0.534 | 2.6 |
| H = 2 (frozen) | 0.0237 | 0.504 | 0.000 | 0.474 | 4.8 |
| H = 4 | 0.0069 | 0.502 | 0.004 | 0.446 | 6.2 |
| K = 4 | 0.0164 | 0.441 | 0.013 | 0.510 | 2.4 |
| K = 16 | 0.0077 | 0.574 | 0.005 | 0.553 | 9.3 |

**Reading:**
* **No monotone benefit from a longer horizon or more candidates** at this sample size. This matches toy result C3 (more candidates is not monotonically closer to π) and Lin et al. 2013, Prop. 7.
* **Cost grows roughly linearly in H and K.**

## 5. Trigger choice (NA‑adaptive, same λ)

| Trigger | Fires on (DEV‑A) | DEV‑A OD (non‑empty) | DEV‑A S10 | DEV‑B OD | DEV‑B S10 | s / tile (DEV‑A) |
|---|---|---|---|---|---|---|
| `deg1` (frozen; best VoC capture) | 43 % | 0.0237 | 0.504 | 0.000 | 0.474 | 4.8 |
| random, same rate | 44 % | 0.0164 | 0.427 | 0.002 | 0.434 | 4.3 |
| `island` | 16 % | 0.0045 | 0.487 | 0.002 | 0.589 | 2.1 |
| `water` | 1 % | 0.0018 | 0.539 | 0.002 | 0.522 | 0.8 |
| `steep` | 10 % | 0.0071 | 0.463 | 0.008 | 0.557 | 1.4 |
| `feas_low` | 6 % | 0.0082 | 0.595 | 0.012 | 0.573 | 0.9 |
| everywhere (NA‑fixed) | 100 % | 0.0094 | 0.350 | 0.004 | 0.390 | 8.5 |

**This is the main negative ablation result.**
* **The VoC criterion did not identify the decisions that matter for outcomes.** The trigger chosen by value of computation (`deg1`, 78 % of VoC captured at 43 % of decisions) is no better than random allocation at the same rate on either set. This was also the outcome of the pre‑registered pilot validation and of H3 on GEN2.
* **Cheap triggers do as well or better on DEV‑A connectivity**, at 2–6× lower cost:
  * `water` fires on 1 % of decisions;
  * `island` fires on 16 %.

  These differences are within noise at n = 25 and were not tested. They are reported because they point the same way as the confirmatory result.
* **Interpretation (not tested).** VoC was measured as the total‑variation change in the *next selection* caused by the twist. A large local change in selection probabilities does not imply a change in the final network's connectivity. Many such changes are between near‑equivalent continuations.

## 6. KL trust region (NA‑myopic)

| ε | DEV‑A OD (non‑empty) | DEV‑A S10 | DEV‑A Jaccard | DEV‑B OD | DEV‑B S10 | DEV‑B Jaccard |
|---|---|---|---|---|---|---|
| none (NA‑myopic, m = 1) | 0.0120 | 0.606 | 0.628 | 0.007 | 0.532 | 0.773 |
| 0.05 | 0.0139 | 0.528 | 0.622 | 0.007 | 0.495 | 0.759 |
| 0.2 | 0.0064 | 0.552 | 0.634 | 0.003 | 0.518 | 0.748 |

**Reading:**
* **The trust region reduces the S10 cost of the myopic tilt** (DEV‑A 0.61 → 0.53 at ε = 0.05) **without losing connectivity**.
* **It does not increase diversity.** Jaccard is unchanged within noise.
* **Diversity was never at risk.** The H4 ratio is 0.97 on GEN2 and 0.98 on DEV‑B, so the trust region is not needed for diversity at the calibrated λ. Its use would be as a realism control at larger λ (not tested).

## 7. Inference variants on the confirmation sets (from the main tables)

The confirm runs (TN_s0–s2, seeds 1–3) also act as ablations of the sampler. All values are on GEN2.

| | OD (land) | Dead‑end | S10 | Density / real | s / tile |
|---|---|---|---|---|---|
| M\* (no network potential) | 0.037 | 0.054 | 0.369 | 1.12 | 0.64 |
| NA‑myopic (stop weight only) | 0.010 | 0.022 | 0.501 | 1.34 | 0.86 |
| NA‑fixed (roll‑outs at every decision) | 0.011 | 0.019 | 0.370 | 1.21 | 9.75 |
| NA‑SMC (properly weighted, 4 particles) | 0.012 | 0.014 | 0.332 | 1.11 | 2.74 |
| M\*+repair (post‑hoc A\*) | 0.006 | 0.044 | 0.424 | 1.15 | 0.64 |

**Reading:**
* **Connectivity is the same across the network‑aware samplers.** The single‑decision stop weight already provides it.
* **Realism and density are not the same:**
  * The single‑chain myopic and adaptive variants inflate density and S10.
  * Full roll‑outs or SMC re‑weighting remove that inflation.
  * SMC does so at 28 % of the cost of full roll‑outs.
