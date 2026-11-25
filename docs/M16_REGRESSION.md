# M16 part 1: did terrain adaptation regress? Milestone replay and diagnosis

* **Sources.** Numbers come from `outputs/m16/replay_report.md` (`scripts/m16_replay_report.py`, plan
  `configs/m16_plans/replay.json`, seeds 101–105). The milestone comparison is `docs/M16_MILESTONE_TABLE.md`.
* **Exposure measure.** The share of ground street length steeper than 15 %, over 10 m sub-spans of the true raster,
  unless stated otherwise.
* **CIs.** Two-level bootstrap (sites, then seeds). With six sites per set they are coarse.

## 1. Short answer

**Yes, against the M12 demos, and the cause is a design decision, not a code change.**

* On the demo landscapes:
  * the M12 configuration (M12 loop + hard per-class grade limits with replanning, "M12-HR") gives a > 15 % share
    of **0.050**;
  * every later sampler gives **0.19–0.22**: A 0.221, M\* 0.212, NA-fixed 0.203, NA-SMC 0.194. The paired increase
    over M12-HR is +0.14 to +0.17 (each CI excludes 0).
* The same M12 loop **without** its rules gives 0.212, identical to M\*. The difference is entirely the hard grade
  rules, which M13 replaced with soft potentials (weight 0.5) by design. Those potentials have no measurable effect:
  M\* − A = +0.006 [−0.020, +0.040] on the demo landscapes and +0.006 [−0.026, +0.047] on the analytic sites.
* The code did not regress. Archived M14 runs replay bit for bit with the M16 code: A, M\*, M\*+repair, P and
  NA-fixed reproduce 95/95 graphs (`outputs/m15/audit/replay_m16_*.json`; the remaining labels are in the same
  file). Raster units, interpolation, slopes and model-input orientation pass 68 analytic checks
  (`tests/test_m16_terrain_checks.py`).

## 2. Historical configurations (R1; each milestone's own loop, learned start head)

| Configuration | street km | > 15 % | > 10 % | grade p90 (M13, 30 m) | OD unreachable |
|---|---|---|---|---|---|
| M12-HR (every M12 demo) | 2.66 | **0.050** | 0.210 | 0.126 | 0.112 |
| M12 loop, no rules | 3.64 | 0.212 | 0.360 | 0.176 | 0.012 |
| A (M13 plain sampler) | 2.87 | 0.221 | 0.361 | 0.166 | 0.079 |
| M\* (M13) | 2.66 | 0.212 | 0.350 | 0.160 | 0.004 |
| NA-fixed (M14) | 3.31 | 0.203 | 0.343 | 0.156 | 0.000 |
| NA-SMC (M14) | 3.14 | 0.194 | 0.354 | 0.163 | 0.000 |
| P (procedural) | 3.13 | 0.061 | 0.195 | 0.140 | 0.128 |

**Demo selection.** The M12 single-tile demos kept the densest of four samples; later demos used the first. This
changes length (M12-HR 2.73 → 3.81 km), not exposure (0.059 → 0.055).

**Demo terrain and seeds differed.** The M12 demos used gentler scenarios than the M14 demos and seed 2041. No
scenario-and-seed pair is shared.

**Visual check.** `outputs/m16/figures/fig_replay_demo.png`, seed 101. M12-HR streets avoid the steep hillside
(0.16 against 0.59–0.80) and the mountain pass (0.10 against 0.32–0.48).

## 3. Matched comparisons, one component at a time (R2; fixed start, same seeds and evaluator)

| Contrast (paired) | Analytic sites (sloped + river) | Demo landscapes |
|---|---|---|
| M\* − A (soft terrain potentials + hard rules) | +0.006 [−0.026, +0.047] | +0.006 [−0.020, +0.040] |
| NA-fixed − M\* (network potentials + look-ahead) | −0.027 [−0.077, +0.001] | −0.006 [−0.043, +0.024] |
| M\* + M12 grade limit (`real_p99`) − M\* | −0.044 [−0.111, +0.001] | **−0.115 [−0.225, −0.032]** |
| NA-fixed + M12 grade limit − NA-fixed | −0.021 [−0.053, 0.000] | **−0.156 [−0.306, −0.039]** |
| M\*: TNlp − TN (low-pass terrain input) | −0.033 [−0.083, 0.000] | **−0.089 [−0.197, −0.021]** |
| M\*: TNse − TN (street-erased training DEMs) | −0.048 [−0.111, −0.000] | **−0.146 [−0.334, −0.020]** |
| A: true − flattened input (the model's own terrain response) | −0.056 [−0.140, +0.007] | +0.003 [−0.125, +0.137] |
| M\*: true − flattened input | −0.088 [−0.217, 0.000] | +0.008 [−0.125, +0.166] |
| NA-fixed: true − flattened input | −0.094 [−0.203, 0.000] | −0.038 [−0.184, +0.084] |

**Mismatch control** (analytic sites, the terrain shown rotated 90°): exposure is higher by +0.09 to +0.14 than
with the true terrain (A, M\*, NA-fixed).

### Reading

1. **The hard grade limit restores most of the M12 behaviour, at a cost.** NA-fixed with the limit reaches 0.042 on
   the demo landscapes, close to M12-HR's 0.050. But OD unreachability rises:
   * M\*: 0.042 → 0.177;
   * NA-fixed: 0.008 → 0.056.

   M12-HR also replans (redraw, halve, turn onto the contour, far bank), which the M13+ samplers do not.
2. **The learned model's own terrain response is real but narrow.** On simple analytic terrain (ridge, valley) the
   true input lowers exposure by 0.06–0.09 compared with flattened input. On the procedural demo landscapes it does
   nothing measurable.
3. **The terrain input of the backbone matters more than the sampler.**
   * TNse brings M\* to 0.058 on the demo landscapes and TNlp to 0.115, against TN's 0.204.
   * A plausible explanation, not tested here: TN learned partly from street beds imprinted in urban DEMs (the M12
     leakage), which landscapes without streets do not have.
   * TNse is not free: on the analytic sites 12 % of its M\* networks were empty and OD unreachability rose to 0.12.
   * Protocol amendment 1 adds TNse as a secondary backbone in the confirmatory runs.
4. **Not a regression:** the M14 network potentials (NA-fixed − M\*) do not increase exposure.

## 4. What changed between milestones that can matter

Evidence is in `docs/M16_MILESTONE_TABLE.md` §6:
* hard rules with replanning (M12) against soft potentials (M13+);
* demo selection, terrain and seeds;
* the evaluator. `grade_p90` and `share_gt10` keep their names but changed definition (M12 edge mean against M15
  10 m sub-spans). M16 reports both the M13 30 m measure and the M15 sub-span shares.

## 5. Limits

* 5 seeds × 6 sites per set. The demo landscapes were inspected in earlier milestones; this is a diagnosis, not a
  confirmatory test.
* GLO-30 / MDT05 real terrain is not part of this replay. The M15 real pilot already showed that the DEM source
  changes network length.
