# Milestone 17 protocol: junction placement in sequential street generation

* **Status.** Written 2026-11-22 before any M17 diagnosis run. §5 (fix selection), §6 (holdout) and §7 (margins)
  are frozen before the holdout runs; amendments go in §9 with date and reason.
* **Scope.** M16's confirmatory runs are paused and unaffected. All historical samplers, configs, checkpoints and
  results stay reproducible through their entry points: every M17 option defaults to off.

## 1. Question

Does sequential street generation create excess close junctions? If so:
* which edit creates them;
* does topology-aware junction reuse remove the excess while keeping connectivity and legitimate short blocks;
* does a post-hoc topological cleanup explain the same improvement?

## 2. Definitions (`graphs/junctions.py`, `docs/M17_METRICS.md`)

* **Junction:** a node with ≥ 3 unique neighbours, after merging duplicate nodes (< 5 cm) and dropping self-loops and
  repeated edges. Bends and shape points are degree 2. Grade-separated crossings share no node.
* **Focal junctions:** ≥ 40 m inside the evaluation window (minus-sampling edge correction). Their neighbours are all
  junctions of the window, or of the buffered domain when available.
* **close15:** share of focal junctions with another junction at pairwise distance < 15 m.
  * nn p10 = 10th percentile of the focal nearest-neighbour distances.
  * A sweep 5–40 m is always reported, pooled and window-averaged, both labelled, with denominators.
* **consol10:** focal junctions ÷ topology-aware clusters. Two junctions are linked if pairwise ≤ 10 m and an
  at-grade path ≤ 15 m joins them; single linkage, largest diameter reported.
  * This is NOT OSMnx's tolerance, which is a per-node buffer radius: tolerance r groups nodes < 2r apart, checked
    against OSMnx 2.1.1 in the tests.
* **Representations,** always side by side:
  * RAW: the generated graph as produced.
  * REBUILT: passed through the same `build_graph` as the real data (30 m resampling, 8 m greedy snap).
  * Real graphs are REBUILT by construction.

## 3. Development data (recurring terrain pack)

* The named windows of `configs/m16_pack`: KAIST, Seoul Haebangchon, Madrid Lavapiés, El Escorial, on GLO-30; Madrid
  and El Escorial also on MDT05.
* Seeds 1–10, fixed start at the tile centre, backbone TN_s0.
* Matched real references: the same windows' public streets (training road definition), plus the M15 pilot (120
  windows) and the style references for context.

## 4. Diagnosis (step 2; `configs/m17_plans/diagnose.json`)

**Replay.** A, M\*, M13-SMC, NA-myopic, NA-fixed, NA-adaptive, NA-random, NA-SMC and M\* + organic / radial (M16
frozen settings) on the same windows and seeds, with the edit event log (`graphs/junction_events.py`). M12-HR uses
the legacy loop: metrics only, no event log.

**Three hypotheses, separated by matched interventions:**

| | Hypothesis | Test |
|---|---|---|
| H1 | Insertion geometry | On the same states and moves, the edit class of every close junction birth (crossing split, T-snap, merge, u-branch); how often the topology-preserving reuse of the nearby junction was feasible; how often a feasible candidate without a close birth existed |
| H2 | Network reward | NA-fixed and NA-SMC with the dead-end and island weights × {0, 0.5, 1, 2}, same proposals and budget. Close births as a function of the weight, and the share of chosen closures (`lw_net` > 0) creating a close birth, compared with non-closure edits |
| H3 | Density / style | Stop-bias sweep {−0.5, 0, +0.5, +1} for M\* and NA-SMC; organic and radial at stop bias +1 / +2. close15 against focal junction density (matched-density comparison by binning on density), unconditional outcomes, road length, coverage, empty runs |

**Training audit.** Teacher-forced sequences are breadth-first walks over planar real graphs. Every step ends on a
real node, so crossing splits and T-snaps never occur in training, while merges (loop closing) do. This is verified
by code inspection (`graphs/sequence.py`) and stated as such.

## 5. Fixes (step 3), each only if the diagnosis supports it

| Fix | What it does | Calibration |
|---|---|---|
| F0 post-hoc | Topology-aware consolidation of the final graph (cleanup pass baseline) | τ = the F1 radius |
| F1 online reuse | When a candidate would create a junction within r of an existing junction j, it is replaced by the move to j (same insertion rules) if: same level; no duplicate edge or self-loop; the resulting edge ≥ 8 m; turn angle at j ≥ 30° from every incident branch; hard rules pass; j is not already joined to u. Otherwise the original edit stays | r from {6, 9, 12, 15} m on development data (sensitivity reported) |
| F2 reward | Conditional on H2. The closure reward counts only when the closure shortens network distance between its ends by ≥ 25 % (not for a short redundant connector) | heuristic, labelled as such |
| F3 training | Conditional, pilot only, if inference fixes leave a reproducible gap | — |

* **Fixed spacing (ablation):** a hard minimum spacing r on new junctions.
* **Comparisons:** original, F0, F1, F2, F1 + F2 where applicable, with runtime and candidate evaluations. One
  baseline keeps the same computational budget.

## 6. Holdout (frozen before running)

* **Fresh geographic holdout metros** never used in any milestone and not among the declared held-out metros of M12 /
  M15. They are listed in `configs/m17_holdout.json` with the window sampling rule.
* **Real references:** the same windows' OSM public streets (training road definition), matched by road definition,
  window and scale.
* **Uncertainty:** paired windows and seeds; bootstrap over metros (geographic blocks); seeds are repeated draws.

## 7. Outcomes and margins (frozen before the holdout)

**Primary.** Excess close15 = generated − matched real (window-level, REBUILT representation), and its reduction by
each fix. Overcorrection (below the real value) is reported.

**Companions.** nn p10 and consol10, in both representations.

**Preservation, with meaningful-loss margins (frozen in §9 before the holdout):**
* same-land OD unreachable;
* coverage_60;
* street km and junction density;
* share > 15 % grade;
* block median log area and shape;
* Jaccard diversity;
* empty and failed runs.

Unconditional effects are reported alongside matched-density comparisons.

## 8. Rendering

* Matched before / after maps on the same seeds.
* Junction-event overlays (close births coloured by edit class).
* Typical and failure cases, selected by a fixed rule:
  * typical: the median window by excess close15;
  * failure: the window with the largest remaining excess after the fix.

## 9. Amendments and frozen values

(none yet)
