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

**Amendment 1 (2026-11-22).** Made after the diagnosis (`outputs/m17/diagnosis.md`) and before any fix run.

**Diagnosis verdicts:**
* **H1 supported.** T-snaps and crossing splits make up 45–63 % of the close pairs. Reuse was feasible in 52–88 % of
  the split cases. u-branch births account for 23–41 % and are not addressed by F1.
* **H2 partly supported.** Close junctions rise with the network weight. Chosen closures are only 0.1–0.2 % of edits,
  so the mechanism is the dead-end STOP penalty (density), not closure rewards.
  * F2 is therefore NOT adopted; it is run on development data as a check only.
* **H3.** At matched density, A / M\* / M13-SMC / NA-fixed / NA-random show no excess. The NA SMC, myopic and
  adaptive variants show +0.03–0.05; organic +0.05 at a realistic density.
* **F3 (training) not run.** Training graphs already carry an 8 m snap, and the gap is attributable to
  generation-only edits.

**Development fix runs** (`configs/m17_plans/fixdev.json`):
* M\*, NA-fixed, NA-SMC, M\* + organic, M\* + radial;
* F1 reuse at r ∈ {6, 9, 12, 15} m;
* fixed-spacing ablation at r ∈ {9, 12, 15} m;
* F2 check (closure gain 0.25) for NA-fixed and NA-SMC;
* F0 post-hoc consolidation of the original graphs at τ ∈ {6, 9, 12, 15}.

All use the same sites, seeds and candidate budget (K = 8).

**Preservation margins** (frozen now), each a mean change against the original on the same windows and seeds:

| Measure | Margin |
|---|---|
| same-land OD unreachable | ≤ +0.02 |
| coverage_60 | ≥ −5 % relative |
| street km | ≥ −10 % relative |
| empty share | ≤ +0.05 |
| share > 15 % grade | ≤ +0.02 |
| block median log area | \|Δ\| ≤ 0.25 |

**F1 radius rule** (one r for all samplers):
* the smallest r whose mean reduction of the REBUILT excess-vs-window close15, over the five methods, is ≥ 50 %
  of the excess, with every margin met for every method;
* otherwise the r with the largest mean reduction meeting all margins;
* ties go to the smaller r.

If no r meets the margins, F1 is reported as failing on development data and the holdout tests F0 and the spacing
ablation only.

**Frozen before the holdout (2026-11-22).**
* F1 radius **r = 15 m**, by the amendment-1 rule: 51 % mean reduction on development data with all margins met.
  This is the largest radius tested.
* Holdout comparisons: original, F1 r = 15, spacing ablation r = 15, F0 post-hoc τ = 15 (as declared, τ = the F1
  radius), and F0 τ = 9 as a labelled secondary.
* F2 is not run on the holdout. On development data it changed nothing for NA-fixed (the closure condition never
  fired) and only noise for NA-SMC.
* Seeds 101–103 for all 54 holdout windows.

**Amendment 2 (2026-11-22).** Made after the holdout, at the owner's request: generated networks have too many 5+-way
junctions. Real holdout windows have 0.7 % of junctions with 5+ branches; generated ones 2.3–12 %, and F1 / F0 do not
lower it.

* **F4 degree prior** (`graphs/junction_reuse.degree_lw`): a soft potential from the training junction-degree
  distribution (`configs/m17_degree_prior.json`). Raising a node to degree k ≥ 5 costs λ · log(π(k)/π(k−1)):
  −3.45 nats at k = 5, more for further branches. It applies to every candidate, F1-redirected moves included.
* **λ rule** (development windows, seeds 1–10, combined with F1 r = 15): the smallest λ ∈ {0.5, 1, 2} that meets all
  three conditions:
  * the mean over the five samplers of the pooled 5+-way share of focal junctions (REBUILT) is ≤ 2 × the training
    share (2 × 1.4 % = 2.8 %);
  * all amendment-1 margins are met;
  * the excess close15 is not worse than F1 alone by more than 0.02 for any sampler.

  If none qualifies, the λ with the lowest 5+-way share meeting the margins.
* **Holdout runs with the chosen λ:** F4 alone and F1 + F4, same windows and seeds. These are labelled
  **EXPLORATORY**, because the holdout is being reused after its first evaluation.
* **Additional outcomes:** the pooled 5+-way and 4-way shares of focal junctions.

**Amendment 2 result (development, 2026-11-22).**
* **No λ met the rule.** F4 lowered the 5+-way share to the real level (1.4 %) or below for every sampler.
* **But for NA-SMC and M\* + organic it enlarged blocks beyond the block-area margin** (Δ log area +0.36 to +0.53).
  For organic it also cut street length by more than 10 % (it partly reverses organic's runaway density).
* M\*, NA-fixed and radial met all margins at every λ.
* **No λ is selected.** At the owner's request a holdout comparison is run at λ = 1 (the middle value), labelled
  EXPLORATORY and NOT rule-selected.

