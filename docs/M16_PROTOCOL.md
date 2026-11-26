# Milestone 16 protocol: terrain adaptation and recognisable street patterns

* **Status.** Written 2026-11-21 before any development-sweep or confirmatory result. Amendments go in §10 with date
  and reason. Earlier milestones' protocols are unchanged; the M15 critique is `docs/M15_POSTHOC_CRITIQUE.md`.
* **Labels.** Every M16 result is labelled **development**, **confirmatory (synthetic)**, **recurring comparison
  (named real sites)** or **exploratory**.

## 1. Questions

1. **Regression.** Did terrain adaptation get worse between M12 and M15? (Part 1; diagnosis, not a hypothesis test.)
2. **Hypothesis H.** The learned turtle sometimes runs out of candidates that satisfy both terrain and street-pattern
   constraints; targeted proposals recover viable continuations at comparable compute and coverage.
3. **Recognisability.** Are the resulting street patterns recognisable at useful coverage?
4. **Added value.** What does the method add over the strongest matched baseline?

## 2. Sites and seeds

| Set | Role | Content |
|---|---|---|
| `syn_dev` | development | the six M15 development conditions (flat, incline 8 %, incline 15 %, ridge, valley, river) |
| `syn_eval` | replay and diagnosis only | the M15 evaluation sites (inspected in M15, so not confirmatory) |
| `demo_scen` | replay only | the procedural landscapes of the M12 / M14 demo cities |
| `syn_conf` (+ `_mismatch`, flattened input) | **confirmatory** | fresh instances of the six conditions, declared in `experiments/sites.py` before any M16 run |
| named real sites (`configs/m16_pack`) | recurring comparisons and demos | KAIST centre window and campus-wide domain, Seoul Haebangchon, Madrid Lavapiés, El Escorial |
| style references (`configs/m16_style_refs`) | threshold calibration (`calib`) and blinded sorting (`sort`) | real OSM windows labelled by documented typology; no held-out metro, no USA |

* **Seeds.** Development 201–205. Confirmatory 301–320 (20 seeds; inference seeds are repeated samples of the same
  site). Replays 101–105.
* **Real-terrain confirmation is not run.** It needs the held-out metros, which need the owner's authorisation. The
  named sites were chosen for demonstration; one is a deliberately steep stress test.
* **Seoul.** It was a declared held-out metro in M15. The M16 brief names a Seoul site explicitly, so that one window
  is used as a recurring comparison. The M15 Korean held-out pool therefore needs a replacement metro before any
  confirmatory real benchmark.

## 3. Methods

Samplers M\* and NA-fixed, each in four modes, all at temperature 0.5 with a fixed start at the tile centre:

| Mode | Settings |
|---|---|
| off | no style |
| rerank | M15 heading reranking (`configs/m15_style.json`) |
| proposals | rerank + style-aware proposals (`prop_mode = style`) |
| proposals + topology | rerank + proposals + topology rules (`prop_mode = style_topo`) |

* **Proposal weighting.** Heuristic, with pattern strength α (`graphs/style_proposals.py`). It is a heuristic
  selection with no inference guarantee.
* **Additional comparisons (confirmatory set).**
  * **Matched-compute unguided pool.** M\* and NA-fixed, rerank, with K = 8 + the mean number of style candidates
    per decision measured in the matching proposals run (rounded up).
  * **Rule-only ablation.** Proposals + topology with the learned candidates removed (`prop_rule_only`).
  * **Importance-weighted variant.** Proposals + topology with `prop_weight = is` (target unchanged).
  * **Trigger.** M\* only: adaptive trigger `deficit:2` against always-on and against a random trigger at the
    adaptive trigger's measured firing rate.
  * **Cul-de-sac dead-end ablation.** NA-fixed cul-de-sac proposals + topology with and without
    `dead_exempt_local`.
  * **SMC samplers.** M13-SMC and NA-SMC, style off and proposals + topology.
* **Separate sweeps, after the confirmatory runs.** Displacement temperature {0.35, 0.5, 0.8} and stop bias
  {−0.5, 0, +0.5}, on M\* grid and cul-de-sac proposals + topology. Secondary.

## 4. Development selection rule (declared before the sweep)

**Grid.** α ∈ {0.5, 0.8} per (style, sampler, mode), on `syn_dev` with seeds 201–205.

**Constraints.**
* empty share ≤ style-off empty share + 0.05;
* mean `coverage_60` ≥ 0.9 × style off;
* mean seconds per seed ≤ 3 × rerank.

**Objective.** Maximise the style's controllability check, which is a rule-level quantity, not a confirmatory
outcome:

| Style | Controllability check |
|---|---|
| grid | `field_alignment` |
| organic | `field_alignment` |
| radial | radial + tangential share |
| cul-de-sac | `deadend_per_km` |

**Tie-breaks.** Within 0.01 of the best, the smaller α. If no setting meets the constraints, the one with the highest
coverage.

**Topology parameters** come from the `calib` real references, not from the sweep:
* grid lattice spacing and ring spacing = √(median block area) of the grid references, rounded to 8 m;
* cul-de-sac `branch_len_m` = median terminal length ÷ 3 of the cul-de-sac references, clipped to [25, 60] m;
* `term_depth` = 3;
* spokes 8; continuation strength γ = 2 nats.

They are fixed before the sweep.

## 5. Outcomes (confirmatory, `syn_conf`, 20 seeds)

Independent network outcomes (`graphs/network_outcomes.py`, `docs/M16_METRICS.md`) are **not** optimised by any rule:

| Pattern | Primary outcomes (direction expected) |
|---|---|
| grid | `block_rect_share` ↑, `stroke_xjunc_lw` ↑ (multi-block continuity), `share_4way` ↑, `circuity_200_350` ↓ |
| organic | `block_logA_sd` ↑, `jangle_right_share` ↓, `share_3way` ↑, `block_phi_med` ↓ |
| radial | `spokes_long` ↑, `ring_cover_max` ↑, `sector_circ` ↓ |
| cul-de-sac | `tree_len_share` ↑, `term_attached_share` ≥ 0.9, `od_unreach` ≤ style off + 0.02 |

* **Controllability checks** (reported, never used as evidence of a pattern): field alignment, R4, radial share,
  dead ends per km.
* **Recognisability (primary).** A nearest-centroid classifier is fitted on the `calib` references only. It uses the
  union of the primary outcomes, standardised by the pooled within-style SD of the references. A generated network
  is "recognised" when the classifier returns the requested style.
  * **Threshold for "recognisable".** Recognition rate ≥ 0.5 (twice chance), and the lower 95 % bound above the rate
    of the same sampler in rerank mode.
  * **"At useful coverage".** Mean `coverage_60` ≥ 0.8 × style off, and empty share ≤ style off + 0.05.
  * **Sanity check.** The classifier's leave-one-out accuracy on the `calib` references is reported. If it is below
    0.6, recognisability is reported as **inconclusive**.
* **Blinded style sorting (people).** A kit with standardised map styling is prepared:
  * real `sort` references mixed with generated maps from predefined seeds;
  * options grid / organic / radial / cul-de-sac / mixed / uncertain;
  * answer key stored separately.

  It is run only when raters are available. Until then it is reported as not run.
* **H, per decision.** Logged in the run files:
  * decisions where no feasible learned candidate lies within 15° of the pattern (`no_aligned_model`);
  * the share of those where a feasible style candidate exists (`rescued_aligned`);
  * the source of every accepted move;
  * evaluations per decision and seconds.

  H is supported if proposals + topology raises the recognition rate over the matched-compute unguided pool (paired
  by site and seed, lower bound > 0) at comparable coverage.
* **Terrain.** On the five sloped / river conditions:
  * share of ground street > 15 % and > 10 % (10 m sub-spans);
  * M13 `grade_p90_m13` and `contour_align`;
  * bridges, street km, `coverage_60`, empty count;

  each for true input, flattened input and 90° mismatched input under the same evaluation terrain. Outcomes are
  reported raw and density-matched: regression on log street km, and the share metrics, which are per km.
* **Connectivity and diversity.** `od_unreach` (fixed point sampling; points in water excluded), circuity,
  Jaccard diversity across seeds, and coverage of the real-reference metric distributions.

## 6. Statistics

* Paired differences (same site and seed) with a two-level bootstrap: sites resampled, then seeds within site;
  5 000 resamples.
* Per-site results are always shown.
* With six synthetic sites, between-site uncertainty is coarse; CIs are labelled accordingly.
* No multiplicity correction is claimed. Each primary outcome is reported with its CI, and the verdicts follow the
  rules above without re-definition.

## 7. Part 1 replay and regression diagnosis

* **First, historical configurations** (`configs/m16_plans/replay.json`, block R1): each milestone's own loop and
  start rule on the demo landscapes.
* **Then matched comparisons** (block R2): same sites, seeds, fixed start, budget and evaluator.
* **One component swapped at a time:**
  * the M12 hard grade limit alone (`grade_cap = real_p99`);
  * the M12 loop against committed-step sampling (A);
  * terrain input (TN / TNlp / TNse; flattened; mismatched);
  * densest-of-4 against first-sample selection, computed from the same runs.
* **Bit-identity replays** of the archived M14 runs: `scripts/m15_replay_baseline.py` with all labels.
* **Units and slopes** are checked against analytic terrain (`tests/test_m16_terrain_checks.py`).
* **A regression is attributed** only after a matched replay reproduces it.

## 8. Rendering

Figures are drawn from stored graphs and terrain only, with the same extent, scale, hillshade and road styling, and
predefined seeds (the first two confirmatory seeds plus the weakest and any empty run by a fixed rule). PNG and PDF
are produced, with run IDs and the render command.

## 9. What is not done

* No sealed test.
* No held-out metro.
* No paid services.
* Forest, soil and weather are deferred.

## 10. Amendments

**Amendment 1 (2026-11-21).** Made after the part 1 replay results and before any development-sweep summary or
confirmatory run.

* **Finding behind it.** On the demo landscapes, the street-erased backbone gave M\* a > 15 % share of 0.058 against
  0.204 for TN (paired −0.146 [−0.334, −0.020]). This is close to the M12 hard-rule configuration (0.050). On the
  analytic sites the same backbone produced some empty networks (0.12 for M\*).
* **Change.** TNse_s0 is added to the confirmatory plan as a **secondary backbone**:
  * M\* and NA-fixed;
  * style off and proposals + topology for every style;
  * the α frozen for TN_s0 (no separate tuning);
  * `syn_conf`, the same seeds.
* **Unchanged.** Primary verdicts stay on TN_s0.

**Amendment 2 (2026-11-21).** Made after the development sweep, before freezing and before any confirmatory run.

* **The runtime constraint** "seconds per seed ≤ 3 × rerank" (§4) is replaced by "candidate evaluations per seed
  (learned + style, look-ahead roll-outs included) ≤ 3 × rerank".
* **Reason.** Wall-clock times were confounded by up to 12 concurrent jobs: NA-fixed reranking ranged from 52 to
  738 s per seed for comparable work. Evaluations are deterministic and measure the same intended quantity.
* **A filter bug is also fixed.** The selection had included the adaptive-trigger development runs in the
  always-on choice. Only always-on runs are eligible, as §4 intended.
* **Seconds are still reported.**
