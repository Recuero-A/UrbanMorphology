# Milestone 16 method: style-aware proposals and topology rules

* **Scope.** The learned terrain-aware turtle (the M12 TerrainGraphTurtle checkpoint, unchanged) and the M13 / M14
  samplers (M\*, NA-fixed, M13-SMC, NA-SMC) are unchanged.
* **What M16 adds.** An optional source of extra candidates and a few network rules, applied inside the same
  candidate-evaluation loop.
* **Code.** `graphs/style_proposals.py` (targets, densities, trigger) and `graphs/tc_sampler.py` (`add_proposals`,
  the cul-de-sac rules in `evaluate_candidate`, the continuation rule).
* **Defaults.** With `prop_mode = ""` every earlier sampler is bit-identical: the M14 archive replays 95/95 for M\*
  and NA-fixed (`outputs/m15/audit/replay_m16_proposal_code.json`).

## 1. Why proposals, not only reranking

**M15 reranked the K = 8 learned candidates by a heading field.** Reranking can only choose among what the model
proposes. Two development diagnostics bear this out:
* the learned candidates rarely include a pattern move, so the decision log often records
  `no_aligned_model` = 1: no feasible learned candidate within 15° of any pattern direction;
* the model gives pattern moves (exact axis directions, lattice-length steps, ring chords) a log-density 7–47 nats
  below its own typical draw (`style_plaus_*`, flat development site).

**Grids are topological.** Boeing (2019) reports r(φ, 4-way share) = 0.63 across 100 cities: grids have through
streets, crossings and few dead ends (`docs/M16_LITERATURE.md`). A heading-only term cannot create a crossing.

## 2. Proposals

At every decision (a popped node u with incoming heading h), up to `prop_m` = 4 style candidates are added to the K
learned ones. Each is a displacement with a class (drawn from the model's class head, or fixed by the rule), ground
only, jittered by `prop_sigma_m` = 1 m. Lengths default to the median length of the decision's feasible learned
candidates (L0, clipped to 12–60 m), so proposals keep the model's step scale.

| Pattern | Proposals (`prop_mode = style`) |
|---|---|
| grid | the cross-field axis closest to h (straight) and the two perpendicular axes (turns) |
| organic | straight along the local organic field direction ± N(0, 8°); one branch (30 %: two) at 60–120° on a random side; lengths × lognormal(0, 0.3) |
| radial | spokes out and in, held on the ray through u; ring arcs (chords on the circle of u's radius); at the centre, 8 spokes on fixed bearings |
| cul-de-sac | from a collector (major / mid) or the start: local branches at 90 ± 12° to both sides (class local, `branch_len_m`) and the collector continuation (class mid); on a local branch, a short continuation while depth allows |

Directions within 30° of an edge already at u are skipped. Mixed patterns (e.g. `prop_pattern = "radial+culdesac"`)
concatenate the targets.

## 3. Topology rules (`prop_mode = style_topo`)

**Grid lattice.** A lattice of spacing s (`grid_spacing_m`) aligned with θ0, anchored at the tile centre.
* Streets are held on their lattice line (lateral snap when within s/4).
* Steps end on the next lattice crossing, split into near-L0 steps.
* Turns are proposed only where a perpendicular lattice line passes (4 m).
* A closure candidate reaches a street ahead within 1.5 s (T- or X-junction).

**Radial.** Ring radii at multiples of `ring_spacing_m`.
* Spokes stay on their ray and step to the next ring.
* Ring arcs are proposed only at nodes on a ring.

**Organic.** Infill closures: a candidate reaching an existing street within 2.5 L0 along the straight or branch
direction, creating T-junctions and uneven blocks.

**Cul-de-sac.** Hard rules that apply to EVERY candidate, learned ones included:
* a local / minor edge whose branch is rooted on a collector may not exceed `term_depth` = 3 local edges ("terminal");
* such an edge may not close a loop ("local_loop");
* branch roots on a collector are at least `branch_gap_m` = 60 m apart.

**Continuation.** While a feasible pattern continuation exists, the arm's stop odds fall by e^−γ (γ = 2 nats).
Continuations are:
* grid: straight or a lattice turn;
* radial: spoke or ring;
* organic: straight;
* cul-de-sac: collector or a spaced branch.

**Parameters** (`configs/m16_topology.json`) come from the real `calib` references, not from generated output:
* s = √(median grid block area);
* branch step = median terminal length ÷ 3.

## 4. Weighting the pool and what it means for inference

**Heuristic (default).** Selection mass is split by source:
* feasible style candidates share α (the pattern strength, `prop_alpha`), each weighted by its terrain / network /
  M15 style weights within the group;
* feasible learned candidates share 1 − α;
* when one group has no feasible member, the other takes all the mass, so proposals never stop an arm the learned
  pool would continue.

The stop decision uses Z = (1 − α) · mean_learned w + α · mean_style w. **This changes the target distribution:**
* the result is no longer p_model × potentials;
* no SMC or importance-sampling guarantee holds for the styled output;
* the continuation rule also changes stop probabilities directly.

**Importance-weighted variant (`prop_weight = is`).** Each candidate gains log p(x) − log Q(x) with
Q = (K p + M q)/(K + M), the balance-heuristic mixture of the model and the style proposal:
* q = equal-weight Gaussians of σ = 1 m about the targets × the class rule × ground only;
* selection and the stop normaliser then target p_model × potentials as without proposals.

Proposals can then only reduce variance. Because the model gives pattern moves little density, this variant is
expected to change little; it is a test of H, not a candidate method.

## 5. Ablations and controls

* **Rule-only proposals** (`prop_rule_only`): the learned candidates are removed from the pool. Close to a
  procedural generator under the same potentials.
* **Matched-compute unguided pool:** K = 8 + the mean number of style candidates per decision.
* **Adaptive trigger** (`deficit:2`: proposals only when fewer than 2 feasible learned candidates are aligned)
  against a random trigger at the same rate.
* **Cul-de-sac dead-end ablation** (`dead_exempt_local`): the NA dead-end penalty is waived for a STOP that ends a
  rooted local branch. Without it, the NA potential (−1.81 nats per interior dead end) opposes intended cul-de-sacs
  (`docs/M15_POSTHOC_CRITIQUE.md` §4).

## 6. What is not new

**Already established elsewhere:**
* rule-generated candidates: Parish & Müller 2001, Chen et al. 2008, CitySprout, CityStack;
* snapping and closure rules;
* learned-skeleton-plus-rules pipelines: RoadWeaver 2026.

**What M16 contributes** is a narrow combination:
* rule candidates in one pool with a learned, terrain-conditioned turtle;
* selection under shared terrain and network potentials with hard water and grade rules;
* evaluation with outcomes the rules do not target, with matched-compute and rule-only controls.
