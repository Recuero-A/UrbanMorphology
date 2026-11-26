# M19 log: can the model build gentle roads that still reach places on steep terrain?

One ordered log of decisions (D) and findings (F), as `docs/M18_LOG.md`. Baseline: the frozen M18 setting (M18 D14b).
Standing rules: development data only; the sealed test and the remaining held-out metros stay unused; every design is
written here before its runs; commits by Recuero-A without AI attribution.

## 2026-11-26

* **D1. Owner decision and its reason.**
  * The decision: freeze M18 and test one new idea. The turtle should be able to propose a route around a hill
    before it decides to stop. First, find out with a diagnostic whether the limit lies in the proposals or in the
    scoring.
  * The evidence:
    * M18 does not show that UrbMorph is more realistic or more terrain-adaptive overall than the external
      generators (M18 F26, F28).
    * At Machu Picchu the D12 potentials raised TAS from 40.9 to 74.7, while geographic connectivity fell from 0.76
      to 0.27.
    * At Petra, TNg is competitive on grade, but in whole-town mode it builds about three times the reference
      flat-land density.
  * The hypothesis. The sampler chooses among 8 short learned moves (or STOP). A viable hillside street may need to
    follow a contour for several steps before turning uphill. If such a continuation rarely appears among the
    proposals, then raising the terrain penalty makes STOP more attractive. That would cost connectivity, as observed.
* **D2. Proposal diagnostic, design fixed before any run.**
  * **Generator.** The frozen M18 baseline, unchanged, at its stored thread counts.
    * Tile mode: NA-SMC on TNg_s0, inference seed 1, 2 threads, on all 95 GEN2 tiles and on the 16 natural windows.
    * Whole-town mode: Machu Picchu and Petra, seed 1, 1 thread, each with the other site's D14 δ.
    * The diagnostic only observes. Its extra proposals use their own random generator and a copy of the state.
      The integrity check: every diagnostic run must reproduce the stored M18 run bit for bit.
  * **Events.**
    * An event is a STOP chosen at a node u that leaves u a closed interior dead end: one street, off the tile
      border, not queued.
    * u must lie on hillside land: landform slope ≥ 15 % (the sampler's `steep_slope`).
    * Every event in every particle is recorded.
  * **Destination.** It uses only the state at that moment; no evaluation information is used.
    * Candidates: the nearest node v of the partial network at 30–250 m straight distance that is either in another
      component, or reachable from u only by a network path longer than 3 × the straight distance.
    * An event with no such v is recorded as "no target" and is described only.
  * **The three proposal sets, on the identical state.**
    * (a) The 8 original learned candidates of that decision, with their full weights.
    * (b) 64 learned candidates drawn from the same decoder output (the same mixture, class and structure heads),
      scored by the same `candidates`.
    * (c) A terrain-aware route from u to v.
      * Method: Dijkstra on the tile's terrain grid (4 m, 16 neighbours). Cost = length × (1 + (grade / 0.08)²).
        Water is forbidden, and so is any piece steeper than 21.8 %, the sampler's hard cap for local streets
        (`configs/m12_feasibility.json`).
      * The route is viable if it is found and is at most 3 × the straight distance + 60 m long.
      * Its first step is the point along the route at the median length of the feasible learned candidates
        (20 m if there are none). It is scored by the same `candidates`, with the class most likely under the model.
  * **Measures, per event with a viable route ("eligible").**
    * Aligned feasible move: a candidate with finite weight whose end point lies within 30° of the route's first leg
      (the route point at 20 m).
    * p_orig = share of eligible events where (a) holds an aligned feasible move.
    * p_big = the same share for (b).
    * p_route = share of eligible events where the route's first step is feasible under the sampler's own rules.
    * Also reported:
      * the shares of events with a destination and with a viable route;
      * P(STOP) as the sampler computed it, and the selection mass on aligned moves when they exist;
      * the route's length, grade p90 and detour.
  * **Decision rule** (fixed now; pooled over the four runs, and also reported per run).
    * Proposal gap: p_orig < 0.25 and p_route ≥ 0.5. Next step: router proposals mixed with the learned ones, with
      importance weights. Matched-compute learned proposals and post-hoc routing are baselines.
      * If also p_big ≥ 0.5, more learned proposals may already suffice, and that becomes the main comparison.
    * Scoring problem: p_orig ≥ 0.5. The turtle proposes such moves and the weights discard them. Next step: change
      the candidate scoring instead.
    * Otherwise mixed: both shares are reported, and the owner decides.
    * Fewer than 50 eligible events pooled: inconclusive.
  * **Runner.** `scripts/m19_proposal_diag.py` → `outputs/m19/proposal_diag/`.
