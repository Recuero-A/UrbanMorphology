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
* **F1. Proposal diagnostic: by D2's rule the result is "mixed"; the limit is in both the proposals and the
  scoring** (`outputs/m19/proposal_diag/`; `scripts/m19_proposal_diag.py`).
  * **Integrity.** All four replays are bit-identical to the stored M18 runs: GEN2, the natural windows, Machu Picchu
    and Petra. The observer changed nothing.
  * **Data.** 1,213 STOPs left hillside dead ends. 955 had a destination in the partial network, and 703 were
    eligible (the router found a viable route).

    | Run | Eligible | p_orig (8 learned) | p_big (64 learned) | p_route (router step feasible) |
    |---|---|---|---|---|
    | GEN2 | 243 | 0.36 | 0.56 | 0.95 |
    | Natural windows | 110 | 0.38 | 0.64 | 0.95 |
    | Machu Picchu | 41 (< 50: inconclusive alone) | 0.37 | 0.71 | 0.90 |
    | Petra | 309 | 0.41 | 0.70 | 0.95 |
    | **Pooled** | **703** | **0.39** | **0.64** | **0.95** |

  * **Decision rule (D2).** p_orig is between 0.25 and 0.5, so the result is **mixed**: both shares are reported and
    the owner decides. The shares agree across the runs.
  * **Descriptive, not part of the rule.**
    * In 61 % of eligible stops, no aligned feasible move was among the 8 proposals. Drawing 64 learned proposals
      finds one in 43 % of those stops. The router's first step is feasible in 95 % of stops.
    * The weights raise the stop probability.
      * The model's own stop probability has median 0.15; after the terrain and network weights, the sampler's
        P(STOP) has median 0.34 (median ratio 1.8).
      * The weights raised it in 71 % of stops.
    * Where an aligned move was proposed, it got a median 14 % of the selection mass, and STOP got 38 %.
    * The router's routes: median detour 1.95; median route grade p90 10 %.
    * 99 % of stops had at least one feasible learned candidate, so the stops are not forced by infeasibility.
  * **Reading.**
    * The turtle often does not offer the gentle continuation toward a nearby part of the network, although one
      exists. More learned samples recover about half of these cases; a terrain router recovers almost all.
    * When the turtle does offer it, the weights favour stopping.
    * So a proposal gap and a scoring effect both hold. Q1 asks the owner which to test.
* **Q1.** Next experiment (owner decision, D2 "mixed"):
  * (a) router proposals mixed with the learned ones, with importance weights, against matched-compute learned
    proposals and post-hoc routing;
  * (b) more learned proposals alone (K = 64);
  * (c) a scoring change: a stop decision that counts the value of a gentle continuation, e.g. a router-estimated
    twist.
* **D3. Owner decision (2026-11-26).** Option 1 for M19, with a scoring-only control.
  * The paper question: does giving the turtle a terrain-feasible route *during growth* improve access on steep land
    beyond what the same router achieves *after* growth, at comparable road length and compute?
  * The plan, in order:
    1. correct the scientific record;
    2. verify what the diagnostic measures;
    3. run a 6-arm development comparison with a frozen choice of one scoring change;
    4. make geographic access on steep land the decisive endpoint, at matched road length, for tiles and whole towns
       separately;
    5. freeze once, then confirm on untouched geography (owner authorisation needed before any sealed or held-out
       data are opened).
* **F2. Corrections to the scientific record** (checked against `docs/M14_EXPERIMENTAL_PROTOCOL.md`,
  `docs/M18_AUDIT.md` §5, `docs/M18_LOG.md`).
  * **Attribution of the connectivity results** (GEN2, TN backbones).

    | Contrast | Estimate [95 % CI] | How obtained | Status |
    |---|---|---|---|
    | NA-fixed − M\* (M14 H1) | **−0.0259 [−0.0363, −0.0154]**, p < 10⁻⁴, 34 blocks | M14 confirmatory generation (3 × 3 × 95). Re-scored in M18 from the stored graphs with M14's random stream: 1,707 of 1,710 graphs identical. Not regenerated | pre-registered primary |
    | NA-SMC − M\* | **−0.019 [−0.029, −0.007]** | M18 audit: stored graphs re-scored; paired tile differences averaged per 3 km block | descriptive (not an M14 primary test) |
    | M\* + post-hoc A\* repair − M\* | **−0.025 [−0.034, −0.017]** | same as the row above | descriptive |
    | NA-adaptive − M\* on TNlp and TNse (M14 H5, leak control) | CI excludes 0 in 3 of 3 seeds each; Holm-significant | M14 generation | pre-registered |
    | NA-SMC − M\* on TNg (M18 F21) | −0.009 [−0.021, +0.009] | M18 generation | descriptive |

    * **Endpoint.** `od_unreach_land` uses pairs of the generated graph's own nodes on the same land region (M18 F3),
      not fixed geographic points.
    * **Erratum.** My chat summaries on 2026-11-26 attributed −0.026 to NA-SMC. That was wrong: −0.026 is NA-fixed vs
      M\*, and NA-SMC's estimate is −0.019. The repository documents already had it right (`docs/M18_PLAN.md`).
    * Post-hoc repair gives the largest reduction in unreachability. Its known costs are terrain-blind connectors
      (M14: detour 1.59, unbridged links).
  * **"No cost" (M18 F2)** was a no-measurable-difference statement, not non-inferiority. NA-SMC − M\*, with 95 %
    block-bootstrap intervals:
    * km/km²: −0.39 [−0.88, +0.10];
    * junctions/km²: −4.7 [−9.5, +0.1];
    * close junctions < 15 m: +0.002 [−0.008, +0.012];
    * links < 15 m: −0.005 [−0.010, +0.001];
    * blocks/km²: +2.0 [−0.9, +5.0];
    * share of blocks < 2,000 m²: −0.052 [−0.101, +0.009].

    No preservation margins were prespecified for these statistics. M14's prespecified guard (S10_cond ≤ M\* + 0.05;
    density ratio within ±25 %) applied to H1 and H3 only.
  * **M18's terrain result.** 2.22 → 1.12 means better agreement with the targeted per-slope curves: the score the
    potentials were fitted to match, on training tiles. Alongside it:
    * three steep-land preservation checks were missed (M18 F28);
    * whole-town density stayed too high, and fixing it broke connectivity (F27, F28);
    * the untargeted climb outcome was not improved (GEN2 climb gap −0.02 [−0.19, +0.14]), and every generator is
      still short of climbs ≥ 100 m (F24).
  * **LiDAR street-imprint effect**, by metric and control (only what was measured).

    | Metric | Control | Effect |
    |---|---|---|
    | S_T (terrain-response gap, city tiles) | street strips erased vs random strips of equal area (M18 F17) | +0.122 [+0.025, +0.219] (A), +0.128 [+0.033, +0.222] (M\*) on top of random-strip erasure (+0.178 A, +0.116 M\*) |
    | S_T | street-erased / low-pass input (M13) | ≈ 0.6–0.7 instead of ≈ 0.4 |
    | S_T | GLO-30 input (TNg) vs LiDAR (TN), GEN2 (M18 F21) | 0.36–0.55 vs 0.19–0.38 |
    | Carriageway probe AUC from terrain alone (whole tile / inside corridors) | raw, low-pass, street-erased, random-erased, landform, GLO-30 | 0.78/0.75, 0.66/0.56, 0.74/0.61, 0.71/0.65, 0.51/0.52, 0.59/0.53 |
    | Terrain response with no imprint possible (natural windows; M18 F20, F21) | TN / TNg vs terrain-blind N0 | share > 15 %: TN 0.13–0.16, N0 0.27. Density on > 30 % land: TN ≈ 3, N0 13–15 (real 4.0). Contour index: TNg 0.47–0.52, TN 0.38–0.42 |
    | Connectivity (M14 H5) | TNlp, TNse | the H1-type effect persists |

    Not tested for imprint: the M18 terrain-realism score (F22) and TAS.
  * **Literature comparisons** describe the papers in `docs/literature/` (about 30 notes, free sources). This is not
    a systematic review.
* **D4. Verification of the diagnostic (step 2), fixed before its runs.**
  * **Replays.** As D2: the frozen M18 baseline, bit-identical, the same four runs, with extended logging.
  * **Counts.** Raw numerators and denominators for every share. Independent units: GEN2's 3 km blocks, each natural
    window, each site. Event shares are bootstrapped over these units (B = 1000).
  * **Frequency and land.**
    * Eligible frontiers: every pop of a node that ends one street (degree 1, off the border) on land with slope
      ≥ 15 %.
    * Reported: the share of eligible frontiers whose first decision is STOP (the D2 events).
    * Affected land: eligible land (dry, valid, slope < 30 %) within 120 m of an event node and more than 60 m from
      every street of the final network, as a share of all eligible land.
  * **Same horizon for learned and router proposals.** Both get the same destination (D2) and the same physical
    horizon: a length budget of 3 × the straight distance + 60 m.
    * Router: first-step support (D2), plus a complete feasible continuation. The route is cut into steps of the
      median learned step length, then committed in order on a copy of the state under the sampler's own rules; all
      steps must be feasible.
    * Learned: 32 roll-outs per event from the same state. Each step draws the model's 8 proposals, keeps the
      feasible ones, and picks one by the frozen weights, with STOP excluded. The roll-out ends at the budget, at
      40 steps, or when no candidate is feasible.
      * First-step support: an aligned feasible move among the 8 (as D2).
      * Complete continuation: the roll-out ends within 25 m of the destination or joins its component.
  * **Selection.**
    * Per candidate, the weight components: M13 terrain potentials, degree prior, network term, terrain-moment term.
    * Per candidate, the selection probability, against STOP's.
    * 1,000 paired draws of the decision per event, giving the empirical STOP frequency. An event counts as
      systematic if P(STOP) > 0.5 and as occasional if P(STOP) < 0.2.
  * **Stored states.** Every event's state, decoder output and candidates are saved (compressed). The classification
    is then recomputed from the stored states and must match the logged one exactly.
