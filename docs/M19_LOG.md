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
  * **Amendment** (found while matching D2's events; only the event counts had been compared, nothing else).
    * An eligible frontier is the pop of a node with **at most one** street, off the border, on land ≥ 15 %: a
      street end, or a new street's start node.
    * The events are D2's exactly: a STOP during that pop that leaves the node a closed dead end. That includes a STOP
      after an arm that added no street.
    * With this definition the natural-window replay finds D2's 200 events one for one.
    * Runner: `scripts/m19_diag_verify.py` → `outputs/m19/diag_verify/`.
* **F3. Verification of the diagnostic (D4)** (`outputs/m19/diag_verify/`).
  * **Integrity.**
    * The four replays are bit-identical to the stored M18 runs.
    * The re-scored candidates equal the live weights.
    * The events equal D2's one for one: 1,213.
    * Recomputing every classification from the stored states reproduces it exactly (0 mismatches).
  * **Frequency.** 1,213 of 13,576 eligible steep frontiers end in a STOP that leaves a dead end: 8.9 %, between
    7 % (GEN2) and 12 % (Petra).
  * **Land affected.** Eligible land within 120 m of an event, farther than 60 m from every street, as a share of the
    eligible land:

    | GEN2 | Natural windows | Petra | Machu Picchu |
    |---|---|---|---|
    | 4.1 % (0.97 km²) | 13.4 % (0.45 km²) | 4.8 % (0.04 km²) | 35 % (0.03 km²; little eligible land) |

  * **Shares** over the 703 eligible events. Intervals: block bootstrap over 14 GEN2 blocks with events, 13 natural
    windows and the 2 sites; the sites are single units.

    | Proposal set | Measure | Share |
    |---|---|---|
    | Learned, 8 (p_orig) | aligned first step | 0.39 |
    | Learned, 64 (p_big) | aligned first step | 0.64 |
    | Router | first step feasible | 0.95 |
    | Router | complete route feasible under the sampler's rules | 0.58 |
    | Learned roll-outs (32, same destination and length budget) | at least one reaches the destination | 0.66 (465/703) [0.62, 0.70] |
    | Learned roll-outs | a single roll-out reaches the destination | 0.18 [0.15, 0.21] |

    * Router failures when committed in steps: 146 steps dropped as unrepresentable and 139 joins to another street
      before the destination.
    * Learned roll-outs end mostly by joining *some other* street (56 % of roll-outs). That also closes the dead end.
    * **Correction to F1's reading.** Over a matched physical horizon, the learned model reaches the destination
      about as often as the router's committed route (0.66 with 32 tries against 0.58). The first-step comparison
      (0.39 vs 0.95) overstated the proposal gap.
  * **Selection.**
    * Median P(STOP) is 0.41; 43 % of events have P(STOP) > 0.5 (systematic) and 31 % have P(STOP) < 0.2
      (occasional).
    * 1,000 paired redraws reproduce the probabilities (largest deviation 0.048).
    * The terrain-moment term carries the asymmetry: median candidate component −2.63, M13 potentials −0.20, network
      and degree terms 0. STOP pays only the dead-end weight, median −0.64.
    * So a continuation is charged its immediate terrain cost with no credit for later closing the dead end, which
      STOP's weight does count.
* **D5. Development comparison of the arms, design fixed before any arm run.**
  * **Inherited configuration** (frozen M18, D14b).
    * Tiles: NA-SMC with J's parameters (`configs/m18_terrain_potentials_TNg_joint.json`, with F1 + F4).
    * Whole towns: J plus the other site's D14 δ.
    * The primary backbone is TNg.
  * **Arms.**

    | Arm | Proposals | Scoring | Notes |
    |---|---|---|---|
    | A0 | learned (K = 8) | frozen | the M18 baseline |
    | A1 | learned + router | frozen | |
    | A2 | learned | horizon-matched look-ahead twist | |
    | A3 | learned + router | the twist | |
    | A4 | learned, K = 64 at frontier decisions | frozen | larger learned search |
    | A5 | as A0 | frozen | then the same router applied after growth |

  * **Where the online changes act** (A1–A4): at decisions at an eligible frontier, as defined in D4 (a node with at
    most one street, off the border, landform slope ≥ 15 %). Elsewhere every arm is A0.
  * **Router proposals** (A1, A3).
    * They need a destination (D2's rule: only the partial graph is used) and a viable route (D2's router).
    * Two proposals per decision: jittered copies (σ = 3 m) of the route point at distance L0. L0 is the median length
      of the feasible learned candidates (20 m if none). The class is drawn from the model's class head; the street
      is at ground level.
    * Weights: every candidate's weight is multiplied by p/Q, with Q = (K p + M q)/(K + M). Here p is the model's
      density of the candidate's displacement × class × structure, and q is the Gaussian × class probability (ground
      only; `style_proposals.logpdf_d` / `logq_d`).
    * Proposals are one step at a time (receding horizon): no multi-step or variable-length move enters the target,
      so the sampler's target distribution is unchanged.
    * Roll-outs (A3) use learned proposals only.
  * **The look-ahead twist** (A2, A3). This is the scoring change chosen from F3's traces.
    * It uses the M13/M14 approximate twisted SMC: roll-outs with the network potentials and the STOP weight inside;
      the twist is corrected when the node is popped.
    * It is computed at frontier decisions only, with horizon H = 8 steps (the 75th percentile of route length in
      learned steps, F3) and 4 candidates per roll-out step (as NA-fixed).
    * This puts the continuation and STOP on a comparable horizon. The target is unchanged.
    * Natural termination is checked: rounds used, and runs that reach the round cap.
  * **A4.** 56 extra learned candidates from the same decoder output at frontier decisions. All are drawn from p, so
    no correction is needed.
  * **A5.**
    * Applied to the final state of A0's chosen particle.
    * Candidate connectors: every eligible dead end (one street, off the border, ≥ 15 %), with D2's destination and
      router on the current graph.
    * Connectors are added in order of increasing route length; each is committed in steps of L0 under the sampler's
      own rules (as D4).
    * A graph snapshot is saved after every connector (tiles) or every 10 connectors (whole towns).
    * Online and post-hoc routing share the same terrain, router cost, grade cap, water rule and crossing rules.
  * **Compute logged:** model forward passes, candidate evaluations, router calls and search-window nodes, wall time,
    total street length. K = 64 is a configuration; equal compute is a measurement, shown as access against
    compute.
  * **Primary endpoint: steep-land geographic access.**
    * Eligible land: dry, valid, inside the tile or the city limits, landform slope < 35 %. Real streets still cover
      62 % of land steeper than 30 % (M18 F22).
    * Steep band: 15–35 %. 15 % is the sampler's steep threshold; real coverage falls from 0.87 (10–20 %) to 0.80
      (20–30 %) and 0.62 (> 30 %).
    * Fixed points, seeded by crc32(unit id) and never shown to any generator: per tile, 40 points on steep eligible
      land and 40 on any eligible land; per site, scaled by eligible area / 0.262 km².
    * Pairs: at least one steep endpoint; 150–500 m straight; on the same dry-land region.
    * A pair is served if:
      * both endpoints are within 60 m of a street (the coverage radius of F22);
      * the network joins them (closed window on tiles; open graph for whole towns);
      * the detour, (network length + the two joins) / straight distance, is ≤ 4.
    * Failure modes are reported separately: an endpoint not reached, disconnected, detour > 4. Empty generations
      stay in the denominator.
  * **Units.**
    * Tile scale: the 39 GEN2 tiles (14 blocks) with ≥ 0.01 km² of steep eligible land, and the 16 natural windows;
      TNg_s0–s2 × inference seeds 1–2, paired; block bootstrap over the 30 units (B = 1000).
    * Whole-town scale: Machu Picchu and Petra, TNg_s0, seeds 1–3; descriptive, with only 2 units.
    * The two scales are analysed separately.
  * **Contrasts.**
    * Each arm − A0, unconditional.
    * Each online arm − A5 at matched total street length. A5's access is interpolated along its snapshots at the
      arm's length; if the arm is shorter than A0, it is compared with A0.
    * Each online arm − A4.
    * Plot: steep access against total street length, with A5's curve.
  * **Preservation margins.**
    * Flat-land density (0–5 % and 5–10 %): |arm − A0| ≤ 1 between-city SD of the reference (3.2 and 2.4 km/km²).
    * Steep-land coverage (20–30 %, > 30 %): arm − A0, lower CI bound ≥ −0.02.
    * Upper-tail grade (length-weighted grade p90, F22 smoothing): arm − A0 ≤ max(1.0 percentage point, 1 SD of
      A0's per-unit values across its 6 runs), computed from A0 before any other arm is evaluated.
    * Independent outcome, outside every objective: sustained climbs (the share of length in runs ≥ 100 m, and runs
      per km) against real GEN2.
  * **Mechanism:** router proposals offered, selected, completed (reaching the destination) and used by served
    pairs; whether a connector joins a previously separate component or doubles an existing link.
  * **Selection at each scale.**
    * An arm qualifies if:
      * its steep access exceeds A0's (tiles: CI excludes 0; whole towns: positive in all 6 site × seed pairs); and
      * the preservation margins hold.
    * If A1 or A3 qualifies and also beats A5 at matched length and beats A4 (CIs exclude 0 in its favour; whole
      towns: all 6 pairs), online routing is selected.
    * Otherwise the simplest qualifier is selected, in the order A5 < A4 < A2 < A1 < A3.
    * If none qualifies, the round closes and the trade-off is reported.
  * **Secondary.** TN_s0 and TNse_s0 (imprint check) on the selected arm vs A0, tile scale, seeds 1–2; descriptive.
  * **Next.** D6 (freeze, confirmation protocol) is written after this comparison and before any holdout is opened.
    Opening sealed or held-out data needs the owner's explicit authorisation.

## 2026-11-27

* **F4. D5 development comparison: online routing does not earn its complexity; the router after growth is selected
  at tile scale** (`outputs/m19/arms/`; `scripts/m19_arms.py`, `m19_eval.py`, `m19_maps.py`).
  * **Integrity.** A0 run through the arm runner is bit-identical to the stored M18 run, so the arm code is inert when
    off. All 90 runs and the 8 secondary runs completed.
  * **Tile scale.** 330 unit-runs: 39 GEN2 tiles + 16 natural windows × TNg_s0–s2 × seeds 1–2; 30 blocks. Steep-land
    access with 95 % block-bootstrap intervals:

    | Arm | Access | − A0 | − A5 at matched length | − A4 | km | Steep coverage 20–30 / > 30 % (Δ) | Qualifies |
    |---|---|---|---|---|---|---|---|
    | A0 frozen | 0.480 | | | | 940 | | |
    | A1 router proposals | 0.484 | +0.004 [−0.030, +0.032] | +0.001 [−0.034, +0.027] | −0.014 [−0.058, +0.013] | 935 | −0.002 / +0.004 | no |
    | A2 look-ahead twist | 0.425 | **−0.055 [−0.076, −0.027]** | −0.059 [−0.079, −0.033] | −0.073 | 880 | **−0.070 / −0.072** | no |
    | A3 both | 0.411 | **−0.069 [−0.100, −0.028]** | −0.073 [−0.104, −0.033] | −0.087 | 872 | **−0.091 / −0.108** | no |
    | A4 K = 64 | 0.498 | +0.018 [−0.010, +0.051] | +0.013 [−0.015, +0.044] | | 959 | +0.019 / +0.030 | no (gain CI includes 0) |
    | **A5 router after growth** | 0.492 | **+0.012 [+0.005, +0.023]** | | | 965 | +0.003 / +0.002 | **yes** |

    * Every arm keeps flat-land density within its margin.
    * A5's grade p90 rises by +0.5 pp; the margin is 1.47 pp (the A0 per-unit SD).
    * Climbs ≥ 100 m: share 0.032 (A0) and 0.037 (A5), against 0.04 or more for real GEN2 (M18 F24). There is no
      margin on this outcome.
    * Failure modes (A0): an endpoint not reached 0.485, disconnected 0.022, detour > 4 0.014.
      * On tiles, a pair joined only through the outside of the tile counts as served (D5), so tile access mostly
        measures reach.
    * In 156 of 330 rows, A1 is longer than all of A5's connectors combined. There A5 is compared at its maximum,
      which favours A1, and A1 still does not beat A5.
  * **Selection (D5).**
    * Only A5 qualifies, and no online arm beats both A5 and A4, so the simplest qualifier, **A5**, is selected at
      tile scale.
    * The look-ahead twist (A2) — the scoring change chosen from F3's traces — *lowers* access and steep coverage. Over
      a horizon, continuing up steep land accumulates terrain penalties, so stopping looks better still.
  * **Whole towns** (Machu Picchu and Petra, TNg_s0, seeds 1–3; descriptive with 2 units).

    | Site and seed | A0 | A1 | A2 | A3 | A4 | A5 |
    |---|---|---|---|---|---|---|
    | Petra s1 | 0.31 | 0.26 | 0.21 | 0.15 | 0.55 | **0.50** |
    | Petra s2 | 0.21 | 0.35 | 0.13 | 0.19 | 0.24 | **0.43** |
    | Petra s3 | 0.27 | 0.59 | 0.24 | 0.35 | 0.17 | **0.52** |
    | Machu Picchu s1 | 0.00 | 0.24 | 0.06 | 0.15 | 0.01 | 0.00 |
    | Machu Picchu s2 | 0.14 | 0.06 | 0.07 | 0.00 | 0.00 | 0.14 |
    | Machu Picchu s3 | 0.00 | 0.10 | 0.03 | 0.07 | 0.09 | 0.00 |

    * Machu Picchu has only 81 pairs. Its frozen network is parallel contour streets that never join, and A5 finds
      almost no viable connector.
    * No arm is above A0 in all 6 site × seed runs, so **no arm qualifies at whole-town scale**.
  * **Mechanism.**
    * A1 kept 587 router steps over all tile runs (31,225 router calls) and 294 in the towns. Under correct weighting,
      the model's low probability for gentle detours keeps them rare.
    * In the D5-rule maps, the representative tile "success" and "failure" of A1 against A5 kept 1 and 0 router steps.
      Per-unit differences between online arms are mostly sampling divergence: 165 positive against 164 negative.
    * A5: 311 of 393 tile connectors and 152 of 188 town connectors are complete. At Petra they join separate pieces
      (served share +0.19 to +0.25 for about 1–2 km of street).
  * **Compute** (tiles, summed).
    * A5: 906 router calls, about 2 s.
    * A1: +31,225 router calls, about the same model passes as A0.
    * A2 and A3: about 4.3× A0's forward passes and 3× its candidate evaluations.
    * A4: 2.1× A0's candidate evaluations.
    * Wall times are confounded by running in parallel.
  * **Secondary** (descriptive; tiles, s0, seeds 1–2). A5 − A0 is +0.012 (TNg), +0.021 (TN) and +0.024 (TNse,
    street-erased input). The post-hoc gain does not depend on the street imprint.
  * **Verdict on the paper question** (development data only).
    * At tile scale, online routing does **not** earn its additional complexity over terrain-aware repair. Properly
      weighted router proposals change nothing, and the comparable-horizon scoring makes things worse.
    * The same router applied after growth gives a small, significant gain in steep-land access (+0.012) at almost no
      compute.
    * At whole-town scale the evidence is two sites and mixed:
      * the router after growth roughly doubles access at Petra in all 3 seeds;
      * nothing works at Machu Picchu, where the frozen generator's town is fragmented into parallel contour streets.
    * The positive finding is a documented win for the simpler baseline.
* **Q2. Confirmation (D5 → D6), owner decision needed.**
  * By D5, A5 (the router after growth) is the selected tile-scale method, and a fresh-holdout confirmation would
    test A5 − A0 (with A1 and A4 as comparators) on untouched geography.
  * The effect is small (+0.012), so a confirmation needs many hilly units.
  * It also needs fresh metros (data acquisition: OSM and GLO-30) or the sealed M12 test tiles. Opening either needs
    the owner's explicit authorisation.
  * Alternatives: close M19 on the development verdict, or extend the whole-town evidence with more development
    sites first.
