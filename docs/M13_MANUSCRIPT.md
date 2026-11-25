# Terrain-Constrained Sampling for Autoregressive Road-Graph Generation

*Draft manuscript (Milestone 13). Results are taken only from `outputs/m13/FINAL_REPORT.md` (development data).*
*No result in this draft comes from the held-out test splits, which have not been read.*

## Abstract

Autoregressive road-graph generators can be conditioned on terrain, but nothing at sampling time makes a generated
street buildable. We study whether probabilistic inference over a fixed, trained generator improves its response to
terrain. The target reweights the model by hard feasibility constraints and by density-ratio potentials calibrated on
real streets. It is sampled with a properly weighted K-candidate proposal, Monte Carlo look-ahead, beam search and
sequential Monte Carlo.

We pre-registered the evaluation, which controls for two confounds:
* **terrain leakage:** urban DEMs contain street beds, so we add low-pass and street-erased inputs;
* **street density:** a sparser network trivially violates less.

On development data, the selected one-step sampler lowers a slope-binned terrain-response score by 9–25 % per
training seed, and by 6–17 % on three unseen cities. Its block-bootstrap intervals include zero for most seeds. After
Holm correction, no confirmatory hypothesis is supported.

Three findings explain this:
* with leakage-controlled inputs, every sampler's terrain fit degrades sharply;
* at matched density, the advantage disappears;
* look-ahead does not reduce premature termination or fragmentation.

Particle methods reach lower scores only by building fewer streets. What constrained sampling does provide is the
removal of infeasible infrastructure: unbridged water crossings and spurious bridges. Realism and diversity are
preserved. We release the protocol, including slope-binned response curves, connectivity with unreachable pairs, and
leakage controls, as the more durable contribution.

## 1. Introduction

**The problem.** Procedural and learned street-network generators are used for urban simulation, games and
planning studies. On real terrain, a plausible network must respond to topography:
* streets climb at bounded grades;
* they turn along contours on steep land;
* they cross rivers on structures;
* they remain connected.

**What learned generators do today.** Autoregressive graph generators such as Neural Turtle Graphics (Chu et al.
2019) can be conditioned on terrain. Terrain-conditioned learned generators exist in raster form (Fang et al. 2022;
Gu et al. 2024) and in vector form (Zhou et al. 2024). These models learn a distribution, but nothing at sampling
time ensures that a generated step is physically buildable.

**Our earlier finding.** Combining a learned turtle model with deterministic feasibility rules either starved the
network (rules that drop infeasible steps) or required heuristic replanning that fragmented it.

**Question.** We study whether *probabilistic inference* over a fixed, trained generator addresses this:
constrained sampling, sequential Monte Carlo (SMC) and look-ahead.

**Framework.**
1. Generation becomes sampling from a target that reweights the model by hard feasibility constraints and soft
   potentials.
2. The soft potentials are **density ratios calibrated on real streets**. Their expectation under the proposal is 1,
   so the reweighting corrects the terrain response without the systematic pull toward shorter networks that
   positive energies cause.
3. K model candidates per decision are resolved by the **same geometric rules that insert them**, and the decoder
   is conditioned on the committed edge.
4. A properly weighted K-candidate proposal (Naesseth et al. 2015) serves single-chain and SMC samplers; Monte Carlo
   roll-outs provide approximate twists for look-ahead (Lin et al. 2013; Zhao et al. 2024).

**Evaluation.** We test the framework with a pre-registered protocol that controls for terrain leakage (an urban
bare-earth DEM still contains street beds) and for density (a generator that builds fewer streets trivially has
fewer violations).

**Contributions** (retained only if supported; see §7):
* **C1** a terrain-constrained inference framework for autoregressive road-graph generation, with calibrated
  density-ratio potentials and state-consistent commitment. It removes infeasible infrastructure; its effect on
  terrain response is **not** significant (§6);
* **C2** a negative evaluation of look-ahead for terrain-induced premature termination and fragmentation (§6.3);
* **C3** a leakage-controlled, terrain-stratified, density-matched evaluation protocol: response curves,
  connectivity with unreachable pairs counted, street-erased and low-pass DEM controls. It shows that an apparent
  improvement can be explained by leakage and density.

We do **not** claim the first terrain-aware neural road generator, the first neural vector road generator, or the
first look-ahead search for road graphs (§2).

## 2. Related work

* **Learned street generation.**
  * NTG (Chu et al. 2019): queue-based autoregressive graphs.
  * Birsak et al. 2022: transformer index fields, sliding windows; land / water conditioning, no terrain.
  * Fang et al. 2022: topography-aware GAN, raster.
  * Gu et al. 2024: conditional diffusion with elevation and slope, raster.
  * Zhou et al. 2024: elevation → density map → transformer vector graph.
  * We read NTG and Birsak in full; Fang, Gu and Zhou only as abstracts (access), so we do not characterise their
    decoding.
* **Terrain routing.** Galin et al. 2010 and 2011: anisotropic least-cost paths with hard thresholds and soft
  slope, curvature and water costs. Emilien et al. 2012: villages on terrain, with road re-use. These are evaluated
  visually or by path cost.
* **Search for road graphs.** Anagnostidis et al. 2023: tree search with a learned policy for road-graph
  *extraction* from imagery.
* **Constrained decoding.**
  * SMC steering of language models: Lew et al. 2023; Loula et al. 2025.
  * Twisted SMC: Zhao et al. 2024.
  * A*-style look-ahead decoding: Lu et al. 2022.
  * Look-ahead SMC: Lin, Chen & Liu 2013.
  * Proper weighting: Naesseth et al. 2015.
  * Constrained graph generation by projection or solvers: PRODIGY, ConStruct, NSGGM.
* **Our position.** We apply SMC-style constrained inference to a terrain-conditioned road-graph generator. The
  terrain-specific parts are the potentials, the committed-geometry transition and the evaluation.

## 3. Problem formulation

* **Generation.** A sequence of arm decisions at popped nodes: STOP, or (displacement, class, structure).
* **Transition.** Deterministic geometric rules: clip, junction at crossings, merge within 8 m, T-junction, drop
  short steps; bridges and tunnels pass over or under.
* **Target.**

  π(a₁:T) ∝ Π p_θ(a_t | s_t) · 1[hard] · exp(λ·φ)

  * Hard constraints are genuine impossibilities: ground over water, ending in water, unsupported structures, spans
    beyond any observed, degenerate crossing angles, unrepresentable steps.
  * Grade is **not** hard.
  * φ are log density ratios of real to proposal feature distributions (grade by class and slope bin; contour
    alignment by slope bin), plus span and angle survival terms for structures.

  `docs/M13_METHOD.md` §1–§2.

## 4. Method

* **K-candidate proposal (exact local weighting).**
  * The local normaliser Ẑ = p_stop + (1 − p_stop) · mean_k w_k is Rao-Blackwellised over the stop decision.
  * Selection is ∝ w among candidates drawn from p_θ.
  * We note the common error of selecting ∝ p_θ·w, which targets p_θ²·w.
* **Look-ahead.** Roll-outs of the new road's continuation for H steps with M candidates each; ψ̂ = Π Ẑ_h.
  * Stops by model choice and reconnections are not penalised.
  * Forced dead ends drive ψ̂ toward 0.
  * In SMC, twists are divided out when the node is popped (approximate twisted SMC).
* **Samplers:** standard, top-k, full temperature, filter, energy, energy + look-ahead, beam (stop sampled,
  length-normalised pruning), SMC, SMC + look-ahead. Frontier schedulers.
* **Tile-free generation.** One graph and one terrain field; the start head runs on sliding 512 m windows.
* **Procedural repair** is reported separately.

## 5. Experimental setup

* **Data.**
  * Training: 685 tiles (Spain, US, Japan). 3D-aware OSM graphs; DEMs from IGN MDT05 and USGS 3DEP.
  * Development: 52 tiles with a DEM, split by geographic blocks into DEV-A (selection) and DEV-B (confirmation).
  * GEN: Girona, Manresa and San Francisco; terrain-blind random tile sample.
* **Backbones.** TN (raw DEM input), TN-lp (σ = 16 m low-pass), TN-se (street-erased DEM, corridor factor 4) and
  N0 (terrain-blind); 3 training seeds each.
* **Metrics** (pre-registered):
  * primary: S_T, RC3 (slope-binned response curves, bins weighted equally), premature share and giant-component
    share, S10;
  * secondary: circuity with unreachable pairs, joint energy distance and MMD, infrastructure, earthwork proxy,
    diversity, compute.
* **Statistics.** Paired block bootstrap over geographic blocks; training seeds analysed separately; Holm across
  the confirmatory family.

## 6. Results

All numbers are from `outputs/m13/FINAL_REPORT.md`. The test splits were not used.

**6.1 Selection (DEV-A, TN_s0, rule pre-registered).**
* M* = E* = energy-guided sampling with K = 8 and λ = 0.5: S_T 0.338 vs 0.570 for standard sampling A, density 0.93
  of real, premature share 0.12. H* = 1.
* SMC with look-ahead had a lower S_T (0.286) but missed the ±15 % density criterion.
* Beam search starved the network (density 0.62, premature share 0.32).

**6.2 H1 (DEV-B).** Δ S_T (M* − A) per training seed, with 95 % block-bootstrap intervals:

| Backbone | s0 | s1 | s2 |
|---|---|---|---|
| TN | −0.110 [−0.257, +0.006] | −0.046 [−0.236, +0.123] | −0.041 [−0.183, +0.225] |
| TN-lp | −0.017 | +0.002 | −0.193 [−0.293, −0.092] |
| TN-se | −0.109 | −0.141 | −0.090 |

* TN: −25 %, −12 %, −9 %.
* TN-se: −16 %, −19 %, −13 %, with every interval including 0.
* None of H1, H1-lp and H1-se is supported.
* Absolute fit degrades sharply under the leakage controls (mean S_T of A: TN 0.42, TN-lp 0.63, TN-se 0.71).

**6.3 H2.**
* Look-ahead at H* = 1 changes premature share by −0.04 to 0 and giant share by −0.04 to +0.003.
* It meets the criterion in 0 of 3 seeds, at 5× the compute.
* Longer horizons failed selection on DEV-A.

**6.4 H3 (supported).** S10_cond(M*) − S10_cond(A) = +0.017, +0.007, −0.049. Diversity ratio ≥ 0.8 in all seeds.

**6.5 H4 (GEN: Girona, Manresa, San Francisco; 100 tiles).** Δ S_T for TN: −0.021, −0.062, −0.065 (−6 %, −17 %,
−16 %). The interval is below 0 for one seed. Holm across {H1, H1-lp, H1-se, H2, H4}: no hypothesis significant
(smallest median p = 0.071, for H4).

**6.6 Density-matched.**
* Stop biases calibrated on DEV-A bring both samplers to real density.
* On DEV-B, Δ S_T = +0.026, +0.025, +0.034; every interval includes 0.
* The calibration transfers imperfectly: M*_dm density is 1.20–1.24 vs 0.99–1.12 for A_dm.
* Particle methods with lower S_T than M* build 14–42 % fewer streets: top-k density 0.84, SMC 0.86, beam 0.58.

**6.7 Infrastructure (H5, descriptive; 3 river tiles).**
* Unbridged river crossings per tile: A 0.19–0.43, M* 0.
* Bridges per tile: A 3.2–3.9, M* 0.26–0.42, real 0.85. M* removes over-bridging but under-bridges.

**6.8 Compute.** Seconds per tile and sample:

| Sampler | s |
|---|---|
| A (standard) | 0.25 |
| M* | 0.52 |
| E + look-ahead | 2.5 |
| SMC | 2.9 |
| SMC + look-ahead | 8.5 |

**6.9 Global vs windowed (synthetic, 1536 m).**

| Method | Giant share | Density |
|---|---|---|
| Windowed | 0.74 | 14.2 |
| Global A | 0.68 | 9.4 |
| Global M* | 0.41 | 7.3 |

Tile-free generation reduces the density artefact at window seams (near / far 1.79 → 1.38–1.42). It does not reduce
fragmentation.

## 7. Ablations

DEV-B; TN_s0 unless noted.
* **Terrain-blind backbone (N0, 3 seeds).**
  * M* lowers S_T significantly: Δ −0.225, −0.270, −0.202, every interval below 0.
  * The terrain-conditioned backbone matters more: N0 + M* → TN + M*, Δ −0.73 to −0.85.
* **Constraint types (S_T).**

  | Constraints | S_T |
  |---|---|
  | none | 0.437 |
  | hard only | 0.399 |
  | soft only | 0.349 |
  | hard + soft (M*) | 0.327 |

  No pairwise interval excludes 0. Soft-only keeps the bridge excess (3.9 per tile); hard constraints remove it.
* **Horizon:** H = 1 gives 0.335 vs 0.327 for H = 0.
* **λ_infra = 0 is identical to M\*.** All feasible bridges on DEV-B have short spans and no acute water crossing, so
  the structure potentials are always 0.
* **Scheduler:** stochastic 0.359 vs fifo 0.327.
* **Procedural repair:**
  * giant share 0.90 → 0.99, unreachable pairs 0.25 → 0.04;
  * S_T 0.327 → 0.630 (interval [+0.14, +0.53]): the connectors fit terrain poorly.

## 8. Limitations

Known before results:
* **Infrastructure data is sparse:** 107 / 40 / 17 / 4 bridge runs by class and only 12 measured crossing angles.
  Viaduct claims are exploratory.
* **The leakage controls are imperfect.**
  * The low-pass removes real fine relief.
  * Street erasure uses the real street geometry, fills 40 % of each tile, and transplants roughness. It is a
    sensitivity analysis, not pre-development terrain.
* **Development data is small:** 25 + 27 tiles.
* **Look-ahead is local:** it covers one road continuation and guarantees no global connectivity.
* **Global generation uses tile-relative position features**, an approximation.
* **Three key related papers were read only as abstracts.** No learned external baseline was reproduced (no code or
  weights).
* **Small evaluation sets give wide intervals:** DEV-B has 27 tiles in 14 blocks; GEN has 100 tiles in 24 blocks.
* **The soft structure potentials never bound on DEV-B**, so their effect is untested.
* **The pre-registered failure rule is degenerate when A's premature share is 0:** one tile triggers it. Reported
  without changing any verdict.

## 9. Conclusion

* **The selected sampler.** Constrained inference over a trained autoregressive road generator reliably removes
  infeasible structures. Under a leakage-controlled, density-matched, pre-registered evaluation, it does not
  significantly improve the network's response to terrain.
* **Look-ahead** did not reduce premature termination or fragmentation.
* **Particle methods** traded density for apparent fit.
* **The terrain-conditioned backbone matters far more than the sampler,** and part of its fit comes from street
  beds in urban DEMs.
* **Future work** should use pre-development terrain or erased-DEM training throughout, a larger set of river and
  valley tiles, and global (not tile-relative) position features.

## References

See `docs/M13_LITERATURE_REVIEW.md` §2 (with access status per work).
