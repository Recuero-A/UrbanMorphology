# Terrain-Constrained Sampling for Autoregressive Road-Graph Generation

*Draft manuscript (Milestone 13). Bracketed items `[RESULT …]` are filled only from `outputs/m13/FINAL_REPORT.md`.*
*No result in this draft comes from the held-out test splits, which have not been read.*

## Abstract

`[RESULT: written last, from the confirmed findings only.]`

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
* `[CANDIDATE C1]` a terrain-constrained inference framework for autoregressive road-graph generation, with
  calibrated density-ratio potentials and state-consistent commitment;
* `[CANDIDATE C2]` an evaluation of look-ahead for terrain-induced premature termination and fragmentation;
* `[CANDIDATE C3]` a leakage-controlled, terrain-stratified, density-matched evaluation protocol: response curves,
  connectivity with unreachable pairs counted, street-erased and low-pass DEM controls.

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

* `[RESULT 6.1: selection on DEV-A: M*, E*, H*; table.]`
* `[RESULT 6.2: H1 / H1-lp / H1-se: S_T, M* vs A per training seed.]`
* `[RESULT 6.3: H2: look-ahead vs one-step: premature share, giant share.]`
* `[RESULT 6.4: H3: S10_cond and diversity.]`
* `[RESULT 6.5: H4 on GEN.]`
* `[RESULT 6.6: density-matched comparison.]`
* `[RESULT 6.7: infrastructure (descriptive, H5).]`
* `[RESULT 6.8: compute vs realism.]`
* `[RESULT 6.9: global vs windowed (X5, synthetic).]`

## 7. Ablations

`[RESULT: N0 + M*; constraint types; horizon; λ_infra = 0; scheduler; repair.]`

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
* **Three key related papers were read only as abstracts.**

## 9. Conclusion

`[RESULT: written last.]`

## References

See `docs/M13_LITERATURE_REVIEW.md` §2 (with access status per work).
