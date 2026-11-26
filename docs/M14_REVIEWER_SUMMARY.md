# UrbanMorph — summary for an external reviewer (status 2026-11-20)

This document is self-contained. It assumes expertise in generative modelling of spatial structure, but no knowledge
of this project. Every number comes from development data. The sealed test splits have not been evaluated.
Pointers to the detailed documents are given at the end.

## 1. Field and question

**Field.** Data-driven procedural modelling of cities. It sits between three areas:
* computer graphics (procedural and inverse-procedural content generation);
* machine learning (autoregressive graph generation and inference-time control of a learned prior: SMC,
  look-ahead, potentials);
* urban science / GIScience (quantitative street-network morphology).

**Overall question.** Can a learned generator produce street networks that are realistic, diverse, respond to
terrain and water, and generalize to cities it was not trained on?

**Question of the most recent milestone (M14).** Learned sequential road generators make local decisions. The
resulting networks are fragmented: they contain disconnected pieces and dead ends that cannot be closed. Can
inference-time, network-level control fix this without retraining and without losing realism or diversity?

## 2. System under study

**Data.**
* 512 m × 512 m tiles on a 4 m grid, built from OpenStreetMap street graphs and national LiDAR DEMs (Spain IGN
  MDT05, US USGS 3DEP).
* Spatial splits: 3 km blocks are assigned whole to a split, with a 1 km buffer between splits. Test-tile IDs are
  hash-locked in the test suite.

**Generator ("TerrainGraphTurtle").** An autoregressive vector road-graph model in the style of Neural Turtle
Graphics (Chu et al., ICCV 2019).
* A breadth-first queue of frontier nodes is expanded one road step (~26 m) at a time.
* A GRU encodes the incoming path.
* A CNN encodes a local terrain raster.
* Heads predict: stop / continue, a mixture-density displacement, the road class, and the structure (ground,
  bridge, tunnel).
* **Conditioning is terrain only:** elevation, slope and water. There is no city label, no style token, and no
  surrounding street context.

**Backbones.** Three variants, each trained with three seeds:

| Backbone | Elevation input | Role |
|---|---|---|
| TN | Raw DEM | Main model |
| TNlp | Low-passed DEM (σ = 16 m) | Leakage control |
| TNse | DEM with street beds erased | Leakage control |

The leakage controls exist because urban bare-earth DEMs contain the imprint of the real streets (cuts and
embankments), so the raw-DEM model can partly see the answer.

**Base sampler M\* (from M13).**
1. At each decision, draw K = 8 candidate steps from the model.
2. Apply hard feasibility rules: no unbridged water crossings, no infeasible grades, snapping and merging.
3. Apply soft terrain potentials.
4. Select one candidate in proportion to model probability × potentials.

## 3. Method under test (M14)

**Network potentials.** The tilted target is

> π_λ(τ) ∝ p_θ(τ) · G_terrain(τ) · exp(−λ_D · D(τ) − λ_I · I(τ))

where D counts *closed* dead ends and I counts *closed* islands (components that can no longer grow).

* **Proposition 1 (telescoping).** Both counts decompose exactly into per-step increments. A node or component
  can only become "closed" at an identifiable step, so the factor can be applied incrementally. This is tested to
  machine precision, including closures that happen without an explicit STOP.
* **Fitting λ.** λ is fitted by iterative-scaling moment matching (I-projection form) on 48 training tiles, so
  that generated dead-end and island rates match the real rates. The result is λ_D = 1.81 and λ_I = 0.
* **Caveat.** The moments are matched under the *implemented* sampler, not the exact tilt. So the
  minimum-KL interpretation holds only approximately.

**Samplers compared.** All use the same backbone and potentials.

| Sampler | How it draws a sample |
|---|---|
| NA-myopic | M\* plus the network factor at each step |
| NA-fixed | Plus a twisted selection: each candidate is rolled out H = 2 steps (4 candidates per roll-out step), and the estimated future normaliser multiplies its weight. This is a heuristic, truncated twist |
| NA-adaptive | Roll-outs only where a trigger fires. The trigger (`deg1`) was selected on development data by a value-of-computation proxy |
| NA-random | Roll-outs at random decisions, at the adaptive trigger's firing rate |
| NA-SMC | N = 4 particles, multiplied by an unbiased estimate of the local normaliser, resampled when ESS < N/2, no look-ahead. Asymptotically exact for π_λ; biased at N = 4 |

**Baselines.**

| Baseline | What it is |
|---|---|
| A | Unconstrained ancestral sampling |
| M\* | The M13 base sampler (§2) |
| M\*+repair | M\* followed by A\*-style post-hoc connectivity repair |
| M13-SMC | SMC without network potentials |
| P | Terrain-aware procedural generator, density-matched to training tiles |

**Not run:** an external learned baseline. Raster methods would need retraining and take ground-truth junctions
as input, and NTG has no released code.

## 4. Protocol

**Pre-registration.** Hypotheses, the primary metric, margins and the statistical tests were fixed before the
confirmatory runs. Amendments made after the pilot are recorded with their reasons.

**Primary set: GEN2.** 95 tiles from four cities that had not been inspected before the confirmatory runs:
* Alcoi and Tarragona (Spain), with hilly and partly organic fabric;
* Sacramento and Santa Barbara (USA), with grid fabric.

**Secondary sets:**
* GEN: Girona, Manresa, San Francisco (100 tiles);
* DEV-B: 27 validation tiles from the training cities' validation blocks.

**Runs.** 3 backbones × 3 training seeds × 3 inference seeds.

**Primary outcome.** Same-land OD unreachability: the share of random point pairs that lie on the same dry-land
region but have no network path between them.

**Realism.**
* S10, the mean W1/IQR gap to the real tiles over 10 network statistics. The real-vs-real noise floor on GEN2 is
  0.293.
* S_T, a terrain-response gap.
* Energy distance on joint features.

**Diversity.** Mean pairwise Jaccard distance of the 8 m occupancy grids of samples from the same tile.

**Inference.**
* The unit is the geographic block (3 km).
* Paired block sign-flip randomisation and block bootstrap CIs.
* Holm correction over the primary family.

**Density.** Density is a known confounder of S10, so three analyses were run:
* regression adjustment;
* stop-bias calibration to matched density;
* density-matched re-runs.

## 5. Confirmatory results (GEN2 primary unless stated)

| Hypothesis (pre-registered test) | Result | Verdict |
|---|---|---|
| **H1.** Network-aware look-ahead improves connectivity (NA-fixed − M\* on OD unreachability < 0) | Δ = −0.026 [−0.036, −0.015], p < 10⁻⁴ (0.037 → 0.011, about −70 %). Negative in 9/9 set × seed cells, 7/7 cities, 7/7 terrain strata. Also DEV-B −0.021 (p = 5·10⁻⁴) and GEN −0.017 (p < 10⁻⁴) | **Supported** |
| **H2.** Strong planning damages realism (S10 rise from λ×0 to λ×4 exceeds the noise floor; tested on DEV-B) | S10 at λ × {0, 0.5, 1, 2, 4} = 0.41 / 0.35 / 0.53 / 0.62 / 0.73. Rise +0.32 < floor 0.42 | Not supported (trend in the predicted direction) |
| **H3.** Adaptive planning improves the trade-off (non-inferior to fixed at ≤ 0.7× cost, *and* better than a random trigger) | Cheaper (0.55×) and equally connected, but S10 +0.09 to +0.12 vs fixed (margin 0.02). Indistinguishable from a random trigger at the same rate (p = 0.66 / 0.49 / 0.81) | **Not met** (main negative result) |
| **H4.** Diversity is preserved (Jaccard ≥ 0.9 × A) | Ratio 0.974 (GEN2), 0.994 (DEV-B), 0.987 (GEN) | **Met** |
| **H5.** The effect survives leakage controls (CI excludes 0 in ≥ 2/3 seeds for each of TNlp and TNse) | 3/3 seeds for both. Holm-significant. One TNlp seed has sign-flip p = 0.072 | **Supported** |

**Trade-off by sampler (GEN2, mean over 3 × 3 runs):**

| Method | OD unreach. | Islands / tile | S10 | Energy dist. | Density / real | Cost vs M\* |
|---|---|---|---|---|---|---|
| M\* | 0.037 | 0.28 | 0.369 | 0.65 | 1.12 | 1× |
| M\*+repair | 0.006 | 0.02 | 0.424 (S_T +0.17) | 0.70 | 1.15 | — |
| M13-SMC | 0.048 | 0.23 | **0.309** | 0.52 | 0.90 | — |
| NA-myopic | 0.010 | 0.08 | 0.501 | 0.48 | 1.34 | 1.3× |
| NA-fixed | 0.011 | 0.07 | 0.370 | **0.26** | 1.21 | 15× |
| **NA-SMC** | 0.012 | **0.02** | 0.332 | 0.27 | 1.11 | 4.3× |

Real GEN2 networks have OD unreachability 0.047.

**Reading of the results.**
* NA-SMC is the best trade-off: the full connectivity gain, no realism loss, and the lowest density inflation.
* Post-hoc repair reaches the lowest OD, but by adding straight connectors that ignore terrain (detour 1.59,
  unbridged links).

**Caveats the authors consider material.**
1. **Realism differences are small.** Differences in S10 between methods are inside the real-vs-real noise floor,
   and they reverse under density matching. The only realism claim made is "no degradation" for NA-fixed and
   NA-SMC.
2. **"More connected" is not "more real" on GEN2.** Real GEN2 networks are themselves fragmented (Tarragona
   0.128 on non-empty tiles, against 0.000–0.028 for every generator). The claim is "removes the generator's fragmentation deficit", not "moves every statistic toward real".
3. **Analysis bugs fixed after confirmation.** Three were found: cross-set data contamination, a run lookup that
   picked the wrong run, and mixed seed counts in two-seed analyses. All moved the analysis toward the
   pre-registered specification, and none changed a verdict. They are documented in M14_FAILURE_ANALYSIS §4.
4. **The backbone's start head failed on 5 GEN tiles.** It produced empty output for every method.

## 6. Exploratory (not pre-registered): does the generator adapt its *style*?

**Motivation.** Visually, generated towns look "too linear" and not organic. Classic procedural systems (Parish &
Müller, SIGGRAPH 2001) switch style by explicit global goals such as grid or radial patterns. Our model has no
style input, so any adaptation must come through the terrain channels.

**Per-tile style statistics (real vs generated, same tile):**
* Orientation order φ (Boeing, *Applied Network Science* 2019): 0 = uniform orientations, 1 = an ideal grid.
* Share of 4-way junctions.
* Junction-angle deviation from 90°.
* Link sinuosity (junction-to-junction path length ÷ straight distance).
* Median turn at interior vertices.

**Adaptation measures:**
* Spearman ρ across tiles;
* the OLS slope of generated on real (1 = follows fully, 0 = the same style everywhere);
* block-bootstrap CIs.

**References:** a style-blind procedural generator (P), and a "city oracle" that predicts every tile by its city's
real mean.

**φ adaptation on GEN2 (TN_s0):**

| Method | φ ρ | φ slope |
|---|---|---|
| P | 0.17 | 0.05 |
| city oracle | 0.49 | 0.25 |
| M\* | 0.73 [0.57, 0.84] | 0.73 [0.56, 0.86] |
| NA-fixed | 0.77 [0.64, 0.84] | 0.74 |
| NA-SMC | 0.68 [0.52, 0.77] | 0.70 |

GEN and DEV-B are similar: ρ 0.6–0.8 and slope 0.6–0.85.

**Findings.**
1. **The learned generators adapt their orientation style per tile, not only per city.** ρ is well above both the
   city oracle and P. The network potentials do not reduce this adaptation.
2. **Part of it is leakage.** With street-imprint-free DEMs, φ ρ falls from 0.71–0.75 to 0.47–0.66 (10–35 %),
   and 4-way-junction ρ from about 0.57 to about 0.22. Part of the apparent adaptation is the model reading street
   beds in the DEM.
3. **Contrast is compressed and shifted.** Generated φ is below real in every city, by about 0.1 to 0.2:

   | City | Real φ | Generated φ |
   |---|---|---|
   | Sacramento | 0.78 | 0.62–0.71 |
   | Alcoi | 0.47 | 0.22–0.37 |
   | Girona | 0.41 | 0.25–0.38 |

   Junction density shows the same compression: Alcoi real 119 per km² vs generated 80–90, and Sacramento real 49
   vs generated 59–68. The generator regresses toward a "middle" style: grids are not crisp, and dense organic
   cores are not fine-grained.
4. **Why it looks "linear".** Real streets in this data are piecewise straight: the median turn at an interior
   vertex is 0–2°. Generated streets have a 4–10° median turn at every ~26 m step, while their overall sinuosity
   is similar to real (1.08–1.27 vs 1.05–1.22). So generated roads are straight-ish lines with uniform jitter,
   not deliberate curves. Sinuosity and turn adaptation are weak: ρ is about 0.3–0.5.
   * *Hypothesis (untested):* the training recipe adds 2.5 m positional noise (`noise_m`), which at 26 m steps
     corresponds to roughly this turn magnitude.

**Qualitative 3D demonstrations (one sample each, synthetic terrain, illustrative only).**
* On flat terrain, raising the displacement temperature from 0.5 to 0.8–1.0 yields less regular layouts for
  NA-SMC and NA-fixed.
* At the same higher temperatures, the single-chain samplers (A, M\*, NA-myopic) terminated after one road in
  most cases.
* On difficult synthetic relief, M\* and A often terminated early, while the network-aware samplers produced
  connected towns.

None of this is quantified yet.

## 7. Questions we would like the reviewer to judge

1. Is the novelty claim appropriately narrow? The claim is exact telescoping network-count potentials,
   moment-matched and applied at inference to a learned terrain-conditioned vector road generator. The machinery
   (count potentials, I-projection, twisted SMC) is established. Three recent works (Zhou et al. 2024 and two
   2025 diffusion works) could be read only as abstracts or metadata.
2. Is same-land OD unreachability the right primary connectivity metric? It is chosen to avoid rewarding
   unbridged water crossings.
3. Is S10 (W1/IQR over 10 statistics) an adequate realism measure, given its density sensitivity? What would you
   use instead? Candidates are per-statistic matched-density comparisons, or a learned feature distance like
   NTG's FID variant.
4. Is the style-adaptation analysis (§6) a meaningful complement? Should explicit style conditioning be added?
5. Which external learned baseline would you consider mandatory?

## 8. Where to look

| Topic | File |
|---|---|
| Formulation, propositions, what each sampler targets | `docs/M14_MATHEMATICAL_FORMULATION.md` |
| Protocol and amendments | `docs/M14_EXPERIMENTAL_PROTOCOL.md` |
| All confirmatory numbers | `outputs/m14/REPORT.md`, `docs/M14_STATISTICAL_ANALYSIS.md`, `docs/M14_GENERALIZATION_RESULTS.md` |
| Failures and post-hoc fixes | `docs/M14_FAILURE_ANALYSIS.md` |
| Style adaptation (this §6) | `outputs/m14/style/STYLE_ADAPT.md`, `scripts/m14_style_adapt.py`, `outputs/m14/figures/fig_style_adapt_gen2.png` |
| Sampler implementation | `src/urbanmorph/graphs/m14_sampler.py` (NetSampler), `src/urbanmorph/graphs/m14_metrics.py` |
| Manuscript draft and internal review | `docs/M14_PAPER_DRAFT.md`, `docs/M14_FINAL_REVIEW.md` |
