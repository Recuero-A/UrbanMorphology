# Milestone 15 protocol: style guidance on the existing samplers (pilot)

Status: **frozen 2026-11-20, before any evaluation run.** Development tuning (style strengths) used only the
development synthetic sites and seeds 1–3 (`outputs/m15/dev_select.md`). Every M15 result is labelled **pilot**.
Amendments are appended in §8 with their reason and date; nothing above §8 changes after evaluation runs start.

## 1. Question and scope

> Can a turtle-graph generator adapt street growth to terrain while preserving a requested street morphology, and
> produce a range of plausible layouts on unseen sites?

* **Scope:** neighbourhood-scale (512 m tiles) vehicular street networks. Terrain proxies are elevation and water only.
  Forest, soil, rock and weather are deferred.
* **Not new in M15:** the generator (M12 TerrainGraphTurtle), the M13 terrain-aware sampler M\*, M13-SMC, and the
  M14 network-aware samplers. They are kept as they are and named in `src/urbanmorph/experiments/methods.py`.
* **New in M15:** optional style guidance (`src/urbanmorph/graphs/style.py`), added as one more factor on each
  candidate's weight inside any of these samplers. Also new: metrics, fixtures, a tracer baseline, and the benchmark.
* **Literature and overlap:** `docs/M15_LITERATURE.md`. Style heading fields, mixtures and terrain-aware growth are
  established in procedural work (Parish & Müller 2001; Chen et al. 2008; Galin et al. 2010). Learned style
  conditioning without terrain exists (RoadWeaver 2026). Any novelty is limited to the combination and its matched
  evaluation.

## 2. Synthetic interventions (first experiment)

**Sites.** `src/urbanmorph/experiments/sites.py`. Each condition has one development instance (used for tuning only)
and one evaluation instance with different azimuths and parameters:

| Condition | Evaluation instance |
|---|---|
| flat | 80 m |
| incline 8 % | azimuth 25° |
| incline 15 % | azimuth 300° |
| Gaussian ridge | 40 m high, 150 m wide, azimuth 15° |
| Gaussian valley | 40 m deep, 150 m wide, azimuth 130° |
| water barrier | straight 40 m river, azimuth 40° |

**Fixed across all runs:**
* Domain: 512 m tile with a 256 m context ring, 4 m raster.
* Start rule: the learned start head with the same seed.
* Proposal count: K = 8.
* Look-ahead budget: H = 2, 4 roll-out candidates per step.
* Round budget: 1500.
* Water: the shared M13 eligibility rule (no ground edge over or ending in water; bridges only over rivers, span ≤
  class p99, crossing angle ≥ real p5), applied identically to every variant, the tracer included.

**Seeds.** Evaluation inference seeds 101–110: paired, the same for every variant. Backbone TN_s0. The TN_s1
replication is a secondary check (§5).

**Styles** (`configs/m15_style.json`, strengths frozen by the pre-written rule on development data):

| Style | Setting |
|---|---|
| grid | θ0 = 0°, κ = 4 |
| organic | smooth cross field, θ0 = 0°, RMS 45°, wavelengths 250–500 m, κ = 8 |
| radial | centre = tile centre, κ = 8 |
| cul-de-sac | stop bias +0.25, loop-closure penalty 2 nats |
| off | no style factor |

Strengths apply to every sampler and to the tracer. The style factor is normalised within each decision, so it
changes which continuation is chosen, not whether an arm stops; termination is a separate control. Local and minor
streets get full strength; mid and major streets get 0.5 and 0.25. The declared scenario per site is the style name,
θ0 and the centre; nothing else about any reference network is given.

**Variants:**

| Group | Variants |
|---|---|
| Core (the 2 × 2) | M\* style off / on; NA-fixed style off / on; every style |
| Existing comparisons (style off) | A, NA-myopic, NA-SMC, M13-SMC, P (procedural) |
| Morphology-only | M\* + style with soft terrain potentials off (`soft = False`; hard rules unchanged) |
| Hard-mask control | M\* + style and NA-fixed + style with grade ceiling `real_p99` (class p99 of real streets) |
| External baseline | Chen-inspired cross-field tracer (`src/urbanmorph/graphs/field_tracer.py`), each style, same graph operations, same hard rules, d_sep 78 m, step 26 m, 6 km length budget. It is not the authors' code |
| Negative control | M\* and NA-fixed, style off and grid, shown terrain rotated 90° about the tile centre (same condition, other direction), scored on the true terrain. Flat excluded |
| Grade-ceiling sensitivity | NA-fixed + grid with ceiling `real_p99`, 0.12, 0.08 |
| Ablation | M\* + grid with the un-normalised style factor |
| Temperature (separately) | M\* and NA-fixed, style off and grid, displacement temperature 0.8 and 1.0 (0.5 is the base). Every other setting is unchanged, so the effect of temperature is not mixed with that of style |

## 3. Outcomes (pre-declared) and smallest effects worth reporting

Metrics: `src/urbanmorph/graphs/style_metrics.py`, definitions and units in `docs/M15_METRICS.md`. Everything is
scored on the true terrain.

**O1. Style control on flat terrain** (paired style on − off; independent descriptors):

| Style | Descriptor | Smallest effect |
|---|---|---|
| grid | orthogonal order R4 | +0.05 |
| radial | radial + tangential length share about the declared centre | +0.05 |
| cul-de-sac | loops per km² | −5 |
| cul-de-sac | interior dead ends per km | +0.3 |
| organic | R4 below the grid-style R4 | 0.05 lower |

Field alignment (the generator's own objective) is reported as a controllability check only.

**O2. Morphology retention on terrain.** The O1 effect pooled over the five non-flat conditions, divided by the
flat-terrain effect.
* Retention ≥ 0.5 is non-inferior. The margin is half the flat effect: the development sweep's 18 tile-samples per
  setting gave noisy effects, and a reduction to half is the largest loss we would still call "retained".
* Reported per sampler (M\*, NA-fixed) and for the tracer.

**O3. Do NA-fixed's connectivity gains survive?**
* Paired NA-fixed − M\* on same-land OD unreachability, with style on (pooled over styles and conditions), next to
  the same contrast with style off.
* The gain survives if the style-on contrast is ≤ −0.01 (the smallest effect worth reporting; M14's GEN2 effect was
  −0.026), and its interval excludes 0.

**O4. Terrain exposure.**
* Measures: the length share of ground streets steeper than 15 % (10 m sub-spans on the true raster), and the
  length-weighted p90 grade. Conditions: incline 8 %, incline 15 %, ridge, valley.
* (a) Style on − off for each sampler: non-inferiority margin +0.02 absolute.
* (b) Each sampler + style vs the tracer, and vs the negative control: superiority (lower exposure).
* (c) Against the hard-mask control, exposure is compared below the cap, so compliance alone cannot win: we report
  the share of streets with grade above 10 % but under the cap, and the achieved length.
* Achieved street length and 60 m coverage are always reported next to exposure. A variant that reaches lower
  exposure with < 0.8 × the comparison's street length is flagged "fewer roads", not counted as better.

**Also reported:** validity (empty outputs, failures, budget hits, rejected candidates by reason), runtime, diversity
(mean pairwise Jaccard distance across the 10 seeds of a site), and all runs in every denominator.

## 4. Statistics (synthetic)

* Six synthetic sites are not a population of cities. Seeds are repeated draws on a fixed site, never independent
  sites.
* Effects are paired by (site, seed). Intervals are percentile bootstraps (5000 draws) resampling seeds within sites
  (stratified by site).
* Per-site effects are shown, and a leave-one-site-out range is reported.
* No claim about real-world generalisation is made from synthetic interventions; they are evidence about model
  behaviour.

## 5. Real-city pilot (descriptive; development metros only)

**Development metros.** Six metros in three countries:

| Country | Metros | Tiles |
|---|---|---|
| Spain | Girona, Manresa | the M13 GEN tiles |
| South Korea | Busan, Daejeon | the M2 v1 pilot tiles |
| Peru | Arequipa, Cusco | the M2 v1 pilot tiles |

**Window selection.** Windows were chosen by the earlier GHSL built-up rules (M2 v1 stratified design; M13 GEN rule),
never by street metrics. Flat and hilly strata are assigned afterwards from GLO-30 landform slope: median slope
< 5 % flat, ≥ 10 % hilly, in between mixed. This is independent of the street network.

**Elevation.** Copernicus GLO-30 for every metro: the same source, a DSM at 30 m, used as a terrain proxy.
* Building and vegetation artefacts are documented.
* The model input is GLO-30 bilinearly resampled to 4 m.
* Grade exposure on real sites uses 30 m sub-spans (the source resolution).
* For Spain, the IGN MDT05 bare-earth DTM is a sensitivity check.

**OSM.** The cached, dated Geofabrik extracts listed in the manifest.
* Road classes, projection (UTM per metro), intersection simplification and boundary rules are the project's M2/M12
  ones, shared by real and generated graphs.
* Border nodes are kept, and boundary dead ends are censored.
* Bridges and tunnels follow the M12 structure labels and the shared eligibility rule.
* Korea and Peru were dropped in M2 because OSM *building* mapping was incomplete. Street completeness there is not
  independently verified; this is a stated limitation.

**Tasks.**
* **A (unconditional):** the model gets the terrain and the window only, style off. Real-data resemblance uses
  features outside every generator objective: log median and log p90 interior block area, 3-way and 4-way junction
  shares, and OD circuity in the 100–200 and 200–350 m bands. Unreachable OD pairs are counted separately.
* **B (declared style, a separate task):** grid style with θ0 declared from the real network's dominant orientation,
  on windows whose real R4 ≥ 0.5. R4 is then a conditioned quantity and is excluded as evidence.

**Fidelity.** Precision, recall, density and coverage (Kynkäänniemi et al. 2019; Naeem et al. 2020) in
real-standardised feature space, k = 5. They are read against a real-vs-real benchmark of metro-split halves.
Coverage-retention target: generated coverage ≥ 0.8 × real-vs-real coverage.

**Units.** City-level paired effects, windows nested in metros, seeds as repeated draws. With six metros the pilot is
descriptive. Leave-one-metro-out ranges are reported, and a variance-based sample-size calculation for a confirmatory
benchmark is given.

**Held-out metros (declared, reserved, not built or evaluated in M15):**
* Spain: Lleida and Alicante;
* South Korea: Seoul and Daegu;
* Peru: Lima and Trujillo;
* external country: the USA (Sacramento, Santa Barbara; 3DEP and GLO-30).

Building or evaluating them needs explicit authorisation from the project owner, as for the earlier sealed test
splits (which stay untouched).

## 6. Claim rule (fixed now)

A defensible M15 claim needs all three of the following:
1. improvement over the strongest matched baseline in terrain exposure at useful coverage (O4 b, without the "fewer
   roads" flag);
2. morphology retained within the O2 margin;
3. held-out distributions with adequate diversity and coverage.

The third cannot be met in M15, because held-out metros are reserved. Otherwise the claim is revised to describe
where and how control breaks down, with the evidence.

## 7. Reproduction

```
python scripts/m15_plans.py eval            # writes configs/m15_plans/eval.json from this protocol
python scripts/m15_run.py --plan configs/m15_plans/eval.json --worker k/n
python scripts/m15_report.py                # outputs/m15/REPORT.md, figures
```

## 8. Amendments

**Amendment 1 (2026-11-20).** Fixed after the protocol commit, before any real-city run and while only synthetic
evaluation runs were in progress. It adds detail to §5 and changes nothing above.
* The real-pilot plan is `scripts/m15_plans.py real`.
  * Task A: A, M\*, NA-fixed, NA-SMC and P, style off, on all 120 usable development windows.
  * Task B: M\* and NA-fixed with the declared grid on the 64 windows with real R4 ≥ 0.5. The off arm is the Task A
    run on the same windows.
  * DEM sensitivity: M\* and NA-fixed, style off, on the 60 Spanish windows with IGN MDT05.
* Seeds 101–103 for all of these.
* GLO-30 grades use 30 m sub-spans.
* For real windows the radial centre is estimated by the fixed rule (`estimate_centre`); no centre is declared.
* All 120 development windows passed preparation: the manifest lists no exclusions.
* Slope strata from GLO-30 (flat / mixed / hilly):

  | Metro | Flat | Mixed | Hilly |
  |---|---|---|---|
  | Girona | 6 | 17 | 7 |
  | Manresa | 1 | 25 | 4 |
  | Busan | 8 | 5 | 2 |
  | Daejeon | 6 | 8 | 1 |
  | Arequipa | 2 | 10 | 3 |
  | Cusco | 3 | 4 | 8 |

**Amendment 2 (2026-11-20).** Made after most of the synthetic evaluation runs had completed and a partial report
had been produced, so it was made with knowledge of partial results; it changes how starts are placed, not any
outcome, margin or analysis.

* **The deviation.** §2 states that the start rule is "the learned start head with the same seed". The project
  brief requires fixed starting points.
* **The problem.** With the learned start, the start head ended the tile before placing any street:
  * on the flat evaluation site in 3 of 10 seeds;
  * on the water site in 8 of 10 seeds.

  It did so identically for every sampler, since the start decision precedes any candidate scoring. These empty
  outputs make the flat-terrain (O1) and water-barrier results mostly unusable.
* **The fix.** All synthetic evaluation runs are repeated with a fixed starting point: the first start node is placed
  at the tile centre, or at the nearest dry point on a 4 m grid (`SamplerConfig.fixed_start = "centre"`). Further
  starts use the learned start head as before.
* **Unchanged:**
  * the style strengths, frozen on development sites with the learned start;
  * the real-city pilot, which keeps the learned start because it evaluates the whole generator (1–2 % empty
    outputs there).
* **Reporting.** The amended runs are primary. The as-run learned-start runs are kept under their own run IDs. They
  are summarised in an appendix of the report (empty outputs and the O3 contrast), and they are incomplete: the
  temperature and TN_s1 runs were stopped when the amendment was made.
