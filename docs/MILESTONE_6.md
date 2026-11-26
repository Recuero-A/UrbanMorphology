# Milestone 6: Hybrid neural-procedural generation (roads → buildings → green)

## Answer in brief

* **Buildings and green spaces are now generated** as parameterised polygons and exported as
  GeoPackage / GeoJSON, ready for 3D. On real street networks they match the real distributions
  of all 9 measured properties (within 1 IQR):
  * building count and coverage;
  * footprint size, aspect and rectangularity;
  * alignment with the street and distance to it;
  * clustering;
  * green coverage.

  They are always valid: 0 invalid polygons, 0 overlaps, 0 buildings on roads.
* **The network beats the statistical baseline only marginally** (mean W1/IQR 0.50 / 0.54 vs 0.55).
* **It does not preserve country differences** (pre-registered B4 fails). It produces
  similar buildings everywhere: median footprint ≈ 67 m² in every country, against 257 m² in
  Spain and 67 m² in Japan. Per-tile building counts correlate only weakly with the real tile
  (Spearman 0.41). The street network alone does not tell the model how built-up a tile is.
* **Roads v2 did not fix street structure**, although its likelihood improved strongly
  (validation NLL −588 / −510 per tile vs −464 / −474 in M5). Junction-angle deviation is
  still 13–16° (real 3.4°) and orientation order φ 0.10 (real 0.59). The turtle frame makes
  *teacher-forced* steps more accurate (share of > 20° errors 30 % → 24 %; median error
  3.9° → 2.1°), but errors **compound when the model continues from its own output**
  (exposure bias). The crossing / T-junction rules add the most skewed junctions (median 25.5°).
  My M5 hypothesis (absolute frame = main cause) was therefore **wrong or incomplete**.
* **Pre-registered decisions:** roads v2 **not met** (C2 structure fails, both seeds); buildings and
  green **not met** (B4 country differences fail; B1, B2, B3 and B5 pass).


Plan and literature: `docs/MILESTONE_6_PLAN.md`, `docs/HYBRID_GENERATION_LITERATURE.md`.
Milestone 5 is kept unchanged: its code paths, checkpoints and results are untouched; v2 is
selected by flags.

## 1. What is learned and what is programmed

| Stage | Learned (neural network, from real tiles) | Programmed (explicit rule) |
|---|---|---|
| A roads v2 | where components start; each step's **turn angle and length relative to the incoming street**; branching; stopping; **street class** | clip at the tile edge; join within 8 m; split at crossings; T-junctions |
| B land | — | road surface = every edge buffered by its class's median OSM width (training data: 9.75 / 10 / 8 / 6 m); land = tile minus road surface |
| C buildings | **count** (when to stop), **where** (per 4 m pixel), **orientation relative to the street**, **size**, **shape** (20 ray fractions) | made valid; clipped to the tile; rejected if > 25 % on the road surface or > 5 % overlap with another building |
| D green | the same, in a second phase conditioned on the roads and buildings | trimmed to land not covered by roads or buildings; rejected if < 20 m² remain or > 5 % overlap with other green |

**Thresholds come from real training tiles, not taste:**

* 14.6 % of real buildings touch the class-width road surface and 6.7 % lie more than 10 % on it.
  A "no overlap" rule would therefore contradict the data; 25 % is the tolerance.
* Real buildings touch but essentially never overlap (median overlap 0), hence 5 %.

**Bias check:**

* No rule prefers grids, straight roads, right angles, rectangles, uniform sizes or one connected
  component.
* Road width is drawn per class, from data.
* Every rule firing is counted and reported.

## 2. Stage A: roads v2 (turtle frame + street classes)

**The change.** Milestone 5 predicted each step in absolute (dx, dy) coordinates. Now the step,
the existing arms and the 48 × 48 crop are all expressed in the frame of the street that reached
the node (`sequence.py`, `turtle=True`):

* "straight on" is (+30 m, 0) at any compass heading;
* "a side street at 90°" is one learned mode, not 360 separate ones.

This is **rotation equivariance by construction**: it makes angles *relative* and lets the data
decide which angles occur (tested: crops and steps are unchanged under rotation). It does not
prefer 90°.

Each step also predicts the **street class** (OSM highway → major / mid / local / minor).

**Training changes:**

* cosine learning-rate schedule (M5's validation NLL was noisy at a constant learning rate);
* the start head is updated at every step (in M5, once per chunk).

**What the start head can learn.** Its validation NLL stays near the "uniform location" value in
both M5 and v2 (≈ 15.5 nats per tile). With random start nodes there is little to learn about
start locations. The number of components is decided by its STOP probability, checked in the
evaluation below.

## 3. Stages C and D: the object placer

**Representation (`layout/objects.py`).** Each building or green polygon is described by:

* an anchor (centre of its minimum rotated rectangle);
* an orientation relative to the nearest street (that street's direction is chosen so the object
  lies on its left, so an L-shape's facing is preserved);
* length ≥ width;
* 20 ray fractions (16 uniform rays + 4 rays to the rectangle corners):
  * a rectangle is reproduced exactly;
  * L, U, wedge and irregular shapes are reproduced approximately.

**Fidelity on real training buildings (median IoU, 10th percentile):**

| | Buildings | Green |
|---|---|---|
| Japan | 0.998, 0.95 | 0.97 |
| Spain | 0.987, 0.90 | 0.97 |
| US | 0.980, 0.90 | 0.95 |

**Model (`models/placer.py`, 1.50 M params).** It is autoregressive: all buildings in random
order, then STOP, then all green spaces, then STOP. Each step sees a 4-channel 128 × 128 canvas
(roads, buildings so far, green so far, phase).

| Part | Network | Output |
|---|---|---|
| Where | U-Net (+ coordinate channels) | logit per 4 m pixel (anchor uniform inside the pixel); STOP logit |
| What | CNN on a 32 × 32 crop rotated to the nearest street + U-Net feature at the pixel + 10 scalars (distance to street, street class, phase, counts) | orientation: mixture of 6 axial von Mises; log size: mixture of 6 bivariate Gaussians; shape: mixture of 4 diagonal Gaussians over 20 logit ray fractions |

**Training:**

* teacher-forced on **real** street networks of the 565 training tiles, with 8 symmetries and a
  fresh object order each epoch;
* 10 % of steps per epoch; cosine learning-rate schedule;
* early stopping on the validation NLL per object (19 validation tiles × 2 symmetries).

**P1, the statistical baseline (no neural network).** It uses the same rules:

* counts are drawn from real tiles;
* locations follow the real density of objects as a function of distance to the nearest street;
* attributes are resampled from real objects, relative to their street.

**What P2 must beat.** Simple statistics already give realistic-looking footprints, so P2 has to
do better than P1.

## 4. Evaluation protocol and decision criteria (fixed before generating samples)

**References:**

* real validation tiles (19) are primary;
* real training tiles (565) are a secondary reference;
* test tiles are never used.

**Part 1, roads v2 against M5.** The same protocol and criteria as M5 (C1–C4; success = both
seeds). Additionally reported: junction-angle deviation, single-component share, and class shares.

**Part 2, buildings and green on REAL validation street networks (conditional).** For P1 and P2
(both seeds): 4 samples per validation tile, 76 layouts per method.

* Per-tile metrics:
  * building count and coverage;
  * median log area, median aspect, median rectangularity;
  * share aligned with the street (parallel or perpendicular, ±10°);
  * median distance to the street;
  * Clark–Evans ratio (clustering);
  * green coverage.
* **B1:** P2 W1/IQR to real validation ≤ 1 for ≥ 7 of these 9 metrics (both seeds).
* **B2:** P2's mean W1/IQR is lower than P1's (the network adds something; both seeds).
* **B3:** validity:
  * 0 invalid polygons;
  * overlapping building pairs per building ≤ the real value;
  * share of buildings > 25 % on road ≤ the real value.
* **B4:** country differences preserved: for building count, median log area and distance to the
  street, the ranking of the three countries' medians in P2 equals the real ranking (both seeds).
* **B5:** no memorisation: median nearest-training-tile IoU (3-channel raster, 8 symmetries)
  ≤ real validation's + 0.10.
* **Stage C/D success** = B1–B5.

**Part 3, the full pipeline.**

* **D:** roads v2 seed 0 → P2 seed 0, 64 tiles, against the raster baselines from the same brief:
  * **A:** Spatial VAE C2 with z ~ N(0, I);
  * **B:** C2 + PixelCNN prior (M4).
* **Shared raster metrics:**
  * road topology (pieces, floating fragments, junctions, φ);
  * building and green coverage;
  * building pieces and median piece area;
  * diversity and memorisation.
* **Not a strictly fair comparison:**
  * D has explicit geometry and staged conditioning;
  * A and B generate pixels jointly;
  * D's rasters use per-class median widths, while real rasters use per-street OSM widths.

  Reported, not used for a decision.

## 5. Results

### 5.1 Training

| Model | Params | Best epoch / run | Time (MPS) | Validation NLL |
|---|---|---|---|---|
| roads v2 seed 0 / 1 | 1.39 M | 55 / 60, 60 / 60 | 35 / 39 min | −588.5 / −510.2 per tile (M5: −464.4 / −474.3) |
| placer seed 0 / 1 | 1.50 M | 30 / 30, 29 / 30 | 63 / 46 min | 18.44 / 18.94 per object |

* Roads v2 seed 1 was still improving at epoch 60 (not fully converged).
* The NLLs of v2 and M5 are comparable: rotation has Jacobian 1. v2 also pays for street classes,
  so its advantage is, if anything, understated.

### 5.2 Part 1: roads v2 (256 tiles per seed; medians; `outputs/m6/results.json`)

| | Pieces | Floating | Components | Junctions/km² | Dead-end ratio | Loops/km² | φ | Link (m) | Street km/km² | Junction-angle dev. | Single-component tiles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Real val | 2 | 0 | 2 | 76 | 0.056 | 31 | 0.59 | 81 | 17.4 | 3.4° | 42 % |
| M5 seed 0 | 1 | 0 | 1 | 328 | 0.078 | 181 | 0.15 | 44 | 25.3 | 14.4° | 88 % |
| **v2 seed 0** | 1 | 0 | 1 | 378 | 0.173 | 196 | 0.10 | 43 | 30.7 | 15.7° | 73 % |
| **v2 seed 1** | 1 | 0 | 1 | 385 | 0.099 | 227 | 0.11 | 40 | 29.0 | 13.7° | 81 % |

**Other measures:**
* Street-class shares (major / mid / local / minor): real 0.12 / 0.20 / 0.67 / 0.01;
  v2 0.12–0.13 / 0.16–0.19 / 0.62 / 0.07–0.10. The classes are learned well.
* Diversity 0.91 / 0.92 (real 0.94).
* Nearest-train road IoU 0.148 / 0.145 (real 0.137). No memorisation.

**Decision C1–C4** (same as M5): C1 ✅, C2 ❌ (2 of 6 metrics within 1 IQR), C3 ✅, C4 ✅,
giving **not met** for both seeds.

**Diagnosis** (`outputs/m6/junction_angles.png`):

| Test | M5 | v2 |
|---|---|---|
| Teacher-forced first step at real nodes: median angular error | 3.9° | **2.1°** |
| … share > 20° | 30.5 % | **23.7 %** |
| Free sampling, junction-angle deviation, junctions made by the model | 14.4° (pooled) | 14.4° |
| … junctions made by the crossing / T rules (≈ 24 % of junction gaps) | — | **25.5°** |
| Sampling outcomes: new node / joined node / split street / dropped | 80 / 10 / 6 / 4 % | 80 / 9 / 6 / 5 % |

**Reading:**
* The model predicts the next step of a **real** network well, and better than M5.
* Streets slowly bend (about 2° median error per step, compounding over 10+ steps per street).
  The model never saw its own imperfect drawings in training.
* Bent streets then meet other streets at odd angles, and the crossing rule turns each meeting
  into a skewed junction.
* Lower NLL did **not** mean better networks (cf. Milestone 4).

### 5.3 Part 2: buildings and green on REAL validation street networks (76 layouts per method)

| | Count | Coverage | Median area | Aspect | Rectangularity | Aligned ±10° | Dist. to street (m) | Clark–Evans | Green cov. | Mean W1/IQR |
|---|---|---|---|---|---|---|---|---|---|---|
| Real val (19 tiles) | 462 | 0.273 | 105 m² | 1.73 | 0.97 | 0.84 | 22.0 | 1.03 | 0.012 | — |
| P1 statistical | 393 | 0.141 | 58 m² | 1.63 | 0.99 | 0.87 | 19.6 | 1.11 | 0.008 | 0.55 |
| **P2 seed 0** | 382 | 0.182 | 66 m² | 1.57 | 0.93 | 0.86 | 22.2 | 1.07 | 0.017 | **0.50** |
| **P2 seed 1** | 364 | 0.146 | 66 m² | 1.65 | 0.94 | 0.87 | 20.7 | 1.04 | 0.023 | **0.54** |

**Validity:**

* Real: share > 25 % on road 0.002; overlapping pairs 0.
* P1 and P2: 0 invalid polygons, 0 overlaps, 0 buildings > 25 % on road.

**How often each rule fired:**

| | Accepted | Rejected: overlap | Rejected: on road | Locations given up |
|---|---|---|---|---|
| P2 seed 0 | 34,241 | 41,093 | 6,977 | 9,585 |
| P1 | 35,485 | 118,871 | 22,122 | 31,836 |

The network proposes far fewer invalid buildings than the statistics-only baseline: it has
learned where there is free land.

**Diversity and memorisation (3-channel raster):**

| | Diversity | Nearest-train IoU |
|---|---|---|
| Real val | 0.908 | 0.186 |
| P1 | 0.935 | 0.144 |
| P2 | 0.932 / 0.935 | 0.143 / 0.144 |

**Per country (medians; the generator sees only the roads, not the country):**

| | Count real / P2 s0 / P1 | Median area (m²) real / P2 s0 | Dist. to street real / P2 s0 | Aligned real / P2 s0 |
|---|---|---|---|---|
| Spain | 89 / 527 / 361 | 257 / 67 | 27.4 / 21.8 | 0.81 / 0.85 |
| United States | 507 / 342 / 466 | 104 / 70 | 24.8 / 23.2 | 0.96 / 0.87 |
| Japan | 656 / 332 / 274 | 67 / 63 | 17.4 / 21.7 | 0.73 / 0.86 |

* **Learned STOP:** the building phase ended by the learned STOP on all 19 validation tiles, never
  by the give-up rule.
* **Per-tile counts:** generated vs real counts have Spearman ρ = 0.41.

**Decision:**

| | B1 realism | B2 beats P1 | B3 validity | B4 country ranking | B5 memorisation | Success |
|---|---|---|---|---|---|---|
| P2 seed 0 | ✅ 9 / 9 | ✅ | ✅ | ❌ | ✅ | **no** |
| P2 seed 1 | ✅ 9 / 9 | ✅ | ✅ | ❌ | ✅ | **no** |

**Reading:**
* **Pooled distributions match, but each tile is "average".** Japan is approximated well;
  Spanish large-footprint blocks (Eixample perimeter blocks, industry) and US rows of houses are
  not reproduced.
* **B1 was too lenient.** It is computed on the pooled distribution, so averaging styles can
  still pass it.
* **The likely cause is that roads under-determine buildings.** The same street grid hosts
  warehouses, perimeter blocks or detached houses. With no country or density input, the model
  learns the mixture.

### 5.4 Part 3: full pipeline vs raster baselines (64 tiles each; raster medians; context only)

| | Pieces | Floating | Junctions/km² | φ | Building cov. | Green cov. | Building pieces | Median piece | Diversity | Nearest-train IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| Real val (OSM widths) | 2 | 0 | 111 | 0.70 | 0.273 | 0.012 | 141 | 224 m² | 0.910 | 0.183 |
| A: Spatial VAE, N(0, I) | 18 | 7 | 275 | 0.48 | 0.360 | 0.118 | 60 | 660 m² | 0.880 | 0.178 |
| B: VAE + PixelCNN prior | 10 | 4 | 158 | 0.64 | 0.246 | 0.083 | 44 | 596 m² | 0.925 | 0.189 |
| **D: hybrid** | **1** | **0** | 355 | 0.32 | 0.148 | 0.023 | **172** | **96 m²** | 0.935 | 0.155 |

**D vs the raster baselines:**
* **Better:** connectivity, separate buildings of realistic size, realistic green amount, validity.
* **Worse:** street orientation order and junction density; too little building coverage.

**Not a fair comparison:** D generates explicit geometry in stages, while A and B generate all
pixels jointly; D's widths are per-class medians.

Figures:
* `outputs/m6/layouts_on_real_roads.png`
* `outputs/m6/full_pipeline.png`
* `outputs/m6/roads_v2_samples.png`
* `outputs/m6/junction_angles.png`
* `outputs/m6/building_metrics.png`
* `outputs/m6/raster_comparison.png`

Exports: `outputs/m6/export/generated_{0..3}.gpkg` (layers: roads with class and width,
road_surface, buildings, green) and `generated_0.geojson`.

## 6. What this means and the next experiments (proposed, not run)

1. **Exposure bias in roads.** Evidence: teacher-forced is good, free-running is bad.
   * Train on the model's own rollouts: scheduled sampling, or DAgger-style correction targets.
   * Or condition on a per-tile **orientation field / latent**, drawn first, so streets share
     directions across the tile. This is NTG-independent and closer to Yang et al. 2013's cross
     fields, but learned.
   * Re-examine the crossing rule. Option: only allow crossings the model "intended", i.e. steps
     that end within a few metres of a street.
2. **Buildings need context the streets do not carry.**
   * Condition the placer on country (one-hot) and on the GHSL density stratum we already sampled
     with.
   * Re-test B4 with per-country W1, not pooled.
   * This is cheap: the same model with 5 extra inputs.
3. **The dataset expansion** (plan §5) becomes useful once these inputs exist. More tiles per
   country and per density stratum is exactly what conditioning needs.

## 7. Limitations

* 2 seeds; 19 validation tiles (6–7 per country).
* Per-country results are therefore indicative.
* Roads v2 seed 1 was not fully converged.
* Road surface uses per-class median widths.
* Buildings below 10 m² and green below 20 m² are excluded.
* No parcels (OSM rarely maps them).
* Footprints must be star-shaped from the rectangle centre (median IoU 0.98–1.0, 10th percentile
  ≈ 0.90).
* Green spaces are mostly small gardens and lawns in OSM.
* Generated layouts have no geographic location (local metres).

## 8. Reproduce

```
python scripts/m6_build_roads.py && python scripts/m6_build_objects.py
python scripts/m6_train_roads.py --seed 0 && python scripts/m6_train_roads.py --seed 1      # ~35-40 min each (MPS)
python scripts/m6_train_placer.py --seed 0 && python scripts/m6_train_placer.py --seed 1    # ~45-65 min each
python scripts/m6_evaluate.py                                                               # ~10 min
pytest tests/test_graph_gen.py tests/test_layout.py
```

