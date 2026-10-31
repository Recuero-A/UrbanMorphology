# Milestone 10 — Block-level generation on real streets

Pre-registration (§5) was written 2026-10-31, before any experiment in this document was run. Results are in
§6 and in `MILESTONE_10.md`.

## 1. Why blocks, and why on real streets

* **The literature conditions on given streets.** GlobalMapper and the npj 2026 model generate buildings per
  block from the block shape (`M10_LITERATURE_REVIEW.md` §1.3–1.4).
* **Separating the two problems.** Generating buildings on real streets removes street-generator errors, so
  building quality is measured on its own. The two are combined only in Phase 5.
* **Our data:** about 6.7k closed blocks in train and validation (`outputs/m10/audit/dataset_audit.json`), plus open
  blocks cut by the tile window.

## 2. Land-use audit: the "no frontage → green" rule is wrong

`scripts/m10_landuse_audit.py` → `outputs/m10/landuse/landuse_audit.json`, `landuse.png`. Training tiles only.

### 2.1 Observed cadastral parcels

Parcels cut by the tile window are excluded. Each parcel is classified as built, public green, private green, or
unbuilt other:

| | Built | Public green | Private green | Unbuilt other |
|---|---|---|---|---|
| Spain, street parcels (n = 19,577) | 0.886 | 0.013 | 0.002 | 0.099 |
| **Spain, parcels without frontage** (n = 3,809) | **0.743** | 0.031 | 0.002 | 0.224 |
| US, street parcels (n = 29,490) | 0.972 | 0.003 | 0.001 | 0.024 |
| **US, parcels without frontage** (n = 5,481) | **0.936** | 0.009 | 0.000 | 0.055 |

### 2.2 Closed-block interiors

The interior is the part of a closed block more than 20 m from its edge. Area shares:

* built: 0.31–0.55;
* public green: 0.00–0.09;
* private / unknown-access gardens: 0.00–0.04;
* **untagged unbuilt** (courtyards, yards, parking, vacant land): **0.35–0.66**.

### 2.3 Consequences

1. **M9b's rule fails on the data.** The rule turns land without street access into green, which caused M9b's
   40% green coverage. In the observed data, land without frontage is mostly *built*; green is 1–3%.
2. **"Unbuilt" is not "green".** Most unbuilt interior land carries no green tag.
   * OSM cannot separate vacant parcels, paved courtyards and private yards. This audit therefore reports
     them together as *unbuilt other*.
   * Separating them would need cadastral land-use codes. Spanish cadastre parcels have use codes in a
     different product, which would be a new download; it is not done, per the constraints.
3. **Categories used from now on:**
   * *public green* (park, playground, …, unless access is private);
   * *private green* (gardens, or green with private access);
   * *unbuilt other* (yards, courtyards, vacant or paved land, unclassified);
   * *built*.

## 3. Procedural fix (method BF)

`src/urbanmorph/layout/tidy_parcels.py` adds `green_fix`, used as `--m10` mode in `scripts/m9_evaluate.py`.
Everything else is M9b (method B) unchanged.

* **Land without street access** (no-access pieces of M9b, and block remainders of M9 that are ≥ 400 m²):
  * it becomes green only with the observed probability P(green | no frontage) — Spain 0.033, US 0.0095;
  * otherwise, if it is no bigger than one typical lot, it merges into the neighbouring lot as a backyard
    (unbuilt);
  * otherwise it becomes interior building lots.
* The learned TZ zoning model and the "none" block type are unchanged; they were learned from data.

## 4. Learned block generator (BlockGen)

### 4.1 Design

The design follows GlobalMapper's "condition on the block" idea, but with a raster canvas so that inpainting
is natural.

* **Canonical frame.** The block's minimum rotated rectangle: long axis along x, at 2 m per pixel, in a
  128 × 128 window (256 m) centred on the block.
* **Larger blocks** are generated in overlapping windows. Each later window is *outpainted* from the
  buildings already placed, using the same known-building channel as inpainting.
* **Inputs** (4 channels):
  * block mask;
  * distance to the block edge (frontage depth);
  * known buildings;
  * known-region mask.

  Plus a learned city embedding (9 cities).
* **Model.** A small conditional DDPM U-Net over the building mask:
  * about 3–5M parameters;
  * cosine schedule;
  * 50-step DDIM sampling.

  The literature's caution about diffusion (§3 of the review) concerns large pretrained raster models and
  streets. Here the canvas is small and binary, and inpainting is the explicit requirement.
* **Training.**
  * Data: training tiles only. Every block, or a random window inside a large block. Random flips of the
    canonical frame.
  * The known region is random: none (50%), a half-plane (25%), or random building dropout (25%).
* **Vectorising.** Connected components ≥ 10 m² become polygons, simplified at 1 m and clipped to the block.

### 4.1b Changes made during implementation (before any evaluation; criteria untouched)

| Planned | Implemented | Why |
|---|---|---|
| 4 input channels, including distance to the block edge | **3 channels**: block mask, known buildings, known-region mask | The U-Net sees the block mask at every scale, so frontage depth is derivable from it. This saves one channel per window in the data file. |
| Known region: none, half-plane, or building dropout | none (50%), half-plane through a random point (25%), **random rectangle** (25%) | A rectangle also exercises the outpainting window shape. |
| "3–5M parameters" | 4,159,169 parameters (channels 32, 64, 128, 128), loss weight 0.05 outside the block, AdamW 2e-4, EMA 0.999 | — |
| Training length not fixed | **20,000 steps**, batch 32, both seeds (5.6 h each on MPS while sharing the GPU) | Chosen before training from the 1-epoch smoke run, so as to fit the session. Validation loss was still falling slowly (0.0181 at 16k, 0.0175 at 20k). |

### 4.2 Evaluation

* **Tiles:** the M9 evaluation tiles — 10 validation tiles with observed parcels (Spain 6, US 4).
* **Reference:** real validation streets, compared with the M9 building metrics (`tile_row`: count, coverage,
  log area, aspect, rectangularity, alignment, distance to road, Clark–Evans).
* **Samples:** 2 seeds, 4 samples per tile. BlockGen is also reported on all 19 validation tiles, as a
  reference.

## 5. Pre-registered criteria

### 5.1 BF vs B

Same 10 tiles, 2 seeds, 4 samples per tile. Criteria must hold for both seeds:

* **F1** W1/IQR of green coverage for BF is below B's, and ≤ 1.0.
* **F2** The built-parcel share is within 0.10 of the observed share, in both Spain and the US. This is M9b's
  `built_parcels` criterion.
* **F3** The median building count is within ±25% of real. This is M9's G5.
* **F4** Validity (G1) holds, and at most 1 of the 6 primary M9 metrics is worse than B's by more than 0.1.

### 5.2 BlockGen (G) vs the best procedural method (B or BF, chosen per seed)

* **G1** S_B (the mean W1/IQR over the 8 building metrics, green excluded) is lower than the procedural
  method's, for both seeds.
* **G2** Validity: less than 1% of generated building area lies outside its block (on roads), and no
  overlapping buildings.
* **G3** The median building count is within ±25% of real, and the building-coverage W1/IQR is ≤ 1.0.

### 5.3 Inpainting (I)

This is a conditional experiment.

* **Setup.**
  * On each validation block, real buildings on one side of a random line through the block centroid are
    given.
  * The other side is generated: (a) by inpainting, and (b) by unconditional BlockGen on the same block, for
    comparison.
* **I1** On the hidden side, the median absolute error per block of building coverage is lower for
  inpainting than for unconditional generation. The paired tile-bootstrap 95% interval of the difference
  must lie below 0, for both seeds.
* **I2** The same holds for building count, without the bootstrap requirement.

## 6. Results (2026-11-01)

### 6.1 BF vs B (`scripts/m9_evaluate.py --m10` → `outputs/m10/procedural_fix/results.json`, `decision`)

Same 10 tiles, 4 samples per tile. Real values: 381.5 buildings per tile, green coverage 0.013, observed
built-parcel share 0.93 (Spain) and 0.92 (US).

| | B, s0 / s1 | BF, s0 / s1 | Criterion | BF result |
|---|---|---|---|---|
| Green coverage (median) | 0.40 / 0.39 | 0.050 / 0.065 | F1: W1/IQR below B's and ≤ 1.0 (B 4.5, BF 0.32–0.36) | **pass** |
| Built-parcel share, Spain / US | — | 0.48 / 0.82 and 0.47 / 0.85 | F2: within 0.10 of observed | **FAIL** |
| Buildings per tile | 114 / 132 | 276.5 / 291.5 (−27.5% / −23.6%) | F3: within ±25% | **FAIL** for s0, pass for s1 |
| Validity, and primary metrics worse than B by > 0.1 | — | valid; 0 of 6 worse | F4 | **pass** |

**BF fails** (F2 both seeds, F3 seed 0).
* What it fixes: the green problem, with 6–8× less green than B, and it more than doubles the building count.
* Why it falls short: interior lots are created, but many get no building (`not_fitted`). The Spanish
  built share stays near 0.47, against 0.93 observed.

### 6.2 BlockGen (G) on real streets (`scripts/m10_eval_blocks.py` → `outputs/m10/blocks/eval.json`)

| | Real (10 M9 tiles) | BlockGen s0 / s1 | BF s0 / s1 |
|---|---|---|---|
| S_B (8 building metrics; lower is better) | — | 1.034 / 1.065 | 0.640 / 0.605 |
| Buildings per tile | 381.5 | 96 / 97.5 | 276.5 / 291.5 |
| Building coverage | 0.29 | 0.45 / 0.50 | — |
| Rectangularity | 0.92 | 0.72 / 0.72 | — |
| Share of buildings aligned to the street (10°) | 0.87 | 0.63 / 0.61 | — |
| Validity: area on roads / overlaps | — | 0 / 0 | — |

* **G1 FAIL** (S_B is above BF's for both seeds). **G2 pass.** **G3 FAIL** (count −75%, coverage W1/IQR 1.6–1.8).
* **BlockGen fails.**
* On all 19 validation tiles (reference only): S_B 0.94 / 1.05, with 158 / 139 buildings against 462 real.
* Per-tile sampling time: 4–6 s on MPS.

**Why BlockGen fails** (`outputs/m10/blocks/roundtrip.json`, `blockgen_vs_real.png`):
1. **The binary mask caps the count.**
   * Drawing the *real* buildings of each validation block on the canvas and vectorising them back drops the
     count from 462 to 235 per tile (−49%), and rectangularity from 0.97 to 0.82.
   * Attached buildings (shared walls, as in Spanish perimeter blocks and NYC row houses) become one component.
   * So about half of the count gap (462 → 235) is the representation, and the other half (235 → about 150) is the model.
2. **Over-filling.**
   * Coverage is 0.46–0.50, against 0.25 for the round trip.
   * Blocks that are empty in reality (parks, vacant land: Madrid e877, Valencia e1427) get filled. The model
     has no input that says a block is open space. In BF, the learned zoning model supplies this.
3. **Ragged edges.** At 2 m per pixel with 50 DDIM steps, footprints are not rectilinear (rectangularity 0.72).

### 6.3 Inpainting (I), conditional

239 blocks on 17 tiles. Per-block absolute error on the hidden side, tile means:

| | Seed 0 | Seed 1 |
|---|---|---|
| Coverage error, median: inpainting / unconditional | 0.166 / 0.285 | 0.132 / 0.382 |
| Paired tile difference, 95% CI | [−0.150, −0.049] | [−0.316, −0.170] |
| **I1** | **pass** | **pass** |
| Count error, median: inpainting / unconditional | 3 / 4 | 3 / 3 |
| **I2** (the median must be lower) | **pass** | **FAIL** (tie) |

**I1 passes for both seeds; I2 fails for seed 1.**
* Given half a block, BlockGen continues it far better than it generates from scratch: the coverage error
  roughly halves (`inpainting.png`).
* This is the strongest BlockGen result, and the one that motivates editing workflows.
