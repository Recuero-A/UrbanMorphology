# Milestone 3b: Improving generative urban layouts through controlled experiments

**Date:** 2026-10-30 · **Branch:** `milestone-3b` (Milestone 3 frozen at commit `fc35d7f`;
baseline numbers in `outputs/m3/BASELINE_M3.{md,json}`).
**Data used:** training and validation only. The 19-tile test split was never loaded.

## Summary

* **The biggest single change was the switch to a spatial latent** (16 × 8 × 8 instead of one
  32-number vector). Validation road IoU rose from 0.11 to 0.72 on the same expanded data, and
  reconstructions now recover real street grids.
  *Caveat:* the change also raised latent capacity 32× (from 32 to 1,024 numbers), so it cannot be
  attributed to spatial structure alone.
* **More data helps, and more so for the spatial model:** +0.06 road IoU for the global latent,
  +0.27 for the spatial latent.
* **Loss modification:** weighting the road BCE ×3 (C2) gave a further, seed-robust +0.09 road IoU
  over B. A soft-Dice term (C1) gave +0.03 (single seed; suggestive only).
* **Generation is still not realistic.** Prior samples from the spatial VAE contain street-like
  *texture*, with plausible road widths (≈ 10–12 m) and coverage, but they are over-branched
  (2.5–3.5× the real intersection density), lack long continuous streets, and are globally
  incoherent. This is consistent with a prior that is **factorised over positions**.

---

## 1. Hypotheses

| | Hypothesis | Test |
|---|---|---|
| H1 | The baseline underfits partly because 97 tiles are too few | A: same model and objective on 565 tiles |
| H2 | One global 32-number latent cannot carry 1–5 px street geometry; a spatial latent can | B vs A (same data and objective) |
| H3 | Roads are under-weighted by pixel-wise BCE; a road-aware objective improves them | C1 (Dice) and C2 (road BCE ×3) vs B |
| H4 | Unrealistic prior samples come from a mismatch between prior and aggregate posterior | prior vs fitted-posterior samples vs reconstructions, plus alignment statistics |

## 2. Dataset expansion (training only)

**Feasibility audit** (`scripts/m3b_expansion_plan.py`, cached snapshot 2026-10-28, no network):

| spacing between train tile centres | possible new tiles | Barcelona | Valencia |
|---|---|---|---|
| 1,024 m (no adjacent tiles) | 2,373 | 28 | 20 |
| 1,536 m (M2 rule) | 1,015 | 9 | 5 |

**Chosen (approved): 1,024 m spacing, cap of 60 per city, giving +468 tiles.** Training grows
from 97 to 565 tiles (BCN 28, VLC 20, every other city 60). Rules, each enforced by assertions
(`scripts/m3b_build_expansion.py`) and tested (`tests/test_m3b.py`):
* same population as M2 (GHSL-eligible cells of the same study areas), same strata, new seed;
* no tile inside a 3 km block that contains a validation or test tile;
* every tile ≥ 1,000 m (edge to edge) from every validation and test tile;
* no reuse of any existing cell, and no two training tiles adjacent (centre distance ≥ 1,024 m);
* same processing and quality rules as M2. All 468 tiles were usable. Warnings: 39 incomplete
  buildings, 11 channel overlap, 6 no roads; info: 84 empty green.

**Unchanged:** `samples.csv`, the split assignment (97 / 19 / 19), every validation and test tile.
The expansion lives in a separate index, `data/processed/m2/samples_expansion_m3b.csv`.
**Balance:** Spain makes up 140 of 565 training tiles (25%) because Barcelona and Valencia cannot
supply more. Within-train spatial autocorrelation is higher than in M2 (tiles 1,024 m apart rather than 1,536 m).

## 3. Experimental design

Every run = the frozen M3 config (`configs/m3_vae.toml`: Adam lr 1e-3, batch 16, 8-fold
augmentation, `sqrt(neg/pos)` positive weights, KL warm-up 20 epochs, β = 0.5, ≤ 300 epochs,
early stopping on validation loss with patience 40) **plus only the overrides in
`configs/m3b_experiments.toml`**.

| run | data | architecture | latent | params | loss change | best epoch / run | train time |
|---|---|---|---|---|---|---|---|
| baseline `vae_z32_b0.5` | 97 | global ConvVAE | 32 | 3,665,891 | none | 272 / 300 | 66 s |
| A `A_global_z32_exp` | 565 | global ConvVAE | 32 | 3,665,891 | none | 124 / 164 | 375 s |
| control `B_spatial16_orig97` | 97 | SpatialVAE | 16×8×8 | 2,122,435 | none | 196 / 236 | 79 s |
| B `B_spatial16_exp` (+ seed 1) | 565 | SpatialVAE | 16×8×8 | 2,122,435 | none | 295, 288 | 690 s, 547 s |
| C1 `C1_spatial16_dice` | 565 | SpatialVAE | 16×8×8 | 2,122,435 | + soft Dice (weight 1) | 287 | 542 s |
| C2 `C2_spatial16_roadx3` (+ seed 1) | 565 | SpatialVAE | 16×8×8 | 2,122,435 | road BCE × 3 | 298, 289 | 554 s, 609 s |

Total training ≈ 57 min on Apple Silicon (MPS). Positive weights are recomputed from the
training-set pixel frequencies, so they change slightly when the data change (part of "more data").

### Architecture difference (B vs the M3 VAE)

* **Global ConvVAE (M3):** conv trunk → (256, 8, 8) → *flatten* → linear layers → μ, log σ² ∈ ℝ³²
  → linear → (256, 8, 8) → upsampling decoder. All geometry must pass through 32 numbers, and
  1.6M of the parameters are in the linear layers.
* **SpatialVAE (B):** the same conv trunk → (256, 8, 8) → **1×1 conv** → μ, log σ² ∈ ℝ^(16×8×8)
  → 3×3 conv → the same upsampling decoder. Each latent position describes a 64 m × 64 m patch
  (with a larger receptive field). KL is summed over all 1,024 elements against N(0, I).
* **What else changes (confounds):** latent capacity (32 → 1,024 numbers), parameter count
  (3.67M → 2.12M), and the prior, which is now *factorised over positions* (no spatial correlation).

### Loss modifications

* **C1, soft Dice:** `Dice_c = (2Σpt + 1) / (Σp + Σt + 1)` per channel. The term added is
  `1 · H·W · (1 − Dice_c)`, so its magnitude is comparable to the per-channel BCE sum. Dice
  rewards overlap regardless of class frequency, but **it knows nothing about connectivity**:
  broken roads can still score well.
* **C2, per-channel weighting:** the road channel's BCE sum is multiplied by 3. This also says
  nothing about topology.

## 4. Evaluation protocol (identical for all runs: `scripts/m3b_evaluate.py`)

On the 19 original validation tiles, threshold p ≥ 0.5:
* **Reconstruction (from the posterior mean):** per-channel BCE, IoU, coverage error and bias.
* **Road network** (new, `src/urbanmorph/metrics/road_raster.py`): Zhang–Suen skeleton with spurs
  ≤ 12 m pruned, then:

| metric | meaning | limitation |
|---|---|---|
| mean road width = area / skeleton length | streets ≈ 3–20 m; blobs have huge "width" | thick avenues and plazas also widen it |
| intersections / km² | junction-pixel clusters | ≈ 1.8× the vector-graph count on real tiles (parallel carriageways, junction splits); rank correlation 0.80, so use it for **comparison only** |
| skeleton components | fragmentation | tile edges cut real networks |
| mean segment length | continuity between junctions | sensitive to spurious junctions |
| dead-end ratio, largest-component share | network shape | tile-edge endpoints count as dead ends |

  On the real validation tiles: road width 8.4 m, 111 raster intersections/km², 3 components,
  segments 30 m.
* **Generation (64 samples each):** **prior** z ~ N(0, I) (the actual generative model) and
  **fitted** z ~ a diagonal Gaussian fitted to the training posterior means. The fitted version is
  a **diagnostic only, not a valid way to sample from the VAE.** For each: coverage, diversity,
  nearest-training-tile IoU (memorisation), road-network metrics.
* **Prior alignment:** std of posterior means per dimension (prior: 1), posterior σ, KL of the
  fitted aggregate posterior to N(0, I) per dimension, active dimensions.

## 5. Results (`outputs/m3b/comparison_all.{csv,md}`, `outputs/m3b/iou_summary.png`)

### Reconstruction (validation IoU; mean over seeds where two)

| run | roads | buildings | green | road BCE | recon road width | recon intersections/km² |
|---|---|---|---|---|---|---|
| baseline (global, 97) | 0.046 | 0.252 | 0.413 | 0.452 | 55.6 m | 8 |
| A (global, 565) | 0.106 | 0.287 | 0.537 | 0.430 | 41.6 m | 8 |
| control (spatial, 97) | 0.450 | 0.426 | 0.658 | — | 19.9 m | — |
| B (spatial, 565), 2 seeds | **0.718 / 0.728** | 0.596 / 0.602 | 0.768 / 0.786 | 0.123 / 0.121 | 11.7 / 11.1 m | 153 / 179 |
| C1 (+Dice), 1 seed | 0.745 | 0.601 | 0.869 | 0.124 | 11.2 m | 164 |
| C2 (road ×3), 2 seeds | **0.818 / 0.812** | 0.607 / 0.582 | 0.633 / 0.751 | 0.073 / 0.077 | 9.7 / 10.2 m | 149 / 153 |
| *real validation tiles* | | | | | *8.4 m* | *111* |

Seed-to-seed variation: road IoU ±0.01, building IoU ±0.01–0.03, **green IoU up to ±0.12**.
Validation has only 4% green pixels, so green differences of < 0.15 are not interpretable.

### Generation (64 samples; real validation: roads 0.138, buildings 0.271, green 0.040)

| run | prior coverage (roads / bldg / green) | prior road width | prior intersections/km² | fitted-diagnostic intersections/km² | diversity | nearest-train IoU |
|---|---|---|---|---|---|---|
| baseline | 0.004 / 0.136 / 0.226 | 32 m | 0 | 8 | 0.70 | 0.16 |
| A | 0.021 / 0.117 / 0.144 | 35 m | 0 | 4 | 0.95 | 0.18 |
| B (seed 0 / 1) | 0.206 / 0.295 / 0.054 · 0.225 / 0.313 / 0.053 | 11.9 m | 296 / 317 | 263 / 261 | 0.90 | 0.17 |
| C1 | 0.255 / 0.267 / 0.009 | 11.9 m | 381 | 336 | 0.90 | 0.16 |
| C2 (seed 0 / 1) | 0.148 / 0.358 / 0.120 · 0.174 / 0.397 / 0.103 | 9.8–10.7 m | 275 / 301 | 208 / 229 | 0.88 | 0.18 |

No model memorises: the nearest-training-tile IoU of samples (0.16–0.18) matches that of real
validation tiles (0.16).

### Prior alignment (H4)

| run | std of μ per dim (prior 1) | posterior σ | aggregate KL / dim | active dims |
|---|---|---|---|---|
| baseline | 3.28 | 0.10 | 4.57 | 32/32 |
| A | 2.31 | 0.06 | 1.53 | 32/32 |
| B | 0.97 | 0.13 | 1.12 | 640/1024 |
| C2 | 0.99 | 0.09 | 0.02 | 1024/1024 |

## 6. Interpretation (what can and cannot be attributed)

1. **More data (A vs baseline; control vs B).** With the architecture held fixed, more data
   improved every channel: global +0.06 road IoU, spatial +0.27. The data effect **depends on the
   architecture** (an interaction), so a global VAE cannot show what more data does for a spatial
   model. "More data" here also changes optimisation steps per epoch and the positive weights,
   so it is not a pure sample-size effect.
2. **Architecture (B vs A; control vs baseline).** This is the dominant effect: +0.40 road IoU at
   97 tiles and +0.61 at 565 tiles. Reconstructed road widths drop from 42–56 m (blobs) to
   11–12 m (streets, real 8.4 m). **It cannot be credited to "spatial structure" alone**, because
   latent capacity rose 32× and the parameter count fell. The next control is a global latent with
   1,024 dimensions, or a spatial latent of 1 × 8 × 8 (64 numbers).
3. **Loss (C1, C2 vs B; data and architecture fixed).** Road BCE ×3 improved road IoU by +0.09
   (two seeds each, ≈ 9× the seed variation), and road BCE and road width moved towards the real
   values. It also produced fewer spurious junctions in samples (275–301 vs 296–317 for B), but
   still 2.5× too many. Dice gave +0.03 on roads (single seed, ≈ 2–3× the seed variation) and
   *more* spurious junctions in samples (381). **Neither change addresses topology**: both
   improve pixel overlap, and the intersection over-count shows that network structure is still wrong.
4. **Latent sampling (H4).**
   * *Global VAEs:* yes, misaligned. The posterior means spread 2–3× wider than the prior. Samples
     from the fitted posterior have more realistic coverage than prior samples, but they still
     contain no streets, because the decoder itself cannot draw streets.
   * *Spatial VAEs:* the **marginal** alignment is good (std of μ ≈ 1), and prior and fitted
     samples look alike. Yet both are over-branched and globally incoherent. The remaining problem
     is **not** marginal misalignment but the **missing spatial dependence** in a prior that is
     factorised over positions: each 64 m patch is sampled independently, so streets do not
     continue across patches and grids have no consistent orientation.
   * The fitted-posterior samples are a diagnostic and are not presented as successful VAE sampling.

**Answer to "do any generated neighbourhoods look realistic?":** no. Reconstructions of real
tiles are now good, which is a learned *compression* of urban layouts. Unconditional prior samples
have realistic *local statistics* (road width, coverage) but unrealistic *network structure*.

## 7. Visualisations

* `outputs/m3b/comparison_grid.png`: the same 3 validation tiles, 3 prior samples and 3
  fitted-diagnostic samples for every model (same seeds).
* `outputs/m3b/iou_summary.png`: per-channel IoU for every run, with seeds marked.
* `outputs/m3b/figures/<run>/{reconstructions,samples_prior,samples_fitted_diagnostic}.png`.
* `outputs/m3/runs/<run>/history.json`: training histories.

## 8. Limitations

* Validation has 19 tiles (green especially is noisy), and C1 has one seed while B and C2 have two.
* B confounds spatial structure with latent capacity and parameter count.
* Raster intersection counts are biased upwards relative to the vector graph (≈ 1.8× on real tiles).
* Road widths in the labels are partly default values (Japan most of all), so a "realistic width"
  means close to the *label* width.
* Rotation augmentation removes north-up orientation, by design.
* Expansion tiles are denser (1,024 m spacing), so effective sample diversity is below 565 independent tiles.
* Every model was trained to ≤ 300 epochs. The spatial models were still improving at the cap
  (best epochs 287–298).

## 9. Conclusions and next experiment

* What improved reconstructions: **spatial / high-capacity latent ≫ more data > road-weighted loss**.
* What remains poor: **generation**, specifically street-network coherence.
* **Recommended next experiment (Milestone 4):** keep the spatial VAE (C2) as an encoder/decoder,
  and replace the factorised N(0, I) prior with a **learned spatial prior over the 16 × 8 × 8
  latent grid**. Options: an autoregressive prior (PixelCNN-style), or a small *latent diffusion*
  model over the 8 × 8 latents. That directly targets the cross-patch dependence that is missing.
  Pair it with the capacity control (a global latent with 1,024 dimensions) and a topology-aware
  check (skeleton intersections and segment lengths against real tiles) as the primary generation metric.
* Do not touch the test split until a final model is chosen.

## 10. Reproduce

```bash
python scripts/m3b_expansion_plan.py                      # feasibility report
python scripts/m3b_expansion_plan.py --write --per-city 60 --spacing 1024
python scripts/m2_download.py --plan data/processed/m2/sampling/expansion_m3b.csv   # local, ~12 min
python scripts/m3b_build_expansion.py                     # processing + split assertions
python scripts/m3b_experiments.py                         # A, B, C1, C2 + evaluation (~40 min)
python scripts/m3b_seed_check.py B_spatial16_exp C2_spatial16_roadx3 --seed 1
python scripts/m3b_data_control.py                        # spatial VAE on the original 97 tiles
python -m pytest -q                                        # 87 tests
```
