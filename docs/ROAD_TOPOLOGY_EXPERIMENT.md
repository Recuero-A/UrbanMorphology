# Road-topology experiment: can soft-clDice improve connectivity without reducing diversity?

**Date:** 2026-10-30 · **Branch:** `road-topology` (from `milestone-3b`, commit `6dbdde4`).
**Data:** 565 training / 19 validation tiles. The test split was not used. Existing checkpoints
and results are preserved.
**Generated tables:** `outputs/topology/results_tables.md`. **Raw results:** `outputs/topology/eval/*.json`.

## Answer in brief

* **No.** In this setup, soft-clDice did **not robustly improve road connectivity**, either in
  reconstruction or in generation.
* At a strong weight it introduced a mild, consistent **over-connection bias**: more junctions and
  fewer dead ends, in all three countries. At a very strong weight it collapsed into **flooding the
  tile with road**.
* Diversity and between-country differences were largely **preserved at weak and strong λ**, with
  a small, consistent drop in sample diversity at λ = 1.
* **The fragmentation of generated layouts is a generation (prior) problem, not a reconstruction
  problem.** Reconstructions are already topologically close to the truth (centreline agreement
  clDice ≈ 0.93), and a loss on reconstructions cannot fix how latent codes are sampled.
* The diagnosis also shows that the fragmentation is **not caused by seams between latent cells**.
  It reflects missing *non-linear* dependence in the prior.

---

## 1. Hypotheses

| | Hypothesis | Verdict |
|---|---|---|
| H1 | clDice on roads improves paired reconstruction topology (road pieces, interior fragments, loops) | **Not supported** (no effect beyond seed variation; strong λ worsens intersections) |
| H2 | It improves the topology of unconditional samples | **Not supported** (distributions essentially unchanged at λ ≤ 1) |
| H3 | It does not impose structure: no grid bias, no loss of dead ends or diversity, no country homogenisation | **Partly:** no grid bias, country differences preserved; but strong λ adds junctions, removes dead ends, and slightly lowers diversity |
| H4 | Generation fragmentation comes from discontinuities at latent-cell seams under the factorised prior | **Rejected:** no seam effect; the cause is missing non-linear dependence in the prior |

## 2. Implementation

* **Starting point (A):** C2 = SpatialVAE, latent 16 × 8 × 8, 2,122,435 parameters, β = 0.5 with
  KL warm-up, √(neg/pos)-weighted BCE with road ×3, 8-fold augmentation, 300 epochs, early stopping
  on validation. Prior N(0, I), independent per latent element.
* **soft-clDice** (`src/urbanmorph/models/cldice.py`): this reproduces the official PyTorch code of
  Shit et al. (CVPR 2021; github.com/jocpae/clDice, MIT) operation for operation:
  * soft erosion = min-pool over 3×1 and 1×3; soft dilation = 3×3 max-pool;
  * soft skeleton built from `relu(x − open(x))` over 10 iterations;
  * Tprec = (|S_pred·V_true| + 1)/(|S_pred| + 1), Tsens = (|S_true·V_pred| + 1)/(|S_true| + 1);
  * loss = 1 − 2·Tprec·Tsens/(Tprec + Tsens).
* **Objective:** L = L_C2 + λ · (128·128) · mean_tiles(1 − clDice_roads).
  * applied to roads only; building and green terms are unchanged (tested);
  * the H·W factor puts λ = 1 on the scale of the summed BCE.
* **Deviation from the official code:** sums are taken *per tile* and then averaged, rather than
  over the whole batch, so tiles with long networks do not dominate. With a batch of 1 the result
  is identical to the official code (tested).
* **Tests** (`tests/test_cldice.py`, 6 tests):
  * equals the official implementation;
  * empty prediction and empty target gives 0, and hallucinated or missed roads give > 0.5;
  * the skeleton is about 1 px wide;
  * buildings and green are unaffected;
  * the clDice term alone produces encoder gradients;
  * **a 24-pixel gap and a 24-pixel thinning have identical Dice loss (0.057 vs 0.057), while
    clDice penalises only the gap (0.058 vs 0.000).**
* No term prefers straight roads, grids or particular angles.

## 3. Ablations and selection

All runs are identical to C2 except `cldice_weight` (`scripts/topo_experiments.py`). They use the
same data, architecture, seeds (0 and 1) and 300-epoch budget. Training takes about 7–15 min per
run on MPS (6 new runs, ≈ 63 min in total).

**Pilot (seed 0):** λ ∈ {0.1, 0.3, 1, 3}. **Selection rule fixed before seeing results,**
applied to paired validation reconstructions only:
* **weak** = the smallest λ that lowers the road-piece error by more than the baseline's
  seed-to-seed spread (0.26);
* **strong** = the largest λ that keeps building and green IoU within 0.02 of the worse baseline
  seed and road-coverage bias within ±0.02.

**Outcome:**
* weak = **λ 0.1**: road-piece error 0.58, against 0.92 for the baseline;
* λ 3 is degenerate under any reading (road-coverage bias +0.52);
* **Deviation (disclosed):** read literally, the bias limit would exclude λ 0.3 (+0.021) and λ 1
  (+0.024), although the baseline itself sits at +0.018, which would make strong = weak. I applied
  the bias criterion *relative to the baseline*, like the IoU criteria, which gives strong =
  **λ 1** (bias +0.006 relative to the baseline). λ 0.3 is reported from the pilot only.

**Seed 1 for λ 0.1 and λ 1 showed that the pilot's apparent improvement at λ 0.1 was seed noise**
(see below).

## 4. Metrics (`src/urbanmorph/metrics/road_raster.py`)

| metric | definition | limitation |
|---|---|---|
| road pieces (Betti-0, ≥ 10 px) | 8-connected road components | tile edges cut real networks, so 1 is **not** the target (real median 2, range 1–9) |
| interior fragments | pieces that touch no tile edge | boundary-aware; real median 0, but Spanish tiles have genuine ones (pedestrian zones, plazas) |
| loops / km² (Betti-1) | 4-connected background regions not touching the edge, ≥ 2 px | depends on 4/8-connectivity; merged parallel carriageways can hide loops |
| intersections / km² | junction-pixel clusters of the Zhang–Suen skeleton | ≈ 1.8× the vector-graph count on real tiles (rank correlation 0.80): comparisons only |
| dead-end ratio | end points / (end points + junctions) | tile-edge cuts count as ends |
| orientation order φ, entropy | PCA direction of skeleton segments → 18 axial bins (same code as the vector metric) | curved segments are averaged; rank correlation with the vector φ is **0.74** on real tiles |
| hard clDice | clDice with exact skeletons on binary masks | pixel-based; ignores which piece connects to which |

Pixel resolution (4 m) and labelled widths (partly default values) limit every raster metric.
Raster numbers are compared with *raster* numbers of real tiles, never with vector values.

## 5. Results

### 5.1 Reconstruction (paired, 19 validation tiles; mean ± half-range over two seeds)

| group | road IoU | hard clDice | road-piece error | interior-fragment error | loops err /km² | intersections err /km² | dead-end err | φ err | road cov. bias | building IoU | green IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: C2 (λ 0) | 0.815 ± 0.003 | 0.925 ± 0.005 | 0.92 ± 0.13 | 0.50 ± 0.08 | 21.7 ± 0.8 | 32.5 ± 1.8 | 0.087 | 0.058 | +0.017 | 0.595 ± 0.012 | 0.69 ± 0.06 |
| B: λ 0.1 | 0.818 ± 0.004 | 0.926 ± 0.002 | 0.84 ± 0.26 | 0.58 ± 0.05 | 23.0 ± 0.3 | 39.2 ± 6.2 | 0.097 | 0.061 | +0.019 | 0.592 ± 0.003 | 0.67 ± 0.01 |
| C: λ 1 | 0.789 ± 0.015 | 0.928 ± 0.012 | 1.11 ± 0.26 | 0.63 ± 0.21 | 20.4 ± 0.7 | 44.8 ± 1.6 | 0.094 | 0.076 | +0.024 | 0.567 ± 0.002 | 0.71 ± 0.00 |
| pilot λ 0.3 | 0.801 | 0.933 | 0.63 | 0.42 | 22.3 | 45.0 | 0.107 | 0.049 | +0.021 | 0.572 | 0.72 |
| pilot λ 3 | **0.190** | 0.290 | 1.74 | 0.58 | 64.4 | 120.7 | 0.145 | 0.204 | **+0.523** | 0.452 | 0.69 |

* **The baseline already reconstructs topology well** (clDice 0.925, about 1 road piece of error
  per tile), so there is little room for improvement.
* At λ 0.1 and λ 1, the topology errors are within or worse than the seed variation. The λ 0.1
  pilot gain (0.58) was not reproduced by seed 1.
* **λ 1 is consistently worse** on intersections (+12/km² error), road IoU (−0.026) and building
  IoU (−0.028).
* **λ 3 floods the tile with road** (failure case, `outputs/topology/samples_by_lambda.png`,
  bottom row). The soft skeleton of a large blob is short and lies entirely inside true-road
  neighbourhoods, which can satisfy clDice while destroying everything else.

### 5.2 Generation (256 prior samples per seed vs real tiles; median [5th–95th percentile])

| set | road pieces | interior fragments | intersections /km² | loops /km² | dead-end ratio | φ | diversity |
|---|---|---|---|---|---|---|---|
| real validation (19) | 2 [1–8] | 0 [0–4] | 111 [42–290] | 34 [3–206] | 0.31 [0.17–0.48] | 0.70 [0.52–1.00] | — |
| real training (565) | 2 [1–9] | 0 [0–5] | 130 [20–316] | 38 [0–168] | 0.31 [0.14–0.60] | 0.84 [0.42–1.00] | — |
| A: C2 (λ 0) | 16 [9–23] | 6.5 [2–13] | 282 [216–353] | 34 [15–61] | 0.34 [0.27–0.43] | 0.47 [0.34–0.61] | 0.875 ± 0.005 |
| B: λ 0.1 | 18 [11–25] | 8 [3–13] | 271 [200–332] | 27 [8–53] | 0.36 [0.28–0.45] | 0.47 [0.33–0.62] | 0.875 ± 0.004 |
| C: λ 1 | 15 [9–22] | 6 [2–12] | 252 [187–315] | 31 [11–57] | 0.36 [0.28–0.44] | 0.47 [0.33–0.61] | **0.858 ± 0.004** |
| λ 3 (degenerate) | 2 [1–4] | 0 [0–1] | 31 [15–50] | 0 [0–4] | 0.30 [0.10–0.56] | 0.86 [0.70–0.99] | 0.727 |

* **Generation did not change in any meaningful way** with λ ≤ 1. All models produce about 8×
  too many road pieces, 6–8 floating fragments (real: 0), about 2× the real intersection density,
  and *lower* orientation order than real streets (φ 0.47 vs 0.70–0.84).
* λ 3 matches the real piece count, but only because roads cover 59% of the tile. **Matching a
  connectivity statistic is not evidence of realism.**
* **Generated networks are less diverse than real ones in every model.** The 5–95% ranges of
  loops, dead-end ratio and φ are much narrower than the real ranges
  (`outputs/topology/generation_distributions.png`). This homogeneity is present without clDice.
  λ 1 lowers pairwise diversity slightly but consistently (0.875 → 0.858, ≈ 4× the seed variation).

### 5.3 Structural bias per country (reconstructions of the 565 training tiles; prediction − truth)

| | Spain (140) | United States (213) | Japan (212) |
|---|---|---|---|
| φ shift: A / B / C | −0.055 / −0.049 / −0.054 | −0.014 / −0.014 / −0.014 | −0.027 / −0.024 / −0.021 |
| intersections shift: A / B / C (/km²) | +38 / +41 / **+44** | +11 / +15 / **+22** | +14 / +17 / **+25** |
| dead-end ratio shift: A / B / C | −0.112 / −0.122 / **−0.129** | −0.044 / −0.055 / **−0.074** | −0.031 / −0.031 / **−0.042** |
| loops shift: A / B / C (/km²) | −16.8 / −13.4 / −13.3 | −5.3 / −4.5 / −4.1 | −9.1 / −9.1 / −8.9 |
| interior-fragment shift: A / B / C | −1.61 / −1.89 / −1.86 | −0.23 / −0.28 / −0.21 | −0.25 / −0.26 / −0.26 |
| spread ratio of intersections (pred/true): A / C | 1.13 / 1.14 | 1.10 / 1.20 | 1.04 / 1.10 |

Between-country share of variance (η², prediction vs truth) on training reconstructions:

| | φ | entropy | intersections | loops | dead-end ratio | interior fragments |
|---|---|---|---|---|---|---|
| truth | 0.22 | 0.27 | 0.31 | 0.26 | 0.24 | 0.10 |
| A | 0.24 | 0.30 | 0.29 | 0.35 | 0.26 | 0.05 |
| B | 0.24 | 0.29 | 0.28 | 0.34 | 0.24 | 0.03 |
| C | 0.24 | 0.29 | 0.27 | 0.34 | 0.23 | 0.03 |

* **No grid bias:** φ shifts are unchanged by clDice in every country.
* **Over-connection bias at strong λ:** extra junctions grow with λ in *all three* countries
  (US +11 → +22, Japan +14 → +25), and dead ends are removed (US −0.044 → −0.074). This is the
  "removes legitimate dead ends" effect the experiment was designed to detect. It is small at
  λ 0.1 and clear at λ 1.
* **Loops:** clDice slightly reduces the baseline's under-prediction of loops in Spain (−17 → −13).
* **Variation is not compressed:** spread ratios are ≥ 1. Between-country differences are kept
  (η² close to the truth), so countries are not pushed towards a global average. Exception:
  *interior fragments*. The baseline already removes Spain's genuine interior fragments (η² 0.10 →
  0.05), and clDice reduces this slightly further (0.03).
* The validation-only per-country tables (n = 6–7) are in `results_tables.md`. They are too small
  to interpret beyond confirming the direction of the training-set results.

### 5.4 Spatial prior diagnosis (`scripts/topo_prior_diagnosis.py`, C2 seed 0)

| | real val | reconstructions | prior N(0, I) | full-covariance Gaussian (diagnostic) |
|---|---|---|---|---|
| road break rate at latent-cell seams / at mid-cell control lines | 0.78 | 0.79 | **1.07** | 1.02 |
| road end points at seams (1 = chance) | 0.65 | 0.73 | **0.94** | 0.95 |
| road pieces (median) | 2 | 2 | 17 | 15 |
| interior fragments | 0 | 0 | 7 | 8 |
| intersections /km² | 111 | 149 | 271 | 210 |
| φ | 0.70 | 0.63 | 0.49 | 0.60 |

* **There is no seam effect.** Roads in prior samples break no more often at the boundaries of the
  64 m latent cells than in the middle of cells (ratio 1.07), and end points are not concentrated
  there. The decoder's convolutions blend neighbouring cells (`prior_seams_*.png`).
* **The linear correlation between neighbouring latent cells of real tiles is weak:** 0.12 at
  distance 1, falling to 0.06 at distance 4 (`latent_neighbour_correlation_*.png`).
* A full-covariance Gaussian fitted to real codes (a *diagnostic*, not a VAE sampler) moves
  intersections (271 → 210) and φ (0.49 → 0.60) towards real values, but **does not reduce
  fragmentation** (15 pieces, 8 interior fragments).
* **Conclusion:** a learned spatial prior **is necessary**, and it must model **non-linear**
  dependence. "Which combinations of latent cells decode to a connected network" is not captured
  by a Gaussian with correlated neighbours.

## 6. Failure cases

1. **λ 3: road flooding** (connected, but 59% road). A topology loss can be satisfied by trivial,
   unrealistic outputs.
2. **λ 0.1: a pilot "improvement" that did not replicate.** One seed is not evidence.
3. **λ 1: over-connection.** It adds junctions and removes dead ends across all countries, at a
   small cost in road and building IoU.
4. **Every model:** homogeneous, over-branched generated networks with floating fragments and low φ.

## 7. Conclusions

* Topology-aware *reconstruction* losses are the wrong lever for this model's main failure. The
  autoencoder already preserves topology, and the problem lies in **sampling the latent grid**.
* Within the tested range, clDice did not reduce diversity in a way that matters at λ ≤ 0.1. At
  λ ≥ 1 it introduces an over-connection bias, and at λ 3 it degenerates. If clDice is kept, keep
  it **weak** (≤ 0.1), and only for reconstruction quality.
* Connectivity statistics alone (piece counts) are misleading: the degenerate model matches them best.

## 8. Recommended next experiment: a learned autoregressive prior

Keep the trained **C2 encoder/decoder frozen** (λ = 0; it reconstructs topology well) and replace
N(0, I) with a **small autoregressive prior over the 8 × 8 × 16 latent grid**:
* **Model:** PixelCNN-style masked convolutions over the 8×8 positions. At each position, it
  predicts the 16-channel latent with a mixture of Gaussians, conditioned on all previously
  generated positions. Roughly 0.2–0.5M parameters, trained on posterior means (or samples) of
  565 × 8 augmented training tiles (4,520 grids). That is small enough for the dataset and a
  laptop (a few minutes per run).
* **Why:** it models exactly the non-linear dependence between cells that the diagnosis showed
  to be missing, with no handcrafted street rules.
* **Alternative:** VQ-VAE codes + a discrete autoregressive prior (the VQ-VAE-2 approach),
  discussed but not proposed first, because it would require retraining the autoencoder.
* **Evaluation:** this protocol unchanged, as the primary generation test: piece/fragment/loop/
  intersection/φ distributions vs real tiles, diversity, and per-country checks on the latent
  codes. Then, if useful, a distribution-level topology objective in the spirit of TopoGAN
  (Wang et al., ECCV 2020).
* Diffusion remains out of scope, and the test split stays unused.

## 9. Reproduce

```bash
python -m pytest -q tests/test_cldice.py
python scripts/topo_experiments.py --lambdas 0.1 0.3 1 3 --seeds 0     # pilot (~45 min)
python scripts/topo_experiments.py --lambdas 0.1 1 --seeds 1           # second seed (~25 min)
python scripts/topo_evaluate.py C2_spatial16_roadx3 C2_spatial16_roadx3_seed1 T_cl0.1_s0 T_cl0.3_s0 T_cl1_s0 T_cl3_s0 T_cl0.1_s1 T_cl1_s1
python scripts/topo_prior_diagnosis.py C2_spatial16_roadx3
python scripts/topo_report.py                                          # tables + figures
```

## References (verified)
* Shit, Paetzold, et al. clDice: A Novel Topology-Preserving Loss Function for Tubular Structure
  Segmentation. CVPR 2021. arXiv:2003.07311. Code: github.com/jocpae/clDice (MIT).
* Wang, Liu, Samaras, Chen. TopoGAN: A Topology-Aware Generative Adversarial Network. ECCV 2020.
* Hu, Li, Samaras, Chen. Topology-Preserving Deep Image Segmentation. NeurIPS 2019.
* Boeing. Urban Spatial Order: Street Network Orientation, Configuration, and Entropy. Applied Network Science 2019.
