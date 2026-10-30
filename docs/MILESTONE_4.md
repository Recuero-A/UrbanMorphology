# Milestone 4: Learning coherent urban structure with a learned latent prior

**Date:** 2026-10-30 · **Branch:** `milestone-4` (the topology experiment is committed and pushed:
`road-topology`, `407b677`). **Data:** 565 training / 19 validation tiles. The test split was not
used. The C2 autoencoder is **frozen**, and no reconstruction loss was changed.

## Answer in brief

* A learned autoregressive prior over the 8 × 8 × 16 latent grid **clearly improves** generated
  layouts over N(0, I) and over a fitted Gaussian:
  * road pieces 17 → 9–10 (real 2);
  * floating fragments 7 → 4 (real 0);
  * intersections 275 → 175–187 /km², about the decoder's own reconstruction level of 195;
  * orientation order 0.48 → 0.63 (real 0.70);
  * coverage closer to real, with no loss of diversity and no memorisation.
* **It does not meet the pre-registered "substantial improvement" criterion.** Floating fragments
  did not move halfway towards the real value with either seed, and road pieces did so with only
  one seed. Generated networks remain fragmented.
* **Decision (step 7):** *do not scale up yet.* A **graph-based road generator inspired by Neural
  Turtle Graphics** is proposed below as a design, not implemented, for review.

---

## 1. Motivation

The road-topology experiment showed that:
* C2 reconstructs road topology well (centreline clDice ≈ 0.93), but prior samples are fragmented;
* the cause is not seams between latent cells, and the linear correlation between neighbouring
  latent cells is weak (0.12);
* a Gaussian with full covariance helped orientation and intersections but not fragmentation.

**Hypothesis:** the missing ingredient is the *non-linear joint distribution* of the latent cells.
Modelling p(z) autoregressively should produce latent grids that decode to more coherent networks.

## 2. Related methods: how they model spatial dependence

| method | representation | dependence model | relation to this experiment |
|---|---|---|---|
| **VQ-VAE** (van den Oord, Vinyals, Kavukcuoglu, NeurIPS 2017) | grid of **discrete** codebook indices | uniform prior while training the autoencoder, then a **PixelCNN** fitted afterwards: p(z) = Π p(z_i \| z_<i), each a **categorical** over K codes | same two-stage idea, but our latents are **continuous**, so each conditional must be a density, not a softmax |
| **Neural Turtle Graphics** (Chu et al., ICCV 2019) | road **graph** (control points + segments) | autoregressive growth: each new node/edge is attached to the existing graph, conditioned on it | connectivity by construction; basis of the proposal in §8 |
| **GraphRNN** (You, Ying, Ren, Hamilton, Leskovec, ICML 2018) | generic graph as a node sequence (BFS ordering) | graph-level RNN + edge-level RNN: p(G) = Π p(edges of node i \| earlier nodes) | explicit topology but no geometry; evaluated with MMD on graph statistics, a useful evaluation idea |

**Continuous vs discrete autoregressive priors.** Applying a categorical PixelCNN loss to our
continuous 16-dimensional latents would not define a valid probability density. We keep the
autoregressive factorisation, but each conditional p(z_i | z_<i) is a **mixture of diagonal
Gaussians** over the 16-dimensional vector, and training minimises the exact continuous negative
log-likelihood. (Mixture likelihoods in PixelCNN++, Salimans et al. ICLR 2017, follow a similar
idea for discretised pixel values.)

## 3. Mathematical formulation

* Frozen C2: encoder q(z|x) = N(μ(x), diag σ²(x)), z ∈ ℝ^(16×8×8); decoder p(x|z).
* Prior over positions i = 1…64 in raster order:
  **p_θ(z) = Π_i Σ_k π_k(z_<i) Π_{d=1}^{16} N(z_{i,d}; m_{k,d}(z_<i), s_{k,d}(z_<i)²)**, with K = 5.
* Training data: the aggregate posterior. For each epoch and tile, z = μ(x) + σ(x)·ε for the 565
  training tiles × 8 symmetries (4,520 grids, a fresh ε each epoch).
* Objective: minimise −E[log p_θ(z)] (exact NLL; codes standardised per channel, with the change
  of variables included in the reported log-likelihood).
* Generation: ancestral sampling z_1, …, z_64 ~ p_θ, then x = decoder(z), thresholded at p ≥ 0.5.
  Temperature 1 only; no tuning to make samples look nicer.

## 4. Architecture (`src/urbanmorph/models/prior.py`)

* **Gated PixelCNN** (vertical and horizontal stacks): 6 gated layers, 32 channels, vertical
  kernel 2×5, horizontal kernel 1×2 (excluding the current position in the first layer), and a
  head of two 1×1 convolutions producing K·(1 + 2·16) = 165 values per position. **173,733 parameters.**
* **Mask verification (`tests/test_prior.py`):** a float64 Jacobian test shows that the mixture
  parameters at position p depend on **no** position q ≥ p (no future access) and on **every**
  position q < p (no blind spot).
  * A first version of this test reported a false blind spot: artificially large weights saturated
    tanh/sigmoid and made gradients underflow to exactly 0. The fixed test uses the default
    initialisation in float64.
* Other tests check the log-density against a hand-computed Gaussian, the change of variables for
  standardisation, and that a toy dataset with long-range copying is learned and reproduced in samples.
* **Training:** Adam (lr 1e-3), batch 64, early stopping on validation NLL (19 tiles × 8 symmetries,
  fixed noise; patience 20). Best epochs 103 and 109; ≈ 67 s per seed on MPS.

## 5. Experimental controls

All priors feed the **same frozen decoder** (C2, seed 0) and are scored with the **same metric
code** as the road-topology experiment. 256 samples each, fixed random seeds.

| | prior | training |
|---|---|---|
| A | N(0, I) | none (the VAE's own prior) |
| B | Gaussian with full covariance over all 1,024 elements, 10% shrinkage | fitted to the same 4,520 z ~ q(z\|x) |
| C | Gated PixelCNN, seeds 0 and 1 | §4 |
| ref | decoded posterior means of training tiles | an upper bound for this decoder; these **are** training tiles, not generation |

**Decision criteria fixed before the results:** "substantial improvement" means C moves the
medians of road pieces, floating fragments **and** intersections at least halfway from A towards
real validation (both seeds), with diversity ≥ real, nearest-training-tile IoU ≤ real + 0.10, and
every country's share within 15 percentage points of its training share.

## 6. Results (`outputs/m4/results.json`; median [5th–95th percentile]; W1 = Wasserstein distance to real validation in units of its IQR)

| set | road pieces | floating fragments | intersections /km² | loops /km² | dead-end ratio | φ | orientation entropy | road coverage |
|---|---|---|---|---|---|---|---|---|
| **real validation** | 2 [1–8] | 0 [0–4] | 111 [42–290] | 34 [3–206] | 0.31 [0.17–0.48] | 0.70 [0.52–1.00] | 1.89 [0.55–2.21] | 0.13 [0.04–0.25] |
| A: N(0, I) | 17 [11–23], W1 9.6 | 7 [3–13], W1 6.9 | 275 [209–340], W1 0.85 | 31 [15–61] | 0.36 [0.28–0.44] | 0.48 [0.35–0.62], W1 0.76 | 2.27 [2.05–2.47] | 0.15 [0.11–0.19] |
| B: fitted Gaussian | 15 [7–23], W1 8.5 | 7 [2–15], W1 7.1 | 196 [57–368], W1 0.38 | 23 [0–88] | 0.41 [0.27–0.63] | 0.59 [0.45–0.82], W1 0.37 | 2.10 [1.61–2.33] | 0.12 [0.04–0.25] |
| **C: PixelCNN s0** | **10** [4–16], W1 4.7 | **4** [1–8], W1 3.6 | **175** [19–343], W1 0.22 | 23 [0–80] | 0.36 [0.24–0.65] | **0.63** [0.37–0.97], W1 0.25 | 2.03 [0.90–2.44] | 0.12 [0.02–0.23] |
| **C: PixelCNN s1** | **9** [4–16], W1 4.7 | **4** [1–8], W1 3.5 | **187** [46–347], W1 0.33 | 27 [0–88] | 0.35 [0.22–0.56] | **0.63** [0.37–0.89], W1 0.30 | 2.03 [1.43–2.44] | 0.13 [0.04–0.24] |
| ref: decoded train codes | 2 [1–6] | 0 [0–2] | 195 [26–358] | 53 [0–165] | 0.22 [0.11–0.44] | 0.69 [0.36–0.98] | 1.91 [0.91–2.45] | 0.17 [0.04–0.28] |

| set | coverage roads / buildings / green | diversity | nearest-train IoU (median) | latent NN distance | prior NLL on val codes (nats/dim) |
|---|---|---|---|---|---|
| real validation | 0.138 / 0.271 / 0.040 | 0.910 | 0.183 | 1.10 | — |
| A | 0.150 / 0.360 / 0.115 | 0.880 | 0.176 | 1.17 | 1.445 |
| B | 0.126 / 0.316 / 0.101 | 0.908 | 0.186 | 1.13 | 1.387 |
| C s0 | 0.118 / 0.258 / 0.110 | 0.923 | 0.188 | 1.15 | 1.177 |
| C s1 | 0.133 / 0.300 / 0.096 | 0.911 | 0.190 | 1.18 | 1.178 |

**Decision criteria:**

| | pieces halfway | fragments halfway | intersections halfway | diversity ≥ real | memorisation OK | country shares | verdict |
|---|---|---|---|---|---|---|---|
| C s0 | ✗ (10 vs threshold 9.5) | ✗ (4 vs 3.5) | ✓ | ✓ | ✓ | *uninformative* (classifier at chance) | **not met** |
| C s1 | ✓ (9) | ✗ (4 vs 3.5) | ✓ | ✓ | ✓ | *uninformative* | **not met** |

## 7. Diagnosis (step 6)

1. **Road connectivity: improved, not solved.** Fragmentation is roughly halved (pieces −45%,
   floating fragments −43%), consistently across both seeds. Samples (`outputs/m4/samples_by_prior.png`)
   show longer continuous streets and block-like arrangements, but still 4 floating fragments per
   tile where real tiles have 0. The reference row proves the decoder can produce connected
   networks (2 pieces, 0 fragments), so **the gap is in the prior**, not the decoder.
2. **Diversity is preserved.** Pairwise diversity is 0.91–0.92, at or above real (0.91) and above
   N(0, I) (0.88). The 5–95% ranges of φ, entropy, road coverage and intersections now *widen*
   towards the real ranges, instead of the narrow bands of N(0, I). This is the opposite of mode collapse.
3. **Other characteristics improve:** orientation order and entropy move towards real values;
   building coverage 0.26–0.30 (real 0.27; A over-produced buildings at 0.36). Green coverage is
   still too high (0.10–0.11 vs 0.04 in validation; training tiles have ≈ 0.10, so part of this is
   a validation-sample effect). Loops are slightly *under*-produced (23–27 vs 34), and the
   dead-end ratio stays slightly high.
4. **No memorisation detected.** The nearest-training-tile IoU of samples (0.188–0.190) matches
   that of real validation tiles (0.183), and latent nearest-neighbour distances (1.15–1.18) are
   not smaller than those of validation codes (1.10). A 174K-parameter prior trained on 4,520 grids
   does not copy tiles.
5. **Structural bias per country: could not be measured for generation.** The country classifier
   on latent codes reaches **0.37** validation accuracy, equal to the majority-class share, so it
   carries no country information. It assigns 75–85% of samples to "Japan" for *every* prior,
   including N(0, I), which is an artefact. The country criterion is therefore **uninformative**,
   not failed. This matches Milestone 3 (latent codes are not country-separable). A proper test
   needs a **country-conditioned** prior.
6. **NLL vs generation quality.** The prior with the lowest NLL (C) also gave the best samples
   here, but the two C seeds have identical NLL (1.177 vs 1.178) and differ in which metric is
   better (pieces vs intersections). NLL ranks the families, not the details. It was reported, not
   used for selection.

## 8. Decision and recommended next step

Under the pre-registered rule, the autoregressive prior **failed** the "substantial improvement"
test, so the dataset and model are **not** scaled up now. A proportionate reading: it is the
first change that has moved *generation* (not only reconstruction) towards real street networks
without reducing diversity. Scaling it remains a plausible secondary option (a larger prior,
temperature analysis, more tiles), but it does not change the representation. Roads remain
pixels, and connectivity remains a statistical outcome.

### Proposal (for review, not implemented): an NTG-inspired graph road generator

* **Idea:** generate roads as a **graph** first, then rasterise them and condition the existing
  decoder (or a conditional decoder) for buildings and green on it. Connectivity then holds by
  construction, without imposing grids or angles.
* **Data:** we already have the vector street graphs of all tiles (OSMnx, clipped, the same data
  the raster came from), so no new downloads are needed.
* **Model (small):** following NTG, an encoder–decoder RNN that grows the graph node by node.
  Each step predicts:
  1. which existing node to extend (a pointer / attention over the frontier);
  2. a displacement (Δx, Δy) with a mixture density (no preferred angles);
  3. a stop/branch decision.

  Optionally, add GraphRNN-style edge prediction to close loops (merging with nearby nodes).
* **Evaluation:**
  * the same topology protocol, plus MMD on graph statistics (degree, intersection density,
    segment lengths, orientation), as in GraphRNN;
  * memorisation checks;
  * per-country distributions, with real per-country graphs available directly.
* **Risks:**
  * 565 training graphs is little data;
  * boundary truncation (streets that leave the tile) must be modelled explicitly;
  * graph ordering (BFS vs spatial) affects learning.
* **A cheaper intermediate option** to discuss first: a **country-conditioned** PixelCNN prior,
  which would make per-country evaluation of generation possible at negligible cost.

## 9. Limitations

* One frozen decoder (C2 seed 0); two prior seeds; 19 validation tiles as the primary real reference.
* The decoder itself over-produces intersections (decoded training codes: 195/km² vs 130 real
  raster), so "real" values are not fully attainable through this decoder.
* Raster topology metrics inherit the limitations documented in `ROAD_TOPOLOGY_EXPERIMENT.md`
  (4 m pixels, skeleton artefacts, default road widths).
* Augmentation removes absolute orientation, so orientation *order* (φ) is comparable but compass direction is not.
* The country analysis of generation was not possible (classifier at chance).

## 10. Reproduce

```bash
python -m pytest -q tests/test_prior.py
python scripts/m4_train_prior.py --seed 0      # ~70 s on MPS
python scripts/m4_train_prior.py --seed 1
python scripts/m4_evaluate.py                  # priors A, B, C + reference -> outputs/m4/
```

## References (verified)
* van den Oord, Vinyals, Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS 2017 (arXiv:1711.00937).
* Chu et al. Neural Turtle Graphics for Modeling City Road Layouts. ICCV 2019.
* You, Ying, Ren, Hamilton, Leskovec. GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models. ICML 2018, PMLR 80:5708–5717.
* van den Oord et al. Conditional Image Generation with PixelCNN Decoders (Gated PixelCNN). NeurIPS 2016.
* Salimans, Karpathy, Chen, Kingma. PixelCNN++. ICLR 2017.
