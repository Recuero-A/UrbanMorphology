# Milestone 3: First Generative Model (convolutional VAE baseline)

**Status (2026-10-30):** the VAE trains, reconstructs and samples, and it is compared with an AE
and a trivial baseline. The run is reproducible: a repeat with the same seed gave bit-identical
results. **The model does not yet produce realistic neighbourhoods.** It learns coarse building
and green mass, but not the street network. This is a baseline result, not a claim of novelty or
realism. All numbers below are on the **validation** split (19 tiles); the test split has not been used.

---

## 0. Dataset audit before training (`scripts/m3_dataset_audit.py` → `outputs/m3/dataset_audit.md`)

The audit was re-derived from the saved arrays instead of relying on earlier tests:

* 135 tiles, all `(3, 128, 128)` uint8 with values in {0, 1}.
* Train/val/test tiles are ≥ 1,024 m apart edge to edge (buffer 1,000 m); any two tiles are ≥ 1,024 m apart.
* No exact duplicates. The highest pairwise building IoU is 0.415 (Barcelona vs Valencia block
  patterns), so there are no near-duplicates.
* Balance: 10–11 train, 2–3 val and 2–3 test tiles per city. **But** the density mix differs:
  val is 10 high / 6 low / 3 medium, test is 4 / 9 / 6. Validation-based model selection is therefore noisy.
* Positive-pixel fractions (train): roads 0.134, buildings 0.250, green 0.095. Validation green is only 0.040.
* Quality flags: 14 / 2 / 2 tiles flagged `buildings_incomplete_suspected` in train / val / test.
* Road widths: 77% of Japanese road segments use class-default widths (Spain 42%, US 66%). The
  model learns these approximate widths as if they were real.

**The dataset was not changed** for Milestone 3.

## 1. Architecture (`src/urbanmorph/models/vae.py`)

```
input (B, 3, 128, 128) binary
Encoder: 4 x [Conv 4x4 stride 2 -> GN -> SiLU -> Conv 3x3 -> GN -> SiLU]
         channels 3 -> 32 -> 64 -> 128 -> 256, spatial 128 -> 64 -> 32 -> 16 -> 8
         flatten (256*8*8 = 16,384) -> Linear -> mu (32), Linear -> log sigma^2 (32)
Latent:  z = mu + sigma * eps, eps ~ N(0, I)                    (reparameterisation trick)
Decoder: Linear 32 -> 256*8*8 -> 4 x [Upsample x2 (nearest) -> Conv 3x3 -> GN -> SiLU -> Conv 3x3 -> GN -> SiLU]
         channels 256 -> 128 -> 64 -> 32 -> 32, spatial 8 -> 128
         Conv 1x1 -> 3 logits;  probabilities = sigmoid(logits)
```

| | parameters |
|---|---|
| encoder convolutions | 1,475,904 |
| encoder heads (mu, log sigma², z = 32) | 1,048,640 |
| decoder linear | 540,672 |
| decoder convolutions + output | 600,675 |
| **VAE total (z = 32)** | **3,665,891** |
| VAE total (z = 16) | 2,879,427 |
| AE baseline (z = 32, no log sigma² head) | 3,141,571 |

Design choices: nearest-neighbour upsampling + convolution (not transposed convolutions) avoids
checkerboard artefacts. GroupNorm works with small batches. The loss uses logits
(`BCEWithLogits`), which is numerically safer than taking logs of sigmoid outputs. One training
step on a batch of 16 takes ≈ 33 ms on Apple Silicon MPS; there is an automatic CPU fallback.

## 2. Objective (`src/urbanmorph/models/losses.py`)

The VAE maximises the evidence lower bound (ELBO). We minimise its negative:

  **L(x) = E_q(z|x)[ −log p(x|z) ] + β · KL( q(z|x) ‖ N(0, I) )**

* **Reconstruction:** each pixel and channel is a Bernoulli variable with probability
  `sigmoid(logit)`. The term is the binary cross-entropy *summed* over the 3 × 128 × 128 pixels
  (so it is on the same scale as the KL, which is summed over latent dimensions) and averaged over
  the batch. **Channel imbalance:** positive pixels are weighted by `sqrt(neg/pos)` per channel,
  computed on the training set: roads 2.54, buildings 1.73, green 3.09. The square root is a
  compromise. With plain BCE, thin roads are cheapest to ignore; with the full `neg/pos`
  weighting, the model over-predicts positives.
* **KL term:** for a diagonal Gaussian it has a closed form,
  `KL = −½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`.
  *Why it is needed:* without it, the encoder can put each tile anywhere in latent space with
  near-zero variance. That makes a deterministic autoencoder, whose codes have no known
  distribution to sample new layouts from. The KL pulls every posterior towards N(0, I), so that
  sampling z ~ N(0, I) lands where the decoder has learned something.
* **β** sets the balance between the two terms. A large β gives a smoother, better-matched latent
  space but blurrier reconstructions: the encoder is penalised for storing information, so detail
  is discarded first (here, thin streets). In the extreme the posterior collapses to the prior and
  the decoder ignores z. A small β gives sharper reconstructions, but the codes drift away from
  N(0, I), and prior samples land in regions the decoder has never seen.
* **KL warm-up:** β rises linearly from 0 over the first 20 epochs, so the decoder learns to use z
  before the KL term starts compressing it.

## 3. Training (`src/urbanmorph/models/train.py`, `configs/m3_vae.toml`)

| setting | value |
|---|---|
| data | M2 splits: 97 train / 19 val tiles (test untouched) |
| augmentation | training only: random one of the 8 symmetries of the square (4 rotations × mirror). Valid layouts, but absolute north-up orientation is lost. |
| optimiser | Adam, lr 1e-3, batch 16 |
| schedule | up to 300 epochs; early stopping on validation loss (patience 40, counted after warm-up); best checkpoint kept |
| seeds | Python / NumPy / PyTorch seed 0; seeded DataLoader shuffling and augmentation |
| checkpoints | `checkpoints/m3/<run>/best.pt` (git-ignored); histories in `outputs/m3/runs/<run>/` |

**Procedure followed:**
1. Smoke test (2 epochs, VAE and AE).
2. Short run (60 epochs, 420 steps): IoU for roads 0.00, buildings 0.16. The model was close to
   the trivial baseline (`outputs/m3/evaluation_val_short60.md`).
3. **Overfitting diagnostic:** the AE memorises 8 training tiles to IoU 1.00 on all channels in
   1,000 steps. The code, loss and capacity can therefore represent 1–5 px roads; the difficulty
   is generalising.
4. Approved sweep (`scripts/m3_sweep.py`): VAE z ∈ {16, 32} × β ∈ {0.1, 0.5, 1.0}, plus AE z ∈ {16, 32}.
5. Selection: the VAE with the lowest **validation** negative ELBO at β = 1 → `vae_z32_b0.5`.

### Sweep (validation, weighted reconstruction in nats per tile)

| name | best_epoch | epochs_run | train_recon | val_recon | val_kl | val −ELBO (β=1) | IoU roads | IoU buildings | IoU green |
|---|---|---|---|---|---|---|---|---|---|
| vae_z16_b0.1 | 262 | 300 | 26293 | 26080 | 343 | 26423 | 0.036 | 0.174 | 0.412 |
| vae_z16_b0.5 | 221 | 261 | 26736 | 26319 | 127 | 26446 | 0.000 | 0.134 | 0.428 |
| vae_z16_b1.0 | 256 | 296 | 26337 | 26043 | 75 | 26118 | 0.006 | 0.183 | 0.415 |
| ae_z16 | 298 | 300 | 26408 | 25957 | 0 | 25957 | 0.000 | 0.173 | 0.433 |
| vae_z32_b0.1 | 192 | 232 | 25559 | 25709 | 490 | 26198 | 0.018 | 0.238 | 0.378 |
| **vae_z32_b0.5** | 272 | 300 | 25617 | 25780 | 191 | **25971** | 0.046 | **0.252** | 0.413 |
| vae_z32_b1.0 | 199 | 239 | 26224 | 26056 | 97 | 26153 | 0.025 | 0.203 | 0.485 |
| ae_z32 | 196 | 236 | 25238 | 25229 | 0 | 25229 | 0.079 | 0.176 | 0.511 |

**Train ≈ validation in every run, so the models underfit; they do not overfit.** Training curves
were still slowly decreasing at 300 epochs, with occasional instability spikes
(`outputs/m3/loss_curves.png`).

## 4. Evaluation (`src/urbanmorph/models/evaluate.py`, `scripts/m3_evaluate.py`)

* **Per-channel BCE:** unweighted, in nats per pixel, from the posterior-mean reconstruction.
* **IoU** at threshold **p ≥ 0.5** (a tile where both prediction and truth are empty counts as 1).
* **Coverage error:** |mean(mask) − true coverage| per channel, plus the signed bias.
* **Street connectivity:** 8-connected components of the road mask. A real 512 m window is mostly
  one connected network (largest component ≈ 99% of road pixels).
* **Diversity:** mean pairwise layout distance (1 − mean channel IoU) among samples.
* **Memorisation:** each sample's highest IoU with any training tile under any of the 8 symmetries.
* **Latent probes** (train codes → validation codes): 5-NN city and country accuracy, and ridge-regression R² for morphology metrics.

## 5. Results (validation, `outputs/m3/evaluation_val.md`)

### Reconstruction

| model | channel | BCE | IoU | coverage abs. error | bias |
|---|---|---|---|---|---|
| VAE (z32, β 0.5) | roads | 0.452 | 0.046 | 0.107 | −0.107 |
| | buildings | 0.567 | **0.252** | 0.106 | +0.002 |
| | green | 0.091 | 0.413 | 0.013 | +0.010 |
| AE (z32) | roads | 0.450 | **0.079** | 0.085 | −0.069 |
| | buildings | 0.572 | 0.176 | 0.133 | −0.049 |
| | green | **0.077** | **0.511** | 0.013 | +0.010 |
| trivial (mean coverage) | roads | **0.401** | 0.000 | 0.138 | −0.138 |
| | buildings | 0.585 | 0.000 | 0.271 | −0.271 |
| | green | 0.189 | 0.263* | 0.040 | −0.040 |

\* Trivial green IoU comes from tiles where the truth is empty and the constant prediction (p < 0.5)
is empty too, which counts as 1.

* Both models beat the trivial baseline on **buildings** (IoU 0.18–0.25 versus 0; coverage error
  roughly halved) and on **green** (BCE 0.08–0.09 versus 0.19).
* **Roads are not learned.** Road IoU is ≤ 0.08, and the unweighted road BCE is *worse* than the
  constant baseline (0.45 versus 0.40). The square-root weighting pushes road probabilities up
  without placing them correctly, which costs calibration. Reconstructions show building and green
  *blobs* in roughly the right places, with no street network (`outputs/m3/reconstructions_val.png`).
* Road connectivity: real validation tiles have a median of 2 road components (largest = 99% of
  road pixels). VAE reconstructions have 1 component (largest = 72%) and AE reconstructions 2
  (80%). VAE prior samples have a median of **0** road components above 10 px. **Caveat:** a single
  road-coloured blob also counts as "one connected component", so component counts cannot tell a
  street network from a blob. A skeleton/graph-based metric is needed (see next steps).
* Latent interpolation (`outputs/m3/interpolation_vae.png`) changes smoothly between two tiles
  (the latent space is continuous), but every intermediate frame is made of blobs.

### Generation (64 samples each)

| set | diversity | nearest-train IoU (median / max) | road components (median) | coverage roads / buildings / green |
|---|---|---|---|---|
| real validation tiles | 0.910 | 0.164 / 0.453 | 2 | 0.138 / 0.271 / 0.040 |
| VAE, z ~ N(0, I) | 0.696 | 0.157 / 0.238 | 0 | **0.004** / 0.136 / **0.226** |
| VAE, z ~ Gaussian fitted to training codes | 0.843 | 0.195 / 0.542 | 0 | 0.032 / 0.300 / 0.123 |
| AE, z ~ Gaussian fitted to training codes | 0.869 | 0.222 / 0.427 | 2 | 0.061 / 0.238 / 0.135 |

* **There is no memorisation:** samples are no closer to training tiles than real validation tiles are.
* **Prior samples are not plausible.** They have almost no roads and far too much green
  (`outputs/m3/samples_vae.png`). The cause is a **prior / aggregate-posterior mismatch**: across
  training tiles, the posterior means have a per-dimension std of ≈ 3.3, against 1 for the prior,
  and each posterior is narrow (σ ≈ 0.12). With β ≤ 1 against a reconstruction term summed over
  49,152 pixels, the KL is too weak to make N(0, I) cover the region where real tiles are encoded.
  Sampling the same decoder from a fitted Gaussian (a control that needs no retraining) fixes the
  building coverage and diversity, but not the roads. Both problems are real; the missing street
  structure is the larger one.

### Latent space

| latent | kNN city (chance) | kNN country (chance) | R² bldg cov | R² road cov | R² green cov | R² intersections | R² φ | R² footprint |
|---|---|---|---|---|---|---|---|---|
| VAE | 0.16 (0.16) | 0.42 (0.37) | −0.18 | −0.05 | **0.86** | 0.08 | −0.75 | −0.29 |
| AE | 0.16 (0.16) | 0.53 (0.37) | −0.37 | −0.20 | 0.40 | −0.07 | −0.78 | −0.56 |

* **Cities are not distinguishable** in latent space (k-NN accuracy equals chance). Country is
  barely above chance, which with 19 validation tiles is within noise (`outputs/m3/latent_pca_vae.png`).
* The latent space clearly encodes **green coverage** (R² 0.86). It does not encode
  street-network properties (intersection density, orientation order), which is consistent with
  the missing roads. Orientation could not be learned anyway: rotation augmentation removes absolute orientation.
* All 32 latent dimensions are active (no posterior collapse).

### Answers to the research questions (baseline, 97 training tiles)

1. *Can a compact network reconstruct real layouts?* **Only coarsely.** It recovers building and
   green mass and their coverage; it does not recover streets.
2. *Does the latent space capture meaningful characteristics?* **Partly.** Green coverage, yes;
   street structure and city identity, no.
3. *Do samples look plausible?* **No.** Prior samples lack roads and over-produce green.
   Visually smooth blobs are not evidence of a successful generative model.
4. *Do cities occupy distinguishable regions?* **No evidence** at this data size.

## 6. Limitations

* **97 training tiles.** That is far too few to learn street-network structure that transfers to
  unseen neighbourhoods; the models underfit even the training set.
* **A single global 32-number latent** has to encode the exact geometry of 1–5 px streets across
  hundreds of distinct augmented layouts. Fine detail is what a global bottleneck discards first.
* Training was **stopped at 300 epochs while still slowly improving**, and there were instability spikes.
* **Pixel-wise BCE** treats a road shifted by 2 px as entirely wrong, so the model "hedges" with a blurred average.
* Validation has only 19 tiles, with a density mix that differs from test, so model selection is noisy.
* Road channel widths are partly artefacts (default widths), especially in Japan.
* Rotation augmentation removes north-up orientation information, by design.
* The sweep is small (one seed per configuration).

## 7. Recommended next steps (Milestone 3b / 4)

1. **More training data** (approved for later): extract additional training tiles from the same
   cached snapshot, respecting the 1 km buffer from validation and test, and measure the effect of data size.
2. **A spatial latent** (e.g. a 16 × 16 × c feature map instead of one global vector) or a
   hierarchical VAE, so that local street geometry does not have to squeeze through 32 numbers.
3. **Better priors:** a higher β, or a two-stage approach (fit a density to the aggregate posterior),
   or a learned prior.
4. **Structure-aware losses or metrics:** a topology/connectivity loss for roads, and graph metrics
   computed on skeletonised road masks.
5. Only then a diffusion model, compared against this baseline with the same metrics.
6. Use the test split only once, for the final chosen model: `python scripts/m3_evaluate.py --split test --final`.

## 8. Reproduce

```bash
pip install -e ".[dev,ml]"
python scripts/m3_dataset_audit.py
python scripts/m3_train.py --smoke                     # 2-epoch check
python scripts/m3_sweep.py                             # 8 runs, ~8 min on Apple Silicon
python scripts/m3_evaluate.py --vae vae_z32_b0.5 --ae ae_z32
python -m pytest -q
```
