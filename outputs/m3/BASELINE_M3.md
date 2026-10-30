# Frozen Milestone 3 baseline (validation split, 19 tiles) — do not overwrite

Model: `vae_z32_b0.5` (ConvVAE, global latent z=32, beta=0.5, sqrt pos_weight, 97 train tiles, 8-fold augmentation, best epoch 272); AE baseline `ae_z32`. Threshold p >= 0.5. Source: `BASELINE_M3_val.json`.

| model | channel | BCE | IoU | coverage abs err | bias |
|---|---|---|---|---|---|
| VAE | roads | 0.4516 | 0.0457 | 0.1072 | -0.1072 |
| VAE | buildings | 0.5667 | 0.2521 | 0.1064 | +0.0015 |
| VAE | green | 0.0910 | 0.4134 | 0.0132 | +0.0101 |
| AE | roads | 0.4496 | 0.0793 | 0.0853 | -0.0688 |
| AE | buildings | 0.5715 | 0.1756 | 0.1327 | -0.0487 |
| AE | green | 0.0770 | 0.5113 | 0.0126 | +0.0097 |
| trivial_mean_coverage | roads | 0.4011 | 0.0000 | 0.1379 | -0.1379 |
| trivial_mean_coverage | buildings | 0.5852 | 0.0000 | 0.2708 | -0.2708 |
| trivial_mean_coverage | green | 0.1894 | 0.2632 | 0.0398 | -0.0398 |

| samples | diversity | nearest-train IoU median | road comps median | coverage roads/bldg/green |
|---|---|---|---|---|
| real_eval_tiles | 0.910 | 0.164 | 2 | 0.138 / 0.271 / 0.040 |
| VAE_prior_samples | 0.696 | 0.157 | 0 | 0.004 / 0.136 / 0.226 |
| VAE_fitted_gaussian_samples | 0.843 | 0.195 | 0 | 0.032 / 0.300 / 0.123 |
| AE_gaussian_samples | 0.869 | 0.222 | 2 | 0.061 / 0.238 / 0.135 |

VAE latent: kNN city 0.16 (chance 0.16), country 0.42 (chance 0.37); R² green 0.86; aggregate posterior std/dim 3.28.
