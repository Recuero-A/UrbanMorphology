# Milestone 3 evaluation (val, 19 tiles, threshold p ≥ 0.5)

| model | channel | BCE (nats/px) | IoU | coverage abs. error | coverage bias |
|---|---|---|---|---|---|
| VAE | roads | 0.457 | 0.000 | 0.138 | -0.138 |
| VAE | buildings | 0.623 | 0.163 | 0.158 | -0.099 |
| VAE | green | 0.126 | 0.384 | 0.024 | +0.012 |
| AE | roads | 0.470 | 0.000 | 0.138 | -0.138 |
| AE | buildings | 0.593 | 0.030 | 0.254 | -0.254 |
| AE | green | 0.137 | 0.406 | 0.034 | +0.031 |
| trivial_mean_coverage | roads | 0.401 | 0.000 | 0.138 | -0.138 |
| trivial_mean_coverage | buildings | 0.585 | 0.000 | 0.271 | -0.271 |
| trivial_mean_coverage | green | 0.189 | 0.263 | 0.040 | -0.040 |

| set | diversity | nearest-train IoU (median / max) | road components (median) | largest road component share |
|---|---|---|---|---|
| real_eval_tiles | 0.910 | 0.164 / 0.453 | 2 | 0.99 |
| VAE_prior_samples | 0.587 | 0.189 / 0.456 | 0 | nan |
| AE_gaussian_samples | 0.590 | 0.174 / 0.417 | 0 | nan |

| latent | kNN city acc. (chance) | kNN country acc. (chance) | R² buildings_coverage | R² roads_coverage | R² green_spaces_coverage | R² intersection_density_per_km2 | R² orientation_order_phi | R² building_area_median_m2 |
|---|---|---|---|---|---|---|---|---|
| VAE | 0.05 (0.16) | 0.37 (0.37) | 0.04 | -0.00 | 0.79 | 0.00 | -0.49 | -0.13 |
| AE | 0.05 (0.16) | 0.42 (0.37) | -0.45 | -0.02 | 0.35 | -0.09 | -0.42 | -0.35 |
