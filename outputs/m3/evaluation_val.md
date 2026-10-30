# Milestone 3 evaluation (val, 19 tiles, threshold p ≥ 0.5)

| model | channel | BCE (nats/px) | IoU | coverage abs. error | coverage bias |
|---|---|---|---|---|---|
| VAE | roads | 0.452 | 0.046 | 0.107 | -0.107 |
| VAE | buildings | 0.567 | 0.252 | 0.106 | +0.002 |
| VAE | green | 0.091 | 0.413 | 0.013 | +0.010 |
| AE | roads | 0.450 | 0.079 | 0.085 | -0.069 |
| AE | buildings | 0.572 | 0.176 | 0.133 | -0.049 |
| AE | green | 0.077 | 0.511 | 0.013 | +0.010 |
| trivial_mean_coverage | roads | 0.401 | 0.000 | 0.138 | -0.138 |
| trivial_mean_coverage | buildings | 0.585 | 0.000 | 0.271 | -0.271 |
| trivial_mean_coverage | green | 0.189 | 0.263 | 0.040 | -0.040 |

| set | diversity | nearest-train IoU (median / max) | road components (median) | largest road component share |
|---|---|---|---|---|
| real_eval_tiles | 0.910 | 0.164 / 0.453 | 2 | 0.99 |
| VAE_prior_samples | 0.696 | 0.157 / 0.238 | 0 | 0.51 |
| VAE_fitted_gaussian_samples | 0.843 | 0.195 / 0.542 | 0 | 0.60 |
| AE_gaussian_samples | 0.869 | 0.222 / 0.427 | 2 | 0.67 |

| latent | kNN city acc. (chance) | kNN country acc. (chance) | R² buildings_coverage | R² roads_coverage | R² green_spaces_coverage | R² intersection_density_per_km2 | R² orientation_order_phi | R² building_area_median_m2 |
|---|---|---|---|---|---|---|---|---|
| VAE | 0.16 (0.16) | 0.42 (0.37) | -0.18 | -0.05 | 0.86 | 0.08 | -0.75 | -0.29 |
| AE | 0.16 (0.16) | 0.53 (0.37) | -0.37 | -0.20 | 0.40 | -0.07 | -0.78 | -0.56 |
