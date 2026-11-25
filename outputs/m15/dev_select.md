# M15 development sweep (syn_dev, TN_s0, M*, seeds 1-3)

Style off: street_km 2.15, dead ends / km 1.45, loops / km2 17.4

| Style | Setting | field alignment | gain | street_km | orth R4 | radial+tang. | dead ends / km | loops / km2 |
|---|---|---|---|---|---|---|---|---|
| grid | kappa 0.5 | 0.728 | +0.058 | 1.98 | 0.75 | 0.43 | 1.18 | 10.0 |
| grid | kappa 1 | 0.692 | +0.022 | 2.22 | 0.60 | 0.46 | 1.36 | 17.6 |
| grid | kappa 2 | 0.824 | +0.154 | 2.06 | 0.71 | 0.39 | 1.41 | 16.7 |
| grid | kappa 4 **chosen** | 0.856 | +0.186 | 2.04 | 0.80 | 0.42 | 2.76 | 15.7 |
| grid | kappa 8 | 0.873 | +0.203 | 1.91 | 0.85 | 0.46 | 1.46 | 11.0 |
| grid | off (alignment vs this field) | 0.670 | — | 2.15 | | | | |
| organic | kappa 0.5 | 0.506 | +0.007 | 1.32 | 0.64 | 0.49 | 2.45 | 4.3 |
| organic | kappa 1 | 0.506 | +0.006 | 1.75 | 0.65 | 0.46 | 1.68 | 8.6 |
| organic | kappa 2 | 0.558 | +0.059 | 2.37 | 0.55 | 0.50 | 1.56 | 29.8 |
| organic | kappa 4 | 0.557 | +0.058 | 2.58 | 0.54 | 0.49 | 1.23 | 43.4 |
| organic | kappa 8 **chosen** | 0.604 | +0.105 | 1.97 | 0.58 | 0.54 | 4.00 | 26.2 |
| organic | off (alignment vs this field) | 0.499 | — | 2.15 | | | | |
| radial | kappa 0.5 | 0.529 | -0.037 | 2.30 | 0.59 | 0.48 | 1.66 | 20.3 |
| radial | kappa 1 | 0.526 | -0.041 | 1.83 | 0.70 | 0.48 | 2.16 | 9.1 |
| radial | kappa 2 | 0.575 | +0.009 | 2.19 | 0.60 | 0.53 | 1.77 | 19.6 |
| radial | kappa 4 | 0.608 | +0.042 | 2.25 | 0.58 | 0.59 | 1.73 | 17.9 |
| radial | kappa 8 **chosen** | 0.699 | +0.133 | 1.76 | 0.57 | 0.70 | 2.28 | 18.4 |
| radial | off (alignment vs this field) | 0.566 | — | 2.15 | | | | |
| culdesac | stop_bias 0.25, loop 2 (eligible) | | | 2.20 | 0.70 | | 1.99 | 15.3 |
| culdesac | stop_bias 0.5, loop 2 | | | 1.58 | 0.67 | | 1.59 | 6.2 |
| culdesac | stop_bias 0.5, loop 4 | | | 1.75 | 0.63 | | 1.71 | 7.9 |
| culdesac | stop_bias 1.0, loop 4 | | | 1.25 | 0.64 | | 2.48 | 4.1 |

Chosen (rule in configs/m15_style.json): {"grid": {"style_kappa": 4, "flag": ""}, "organic": {"style_kappa": 8, "flag": ""}, "radial": {"style_kappa": 8, "flag": ""}, "culdesac": {"stop_bias": 0.25, "style_loop": 2, "flag": ""}}
