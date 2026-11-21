# KAIST campus: different cities, one change at a time (EXPLORATORY demo; GLO-30 DSM; anchors = real arterials)

Seeds [1, 2, 3]; window scores = mean over the 3 x 3 windows of 512 m; > 15 % / > 10 % over 30 m sub-spans (the GLO-30 resolution). Style classifier fitted on the real calib references.

| Variant | street_km | coverage_60 | xwin_unreach | seam_dead_ratio | share_4way | share_3way | deadend_per_km | loops_per_km2 | block_rect_share | block_phi_med | block_logA_med | jangle_right_share | stroke_xjunc_lw | tree_len_share | phi_boeing | circuity_200_350 | share_gt15 | share_gt10 | classifier says |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P (rules only) | 34.08 | 0.90 | 1.00 | 9.57 | 0.20 | 0.72 | 0.45 | 39.42 | 0.00 | 0.40 | 9.14 | 0.15 | 0.48 | 0.06 | 0.15 | 1.39 | 0.03 | 0.08 | organic 100% |
| A (learned) | 39.39 | 0.83 | 0.11 | 1.09 | 0.20 | 0.79 | 2.83 | 53.26 | 0.01 | 0.28 | 6.18 | 0.26 | 1.32 | 0.20 | 0.28 | 1.47 | 0.05 | 0.12 | organic 89%, culdesac 11% |
| M* (+terrain rules) | 35.24 | 0.75 | 0.12 | 0.93 | 0.20 | 0.79 | 2.69 | 48.18 | 0.01 | 0.26 | 6.33 | 0.28 | 1.28 | 0.20 | 0.38 | 1.41 | 0.05 | 0.12 | organic 81%, culdesac 19% |
| M* TNse (terrain input) | 32.44 | 0.73 | 0.67 | 1.13 | 0.20 | 0.78 | 3.46 | 36.31 | 0.03 | 0.27 | 6.00 | 0.28 | 0.89 | 0.27 | 0.45 | 1.32 | 0.05 | 0.13 | organic 63%, culdesac 37% |
| M* + grid | 48.77 | 0.87 | 0.01 | 0.51 | 0.31 | 0.65 | 0.85 | 100.31 | 0.27 | 0.39 | 7.59 | 0.55 | 2.94 | 0.05 | 0.74 | 1.36 | 0.09 | 0.19 | organic 96%, radial 4% |
| M* + organic | 166.53 | 0.91 | 0.00 | 0.66 | 0.41 | 0.45 | 1.07 | 1376.96 | 0.14 | 0.42 | 5.53 | 0.46 | 3.93 | 0.03 | 0.04 | 1.25 | 0.14 | 0.25 | radial 74%, organic 26% |
| M* + radial | 157.18 | 0.90 | 0.00 | 0.84 | 0.39 | 0.53 | 1.04 | 1066.98 | 0.34 | 0.38 | 5.88 | 0.57 | 5.30 | 0.03 | 0.26 | 1.29 | 0.13 | 0.25 | radial 78%, organic 22% |
| M* + culdesac | 22.61 | 0.66 | 0.72 | 0.84 | 0.17 | 0.80 | 4.64 | 8.19 | 0.01 | 0.24 | 6.66 | 0.29 | 1.05 | 0.39 | 0.38 | 1.28 | 0.05 | 0.12 | culdesac 56%, organic 44% |
| real KAIST: public streets | 15.33 | 0.53 | 0.37 | 2.67 | 0.13 | 0.87 | 1.60 | 4.66 | 0.00 | 0.22 | 6.73 | 0.29 | 1.10 | 0.27 | 0.69 | 1.24 | 0.02 | 0.08 | organic 56%, culdesac 44% |
| real KAIST: vehicular (+service) | 41.68 | 0.85 | 0.48 | 0.73 | 0.15 | 0.85 | 6.06 | 27.55 | 0.15 | 0.37 | 7.11 | 0.44 | 1.62 | 0.30 | 0.52 | 1.60 | 0.04 | 0.13 | culdesac 56%, organic 44% |
