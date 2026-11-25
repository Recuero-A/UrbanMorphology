# M18: terrain response on natural terrain without street beds (leakage control b)

Windows: 16 (rule in `windows.json`); script `scripts/m18_natural_terrain.py`; figure `fig_natural.png`.

| Method [backbone] | runs | km/km² | share > 15 % | G | C (contour) | S (siting) | TAS |
|---|---|---|---|---|---|---|---|
| P [P] | 2 | 12.9 | 0.092 | 0.91 | 0.50 | 0.69 | 70.1 |
| M13-SMC [TN] | 6 | 9.4 | 0.125 | 0.87 | 0.40 | 0.77 | 68.2 |
| NA-fixed [TN] | 6 | 14.1 | 0.126 | 0.87 | 0.41 | 0.76 | 68.1 |
| NA-SMC [TN] | 6 | 13.4 | 0.137 | 0.86 | 0.42 | 0.74 | 67.3 |
| M* [TN] | 6 | 12.6 | 0.138 | 0.86 | 0.39 | 0.76 | 67.1 |
| M* [TNlp] | 6 | 12.3 | 0.126 | 0.87 | 0.49 | 0.65 | 67.1 |
| A [TN] | 6 | 12.2 | 0.146 | 0.85 | 0.38 | 0.77 | 66.7 |
| NA-myopic [TN] | 6 | 16.8 | 0.161 | 0.84 | 0.40 | 0.71 | 65.1 |
| M* [TNse] | 6 | 8.8 | 0.147 | 0.85 | 0.59 | 0.47 | 63.9 |
| M* [N0] | 6 | 14.4 | 0.268 | 0.73 | 0.45 | 0.53 | 57.0 |
