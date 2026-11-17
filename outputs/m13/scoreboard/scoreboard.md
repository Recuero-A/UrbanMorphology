# M* vs A scoreboard (development data, TN backbone; exploratory)

Mean over 3 training seeds; [min, max] across seeds. Winner = better on the mean (closer to real where that is the target). Not significance: see the CI section.


## DEV-B (27 tiles, confirmation set)

| Metric | target | real | M* | A | better |
|---|---|---|---|---|---|
| S_T terrain score | lower | — | 0.354 [0.327, 0.398] | 0.42 [0.384, 0.438] | **M*** |
| S10 street-network realism | lower | — | 0.418 [0.416, 0.42] | 0.426 [0.401, 0.466] | **M*** |
| joint realism: energy distance | lower | — | 0.603 [0.582, 0.641] | 0.719 [0.588, 0.812] | **M*** |
| joint realism: MMD | lower | — | 0.0752 [0.0705, 0.0833] | 0.0934 [0.0776, 0.111] | **M*** |
| street density / real | 1.0 | 1 | 1.12 [1.08, 1.16] | 1.13 [1.1, 1.21] | **M*** |
| largest connected piece | ≈ real | 0.776 | 0.86 [0.836, 0.898] | 0.874 [0.863, 0.882] | **M*** |
| intersections per km² | ≈ real | 70.2 | 96.9 [96.2, 97.7] | 96.3 [91.8, 102] | **A** |
| dead-end ratio | ≈ real | 0.0253 | 0.0474 [0.0444, 0.0491] | 0.0496 [0.0426, 0.0544] | **M*** |
| contour alignment | ≈ real | 0.493 | 0.5 [0.477, 0.517] | 0.491 [0.474, 0.509] | **A** |
| bridges per tile | ≈ real | 0.852 | 0.346 [0.259, 0.417] | 3.67 [3.23, 3.93] | **M*** |
| unbridged river crossings per tile | lower | 0 | 0 [0, 0] | 0.309 [0.185, 0.426] | **M*** |
| diversity between samples | higher | — | 0.744 [0.738, 0.752] | 0.751 [0.746, 0.756] | **A** |

Count of metrics: M* better 9, A better 3, tie 0.

Paired block-bootstrap 95 % CI of M* − A, per training seed (negative = M* better):

* S_T: s0: -0.110 [-0.254, +0.006]; s1: -0.046 [-0.221, +0.105]; s2: -0.041 [-0.179, +0.229]
* S10_cond: s0: +0.017 [-0.068, +0.073]; s1: +0.007 [-0.046, +0.058]; s2: -0.049 [-0.120, +0.039]

## GEN (100 tiles: San Francisco, Girona, Manresa; never trained on)

| Metric | target | real | M* | A | better |
|---|---|---|---|---|---|
| S_T terrain score | lower | — | 0.32 [0.307, 0.344] | 0.37 [0.331, 0.409] | **M*** |
| S10 street-network realism | lower | — | 0.384 [0.361, 0.402] | 0.406 [0.396, 0.413] | **M*** |
| joint realism: energy distance | lower | — | 0.77 [0.571, 0.9] | 0.82 [0.664, 0.992] | **M*** |
| joint realism: MMD | lower | — | 0.108 [0.0783, 0.126] | 0.116 [0.0921, 0.142] | **M*** |
| street density / real | 1.0 | 1 | 1.05 [1.02, 1.09] | 1.08 [1.06, 1.11] | **M*** |
| largest connected piece | ≈ real | 0.921 | 0.894 [0.885, 0.905] | 0.913 [0.901, 0.921] | **A** |
| intersections per km² | ≈ real | 117 | 131 [129, 135] | 139 [136, 142] | **M*** |
| dead-end ratio | ≈ real | 0.0173 | 0.0488 [0.0437, 0.0535] | 0.049 [0.0457, 0.055] | **M*** |
| contour alignment | ≈ real | 0.537 | 0.52 [0.511, 0.528] | 0.537 [0.529, 0.549] | **A** |
| bridges per tile | ≈ real | 2.08 | 0.396 [0.315, 0.44] | 3.63 [2.88, 4.24] | **A** |
| unbridged river crossings per tile | lower | 0.06 | 0 [0, 0] | 0.658 [0.65, 0.662] | **M*** |
| diversity between samples | higher | — | 0.798 [0.794, 0.802] | 0.801 [0.797, 0.808] | **A** |

Count of metrics: M* better 8, A better 4, tie 0.

Paired block-bootstrap 95 % CI of M* − A, per training seed (negative = M* better):

* S_T: s0: -0.021 [-0.077, +0.023]; s1: -0.062 [-0.109, +0.003]; s2: -0.065 [-0.111, -0.025] (significant)
* S10_cond: s0: -0.036 [-0.091, +0.014]; s1: -0.018 [-0.046, +0.021]; s2: -0.011 [-0.050, +0.043]
