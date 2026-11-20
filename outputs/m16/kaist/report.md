# KAIST campus-wide pilot (recurring comparison; GLO-30 DSM; style off)

Domain 1536 m x 1536 m, origin UTM 52N 352344, 4025483; seeds [1, 2, 3]. Anchors: 206 OSM trunk / primary / secondary / tertiary edges (supplied, not generated).

| Approach | street km | 60 m coverage | cross-window OD unreachable | seam dead-end ratio | seconds |
|---|---|---|---|---|---|
| anchored M* | 35.24 | 0.749 | 0.111 | 0.93 | 16 |
| global M* | 26.63 | 0.645 | 0.459 | 0.99 | 10 |
| tiles M* | 36.92 | 0.834 | 0.999 | 7.97 | 5 |
| tiles NA-fixed | 39.13 | 0.858 | 0.999 | 35.70 | 71 |
| real streets (training definition) | 15.33 | 0.530 | 0.367 | 2.67 | – |
| real vehicular (+ service) | 41.68 | 0.846 | 0.477 | 0.73 | – |

Seam dead-end ratio: dead ends per km within 24 m of an internal window edge / elsewhere (1 = no seam artefact). Anchored runs include the supplied anchors in their street km. Real references: the pack's 9 clipped windows re-joined at their shared cut points (0.5 m).
