# Milestone 15 metric definitions

**Code.** `src/urbanmorph/graphs/style_metrics.py` (per graph) and `src/urbanmorph/metrics/fidelity.py` (sets of
graphs). Tests: `tests/test_m15_style.py`, on analytic fixtures and deliberately wrong outputs.

**Conventions.**
* Graphs are in tile metres: a projected (UTM) 512 m × 512 m tile, x east, y north.
* An **edge** is one stored graph edge: about 26 m when generated, resampled when real.
* A **link** is a junction-to-junction chain of edges.
* **Border nodes** are those within 1 m of the tile edge. They are recomputed geometrically for real and generated
  graphs alike.
* **Interior** means not on the border. Dead ends at the tile boundary are censored.
* All lengths are of the 2D centreline.

## Style descriptors (joint, never one alone)

| Style | Metric | Unit | Definition | Fixture check |
|---|---|---|---|---|
| grid | `orth_order` (R4) | – | \|Σ l·e^{4iθ}\| / Σ l over edges (θ = edge bearing, l = length). 1 = only two perpendicular directions (any rotation); 0 = uniform | Grid at 0° and 30°: 1.000 |
| grid | `share_4way` | – | 4-way junctions / interior junctions (degree ≥ 3) | Grid: > 0.7 |
| all | `share_3way`, `share_5plus` | – | Likewise for 3-way and 5+-way junctions | – |
| organic | `sinuosity` | – | Length-weighted mean of link length ÷ chord, for links with chord ≥ 30 m | Grid: 1.0 |
| organic | `turn_med_deg` | degrees | Median absolute heading change at interior degree-2 vertices | Grid: 0 |
| organic | `meshedness` | – | (E − V + C) / (2V − 5) | – |
| radial | `radial_share`, `tangential_share` | – | Length share of edges within ±20° of the radial / ring direction about a centre. Edges within 30 m of the centre are excluded. Chance level 0.222 each | Spoke-and-ring at its centre: sum > 0.9 |
| radial | `centre_x`, `centre_y` | m | Synthetic: the declared centre (tile centre). Real: `estimate_centre`, the 32 m grid point (64 m from the edge) maximising radial + tangential share, one rule for every graph | Recovered within 32 m |
| cul-de-sac | `deadend_per_km` | 1/km | Interior degree-1 nodes per km of street | Dendritic > 1, grid 0 |
| cul-de-sac | `loops_per_km2` | 1/km² | Cyclomatic number E − V + C per km² of tile | Dendritic 0, grid > 50 |

**Bearing entropy alone** (orientation φ, M2/M12) is not used as a style descriptor: it cannot separate radial from
organic layouts.

## Controllability check (not independent evidence)

| Metric | Unit | Definition |
|---|---|---|
| `field_alignment` | – | Length-weighted mean of (1 + cos 4(θ − θ_field)) / 2 against the requested style field (mixtures weighted). 1 = every street on the cross field; 0.5 = uniform bearings. This is the generator's own objective, so it is only a controllability check |

## Real-data resemblance (features outside every generator objective)

| Metric | Unit | Definition |
|---|---|---|
| `block_area_med`, `block_area_p90`, `blocks` | m², count | Interior faces of the planar street graph (shapely polygonize); faces within 2 m of the tile edge are censored |
| `circuity_{lo}_{hi}` | – | Median network ÷ Euclidean distance between random node pairs in fixed bands 100–200, 200–350 and 350–500 m. Connected pairs only, up to 300 pairs per band |
| `unreach_{lo}_{hi}` | – | Share of those pairs with no network path. Reported separately, never folded into circuity |
| `street_km` | km | Total street length in the tile |
| `edges`, `bridges`, `empty` | count | Validity |

**Fidelity** (`metrics/fidelity.prdc`):
* Per-tile vectors of [log block_area_med, log block_area_p90, share_3way, share_4way, circuity_100_200,
  circuity_200_350] are standardised with the real set's mean and standard deviation.
* Balls use k = 5 nearest neighbours.

| Measure | Definition |
|---|---|
| Precision (Kynkäänniemi et al. 2019) | Share of generated tiles inside a real ball |
| Recall (Kynkäänniemi et al. 2019) | Share of real tiles inside a generated ball |
| Density (Naeem et al. 2020) | Mean number of real balls containing a generated tile, ÷ k |
| Coverage (Naeem et al. 2020) | Share of real balls containing at least one generated tile |

* The benchmark is `real_vs_real`: the same numbers for random metro-level halves of the real set (20 splits).
* Tiles with a non-finite feature, such as no interior block, are dropped and counted.

## Terrain exposure

| Metric | Unit | Definition |
|---|---|---|
| `grade_p90` | – (m/m) | Ground edges only (bridges and tunnels excluded), sampled every 2 m on the true elevation raster (bilinear). Directional grade \|Δz\| / Δs over consecutive sub-spans of 10 m (4 m synthetic rasters) or 30 m (GLO-30, its resolution). Length-weighted 90th percentile |
| `share_gt10`, `share_gt15` | – | Length share of those sub-spans steeper than 10 % / 15 % |
| `grade_nodata_share` | – | Length share of sub-spans touching no-data. They are excluded from the above and reported, never silently dropped |
| `ground_km` | km | Ground street length scored |

Fixture check: on a 10 % analytic incline, an edge along the fall line scores 0.100 and one along the contour 0.000.

## Connectivity and coverage (from M14, unchanged)

`od_unreach_land`, `islands`, `closed_giant`, `coverage_60` (`src/urbanmorph/graphs/m14_metrics.py`,
`docs/M14_EXPERIMENTAL_PROTOCOL.md`).

## Diversity

The mean pairwise Jaccard distance of 8 m occupancy grids across the seeds of one site
(`src/urbanmorph/graphs/m13_metrics.py`). It is measured separately from the temperature analysis.
