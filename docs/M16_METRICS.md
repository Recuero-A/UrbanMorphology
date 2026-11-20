# Milestone 16 metrics: definitions and units

* **Scope.** Every graph is a RoadGraph in tile metres: a 512 m projected tile, x east, y north. The KAIST campus
  domain is 1536 m. Real and generated graphs go through the same code.
* **Censoring.** Tile-edge nodes are recomputed geometrically for every graph (`is_border`, 6 m), so dead ends and
  faces created by the window boundary are censored identically.
* **Tests.** `tests/test_m16_outcomes.py` checks fixtures with known answers and rotation / clipping sensitivity;
  `tests/test_m16_terrain_checks.py` checks units and slopes.

## 1. Independent network outcomes (`graphs/network_outcomes.py`)

No style rule optimises these.

| Name | Definition | Unit / range | Reference (fixture) |
|---|---|---|---|
| `block_logA_med`, `block_logA_sd` | median and SD of ln(area) of interior faces (polygonized street graph; faces touching the tile edge censored) | ln m² | 64 m grid: ln 4096, SD 0 |
| `block_phi_med` | median shape factor Φ = area / area of the minimum bounding circle (Louf & Barthélemy 2014, eq. 3.2) | 0–1 | square 2/π = 0.637; 2:1 rectangle 0.51 |
| `block_rect_share` | faces that simplify (3 m) to 4 corners each within 90 ± 15° | share | grid 1 |
| `jangle_right_share`, `jangle_sd_deg` | angles between bearing-consecutive edges at interior junctions (degree ≥ 3): share within 90 ± 15°, SD | share, ° | grid 1 / 0; T-junction 2/3 |
| `stroke_len_lw` | strokes (natural streets): at every node incident edges are paired greedily by smallest deflection < 30° and chained. Length-weighted mean stroke length | m | full-width grid line 512 |
| `stroke_top10_share` | length share of the longest 10 % of strokes (hierarchy) | share | – |
| `stroke_xjunc_lw` | length-weighted mean number of junctions a stroke passes through (multi-block continuity) | count | crossing: 1 |
| `through_share` | interior junctions where some stroke continues straight through | share | grid 1; tree 0 |
| `tree_len_share` | street length outside the 2-core (degree-1 nodes pruned repeatedly; tile-edge nodes never pruned) | share | tree 1; grid 0 |
| `term_len_med` | median network distance from an interior dead end to the 2-core | m | – |
| `term_branch_med` | median junctions on that path (terminal branch depth) | count | – |
| `term_attached_share` | interior dead ends whose tree reaches a 2-core node (collector access) | share | – |
| `spoke_span_lw` | strokes whose length-weighted \|cos(bearing − radial)\| ≥ cos 20° are spokes; length-weighted radial extent ÷ 256 m | 0–1+ | spoke-and-ring (r ≤ 192): 0.75 |
| `spokes_long` | spokes spanning ≥ 0.4 × 256 m (opposite spokes through the centre form one stroke) | count | 8 spokes: 4 |
| `ring_cover_max` | largest angular coverage of ring-direction edges in any 48 m radius band starting at 48 m | 0–1 | rings: 1 |
| `sector_circ`, `sector_unreach` | median circuity and unreachable share of node pairs in different quadrants about the centre, ≥ 100 m apart | ratio, share | – |
| `phi_boeing` | orientation order 1 − ((H − ln 4)/(ln 36 − ln 4))², H the entropy of the length-weighted bidirectional bearing histogram, 36 bins of 10° centred on multiples of 10° (Boeing 2019) | 0–1 | grid 1; uniform ≈ 0 |

## 2. Inherited metrics

From M13–M15; definitions are unchanged and the names are kept:
* M15 morphology (`style_metrics.morphology_row`): `orth_order` (R4), junction shares, `sinuosity`, `turn_med_deg`,
  radial / tangential shares, `deadend_per_km`, `loops_per_km2`, `meshedness`, OD circuity bands, block areas, grade
  exposure.
* M14 connectivity: `od_unreach_land` (fixed point sampling, points in water excluded), `closed_giant`,
  `coverage_60`.
* M13 terrain:
  * `grade_p90_m13`: per-edge maximum grade over sliding 30 m spans, length-weighted p90;
  * `contour_align`: |cos| between edge and terrain gradient, edges on slope ≥ 5 %.

**Same name, different definition.** These were not merged (`docs/M16_MILESTONE_TABLE.md` §5):
* `grade_p90`: M12 uses the edge-mean grade; M15 uses 10 m sub-spans.
* `share_gt10`: whole edge in M12, sub-span share in M15.

M16 reports the M15 sub-span shares (`share_gt15`, `share_gt10`, 10 m spans on 4 m rasters) and the M13 30 m
measure side by side.

## 3. Controllability checks

These are the rules' own objectives. They show that a rule acts; they are never evidence of a pattern:
* `field_alignment` = (1 + cos 4(θ − θ_field))/2, length-weighted;
* R4;
* radial + tangential share;
* dead ends per km.

## 4. Proposal diagnostics (sampler log, per tile)

| Name | Meaning |
|---|---|
| `prop_decisions`, `prop_skipped` | decisions with proposals on; decisions where the trigger did not fire |
| `cand_style`, `feas_style`, `feas_model` | style candidates generated; feasible style and learned candidates |
| `no_aligned_model` | decisions with no feasible learned candidate within 15° of a pattern direction |
| `rescued_aligned` | of those, decisions with a feasible style candidate |
| `rescued` | decisions with no feasible learned candidate but a feasible style candidate |
| `chosen_model`, `chosen_style`, `chosen_<label>` | sources of accepted moves |
| `style_hard_<reason>` | style candidates rejected by a hard rule |
| `hard_terminal`, `hard_local_loop` | cul-de-sac topology rules |
| `topo_continue` | decisions whose stop odds fell because a pattern continuation was feasible |
| `style_plaus_sum / style_plaus_n` | mean model log-density of style candidates relative to the median learned candidate (nats) |

## 5. Recognisability

**Classifier.** A nearest centroid in the space of the primary outcomes (`docs/M16_PROTOCOL.md` §5):
* standardised by the pooled within-style SD;
* fitted on the `calib` real references only (`configs/m16_style_refs`).

**Recognition rate.** The share of generated networks classified as the requested style; chance is 0.25.
