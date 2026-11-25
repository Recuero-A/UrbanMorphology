# Milestone 9: tidy subdivision, green zoning, standard building shapes

**Status (2026-10-31): T and TZ both FAIL their pre-registered criteria** (docs/MILESTONE_9_PLAN.md §4,
applied unchanged).

What T does achieve:
* tidy rows of similar lots;
* buildings stretched from standard templates, never clipped (0 clipped, against 8,083 in C);
* improves on C on 4 of 6 building metrics;
* passes G1–G4.

What it misses:
* about 65 % too few buildings, because 22 % of building lots get no building;
* too many "single" blocks, which give very large buildings;
* far too much green;
* realistic US lots.

## 1. What was built

| Piece | File | Learned / programmed |
|---|---|---|
| 7 standard templates (rectangle, L, T / wing, Z, U, H, cross), 337 classes | `src/urbanmorph/layout/templates.py` | programmed; fit to real footprints in §2.2 of the plan |
| Training records: blocks, street sides, parcel zoning, best template per real building | `scripts/m9_records.py` | from OBSERVED train parcels |
| BlockNets: 4 MLPs for block type, lots per side + rear ratio, green zoning, template choice | `src/urbanmorph/models/block_net.py`, `scripts/m9_train.py` | learned |
| Tidy generator | `src/urbanmorph/layout/tidy_parcels.py` | see its docstring |
| Subdivision metrics S1–S4, identical code for observed and generated parcels | `src/urbanmorph/layout/subdivision_metrics.py` | — |
| Evaluation | `scripts/m9_evaluate.py` → `outputs/m9/eval/` | — |

Training: both seeds took about 15 s each on CPU (checkpoints in `checkpoints/m9/`, not committed).

| Net (validation NLL, best epoch) | Seed 0 | Seed 1 |
|---|---|---|
| type (67 val blocks) | 0.496 (27) | 0.501 (19) |
| side (371 val sides) | 1.950 (37) | 1.949 (36) |
| zone (2,646 val parcels) | 0.0300 (23) | 0.0301 (17) |
| shape (1,365 val buildings) | 4.008 (33) | 4.031 (14) |

Coverage, number of buildings and setbacks come from the M8 ParcelNet (seed matched).

### Changes made during implementation (on train tiles, before evaluation; criteria untouched)

1. **Remainder rule.** The plan turned every block remainder of 400 m² or more into green. On train
   tiles this covered half of some tiles, in large blocks and blocks cut by the tile edge. Now:
   * pieces of 400–4,000 m² become green (4,000 m² ≈ the 75th percentile of real green parcels);
   * larger pieces are split recursively into interior lots of about the block's median lot area.
2. **Zoning without the neighbour feature.** With one green neighbour the learned probability of
   green is 0.6–0.7. Sampled sequentially, green therefore spread lot to lot: 25–40 % of parcels
   became green on train tiles, against 1–3 % in real data. Zoning is now sampled from the parcel's
   own features only.
3. **Template fit on corner lots.** The template envelope is taken from the clipped lot. The template
   tries the centre, left and right positions before shrinking.
4. **Rear ratio definition.** The rear ratio is the distance from the edge to the lot's rear, divided
   by the block depth. The plan's §2.1 used lot depth instead. The new definition makes back-to-back
   US lots read about 0.5 rather than 0.42, because US parcel lines start about 5 m behind the road
   surface. S2 uses the same definition for observed and generated parcels.

The template labels are fitted in the lot frame (x along the street). In that frame only 59 % (Spain)
and 73 % (US) of real buildings fit a template at IoU ≥ 0.9; in each building's own frame the figure
was 82.7 %. Real buildings are often not parallel to their street, while M9 places them parallel.

## 2. Results

Evaluation setup:
* 10 validation tiles with observed parcels (Spain 6, US 4);
* 2 seeds, 4 samples per tile, real street networks;
* W1 / IQR against the real data of the same tiles;
* outputs: `outputs/m9/eval/results.json`;
* figures: `zoom_C_T_TZ.png` and `full_C_T_TZ.png`, which draw generated parcels dashed and
  observed parcels in red.

### Buildings and green

| | C s0 | C s1 | T s0 | T s1 | TZ s0 | TZ s1 | real |
|---|---|---|---|---|---|---|---|
| buildings per tile (median) | 333 | 336 | 134.5 | 141 | 126.5 | 139.5 | 381.5 |
| building coverage | 0.371 | 0.369 | 0.177 | 0.153 | 0.134 | 0.139 | 0.289 |
| green coverage | 0.011 | 0.007 | 0.097 | 0.088 | 0.113 | 0.135 | 0.013 |
| building area p99 (m²; limit 1.25 × real = 1,526) | 3,190 | 3,022 | 2,030 | 2,102 | 1,657 | 1,405 | 1,221 |
| clipped to parcel | 8,083 | 8,218 | 0 | 0 | 0 | 0 | |
| building lots without a building (not fitted) | 0 | 0 | 22.1 % | 22.4 % | 21.8 % | 22.4 % | |
| acceptance of proposed buildings | 0.968 | 0.968 | 0.999 | 0.996 | 0.996 | 0.999 | |

| W1 / IQR (primary building metrics) | C s0 | C s1 | T s0 | T s1 | TZ s0 | TZ s1 |
|---|---|---|---|---|---|---|
| nn_aligned_5deg | 0.21 | 0.21 | 0.25 | 0.32 | 0.27 | 0.30 |
| spacing_cv | 1.43 | 1.34 | **0.63** | **0.68** | 0.86 | 0.50 |
| setback_std | 2.63 | 0.91 | **0.25** | **0.29** | 0.24 | 0.19 |
| frontage_share | 1.35 | 1.35 | **0.52** | **0.48** | 0.54 | 0.44 |
| frontage_size_similarity | 1.10 | 1.14 | **0.65** | **0.61** | 0.71 | 0.65 |
| pooled_log_area | 0.17 | 0.17 | 0.22 | 0.18 | 0.17 | 0.15 |
| building coverage | 1.47 | 1.21 | 1.05 | 1.03 | 1.25 | 1.23 |
| green coverage | 0.40 | 0.33 | 0.87 | 0.62 | 0.90 | 1.03 |
| green rectangularity | 0.22 | 0.26 | 0.49 | 0.47 | 0.41 | 0.31 |

### Parcels (W1 / IQR against observed parcels)

| | C s0 | T s0 | T s1 | TZ s0 |
|---|---|---|---|---|
| Spain: log area / frontage / depth / coverage / front setback / buildings | 0.60 / 0.29 / 0.30 / 1.08 / 0.25 / 0.27 | 0.69 / 0.84 / 0.42 / 2.30 / 1.17 / 0.59 | 0.60 / 0.71 / 0.39 / 2.38 / 1.01 / 0.62 | 0.72 / 0.82 / 0.42 / 2.36 / 0.92 / 0.61 |
| Spain: metrics ≤ 0.5 | 4 | 1 | 1 | 1 |
| US: log area / frontage / depth / coverage / front setback / buildings | 0.87 / 0.59 / 0.81 / 0.44 / 0.36 / 0.70 | 0.47 / 1.19 / 0.72 / 1.03 / 0.21 / 0.77 | 0.50 / 1.21 / 0.78 / 0.98 / 0.24 / 0.76 | 0.47 / 1.21 / 0.72 / 1.06 / 0.22 / 0.79 |
| US: metrics ≤ 0.5 | 2 | 2 | 2 | 2 |
| Moran's I Spain / US (observed 0.52 / 0.57; not gating) | 0.28 / 0.28 | 0.18 / 0.25 | 0.24 / 0.23 | 0.20 / 0.24 |

### Subdivision (S1–S4)

| | observed | C s0 | T s0 | T s1 |
|---|---|---|---|---|
| Spain S1 width spread, median (W1 / IQR) | 0.59 | 0.68 (0.32) | 0.58 (**0.17**) | 0.57 (**0.40**) |
| US S1 width spread, median (W1 / IQR) | 0.20 | 0.73 (1.39) | 0.42 (**0.38**) | 0.38 (**0.48**) |
| Spain S2 rear ratio, median (W1 / IQR) | 0.46 | 0.38 (2.74) | 0.50 (2.70) | 0.50 (3.48) |
| US S2 rear ratio, median (W1 / IQR) | 0.49 | 0.40 (133.5) | 0.50 (145.7) | 0.50 (132.3) |
| Spain S3 types none / single / few / rows (TV) | 16 / 4 / 4 / 76 % | 0 / 1 / 10 / 89 (0.19) | 13 / 28 / 5 / 54 (0.25) | 12 / 23 / 7 / 58 (0.22) |
| US S3 types (TV) | 24 / 0 / 0 / 76 % | 0 / 0 / 1 / 99 (0.24) | 0 / 25 / 6 / 69 (0.31) | 0 / 25 / 4 / 71 (0.29) |
| S4 interior leftover, median Spain / US | 0 / 0.0028 | 0 / 0 | 0.0002 / 0.00005 | 0.0002 / 0.00005 |

Notes:
* The observed side comes from only 50 Spanish and 17 US closed blocks on these 10 tiles.
* **S2 cannot be used as a metric.** Real US rear ratios are concentrated (IQR 0.03, median 0.49),
  and both C and T produce rare extreme ratios. The block depth is measured along the normal from an
  edge's midpoint, so on concave blocks it can be very small, which inflates the ratio. The medians
  (0.50 vs 0.49) agree, but the W1 is dominated by these outliers. The criterion fails as defined.
* **S4 fails by 0.02 % of block area** in Spain (0.0002 vs exactly 0 observed). This most likely
  comes from the 1 cm gaps the geometry code leaves between lots (not confirmed). Leftover space is
  practically gone, but the rule as written fails.

### Decisions

| Criterion | T s0 | T s1 | TZ s0 | TZ s1 |
|---|---|---|---|---|
| parcel realism (≥ 4 / 6 ≤ 0.5, both countries) | no | no | — | — |
| tidiness (S1, S2 ≤ 0.5; S3 TV ≤ 0.15) | no (S2, S3) | no (S2, S3) | — | — |
| no leftovers (S4 ≤ observed) | no (+0.0002) | no (+0.0002) | — | — |
| no giant buildings (p99 ≤ 1,526 m²) | no (2,030) | no (2,102) | — | — |
| mechanism (0 clipped, 0 acute, acceptance ≥ 0.95) | yes | yes | — | — |
| improves on C on ≥ 4 / 6 | yes (4) | yes (4) | — | — |
| green coverage W1 ≤ 0.5 | — | — | no (0.90) | no (1.03) |
| green shape W1 ≤ 0.5 | — | — | yes (0.41) | yes (0.31) |
| parcel realism within +0.05 of T | — | — | yes | yes |
| G1 / G2 / G3 / G4 | yes / yes / yes / yes | yes / yes / yes / yes | yes / yes / yes / yes | yes / **no** / yes / yes |
| G5 density | **no** (−65 %) | **no** (−63 %) | **no** | **no** |
| **Success** | **no** | **no** | **no** | **no** |

## 3. Failure analysis (to test on train tiles; not yet tested)

1. **Empty lots.**
   * 22 % of building lots get no building: the template does not fit even after shrinking.
   * With ParcelNet's unbuilt parcels, this gives 65 % too few buildings and low Spanish parcel
     coverage (W1 2.3).
   * A likely cause: the envelope is the lot's axis-aligned extent in the street frame. For clipped
     or skewed lots the template overhangs the lot, so the containment test fails.
   * The template choice ignores the lot's true shape.
2. **Too many "single" blocks:** 23–28 % generated, against 0–4 % observed on these tiles and 7–12 % on
   train tiles. Single-parcel blocks carry the very large buildings (p99) or stay almost empty
   (Valencia in the figure).
3. **Green is 7–10× real.**
   * Every "none" block of 400 m² or more becomes green, as do remainder pieces of 400–4,000 m².
   * Real green sits in a few large parcels (§2.3 of the plan), not in every leftover.
4. **US frontage too wide** (W1 1.2): generated US sides get fewer lots than real (median 7).
5. **S2 must be redefined** before it can be used, for example with a robust rear ratio or depth
   measured to the block's medial axis. That change would need a new pre-registration.

## 4. What did work

* **Tidy rows.** Widths along a street side match reality much better than C: S1 W1 ≤ 0.5 in both
  countries and both seeds, against 1.39 for C in the US. The New York and Los Angeles panels show
  regular rows.
* **Template buildings.** Zero buildings clipped, zero acute angles. Every building is a stretched
  standard shape, and setback and frontage regularity are much closer to real (setback_std W1
  0.25 vs 0.91–2.63 for C).
* **Green parcels have clean shapes** (rectangularity W1 0.31–0.49, criterion met in TZ). The problem
  is the amount of green, not its shape.

---

# M9b: street-access splitting, no-access parcels → green, inscribed-rectangle fit

**Status (2026-10-31): B FAILS its pre-registered criteria** (plan §6, applied unchanged), for both seeds.

What B fixes:

* the fit: 3.4–3.9 % of building lots without a building, against 22 % for TZ (criterion ≤ 5 % met);
* giant buildings: p99 1,055–1,207 m², against 1,405–1,657 m² for TZ (limit 1,526 m²; met);
* leftovers: none (met).

What B breaks: the user's rule turns about 40 % of the tile area into green (real: 1.3 %), so buildings
drop further, to 114–132 per tile.

Outputs: `outputs/m9/eval_b/` (`results.json`, `zoom_TZ_B.png`, `full_TZ_B.png`) and
`outputs/compare/b_cities_seed0.png` (5 cities from scratch).

## Implementation notes

These choices were made on train tiles, before the evaluation:

* **Tile-edge access.** A parcel touching the tile edge for ≥ 4 m counts as having street access,
  because the 512 m window cuts it and its street is probably outside. Without this, every interior
  lot of a tile-edge block became green: 187 green parcels on one train tile.
* **One park per area.** Adjacent no-access pieces are merged before the green-or-merge decision, so
  they form one park instead of a grid of green lots.
* **Minimum lot width.** The access split rejects cuts that leave a piece narrower than 5 m. Without
  this, a narrow block on a train tile was cut into thin strips.
* **Drawing bug fixed** in `layout/export._fill`. Polygon holes were painted in the ground colour,
  which hid green drawn inside them, so large green parcels appeared white. Holes are now cut out of
  the patch.
  * Metrics were never affected; they are computed on geometry.
  * The M9 evaluation was rerun with identical numbers and decisions; its figures, the TZ cities and
    the M9b figures were regenerated.

## Results (same 10 validation tiles, 2 seeds; these tiles were already seen in the M9 figures)

| | TZ s0 | TZ s1 | B s0 | B s1 | real / observed |
|---|---|---|---|---|---|
| buildings per tile | 126.5 | 139.5 | 114 | 132 | 381.5 |
| building coverage | 0.134 | 0.139 | 0.094 | 0.096 | 0.289 |
| green coverage | 0.113 | 0.135 | **0.399** | **0.393** | 0.013 |
| building area p99 (m²) | 1,657 | 1,405 | **1,207** | **1,055** | 1,221 |
| not-fitted share | 0.218 | 0.224 | **0.039** | **0.034** | |
| built share of non-green parcels, Spain | 0.40 | 0.37 | 0.48 | 0.49 | 0.93 |
| built share of non-green parcels, US | 0.74 | 0.76 | 0.86 | 0.89 | 0.92 |
| S2b Spain / US (W1 / max(IQR, 0.05)) | 0.15 / 1.26 | 0.12 / 1.17 | 0.29 / 2.45 | 0.25 / 2.21 | |
| S1 Spain / US | 0.26 / 0.37 | 0.66 / 0.38 | 0.62 / 0.88 | 0.33 / 0.87 | |
| Spain parcel coverage W1 | 2.36 | 2.44 | 2.37 | 2.28 | |
| US frontage W1 | 1.21 | 1.20 | 1.85 | 1.80 | |
| nn_aligned_5deg W1 | 0.27 | 0.30 | 0.22 | 0.24 | |
| green rectangularity W1 | 0.41 | 0.31 | 0.20 | 0.17 | |

| Criterion (plan §6) | B s0 | B s1 |
|---|---|---|
| built share within 0.10 of observed | no (Spain 0.48 vs 0.93) | no |
| G5 density | no | no |
| parcel realism | no | no |
| tidiness (S1, S2b, S3) | no | no |
| no leftovers | **yes** | **yes** |
| no giant buildings | **yes** | **yes** |
| mechanism (0 clipped, 0 acute, acceptance ≥ 0.95, not-fitted ≤ 0.05) | **yes** | **yes** |
| improves on TZ on ≥ 4 / 6 | no (2 / 6) | no (2 / 6) |
| G1 / G2 / G3 / G4 | yes / no / no / yes | yes / no / no / yes |
| **Success** | **no** | **no** |

## Diagnosis (to test on train tiles)

1. **Green is 30× real.** The no-access rule turns every block interior behind the street lots into
   green. Generated street lots are shallower than the blocks are deep, especially on irregular blocks.
   **The rule conflicts with the real data**: 16 % of real parcels have no frontage and 54–68 % of
   those are built.
2. **Spanish lots are left unbuilt too often:** built share 0.48 against 0.93. ParcelNet's "built"
   probability for generated Spanish lots is too low, and Spanish parcel coverage W1 stays at 2.3–2.4
   in every method. ParcelNet sees generated lots whose context (block size, neighbours) differs from
   real ones.
3. **US frontage gets worse with splitting** (W1 1.8): the access split makes wide street lots on
   irregular blocks.

## Options (need the user's decision)

* (a) Deeper street lots (back-to-back to the block centre), so that little interior remains. This
  keeps the user's rule.
* (b) Only no-access pieces larger than a park-size threshold become green; smaller ones become
  built interior lots, as in the real data.
* (c) Recalibrate ParcelNet's built probability for Spain.
