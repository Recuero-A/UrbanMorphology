# Milestone 12 evaluation protocol (pre-registered)

**Pre-registered 2026-11-04, before any M12 model was trained or sampled.** Changes after this date go in §8, with
their date and reason. Criteria are never changed after results are seen.

## 1. Data, splits and units

* **Unit:** the 512 m tile. Statistics are computed per tile, then compared across tiles.
* **Training:** the 565 M2 / M3b train tiles plus 120 steep Los Angeles tiles (M12, selected by measured landform
  slope; `scripts/m12_select_regions.py`).
* **Development (model selection, thresholds, all reported M12 numbers):**
  * `val`: the 19 M2 validation tiles (12 with a DEM);
  * `val_t`: 40 San Diego tiles, a **held-out city** never used for training (23 steep, 7 moderate, 10 flat at
    screening).
* **Untouched until authorised:** `test` (19 M2 tiles) and `test_t` (34 San Diego tiles, not even built).
* **Thresholds** (feasibility, normalisation) come from **training tiles only**.
* **Japan:** water and streets only (no DEM, GSI licence unresolved); Japanese tiles have no terrain metrics.

## 2. Models

| ID | Steps from | Feasibility | Notes |
|---|---|---|---|
| N0 | TerrainGraphTurtle, no terrain inputs (3D-aware graphs, structure head) | — | terrain-blind control, 2 seeds |
| R | N0 | rules: infeasible steps dropped; bridges and viaducts set by rules | |
| TN | TerrainGraphTurtle, terrain inputs elev + slope + water | — | 2 seeds |
| TN+R | TN | rules as R | |
| TN+HR | TN | heuristic replanning (§5) | |
| TN+LR | — | — | only with a defensible objective (spec §10.5); see the final report |
| P | procedural: least-cost paths with slope, curvature and water costs (Galin-style) | built in | no learning |
| TN-elev, TN-slope, TN-water, TN-elev+slope | ablations | — | 1 seed each |

All learned models share the N recipe, budget and data order (`scripts/m12_train.py`). **Sampling temperature
T = 0.5** for every learned model (fixed in advance: the M10 setting; not tuned on M12 data).

## 3. Metrics

**S10 (unchanged definition, M10 protocol §2.2):** 10 street metrics, W1/IQR against real tiles, mean.

**S_T (terrain score, new):** mean of the following W1/IQR terms over tiles with a DEM, against the real streets of
the SAME tiles' population:
1. `grade_p90`: length-weighted p90 of the mean edge grade (ground edges; the terrain profile sampled every 2 m);
2. `share_gt10`: share of street length whose maximum 10 m sub-span grade exceeds 10%;
3. `contour_index`: length-weighted mean |cos| of the angle between the street and the terrain gradient, on edges
   whose landform slope (32 m) exceeds 5% (1 = straight up the slope, 0 = along the contour);
4. `steep_density_ratio`: street length per km² on cells with landform slope > 10%, divided by the tile's street
   length per km² (tiles with ≥ 10% such cells only);
5. `circuity`: median network / straight-line distance between 50 random node pairs 200–400 m apart.

**Water and infrastructure (reported separately; few water tiles):**
* unbridged street–water crossings per km of river;
* bridges per km of river;
* bridge spans;
* crossing angles;
* viaducts per tile, with span, clearance and approach grades.

**Network quality:**
* components;
* dead-end ratio;
* junction angle deviation;
* street length per km²;
* class continuity (share of degree-2 nodes whose two edges share a class).

**Failure measures:**
* premature termination: generated street length < 50% of the real tile's;
* empty area: share of 64 m cells farther than 100 m from any street, real vs generated;
* bridges per km² relative to real;
* replanning attempts per step;
* failed proposals;
* circuity > 2.

## 4. Regimes

* **A, conditional real terrain (primary for terrain claims).** Every development tile with a DEM (`val` with a
  DEM ∪ `val_t`) and every model generates 4 samples per training seed on the tile's real terrain and water. Per
  tile, the metric is the mean over samples. Labelled CONDITIONAL. No real street of the tile is given.
* **B, unconditional.** 256 tiles at sampling seed 123, as M10, for S10.
  * Terrain models receive terrain from TRAINING tiles drawn at random (seeded), with a random symmetry.
  * Terrain-blind models receive nothing.
* **C, stress tests (exploratory).** The 10 procedural scenarios × 5 seeds; violation rates, street length,
  bridges against river width, failures. Synthetic performance is not evidence of real-world generalisation.

## 5. Heuristic replanning (HR), fixed before results

When a proposed step is not `buildable`:
1. up to 4 redraws from the SAME learned step distribution;
2. halve the step;
3. rotate toward the contour (± the direction perpendicular to the terrain gradient nearest the proposal);
4. for a step ending in water: extend to the far bank when that is within the class span limit (bridge);
5. stop the arm.

At most 8 evaluations per step (bounded; no loops). Every attempt is counted.

## 6. Hypotheses and criteria (both N0 / TN seeds must agree)

* **H1 (terrain adaptation).** In regime A, TN improves S_T over N0 by ≥ 20%, and the 95% paired tile-bootstrap
  interval of ΔS_T (TN − N0) lies below 0.
* **H2 (no realism loss).** In regime B, S10(TN) ≤ S10(N0) + 0.10.
* **H3 (learned vs rule).** In regime A, TN+R improves S_T over R (= N0+R) by ≥ 10%, interval below 0. This tests
  whether terrain inputs matter beyond the rules.
* **H4 (replanning).** TN+HR reduces premature termination relative to TN+R (paired interval below 0) without
  worsening S_T by more than 0.05.
* **H5 (learned bridge decisions).** On development steps that cross water, the structure head's bridge
  probability separates bridged from ground crossings with AUC ≥ 0.8. In generation, the share of water crossings
  that the MODEL marks as bridges (before any rule) is reported against the rule-only share.

Secondary / descriptive:
* the ablations;
* P against all;
* per-city and per-terrain-class (flat / moderate / steep) results;
* resolution sensitivity (grades from 4 m vs native).

## 7. Statistics and reporting

* **Bootstrap.** Paired tile bootstrap, 2,000 resamples, over the regime-A tiles. Seeds are reported separately and
  never pooled into independent samples. Samples of one tile are averaged first (they are not independent).
* **Effect sizes:** relative change of S_T and S10; per-metric W1/IQR.
* **Reporting:** all runs, including failures and collapsed seeds.
* **Coverage limits:** every claim states its tile count. Valleys (11 train tiles), viaducts (0 real) and water
  (2–4 validation tiles) **cannot support quantitative claims**; they are reported as descriptive and stress-test
  results only.

## 8. Changes after pre-registration

1. **2026-11-04, before any model was trained.** H5 needs negatives at water crossings, but untagged ground
   crossings of water are masked as UNKNOWN by design: a missing tag is not evidence of a ground crossing, and fords
   are absent in the data.
   * H5 is therefore computed as the AUC of the bridge probability over all labelled development steps (bridge vs
     non-bridge, OSM default),
   * plus the model's bridge rate on the real bridges over water (recall at water).

   The criterion (AUC ≥ 0.8) is unchanged.
2. **2026-11-04, AFTER the regime A / B results (leakage control).** The example figure showed TN reproducing
   real street curves on held-out terrain. An urban bare-earth DEM still contains the street beds: on real street
   pixels the DEM curvature (|Laplacian|) is 0.53× its off-street value, median over 52 development tiles
   (`outputs/m12/audits/dem_street_imprint.json`). TN can therefore partly read existing streets from the terrain,
   which **confounds H1 and H3**.
   * Added: **TN-lp**, the same model and recipe, whose terrain INPUT is Gaussian low-passed (σ = 16 m), 2 seeds.
     Rules and metrics still use the unfiltered DEM.
   * H1/H2/H3 are re-tested as H1lp/H2lp/H3lp with unchanged criteria.
   * The original H1–H3 verdicts are reported, marked as confounded.
   * The stress tests (regime C, procedural terrain) are free of this leakage by construction.
3. **2026-11-04, after regime C (scenario bug).** In `narrow_river`, `wide_river` and `deep_valley`, the valley
   axis lay on the tile's western EDGE (offset 0 for an axis at 90°), not across the tile as intended.
   * Fixed: offset −256 m, so the axis is at x = 256.
   * Regime C is rerun for every model; the earlier regime C results are superseded and not reported.
4. **2026-11-04, exploratory (not part of any hypothesis).** **HRV** = HR plus a dry-valley "far-rim" extension
   tried first and built as a viaduct. It was added because HR, as pre-registered, never spans a valley.
