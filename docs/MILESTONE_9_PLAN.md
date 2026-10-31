# Milestone 9 plan: tidy subdivision, green zoning, standard building shapes

**Written 2026-10-31, before implementing anything in M9.** The success criteria in §4 are fixed now
and will be applied unchanged. All measurements below use TRAIN tiles, except the shape-fit check,
which uses the same 4000 validation footprints as the M8 library evaluation. The test split is unused.

## 1. Why: the M8 failures this milestone targets

M8 C / D (docs/MILESTONE_8.md §6) failed. The figures showed:

* irregular lot widths along a street;
* leftover space and oversized interior parcels in the middle of blocks, each filled by one
  block-sized building (building coverage 0.37 vs 0.29 real);
* blob-like green spaces placed by the fallback network, unrelated to parcels;
* buildings clipped to their parcel, so 70 % of them lost their library shape.

User design requests (2026-10-31):

* tidy rows of similar rectangular lots, with no awkward gaps;
* some parcels zoned as green space;
* a small library of standard shapes (rectangle, L, T, U, cross …);
* no empty courtyards: large leftovers become green.

## 2. Evidence gathered before planning

### 2.1 Real subdivision

Source: `scripts/m9_subdivision_analysis.py` → `outputs/m9/analysis/subdivision.json`.
Scope: closed blocks only; observed parcels.

| | Spain | US |
|---|---|---|
| Closed blocks | 1,428 | 646 |
| Block type share: rows (≥ 4) / few (2–3) / single / none | 53 / 9 / 12 / 26 % | 72 / 6 / 7 / 16 % |
| Area share of rows blocks | 73 % | 89 % |
| Median area of "none" blocks | 734 m² | 818 m² |
| Lots per street side (median) | 4 | 7 |
| Lots within ±15 % of their side's median width (random lots: 33 / 29 %) | 50 % | 71 % |
| Width robust spread IQR / median, same side (random: 0.79 / 0.74) | 0.52 | 0.19 |
| Lot depth / block depth, median (0.5 = back-to-back) | 0.36 | 0.42 |

Notes:

* "None" blocks are mostly small slivers between roads.
* US parcel lines sit about 5 m behind the modelled road surface (the sidewalk). That gap explains
  the 0.42 depth ratio, and part of the 20 % of block area outside parcels.

### 2.2 Standard shapes

Source: `scripts/m9_checks.py` → `outputs/m9/analysis/template_fit.json` and `templates.png`.

The library has 7 parametric types: rectangle, L, T / wing, Z, U, H and cross. Cut sizes are 0.15, 0.25,
0.35 or 0.45 of the side, in every orientation. A shape is always stretched to the building's own
length and width.

| Library | Validation footprints with IoU ≥ 0.9 | Median IoU |
|---|---|---|
| Rectangle only | 68.0 % | 0.969 |
| **7 standard types** | **82.7 %** | 0.978 |
| M8 100-shape library | 87.5 % (M8 reported 87.6 %) | 0.980 |

Best-fitting type per footprint: rectangle 52 %, L 33 %, T / wing 10 %, Z 3 %, U 1 %, cross 1 %, H 0.4 %.

### 2.3 Green versus parcels

Source: `outputs/m9/analysis/green_parcels.json`.

| | Spain | US |
|---|---|---|
| Real green area inside observed parcels | 77 % | 84 % |
| … inside parcels that are ≥ 50 % green ("green parcels") | 68 % | 75 % |
| Share of parcels that are green parcels | 3.1 % | 0.8 % |
| Green-parcel area, median (IQR) | 1,135 m² (301–3,315) | 1,252 m² (380–3,839) |

Most real green belongs to parcels that are mostly green, so zoning parcels as green matches the data.

### 2.4 Literature (verified 2026-10-31)

* C. A. Vanegas, T. Kelly, B. Weber, J. Halatsch, D. G. Aliaga, P. Müller, *Procedural Generation of
  Parcels in Urban Modeling*, Computer Graphics Forum 31(2), 2012, doi:10.1111/j.1467-8659.2012.03047.x.
  Procedural block subdivision with style parameters, compared with real parcel data.
* L. Xu, Y. Xiangli, et al., *BlockPlanner: City Block Generation with Vectorized Graph
  Representation*, ICCV 2021. A VAE over a ring-topology graph of land lots with a land-use category
  per lot (NYC-Block dataset).
* L. He, D. Aliaga, *GlobalMapper: Arbitrary-Shaped Urban Layout Generation*, ICCV 2023. Building
  layouts within arbitrary blocks, via skeletonisation and graph attention networks.

M9 is not claimed to be novel. Its subdivision follows the strip / equal-division idea of procedural
parcel generation, with learned parameters.

## 3. Design

M9 is evaluated on real validation street networks; generated-street examples are shown for
illustration. Japan has no parcels and is not part of M9.

### 3.1 BlockNet: per block and per street side (MLP; learned)

1. **Block type** ∈ {rows, few, single, none}.
   * Inputs: country, block area, rectangularity, compactness, number of street sides, street classes.
   * A "none" block becomes **green** if it is ≥ 400 m², otherwise it is left as road verge.
2. **Number of lots N per street side** (categorical 1 … 20) and **depth ratio**.
   * Inputs: block features, side length, street class.
   * The side is cut into N **equal-width** lots, with a small learned jitter (robust spread fitted
     per country).
   * Lots are extruded to the depth ratio × block depth. A ratio above 0.45 is snapped to "meet the
     opposite lots" (back-to-back) when the far side is also a street.
3. **Block interior.**
   * Leftover pieces ≥ 400 m² become green parcels.
   * Smaller pieces are merged into the adjacent lot with the longest shared boundary.
   * A leftover interior parcel therefore never receives buildings.

Training labels: block type, N per side and depth ratio come from the observed train blocks (§2.1
pipeline).

### 3.2 Zoning: land use per parcel (learned)

* Output: a label per parcel ∈ {building, green}.
* Inputs: country, parcel area, interior flag, block type, and the share of green among
  already-assigned neighbours.
* Labels: observed train parcels, green if ≥ 50 % of the parcel is covered by real green.
* A green parcel becomes a green polygon: the parcel inset by 0.5 m.
* The M8 fallback no longer places green in M9-Z.

### 3.3 Standard building shapes (learned choice, stretched, never clipped)

* The lot's building envelope is the lot minus its front, side and rear setbacks, all from
  ParcelNet.
* A network head picks the type (7), cut sizes (4 × 4) and orientation (4), given lot width, depth,
  coverage target and country.
  * Labels: the best-fitting template of each real building inside an observed train parcel.
* The template is stretched to the envelope width × the depth needed for the coverage target, capped
  at the envelope. It is never clipped. If it cannot fit, it is shrunk.

### 3.4 Methods compared

| Method | Subdivision | Green | Buildings |
|---|---|---|---|
| C (M8 baseline, rerun) | M8 rules | M8 fallback | 100-shape library, clipped |
| **T** | BlockNet tidy subdivision, interior ≥ 400 m² → green | M8 fallback + interior green | 7 templates, stretched |
| **TZ** | as T | zoning only (no fallback green) | as T |

There are 2 training seeds per network, 4 samples per tile, and the same 10 validation tiles with
observed parcels (Spain 6, US 4).

## 4. Pre-registered success criteria

### New subdivision metrics

Each metric is a per-block or per-side distribution, compared as W1 / IQR against the observed parcels
of the same tiles, per country:

* **S1** width robust spread per street side (≥ 3 lots);
* **S2** depth ratio per street lot;
* **S3** block-type shares: total variation distance;
* **S4** interior leftover: share of a closed block's area that is neither parcel nor green and lies
  more than 8 m from the block edge. The 8 m margin keeps the US sidewalk gap out of the metric.

### T succeeds if, for both seeds:

1. **Parcel realism:** W1 / IQR ≤ 0.5 on ≥ 4 of the 6 M8 parcel metrics, in **both** Spain and the US.
2. **Tidiness:** S1 and S2 each have W1 / IQR ≤ 0.5 per country; S3 TV ≤ 0.15 per country.
3. **No leftovers:** the median S4 is ≤ the observed median S4 per country. This is the user's
   preference; "no worse than real" is the realism floor.
4. **No giant buildings:** the 99th percentile of building area is ≤ 1.25 × the real 99th percentile
   of the same tiles.
5. **Mechanism:**
   * 0 buildings clipped to their parcel;
   * every building is a template, with 0 acute angles;
   * ≥ 95 % of proposed buildings are accepted.
6. **Buildings:** improves on C on ≥ 4 of the 6 M8 primary metrics.
7. **Guardrails:** G1–G5 as defined in M8.

### TZ succeeds if, for both seeds:

1. Green coverage W1 / IQR ≤ 0.5.
2. **Green shape:** median green-patch rectangularity, W1 / IQR ≤ 0.5 against real green on the same
   tiles.
3. Mean parcel-realism W1 no more than 0.05 worse than T.
4. G1–G5.

### Reported but not gating

* Moran's I of parcel coverage (it failed in M8; reported for continuity).
* Japan.
* Generated-street examples.

## 5. Order, cost, approvals

1. Implement BlockNet, the templates and zoning, with tests. Small MLPs train in minutes on the
   Mac; no large download and no expensive training.
2. Evaluate C, T and TZ. Write the results into a new `docs/MILESTONE_9.md`.
3. Figures: every final figure draws the parcels, with green parcels filled.
4. Report before adding architectural complexity, such as a graph network over lots in the style of
   BlockPlanner.

## 6. M9b pre-registration (written 2026-10-31, after the M9 results, before implementing M9b)

### Design (user request)

1. **Street-access splitting.**
   * Any street-touching parcel larger than 3 × the country's median real street-lot area (Spain
     246 m², US 369 m²; observed train parcels) is split recursively along its minimum rotated
     rectangle. This is the OBB subdivision of Vanegas et al. 2012.
   * A cut is kept only if every piece keeps ≥ 4 m of street frontage; otherwise the other axis is
     tried, and if both fail the piece stays.
   * Splitting stops at ≤ 1.5 × the median area.
   * Pieces deeper than 2 × the median real lot depth (Spain 19.6 m, US 30.8 m) have their back cut
     off at that depth.
2. **Parcels without street access** (frontage < 4 m) become green if they are ≥ 400 m². Smaller ones
   are merged into the adjacent lot with the longest shared boundary. This is the user's rule; the
   real data has 16 % no-frontage parcels, 54–68 % of them built (§M9 answer of 2026-10-31).
3. **Building fit.** The template is stretched inside the largest street-frame rectangle that truly
   fits inside the lot minus its setbacks (0.5 m grid), instead of the lot's outer extent.

Debugging uses train tiles only. **The 10 validation tiles were already seen in the M9 evaluation
figures**, so the M9b evaluation on them is not blind to M9's failure modes.

### Method B (= TZ + the three changes), 2 seeds, same 10 tiles. Success, for both seeds:

1. **Built parcels:** the share of non-green parcels that have ≥ 1 building is within 0.10 of the
   observed share of parcels with a building on the same tiles, per country.
2. **Density:** G5 (median buildings per tile within ±25 % of real, building coverage W1 ≤ 1).
3. **Parcel realism:** ≥ 4 of 6 M8 parcel metrics with W1 / IQR ≤ 0.5, both countries.
4. **Tidiness:** S1 W1 ≤ 0.5 and **S2b** W1 ≤ 0.5 per country.
   * S2b replaces S2 (§M9 results). Block depth is the median over normals at 25, 37.5, 50, 62.5
     and 75 % of the edge; ratios are clipped to [0, 1.5]; W1 is divided by max(IQR, 0.05).
   * S3 TV ≤ 0.15.
5. **No leftovers:** median S4 ≤ observed + 0.001.
6. **No giant buildings:** p99 building area ≤ 1.25 × real.
7. **Mechanism:**
   * 0 clipped, 0 acute;
   * acceptance ≥ 0.95;
   * template not-fitted share ≤ 0.05.
8. **Improves on TZ** (same seed) on ≥ 4 of 6 primary building metrics.
9. **Guardrails:** G1–G4.

Reported, not gating: green coverage and green shape (the user's green rule is expected to raise green
above real), Moran's I, and generated-street examples.
