# Milestone 8: Parcel-aware neural-procedural generation

Research plan, literature and Phase 1 diagnosis: `docs/MILESTONE_8_PLAN.md`,
`docs/PARCEL_AWARE_GENERATION_LITERATURE.md`.

**Approved:**
* run order A → B1 → B2 → C → D, ≥ 2 seeds per learned model;
* observed parcels from Spain and the US only; Japan kept separate;
* inferred parcels never treated as ground truth.

**Splits are frozen:** `tests/test_split_integrity.py` stores SHA-256 fingerprints of the
train / val / test tile IDs (565 / 19 / 19). Parcels are fetched only for train and val tiles.

<!-- RESULTS-BRIEF -->

## 1. Data provenance (observed parcels)

| Source | Our tiles | Licence handling | Status |
|---|---|---|---|
| Spain: Dirección General del Catastro, INSPIRE CadastralParcels (Barcelona 08900, Madrid 28900, Valencia 46900; GML, ETRS89 / UTM) | 146 train + val tiles; 131 with > 50 % of the area covered by parcels (median 0.70; the rest is mostly public streets) | licence v1.0 (2016): transformed use allowed, raw data not redistributable, not to be called cadastral cartography; raw files in `data/raw/parcels/` (git-ignored); provenance JSON (URL, time, SHA-256, licence) per file | fetched (73 MB) |
| US: NYC DCP MapPLUTO tax lots (ArcGIS service, per-tile queries) | 73 NYC train + val tiles | free public data; cited | fetched |
| US: LA County Assessor Parcel Boundaries (ArcGIS service, per-tile queries) | LA train + val tiles | County GIS terms grant copy / adapt with citation, but the service notes records are for sale, so treated conservatively: local research use, derived statistics only | fetching |
| US: Chicago (Cook County) | — | **excluded**: licence unspecified / personal use | — |
| Japan | — | kept separate (registry maps mix arbitrary local coordinates; coverage unknown) | — |

**Labels used everywhere:** *observed* (cadastre), *inferred* (algorithm; not used in M8 learning),
*generated* (model).

## 1b. Observed parcels: what real subdivisions look like (train + val tiles; `outputs/m8/parcels/`)

Measured with the programmed rules of `layout/parcels.py`:
* frontage = outline facing a street (not shared with another parcel, ≤ 12 m from a centreline);
* setbacks only for parcels with exactly one building;
* shape statistics only for parcels not cut by the tile edge.

These are **observed** parcels; nothing here is inferred.

| City | Tiles | Parcels | Area p10 / p50 / p90 (m²) | Frontage (m) | Depth (m) | No street frontage | Parcels with 0 / 1 / ≥ 2 buildings | Coverage of built parcels | Front setback (m) | Side setback (m) | Rear setback (m) | Buildings spanning > 1 parcel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Barcelona | 41 | 11,592 | 84.77 / 242.05 / 949.82 | 4.71 / 11.24 / 41.66 | 9.53 / 21.48 / 39.32 | 0.12 | 0.21 / 0.72 / 0.08 | 0.43 / 0.88 / 1 | 0 / 0 / 1.67 | 0 / 0 / 0 | 0 / 0 / 4.27 | 0.09 |
| Madrid | 72 | 11,966 | 78.85 / 237.3 / 1372.13 | 5.5 / 16.38 / 58.86 | 8.01 / 17.83 / 40.47 | 0.19 | 0.28 / 0.66 / 0.06 | 0.36 / 0.82 / 0.99 | 0 / 0 / 4.02 | 0 / 0 / 0.13 | 0 / 0 / 6.29 | 0.16 |
| Valencia | 33 | 6,814 | 90.21 / 255.32 / 1108.12 | 5.55 / 14.09 / 51.68 | 8.34 / 18.05 / 33.48 | 0.17 | 0.32 / 0.64 / 0.04 | 0.6 / 0.9 / 0.99 | 0 / 0 / 0.62 | 0 / 0 / 0 | 0 / 0 / 0.87 | 0.31 |
| New York | 73 | 27,227 | 159.41 / 240.91 / 559.51 | 4.66 / 7.14 / 35.75 | 15.78 / 30.56 / 33.62 | 0.10 | 0.04 / 0.73 / 0.23 | 0.26 / 0.4 / 0.66 | 0.4 / 3.77 / 7.57 | 0 / 0 / 1.76 | 0.44 / 8.67 / 14.38 | 0.33 |
| Los Angeles | 73 | 18,547 | 449.24 / 647.3 / 1674.83 | 11.19 / 15.3 / 48.08 | 18.29 / 39.04 / 49.93 | 0.24 | 0.14 / 0.42 / 0.44 | 0.24 / 0.38 / 0.57 | 0.55 / 5.44 / 9.41 | 0.04 / 0.82 / 2.77 | 0.55 / 7.41 / 18.81 | 0.02 |

**Reading:**

* **Spain: a fully built pattern.** The median coverage of built parcels is 0.82–0.90, and the
  median front, side and rear setbacks are **0 m**: buildings fill their parcels and share walls.
  This is the medium-density European block.
* **US: detached houses on regular lots.**
  * NYC lots: 7 m frontage × 31 m depth. LA lots: 15 × 39 m.
  * Median front setbacks 3.8 m (NYC) and 5.4 m (LA); rear setbacks 7–9 m.
  * 22–44 % of parcels hold ≥ 2 buildings (garages, rear units).
* **The two countries are opposite** on coverage and setbacks. A universal setback rule would be
  wrong for both.
* **Buildings spanning several parcels:**
  * 33 % in NYC and 31 % in Valencia: OSM often maps a row of attached houses as one building.
  * 16 % in Madrid; 2 % in LA.

  So the OSM-building ↔ parcel relation is **not** one-to-one, and parcel-conditioned generation
  must allow it.
* **Caveat:** Spanish OSM buildings are often traced from cadastral footprints, which is one reason
  for the near-zero setbacks. The setback metric is a programmed approximation; see the rules above.

## 2. Pre-registered success criteria (written before implementing B1)

**Set-up:** all experiments place buildings on the **real validation street networks** (19 tiles ×
4 samples per method, sampling seed 21), against real OSM buildings of the same tiles. They are
compared with **A = Milestone 7** (frontage seed s + P2 seed s, strict rules) re-run with identical
settings.

### Primary realism set (per-tile metrics; W1 to real in units of the real IQR, lower is better)

| # | Metric | Meaning |
|---|---|---|
| P1 | `nn_aligned_5deg` | share of buildings aligned within 5° (mod 90°) with their nearest neighbour |
| P2 | `spacing_cv` | regularity of the rhythm along frontages |
| P3 | `setback_std_m` | regularity of the building line |
| P4 | `frontage_share` | share of buildings within 20 m of a street |
| P5 | `frontage_size_similarity` | share of consecutive frontage pairs within a factor of 1.5 in area (real: 46–71 %) |
| P6 | pooled log-area distribution | W1 between the generated and real per-building log-area distributions (IQR units) |

### Criteria

| Criterion | Rule |
|---|---|
| **Improvement over the predecessor** (B1 vs A, B2 vs B1) | lower distance to real on **≥ 4 of the 6** primary metrics, for **both** seeds |
| **Mechanism (B1 only)** | ≥ 75 % of proposed frontage buildings accepted (A: 51 %) **and** fallback share ≤ 50 % of final buildings (A: 73 %), both seeds |
| G1 validity | 0 buildings with > 1 m² on the road surface, 0 overlapping building pairs (> 1 %), 0 invalid polygons, 0 footprints with an angle < 45° |
| G2 no regression | ≥ 7 of the 9 M6 building / green metrics within 1 IQR of real |
| G3 diversity | 3-channel diversity ≥ real − 0.05 |
| G4 memorisation | median nearest-train IoU ≤ real + 0.10 |
| Reported, no threshold | per-country distances on P1–P6 (Spain / US / Japan separately); acceptance rates; rule firings; shrink-to-fit counts |

An experiment **succeeds** if it improves on its predecessor and satisfies G1–G4 (B1 also the
mechanism criterion). Otherwise the result is reported as negative and the next step is
reconsidered before adding complexity.

**Parcel experiments C / D** add parcel-specific criteria (parcel area / frontage / depth and
front / side / rear setbacks and coverage vs **observed** parcels on Spanish and US validation
tiles). They will be fixed here **before** C is implemented, after B1 / B2 are reported.

## 3. Experiment B1: edge-parameterised frontage (no parcels)

**What changed vs A (Milestone 7).** Only the frontage step: what it predicts, and the correction.
The network, data, budget and fallback (M6 P2) are the same.

* **Learned from real data:**
  * a **signed** edge-to-edge gap (asinh scale), so attached (0), spaced (> 0) and staggered (< 0)
    buildings are all possible;
  * the front distance from the street centreline;
  * size and orientation;
  * a "plain rectangle" flag (46 % of real frontage buildings), with the 20-ray shape otherwise.
* **Programmed:**
  * placement by edges;
  * **shrink-to-fit** (0.9 / 0.8 / 0.7 about the centroid) before rejection.
* **Why the gap is signed:** in real data, 31–78 % of consecutive frontage pairs have
  *overlapping* along-street extents (staggered at different depths). Forcing gaps ≥ 0 would have
  been an arbitrary constraint.
* **Training:** best validation epochs 25 / 24 (seeds 0 / 1), about 11 min each.

**Results** (`outputs/m8/eval/results_A_B1.json`; W1/IQR to real validation, lower is better;
medians in brackets):

| | Accepted | From fallback | Aligned < 5° | Spacing CV | Setback std | Frontage share | Size similarity | Log-area distribution |
|---|---|---|---|---|---|---|---|---|
| Real | — | ≈ 0.2 | (0.90) | (0.34) | (3.5 m) | (0.68) | (0.45) | — |
| A seed 0 | 0.52 | 0.73 | 1.85 (0.54) | 0.35 (0.49) | 0.55 (4.3) | 0.20 (0.75) | 0.94 (0.31) | 0.16 |
| A seed 1 | 0.54 | 0.70 | 1.82 (0.54) | 0.42 (0.52) | 0.58 (4.3) | 0.21 (0.75) | 0.93 (0.31) | 0.21 |
| B1 seed 0 | 0.57 | 0.75 | 1.80 (0.55) | 0.45 (0.52) | 0.66 (4.7) | 0.16 (0.74) | 0.97 (0.30) | 0.16 |
| B1 seed 1 | 0.61 | 0.58 | 1.77 (0.55) | 0.50 (0.52) | 0.59 (4.3) | 0.29 (0.81) | 0.92 (0.32) | 0.25 |

**Guardrails, both seeds:**

* **G1 validity:** 0 buildings on the road, 0 overlaps, 0 invalid, 0 acute.
* **G2 no regression:** 9 / 9 within 1 IQR.
* **G3 diversity:** 0.92.
* **G4 memorisation:** ✅.

**Decision:** **not met.**

* Improvement on 3 / 6 (seed 0) and 2 / 6 (seed 1) of the primary metrics; ≥ 4 required.
* The mechanism criterion fails: acceptance is 0.57 / 0.61, against ≥ 0.75 required.

**Diagnosis** (instrumented B1 frontage phase, seed 0, validation streets; rule counts include
shrink attempts):

| Outcome | Share |
|---|---|
| Accepted | 56.7 % |
| Rejected: acute angle | **21.4 %** |
| Rejected: overlap (58 % with the previous building on the frontage) | 12.0 % |
| Rejected: on road | 6.4 % |
| Rejected: too small | 3.4 % |

* **Acute rejections come from the shape sampler itself:** 1,892 of 1,933 acute rejections had an
  acute angle **before** any trimming. The rectangle flag removed part of the problem, but sampled
  non-rectangular shapes (20 independent ray values) remain spiky.
* **Overlaps fell** (18.5 % → 12 %) but remain. The model reproduces real staggering (38 % of
  sampled gaps < 0, as in the data), but its setback is not always large enough to put a staggered
  building behind its neighbour.

**What B1 shows:**

* Geometric incoherence of the *gap* was only part of the problem.
* The **shape representation** is now the largest single source of rejections, for M7 and B1 alike.
* This is the main candidate for a next fix. It is *not* parcel-related, and it is not added
  silently: it is reported here first.

## 4. Experiment B2: neighbour-aware fallback (no parcels)

**What changed vs B1:** only the fallback (the M6 placer that fills the rest of the tile after the
frontage phase). B1's frontage model is reused unchanged.

* The attribute head now also sees the 8 nearest already-placed buildings within 60 m (offset
  along / across the street, orientation, log size, distance; a set encoder with sum pooling).
* It is trained for its real role: the tile's real frontage buildings are already placed, and it
  learns only the remaining (interior) buildings and green spaces.
* Code: `scripts/m8_train_b2.py`, `placer_data.tile_samples_b2`, `ObjectPlacer(neighbours=8)`.
  Checkpoints: `checkpoints/m8/fallback_nb_s{0,1}`. Evaluation:
  `outputs/m8/eval/results_A_B1_B2.json`, `zoom_A_B1_B2.png`.

**Training:** 30 epochs per seed (about 48 min each).

* Seed 1 trained normally: best validation NLL 12.25 per object (epoch 29).
* **Seed 0 diverged at epoch 17** (loss 16 → 50, never recovered). Its checkpoint is the best
  epoch before that, epoch 8 (NLL 14.31), so seed 0 is **undertrained**. The same recipe was
  unstable in M6 (that placer spiked to 51 at epoch 7 and recovered), so the cause is the shared
  recipe (mixture likelihood, learning rate 5e-4), not the neighbour inputs (all scaled to about ±2).
  Seed 0 was **not** retrained; it is reported as it is.

**Results** (W1 to real, IQR units, lower is better):

| | P1 align | P2 spacing CV | P3 setback std | P4 frontage share | P5 size sim. | P6 log area | buildings |
|---|---|---|---|---|---|---|---|
| B1 seed 0 | **1.80** | 0.45 | 0.66 | 0.16 | 0.97 | 0.16 | 40,348 |
| B1 seed 1 | **1.77** | 0.50 | 0.59 | 0.29 | 0.92 | 0.25 | 39,405 |
| B2 seed 0 | 2.18 | **0.16** | **0.35** | **0.16** | **0.72** | **0.15** | 16,623 |
| B2 seed 1 | 1.93 | **0.31** | **0.40** | **0.23** | **0.77** | **0.20** | 22,078 |

(P4, seed 0: 0.157 vs 0.164 before rounding.)

| Criterion | B2 seed 0 | B2 seed 1 |
|---|---|---|
| Better than B1 on ≥ 4 of 6 | yes, 5 of 6 (not alignment) | yes, 5 of 6 (not alignment) |
| G1 validity | pass (0 / 0 / 0 / 0) | pass |
| G2 (≥ 7 of 9 M6 metrics within 1 IQR) | pass, 8 of 9 (coverage 1.38) | pass, 8 of 9 (coverage 1.22) |
| G3 diversity (real − 0.05 = 0.87) | pass (0.938) | pass (0.925) |
| G4 memorisation | pass (0.144) | pass (0.146) |
| **Pre-registered decision** | **success** | **success** |

Medians (real → B2 seed 0 / seed 1):

* spacing CV 0.34 → 0.40 / 0.47 (B1: 0.52);
* setback std 3.5 m → 3.4 / 3.7 m (B1: 4.3–4.7);
* frontage size similarity 0.45 → 0.34 / 0.35 (B1: 0.30–0.32);
* alignment 0.90 → 0.52 / 0.52 (B1: 0.55).

Per country, P2 spacing improves mostly in the US (1.2–1.4 → 0.4–0.9). Spain's spacing remains far
from real (2.6–3.3); alignment is worse than B1 in all three countries.

**Important caveat, which the pre-registered criteria did not catch: B2 builds far fewer buildings.**

* B2 produces 41–56 % of B1's building count (median per tile: 120–316 vs about 460 real).
  Building coverage is the G2 metric that fails.
* Building count still passes G2 (0.41–0.53 IQR), because real counts vary a lot between tiles.
* Likely cause, a train / generation mismatch:
  * in training, the fallback sees **all** real frontage buildings (about two-thirds of real buildings are within 20 m of a street);
  * in generation, B1's frontage phase places far fewer (about 57 % acceptance);
  * the fallback learned to place only interior buildings, so it leaves the missing street fronts
    empty.
* Part of B2's improvement on P2 / P3 / P5 may therefore come from **fewer** buildings (less
  fallback clutter near streets) rather than better placement.

**Verdict:** B2 meets its pre-registered criterion with both seeds, but it is **not a clean
improvement**:

* density is about halved;
* alignment, the largest remaining gap to real (0.52 vs 0.90), got worse;
* seed 0 is undertrained.

Candidate fixes, none applied, reported first:

1. train the fallback with a random subset of real frontage buildings removed, matching the
   generated acceptance rate;
2. raise frontage acceptance first, by fixing the shape representation (§3);
3. retrain B2 seed 0 at a lower learning rate (a stated deviation).

<!-- RESULTS -->
