# Milestone 7: Constraints and relative building patterns

## Answer in brief

* **Constraints: solved.** In 36,730 generated buildings (M7 seed 0) on real validation streets:
  * 0 lie on a road (> 1 m²);
  * 0 overlapping building pairs;
  * 0 invalid polygons;
  * 0 footprints with an angle < 45°.

  Milestone 6 had 2,810 buildings on roads, 4,505 overlapping pairs and 17 % acute footprints.
  The 9 realism metrics of M6 stay within range (S3 ✅).
* **Relative patterns: partly learned.** The frontage model (rows along each street side) moves
  all 4 pattern metrics towards real:

  | Metric | Real | M6 | M7 |
  |---|---|---|---|
  | Within 20 m of a street | 0.68 | 0.59 | 0.75 |
  | Building-line irregularity (setback std) | 3.5 m | 4.8 m | 4.3 m |
  | Rhythm irregularity (spacing CV) | 0.34 | 0.56 | 0.49–0.52 |
  | Neighbour alignment (within 5°) | **0.90** | 0.49 | **0.54** |

  Three of these are within 1 IQR of real. **Neighbour alignment is still far from real.**
* **Pre-registered decision: not met** (S2 fails on neighbour alignment, both seeds and both
  variants).
* **Second variant M7b** (each building predicted as a change from the previous one): did **not**
  improve alignment (0.52–0.53).
* **Diagnosis:**
  * Along a frontage, the model's step-to-step orientation matches real data (83–87 % within 5° vs
    84 % real, teacher-forced).
  * The orientation drawn equals the one intended (median error 0.02°).
  * Trimming rarely rotates buildings (0.9 % by > 2°).
  * The misalignment arises in **free generation**: 31 % of frontage buildings are trimmed and many
    are rejected, so rows are broken. Gaps are filled by M6 interior buildings, which know nothing
    about their neighbours' orientation.


**The request:**
1. nothing may lie on top of anything else (e.g. houses on roads);
2. buildings should follow the patterns of their neighbours ("irregular, but not chaos");
3. very acute angles in buildings should be strongly discouraged.

Milestones 5–6 stay unchanged and reproducible; everything new is opt-in (`strict=True`, new modules).

## 1. Evidence from real tiles (training data, before any modelling)

| Property (real buildings) | Spain | US | Japan |
|---|---|---|---|
| Aligned within 5° (mod 90°) of the nearest neighbour | 78 % | 92 % | 89 % |
| Within 20 m of a street (polygon to centreline) | 71 % | 67 % | 82 % |
| Coefficient of variation of nearest-neighbour distance | 0.63 | 0.69 | 0.54 |
| Footprints with an interior angle < 45° (raw OSM outlines, < 0.5 m detail removed) | ≈ 0.1 % | ≈ 0 % | ≈ 0 % |
| … after our 20-ray encoding | 7.3 % | 2.2 % | 2.2 % |

Milestone 6 generated footprints (P2): **18.8 % have an angle < 45°**, 35 % < 60°. The acute
angles come from sampling the 20 shape values independently and from the encoding; they are not in
real buildings.

So buildings are strongly **ordered relative to each other and to the street**, and acute corners
are essentially absent.

## 2. Literature (verified; added to `HYBRID_GENERATION_LITERATURE.md`)

* **Zhang, Ai, Stoter, Kraak & Molenaar (2013), *GeoInformatica* 17(1)**: recognises collinear
  and curvilinear building alignments in topographic maps. The related typology covers collinear,
  curvilinear, align-along-road, grid-like and unstructured patterns. This gives us the
  **vocabulary** and the **metrics**: neighbour alignment, alignment along roads.
* **Strauss (1975), *Biometrika* 62**: pairwise-interaction point processes. Objects closer than r
  inhibit each other; a hard core forbids overlap. The statistical form of
  "irregular but not random": what matters is **relative** position, not absolute.
* **GlobalMapper (He & Aliaga, ICCV 2023), BlockPlanner (Xu et al., ICCV 2021)**: buildings are
  generated in **rows / rings along block boundaries**, each relative to its neighbours.
* **Para et al. (ICCV 2021), House-GAN (Nauata et al., ECCV 2020)**: relations between elements
  (adjacency, alignment) as explicit graph constraints.

**What we adopt:**
* **The GlobalMapper / BlockPlanner idea**, simplified to street **frontages**: one sequence per
  street side, each building predicted from the previous two.
* **A Strauss-like hard core** as a programmed rule: no overlap.
* **No novelty claimed.**

## 3. What changed

| Part | Learned | Programmed |
|---|---|---|
| **Frontage model** (`layout/frontage*.py`, `models/frontage_net.py`, 0.64 M params) | per street side, in order along the street: STOP, or the next building's **gap** from the previous one and **setback** (joint mixture), **orientation relative to the street**, size, shape, conditioned on a street-aligned crop and on the **previous two buildings** (rhythm, building line) | street links and frontages; assignment of real buildings (≤ 20 m from the street); order of frontages (major streets first, longest first) |
| **Interior buildings + green** | the Milestone 6 placer P2 (unchanged weights), continuing from the frontage layout | — |
| **Strict rules** (`place.py`, `strict=True`) | — | buildings are **trimmed to the road edge** (rejected if > 20 % is cut); building–building overlap ≤ 1 % (touching allowed); green trimmed to free land, ≤ 1 % overlap; **interior angle ≥ 45°** |

**Why a frontage model and not just more inputs to P2?**
* P2 chooses positions with a heat map over the whole tile, in random order. A row of houses is
  then a coincidence of independent choices.
* Along a frontage, "15 m after the previous house, same setback, same angle" is a single,
  easily learned step. This is the mechanism the literature uses (rows / rings).

**Honest limitations of this design:**
* The interior placer is the M6 model. It was trained to place *all* buildings in random order,
  not "the rest after frontages" (a mismatch).
* The frontage order is programmed.
* A building that the rules reject still advances the row position.

## 4. Evaluation and decision criteria (fixed before generating any Milestone 7 samples)

**Set-up:** buildings and green on the **real validation street networks** (19 tiles × 4 samples
per method), against the real validation buildings (raw OSM polygons).

**Methods:**
* M6 P2 (M6 rules);
* M6 P2 + strict rules (isolates the effect of the rules);
* **M7** = frontage model + P2 interior + strict rules.

Seeds 0 / 1 are paired (frontage seed i with P2 seed i). The full pipeline (roads v2 → M7) is shown
for illustration.

**Metrics:**
* the 9 building / green metrics of Milestone 6;
* pattern metrics (`frontage.pattern_metrics`):
  * `nn_aligned_5deg`;
  * `frontage_share`;
  * `setback_std_m` (regularity of the building line);
  * `spacing_cv` (regularity of the rhythm);
  * `acute_share`;
* validity: buildings with > 1 m² on the road surface, building pairs overlapping > 1 %, invalid
  polygons.

**Criteria (all must hold for both seeds):**

| | Criterion |
|---|---|
| S1 constraints | 0 buildings with > 1 m² on the road surface, 0 overlapping pairs (> 1 %), 0 invalid polygons, acute share ≤ 1 % |
| S2 patterns | W1/IQR to real ≤ 1 for all 4 pattern metrics **and** M7 closer to real than M6 + strict rules on ≥ 3 of the 4 |
| S3 no regression | ≥ 7 of the 9 M6 building / green metrics within 1 IQR; nearest-training-tile IoU ≤ real + 0.10 |
| (S4, reported) | the country ranking criterion B4 of M6 (expected to still fail: no country input) |

## 5. Results (`outputs/m7/results.json`; real validation streets; 76 layouts per method; medians)

### 5.1 Training

| Model | Best epoch / run | Validation NLL per step | Note |
|---|---|---|---|
| M7 frontage seed 0 | 40 / 40 | −3.24 | |
| M7 frontage seed 1 | 39 / 40 | 6.76 | the first run early-stopped at epoch 12 during a transient spike of the shape term; it was retrained for all 40 epochs (logs of both runs kept) |
| M7b relative seed 0 / 1 | 32 / 40, 39 / 40 | 6.76 / 0.67 | |

* Each seed takes about 11 minutes on MPS.
* The seeds differ almost only in the **shape** term (sharp vs broad shape modes). Gap/setback,
  angle, size and stop agree within a few hundredths.

### 5.2 Constraints (counts over all 76 layouts; seed 0)

| | Buildings | On road (> 1 m²) | Overlapping pairs (> 1 %) | Acute (< 45°) | Invalid |
|---|---|---|---|---|---|
| Real (raw OSM, class-width roads) | — | 3.9 % of buildings per tile* | 0 | 1.6 % (20-ray encoded) / ≈ 0.1 % (raw) | 0 |
| M6 P2 | 33,386 | 2,810 | 4,505 | 17.4 % | 0 |
| M6 P2 + strict rules | 31,478 | **0** | **0** | **0 %** | 0 |
| M7 frontage | 36,730 | **0** | **0** | **0 %** | 0 |
| M7b relative | 35,973 | **0** | **0** | **0 %** | 0 |

\* Real buildings "on the road" are an artefact of drawing every street with its class's median
width.

**A bug was found and fixed during evaluation.** A building was only trimmed if it lost > 0.1 % of
its area, so 1 building (1.5 m² on the road) slipped through in the first run. Trimming now
happens on any overlap; test added. The first-run results are kept in
`results_v1_before_fix.json`.

**How often the rules fired (M7 seed 0):**

| Accepted | Trimmed at the road edge | Rejected: overlap | Rejected: acute | Rejected: on road |
|---|---|---|---|---|
| 36,730 | 10,928 | 58,050 | 17,113 | 11,765 |

The rules do a lot of work: the network still proposes many invalid buildings.

### 5.3 Patterns and realism (W1/IQR to real in brackets)

| | Neighbour aligned < 5° | Within 20 m of street | Setback std (m) | Spacing CV | Count | Coverage | Median area | Clark–Evans |
|---|---|---|---|---|---|---|---|---|
| Real | 0.90 | 0.68 | 3.5 | 0.34 | 462 | 0.27 | 105 m² | 1.03 |
| M6 P2 | 0.49 (2.21) | 0.59 (0.24) | 4.8 (0.71) | 0.56 (0.62) | 370 | 0.17 | 67 m² | 1.07 |
| M6 P2 + strict | 0.49 (2.21) | 0.60 (0.27) | 4.8 (0.72) | 0.54 (0.51) | 360 | 0.15 | 65 m² | 1.08 |
| **M7 s0 / s1** | 0.54 / 0.54 (1.85 / 1.82) | 0.75 / 0.75 (0.20 / 0.21) | 4.3 / 4.3 (0.55 / 0.58) | 0.49 / 0.52 (0.35 / 0.42) | 426 / 420 | 0.20 / 0.18 | 69 / 66 m² | 1.07 / 1.03 |
| **M7b s0 / s1** | 0.53 / 0.52 (1.83 / 1.93) | 0.73 / 0.76 (0.16 / 0.24) | 4.6 / 4.4 (0.61 / 0.59) | 0.53 / 0.50 (0.44 / 0.43) | 433 / 396 | 0.21 / 0.18 | 68 / 64 m² | 1.07 / 1.03 |

* **Diversity:** 0.92–0.93 (real 0.91).
* **Nearest-train IoU:** 0.14–0.16 (real 0.19). No memorisation.

**Per country (real → M7b seed 0):**

| | Neighbour aligned | Setback std | Spacing CV | Count | Median area |
|---|---|---|---|---|---|
| Spain | 0.81 → 0.53 | 3.7 → 4.3 m | 0.31 → 0.53 | 89 → 306 | 257 → 68 m² |
| US | 0.96 → 0.60 | 1.8 → 5.0 m | 0.29 → 0.53 | 507 → 510 | 104 → 70 m² |
| Japan | 0.85 → 0.51 | 4.2 → 4.0 m | 0.53 → 0.46 | 656 → 438 | 67 → 62 m² |

US suburbs (very regular rows: setback std 1.8 m) are the furthest from real; Japan is closest.

### 5.4 Decision

| | S1 constraints | S2 patterns (within 1 IQR; closer than M6 + strict) | S3 no regression | S4 country | Success |
|---|---|---|---|---|---|
| M7 seed 0 | ✅ | ❌ (3 / 4 within; 4 / 4 closer) | ✅ 9 / 9 | ❌ | **no** |
| M7 seed 1 | ✅ | ❌ (3 / 4; 3 / 4) | ✅ 9 / 9 | ❌ | **no** |
| M7b seed 0 | ✅ | ❌ (3 / 4; 4 / 4) | ✅ 9 / 9 | ❌ | **no** |
| M7b seed 1 | ✅ | ❌ (3 / 4; 3 / 4) | ✅ 9 / 9 | ❌ | **no** |

**Note on the M7b decision.** M7b was added *after* seeing M7's results, against the same
criteria and validation tiles. Its result is therefore a second look at the validation set, not an
independent confirmation.

### 5.5 Diagnosis of the alignment gap

| Test | Real | Generated |
|---|---|---|
| Orientation change between consecutive frontage buildings, mod 90°, share < 5° (teacher-forced) | 84 % | 83–87 % |
| Same, absolute (real neighbours) | 88 %, median 0.77° | — |
| Drawn vs intended orientation (shape noise) | — | median 0.02° |
| Buildings re-oriented > 2° by trimming | — | 0.9 % |
| Frontage buildings trimmed at the road edge | — | 31 % |
| Nearest-neighbour aligned, frontage buildings only (M7) | 87 % | 61 % |

**Reading:**
* The **learned relation is right**.
* The gap appears when rows are **built and edited** in free generation:
  * rejections break rows;
  * interior buildings from the M6 placer have no notion of neighbour orientation;
  * small deviations propagate along the row.

  This is the same exposure-bias pattern as the roads (M6).

## 6. Next steps (proposed, not run)

1. **Give the interior placer the neighbours.** Feed it the orientations and offsets of the k
   nearest placed buildings, and make its orientation relative to the nearest building (as in
   the frontage model). This targets the measured failure directly.
2. **Make rejection part of learning.** Train on the model's own rollouts, where rows have gaps
   (DAgger / scheduled sampling). Or propose a few candidates and pick the one most consistent with
   the neighbours. That is still learned, because the score would be the model's own likelihood.
3. **Country + density conditioning** (still open from M6, S4), then the dataset expansion.

## 7. Reproduce

```
python scripts/m7_train_frontage.py --build && python scripts/m7_train_frontage.py --build --relative
python scripts/m7_train_frontage.py --seed 0 && python scripts/m7_train_frontage.py --seed 1
python scripts/m7_train_frontage.py --relative --seed 0 && python scripts/m7_train_frontage.py --relative --seed 1
python scripts/m7_evaluate.py          # ~13 min
pytest tests/test_layout.py
```

Figures:
* `outputs/m7/zoom_patterns.png`
* `outputs/m7/layouts_on_real_roads.png`
* `outputs/m7/pattern_metrics.png`
* `outputs/m7/full_pipeline.png`

Exports: `outputs/m7/export/*.gpkg`.

