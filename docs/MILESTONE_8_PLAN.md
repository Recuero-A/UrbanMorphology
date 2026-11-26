# Milestone 8 plan: parcel-aware neural-procedural generation (for approval; nothing trained)

Companions:
* `docs/PARCEL_AWARE_GENERATION_LITERATURE.md` (literature);
* `outputs/m8/analysis/phase1_analysis.json`, `blocks_examples.png` (Phase 1 measurements;
  script `scripts/m8_analysis.py`).

## 1. Phase 1 findings

### 1.1 Why Milestone 7's patterns break (instrumented run on the 19 validation street networks)

| Measurement | Value |
|---|---|
| Frontage buildings proposed / accepted | 4,902 / **51 %** |
| Rejected: overlap | 18.5 %; **77 % of these overlap the previous building on the same frontage** (median overlap 24 %) |
| Rejected: acute angle | 17.1 % |
| Rejected: on road (> 20 % would be cut) | 9.0 % (median 30 % on the road) |
| Trimmed, then rejected by another rule | 3.2 % |
| Accepted buildings that were trimmed | 19 % |
| **Final buildings from the interior fallback placer** | **73 %** (real: ≈ 20 % of buildings are not on a frontage) |
| Neighbour alignment < 5°: frontage buildings / fallback buildings | 0.56 / 0.54 |

**Diagnosis.** The failure is mainly **geometric incoherence**, not missing context:

1. **Overlaps with the predecessor.** The model predicts the gap between building *centres*
   independently of the building's *length along the street*, so consecutive buildings overlap.
2. **Acute corners.** The 20 independent shape values make spiky shapes.
3. **Setbacks too small** for the drawn road width.
4. **Fallback dominates.** Each rejection leaves a hole, and the fallback placer, which has no
   notion of neighbours, then produces most of the final layout. So in M7 the frontage model
   barely shaped the result.

### 1.2 Urban blocks (Stage A analysis, 189 training tiles; `layout/blocks.py`)

Rule: block = tile minus the road surface (road boundaries, not centrelines). It needs no closed
cycles, and a block touching the tile edge is **open** (true size unknown).

| | Spain | US | Japan |
|---|---|---|---|
| Blocks per tile (median) | 23 | 18 | 43 |
| Open (cut by the 512 m window) | 62 % | 74 % | 48 % |
| Closed block area p10 / p50 / p90 (m²) | 452 / 3,224 / 12,722 | 881 / 12,576 / 20,359 | 790 / 2,898 / 6,822 |
| Closed block rectangularity (median) | 0.85 | 0.95 | 0.90 |

**Implication:** most blocks are truncated by the tile edge. Parcel statistics must come from
closed blocks, or from data larger than one tile.

### 1.3 Building sizes (real OSM, training tiles; 9,164 / 26,193 / 43,487 buildings)

| | Spain | US | Japan |
|---|---|---|---|
| Area p10 / p50 / p90 (m²) | 50 / 173 / 675 | 32 / 99 / 295 | 26 / 62 / 186 |
| Width × length (median, m) | 10.5 × 19.8 | 7.1 × 14.4 | 6.3 × 10.7 |
| Consecutive on the same frontage within a factor of 1.5 in area (random pairs in the same tile) | 51 % (30 %) | **71 % (41 %)** | 46 % (34 %) |
| Nearest neighbour within ×1.5 (random) | 48 % (29 %) | 64 % (35 %) | 47 % (33 %) |
| Same block within ×1.5 (random) | 38 % (30 %) | 40 % (37 %) | 39 % (35 %) |
| Moran's I of log area: nearest neighbour / same frontage / same block | 0.35 / 0.31 / 0.17 | 0.47 / 0.36 / 0.07 | 0.20 / 0.15 / 0.06 |

**Reading.** Size similarity is real but **local**: along a frontage and between nearest
neighbours, strongest in US suburbs and weakest in Japan. It almost disappears at block scale, so
large buildings beside small ones are normal. A model should therefore condition each building
on its **frontage neighbours**, not on a block-wide size.

### 1.4 Where parcels come from

| Country | Source | Our cities | Licence (read from the official text where available) | Caveats |
|---|---|---|---|---|
| Spain | Dirección General del Catastro, INSPIRE Cadastral Parcels (ATOM, per municipality, GML), updated twice a year | Barcelona, Madrid, Valencia ✔ (all of Spain except the Basque Country and Navarra) | Licence text v1.0 (2016), read directly: use **including commercial is allowed for *transformed* products**; the **original data may not be redistributed**; derived products must not be called "cadastral cartography" | Use locally; publish only derived statistics and figures; raw parcels never committed |
| US | NYC MapPLUTO (free and public); LA County Enterprise GIS (terms: copy, publish, adapt with citation); Cook County Assessor "Parcel Universe" (public portal, **licence not specified**) | New York ✔, Los Angeles ✔, Chicago ? | per county; Chicago must be checked before any redistribution | Parcels are tax lots; condominiums and air rights are complicating cases |
| Japan | Ministry of Justice registry maps (登記所備付地図), map-XML via the G-Space Information Center (2023–), per municipality | Tokyo, Osaka, Nagoya ? | terms allow reproduction, adaptation and commercial use with attribution (catalogue label inconsistent, so the official terms will be re-read before use) | Mix of **public and arbitrary local coordinates**; arbitrary-coordinate sheets cannot be georeferenced; coverage in dense areas unknown until measured |

**Conclusion:**
* **Observed parcels are obtainable** for Spain and for 2–3 US cities.
* **Japan is uncertain.**
* So parcel *learning and validation* can be honest for Spain and the US. For Japan we would
  report only internal consistency, or inferred parcels **labelled as inferred**.

## 2. Proposed architecture (road → block → parcel → building)

| Stage | Programmed rule | Learned from data | Neural network output |
|---|---|---|---|
| Roads | (M6 roads v2, unchanged) | | |
| A. Blocks | tile minus road surface; open / closed; slivers dropped | — | — |
| B. Parcels | **subdivision procedure**: frontage strips along each block side (Vanegas skeleton-strip idea, simplified) cut at sampled frontage widths; rear line at a sampled depth; interior or corner remainders become their own parcels | **frontage width, depth, irregularity, parcels per side** (distributions per country / morphology, from **observed** parcels) | (Experiment E only) a small network choosing these parameters per block side from context |
| C. Buildings in parcels | containment in the parcel; no overlaps; angle ≥ 45° | occupancy and coverage distributions | per parcel: **occupied?**, number of buildings, **front / side / rear setbacks, coverage**, orientation, shape. A mixture density network conditioned on parcel geometry, street class and the **previous parcel's building** (the local size similarity found in §1.3) |
| Corrections | **shrink-to-fit inside the parcel** (scale down to ≤ the parcel minus setbacks) instead of rejection; counted and reported | — | — |

**Why parcels can address the diagnosis:**
* **Overlaps between consecutive buildings** become impossible by construction (each building
  lives in its own parcel).
* **Rejection becomes adjustment** inside a known space.
* **Coverage and setbacks** become explicit, learnable quantities.

**The risk,** and why Experiment B exists: the same benefits might come from simply making the
frontage geometry coherent (edge-to-edge gaps) plus a neighbour-aware fallback, **without
parcels**.

**Bias risks (procedural subdivision):**
* rectangular strips can impose regularity;
* mitigation:
  * irregularity is a learned parameter;
  * consolidated parcels (one building over several) and empty parcels are allowed;
  * the diversity metrics penalise uniform subdivision.

## 3. Controlled experiments (one factor at a time; ≥ 2 seeds; same validation tiles and metrics)

| Exp. | Change vs previous | Needs parcels? | Answers |
|---|---|---|---|
| **A** | Milestone 7 (frozen) | no | baseline |
| **B1** | frontage step re-parameterised: **edge-to-edge gap ≥ 0 + frontage length** (no overlap with the predecessor by construction); shapes from a smoother space (rectangle + learned corner notches); shrink-to-fit instead of rejection | no | how much of the gap is geometric incoherence |
| **B2** | B1 + **neighbour-aware fallback** (orientations, sizes and offsets of the k nearest placed buildings) | no | secondary Q7 baseline: neighbours without parcels |
| **C** | B2 + **procedural parcels** (fixed, hand-set frontage / depth per country from literature-free defaults) + buildings conditioned on parcels | generated only | does the parcel structure itself help? |
| **D** | C with frontage / depth / irregularity / occupancy / setback / coverage **distributions learned from observed parcels** (Spain, US) | observed (train) | do learned parcel statistics help? |
| **E** | a neural network chooses parcel parameters from block context | observed | only if D shows parcel statistics matter **and** C → D improves |
| **F** | combination of what helped | — | final |

Each of B–F is compared with its predecessor and with A. Country conditioning is a separate
ablation on the best model:
* (a) none;
* (b) country one-hot;
* (c) city;
* (d) morphology descriptors we already have (GHSL density × era stratum, local street density).

All variants have the same parameter count.

**Pre-registration** (written into `docs/MILESTONE_8.md` before any samples). An improvement
counts if, for **both seeds**:
* the W1/IQR to real validation improves on ≥ 3 of these 5: neighbour alignment, spacing CV,
  setback std, building-area distribution, local size similarity;
* there is no regression beyond 1 IQR on the 9 M6 metrics;
* there are zero validity violations;
* diversity is ≥ real − 0.05.

**Parcel-realism metrics** (area, frontage, depth, aspect, orientation, parcels per block,
irregularity) are reported **only** against observed parcels (Spain, US, validation tiles). They
are never against inferred ones.

## 4. Costs

| Item | Estimate | Approval needed |
|---|---|---|
| **Parcel data pilot**: Catastro (Barcelona, Madrid, Valencia municipalities), NYC MapPLUTO, LA County via ArcGIS REST limited to our tiles, Cook County, 1 Japanese ward as a coverage / coordinate check | download ≈ 1–3 GB, ≈ 30–60 min including clipping to our ~600 tiles; raw data kept outside git | **yes** (network download) |
| Stage A (blocks) | done for analysis; productionise + tests ≈ 0.5 day | — |
| B1 / B2 (retrain frontage + fallback) | ≈ 15 min (frontage) + ≈ 45 min (fallback) per seed on MPS | yes (training) |
| C (procedural parcels + parcel-conditioned building MDN) | implementation ≈ 1.5 days; training ≈ 20 min per seed | yes |
| D (fit distributions from observed parcels) | minutes | — |
| Evaluation per experiment | ≈ 15 min | — |
| E (neural parcel parameters) | only if justified; ≈ 30 min per seed | later |
| Dataset expansion | still pending (M6 plan §5); parcel availability argues for adding Spanish and US cities first | separate |

## 5. Decisions requested

1. **Parcel data pilot:** download and clip observed parcels for our tiles (Spain, US; Japan check).
2. **Run order A → B1 → B2 → C (→ D):**
   * B first, because the diagnosis says most of M7's failure is geometric incoherence;
   * if B2 already closes the gap, parcels may not be needed for alignment;
   * parcels may still be needed for coverage and setbacks.
3. **Whether to keep Japan in parcel experiments** with inferred parcels, clearly labelled, or to
   restrict parcel learning to Spain and the US.
