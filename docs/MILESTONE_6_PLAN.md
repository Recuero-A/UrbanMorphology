# Milestone 6 plan: hybrid neural-procedural city generation (for approval, nothing trained yet)

Companion documents:
* `docs/HYBRID_GENERATION_LITERATURE.md` (literature review);
* `docs/MILESTONE_5.md` (graph generator results).

## 1. What is learned and what is programmed

| Part | Learned (neural network, from data) | Programmed (explicit rule) |
|---|---|---|
| Roads (Stage A) | where components start; each step's length and turn; branching; stopping; street class | clip at tile edge; snap within 8 m; split at crossings; T-junctions |
| Blocks (B) | — | polygonise the faces of the road graph minus road buffers (computational geometry) |
| Buildings (C) | count, position, orientation, size and shape parameters, order | inside the block; no overlap with roads or other buildings; valid polygon |
| Green (D) | later | later |

**Principle:** rules check validity ("can this exist?"), never style ("should it look like a grid?").
Every rule is listed, and each experiment reports how often each rule fired.

## 2. Assessment of the Milestone 5 backbone (see MILESTONE_5.md)

Pre-registered verdict: **not met** (both seeds). In detail:
* **Connectivity: ✅** 0 floating fragments; versus M4 (4 fragments) this is the clearest gain so far.
* **Diversity: ✅** 0.92 vs 0.94 real.
* **Memorisation: ✅** none.
* **Structure: ❌** junctions 4×, loops 6×, φ 0.15 vs 0.59, and 88–91 % single-component tiles vs
  42 % real.

The diagnosis points to the **absolute-frame step parameterisation**, a simplification of NTG.
So M5 **is a suitable backbone after a targeted fix**, not something to rebuild. Stage A of the
prototype is that fix, evaluated with the unchanged M5 protocol before any buildings are added.

## 3. Candidate architectures

| | **H1: hierarchical vector pipeline** (recommended) | **H2: graph roads + raster buildings** | **H3: one transformer over all elements** |
|---|---|---|---|
| Roads | M5 GraphTurtle, steps in the local turtle frame, plus a street-class head | same as H1 | tokens of a joint sequence |
| Blocks | polygonise roads (programmed) | — | implicit |
| Buildings | autoregressive footprint placer per block: anchor, orientation relative to the nearest road, size, shape as a radial polygon (16 radii); conditioned on a local raster (roads, already-placed buildings, block mask) | conditional raster model p(building pixels \| road raster), reusing the M3 VAE code; then vectorise by contour tracing and simplification | buildings and roads as tokens; a decoder-only transformer, as in COHO / BlockPlanner |
| Validity | by construction (rejection plus rules) | not guaranteed: merged buildings, vectorisation artefacts | learned only, unless rules are added |
| Data needs | modest: about 25k buildings already in 565 tiles | modest | high (thousands of tiles) |
| Bias risk | rules could leak style; mitigated by a rule audit | raster smoothing → blobs | none built in; overfits small data |
| Cost on an M-series Mac | ~10–20 min training per stage | ~1 h (M3 scale) | hours to days |
| Scientific value | clean answers to "does explicit geometry help?" and "do rules reduce diversity?" | answers "does vector conditioning help the raster?" | best long-term, worst now |

**Recommendation: H1.**
* **Simplest meaningful option:** each stage can be tested on its own, and it reuses M5.
* **Fits our data size:** it can be trained on the data we already have.
* **H2** is a useful ablation later: same roads, raster buildings.
* **H3** fits only after the dataset expansion.

## 4. Recommended first prototype (scope of the first approval)

1. **Fix Stage A**, a small change to M5. The diagnosis showed absolute (dx, dy) steps cause
   skewed junctions. Two changes:
   * predict each step in the turtle frame of the incoming street (turn angle + length, mixture
     of von Mises × log-normal), as in NTG;
   * add a street-class head (OSM highway → 4 classes) so road width comes from data, not a
     constant.

   Retrain 2 seeds (about 10 min each) and re-run the M5 evaluation unchanged.
2. **Road geometry:** buffer each edge by its class's width, sampled from the real per-country
   width distribution (programmed). Polygonise into blocks: shapely `polygonize` on the road
   lines, minus the road buffers.
3. **Buildings, two levels, so we can measure what the network adds:**
   * **P1 (statistical baseline):** sample building attributes from per-country empirical
     distributions conditioned on block-boundary distance (no neural network). Place them with
     the same validity rules.
   * **P2 (neural placer):** a small autoregressive model, trained on real blocks and
     teacher-forced, like M5. Placement order: along the block boundary, then inward.
     * Input: a 64 × 64 local crop with channels for road, block, placed buildings and distance
       to the boundary.
     * Output: anchor cell; angle relative to the nearest road (mixture); log length and log
       width; 16 radial shape parameters; STOP.
     * About 1–2 M parameters.
4. **Render** as a 2D map (matplotlib) and **export** GeoPackage / GeoJSON layers: roads with
   class and width, blocks, buildings. These layers are ready for 3D extrusion later.
5. **Evaluate** on validation tiles (§6), per country.

**Conditioning honesty:** P1 and P2 are first trained and evaluated on **real** road networks
(conditional), then run on **generated** roads (unconditional pipeline). Raster baselines
(VAE, VAE + prior) generate everything jointly, so comparisons are labelled by what each model
receives.

## 5. Dataset expansion proposal (estimate only; nothing processed)

| Item | Current | Proposed |
|---|---|---|
| Source | 9 city clips from cached Geofabrik extracts (one snapshot) | same 9 regional extracts (4.5 GB, **cached, no download**): more cities inside them |
| Cities | 3 per country | ~10 per country. Candidates in the cached extracts:<br>• ES: Barcelona metro (Sabadell, Terrassa), Girona, Tarragona, Lleida, Alcalá, Móstoles, Alicante, Elche, Castellón<br>• US: San Diego, Riverside, Santa Barbara, Springfield, Peoria, Rockford, Buffalo, Rochester, Syracuse, Albany<br>• JP: Yokohama, Saitama, Chiba, Kyoto, Kobe, Shizuoka, Hamamatsu, Niigata |
| Tiles | 603 (565 train) | ~3,000–4,500 (about 100–150 per city, ≥ 512 m apart) |
| Disk | 1.8 GB raw + 0.3 GB processed | +~12 GB raw, ~2 GB processed (577 GB free) |
| Time | — | city clipping ~1–2 min × ~25 cities; tile extraction ~1 s per tile (measured, M3b); processing ~1–2 s per tile. **Total ≈ 2–4 h CPU**, measured on a 50-tile pilot first |
| Building completeness | GHSL ratio check per city (M2) | same check. Expect smaller Japanese and US cities to be less complete; such cities are excluded or flagged before sampling |
| Leakage | 3,072 m spatial blocks, 1 km buffer | new cities: whole city in train **or** a new held-out city. Existing val/test cities: new tiles only ≥ 1 km outside val/test blocks. **Existing test split untouched** |
| Effective diversity | 9 cities, ~70 spatial blocks | tiles within one 3 km block are strongly correlated. We will report the number of independent blocks and cities (expected ~30 cities, ~400–600 blocks) and use **leave-one-city-out** for generalisation |
| Sampling bias | GHSL-stratified (density × era) | same strata per city; report per-stratum counts; OSM completeness bias (rich areas mapped better) reported, not hidden |

**Approval needed** before the pilot and before the full run.

## 6. Evaluation plan

| Family | Metrics | Reference |
|---|---|---|
| Roads | components, floating components, junctions/km², dead-end ratio, loops, link length, φ, entropy, junction-angle deviation, street-class shares | real val per country (W1/IQR), GraphRNN-style MMD |
| Buildings | area and aspect distributions, rectangularity, #vertices, coverage, orientation relative to the nearest road, building–road distance, nearest-neighbour spacing, Ripley's K / pair-correlation (clustering) | same |
| Validity | invalid polygons, building–road overlaps, building–building overlaps, buildings outside blocks, impossible intersections (crossings without nodes), how often each rule fired | must be ~0 for H1; reported for the raster baselines |
| Diversity | 1 − pairwise IoU; per-country spread (IQR) of each metric versus real | real val |
| Memorisation | nearest training tile (8 symmetries), road and building IoU | real val's own value + 0.10 |
| Generalisation | gap between seen-city validation and a held-out city | — |

**Baselines:**
* A: Spatial VAE (C2).
* B: C2 + PixelCNN prior (M4).
* C: M5 GraphTurtle (roads only).
* D: H1.

Raster models are compared via rasterisation of D's vectors. Vector-only metrics (angles,
polygons) are reported only for vector models. **Not a fair comparison** in the strict sense:
D has explicit geometry and staged conditioning. This is stated wherever results appear.

Per-country evaluation: needs a country condition (one-hot) on Stage A and C, or per-country
models. Planned as a one-hot input, with "country = unknown" dropout.

## 7. Costs

| Step | Implementation | Compute |
|---|---|---|
| Stage A fix (turtle frame + class head) | ~0.5 day | 2 × ~10 min train, ~5 min evaluation |
| Blocks + validity rules + export | ~0.5 day | seconds per tile |
| P1 statistical buildings | ~0.5 day | minutes |
| P2 neural placer | ~1 day | 2 × ~20 min train (estimate, measured before running) |
| Evaluation + docs | ~1 day | ~15 min |
| Dataset expansion (separate approval) | ~1 day | 2–4 h CPU |
