# Milestone 12: terrain-aware Neural Turtle Graphics

**Status (2026-11-04): development complete (Phases 1–8, §13–§17). Waiting at gate 3: the one-time held-out test needs explicit authorisation.** Phase 1–2 history: §1–§12.
* DEMs were downloaded for the Spanish and US train + validation tiles only.
* Japan has no terrain (GSI licence unresolved).
* Models trained: N0, TN, TN-lp × 2 seeds, plus 4 ablations. Evaluation is on development data only.
* The held-out test splits have not been read (`test_t` is not built).
* Results: `M12_RESULTS.md`.

Detail documents:
* `M12_LITERATURE_REVIEW.md`
* `M12_DATA_PROTOCOL.md`
* `M12_MODEL_ARCHITECTURE.md`, `M12_EVALUATION_PROTOCOL.md` (pre-registered), `M12_RESULTS.md`, `M12_PROGRESS_LOG.md`
* `M12_TERRAIN_BRIEF.md` (the summary of M1–M11 written before this milestone)

---

## 1. Phase 1: what was done

| Item | Result | Where |
|---|---|---|
| Repository inspection | Integration points identified (§2) | — |
| Literature (17 works; 9 read in full or in part) | Comparison table, qualified novelty assessment | `M12_LITERATURE_REVIEW.md` |
| Terrain and water audit, 565 train + 19 val tiles, OSM only | Water, coastline, bridges, tunnels, crossings, tag coverage | `scripts/m12_audit_terrain.py`, `outputs/m12/audits/` |
| Elevation-source investigation (no download) | DTM candidates per country with datum and licence status | `M12_DATA_PROTOCOL.md` §2 |
| Checkpointing and automatic resume (requested 2026-11-04) | Implemented and tested before any training | `src/urbanmorph/models/checkpoint.py`, `tests/test_checkpoint.py` |
| Dependencies | `rasterio` 1.5.2 and `pyproj` present. `scipy` absent (only needed for convenience in hydrology; NumPy suffices). No large application needed | — |

## 2. Repository inspection: where terrain enters

| Component | Current behaviour | M12 change (planned; baseline untouched) |
|---|---|---|
| `graphs/sequence.py` `crops_turtle` | 3-channel 48 × 48 crop (streets, open nodes, outside), rotated to the heading | Add terrain channels sampled on the **same rotated grid**. The gradient vector is rotated by −heading |
| `models/graph_gen.py` `GraphTurtle` | `Conv2d(3, …)` input; scalars; GRU decoder; 10-component mixture, stop, class | New `TerrainGraphTurtle` subclass with `3 + k` input channels and terrain scalars. `GraphTurtle` keeps its exact behaviour and checkpoints |
| `graphs/generate.py` | Geometric rules after each step: clip, split at crossings, merge 8 m, T-junction, drop short | A feasibility hook *before* the rules (rules-only R) and a bounded replanning loop (HR). The baseline path is unchanged when the hook is absent |
| `graphs/road_graph.py` `build_graph` | **Planar**: every crossing within 8 m is merged | A 3D-aware variant keeps `bridge`, `tunnel` and `layer`, so grade-separated crossings are not junctions. 76% of bridge length crosses non-water (audit) |
| `scripts/m11_big_city.py` | Windowed growth with `given` / `frontier` | Each window samples crops from one city-wide terrain field |
| `city3d/*`, Blender, Godot | Flat z = 0 | Phase 8 |

## 3. Scientific findings of Phase 1

1. **Terrain-conditioned learned street generation already exists, for raster models** (DeepStreet 2020; Fang 2022;
   Gu 2024).
   * Adding elevation channels is therefore not a contribution.
   * Naively added terrain channels helped little in two independent studies (Fang 2022, Model 1; Gu 2024).
   * The defensible contribution space is: sequential graph decisions with feasibility feedback, explicit
     infrastructure actions, and 3D-consistent graphs. This holds only after a systematic search.
2. **Our data contain water mainly in Japan.**
   * 81 training tiles have ≥ 1% water, 57 of them in Tokyo, Osaka and Nagoya; 23 tiles are coastal.
   * **Validation has water in only 2–4 of its 19 tiles**, too few for any water or bridge claim.
3. **OSM crossing labels are usable.** About 90% of street–water crossings are bridge-tagged, and untagged ones will
   be "unknown".
4. **Relief cannot be assessed yet.** The tiles were sampled from dense, mostly flat built-up areas. Whether they
   contain moderate or steep slopes is unknown until a DTM is fetched. This is the main risk to the milestone.
5. **Checkpointing.**
   * Interrupted and resumed training equals uninterrupted training. Parameters match to 1e-7 (CPU; exact in
     practice) for interruptions mid-chunk, at a chunk end and at an epoch end, and for two interruptions in a
     row.
   * Disabling RNG restoration makes these tests fail, so they detect a broken resume.

## 4. Checkpoint and resume contract (all M12 training scripts)

`Checkpointer` in `src/urbanmorph/models/checkpoint.py`.

**When it saves**
* every 300 s (`--ckpt-interval`) at the next safe point (after an optimiser step);
* at the end of every epoch;
* on Ctrl+C or SIGTERM. A second signal aborts immediately.

**What it saves:** model, optimiser, optional scheduler and gradient scaler, user state (epoch, global step,
position in the epoch, the data-generator state at the start of the current chunk, best score, history), the RNG
states (Python, NumPy, torch CPU and MPS / CUDA, named generators), and the config with its SHA-256.

**Crash safety**
* Each save goes to a temporary file, is fsynced, then `os.replace`d. The old `latest.pt` becomes `previous.pt`.
* A corrupt or truncated `latest.pt` falls back to `previous.pt`.
* **Resuming with a different configuration is refused.**

**Logs.** `RunLog` is append-only JSON lines. A resume writes a `resume` event and continues; nothing is
overwritten.

**Usage:** `--resume latest` or `--resume PATH`.

## 5. Unresolved risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Training tiles may be too flat | TN cannot learn slope behaviour, whatever its design | Measure first (gate 2). Expand with terrain-diverse cities chosen by measured diversity |
| 19 validation tiles, 2–4 with water | No statistical power for terrain or water metrics | New held-out regions (whole cities) for terrain evaluation; the existing test split untouched |
| Terrain signal is weak in learned models (literature) | TN ≈ N0 | Pre-registered ablations, a shuffled-terrain control, and R as the honest comparison |
| Learned replanning has no obvious supervision | Cannot be trained without fabricated labels | Heuristic replanning first. The learned variant only with a ranking objective whose assumptions are documented (spec §10.5) |
| A planar graph loses grade separation | False junctions near overpasses | A 3D-aware graph build (Phase 2), measured before and after |
| Mixed vertical datums | Wrong elevations across countries | Per-country orthometric conversion; slopes only across countries |
| DEM5A survey-result terms (Japan), Korean DEM unknown | Licence or availability gaps | Verify before download. Exclude regions without a compatible licence |

## 6. Estimated next-stage compute (estimates; the pilot will measure them)

| Stage | Estimate |
|---|---|
| DEM fetch for train + val boxes of the nine cities (about 584 tiles × (512 + 2·256 m)² ≈ 610 km²) | Hours of wall time at service rate limits; volume small at 4–5 m. **The exact size and time need the provider indices: verify** |
| Terrain sample building (DEM sampling, water rasters, 3D-aware graphs) | Under 1 h CPU for the current tiles |
| TN training, current data | About 80–100 min per seed on MPS (N took 79 min; extra crop channels add a little CNN cost) |
| Full plan: N0 retrain + TN (2 seeds each) + 4 single-seed channel ablations, on 2× data | About 10 runs × about 3 h ≈ 30 MPS-hours sequential (2–3 runs can share the GPU) |

## 7. Decisions requiring approval (gate 1)

1. **DEM download for the nine existing cities**, train + validation bounding boxes only (+256 m margin):
   * IGN MDT05 (Spain), USGS 3DEP 1 m (US), GSI DEM5A with DEM10B fallback (Japan);
   * through bbox or tile services, after the licence checks marked *(verify)* in `M12_DATA_PROTOCOL.md`.
2. **Then the terrain diversity audit:** elevation range, slope quantiles, street-grade distributions, measured
   per city and split.
3. **Expansion choice deferred until 2.** Are Seoul and Busan acceptable in principle as street-only cities, given
   M2 excluded them for building completeness? Are new OSM downloads acceptable (San Francisco, Rio de Janeiro)?
4. **The systematic literature search** (Scopus or Google Scholar style queries), recorded in
   `M12_LITERATURE_REVIEW.md` §6, before any novelty statement.
5. **Optional:** install `scipy` (small pip package) for hydrology convenience. NumPy-only is possible.


---

# Phase 2 (approved 2026-11-04, conditionally): data, graph correction, audit, literature

## 8. What was done

| Item | Result | Where |
|---|---|---|
| Licence, CRS, datum and provenance checks, then DEM download | Spain (IGN MDT 5 m, CC BY 4.0) and US (3DEP 1 m, public domain): 365 tiles, no missing data. **Japan held** | `scripts/m12_fetch_dem.py`, `M12_DATA_PROTOCOL.md` §5 |
| Terrain representation | `Terrain` (heightfield, bilinear sampling, gradient, slope, along-segment profiles, provenance, native resolution) + DEM loader | `src/urbanmorph/terrain/{heightfield,data}.py` |
| **3D-aware street graphs** | Bridge / tunnel / layer from the OSM ways (matched by way id; multi-way lines cut along their tagged members); grade-separated crossings are no longer junctions; metadata kept through `transform`, `subgraph` and generation | `graphs/levels.py`, `graphs/road_graph.py`, `graphs/generate.py`, `graphs/crossings.py`, `scripts/m12_build_structures.py` |
| Crossing classification | at-grade (shared node, confirmed), portal, grade-separated (confirmed by tags), **ambiguous** (no shared node, no tag); water: bridge / tunnel / ford / **unknown** (never assumed ground) | `graphs/crossings.py` |
| Effect on GraphTurtle data | §9 | `scripts/m12_graph_levels_effect.py`, `outputs/m12/graph_levels/` |
| Terrain-diversity audit | §10 | `scripts/m12_terrain_audit.py`, `outputs/m12/audits/terrain_audit.*` |
| Scripted literature search + citation chasing | 844 + 552 records, 20 included, recall 5/14 | `scripts/m12_lit_search.py`, `scripts/m12_lit_citations.py`, `M12_LITERATURE_REVIEW.md` §7 |
| Tests | 177 → **194 passing**: overpass, tunnel, river bridge, ordinary intersection, ambiguous crossing, multi-way cut, generation under a given bridge, terrain analytic cases | `tests/test_levels.py`, `tests/test_terrain.py` |
| SciPy | 1.18.1 installed (used for the TPI filter) | — |

**Checkpointing.** The Phase 1 `Checkpointer` and its tests are unchanged and pass.

## 9. The road-graph correction and what it changes

**Regression check.** Without vertical metadata, the graph build and the generator are **bit-identical** to the
previous commit:
* same hash over 40 real tiles (graph build);
* same hash for 9 sampled graphs, including a `given` network, with the N checkpoint at T = 0.5 (generator).

The M10/M11 results are therefore reproducible exactly.

**Effect on the training and validation graphs** (old = planar snap, what N was trained on; new = 3D-aware):

| | Train (565 tiles) | Val (19) |
|---|---|---|
| Tiles whose graph changed | 180 (32%) | 8 |
| Junctions old → new | 17,307 → 17,133 (−1.0%) | 574 → 574 |
| Grade-separated crossings now kept apart | 687 | 43 |
| Ambiguous crossings (no node, no tag) | 47 | 2 |
| Portal junctions (bridge ends, ramps) | 119 | 12 |
| Street length on bridges / in tunnels | 1.45% / 0.28% | 3.5% / 0.16% |
| Shift of the S10 reference (W1/IQR, old vs new real metrics) | **0.022** | **0.055** |

**Per city.** Osaka changes most: 175 grade-separated crossings and 2.8% of street length on bridges. Chicago
changes least.

**Interpretation and recommendation**
1. **The correction is small on aggregate:** a 0.02–0.05 S10 shift against the N gap of 0.30. It is concentrated
   in Japanese and Spanish tiles with viaducts and ramps.
2. **M10/M11 results stand as reported.** Their reference was the planar data, which N was trained and evaluated
   on consistently.
3. **For M12, the terrain-blind baseline N0 must be retrained on the 3D-aware graphs.** TN will be trained on
   them, and a controlled comparison needs identical street data. The shift is small but not zero, and it is
   largest where terrain matters (viaducts, ramps). This is part of the training plan at the next gate; it has not
   been run.
4. **Remaining mismatch, documented.** Generated streets are at ground level, and generation rule 2 still turns a
   crossing of two *generated* streets into a junction. Real overpasses in the training sequences can therefore be
   reproduced only as junctions until an infrastructure action exists (spec §8.3).

## 10. Terrain-diversity audit (Spain + US: 353 train, 12 val tiles; Japan pending)

| | Train | Val |
|---|---|---|
| Relief p95−p5 per tile, median (p90) | 6.9 m (27.6 m) | 6.1 m (22.5 m) |
| Landform slope p90 (32 m), median (p90) | 4.6% (13.5%) | 4.9% (10.8%) |
| Landform class flat / moderate / steep | 187 / 136 / **30** | 6 / 5 / **1** |
| Street grade p90 (ground edges), median (p90) | 2.7% (7.5%) | 1.8% (7.2%) |
| Tiles whose p90 street grade > 10% | **21** | 1 |
| Valley tiles (relief ≥ 40 m and TPI < −5 m on ≥ 10%) | **11** | 1 |
| Bridge runs / over water / **dry, clearance ≥ 10 m (valley-spanning)** | 133 / 10 / **0** | 7 / 2 / 0 |
| Street–water crossings: bridge / unknown | 10 / 4 | 2 / 0 |

**By city.**
* Steep landform tiles: Los Angeles 17, Barcelona 6, Madrid 5, New York 3.
* Highest relief: Barcelona 116 m, Los Angeles 97 m.
* Chicago and Valencia are almost flat (median relief 2–3 m).

**Water and bridges, all nine cities** (OSM, Phase 1): 84 training tiles with a river polygon, 64 with a bridge
over water, mostly in Japan. Validation: 2–4 tiles.

**Adequacy.** Rule fixed before the results: ≥ 30 train tiles from ≥ 2 cities and ≥ 10 blocks for training; ≥ 5
validation tiles for evaluation.

| Feature | Train tiles / cities / blocks | Val tiles | Training | Evaluation |
|---|---|---|---|---|
| Moderate or steep landform | 166 / 6 / 102 | 6 | adequate | adequate (barely) |
| **Steep landform** | 30 / 4 / 26 | 1 | adequate (at the threshold) | **inadequate** |
| **Street grades p90 > 10%** | 21 / 4 / 20 | 1 | **inadequate** | **inadequate** |
| **Valleys** | 11 / 2 / 10 | 1 | **inadequate** | **inadequate** |
| **Valley-spanning bridges (viaducts)** | 0 | 0 | **absent** | **absent** |
| Rivers (all 9 cities, OSM) | 84 (mostly Japan) | 2 | adequate | **inadequate** |
| Bridges over water (all 9 cities, OSM) | 64 | 3 | adequate | **inadequate** |
| Lakes / coastline | 54 / 23 | 0 / 1 | adequate / borderline | **inadequate** |

**Answer to the gate question.** The current dataset represents **flat and moderate terrain well**.
* Steep terrain is present only at the minimum.
* Valleys and steep streets are under-represented.
* Valley-spanning viaducts are absent.
* Rivers and water bridges are adequate for training, but mostly in Japan, whose DEM is pending.
* **No terrain or water feature has enough validation tiles for an evaluation claim.**

Two quality limits:
* the Spanish DEM mosaic is terraced in places (§5.3 of the data protocol), so street grades in Valencia and parts
  of Barcelona are uncertain by about ±3%;
* Japan has no DEM yet.

## 11. Recommendations (only where coverage is insufficient)

1. **Steep streets, valleys, viaducts and terrain evaluation sets: add terrain-diverse regions**, chosen by measured
   DEM statistics before selection. In order of data readiness:
   * **San Diego and the hillside parts of Los Angeles**: OSM cached (SoCal), 3DEP lidar, canyons and canyon
     bridges;
   * **San Francisco**: 3DEP lidar 1 m verified; needs the NorCal OSM extract;
   * **Kobe, Yokohama, Kyoto**: OSM cached; depends on the GSI decision.

   Hold out at least one whole hilly city for evaluation.
2. **Not recommended now.**
   * **Rio de Janeiro**: no open bare-earth DTM found; OSM not cached; completeness doubtful.
   * **Seoul and Busan**: DEM access needs a login, and the licence and resolution are unverified; road
     completeness has no reference; building completeness is known to be poor. They could be reconsidered for
     street-only training once both are verified.
3. **Rivers and bridges:** covered in Japan for training. Evaluation needs more wet validation regions, e.g. a
   held-out Japanese river city if GSI is approved.
4. **Spanish DEM quality:** consider IGN MDT02 (2nd LiDAR coverage) for Barcelona and Madrid if Spanish street
   grades are used for training targets. Verify first that it is float data.

## 12. Decisions requiring approval (gate 2)

1. **Japan DEM:** (a) apply to or ask GSI about 使用承認 for research use of DEM5A, (b) use FABDEM (30 m,
   CC BY-NC-SA) for Japan as a declared lower-quality arm, or (c) exclude Japanese terrain.
2. **Dataset expansion** for steep terrain, valleys and viaducts (§11.1). Which regions? This requires DEM
   downloads, and an OSM download for San Francisco.
3. **Spanish MDT02** for street-scale grades (optional; §11.4).
4. **Library-database search** (Scopus / WoS via KAIST) and **full text of CAVW 2024** (doi 10.1002/cav.2267, via
   KAIST access), before any novelty statement.
5. **After 1–2:** procedural terrain (Phase 3) and feasibility baselines (Phase 4) are CPU-only and can proceed.
   **Training** (N0 retrain on 3D-aware graphs, TN) waits for its own gate with a measured pilot (spec §12).

---

# Phases 3–8 (approved 2026-11-04, autonomous): terrain, feasibility, models, evaluation, 3D

All numbers are on development data. Full tables: `M12_RESULTS.md`. Design decisions: `M12_MODEL_ARCHITECTURE.md`.
Dated log, including the mistakes: `M12_PROGRESS_LOG.md`.

## 13. What was done

| Phase | Deliverable | Where |
|---|---|---|
| Data expansion | 120 steep Los Angeles train tiles; 40 San Diego `val_t` tiles (a held-out city); 34 `test_t` tiles selected but **not built**; selection by measured 32 m landform slope, seed 20261104 | `scripts/m12_select_regions.py`, `m12_build_tiles.py`, `M12_DATA_PROTOCOL.md` §5 |
| 3. Procedural terrain | Seamless, seeded landscapes in world coordinates (fBm, ridged noise, valleys, basins); priority-flood and D8 hydrology; rivers with hydraulic-geometry widths; lakes only from deep, large depressions; sea; 10 named scenarios | `terrain/synthesis.py`, `terrain/hydrology.py` |
| 4. Feasibility | Profile every 2 m; per-class grade limits from training streets (30 m sub-span); bridge span, angle and deck grade; lake / sea obstruction; viaduct clearance | `terrain/road_feasibility.py`, `configs/m12_feasibility.json` |
| 5. Models | TerrainGraphTurtle: GraphTurtle + rotated terrain crops (elev, slope, water, valid) + structure head (ground / bridge / tunnel); policies none / R / HR (pre-registered) / HRV (exploratory); procedural baseline P | `models/terrain_graph_turtle.py`, `graphs/terrain_generate.py`, `graphs/procedural_terrain.py` |
| 5. Training | N0, TN, TN-lp × 2 seeds; 4 single-channel ablations; checkpointed and resumable; resume verified bit-identical | `scripts/m12_train.py`, `m12_train_all.py`, `models/checkpoint.py` |
| 6. Evaluation | Regimes A (conditional real terrain, 52 tiles), B (unconditional S10), C (50 synthetic landscapes), H5 (structure head); paired tile bootstrap | `scripts/m12_evaluate.py` |
| 7. 3D | Terrain meshes, draped roads, buildings on slopes, water, bridge decks and piers, viaducts; Blender renders; Godot collision check; a 3 × 3 landscape | `city3d/terrain3d.py`, `scripts/m12_build_city3d.py`, `m12_big_landscape.py`, `m12_showcase.sh` |
| 8. Final test | One-time pipeline that refuses to run without explicit authorisation; development outputs frozen | `scripts/m12_final_test.sh` |

Tests: 231 pass (`pytest tests/`).

## 14. Results in one table

See `M12_RESULTS.md` §1 for the narrative and every caveat.

Pre-registered hypotheses. Both seeds must agree. Development data only.

| Hypothesis | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|
| H1: TN improves S_T over N0 by ≥ 20% | −68.5% | −68.9% | pass, confounded by leakage |
| **H1lp**: the same with TN-lp (leakage control) | −53.1% | −49.9% | **pass** |
| H2: S10(TN) ≤ S10(N0) + 0.10 | 0.312 vs 0.328 | 0.319 vs 0.333 | pass |
| **H2lp** | 0.305 vs 0.328 | 0.314 vs 0.333 | **pass** |
| H3: TN+R improves S_T over R by ≥ 10% | −50% | −52% | pass, confounded |
| **H3lp** | −21%, CI upper +0.045 | −33% | **fail** |
| H4: HR reduces premature termination, ΔS_T ≤ 0.05 | ΔS_T +0.095 | +0.089 | **fail** |
| H5: structure-head bridge AUC ≥ 0.8 | 0.967 | 0.967 | pass (TN-lp: 0.949 / 0.943) |

**What the evidence supports**
1. Terrain inputs make a learned turtle generator avoid steep streets on held-out real terrain. This survives the
   leakage control at about three quarters of the raw effect.
2. It does so without losing street realism.
3. Its structure head separates bridges from ground steps.

**What it does not support**
1. That terrain inputs help beyond hard feasibility rules (H3lp).
2. That the pre-registered heuristic replanning keeps realism (H4).
3. Any learned viaduct placement.


## 15. What did not work, or is not supported

1. **Leakage.** Urban DEMs contain the street beds, so TN partly reads the answer (`M12_RESULTS.md` §4).
2. **H4 failed.** Heuristic replanning fills the tile but lengthens routes beyond the pre-registered allowance.
3. **TN over-bridges** on synthetic water (42–60 per landscape) and ignores grade limits on steep scenarios.
4. **R starves the network.** It avoids violations by not building. It is not a success.
5. **Viaducts are rule-only.** There are 0–2 real examples to learn from. The exploratory HRV over-builds them (16
   decks in one city).
6. **TN+LR was not trained.** There is no defensible objective (design decision 9).
7. **`large_lake` is degenerate.** The lake fills the tile.
8. **Data.** San Francisco, MDT02 and FABDEM were not used. Japan has no terrain (GSI licence). Spanish DEMs show
   terracing.

## 16. Walk the cities (Godot 4)

```
G=~/Downloads/Godot.app/Contents/MacOS/Godot
C="$PWD/outputs/m12/cities"
$G --path godot/viewer -- --glb "$C/m12_wide_river3_TN_s0_HR_s2041/scene.glb"  --viewer "$C/m12_wide_river3_TN_s0_HR_s2041/viewer.json"
$G --path godot/viewer -- --glb "$C/m12_deep_valley0_TN_s0_HRV_s2041/scene.glb" --viewer "$C/m12_deep_valley0_TN_s0_HRV_s2041/viewer.json"
$G --path godot/viewer -- --glb "$C/m12_USA_SAN_e958_n7125_TN_s0_HR_s2041/scene.glb" --viewer "$C/m12_USA_SAN_e958_n7125_TN_s0_HR_s2041/viewer.json"
$G --path godot/viewer -- --glb "$C/big_TN_s0_HR_rolling_hills4_3x3_s2050/scene.glb" --viewer "$C/big_TN_s0_HR_rolling_hills4_3x3_s2050/viewer.json"
$G --path godot/viewer -- --glb "$C/big_TNlp_s0_HR_alpine_valley1_3x3_s2050/scene.glb" --viewer "$C/big_TNlp_s0_HR_alpine_valley1_3x3_s2050/viewer.json"
```

**Alpine showcase** (added 2026-11-05 on request; showcase only, not part of any result). `alpine_valley` is a
glacial U-valley: about 1,050 m of abrupt ridged relief (median slope 68%) around a flat, meandering floor about
250 m wide.
* Streets: TN-lp + HR; 429 segments, 76 steps dropped for grade.
* Buildings: 702 kept, all watertight and none floating; about 1,900 BF buildings removed on slopes over 35%.
* Most streets stay on the valley floor. On the walls, replanning builds only short, disconnected arms: it has no
  switchbacks, the known HR limit.
* BF parks are not slope-filtered, so some green blocks sit on cliffs.
* The landscape needed two new optional terrain parameters, `valley_power` and `valley_floor_damp`, plus
  `valley_meander_m`. All default to the previous behaviour; the 10 pre-registered scenarios are bit-identical
  (hash-checked).
* The Godot ray check now casts from above the scene's bounding box (it started at 500 m, inside these mountains).

Run these from the repository root: Godot's `--path` changes the working directory, so the paths must be
absolute. Controls: F toggles walk / fly; WASD move; Space jumps; the mouse wheel sets the speed; Q quits.

Other cities are in `outputs/m12/cities/`. The contact sheet is `outputs/m12/figures/cities.png`.

## 17. Decisions requiring approval (gate 3: final)

1. **The one-time held-out test evaluation:** `M12_TEST_AUTHORISED=yes bash scripts/m12_final_test.sh`.
   * Everything it uses is frozen: checkpoints, thresholds, normalisation, T = 0.5, metrics and criteria.
   * It builds `test_t` and fetches DEMs for the test tiles (USGS 3DEP and IGN; same licences as before).
   * Its output is reported as-is, once.
2. **Open items before any publication claim**:
   * the full text of CAVW 2024 (doi 10.1002/cav.2267), via KAIST access;
   * a library-database search (Scopus / Web of Science);
   * the Japanese GSI licence decision.
