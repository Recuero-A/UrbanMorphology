# Milestone 11: procedural 3D city generation and interactive visualization

**Status (2026-11-01).** A reproducible pipeline turns UrbanMorph 2D layouts, real or generated, into validated 3D
scenes. It renders them in Blender and exports them to GLB, which loads into a Godot 4 viewer with walking and flying.

* **Geometry.** Every building solid in all six tiles (2,823 buildings) is watertight and outward, with:
  * no floating objects;
  * no missing roofs;
  * no degenerate triangles;
  * no façade opening outside its wall.
* **Style.** The architecture is procedural (heights, roofs, split-grammar façades, sidewalks, trees, lights). No
  neural network was trained.
* **Scope of the claims.**
  * **Photorealism is not claimed.** No perceptual study was run.
  * Renders show what the rules produce. They are not evidence of morphological accuracy.
  * There were no pre-registered criteria for 3D quality; all numbers are descriptive.

Detail documents:
* `M11_3D_LITERATURE.md`
* `M11_3D_ARCHITECTURE.md`
* `M11_PROCEDURAL_BUILDINGS.md`
* `M11_RENDERING_PIPELINE.md`

Numbers come from `outputs/m11/evaluation.json`, the per-tile `validation.json`, `blender_timing*.json`,
`godot_*.json`, and `outputs/m11/audit/`. **Test tiles were never read**: real tiles are validation tiles, and
`m11_build_city.py` refuses any other split.

## 1. What was built

| Stage | File | Notes |
|---|---|---|
| Environment and tag audit | `scripts/m11_height_audit.py` | macOS 26.6, M5 Pro, 48 GB, Blender 5.2.3 LTS (Eevee and Cycles on Metal), Godot 4.7.2; heights and `roof:shape` coverage |
| City format v0.1 | `city3d/format.py`, `assemble.py`, `convert.py` | Streets with connectivity, blocks, parcels with street access, buildings, land use, provenance; synthetic tiles get no geographic coordinates |
| Heights | `city3d/heights.py`, `scripts/m11_height_priors.py`, `m11_calibrate_heights.py` | observed → derived from levels → synthetic, block-correlated, calibrated on training tiles |
| Roofs and styles | `city3d/style.py` | From local context (floors, area, attachment), not nationality |
| Solids | `city3d/mesh.py`, `validate.py` | Constrained Delaunay caps; L, U, courtyards; flat, gable, hip, pyramid |
| Façades | `city3d/facade.py` | Party-wall occlusion → floors → bays → window / door / shop / balcony; cornice; parapet |
| Environment | `city3d/environment.py` | Sidewalks (raised, only on free land), markings, crossings, street and park trees, streetlights (instanced) |
| Scene and cameras | `city3d/scene.py`, `cameras.py` | Batched groups, procedural PBR material table, validation report |
| Blender | `blender/m11_scene.py` | Sky + sun, 4 cameras, Eevee or Cycles, `.blend` + GLB |
| Godot | `godot/viewer/`, `scripts/m11_godot.py` | Runtime GLB loading, trimesh collision, fly / walk, headless check, benchmark |
| One command | `scripts/m11_pipeline.py` | Build → Blender → Godot → evaluation for the six tiles |
| Tests | `tests/test_m11.py` | 21 tests; full suite 167 passed |

## 2. Tiles

**Real (validation split).**

| Tile | Character |
|---|---|
| Barcelona `ESP_BCN_e837_n8952` | Dense perimeter blocks, observed cadastral parcels (local use only) |
| Los Angeles `USA_LAX_e758_n7370` | Low-density suburban |
| New York `USA_NYC_e1149_n8784` | Dense rowhouses |
| Tokyo `JPN_TYO_e764_n7739` | Dense mixed residential |

**Generated (M10 recommendation: GraphTurtle N at T = 0.5, then BF).**

| Tile | Parcel model |
|---|---|
| `gen_N_T05_s2026_00_spain` | Spanish BF |
| `gen_N_T05_s2026_01_united_states` | US BF |

Synthetic attributes use attribute seed 0.

Renders, GLBs and `.blend` files are in `outputs/m11/<tile>/` and are not committed: they are generated assets,
and the real tiles' city files contain licence-restricted parcels.

## 3. Data audit: heights and roofs

**Heights.** Share of buildings with an OSM `height` or `building:levels` tag, train + validation tiles:

| City | Share |
|---|---|
| New York | 0.98 |
| Los Angeles | 0.98 |
| Barcelona | 0.72 |
| Osaka | 0.68 |
| Chicago | 0.53 |
| Nagoya | 0.50 |
| Valencia | 0.48 |
| Tokyo | 0.42 |
| Madrid | 0.21 |

* Implied storey height (height ÷ levels) varies by city: 3.2 m in Osaka to 4.1 m in Nagoya, where ≥ 500
  buildings carry both tags (Tokyo 3.6 m, New York 4.0 m).
* So there is no universal floor height; 3.0 m is used only as a flagged default.
* In the rendered real tiles, 92–100% of buildings have observed or derived heights.

**Roofs.** `roof:shape` is tagged on ≤ 3.4% of buildings in every city clip, so **all roofs are synthetic**.

## 4. Geometric correctness (all six tiles)

| | BCN | LAX | NYC | TYO | gen ES | gen US |
|---|---|---|---|---|---|---|
| Buildings | 578 | 334 | 772 | 605 | 215 | 319 |
| Non-watertight / inward / non-manifold / degenerate | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| Floating / missing roof / roof fallback | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| Parapets / sidewalks watertight | 571/571, 199/199 | 230/230, 201/201 | 498/498, 118/118 | 351/351, 225/225 | 182/182, 135/135 | 153/153, 247/247 |
| Intersecting building pairs (> 0.5 m²) | 2 | 0 | 1 | 0 | 0 | 0 |
| Buildings on the road surface (> 1 m²) | 93 | 17 | 0 | 59 | 0 | 0 |
| Road surface coverage, down-facing triangles | 1.000, 0 | 0.999, 0 | 1.000, 0 | 1.000, 0 | 0.999, 0 | 0.999, 0 |
| Props inside buildings, props on roads | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| Façade openings outside their wall | 0 | 0 | 0 | 0 | 0 | 0 |

**The overlaps and road conflicts are in the source data, not the mesher.**
* Real tiles: OSM building footprints overlap OSM road polygons, whose widths are partly estimated
  (Barcelona's road areas often include the pavement).
* Generated tiles have none, because BF places buildings by construction.

**Crack-free roads.**
* Roads are one triangulated surface (coverage 0.999–1.000 of the source polygon).
* Nothing is coplanar with anything else, so there is no z-fighting by construction.

## 5. Architectural plausibility

### 5.1 Façades

| | BCN | LAX | NYC | TYO | gen ES | gen US |
|---|---|---|---|---|---|---|
| Doors: buildings with exactly one door at z = 0 | 578/578 | 334/334 | 772/772 | 604/605 (1 building has no visible wall wide enough) | 215/215 | 319/319 |
| Upper-storey height, 5–50–95% (m) | 2.8 / 3.0 / 3.0 | 2.4 / 2.9 / 3.6 | 2.7 / 3.4 / 5.5 | 2.6 / 3.2 / 4.8 | 2.8 / 3.0 / 3.0 | 2.8 / 3.0 / 3.0 |
| Storeys outside 2.4–4.5 m | 0% | 2% | 14% | 8% | 0% | 0% |
| Bay width 10–50–90% (m) | 2.0 / 3.4 / 4.6 | 2.2 / 3.1 / 4.3 | 2.3 / 3.1 / 4.4 | 2.3 / 3.2 / 4.2 | 2.8 / 3.7 / 4.7 | 2.4 / 3.2 / 4.0 |
| Doors opening onto road (no sidewalk) | 34% | 7% | 0% | 12% | 20% | 20% |
| Block façade-family agreement (median) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Distinct façade configurations per building | 0.92 | 0.87 | 0.80 | 0.87 | 0.93 | 0.92 |

* **Floors are consistent by construction:** equal upper storeys, a taller ground floor only for shop typologies,
  and no storey under 2.4 m.
* **Storeys outside 2.4–4.5 m** come from real buildings whose observed OSM height implies tall storeys (for
  example, 11 m with 2 levels in New York).
* **Window spacing is regular within a façade** (equal bays) and varies between blocks.
* **Party walls get no openings:** a test covers this. The real Barcelona tile has 17.2 km of party wall.

### 5.2 Heights: does the block-correlated synthetic baseline help?

**Setup.** Holdout on the 15 validation tiles with ≥ 50 buildings of known floors. 20% of known buildings are
hidden, over 5 folds. Tile medians of the mean absolute error, in floors:

| Method | MAE (floors) |
|---|---|
| Prior median per area bin (deterministic) | **0.72** |
| Block-conditioned median (neighbours' percentile, no noise) | 0.80 (better than the prior median on 5 of 15 tiles) |
| Independent draw from the prior | 0.96 |
| Block-conditioned draw (the generator, u_sd 0.20) | 1.17 (better than independent on 6 of 15) |

**Within-block floor spread (median SD).**

| Floors | SD |
|---|---|
| Observed | 0.80 |
| Block-correlated synthetic | 0.58 |
| Independent synthetic | 0.79 |

**Findings.**
* **Neighbour information does not beat the plain prior median** at predicting held-out floors.
* Sampling noise costs accuracy, as expected for a generator.
* Even after calibration on training tiles, the block-correlated sampler is *more uniform within blocks* than
  observed data. Independent sampling matches the observed spread better.
* The spatial-correlation assumption from the specification ("avoid assigning heights independently") is **not
  supported by these data at block level** for floor counts.
* It is kept for visual coherence (neighbours look related). A learned, context-aware height model (§8) should
  be judged against the prior-median baseline.

### 5.3 Roofs

| | BCN | LAX | NYC | TYO | gen ES | gen US |
|---|---|---|---|---|---|---|
| Pitched share | 1% | 31% | 35% | 42% | 10% | 52% |

* All pitched roofs sit on near-rectangles, regularised up to 15% for detached houses.
* Regularising never added road overlap: 36 houses in LA, 0 m² extra.
* Irregular footprints fall back to flat roofs.

## 6. Performance (M5 Pro, 48 GB; Eevee 64 samples; 1600 × 900, aerial 1600²)

| | BCN | LAX | NYC | TYO | gen ES | gen US |
|---|---|---|---|---|---|---|
| Triangles (instances) | 689k (362) | 126k (867) | 248k (985) | 249k (638) | 323k (365) | 165k (972) |
| Layout → city.json (s) | 0.7 | 0.5 | 0.5 | 0.5 | 2.4 (incl. street sampling + BF) | 3.0 |
| Meshes + validation (s) | 5.5 | 1.7 | 3.2 | 3.2 | 2.0 | 5.5 |
| Eevee render, aerial / oblique / street / close-up (s) | 1.2 / 0.8 / 0.7 / 0.7 | 1.1 / 0.7 / 0.6 / 0.6 | 1.2 / 0.8 / 0.7 / 0.6 | 1.1 / 0.8 / 0.7 / 0.6 | 1.1 / 0.8 / 0.7 / 0.7 | 1.2 / 0.8 / 0.6 / 0.6 |
| Blender peak memory (MB) | 1,159 | 770 | 893 | 869 | 899 | 812 |
| GLB / .blend (MB) | 57.5 / 20.5 | 9.8 / 4.1 | 20.0 / 8.6 | 20.0 / 8.3 | 27.6 / 10.2 | 13.0 / 5.3 |
| Godot load / collision build (ms) | 164 / 663 | 49 / 134 | 83 / 255 | 66 / 230 | 77 / 291 | 64 / 162 |
| Godot frame rate (20 s path) | ≥ 120* | ≥ 120* | ≥ 120* | ≥ 120* | ≥ 120* | ≥ 120* |
| Godot CPU render time (median, ms) / draw calls | 0.77 / 263 | 0.97 / 276 | 0.91 / 226 | 0.86 / 261 | 0.58 / 172 | 0.90 / 224 |

\* Capped by the 120 Hz display. Godot's GPU timer returns 0 on Metal, so GPU time is unavailable.

**Cycles** (Metal GPU, Barcelona) takes 3.9–10.3 s per view at 64–256 samples with denoising, against 0.7–2.6 s
for Eevee. The first Cycles render adds about 90 s of one-time kernel compilation.

**Whole pipeline:** about 4–9 s build plus 4 s Blender per tile; Godot check and benchmark 145 s for all six.

**Optimisation used:** batching (one object per material class), material reuse (a shared table), and instancing
(trees and lights). There was no need yet for level of detail or culling, since one tile runs at the display cap.

## 7. Real vs generated layouts, rendered by the same pipeline

The renderer is identical for both, so differences come from the 2D layouts:

| | BCN real | gen ES | LAX real | NYC real | gen US |
|---|---|---|---|---|---|
| Buildings | 578 | 215 | 334 | 772 | 319 |
| Height median / 90% (m) | 18 / 30 | 18 / 33 | 6 / 11 | 9 / 11 | 6 / 12 |
| Party wall per building (m) | **29.8** | **0.8** | 0.4 | 2.4 | 0.4 |
| Windows per building | 56 | 102 | 22 | 24 | 30 |
| Doors onto road | 34% | 20% | 7% | 0% | 20% |
| Pitched roofs | 1% | 10% | 31% | 35% | 52% |

* **Shared walls.** The 3D view exposes a layout difference that the 2D building metrics of M9/M10 only hinted
  at. Real Barcelona is continuous perimeter blocks (about 30 m of shared wall per building). The generated
  Spanish tile is detached buildings standing apart, with more exposed façade (twice the windows per building).
* **Heights** match the real Spanish tile well. That is by construction: synthetic heights are drawn from Spanish
  training priors.
* **US generated tile:** closer to Los Angeles than to New York in party walls and heights. It has more pitched
  roofs because more of its footprints are rectangles (BF templates).

## 8. Completion criteria (specification §25)

| # | Criterion | Status |
|---|---|---|
| 1 | Valid 3D scene from an existing 2D layout | **Met**: 6 tiles, 0 invalid solids |
| 2 | Plausible heights and roofs | **Partly met.** Heights are observed or derived for 92–100% of real buildings; synthetic heights match priors but are less accurate than the prior median (§5.2). Roofs are synthetic (no data); pitched roofs only on near-rectangles. |
| 3 | Roads and buildings align | **Met for generated tiles** (0 conflicts). Real tiles keep OSM's own building/road overlaps (17–93 buildings), which are reported. |
| 4 | Architectural details geometrically consistent | **Met**: 0 openings outside walls, no openings on party walls, consistent floors, watertight parapets. One door per building at ground level, except 1 of 2,823 buildings with no visible wall wide enough for a door. |
| 5 | Blender aerial and street-level renders | **Met**: 4 views per tile, Eevee and Cycles |
| 6 | Interoperable export | **Met**: GLB; frame conversion tested |
| 7 | Explorable in Godot | **Met**: runtime loading, fly / walk with collision; checked headless on all tiles |
| 8 | Geometry validation tests pass | **Met**: 21 M11 tests; 167 total |
| 9 | Reproducible on Apple Silicon | **Met**: one command, seeded, deterministic arrays (tested) |
| 10 | Results and limitations documented | **Met**: this file and the detail documents |

## 9. Limitations

1. **No visual-realism study.** Realism was judged by inspection only. The renders are stylised: flat-colour
   façades in the GLB, panel windows with no depth, low-poly trees.
2. **Synthetic roofs everywhere.** Only rectangles get pitched roofs (no straight skeleton), and there are no
   dormers or chimneys.
3. **Heights.** The block-correlated synthetic sampler is less accurate than the prior median. Block-level
   correlation of floors is weaker than assumed (§5.2).
4. **Sidewalks depend on the road convention.** Where OSM road polygons include pavements (Barcelona), or BF
   puts buildings at the road edge, doors open onto asphalt (20–34%).
5. **Generated layouts lack attached buildings and have no Japanese parcel model.** These are upstream (M10)
   limits that the 3D view makes visible.
6. **Single tile, flat terrain.** There are no tile seams, terrain or LOD yet.
7. **Measurement limits.** Frame rate is capped at 120 Hz, and GPU time is unavailable on Metal in Godot 4.7.
8. **Cycles brick colour.** Cycles brick read grey at first. The cause was light mortar covering a third of the
   area, and it was fixed before the final renders.

## 10. Recommended next steps

1. **Attached-building generation (upstream).** The largest real-vs-generated gap seen in 3D is party walls: 30 m
   against 0.8 m per building in Spain. A BF variant with perimeter-block typologies, or a learned block typology,
   would close it.
2. **Learned height model.** Justified now: observed or derived heights exist for 42–98% of buildings in 8 of 9
   cities. It must beat the prior-median MAE of 0.72 floors (§5.2) on held-out validation buildings.
3. **Straight-skeleton roofs** for irregular footprints. This would remove the rectangle restriction (pure Python,
   or a small dependency).
4. **Façade depth.** Recessed openings via wall splitting, and texture baking for the GLB, so Godot shows the
   procedural textures.
5. **Multi-tile cities.** Stitch neighbouring tiles (generated tiles need seam-aware streets), with LOD and
   occlusion culling.
6. **A small perceptual study** (real vs generated renders, blind pairwise), before any visual-realism claim.
