# Milestone 11 — 3D architecture and the structured city format

Status: designed in Phase A and **implemented** (2026-11-01). Differences from the design are listed in §3.3.
Results are in `MILESTONE_11.md`.

## 1. Environment audit (2026-11-01)

| Item | Found | Assessment |
|---|---|---|
| OS / CPU | macOS 26.6.2, arm64, Apple M5 Pro (5 performance + 10 efficiency cores) | — |
| GPU | Apple M5 Pro, 16-core GPU, Metal 4 | Eevee and Cycles (Metal) supported |
| RAM / disk | 48 GB / 574 GB free | Ample for single tiles, and for city-scale scenes later |
| Project Python | 3.12.2 venv: shapely 2.2.0 (GEOS 3.14.1), geopandas 1.2, pyproj 3.8, pyogrio 0.13 (GDAL 3.12.4, OSM driver), numpy 2.5, torch 2.15 | Constrained Delaunay with holes works (`shapely.constrained_delaunay_triangles`). No straight skeleton. No trimesh (not needed: §4.3). |
| Blender | **5.2.3 LTS** at `/Applications/Blender.app` (official build, 2026-10-22) | Headless `-b --python` works. Bundled Python 3.13.13 + numpy 2.3.4 (no shapely). Eevee render of a test scene: 2.4 s. Cycles devices: CPU and **METAL** (16-core GPU). glTF exporter `io_scene_gltf2` present. |
| Godot | **Not installed** | Needed only for Phase E (§6) |
| osmium | `/opt/homebrew/bin/osmium` | Used for the roof-tag audit |

**Installing Godot (Phase E; not done automatically).**
1. Download the latest **stable Godot 4.x, standard edition (not .NET)** for macOS from the official page:
   https://godotengine.org/download/macos/ (universal binary, Apple Silicon native, about 150 MB unpacked).
   Or: `brew install --cask godot`.
2. Move `Godot.app` to `/Applications`.
3. Check: `/Applications/Godot.app/Contents/MacOS/Godot --version`.
4. Godot 4 imports `.glb` natively, with no plug-in.

No other software is needed. Blender stays at 5.2.3 LTS: the pipeline uses only stable `bpy` API (mesh from
arrays, Principled BSDF, sun and sky, cameras, `export_scene.gltf`).

## 2. Pipeline and module boundaries

```
UrbanMorph 2D geometry ──► city.json (versioned, §3)          [project venv; shapely]
   (real OSM tile | generated M10 tile)
        │
        ▼
geometry validation ──► 3D generation ──► meshes.npz + validation.json   [project venv; numpy + shapely]
   (heights, roofs, façades, roads, sidewalks, ground, props)
        │
        ▼
Blender (bpy, headless): build objects, materials, lights, cameras ──► renders (PNG) + scene.glb + scene.blend
        │
        ▼
Godot 4: viewer project loads scene.glb (first-person / fly camera, collision)
```

* **The neural generator never imports Blender.**
  * It writes `city.json`.
  * The 3D stage reads only `city.json`.
  * Blender reads only `meshes.npz` plus a material table.
* **Why this split:**
  * Each stage can be replaced alone, for example a learned height or façade model replacing the procedural
    one.
  * All geometry code is unit-testable in the normal venv, without `bpy`.

**Planned code layout**

| Path | Contents |
|---|---|
| `src/urbanmorph/city3d/format.py` | City-format dataclasses, JSON read/write, schema version, validation |
| `src/urbanmorph/city3d/from_real.py` | Real OSM tile (validation split only) → city.json; observed parcels where licensed |
| `src/urbanmorph/city3d/from_generated.py` | M10 output (GraphTurtle N, T = 0.5 + BF) → city.json |
| `src/urbanmorph/city3d/heights.py` | Observed / from-levels / synthetic heights, with provenance |
| `src/urbanmorph/city3d/mesh.py` | Extrusion, triangulation, roofs, road / sidewalk / ground surfaces |
| `src/urbanmorph/city3d/facade.py` | Split-grammar façades (prototype 2) |
| `src/urbanmorph/city3d/validate.py` | Watertightness, manifoldness, normals, overlaps, floating objects |
| `scripts/m11_build_city.py` | Tile → city.json → meshes.npz + validation.json |
| `blender/m11_scene.py` | Runs inside Blender: npz → scene, renders, GLB export |
| `godot/viewer/` | Minimal Godot 4 project |

## 3. Structured city format, `urbanmorph-city` v0.1

* One JSON file per tile.
* **Units:** metres, and degrees for angles.
* **Frame:** the local tile frame (§3.2).
* **Geometry:** coordinate arrays, not WKT. A polygon is `[exterior, hole1, …]`, each ring a list of
  `[x, y]`, exterior counter-clockwise, holes clockwise.
* **Provenance:** every attribute that can be observed or inferred carries a source label, one of
  `observed` (from data), `derived` (computed deterministically from observed data, e.g. height from levels),
  `generated` (output of an UrbanMorph model), or `synthetic` (sampled by a procedural rule).

```jsonc
{
  "format": "urbanmorph-city", "version": "0.1",
  "metadata": {
    "tile_id": "ESP_BCN_e837_n8952" | "gen_N_T05_seed2026_03",
    "kind": "real" | "generated",
    "frame": "local_tile",                 // x east, y north, z up, metres, origin = tile SW corner
    "tile_size_m": [512, 512],
    "crs": "EPSG:25831" | null,            // real tiles only; null for generated (no geography invented)
    "origin_crs_xy": [428544.0, 4583424.0] | null,
    "seed": 2026 | null,
    "provenance": {"streets": "OSM 2026-10-28 (Geofabrik)" | "GraphTurtle N_noise_s0, T=0.5", "...": "..."},
    "units": {"length": "m", "angle": "deg"},
    "storey_height_m": {"value": 3.2, "source": "osaka median height/levels, n=12178"}
  },
  "streets": [{"id": "s12", "centerline": [[x, y], ...], "width_m": 8.0, "width_source": "observed|class_default",
               "class": "major|mid|local|minor", "nodes": ["n3", "n9"]}],
  "nodes":   [{"id": "n3", "xy": [x, y], "degree": 3, "border": false}],
  "blocks":  [{"id": "b4", "polygon": [...], "streets": ["s12", "s15"]}],
  "parcels": [{"id": "p88", "block": "b4", "polygon": [...], "street_access": true, "front_street": "s12",
               "use": "built|public_green|private_green|other_open", "source": "observed|generated"}],
  "buildings": [{"id": "B210", "footprint": [...], "parcels": ["p88"],
                 "height_m": 15.0, "height_source": "observed|derived_levels|synthetic",
                 "floors": 5, "floors_source": "observed|derived_height|synthetic",
                 "roof": {"type": "flat|gable|hip|skillion", "pitch_deg": 30, "ridge_axis_deg": 12, "source": "synthetic|observed"},
                 "style": {"facade_material": "plaster", "palette": 3, "bay_width_m": 3.6, "balconies": true,
                           "attached_edges": [2, 5], "source": "synthetic"}}],
  "landuse": [{"id": "g7", "polygon": [...], "class": "public_green|private_green|built|other_open",
               "source": "observed|generated"}]
}
```

### 3.1 Rules

* **Synthetic tiles get no geographic coordinates.** `crs` and `origin_crs_xy` stay `null`; the local frame
  is the only frame.
* **Real tiles:** the observed buildings, streets and green come from the tile's GeoPackage, already clipped
  and in local coordinates.
* **Restricted data.** Observed Spanish cadastral parcels may be used locally (licence: transformed use
  allowed, no redistribution). City files containing them stay in `outputs/m11/` and are **never committed**.
* **No test-split tiles.** Real prototypes use **validation** tiles.

### 3.2b Implemented v0.1: fields added during implementation (2026-11-01)

| Where | Field | Why |
|---|---|---|
| top level | `road_surface` (polygons) | The exact road surface used for blocks, sidewalks and meshes. Real tiles use observed OSM road polygons; generated tiles use the union of street pieces at class widths, identical to M10's `road_polygons`. |
| streets | `segment_widths_m`, `segment_classes`, `width_source` | Street class flickers along chains (M10 finding), so widths are stored per segment. `class` is the majority class. |
| buildings | `eave_m`, `typology`, `roof.rise_m`, `roof.footprint_regularised` | `height_m` is the top of the roof (the OSM convention); `eave_m` is the wall top. |
| buildings | `style.{bay_width_m, window_*_ratio, balconies, balcony_pattern, cornice, ground_floor, attached_to}` | Façade-grammar parameters (`M11_PROCEDURAL_BUILDINGS.md` §4). |
| parcels | `use_source` | The parcel polygon may be observed while its use is derived. |
| metadata | `parcel_model_country` (generated tiles) | Which BF parcel model built the tile. Not a geographic claim. |
| land use | `built` is not emitted as polygons | Building footprints are the built land. Land-use polygons are public green, private green and other open land. |

### 3.3 Differences from the Phase A design

* Cameras are placed in the Python stage (`city3d/cameras.py` → `cameras.json`), because a clear line of sight
  needs Shapely, which Blender's Python lacks.
* Blender reads only `meshes.npz` and `cameras.json`.

### 3.2 Coordinate frames

| System | Axes | Notes |
|---|---|---|
| city.json, meshes.npz | x east, y north, z up, metres | Origin at the tile's SW corner. Ground at z = 0. |
| Blender | x, y, z with Z up, metres (1 BU = 1 m) | Identical to the city frame: no transform. |
| glTF / GLB | Y up, metres, right-handed | The Blender exporter applies (x, y, z) → (x, z, −y), "+Y up". |
| Godot 4 | Y up, metres, −Z forward | Imports glTF unchanged: 1 unit = 1 m. |

A test round-trips a known point through the npz → Blender → GLB chain (§5).

## 4. Smallest viable implementation (prototype 1)

Goal: one **real** validation tile and one **generated** tile, each with extruded buildings, simple roofs, road
surfaces, basic materials, sun lighting, one aerial render, and one GLB.

### 4.1 Heights (transparent baseline, priority order)

1. **Observed:** the OSM `height` tag (metres; parsed for `m` and `ft`). Note that OSM `height` is to the
   top of the roof.
2. **Derived from levels:** `building:levels` × a per-city storey height.
   * Where a city has ≥ 500 buildings with both tags, its median `height / levels` is used:
     Tokyo 3.6 m, Osaka 3.2 m, Nagoya 4.1 m, New York 4.0 m.
   * Otherwise a documented default of **3.0 m** is used, flagged in metadata. Barcelona's 2.7 m and
     Madrid's 3.0 m come from only 21 and 31 buildings.
   * Source: `outputs/m11/audit/height_audit.json`.
3. **Synthetic:** sampled per **block**, so that neighbours are correlated.
   * Each block draws a base floor count from its city's observed-floors distribution in *training* tiles.
   * Each building varies around the block base by ±1 floor.
   * Footprints under 40 m² are capped at 2 floors.
   * Generated tiles have no city, so they use the country chosen for their BF parcels.

**Measured observed coverage** (train + validation tiles), share of buildings with a height or level tag:

| City | Share |
|---|---|
| New York | 0.98 |
| Los Angeles | 0.98 |
| Barcelona | 0.72 |
| Osaka | 0.68 |
| Chicago | 0.53 (its few `height` tags are unusable: median 150 m, n ≈ 5) |
| Nagoya | 0.50 |
| Valencia | 0.48 |
| Tokyo | 0.42 |
| Madrid | 0.21 |

`roof:shape` is tagged on at most 3.4% of buildings in any city clip, so **roofs are synthetic everywhere**,
and labelled so.

### 4.2 Geometry

* **Buildings**
  * Make each footprint valid (`make_valid`), drop slivers under 10 m², and simplify at 0.2 m.
  * Ground cap: constrained Delaunay triangles, which keeps holes and concave shapes.
  * Walls: one quad per ring edge, holes included, so courtyards get walls.
  * Roof cap at the eave height.
  * Faces outward; normals checked.
* **Roofs**
  * flat (any footprint);
  * gable and hip only where the footprint fills ≥ 85% of its minimum rotated rectangle, built over that
    rectangle and clipped to the walls' eave line;
  * otherwise flat;
  * pitch 25–35°, ridge along the long axis.
* **Roads**
  * The road surface is the union of centerline buffers at the class widths (the M6 `road_polygons`
    code), triangulated as **one** surface at z = 0.02 m.
  * One surface means no cracks at junctions, and no overlapping road faces (no z-fighting).
* **Ground**
  * Tile plane at z = 0.
  * Green land use at z = 0.01 m.
  * Parcels and other open land left as ground for now.

### 4.3 Validation (no new dependencies)

Computed on the arrays in `validate.py`:
* **Watertight:** every edge is used exactly twice, with opposite orientation.
* **Manifold vertices.**
* **Normals:** non-degenerate, and outward (signed volume > 0).
* **Areas:** no zero-area triangles.
* **Overlaps:** footprint pairwise overlap > 0.5 m² counts as intersecting buildings.
* **Floating objects:** minimum z ≠ 0.
* **Road/building intersection area.**

Results go to `outputs/m11/<tile>/validation.json`.

### 4.4 Blender stage

`blender --background --python blender/m11_scene.py -- --tile <dir> --seed <s>`:
1. Build one mesh object per material class (buildings, roofs, roads, ground, green). This batches objects
   and reuses materials.
2. Principled BSDF materials, controlled per-building colour variation through a vertex-colour attribute.
3. Sun plus Nishita sky.
4. Orthographic aerial camera; Eevee.
5. Export `scene.glb`.

## 5. Plan after prototype 1 (spec §21–25)

| Phase | Steps | Done when |
|---|---|---|
| B (core geometry) | Prototype 1 above. Tests: extrusion, roofs, triangulation, watertightness, frames, determinism. | Both tiles valid (0 non-watertight meshes, 0 floating). Aerial renders and GLBs exist. |
| C (architecture) | Split-grammar façades (floors → bays → window, door, balcony); party-wall occlusion (no openings on walls shared with neighbours); hip, gable and skillion variety; material library (procedural PBR, no downloaded textures). | Façade-consistency metrics (§7 of `MILESTONE_11.md`) computed; 0 openings crossing wall edges or party walls. |
| D (environment) | Sidewalks (frontage strip between road and parcel, only where space ≥ 1.5 m); trees in public green and along wide streets (instanced); streetlights; road markings; street-level and close-up cameras; Cycles vs Eevee comparison. | Street-level renders. Props never intersect buildings or roads. |
| E (interaction) | Godot installed by the user; `godot/viewer` with fly and first-person camera, mouse look, adjustable speed, collision from building and road meshes; FPS measurement. | GLB explorable; FPS logged. |
| F (evaluation) | Geometry validity, architectural plausibility, the real-vs-generated controlled comparison, performance. | `MILESTONE_11.md` with limitations. |

**No neural network is trained in M11.** The learned height model is listed as a justified *next* step, because
observed heights exist for 42–98% of buildings in 7 of 9 cities.
