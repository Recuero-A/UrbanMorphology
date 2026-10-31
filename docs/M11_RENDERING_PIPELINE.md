# Milestone 11 — Rendering, export and interactive exploration

Code: `blender/m11_scene.py` (runs inside Blender), `godot/viewer/` (Godot 4 project), `scripts/m11_godot.py`,
`scripts/m11_pipeline.py`. Every number below comes from `outputs/m11/<tile>/blender_timing*.json`,
`godot_check.json`, `godot_fps.json` or `outputs/m11/evaluation.json`.

## 1. Stages and commands

```
python scripts/m11_height_audit.py          # once: tag coverage            -> outputs/m11/audit/height_audit.json
python scripts/m11_height_priors.py         # once: floor priors (train)     -> outputs/m11/audit/height_priors.json
python scripts/m11_calibrate_heights.py     # once: within-block spread      -> height_calibration.json, priors["u_sd"]
python scripts/m11_pipeline.py              # build -> Blender -> Godot -> evaluation, six tiles
```

**Per tile:**
1. `scripts/m11_build_city.py` (project venv) writes:
   * `city.json` (format v0.1);
   * `meshes.npz` (one triangle group per material class: vertices, faces, per-face material and building id,
     per-corner UVs in metres);
   * `cameras.json`, `viewer.json`;
   * `validation.json` (+ per-building report).
2. `blender -b --factory-startup --python-exit-code 1 --python blender/m11_scene.py -- --tile <dir>` builds
   the scene and renders `renders/{aerial,oblique,street,closeup}.png`. It writes:
   * `scene.blend`;
   * `scene.glb`;
   * `blender_timing.json`.

   `--python-exit-code 1` matters: without it, Blender exits 0 even when the script fails.
3. `scripts/m11_godot.py` writes:
   * `godot_check.json` (headless import and collision test);
   * `godot_fps.json` (windowed 20 s benchmark).

**Reproducibility.**
* All random choices (heights, roofs, styles, trees) are made in step 1 from the city seed.
* Blender and Godot draw no random numbers.
* Two builds from the same seed give identical arrays (`tests/test_m11.py::test_scene_is_valid_and_deterministic`).

## 2. Coordinate frames and units

| Stage | Frame |
|---|---|
| city.json, meshes.npz, Blender | x east, y north, z up; metres; origin at the tile's SW corner |
| GLB, Godot | glTF Y-up: (x, y, z) → (x, z, −y); metres |

The exporter's `export_yup=True` does the conversion. A test exports a known box from Blender and reads the GLB
position bounds back: x 10…30, height −0.05…9.9, z −130…−120
(`tests/test_m11.py::test_glb_export_converts_z_up_metres_to_y_up`). In Godot, every tile's bounding box is
x ≈ 0…512, z ≈ −512…0 (`godot_check.json`).

## 3. Lighting and depth precision

**Daylight.**
* A physical sky (Blender 5.2 `MULTIPLE_SCATTERING`) for fill light, plus a sun at 42° elevation (4 W/m², 0.6°
  angular size, so shadows are soft).
* AgX view transform at exposure −1.
* **Calibration**, from test renders on 2026-11-01:
  * at sky strength 0.35, the sky alone was as bright as the sun alone and turned everything blue-grey with no
    visible shadows;
  * at 0.08, the sky gives blue fill in shadows, and exposure −1 stops mid-grey roofs clipping to white.

**Surround.** The physical sky is black below the horizon, so renders get a neutral ground frame *around* the tile.
It is never under the tile, because a plane 10 cm below the ground z-fought with the roads at 450 m camera distance.
It is excluded from the GLB.

**Depth layout** (no coplanar overlaps anywhere):

| Surface | Height |
|---|---|
| Base plane | −0.02 m |
| Roads and green | 0 (they never overlap) |
| Sidewalk slabs | to +0.15 m |
| Markings | +0.03 m |
| Façade panels | 4 cm in front of walls |

Camera near-clip is 1 m for the aerial and oblique views and 0.1 m at street level.

## 4. Rendering engines on Apple Silicon (M5 Pro, 16-core GPU, Metal)

Barcelona tile, about 700k triangles, 1600 × 900 (aerial 1600²):

| Engine, samples | Oblique | Street | Close-up | Note |
|---|---|---|---|---|
| Eevee, 64 (default) | 0.7 s | 0.7 s | 0.7 s | Aerial 1.0–1.4 s |
| Eevee, 256 | 2.6 s | 2.3 s | 2.3 s | |
| Cycles (Metal GPU), 64 + denoise | 98 s* | 3.9 s | 4.4 s | *first render includes one-time Metal kernel compilation |
| Cycles (Metal GPU), 256 + denoise | 5.8 s | 7.3 s | 10.3 s | |

* Eevee is 5–15× faster and is the iteration engine.
* Cycles gives physically correct soft shadows and inter-reflection (`renders/*_cycles256.png`). At this geometric
  detail, the visual difference is mostly shadow softness: the limiting factors are geometry and materials, not
  samples.
* Higher sample counts do not add realism the scene does not model (no façade depth, no weathering, no clutter).

## 5. GLB export

* Mesh objects only (no cameras or lights), modifiers applied, normals and UVs exported.
* **Materials** are made plain before export: Principled base colour, roughness and metallic, with the procedural
  node links removed. glTF cannot carry procedural textures.
* **Trees and streetlights** are linked duplicates, exported as many nodes sharing one mesh.
* Sizes per tile: `MILESTONE_11.md` §6.

## 6. Godot 4.7 viewer (`godot/viewer`)

**Loading.** The GLB is loaded at runtime (`GLTFDocument.append_from_file`), so no import step is needed per city.

**Collision.** Each mesh gets a trimesh collision shape (`create_trimesh_collision`).

**Controls.**

| Input | Action |
|---|---|
| Mouse | Look |
| WASD | Move |
| Space / Shift | Up / down (fly) or jump (walk) |
| F | Toggle fly ↔ first-person walking (capsule character, 1.8 m, gravity) |
| Mouse wheel or + / − | Change speed (1–400 m/s) |
| Esc | Release / capture the mouse |
| Q | Quit |

**Environment.** Sun with shadows (600 m distance), procedural sky, SSAO, AgX tonemapping.

**Run it:**
```
~/Downloads/Godot.app/Contents/MacOS/Godot --path godot/viewer -- \
    --glb "$PWD/outputs/m11/ESP_BCN_e837_n8952/scene.glb" --viewer "$PWD/outputs/m11/ESP_BCN_e837_n8952/viewer.json"
```

**`--check` (headless):**
* 64 downward rays over an 8 × 8 grid must hit the collision mesh;
* a walking capsule dropped 1 m above the street spawn must come to rest on the floor;
* the bounding box is reported.

**`--benchmark N` (windowed):**
* a fixed path: an aerial orbit for the first half, then a street-level glide;
* frame times are recorded after 1 s of warm-up, with draw calls, primitives and memory.

**Measurement limits.**
* macOS caps windowed frame rate at the display refresh (120 Hz ProMotion), so "120 fps" means *at least* 120.
* Godot's GPU timer returned 0 on the Metal backend, so it is reported as unavailable.
* The CPU render time per frame is reported instead.
