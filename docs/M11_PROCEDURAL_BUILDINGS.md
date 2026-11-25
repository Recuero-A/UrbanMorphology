# Milestone 11 — Procedural buildings, façades and environment

Code: `src/urbanmorph/city3d/` (`heights.py`, `style.py`, `mesh.py`, `facade.py`, `environment.py`, `materials.py`).
Results: `MILESTONE_11.md` and `outputs/m11/evaluation.json`.

Nothing here is learned. Every rule is a documented, configurable assumption, and every attribute in the city file
says where it came from: `observed`, `derived`, `generated` or `synthetic`.

## 1. Building solids (`mesh.py`)

* **Footprint cleaning.**
  * `make_valid`, then keep the largest part.
  * Simplify at 0.05 m.
  * Drop holes under 1 m² and footprints under 10 m².
  * Orient: exterior counter-clockwise, holes clockwise.
* **Caps** use Shapely's constrained Delaunay triangulation. It handles concave outlines and courtyards, and adds
  no vertices, so cap and wall vertices coincide exactly. The code raises an error if a vertex ever appears.
* **Walls:** one quad per ring edge, holes included, so courtyards get inward-facing walls. Normals face outward by
  construction: the right of edge a→b on a counter-clockwise exterior or a clockwise hole.
* **Validity, checked per solid:**
  * every directed edge appears once and its reverse once (watertight and consistently oriented);
  * no edge shared by more than two faces;
  * no zero-area triangles;
  * signed volume > 0 (outward).

  The tests check exact volumes: rectangle, L, U, courtyard, gable, hip, pyramid.
* **Base:** solids start 5 cm below road level, sunk into the base plane, so nothing floats and there is no gap at
  the wall foot.

## 2. Heights and floors (`heights.py`)

| Priority | Source | Rule |
|---|---|---|
| 1 | `observed` | OSM `height` (metres; `m` and `ft` parsed), kept if 2–400 m. OSM height is to the top of the roof, so the eave = height − roof rise. |
| 2 | `derived` | `building:levels` × city storey height. That height is the city's median height ÷ levels where ≥ 500 buildings carry both tags (Tokyo 3.6 m, Osaka 3.2 m, Nagoya 4.1 m, New York 4.0 m); elsewhere a documented default of 3.0 m. |
| 3 | `synthetic` | Floors drawn from the empirical floor percentiles of the building's city (real tiles) or country (generated tiles), per footprint-area bin (< 60, 60–150, 150–400, 400–1,500, ≥ 1,500 m²), from **training tiles only**. |

**Spatial correlation (synthetic floors).**
* Each block gets one percentile u:
  * the mean percentile of the block's buildings with known floors, when there are any;
  * otherwise uniform.
* Each building draws u + N(0, u_sd).
* u is clipped to [0.01, 0.99]. Extreme training outliers (a 29-floor building) would otherwise set whole blocks.
* Footprints under 40 m² get at most 2 floors.
* **u_sd = 0.20**, calibrated on 80 **training** tiles so that the synthetic within-block floor spread matches the
  observed one (0.58 vs 0.63 floors; `outputs/m11/audit/height_calibration.json`).
  * The first value, 0.08, over-smoothed blocks: 0.47 against 0.80 observed, on validation.

**Coverage of observed or derived floors** (train + validation tiles; `outputs/m11/audit/height_audit.json`):

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

Chicago's handful of `height` tags (median 150 m, n ≈ 5) are excluded from the priors.

## 3. Roofs and styles from local context (`style.py`)

* **Typology** comes from floors, footprint area and attachment (touching another footprint within 0.3 m), never from
  country:

  | Typology | Rule |
  |---|---|
  | detached_low | ≤ 2 floors, not attached, < 400 m² |
  | attached_low | ≤ 3 floors, otherwise |
  | midrise | 4–8 floors |
  | highrise | ≥ 9 floors |

* **Roofs.**
  * P(pitched | typology) = 0.75 / 0.35 / 0.10 / 0.0; hip:gable = 0.4:0.6; pitch 25–35°; ridge along the long axis.
  * **This is an assumption, not data.** `roof:shape` is tagged on ≤ 3.4% of buildings in every city clip.
  * Pitched roofs need a near-rectangular footprint: symmetric difference with the minimum rotated rectangle
    ≤ 5% of the area.
  * Detached buildings may be **regularised** to that rectangle up to 15%, only if the rectangle touches no
    other footprint. The roof is flagged `footprint_regularised`, and the city file keeps the original footprint.
  * Everything else gets a flat roof with a 0.9 m parapet.
  * There is no straight skeleton, so no hip roofs on L-shapes yet.
* **Controlled variation.**
  * One façade material family per block, drawn by the block's dominant typology: plaster, brick, stone, concrete,
    wood siding.
  * A palette variant per building, within ±1 of the block's base.
  * Façade-grammar parameters are drawn per block, and each building departs from the block's binary choices
    (balconies, cornice, shop ground floor) with probability 0.15.
  * Neighbouring buildings are therefore related, not random.

## 4. Façade grammar (`facade.py`)

A small CGA-style rule set (Wonka et al. 2003; Müller et al. 2006), applied to the walls of the **actual solid**
(the rectangle, for pitched roofs):

```
Wall   -> occlusion(neighbouring footprints within 0.45 m)    -> visible runs   (party walls: nothing)
Run    -> split(y){ ground floor | upper floor* }              (ground 1.3 x storey, max 4.2 m, for shop typologies;
                                                                upper floors equal; no storey under 2.4 m)
Floor  -> repeat(x){ bay } between 0.6 m corner margins        (bay count = round(run / target width): equal bays)
Bay    -> door        (one per building: middle bay of the longest street-facing visible wall)
        | shop window (ground floor, street-facing, shop typologies)
        | window      (sill 0.9 m; width 0.42-0.60 x bay; height 0.50-0.62 x floor)
          [+ balcony: street-facing, floors >= 1, pattern all / alternate / first floor; 0.6 m slab + metal railing;
           skipped if it would reach into another footprint]
Wall top -> cornice band (alternating 1 cm offsets so the pieces at corners are never coplanar)
Flat roof -> parapet ring (0.25 m wide, 0.9 m high; a watertight solid)
```

* **Openings are panels, not cuts.** Glass and four frame strips sit 4 cm in front of the wall. Walls stay
  watertight, and the gap is above Eevee's depth precision at every camera used (§3 of
  `M11_RENDERING_PIPELINE.md`).
* **Checks, all counted in `validation.json` → `facade`:**
  * openings outside their wall rectangle: **0 by construction**;
  * party-wall length;
  * door bottoms at z = 0;
  * balcony clashes;
  * floors capped because OSM height and levels disagree.

## 5. Environment (`environment.py`)

| Element | Rule | Guard |
|---|---|---|
| Sidewalks | A band 3.0 / 2.5 / 2.0 / 0 m wide (major / mid / local / minor) beyond the road edge, minus roads and buildings, with slivers < 1.2 m removed. A raised slab, top at 0.15 m. | Only on free land. Some streets therefore have no sidewalk (where buildings meet the road). |
| Centre lines | 3 m dashes every 9 m on streets ≥ 8 m wide, none within (width + 4 m) of a junction. | Dash centre must lie on the road surface. |
| Crossings | Zebra stripes 3 m past each junction end. | Each stripe must lie on the road. |
| Street trees | Every 9 m along the sidewalk centre line (major / mid / local streets). | On sidewalk core, ≥ 2.5 m (canopy) from buildings, never on the road, ≥ 5 m from other trees. |
| Park trees | Minimum-distance sampling (9 m) in public green. | ≥ canopy radius from buildings. |
| Streetlights | Every 56 m, alternating sides, arm over the road. | On sidewalk, ≥ 0.5 m from buildings. |
| Private green | Shaded differently from public green; no trees placed. | — |
| Other open land | Left as ground. Empty parcels are **not** turned into grass. | — |

Trees and lights are **instanced**: one mesh each, with linked duplicates in Blender and shared meshes in the GLB.

## 6. Materials (`materials.py`)

* All procedural (Blender nodes: noise, brick, tile, bands).
* **No downloaded textures, so no licence issues.** Every material is defined in code.
* Colours are linear RGB converted from muted sRGB palettes.
* Façade families: plaster (6 tones), brick (5), stone (4), concrete (4), wood siding (5).
* Roofs: bitumen / concrete (3), terracotta / slate tile (4).
* Also: asphalt, pavement, ground, public and private grass, glass, frames, doors, trim, balconies, metal, lamp,
  road markings.
* **For the GLB**, procedural links are removed and each material keeps its base colour, roughness and metallic
  value. glTF has no procedural textures, so Godot shows flat colours.

## 7. Known limitations

1. **Roofs.**
   * Pitched roofs only on (near-)rectangles: no straight skeleton, so L-shaped houses get flat roofs.
   * Roof types are synthetic everywhere.
2. **Façades.**
   * Openings are panels on the wall, with no depth or reveals.
   * Gable walls and pitched-roof attics get no windows.
   * The same rules are used on all sides of a building (front and back differ only through the street and
     party-wall tests).
3. **Sidewalks follow the layout's road convention.** OSM road polygons in Barcelona often already include the
   pavement, and BF places buildings at the road edge. Both leave doors opening onto asphalt: 34% of doors in the
   real Barcelona tile, 20% in the generated Spanish tile.
4. **Detached generated buildings.** Generated Spanish tiles contain detached buildings (0.8 m of party wall per
   building, against 29.8 m in real Barcelona). This is a property of the M10 BF layouts that the 3D view makes
   obvious. It is not a façade rule.
5. **Heights.** The block-conditioned sampler is *less* accurate than the plain prior median on held-out floors
   (`MILESTONE_11.md` §5.2). It exists for coherent variation, not for prediction.
