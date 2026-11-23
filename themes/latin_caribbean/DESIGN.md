# Latin Caribbean theme: design

A rendering theme for the 3D cities, kept outside the milestones. The same city file (streets, parcels, buildings,
land use) renders in two ways:

* `standard`: the Milestone 11 buildings;
* `latin_caribbean`: a Spanish-colonial Caribbean town of colourful lime-plastered houses with tile roofs, wooden
  and iron balconies, domed churches on plazas, palms, flowers and jungle, cobbled streets, and beaches below
  jungle mountains.

**References.** The three images in `architectural features/` are the brief. IMG_3376 is the favourite: a hillside
sea town at sunset with a domed coral church. IMG_3377 is the ochre and coral hill town with blue domes. IMG_3378
is Cartagena: magenta, yellow and turquoise façades with wooden balconies. `RESEARCH.md` collects the web research:
Cartagena rules, 14 reference churches, plants, terrain and lighting, each claim tagged [V]/[S]/[E]/[R] for how far
to trust it.

## 1. How to use it

| What | Command |
|---|---|
| Kit preview (hand-laid showcase street) | `.venv/bin/python themes/latin_caribbean/scripts/preview_kit.py [--seed 3] [--light sunset\|midday]` |
| Coastal town (mountains + beach; improved NA-fixed streets) | `.venv/bin/python themes/latin_caribbean/scripts/coastal_town.py [--method NA-fixed] [--seed 1]` |
| KAIST jungle village (M\* + organic, F1 + F4; real main roads; dense jungle; rivers) | `.venv/bin/python themes/latin_caribbean/scripts/kaist_village.py [--dense]` (`--dense`: colonial row houses, plazas with churches) |
| KAIST village, another street generator, giant towers | `kaist_village.py --method NA-adaptive --version original --giant 165,125,95 --giant-lottery 60,85` (see §5.1) |
| Any big 3 × 3 city with the theme | `scripts/m16_city3d_big.py ... --theme latin_caribbean [--terrain-npz domain.npz]` |
| In code | `city3d.scene.build_scene(city, theme="latin_caribbean")`, then `apply_terrain`, then `latin.build.dress_terrain` |
| Blender light | `blender/m11_scene.py -- --tile DIR --light sunset` (default for the theme), `midday`, or `default` (M11) |
| Godot light | `godot/viewer -- --glb ... --light sunset\|midday` |

* Textures are generated procedurally (`city3d/latin/textures.py`; no downloaded assets) into `textures/`, and
  regenerated on demand.
* Previews go to `previews/`. Theme tests: `tests/test_latin_theme.py`.

## 2. Palette

The palette is set in `city3d/latin/palette.py`. Colours were sampled from the references and corrected for their
sunset tint (RESEARCH.md §2 suggests desaturating 15–25 % for a warm sun).

| Role | Colours | Rule |
|---|---|---|
| Walls | amarillo `#f2c14e`, ocre `#e9a93f`, naranja `#e58a3f`, coral `#e5694a`, salmón `#ef9877`, terracota `#c9573a`, rosa `#e0577f`, rosa palo `#f0a3a8`, turquesa `#45b3ad`, azul `#5f97b4`, celeste `#8ccbe0`, menta `#9fd1a2`, lila `#b19ad0`, crema `#f3dfb8`, blanco `#f3eee4` | Weighted, with warm colours most frequent. An attached neighbour never repeats the colour. White and cream stay at about 14 % or less (RESEARCH.md §7) |
| Trims | blanco 75 %, crema 20 %, gris 5 % | Pilasters, cornices, window and door surrounds |
| Plinth (zócalo) | stone grey, terracota, dark blue, dark green, white, dark ochre | On 70 % of houses, 0.8–1.2 m high |
| Carpentry | mahogany, light and dark wood, green, turquoise, blue, white | The documented Cartagena set |
| Roofs | barrel tiles, new and weathered (moss) | 65 / 35 % |
| Domes | terracotta fish-scale tiles, blue, yellow, white | Plus white ribs on 80 % |

## 3. Textures (light texture on vibrant colour)

* **Grey textures tinted by the material colour:** plaster, wood, doors, shutters, balusters. In glTF this is
  baseColorTexture × baseColorFactor, so one image serves every wall colour.
* **Coloured textures:**
  * paving: barrel tiles, dome tiles, cobbles, stone pavers, basket-weave plaza bricks;
  * ground: sand, grass, soil, jungle floor;
  * canopies: leaves, bougainvillea, flamboyant.
* **Cut-outs (alpha mask):** palm fronds, fan leaves, banana leaves, flower cards, turned balusters, wrought-iron
  railings.
* **Seamless:** every surface texture uses periodic noise and periodic Voronoi.
* **Mapping:** walls are UV-mapped in metres. Tile roofs follow the slope (u along the eave, v up the slope).
  Domes follow the meridians.

## 4. Houses (`city3d/latin/kit.py`, styles in `build.py`)

* **Heights.** Ground floor 3.8–4.3 m, upper floors 3.3–3.7 m (colonial ceilings). At most the city's own floor
  count, mostly 1–3 floors.
* **Roofs.**
  * Detached near-rectangles: hip (55 %), gable (30 %) or flat (15 %), with 0.45–0.65 m eaves, fascia, ridge and
    hip caps.
  * Any other footprint: a tiled skirt sloping out from a flat clay azotea, mitred at every corner. This gives the
    sea of terracotta seen from above.
  * Flat roofs get a parapet and coping.
* **Façades.** Only on visible wall runs; party walls stay blank.
  * Plinth band, corner pilasters, string courses and a two-step cornice.
  * Doors: wooden double doors with moulded surrounds, some arched, on a stone step.
  * Windows: tall, with mouldings, sill and lintel, and open or half-closed louvred shutters.
  * Ground floor of street walls: Cartagena projecting wooden grilles (*rejas*), half with a small tile roof
    (*tejaroz*).
* **Balconies.** One sub-style per block (55 % Cartagena wood, otherwise iron).
  * Wood: long balconies on corbels with turned balusters, posts and a tile roof.
  * Iron: small slab balconies with wrought-iron railings.
  * Only on the first upper floor of street walls, never on every floor (RESEARCH.md §7).
* **Cost.** About 20 triangles per window: mouldings are front, top and bottom faces only, and the surrounds are flat.

### 4.1 Dense colonial infill (`city3d/latin/infill.py`; `--infill colonial`)

The Milestone 10 parcels and buildings are sparse for a colonial town. With `--infill colonial`, every street block
of the village is rebuilt as attached row houses.

* **Sidewalk.** Colonial sidewalks are 1.6 m (local and minor streets), 1.8 m (mid) and 2.5 m (major).
* **Lots.**
  * Each block gets lots 7–12 m wide along the street, in a perimeter band 11–15 m deep.
  * The lots are Voronoi cells of seeds spaced along the line half a band-depth inside the block edge. Their
    boundaries are perpendicular to the street, and corner lots take the corners.
  * A block thinner than the band is cut across its whole depth (houses with two street fronts). A block under
    45 m² holds one house.
* **Patios and paving.** The inside of the band becomes a patio garden. Everything else in the block (sidewalk edge,
  leftovers) is paved.
* **Lanes** (`--lanes M`, `infill.split_block`; added 2026-11-23). Some street generators leave blocks far larger
  than a manzana: NA-adaptive left patios of up to 9 ha, 28 ha in all. With this option, a block larger than
  1.6 × M² is first cut by straight paved lanes (callejones), 4 m wide, about every M m along both axes of its minimum
  rectangle. The lanes join the streets around the block. Each sub-block then gets its own ring of lots and its own
  patio. With M = 60, a 300 × 300 m block becomes 25 sub-blocks with patios of at most about 1,200 m².
* **Plazas** (`infill.plaza_sites`, three per map, at least 280 m apart). Large compact blocks (1,500–7,000 m²) deep
  inside the village and within one tile.
  * A church (≤ 44 × 22 m) takes one end of the block, its façade toward the block centre.
  * The rest is a paved plaza with royal palms, a fountain and flamboyants.
  * The usual church lottery adds a few more churches elsewhere.

KAIST village with the infill: 3,361 row houses on 910 blocks, 38 patios, 3 plazas, 6 churches (5.7 times the
M10 layout's 593 buildings).
* **Godot (1600 × 900, sunset):** 102 fps on average, p95 frame 86 fps, 27 s to load.
* **Size:** 3.2 M triangles and 21.8 k plants.

## 5. Churches (`city3d/latin/church.py`)

* **Site.** A building merged with its attached neighbours, fitted with a rectangle at least 20 × 11 m. At most one
  per 512 m tile, at least 260 m apart. The façade faces the nearest street.
* **Plan and body.**
  * Nave of 8–16 m, eave 9–15 m, gable tile roof.
  * Baroque façade rising above the roof: four pilasters on pedestals, a two-step entablature, curved shoulders, a
    pediment, an arched portal between columns, an oculus or choir window, a niche, pinnacles, an iron cross and
    entrance steps.
* **Towers.** One (beside the façade) or two.
  * Each has a shaft with slit windows, one or two belfry stages with arched openings, and corner pinnacles.
  * Caps: a cupola on a drum (or a bulb), or a tile pyramid, then a lantern and a cross.
* **Crossing.** A 16-sided drum with arched windows and pilasters, a ribbed dome (radius 0.4 × nave width) and a
  lantern (0.2 × the drum radius).
* **Optional parts.** Transepts and a semicircular apse.
* **Size.** A crossing dome reaches about 36 m. That is 2–2.6 × the eave, the research range for towers
  (RESEARCH.md §3.2).

### 5.1 Giant towers (`church.giant_tower`; user request 2026-11-23, "futuristic tropical Caribbean")

Not a colonial building type. It is the Cartagena cathedral tower and the towers of IMG_3376 / IMG_3377 scaled to
60-165 m, keeping the colonial ornament.

* **Site.** One tower on the left of the façade, base side 0.45 × the church width (7-15 m). The nave keeps at least
  8 m. With `--giant`, the plaza church sites may be up to 34 m wide (22 m otherwise).
* **Shaft** (about 45 % of the height):
  * stone plinth, white corner pilasters, white string courses about every side length;
  * one or three tall arched windows per panel, small iron balconies on every other panel;
  * a clock with a gilt ring in the top panel.
* **Four stages, each on a balustraded terrace with pinnacles at its corners.**
  * Two square belfries (0.78 and 0.62 × the base side): two or one arched openings per face, at most 2.8 × as tall
    as wide, with an oculus below when the stage is tall.
  * Two octagonal stages in an accent colour (white, cream or yellow): arched openings on every face, then on every
    other face.
  * The two lower terraces are roof gardens: fan palms at the corners and bougainvillea. They are instances placed on
    the building base by `dress_terrain`.
* **Top.** A ribbed tile cupola, a lantern, a gilt ball and a gilt cross.
* **Height.** Stage heights stretch up to 1.5× so the upper part is about 55 % of the total height.
* **Triangles.** About 10-15 k per tower.
* **Heights.** `kaist_village.py --giant 165,125,95` sets the plaza churches, tallest first.
  `--giant-lottery lo,hi` gives the other churches (sites at least 15 m wide) a tower of uniform random height.
* **Views.** The script adds three views to `cameras.json` (Blender) and `viewer.json` (`shots`, Godot `--shot`):
  * `tower`: from the far side of the main plaza, looking up;
  * `alley`: down a straight street with a clear line of sight to the tallest tower;
  * `skyline`: from the sun side, over the jungle.
  The Godot walk starts in the main plaza, facing the tower.

## 6. Plants and furniture (`city3d/latin/plants.py`)

| Prototype | Triangles | Where |
|---|---|---|
| Coconut palm (straight, leaning) | 444 | Mid streets, gardens, parks, jungle edge, beach (leaning) |
| Royal palm (swollen grey trunk, green crownshaft) | ~560 | Major streets, plaza rings |
| Fan palm, banana | 84 / 372 | Gardens, parks, jungle |
| Bougainvillea, shrub | 96 | Gardens, plaza edges, jungle understorey |
| Flamboyant (wide flat crown) | ~1,400 | Plaza centres, parks |
| Jungle trees (two greens) | ~500 | Free land at least 28 m from the town, every ~8 m, with shrubs every 6 m |
| Flower cards (heliconia, hibiscus colours) | 6 | Plaza edges, gardens, parks |
| Colonial iron lamp | 44 | Every street, ~39 m apart |

* **Dense jungle** (`--jungle dense`, `--village-npz`): outside a village mask there are no buildings or yards.
  A closed canopy of 11 m patches (190 triangles each) every ~8.5 m, with emergent palms and flamboyants every
  ~22 m, reaches to 12 m from the buildings and 5 m from the roads.
* **Plazas.** Public green smaller than 6,000 m² is paved as a brick plaza with a ring of royal palms, a stone
  fountain (an octagonal basin with an upper bowl, 136 triangles), flamboyants and flowers. Larger green is a
  tropical park.
* **Open land in town** (lots without buildings, verges): scattered palms, bananas, bushes and the odd flamboyant.
* **Streets.** Cobbles, except asphalt with markings on major streets.

## 7. Terrain (`latin.build.dress_terrain`, scenario `caribbean_coast`)

* **Dressing.** Sand on the terrain up to 2.8 m above the sea and within 90 m of it. Rock above 38°. Jungle floor on
  free land. Turquoise water. No jungle on the sand. Leaning coconut palms along the beach.
* **The `caribbean_coast` landscape** (showcase-only, never used in any evaluation):
  * sea and a bay to the south;
  * a 3 % coastal plain with ±20 m hills and a 3–4 % beach band;
  * mountains to about 310 m on the north edge, made by an inverted valley with amplified ridged relief;
  * streams from drainage.

## 8. Lighting and performance

| | Blender | Godot |
|---|---|---|
| **Sunset (default)** | Sun 8° high, colour (1, 0.56, 0.3); multiple-scattering sky; AgX Punchy | Sun −8° `#ffb46b`; violet `#6a5f93` to amber `#f3b95d` sky; fog `#e9a36a` with aerial perspective and sun scatter; glow; saturation 1.12 |
| **Midday** | Sun 62° high | Sun −62°; light blue haze |

**Performance (Godot).**
* Plants and lamps arrive as glTF linked duplicates. The viewer turns each shared mesh into MultiMesh chunks of
  128 × 128 m, so draw calls stay low and frustum culling works per chunk.
* Small plants (flowers, shrubs, bougainvillea, banana) and lamps fade out beyond 380 m and cast no shadows.
* No collision on glass, carpentry or metal.
* Textures are 512² (fronds 256 × 1024).
* About 1,100 triangles per house and 3–6 k per church.

Blender rounds crowns, trunks and domes with smooth-by-angle shading.

**Ocean.** When the map has sea, `cameras.json` / `viewer.json` carry `sea_z`. Blender (render only) and Godot then
add a water plane to the horizon.

**Measured** (coastal town, 3 × 3 tiles, 837 buildings, 1.74 M triangles, 27.6 k plants in 1,304 MultiMesh chunks;
Godot 4.7.2, Apple M5 Pro, 1600 × 900, sunset, fly-over benchmark):
* **Frame rate:** mean 117 fps; p95 frame 11.9 ms (84 fps); 4.2 k draw calls.
* **Load time:** 48 s, because the GLB is 181 MB with one glTF node per plant.
* **Next step:** write plant transforms to a sidecar file instead of glTF nodes, to load faster.

## 9. Known limits and next steps

* Arcades (*portales*), interior patios, a plaza kiosk or fountain, and stepped streets on steep slopes are not
  modelled yet.
* There is no LOD for houses beyond Godot's per-chunk culling.
* Blender has no aerial haze: a world volume blacked out EEVEE at every tested density, so the haze lives in Godot's
  fog.
* Wall colours are drawn per building, not by an urban rule (for example, warmer near the plaza).
