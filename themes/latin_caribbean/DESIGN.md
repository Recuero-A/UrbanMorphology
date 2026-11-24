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
| KAIST village with a Latin skyscraper district and red churches | `kaist_village.py --method NA-adaptive --version original --giant 165,125,95 --giant-lottery 60,85 --lanes 60 --district 10 --church-colour rojo --church-dome cupula_naranja` (§5.2) |
| The same downtown with six landmark towers and four photo churches (a separate city; §5.5) | the downtown command plus `--landmarks "r1c1=golden:rojo,coral:bermellon,eagle:coral,coral:rojo_ladrillo,golden:coral;r2c1=eagle:rojo" --photo-churches "r0c1=gothic,r1c0=gothic,r1c2=bahia,r0c2=bahia"` |
| Any big 3 × 3 city with the theme | `scripts/m16_city3d_big.py ... --theme latin_caribbean [--terrain-npz domain.npz] --cities-dir outputs/latin_caribbean/cities/<site>` |
| In code | `city3d.scene.build_scene(city, theme="latin_caribbean")`, then `apply_terrain`, then `latin.build.dress_terrain` |
| Landmark towers and the two photo churches on a showcase plaza (sheets, reference comparisons, sunset photos, Godot; §5.3-5.4) | `.venv/bin/python themes/latin_caribbean/scripts/preview_landmarks.py [--height 150] [--colours wall=wall:rosa,...]` |
| Sunset photo tour of the KAIST downtown village (12 photos, Blender + Godot; §8.1) | `.venv/bin/python themes/latin_caribbean/scripts/photo_tour.py [--only 01_hill_golden_hour] [--res 2400] [--samples 128]` |
| Sunset showcase: six photos each of the landmark city and the coastal town (§8.2) | `.venv/bin/python themes/latin_caribbean/scripts/showcase.py [--city landmarks\|coast\|all] [--only name]` |
| Blender light | `blender/m11_scene.py -- --tile DIR --light sunset` (default for the theme), `midday`, or `default` (M11) |
| Godot light | `godot/viewer -- --glb ... --light sunset\|midday` |

* **Where the cities are.** The theme's 3D cities live in `outputs/latin_caribbean/cities/kaist/` (KAIST: the themed
  campus, the villages, the downtown, the landmark city) and `outputs/latin_caribbean/cities/coast/` (the coastal
  town), next to their street and terrain data (`outputs/latin_caribbean/kaist_village/`, `.../coast/`). They moved
  there from `outputs/m16/cities/` on 2026-11-23. The research cities of the milestones stay in
  `outputs/m16/cities/`, where their reports point.
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

### 5.1 Giant towers (`church.giant_tower`; user requests 2026-11-23)

Not a colonial building type. It is a Spanish-colonial bell tower scaled to 60-165 m.

**Version 2.** The user said the first version looked like a mosque. It was very slender, had octagonal upper
stages and had a balustraded ring at every setback, and that is a minaret's silhouette. The tower is now square in
plan up to the cupola, as in Cartagena, Havana, Lima and Puebla, and in the user's red reference tower. It is also
stockier, and its decoration is concentrated in the belfries.

* **Site.** One tower on the left of the façade, base side 0.55 × the church width (8-20 m). The nave keeps at least
  9 m. With `--giant`, the plaza church sites may be up to 34 m wide (22 m otherwise).
* **Shaft** (about half the height). A mostly plain wall: stone plinth, cream quoins and string courses about every
  1.4 side lengths. The panels alternate a small arched window with a niche on a sill.
* **Clock stage.** The same side as the shaft. A clock with gilt and cream rings and iron hands on every face, and
  pilasters at the corners.
* **Belfry 1** (0.86 × the base side) on the only balustraded balcony:
  * corner urns and bougainvillea planters on the balcony;
  * three or two arches per face with paired pilasters;
  * a bronze bell on a wooden yoke in every arch (the strongest "bell tower, not minaret" cue);
  * an entablature at the top.
* **Belfry 2** (0.7 × the base side). Baroque scrolls (volutes) and urns at its corners, then one or two belled
  arches per face.
* **Top.** A low cream attic with corner urns, a drum with oculi, and a ribbed cupola: orange with
  `--church-dome cupula_naranja`, or tile, blue or yellow. Then a lantern, a gilt ball and an iron cross.
* **Height.** The three stages take about half of the height (stretched 0.75-1.35×); the shaft takes the rest.
* **Triangles.** About 12-14 k per tower.
* **Colour.** `--church-colour rojo` makes every church the reference's vermilion (#cc3a28). The giant tower's trim is
  cream.
* **Crosses (all churches).** `kit.cross` now stands on a square post. `beam()` builds a box with zero width on a
  vertical line, so every earlier church cross showed only its crossbar.
* **Heights.** `kaist_village.py --giant 165,125,95` sets the plaza churches, tallest first.
  `--giant-lottery lo,hi` gives the other churches (sites at least 15 m wide) a tower of uniform random height.
* **Views.** The script adds three views to `cameras.json` (Blender) and `viewer.json` (`shots`, Godot `--shot`):
  * `tower`: from a street 90-220 m away with a clear view of the top 75 % of the tower. The far side of the
    main plaza is the fallback: from about 50 m, the tower's upper 70 m is foreshortened into a sliver.
    Lines of sight must clear every building (by its height) and any low crown or fountain within 35 m of the
    camera;
  * `alley`: down a street with a clear line of sight to the tallest tower;
  * `skyline`: from the sun side, over the jungle.
  The Godot walk starts in the main plaza, facing the tower.

### 5.2 Latin skyscrapers and their district (`city3d/latin/skyscraper.py`, `infill.tower_site`; 2026-11-23)

The user's reference picture is a vermilion tower with cream trim and an orange dome, in Spanish-colonial revival /
Beaux-Arts style, among cream arcaded buildings, palms and bougainvillea. Colours were sampled from the image: walls
#dc5827, dome #df610d, trim #f4c2a1 and glass #432d1d under its sunset light. The albedos are redder or paler to
compensate: `tower:rojo` #cc3a28 (plus bermellón and rojo ladrillo for variety), `cupula_naranja` #e2672a,
`trimc:crema` and `glass_warm` #5e3b24.

* **Podium** (cream, three storeys):
  * a stone plinth and an arcade (portal) of warm shop windows;
  * two floors of windows with surrounds, with iron balconies on the first;
  * a cornice, then a balustraded roof terrace with urns around the shaft.
* **Shaft** (red, set back 0.8-1.4 m):
  * cream quoins and a cream pilaster between bays (bays about 5 m wide);
  * stages of 3-5 storeys, each ending in a projecting cornice with a wrap-around balustraded balcony;
  * window rhythm: rectangular-window stages (surround, sill, lintel) with a round-arched top storey, alternating with
    stages of two-storey arched windows.
* **Loggia.** One or two storeys of tall arches, an entablature, a balustrade and corner urns.
* **Crown:**
  * an octagonal cream drum with warm arched windows;
  * a ribbed dome: orange 65 %, tile 20 % or blue 15 %;
  * a lantern and a gilt finial.
* **Size.** 6-21 k triangles for 30-130 m. The height is within about 3 m of the target, because it is rounded to
  whole storeys.
* **District** (`kaist_village.py --district N [--district-h lo,hi]`):
  * The smallest disk around the main plaza that holds N streets. A street is a natural stroke: it continues through a
    junction when the deflection is under 35°, and strokes under 40 m don't count.
  * In blocks inside the disk, `colonial_infill` cuts wide, deep lots (frontage 18-26 m, depth 18-24 m). Big blocks
    are first cut by lanes every 1.5 × `--lanes`.
  * Each lot gets one skyscraper on the largest rectangle that fits inside it (1 m clear of its neighbours, sides
    11-30 m).
  * Height: lo-hi from the edge to the centre (×0.8-1.12 at random), and at most 7.5 × the narrower side.
  * Lots too small for a tower, and the block cores, become gardens.
  * The plazas and churches stay. The giant cathedral tower (165 m) stays the tallest.
* **Views:**
  * `skyscraper`: up at the tallest skyscraper from a street with a clear line of sight;
  * `downtown`: along the district's longest street, pitched up 24°;
  * `district`: an oblique view from the sun side.

### 5.3 Landmark towers (`city3d/latin/landmarks.py`; user references 2026-11-23)

The user's six study sheets (`architectural features/thin`, `architectural features/cohesive`) show three designs,
each in two forms.

**Version 1 was rejected.** The user found it "so plain": its relief stood only 10-30 cm off the walls, its ornament
was flat cut-out textures, its crowns were simple boxes, and the relief on the Coral gable was stretched.

**Version 2** is built with a relief kit (`city3d/latin/relief.py`), in which every element has real depth:

* **Profile sweeps:** cornices, string courses, bases, architraves, archivolts, raking cornices and dome ribs. Faces
  are oriented from the profile tangent.
* **Extruded 2D ornament:** carved lace (an anthemion with vine scrolls, acanthus leaves, rosettes and C-scrolls,
  drawn at each panel's own proportions, never stretched), cartouches, scallop shells, scroll consoles, volute
  buttresses, swan-neck pediments and rose tracery.
* **Lathe solids:** columns with Attic bases and Corinthian capitals (two rows of leaves and corner volutes),
  balusters, urns, bulbs, onion finials, bells and statues.
* **Ornament wrapped onto domes** (`wrap_dome`).
* **Windows** show interiors. `glass_lit` is an atlas of four cells (lit with curtains and a lamp; lit with a plant;
  half-lit; dark), and each window takes one cell, chosen from its position.

| Design | Kept from the references |
|---|---|
| **Coral Lantern** | gold-yellow walls and a coral strip of balconies between deep cream pilasters; a crown on scroll consoles (an arcade with balustraded balconies, then an oculus stage with cartouches and volute ears); corner turrets with obelisks; a coral ribbed dome with aedicule dormers (coral pilasters, balcony, pointed pediment with cream lace, volute wings); a two-tier lantern; a banded coral spire with a needle |
| **Eagle Belfry** | ivory walls; green marble columns with peach Corinthian capitals (on the shaft, the chamfers and both belfries); a chamfered two-stage crown; arched bays (marble archivolt, bronze bell, round green balcony on a peach corbel); oculi under hoods; heavy peach swan-neck pediments; urns; layered entablatures; volute buttresses; a green ball balustrade; a stepped peach dome; a bronze eagle with fingered wings |
| **Golden Clock Tower** | gold walls between cream piers and pilasters; a console cornice with a balustrade, cypresses and gilt urns; a clock stage (teal dial in a bold bezel, carved C-scrolls, lace panels, balcony); a belfry of three gilt bells (Corinthian columns, balustrades, broken pediment, cartouche); an orange ribbed dome with lace between the ribs and teal oculi; onion finials; a colonnaded lantern with a gilt cupola |

* **Thin form.** A wider shaft (22-23 m) under a crown about 1.2 × wider. The user chose this ratio: it gives the
  proportions of the city's skyscrapers.
* **Cohesive form.** One width (20-22 m), with a tier repeated up the shaft:
  * Coral: a triple arcade, then a pointed gable with lace.
  * Eagle: a window grid under a hood, then an arched bay with a round balcony.
  * Golden: three storeys by three bays with balconies, a segmental hood with a shell and lace, and teal ovals.
  * The cohesive Golden tower has one clock only, on the dome, as the sheet notes.
* **Height** is calibrated to the target by whole storeys or whole tiers: within about 2 m for thin forms, about 5 m
  for cohesive forms.
* **Cost: 225-330 k triangles per tower at 150 m**, about 10 × version 1. That is fine for a few landmarks. Placing
  many in a city needs a lighter level of detail (see §9).
* **Colours** follow a scheme of roles (wall, accent, trim, base, band, dome, spire, glass, lit, metal, rail, rail3d,
  column, frame, ornament). `landmark(..., colours={"wall": "wall:rosa"})` repaints a tower in any house colour.
  Landmark-only colours live in `palette.LANDMARK`, `LANDMARK_DOMES` and `MARBLES`. Never append them to
  `WALLS`/`DOMES`: that would shift the house and church lotteries.
* **Studio light** (`m11_scene.py --light studio`): a warm uniform ambient, a key sun and a fill sun, the Standard
  view transform, and an unfilmed sky composited on paper. AgX desaturated the yellows and corals.
* **Preview** (`preview_landmarks.py`) writes:
  * `sheets/`: one reference-style sheet per building;
  * `compare/`: each model beside its reference;
  * `photos/`: sunset photos (`photo_tour.py`);
  * a GLB with `viewer.json` for Godot, and Godot shots.

### 5.4 Churches from the user's photos (`city3d/latin/church_styles.py`; 2026-11-23)

Reference photos: `references/church_gothic_blue.png` and `references/church_bahia_rosario.png`.
Call: `church_style(style, center, axis, L, W, rng, colours)`. The facade is centred on `center` and the nave runs
along `axis`.

* **gothic.** A cobalt-blue neo-Gothic church with white trim:
  * twin square towers with white colonnettes, string courses, long louvred twin lancets, a white clock and a
    Lombard band;
  * octagonal slate spires with white bands, gabled lucarnes and four open corner pinnacles;
  * a round-arched portal with a stepped archivolt;
  * a rose window with cut tracery (a rosette, twelve petals and twelve flowers) on white pilasters;
  * a gable with a white notched band, a medallion and an apex statue, with pinnacle turrets at its feet;
  * a buttressed nave with a tile roof.
  * Size: 42 m to the cross for a 26 m facade, about 44 k triangles.
* **bahia.** The Pelourinho baroque church (Salvador): aqua walls and sandstone pilasters, cornices and frames.
  * Body: three doors and three balconied windows in eared frames with curved and broken hoods, palmettes and
    scrolled aprons; oval oculi in carved cartouches.
  * Frontispiece: S-curved, with volutes, urns, an oculus and a cross.
  * Towers: belfries with bronze bells, a big cornice, four stacked-bulb pinnacles with terracotta flames, and a
    tiled, ribbed stacked-bulb cupola.
  * Size: 36 m to the cross, about 48 k triangles.
* **Not yet in the city.** Both join the church lottery when the landmarks are placed.

### 5.5 The landmark city (`kaist_village.py --landmarks --photo-churches`; user request 2026-11-23)

This is a separate city: `outputs/latin_caribbean/cities/kaist/kaist_village_landmarks_NA-adaptive_s1_latin`. The
downtown city it starts from (§5.2) is kept unchanged. Streets, lots, plazas, district and seeds are the same, so the
two cities differ only in the buildings swapped below.

* **Landmarks.** Theme option `landmarks` = {tile: ["design:wall", ...]}.
  * Selection, per tile: the tallest district skyscrapers whose narrower side is ≥ 15 m and height ≥ 60 m, at least
    60 m apart.
  * Each becomes the cohesive form (§5.3), with `landmark_on_lot`: the widest shaft whose whole tower stays within
    the skyscraper's rectangle (+0.4 m). These lots are 15-18 m, so shafts are 11-13 m, about 1:7-1:10. That is the
    slenderness of the cohesive sheets.
  * The tiers adapt to narrow shafts: two windows instead of three below 17 m, slimmer marble columns, side ovals
    only above 15 m. The crowns keep human-scale minimums (Eagle bays ≥ 8.5 m, Golden belfry ≥ 7 m, dormer mouldings
    in proportion).
  * Colours (`landmarks.city_colours`; user request 2026-11-24: one cohesive, tropical downtown with the red towers).
    * Walls: the reds of the Latin skyscrapers (`palette.TOWER_WALLS`: rojo, bermellón, rojo ladrillo) or a warm house
      colour (coral, terracota).
    * Trim and podium: cream (`trimc:crema`, `podium`), as on those towers.
    * Accents: yellow, ochre or orange. Domes: orange or tile.
    * Tropical touches: green or turquoise marble on the Eagle towers, turquoise frames and balusters on the Golden
      towers, wooden balconies on the Coral towers.
    * No cream or pale walls. The first version had cream, yellow, turquoise and pale-pink towers, and they did not
      mix with the red downtown.
    * The Golden towers' piers, columns, belfry and drum take the wall colour. They were solid cream and read as white
      blocks.
  * This city: Golden 122 m red, Eagle 92 m coral, Coral 86 m brick red, Coral 83 m vermilion, Golden 78 m coral,
    Eagle 64 m red.
* **Photo churches.** Theme option `photo_churches` = {tile: style} replaces the lottery church of that tile when its
  site is ≥ 18 m wide.
  * The facade sits 2.4 m inside the site, so the towers do not reach the street.
  * Heights scale with the facade width (0.82-1.1 ×). The rose window shrinks to fit between the string course and
    the cornice.
  * This city: Gothic ×2 (42 m, 36 m) and Bahia ×2 (33 m, 30 m) on four of the five lottery sites. The three plaza
    churches with giant towers and one lottery church stay red.
* **Bahia colours** (user requests 2026-11-23 and 2026-11-24): house-palette turquoise walls, glazed terracotta
  bulbs as on the city's domes, and terracotta flames.
  * The stone frames and pinnacles are the palette's warm grey (`trimc:gris`).
  * The first recolour used a deep golden ochre. It was too saturated against the city and was dropped.
  * The photo's own colours are kept in `church_styles.PHOTO_COLOURS` for the reference sheets.
* **Views:**
  * `landmark`: up at the tallest landmark from a street;
  * `landmarks`: over the district from the sun side;
  * `church_gothic` and `church_bahia`: a street view of the first of each.
  The photo tour (§8.1) adds all four.
* **Cost.** 3.51 M triangles in Blender (the downtown city: 2.36 M). Godot: 3.9-4.0 M triangles; 72-126 fps mean and
  63-70 fps at the 95th-percentile frame over three runs (the mean varies with machine load), against 103 / 77 for
  the downtown (Apple M5 Pro, 1600 × 900).

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

### 8.1 Sunset photo tour (`scripts/photo_tour.py`, `blender/photo_tour.py`; user request 2026-11-23)

Twelve photographs of the downtown village (§5.2), taken like an evening walk: the sun sinks from 7° to 0.3° from
photo to photo (it stays at azimuth 140°, the north-west, as in every sunset render). Each viewpoint is placed by
hand, then checked in low-resolution scouting renders (four rounds; garden, fountain, downtown-street and second-plaza
views were dropped because a flamboyant crown or a wall filled the frame). Placement helpers:
* `Ground.spot` finds the free point nearest a wish: off every building, at least 7 m from a low crown, with the first
  metres of the line of sight clear.
* `shore_spot` finds the pond's edge with the longest stretch of water towards the target, so the towers reflect.
* `balcony_point` puts the camera on the first belfry's balcony of the cathedral tower.
* `terrace_point` puts it on a skyscraper's crown terrace; its height comes from a Blender ray cast.

`blender/photo_tour.py` opens the city's `scene.blend` and changes only light and camera:
* **Sky.** The physical sky gets a procedural cloud layer. Clouds take half the sky's light plus the reddened low
  sun: pink-salmon away from the sun, gold around it. Thick parts self-shadow. Rays below the horizon see the
  horizon, so there is no black seam.
* **Sun.** The lamp's strength and colour follow its elevation (1.2 W at 0°, colour (1, 0.28, 0.12); 6 W at 8°).
  An optional sun disc is drawn in the sky shader.
* **Haze.** Each photo first renders a sky-only image with every view ray flattened onto the horizon: this is the
  colour of the haze in that direction. The compositor then mixes each pixel towards it by
  1 − exp(−d / haze_m − max(d − 1.8 km, 0) / 2.5 km), where d is depth. The second term dissolves the flat land past
  the map's edge into the horizon.
  * The depth is eroded by one pixel, because anti-aliased land/sky edge pixels otherwise went unhazed (a dashed dark
    line on the horizon).
  * The sky itself is never hazed.
* **Finish.** Warm skyscraper windows and lanterns glow (emission, about 0.6–1.0); a light bloom.

Godot renders the same cameras under its own sunset light (`photos/godot_<name>.png`, no clouds).

**Sky direction (found here).** Blender's sky texture reads `sun_rotation` as a compass bearing (clockwise from
north; measured: rotation 0 is brightest towards +y, 140° is brightest at a math angle of 310°). The m11 sun lamp
uses a math angle (counter-clockwise from east). `blender/m11_scene.py` passes the same 140° to both, so in every
m11 render the sky's bright side is opposite the side the sun lights. `photo_tour.py` passes 90° − az to the sky;
`m11_scene.py` is unchanged.

### 8.2 Sunset showcase (`scripts/showcase.py`; user request 2026-11-23)

Six hand-placed photos per city, rendered by `blender/photo_tour.py`, each with its own sun.

* **Landmark city** (sun towards the north-west, as in every KAIST render): the towers front-lit from the north-west
  hills (78 mm), the backlit skyline with the sun beside the towers, a raking-light drone from the south-west, the red
  Golden tower behind a colonial balcony, the towers above the rainforest canopy, the Bahia church at dusk.
* **Coastal town** (the sun sets over the sea, south-west): the town from the beach, a drone over the bay, the vista
  from the mountain road, the beach promenade, the sun into the sea over the roofs, palms on the beach against the
  sunset.
* **Placement rules.** Street cameras keep 8 m clear of any building (`open_spot`), so no wall or balcony fills the
  frame. The coastal drone looks down steeply enough that the map's edges stay out of the frame.
* **Lesson.** Moving the theme's cities broke the texture paths inside their `scene.blend` files: Blender saves
  image paths relative to the `.blend`. All seven were repointed, and `m11_scene.py` now saves absolute paths. The
  GLBs embed their images, so Godot was never affected.

## 9. Known limits and next steps

* Arcades (*portales*), interior patios, a plaza kiosk or fountain, and stepped streets on steep slopes are not
  modelled yet.
* There is no LOD for houses beyond Godot's per-chunk culling.
* `m11_scene.py` has no aerial haze (a world volume blacked out EEVEE at every tested density); only the photo tour
  adds it, in the compositor (§8.1). The m11 sky's sun sits opposite the sun lamp (§8.1).
* Wall colours are drawn per building, not by an urban rule (for example, warmer near the plaza).
* **Landmark towers (§5.3): what still differs from the reference sheets.**
  * The carved lace is a 2.5D extrusion, not sculpted relief.
  * The Corinthian leaves are simple triangles.
  * The eagle is low-poly.
  * The Eagle crown is chamfered but not as deeply layered as the drawing.
  * The thin Golden tower has a clock on all four faces.
  * At 225-330 k triangles per tower they need a lighter level of detail before many go into a city.
* **Photo churches (§5.4).** The Bahia cupola tiles are uniform, not weathered. Neither church has an interior.
