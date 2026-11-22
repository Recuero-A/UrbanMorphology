# Latin Caribbean theme: design reference

This file is for the 3D artist or programmer who builds the optional "Latin Caribbean" theme (buildings, roofs, balconies, churches, vegetation, beach and mountain terrain) in Blender 5.2 and Godot 4.7 (Forward+). Compiled 2026-11-22.

**Evidence tags:**
- **[V]** verified in the fetched source, cited as `[Sn]` (list at the end).
- **[S]** seen only in a search-engine summary. The page itself was blocked (403) or not opened.
- **[E]** estimate or unverified. Treat it as a tunable default.
- **[R]** sampled from the owner's reference images. Each value is the median of a 9–15 px patch from `architectural features/IMG_3376–3378.jpg`.

The reference images are stylised illustrations (phone screenshots). Their colours already include golden-hour tint and posterisation.

---
## 1. Style summary: Spanish colonial Caribbean

### 1.1 Towns
| Town | What to borrow | Tag |
|---|---|---|
| Cartagena de Indias (CO) | Casa baja / casa alta around a patio; wooden balconies; coral-stone churches; strong lime colours (back since the 1980s) | [V] [S1][S3][S4] |
| Mompox (CO) | Santa Bárbara church: octagonal "Moorish-Baroque" bell tower, yellow front | [S] [S16] |
| Santa Marta (CO) | Founded 1525; 18th-c. cathedral; town on a bay at the foot of the Sierra Nevada | [V] [S4][S37] |
| Havana Vieja (CU) | Baroque + neoclassical; coral-block cathedral; mansions with an *entresuelo* (low mezzanine) for servants | [V]/[S] [S7][S23] |
| Trinidad (CU) | Pastel houses, wrought-iron grilles, cobbles; turned-wood window *barrotes* | [V] [S19]; barrotes [S] |
| Santo Domingo, Zona Colonial (DO) | Calle Las Damas (1502, oldest paved street in the Americas); coral-limestone facades | [V] [S8][S24] |
| Old San Juan (PR) | "Vibrantly-colored" facades, wooden **and** metal balconies, flat-roofed brick/stone buildings, blue cobbles | [V] [S18] |
| Granada / León (NI) | Big neoclassical cathedrals on plazas; León has a white roof of many small domes | [V] [S10][S11] |
| Antigua Guatemala | "Seismic baroque": thick walls, low towers [E]; lime-yellow walls with white stucco lace (*ataurique*) | [V]/[S] [S12] |
| Campeche (MX) | Walled port; Baroque/neoclassical cathedral (1540–1760) | [V] [S15] |
| Puerto Vallarta (MX) | Cobbles; narrow coastal plain at the foot of the Sierras Cuale and San Sebastián; church tower with a crown | [V] [S14][S38] |

### 1.2 Building elements and typical dimensions
| Element | Facts / values | Tag |
|---|---|---|
| Typologies | **Casa baja**: 1 floor, lower or middle-class owners. **Casa alta**: 2 floors; the ground floor holds commerce, offices, storage and servants, and the owner lives upstairs. Shops sit in corner houses. Plans are C, O, U or L around a patio, with a side *zaguán* (entrance passage) and an *aljibe* (cistern) in the patio. | [V] [S1] |
| "Casa de balcón", "accesoria" | Common labels in the literature, but I found no definition in the sources I opened. If used: accesoria = 1-bay rental unit, 1 floor or 2 with an *altillo* (loft). | [E] |
| Floors | 1–2 in Cartagena [V S1]. Havana mansions: 2–3, including the entresuelo. Small-town infill: 1. | [V]/[S] |
| Room / storey height | Cartagena rooms are 3.0–4.0 m tall (Bogotá 2.5–3.0) [V S1]. Modelling defaults: ground floor of a casa alta 4.0–5.0 m, upper floor 3.5–4.5 m, entresuelo 2.2–2.8 m. | [V] + [E] |
| Walls | Adobe, or brick with stone, ≈30 cm (varies) [V S1]. Palaces 0.5–0.8 m [E]. Finish: smooth lime-and-sand render; stone or tile cladding is not allowed [V S2]. | [V] |
| Facade compositions | Portón + door; portón + 2 windows; portón + 3 windows [V S1]. Opening axis spacing ≈2.5–4 m [E]. | [V]/[E] |
| Openings | Always **taller than wide** (now a Cartagena rule) [V S1, S2]. Lintels are plastered over and hidden [V S2]. Window ≈1.0–1.4 × 2.0–2.8 m; portón ≈1.8–2.6 × 3.2–4.2 m; door ≈1.1–1.4 × 2.5–3.0 m [E]. | [V]/[E] |
| Windows | Three types: *de panza* (bellied), *de caja* (box), *de reja* (grille). Turned wood spindles (*bolillos*) in 3–4 sections. Above: a *tejaroz* (small tile roof). Below: a masonry sill shaped as cup, belly or shelf. Inside: seats. | [V] [S1] |
| Rejas (window grilles) | In Cartagena, wood replaced iron because sea air corroded iron; the *reja volada* projects from the wall face [V S1]. Cuba: metal bars [V S20]. Trinidad: wrought iron [V S19]. Projection 0.15–0.40 m [E]. | [V]/[E] |
| Wooden balcony | Parts: *canes* (corbels), railing, *pies derechos* (posts), *zapatas* (bracket caps), *tejadillo* (roof) [V S1]. Types: covered (1–6 modules), corner, patio-facing, *tribuna* [V S1]. Climbers on balconies [V S1]. **Max projection 1.20 m** (Cartagena rule; the brochure misprints "1.20 cm") [V S2]. Rail ≈1.0 m with balusters ≈0.10–0.12 m apart; corbel pitch 0.6–0.9 m; posts 2.2–2.6 m [E]. | [V]/[E] |
| Iron balcony | Old San Juan mixes wooden and metal balconies [V S18]. Depth 0.4–0.8 m; thin slab on iron brackets; rail 0.9–1.0 m [E]. | [V]/[E] |
| Tejaroz / tejadillo | Small tile roof over a window [V S1]. Projection 0.4–0.6 m on 2–3 wood brackets, 1–2 tile courses [E]. | [V]/[E] |
| Eaves / cornice | *Alero sobre canecillos* (eave carried on corbels) [V S1]. Overhang 0.5–1.0 m [E]. Flat-roof towns use a moulded cornice with a 0.6–1.0 m parapet [E]. | [V]/[E] |
| Zócalo, portadas, pilasters | Stone or brick plinth (*zócalo*) [V S1]; height 0.6–1.0 m, darker band [E]. Stone or brick arches and door surrounds (*portadas*) [V S1]. Corner and party-wall pilasters plus a white trim band [R]. | [V]/[R] |
| Patio | Organises the whole house and holds the aljibe [V S1]. Unpaved patios should be planted [V S2]. Galleries on 1–2 sides, ≈2.5–3.5 m deep [E]. | [V]/[E] |
| Pitched roof | Timber *par y nudillo* (collar-beam truss) or *par e hilera* (simple rafter). Thatch was later replaced by clay tile [V S1]. Curved "Arab" tile: one half-cone piece used as both channel (*canal*) and cover (*cobija*) [V S25]. Pitch **20–30°** [E]. Modern Colombian clay-tile makers require ≥16–21° (29–40 %) [S]. Tile ≈46 × 28 cm ("teja colonial caribe" retail size) [S]. | [V]/[E] |
| Flat roof (*azotea*) | Old San Juan is flat-roofed [V S18]. The Cartagena cathedral aisles first had flat azoteas [V S4]. Havana azoteas behind parapets [E]. | [V]/[E] |
| Portales, mediopuntos | Cuba: arcades (*portales*) for sun and rain [V S20]. Stained-glass half-round fanlights (*mediopuntos*) with louvred shutters (*persianas*) [S]. | [V]/[S] |
| Colour practice | Lime white or earth colours, with good lime quarries near Cartagena [V S1]. Only lime + mineral powders are allowed [V S2]. History: Cartagena was **ordered all-white** in the 1970s–80s "to resemble Andalusian towns". Paint layers underneath showed yellows, blues and yellow ochre. Colour returned in the early 1980s (e.g. "rojo quemado", burnt red) [V S3]. Havana: more intense colours until the 19th c., lighter later; a 2019 chart lists 164 colours; a rule asks for "medios colores" (mid tones, not white or loud) [S S26]. | [V]/[S] |
| Carpentry colours | Natural wood, or oil paint in white, mahogany, green, blue, *rojo almagre* (red ochre) or grey [V S2]. Historic shutters: light blue, green, pink, rojo almagre [V S1]. | [V] |

## 2. Colour palettes
### 2.1 Per-image palettes [R]
| Image | Palette (hex, roughly in order of area) |
|---|---|
| **IMG_3376** (favourite, golden-hour hillside) | sky `#F29B50` · low sky `#F3B95D` · sun `#FCFCC0` · far ridge `#EDA554` · mid ridge `#C77A61` · mauve ridge `#AE5C69` · sea `#57918B` · deep sea `#27474B` · dome `#DE4D35` · church facade `#DB6A3C` · tower `#E47240` · shaded red wall `#902E31` · cream wall `#F1BE86` · roof lit `#DC704C` · roof `#A65136` · roof shade `#9E4A32` · bougainvillea `#DA6465` / `#A14346` · palm `#243531` · jungle `#1B2625` · path `#E5AC87` |
| **IMG_3377** (ochre/teal town, blue domes) | sky `#D88D39` · low sky `#EFBC48` · sun `#EDD3AD` · slate mountain `#3B565F` / `#567379` · blue dome `#60858B` · teal wall `#698C92` · lime white `#D8D0C5` · ochre tower `#F0BC50` · orange tower `#EA9C39` · orange wall `#E18031` · coral wall `#E76A44` · red wall `#E36440` · brown tile `#936349` · olive palm `#8D7D4F` / lit `#C4B58A` · red flowers `#A44037` · openings/iron `#1F2427` |
| **IMG_3378** (Cartagena street) | yellow `#F4C946` · magenta `#BC2D4B` · pink `#ED6F70` · turquoise `#56B2B1` · deep teal `#478E87` · cream `#F9E6D5` · door orange `#E96E31` · corbel wood `#B44E35` · dark wood `#6D2112` · tejaroz `#A6412A` · banana `#3B5B3A` · lit leaf `#A3AA5E` · heliconia `#C82E4D` · orange flower `#E18437` · step tile `#EB904B` · paving `#E5DBCE` |

### 2.2 Named master palette (theme defaults)
| Name | Hex | Role | Src |
|---|---|---|---|
| Cal White | `#F2ECE0` | Trims, pilasters, cornices, white walls (albedo) | [E] from `#D8D0C5`/`#F9E6D5` |
| Crema | `#F1BE86` | Base wall (most common in 3376) | [R] |
| Amarillo Cartagena | `#F4C946` | Wall | [R] |
| Ocre | `#F0BC50` | Wall, church body | [R] |
| Naranja | `#E18031` | Wall | [R] |
| Coral | `#E76A44` | Wall, church facade | [R] |
| Terracota Iglesia | `#DB6A3C` | Church facade, tower | [R] |
| Rojo Almagre | `#902E31` | Deep wall, zócalo, carpentry | [R] (named in [S1][S2]) |
| Fucsia | `#BC2D4B` | Accent wall (≤10 % of facades) | [R] |
| Rosa | `#ED6F70` | Wall | [R] |
| Azul Colonial | `#698C92` | Wall | [R] |
| Turquesa | `#56B2B1` | Shutters, balcony rails, window frames | [R] |
| Verde Postigo | `#478E87` | Shutters, doors | [R] |
| Madera Caoba | `#B44E35` / `#6D2112` | Balconies, corbels, panelled doors (lit / shade) | [R] |
| Puerta Naranja | `#E96E31` | Painted doors | [R] |
| Hierro | `#1F2427` | Iron rejas, rails, open windows | [R] |
| Teja | `#DC704C` → `#A65136` → `#936349` | Roof tile: new → weathered | [R] |
| Cúpula Roja | `#DE4D35` | Tiled or painted domes | [R] |
| Cúpula Azul | `#60858B` | Blue-grey domes | [R] |
| Palma | `#243531` (shade) / `#4A6742` / `#A3AA5E` (lit) | Palm and banana leaves | [R] |
| Buganvilla | `#DA6465` / `#A14346` | Bougainvillea masses | [R] |
| Heliconia | `#C82E4D` / `#E18437` | Flower accents | [R] |
| Mar | `#57918B` (near) / `#27474B` (far) | Sea | [R] |

**Albedo tip [E].** The [R] wall and sky colours already contain the golden-hour tint. As PBR base colour under a warm sun they would double the orange. For albedo, desaturate ≈15–25 % and move hue ≈5° toward neutral (e.g. Crema `#F1BE86` → `#EAD6B4`).

### 2.3 Documented palettes
| Palette | Hex | Tag |
|---|---|---|
| **Cartagena IPCC suggested palette** (exact vector fills read from the brochure; screen approximations, not paint specs) | Amarillo Colonial `#F49D2D` `#FCB12A` `#FFD175` · Rojo Colonial `#D8585E` `#E58B8E` `#EFADAC` · Verde Colonial `#649248` `#ABCE95` `#C4DAB3` · Blanco Cal `#F1F2F2` `#FFFFFF` · Azul Colonial `#0A86C8` `#62A8D7` `#ABCCE7` | [V] [S2] |
| Havana 2019 chart (164 colours) | No hex values published in what I could reach | [S] [S26] |

## 3. Churches
### 3.1 Reference churches
| Church (city) | Date | Plan | Towers / stages | Dome | Facade, colour, dims | Tag |
|---|---|---|---|---|---|---|
| Catedral Santa Catalina (Cartagena) | 1577–1612 | Basilica, 3 naves, 7 pairs of Tuscan columns; tall nave lit by round oculi; octagonal chancel; timber roofs | 1 square tower at the entrance end (Gospel side). The original bell stage had round arches between pilasters. Lelarge raised it and added dome + lantern (1908) | Tower dome + lantern | Lower portal of paired free Corinthian columns between buttresses; turrets with conical brick spires; Tierrabomba coral stone. Tower is ochre/terracotta [E] | [V] [S4][S5] |
| Santo Domingo (Cartagena) | Late 16th c.; roof arches c.1630 | Single wide nave, side chapels, short transept, shallow octagonal chancel; barrel vault on thick transverse arches; 5 external buttresses | 1 brick side tower; bell stage turned 90° to the facade ("twisted by the devil"); round-arched openings between pilasters | Planned over the crossing, never built | Curved cornice following the vault; ball finials; round oculus | [V] [S4] |
| San Pedro Claver (Cartagena) | 1580–1654 | Nave | Two square towers flanking an arched portal [S] | Replaced 1921 by Lelarge | Coral stone [S] | [V] [S6] |
| Santa Bárbara (Mompox) | 1613; rebuilt c.1733 (sources disagree) | Nave [E] | Octagonal "Moorish-Baroque" tower; balcony ring and small cupola [E] | Cupola on tower | Yellow front with palm and floral reliefs [S] | [S] [S16] |
| Catedral San Cristóbal (Havana) | 1748–1777 | Latin cross, nave + 2 aisles, 8 chapels | **2 asymmetric** towers | Over the crossing, not visible from the plaza; 8 flying arches | Baroque; coral blocks with visible fossils; orange tile roofs; **30 × 49 m** | [V] [S7] |
| Catedral Primada (Santo Domingo) | 1512–1541 | 3 naves, Gothic rib vaults | Tower stopped in 1547 | None (aisle vaults read as domes) | Plateresque portal; golden coral limestone; **54 m long, 23 m wide, vault 16 m** | [V] [S8] |
| San Juan Bautista (Old San Juan) | 1535–42; rebuilt 1802; facade 1905 | Nave + side chapels [E] | 2 small towers [E] | Crossing dome [E] | Remains of a Gothic vault; cream/white with white trim [E] | [V] [S9] |
| Catedral de Granada (NI) | 2nd temple 1751; dome iron frame 1916; 2nd tower 1931 | Nave and aisles [E] | 2 | Central dome | Neoclassical; 3,614.87 m²; yellow/ochre body, white trim, red dome [E] | [V] [S10] |
| La Merced (Granada NI) | 1534/39; facade 1781–83 | Nave [E] | 1 climbable tower | — | Baroque facade; white/tan [S] | [S] [S17] |
| Catedral de León (NI) | 1747–1814; design 1762 (Porres) | Rectangular, 10 bays, large dome over the crossing | 2 towers with "Chinese domes" | **34 roof domes** | Neoclassical/Baroque on a terrace; round central pediment with statue; roof painted white [S] | [V] [S11] |
| La Merced (Antigua GT) | 1749/51–1767 | Nave [E] | 2 towers, low and stout [E] ("seismic baroque") | Dome | Stucco *ataurique*; lime-yellow with white trim [S]; survived the 1773 quake | [V] [S12] |
| Santa Prisca (Taxco) | 1751–1759 | Very narrow Latin cross | 2 Churrigueresque towers | Octagonal drum, ribbed ("gajos") dome + lantern, Talavera tiles [S] | Solomonic columns; pink stone [S]; **height 40.40 m** | [V] [S13] |
| N. S. de Guadalupe (Puerto Vallarta) | 1930–1940 | Nave [E] | 1 tower topped by a **crown** (fell in 1995, rebuilt 2009) | — | Neoclassical; red-brick tower, white body [E] | [V] [S14] |
| Catedral de Campeche | 1540–1760 | Nave [E] | 2 towers [E] | Dome [E] | Baroque with neoclassical elements | [V] [S15] |

Two layout rules for churches:
- **Plaza:** the Laws of the Indies ideal plaza is about 600 × 400 ft (≈183 × 122 m). The church stands on its own block, separated from the plaza by a street, facing east, and arcades line the plaza streets [V S22].
- **Earthquake baroque:** towers lower and stouter, thick walls and buttresses, sometimes a detached campanile. The source describes this for the Philippines [V S21].

### 3.2 Procedural recipe
All ranges are [E] unless they cite an anchor. Anchors are [V] values from the table above.

| Component | Parameter: range (default) | Notes |
|---|---|---|
| Type | Parish (single nave) 70 %, cathedral (3 naves) 30 % | Santo Domingo vs. cathedral |
| Nave | Width 8–14 m (10); length 25–55 m (35); eave 9–14 m (11) | Primada 23 × 54 m, vault 16 m; Havana 30 × 49 m |
| Bays | 5–7 bays with buttress pilasters (6) | 5 buttresses; 7 column pairs |
| Nave roof | Tiled gable 22–30° (26°) or vault + flat terrace with parapet | |
| Aisles (cathedral) | Width 0.45–0.6 × nave; height 0.6–0.75 × nave | |
| Transept + crossing | Arm depth 0.5–1 × nave width; octagonal chancel 0.8 × nave width | |
| Facade | 2–3 horizontal tiers with cornices; paired pilasters or columns; arched portal 3–4.5 m wide × 5–7 m high; oculus Ø 1.5–3 m above | Corinthian paired columns, oculus |
| Facade top | Curved gable following the vault / triangular pediment / round pediment + statue / *espadaña* (bell-gable with 2–3 arched bell openings, for chapels) | Santo Domingo, León |
| Tower layout | One tower beside the facade (Gospel/left) 50 %, twin symmetric 30 %, twin asymmetric 10 %, detached campanile 10 % | Cartagena, León/Taxco, Havana, ref 3376 |
| Tower shaft | Square side 4.5–7 m (5.5); shaft height = nave eave ±10 % | |
| Bell stages | 1–3 stages; each 0.8–1.0 × side tall and 5–10 % narrower; 1–2 round-arched openings per face, pilasters at corners, cornice + corner pinnacles or balls | Ref 3376: 2 stages, 2 + 3 arches |
| Tower total | 1.6–2.6 × nave eave (≈25–40 m) | Santa Prisca 40.4 m |
| Tower cap | Half-orange dome + lantern + cross / octagonal stage + cupola / pyramid or cone in brick / crown | Cartagena, Mompox, PV |
| Crossing dome | Drum: octagonal or round, height 0.4–0.7 R, 8 windows; dome R = 0.4–0.5 × nave width; 8–16 ribs (12); profile 1.0–1.15 × R (slightly raised) | Taxco ribs; ref 3376 red ribbed dome |
| Lantern | Radius 0.15–0.22 R; height 0.9–1.3 × its diameter; 4–8 arched openings; small dome + ball + cross | Refs 3376 and 3377 |
| Dome finish | Glazed tile (red `#DE4D35`, blue-grey `#60858B`, Talavera blue-white), lime white, or plain terracotta | Taxco, León, refs |
| Cross | Wrought iron, 1.5–3 m (2), on a ball finial | |
| Colour | Body from the wall palette (ocre/coral/terracota/white); pilasters, cornices and openings in Cal White | La Merced yellow + white |
| Placement | On the plaza (separated by a street), facing the plaza; 2–6 palms or a flamboyant tree in the plaza | [S22] |

## 4. Vegetation
Sizes are [V] from the cited source. Hex values are [R] where a reference image shows the plant, otherwise [E].

| Plant | Height | Crown / spread | Form notes | Colours (hex) | Where in town |
|---|---|---|---|---|---|
| Coconut (*Cocos nucifera*) | ≤30 m [S27]; 15–18 m (50–60 ft) [S28] | 4.6–7.6 m (15–25 ft) [S28] | Fronds 4–6 m, leaflets 0.6–0.9 m [S27]; curved, leaning trunk [E] | Fronds `#243531` / `#4A6742` / `#A3AA5E` [R]; trunk `#8A7A66` [E] | Beachfront and sandy soils [S27]; plazas, promenades |
| Royal palm (*Roystonea regia*) | 15–21 m (50–70 ft) [S29] | 6–7.6 m [S29] | Smooth grey-white trunk up to 0.6 m wide, bulge below the crown; green crownshaft; ~15 leaves up to 4 m [S29][S30] | Trunk `#C9C4B8`, crownshaft `#6E9A3C`, leaves `#3F6B35` [E] | Streets, parks, plazas [S29]; Cuba's national tree [S30] |
| Fan palm (*Sabal palmetto*; *Washingtonia robusta* for MX) | ≤24 m [S31]; W. robusta 21–30 m [S] | 2.5–4 m [E] | Costapalmate fans; W. robusta native to Baja/Sonora, planted elsewhere [S] | `#354F4B` [R] (3376 foreground fans) | Patios, plazas; foreground framing |
| Banana / plantain (*Musa*) | Up to 9 m (genus) [S32]; typical 3–6 m [E] | 2–4 m clump [E] | Pseudostem; leaves up to 7 m in one subspecies [S32]; wind-torn leaves [E] | `#3B5B3A` / lit `#A3AA5E` [R] | Patios, back yards, roadside clumps |
| Heliconia | 0.5–4.5 m [S33] | 1–2 m [E] | Leaves 0.15–3 m; erect or hanging bract spikes [S33] | Bracts `#C82E4D` / `#E18437` [R] | Patio beds, street planters, forest edge |
| Bougainvillea | 1–12 m [S34] | Climber [S34] | Thorny vine; bracts pink, magenta, purple, red, orange, white, yellow [S34] | `#DA6465` / `#A14346` [R] | Walls, fences [S34]; balconies [S1]; pergolas [E] |
| Hibiscus (*H. rosa-sinensis*) | 2.5–5 m [S35] | 1.5–3 m [S35] | Glossy leaves; flowers ≈10 cm [S35] | Flowers `#D7263D`, leaves `#2F5D2A` [E] | Hedges, patios, front planters |
| Flamboyant (*Delonix regia*) | 10.7–12.2 m (35–40 ft) [S36] | **12–18 m** (40–60 ft), wider than tall [S36] | Fine bipinnate leaves; scarlet flowers 8–11 cm; pods ≤60 cm; leafless in the dry season [S37] | Flowers `#E0442B`, leaves `#5E8C3A` [E] | Plaza centres, avenues; shade tree [S37] |
| Almendro (*Terminalia catappa*) | ≤35 m [S38]; 10–20 m in towns [E] | 10–15 m [E] | Branches in tiers (pagoda form); leaves 15–25 cm that turn red before falling [S38] | `#3F6E2E` / `#B5452E` [E] | Beach roads, plazas (deep shade) [S38] |
| Jungle mass (generic) | Canopy 15–30 m [E] | Continuous | Layered understorey, lianas, palms poking through [E] | `#1B2625` / `#243332` / `#374331` [R] | Slopes behind and above the town |

Cartagena houses used climbers, palms and small shrubs for shade [V S1].

## 5. Coastal mountain + beach terrain
### 5.1 Profile from sea to summit
| Zone | Width / elevation | Slope | Surface colour | Tag / src |
|---|---|---|---|---|
| Deep sea | Depth >20 m; Banderas Bay ≈900–1,436 m deep | — | `#27474B` [R] → `#1B4F6B` [E] | [S] |
| Shallow shelf | 30–300 m wide, 0–8 m deep | Nearshore ≈0.01 (dissipative) | Turquoise `#3FB8C0` (Caribbean) [E] / `#57918B` [R]. Pale sand reflects light, so shallow water looks lighter [S S42] | [V] [S41] |
| Surf / swash | 5–20 m | — | Foam `#F2F4EE` [E] | [E] |
| Beach | 15–60 m wide [E] | Beach face 0.03 (fine sand) to 0.10–0.20 (coarse) [V S41] | Coral sand `#E9DFC7`; golden `#D8BE8F`; darker Pacific sand `#B49D7E`; wet ≈0.7× value [E] | [V]/[E] |
| Backshore / berm | 1–3 m above sea level | 1–4° | Sand + coconut, almendro | [E] |
| Coastal plain (town) | 0–40 m elevation; 150–800 m deep | 0–6° | Streets, plaza near the port. Santa Marta is at 15 m [V S37]. PV sits on a "narrow coastal plain" [V S38] | [V]/[E] |
| Foothills | 40–300 m | 8–20° | Dry forest (Santa Marta, below ~500 m) [V S39]; houses stepping uphill (ref 3376) | [V]/[E] |
| Mountain wall | 300–1,500+ m | 20–40° | Jungle `#243332`/`#374331` [R]; rock `#8A8178` [E]; montane forest 500–900 m, cloud forest >1,000 m [V S39] | [V]/[E] |
| Distant ridges | Layered silhouettes | — | Golden-hour haze `#EDA554` → `#C77A61` → `#B47A6F`/`#AE5C69` [R]; daylight haze slate `#567379` [R] | [R] |

### 5.2 Real-world anchors
- **Sierra Nevada de Santa Marta:** ≈5,700 m high, about 42 km from the Caribbean shore. It is the highest coastal range in the tropics [V S39]. That works out to an average gradient of ≈13.6 % (≈7.7°); local slopes are far steeper.
- **Puerto Vallarta:** sits at the foot of the Sierras Cuale and San Sebastián (Sierra Madre Occidental). The Cuale, Pitillal and Ameca rivers cross the plain [V S38]. Banderas Bay has about 100 km of coastline [V S40].

**Generator recipe [E].** For a town tile of about 1 × 1 km:
1. Sea along one edge.
2. A 20–50 m beach.
3. A 200–500 m plain sloping 2–5°.
4. A 10–20° foothill band 200–300 m deep where houses step up in terraces.
5. Behind that, ridges rising at 25–40° to 400–900 m.
6. Optionally 2–3 background ridge silhouettes, each lighter and more hazy than the one in front.

## 6. Rendering and lighting (golden hour)
### 6.1 What the references show
- **Golden-hour physics:** sun elevation ≈0–6° above the horizon (some tools use −4° to +6°). Colour temperature is ≈3,500 K in the golden hour and ≈2,000 K at the horizon; blue hour follows once the sun is below the horizon [V S43].
- **IMG_3376 [R]:**
  - Backlit scene: the sun disc (`#FCFCC0`) sits just above the ridge, at about 3–5° [E].
  - Sky gradient: zenith `#F29B50` → horizon `#F3B95D`.
  - Palms are near-silhouettes (`#152024` to `#243531`).
  - Strong aerial perspective with a magenta-mauve haze.
  - Lit walls are warm cream `#F1BE86`; shadows go to deep red/violet (`#902E31`, `#522C34`), not grey.
- **IMG_3377 [R]:** flat ochre sky `#D88D39`→`#EFBC48` and a soft sun `#EDD3AD`. The shadowed mountain is a complementary slate blue `#3B565F`, so shadows read cool against warm light.

### 6.2 Engine settings
| Engine | Settings |
|---|---|
| **Godot 4.7 (Forward+)** | **Sun:** DirectionalLight3D pitched −3° to −8° [E], colour `#FFB46B`, energy 1.0–1.5 [E], shadows on. Sky materials take the sun's direction, colour and energy from the first DirectionalLight3D [V S45][S46].<br>**Stylised sky:** ProceduralSkyMaterial: sky_top_color `#E98A4A`, sky_horizon_color `#F3B95D`, ground_horizon_color `#C77A61`, sky_curve 0.10–0.20, sun_angle_max 30 [E] (defaults in [V S45]).<br>**Physical sky:** PhysicalSkyMaterial (defaults: turbidity 10, mie_coefficient 0.005, mie_eccentricity 0.8) [V S46]; raise Mie for a halo [E].<br>**Fog:** Environment depth fog + height fog with **Sun Scatter** and **Aerial Perspective** for layered ridges [V S44]; fog colour `#E9A36A`, height fog over the sea [E].<br>**Tonemap:** AgX keeps hues better as colours brighten [V S44]; set white 6–8 for photoreal [V S44].<br>**Extras:** glow (bloom) for the sun disc; SSAO; SDFGI works in Forward+ only [V S44]. |
| **Blender 5.2** | **Sky:** Sky Texture in Single Scattering (the old Nishita model) or Multiple Scattering. Sun Elevation 2–6°, Sun Rotation chosen to put the disc behind the church, Altitude 0 km (beach), Air 1, Aerosols 2–4 for a warm haze, Ozone 1 [E]. These skies are very bright by default, so lower Exposure [V S47].<br>**Colour management:** AgX has been the default view transform since 4.0 [V S48].<br>**Aerial perspective:** Mist pass or a low-density volume scatter [E].<br>**Stylised option:** world colour gradient `#F29B50`→`#F3B95D` plus a Sun lamp at `#FFB46B` [E]. |

## 7. Authenticity do / don't
**Do**
- Use one wall colour per house, white or Cal White trims, and dark-wood or painted carpentry. Neighbouring houses should contrast in hue [R][E]. Walls stay lime-based [V S2].
- Make every opening vertical and plaster over lintels [V S2]. In Cartagena-style districts use wooden doors, windows and rejas (iron corroded by the sea) [V S1]. Iron grilles and balconies suit Cuban and Puerto Rican districts [V S19][S18].
- Put wooden balconies on **upper floors** facing the street, ≤1.2 m deep [V S2], mainly on casas altas (implied by [S1]) [E]. Ground floors get rejas, portones and shops on corners [V S1].
- Use a tejaroz over ground-floor windows, a zócalo band, and eaves on corbels [V S1].
- Set the church on the plaza, as its own block separated by a street, with palms or a flamboyant in the plaza [V S22].
- Weather the roofs: mix new and old tiles (`#DC704C`→`#936349`), especially on hillside towns seen from above [R].

**Don't**
- Don't paint the whole town white. Cartagena's white mandate imitated Andalusian villages and was later reversed [V S3]. Keep pure white to ≤15–20 % of facades [E].
- Don't use synthetic or neon colours, or stone and tile cladding on facades. Colours come from lime + mineral pigments [V S2].
- Don't put balconies on every floor of every house, or use glass box balconies. Bogotá's glazed "gabinetes" [V S1] and Lima's box balconies [E] belong to other regions.
- Don't mix Mexican hillside (Puerto Vallarta: crown tower, cobbles) and Cuban flat-roof Havana idioms in one block. Pick one sub-style per district [E].
- Don't make tall spires or Gothic pinnacles. Towers end in domes, cupolas or pyramids; in seismic zones they are low and stout [V S4][S21].

---
## Sources
One line per group; entries are separated by " · ". "snippet" means only a search summary was seen.

- **Architecture & colour:** [S1] López & Sierra, *Estoa* 12 (2018) https://publicaciones.ucuenca.edu.ec/ojs/index.php/estoa/article/download/1556/1225/4684 · [S2] IPCC Cartagena, *Cartilla de descontaminación visual* https://ipcc.gov.co/wp-content/uploads/2023/pdf/cartilla_descontaminacion_visual_del_centro_historico_de_cartagena_0.pdf · [S3] El Universal 2013, "Cartagena blanca" https://www.eluniversal.com.co/suplementos/dominical/2013/04/14/lente-de-la-nostalgia-cartagena-blanca/ · [S20] https://en.wikipedia.org/wiki/Architecture_of_Cuba · [S21] https://en.wikipedia.org/wiki/Spanish_Colonial_architecture · [S22] ArchDaily, Laws of the Indies https://www.archdaily.com/1024343/the-standardized-planning-of-latin-american-cities-tracing-the-blueprint-of-the-laws-of-the-indies · [S25] https://es.wikipedia.org/wiki/Teja · [S26] Diario de Cuba (snippet, 403) https://diariodecuba.com/cultura/1696764015_50243.html
- **Towns:** [S18] https://en.wikipedia.org/wiki/Old_San_Juan · [S19] https://en.wikipedia.org/wiki/Trinidad,_Cuba · [S23] https://en.wikipedia.org/wiki/Old_Havana · [S24] https://en.wikipedia.org/wiki/Ciudad_Colonial_(Santo_Domingo)
- **Churches:** [S4] Isaza Londoño, *Apuntes* 17, "Dos iglesias cartageneras del siglo XVI" https://revistas.javeriana.edu.co/index.php/revApuntesArq/article/download/9072/7376/34309 · [S5] https://en.wikipedia.org/wiki/Cartagena_Cathedral,_Colombia · [S6] https://en.wikipedia.org/wiki/Iglesia_de_San_Pedro_Claver,_Cartagena · [S7] https://en.wikipedia.org/wiki/Havana_Cathedral · [S8] https://en.wikipedia.org/wiki/Cathedral_of_Santa_María_la_Menor · [S9] https://en.wikipedia.org/wiki/Cathedral_of_San_Juan,_Puerto_Rico · [S10] https://en.wikipedia.org/wiki/Our_Lady_of_the_Assumption_Cathedral,_Granada · [S11] https://en.wikipedia.org/wiki/León_Cathedral,_Nicaragua · [S12] https://en.wikipedia.org/wiki/Iglesia_de_La_Merced,_Antigua_Guatemala (colour: snippet, https://www.frommers.com/destinations/antigua-guatemala/attractions/iglesia-la-merced) · [S13] https://en.wikipedia.org/wiki/Church_of_Santa_Prisca_de_Taxco (dome and stone: snippet, https://programadestinosmexico.com/2/es/arquitectura/templo-de-santa-prisca-taxco/) · [S14] https://en.wikipedia.org/wiki/Church_of_Our_Lady_of_Guadalupe_(Puerto_Vallarta) · [S15] https://en.wikipedia.org/wiki/Campeche_Cathedral · [S16] https://www.komoot.com/highlight/6337697 (user comments, low authority) · [S17] Lonely Planet (snippet) https://www.lonelyplanet.com/points-of-interest/iglesia-la-merced/432722
- **Vegetation:** [S27] https://en.wikipedia.org/wiki/Cocos_nucifera · [S28] UF/IFAS ST-177 (snippet) https://webarchive.library.unt.edu/eot2008/20090116081049/http:/hort.ifas.ufl.edu/trees/COCNUCA.pdf · [S29] UF/IFAS ST574 (snippet) https://ask.ifas.ufl.edu/publication/ST574 · [S30] https://en.wikipedia.org/wiki/Roystonea_regia · [S31] https://en.wikipedia.org/wiki/Sabal_palmetto · [S32] https://en.wikipedia.org/wiki/Musa_(genus) · [S33] https://en.wikipedia.org/wiki/Heliconia · [S34] https://en.wikipedia.org/wiki/Bougainvillea · [S35] https://en.wikipedia.org/wiki/Hibiscus_rosa-sinensis · [S36] UF/IFAS ST228 (snippet) https://ask.ifas.ufl.edu/publication/ST228 · [S37] https://en.wikipedia.org/wiki/Delonix_regia and https://en.wikipedia.org/wiki/Santa_Marta · [S38] https://en.wikipedia.org/wiki/Terminalia_catappa and https://en.wikipedia.org/wiki/Puerto_Vallarta. For [S37] and [S38], plant facts come from the first page and town facts from the second.
- **Terrain:** [S39] https://en.wikipedia.org/wiki/Sierra_Nevada_de_Santa_Marta · [S40] https://en.wikipedia.org/wiki/Banderas_Bay · [S41] Bosboom & Stive, *Coastal Dynamics* §7.3.2 https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/07%3A_Cross-shore_transport_and_profile_development/7.03%3A_Morphodynamics_of_the_upper_shoreface/7.3.2%3A_Beach_states · [S42] NASA Earth Observatory (snippet) https://science.nasa.gov/earth/earth-observatory/beautiful-cuba-144390/
- **Lighting & engines:** [S43] https://en.wikipedia.org/wiki/Golden_hour_(photography) · [S44] https://docs.godotengine.org/en/stable/tutorials/3d/environment_and_post_processing.html · [S45] https://docs.godotengine.org/en/stable/classes/class_proceduralskymaterial.html · [S46] https://docs.godotengine.org/en/stable/classes/class_physicalskymaterial.html · [S47] https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/sky.html · [S48] https://developer.blender.org/docs/release_notes/4.0/color_management/
- **Other [S] snippets used:**
  - Havana entresuelo: https://repositoriodigital.ohc.cu/files/original/40927ec2f0ac6da969f605e5dd46a12e8d6d625d.pdf
  - Mediopunto and persianas (FIU thesis): https://core.ac.uk/download/159753796.pdf
  - Trinidad wooden barrotes: https://travel2unlimited.com/?p=21301
  - Clay-tile minimum pitch: https://www.homecenter.com.co/homecenter-co/content/tejas-coloniales/
  - Tile size: https://www.easy.cl/teja-de-arcilla-colonial-caribe-28x46-cm-1416550/p
  - Banderas Bay depth: https://puertovallarta.net/fast_facts/banderas-bay and https://www.banderasnews.com/0808/vl-sospecial.htm
  - Santa Bárbara (Mompox) colour: https://www.viamichelin.co.uk/maps/tourist-attractions/poi/santa_cruz_de_mompox-_-bc6423ce433c
