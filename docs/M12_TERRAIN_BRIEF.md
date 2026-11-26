# Brief for Milestone 12: terrain-aware street generation

**Purpose.** This brief gives a reader (a human collaborator or another LLM) enough context to write the
implementation prompt for Milestone 12. Section 1 summarises what exists. Sections 2–6 collect the ideas, constraints
and open questions for the terrain milestone. **Nothing in this document has been implemented.** It is written so
that the result could be publishable: every claim must be traceable to a file, criteria must be fixed in advance, and
nothing can be claimed as new without a literature check.

Repository: `github.com/Recuero-A/UrbanMorphology`, branch `milestone-11` (last commit `a08894e`). Python 3.12,
PyTorch on Apple MPS (M5 Pro, 48 GB), Blender 5.2.3, Godot 4.7.2. Free tools only. 167 tests pass.

---

## 1. What has been done (M1–M11)

### 1.1 Data

| Item | Value |
|---|---|
| Unit | 512 × 512 m tile in local UTM. Raster form: 128 × 128 px at 4 m/px. Vector form: street graph, buildings, green, parcels |
| Source | OSM Geofabrik extracts, one replication state (2026-10-28T21:21:02Z), clipped locally with osmium |
| Cities | Spain (Barcelona, Madrid, Valencia), US (New York, Chicago, Los Angeles), Japan (Tokyo, Osaka, Nagoya) |
| Sampling | Stratified (GHSL density × urbanisation era) inside each city's main built-up area (GHSL built-up ≥ 0.15) |
| Splits | **565 train / 19 validation / 19 test**. Validation and test tiles are ≥ 1,536 m from every training tile. Tile IDs are fixed by SHA-256 in `tests/test_split_integrity.py` |
| Train tiles per city | BCN 39, MAD 70, VLC 31, CHI 71, LAX 71, NYC 71, TYO 71, OSA 70, NGO 71 |
| Effective sample size for streets | About 300 independent 3 km spatial blocks, not 565 (M10 audit) |
| Parcels (M8) | Observed parcels only from Spain (Catastro INSPIRE; licence v1.0: raw data never committed) and the US (NYC MapPLUTO, LA County). Chicago is excluded (licence unclear). Japan has none |
| **Terrain** | **None.** All work so far assumes flat ground. There is no water layer, and the raster channels are roads, buildings and green only |

**Bias that matters for M12.** The tiles were drawn from dense built-up areas of mostly flat cities. Steep, hilly
and riverside land is under-represented. This has not yet been measured (see §3.1).

### 1.2 Model history (each milestone has a `docs/MILESTONE_*.md` with pre-registered criteria and results)

| M | What | Outcome |
|---|---|---|
| 1–2 | OSM to raster pipeline; 9-city pilot dataset | Done |
| 3 / 3b | Convolutional VAE; a spatial latent and road-weighted loss | Road reconstruction IoU 0.05 → 0.82; samples not realistic |
| 4 | PixelCNN prior over the VAE latent | Fragmentation halved; target not met |
| 5 | **NTG-inspired graph generator** (Chu et al., ICCV 2019) | Connected by construction; absolute-coordinate steps give skewed junctions |
| 6 | GraphTurtle v2 (steps in the turtle frame) + building placer | Valid buildings; exposure bias found |
| 7 | Frontage model; hard constraints | 0 overlaps, 0 buildings on roads |
| 8 | Parcel-aware generation (ParcelNet) | Failed pre-registered criteria; alignment improved |
| 9 | Tidy subdivision, BlockNets (4 MLPs), templates | Failed criteria; tidy rows achieved |
| 10 | Street candidates A/B/N/C; BF blocks; BlockGen DDPM | **N (denoising context noise) best**: S10 1.23 → 0.50; P3 fails for all |
| 11 | Procedural 3D (watertight meshes, façades, sidewalks, trees) → Blender → GLB → Godot walk/fly | Geometry valid; no photorealism claimed |
| post-11 | Temperature study; 5 new tiles; **3 × 3-tile cities** (1.5 km; seeds 2033–2035) by windowed growth | All geometric checks pass; blocks are cut at the 512 m seams |

### 1.3 The street generator that M12 extends: GraphTurtle N

Code: `src/urbanmorph/models/graph_gen.py` (`GraphTurtle`), `src/urbanmorph/graphs/{generate,sequence,road_graph}.py`,
`scripts/m10_train_roads.py --variant noise`, config `configs/m6_roads.toml`.

**Generation loop**
* A queue of open nodes. For each popped node, the model sees:
  * a **48 × 48 crop at 4 m/px (192 m)**, rotated so the incoming street points up (the turtle frame). It has 3
    channels: streets drawn so far, open nodes, outside the tile;
  * the node's existing arms and some scalar features.
* A CNN encoder feeds a GRU decoder. The decoder emits up to `MAX_T` new arms. Each arm has a 2D step from a
  **mixture of 10 bivariate Gaussians**, a stop logit and a street class (major, mid, local, minor).
* A start head (a 32 × 32 grid of 16 m cells) seeds new components.

**Geometric rules after each step** (not learned): clip to the tile; split at crossings; merge with a node within
`MERGE_M` = 8 m; form a T-junction with a street within 8 m; drop steps shorter than 8 m; otherwise add a node.
About half of all junctions come from these rules (M10 diagnosis). **This rule layer is the natural place for
terrain feasibility checks.**

**Training**
* Teacher forcing on real traces, with 8 symmetries.
* Variant N jitters context nodes with σ = 2.5 m (border nodes fixed) while targets stay clean. This fights exposure
  bias.
* Parameters: hidden 256, 1,381,507 parameters, 60 epochs (best epoch 59), about 79 min on MPS.
* Learning rate: cosine from 1e-3 to 5e-5, with a **mandatory 300-step warmup** (without it the decoder ReLU dies,
  giving a constant stop head; check `decoder_units_alive`).

**Sampling.** Temperature T scales every Gaussian σ (`log σ + log T` in `generate.py`). T = 0.5 gives the best S10;
this is exploratory.

**Windowed growth for multi-tile cities** (`scripts/m11_big_city.py`)
* 512 m windows overlapping by 128 m.
* Existing streets are passed in as `given`; dangling ends on earlier windows' edges are passed as `frontier` nodes,
  with their heading.
* Stubs on internal edges are pruned.
* A global terrain field fits this design directly: each window samples its crop from the same city-wide raster.

### 1.4 Evaluation protocol (`docs/M10_EVALUATION_PROTOCOL.md`) — reuse it

* **The tile is the unit.** Metrics are computed per tile, then distributions are compared across tiles.
* **S10** (lower is better; 0 means identical):
  * ten per-tile street metrics: orientation order φ, orientation entropy, junction-angle deviation, intersections
    per km², dead-end ratio, mean link length, street length per km², loops per km², share of 4-way junctions,
    number of components;
  * for each metric, the Wasserstein-1 distance between the 256 generated tiles and the 19 real validation tiles,
    divided by the validation IQR;
  * S10 is the mean of the ten.
* **Statistics.** Paired tile bootstrap, 2 training seeds, pre-registered pass/fail criteria (P1–P4), and a dated
  change log for any deviation.
* **Results so far.** A 1.23; N 0.50; N at T 0.5 0.30–0.33 (exploratory; **test-split confirmation still pending
  and not approved**). Real junction deviation is 4.2°; N at T 0.5 reaches 5.1–5.5°.
* **Conditional vs unconditional.** Unconditional runs receive no validation or test statistics. A run that uses a
  validation tile's real structure (here, its real terrain) must be labelled *conditional* and reported separately.

### 1.5 Downstream stages (rule-based, run on any street graph)

* **BF:** parcels and buildings (ParcelNet + BlockNets + procedural fixes; Spain and US only).
* **3D:** city format v0.1 → meshes → GLB.
* **What these stages assume:** a flat z = 0 plane everywhere (ground, roads, sidewalks, building bases, Godot
  collision).

### 1.6 Known open issues inherited by M12

* Test-split confirmation of N at T 0.5.
* Street class "flickers" along one street.
* Junction deviation fails P3 for every model.
* Generated Spanish blocks lack attached buildings (party wall 0.8 m against 29.8 m per building).
* Blocks are cut at multi-tile seams.

---

## 2. Goal and scope of M12 (from the user's brainstorm)

**Goal.** Generate a terrain (relief plus rivers and lakes) and a street network that responds to it:
* streets avoid or follow steep slopes;
* streets do not enter water except on bridges;
* the generator must cope with extreme cases: steep slopes, wide rivers, large lakes or coastlines.

The result should be shown in 3D and walkable in Godot.

**Scope assumptions**
1. **Solid ground everywhere.** No soft, unstable or unbuildable soil is modelled; buildability depends only on
   slope and water. Geotechnics is future work.
2. Terrain is a single-valued heightfield. Overhangs, caves and tunnels-as-geometry are out of scope; tunnels may
   appear only as graph edges, if at all.
3. **Main contribution: the streets.** BF and 3D need only the minimum terrain adaptation (§4.5).

**Candidate research question.** *Does conditioning a learned turtle street generator on terrain (elevation, slope,
water) reproduce how real street networks respond to relief and water better than (a) a terrain-blind generator and
(b) the same generator with only hard feasibility rules?*

---

## 3. Data: what exists and what would be needed

### 3.1 Terrain information in OSM (measured on the cached Cataluña extract: 37.6M nodes, 3.2M ways)

| Tag | Count | Usefulness |
|---|---|---|
| `ele` | 21,254 (mostly peaks and natural points) | Points only; not a terrain surface |
| `incline` | 13,741 of 1.2M highway objects (1.1%) | Too sparse to train on; usable as a spot check |
| `waterway` (lines) | 87,506 | River and stream centrelines |
| `natural=water` (polygons) | 14,419 | Rivers (riverbanks), lakes, reservoirs; gives true water width |
| `natural=coastline` | 3,344 | Sea boundary |
| `bridge` / `tunnel` / `layer` | 19,854 / 29,600 / 53,265 | Ground truth for crossings |
| `highway=steps` | 19,040 | A weak signal for steep pedestrian links |

**Conclusion.**
* OSM gives water and crossings, but **not elevation.** Elevation needs a DEM.
* Water and bridge data come from the cached extracts, so no download is needed.
* **The current tiles drop water entirely.** It has to be added as a new channel.

### 3.2 Elevation sources (all need explicit user approval before download; verify each licence)

| Source | Resolution | Note |
|---|---|---|
| Copernicus GLO-30 | 30 m, global | **A surface model (DSM)**: in cities it includes buildings and trees. Unsuitable for street-scale slope without correction |
| SRTM | 30 m | Surface-like; older; voids |
| Spain IGN MDT05 / MDT02 | 5 m / 2 m, LiDAR **terrain model (DTM)** | Licence to verify (expected open with attribution) |
| USGS 3DEP | 1 m or 10 m DTM | US public domain (verify) |
| Japan GSI | 5 m / 10 m DEM | GSI terms of use (verify) |

**Recommendation.**
* Use national DTMs at 2–10 m, resampled to 4 m to match the crop grid.
* A 30 m DEM blurs street-scale grades. Report the effect of resolution as a sensitivity analysis.
* Raw DEM files stay outside version control, like the cadastral data.

### 3.3 Terrain variety (the main risk)

A terrain-aware model can only learn terrain effects that its training data contains. Before any modelling:
* **Audit the slope and water distribution** of the 565/19/19 tiles: per-tile median and 90th-percentile slope,
  relief, and water share.
* **If the tiles are too flat,** expansion candidates are already in the cached extracts (no OSM download):
  * Kobe, Kyoto and Yokohama (backups in the M2 config);
  * San Diego;
  * Barcelona's hill districts (Collserola, Montjuïc);
  * LA hillsides;
  * Seoul and Busan (South Korea extract). These were excluded in M2 for building completeness, which matters less
    for street-only training.
* **Rules for any expansion.**
  * It is a dataset expansion, so it needs approval.
  * The existing test split stays untouched.
  * New cities should be held out by city (leave-one-city-out) so that terrain generalisation can be tested.
* **Stratify every evaluation by slope class** (e.g. < 5%, 5–15%, > 15% median grade per tile) and by water
  presence. Otherwise flat tiles dominate the averages.

---

## 4. Method ideas

### 4.1 Terrain for synthetic cities (unconditional generation needs terrain too)

The options are listed from most to least defensible.

**(a) Real DEM patches** cut from training-area DEMs, never from validation or test areas. They are realistic by
definition and the simplest to defend in a paper. This is the recommended default.

**(b) Procedural terrain** for controlled stress tests:
1. Fractal noise (fBm, Perlin or simplex) for relief, with parameters for amplitude, roughness and dominant
   wavelength.
2. Hydraulic or thermal erosion, or a stream-power uplift–erosion model (Cordonnier et al. 2016), so valleys
   look realistic.
3. **Hydrology from the heightfield:**
   * fill depressions (priority-flood, Barnes et al. 2014). Filled depressions become **lakes**;
   * D8 flow directions, then flow accumulation, which gives drainage area A;
   * **rivers** are cells where A exceeds a threshold;
   * **river width** from hydraulic geometry: w = a·Q^b with b ≈ 0.5 (Leopold & Maddock 1953), taking discharge Q
     as proportional to drainage area A. a and b are fitted to OSM river polygons in the training regions;
   * carve each channel into the DEM.

   This gives rivers whose width is physically consistent instead of hand-set.
4. Stress parameters to sweep: maximum slope (0–60%), river width (5–500 m), lake size, coastline.

**(c) A learned terrain model** (e.g. diffusion on DEM patches). Not justified at our data size; mention as future
work only.

### 4.2 Terrain-conditioned GraphTurtle (the core experiment)

**New crop channels.** Rotate every channel with the turtle frame. Note that the slope *vector* must be rotated
too, not just its raster.

| Channel | Encoding |
|---|---|
| Relative elevation | Δz = z(cell) − z(current node), scaled, e.g. ÷ 20 m and clipped |
| Slope magnitude | Percent grade |
| Water mask | 1 = water |
| Optional | Existing-bridge mask, or a coastline/sea mask |

**New scalars**
* the terrain gradient at the node in turtle coordinates (∂z/∂forward, ∂z/∂left);
* the node's elevation relative to the tile median.

**Training**
* Use the same N recipe (context noise, 8 symmetries, warmup), with DEM and water rasters paired to every real tile.
* Under the symmetries, the DEM must be transformed together with the streets.
* **Retraining is justified here:** a new input modality is exactly the "compelling research justification" the
  project rules require. It still needs approval before it starts.

**What it should learn from data**
* streets that follow contours on steep ground, and switchbacks;
* longer, more winding links on hillsides;
* lower street density on steep ground;
* water crossed only at bridges that are roughly perpendicular to the river.

### 4.3 Hard feasibility layer (extends the existing rules in `generate.py`)

**Grade.** Reject or resample a step whose grade |Δz| / length exceeds G_max. Prefer **truncated sampling**
(redraw from the mixture, at most k tries, then stop the arm) to plain rejection, and report how often each
happens.

**Water.**
* A step that crosses water is allowed only as a **bridge**, and only if all of these hold:
  * its span is ≤ L_bridge;
  * its angle to the river axis is ≥ θ_min;
  * it is at least D_min from another bridge on the same river.
* Otherwise the arm stops.
* A lake or the sea is a hard boundary (no bridges, or only below a small L_bridge).

**Data-derived thresholds.** G_max, L_bridge, θ_min and D_min must be **estimated from the training data**:
* G_max: a high quantile of real edge grades from the DEM;
* L_bridge, θ_min, D_min: OSM `bridge=*` edges intersected with water polygons.

Sanity-check them against design references. Road-design manuals (e.g. the AASHTO *Green Book*) set maximum grades
by design speed and terrain type; steep urban streets in hilly cities exceed 15–20%. Verify any figure before
citing it. Fixing thresholds from data, before seeing results, avoids tuning them on validation.

### 4.4 Experiment design (pre-register before any training)

**Models.** All use the same training recipe, 2 seeds and 256 samples per seed.

| ID | Model | Role |
|---|---|---|
| N0 | Current N, terrain-blind; streets laid over the terrain afterwards | Reference |
| R | N0 + the hard feasibility layer (§4.3) | Rules only |
| TN | Terrain-conditioned GraphTurtle (§4.2), without the layer | Learned only |
| TN+R | TN with the feasibility layer | Main candidate |
| G | Procedural least-cost-path baseline in the style of Galin et al. 2010/2011 (anisotropic cost from slope, curvature and water; bridges as costed edges) | Optional classical baseline |

**Metrics**
1. **S10**, unchanged, overall and **per slope/water stratum**.
2. **New terrain metrics** (define them exactly in the protocol):
   * distribution of edge grades (median, 90th and 99th percentile; share above G_max);
   * **contour alignment**: the angle between each street and the contour, as a function of local slope;
   * street length and intersection density per slope bin;
   * circuity (network over straight-line distance) per slope bin;
   * share of street length in water: should be 0 except bridges;
   * bridges per km of river and per km² of tile; bridge length; bridge angle to the river;
   * connectivity across water (components on each bank).
   * The terrain metrics enter a separate score (call it S_T) using the same W1/IQR method, so S10 stays comparable
     with M10.
3. **Validity:** violation counts (grade above G_max, non-bridge water crossings). These must be 0 for R and TN+R.

**Evaluation regimes**
* **Unconditional:** terrain from training-area DEM patches or procedural terrain; compared with real validation
  tiles by stratum.
* **Conditional:** generate on the real validation tiles' terrain and water; labelled conditional and reported
  separately. This regime best shows "response to terrain", because real and generated streets share the ground.
* **Stress tests** (exploratory, no pass/fail): procedural slope and river-width sweeps. Report the violation rate,
  the share of the tile left empty, bridge count against river width, and the failure modes, with figures.

**Statistics.** Paired tile bootstrap, criteria fixed in advance (e.g. TN+R improves S_T over R by ≥ 20% with a
95% interval below 0, while S10 does not worsen by more than 0.1). The **test split is used once, at the end, with
approval.**

**Ablations.** Remove each channel in turn (elevation, slope, water); compare DEM resolutions (4 m from DTM vs 30 m).

### 4.5 Minimum terrain adaptation in BF, 3D and Godot

**Parcels and buildings (BF)**
* Exclude water, and slopes above a buildability threshold S_build (from data: the slope distribution under real
  building footprints).
* Otherwise leave BF unchanged and list slope-aware lot orientation as future work.

**3D**
* A terrain heightfield mesh, with UVs and materials.
* **Roads draped with a smoothed long profile.** Clamp the grade to G_max and add a small cut/fill, so roads do not
  follow DEM noise.
* Sidewalks follow the road profile.
* **Buildings sit on a plinth**: the base at the lowest ground point under the footprint, stepped on steep lots.
* Water surface polygons; simple bridge decks with piers.

**Godot**
* Heightfield collision for the terrain, plus trimesh collision for everything else.
* The existing headless checks (ray hits, walker on floor) extended to sloped ground.

**Multi-tile cities.** One global terrain raster; the windowed growth samples crops from it.

---

## 5. Literature

**Verified in this project's earlier search** (DOI or venue checked; the content still needs reading in full before
citing):

| Work | Relevance |
|---|---|
| Parish & Müller 2001, SIGGRAPH "Procedural Modeling of Cities" | L-system roads; global goals + local constraints (water, elevation); bridges over short spans |
| Chen et al. 2008, ACM TOG 27(3), doi 10.1145/1360612.1360702 | Tensor-field street modelling; fields can follow terrain and rivers |
| Galin et al. 2010, CGF 29(2):429–438, doi 10.1111/j.1467-8659.2009.01612.x | Roads as anisotropic shortest paths; cost from slope, curvature, water; bridges and tunnels |
| Galin et al. 2011, CGF 30(7) | Hierarchical road networks on terrain |
| Emilien et al. 2012, Visual Computer 28:809–818, doi 10.1007/s00371-012-0699-7 | Villages and roads adapted to terrain slope |
| Chu et al. 2019, ICCV "Neural Turtle Graphics" | The base of our generator; does not model terrain |
| DeepStreet, arXiv 2010.04365 | Learned street generation (check its terrain handling) |
| Fang et al. 2022, IJGIS 36(10):2035–2059, doi 10.1080/13658816.2022.2072849 | Learned street generation (check its terrain handling) |
| Gu et al. 2024, ISPRS IJGI 13(6):203 | Check its terrain handling |
| CityDreamer, CVPR 2024 | Unbounded 3D city generation from generated semantic and height maps (check how it treats ground relief) |

**Well-known, to verify before citing:**
* Génevaux et al. 2013, ACM TOG 32(4): terrain from hydrology;
* Cordonnier et al. 2016, CGF 35(2): uplift and fluvial erosion;
* Galin et al. 2019, CGF 38(2): a review of digital terrain modelling;
* Barnes et al. 2014: priority-flood depression filling;
* Leopold & Maddock 1953, USGS Professional Paper 252: hydraulic geometry;
* Allen & Pavelsky 2018, *Science*: the GRWL global river widths.

**How the literature handles the user's two questions, as far as known.**
* **Slope:** a cost that rises with grade, plus a maximum grade above which a road is impossible (Galin 2010).
* **Rivers:** a hard boundary with bridges allowed up to a maximum length or a bridge cost (Parish & Müller 2001;
  Galin 2010).
* **Lakes or sea:** treated as obstacles.
* **Gap:** none of these is a *learned* generator that infers the response to terrain from real data. **That is the
  plausible gap, but it may only be claimed after a systematic search** (Scopus or Google Scholar, ≥ 3 queries,
  2019–2026), recorded in a `M12_LITERATURE.md` file.

---

## 6. Standing constraints (carry into the prompt verbatim)

* Do not modify or read the held-out test split; use it once at the end, with approval.
* Do not use validation or test statistics as conditioning in unconditional evaluation. Real validation terrain is a
  conditional evaluation and must be labelled as one.
* Do not download data (DEMs included) or expand the dataset without approval. Record licences and provenance; keep
  restricted raw data outside git.
* Do not commit datasets, checkpoints, DEMs, GLBs or renders.
* Wait for approval before substantial training. Always use the 300-step warmup and check `decoder_units_alive`.
* Do not claim novelty without the systematic search, photorealism without a perceptual study, or success without
  tests and pre-registered criteria.
* Do not invent geographic coordinates for synthetic cities.
* Preserve all previous milestone outputs. Final 2D figures must draw parcels.
* Free tools only; macOS on Apple Silicon. The user wants concepts explained and answers concise.

**Suggested M12 phases** (each ends with a report and waits for approval):
1. Literature search and terrain/water audit of the existing tiles.
2. Data plan (DEM sources and licences, expansion cities).
3. Pre-registration.
4. Terrain synthesis and hydrology.
5. Feasibility layer (R).
6. Terrain-conditioned training (TN).
7. Evaluation and stress tests.
8. 3D and Godot terrain.
9. Test-split confirmation.
