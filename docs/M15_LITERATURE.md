# Milestone 15 literature review: terrain-adaptive street growth with style control

Compiled on 2026-11-20, before any M15 implementation. It extends `docs/M14_LITERATURE_AND_NOVELTY.md` and `outputs/m14/literature/road_generation.md`.

**Research question.** Can a turtle graph adapt street growth to terrain while preserving a requested street morphology and producing a range of plausible layouts on unseen sites?

**Access levels.**
* **F**: full text read.
* **F\***: full text read through a summarising web-fetch tool, so details are slightly less certain.
* **A**: abstract or catalogue record only.
* **N**: none.

Nothing marked A or N supports a technical claim here.

## 1. Retrieval log

| Paper | Access | Source / notes |
|---|---|---|
| Parish & Müller 2001 | F | Author PDF (cgl.ethz.ch). M14 had only the abstract. |
| Chen et al. 2008 | F | Author PDF (oregonstate.edu/~zhange); project page and code zip checked. |
| Galin et al. 2010 | F | Author PDF (perso.liris.cnrs.fr/egalin). Cost, results and conclusion sections read; the rest searched by keyword. |
| CitySprout | F\* (paper), F (supplement) | "CitySprout" is the tool's name. The paper is Dobbie & Johansson, *EPJ Data Science* 2026, read via publisher HTML. The Research Square preprint supplement was read as a PDF. |
| CityStack | F | Found via OpenAlex: an ISPRS Annals 2026 paper. Queries that returned nothing: arXiv API `all:CityStack` and `ti:CityStack`; Crossref "CityStack"; web searches for "CityStack" with arxiv, street network, or 2026. |
| CaRoLS | A | Crossref and Semantic Scholar metadata (no abstract), plus an abstract summary from a library catalogue record seen through a search engine. ScienceDirect returned 403; OpenAlex marks it closed access. |

## 2. Per-paper notes

### 2.1 Parish & Müller 2001 (F)

Y. I. H. Parish, P. Müller, "Procedural Modeling of Cities", *SIGGRAPH 2001*, 301–308, doi:10.1145/383259.383292.

* **Representation:** vector graph grown by an extended L-system. Procedural.
* **Terrain:**
  * Water and parks are hard masks. A road end inside them is pruned, then rotated "within a maximal angle", which makes roads follow coastlines.
  * Highways may cross illegal areas "up to a specified length", which become bridges or tunnels.
  * Elevation acts only through the *San Francisco* global goal: main roads follow least elevation, and short connectors follow the steepest elevation.
  * There is no grade limit.
* **Style:**
  * Rules: Basic (follows population; no pattern), New York (grid with a global or local angle and a maximum block size), Paris (radial around a computed or manual centre), San Francisco.
  * **Mixing:** "If more than one pattern rule is active at a given location, all of them are evaluated. The proposed parameter values are summed up and weighted according to the value in the input image grey scale map" (Fig. 7 blends New York and Paris).
  * No cul-de-sac control.
* **Hierarchy:** highways seek population peaks by ray sampling; streets fill the space between them using the dominant pattern.
* **Connectivity:** crossings are cut back to make an intersection, and ends near a road or crossing are extended to join it ("the dead end road is the exception").
* **Code:** none (the method became the commercial CityEngine).
* **Evaluation:** qualitative only. Manhattan is shown beside the real map; no metrics, baselines or user study.

### 2.2 Chen, Esch, Wonka, Müller & Zhang 2008 (F)

"Interactive Procedural Street Modeling", *ACM TOG* 27(3), Art. 103, doi:10.1145/1360612.1360702.

* **Representation:** a tensor field on a 512×512 grid. Streets are hyperstreamlines of its eigenvector fields, and their intersections form a vector graph. Procedural and interactive.
* **Terrain:**
  * Inputs: water, park/forest, height and population maps.
  * A *boundary field* aligns roads with water edges.
  * A *heightfield field* sets the minor eigenvector along ∇H, so roads run along contours or down the fall line. This is a heading preference, not a grade limit.
  * Narrow water may be bridged, depending on bridge length and crossing angle.
* **Style:**
  * Grid and radial (centre element) basis fields, plus the boundary and height fields.
  * **Composable by construction:** T(p) = Σᵢ exp(−d‖p−pᵢ‖²)Tᵢ(p) (Eq. 3).
  * Organic irregularity comes from Perlin-noise rotation fields R1–R3 and graph noise (occasional dead ends).
  * Brush strokes and smoothing are also available.
  * No cul-de-sac control.
* **Hierarchy:** major roads are traced first. Minor roads are traced per region with a field that may differ and be discontinuous across major roads. **This is the closest precedent for different styles on the backbone and on local streets.** Density maps set the line spacing d_sep.
* **Connectivity:** major and minor lines are traced interleaved, because independent tracing "often leads to a disconnected street network". Tracing continues for d_lookahead to find an intersection.
* **Code:** `streetmodeling.zip`, linked from sci.utah.edu/~chengu/street_sig08 and reachable today: a Visual Studio C++ project (58 .cpp and 62 .h files, dated 2010) plus a manual. Not built.
* **Evaluation:** qualitative cities (Taipei, Portland, Manhattan) and timings (about 5 min per layout plus 30–60 min of tuning). No metrics, baselines or user study.

### 2.3 Galin, Peytavie, Maréchal & Guérin 2010 (F)

"Procedural Generation of Roads", *Computer Graphics Forum* 29(2), 429–438, doi:10.1111/j.1467-8659.2009.01612.x.

* **Representation:** one road between two given points. It is a discrete anisotropic shortest path using k-neighbourhood segment masks (k = 5 recommended), fitted with clothoids. Procedural optimisation.
* **Terrain:**
  * The cost is a weighted sum of transfer functions of directional slope, curvature, water depth and vegetation.
  * Each transfer function "should return infinity" above a threshold κ₀. **The maximum grade is therefore a hard limit**; below it, slope is a soft cost.
  * Bridges and tunnels are costed in the same search. A low maximum slope makes the search choose a tunnel (Fig. 22).
* **Style, hierarchy and connectivity:** none, since it builds one path. Galin et al. 2011 adds levels.
* **Code:** none released (the authors' implementation is C++).
* **Evaluation:** tables of cost, time and length against mask size (under 1 s on a 100×100 grid). It claims to compare "favorably" with McCrae & Singh 2009, without numbers. No real-data comparison.

### 2.4 CitySprout: Dobbie & Johansson 2026 (F\* / F)

"Comparing urban network morphologies: a new living metric and a framework for evaluating procedurally generated city layouts", *EPJ Data Science* 15, Art. 76 (published 2026-08-31), doi:10.1140/epjds/s13688-026-00694-7. Preprint: Research Square, doi:10.21203/rs.3.rs-7492997/v1.

* **Representation:** planar NetworkX graph in arbitrary units (default merge radius 0.8). Procedural, stochastic, parametric L-system.
* **Terrain:** **none.** Geography is omitted "to ensure that the structural properties of each grammar are not affected by other factors". The conclusion only remarks that organic layouts *could* be "molded around … rivers and areas of steep terrain".
* **Style:**
  * Four fixed grammars: Organic, Grid, Hex (six radial arms with grid infill) and Line, one per city.
  * Each grammar is a set of production rules (deflection angles, lengths, probabilities, minimum distances).
  * No mixing was found; no cul-de-sac control.
* **Hierarchy:** main, large and small roads.
* **Connectivity:** crossing edges are cut back to the intersection, and close nodes are merged. The authors say this yields a planar, connected network.
* **Code:** github.com/camdobbie/CitySprout, in Python. No licence file is visible. Not run here.
* **Evaluation:**
  * 40 cities per grammar.
  * Six OSMnx cities (London, Rome, Tokyo, Chicago, Phoenix, Salt Lake City), 31 hexagons each.
  * Metrics: a *living metric* (ht-index of road betweenness), population density (taken as proportional to road length) and **mean circuity** (network distance / straight-line distance).
  * Medians and IQRs, leave-one-seed-out stability, no significance tests, no external baseline.

### 2.5 CityStack: Post, Ledoux & Patil 2026 (F)

"A proposed framework for generating plausible cities", *ISPRS Annals* XII-4/W1-2026, 277–284 (3D GeoInfo 2026), doi:10.5194/isprs-annals-XII-4-W1-2026-277-2026. It is based on Post's 2024 TU Delft MSc thesis *The City Stack*.

* **Representation:** a 100 m typology-grid raster, with vector minor roads grown by an adapted Parish–Müller L-system.
* **How it is learned:**
  * A GMM over street-morphology metrics from 43 OSM cities gives 13 road typologies.
  * Simulated annealing rearranges the typology grid.
  * Segment length, angle and branching are "sampled from the typology template statistics of the local typology grid cell".
* **Terrain:** **not implemented.** Roads are to be "constrained by terrain, water bodies, and previously generated segments (but this is future work as it is not yet implemented)".
* **Style:** data-derived typologies (for example regular grid, curvy, disconnected), varying per cell. There is no weighted blending and no radial or cul-de-sac control.
* **Hierarchy:** minor roads only.
* **Connectivity:** not yet implemented.
* **Code:** github.com/OliverJPost/CityStack (`citypy`). Not run.
* **Evaluation:**
  * Typology interpretability.
  * About 95% similarity of an annealed grid to the input grid (Breda).
  * In the authors' words: "We do not yet provide … a quantitative comparison between final generated geometries and real cities".
  * Useful metrics: **continuity** (stretch length / 500 m) and **stretch curvilinearity**.

### 2.6 CaRoLS: Feng, Li, Li, Li & Shen 2025 (A only)

"CaRoLS: Condition-adaptive multi-level road layout synthesis", *Computers & Graphics* 133, 104451, doi:10.1016/j.cag.2025.104451 (DOI verified via Crossref).

* **From the abstract summary only:**
  * a VAE pre-trained on real road layouts;
  * latent diffusion, optionally conditioned on "an image with natural and socio-economic information";
  * a condition-aware decoder block replacing the U-Net decoder blocks;
  * an Australian metropolitan dataset, on which it reportedly beats general and specialised methods.
* **Not verified:** representation (probably raster, given the VAE and U-Net), terrain channels, style control, hierarchy ("multi-level" is in the title), connectivity, metrics, held-out sites.
* **Code:** none found.
* **Sibling paper:** DRoLaS (ICMR 2025, doi:10.1145/3731715.3733316) remains N (ACM returned 403).

### 2.7 Closest 2024–2026 competitors (search, not exhaustive)

| Work | Access | Terrain | Style control |
|---|---|---|---|
| RoadWeaver, Li et al., arXiv 2608.11580 (2026) | F\* | **none** | **Learned:** an 11-D condition (6-D style code, density, gridness, radialness, organicness, bearing entropy) via AdaLN, then tensor-field growth. No attribute formulas, no mixing tested, no held-out split, code "upon acceptance". |
| Kasiński & Szufel, *Urban Planning* 11:12126 (2026), doi:10.17645/up.12126 | abstract + keyword search of PDF | none found | An LLM prompted with "grid-like, radial, organic", calibrated by the Wasserstein distance of segment lengths. |
| Zhou et al., *CAVW* 35(3) e2267 (2024) | A (M14) | elevation, water, vegetation | unknown. This is the closest learned + terrain vector work. |
| Fang 2022 (IJGIS); Gu 2024 (IJGI) | F (M14) | elevation and slope | none (raster; ground-truth junctions given as input) |

Web, arXiv, Crossref and OpenAlex searches found **no 2024–2026 work that combines terrain adaptation with user-chosen street morphology**. Scopus and Web of Science were not available.

## 3. Capability comparison

Y = verified · N = no · ? = not verified · (y) = partial.

| | Vector | Learned prior | Terrain-adaptive growth | Water | Grade limit | Grid | Organic | Radial | Cul-de-sac | Mixed styles | Hierarchy | Connectivity | Quant. real-data eval | Held-out sites | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Parish & Müller 2001 | Y | N | (y) SF rule | Y prune / rotate / bridge | N | Y | (y) Basic | Y | N | **Y** weighted maps | Y | snapping | N | N | N |
| Chen et al. 2008 | Y | N | Y height heading | Y boundary field, bridges | N | Y | (y) noise | Y | N | **Y** RBF sum | **Y** separate fields | interleave, look-ahead | N | N | Y (C++ 2010) |
| Galin et al. 2010 | one road | N | Y cost | Y cost | **Y** (∞ above κ₀) | N | N | N | N | N | N | n/a | N | N | N |
| CitySprout 2026 | Y | N | N | N | N | Y | Y | (y) Hex | N | N | Y | merge / cut | Y (6 cities) | n/a | **Y Python** |
| CityStack 2026 | Y | (y) GMM | N (future) | N (future) | N | Y | Y | N | N | (y) per cell | N | N (future) | N (stated) | n/a | Y |
| CaRoLS 2025 | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | N found |
| RoadWeaver 2026 | Y | Y | N | N | N | Y attr. | Y attr. | Y attr. | N | not tested | ? | A\* repair | N | N | promised |
| **UrbanMorph M15 (planned)** | Y | Y (no style input) | Y | Y hard | Y hard | planned | planned | planned | planned | planned | backbone vs. local | M14 potentials | planned | planned | this repo |

## 4. Evaluation protocols

| Work | Data | Metrics | Baselines | Statistics |
|---|---|---|---|---|
| Parish & Müller | synthetic maps, Manhattan | none | none | none |
| Chen et al. | real water / height maps | timing | none | none |
| Galin et al. | example terrains | cost, time, length | qualitative only | none |
| CitySprout | 4 grammars × 40 cities; 6 OSMnx cities | living metric, density, mean circuity (grid reference 4/π) | none | medians, IQR, seed stability |
| CityStack | 43 OSM cities | GMM / BIC; landscape statistics by distance to centre | none | one example |
| RoadWeaver | about 58k OSM tiles, 144 cities | LCC, reachability, dead-end and cycle ratios | MetaDrive, RoadGen, HDMapGen | ± undefined |
| CaRoLS | Australian metro (abstract) | ? | ? | ? |

None of these papers measures style fidelity as a function of terrain difficulty, or tests style control on held-out real terrain.

## 5. Overlap and novelty

**Overlap:**

1. **Style heading fields are established procedural work.**
   * Grid and radial fields, weighted blending, terrain- or boundary-aligned headings and organic rotation noise all appear in Chen 2008.
   * Pattern rules blended by weight maps appear in Parish & Müller.
   * Separate backbone and local fields appear in Chen 2008.
   * **The M15 grid, radial and organic potentials and their mixtures are not new ideas.**
2. **Terrain adaptation is established.** Hard water masks with prune, rotate or bridge (Parish & Müller); height-aligned headings (Chen); slope cost with a hard maximum (Galin). Learned terrain conditioning: Fang 2022, Gu 2024, Zhou 2024 (unread), and our M12–M14.
3. **Learned style conditioning exists.** RoadWeaver conditions on gridness, radialness and organicness. NTG and Birsak use city identity. CityStack uses data-derived typologies.
4. **The sampler machinery is prior work.** K-candidate potentials, the M\* sampler, NA-fixed look-ahead, SMC and network potentials come from M13/M14 and the established SMC and look-ahead literature. NTG already multiplies its prior by an external likelihood at decode time. **M15 adds a potential, not a sampler.**
5. **Cul-de-sacs are handled by rules everywhere.** Separate termination and loop-closure controls were not seen as named controls, but this is a small difference.

**What may remain new (conservative).** Among the works read, none:
* composes user-specified style potentials (grid, organic, radial, cul-de-sac, and mixtures, applied differently to backbone and local streets) at inference time;
* applies them on a learned, terrain-conditioned vector prior trained without style input;
* does so under hard water and grade feasibility;
* evaluates the result quantitatively on held-out real sites for the trade-off between style fidelity, terrain compliance, realism and diversity, against matched procedural baselines.

Any novelty claim is limited to this **combination and its matched evaluation**. It remains conditional on:
* reading CaRoLS, DRoLaS and Zhou 2024 in full;
* RoadWeaver's attribute definitions;
* a Scopus or Web of Science search.

## 6. Recommended metrics

1. **Grid order.**
   * A length-weighted bearing histogram folded mod 90°, scored against the requested axis.
   * The share of intersection angles within ±10° of 90°.
   * Sources: RoadWeaver names "bearing entropy" and "gridness", without formulas. CityStack's regular-grid template has angles "centred near 90 degrees".
2. **Organic.** CityStack's continuity and stretch curvilinearity, computed on merged stretches. CitySprout's mean circuity, with 4/π as the ideal-grid reference.
3. **Radial.** Length-weighted angular deviation from the radial or tangential direction about the declared centre, binned by distance to the centre (the binning follows CityStack). **No paper read defines radialness numerically**, so this metric is our own and must be labelled as such.
4. **Cul-de-sac.** Dead-end ratio, cycle ratio, LCC ratio and reachability, as defined by RoadWeaver. Use the M14 window-closed variants, so that boundary stubs are not counted as dead ends. Add mean circuity to measure the detour cost.
5. **Terrain compliance.**
   * Length share above the grade limit, and the grade distribution along roads (Galin's directional slope).
   * Water-crossing count and length (Parish & Müller and Chen bridge rules).
   * **Style scores stratified by slope bin**, to show where terrain overrides style. Fang 2022's flat-versus-hilly split is the precedent.

## 7. Accessible external baselines

* **CitySprout.** The Python code is public (not run here), but it has no terrain or water, uses arbitrary units, offers radial only through Hex, has no cul-de-sac control and shows no licence. It is usable only as a style-only reference after scale calibration with our feasibility mask added, and must be reported as a modified baseline.
* **Chen-2008-style tensor-field tracer (recommended).** The paper specifies enough to re-implement it:
  * grid, radial, boundary and height basis fields;
  * the RBF sum (Eq. 3) and rotation noise;
  * hyperstreamline tracing with d_sep, d_lookahead and the five stopping criteria;
  * major roads first, then minor roads per region.

  It is the only method read that already combines style and terrain. On the same DEM, water and feasibility rules, it is a matched baseline for "style + terrain without a learned prior". The original C++ code can serve as a reference but is unlikely to build unchanged on macOS.
* **Parish–Müller global goals** are simpler to implement but weaker on terrain. CityStack has no terrain, so it cannot serve as a terrain baseline.

## 8. Open items

1. Full texts of CaRoLS, DRoLaS and Zhou 2024 (KAIST library).
2. RoadWeaver attribute formulas.
3. A Scopus or Web of Science search.
4. A single CitySprout run, to confirm it executes and to fix its unit scale.
