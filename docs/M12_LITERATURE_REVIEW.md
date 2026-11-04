# Milestone 12 literature review: terrain-aware street-network generation

**Date:** 2026-11-04 · **Status:** Phase 1. This is a documented web search, not a systematic review (§6).

**Reading levels used below**
* **Full** — full text read.
* **Partial** — full text available, relevant sections read.
* **Abstract** — only the abstract or catalogue record was available.

Architecture is described only for papers read in full or in part. Full texts are kept locally under the session
scratchpad and are not committed (copyright).

---

## 1. Comparison table

| # | Work | Read | Output | Sequential | Terrain input | Water / bridges | Connectivity guaranteed |
|---|---|---|---|---|---|---|---|
| 1 | Chu et al. 2019, *Neural Turtle Graphics*, ICCV | Full | Planar graph | Yes (node queue, RNN encoder and decoder) | **None** | None | Yes, by construction |
| 2 | Parish & Müller 2001, *Procedural Modeling of Cities*, SIGGRAPH | Full | Graph (extended L-system) | Yes (road-end expansion) | Elevation map; land/water/vegetation map; population map | Road ends in water are pruned or rotated; **highways may cross illegal areas up to a set length**, flagged and later built as a bridge or a pair of tunnel portals | Yes (intersection search) |
| 3 | Chen et al. 2008, *Interactive Procedural Street Modeling*, ACM TOG 27(3) | **Not accessed** (the PDF link returned 404) | Graph traced from tensor fields | Streamline tracing | Reported (secondary sources) to align fields with boundaries and terrain; **unverified** | Unverified | — |
| 4 | Galin et al. 2010, *Procedural Generation of Roads*, CGF 29(2):429–438 | Abstract; cost terms via #5 (full text could not be downloaded, §6) | Path between two given points | No (one optimal path per pair) | Directional slope cost along the road; curvature cost | Water and vegetation costs; the path decides which segments become **bridges or tunnels** | — |
| 5 | Galin et al. 2011, *Authoring Hierarchical Road Networks*, CGF 30(7):2021–2030 | Partial | Hierarchical graph (highways, major, minor) of geodesic paths between settlements | Hierarchy level by level; paths are global optima, not sequential decisions | Cost c(p, ṗ, p̈) = weighted sum of vegetation v(p), water height w(p), directional slope s(p, ṗ), curvature g(p, ṗ, p̈), plus a user control cost h(p) | Generates geometry for **tunnels, bridges and interchanges**; merges nearby paths (Steiner nodes) | Yes (graph of paths) |
| 6 | Emilien et al. 2012, *Procedural Generation of Villages on Arbitrary Terrains*, Visual Computer 28:809–818 | Abstract | Settlements, roads and buildings | Iterative growth from interest maps | Building geometry adapts to local slope; roads cite Galin's max-slope paths | — | — |
| 7 | Fang, Yang & Jin 2020, *DeepStreet*, arXiv 2010.04365 | Full | **Raster** (inpainting a masked region, 256 px at 5 m) | No | One channel: SRTM elevation (~30 m), rescaled 0–255 | None | No; failure cases leave dead ends |
| 8 | Fang, Jin & Yang 2020, *Incorporating planning intelligence into deep learning*, arXiv 2010.04536 | Partial | Raster (inpainting) | No | Elevation + slope-aspect channels (SRTM) | None | No |
| 9 | **Fang et al. 2022**, *A topography-aware approach to the automatic generation of urban road networks*, IJGIS 36(10):2035–2059, doi 10.1080/13658816.2022.2072849 | Full | Raster (512 m patch at 2 m = 256 px; masked infill of 48–64 px) | No | 7 channels from a 30 m DEM (elevation, slope, aspect, 4 hillshades) at 2, 8 and 32 m; a learned gate and attention module (GE) and a geo-merging bypath | **None**: "waterways and drainage channels are not included" (§7) | No (raster); a graph is extracted for evaluation |
| 10 | **Gu et al. 2024**, *Generating Urban Road Networks with Conditional Diffusion Models*, IJGI 13(6):203, doi 10.3390/ijgi13060203 | Full | Raster (1 km patch, 128 px at 7.81 m) → post-processed to vector | No (DDPM) | SRTM 30 m elevation and slope (bilinear to 7.81 m, min–max to 0–255) + 9-class land use + **real intersections** | Water as a land-use class; "does not seem to be effective at predicting the bridge" (Fig. 6, case 4) | No |
| 11 | **Birsak et al. 2022**, *Large-Scale Auto-Regressive Modeling of Street Networks*, arXiv 2209.00281 | Full | Graph via a VQ-VAE index field → decoder → thinning and tracing | **Yes** (transformer over an index field, sliding window) | **None**: "we do not consider terrain as conditioning information … difficulty of obtaining training data" | **Land–water map** as conditioning; bridges not discussed | Graph tracing; not guaranteed |
| 12 | Li et al. 2024, *RoBus* dataset, arXiv 2407.07835 | Partial | Dataset (6-channel tiles: roads, buildings, heights, water, greenery, density) and baselines | — | No elevation channel (building heights only) | Water bodies as a channel | — |
| 13 | Kim, Jha, Schonfeld & Kim 2007, J. Transp. Eng. 133(2):71 | Abstract (secondary) | One highway alignment (3D) | Optimisation (genetic algorithm) | Earthwork cost on terrain | Chooses a **bridge or tunnel** versus embankment or cut | — |
| 14 | KGDRL-HAO 2026, Computer-Aided Civil and Infrastructure Eng. (Chalmers record) | Abstract | One highway alignment | RL (soft actor-critic) | Terrain-based costs | Design constraints | — |
| 15 | Dong et al. 2025, *DRoLaS*, ICMR 2025, doi 10.1145/3731715.3733316 | Bibliographic record only | Hierarchical road layouts (diffusion) | ? | Unknown | Unknown | Unknown |
| 16 | GraphWalker, IJCAI 2025 (*High-Fidelity Road Network Generation with Latent Diffusion Models*) | Abstract | Road graph from trajectories | — | None evident | — | — |
| 17 | CityDreamer (CVPR 2024), GaussianCity (CVPR 2025) | Abstract | 3D city scenes (neural fields, Gaussian splatting) | — | Not described in the abstracts | Not described | — |

**Datasets, metrics and code** (papers read in full or in part)

| Work | Data | Metrics | Code |
|---|---|---|---|
| Fang 2022 | Florence and Perugia (train), Rome and Siena (val/test); about 56k samples, 9,033 evaluated | MAE and MSE of the raster; length-weighted metric choice and integration errors after graph extraction | Data and code on figshare (doi 10.6084/m9.figshare.19606960) |
| Gu 2024 | 5 US cities (Chicago, Los Angeles, New York, Phoenix, Washington); 1 km patches with ≥ 2 km of road | FID, F1, IoU, difference of average degree, difference of average road length; regressions of road length on water, mountain and vegetation area | Stated as available (figshare) |
| Birsak 2022 | OSM: whole cities, US-wide | Comparison with CityEngine statistics; qualitative large-scale results | Not stated |
| NTG 2019 | Cities from OSM; SpaceNet for parsing | Distributional road metrics | Project page nv-tlabs.github.io/NTG |

---

## 2. What the literature establishes

1. **Feeding terrain rasters to a learned street generator has been done**, for raster models:
   * DeepStreet (2020): elevation channel;
   * Fang 2020 (arXiv 2010.04536): elevation and aspect;
   * Fang 2022: 7 channels with learned gating;
   * Gu 2024: elevation and slope.

   **A new contribution cannot be "elevation channels in a CNN."**
2. **Naively adding terrain channels did not help** in Fang 2022 (Model 1, "simply attaching seven channels"), and
   Gu 2024 found the terrain factors had a smaller effect than land use and given intersections. Two consequences
   for M12:
   * TN must be checked for whether it uses terrain at all: channel ablations, and sensitivity of the outputs to
     shuffled terrain;
   * TN must be compared with the rules-only control R.
3. **Fang 2022 reports a learned slope threshold of about 9% (5.14°)**, above which the gate pays more attention to
   terrain. This is a data-derived reference point for the grade thresholds of §9 of the M12 specification, from
   Italian cities only.
4. **Water has been treated as conditioning in an autoregressive graph model** (Birsak 2022, land–water map) and as
   a land-use class in a raster DDPM (Gu 2024). **Neither predicts bridges**: Gu reports failure at a bridge, and
   Birsak does not discuss them.
5. **Bridges, tunnels and slope costs are standard in procedural and optimisation methods:**
   * Parish & Müller 2001: water crossings flagged up to a length, then built as bridges or tunnels;
   * Galin 2010/2011: anisotropic slope, curvature and water costs, with bridges, tunnels and interchanges;
   * alignment optimisation (Kim et al. 2007; RL methods 2021–2026).

   These are hand-designed costs or single-route optimisers. They do not learn the street-network distribution from
   data.
6. **Every learned method found is raster-based, or graph-based without terrain.**
   * The raster methods guarantee neither connectivity nor planar topology. Fang and Gu evaluate inpainting with
     *given junctions or intersections*, a much easier conditional task than generation.
   * No learned method found represents grade separation (one street passing over another without a junction).

---

## 3. Novelty assessment (qualified)

**Within the searches listed in §6, none of the works found does the following:**
* (a) a **sequential neural graph generator** (NTG-like) whose step decisions are conditioned on a DEM and water;
* (b) explicit **infrastructure actions** (ground, bridge, viaduct, stop) attached to generated edges;
* (c) **feedback-based replanning** in which the result of a terrain feasibility check is fed back into the next
  neural proposal;
* (d) a **3D-consistent (non-planar) street graph** in which crossings at different levels are not junctions.

**Not claimed**
* (a) alone is close to Fang 2022 and Gu 2024 in idea (terrain-conditioned learned generation); it differs only in
  output representation.
* Rules for bridges and slopes (R, HR) are prior art (Parish & Müller; Galin).
* Any defensible novelty lies in **(c) and the evaluation of (b)–(d)**, and only if they work and survive a
  systematic search.

**Evidence still missing before any claim**
1. A systematic search in Scopus or Web of Science and Google Scholar (we only have a web search engine).
2. Full reading of Galin 2010 and Chen 2008, and of DRoLaS, whose content is unknown.
3. The 2025–2026 SIGGRAPH, CVPR and ICCV proceedings on city or road generation.

---

## 4. Consequences for the M12 design

* **Baselines.**
  * P (procedural): Galin-style anisotropic least-cost paths with slope, curvature and water costs, and bridges as
    costed segments.
  * R: the rule layer of Parish & Müller (prune or rotate, then a bridge up to a maximum length).

  Both are faithful reimplementations of published mechanisms, not strawmen.
* **Terrain channels.** Include an **elevation-only** and a **slope-only** ablation (Fang 2022 found aspect and
  hillshades carry little extra signal; check this on our data rather than assume it).
* **Evaluate with metrics that do not reward copying given structure.** Fang and Gu condition on real junctions.
  Our unconditional regime has no such input, and the conditional regime gives terrain only.
* **Expected difficulty.** Two independent groups found terrain signals weak in learned models. A negative TN
  result is plausible and must be reportable.

---

## 5. Engineering references for thresholds (to verify before use)

* Street grade limits depend on road class and design speed (e.g. the AASHTO *Green Book*); local streets in hilly
  cities exceed 15–20%. **No figure is used until it is read from the source.**
* Thresholds will be estimated from our own DEM-paired data (quantiles of observed edge grades per class), then
  compared with these references.

---

## 6. Search log and gaps

**Queries run (web search, 2026-11-04)**
1. Fang 2022 IJGIS DOI;
2. Gu 2024 IJGI 13(6) 203;
3. terrain-aware road network generation deep learning elevation conditioned graph (extended);
4. Galin 2010 anisotropic shortest path, slope, bridges, tunnels;
5. diffusion model road network generation terrain elevation conditioned 2025 (extended);
6. highway alignment optimisation deep RL, terrain, bridges, tunnels;
7. autoregressive graph generation street network transformer conditioned on terrain or land use 2024–2025
   (extended);
8. procedural road generation terrain bridge placement survey;
9. street network generation hilly terrain slope learned model 2025–2026 (extended);
10. DRoLaS;
11. 3D city generation terrain roads bridges 2025;
12. Emilien 2012 villages.

**Not accessed**
* Chen 2008: author page returned 404;
* Galin 2010: the author-hosted PDF timed out repeatedly and a retry is running;
* DRoLaS (ACM, paywalled);
* KGDRL-HAO and Kim 2007: only abstracts were seen.

**Search terms not yet covered:** "bridge-aware road generation", "neural road planning", "learned road
alignment" beyond highway optimisation, and "terrain-conditioned diffusion" for roads beyond Gu 2024. These must be
covered by the systematic search.
