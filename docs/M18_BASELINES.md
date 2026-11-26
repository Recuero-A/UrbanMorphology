# Milestone 18: external baselines and venues (phase 1c)

Compiled on 2026-11-25 from primary pages: official repositories through the GitHub API and raw files, full-text papers where noted, and official calls for papers. A research sub-agent did the search. These facts were then re-checked by hand on the official pages:

* the Eurographics 2027 short-paper deadline (Jan 22, 2027, 23:59 UTC; 4 pages);
* the SIGGRAPH 2027 deadlines (form Jan 21, paper Jan 28, upload Jan 29, 2027; 22:00 UTC);
* the GIScience 2027 deadlines (full papers Jan 29, short papers Apr 2, 2027; AoE);
* the MIT licence of `t-mw/citygen`;
* the GPL-3 licence and version 2.0.13 of `leastcostpath`.

Hour figures are estimates; nothing was installed or run.

---


How this was checked: official repos were read through the GitHub API and raw files. Papers were read as full-text PDFs where noted, otherwise from the abstract. Venue facts come from the official CFP pages. Hour figures are my own estimates; nothing was installed or run.

- **"not verified"** means I could not confirm the fact on a primary page.
- **[abstract only]** means the assessment comes from the abstract, a README or a third-party mirror of the abstract, not the full paper.

**Matched-input protocol (M18):** every method gets the same site boundary, DEM and water mask (512 m tile, 4 m grid), starting anchors, road classes, metric scale and permitted crossings.

---

## Recommendations

**Terrain-aware external baseline: `leastcostpath` (R, CRAN v2.0.13, GPL-3).** It belongs to the Galin-2010 family: least-cost roads over terrain.
- **Setup:** use the `"wheeled transport"` slope cost (critical slope `crit_slope`, default 12 %) plus `max_slope`, with 16-, 32- or 48-cell neighbourhoods. Water becomes a barrier, except at the permitted crossing cells. Build anchor-to-anchor routes with `create_FETE_lcps` (all pairs), then merge them into a planar graph ourselves.
- **Why this one:**
  - It is maintained and on CRAN.
  - Its anisotropic, vehicle-oriented slope costs come from the literature.
  - It takes a DEM plus our anchors directly and returns vector lines.
  - It runs on Apple Silicon without CUDA.
  - No faithful Galin 2010 or 2011 code exists.
- **Main fairness threat: scope mismatch.** It only routes trunks between anchors. It has no local-street infill and no curvature term. It also leaves grid-path artefacts, and the merge/simplify post-processing is ours. So it will score well on slope compliance and badly on morphology by construction.
- **Mitigation:**
  - Compare it only on the matched road classes that connect anchors.
  - Give it the same tuning budget as our method (sweep `crit_slope` and the neighbourhood size).
  - Publish the post-processing code.
- **Optional, higher-fidelity alternative:** a labelled Galin 2010 reimplementation on `skimage.graph.MCP_Flexible`, about 20–35 h.

**Morphology baseline: `t-mw/citygen` (TypeScript, MIT).** It is the most-used open implementation of Parish & Müller 2001.
- **Why this one:**
  - It is the canonical P&M-style street growth.
  - It has a clear licence and was pushed in 2026-04.
  - Its `localConstraints()` hook is the natural place to add P&M's own water and elevation constraints.
  - It runs on Node with no GPU.
- **Main fairness threat:** we must add the water and slope constraints, the density map, metric scaling and seeding ourselves. Its terrain behaviour and parameter tuning are therefore our design choices, which risks a weakened baseline.
- **Mitigation:**
  - Implement only the constraints described in the P&M paper.
  - Tune on held-out tiles with the same realism metric and the same budget as our method.
  - Release the patch.

**Learned baseline: none is clean.** No 2023–2026 learned vector road generator with released code and weights accepts DEM or water inputs.
- The only NTG code is an unofficial repo with no licence (`Cewein/Neural-Turtle-Graphics`). Its single-suburb weights appear to be from Balmain, Sydney (inferred from the weight filename).
- Use it only as a secondary sanity check, and only with the author's permission.
- Reviewers would also see it as an ablation of our own base model.

---

## Part A: external baseline feasibility

### A1. Terrain-aware road construction

| Candidate | Code / licence | Language and dependencies; last commit | macOS / Apple Silicon, no CUDA | Weights | Inputs | Output | Terrain / water | Published example or sanity check | Effort as-is / adapted (h) |
|---|---|---|---|---|---|---|---|---|---|
| **Galin, Peytavie, Maréchal, Guérin 2010**, "Procedural Generation of Roads", CGF 29(2):429–438, doi 10.1111/j.1467-8659.2009.01612.x | **No code found** (searched GitHub and the PDF text has no code link) → **reimplementation** | Python + numpy/heapq, or `skimage.graph.MCP_Flexible` (custom offsets = k-neighbourhood masks; overridable `travel_cost`) | Yes | n/a | Terrain, water/river/lake, forest maps; start and end point | Vector polyline, then road/bridge/tunnel geometry | **Yes.** Anisotropic A* on a grid with k-neighbourhood path-segment masks. Cost is a sum of transfer functions over slope (threshold → ∞), curvature, water (bridges), vegetation and tunnels, integrated along each segment (read in the full-text PDF). | Qualitative: switchbacks appear as the slope threshold drops; bridge vs detour as the water cost varies | Reimplementation **20–35**; matched protocol **+4–8** |
| **Galin, Peytavie, Guérin, Beneš 2011**, "Authoring Hierarchical Road Networks", CGF 30(7):2021–2030 (Pacific Graphics 2011), doi 10.1111/j.1467-8659.2011.02055.x. This is the "Beneš" (Bedřich) hierarchical paper. | **No code found** → reimplementation | as above | Yes | n/a | Terrain + settlements (cities, towns, villages) | Vector network (highways, primary, secondary) plus interchanges | Yes: a geometric graph under a non-Euclidean (terrain-cost) metric, then path merging into junctions [abstract only] | Qualitative hierarchy on a large terrain | **+30–50** on top of the 2010 reimplementation |
| **Jan Beneš, Wilkie, Křivánek 2014**, "Procedural Modelling of Urban Road Networks", CGF 33(6), doi 10.1111/cgf.12283. The other "Beneš" (urban, traffic-driven) paper. | **No code** (project page lists only the PDF and DOI) | — | — | n/a | Elevation map classified into land, water and forbidden areas; neighbouring cities; traffic function | Vector network | **Yes.** Lattice pathfinding graph; edges rejected over water, forbidden areas or above a maximum slope; cost = LENGTH(e)·max(1, O1·E_L,max); bridge edges as in Galin 2010; harbours (read in the full-text PDF). | Qualitative city growth | Full reimplementation **60–100** (traffic simulation); major-road lattice only **10–15** |
| **`leastcostpath`** (J. Lewis) — https://github.com/josephlewis/leastcostpath, CRAN https://cran.r-project.org/package=leastcostpath | **GPL-3** (DESCRIPTION file and CRAN) | R ≥ 3.4: terra, sf, igraph, foreach, gstat, Matrix, doParallel. v2.0.13, 2025-06-09 (on CRAN, not archived). | Yes in principle (CRAN arm64 binaries for terra/sf); not tested | n/a | DEM raster + origin/destination points; optional barrier and cost updates | sf LINESTRINGs (vector); cost corridors; LCP density | **Yes.** About 40 slope cost functions, including `"wheeled transport"` (crit_slope), Herzog, Llobera–Sluckin and Tobler; `max_slope`; 4–48 neighbours. Water: as a user barrier (set to 0) via `update_values`/`replace_values`. | Package examples; case studies listed in the README (e.g. Lewis 2021 Roman-road probabilistic LCP, JAMT; data availability not verified) | **2–4 / 8–14.** R↔Python via GeoTIFF/GPKG; water barrier with permitted crossing cells; anchors; merge and planarize overlapping LCPs with shapely/momepy; class labels. |
| **GRASS GIS `r.walk` / `r.cost` + `r.path`** — https://grass.osgeo.org/grass-stable/manuals/r.walk.html | GPL (GitHub API reports NOASSERTION; licence file not verified) | C; very active (pushed 2026-11-25) | Apple Silicon builds: not verified | n/a | DEM + friction raster + start/stop points | Raster cost and direction → path (vector after conversion) | Yes: anisotropic (Aitken/Langmuir walking-time model), knight's move (`-k`); null cells excluded (usable as a barrier) | Manual examples | **3–6 / 8–12.** The walking model is not a road model, so a custom friction is needed. |
| **scikit-image `skimage.graph`** (MCP, MCP_Geometric, MCP_Flexible) | BSD-3 (licence file not verified; GitHub API: NOASSERTION) | Python; active | Yes | n/a | Cost array, custom offsets | Index path | Engine only. MCP_Flexible allows overridden travel costs and arbitrary neighbour offsets, so it can host Galin's masks. Curvature needs extra state. | — | Engine for the Galin reimplementation (above) |
| **`otto-link/AlphaModelRoadNetwork`** — https://github.com/otto-link/AlphaModelRoadNetwork | **GPL-3.0** | Python: numpy, scipy, matplotlib; pushed 2023-08-05 | Yes | n/a | Heightmap PNG + city positions and sizes | Vector edges with weights | Partial. Molinero & Hernando 2020 alpha model: Delaunay over random dummy nodes, 3D distance √(d²+dz_weight·dz²), α discount for reusing roads (path merging). No water and no slope cap. | `python3 main.py` with `data/hmap.png` (seed 1) reproduces the README video | **1–2 / 6–10.** Replace dummy nodes with the 4 m grid; add water and crossings, anchors and a slope cap. |
| WhiteboxTools CostDistance/CostPathway — https://github.com/jblindsay/whitebox-tools | MIT | Rust; pushed 2026-05-26 | Yes (not tested) | n/a | Cost raster + sources | Raster path | Isotropic cost raster only (not verified in the docs) | — | 2–3 / 8–12 (weaker than leastcostpath) |
| `dunkean/magna-urbis` (relief → rivers → roads → towns) | AGPL-3.0 | TypeScript + Rust/WASM; pushed 2026-11-24 | Experimental arm64 DMG | Neural relief bundled | Own or imported relief | Map/3D | Yes (per README) | Not a published research method | Not assessed; weak as an academic baseline |

### A2. Procedural street form and local infill

| Candidate | Code / licence | Language and dependencies; last commit | macOS, no CUDA | Weights | Inputs | Output | Terrain / water | Published example or sanity check | Effort as-is / adapted (h) |
|---|---|---|---|---|---|---|---|---|---|
| **CitySprout** — Dobbie & Johansson, "In Pursuit of the Optimal City: Evaluation of Procedurally Generated City Layouts", Research Square preprint doi 10.21203/rs.3.rs-7492997/v1 (peer review not verified). Read from its supplementary material only. | https://github.com/camdobbie/CitySprout — **no licence file** (all rights reserved by default) | Python: networkx, matplotlib, scipy, numpy, shapely, tqdm, pandas, seaborn; 23 commits; pushed 2026-02-27 | Yes | n/a | Grammar (Organic, Grid, Hex, Line), seed, `intersectRadius`, size limits, stop criteria; starts from one node at (0,0) | NetworkX graph (vector), road types m/l/s | **No** | `exampleUsage.py`, `generatePaperFigures.py` | **1–2 / 10–16.** Anchors, boundary clip, water/slope checks, metres, class map. Licence permission needed. |
| **CityStack** — Post, Ledoux, Patil, "A proposed framework for generating plausible cities", ISPRS Annals XII-4/W1-2026, 277–284, published 2026-09-28, doi 10.5194/isprs-annals-XII-4-W1-2026-277-2026 (read in full text) | https://github.com/OliverJPost/CityStack (umbrella: 7 commits, last 2025-04-27, no licence shown). Submodules: `citypy` (Python, **GPL-3.0**, 2025-04-25) and `citystackgen` (Rust, **GPL-3.0**, 2024-11-18). | Python CLI + Rust CLI (Rerun GUI) | Probably (Rust and Python); not tested | n/a | OSM roads + Overture footprints for 43 cities → typology grid (100 m cells) → simulated annealing → P&M-style growth sampled from local typology statistics | Generated roads + building footprints; typology grid | **No.** The paper says the terrain and water constraint "is future work as it is not yet implemented". | Breda typology grid: about 1.3 M iterations reach 95 % similarity | **6–12 / 20–30.** The generator code predates the paper, so its match to the paper is not verified. Add terrain, water and anchors. |
| **Chen, Esch, Wonka, Müller, Zhang 2008**, "Interactive Procedural Street Modeling" (SIGGRAPH 2008; full text grepped for terrain/water terms) | **Official source:** http://www.sci.utah.edu/~chengu/street/streetmodeling.zip (archive dated 2010-10-22). C++ **MFC / Visual Studio 2005** GUI ("IBFV2D"); "copyright @2007 Guoning Chen"; **no licence file**; bundles Numerical Recipes headers (NR licence is restrictive). | C++/MFC/OpenGL/GLUT | **No** (Windows MFC) | n/a | Water map W, park/forest map F, **height map H**, population P, user tensor design | Vector streamlines (major and minor) | **Yes:** heightfield basis field from the elevation gradient; water boundary fields; bridges | Paper figures (Benue River, Willamette River) | Port the core **40–80**, or Python reimplementation **30–50** |
| ↳ third-party: **`ProbableTrain/MapGenerator`** (tensor field, in the style of Chen 2008) — https://github.com/ProbableTrain/MapGenerator | **LGPL-3.0** | TypeScript + Gulp; 1,490 stars; pushed 2024-05-09 | Yes (browser/Node) | n/a | Grid and radial basis fields, noise, procedural sea and river | SVG/PNG/STL; streamlines (vector internally) | Water only (procedural); **no heightfield basis** (verified in `basis_field.ts`: Grid and Radial only) | Hosted demo (probabletrain.itch.io/city-generator) | **1–2 / 20–35.** Headless core, heightfield basis, our water mask, anchors. |
| **Parish & Müller 2001** | No official code (CityEngine is commercial, excluded) | — | — | — | — | — | Paper: water and elevation constraints, plus a least-elevation pattern | — | — |
| ↳ **`t-mw/citygen`** — https://github.com/t-mw/citygen; demo and write-up http://tmwhere.com/city_generation.html | **MIT** (© 2015 Tobias Mansfield-Williams) | TypeScript + Vite; 252 stars; pushed 2026-04-08. Godot 3 port `t-mw/citygen-godot` (GDScript, 2021). | Yes (Node) | n/a | Simplex-noise population heatmap; highway/normal segments; priority queue with delays | Segments (vector) | **No** (code grep: heatmap only); `localConstraints()` is the hook to add them | Live demo | **0.5–1 / 12–20.** Headless runner 4–6; water/slope constraints 3–5; anchors, boundary, scale, classes 4–6; GeoJSON export 1–2. |
| ↳ **`josauder/procedural_city_generation`** — https://github.com/josauder/procedural_city_generation | **MPL-2.0** | Python 2/3; numpy, scipy, matplotlib, PyQt4 GUI; Blender visualisation; 597 stars; pushed 2023-01-12 | Likely, after fixes. PyQt4 is obsolete; float-index code in `getRule.py` will probably fail on modern NumPy (inferred from code). | n/a | Rule image (red = grid, green = organic, blue = radial), population density image, axiom start vertices, borders | Vertex graph (pickled); blocks, lots, buildings | **No** for growth (verified in `getRule.py`/`check.py`); the heightmap is used only for 3D | `test/Tester.py`; documented example | **2–6 / 12–18** |
| ↳ `phiresky/procedural-cities` | AGPL-3.0 | TypeScript implementation + survey (2020-11-12) | Yes | n/a | P&M-style | Vector | No | Demo | Not preferred (AGPL; stale) |

### A3. Learned road generation

| Candidate | Code / licence | Language and dependencies; last commit | macOS, no CUDA | Weights | Inputs | Output | Terrain / water | Sanity check | Effort as-is / adapted (h) |
|---|---|---|---|---|---|---|---|---|---|
| **Neural Turtle Graphics** (Chu et al., ICCV 2019, doi 10.1109/ICCV.2019.00462) | Official https://github.com/nv-tlabs/NTG contains only a README and `docs/` (project page): **no code, no weights**. | — | — | No | — | — | — | — | — |
| ↳ unofficial **`Cewein/Neural-Turtle-Graphics`** — https://github.com/Cewein/Neural-Turtle-Graphics | **No licence file** | PyTorch ≥1.8, networkx, osmnx, numpy, pyyaml, tqdm, matplotlib, pandas; 17 commits; 2025-04-29 to 2025-05-08 | Yes on CPU (device = cuda or cpu, no MPS; fine for GRUs). The config's weight path uses Windows backslashes, so it needs a fix. | Yes: `example/NeuralTurtleGraphics_model_blamain.pth`, apparently from Balmain, Sydney OSM (inferred from the path) | OSM `.osm` files; seed edges; max degree, minimum angle, planarity constraints | NetworkX graph (vector) | **No** | `output.png`, `output_train_glebe.png` in the repo | **2–4 / 15–25** plus retraining on our cities. Licence permission needed. |
| **CaRoLS** — Feng, Li, Li, Li, Shen, *Computers & Graphics* 133:104451 (2025), doi 10.1016/j.cag.2025.104451 **[abstract only, via a third-party mirror]** | **No code found** | — | — | No | Image of natural and socio-economic conditions (optional, unconditional also supported) | Raster layout image (VAE latent + diffusion) | Natural-condition channels; DEM not confirmed | Australian metropolitan dataset | Reimplementation **80–150** plus training; not recommended |
| **RoBus** — Li et al., *Computers* (MDPI) 15(8):488, published **2026-07-30**, doi 10.3390/computers15080488. Article licence CC BY 4.0 (Crossref); arXiv 2407.07835 (2024). | https://github.com/tourlics/RoBus_Dataset (Python, **no licence file**, pushed 2026-09-07). **Dataset licence not stated**; built from OSM (ODbL) and Microsoft footprints, so the implications are not verified. Data on Baidu Pan (Europe, US, Australia); extraction codes by email with your institution (not requested). | Data-preparation pipeline; pix2pix/CycleGAN copy; csv2xodr (SUMO/CARLA) | Yes for scripts | Baseline weights not listed | — | **72,400 paired samples, about 80,000 km²**. Aligned rasters: primary and secondary roads, land use (water, woods), road density, buildings with heights, POIs. Also NetworkX topology from skeletonised road rasters, text prompts and per-tile metrics. | Water yes; **no DEM** (verified from the README pipeline) | Reported: ControlNet road FID 20.78; topology-aware road baseline with traffic-convenience score 0.83 | As a realism reference only. A pix2pix baseline with an added DEM channel would be our own design: 10–20 |
| **GraphWalker** — Wang, Wen, Min, Luo, IJCAI 2025, 6307–6315, doi 10.24963/ijcai.2025/702 [abstract only] | Paper cites `github.com/JinmingWang/GraphWalker` (not checked). Found `JinmingWang/RNGen_GraphWalker` (MIT, Python 3.12, torch 2.5.1+cu121, pushed 2025-10-27). | — | Probably CPU/MPS with changes | Not verified | **GPS trajectories** (Tokyo/Shanghai) | Vector graph | No | — | **Not matchable** (we have no trajectories) |
| Birsak, Kelly, Para, Wonka 2022, arXiv 2209.00281 | Only evaluation code: https://github.com/twak/streetz (Apache-2.0, 2023-05-15) | Python | Yes | No generator | — | — | — | — | Useful **metrics** (circuity, block shape, transport ratio, betweenness, PageRank) |
| Belli & Kipf 2019 GGT (image-conditioned road extraction) | https://github.com/davide-belli/generative-graph-transformer (MIT, 2024-09-05) | PyTorch | Yes | Not verified | Image | Graph | — | — | **StreetMover distance** metric only |
| CityDreamer (CVPR 2024) | https://github.com/hzxie/CityDreamer, NTU S-Lab License 1.0 | PyTorch 1.13.1 + CUDA 11.7 + custom CUDA extensions | **No** without porting the extensions | Yes | — | Raster layout (not verified) | — | — | Not suitable |
| Elite Lanes (GECCO 2026 poster, arXiv 2603.20964); Niu et al. arXiv 2509.23804; CityGen arXiv 2312.01508 (CVPR 2025 workshop); Yang, Zhu, Feng arXiv 2305.08186 (cGAN → graph) | **No code found** [abstract only] | — | — | — | — | — | — | — | Not candidates |

---

## Part B: venues

### Is the Eurographics 2027 full-paper deadline past?

**Yes.** The official CFP (https://eg2027.isti.cnr.it/call-for-papers/) gives:
- abstract and submission form: Sep 25, 2026
- full paper: Oct 1, 2026, extended to **Oct 2, 2026, 00:59 AoE**
- reviews released: Nov 23, 2026
- rebuttal due: Nov 30, 2026

The next full-paper chance is Eurographics 2028, whose deadline is not yet announced. The EG 2027 short-paper and STAR calls are separate (STARs closed Oct 19, 2026).

### Venue table

| Venue | Official URL | Deadline (verified only) | Length / format | Review | Fit (i): method paper on topology control | Fit (ii): realism-vs-terrain benchmark |
|---|---|---|---|---|---|---|
| **EG 2027 full papers** (Lucca, May 10–14, 2027) | https://eg2027.isti.cnr.it/call-for-papers/ | **Passed:** Oct 2, 2026, 00:59 AoE | No maximum length; CGF format; in-person presentation | Double-blind, two-step, 5 reviews, rebuttal | **High** (generative AI and modelling in scope; the Galin terrain lineage lives here) | Medium-high ("welcomes new datasets and benchmarks") |
| **EG 2027 short papers** | https://eg2027.isti.cnr.it/call-for-short-papers/ | **Jan 22, 2027, 23:59 UTC**; notification Mar 26; camera-ready Apr 19 | **4 pages max**; anonymous; EG Digital Library; in-person, 10-minute talk | Anonymous; reviewer model not stated | Medium (compact SMC/potentials result) | Medium |
| **Pacific Graphics 2027** | Not yet announced (no PG2027 site found). Reference: https://pacificgraphics2026.github.io/cfp.html | **Not yet announced.** PG2026 had abstract Jun 1 and paper Jun 8, 2026 (AoE). | PG2026: journal track (CGF), 10–12 pages recommended; conference track (EG DL), 7–8 pages | PG2026: double-anonymous, rebuttal | High | Medium |
| **SIGGRAPH 2027 Technical Papers** (Anaheim, Aug 8–12, 2027) | https://s2027.siggraph.org/program/technical-papers/ | Submission form **Jan 21**, paper **Jan 28**, upload **Jan 29, 2027** (22:00 UTC); reviews Mar 8; rebuttal Mar 15; decisions Apr 5 | No maximum length. A Conference-paper acceptance needs ≤ 9 pages excluding references (final ≤ 10). ACM `acmtog` two-column. | 4 double-blind reviews; one track, Journal (TOG) or Conference acceptance | Medium (high bar; needs strong visuals and novelty) | Low |
| **CGF direct (regular issues)** | https://www.eg.org/wp/?p=149 (submit at https://wiley.atyponrex.com/dashboard?siteName=CGF) | Rolling | About 10 pages recommended excluding references (https://www.eg.org/wp/?p=4287) | Anonymity not required (p=4287); rest of the review model not verified. Presentation at EG: not verified. | **High** | Medium |
| **GIScience 2027** (Tongji University, Shanghai, Oct 18–22, 2027) | https://giscience2027.org/ (frames https://geods.github.io/GIScience2027/) | Full papers **Jan 29, 2027 AoE**; short papers **Apr 2, 2027 AoE**; abstracts and demos May 14; workshop proposals Feb 12; submission links TBA | Full: about 18 pages, LIPIcs (open access, CC BY). Short: about 6 pages, Zenodo. | Double-blind, at least 3 PC reviews | Low-medium | **Medium-high** |
| **ACM SIGSPATIAL 2027** | Not yet announced (sigspatial2027.sigspatial.org does not resolve; sigspatial.org shows no 2027 announcement). Reference: https://sigspatial2026.sigspatial.org/research-submission.html | **Not yet announced.** 2026: abstract May 29, paper Jun 5, 2026 (PT). | 2026: 10 pages + 2 appendix pages; ACM two-column; "Experiment, Benchmark & Experience" paper type | 2026: **single-blind** | Low-medium | **Medium-high** (E/B&E track) |
| **IJGIS** (journal) | https://www.tandfonline.com/journals/tgis20/about-this-journal (403 to my fetcher) | n/a | Not verified | Double-anonymised according to the T&F page as indexed by search; **not verified on the page** | Low-medium | Medium |
| **Computers, Environment and Urban Systems** (journal) | https://shop.elsevier.com/journals/computers-environment-and-urban-systems/0198-9715 | n/a | Not verified (Guide for Authors blocked) | **Not verified** | Medium | **High.** Scope (verified): "interdisciplinary … computer-based research on urban systems" with a geospatial focus. |
| **Environment and Planning B** (journal) | https://journals.sagepub.com/home/epb. Guidelines read through SAGE's mirror: https://sage.cnpereading.com/storage/ext/submission-guidelines/page/author-instructions_EPB.html | n/a | Original articles about 8,000 words, including references. **Urban Data/Code** articles ≤ 3,000 words (software or data must be in a public repository). | **Double-anonymised; two independent reviews needed** (verified in guidelines). Scope from the SAGE description (aims page 403): urban analytics, GIS, simulation, analytical methods for planning and design. | Low-medium | **High** (also Urban Data/Code for a benchmark release) |

### Recent (2025–2026) related papers per graphics venue, to judge fit

**Finding:** I found **no road or street-network generation paper** in any of these:
- EG 2025/2026, PG 2025/2026 or CGF regular issues 2025/2026. Checked with an EG Digital Library title search for road, street, urban, city, layout, procedural, terrain, village and settlement.
- SIGGRAPH 2025/2026 or SIGGRAPH Asia 2025. Checked against the Ke-Sen Huang lists.

The closest active community is the Galin, Guérin and Peytavie terrain-modelling group at EG and CGF, which is the Galin-2010/2011 lineage. The nearest neighbours are below.
- **Eurographics:**
  - EG 2026, "Terrain Synthesis and Authoring based on Iso-Contours" (Huftier et al.), https://diglib.eg.org/handle/10.1111/cgf70389
  - EG 2026, "Authoring Terrestrial Planets with Diffusion Models" (Borg, Gain, Guérin, Peytavie, Cani, Galin), https://doi.org/10.1111/cgf.70390
  - EG 2026, "HeatMat: Simulation of City Material Impact on Urban Heat Island Effect", https://diglib.eg.org/handle/10.1111/cgf70376
  - EG 2025 STAR, "Terrain Descriptors for Landscape Synthesis, Analysis and Simulation" (Argudo, Guérin, Schott, Galin; CGF 44(2)), https://doi.org/10.1111/cgf.70080
- **Pacific Graphics:**
  - PG 2026 conference paper, "Atmosphere-Aware Intrinsic Decomposition from a Single Terrain Image with Latent Diffusion Models", https://doi.org/10.2312/pg.20261032 (weak match)
  - Nothing on roads or cities in PG 2025/2026. The historical precedent is Galin 2011 (PG 2011).
- **SIGGRAPH / SIGGRAPH Asia:**
  - SIGGRAPH 2026, "InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation", https://doi.org/10.1145/3799902.3811080 (code: https://github.com/xandergos/terrain-diffusion)
  - SIGGRAPH Asia 2026 (conditionally accepted), "GreenScapes: Optimized Vegetation Configurations in Cities", https://greenscapes-project.github.io/
  - SIGGRAPH Asia 2025, "CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians", https://citygo-weihang.github.io/page/
  - SIGGRAPH 2025, "Aerial Path Online Planning for Urban Scene Updation", https://vcc.tech/research/2025/DroneUpdate
- **CGF regular issues:**
  - Liu & Benes, "Single-Shot Example Terrain Sketching by Graph Neural Networks", CGF 44(1), https://doi.org/10.1111/cgf.15281
  - Perche, Guérin, Galin, Peytavie, "Vector-Based Terrain Modelling", CGF 44(6), https://doi.org/10.1111/cgf.70160
  - I am treating issues 1 and 6 as regular issues; that is inferred from the issue numbers.
- **Outside the five graphics venues:** CaRoLS (learned road layouts) went to *Computers & Graphics* (2025), a further graphics-journal option.

**Implications:**
- **Method paper (i):** fits EG/CGF best when framed as terrain-aware modelling.
  - EG 2027 full has passed. The options are CGF direct (rolling), EG 2027 short (Jan 22, 2027), SIGGRAPH 2027 (Jan 28, 2027) or PG 2027 (dates not yet announced).
- **Benchmark (ii):** the best verified-date venue is GIScience 2027 (Jan 29 full, Apr 2 short). EPB (including Urban Data/Code) and CEUS are the journal routes. SIGSPATIAL's E/B&E track is an option once the 2027 call appears.

---

## Sources (opened)

**Venues**
- EG 2027 home, CFP, short papers and STARs: https://eg2027.isti.cnr.it/ · https://eg2027.isti.cnr.it/call-for-papers/ · https://eg2027.isti.cnr.it/call-for-short-papers/ · https://eg2027.isti.cnr.it/?p=1538
- SIGGRAPH 2027: https://s2027.siggraph.org/ · https://s2027.siggraph.org/program/technical-papers/
- PG2026 CFP: https://pacificgraphics2026.github.io/cfp.html
- GIScience 2027: https://giscience2027.org/ · https://geods.github.io/GIScience2027
- SIGSPATIAL: https://sigspatial2026.sigspatial.org/research-submission.html · https://www.sigspatial.org/
- CGF: https://www.eg.org/wp/?p=149 · https://www.eg.org/wp/?p=4287 · https://www.eg.org/wp/guidelines/
- CEUS: https://shop.elsevier.com/journals/computers-environment-and-urban-systems/0198-9715
- EPB guidelines: https://sage.cnpereading.com/storage/ext/submission-guidelines/page/author-instructions_EPB.html

**Paper lists**
- https://www.realtimerendering.com/kesen/sig2025.html, sig2026.html, siga2025Papers.htm, siga2026Papers.htm, eg2025Papers.htm, eg2026Papers.htm
- EG Digital Library search API (diglib.eg.org/server/api/discover/search/objects)
- Crossref

**Papers**
- Galin 2010 PDF: https://perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf
- Beneš 2014 PDF: https://cgg.mff.cuni.cz/~jaroslav/papers/2014-cities/2014-cities-paper.pdf and https://cgg.mff.cuni.cz/?p=1672
- Chen 2008 project and PDF: https://www.sci.utah.edu/~chengu/street_sig08/street_project.htm
- CityStack PDF: https://isprs-annals.copernicus.org/articles/XII-4-W1-2026/277/2026/
- CitySprout supplement: https://assets-eu.researchsquare.com/files/rs-7492997/v1/5f3ab11b2c10c35e840de7ca.pdf
- arXiv abstracts: 2305.08186, 2603.20964, 2509.23804, 2312.01508

**Repos**
- All repo READMEs, licences and dates were read through the GitHub API or raw files (URLs in the tables).
- leastcostpath on CRAN: https://cran.r-project.org/package=leastcostpath
- r.walk manual: https://grass.osgeo.org/grass-stable/manuals/r.walk.html
- skimage.graph docs: https://scikit-image.org/docs/stable/api/skimage.graph.html
