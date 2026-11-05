# Milestone 13 literature review: constrained, terrain-aware autoregressive road-graph generation

Search date: **2026-11-05**.

Prior reviews: `M12_LITERATURE_REVIEW.md` (terrain and road generation; scripted Semantic Scholar / arXiv search, §7).

This review adds the inference-side literature that the M13 method builds on:
* constrained decoding, sequential Monte Carlo (SMC), look-ahead search;
* constrained graph generation;
* the procedural terrain-routing literature, read in full where accessible.

**Rule:** method details are stated only from full text we read. Papers read only from abstracts or listings are
marked **A** (abstract-only), and no architectural claim is made about them.

## 1. Search protocol

**Databases / tools**
1. A general web search engine (the WebSearch tool; US index).
2. The arXiv API (`export.arxiv.org/api/query`, title search).
3. Publisher and repository landing pages (Eurographics DL, HAL, PMLR, ICLR, MDPI, Taylor & Francis mirrors).
4. Author-hosted PDFs (LIRIS).

No library database was available: Scopus / Web of Science need KAIST access (an open item since M12).

**Queries, all run 2026-11-05**

| # | Query | Purpose |
|---|---|---|
| Q1 | `Zhou "Transformer-Based Graph Generation Model for Vectorized Road Design" 2024 Computer Animation and Virtual Worlds` | priority paper 4 |
| Q2 | `Fang 2022 "topography-aware" automatic generation urban road networks` | priority 2 |
| Q3 | `Gu 2024 "Generating Urban Road Networks with Conditional Diffusion Models"` | priority 3 |
| Q4 | `Birsak "Large-Scale Auto-Regressive Modeling of Street Networks" code` | priority 5 |
| Q5 | `sequential Monte Carlo steering language models probabilistic programs constrained generation Lew 2023` | SMC decoding |
| Q6 | `terrain-aware road network generation deep learning 2025` | recent terrain work |
| Q7 | `NeuroLogic A*esque decoding lookahead heuristics constrained text generation` | look-ahead decoding |
| Q8 | `twisted sequential Monte Carlo probabilistic inference language models Zhao 2024` | twisted SMC |
| Q9 | `Loula 2025 syntactic and semantic control of large language models via sequential Monte Carlo` | SMC control |
| Q10 | `plug-and-play controllable graph generation constraints diffusion PRODIGY Sharma 2024` | constrained graph generation |
| Q11 | `Galin Peytavie Marechal Guerin "Procedural Generation of Roads" 2010 pdf anisotropic shortest path` | priority 7, 8 |
| Q12 | `Emilien Bernhardt Peytavie Cani Galin "Procedural generation of villages on arbitrary terrains" pdf` | priority 9 |
| Q13 | `constrained autoregressive graph generation hard constraints sampling valid graphs` | constrained graph generation |
| Q14 | `street network generation 2026 terrain slope generative model urban` (extended search) | recency, to Nov 2026 |
| Q15 | `road alignment optimization terrain earthwork cut and fill vertical alignment heuristic` | earthwork proxy |
| Q16 | `bridge location river crossing spacing urban street network empirical` | infrastructure metrics |
| Q17 | `Boeing street network grade elevation indicators global urban street networks circuity hilly` | metrics |
| Q18 | `sequential Monte Carlo graph generation particle filter generative model structured objects` | SMC on graphs |
| Q19 | `Lin Chen Liu "Lookahead strategies for sequential Monte Carlo" Statistical Science 2013` | look-ahead SMC theory |
| Q20 | `Naesseth Lindsten Schön "Nested Sequential Monte Carlo Methods" ICML 2015 proper weighting` | multi-candidate weighting |
| Q21 | arXiv API title search: `"Neural Turtle Graphics"`, `"Mastering Spatial Graph Prediction"` | IDs for priorities 1, 6 |

**Inclusion:** methods that generate or predict road / street graphs or rasters, conditioned on terrain or
geography; inference algorithms for constrained autoregressive generation (SMC, look-ahead, beam or tree search);
constrained graph generation; terrain-aware routing; empirical street-network metrics with elevation or bridges.

**Exclusion:**
* road *extraction* from imagery, except as search-method prior art (priority 6);
* traffic simulation networks;
* 3D road-surface reconstruction;
* hiking-trail inference from GPS;
* non-English full texts.

**Duplicates:** removed by DOI or arXiv ID. Q1, Q3 and Q14 returned the same Gu 2024 and Fang 2022 records;
mirrors (ResearchGate, Bohrium, DeepAI, aggregators) were collapsed into their primary record.

**Counts:**
* about 120 result links screened;
* 34 distinct records after de-duplication;
* 24 included (§2);
* 10 excluded as off-topic (road extraction from trajectories or imagery, simulation networks, 3D road surfaces).

**Full-text access**
* Read in full (text extracted with `pdftotext` from the downloaded PDFs):
  * Galin et al. 2010;
  * Emilien et al. 2012;
  * Chu et al. 2019 (arXiv 1910.02055);
  * Birsak et al. 2022 (arXiv 2209.00281);
  * Anagnostidis et al. 2023 (arXiv 2210.00828);
  * Lew et al. 2023 (arXiv 2306.03081);
  * Zhao et al. 2024 (arXiv 2404.17546);
  * Lu et al. 2022 (arXiv 2112.08726);
  * Loula et al. 2025 (arXiv 2504.13139).
* Not accessible:
  * **Gu et al. 2024** — MDPI returned HTTP 403 to our client and no mirror was used;
  * **Fang et al. 2022** — the HKU repository host was unreachable, the publisher is paywalled, and only the
    code / data record on Figshare was found;
  * **Zhou et al. 2024** (CAVW, doi 10.1002/cav.2267) — paywalled; institutional abstract pages only.

  We did not try to bypass access controls.

## 2. Included works

| # | Work | Venue | Access | What it does (from what we read) | Overlap with M13 |
|---|---|---|---|---|---|
| 1 | Chu et al., Neural Turtle Graphics | ICCV 2019 | full | Autoregressive graph generation from a queue of active nodes; an encoder of incoming paths, a decoder of outgoing nodes; metrics: perceptual, urban-planning (density, connectivity, reach, convenience), diversity (road overlap with real). No terrain. | The backbone family (our GraphTurtle, M5–M12). Queue-based global growth is already in NTG. |
| 2 | Fang et al., topography-aware generation | IJGIS 2022 | **A** | GAN-based generator with a "geo-extractor" and "geo-merging bypath" combining geometric and topographic inputs; code and data on Figshare. | Terrain-conditioned learned generation (raster). Prior art for terrain conditioning. |
| 3 | Gu et al., conditional diffusion | IJGI 2024 | **A** | Image diffusion conditioned on four context factors (listings: land use, elevation, slope, intersections), converted to graphs by post-processing; reports intersections as most influential and terrain inputs not always helpful. | Terrain-conditioned raster generation; no constrained sampling reported in the abstract. |
| 4 | Zhou et al., transformer graph generation for vectorized road design | CAVW 2024 | **A** | Elevation, water and vegetation maps → density map (attention image translation) → transformer converting the density map into a vector road graph; targets non-urban roads. | **Closest prior art**: terrain-conditioned *vector* graph generation. Whether it uses constrained decoding or search is **unknown** (no full text). |
| 5 | Birsak et al., large-scale autoregressive street networks | arXiv 2022 | full | VQ-VAE index field plus a transformer prior, decoded to a graph; conditioned on land/water, density and higher-priority streets; sliding-window generation for large areas; states explicitly that terrain is not considered. | Large-area sliding windows (our M11/M12 windows); no terrain; no constrained search. |
| 6 | Anagnostidis et al., Mastering Spatial Graph Prediction of Road Networks | ICCV 2023 | full | Road-graph *extraction* from aerial images as an MDP; a policy and value network with tree search (MuZero-style) maximizing graph-level rewards. | **Prior art for search-based neural road-graph construction** (reward-driven tree search). Different task (prediction from imagery), no terrain feasibility. |
| 7 | Galin et al., Procedural Generation of Roads | CGF 2010 | full | Single road between two points; weighted anisotropic shortest path (A*) on a grid with k-neighbourhood segment masks (gcd = 1); the cost is a sum of transfer functions of slope, curvature, water and vegetation; a transfer function returns infinity above a threshold (hard constraint); bridge and tunnel costs from height, slope and water depth. Evaluated by timings and path cost, not realism against real networks. | Terrain-routing baseline family; hard plus soft cost decomposition. |
| 8 | Galin et al., Authoring Hierarchical Road Networks | CGF 2011 | listing | Extends the 2010 method to hierarchical networks. | Procedural network baseline. |
| 9 | Emilien et al., villages on arbitrary terrains | TVC 2012 | full | Interleaved settlement seeding and road creation; anisotropic least-cost roads with re-use of existing roads (a discounted cost on existing segments); anisotropic land conquest for parcels; slope-adapted buildings; validated against real village layouts. | Roads and buildings adapted to slope; road re-use is a connectivity mechanism. |
| 10 | Lew et al., SMC steering of LLMs | ICML-W 2023 | full | Constrained generation as posterior inference: SMC with the LM as proposal and constraint potentials, resampling by weight. | **Core inference prior art.** |
| 11 | Zhao et al., twisted SMC | ICML 2024 | full | SMC with twist functions estimating future potential; learned twists via a contrastive objective; bidirectional log-partition bounds. | Prior art for future-aware weighting (our look-ahead twist). |
| 12 | Loula et al., syntactic and semantic control via SMC | ICLR 2025 | full | SMC with inference-time constraints and potentials across code, SQL and molecule domains. | SMC control in structured domains. |
| 13 | Lu et al., NeuroLogic A*esque decoding | NAACL 2022 | full | Beam search with A*-like look-ahead estimates of future constraint satisfaction. | Prior art for look-ahead in constrained decoding. |
| 14 | Lin, Chen & Liu, Lookahead strategies for SMC | Stat. Sci. 2013 | listing | Look-ahead weighting, sampling and resampling in SMC. | The statistical foundation for Monte Carlo look-ahead twists. |
| 15 | Naesseth et al., Nested SMC | ICML 2015 | listing | Proper weighting: approximate proposals remain valid with weights equal to estimated normalisers. | Justifies our K-candidate proposal with weight = mean candidate potential. |
| 16 | Sharma et al., PRODIGY | ICML 2024 | listing | Projection of diffusion samples onto constraint sets for graph generation. | Constrained graph generation (diffusion, not autoregressive). |
| 17 | ConStruct (Madeira et al.) | 2024 | listing | Structurally constrained discrete diffusion (planarity, acyclicity). | Constrained graph generation. |
| 18 | GraphARM (Kong et al.) | ICML 2023 | listing | Autoregressive diffusion for graphs; constrained experiments via degree checks. | Rejection-style constraints in autoregressive graph models. |
| 19 | Neuro-symbolic constrained graph generation (NSGGM / NSPSG) | 2026 | **A** | Solver-based projection for hard constraints (molecules). | Hard-constraint satisfaction by construction. |
| 20 | Boeing, street network models and indicators for every urban area | Geogr. Anal. 2021 | listing | Global OSM networks with node elevations (SRTM / ASTER) and edge grades; indicators include circuity and hilliness. | Grade-based metrics; an empirical reference for grade statistics. |
| 21 | Plasencia-Lozano, bridge frequency vs river width | EPB 2018 | **A** | Reported power-law decline of bridge frequency with river width; derives inter-bridge distance. | Supports crossing-rate-by-width and crossing-spacing metrics. |
| 22 | Vertical alignment / earthwork optimisation (Aruga et al. 2005; Ghanizadeh & Heidarabadizadeh 2020) | various | **A** | Earthwork as cut/fill between road profile and terrain; objective choice matters. | Grounds the earthwork *proxy* (not an engineering estimate). |
| 23 | AASHTO "Green Book" | standard | not read | Terrain classes (level / rolling / mountainous); maximum grades by class and design speed. | Background for class-specific grade guidance; not used numerically. |
| 24 | Bloem-Reddy & Orbanz, random-walk network models and SMC | JRSS-B 2018 | listing | SMC for network-growth models. | SMC over graph-growth histories (statistical modelling, not generation under constraints). |

## 3. What the literature establishes

1. **Terrain-conditioned learned road generation exists** in raster (Fang 2022; Gu 2024) and in vector graphs
   (Zhou 2024). Our M12 TerrainGraphTurtle is another instance. We make **no** claim to be first.
2. **Neural vector road-graph generation exists** (NTG 2019; Birsak 2022; Zhou 2024). No first claim.
3. **Search and look-ahead for road graphs exist:**
   * reward-driven tree search for road-graph prediction (Anagnostidis 2023);
   * A* least-cost terrain routing (Galin 2010, 2011; Emilien 2012);
   * look-ahead decoding for sequences (Lu 2022).

   No claim to be the first look-ahead method for road graphs.
4. **SMC for constrained autoregressive generation is established** for language and code (Lew 2023; Zhao 2024;
   Loula 2025), with look-ahead SMC theory from Lin et al. 2013 and proper weighting for nested or approximate
   proposals from Naesseth et al. 2015.
5. **Procedural terrain routing uses hard thresholds (infinite cost) plus soft costs** for slope, curvature,
   water and structures (Galin 2010). It is evaluated visually or by path cost, not against real network
   statistics.
6. **Quantitative evaluation of terrain response against real cities** was not found in any generator we read in
   full. NTG and Birsak evaluate flat-domain statistics; Fang and Gu evaluate image or graph similarity; Galin and
   Emilien evaluate visually or against a few real layouts.

## 4. Claim-by-claim novelty matrix

Novelty status is relative to this search only (recall limited: no library database; three key papers
abstract-only).

| Candidate claim | Closest prior art | Overlap | Status for M13 |
|---|---|---|---|
| Terrain-conditioned neural road generator | Fang 2022; Gu 2024; Zhou 2024 | full | **Not claimed.** |
| Neural vector road-graph generator | NTG; Birsak; Zhou | full | **Not claimed.** |
| Look-ahead or tree search for road graphs | Anagnostidis 2023; Galin 2010; Lu 2022 | full for the concept | **Not claimed.** |
| SMC / constrained probabilistic inference for autoregressive generation | Lew 2023; Zhao 2024; Loula 2025 | the inference framework is established | **Not claimed as a method.** We *apply* it. |
| Applying properly weighted constrained SMC, with Monte Carlo look-ahead twists, to terrain-feasible road-**graph** generation, where the potentials are real-data-calibrated terrain costs and geometric post-processing is inside the transition | none found; Zhou 2024 unread | partial (each ingredient exists) | **Candidate contribution**, conditional on reading Zhou 2024 in full and a library search. Phrase as "we apply and evaluate", not "we introduce SMC". |
| State-consistent commitment: the decoder is conditioned on the post-processed accepted edge | (engineering correctness; implicit in any exact sampler) | — | Not a contribution; a correctness requirement, tested. |
| Leakage-controlled, terrain-stratified evaluation (low-pass and street-erased DEM controls, slope-response curves, density-matched comparison) | none found among works read in full | low | **Candidate contribution** (benchmark and protocol). |
| Network-level cost or connectivity for bridge placement in a *learned* generator | Galin 2010 (procedural bridge costs); Plasencia-Lozano 2018 (empirical frequency vs width) | partial | Secondary, exploratory; claim only with data support. |
| Tile-free global queue | NTG (global queue); Birsak (sliding windows) | full | **Not claimed**; engineering, evaluated. |

## 5. Consequences for the M13 design

1. **Formulate inference as SMC** with a clear target, rather than as heuristic replanning, following Lew 2023 and
   Loula 2025.
   * Report exact versus approximate status explicitly.
   * Valid: the K-candidate proposal with weight equal to the mean candidate potential (proper weighting, Naesseth
     2015).
   * Approximate: Monte Carlo look-ahead twists (Lin 2013, Zhao 2024), with estimated rather than learned twists.
2. **Hard versus soft follows Galin 2010** (infinite cost above physical thresholds), but steep grades are soft:
   streets are built on steep land with switchbacks and earthwork. The soft costs are calibrated on real training
   streets, not set by hand.
3. **Evaluation must go beyond the cited works:**
   * terrain-response curves;
   * connectivity with unreachable pairs counted;
   * crossing rate by river width (Plasencia-Lozano 2018);
   * an earthwork proxy (alignment literature);
   * density-matched comparisons.
4. **External baselines** (§6): NTG is the backbone family; Galin-style anisotropic routing is implementable
   from the full text; the others lack accessible code or weights for our data, or full text.

## 6. External baseline availability

| Method | Code / weights | Data compatibility | Decision |
|---|---|---|---|
| NTG (Chu 2019) | project page only; no official training code found | NTG's own data not used | Our GraphTurtle (N0 / TN) is an NTG-family reimplementation; **not** claimed to reproduce NTG exactly. |
| Birsak 2022 | "code available" per aggregators; repository not located in this search | needs VQ-VAE plus transformer training on index fields; no terrain input | Not reproduced in M13 (compute and time); documented. |
| Zhou 2024 | none found | — | Not reproducible (no code, no full text). |
| Fang 2022 | Figshare data and code record | raster GAN; needs its preprocessing | Investigated only; not reproduced (no full text to confirm the architecture; raster-to-graph conversion would add a confound). |
| Gu 2024 | not found | raster diffusion | Not reproduced. |
| Galin 2010 / 2011 | none (procedural) | implementable from the full text | The M12 procedural baseline **P** (Poisson junctions, Gabriel graph, slope- and water-costed Dijkstra paths on an 8 m grid) follows Galin's cost decomposition **without** the curvature term or the k-masks; it is not claimed to reproduce Galin et al. A curvature-aware reimplementation is listed as future work. |

## 7. Open items before any publication claim

1. Read Zhou et al. 2024 in full (KAIST library access).
2. Read Gu et al. 2024 and Fang et al. 2022 in full.
3. Repeat Q1–Q21 in Scopus and Web of Science; add forward citations of Galin 2010, NTG and Zhou 2024.
4. Confirm AASHTO maximum grades if they are used numerically (they are not, in M13).
