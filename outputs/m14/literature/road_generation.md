# M14 literature check: road / street network generation (terrain-aware and learned)

Compiled 2026-11-17. Sources were fetched with WebSearch/WebFetch/curl. Bibliographic fields were checked against Crossref or OpenAlex wherever possible.

**Access-level legend**
- **FULL**: I read the full text myself (PDF text extracted and read or grepped section by section).
- **ABSTRACT**: I read only the abstract and metadata, sometimes plus third-party descriptions.
- **NONE**: I could not get the text.

Anything not confirmed from a primary source is marked **not verified**.

## Summary table

| # | Work | Access | Output repr. | Learned? | Terrain inputs | Search / constrained decoding | Diversity eval | Unseen-city eval | Code |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Zhou et al. 2024 (CAVW) | ABSTRACT | vector road graph (via density map) | yes (image translation + transformer) | elevation, water, vegetation maps | not verified | not verified | not verified | none found |
| 2 | Gu et al. 2024 (IJGI) | FULL | raster 128×128, then thinned and vectorized | yes (conditional DDPM) | elevation, slope, land use (incl. water) + GT intersections | none | not evaluated | **no** (spatially disjoint tiles in the same city, per-city models) | figshare (data + code) |
| 3 | Fang et al. 2022 (IJGIS) | FULL | raster inpainting (256×256 px) | yes (GAN inpainting) | elevation, slope, aspect, 4 hillshades (no water) | none | not evaluated | **yes** (train Florence+Perugia, test Rome+Siena) | figshare (data + code) |
| 4 | Chu et al. 2019 NTG (ICCV) | FULL | vector planar graph | yes (GRU enc/dec, autoregressive) | none | local hard checks only (merge radius, max degree/density, min angle); product with CNN likelihood for parsing | yes (Chamfer-like novelty metric) | generation: no; aerial parsing: yes (held-out city) | official repo has no code |
| 5 | Birsak et al. 2022 (arXiv) | FULL | VQ index field → distance field → traced graph | yes (VQ-VAE + transformer) | land/water map only (terrain explicitly excluded) | none (sliding-window AR sampling) | qualitative only | qualitative style transfer (Istanbul) | none found |
| 6a | Hartmann et al. 2017 StreetGAN (WSCG) | ABSTRACT | raster patches → vector | yes (GAN) | none (per abstract) | none | not verified | not verified | not verified |
| 6b | Owaki & Machida 2020 RoadNetGAN (ICONIP) | ABSTRACT | planar graph via random walks | yes (conditional NetGAN) | none (per abstract) | not verified | not verified | not verified | not verified |
| 6c | Yang, Zhu, Feng 2023/2025 (arXiv 2305.08186) | FULL (arXiv v2) | raster → graph | yes (AE + cGAN) | elevation (+ population, land use) | none | dropout at test, qualitative | no (70/30 split of 3 Australian cities) | not found |
| 6d | Wang et al. 2025 GraphWalker (IJCAI) | FULL (skimmed) | graph from walks | yes (latent diffusion) | none (trajectory-conditioned) | none | — | yes (train on 1 city, test on 3) | GitHub |
| 6e | Li et al. 2024/2026 RoBus (dataset) | ABSTRACT | images/graphs | baselines | not verified | not verified | — | — | GitHub |
| 6f | DRoLaS 2025 (ICMR), CaRoLS 2025 (C&G) | metadata only | not verified | diffusion (DRoLaS) | not verified | not verified | not verified | not verified | not verified |
| 6g | Li et al. 2026 RoadWeaver (arXiv) | FULL (skimmed) | vector road graph → HD map | yes (VQ-VAE + masked transformer) + procedural | none | post-hoc A* reconnection of dangling endpoints + LCC filtering | — | — | not checked |
| 7a | Parish & Müller 2001 (SIGGRAPH) | ABSTRACT + secondary | vector graph | procedural (L-system) | land/water, population density (+ elevation per secondary sources) | rule-based local constraints | n/a | n/a | n/a |
| 7b | Chen et al. 2008 (TOG) | FULL | vector graph from tensor field | procedural / interactive | water, park/forest, height map, population density | look-ahead to find intersections; bridge rules | via noise fields (user-driven) | n/a | project page reportedly has code (not verified) |
| 7c | Galin et al. 2010 (CGF) | FULL (abstract + intro + method skim) | single 3D road between two points | procedural optimization | slope, rivers, lakes, forests, curvature | anisotropic shortest path (global optimum per road) | n/a | n/a | n/a |
| 7d | Galin et al. 2011 (CGF) | FULL (abstract + intro + method skim) | hierarchical inter-city road network | procedural optimization | slope, water, vegetation (cost) | complete graph → non-Euclidean proximity graph → path merging | parameter-controlled density/pattern | n/a | n/a |
| 8 | Anagnostidis et al. 2023 (ICCV) | FULL (arXiv v1, 2022) | vector spatial graph | yes (AR transformer + MuZero) | none (satellite image input) | **yes: MCTS / MuZero look-ahead at train and test** | — | standard splits (not unseen-city) | none found |
| + | Darvariu et al. 2023 (Proc. R. Soc. A) | ABSTRACT | edge additions to an existing graph | RL / planning | none | MCTS (UCT variants) | — | — | not checked |

---

## 1. Zhou et al. 2024: transformer-based graph generation for vectorized road design

**Citation (verified via Crossref and OpenAlex):** Peichi Zhou, Chen Li, Jian Zhang, Changbo Wang, Hong Qin, Long Liu. "A novel transformer-based graph generation model for vectorized road design." *Computer Animation and Virtual Worlds* 35(3), e2267, May 2024. DOI: 10.1002/cav.2267.
- The exact title starts with "A novel ..."; the brief's short title omits that.
- ECNU's Pure portal lists the keywords "generative model; landscape modeling; non-urban road network".

**Access: ABSTRACT only.**
- The Wiley PDF and full-text pages returned 403 or a Cloudflare challenge.
- Semantic Scholar marks it CLOSED. OpenAlex says "bronze OA", but the pdfdirect link was blocked.
- I found no arXiv version, author copy, or code.

**From the abstract (Crossref/OpenAlex):**
- It is a framework "for automatic and interactive road production driven by geographical maps (including elevation, water, vegetation maps)".
- The authors "construct a dataset that includes road graphs, density map and their corresponding geographical maps".
- A "density map generation network based on image translation model with an attention mechanism" predicts a road density map.
- "we employ the transformer architecture to evolve density maps to road graphs".
- The outcome is "a vectorized road network".

**Answers:**
- **(a) What it generates:** a vector road graph, produced in two stages: geographic maps → density map → transformer → graph.
- **(b) Distribution modelled:** a learned model. It is **not verified** whether the transformer is autoregressive or one-shot, or whether sampling is stochastic.
- **(c) Terrain:** elevation, water and vegetation maps, per the abstract. The keyword "non-urban road network" suggests the target is rural or landscape roads, but that is **not verified**.
- **(d) Global planning, search or constrained decoding:** **not verified**. The abstract mentions none.
- **(e) Diversity:** **not verified**.
- **(f) Unseen geography:** **not verified**.
- **(g) Code:** none found. GitHub and web searches found nothing.

**Relevance:** This is the closest prior art in *problem setup*: learned, terrain-conditioned, vector road graph output. Its full text must be obtained (for example through KAIST library access) before claiming novelty on any specific point.

## 2. Gu et al. 2024: urban road networks with conditional diffusion models

**Citation (verified via Crossref, OpenAlex and the PDF):** Xiaoyan Gu, Mengmeng Zhang, Jinxin Lyu, Quansheng Ge. "Generating Urban Road Networks with Conditional Diffusion Models." *ISPRS International Journal of Geo-Information* 13(6), 203, 2024. DOI: 10.3390/ijgi13060203. Open access, CC BY.

**Access: FULL.**
- I read the 19-page PDF from MDPI's resource CDN (mdpi-res.com, official MDPI host).
- Some pages carried an overlapped review layer, which garbled part of the extraction; all sections were still readable.

**Answers:**
- **(a) What it generates:** raster road images (1×128×128 px for 1 km × 1 km, 7.81 m/px).
  - These are converted to vectors by post-processing (§3.3): binarize at 127, thin with Zhan's algorithm, then vectorize.
  - "We can reconstruct the road graphs based on these road vectors."
- **(b) Distribution modelled:** a learned conditional DDPM with a U-Net, 1000 steps, following Saharia et al. (§3.2).
- **(c) Conditions (§3.1, Table 1):** four "context factors":
  - land use: 9 ESRI Sentinel-2 classes, including water;
  - elevation: SRTM 30 m;
  - slope: derived from elevation;
  - **road intersections**: "Nodes with a degree greater than 1 were extracted from the road networks as intersections" (§3.1).
  - Intersections therefore come from the ground-truth OSM network, which is a strong structural hint. The ablation shows that removing intersections hurts most (§4.4: "Model 4 … implying that intersections are the most important guide").
  - Elevation and slope are min-max normalized *per city* (Eq. 1).
- **(d) Global planning, search or constrained decoding:** none. Connectivity is not enforced; topology is only measured via DAD.
- **(e) Diversity:** not evaluated. Fidelity and diversity are named as goals in §1, but no multi-sample diversity metric is reported.
- **(f) Unseen geography:** **No.**
  - The setup covers five US cities (Chicago, LA, New York, Phoenix, Washington). "For each city, we trained diffusion models" (§4.2), so each city has its own model.
  - Each city is split 80/20 by spatially non-overlapping squares (§4.1): "our testing could be performed completely on the untouched spatial area".
  - No cross-city transfer is tested.
- **Metrics (§3.4):**
  - image-based: FID, F1, IoU;
  - network-based: DAD (difference in average degree) and DARL (difference in average road length).
  - It also includes a regression analysis of obstacles (water, mountain, vegetation area vs. road length; §4.3).
- **Limitations (§5):** "we generate a local road network within a grid with the size of 1 km × 1 km each time" and "the resolution of the generated image is limited to 128 × 128 pixels".
- **(g) Code:**
  - Data Availability: "The codes and data that support this work are available at … https://doi.org/10.6084/m9.figshare.24618942.v1".
  - Figshare API: item "diffusion-road-dev", 367 MB .7z, CC BY 4.0, "training and testing datasets of five cities in U.S. and the Python codes".

## 3. Fang et al. 2022: a topography-aware approach to automatic urban road network generation

**Citation (verified via Crossref, OpenAlex and the PDF):** Zhou Fang, Jiaxin Qi, Lubin Fan, Jianqiang Huang, Ying Jin, Tianren Yang. "A topography-aware approach to the automatic generation of urban road networks." *International Journal of Geographical Information Science* 36(10), 2035–2059, 2022. DOI: 10.1080/13658816.2022.2072849.

**Access: FULL.**
- I read the 26-page published PDF hosted on co-author Lubin Fan's personal webpage (lbfan.oss-cn-hangzhou.aliyuncs.com).
- The T&F and NTU repository copies were not accessible; the NTU DR record has only a licence bundle.
- Figshare item 19947510 is the *supplementary material* PDF (metadata only; the download was blocked).

**Answers:**
- **(a) What it generates:** raster *inpainting*. It predicts "missing segments in a street configuration within a pre-defined region" (§1).
  - Samples are 512 m × 512 m at 256×256 px, with rectangular masks of 48–64 px (§4.1).
  - Outputs are converted to graphs only for evaluation (§5.3.2).
- **(b) Distribution modelled:** a learned GAN inpainting model built on DeepFill v2 (Yu et al. 2019), with SN-PatchGAN, three stages, and L1 + hinge adversarial loss (§4).
- **(c) Terrain (§3.1.1):** seven channels: elevation, slope, aspect, and four hillshades (0/90/180/270°).
  - A "geo-extractor" (GE) learns a 0–1 gate plus per-channel attention (§4.2).
  - Other inputs: the surrounding street context and **ground-truth junction heat maps** ("junction guidance").
  - Water is **not** included: "polyline-based hydrologic features such as waterways … are not included" (§7).
- **(d) Global planning, search or constrained decoding:** none. The method is a single feed-forward inpainting pass.
- **(e) Diversity:** not evaluated. Because junctions and topography are supplied, "the predicted street network is expected to be identical to the original one" (§5.3.2).
- **(f) Unseen geography:** **Yes, at the city level.** "samples extracted from Florence and Perugia are used to train the models, while samples extracted from Rome and Siena are used for validation and testing" (§5.1).
- **Metrics:**
  - MAE and MSE (pixel);
  - absolute percentage errors of length-weighted space-syntax "metric choice" and "metric integration" (APEMC, APEMI) after image-to-graph conversion (§5.3.2).
  - Reported result: the flat-vs-hilly gap narrows from 17.02% to 3.56% (APEMC).
- **(g) Code:**
  - "The data and codes … are available at … https://doi.org/10.6084/m9.figshare.19606960.v1".
  - Figshare API: "Final_Code.zip", ~947 MB, CC BY 4.0, published 2022-04-16. I did not download it (figshare's downloader returned 403).
- **Precursors:**
  - DeepStreet (Fang, Yang, Jin, arXiv 2010.04365; read in part), with Barcelona as its case study. It used elevation as a raster channel and a random 80/20 split.
  - "Incorporating planning intelligence into deep learning" (Fang, Jin, Yang, arXiv 2010.04536; abstract read).

## 4. Chu et al. 2019: Neural Turtle Graphics (NTG)

**Citation (verified via Crossref):** Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, Sanja Fidler. "Neural Turtle Graphics for Modeling City Road Layouts." *ICCV 2019* (Oral), pp. 4521–4529. DOI: 10.1109/ICCV.2019.00462. arXiv: 1910.02055.

**Access: FULL** (arXiv v1).

**Answers:**
- **(a) What it generates:** a vector planar graph. Nodes are control points; edges are road segments (§3.1).
- **(b) Distribution modelled:** a learned sequential generative model (§3.2–3.5).
  - A queue of unvisited nodes is kept.
  - The encoder is a bidirectional GRU over K sampled acyclic incoming paths (random walks of length ≤ L), summed to be order-invariant.
  - The decoder is a GRU that emits outgoing nodes one by one, plus "a binary variable which indicates whether another node should be generated" (a stop decision) and optionally the edge type (major/minor).
  - For city generation, Δx and Δy are **discrete** classes at 1 m resolution in [−100, 100] m, predicted independently (§3.4). This is not a mixture density.
  - It is conditioned on a city-ID embedding.
- **Inference-time rules (§3.5):**
  - a new node within 5 m of an existing node is merged, creating loops;
  - "We also find the maximum node degree, maximum node density, and minimum angle between two edges in the training set, and ensure our generation does not exceed these limits."
  - These are local hard feasibility checks only, with no look-ahead.
- **Aerial parsing (§3.3):** "At each decoding step, we multiply the likelihood from CNN with the prior produced by NTG, and sample output nodes from this joint distribution."
  - This is an inference-time product of the learned prior and an external potential, a precedent for potential-guided decoding.
- **(c) Terrain:** none. The RoadNet dataset is OSM data from 17 cities, ~10 km² each.
- **(d) Global planning or search:** none beyond the local checks and the CNN×prior product.
- **(e) Diversity:** yes. The "Diversity metric" is the percentage of road outside a 10 m vicinity versus every GT map, reporting the lowest (§4.1.1).
  - Other metrics: domain-adapted FID and "Urban Planning" features (node density, connectivity as node degree, reach, transportation convenience = Euclidean / shortest-path ratio).
  - Baselines: GraphRNN-2D, PGGAN, CityEngine.
- **(f) Unseen geography:**
  - Generation: no. It is city-conditioned style generation.
  - Aerial parsing on SpaceNet: yes. "4-fold evaluation holding out each city per fold" (Table 4).
- **(g) Code:** effectively **not released**.
  - The official repo github.com/nv-tlabs/NTG contains only README.md and docs/ (checked via the GitHub API). The README's Paper/Video links are empty.
  - Project page: research.nvidia.com/labs/toronto-ai/NTG/ (supplementary PDF only).
  - An unofficial reimplementation exists (github.com/Cewein/Neural-Turtle-Graphics); I did not inspect it.

## 5. Birsak et al. 2022: large-scale auto-regressive modeling of street networks

**Citation (verified via the arXiv API and OpenAlex):** Michael Birsak, Tom Kelly, Wamiq Para, Peter Wonka (KAUST). "Large-Scale Auto-Regressive Modeling Of Street Networks." arXiv:2209.00281, 2022.
- The PDF uses an ACM template with placeholder venue text ("Conference acronym 'XX … Woodstock, NY").
- **No peer-reviewed venue found (not verified).**

**Access: FULL** (arXiv v1).

**Answers:**
- **(a) What it generates:** a VQ-VAE index field, sampled by a transformer "IndexModel" in a sliding window.
  - The index field is decoded to a distance field, then turned into a graph by thresholding, thinning, RoadTracer-style tracing, and "filtering small disconnected parts" (§3.5).
  - It covers areas of 400 km² or more.
- **(b) Distribution modelled:** a learned conditional autoregressive prior over codebook indices (§3).
- **(c) Terrain:**
  - Conditions: land/water map, street density map, and a distance field of existing higher-priority (P1) streets.
  - "we do not consider terrain as conditioning information in the current iteration of the paper. The main reason was the difficulty of obtaining training data." (§5)
- **(d) Global planning or search:** none. Coherence comes from sliding-window conditioning. Disconnected parts are deleted post hoc.
- **(e) Diversity:** qualitative only. "We use our model repeatedly using the same set of condition maps to show the diversity" (§4, Fig. 6).
- **(f) Unseen geography:**
  - Training uses the 50 largest US cities.
  - A qualitative "style transfer" regenerates Istanbul from its density map, high-priority streets and coastline (Fig. 7).
  - Results use ground-truth density maps.
  - There is no quantitative held-out-city benchmark in the main text.
- **(g) Code:** none found (GitHub search).

## 6. Other learned road-network generators

### 6a. StreetGAN
**Citation (OpenAlex and search results):** Stefan Hartmann, Michael Weinmann, Raoul Wessel, Reinhard Klein. "StreetGAN: Towards Road Network Synthesis with Generative Adversarial Networks." WSCG 2017. Repository: hdl.handle.net/11025/29554.
- The exact proceedings volume and pages are **not verified**.

**Access: ABSTRACT** (via search summaries). The repository host otik.zcu.cz did not resolve via DNS from my environment.

**Summary:**
- It rasterizes road-network patches into binary images, trains a GAN on them, then extracts road vectors.
- Gu et al. (§2.2) and Fang et al. (§2) confirm this description.
- No terrain input (per Fang et al. §2: "Most studies do not input any topographic information … (Hartmann et al. 2017 …)").
- Diversity, unseen-city evaluation and code: **not verified**.

### 6b. RoadNetGAN
**Citation (verified via Crossref):** Takashi Owaki, Takashi Machida. "RoadNetGAN: Generating Road Networks in Planar Graph Representation." ICONIP 2020, *Communications in Computer and Information Science*, pp. 535–543. DOI: 10.1007/978-3-030-63820-7_61.

**Access: ABSTRACT** (search summaries). The rest is **not verified**.

**Summary:**
- It extends NetGAN with a conditional generator and discriminator, generating random walks with displacement attributes that are assembled into a planar graph.
- No terrain input per the abstract.

### 6c. Yang, Zhu, Feng: city street layout generation via conditional adversarial learning
**Citation:** Lehao Yang, Cui Zhu, Tian Feng. arXiv:2305.08186 (v2). The journal reference given on arXiv is *Bulletin of Science and Technology* 2025, 41(10):83–90+115 (not independently verified).

**Access: FULL** (the v2 PDF is in Chinese with an English abstract).

**Summary:**
- An autoencoder fuses elevation, population density and land use; a cGAN (pix2pix-style) generates 512×512 px (5 m) road images; a graph-extraction module turns them into graphs.
- Data come from Sydney, Melbourne and Brisbane: 2,586 pairs split 70/30 at random, so there is **no unseen-city test**.
- Metrics: connectivity index, intersection density, total length, metric reach, and transport convenience.
- Diversity comes from dropout at test time (rate 0.95).
- The authors note failures to connect across bridges (Fig. 5).
- Code: not found.

### 6d. GraphWalker (Wang et al., IJCAI 2025)
**Citation:** Jinming Wang, Hongkai Wen, Geyong Min, Man Luo. "High-Fidelity Road Network Generation with Latent Diffusion Models." IJCAI-25 (proceedings paper 702).

**Access: FULL** (skimmed).

**Summary:**
- It generates road graphs from GPS **trajectories**: a latent diffusion transformer denoises walks, which are assembled into a graph.
- Not terrain-conditioned. It is closer to map inference than to design.
- It trains on one city and tests on three (Tokyo and others) for cross-city generalization.
- Code: github.com/JinmingWang/RNGen_GraphWalker.

### 6e. RoBus
**Citation:**
- arXiv:2407.07835 (2024): Tao Li, Ruihang Li, Huangnan Zheng, Shanding Ye, Shijian Li, Zhijie Pan.
- Journal version (verified via Crossref): *Computers* 15(8), 488, 2026, DOI 10.3390/computers15080488, "RoBus: A Multimodal Dataset for Controllable Road Networks and Building Layout Generation". OpenAlex lists a slightly different author list for this version.

**Access: ABSTRACT.**

**Summary:**
- A dataset of 72,400 paired samples (images, graphs, text) covering ~80,000 km², with metrics and baselines conditioned on road orientation and building density.
- Code and data: github.com/tourlics/RoBus_Dataset.
- Terrain conditioning: **not verified**.

### 6f. DRoLaS and CaRoLS (Tian Feng group, 2025)
- **DRoLaS (verified via Crossref):** Shenao Dong, Weitao Li, Bo Li, Long Li, Junao Shen, Tian Feng. "DRoLaS: Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts." *ICMR 2025*, pp. 237–245. DOI: 10.1145/3731715.3733316.
- **CaRoLS (researchr listing only):** Tian Feng, Long Li, Weitao Li, Bo Li, Junao Shen. "CaRoLS: Condition-adaptive multi-level road layout synthesis." *Computers & Graphics* 133:104451, 2025. The DOI is **not verified**.
- **Access: NONE** (metadata only). Their conditions, representation, terrain use, connectivity handling and evaluation are all **not verified**.
- These are likely the most recent learned conditional road-layout synthesizers. They are worth fetching through library access.

### 6g. RoadWeaver (Li et al., arXiv 2608.11580, Aug 2026)
**Access: FULL** (skimmed §III–IV).

**Summary:**
- It generates a global road skeleton with a VQ-VAE + conditional masked transformer, conditioned on style code, density, gridness, radialness, organicness and bearing entropy.
- It then runs procedural tensor-field growth to add roads and builds a lane-level HD map.
- The relevant mechanism is a **post-hoc** connectivity repair (§III-C): "dangling endpoints are reconnected to nearby roads through A* search over a cost map derived from the decoded road field … Residual disconnected fragments are subsequently removed through largest-connected-component filtering."
- It reports LCC ratio, reachability (99.8%) and dead-end ratio (10.7%).
- No terrain input. Repair happens after generation and is not built into the sampling.

### Other items found (not road *generation* from terrain; listed for completeness)
- **Barthelemy & Boeing 2025:** "Universal Model of Urban Street Networks", *Phys. Rev. Lett.* 135, 137401 (arXiv 2509.21931; abstract read). A non-learned spanning-tree-plus-edge-addition generative model.
- **Morys-Magiera et al.:** "Elite Lanes", GECCO 2026 poster (arXiv 2603.20964; abstract read). An evolutionary algorithm with connectivity and redundancy constraints for small road networks.
- **Zhou et al.:** arXiv 2409.17049 (urban morphology diffusion); title page only.

## 7. Procedural methods

### 7a. Parish & Müller 2001
**Citation (verified via Crossref):** Yoav I. H. Parish, Pascal Müller. "Procedural Modeling of Cities." *Proc. SIGGRAPH 2001*, pp. 301–308. DOI: 10.1145/383259.383292.

**Access: ABSTRACT** (SIGGRAPH History Archive) plus descriptions in the papers I read in full: Fang 2022 §2, Galin 2011 §1.1, Chen 2008, Birsak 2022. The ACM PDF was blocked.

**One-sentence summary:** An extended L-system grows highways and streets from image maps (the abstract names "land-water boundaries and population density"), then subdivides lots and generates buildings, which became the basis of CityEngine.

**Answers:**
- **Terrain:** water boundaries per the abstract. Fang 2022 §2 says users choose among "basic, New York, Paris, and San Francisco" rule sets "jointly considering topographic constraints", and calls the San Francisco rule topography-aware. Direct elevation handling is **not verified from the primary text**.
- **Connectivity:** handled by local rule-based constraints. The commonly cited intersection snapping is **not verified** from the primary text.
- **Diversity:** via rule parameters and stochasticity; not evaluated.

### 7b. Chen et al. 2008
**Citation (verified via Crossref and the PDF):** Guoning Chen, Gregory Esch, Peter Wonka, Pascal Müller, Eugene Zhang. "Interactive Procedural Street Modeling." *ACM Transactions on Graphics* 27(3), Article 103, 10 pp., Aug 2008 (SIGGRAPH 2008). DOI: 10.1145/1360612.1360702.
- The brief's "TOG/SIGGRAPH" is correct. Note that a 2007 SIGGRAPH *sketch* with the same title exists (Esch et al., DOI 10.1145/1278780.1278822).

**Access: FULL** (authors' PDF, web.engr.oregonstate.edu/~zhange/images/street_sig08.pdf).

**One-sentence summary:** The user designs a tensor field, seeded from water boundaries, height-field gradients, brush strokes and noise. Major and minor roads are traced as hyperstreamlines of its eigenvector fields, producing an editable street graph.

**Answers:**
- **Terrain:** inputs are a water map, a park/forest map, a height map and a population density map (§3). The heightfield constrains road direction via its gradient (§4.2).
- **Connectivity:**
  - "tracing major and minor hyperstreamlines independently often leads to a disconnected street network". The fix is interleaved tracing.
  - "we improve connectivity by continuing the tracing for a distance d_lookahead to search an intersection with other hyperstreamline" (§5).
  - Bridges across narrow water are allowed.
- **Diversity:** via Perlin noise and rotation fields and user editing; not quantitatively evaluated.
- **Code:** a search summary says the project pages (sci.utah.edu/~chengu/street_sig08/) link to source code. I did not open them, so this is **not verified**.

### 7c. Galin et al. 2010
**Citation (verified via Crossref and the PDF):** Eric Galin, Adrien Peytavie, Nicolas Maréchal, Eric Guérin. "Procedural Generation of Roads." *Computer Graphics Forum* 29(2), 429–438, 2010 (Eurographics 2010). DOI: 10.1111/j.1467-8659.2009.01612.x.

**Access: FULL** (author PDF, perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf). I read the abstract and introduction; I grepped the method sections.

**One-sentence summary:** Computes a single road between a start and an end point as a weighted *anisotropic* shortest path on a grid. It uses k-neighbourhood masks, with a cost on slope, curvature, rivers, lakes and forests. Bridges and tunnels are costed consistently, and the path is fitted with clothoid splines and excavated into the terrain.

**Answers:**
- Terrain: yes, heavily.
- Connectivity: one path only, with no network.
- Diversity: via cost parameters.
- This is a global optimum per road (search), not a learned prior.

### 7d. Galin et al. 2011
**Citation (verified via Crossref and the PDF):** Eric Galin, Adrien Peytavie, Eric Guérin, Bedřich Beneš. "Authoring Hierarchical Road Networks." *Computer Graphics Forum* 30(7), 2021–2030, 2011 (Pacific Graphics 2011). DOI: 10.1111/j.1467-8659.2011.02055.x.

**Access: FULL** (author PDF, cs.purdue.edu/homes/bbenes/papers/Galin11CGF.pdf). I read the abstract and introduction; I grepped the method sections.

**One-sentence summary:** Given cities, towns and villages on a terrain, it builds highways, then primary roads, then secondary roads. Each level forms a complete graph of terrain-cost shortest paths (vegetation, water height, slope), which is pruned with an empty-region proximity-graph criterion under a **non-Euclidean** metric. Paths are then merged via a non-Euclidean Fréchet distance to create junctions and Steiner nodes.

**Answers:**
- Terrain: yes.
- Connectivity: by construction, since the network connects all input settlements.
- Diversity: user-controlled density and pattern through the proximity-graph parameters.
- Not learned. Inter-city and rural only, not intra-city streets.

## 8. Anagnostidis et al. 2023: Mastering Spatial Graph Prediction of Road Networks

**Citation (verified via Crossref):** Sotiris Anagnostidis, Aurelien Lucchi, Thomas Hofmann. "Mastering Spatial Graph Prediction of Road Networks." *ICCV 2023*. DOI: 10.1109/ICCV51070.2023.00498. arXiv: 2210.00828.
- Crossref gives pp. 5385–5395; a search snippet said 5408–5418. Use Crossref.
- The CVF open-access PDF exists, but I read the arXiv v1 (Oct 2022), which may differ from the camera-ready.

**Access: FULL** (arXiv v1).

**Answers:**
- **(a) What it does:** road-graph *extraction* from satellite images, not unconditional design.
  - Key points come from a skeletonized segmentation mask.
  - An autoregressive transformer edge decoder (pointer network) adds edges sequentially, giving a vector spatial graph.
- **(b) Distribution modelled:** a learned autoregressive policy, refined with RL.
- **(c) Terrain:** none. The input is a satellite image.
- **(d) Search:** **yes.**
  - "we rely on Monte Carlo Tree Search (MCTS) to simulate entire sequences of actions. We use … MuZero" (§3.2). It uses a learned dynamics network, value and reward heads, and UCB selection.
  - Rewards are increments of topological similarity: APLS and path-, junction- and sub-graph-based F1 against GT during training (Eq. 3).
  - The motivation is that "the effect of each new edge to the reward will be reflected mostly once its whole corresponding road segment has been generated", which is a credit-assignment and look-ahead argument.
  - The search refines "initial sampling probabilities via a tree search" at inference, and it "can be used to refine predictions from another given model" (§5).
- **(e) Diversity:** not applicable (single best extraction).
- **(f) Unseen geography:** SpaceNet and DeepGlobe "same train-test splits" as prior work, plus a synthetic CityEngine occlusion benchmark. There is no explicit held-out-city protocol (**not verified** beyond this).
- **(g) Code:** none found on GitHub. The authors "plan to release" the synthetic dataset.

## Adjacent: Darvariu et al. 2023, "Planning spatial networks with Monte Carlo tree search"

**Citation (verified via Crossref):** Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi. *Proc. R. Soc. A* 479(2269): 20220383, 2023. DOI: 10.1098/rspa.2022.0383. arXiv: 2106.06768.

**Access: ABSTRACT.**

**Summary:** Formulates goal-directed edge addition to an existing spatial graph (efficiency, resilience) as a deterministic MDP solved with UCT-style MCTS. Distance-limited "realizable connections" restrict the action space. Examples are Internet backbones and metro systems; there is no learned generative prior and no terrain.

---

## Closest prior art: inference-time network-aware or adaptive planning on a learned, terrain-constrained road-graph generator

Among the works I could read, **none combines all three of:**
1. a learned road-graph generator (autoregressive or otherwise) as the proposal distribution;
2. terrain feasibility constraints; and
3. inference-time planning that targets network-level properties (connectivity, dead ends, fragments) through look-ahead, SMC or adaptive compute.

The pieces exist separately:

- **Terrain-conditioned learned generators**: Gu 2024, Fang 2022, Yang et al. 2305.08186, and probably Zhou 2024.
  - All except Zhou are raster-first. They condition on terrain only as input channels and do no search.
  - Gu and Fang also feed ground-truth intersections or junctions as conditions.
  - Fang is the only one with a true unseen-city test (two cities).
  - Connectivity is measured afterwards (DAD, space-syntax APE, connectivity index), never optimized during sampling.
- **Learned vector-graph generators**: NTG and Birsak.
  - Neither uses terrain; Birsak explicitly omits it.
  - NTG applies only local hard checks at decode time (merge radius, max degree, density, min angle). Its aerial-parsing mode multiplies the learned prior by an external CNN likelihood at each step. That is a direct precedent for potential-reweighted decoding, but it is greedy and not network-aware.
  - Birsak deletes disconnected parts after generation.
- **Search for network-level objectives on a learned autoregressive graph decoder**: Anagnostidis et al. (ICCV 2023) is the closest *methodological* precedent.
  - It runs MuZero/MCTS look-ahead over edge additions, with topology-metric rewards (APLS, junction/path F1).
  - Its task is image-conditioned extraction against a ground truth, not terrain-constrained design.
  - Its rewards are learned from GT similarity, not intrinsic network properties.
  - It does not evaluate sample diversity or calibrated multi-sample generation.
  - Darvariu et al. (MCTS for spatial network improvement) is a non-generative planning analogue.
- **Post-hoc connectivity repair**: RoadWeaver (2026) does A* reconnection of dangling endpoints on a learned field, plus LCC filtering. Chen et al. 2008 (look-ahead tracing to find intersections) and Galin 2011 (connectivity by construction with terrain-cost paths) are the procedural counterparts.

**Caveat:** Zhou et al. 2024 (closed access) and DRoLaS/CaRoLS 2025 (closed access) could not be read. Before claiming that no prior work does inference-time network-aware planning on a learned terrain-conditioned road-graph generator, their full texts should be checked. Zhou in particular shares our problem setup: terrain maps → learned → vector road graph.

## Retrieval log (for reproducibility)
- **Gu 2024:** mdpi.com HTML returned 403. `https://mdpi-res.com/d_attachment/ijgi/ijgi-13-00203/article_deploy/ijgi-13-00203.pdf` gave the full PDF.
- **Fang 2022:**
  - T&F returned 403.
  - The NTU DR-NTU item has only a LICENSE bundle.
  - Figshare downloads returned 403; metadata came from the API.
  - Full PDF from co-author page: `https://lbfan.oss-cn-hangzhou.aliyuncs.com/personal_webpage/publications/2022-ijis-auto_road_network.pdf`.
  - An informahealthcare.com mirror was not used (blocked as unauthorized).
- **Zhou 2024:** Wiley pdfdirect and full pages returned Cloudflare or 403. Semantic Scholar says CLOSED. Only the abstract (Crossref/OpenAlex) and the ECNU Pure metadata were available.
- **arXiv PDFs read:** 1910.02055, 2209.00281, 2210.00828, 2305.08186, 2608.11580, 2010.04365 (partial), 2010.04536 (partial); IJCAI 2025 paper 702.
- **Galin 2010/2011:** HAL was blocked by Anubis; I used the author-hosted PDFs (LIRIS, Purdue). **Chen 2008:** author PDF (Oregon State).
- **StreetGAN:** the repository host otik.zcu.cz did not resolve via DNS. **Parish & Müller:** ACM PDF blocked; abstract from history.siggraph.org.
