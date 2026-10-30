# Literature Review (Milestone 2, first pass)

**Scope.** Procedural city generation, neural urban-layout generation, diffusion models for
urban morphology, graph-based street-network generation, controllable urban design, generative
modelling of geographic layouts, plus the data and metric work UrbanMorph builds on.

**How entries were verified (2026-10-30).** Every paper below was looked up online: title,
authors, year and venue were checked against the publisher, CVF/ACM/AAAI proceedings, arXiv or an
institutional repository (links given). Method, dataset and limitation notes come from abstracts,
project pages and paper summaries. For several papers I have **not** read the full text; those
notes say "(abstract)". This is a first pass, not a systematic review: there is no search
protocol, no inclusion/exclusion criteria, and recent preprints are under-sampled.

**Convention.** Sections 1–7 describe *established, published research*. Section 8 contains
*our own proposals* and is labelled as such. Nothing here claims UrbanMorph is novel.

---

## 1. Procedural city generation

### Procedural Modeling of Cities
- **Authors:** Yoav I. H. Parish, Pascal Müller · **Year:** 2001 · **Venue:** SIGGRAPH 2001 (pp. 301–308)
- **Link:** https://history.siggraph.org/?p=116923
- **Method:** Extended L-systems grow street networks guided by global goals (patterns such as grid
  or radial) and local constraints (water, elevation). Buildings are generated in the resulting lots.
- **Data:** Image maps (population density, water, elevation) as inputs; no learning.
- **Relevance:** This is the classic rule-based baseline. UrbanMorph's generator must at least
  match a well-tuned procedural model on realism and controllability to justify learning.
- **Limitations:** Rules are hand-designed. Style comes from expert parameter choices, not from data.

### Interactive Procedural Street Modeling
- **Authors:** Guoning Chen, Gregory Esch, Peter Wonka, Pascal Müller, Eugene Zhang · **Year:** 2008
- **Venue:** ACM Transactions on Graphics 27(3), Art. 103 (SIGGRAPH 2008) · **Link:** https://doi.org/10.1145/1360612.1360702
- **Method:** Tensor fields guide street tracing. Users edit the field or the graph directly.
- **Relevance:** Shows that *street orientation* can be an explicit, editable control field. This
  connects directly to our orientation metrics (dominant axis, φ).
- **Limitations:** No learning from real data. Realism depends on the user.

### Inverse Design of Urban Procedural Models
- **Authors:** Carlos A. Vanegas, Ignacio Garcia-Dorado, Daniel G. Aliaga, Bedrich Benes, Paul Waddell
- **Year:** 2012 · **Venue:** ACM TOG 31(6), Art. 168 (SIGGRAPH Asia 2012) · **Link:** https://doi.org/10.1145/2366145.2366187
- **Method:** Users specify target *indicators* (e.g. density-related quantities) and the system
  searches the procedural model's parameters to reach them.
- **Relevance:** This is the closest classical precedent for "generate a layout that satisfies explicit
  planning constraints". Our research question is, in part, a learned version of this.
- **Limitations:** Bound to one hand-written procedural grammar. Expressiveness is limited by the rules.

### Generating Urban Morphologies at Large Scales
- **Authors:** Juste Raimbault, Julien Perret · **Year:** 2019 · **Venue:** ALIFE 2019 (pp. 179–186), doi:10.1162/isal_a_00159
- **Link:** https://arxiv.org/abs/1903.06807
- **Method:** Several morphogenesis models generate district-scale configurations. These are
  calibrated against *morphological indicators* of real European districts (abstract).
- **Relevance:** Methodologically close to our evaluation idea: compare generated and real districts
  through indicator distributions, not only through visual similarity.
- **Limitations:** Rule-based generators, European districts only (abstract).

## 2. Neural urban-layout generation (GAN / VAE / graph networks)

### StreetGAN: Towards Road Network Synthesis with Generative Adversarial Networks
- **Authors:** Stefan Hartmann, Michael Weinmann, Raoul Wessel, Reinhard Klein · **Year:** 2017 · **Venue:** WSCG 2017 (pp. 133–142)
- **Link:** https://otik.zcu.cz/handle/11025/29554
- **Method:** Road-network patches are rasterised to binary images and a GAN is trained on them.
  Graphs are then extracted from the generated images.
- **Relevance:** This is the same "raster first, graph afterwards" route UrbanMorph is taking. It
  checks graph measures (block area, aspect ratio, compactness) on the extracted networks.
- **Limitations:** Roads only, single example networks, early GAN architecture (abstract).

### Modelling Urban Networks Using Variational Autoencoders
- **Authors:** Kira Kempinska, Roberto Murcio · **Year:** 2019 · **Venue:** Applied Network Science 4, Art. 114
- **Link:** https://doi.org/10.1007/s41109-019-0234-0 · **Code:** github.com/kirakowalska/vae-urban-network
- **Method:** A VAE trained on images of street networks from OSM for 1,059 cities with more than
  500k inhabitants. Latent vectors summarise network metrics, and samples resemble real complexity.
- **Relevance:** This is a direct precedent for a VAE on rasterised OSM street images. It is a natural
  baseline for Milestone 3.
- **Limitations:** Streets only (no buildings or green), unconditional, image-level evaluation.

### MetroGAN: Simulating Urban Morphology with Generative Adversarial Network
- **Authors:** Weiyu Zhang, Yiyang Ma, Di Zhu, Lei Dong, Yu Liu · **Year:** 2022 · **Venue:** KDD 2022, doi:10.1145/3534678.3539239
- **Link:** https://arxiv.org/abs/2207.02590
- **Method:** A progressive-growing GAN with a geographical (water-constraint) loss simulates
  metropolitan built-up morphology from physical-geography and other inputs (abstract).
- **Relevance:** Shows that *geographic conditioning* and domain losses help. It operates at the
  city-region scale, much coarser than our 4 m tiles.
- **Limitations:** Coarse resolution (built-up extent, not streets or buildings). Metric details
  were not verified from the full text.

### GANmapper: Geographical Data Translation
- **Authors:** Abraham Noah Wu, Filip Biljecki · **Year:** 2022 · **Venue:** Int. J. Geographical Information Science 36(7):1394–1422
- **Link:** https://doi.org/10.1080/13658816.2022.2041643 · **Code:** github.com/ualsg/GANmapper
- **Method:** Image-to-image GAN translation from road networks (and land use) to building
  footprints, tested in 9 cities.
- **Relevance:** Very close to our channel structure (roads → buildings). The authors explicitly
  suggest it for areas with uncertain OSM data, which is our Lima/Korea completeness problem.
- **Limitations:** Translation, not unconditional generation. Results depend on urban form and scale (paper).

### GlobalMapper: Arbitrary-Shaped Urban Layout Generation
- **Authors:** Liu He, Daniel Aliaga · **Year:** 2023 · **Venue:** ICCV 2023
- **Link:** https://openaccess.thecvf.com/content/ICCV2023/html/He_GlobalMapper_Arbitrary-Shaped_Urban_Layout_Generation_ICCV_2023_paper.html
- **Method:** Graph attention networks generate building layouts inside arbitrary road networks.
  Building rows are skeletonised to map blocks to a canonical form. Supports conditioning on learned priors.
- **Relevance:** A strong vector/graph alternative to our raster representation. Its evaluation
  covers 28 large cities, with a user study.
- **Limitations:** Takes the road network as given (it does not generate streets).

### COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation
- **Authors:** Liu He, Daniel Aliaga · **Year:** 2024 · **Venue:** ECCV 2024, doi:10.1007/978-3-031-72624-8_1
- **Link:** https://arxiv.org/abs/2407.11294 · **Code:** github.com/Arking1995/COHO
- **Method:** A graph masked autoencoder over buildings, blocks, communities and cities, with
  scheduled iterative sampling of 2.5D layouts. Data: 330 US cities.
- **Relevance:** Hierarchical context (neighbouring blocks) matters, which argues against treating
  512 m tiles as independent.
- **Limitations:** US cities only, so cross-regional generalisation is untested.

## 3. Diffusion models for urban morphology

### Text2City: One-Stage Text-Driven Urban Layout Regeneration
- **Authors:** Yiming Qin, Nanxuan Zhao, Bin Sheng, Rynson W. H. Lau · **Year:** 2024 · **Venue:** AAAI 2024, 38(5):4578–4586
- **Link:** https://doi.org/10.1609/aaai.v38i5.28257
- **Method:** Conditional DDPMs regenerate roads and buildings jointly from text, with constraints
  at attribute, geometry and pixel levels. Dataset of about 147K regions (search summary).
- **Relevance:** Joint road+building raster generation with attribute constraints. This is the
  closest published analogue to our planned conditional diffusion model.
- **Limitations:** Text conditioning is less precise than numeric planning targets. Dataset
  geography was not verified.

### ControlCity: A Multimodal Diffusion Model Based Approach for Accurate Geospatial Data Generation and Urban Morphology Analysis
- **Authors:** Fangshuo Zhou, Huaxia Li, Rui Hu, Sensen Wu, Hailin Feng, Zhenhong Du, Liuchang Xu · **Year:** 2024 · **Venue:** arXiv preprint 2409.17049
- **Link:** https://arxiv.org/abs/2409.17049
- **Method:** SDXL-based diffusion. Text and metadata give coarse alignment, and a ControlNet takes
  road-network and land-use images to produce building footprints. 22 cities; reported mean FID 50.94.
- **Relevance:** Shows ControlNet-style spatial conditioning for footprints and multi-city training.
- **Limitations:** Preprint (not yet peer-reviewed as far as I found). FID measures image realism,
  not planning-constraint satisfaction.

### Generative AI for Urban Design: A Stepwise Approach Integrating Human Expertise with Multimodal Diffusion Models
- **Year:** 2025 (arXiv 2505.24260; reported as published in *Building and Environment*) · **Link:** https://arxiv.org/abs/2505.24260
- **Method:** Three ControlNet stages: roads and land use, then buildings, then rendering. Checkpoints for New York and Chicago.
- **Relevance:** A staged pipeline that mirrors our channel order (roads → buildings).
- **Limitations:** Authors, journal details and evaluation **not yet verified**. US cities only.

## 4. Graph-based street-network generation

### Neural Turtle Graphics for Modeling City Road Layouts
- **Authors:** Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, Sanja Fidler
- **Year:** 2019 · **Venue:** ICCV 2019 (Oral) · **Link:** https://openaccess.thecvf.com/content_ICCV_2019/html/Chu_Neural_Turtle_Graphics_for_Modeling_City_Road_Layouts_ICCV_2019_paper.html
- **Method:** An autoregressive model that grows a road graph node by node (RNN encoder-decoder),
  trained on OSM. Supports style imitation of cities and sketch completion.
- **Relevance:** This is the graph-native counterpart to our raster representation. Its street
  topology is valid by construction, which a raster model must earn.
- **Limitations:** Roads only. (Author list verified on mlanthology.org/iccv/2019/chu2019iccv-neural.)

## 5. Generative 3D cities (relevant to the Godot milestone)

### InfiniCity: Infinite-Scale City Synthesis
- **Authors:** Chieh Hubert Lin et al. · **Year:** 2023 · **Venue:** ICCV 2023 (pp. 22808–22818)
- **Link:** https://openaccess.thecvf.com/content/ICCV2023/html/Lin_InfiniCity_Infinite-Scale_City_Synthesis_ICCV_2023_paper.html
- **Method:** Infinite-pixel 2D bird's-eye map synthesis, then octree voxel completion, then neural rendering.
- **Relevance:** Uses the same "2D semantic map → 3D city" decomposition as our roadmap.
- **Limitations:** Focus on visual realism, not planning control.

### CityDreamer: Compositional Generative Model of Unbounded 3D Cities
- **Authors:** Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, Ziwei Liu · **Year:** 2024 · **Venue:** CVPR 2024 (per arXiv and authors' repository)
- **Link:** https://arxiv.org/abs/2309.00610 · **Code:** github.com/hzxie/city-dreamer
- **Method:** Generates extendable semantic maps and height fields, then renders buildings and
  background (roads, green) with separate generators.
- **Relevance:** Its layout stage is essentially our (roads, buildings, green) map plus heights.
- **Limitations:** Visual-quality evaluation. No explicit planning constraints.

## 6. Controllable urban design with learning

### Spatial Planning of Urban Communities via Deep Reinforcement Learning
- **Authors:** Yu Zheng, Yuming Lin, Liang Zhao, Tinghai Wu, Depeng Jin, Yong Li · **Year:** 2023 · **Venue:** Nature Computational Science 3:748–762
- **Link:** https://doi.org/10.1038/s43588-023-00503-5 · **Code:** github.com/tsinghua-fib-lab/DRL-urban-planning
- **Method:** Graph-neural-network policy trained with RL to place land uses and roads, optimising
  planning objectives (e.g. accessibility). Includes a human–AI collaborative workflow. Evaluated on
  a synthetic grid and two Beijing communities.
- **Relevance:** Optimises *explicit planning objectives*, which is the controllability side of our
  question, but without learning a realism prior from many cities.
- **Limitations:** Full text not read (paywalled). Small number of test communities.

## 7. Data and metrics UrbanMorph relies on

### OSMnx (software) and Urban Spatial Order
- **Boeing, G. (2017).** *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems 65:126–139. (Well known; not re-checked in this pass.)
- **Boeing, G. (2019).** *Urban Spatial Order: Street Network Orientation, Configuration, and Entropy.* Applied Network Science 4, Art. 67. doi:10.1007/s41109-019-0189-1. https://arxiv.org/abs/1808.00600
- **Method/relevance:** Defines the orientation entropy and order φ that we compute. Covers 100
  cities and finds φ related to circuity and connectedness.

### A Spatio-Temporal Analysis Investigating Completeness and Inequalities of Global Urban Building Data in OpenStreetMap
- **Authors:** Benjamin Herfort, Sven Lautenbach, João Porto de Albuquerque, Jennings Anderson, Alexander Zipf
- **Year:** 2023 · **Venue:** Nature Communications 14, 3985 · **Link:** https://doi.org/10.1038/s41467-023-39698-6
- **Method:** Machine-learning estimates of OSM building completeness for 13,189 urban agglomerations.
- **Findings:** Only 1,848 urban centres (16% of the urban population) exceed 80% completeness,
  while 9,163 (48% of the urban population) are below 20%. Completeness is particularly low in
  parts of Latin America (among other regions).
- **Relevance:** **Critical.** Our pilot measured OSM/GHSL building-coverage ratios with medians of
  0.74–0.86 in Spain but 0.02 (Lima) to 0.40 (Cusco) in Peru and 0.16–0.37 in Korea. This is
  consistent with Herfort et al. and means cross-country building comparisons from raw OSM are confounded.
  Our v2 response was to restrict the study to well-mapped cities (Spain, US, Japan: 0.63–0.94).
  That trades geographic coverage for comparability, which is itself a selection bias.

### GHS-BUILT-S R2023A
- **Pesaresi, M., Politis, P. (2023).** GHS-BUILT-S R2023A – GHS built-up surface grid, multitemporal (1975–2030). European Commission, JRC. doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA. CC BY 4.0.
- **Relevance:** Our OSM-independent source for stratification (density, era of urbanisation) and for completeness checks.

---

## 8. Our proposals: potentially under-explored questions (NOT established research)

The following are **our hypotheses about gaps**, based on this incomplete first pass. Each needs a
targeted search before we claim it is open.

1. **Data-completeness bias in generative urban models.** Published generators are trained mostly
   on well-mapped regions (US cities in COHO; European/US/Chinese cities elsewhere). We found no
   work that measures how *uneven OSM completeness* (Herfort et al. 2023) leaks into a generative
   model, i.e. whether a model conditioned on "Lima" learns "few buildings" as a style when the
   cause is missing data. **Possible experiment:** train with and without completeness-filtered or
   completed building layers (e.g. GANmapper-style completion or external footprint datasets), then
   compare per-city metric distributions to satellite-derived references (GHSL).

2. **Constraint satisfaction measured jointly with geographic generalisation.** Diffusion work
   reports image metrics (FID) or text/attribute alignment, and inverse procedural design satisfies
   indicators but within one grammar. It is unclear how well a *learned* conditional model hits
   *numeric* planning targets (building coverage, green coverage, intersection density, orientation
   order) **on cities held out from training**, especially across very different regions (Spain,
   Peru, Korea). **Possible experiment:** leave-one-city-out evaluation with metric-error curves as
   the main result, separated from country-classification accuracy.

3. **Does a 4 m raster preserve a valid street network?** StreetGAN extracts graphs from generated
   images, and NTG generates graphs directly. A systematic comparison of graph-level plausibility
   (connectivity, dead-end ratio, intersection density, φ) between raster-generated and
   graph-generated layouts, at a resolution where real streets are 1–3 pixels wide, seems
   under-reported. **Possible experiment:** skeletonise generated road channels into graphs and
   compare their OSMnx statistics with real tiles from the same strata.

## 9. To do for the next pass
- A systematic search (Google Scholar / Semantic Scholar), with logged queries, 2017–2026.
- Read the full texts of Text2City, ControlCity, GlobalMapper and COHO, and record their exact evaluation metrics.
- Check external building datasets (Overture Maps, Microsoft/Google open footprints, national
  cadastres) for licence compatibility (ODbL / CC BY) and coverage in Peru and Korea.
