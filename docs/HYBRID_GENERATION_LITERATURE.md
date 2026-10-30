# Hybrid neural-procedural generation: literature review (Milestone 6)

**Purpose:** decide what UrbanMorph should build for the hybrid road + building generator, and what
is already established. **Verification:** every paper below was checked on its publisher or
proceedings page; links are in the references. Items marked † were verified in earlier milestones
(M4 / M5 reviews). **Novelty:** none is claimed. Our planned extensions are listed separately in §4
and are combinations of known ideas, applied to our data and evaluation.

## 1. Key concepts (short)

* **Procedural generation.**
  * Hand-written rules or programs produce geometry: "grow a street, split a block, extrude a building".
  * Strengths: exact, controllable and always valid.
  * Weakness: the rules encode their author's style.
* **Shape grammar.**
  * A procedural method whose rules rewrite shapes into smaller shapes
    (`Block → Parcel*`, `Parcel → Setback Building`).
  * Results stay structured and valid, but only what the rules allow can be produced.
* **Neural-procedural (hybrid).**
  * The procedure decides **what kind of thing can exist** and keeps geometry valid.
  * A neural network makes the **choices** inside it: which rule, where, how big, at what angle.
    These choices are learned from data instead of being chosen by hand.
* **Graph generation.** A model outputs nodes and edges, usually one at a time (autoregressively).
  Connectivity is explicit rather than inferred from pixels.
* **Geometric primitive.** A parameterised shape, for example a segment (start, length, angle),
  an arc (+ curvature) or a polygon (vertices). The network predicts the parameters, so it is not
  limited to a fixed library of templates.
* **Conditional generation.** The model samples p(buildings | roads) rather than p(buildings);
  the condition is given, not generated.
* **Differentiable vs non-differentiable.**
  * Gradients flow through convolutions and mixture densities.
  * They do not flow through discrete rules such as polygonisation, "snap to node" or "is this
    polygon valid".
  * Hybrids therefore usually either train the neural part on teacher-forced data (as in M5),
    or use RL / sampling-based estimators.
* **Generalisation.** Performance on unseen neighbourhoods.
  * Here this means unseen spatial blocks, because tiles from one city are spatially correlated.
  * Leave-one-city-out is the stricter test.

## 2. Papers

### 2.1 Procedural and grammar-based

| | **Parish & Müller 2001, Procedural Modeling of Cities** (SIGGRAPH) |
|---|---|
| Representation | Road graph; lots; building geometry |
| Generator | Extended L-system. Proposals follow **global goals** (population density, pattern: grid / radial / organic); **local constraints** fix them (snap to nearby intersections, cut at water, shorten) |
| Data | Hand-made input maps (water, density); no learning |
| Constraints | Intersection snapping, legality checks, lot subdivision |
| Evaluation | Visual |
| Limitations | Styles chosen by hand; no distributional realism |
| Relevance | Our M5 sampling rules (snap / cross / T-junction) are exactly the **local constraints** idea. What we add is that the **global goals** are learned instead of chosen |

| | **Stiny & Gips 1972, Shape grammars**; **Müller et al. 2006, Procedural Modeling of Buildings (CGA shape)** (SIGGRAPH) |
|---|---|
| Representation | Shapes plus rewriting rules (split, repeat, component split) |
| Generator | Rule application with hand-set parameters and probabilities |
| Data | None (expert-authored) |
| Constraints | Valid by construction; occlusion queries in CGA |
| Evaluation | Visual / qualitative |
| Limitations | Every style must be authored; parameters are not learned |
| Relevance | Motivates **parameterised primitives + learned parameters** rather than learned pixels |

| | **Vanegas et al. 2012, Procedural Generation of Parcels in Urban Modeling** (Eurographics / CGF) |
|---|---|
| Representation | Block polygons subdivided into parcels |
| Generator | Two procedures: OBB recursive splitting, and straight-skeleton strips. Driven by user parameters (area, width, irregularity) |
| Data | None for generation; compared visually to real parcel maps |
| Constraints | Every parcel gets street access; minimum area / width |
| Evaluation | Visual similarity to real cities |
| Limitations | Parameters set by hand; mostly regular parcels |
| Relevance | Candidate **Stage B** procedure. Its parameters could be **fitted** per country |

| | **Yang et al. 2013, Urban Pattern: Layout Design by Hierarchical Domain Splitting** (SIGGRAPH Asia) |
|---|---|
| Representation | Domain recursively split into streets and parcels |
| Generator | Streamline splitting of a cross field, plus template splitting |
| Data | None (design tool) |
| Constraints | Field smoothness; template geometry |
| Evaluation | Visual / user studies |
| Limitations | Interactive, authored |
| Relevance | A cross field is a natural, **non-grid** way to model local street orientation |

| | **Nishida, Garcia-Dorado & Aliaga 2016, Example-Driven Procedural Urban Roads** (CGF) |
|---|---|
| Representation | Road graph |
| Generator | Grows roads by copying **patches** of an example network, plus procedural growth |
| Data | One example city network at a time |
| Constraints | Snapping / intersection rules |
| Evaluation | Visual and statistical similarity to the example |
| Limitations | Copies the example (no generalisation); a memorisation risk by design |
| Relevance | Shows that a **local-context** growth process can reproduce a style, and warns that patch copying needs a memorisation check |

### 2.2 Neural graph and layout generation

| | **Chu et al. 2019, Neural Turtle Graphics (NTG)** (ICCV) † |
|---|---|
| Representation | Road graph. Nodes plus **turtle steps: displacement relative to the incoming path** |
| Generator | Encoder RNN over a node's incoming paths; decoder RNN emits outgoing steps; node queue; merge of nearby nodes |
| Data | OSM road graphs of real cities (tiles) |
| Constraints | Merge / snap |
| Evaluation | Graph statistics, perceptual study; also used as a prior for road parsing from aerial images |
| Limitations | Roads only; no buildings; no hierarchy / width |
| Relevance | Our M5 backbone. **Our M5 diagnosis (§3) points to NTG's local-frame step parameterisation as the missing part** |

| | **You et al. 2018, GraphRNN** (ICML) † |
|---|---|
| Representation | Adjacency matrix rows in BFS order |
| Generator | Hierarchical RNN (graph-level and edge-level) |
| Data | Generic graphs (no geometry) |
| Constraints | None |
| Evaluation | MMD of degree, clustering and orbit statistics |
| Limitations | No geometry |
| Relevance | BFS ordering; **MMD / Wasserstein on graph statistics** as evaluation |

| | **Xu et al. 2021, BlockPlanner** (ICCV) |
|---|---|
| Representation | One city block as a **ring graph of lots** (each lot a vector of geometry + land use) |
| Generator | Graph VAE with message passing; heads for lot existence, geometry, land use |
| Data | NYC block dataset |
| Constraints | **Losses** (overlap, boundary alignment), i.e. learned rather than enforced |
| Evaluation | Distributional and geometric metrics, user study |
| Limitations | Rectangle-like lots; NYC only; blocks are given |
| Relevance | **Stage B / C** in one model; shows the "condition on a given block" set-up |

| | **He & Aliaga 2023, GlobalMapper** (ICCV) |
|---|---|
| Representation | Buildings of an **arbitrary-shaped block** as a graph. The block is mapped by a **skeleton-based canonical transform** to a shape-independent grid |
| Generator | Graph attention network inside a VAE |
| Data | Building footprints and blocks of many US cities (OSM-derived) |
| Constraints | Canonical transform keeps buildings inside the block |
| Evaluation | Distribution metrics, user study |
| Limitations | US data; building shapes from a small parametric family; roads given |
| Relevance | **The closest prior work to our Stage C**: buildings conditioned on (given) roads and blocks |

| | **He & Aliaga 2024, COHO** (ECCV) |
|---|---|
| Representation | City / community / block / building as one hierarchical graph (2.5D) |
| Generator | Graph masked autoencoder; scheduled iterative sampling |
| Data | 330 US cities |
| Constraints | Learned plus canonical block transform |
| Evaluation | Realism, a context score (neighbouring-block consistency) |
| Limitations | US only; roads given |
| Relevance | Large-scale version of Stage C; the context metric is useful for us |

| | **Para et al. 2021, Generative Layout Modeling using Constraint Graphs** (ICCV) |
|---|---|
| Representation | Layout elements (nodes) plus **constraints** (edges: adjacency, alignment, size) |
| Generator | Transformers generate nodes, then constraint edges; **constrained optimisation** solves the final geometry |
| Data | Floor plans and furniture layouts |
| Constraints | Explicit and solved, not learned |
| Evaluation | Distribution metrics, constraint satisfaction |
| Limitations | Not urban; the optimiser is non-differentiable |
| Relevance | A clean **hybrid template**: neural network for discrete structure, solver for validity |

### 2.3 Neural guidance of programs (neural-procedural)

| | **Ritchie et al. 2016, Neurally-Guided Procedural Models** (NeurIPS) |
|---|---|
| Representation | A procedural program (L-system-like) with random choices |
| Generator | A neural network sets the distribution of each random choice from the partial output |
| Data | Samples from SMC on constraint targets |
| Constraints | Built into the program |
| Evaluation | Speed / quality of constrained generation |
| Limitations | Toy targets; not urban |
| Relevance | Exactly the M5 / M6 structure: **program = growth + validity rules, network = choices** |

| | **Ellis et al. 2018, Learning to Infer Graphics Programs from Hand-Drawn Images** (NeurIPS) |
|---|---|
| Representation | LaTeX-like drawing programs (primitives, loops, conditionals) |
| Generator | CNN proposes primitives; program synthesis finds the program |
| Data | Synthetic and hand-drawn diagrams |
| Constraints | Program semantics |
| Evaluation | Accuracy of the inferred programs |
| Limitations | Simple 2D diagrams |
| Relevance | Program synthesis is a long-term option (e.g. inferring "repeat parcel" structure); too ambitious for now |

### 2.4 Raster / neural-field city generation (for contrast)

| | **Xie et al. 2024, CityDreamer** (CVPR) |
|---|---|
| Representation | Bird's-eye-view semantic map + height field, then neural fields for buildings / background |
| Generator | Separate unbounded layout generator and neural-field renderers |
| Data | OSM + Google Earth imagery (CityGen datasets) |
| Constraints | None explicit for road topology |
| Evaluation | Image realism (FID / KID), 3D consistency |
| Limitations | The layout is raster, with no road-graph validity; not aimed at morphological statistics |
| Relevance | A state-of-the-art **raster** approach: the counterpart of our M3–M4 line |


### 2.5 Relative building patterns and hard constraints (added in Milestone 7)

| | **Zhang, Ai, Stoter, Kraak & Molenaar 2013, Building pattern recognition in topographic data: collinear and curvilinear alignments** (*GeoInformatica* 17(1), 1–33) |
|---|---|
| Representation | Building footprints and proximity graphs |
| Method | Computational geometry + graph theory + perception (Gestalt) rules detect alignments of centres and sides; several algorithms combined |
| Data | Real topographic datasets; validated by an expert survey |
| Constraints | — (recognition, not generation) |
| Relevance | Names the patterns real buildings follow (collinear, curvilinear, along-road, grid-like, unstructured) and how to **measure** them (our `nn_aligned_5deg`, frontage metrics) |

| | **Strauss 1975, A model for clustering** (*Biometrika* 62, 467–475); Kelly & Ripley 1976 |
|---|---|
| Representation | Point patterns |
| Generator | Pairwise-interaction point process: each pair closer than r contributes a factor γ ≤ 1 (inhibition); hard-core variants forbid distances < h |
| Relevance | Formalises "not chaos, not a grid": realism lives in **relative** positions. Our no-overlap rule is a hard core; our frontage model learns the interaction (spacing, alignment) instead of fixing γ |

| | **Nauata et al. 2020, House-GAN** (ECCV) |
|---|---|
| Representation | Room bounding boxes + a bubble (adjacency) graph |
| Generator | Relational GAN; the graph constraint is built into the networks' message passing |
| Data | 117,587 real floor plans |
| Evaluation | Realism, diversity, compatibility with the graph |
| Relevance | Relations between elements as part of the network, not only as post-hoc rules |

**Established vs ours:** row / ring generation along block boundaries is established (BlockPlanner,
GlobalMapper). Our frontage model is a simplified, street-side version trained on our 3-country data,
combined with programmed hard constraints. No novelty is claimed.

## 3. What our own results add (established in this repository)

* **Raster VAE + learned prior (M3–M4):** about 4 floating fragments per tile against 0 real, even with
  a learned prior. Pixels do not encode connectivity.
* **Graph generator (M5):** connectivity is solved by construction (0 floating components). But
  the absolute-frame (dx, dy) parameterisation produces **branches at wrong angles**: junction
  angles deviate a median of 14.8° from 90°/180° (real: 3.4°). The result is too many
  junctions and orientation order φ ≈ 0.14 against 0.59 real. NTG itself predicts steps
  relative to the incoming path (§2.2), which we simplified away.

## 4. Established methods vs our proposed extensions

| Component | Established (cite) | Our extension (not novel as a concept; new in this combination and evaluation) |
|---|---|---|
| Road growth with learned steps | NTG; Ritchie 2016 | Turtle-frame steps + learned **street class / width**; evaluated per country against real tiles |
| Validity via local rules | Parish & Müller 2001 | Same rules shared by data extraction and sampling; we report how often each rule fires |
| Blocks from roads | polygonisation (standard computational geometry) | — |
| Buildings in blocks | BlockPlanner, GlobalMapper, COHO | A small **autoregressive footprint model with a local raster context** (road distance, neighbours), trained on our 3-country data |
| Parcels | Vanegas 2012 (procedural) | Not learned in v1 (we have no parcel data; OSM rarely maps parcels) |
| Evaluation | GraphRNN MMD; BlockPlanner / COHO metrics | Same protocol across raster and vector models, per country, with memorisation checks |

## References

* Parish, Y. I. H., Müller, P. (2001). Procedural Modeling of Cities. *SIGGRAPH 2001*, 301–308. doi:10.1145/383259.383292
* Stiny, G., Gips, J. (1972). Shape grammars and the generative specification of painting and sculpture. *Information Processing 71* (IFIP Congress), 1460–1465.
* Müller, P., Wonka, P., Haegler, S., Ulmer, A., Van Gool, L. (2006). Procedural Modeling of Buildings. *ACM TOG* 25(3), 614–623. doi:10.1145/1141911.1141931
* Vanegas, C. A., Kelly, T., Weber, B., Halatsch, J., Aliaga, D. G., Müller, P. (2012). Procedural Generation of Parcels in Urban Modeling. *CGF* 31(2), 681–690. doi:10.1111/j.1467-8659.2012.03047.x
* Yang, Y.-L., Wang, J., Vouga, E., Wonka, P. (2013). Urban Pattern: Layout Design by Hierarchical Domain Splitting. *ACM TOG* 32(6), 181. doi:10.1145/2508363.2508405
* Nishida, G., Garcia-Dorado, I., Aliaga, D. G. (2016). Example-Driven Procedural Urban Roads. *CGF* 35(6), 5–17. doi:10.1111/cgf.12728
* Chu, H. et al. (2019). Neural Turtle Graphics for Modeling City Road Layouts. *ICCV 2019*. †
* You, J., Ying, R., Ren, X., Hamilton, W., Leskovec, J. (2018). GraphRNN. *ICML 2018*. †
* Xu, L., Xiangli, Y., Rao, A., Zhao, N., Dai, B., Liu, Z., Lin, D. (2021). BlockPlanner: City Block Generation with Vectorized Graph Representation. *ICCV 2021*.
* He, L., Aliaga, D. (2023). GlobalMapper: Arbitrary-Shaped Urban Layout Generation. *ICCV 2023*, 454–464. arXiv:2307.09693
* He, L., Aliaga, D. (2024). COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation. *ECCV 2024*. arXiv:2407.11294
* Para, W., Guerrero, P., Kelly, T., Guibas, L., Wonka, P. (2021). Generative Layout Modeling using Constraint Graphs. *ICCV 2021*, 6690–6700.
* Ritchie, D., Thomas, A., Hanrahan, P., Goodman, N. (2016). Neurally-Guided Procedural Models. *NeurIPS 2016*. arXiv:1603.06143
* Ellis, K., Ritchie, D., Solar-Lezama, A., Tenenbaum, J. (2018). Learning to Infer Graphics Programs from Hand-Drawn Images. *NeurIPS 2018*.
* Zhang, X., Ai, T., Stoter, J., Kraak, M.-J., Molenaar, M. (2013). Building pattern recognition in topographic data: examples on collinear and curvilinear alignments. *GeoInformatica* 17(1), 1–33. doi:10.1007/s10707-011-0146-3
* Strauss, D. J. (1975). A model for clustering. *Biometrika* 62, 467–475.
* Nauata, N., Chang, K.-H., Cheng, C.-Y., Mori, G., Furukawa, Y. (2020). House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation. *ECCV 2020*. arXiv:2003.06988
* Xie, H., Chen, Z., Hong, F., Liu, Z. (2024). CityDreamer: Compositional Generative Model of Unbounded 3D Cities. *CVPR 2024*, 9666–9675.
