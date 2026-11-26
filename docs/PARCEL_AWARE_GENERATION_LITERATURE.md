# Parcel-aware generation: literature review (Milestone 8)

**Verification:** every entry was checked on a publisher, proceedings or institutional page in
this or an earlier milestone (links in References). The search was not exhaustive. "Not found"
means not found in our searches, not "does not exist". **Novelty:** none is claimed.

## 1. Concepts (short)

* **Block:** land enclosed by streets. We compute it from road *boundaries*, not centrelines (§6
  of the plan).
* **Parcel / plot / lot:** a unit of land ownership. It is a legal object recorded in a cadastre,
  and is **not** visible in OSM building data. One parcel can hold zero, one or many buildings, and
  one building can span several parcels.
* **Frontage, depth, setback, coverage:**
  * frontage = parcel width along the street;
  * depth = distance from the street to the rear boundary;
  * setbacks = gaps between the building and the front, side and rear boundaries;
  * coverage = building area / parcel area.
* **Observed vs inferred vs generated parcels:**
  * observed = from a cadastre;
  * inferred = estimated by an algorithm from buildings and roads, which is an *assumption*;
  * generated = sampled by our model.

  These are never mixed up in metrics or figures.

## 2. Works

### 2.1 Procedural modelling with parcels

| | Representation | Generator | Data | Parcel method | Constraints | Evaluation | Limitations | Relevance |
|---|---|---|---|---|---|---|---|---|
| **Parish & Müller 2001**, Procedural Modeling of Cities (SIGGRAPH) | road graph → blocks → lots → buildings | extended L-system for roads; recursive block subdivision | hand-made input maps | recursive splitting of blocks into lots until a size threshold; lots without street access discarded | snapping, legality checks | visual | style chosen by hand | the road → block → lot → building **hierarchy** we adopt |
| **Müller, Wonka, Haegler, Ulmer & Van Gool 2006**, Procedural Modeling of Buildings (ACM TOG 25(3), SIGGRAPH) | CGA shape grammar | split / repeat rules | none (authored) | — (building mass inside a lot) | valid by construction | visual | every style authored | lot → building-mass rules; later 3D export |
| **Aliaga, Vanegas & Beneš 2008**, Interactive Example-Based Urban Layout Synthesis (ACM TOG 27(5), SIGGRAPH Asia) | streets + parcels + per-parcel aerial images | example-based synthesis of street and parcel structure | real city fragments (hundreds to thousands of blocks and parcels) | parcels copied / adapted from examples | fit of examples into new parcels | visual | needs enough parcel variety; copies examples | data-driven parcels from real examples, but no learned model |
| **Vanegas, Kelly, Weber, Halatsch, Aliaga & Müller 2012**, Procedural Generation of Parcels in Urban Modeling (CGF 31(2), Eurographics) | block polygon → parcels | two procedures: OBB recursive splitting, straight-skeleton strips | none for generation; compared to real parcel maps | user parameters: target area, width, irregularity | street access, minimum width / area | visual similarity to real subdivisions | parameters by hand | **our procedural baseline** (Experiment C); its parameters are what we would learn (D) |
| **Yang, Wang, Vouga & Wonka 2013**, Urban Pattern: Layout Design by Hierarchical Domain Splitting (ACM TOG 32(6), SIGGRAPH Asia) | domain recursively split into streets and parcels | streamline splitting of a cross field + template splitting | none (design tool) | hierarchical splitting | field smoothness | visual, user study | authored | orientation fields for non-grid subdivisions |
| **Chen, Song & Ortner 2024**, Hierarchical Co-generation of Parcels and Streets in Urban Modeling (CGF 43(2), Eurographics) | parcels and streets generated together, hierarchically | binary splitting of parcels per level; graph search adds streets so every parcel has access | not learned | hierarchical binary splits | street access for every parcel | visual / design cases | procedural; no distributions learned | most recent procedural parcel work; access constraint |

### 2.2 Neural / learned

| | Representation | Architecture | Data | Parcel / block method | Constraints | Evaluation | Limitations | Relevance |
|---|---|---|---|---|---|---|---|---|
| **Zhang, Shehata, Beneš & Aliaga 2020/21**, Automatic Deep Inference of Procedural Cities from Global-scale Spatial Data (ACM TSAS 7(2)) | city → blocks → (procedural) parcels → buildings | neural networks estimate **procedural parameters** per block from satellite images + road / population / elevation data | global data, many cities | **rule-based** parcel split (two styles); networks choose parameters | procedural validity | approximation of real cities | parcels not compared to cadastres | **neural network chooses parameters, procedure builds parcels**: the pattern of our Experiment D |
| **Chu et al. 2019**, Neural Turtle Graphics (ICCV) | road graph, turtle steps | encoder–decoder RNN, node queue | OSM roads | — | merge / snap | graph statistics | roads only | our road generator (M5–M6) |
| **Xu, Xiangli, Rao, Zhao, Dai, Liu & Lin 2021**, BlockPlanner (ICCV) | one block as a ring + two-tier graph of **land lots** (geometry + land use) | graph VAE with message passing | NYC block dataset | lots generated **inside a given block** (learned) | losses (overlap, boundary) | distributional, user study | one city; lots close to rectangular | the only learned **lot** generator found; single-city |
| **He & Aliaga 2023**, GlobalMapper (ICCV) | buildings of an arbitrary-shaped block on a skeleton-based canonical grid | graph attention network in a VAE | US cities (OSM-derived) | **no parcels**: buildings directly in blocks | canonical transform keeps buildings in the block | distributional, user study | US only; roads given | block-conditioned buildings **without** parcels: the hypothesis our Experiment B tests |
| **He & Aliaga 2024**, COHO (ECCV) | city / community / block / building graph | graph masked autoencoder | 330 US cities | blocks; no parcels | learned | realism, context score | US only | context across neighbouring blocks |

### 2.3 Urban morphology of plots (real cadastral data)

| | Data | Method | Findings | Relevance |
|---|---|---|---|---|
| **Bobkova, Berghauser Pont & Marcus 2021**, Towards analytical typologies of plot systems: quantitative profile of five European cities (EPB 48(4)) | real plots, 5 European cities | k-means on plot size, frontage, compactness (configurational measures) | 7 plot types | which parcel variables to model (**size, frontage, compactness**) and that they cluster into types: conditioning on morphology, not only on country |
| **Zhang, Ai, Stoter, Kraak & Molenaar 2013** (GeoInformatica 17(1)) | topographic building data | alignment-pattern recognition | collinear / curvilinear / along-road / grid patterns | pattern metrics (M7) |

## 3. What is established and what is open

**Established:**

* **The road → block → parcel → building hierarchy:** Parish & Müller.
* **Procedural parcel subdivision** (recursive OBB, skeleton strips, binary hierarchical splits):
  Vanegas 2012, Chen 2024.
* **Neural networks choosing procedural parameters:** Zhang et al. 2020/21.
* **A learned lot generator inside a given block, for one city:** BlockPlanner.
* **Learned building generation inside blocks without parcels:** GlobalMapper, COHO.

**Not found in our searches (open questions relevant to UrbanMorph):**

1. A learned parcel subdivision model trained on **observed cadastral parcels from several
   countries**, evaluated per country against held-out cadastral data.
2. A controlled test of whether **adding parcels** improves building realism **compared with a
   block / neighbour-aware generator without parcels**. GlobalMapper and BlockPlanner each take one
   side; we found no ablation between them.
3. **Probabilistic setback / coverage models** P(setback, coverage | parcel, context) learned from
   observed parcels across morphologies.

These are research questions, not claims. They require observed parcels to be answered
honestly.

## References

* Parish, Y. I. H., Müller, P. (2001). Procedural Modeling of Cities. *SIGGRAPH 2001*, 301–308. doi:10.1145/383259.383292
* Müller, P., Wonka, P., Haegler, S., Ulmer, A., Van Gool, L. (2006). Procedural Modeling of Buildings. *ACM TOG* 25(3), 614–623. doi:10.1145/1141911.1141931
* Aliaga, D. G., Vanegas, C. A., Beneš, B. (2008). Interactive Example-Based Urban Layout Synthesis. *ACM TOG* 27(5), 160. doi:10.1145/1409060.1409113
* Vanegas, C. A., Kelly, T., Weber, B., Halatsch, J., Aliaga, D. G., Müller, P. (2012). Procedural Generation of Parcels in Urban Modeling. *CGF* 31(2), 681–690. doi:10.1111/j.1467-8659.2012.03047.x
* Yang, Y.-L., Wang, J., Vouga, E., Wonka, P. (2013). Urban Pattern: Layout Design by Hierarchical Domain Splitting. *ACM TOG* 32(6), 181. doi:10.1145/2508363.2508405
* Chen, Z., Song, P., Ortner, F. P. (2024). Hierarchical Co-generation of Parcels and Streets in Urban Modeling. *CGF* 43(2) (Eurographics 2024). doi:10.1111/cgf.15053
* Zhang, X., Shehata, A., Beneš, B., Aliaga, D. (2020/2021). Automatic Deep Inference of Procedural Cities from Global-scale Spatial Data. *ACM TSAS* 7(2), 6. doi:10.1145/3423422
* Chu, H. et al. (2019). Neural Turtle Graphics for Modeling City Road Layouts. *ICCV 2019*.
* Xu, L. et al. (2021). BlockPlanner: City Block Generation with Vectorized Graph Representation. *ICCV 2021*.
* He, L., Aliaga, D. (2023). GlobalMapper: Arbitrary-Shaped Urban Layout Generation. *ICCV 2023*, 454–464.
* He, L., Aliaga, D. (2024). COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation. *ECCV 2024*.
* Bobkova, E., Berghauser Pont, M., Marcus, L. (2021). Towards analytical typologies of plot systems: quantitative profile of five European cities. *Environment and Planning B* 48(4), 604–620. doi:10.1177/2399808319880902
* Zhang, X., Ai, T., Stoter, J., Kraak, M.-J., Molenaar, M. (2013). Building pattern recognition in topographic data. *GeoInformatica* 17(1), 1–33.
