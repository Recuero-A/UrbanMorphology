# Milestone 17 literature: junction placement and intersection clustering

Compiled 2026-11-22, before any M17 implementation. Extends `docs/M15_LITERATURE.md` and `docs/M16_LITERATURE.md`
(their Parish & Müller, Chen, CitySprout and RoadWeaver entries are reused). **Problem:** generated networks contain
clusters of junctions that sit very close together.

**Access levels:** **F** full text read · **F\*** full text read through a summarising fetch tool · **K** full text
searched by keyword, with only the relevant passages read · **A** abstract or metadata only · **S** package source code
read. "Ours" marks arithmetic or interpretation added by us, not stated by the source. Page and section numbers refer
to the version named in the Access column.

## 1. Sources

| # | Citation | Access | DOI / URL |
|---|---|---|---|
| B1 | Boeing, G. (2025). Topological graph simplification solutions to the street intersection miscount problem. *Transactions in GIS* 29(3), e70037 (Crossref: published 2025-05) | F (arXiv:2407.00258v4, 29 Apr 2025). Journal typeset version not read. | doi:10.1111/tgis.70037 |
| O1 | OSMnx `consolidate_intersections`, `simplify_graph`, `count_streets_per_node`. Repo venv has **2.1.1**; latest on PyPI is **2.2.0** (2026-11-18); **2.0.7** is the last 2.0.x release | S (docstrings and code from the installed 2.1.1 and from the 2.0.0, 2.0.7 and 2.2.0 wheels) | osmnx.readthedocs.io/en/stable/user-reference.html; github.com/gboeing/osmnx |
| N1 | Chu, H., Li, D., Acuna, D., Kar, A., Shugrina, M., Wei, X., Liu, M.-Y., Torralba, A. & Fidler, S. (2019). Neural Turtle Graphics for modeling city road layouts. *ICCV 2019*, 4521–4529 | F (arXiv:1910.02055v1, main paper). **The supplement and appendix were not read.** | doi:10.1109/ICCV.2019.00462 |
| G1 | Parish, Y. I. H. & Müller, P. (2001). Procedural modeling of cities. *SIGGRAPH 2001*, 301–308 | F (author PDF) | doi:10.1145/383259.383292 |
| G2 | Chen, G., Esch, G., Wonka, P., Müller, P. & Zhang, E. (2008). Interactive procedural street modeling. *ACM TOG* 27(3), Art. 103 | F (author PDF, §6.1) | doi:10.1145/1360612.1360702 |
| G3 | Dobbie, C. & Johansson, A. (2026). CitySprout. *EPJ Data Science* 15, 76; supplement from the Research Square preprint | F (supplement §1); F\* (paper) | doi:10.1140/epjds/s13688-026-00694-7 |
| R1 | Bastani, F., He, S., Abbar, S., Alizadeh, M., Balakrishnan, H., Chawla, S., Madden, S. & DeWitt, D. (2018). RoadTracer. *CVPR 2018*, 4720–4728 (pages from L1's reference list) | F (arXiv:1802.03680v2) | arxiv.org/abs/1802.03680 |
| R2 | He, S., Bastani, F., Jagwani, S. et al. (2020). Sat2Graph: road graph extraction through graph-tensor encoding. *ECCV 2020* | K (arXiv:2007.09547v1, §3.1 read) | arxiv.org/abs/2007.09547 |
| R3 | Xu, Z., Liu, Y., Gan, L., Sun, Y., Wu, X., Liu, M. & Wang, L. (2022). RNGDet. *IEEE T-GRS* (venue from the arXiv comment "Accepted by IEEE TGRS"; not verified) | K (arXiv:2202.07824v2) | arxiv.org/abs/2202.07824 |
| R4 | Xu, Z., Liu, Y., Sun, Y., Liu, M. & Wang, L. (2023). RNGDet++. *IEEE RA-L* (accepted March 2023) | K (arXiv:2209.10150v2); no merge rule found | arxiv.org/abs/2209.10150 |
| L1 | Birsak, M., Kelly, T., Para, W. & Wonka, P. (2022). Large-scale auto-regressive modeling of street networks. arXiv:2209.00281 | K (§3.5 and the statistics appendix) | arxiv.org/abs/2209.00281 |
| L2 | Li, Y. et al. (2026). RoadWeaver. arXiv:2608.11580v1 | F\* (HTML, §III-C and §IV) | arxiv.org/abs/2608.11580 |
| E1 | Barthelemy, M. & Boeing, G. (2024). A review of the structure of street networks. *Findings* | F (arXiv:2409.08016, Table 1) | doi:10.32866/001c.122117 |
| E2 | FHWA (2004). *Signalized Intersections: Informational Guide*, FHWA-HRT-04-091, ch. 8 | F\* (fhwa.dot.gov HTML) | fhwa.dot.gov/publications/research/safety/04091/08.cfm |

**Not checked:** lane-graph and RoadNetTransformer-style extraction works, CityDreamer (raster scene generation, not a
road graph), StreetGAN, GraphRNN applied to roads (beyond NTG's "GraphRNN-2D" baseline), and the TRB *Access
Management Manual*. Nothing in this document relies on them.

## 2. Boeing (B1): why node counts overcount intersections

| Topic | What B1 says (arXiv v4) |
|---|---|
| Definitions (§2.1) | "An intersection exists wherever two or more streets connect with each other and a dead-end is the terminus of a street." A street segment lies "between a pair of intersections and/or dead-ends". **Operational count:** "the non-dead-end nodes (i.e., the subset of nodes with degree > 1)" (§2.2.1), counted after simplification (§3.3). B1 itself does not use "street count" or "unique streets"; those belong to OSMnx (§3 below). |
| Cause 1: nonplanarity (§2.2.1, Fig. 1 left) | Planar line-intersection counts from geometry-only data (e.g. TIGER/Line) count overpasses and underpasses as intersections. Fix: model a **nonplanar** graph and count only topological junctions. |
| Cause 2: intersection complexity (§2.2.2, Fig. 1 centre) | A **divided road** is digitised as two one-way centrelines, so two divided roads crossing give "four nodes (i.e., 2 × 2 centerlines)". A single-roundabout 4-way junction becomes "four three-way intersections". **Slip lanes** and turn-lane digitisation also add nodes. The bias is largest in car-centric neighbourhoods. |
| Cause 3: curve digitisation (§2.2.3, Fig. 1 right) | OSM uses the same node type for curve vertices and for junctions. Fix: edge simplification (§3.1, Fig. 2), which keeps a node only if one of six rules holds: self-loop; all edges in or all out; 1 neighbour; >2 neighbours; 2 neighbours but degree 3 (a one-way/two-way transition); or a user attribute rule. Elbow bends are kept only through rule 6. |
| Geometric consolidation (§3.2.1, Fig. 4) | "geometrically buffers the set of graph nodes with degree > 1 by some user-defined tolerance (i.e., buffer radius)", "merges overlapping node buffers into unified geometries and counts those geometries". The tolerance can be one constant (example "10 meters") or set per node. Fig. 4 shows too small (false negative), appropriate, and too large (two real intersections merge: false positive). |
| Topological consolidation (§3.2.2, Fig. 5) | Geometric merging is wrong when nodes are "spatially proximate but topologically remote". B1's examples are surface streets under a highway overpass and on-ramp ("planar Euclidean distance … approximately 0"), and bollard-diverted 4-way junctions. The method merges by network distance using "a fast heuristic": it forms the geometric clusters, "induces a subgraph on each cluster's set of nodes, then divides this subgraph into its weakly connected components". It then "topologically rebuilds the graph … by reconnecting 'broken' edges to the new consolidated nodes", which gives correct degrees. |
| Chains of buffers | B1 says only that overlapping buffers are merged into unified geometries. Chaining is therefore transitive. B1 does not discuss chain length, and no maximum cluster span is given (ours, confirmed in the code, §3). |
| Order and idempotence (§3.2.2) | Simplify first, then consolidate. Each runs **once** ("can and should only be run once"). |
| Recommended tolerance (§3.2.1, §3.3, §5) | B1 recommends **no universal value.** It says to set the tolerance "according to the irregular urban design standards within the study site — e.g., smaller in fine-grained networks, larger in coarser-grained ones", noting the bias–variance trade-off. The global study used one constant per urban area, taken from the **Atlas of Urban Expansion average road width** of the nearest matched city. The road-width values used are not listed in the text. B1 names per-node tolerance estimation as future work. |
| Numbers (§4) | Global study: 8,910 GHSL urban areas, drivable networks. Before simplification: >160 M nodes and 320 M edges; after: 37 M nodes and 53 M edges (≈77% and 83% reductions). Consolidation reduced the median urban-area count by **12% (978 → 857)**, so the uncorrected count overestimates by **14%**. Median reduction is 6% in Eastern Europe and 13% in Western Europe; Dhaka 7%, Seoul 8%, Sydney and Singapore 26%. The log reduction correlates with log GDP-PPP (r = 0.25) and log night-time lights (r = 0.38), both p < 0.001. |
| Validation (§3.3, §4) | 32 urban areas, stratified, with 120 nodes each (half consolidated, half not), checked by hand against Street View and imagery. **Consolidated nodes:** 98% true positives and 2% false positives; about 5% of the true positives were "insufficiently robust" (partial merges). **Non-consolidated nodes:** 87% true negatives and 13% false negatives. B1 reads this as conservative underfitting. Edge simplification was 100% accurate. |

## 3. OSMnx (O1): exact behaviour

| Item | Behaviour, with docstring quotes (identical in 2.0.0, 2.0.7, 2.1.1 and 2.2.0 unless noted) |
|---|---|
| `tolerance` | Default `10`. "Note: `tolerance` represents a per-node buffering radius. For example, to consolidate nodes within 10 meters of each other, use `tolerance=5`." Two nodes merge directly if their distance is **< 2 × tolerance** (ours: buffers of radius t overlap). It accepts a scalar or a dict per node: "any missing node IDs will not be buffered". 2.2.0 adds a check: values ≤ 0 raise `ValueError`. Use a projected graph in metres. |
| Chaining | Code: `gdf_nodes.buffer(tol).union_all()`, then one cluster per connected polygon. Chaining is **transitive**: n nodes spaced just under 2t apart form one cluster spanning up to (n−1)·2t (ours). |
| `rebuild_graph` | Default `True`. False: "purely geometric (and relatively fast)", and returns a GeoSeries of the **centroids of the merged buffer polygons**. True: "topological (and slower but more accurate)". Each geometric cluster is split into the weakly connected components of its induced subgraph. A split sub-cluster takes the centroid of its own nodes; otherwise the cluster takes the polygon centroid. New IDs; the original IDs are kept in `osmid_original`. |
| `max_length` | **2.1.1 and 2.2.0 only; absent in 2.0.x.** "Ignore edges longer than this when determining whether nearby nodes are topologically connected … prevents long edges that loop outside an intersection, such as cloverleaf ramps, from merging their nearby endpoint nodes." |
| `reconnect_edges` | Default `True`. "If False, the returned graph has no edges (which is faster if you just need topologically consolidated intersection counts)." When True, edges inside a cluster are dropped (original self-loops are kept), and edge geometries are extended to the new node with updated `length`. |
| `dead_ends` | Default `False`: "discard dead-end nodes to return only street intersection points". The code removes nodes with `street_count <= 1` **before** clustering, so the rebuilt graph also loses them and their edges. Use `True` to keep cul-de-sacs. |
| `node_attr_aggs` | Aggregations per attribute, in `pandas.agg` style. Attributes not listed keep their unique values (a list if there are several). Default `{"elevation": numpy.mean}` (the private helper says `"mean"`). `street_count` is recomputed for merged nodes. |
| Grade separation | The base graph is nonplanar, so crossings without a shared node are never intersections. In geometric mode, nodes of a bridge and the street below **do merge** if within 2t. With `rebuild_graph=True` they stay separate unless an edge connects them inside the cluster (for example a short ramp); `max_length` handles long ramps. |
| `count_streets_per_node` | "Count how many physical street segments connect to each node". It uses an undirected view, counts each parallel edge, and counts a self-loop once. It does **not** count distinct named streets. |
| `simplify_graph` | Removes "all nodes that are not intersections or dead-ends". `edge_attrs_differ=["osmid"]` keeps elbow nodes, and `node_attrs_include` keeps tagged nodes. Same six rules as B1 §3.1. |

## 4. Neural Turtle Graphics (N1)

| Topic | What the main paper states |
|---|---|
| Graph model (§3.1) | Nodes are "spatial control points", so bends are nodes too. G is undirected and "planar since all intersections are present in V". Coordinates are in metres. |
| Node merging at generation (§3.2, §3.5) | "If the distance to an existing node is below a threshold (5m in our paper), we do not add the new node to the queue. Instead, an edge is included to connect the current node to the existing node. This enables the generation of cycles." The check is against **any** existing node (junction or control point). The paper also caps **maximum node degree, maximum node density and minimum angle** at the training-set extremes; their values and effects are in the supplement, which was not read. Offsets are discrete at 1 m resolution, limited to ±100 m (§3.4). |
| Crossing handling | No step that splits segments at crossings is described for NTG. For the GraphRNN-2D baseline they "add checks to avoid invalid edge crossing during inference" (§4.1.2). |
| Training data and sequences (§3.5, §4.1) | **Not BFS sequences.** For each node, K random walks of at most L nodes serve as incoming paths. The model predicts all neighbours, sorted counter-clockwise, with teacher forcing. Inference pops nodes from a queue; the pop order is not specified. Data: OSM "road markers" from 17 cities, using the densest 10 km² per city (RoadNet: 233.6k nodes, 262.1k edges). **No intersection consolidation, simplification or T-junction preprocessing is mentioned.** |
| Metrics (§4.1.1, Table 2) | **Perceptual:** FID from an InceptionV3 fine-tuned on city IDs, applied to renders of a 300 m neighbourhood around each node. **Urban planning:** node density within 100, 200 and 300 m; connectivity (node degree); reach (road length within 100, 200 and 300 m); and convenience (Euclidean / Dijkstra distance for pairs more than 500 m apart), scored with a Fréchet distance. **Diversity:** percentage of road outside a 10 m vicinity of the other graph. Parsing (§4.2) uses APLS with a 5 m buffer, on graphs that are RDP-simplified and subdivided to ≤ 30 m. |
| Junction clustering, duplicate nodes, over-dense intersections | **Not mentioned** as a failure mode. The density caps above are the only related item. |

## 5. Related work: how junction spacing is enforced and measured

| Work | Rule for reusing or merging junctions (radius) | Junction quality measured? |
|---|---|---|
| Parish & Müller 2001 (G1, §3.3, §3.3.1, Fig. 8) | The local constraints first check illegal areas, then "search for intersection with other roads or for roads and crossings that are within a specified distance to the segment end". "If the localConstraints function finds a street within the given radius of the end of a segment": "two streets intersect → generate a crossing"; "ends close to an existing crossing → extend street, to reach the crossing"; "close to intersecting → extend street to form intersection". "After a crossing is generated, a road stops growing." **No numeric radius is given.** | No. The evaluation is qualitative (Manhattan). |
| Chen et al. 2008 (G2, §6.1) | Graph nodes are the intersection points of the major and minor hyperstreamlines. Tracing stops when a line comes closer than `d_sep` to an existing line. It continues for `d_lookahead` "to search an intersection" to improve connectivity. `d_sep` is user-set and comes from density maps; no default value is stated in the text read. | No. |
| CitySprout (G3, supplement §1, Fig. 3) | Check 1 moves a new node to the nearest crossing on an existing edge. Check 2: if an existing node is within `intersectRadius` (default **0.8 arbitrary units**), the new node is not created and the parent is linked to the existing node. The stated aim is to prevent "node overcrowding", since without it the city "would become overly dense with nodes … far closer together than is realistic". | No spacing metric. |
| RoadTracer (R1, §3.2) | Step D = **12 m**. "if u is within distance 3D of a vertex v ∈ G, but the shortest distance in G from Stop to v is at least 6D, then we add an edge (u, v) and don't push u". That is a Euclidean radius of 36 m and a graph gap of at least 72 m (ours), and it "prevents small loops". It is a **topology-aware snap** inside sequential generation. | **Junction metric** (§5): a junction is a vertex with ≥ 3 edges. It matches ground-truth and inferred junctions and reports F_correct and F_error. The matching radius is not stated in the text read. |
| Sat2Graph (R2, §3.1) | Vertices are local maxima of the vertexness map above p_thr. Edges snap to the best vertex within **15 m**. Training interpolates to spacing < d m, and stacked-road vertices are shifted to be **> 5 m** apart (non-planar crossings are kept). | TOPO and APLS; no junction-specific metric. |
| RNGDet / RNGDet++ (R3, R4) | Ground-truth labels drop degree-2 vertices ("only intersection vertices (degrees larger than 2) and endpoint vertices"). Label thresholds τ = 40 px and τ′ = 20 px. Initial vertices are peaks of an intersection segmentation map. No inference-time merge radius was found by keyword search. RNGDet notes that peak finding (in VecRoad) fails "when vertices are very close to each other". | Pixel P/R/F, TOPO, APLS; no junction spacing metric. |
| Birsak et al. 2022 (L1, §3.5) | Distance field → threshold → thinning and "road-tracing [RoadTracer]" → small parts filtered. No snap radius. | Node degree, circuity, transport ratio, PageRank; no spacing metric. |
| RoadWeaver 2026 (L2, §III-C) | The road snaps when it "approaches an existing road within the snapping radius" (**no value given**). A* reconnection joins dangling ends; junctions are refined by "junction-aware processing". | Dead-end and cycle ratios, and endpoint alignment error (0.24 m); no junction density or spacing. |

A web search (not exhaustive) found **no generator paper that measures nearest-junction distance or junction
clustering** directly. The closest measures are RoadTracer's junction-by-junction metric, NTG's node density within
100–300 m, and CitySprout's overcrowding rationale.

## 6. Real-world spacing references

| Source | Content | Relevance |
|---|---|---|
| E1, Table 1 (100 cities from Boeing 2019) | Average edge length l₁: median **118.6 m**, range **64.5–537.5 m**. Average degree median 2.93; dead-end share median 14%; 4-way share median 18%. Whether these graphs were consolidated is **not verified**. | Typical segment scale. Spacing under 20 m is about an order of magnitude below the median (ours). |
| B1 §3.3 | Consolidation tolerance tied to **average road width** (Atlas of Urban Expansion). | Clusters of real junctions inside one carriageway width are digitisation artefacts, not separate junctions. |
| E2, §8.2.4 and Table 39; §8.3 | Arterial access spacing tables: e.g. 161 m at 12 access points/km, and "each additional access point per mile … increases the crash rate by about 4 percent". Signals within 0.8 km should be coordinated. The functional area is D1 + D2 + D3 (Fig. 63). | **Arterials only.** Not a local-street minimum. |

We found no verifiable source for a minimum spacing between local-street junctions or for an empirical distribution of
nearest-junction distances. It should be measured on our own consolidated training cities.

## 7. Implications for M17 (ours)

1. **Junction definition.** First run `simplify_graph`-equivalent removal of degree-2 shape points. Then a junction is a
   node with **≥ 3 incident at-grade street segments**: `count_streets_per_node` semantics, with parallel edges counted
   and self-loops counted once. Crossings without a shared node (bridges and tunnels) are excluded. Also report
   B1's count (street_count ≥ 2 after simplification) for comparability. Note that OSMnx counts segments, not distinct
   named streets. A divided road therefore contributes two.
2. **Thresholds.** Report a **pairwise distance d**, for example the share of junctions whose nearest junction is
   closer than d, plus the full nearest-junction-distance CDF compared with real cities. When calling OSMnx, state
   `tolerance = d/2` explicitly. Because buffers chain, also report the largest cluster span.
3. **Baseline.** The standard post-hoc baseline is B1/OSMnx **topological** consolidation (`rebuild_graph=True`,
   `reconnect_edges=True`, `dead_ends=True` to keep cul-de-sacs, `max_length` set if OSMnx ≥ 2.1.1). Apply it
   **identically to real and generated graphs**. Raw OSM over-counts junctions (median +14%, Sydney +26%). If the
   training graphs were not consolidated, some clustering may be learned from the data. Check this before blaming the
   sampler. Geometric consolidation is a weaker variant to ablate.
4. **Novelty.** Reusing existing junctions during sequential growth is old. Parish & Müller extend a street end to
   "an existing crossing" within a radius. NTG connects to any node within 5 m. CitySprout links to an existing node
   within `intersectRadius` to prevent "node overcrowding". RoadTracer already makes the snap **topology-aware**
   (Euclidean ≤ 3D but graph distance ≥ 6D). Any claim must be narrow. Examples: a specific learned or
   learned-plus-rule reuse decision for a GraphTurtle-style model, evaluated against post-hoc B1 consolidation at
   matched d on nearest-junction distributions, degree mix and connectivity. Do not claim "first junction snapping",
   "first topology-aware merging" or "first to address junction clustering". The last is only "not found among works
   searched".
