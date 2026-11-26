# Metrics catalogue: street-network evaluation for UrbanMorph

**Scope.** This catalogue lists the metrics we use now, the ones we plan to adopt, and candidates from the measurement literature. Definitions for literature metrics come from the per-paper notes in this folder. Definitions for our own metrics come from project docs and code (`graphs/m14_metrics.py`, `graphs/terrain_metrics.py`, `graphs/junctions.py`, M10/M13/M15 protocols, SITE_BENCHMARKS). "n.v." marks anything not verified.

**Status key:** **O** = ours, in use; **P** = planned adoption; **C** = candidate from the literature.

**"Real refs?"** says whether published real-city values exist:
- **Pub**: published numbers exist;
- **Data**: an open dataset allows computing them;
- **Ours**: only our own real tiles;
- **None**: nothing found.

Even where Pub or Data exist, compare only on an identical pipeline. Scale, network type, consolidation and DEM all differ between sources.

## A. Catalogue

### Realism

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 1 | **S10** | O | Mean over 10 per-tile metrics of W1(generated, real) / IQR(real). The metrics are φ, H_o, junction-angle deviation, intersections/km², dead-end ratio, mean link length, street km/km², loops/km², degree-4 share and components (M10 §2.2). | unitless | ours; components from Boeing 2019 | Equal weights over **correlated** terms (φ and H are near-redundant; Zhou 2022 r = −0.99 between order and entropy). A small IQR inflates a term. Node-count terms depend on **consolidation** (Boeing 2025). The components term depends on **window clipping**. Marginal only. | Ours |
| 2 | **Typicality** | O | Score of a site's metric vector against the 156 real neighbourhoods (`scripts/m16_realism.py`); "1 = as typical as a real neighbourhood" (M17_RESULTS). Exact formula n.v. here. | unitless | ours | Reference-set composition (city mix, terrain mix) drives the score. Not terrain-conditional. | Ours |
| 3 | **Energy distance** | O | Two-sample energy distance between generated and real joint feature vectors, normalised by real-training means/SDs (M13 protocol). | unitless | ours | Sensitive to feature scaling and to the feature list; dominated by high-variance features. | Ours |
| 4 | Orientation entropy H_o, order φ | O (in S10) | 36 × 10° bins, −5° shift, reciprocal bearings, nats. φ = 1 − ((H_o − 1.386)/(3.584 − 1.386))². | nats; 0–1 | Boeing 2019 | Small-sample downward bias in 1 km tiles. Length-weighted H_w is biased by window shape. φ is non-monotonic below H_o = 1.386: a two-direction network (H_o = 0.693) scores φ ≈ 0.90, lower than a 4-way grid. | Pub (100 cities); Data (10k urban areas) |
| 5 | Edge sinuosity ("circuity" in Boeing) | C | Σ edge length / Σ straight-line distance between adjacent nodes. | ratio ≥ 1 | Boeing 2019/2022; SNDi sinuosity; streetz segment circuity | Depends on simplification (degree-2 nodes). Polyline vs smooth curves. **Not** route circuity. | Pub, Data |
| 6 | Block fingerprint P(Φ\|A) distance | C | Φ = A / A_circumscribed circle. Distance d_α = ∫\|f_α^gen − f_α^real\|dΦ over area bins [10³, 10⁴] and [10⁴, 10⁵] m². | unitless | Louf & Barthelemy 2014 | Ignores dead ends. Clipped faces. OSM face artefacts (dual carriageways, roundabouts). | Pub (clusters only, no numbers) |
| 7 | StreetMover distance | C | Sinkhorn OT cost between equidistant point clouds sampled on two graphs of the **same** site. Squared Euclidean cost, 100 points in the reference code. | length² | Belli & Kipf 2019 | Topology-blind (gaps are cheap). Fixed point count ignores total-length differences. Paired only. | n/a (paired) |

### Terrain adaptation

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 8 | **S_T** | O | Mean W1/IQR over grade_p90, share of length > 10% grade, contour index, steep-land density ratio and route circuity, vs real tiles of the same set. | unitless | ours | Inherits #12 and #20 failure modes. The circuity term drops unreachable pairs. DEM pipeline must match. | Ours |
| 9 | **TAS** = 100·(G + C + S)/3 | O | G = 1 − length share > 15% grade. C = 1 − contour index. S = length share on land below the site's median slope. 10 m pieces on a 16 m-smoothed DSM; ground edges only. | 0–100 | ours | C is undefined on flat land (terrain must be > 5%). S depends on window and land mask. Equal weights are arbitrary. No published baseline for C. | Ours (G partly: Data) |
| 10 | Grade length shares (> 5 / 10 / 15%) and hard violations (> 30%) | P | Length share of 10 m ground sub-spans whose \|grade\| exceeds each threshold. | share | thresholds from [street_grades_real_world.md](street_grades_real_world.md) | Strongly DEM- and sub-span-dependent. Bridges and tunnels must be excluded. | Data (Boeing pipeline only: node-to-node, 30 m DEM) |
| 11 | grade_mean / grade_median, Boeing style | C | Unweighted mean/median over undirected edges of \|Δelev\| / edge length, node-to-node. | % | Boeing 2022 | Inflated by short edges and DEM noise (flat-city floor about 1.2–1.9% grade_mean). Hides within-edge maxima. | Data (10,351 urban areas) |
| 12 | Contour index | O (in S_T and TAS) | Length-weighted \|cos θ\| between street piece and fall line (= street grade / terrain slope), where terrain slope > 5%. | 0–1 | ours (generation rule in Chen 2008 / Nishida 2016 per M18) | No empirical real-city baseline exists in the literature found. Noisy on gentle slopes. Smoothing-scale dependent. | Ours only |
| 13 | Relief-response slopes | C | Within a set, regress tile circuity, dead-end share and P4W on tile relief; compare the generated slope with the real slope. | per unit relief | direction from Yang et al. 2021 (r = 0.612, 0.444, −0.499 across Chinese cities) | Ecological; confounded by city and era. Needs many tiles per relief bin. | Pub (direction only) |
| 14 | Steep-land street density ratio | O (in S_T) | Street density on steep land / density on gentle land (exact threshold per `terrain_metrics.py`). | ratio | ours | Unstable when steep land is a small share of the window. | Ours |

### Connectivity and access

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 15 | **OD unreachability (same land)** | O | Share of sampled **node** pairs (30 sources × ≤ 7 targets, 150–500 m apart) on the same dry-land region with no path on the closed graph (window border joined by a virtual node). Empty network = 1. | share | ours (M14) | Node-weighted (dense parts dominate). Blind to land with no streets. Per-tile SD about 0.03. Different RNG streams for real vs generated (M18_AUDIT §3). | Ours |
| 16 | **Islands** | O | Components with ≥ 40 m of street that do not touch the border and are not the closed giant; count and length share. | count, share | ours | Depends on the window (border-touching components excluded). | Ours |
| 17 | **Dead-end ratio** | O / P | Share of degree-1 nodes (RoadWeaver: "proportion of terminal nodes"). | share | Boeing 2019 P_de; B-LMB 2015; RoadWeaver 2026 | Window clipping creates false dead ends unless border nodes are excluded. Depends on consolidation and service roads. Real values **rise with relief** (Yang 2021). | Pub (Boeing 2019: 0.027–0.395); Data (10k urban areas: p5 0.051, median 0.170, p95 0.338) |
| 18 | **LCC share** | P | Nodes in the largest connected component / all nodes. | share | RoadWeaver 2026 (cites OSMnx) | Node-weighted. Insensitive to small detached pieces in large graphs. Window clipping splits real networks too. | Ours |
| 19 | **Reachability** | P | Fraction of node pairs connected by a path (all pairs on small graphs). | share | RoadWeaver 2026 | Approximately LCC share² for one dominant component. Counts pairs across water as unreachable unless land-masked (use same-land, as #15). | Ours |
| 20 | **Cycle ratio** | P | Share of nodes on at least one cycle (paper). The reference repo uses the 2-core share, which also counts nodes on paths *between* cycles, so it is not identical. | share | RoadWeaver 2026; see [li2026_roadweaver.md](li2026_roadweaver.md) | Definition mismatch (2-core vs cycle membership). Window clipping. | Ours |
| 21 | Dendricity: bridge share and non-cycle share | C | Edge classes: bridge (excluding dead ends), dead end, self-loop, cycle basis. Report the fraction of bridges and of non-cycle edges, by count and by length. | share | Barrington-Leigh & Millard-Ball 2019 | Footpaths change values (SNDi includes them). 20 m merging. | Pub (paradigms; country and city tables of non-cycle length) |
| 22 | Route circuity (banded) | O (S_T) / C | Network / straight-line distance for node pairs in distance bands. Ours: **median** over **reachable** pairs 200–400 m. SNDi: log(Σ S_AB) − log(Σ R_AB) per band. | ratio ≥ 1 | SNDi; Yang 2021 (50,000 random pairs); streetz "transport ratio" | **Unreachable pairs dropped** (streetz, ours) rewards fragmentation. Fix: cap them (M13 `circuity_all` uses 3.0) and report the unreachable share. | Pub (SNDi paradigms 1.24–2.34 at 0–0.5 km) |
| 23 | **Shared geographic OD test** | P | Same random geographic points for real and generated: sample point pairs on the same dry land, snap each to the nearest street within r (e.g. 60 m), and score success when a path exists. Report unreachable, unsnappable and detour separately. | shares | ours (fixes M18_AUDIT §3.1) | r choice. Same RNG stream for both graphs is required. | Ours |
| 24 | Mean node degree, 4-way share | O (in S10) / C | k̄ = 2\|E\|/\|V\|; P_4w = share of degree ≥ 4 nodes. | count, share | Boeing 2019; B-LMB 2015 | Consolidation and degree-2 annealing. | Pub, Data |

### Junction granularity

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 25 | **close15** | O | Share of junctions whose nearest other junction is < 15 m (`junctions.py`). | share | ours | **Consolidation-dependent**: raw generated graphs read 0.01–0.04 higher (M18_AUDIT §3.5). Real OSM has divided-road and roundabout clusters. | Ours (e.g. KAIST real 0.098) |
| 26 | Intersection density, topologically consolidated | C | Nodes of degree > 1 after topological merge within tolerance τ, per km² (land or built-up). | 1/km² | Boeing 2022 (τ = 10 m), Boeing 2025 | τ choice. Denominator (land vs built-up). | Pub (39–284 per km² built-up by subregion); Data |
| 27 | Overcount ratio | C | Raw intersection count / consolidated count − 1. | share | Boeing 2025 | Digitisation detail varies by city (Seoul 8%, Sydney 26%). | Pub (median 14%); Data (median 14.6%) |
| 28 | Median segment length | O (mean link length in S10) / C | Median length of simplified edges. | m | Boeing 2019 | Simplification and consolidation. | Pub (23.2–233.0 m); Data (median of urban areas 84.9 m) |

### Blocks and form

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 29 | **Block area** | O | Areas of closed faces not touching the window. | m² | Louf 2014 | Clipping. Dead ends invisible. OSM artefact faces. | Pub (τ ≈ 2 tail; 10³–10⁵ m² dominant) |
| 30 | **Polsby–Popper** | O | 4πA/P² per block. | 0–1 | ours | Penalises polyline roughness and curved digitisation. | Ours |
| 31 | Louf Φ | C | A / A_circumscribed circle per block (square ≈ 0.64; 4:1 rectangle ≈ 0.30). | 0–1 | Louf 2014 | Depends only on extreme vertices; insensitive to boundary detail. | Pub (Manhattan peaks 0.3 and 0.5, qualitative) |
| 32 | Block aspect, rectangularness | C | Short/long side of the minimum-area bounding rectangle; A / rectangle area. | 0–1 | Birsak 2022 (streetz) | Assumes planar faces; holes not handled. | None |

### Diversity

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 33 | **Jaccard diversity** | O | Mean pairwise Jaccard distance of 8 m occupancy grids across the seeds of one site (M15_METRICS). | 0–1 | ours | Rewards noise as well as structural variety. Should be read together with realism. | n/a |

### Downstream feasibility

| # | Metric | St. | Exact definition | Units | Source(s) | Known failure modes | Real refs? |
|---|---|---|---|---|---|---|---|
| 34 | **Grade-limited routing success** | P | Over the shared OD pairs (#23), the share with a path in the subgraph of edges whose every 10 m ground sub-span has \|grade\| ≤ g_max. g_max = 10% (primary: fire, bus) and 15% (local-street maximum); 20% as sensitivity. | share | thresholds: IFC D103.2, NYSDOT/AASHTO, UN R13-H (see grade note) | Directional grade matters only for trucks (use \|grade\|). Bridges count as passable at deck grade (DSM proxy wrong there). Real cities also fail it on steep sites, so compare with real. | Ours (none published) |
| 35 | Grade-limited detour | C | Network distance under the grade limit / unconstrained network distance, over pairs that succeed. | ratio ≥ 1 | ours | Conditional on success; report with #34. | Ours |
| 36 | Coverage | O | Share of land within 60 m of a street. | share | ours | Land-mask dependent. | Ours |

## B. Recommended primary set (11)

1. **S10.** Keep it for continuity, but drop H_w-like redundancy and consolidate first.
2. **Energy distance** (joint realism).
3. **Dead-end ratio**, computed with border nodes excluded and after consolidation.
4. **Non-cycle (dendricity) length share** (SNDi). It catches loops-on-sticks and tree branches that dead ends and islands miss, and it is close to (1 − cycle ratio) in an edge-based, length-weighted form.
5. **Shared geographic OD unreachability, same land.** It replaces node-pair OD.
6. **LCC share.** Standard in RoadWeaver and OSMnx, so it gives comparability with generation papers.
7. **close15 on topologically consolidated graphs** at τ = 10/15/20 m, with the raw value as a diagnostic.
8. **Block area plus P(Φ\|A) distance** (Louf), with Polsby–Popper as secondary.
9. **G: length share above 15% grade**, plus shares above 5% and 10% and the hard-violation share above 30%, against same-pipeline real tiles.
10. **Contour index (C).** Report it, but state that no published real-city baseline exists; the reference is our own real tiles.
11. **Grade-limited routing success** at 10% and 15%.

Secondary: banded route circuity with capped unreachable pairs, reachability, cycle ratio, Jaccard diversity, typicality, relief-response slopes, and StreetMover (paired site fidelity only).
