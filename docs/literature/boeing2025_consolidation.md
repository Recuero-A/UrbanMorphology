# Boeing 2025: Topological graph simplification and the intersection miscount problem

## 1. Header

- **Citation:** Geoff Boeing. "Topological Graph Simplification Solutions to the Street Intersection Miscount Problem." *Transactions in GIS* 29(3): e70037 (2025). Published online 6 May 2025. Open access, CC BY-NC-ND 4.0 (Crossref licence field).
- **DOI:** 10.1111/tgis.70037. The preprint carries DOI 10.48550/arXiv.2407.00258. An SSRN version (10.2139/ssrn.5194997) and an OSF version (10.31235/osf.io/xf7wm) also exist.
- **URLs opened (2026-11-25):**
  - https://arxiv.org/abs/2407.00258 (v4, 29 Apr 2025; journal-ref "Transactions in GIS, 2025")
  - https://arxiv.org/pdf/2407.00258 (read in full)
  - https://api.crossref.org/works/10.1111/tgis.70037; https://doi.org/10.1111/tgis.70037 (BibTeX)
- **Access level:** **FT** (arXiv v4). Page numbers are preprint pages (PDF page = printed page).

## 2. BibTeX (Crossref via doi.org; article number from the Crossref API)

```bibtex
@article{boeing2025topological,
  title   = {Topological Graph Simplification Solutions to the Street Intersection Miscount Problem},
  author  = {Boeing, Geoff},
  journal = {Transactions in GIS},
  volume  = {29},
  number  = {3},
  pages   = {e70037},
  year    = {2025},
  doi     = {10.1111/tgis.70037}
}
```

## 3. What it measures

The paper asks how far standard data and tools overcount street intersections. It presents the OSMnx algorithms that fix this and measures the bias worldwide.

It names three causes:
1. **nonplanarity** (planar line crossings at overpasses);
2. **intersection complexity** (divided roads, roundabouts, slip lanes);
3. **curve digitisation** (OSM vertices stored as nodes).

## 4. Exact definitions

- **Edge simplification** removes interstitial (degree-2 geometry) nodes and keeps the full edge geometry.
- **Geometric consolidation** buffers nodes of degree > 1 by a tolerance, merges overlapping buffers and counts the merged geometries.
- **Topological consolidation** (p. 12) is the one to use:
  1. cluster nodes geometrically within the tolerance;
  2. induce a subgraph on each cluster;
  3. split it into weakly connected components;
  4. merge only within a component;
  5. rebuild the graph and reconnect edges to the merged nodes, so that degrees are correct.
- **Intersection** means a node with degree > 1 after simplification. (Boeing 2022 instead counts nodes with more than two incident edges; note the difference.)
- **Global assessment:** drivable OSMnx graphs for 8,910 GHSL urban centres.
  - The tolerance is a **per-urban-area constant taken from the Atlas of Urban Expansion average road width** of the nearest Atlas city (p. 13).
  - It is **not a fixed 10 m**; 10 m is only an example (p. 10).

## 5. Key empirical numbers

- Edge simplification removes about 77% of raw nodes (160 M → 37 M) and 83% of edges (p. 14).
- **Median intersection count per urban area falls from 978 to 857** after topological consolidation, a 12% reduction. Equivalently, the unconsolidated count overestimates by **14%** (p. 14).
- Regional medians: Eastern Europe 6%, Western Europe 13% reduction.
- Individual cities: **Seoul 8%**, Dhaka 7%, **Sydney and Singapore 26%** (p. 14).
- The reduction correlates with log GDP-PPP (r = 0.25) and log night-time light (r = 0.38) (pp. 14–15).
- Manual validation (32 urban areas, 120 nodes each):
  - consolidated nodes: 98% true positives, 2% false positives;
  - non-consolidated nodes: 87% true negatives, 13% false negatives.
  - The author calls the parameterisation conservative ("underfitting") (p. 15).
- **Our cross-check** (computed from Dataverse v3.1 indicators, not in the paper): median of intersect_count / intersect_count_clean_topo − 1 across 10,351 urban areas = **14.6%** (p5 4.1%, p95 42.5%). That dataset uses a 10 m topological tolerance.

## 6. Verbatim quotes

- Abstract (p. 1): "Without consolidation, traditional methods would overestimate the median urban area intersection count by 14%. However, this bias varies drastically across regions"
- p. 6: "The intersection of such divided roads produces four nodes (i.e., 2 × 2 centerlines) in models deriving from these data"
- p. 12: "it merges nodes within some tolerance geometrically (as above) into clusters, induces a subgraph on each cluster's set of nodes, then divides this subgraph into its weakly connected components."
- p. 13: "we parameterize the latter algorithm with a per-urban-area constant tolerance, drawn from the Atlas of Urban Expansion's empirical average road width data set"
- p. 14: "which reduces the median urban area intersection count by a further 12%, from 978 to 857 intersections"
- p. 14: "Some urban areas see relatively small reductions, such as Dhaka (7%) and Seoul (8%), whereas others see much larger reductions, such as Sydney and Singapore (26% each)."

## 7. How UrbanMorph could use it

1. **Same consolidation for real and generated graphs before any junction statistic**, using topological consolidation with a stated tolerance. This covers:
   - close15, junction density, k_avg, 3/4-way shares, dead-end ratio, segment length;
   - all S10 terms that count nodes.
2. **close15 depends on consolidation.**
   - Real OSM tiles contain dual-carriageway and roundabout node clusters within 15 m. These are digitisation artefacts, and their frequency varies by city: Seoul 8% versus Sydney 26% overcount.
   - A generator trained on unconsolidated OSM will learn them.
   - Report close15 on (a) raw graphs and (b) topologically consolidated graphs at 10/15/20 m. This matches the M18 note and the M17 "rebuilt vs raw" discrepancy (M18_AUDIT §3.5).
3. **The 14% is a median over urban areas, not a per-tile constant.** Do not "correct" counts by 14%; recompute.
4. **Rebuild degrees after merging.** Merging without reconnecting edges leaves degrees wrong.
