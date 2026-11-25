# Dobbie & Johansson 2025/2026 — CitySprout

## 1. Header

- **Preprint:** Cameron Dobbie, Anders Johansson (University of Bristol). "In Pursuit of the Optimal City: Evaluation of Procedurally Generated City Layouts." Research Square preprint, posted 14 Nov 2025, CC BY 4.0. The cover page says "This is a preprint; it has not been peer reviewed by a journal."
  - DOI: **10.21203/rs.3.rs-7492997/v1**.
- **Published version (title changed):** Cameron Dobbie, Anders Johansson. "Comparing urban network morphologies: a new living metric and a framework for evaluating procedurally generated city layouts." *EPJ Data Science* 15(1), published 31 Aug 2026.
  - DOI: **10.1140/epjds/s13688-026-00694-7**.
  - Licence CC BY-NC-ND (OpenAlex). Article number not verified.
- **URLs opened:**
  - https://www.researchsquare.com/article/rs-7492997/v1
  - https://assets-eu.researchsquare.com/files/rs-7492997/v1_covered_fcbc61ca-ed5b-475c-81f5-680bda132b23.pdf (preprint, read in full)
  - https://assets-eu.researchsquare.com/files/rs-7492997/v1/5f3ab11b2c10c35e840de7ca.pdf (Supplementary Information, read in full)
  - https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-026-00694-7, which redirects to link.springer.com and is **blocked by a bot check**
  - https://link.springer.com/content/pdf/10.1140/epjds/s13688-026-00694-7.pdf (bot check, not obtained)
  - https://api.openalex.org/works/doi:10.1140/epjds/s13688-026-00694-7 (EPJ abstract)
  - Europe PMC REST search (no record)
  - https://research-information.bris.ac.uk/en/publications/2aef5bac-fa4d-4953-a4c7-f60942171c67 (record only, no file)
  - https://api.crossref.org/works?query.bibliographic=… (both DOIs)
  - https://api.github.com/repos/camdobbie/CitySprout and `/git/trees/main?recursive=1`
  - raw files `README.md`, `requirements.txt`, `metrics.py`, `realCities.py` and `lambda_sensitivity_analysis.py` from https://raw.githubusercontent.com/camdobbie/CitySprout/main/
- **Access level:** **FT for the preprint and SI; AO for the EPJ version** (abstract via OpenAlex only). The EPJ method changes, such as the real-city validation and population density, are therefore **not verified** beyond the abstract.

## 2. BibTeX

```bibtex
@article{dobbie2026comparing,
  title   = {Comparing urban network morphologies: a new living metric and a framework for evaluating procedurally generated city layouts},
  author  = {Dobbie, Cameron and Johansson, Anders},
  journal = {EPJ Data Science},
  volume  = {15},
  number  = {1},
  year    = {2026},
  doi     = {10.1140/epjds/s13688-026-00694-7}
}

@misc{dobbie2025pursuit,
  title        = {In Pursuit of the Optimal City: Evaluation of Procedurally Generated City Layouts},
  author       = {Dobbie, Cameron and Johansson, Anders},
  year         = {2025},
  howpublished = {Research Square preprint},
  doi          = {10.21203/rs.3.rs-7492997/v1}
}
```

## 3. What it does

CitySprout is an open-source parametric, stochastic L-system that grows planar road graphs from a single seed node using one of four grammars: Organic, Grid, Hex and Line. The paper scores the generated cities on space efficiency (road density, or population density in the EPJ version), "living structure" (a new living metric for hierarchical betweenness structure) and travel efficiency (mean circuity). It concludes that the "optimal" layout depends on the metric. The EPJ version adds validation of the living metric on six real cities (abstract).

## 4. Learned or procedural, and how

- **Procedural grammar. Not learned.**
- **Nodes:**
  - a road type: m (main), l (large), s (small);
  - a node type: start, Leaf, Twig, Branch;
  - an incoming edge direction.
- **Production rules** (SI Table 4) give:
  - children with deflection angles and distances (fixed values or random ranges);
  - occurrence probabilities;
  - `minDistances` conditions;
  - `newRoad` flags, which group edges into "roads".
- **Growth.** Each iteration applies the rules to every node.
  - **Check 1:** a new edge that crosses existing edges is truncated at the crossing nearest its parent, which creates an intersection.
  - **Check 2:** a new node within `intersectRadius` (default 0.8 arbitrary units) of an existing node is replaced by an edge to that node.
- **Stopping:** iterations, edge count, total road length or population. Population is taken as proportional to road length.
- **Output:** a NetworkX graph (vector) in arbitrary units, starting at (0, 0). No training data, no tiles, no raster.

## 5. Inputs / outputs and terrain

- **Inputs:** grammar, seed, `intersectRadius`, optional `maxWidth`/`maxHeight`, stopping criteria (SI Table 1).
- **Output:** a NetworkX graph with road numbers and road types; plots and animations.
- **Terrain: none.**
  - "This means that the cities have no bridges or tunnels, and are instead a flat road network" (p. 3).
  - The discussion claims the Organic grammar could build around "a lake or river, or a particularly steep area of terrain" (p. 16). This is **not implemented or tested**.
  - No DEM, water or boundary input exists in the SI parameter tables.

## 6. Evaluation

Preprint; the EPJ version is not verified.

- **Design.** 4 grammars × 10 seeds, each city grown to a population of 1,000,000. Synthetic only; no real-data split.
- **Metrics (exact definitions):**
  - **Road density** D_r = L_r / A. L_r is the total edge length; A is the area of an alpha shape around the nodes with α = 20 (arbitrary units). The repo's `calculatePopulation` multiplies road length by 154.15, described as loosely based on London.
  - **Road betweenness** B_R: the number of all-pairs shortest paths (Dijkstra) passing through a road (a group of edges), min-max normalised to [0, 1].
  - **Living metric (LM).** Rank all roads by B_R. For road r, take the β roads immediately below and β above it in the ranking. LM(r) is the number of those 2β roads that share a node with r. LM(G) is the mean of LM(r) over roads, with β = 5.
    - Repo implementation (`metrics.py`, `calculateLivingMetricDict`): the window runs over sorted indices `range(idx-β, idx+β)`, so it includes r's own slot and only β−1 above.
    - Tied betweenness values collapse through a dict inversion.
    - These quirks come from reading the code; their effect is not verified.
  - **Mean circuity** c̄: the mean over all node pairs of c(p, q) = d_N(p, q) / d_L2(p, q), the network shortest-path length over the Euclidean distance.
    - The code averages over the pairs returned by its shortest-path routine.
    - The treatment of unreachable pairs is not stated in the paper (not verified).
- **Results (median, Fig. 7):**

| Metric | Organic | Grid | Hex | Line |
|---|---|---|---|---|
| Road density | 0.6665 | 1.6169 | 1.1665 | 1.3472 |
| Living metric | 0.0518 | 0.0244 | 0.0295 | 0.0186 |
| Mean circuity | 1.2944 | 1.2959 | 1.2374 | 1.135 |

  - IQRs are reported for density and circuity, with violin and box plots.
  - There are no significance tests and no CIs.
  - Across the 40 cities, road density and mean circuity are negatively correlated, with Grid as the outlier (Fig. 11).
- **EPJ addition (abstract only).** The living metric is validated "by applying it to six real cities", and population density replaces road density.
  - The repo's `realCities.py` downloads Royston, Milton Keynes, London, Barcelona, New York and Paris through OSMnx (`network_type="drive"`). Its LM and circuity functions for real cities are stubs ("Not sure how to do this!").
  - Whether these are the six cities in the paper, and how LM was computed there, is **not verified**.
  - A `living_metric_v2` with a λ parameter appears in the repo outputs. Its definition is not verified.
- **Runtime.** Not tabulated. The authors cite "hardware limitations" and say Check 2 becomes slow because it compares every new node with every existing node.

## 7. Verbatim quotes

Preprint pages are the printed page numbers; the PDF page is that number + 1 because of the cover.

- p. 3: "This means that the cities have no bridges or tunnels, and are instead a flat road network."
- p. 4: "The first check ensures that if a new edge crosses an existing one, the new node (at the end of this new edge) is placed at the intersection point with the existing edge, instead of beyond it."
- p. 4: "The second check helps prevent overcrowding by merging any two nodes that are placed too close to each other."
- p. 8: "The α parameter, which controls how tightly the alpha shape surrounds the network, was set to 20."
- p. 8: "The “living structure” of a city describes how well the road network exhibits a clear hierarchical structure from minor to major roads."
- p. 9: "This value (the number of the 2β roads surrounding r in the betweenness ordering that are directly connected to r) becomes the living metric for road r."
- p. 9: "The LM value of the entire city G is defined as the mean living metric value across all roads in G."
- p. 9: "In this case, a β value of 5 was selected."
- p. 9: "The mean circuity, c, across all possible node pairs in a network can be used to quantify how efficient a road network is."
- p. 9: "Whilst we could analyse larger cities, hardware limitations make generating this quantity of larger cities slow."
- p. 16: "Another benefit of the “Organic” grammar which is not represented by any of the 3 metrics is its ability to easily conform to local constraints."
- p. 16: "As a result, a constraint such as a lake or river, or a particularly steep area of terrain, can be easily built around."
- p. 16: "Recall we have cut the development short due to the computational expense of the city generation and analysis."
- p. 17: "Firstly, no metric can universally capture the “quality of life” of a city."
- p. 18: "The results suggest that there is no single optimal city structure."
- SI p. 7: "When the generateCity function is called, a single node with a roadType of m and a nodeType of start is created at position (0, 0)."
- SI p. 8: "The default value of this radius is 0.8 arbitrary units."
- SI p. 9: "Note that as the network becomes large, the sheer quantity of these distance calculations makes performing this check slow."
- EPJ abstract (via OpenAlex): "We introduce a novel “living metric” to quantify the degree of hierarchical living structure (a property associated with organically developed cities) in a city’s road network, and validate it by applying it to six real cities."
- EPJ abstract (via OpenAlex): "We then demonstrate that different urban morphologies perform better at different metrics, highlighting that optimising a road network is inherently metric-dependent and therefore cannot converge on the “optimal” urban form."

## 8. Comparability with UrbanMorph

- **Code.** https://github.com/camdobbie/CitySprout (Python).
  - **No licence file** (GitHub API `license: null`), so all rights are reserved by default and we need the authors' permission to use or modify it.
  - Created 2024-04-15; last push 2026-02-27.
  - Dependencies: networkx, matplotlib, scipy, numpy, shapely, tqdm, pandas, seaborn. `realCities.py` also imports osmnx.
- **macOS without CUDA.** Pure Python, CPU only. Expected to run; not tested.
- **Inputs we could supply:** none natively. Supplying them would mean adding:
  - a metric scale (it uses arbitrary units);
  - start anchors instead of the single (0, 0) seed;
  - boundary clipping;
  - water and slope rejection inside Checks 1/2 or the rule selection.
  All of these would be our design choices.
- **Fair comparison.** Unconditional grammar growth (morphology realism only). Not a completion or terrain baseline. Its grammars are stylised archetypes, not fitted to real cities, so its realism will be poor by design.
- **Retraining:** n/a (no learning). Tuning the grammar parameters to our cities would also be our choice.
- **Effort (estimate):**
  - as-is: **1–2 h**;
  - adapted (anchors, boundary, water/slope checks, metres, class map, export to our metrics): **10–16 h** (consistent with `docs/M18_BASELINES.md`);
  - plus a licence request.

## 9. Ideas to adopt

- **Living metric as a hierarchy-consistency statistic.**
  - Compute LM on real and generated 512 m tiles: do roads of similar betweenness connect to each other?
  - It needs "roads", meaning grouped edges. We would use OSM way or name grouping for real tiles and continuity strokes (e.g. COINS) for generated ones. That grouping choice must be fixed and reported.
  - Fix the off-by-one window and handle ties properly.
- **Mean circuity with an explicit rule for unreachable pairs.**
  - We already flagged this in `docs/M18_LITERATURE.md`. For us: report circuity only over same-component pairs, plus the share of unreachable pairs as a separate number.
  - Circuity is also our natural terrain-adaptation partner: Boeing 2019 found hilly cities more circuitous.
- **Alpha-shape area** for density on irregular, water-cut tiles, instead of the tile area. Relevant where the buildable land on a 512 m tile is small.
- **Metric-dependent trade-off framing.** Present realism vs terrain adaptation as a Pareto set across metrics rather than a single "best". The EPJ abstract makes this argument explicitly.
- **Validate a new metric on real cities first** (EPJ). Before using our terrain-adaptiveness scores on generated networks, show that they separate real hill towns from flat towns.
