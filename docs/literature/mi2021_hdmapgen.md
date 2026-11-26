# Mi et al. 2021 — HDMapGen (RoadWeaver baseline, short note)

## 1. Header

- **Citation:** Lu Mi, Hang Zhao, Charlie Nash, Xiaohan Jin, Jiyang Gao, Chen Sun, Cordelia Schmid, Nir Shavit, Yuning Chai, Dragomir Anguelov. "HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps." *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4225–4234 per Crossref. RoadWeaver cites pp. 4227–4236, probably the CVF open-access pagination (not verified). arXiv:2106.14880v1, 28 Jun 2021 (Waymo, MIT, DeepMind, Google).
- **DOI:** 10.1109/CVPR46437.2021.00421 (Crossref); arXiv DOI 10.48550/arXiv.2106.14880.
- **Identified as RoadWeaver's baseline:** reference [16] in arXiv 2608.11580.
- **URLs opened:**
  - https://arxiv.org/abs/2106.14880
  - https://arxiv.org/pdf/2106.14880 (read in full, including the supplementary pages)
  - https://api.crossref.org/works?query.bibliographic=HDMapGen…
  - https://api.github.com/search/repositories?q=HDMapGen
  - https://raw.githubusercontent.com/agshar96/HDMapGen/main/README.md
  - https://api.github.com/repos/agshar96/HDMapGen
- **Access level:** **FT**.

## 2. BibTeX

```bibtex
@inproceedings{mi2021hdmapgen,
  title     = {HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps},
  author    = {Mi, Lu and Zhao, Hang and Nash, Charlie and Jin, Xiaohan and Gao, Jiyang and Sun, Chen and Schmid, Cordelia and Shavit, Nir and Chai, Yuning and Anguelov, Dragomir},
  booktitle = {2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4225--4234},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.00421}
}
```

## 3. What it does

HDMapGen generates lane-level HD map graphs, i.e. lane centrelines, autoregressively and coarse to fine:
- a global graph of key points (lane endpoints and intersections) with its adjacency;
- a local graph of control points per lane, which gives the curvature;
- a traffic-light attribute per lane.

It is compared with SketchRNN (sequence) and PlainGen (plain graph).

## 4. Learned or procedural, and how

- **Learned: an autoregressive graph generative model (GRAN-style), vector output.**
- **Global graph.** One new node per step. A 7-layer recurrent graph-attention GNN outputs:
  - a mixture of Bernoulli for its edges to earlier nodes;
  - a 2D GMM for its coordinates, with temperature τ;
  - in one of three orders: coordinate-first, topology-first or independent.
  - Training uses teacher forcing with NLL + BCE losses.
- **Local graph.** A single-shot MLP decodes a fixed-length W sequence of control-point coordinates and a validity mask per edge (MSE + BCE).
- **Training data:**
  - Argoverse Miami: 12,000 maps, 200 m × 200 m field of view;
  - Argoverse Pittsburgh: 5,000 maps, same field of view;
  - in-house San Francisco: 6,000 maps, 120 m × 120 m.
  - Preprocessing removes about 70% of redundant control points.
  - Node ordering is by DFS from a random start node.
- **Vector throughout.** No raster stage.

## 5. Inputs / outputs and terrain

- **Inputs:** none at sampling time, apart from the temperature τ and the random seed (unconditional per city model).
- **Output:** lane graph with coordinates, lane control points and traffic-light flags.
- **Terrain: none.** The PDF has no "terrain", "elevation" or "slope" (keyword search). There is no water or boundary.

## 6. Evaluation

- **Topology fidelity.** MMD with a Gaussian kernel using the first Wasserstein distance over degree distributions and Laplacian spectra (GraphRNN style).
- **Geometry fidelity.** Fréchet distance over lane length and orientation features.
- **"Urban planning" features (from NTG).** Connectivity (node degree), node density, reach, and convenience (Dijkstra shortest-path length for node pairs). Each is compared by Fréchet distance (Table 1, Argoverse).
- **Diversity.** Chamfer distance of global graphs, both vs ground truth (novelty) and among samples (internal diversity), across τ (Table 2, in-house data).
- **Train/test split:** not described beyond random sampling of training maps (not verified). No geographic hold-out reported.
- **Statistics:** single numbers; no confidence intervals.
- **Runtime:** 0.20 s per 200 m map (vs 0.89 s PlainGen and 2.28 s SketchRNN). Training on a single V100.
- **In RoadWeaver (about 35–40 nodes):**

| LCC % | Reachability % | Dead-end % | Cycle % | Endpoint error (m) |
|---|---|---|---|---|
| 64.3 | 56.2 | 42.9 | 46.5 | 4.32 |

  These are the worst connectivity numbers among the baselines.

## 7. Verbatim quotes (arXiv v1 PDF pages)

- p. 1: "Compared to HD maps, city layouts are not suitable for autonomous driving applications since they only contain coarse locations of the roads (with a resolution of roughly 10 meters) and lack details such as lanes of the roads or traffic lights."
- p. 5: "We randomly sample 12000 maps with a field-of-view (FoV) of 200m × 200m as the training dataset for Miami and 5000 maps with the same FoV for Pittsburg."
- p. 6: "While the diversity and realism trade-off still exist during generation, we find an empirical bound of 0.2 for τ to guarantee high-quality samples in our experiments."
- p. 6: "However, there are a large number of problematic cut-offs or dead-ends occurring in the generated lanes." (about SketchRNN)
- p. 6: "We use maximum mean discrepancy (MMD) [17] to quantify the similarity with real maps on graph statistics using Gaussian kernels with the first Wasserstein distance."
- p. 7: "We use the common urban planning features, connectivity to represent node degree, node density within a region, reach as the number of accessible lane within a region to evaluate the transportation plausibility, and convenience as the Dijkstra shortest path length for node pairs of our generated maps compared with real maps [10]."
- p. 11 (supplementary): "However, since the model has not been open-sourced yet, we are not able to perform a comparison with NTG on this high-definition map generation task."

## 8. Comparability with UrbanMorph

- **Official code:** none found; the GitHub search shows no Waymo or author repository.
- **Unofficial reimplementation:** https://github.com/agshar96/HDMapGen.
  - Licence: **MIT**. Last push 2024-07-23.
  - PyTorch 1.2.0, GRAN-based.
  - Weights for toy 5×5 grids and a nuPlan subset (README).
  - It is likely, but not verified, that RoadWeaver used it: their development repo has it as a submodule.
- **macOS.** Pure PyTorch, so CPU should work. PyTorch 1.2 is very old and pins would need updating (not tested).
- **Inputs we could supply:** none (no DEM, water, boundary or street context). It is unconditional per training set.
- **Fair comparison:** an unconditional learned vector-graph generator. It is the closest "learned graph AR" relative of UrbanMorph's base model, without terrain.
  - Requires retraining on our 512 m street graphs.
  - Lane-level design (FoV 120–200 m) does not match street-level 512 m tiles.
- **Effort (estimate):**
  - retrain the reimplementation on our OSM graphs and export samples: **20–40 h**;
  - weak as a baseline, because reviewers would see it as an older AR graph model with an unofficial implementation.

## 9. Ideas to adopt

- **MMD over degree distributions and Laplacian spectra** (GraphRNN-style, EMD-based Gaussian kernel). This is a standard graph-generation realism metric we can add next to our statistic distances.
- **Two-level diversity.**
  - Novelty: Chamfer distance to the nearest ground truth.
  - Internal diversity: among samples.
  - Report both across sampling temperature or potential strength, to show the realism–diversity trade-off.
- **Separate "topology" and "geometry" fidelity families.** Graph-only statistics vs length and orientation distributions. A clean way to structure our realism table.
