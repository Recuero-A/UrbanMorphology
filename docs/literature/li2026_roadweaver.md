# Li et al. 2026 — RoadWeaver

## 1. Header

- **Citation:** Yueyuan Li, Zexi Chen, Weijie Xi, Mingyang Jiang, Songan Zhang, Hanyang Zhuang, Ming Yang. "RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation." arXiv:2608.11580v1 [cs.RO], submitted 12 Aug 2026. 8 pages, 6 figures, 2 tables. Peer-reviewed venue: none stated (not verified).
- **DOI:** 10.48550/arXiv.2608.11580 (arXiv-issued, shown on the abstract page).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/abs/2608.11580
  - https://arxiv.org/html/2608.11580 (full text, read in full)
  - https://arxiv.org/pdf/2608.11580 (used for page numbers)
  - https://api.github.com/search/repositories?q=RoadWeaver
  - https://api.github.com/repos/bitxige/tactics2d-roadweaver (plus `/contents/`, `/contents/baselines`, `/contents/src/network_generator`, `/contents/src/network_generator/backbone`, `/contents/eval`, `/contents/eval/metrics`, `/contents/scripts`, `/contents/config`, `/commits`)
  - https://raw.githubusercontent.com/bitxige/tactics2d-roadweaver/main/README.md, `.gitmodules`, `environment.yml`, `config/pipeline_config.yaml`, `eval/metrics/topology.py`, `eval/metrics/route.py`, `eval/metrics/reference.py`, `eval/metrics/__init__.py`, `eval/roadweaver.py`
  - https://api.github.com/repos/WoodOxen/tactics2d, https://api.github.com/repos/WoodOxen/tactics2d/contributors, https://api.github.com/users/SCP-CN-001
- **Access level:** **FT** (full text, arXiv v1). I also read parts of a public development repository; it is not linked from the paper (section 8).

## 2. BibTeX

```bibtex
@misc{li2026roadweaver,
  title         = {RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation},
  author        = {Li, Yueyuan and Chen, Zexi and Xi, Weijie and Jiang, Mingyang and Zhang, Songan and Zhuang, Hanyang and Yang, Ming},
  year          = {2026},
  eprint        = {2608.11580},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2608.11580}
}
```

## 3. What it does

RoadWeaver generates lane-level HD maps for driving simulators from scratch, working coarse to fine:
1. A learned model samples a sparse global road skeleton, conditioned on a style code and structural priors.
2. Procedural tensor-field growth adds branch and local roads.
3. A*-based reconnection and largest-component filtering repair the topology.
4. A rule-based stage builds lanes and junction connectors and exports OSM / OpenDRIVE for Tactics2D.

The target use is closed-loop evaluation of autonomous driving, not urban morphology.

## 4. Learned or procedural, and how

**Hybrid. The skeleton is learned; the local roads, repair and lanes are procedural.**

- **Skeleton representation (raster).** Each OSM road graph is rasterised into a 6-channel "road field" F ∈ R^{H×W×6}: road probability, two orientation channels, junction heatmap, endpoint heatmap, distance field (§III-B). H and W are not stated (not verified).
- **Skeleton model.**
  - A VQ-VAE (trained with reconstruction, codebook and commitment losses) encodes F into a grid of discrete tokens.
  - A **conditional masked Transformer** (MaskGIT-style, AdaLN conditioning) models the token distribution given c ∈ R^11: a 6-d road-style code plus road density, gridness, radialness, organicness and bearing entropy (§III-B).
  - The authors deliberately chose masked (bidirectional, parallel) decoding over autoregressive generation.
- **Raster to vector.** "Road centerlines are then extracted from the decoded field and converted into a vectorized road skeleton through graph reconstruction" (§III-B). The exact extraction method is not given; skeletonisation is not named (not verified). The development repo's config has `simplify_chains` (Douglas–Peucker) and `merge_close_nodes` parameters for this step (repo, not paper).
- **Local roads (vector, procedural).**
  - A structure tensor field is built from skeleton tangents with a Gaussian kernel (Eq. 1–2).
  - Growth fronts are seeded on both sides of the skeleton. Each step moves Δs along the tensor eigen-direction closest to the current heading, blended with inertia, clamped to θ_max and perturbed by a style-dependent Δφ (Eq. 3–5).
  - A front stops at the map boundary, at a maximum length or iteration count, or when it comes within the snapping radius of an existing road (it is then attached).
- **Repair (procedural, post hoc).**
  - Dangling endpoints are reconnected by A* over a cost map from the decoded field.
  - Residual fragments are removed by largest-connected-component filtering.
  - Empty regions are then infilled with more tensor-field roads.
- **Training data.** "approximately 58,000 samples from 144 cities" of OSM road networks. Each sample is "approximately 2 km × 2 km" (§IV-A1). Which cities, the train/test split and the OSM road classes used are not stated (not verified).
- **Final output:** vector road graph, then lane-level HD map (Algorithm 1: refine graph, assign lane configs, boundaries, junction connectors, successor links, geometry repair, topology repair).

## 5. Inputs / outputs and terrain

- **Inputs.** The 11-d condition vector c (style code + 5 structural priors), a random seed, and the map extent (a larger area at fixed node density gives larger maps). There is no boundary polygon, surrounding street context or junction input.
- **Outputs.** Vector road graph, lane-level HD map, OSM and OpenDRIVE files.
- **Terrain: none.**
  - The HTML full text contains no "terrain", "elevation", "slope" or "DEM" (keyword search).
  - "water" also does not appear.
  - Generation is purely 2D.

## 6. Evaluation

- **Datasets.** Training on the OSM set above. Evaluation is on generated maps only. There is no comparison against held-out real maps, so there is **no geographic hold-out or realism-to-real metric**.
- **Scale protocol.** All methods are set to produce about 35–40 graph nodes (intersections, endpoints and turning points count as nodes) for the qualitative and Table II comparison (§IV-B, Fig. 3).
- **Metrics (§IV-C; definitions verbatim in section 7):**
  - **LCC ratio:** share of nodes in the largest connected component.
  - **Reachability:** fraction of node pairs connected by a path.
  - **Dead-end ratio:** share of terminal nodes.
  - **Cycle ratio:** share of nodes on at least one cycle.
  - **Endpoint alignment error:** mean Euclidean gap between lane endpoints that should connect.
  - The development repo implements these as follows (repo code, not the paper):
    - dead-end = degree-1 nodes / all nodes;
    - cycle ratio = nodes in the 2-core / all nodes;
    - reachability: both an exact all-pairs version and a 100-random-OD-pair version exist. Which one produced Table II is not verified.
- **Controllability.** Target vs measured road-node density per km² at targets 10–40 (Fig. 4). Measured density tracks the target. Above about 40 nodes/km² the response saturates.
- **Results (Table II, mean ± sd; number of maps not stated, not verified):**

| Method | LCC % | Reachability % | Dead-end % | Cycle % | Endpoint err (m) |
|---|---|---|---|---|---|
| MetaDrive | 99.5 ± 2.3 | 99.3 ± 3.3 | 43.0 ± 10.1 | 50.5 ± 13.8 | 8.17 ± 2.43 |
| RoadGen | 100.0 ± 0.0 | 100.0 ± 0.0 | 12.8 ± 6.2 | 0.0 ± 0.0 | 4.80 ± 1.29 |
| HDMapGen | 64.3 ± 24.7 | 56.2 ± 28.2 | 42.9 ± 29.4 | 46.5 ± 37.2 | 4.32 ± 0.41 |
| RoadWeaver | 99.9 ± 0.3 | 99.8 ± 0.7 | 10.7 ± 3.6 | 85.2 ± 6.0 | 0.24 ± 0.22 |

- **Baselines.**
  - MetaDrive and RoadGen (rule-based); HDMapGen (data-driven, local lane-level).
  - The paper does not say how the baselines were run or trained. The development repo has baseline submodules: `metadriverse/metadrive`, `agshar96/HDMapGen` (an unofficial reimplementation), `roadgen/RoadGen`, and also `autonomousvision/sledge`, which is not in the paper. From this I infer, but have not verified, that HDMapGen was the unofficial reimplementation.
- **Statistics.** Mean ± sd only. No significance tests and no stated sample count.
- **Runtime.**
  - Complete maps in about 1.39–3.50 s on an RTX 5090 + Ryzen 7 9800X3D, excluding model loading.
  - RoadGen took about 50–370 s, and its valid maps "rarely extend beyond roughly 60 nodes".
  - CPU-seconds comparison in Fig. 5.
- **Simulator test.** 100 generated maps were all imported into Tactics2D (100%). 1,000 random routing tasks: 98.7% routing success, 0.82 ± 0.17 s per route.
- **Not done.** No FID, no MMD/Fréchet comparison with real networks, no user study, no comparison of the generated morphology with real cities.

## 7. Verbatim quotes

Section numbers are from the HTML; page numbers are from the arXiv v1 PDF.

- Abstract (p. 1): "The training code and an out-of-the-box implementation of RoadWeaver will be released upon acceptance."
- §II (p. 2): "However, existing methods are mainly designed for local map synthesis or bounded map patches."
- §III-B (p. 3): "Its six channels consist of one road probability channel, two orientation channels, one junction heatmap, one endpoint heatmap, and one distance field."
- §III-B (p. 3): "Masked token prediction can exploit bidirectional context and refine multiple regions in parallel, enabling more globally coherent layouts and more efficient sampling than autoregressive generation."
- §III-B (p. 3): "Road centerlines are then extracted from the decoded field and converted into a vectorized road skeleton through graph reconstruction."
- §III-C (p. 3): "We expand the generated skeleton using a procedural road-growth strategy [30]." ([30] = Parish & Müller 2001)
- §III-C (p. 4): "The propagation of an active front terminates when it leaves the map boundary, reaches the prescribed maximum length or iteration limit, or approaches an existing road within the snapping radius."
- §III-C (p. 4): "After procedural growth, dangling endpoints are reconnected to nearby roads through A* search [32] over a cost map derived from the decoded road field"
- §III-C (p. 4): "Regions with stronger predicted road evidence are assigned lower traversal costs, encouraging the reconnection paths to remain consistent with the spatial evidence predicted by the generative model."
- §III-C (p. 4): "Residual disconnected fragments are subsequently removed through largest-connected-component filtering."
- §IV-A1 (p. 4): "We train RoadWeaver on real-world road networks collected from OSM, comprising approximately 58,000 samples from 144 cities."
- §IV-A2 (p. 4): "MetaDrive [15] and RoadGen [23] are adopted as rule-based baselines, while HDMapGen [16] is included as a data-driven baseline for local lane-level map generation."
- §IV-B (p. 5): "All methods are configured to generate road graphs containing approximately 35–40 nodes."
- §IV-B (p. 5): "When the target density exceeds approximately 40 road nodes/km2, the density increase becomes less significant due to the increasing topological constraints of dense road networks."
- §IV-C (p. 5): "The LCC ratio denotes the proportion of nodes in the largest connected component [33]."
- §IV-C (p. 5): "Reachability is the fraction of node pairs connected by valid paths [33, 16, 23]." (The PDF orders the citation as [16, 23, 33].)
- §IV-C (p. 5): "The dead-end ratio quantifies the proportion of terminal nodes [33]."
- §IV-C (p. 5): "The cycle ratio measures the proportion of nodes belonging to at least one cycle [33]."
- §IV-C (p. 6): "It also obtains the lowest dead-end ratio among all compared methods, indicating fewer disconnected road extensions."
- §IV-D (p. 6): "Once the available connection queue is exhausted, the assembly process terminates early, so valid maps rarely extend beyond roughly 60 nodes." (about RoadGen)
- §V (p. 7): "Future work will focus on richer traffic semantics, roadside assets, and interactive map editing, and further improve the realism of generated driving environments."

## 8. Comparability with UrbanMorph

- **Official code and weights.** Not released: "will be released upon acceptance" (abstract).
- **Public development repository (found, not announced in the paper).** https://github.com/bitxige/tactics2d-roadweaver
  - **Licence and history:** GPL-3.0. Created 2026-08-01; last push 2026-08-08. Commits are by `SCP-CN-001`.
  - **Contents:**
    - `src/network_generator/backbone/` (VQ-VAE and masked transformer);
    - `growth/` and `topology/`;
    - training scripts (`scripts/train_vq_vae.py`, `train_transformer.py`);
    - baseline evaluators (`eval/*.py`);
    - metric code under the header "Copyright (C) 2026, Tactics2D Authors".
  - **Authorship (inferred, not verified):** probably the authors' own code, because `bitxige` and `SCP-CN-001` are both contributors to `WoodOxen/tactics2d`, the authors' simulator.
  - **No weights.** No checkpoints are in the tree (617 KB repo).
  - **Possibly differs from the paper:** the README still lists the "hierarchical road generator" as "future work", so it may lag or differ from the paper.
- **macOS without CUDA.**
  - The code picks `cuda` if available, else `cpu` (`eval/roadweaver.py`).
  - The environment is Python 3.9 + torch ≥ 2.0, and it also lists `tensorflow>=2.15` and GDAL.
  - Inference on CPU or MPS is plausible but untested.
  - Training the VQ-VAE and transformer on about 58k tiles without a GPU would be slow (not verified).
- **Inputs we could supply.**
  - The 11-d condition vector. We can compute density, gridness, radialness, organicness and bearing entropy from the real OSM tile, which gives an "oracle-statistics" condition. The 6-d style code comes from a CRHD style classifier in the repo; that link is inferred from the repo, not from the paper.
  - **Not supported:** DEM, water, boundary polygon, anchors or surrounding streets.
  - **Tile size mismatch:** RoadWeaver is trained on 2 km tiles and we use 512 m. The paper says the area is adjustable at fixed density.
- **Fair comparison type.** Unconditional (style- and density-conditioned) growth only. It is not a completion baseline and not terrain-aware.
  - Comparing its terrain adaptiveness to ours would measure only that it ignores terrain.
  - The fair axes are morphology realism and topology (dead ends, LCC, cycles) on flat tiles.
  - **Scale caveat:** its maps have about 10–40 road nodes/km² (Fig. 4), a skeleton-level density far sparser than full urban street networks. Node-level statistics would therefore not be comparable without matching the OSM road-class filter.
- **Retraining.** Needed: no weights. Optionally retrain on our 512 m tiles and cities.
- **Effort (estimate):**
  - as-is inference: impossible until weights exist;
  - retrain from the dev repo on our OSM tiles and wire into our metrics: **40–80 h** plus GPU time;
  - reimplement only its **post-hoc repair** (A* reconnection over a cost map + LCC filtering + short-dead-end pruning) on top of UrbanMorph raw samples, as an ablation baseline: **6–12 h**.

## 9. Ideas to adopt

- **Post-hoc repair as a competitor to our in-loop potentials.** RoadWeaver fixes connectivity after generation: A* reconnection, LCC filtering, and, per the repo config, `prune_dead_ends` up to 120 m and `snap_endpoints` 50 m.
  - Apply the same repair to UrbanMorph samples drawn without network potentials.
  - Compare with SMC/look-ahead at matched realism.
  - Report metrics both **before and after** repair. Repair improves LCC and dead-end ratio by construction, so the honest comparison is the realism cost of the repair.
- **Topology metric set with explicit definitions.**
  - LCC ratio; all-pairs reachability (exact, not sampled, on small graphs); dead-end ratio as degree-1 share; cycle ratio as the 2-core share.
  - Compute them on a degree-2-contracted graph, as their repo does.
  - State the contraction rule, because these ratios change a lot with it.
- **Controllability plot.** Target vs measured statistic (their Fig. 4). For us: the potential strength or target grade versus the measured grade distribution, or the target dead-end rate versus the achieved one.
- **Downstream routing check.** 100 maps × 10 random OD routes, reporting routing success rate. We could report OD reachability between same-land nodes in the same way.
- **Mean ± sd across seeds.** Our protocol should go further, with CIs and sample counts, which RoadWeaver does not report.
