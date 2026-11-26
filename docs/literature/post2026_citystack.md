# Post, Ledoux, Patil 2026 — CityStack

## 1. Header

- **Citation:** Oliver J. Post, Hugo Ledoux, Akshay Patil. "A proposed framework for generating plausible cities." *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences* XII-4/W1-2026, pp. 277–284. 21st International 3D GeoInfo Conference, Sofia, 28–29 Sep 2026. Published 2026-09-28 (Crossref). Double-blind peer reviewed on the full paper (page footer). Licence: CC BY 4.0.
  - "CityStack" is the framework name, not the title.
  - The paper is based on Post's TU Delft MSc thesis (2024).
- **DOI:** 10.5194/isprs-annals-XII-4-W1-2026-277-2026.
- **URLs opened:**
  - https://isprs-annals.copernicus.org/articles/XII-4-W1-2026/277/2026/
  - https://isprs-annals.copernicus.org/articles/XII-4-W1-2026/277/2026/isprs-annals-XII-4-W1-2026-277-2026.pdf (read in full)
  - https://api.crossref.org/works?query.bibliographic=A+proposed+framework+for+generating+plausible+cities…
  - https://api.github.com/repos/OliverJPost/CityStack (plus `/contents/`, `/commits`)
  - https://raw.githubusercontent.com/OliverJPost/CityStack/main/README.md and `.gitmodules`
  - https://api.github.com/repos/OliverJPost/citypy and `/contents/`
  - https://api.github.com/repos/OliverJPost/citystackgen (plus `/contents/`, `/git/trees/master?recursive=1`)
  - https://raw.githubusercontent.com/OliverJPost/citystackgen/master/README.md and `Cargo.toml`
  - `src/streets/builder.rs`, `src/streets/streets_experiment.rs`, `src/streets/layer.rs` and `src/main.rs` (raw, keyword search only)
- **Access level:** **FT**.

## 2. BibTeX

```bibtex
@article{post2026citystack,
  title   = {A proposed framework for generating plausible cities},
  author  = {Post, Oliver J. and Ledoux, Hugo and Patil, Akshay},
  journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume  = {XII-4/W1-2026},
  pages   = {277--284},
  year    = {2026},
  doi     = {10.5194/isprs-annals-XII-4-W1-2026-277-2026}
}
```

## 3. What it does

CityStack describes a city as a stack of layers, from natural context at the bottom to points of interest at the top. Constraints flow upward only. The paper implements two layers: minor roads and building footprints.

It works in three steps:
1. Analyse 43 cities with morphological metrics, giving 13 road typologies and 9 building typologies.
2. Encode each city as a 100 m "typology grid".
3. Generate new cities: simulated annealing produces a new typology grid, then Parish–Müller-style growth samples segment length, angle and branching from the local typology statistics.

## 4. Learned or procedural, and how

**Statistics-driven procedural generation. Not learned in the deep-learning sense.** What is fitted to data:
- **Road typologies.**
  - A Gaussian Mixture Model on per-segment morphological metrics plus weighted neighbourhood aggregates, across 43 cities.
  - K = 13 was chosen at the elbow of the BIC curve.
  - New metrics: continuity, stretch curvilinearity, right-neighbour distance and angle deviation.
- **Building typologies.** A supervised gradient-boosting classifier (cited as Dorogush et al. 2018, i.e. CatBoost; inferred from the citation). It is trained on 7 ground-truth cities and applied to the rest, with visual checks.
- **Typology templates.** Per-typology distributions of segment attributes (e.g. intersection angles, segment lengths, curvature) at cell, area and city level.
- **Typology-grid synthesis.** Simulated annealing that swaps cells to match landscape statistics (patch area, core area index, shape index) binned by distance to the city centre.
- **Road growth.** An adapted L-system (Parish & Müller 2001): segment by segment, with parameters sampled from the local cell's template. Only minor roads are generated; major roads are future work.

**Data and representation:**
- **Data:** OSM roads simplified into 4 levels (highway, major road, street, path); Overture building footprints; GHSL Built-H heights (100 m).
- **Tile size:** whole cities; the grid cell is 100 m.
- **Output:** vector road segments (Rust `citystackgen`), polygonised into blocks for buildings. The typology grid is a raster, but the roads are generated as vectors, so no pixel-to-graph step is needed.

## 5. Inputs / outputs and terrain

- **Inputs:** a city's typology template (`.npz` grid plus per-cluster `.json` from `citypy`), or a new annealed grid.
- **Outputs:** minor-road network and building footprints, both 2D. LoD1/2 extrusion is future work.
- **Terrain and water: not implemented.**
  - The paper says roads are "additionally constrained by terrain, water bodies, and previously generated segments", "but this is future work as it is not yet implemented" (p. 281).
  - The natural-context layer is "already present in the dataset for future use" (p. 283).
  - A keyword search of the citystackgen street sources for terrain, water, elevation, slope and height found no terrain handling (only commented-out bounding-box and image-size code).

## 6. Evaluation

- **What is evaluated:** the interpretability and transferability of the typologies, the comparability of typology grids, and whether annealing preserves typology distributions and landscape statistics.
- **Quantitative result:** Breda at 300 m cells. Annealing converges after about 1.3 M iterations at 95% similarity to the input grid, "not shown here". The similarity measure is the annealing objective over landscape statistics.
- **Not evaluated:** generated road geometry against real cities. This is stated explicitly (quote below). There is no train/test or geographic hold-out for generation, no baselines, no statistics or uncertainty, and no user study.
- **Runtime:** road growth produces "300,000 road segments in seconds". Analysis of Paris needed up to 100 GB RAM.

## 7. Verbatim quotes (journal page numbers)

- p. 277: "In this paper, plausibility means morphological consistency with observed urban patterns: a generated city should preserve characteristic typology distributions and spatial organisation without reproducing the source city exactly."
- p. 279: "Continuity measures how far one can travel in a straight direction without a significant turn, defined as the ratio of a road stretch length to the maximum stretch length (500 m)."
- p. 279: "The optimal number of clusters was determined at 13 by the elbow of the Bayesian Information Criterion curve."
- p. 281: "The algorithm initialises a grid with the same typology distribution as the template and iteratively proposes random swaps of two cells with different typology values."
- p. 281: "The objective function measures the similarity between the generated grid and the template using patch area, core area index, and shape index from landscape statistics (McGarigal and Marks, 1995), binned by distance to the city centre to preserve the characteristic centre-to-periphery gradient found in most cities."
- p. 281: "Using the city of Breda (the Netherlands) at one-ninth resolution (300 m cells), the annealing process converges after approximately 1.3 million iterations (not shown here), reaching 95% similarity to the input typology grid."
- p. 281: "At the typology-grid level, this preservation of typology distributions and distance-binned landscape statistics is our operational criterion for plausibility."
- p. 281: "We adapt the L-system approach of Parish and Müller (2001) to use local typology statistics as input parameters, replacing the manually crafted image maps of the original method."
- p. 281: "The algorithm grows the road network segment by segment; at each step, the length, angle, and branching probability of the next segment are sampled from the typology template statistics of the local typology grid cell."
- p. 281: "The road network is additionally constrained by terrain, water bodies, and previously generated segments (but this is future work as it is not yet implemented)."
- p. 281: "The approach is computationally efficient, generating 300,000 road segments in seconds."
- p. 282: "We do not yet provide a full task-based evaluation, nor a quantitative comparison between final generated geometries and real cities across all morphology measures."
- p. 282: "OpenStreetMap road data and derived urban attributes remain spatially uneven in completeness and semantic consistency (Biljecki et al., 2023)"
- p. 282: "The current analysis pipeline processes large cities largely monolithically; for Paris, this required up to 100 GB of RAM."
- p. 283: "The natural context layer (terrain and water), although not used for analysis in this work, is already present in the dataset for future use."

## 8. Comparability with UrbanMorph

- **Code and licences** (GitHub API, checked 2026-11-25):
  - **Umbrella repo** https://github.com/OliverJPost/CityStack: no licence; last push 2025-04-27. It links the thesis at https://resolver.tudelft.nl/uuid:d6dc629c-c75e-4578-b1a1-8ac741133751 (not opened).
  - **`citypy`** https://github.com/OliverJPost/citypy (Python, Poetry): **GPL-3.0**; last push 2025-04-25. It ships `citypy_gmm13_road_clusters.gmm`, apparently the fitted 13-cluster GMM (inferred from the filename).
  - **`citystackgen`** https://github.com/OliverJPost/citystackgen (Rust, Rerun 0.17 GUI, GEOS bindings): **GPL-3.0**; last push 2024-11-18. It includes an example template (`example/Pijnacker_NL.npz` + 13 `cluster_*.json`).
  - **Version risk:** the code predates the paper (Sep 2026), so whether it matches the paper's version is not verified.
  - **Weights:** n/a, apart from the fitted GMM.
- **macOS without CUDA.** Python + Rust, no GPU. Probably runnable on Apple Silicon. Not tested; the GEOS C library and the Rerun viewer are the likely friction.
- **Inputs we could supply.**
  - Typology templates for our cities, by running `citypy` on OSM and Overture.
  - A typology grid over our 512 m tile (about 5×5 cells).
  - **Not supported:** DEM, water, anchors or boundary streets. They would have to be added in the growth loop, and that design would be ours.
- **Fair comparison.** Unconditional statistics-driven growth (morphology realism only). Comparing terrain adaptation would only show that it ignores terrain, unless we add constraints, which turns it into our own variant.
- **Retraining:** no deep training. Re-run the analysis to get templates for our reference cities.
- **Effort (estimate):**
  - as-is on the bundled example: **6–12 h**;
  - adapted to our tiles, metrics, anchors and water mask: **20–30 h** (consistent with `docs/M18_BASELINES.md`).
  - GPL-3.0 is fine for running as a baseline. Code we derive from it and distribute would inherit GPL.

## 9. Ideas to adopt

- **Typology-distribution realism.**
  - Classify generated and real 512 m tiles with the published 13-cluster road-typology GMM, or our own refit.
  - Compare typology histograms, e.g. by Jensen–Shannon distance.
  - This gives an interpretable, named-morphology realism score ("our hill tiles become 'curvy roads' c7 / 'spacious winding' c9 as real hill towns do").
- **Segment metrics** for our statistic set: continuity (stretch length / 500 m), stretch curvilinearity, right-neighbour distance and angle deviation. They target contour-parallel streets, which are directly relevant to contour following.
- **Landscape statistics binned by distance from a centre**, extended to bins by slope class or elevation band: realism measured conditionally on terrain strata.
- **Honest scoping of claims.** They state exactly what is and is not evaluated (p. 282). A model for our own limitations section.
