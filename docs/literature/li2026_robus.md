# Li et al. 2026: RoBus (dataset)

## 1. Header

- **Journal citation:** Tao Li, Ruihang Li, Huangnan Zheng, Heng Chen, Kehan Wang, Wangliang Guo, Hong Li, Shijian Li, Zhijie Pan. "RoBus: A Multimodal Dataset for Controllable Road Networks and Building Layout Generation." *Computers* (MDPI) 15(8):488, published 2026-07-30. Licence CC BY 4.0 (Crossref).
- **Preprint:** Tao Li, Ruihang Li, Huangnan Zheng, Shanding Ye, Shijian Li, Zhijie Pan. "RoBus: A Multimodal Dataset for Controllable Road Networks and Building Layouts Generation." arXiv:2407.07835. **Only v1 exists** (10 Jul 2024), marked "Preprint. Under review".
- **DOI:** 10.3390/computers15080488 (journal). OpenAlex also lists an SSRN record, 10.2139/ssrn.4917011 (not opened).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/abs/2407.07835 (version history)
  - https://arxiv.org/pdf/2407.07835 (full text v1, 18 pages including appendix)
  - https://www.mdpi.com/2073-431X/15/8/488, `/htm` and `/pdf`: **all HTTP 403** (curl and WebFetch)
  - https://api.crossref.org/works/10.3390/computers15080488 (journal metadata and abstract)
  - https://api.openalex.org/works/doi:10.3390/computers15080488
  - https://api.github.com/repos/tourlics/RoBus_Dataset and its git tree; https://raw.githubusercontent.com/tourlics/RoBus_Dataset/main/README.md
  - raw `data_preperation/main_gen_road_density.py` and `data_preperation/batch_align_crop.py`
- **Access level:**
  - **FT** for arXiv v1 (all sections and appendix read);
  - **AO** for the 2026 journal version (Crossref abstract only; MDPI blocked);
  - the repository README and two scripts were read.

  Facts below come from arXiv v1 unless they are marked "journal abstract".

## 2. BibTeX

```bibtex
@article{li2026robus,
  author  = {Li, Tao and Li, Ruihang and Zheng, Huangnan and Chen, Heng and Wang, Kehan and Guo, Wangliang and Li, Hong and Li, Shijian and Pan, Zhijie},
  title   = {{RoBus}: A Multimodal Dataset for Controllable Road Networks and Building Layout Generation},
  journal = {Computers},
  volume  = {15},
  number  = {8},
  pages   = {488},
  year    = {2026},
  doi     = {10.3390/computers15080488}
}
@misc{li2024robus_arxiv,
  author = {Li, Tao and Li, Ruihang and Zheng, Huangnan and Ye, Shanding and Li, Shijian and Pan, Zhijie},
  title  = {{RoBus}: A Multimodal Dataset for Controllable Road Networks and Building Layouts Generation},
  year   = {2024},
  eprint = {2407.07835},
  archivePrefix = {arXiv}
}
```

## 3. What it does

RoBus is a multimodal dataset for generating road networks and building layouts. Its **72,400 paired tiles cover about 80,000 km²** in Australia, China, Europe and the US. It also defines a benchmark (quality, diversity, validity, "urban properties") and baselines. The road baselines are raster:
- pix2pix from land-use maps (Task I);
- a pix2pix variant conditioned on road attributes with a differentiable centreline-Dice topology loss (Task III);
- ControlNet, added in the journal version (journal abstract).

## 4. Learned or procedural, and how

- **The dataset is built procedurally from OSM.** Baselines are learned.
- **Raster stack:**
  - 6-channel GeoTIFF: Road-P, Road-S, water, green, buildings with height, density;
  - **EPSG:3857 at 5 m/px**; tiles **256 × 256 px (about 1.6 km² by the paper's figure)** with 20 % overlap (§3.2).
  - The repository's current `batch_align_crop.py` uses `target_size = 512` and `overlap = 0.3`, and the README says "e.g. 512×512". The released tiling may therefore differ from the paper (not verified).
- **Road classes** (Appendix B):
  - Road-P = motorway, trunk, primary;
  - Road-S = secondary, tertiary, residential, unclassified.
  - Residential streets count as "secondary". Service roads, footways and living streets are not listed.
- **Graph extraction:**
  1. merge Road-P and Road-S;
  2. skeletonise by morphological thinning;
  3. simplify degree-2 nodes with the cosine-smoothness threshold C_tr (Eq. 1) and merge close nodes;
  4. export to NetworkX gpickle.
  - Repository helpers: `trace_skeleton.py` (skeleton-tracing), modified `sknw`.
  - **The vector graphs are derived from rasters, not from the OSM vectors directly.**
- **Blocks:** a "geometric minimal cycle" search with path cutoff 12 (Appendix C, Alg. 1).
- **Per-tile labels:** road density, orientation entropy (Boeing 2019), traffic convenience, junction density, building density and height; text prompts come from templates.
- **Baseline models:**
  - Tasks I/II: pix2pix;
  - Task III: pix2pix with attributes in the U-Net bottleneck, loss λ1·L1 + λ2·GAN + λ3·clDice (Eq. 2);
  - Task IV: cVAE with a GAT, following GlobalMapper, plus building height and density attributes.
- **Size by region** (Table 5):

  | Region | Tiles | Road (km) | Area (km²) |
  |---|---|---|---|
  | Australia | 4,101 | 39,771 | 4,493 |
  | China | 19,243 | 123,227 | 21,211 |
  | Europe | 11,633 | 105,450 | 12,960 |
  | US | 37,429 | 357,496 | 41,278 |

  The list of cities is not given in the text (Melbourne appears in Fig. 4a).

## 5. Inputs/outputs and terrain

- **Channels:** roads (2 classes), water (OSM water, natural=water/wetland/glacier, waterways), greenery (forest, farmland, meadow, wood, …), buildings with height (Microsoft footprints; CNBH for China), and density.
- **The "density" channel is a smoothed road-density heatmap computed from the roads themselves** (`main_gen_road_density.py`: `estimate_density_map(..., sigma=300)` and `cal_geojson_to_density_map`). The paper cites Birsak et al. [1] for it.
  - Task III conditions on "local density maps", which are therefore derived from the target roads (our reading of the code).
- **Terrain:** **no DEM, elevation or slope channel** in the paper or the pipeline. Water enters as a raster channel.
  - The tiles keep their CRS, so an external DEM could be co-registered (our addition).
  - The paper still says that Task I "learned the pattern that roads should not cross mountains" (§5.2), with no elevation input. Presumably this is learned through the green or land-use channel; that is not verified.

## 6. Evaluation

- **Split:** "training, validation, and testing sets with ratios of 8:1:1" (§5.2).
  - The split is random, with **no geographic hold-out described**.
  - Because tiles overlap by 20 %, a random split will place overlapping tiles in both train and test (our inference; not discussed in the paper).
- **Metrics** (§5.1):
  - **Quality:** FID on InceptionV3 features of images. Not domain-adapted; the repository's `inception_download.py` fetches the standard checkpoint.
  - **Diversity:** for roads, Chamfer distance "against all paired road graphs" (citing NTG; higher = more diverse). For buildings, mIoU vs ground truth (lower = more diverse).
  - **Validity:** percentage of samples outside the geometric constraint (Fig. 7).
  - **Traffic convenience:** mean of d_E / d_S (Euclidean / Dijkstra) over all node pairs **more than 300 m apart**. Unreachable pairs are not addressed.
  - **Orientation:** street-bearing entropy (Boeing 2019); lower = more grid-like.
  - **Buildings:** Wasserstein distance for height, count and density.
- **Results, arXiv v1 Table 3:**

  | Variant | FID | DIV | Orientation | Convenience |
  |---|---|---|---|---|
  | Task I (pix2pix) | 27.03 | 22.50 | 3.27 | 0.75 |
  | Task III, λ3 = 0 | 25.83 | 19.98 | 3.25 | 0.77 |
  | Task III, λ3 = 0.3 | 26.98 | 19.05 | 3.27 | **0.83** |
  | RES_5m | 28.97 | — | — | — |
  | RES_10m | 35.28 | — | — | — |
  | US_1w | 21.36 | — | — | — |
  | Global_1w | 26.39 | — | — | — |

  - **Journal abstract:** ControlNet has the lowest road FID (20.78), and "our topology-aware road baseline obtains the highest traffic-convenience score (0.83) at the cost of lower fidelity and diversity".
- **Uncertainty and runtime:** no statistics, confidence intervals or seeds reported; no runtime reported.
- **Application:** generated graphs are exported to OpenDRIVE/SUMO/CARLA (`csv2xodr/`).

## 7. Verbatim quotes (arXiv v1 unless stated)

- p.5, §3.2: "We choose the PseudoMercator projection (EPSG:3857) as coordinate reference system (CRS) to these tif images, with a resolution of 5 meters per pixel." … "we crop the tif images into small tiles with a 20% overlap, each sized at 256 × 256 pixels, corresponding to an actual area of approximately 1.6km2."
- p.6, §3.2 Step 3: "we combine the first two channels of the images, which represent primary and secondary road networks, and then skeletonize the result using morphological thinning methods."
- p.5, §3.2 Stage 1: "To clean the road network data, we rasterize roads into images and then apply thinning to modify the roads, as suggested by [40]."
- p.10, §5.1: "Traffic Convenience is the average value of dE (vi , vj )/dS (vi , vj ) for all node pairs vi , vj that are over 300m, where dE and dS are euclidean distance and Dijkstra shortest distance respectively"
- p.10, §5.1: "Orientation measures the entropy of street bearings [2]. Lower orientation indicates more ordered (such as grid-like) road networks."
- p.10, §5.2: "We divided the dataset into training, validation, and testing sets with ratios of 8:1:1, respectively."
- p.10, §5.2: "It is evident that the model used for Task I has successfully learned the pattern that roads should not cross mountains."
- p.3, §2.1: "Overall, Image-based approaches for road networks generation suffer from modeling topological features, while graph-based approaches struggle to encode spatial information."
- p.11, §5.3: "We conclude that the methods proposed in Task III greatly enhance connectivity by utilizing topological loss."
- Journal abstract (Crossref): "The experiments establish the dataset's usability while revealing complementary trade-offs rather than uniform superiority. ControlNet obtains the lowest road network FID (20.78), whereas our topology-aware road baseline obtains the highest traffic-convenience score (0.83) at the cost of lower fidelity and diversity."

## 8. Comparability with UrbanMorph (as a dataset or reference)

- **Code:** https://github.com/tourlics/RoBus_Dataset.
  - Python, **no licence file** (GitHub API `license: null`), so all rights are reserved; pushed 2026-09-07.
  - Contains the data-preparation pipeline, a copy of pytorch-CycleGAN-and-pix2pix (that folder keeps its own LICENSE), and `csv2xodr`.
  - **No trained baseline weights are listed.**
- **Data:**
  - Large TIFFs on **Baidu Pan** for Europe, US and Australia only; **China is not offered**.
  - Extraction codes require emailing the authors with your institution.
  - **No dataset licence is stated**, although it is derived from OSM (ODbL) and Microsoft building footprints (licence not re-verified here).
  - **Under our no-account, no-login constraint the data are effectively not obtainable.** Whether Baidu Pan downloads need an account was not tested.
- **macOS:** the pipeline scripts are plain Python (rasterio, networkx); pix2pix runs on CPU. Paths are hard-coded Windows-style (`roads\geojson`).
- **DEM:** absent. GeoTIFFs keep their CRS, so Copernicus or LiDAR DEMs could be added (our work).
- **Usefulness as a realism reference:** limited.
  1. Graphs are skeletonised from 5 m rasters, so junction positions and dead-end counts carry raster artefacts that our OSM-vector references do not.
  2. EPSG:3857 "5 m" pixels shrink on the ground by cos(latitude). This is our inference from the projection; the paper does not correct for it. Metric statistics (lengths, the 300 m threshold) are therefore latitude-biased unless reprojected.
  3. There is no terrain, so terrain-adaptiveness cannot be scored.
  - **We already build our references directly from OSM vectors on the same terrain, which is cleaner.**
- **Fair comparison:** not a generator to compare against. Its baselines are raster pix2pix/ControlNet conditioned on land use or road density, not on a DEM. A "pix2pix + DEM channel" baseline would be our own design (10–20 h, estimate, from M18_BASELINES).
- **Effort (estimates):**
  - Implementing RoBus's traffic-convenience (> 300 m) and bearing-entropy definitions on our graphs, for cross-paper comparability: **1–2 h**.
  - Rebuilding a RoBus-style tile set ourselves from OSM for extra cities: **6–10 h** (not needed for M18).

## 9. Ideas we could adopt

- **Report two network statistics with RoBus's exact definitions** (traffic convenience at > 300 m; orientation entropy) so that our numbers sit on a scale others use. Also state our handling of unreachable pairs, which RoBus omits.
- **Document the fidelity vs connectivity trade-off,** as their journal abstract does ("complementary trade-offs rather than uniform superiority"). This mirrors our realism vs dead-end/island potentials and supports reporting a Pareto front rather than a single winner.
- **Geographic hold-out and non-overlapping tiles.** RoBus's random 8:1:1 split on overlapping tiles is a leakage risk worth contrasting with our city-level hold-out.
- **Do not use road-derived density as a condition** (RoBus's density channel). Use only exogenous inputs (DEM, water, boundary) to keep conditioning leak-free; this strengthens our matched-input protocol.
- **Resolution ablation** (5 m vs 10 m: worse FID, higher diversity). An analogous DEM-resolution ablation (4 m LiDAR vs 30 m) would test our terrain-imprint concern.
