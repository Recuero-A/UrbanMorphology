# He & Aliaga 2023: GlobalMapper

## 1. Header

- **Citation:** Liu He, Daniel Aliaga (Purdue University). "GlobalMapper: Arbitrary-Shaped Urban Layout Generation." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 454–464. arXiv:2307.09693 (v1, 19 Jul 2023).
- **DOI:** 10.1109/ICCV51070.2023.00048 (Crossref bibliographic search; title and pages match).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/2307.09693 (full text v1, 11 pages)
  - https://api.github.com/repos/Arking1995/GlobalMapper and its git tree
  - https://raw.githubusercontent.com/Arking1995/GlobalMapper/HEAD/README.md, plus raw `model.py`, `train.py`, `test.py`
  - https://api.crossref.org/works?query.bibliographic=GlobalMapper+Arbitrary-Shaped+Urban+Layout+Generation
- **Access level:** **FT** (arXiv main paper). The supplement was not opened; it is linked from the README at https://openaccess.thecvf.com/content/ICCV2023/supplemental/He_GlobalMapper_Arbitrary-Shaped_Urban_ICCV_2023_supplemental.pdf.

## 2. BibTeX

```bibtex
@inproceedings{he2023globalmapper,
  author    = {He, Liu and Aliaga, Daniel},
  title     = {{GlobalMapper}: Arbitrary-Shaped Urban Layout Generation},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {454--464},
  year      = {2023},
  month     = oct,
  doi       = {10.1109/ICCV51070.2023.00048},
  eprint    = {2307.09693},
  archivePrefix = {arXiv}
}
```

## 3. What it does

GlobalMapper generates **building layouts inside city blocks of a given road network**. It does not generate roads. Each block's buildings become a "stubby" grid graph in a canonical frame, defined by the block skeleton's main axis. A conditional VAE with a graph attention network (GAT) learns the layouts, conditioned on a CNN encoding of the block-shape mask. It is trained on 28 North American cities and can also fill a city from a sparse 5 % prior of known blocks.

## 4. Learned or procedural, and how

- **Model:** a learned conditional VAE with a GAT encoder and decoder (T = 3 attention layers) and a 512-d latent.
  - The block mask (64×64 binary plus a scale channel) is encoded by a 4-layer CNN autoencoder (IoU > 0.98), and that code conditions the decoder.
- **Per building:** existence flag, position, oriented-box width and height, shape type (rectangle, L, U or X, fitted by Powell optimisation) and occupancy ratio. At most N = 120 buildings per block.
- **Canonical spatial transform:** positions and sizes are measured relative to the extended block skeleton (main axis) and its perpendicular minor axes.
- **Data:**
  - OSM from **28 cities** (Chicago, D.C., NYC, Atlanta, …, Houston);
  - **119,236 blocks, 2,513,697 buildings**; split **80 % train / 20 % validation**;
  - minimal cleanup ("simplify parallel multi-lane roads to single edges", remove self-loops).
- **Training:** 10⁵ iterations, about 9 h on one A5000; inference 5.81 ms per block.
- **Output:** vector, i.e. building footprint polygons synthesised from shape parameters.

## 5. Inputs/outputs and terrain

- **Input:** block polygons obtained from an **existing road network**; optionally a sparse prior of known blocks, giving latent interpolation from the k = 5 nearest blocks.
- **Output:** building footprints per block; in a Chicago demo, also heights.
- **Terrain:** **none** (no DEM, slope or water). Water and parks appear only as blank non-block areas.

## 6. Evaluation

- **Set-to-set comparison (Table 1):** 1,000 generated vs 1,000 real layouts. Ours is conditioned on real road networks.
- **Metrics:**
  - **Overlap (%):** overlapping building area;
  - **Out-Block (%):** building area outside the block contour;
  - **WD** of building-count and bounding-box geometry distributions;
  - **FID;**
  - **L-Sim:** a new layout similarity. Blocks are aligned, each building pair is scored S(b1, b2) = MinArea(b1, b2) · 2^(−c‖b1 − b2‖₂) with c = 0.02, and the Hungarian maximum-weight matching is averaged.
- **Baselines:** LayoutVAE, Gupta et al. (layout transformer), VTN and BlockPlanner (partly re-implemented). All were retrained on this dataset.
- **Results, GlobalMapper (Table 1):** L-Sim 22.45, Overlap 1.42 %, Out-Block 0.89 %, FID 14.94, WD(bbx) 1.45, WD(count) 0.06.
- **User study:**
  - **2AFC**, 18 comparisons, with 2 duplicate questions as quality checks;
  - 62 responses, 50 valid after filtering speeders, constant answerers and failed checks;
  - preferred over LayoutVAE, BlockPlanner and VTN in 97.3 %, 86.7 % and 88.0 % of answers.
- **Ablations (Table 2):**
  - one-to-one comparison on 1,000 validation layouts: Overlap, Out-Block, position error and coverage error;
  - adding the canonical spatial transform (CST) and shape encoder (SE) to the baselines;
  - alternative GNN layers.
- **Hold-out, statistics and runtime:** no geographic hold-out (random 80/20 split of blocks); no confidence intervals or seeds. Runtime is given above.

## 7. Verbatim quotes (arXiv v1 pages)

- p.1, Abstract: "Our method generates realistic urban layouts given arbitrary road networks, and enables conditional generation based on learned priors."
- p.5, §4.1: "The total number of city blocks is 119,236, containing 2,513,697 buildings" … "We split the data 80% for training, and 20% for validation. We obtained the datasets by downloading 2D layouts from OpenStreetMap (OSM) [42]."
- p.5, §4.1: "The average inference time is 5.81ms per city block by a single A5000 GPU."
- p.5, §4.2: "Wasserstein distance (WD) indicates the similarity between two distributions. Specifically, we computed the WD between the distributions of building counts and bounding box geometry (location and size, centered around the origin)."
- p.6, §4.2: "The study was performed in a two-alternative forced choice (2AFC) manner." … "We also included 2 random duplicate questions for quality checking. Users that answered differently to the same question were discarded."
- p.8, §5: "Our approach does have some limitations. First, our approach cannot represent city block contours with interior boundaries." … "Our method can handle common dead-end roads, such as cul-de-sac's (see Supplemental), but not all cases of dead-end roads."

## 8. Comparability with UrbanMorph

- **Not a street-network generator, so not a baseline for M18.** It is a downstream consumer of street networks; RoBus Task IV builds on it.
- **Code:** https://github.com/Arking1995/GlobalMapper.
  - Official PyTorch code; **no licence file** (GitHub API `license: null`), so all rights are reserved; last push 2023-10-10.
  - Includes a 10K-block sample, `dataset.tar.gz` (about 50 MB). The full 120K dataset is on a Purdue SharePoint link (not opened).
  - **No pretrained weights in the repository.**
- **macOS without CUDA:**
  - `train.py` and `test.py` hard-code `torch.device('cuda:…')`; switching to CPU or MPS is a small edit.
  - The README pins `torch==1.9.0+cu111` and torch-geometric/torch-sparse/torch-scatter CUDA wheels, so a modern CPU torch plus PyG install is needed (not tested).
- **Inputs we could supply:** block polygons from UrbanMorph's generated networks, i.e. minimal faces of our planar graph clipped to the tile.
- **Retraining:** required (no weights). Training on CPU would be far slower than 9 h on an A5000 (estimate).
- **Effort (estimates):** port and train for 3D deliverables (populating generated blocks), **15–30 h**. Not relevant to the M18 realism/terrain comparison.

## 9. Ideas we could adopt

- **A 2AFC perceptual study with quality controls:**
  - duplicate questions to detect inconsistent raters;
  - removal of speeders and constant answerers;
  - report preference rates per baseline.
  - Use it for "real OSM vs generated network on the same terrain".
- **Matching-based similarity (in the spirit of L-Sim)** for networks. For example, Hungarian matching of junctions within a radius, or of blocks by area and position. This gives a per-site score of closeness to the real network on the same terrain, alongside distribution-level distances.
- **Validity percentages** such as Out-Block, as an analogue of our hard-constraint violations: edges over water, beyond the maximum grade, or outside the boundary.
- **Block-level downstream check:** whether our generated blocks can be populated (block size and shape within the range seen in real cities). This is an indirect realism test of the block shapes the street network induces.
