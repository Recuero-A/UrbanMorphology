# Index: learned and data-driven street-network generators (M18 literature notes)

Compiled 2026-11-25 from free sources only. Each row links to a per-paper note, which carries the URLs, access level (FT, AO, MD), verbatim quotes and the "not verified" flags. Effort figures are estimates; nothing was installed or run.

This index covers only the learned/data-driven batch. Other notes in this folder (`boeing*`, `chen2008_*`, `parish2001_*`) belong to other batches.

**Column meanings:**
- **Vector/raster:** the generator's native output.
- **Terrain inputs:** what enters the model itself.
- **Comparable?:** fit to UrbanMorph's matched-input protocol (512 m tile, 4 m DEM, water mask, anchors).

## Table

| Key | Citation | Model type | Vector / raster | Terrain inputs | Evaluation type | Code / weights / licence | Comparable? | Access | Note |
|---|---|---|---|---|---|---|---|---|---|
| chu2019 | Chu et al., ICCV 2019, Neural Turtle Graphics | Autoregressive GRU encoder-decoder; queue growth; 5 m snapping | Vector | None (city-ID style, optional sketch) | Domain-adapted FID; Fréchet distance on node density, degree, reach and transport convenience; Chamfer diversity; APLS (parsing) | Official repo has no code or weights. Unofficial `Cewein/Neural-Turtle-Graphics`: no licence, CPU, one Balmain weight file (config path broken) | Yes, as a no-terrain ablation of our own base (2–6 h), or completion from the context's degree-1 nodes | FT | [chu2019_ntg.md](chu2019_ntg.md) |
| hartmann2017 | Hartmann et al., WSCG 2017, StreetGAN | Spatial GAN (SGAN), one model per example | Raster at 3 m/px → threshold + morphology + Zhang–Suen thinning → graph | None | Visual; block area, compactness and aspect-ratio histograms | None found (base SGAN: MIT, Theano, deprecated) | No (historical) | FT | [hartmann2017_streetgan.md](hartmann2017_streetgan.md) |
| fang2020 | Fang, Jin, Yang, arXiv 2010.04536 (JUT 29(2) 2022) | Conditional GAN inpainting (encoder-decoder with dilated convs) | Raster, 256 px at 2 m (512 m); no graph extraction | SRTM elevation + slope aspect, concatenated | Pixel l1/l2; error vs context coverage; city hold-out (Prague → Barcelona) | None found | Completion only; superseded by fang2022 | FT (arXiv); MD (JUT) | [fang2020_planning.md](fang2020_planning.md) |
| fang2022 | Fang et al., IJGIS 36(10) 2022 | GAN inpainting (gated conv, SN-PatchGAN) + geo-extractor gate + geo-merging bypaths; vertex heat map | Raster, 256 px at 2 m (512 m); graph only for metrics (method not given) | 7 channels from a 30 m DEM: elevation, slope, aspect, 4 hillshades. **No water** | Pixel MAE/MSE; APEMC/APEMI (length-weighted space-syntax choice and integration); flat-vs-hilly gap; city hold-out (FL+PR → RM+SN) | **figshare 10.6084/m9.figshare.19606960, CC BY 4.0. PyTorch code, base maps and weights for Models 1–3.** T3 is incomplete: model class and an import are missing, there is no T3 `train()`, no metric code, and CUDA is hard-coded | Completion only, with **oracle junctions**. About 16–27 h with pretrained weights | FT | [fang2022_topography.md](fang2022_topography.md) |
| he2023 | He & Aliaga, ICCV 2023, GlobalMapper | Conditional VAE + graph attention network | Vector buildings **given roads** | None | Set metrics (Overlap, Out-Block, WD, FID, L-Sim); 2AFC user study | `Arking1995/GlobalMapper`, no licence, no weights, CUDA hard-coded | No (consumes street networks) | FT | [he2023_globalmapper.md](he2023_globalmapper.md) |
| zhou2024 | Zhou et al., CAVW 35(3) e2267, 2024 | Image translation → road density map → transformer → graph | Vector | Elevation, water, vegetation maps | Not verified | None found | Same setting as ours (terrain → vector) but non-urban; reimplementation 60–120 h | AO | [zhou2024_vecroad.md](zhou2024_vecroad.md) |
| gu2024 | Gu et al., IJGI 13(6):203, 2024 | Conditional DDPM U-Net, one model per city | Raster, 128 px at 7.81 m (1 km) → binarise → Zhan thinning → vectorise (post-processing code not released) | SRTM 30 m elevation + slope (per-city min–max); water as 1 of 9 land-use channels; Model 5 adds a real-junction channel | FID, F1, IoU, DAD/DARL; obstacle-area vs road-length regressions; within-city 80/20 split; no CIs | **figshare 10.6084/m9.figshare.24618942, CC BY 4.0. PyTorch 1.13.1 code + 5 cities of data; no weights**; CUDA hard-coded | **Yes, after retraining.** Use Model 4 (no junctions); 128 px at 4 m = our tile. About 20–30 h + 1–3 days compute | FT | [gu2024_diffusion.md](gu2024_diffusion.md) |
| qin2024 (NEW) | Qin et al., AAAI 2024, Text2City | Guided DDPM + fine-tuned CLIP, roads + buildings | Raster (vectorised only for metrics) | None | SSIM, FID, WD, RLS, BLS; user study (20 people) | `LittleQBerry/Text2City`, no licence, no weights | Completion given surrounding streets, terrain-blind; 30–50 h | FT | [qin2024_text2city.md](qin2024_text2city.md) |
| feng2025 | Feng et al., C&G 133:104451, 2025, CaRoLS | VAE + latent diffusion (third-party description, not verified) | Probably raster (n.v.) | Not verified (third party says land use, elevation, population) | Not verified | None official | No; 80–150 h | MD | [feng2025_carols.md](feng2025_carols.md) |
| dong2025 | Dong et al., ICMR 2025, DRoLaS | Coarse-to-fine diffusion; class-weighted loss; "connectivity refinement" | Not verified | Not verified | Self-built dataset (n.v.) | None found | No | AO | [dong2025_drolas.md](dong2025_drolas.md) |
| jalengg2026 (NEW) | "Groundwork", GitHub prototype 2026 (CaRoLS-style), unpublished | VAE + latent diffusion | Raster, 512 px at 5 m (2.56 km) → skeleton | SRTM elevation (per tile), OSM water, land use | None published; author reports poor outputs | `jalengg/groundwork`, MIT, no weights | Same terrain-only setting; low credibility; 15–30 h + compute | Repo only | [jalengg2026_groundwork.md](jalengg2026_groundwork.md) |
| li2026rw | Li et al., arXiv 2608.11580, 2026, RoadWeaver | VQ-VAE + masked transformer skeleton → procedural tensor-field growth → A\* reconnection + LCC filter | Raster road field → vector graph + lane HD map | None (11-d style and density condition) | Topology validity (LCC, reachability, dead-end, cycle ratio); endpoint error; density control; no realism vs real cities | Official "upon acceptance". Dev repo `bitxige/tactics2d-roadweaver`, GPL-3.0, no weights | Unconditional morphology and topology only; retrain 40–80 h. **Its post-hoc repair as an ablation: 6–12 h** | FT | [li2026_roadweaver.md](li2026_roadweaver.md) |
| li2022md | Li et al., TPAMI 2023, MetaDrive (RoadWeaver baseline) | Procedural block assembly | Vector lanes | None | RL generalisation; no map realism | `metadriverse/metadrive`, Apache-2.0 | No | FT (map parts) | [li2022_metadrive.md](li2022_metadrive.md) |
| yang2024rg | Yang et al., arXiv 2411.19577, RoadGen (RoadWeaver baseline) | Procedural templates | Vector lanes (MATLAB/RoadRunner) | None | Uniqueness rate; simulator usability | `roadgen/RoadGen`, no licence, needs commercial MATLAB | No (not runnable for us) | FT | [yang2024_roadgen.md](yang2024_roadgen.md) |
| mi2021 | Mi et al., CVPR 2021, HDMapGen (RoadWeaver baseline) | Autoregressive hierarchical graph model (GRAN-style) | Vector (lane level, 120–200 m) | None | MMD (degree, spectrum); Fréchet on geometry + NTG urban features; diversity; latency | No official code. Unofficial `agshar96/HDMapGen`, MIT, PyTorch 1.2 | Unconditional only; scale mismatch; 20–40 h | FT | [mi2021_hdmapgen.md](mi2021_hdmapgen.md) |
| post2026 | Post, Ledoux, Patil, ISPRS Annals 2026, CityStack | **Statistics-driven procedural:** 13-cluster road GMM typologies, simulated-annealing typology grid, Parish–Müller growth | Vector roads (100 m typology grid) | None ("not yet implemented") | Typology/annealing similarity only | `citypy` and `citystackgen`, GPL-3.0 (umbrella repo: no licence) | Unconditional morphology only; 6–12 h / 20–30 h | FT | [post2026_citystack.md](post2026_citystack.md) |
| dobbie2026 | Dobbie & Johansson, Research Square 2025 / EPJ Data Sci. 2026, CitySprout | **Procedural** parametric grammars (not learned) | Vector NetworkX | None | Road density (alpha shape), "living metric", circuity; 4 grammars × 10 seeds; median and IQR | `camdobbie/CitySprout`, no licence | Unconditional only; 1–2 h / 10–16 h + permission | FT (preprint + SI); AO (EPJ) | [dobbie2026_citysprout.md](dobbie2026_citysprout.md) |
| li2026robus | Li et al., Computers 15(8):488, 2026 (arXiv 2407.07835), RoBus | **Dataset**; baselines pix2pix + clDice, ControlNet, cVAE + GAT | Raster at 5 m (EPSG:3857) + skeletonised graphs | Water and green channels; **no DEM** | FID, Chamfer diversity, validity, traffic convenience (> 300 m), bearing entropy; random split on overlapping tiles | `tourlics/RoBus_Dataset`, no licence; data on Baidu Pan behind emailed codes | Not a baseline; reuse its metric definitions only | FT (arXiv); AO (journal) | [li2026_robus.md](li2026_robus.md) |

**2024–2026 candidates checked but given no note (no usable code):**
- SCOPE (Guo, Jin, Xu, IJCNN 2026): abstract only, no code.
- Lin & Yao 2026: figshare 10.6084/m9.figshare.33078884 is a 5.3 MB dataset, CC BY 4.0, with no code.
- Yang, Zhu, Feng (arXiv 2305.08186): cGAN conditioned on elevation, population and land use; no code.
- GraphWalker (IJCAI 2025): MIT code, but it builds roads from GPS trajectories, so it is excluded.
- MajutsuCity (CVPR 2026): a semantic-layout generator, not a street-network generator.
- Not re-checked: Ye et al. 2026 (graph CVAE + graph transformer, LNEE).

## Correction to `docs/M18_BASELINES.md`

M18_BASELINES says "Learned baseline: none is clean" and finds no learned vector road generator with code and weights that takes a DEM. That statement is still true as worded. It misses two **raster** terrain-conditioned learned generators that do have code:

- **Fang 2022:** figshare, CC BY 4.0, PyTorch, **with pretrained weights**. It is completion-only with oracle junctions, and its Model 3 code is incomplete.
- **Gu 2024:** figshare, CC BY 4.0, PyTorch 1.13.1, **code + data, no weights**.

M18_LITERATURE's "data + code on figshare, CC BY 4.0" for Fang 2022 is correct. Its "evaluated only by pixel MAE/MSE" is incomplete: the paper also reports APEMC and APEMI, network-level space-syntax errors on reconstructed graphs (pp. 2050–2051).

## Ranking of candidate learned baselines

The three axes are:
- **recency:** publication year;
- **faithful runnability:** can we reproduce the authors' model without inventing parts?
- **terrain input:** does the model take DEM, slope or water?

Procedural entries (CityStack, CitySprout, MetaDrive, RoadGen) and the dataset (RoBus) are excluded.

| Rank | Candidate | Recency | Faithful runnability | Terrain input | Use in UrbanMorph | Effort (estimate) |
|---|---|---|---|---|---|---|
| 1 | **Gu et al. 2024** (diffusion) | 2024 | **Medium-high.** Complete train/test code and the exact data layout; no weights; our own skeletonisation needed; paper says L2 loss, code uses Huber | **DEM + slope + water** (via land use) | **Main learned terrain baseline:** retrain "Model 4" (no junctions) per region on our 512 m tiles (128 px × 4 m). Report Model 5 (real junctions) only as a labelled oracle | 20–30 h + 1–3 days compute |
| 2 | **Fang et al. 2022** (GAN inpainting) | 2022 | **Medium for inference** (weights exist; alias one class, stub one import, patch CUDA). **Low for retraining** (no T3 `train()`) | **7 DEM channels**; no water | **Secondary, completion-only experiment:** infill a 128–192 m hole given context. Oracle-junction and zero-junction variants. Out-of-domain (trained in Italy) | 16–27 h (pretrained); +30–50 h to retrain |
| 3 | **RoadWeaver 2026** | 2026 | **Low-medium.** Dev repo (GPL-3) not yet the official release; no weights; 2 km skeleton-density maps | None | Topology comparison on flat tiles after retraining. **Better use: reimplement its post-hoc A\*/LCC repair on our raw samples** as the "repair vs in-loop potentials" ablation | 6–12 h (repair ablation); 40–80 h + GPU (retrain) |
| 4 | **NTG-style** (our TerrainGraphTurtle with terrain off; or Cewein) | 2019 | **High (ours)** / low (Cewein: no licence, broken config, deviates from paper) | None | No-terrain ablation of our own base; reviewers will see it as an ablation, not an external baseline | 2–6 h (ours); 15–25 h (Cewein, with permission) |
| 5 | **Groundwork 2026** (CaRoLS-style, unpublished) | 2026 | Low. Unpublished, self-reported poor outputs, no weights | DEM + water + land use | At most a reference implementation for a CaRoLS-style baseline | 15–30 h + compute |
| 6 | **HDMapGen** (unofficial) | 2021 | Low-medium. MIT reimplementation, old PyTorch, lane-level scale | None | Weak unconditional AR-graph baseline | 20–40 h |
| 7 | **Text2City** | 2024 | Low. No licence, no weights | None | Terrain-blind completion reference | 30–50 h + compute |

**Not runnable (no code):** Zhou 2024 (the closest DEM→vector precedent; AO only), CaRoLS, DRoLaS, Fang 2020, StreetGAN and SCOPE.
