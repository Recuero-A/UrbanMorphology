# Milestone 10 — Literature review: hierarchical street and block generation

**Scope.** These are the 8 papers named in the M10 specification. Each entry is filled in from the paper
itself, except where the entry says otherwise.

**Access.** Read this before relying on any entry:

| Paper | What was read |
|---|---|
| NTG, BlockPlanner, GlobalMapper, npj 2026, Human-guided diffusion | Full text |
| CaRoLS (C&G 2025) | **Not accessible**: neither the full text nor the abstract. Only bibliographic metadata and one search-engine snippet. |
| GenBuilding-DDPM (AutCon 2025) | **Not accessible**: full text. Abstract and metadata only. |
| Guided discrete diffusion (WSDM 2026) | **Abstract only.** |

A cell reading "n/a (not read)" means the information was not available. It does not mean the paper
lacks it. No novelty claim in M10 rests on these three papers.

## 1. Per-paper summaries

### 1.1 Neural Turtle Graphics (NTG) — Chu et al., ICCV 2019, arXiv:1910.02055

| Field | Content |
|---|---|
| **Objective** | Generate road networks as graphs. Uses: city-like synthesis, aerial-image parsing, interactive editing. |
| **Representation** | Undirected planar graph. Nodes are 2D points; each node has incoming and outgoing paths. |
| **Architecture** | Encoder: a bidirectional GRU over each incoming path (motion vectors). Decoder: a GRU that emits outgoing nodes one at a time. Δx and Δy are each discretised to 1 m bins in [−100, 100] and predicted as **independent** categoricals, with a stop flag. 500 hidden units. |
| **Dataset** | RoadNet: 17 cities in 13 countries, 10 km² per city, 233.6k nodes, taken from OSM. |
| **Conditioning** | Unconditional by default. Optional conditioning on an aerial-image CNN for parsing. Supports user edits. |
| **Losses** | Cross-entropy on the discretised Δx, Δy and the stop flag, trained with teacher forcing. |
| **Sampling** | Breadth-first queue of frontier nodes. New nodes within 5 m of an existing node merge into it. |
| **Metrics** | FID with a domain-adapted network. Urban-planning statistics: node density, connectivity, reach, convenience. Diversity. |
| **Compute** | Adam, lr 1e-3. Hardware is not given in the main text. |
| **Strengths** | Local and scalable. The incoming-path context resembles our turtle frame. Strong local realism in grid cities. |
| **Limitations** | Purely local, so no global structure (no arterial hierarchy). Δx and Δy predicted independently. Trained with teacher forcing only, so exposure bias is untreated. |
| **Relevance** | Our GraphTurtle is NTG-like (continuous mixture output instead of bins, plus a raster context). NTG has the same structural weakness that our diagnosis measures in §3 of `MILESTONE_10.md`. |
| **Code** | Project repository with README and docs only. **No training code.** |

### 1.2 BlockPlanner — Xu et al., ICCV 2021

| Field | Content |
|---|---|
| **Objective** | Generate lot and building layouts for city blocks. |
| **Representation** | Each block is a ring of lots (topology) plus a two-tier graph (lots → buildings). Per node: existence, land use, geometry. |
| **Architecture** | Graph VAE with a GNN encoder and decoder over the ring graph. |
| **Dataset** | 637 Manhattan blocks with 15,541 lots, N ≤ 50 lots per block. |
| **Conditioning** | Block shape. Land-use editing at inference time. |
| **Losses** | L1 on geometry, existence cross-entropy, a geometry-violation (overlap) penalty, KL. |
| **Sampling** | Draw z from the prior, then decode. |
| **Metrics** | User rating, FID on rasterised layouts, L1 distances on statistics. |
| **Compute** | Not a central result. |
| **Strengths** | Explicit lot topology. Land-use editing. |
| **Limitations** | One city, one grid typology. Small N. No code. |
| **Relevance** | The ring-of-lots representation fits Spanish and Japanese perimeter blocks. Only a design reference, since there is no code. |
| **Code** | None released. |

### 1.3 GlobalMapper — He & Aliaga, ICCV 2023, arXiv:2307.09693

| Field | Content |
|---|---|
| **Objective** | Generate building layouts for arbitrary city blocks at city scale. |
| **Representation** | Each block is warped by a canonical skeleton transform onto a grid graph of at most 120 building slots. Each slot has existence, position, size and shape attributes. |
| **Architecture** | Conditional VAE. The block shape enters as a 64×64 mask through a CNN. A GAT runs over the grid graph. z has 512 dimensions. |
| **Dataset** | 119,236 blocks with 2.51M buildings from 28 North American cities. |
| **Conditioning** | Block shape. Optional constraints: building count and style. |
| **Losses** | L2 on geometry (×4), cross-entropy on existence and shape (×1), KL (×0.5). |
| **Sampling** | One forward pass per block: **5.81 ms per block**. |
| **Metrics** | FID 14.94 vs BlockPlanner 39.27. Also L1 on statistics and a user study. |
| **Compute** | About 9 h on one RTX A5000. |
| **Strengths** | Fast. Works for any block shape. Code available. Directly fits our "generate buildings on real streets" setting. |
| **Limitations** | No courtyards or holes. Limited footprint shapes. A fixed maximum building count. North America only. Parcels are not modelled. |
| **Relevance** | **The main design reference for M10 Phase 4.** At our data size it needs a much smaller model (§3 of `M10_BLOCK_GENERATION.md`). |
| **Code** | github.com/Arking1995/GlobalMapper. |

### 1.4 AI-based urban layout generation (3D) — Liu He et al., *npj Urban Sustainability* 6:87 (2026), doi:10.1038/s42949-026-00369-2

| Field | Content |
|---|---|
| **Objective** | A global-scale 3D urban layout generator: building footprints plus heights. |
| **Representation** | GlobalMapper-style block graphs with height attributes. |
| **Architecture** | A scaled-up GlobalMapper: about 130M parameters. |
| **Dataset** | 330 cities: 833,473 blocks and 17.66M buildings. |
| **Conditioning** | **Takes the road network as input.** It does not generate streets. |
| **Losses** | Same family as GlobalMapper, plus a height term. |
| **Sampling** | One pass per block. |
| **Metrics** | FID-type and statistical metrics, per region. |
| **Compute** | Trained on 100 A100 GPUs. |
| **Strengths** | Global coverage, including non-US cities. |
| **Limitations** | Roads are given, not generated. Compute far beyond our budget. Code not yet released. |
| **Relevance** | Confirms that conditioning on real streets is the field's practice for blocks. Out of reach to reproduce. |
| **Code** | "Will be made available" (not released as of 2026-10-31). |

### 1.5 Human-guided urban form generation with multimodal diffusion — Mingyi He et al., *Building and Environment* (2025), doi:10.1016/j.buildenv.2025.113892, arXiv:2505.24260

| Field | Content |
|---|---|
| **Objective** | Generate urban form step by step under human guidance. |
| **Representation** | Raster images of 450 m cells. |
| **Architecture** | Stable Diffusion 1.5 plus ControlNet, in three stages: (1) roads, (2) building footprints, (3) detailed form. |
| **Dataset** | 4,049 NYC cells and 3,209 Chicago cells. |
| **Conditioning** | Text prompts and the previous stage's image. |
| **Losses** | Standard latent-diffusion noise prediction. |
| **Sampling** | Iterative denoising per stage. |
| **Metrics** | FID 49.76 for NYC at stage 2. Also a user study. |
| **Compute** | About 105 GPU-hours on an L40S. |
| **Strengths** | Explicit stagewise hierarchy (roads, then buildings). Controllable. |
| **Limitations** | Raster output: graphs and polygons must be vectorised afterwards. Large pretrained model. US grid cities only. |
| **Relevance** | Supports the hierarchical ordering. The model size and the raster representation do not suit our constraints. |
| **Code** | github.com/Hemy17/Stepwise_GenerativeUrbanDesign. |

### 1.6 CaRoLS — Feng et al., *Computers & Graphics* 133:104451 (2025)

| Field | Content |
|---|---|
| **Access** | **Full text and abstract were inaccessible.** A search snippet mentions a "Diffusion-VAE integration". **Unconfirmed.** |
| **All other fields** | n/a (not read). |
| **Related, by the same group** | DRoLaS (ICMR 2025): coarse-to-fine diffusion for road layouts. Only its metadata was seen. |
| **Relevance** | Possibly a coarse-to-fine diffusion road generator. Not used for any decision until the paper can be read. |
| **Code** | None found. |

### 1.7 GenBuilding-DDPM — Si-Hao Li et al., *Automation in Construction* 179:106491 (2025)

| Field | Content |
|---|---|
| **Access** | **Full text inaccessible.** Abstract and metadata only. |
| **Abstract-level summary** | A multi-channel denoising diffusion model that generates building layouts as rasters. |
| **Other fields** | n/a (not read). |
| **Relevance** | Raster diffusion for buildings, same caveats as 1.5. |
| **Code** | None found. |

### 1.8 Urban Traffic Network Layout Optimization with Guided Discrete Diffusion — Yun … Park, WSDM 2026, doi:10.1145/3773966.3777950

| Field | Content |
|---|---|
| **Access** | **Abstract only.** |
| **Abstract-level summary** | A discrete diffusion model over network adjacency, guided by a learned predictor of traffic performance. Evaluated on synthetic grids, Manhattan and Monaco. |
| **Objective** | **Optimisation of traffic performance, not realism.** |
| **Other fields** | n/a (not read). |
| **Relevance** | Low for realism. Predictor guidance is a possible tool for later controllability. |
| **Code** | Not checked. |

## 2. Comparison

| | Output | Hierarchy | Data scale | Compute | Code | Fits our constraints? |
|---|---|---|---|---|---|---|
| NTG | Road graph | None (local) | 17 cities | Small | No | **Yes**. It is essentially our baseline. |
| BlockPlanner | Lots + buildings | Lot → building | 637 blocks | Small | No | Design idea only. |
| GlobalMapper | Buildings per block | Block → buildings | 119k blocks | 9 h on A5000 | **Yes** | **Yes, if scaled down** |
| npj 2026 | 3D buildings per block | Roads given → blocks | 833k blocks | 100 A100s | Not yet | No (compute) |
| Human-guided diffusion | Raster | Roads → buildings → detail | about 7k cells | 105 GPU-h | Yes | No (size, raster) |
| CaRoLS | Roads (?) | Coarse → fine (?) | ? | ? | No | Unknown |
| GenBuilding-DDPM | Raster buildings | ? | ? | ? | No | Probably not |
| WSDM 2026 | Network adjacency | — | Small | ? | ? | Off-objective |

**Our scale for comparison:** 565 training tiles in 9 cities, about 6.7k closed blocks (`outputs/m10/audit/dataset_audit.json`).

## 3. What the literature implies for M10

1. **No paper here that was read in full generates streets with an explicit arterial-then-local
   hierarchy using a small vector model.**
   * NTG is local only.
   * The diffusion papers are hierarchical but raster-based and large.
   * Whether CaRoLS or DRoLaS do this cannot be confirmed.

   So a hierarchical vector street generator is a reasonable experiment. **We make no novelty claim**,
   because three of the eight papers could not be fully read and this was not a systematic search.
2. **For buildings, the established approach is a GlobalMapper-style model conditioned on block shape,
   placed on real streets** (GlobalMapper; npj 2026). At about 6.7k blocks we need a model roughly
   10–20× smaller than GlobalMapper's, with strong canonicalisation.
3. **Diffusion is not justified as the first step.** The successful diffusion work uses pretrained SD 1.5
   and about 100 GPU-hours, or data 20× larger than ours. A lightweight diffusion street model (baseline D)
   is kept only as a pilot.
4. **Exposure bias is not treated in NTG.** Measuring it first, as the M10 specification requires, is
   consistent with the gap in this literature.

**Recommendation:** see `M10_STREET_ARCHITECTURES.md` §4 and `MILESTONE_10.md` §5.
