# Li et al. 2022/2023 — MetaDrive (RoadWeaver baseline, short note)

## 1. Header

- **Citation:** Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, Bolei Zhou. "MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 45(3):3461–3475 (issue dated March 2023 per Crossref; RoadWeaver cites it as 2022). arXiv:2109.12674 (v1 26 Sep 2021; v3 16 Jul 2022).
- **DOI:** 10.1109/TPAMI.2022.3190471 (Crossref). arXiv DOI 10.48550/arXiv.2109.12674.
- **Identified as RoadWeaver's baseline:** reference [15] in arXiv 2608.11580.
- **URLs opened:**
  - https://arxiv.org/abs/2109.12674
  - https://arxiv.org/pdf/2109.12674 (v3)
  - https://api.crossref.org/works?query.bibliographic=MetaDrive…
  - https://api.github.com/repos/metadriverse/metadrive and `/releases/latest`
  - https://raw.githubusercontent.com/metadriverse/metadrive/main/README.md
  - https://api.github.com/repos/metadriverse/metadrive/git/trees/main?recursive=1
  - https://raw.githubusercontent.com/metadriverse/metadrive/main/metadrive/engine/core/terrain.py
- **Access level:** **FT** for the map-generation parts (§4.1, Algorithm 1, licence paragraph). The RL experiments were not read in detail.

## 2. BibTeX

```bibtex
@article{li2023metadrive,
  title   = {MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning},
  author  = {Li, Quanyi and Peng, Zhenghao and Feng, Lan and Zhang, Qihang and Xue, Zhenghai and Zhou, Bolei},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {45},
  number  = {3},
  pages   = {3461--3475},
  year    = {2023},
  doi     = {10.1109/TPAMI.2022.3190471}
}
```

## 3. What it does

MetaDrive is a driving simulator for generalisable reinforcement learning. Its maps are either procedurally generated (PG) from road blocks or imported from Waymo, Argoverse and OSM-style lane data. PG maps are assembled from parameterised blocks (straight, ramp, fork, roundabout, curve, T-intersection, intersection) by a backtracking search.

## 4. Learned or procedural, and how

- **Procedural. No learning in map generation**; RL is only the downstream use.
- **Block representation.** Each block is a small graph G_ω = {V, E, S, P, ω, Ω} with:
  - lanes;
  - sockets, used to dock onto other blocks;
  - spawn points;
  - a type-specific parameter space (lane count, width, curvature).
- **Block Incremental Generation (BIG), Algorithm 1:**
  1. Uniformly pick a block type and sample its parameters.
  2. Rotate it so its socket docks onto a free socket of the network.
  3. Reject it if any of its edges cross the existing network, and retry, up to T tries.
  4. If all tries fail, remove the last block (backtrack).
  5. Stop at n blocks. A fixed block sequence can also be given.
- **Output is vector**: a lane graph. There are no pixels and no training data.

## 5. Inputs / outputs and terrain

- **Inputs:** number of blocks or a block sequence, parameter ranges, random seed.
- **Output:** a lane-level road network inside the simulator.
- **Terrain:**
  - Not used by the BIG road-layout algorithm. The PDF has no "terrain", "elevation" or "slope" (keyword search).
  - The current code base has a `Terrain` class (`metadrive/engine/core/terrain.py`, Panda3D `BulletHeightfieldShape`, `height_scale`). It is used for rendering and collision.
  - Whether it affects road placement is **not verified**. I assume it does not, because BIG has no terrain term.
- **Water:** none.

## 6. Evaluation

- The paper evaluates RL generalisation across numbers of training maps, safe RL and multi-agent RL. It does **not** evaluate the realism of the maps against real road networks.
- In RoadWeaver, the PG maps (about 35–40 nodes) scored:

| LCC % | Reachability % | Dead-end % | Cycle % | Endpoint error (m) |
|---|---|---|---|---|
| 99.5 | 99.3 | 43.0 | 50.5 | 8.17 |

  RoadWeaver also reports that MetaDrive's resource use rises rapidly as maps grow.

## 7. Verbatim quotes (arXiv v3 PDF pages)

- p. 5: "Each block preserves properties like lanes, spawn points for locating new vehicles, and sockets for interconnect other blocks."
- p. 6: "As illustrated in Algorithm 1, we propose a search-based PG algorithm Block Incremental Generation (BIG), which recursively appends block to the existing road network if feasible and reverts the last block otherwise."
- p. 7: "Maximally T trials will be conducted."
- p. 7: "The stop criterion of BIG is usually the number of blocks."
- p. 12: "Licenses. MetaDrive is released under Apache License 2.0."

## 8. Comparability with UrbanMorph

- **Code.** https://github.com/metadriverse/metadrive.
  - Licence: **Apache-2.0** (GitHub API and paper).
  - Last push 2026-10-16; latest release MetaDrive-0.4.3 (2024-12-06).
- **macOS.** The README says it runs on "Linux/Windows/MacOS". The CPU-only Python install was not tested by me.
- **Inputs we could supply:** none of DEM, water, boundary, anchors or surrounding streets. Block count is the only scale control.
- **Fair comparison:** none for urban realism or terrain. It produces highway- and driving-scenario block assemblies, not neighbourhood street networks. At most it is a "procedural floor" for topology metrics.
- **Retraining:** n/a.
- **Effort (estimate):**
  - export PG graphs to our metric pipeline: **3–6 h**;
  - value for our paper: low.

## 9. Ideas to adopt

- **BIG's local backtracking.** If no feasible block exists after T tries, revert the last block. This is a cheap analogue of our look-ahead; it could serve as a "reject-and-backtrack" ablation against SMC resampling.
- **Socket docking with crossing rejection.** Feasibility by construction. This matches our hard feasibility rules, so there is nothing new to adopt.
