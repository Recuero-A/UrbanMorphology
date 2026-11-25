# Yang et al. 2024 — RoadGen (RoadWeaver baseline, short note)

## 1. Header

- **Citation:** Fan Yang, You Lu, Bihuan Chen, Peng Qin, Xin Peng. "RoadGen: Generating Road Scenarios for Autonomous Vehicle Testing." arXiv:2411.19577v1 [cs.SE], submitted 29 Nov 2024, 7 pages. Peer-reviewed venue: not verified.
- **DOI:** 10.48550/arXiv.2411.19577 (arXiv-issued).
- **Identified as RoadWeaver's baseline:** reference [23] in arXiv 2608.11580 ("Roadgen: generating road scenarios for autonomous vehicle testing. arXiv preprint arXiv:2411.19577").
- **URLs opened:**
  - https://arxiv.org/abs/2411.19577
  - https://arxiv.org/pdf/2411.19577 (read in full)
  - https://roadgen.github.io/
  - https://api.github.com/repos/roadgen/RoadGen and `/contents/`
  - https://api.github.com/orgs/roadgen/repos (404)
- **Access level:** **FT**.

## 2. BibTeX

```bibtex
@misc{yang2024roadgen,
  title         = {RoadGen: Generating Road Scenarios for Autonomous Vehicle Testing},
  author        = {Yang, Fan and Lu, You and Chen, Bihuan and Qin, Peng and Peng, Xin},
  year          = {2024},
  eprint        = {2411.19577},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  doi           = {10.48550/arXiv.2411.19577}
}
```

## 3. What it does

RoadGen builds road scenarios for testing autonomous-driving software by connecting instantiated templates of eight parameterised road components: straight, curve, lane switch, fork, T-intersection, intersection, U-shaped road and roundabout. It favours the least-used components to increase diversity and removes scenarios with duplicate topology. Python templates emit MATLAB scripts, which RoadRunner compiles into HD maps and 3D scenes for SORA-SVL/Apollo.

## 4. Learned or procedural, and how

- **Procedural, rule-based. No learning.**
- **Templates.** 242 templates cover 1–6 lanes and 7 lane-marking types.
- **Algorithm 1 (guided connection):**
  1. Pop an endpoint from a queue.
  2. Pick the least-used compatible component.
  3. Instantiate it within the parameter constraints.
  4. Reject it if its covered area overlaps the scenario.
  5. Repeat until TotalCount components are placed (4–8 in the experiments).
- **Deduplication.** Topology similarity Sim(G1, G2) over component-type adjacency (Definitions 1–4, Eq. 1). Scenarios with Sim = 1 are dropped.
- **Output** is vector (lane geometry in RoadRunner HD map format), not pixels.

## 5. Inputs / outputs and terrain

- **Inputs:** component list, number of components, parameter constraints, endpoint-compatibility candidates, time budget.
- **Outputs:** MATLAB scripts → RoadRunner `.rrhd` → Apollo HD maps and 3D scenes.
- **Terrain: none.** The PDF has no "terrain", "elevation", "slope" or "height" (keyword search). There is no water.

## 6. Evaluation

- **RQ1, diversity.** RoadGen vs RAND (random selection) over 24 h, with 4–8 components per scenario, averaged over 5 runs. Reported: number of deduplicated scenarios, uniqueness rate (after / before dedup), and time to cover all 242 components.
  - RoadGen gives 3.9% more deduplicated scenarios and a 19.3% higher uniqueness rate on average.
  - It covers all components in 1.24 h vs 7.75 h.
- **RQ2, usefulness.** Sampled at 95% confidence and 5% margin. 92–96% of scenarios compiled to 3D scenes; failures came from excessive U-turn curvature. All routing tests in Apollo and the joint simulations succeeded.
- **No realism metrics** against real networks; no geographic data at all.
- **Hardware:** Ubuntu server, 4× RTX 3090.
- **In RoadWeaver (about 35–40 nodes):**

| LCC % | Reachability % | Dead-end % | Cycle % | Endpoint error (m) |
|---|---|---|---|---|
| 100 | 100 | 12.8 | 0.0 | 4.80 |

  Runtime was about 50–370 s, and valid maps "rarely extend beyond roughly 60 nodes".

## 7. Verbatim quotes (arXiv v1 PDF pages)

- p. 1 (abstract): "The key idea is to connect eight types of parameterized road components to form road scenarios with high diversity in topology and geometry."
- p. 2: "This work is focused on the road-level (layer 1) scenario of the 5-layer model [1], which describes the topology and geometry of road scenarios."
- p. 3: "Specifically, we provide a total of 242 unique templates for these eight types of road components."
- p. 4: "Specifically, it uses the covered area of RoadScen and the covered area of d, i.e., CoveredArea and d.getCoveredArea(), to determine whether overlap occurs."
- p. 4: "we first adopt MATLAB to compile the scripts into RoadRunner HD map files (i.e., rrhd files)."
- p. 4: "released the source code of our prototype as well as all the experimental data at our website https://roadgen.github.io/."
- p. 5: "We run the experiment 5 times, and report the average results."
- p. 6 (limitations): "First, we define eight types of typical road components, which is not meant to be exhaustive but is to illustrate the feasibility of our approach."

## 8. Comparability with UrbanMorph

- **Code.** https://github.com/roadgen/RoadGen (Python + MATLAB folders).
  - **No licence** (GitHub API `license: null`), so all rights are reserved by default.
  - Created and last pushed 2023-09-21.
- **Runnable.** It needs **MATLAB + RoadRunner (MathWorks, commercial)**, which violates our free-tools constraint. I treat it as **not runnable** for us.
- **Inputs we could supply:** none (no DEM, water, boundary or street context).
- **Fair comparison:** none. These are driving-test scenarios of 4–8 components, not urban networks. RoadWeaver measured a 0% cycle ratio for RoadGen, i.e. tree-like layouts.
- **Effort:** not recommended. Not estimated beyond "blocked by licence and commercial dependency".

## 9. Ideas to adopt

- **Motif-level duplicate and novelty check.** RoadGen's Sim(G1, G2) counts how many nodes have every incident (type, type) edge duplicated in the other graph.
  - An analogue for us: compare junction-type adjacency motifs (degree-3/4, dead-end, class) between generated and nearest training tiles.
  - This would flag memorisation of training tiles, complementing distribution distances.
- **"Uniqueness rate"** (deduplicated / generated) as a cheap diversity statistic across seeds.
