# Hijzelendoorn 2023: Creating Varied Terrain-Considerate Road Networks on Heightmaps with Directed Alternating Physarum Agents

**Added item: this work is not in `docs/M18_LITERATURE.md`** (nor in any other `docs/*.md`; checked by text search on 2026-11-25). It is a master's thesis (not peer-reviewed) about **terrain-aware path networks**: walking or hiking paths between points of interest on heightmaps. It is included because it generates path networks between supplied points on a DEM, and because it defines terrain-behaviour measures (winding, joining).

## 1. Header

- **Citation:** Jeroen Hijzelendoorn. "Creating Varied Terrain-Considerate Road Networks on Heightmaps with Directed Alternating Physarum Agents." Master's thesis, Game and Media Technology, Utrecht University, 2023. Supervisor S.C.J. Bakkes.
- **DOI:** none. Handle hdl:20.500.12932/44257.
- **URLs opened:**
  - https://studenttheses.uu.nl/handle/20.500.12932/44257: JavaScript app shell only.
  - https://studenttheses.uu.nl/server/api/pid/find?id=hdl:20.500.12932/44257: DSpace item metadata (title, author, date, abstract, supervisor, programme).
  - https://studenttheses.uu.nl/server/api/core/bitstreams/76c7ca49-f9cf-42c1-82db-20c76eba9ee0/content: thesis PDF, 21 pages. Read: §I, §IV.A, §V, §VII.
  - Found via the search-result listing for https://studenttheses.uu.nl/handle/20.500.12932/44257.
- **Access level:** **FT** (read §I, §IV.A, §V, §VII; §III and §VI skimmed).

## 2. BibTeX

```bibtex
@mastersthesis{hijzelendoorn2023dapa,
  author = {Hijzelendoorn, Jeroen},
  title  = {Creating Varied Terrain-Considerate Road Networks on Heightmaps with Directed Alternating Physarum Agents},
  school = {Utrecht University},
  year   = {2023},
  note   = {Game and Media Technology; hdl:20.500.12932/44257}
}
```

## 3. What it does

A slime-mould / ant-colony multi-agent simulation (DAPA) builds walking-path networks between chosen points of interest on heightmap terrain. Paths are made of free-angle line segments rather than grid steps. Agents trade off path length against climbing "effort", and paths heading to the same goal may merge ("joining"). The thesis compares two versions of its own method (v0.5 vs v1) on synthetic terrains.

## 4. How it generates

- **Procedural.** An agent-based stochastic simulation with pheromones. Effort and distance weights are set by hand; nothing is fitted to data.
- **Representation: vector.** The final path is a polyline of short segments with float coordinates. Agents move continuously and deposit pheromones on a grid.
- **Algorithm (§IV, summary):**
  1. Agents start at a node and move forward with random bending.
  2. They sense pheromones (slime-mould sensor model) and head to an automatically chosen goal.
  3. On arrival they alternate and search back for their start.
  4. Extensions:
     - heightmap interaction (an effort term for climbing);
     - between-path pheromone sharing (joining);
     - long-distance and copied pheromones;
     - a probabilistic sensor;
     - a termination condition.
  5. A final path constructor extracts each path.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - heightmap (1 grid cell = 1 m in the experiments; 256² to 2048² cells);
  - node positions;
  - agent counts (300 / 1000 / 2500 / 6000 for node distances of 200 / 400 / 800 / 1600 m);
  - effort and distance weights.
- **Outputs:** a path network between the nodes.
- **Terrain** enters as an effort cost for vertical movement, weighted by an "effort weight" (tested at 0.01, 0.5, 1, 10; and 1000 to force full avoidance).
- **Water and bridges:** none.
- **No grade threshold.**

## 6. Evaluation

- **Synthetic terrains only.** Sloped wall, raised node with a side ramp, inclined planes at 0–50°, an obstacle block, and random terrain.
- **No real path network is compared.** There is no comparison with Galin 2011 because "no code is available" (§V).
- **Measures:**
  - heatmaps of path placement;
  - bend-angle histogram and change of curvature (smoothness);
  - computation time and memory;
  - joined-segment length (consecutive points within 3 m of another path);
  - vertical, horizontal and full path length on inclines (winding);
  - path-length excess over the straight line on uniform-cost terrain (imperfection; target ≤ 110 %).
- **Statistics:** averages over 100 simulations per setting.
- **Outcome (§VII):**
  - smoothness met;
  - the large-scale target (three 1600 m paths in 5 min, 2 GB) not met;
  - joining inconsistent;
  - winding not achieved convincingly.

## 7. Quotes for citation

- p. 1, Abstract: "This thesis presents a novel pathfinding technique for creating terrain-considerate walking-path networks over heightmap terrain called Directed Alternating Physarum Agents (DAPA)."
- p. 1, §I.A: "Heightmap Considerate: paths need to avoid hard-to-navigate terrain like steep mountain sides if a serviceable detour is possible."
- p. 2, §I.A: "Winding: paths must ascend steep slopes in a winding pattern, as seen in real-life mountain scaling paths."
- p. 8, §V: "However, no code is available of this technique in practice, and there was insufficient time left to create an implementation from scratch."
- p. 9, §V.4: "To show the degree of joining, we measure how big this segment is by checking how many consecutive path points are within 3m of the closest point from another path."
- p. 19, §VII: "Only a combination of the two would satisfy the requirement, meaning that v1 by itself does not fulfil it adequately."

A script checked every quote against the PyMuPDF text of the thesis PDF.

## 8. Comparability with UrbanMorph

- **Domain mismatch.** Hiking paths, not urban streets. No road classes, no water.
- **Inputs.** It can take our DEM and anchors.
- **Code:** none linked in the parts I read (not verified elsewhere).
- **Effort (my estimate, labelled):** 30–50 h to reimplement. Not recommended as a baseline.
- **Fair comparison** (if ever): connector routing between supplied points.

## 9. Ideas we could adopt for evaluation

- **Winding test on synthetic inclines.** Put two anchors on an inclined plane (0–50°, as in their §V.5). Measure horizontal path length, maximum and average path incline, and the number of switchback turns as the incline grows. It is a clean, controlled probe of switchback formation for our sampler versus the LCP baselines.
- **Joining / trunk-merging measure:** the length of segments where two routes to the same destination run within 3 m of each other. Use it to measure route merging.
- **Imperfection bound:** the path-length excess over the optimum on uniform terrain is a simple sanity check that terrain-free behaviour stays near-straight.
