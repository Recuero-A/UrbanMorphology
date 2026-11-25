# Galin, Peytavie, Maréchal, Guérin 2010: Procedural Generation of Roads

## 1. Header

- **Citation:** Eric Galin, Adrien Peytavie, Nicolas Maréchal, Eric Guérin. "Procedural Generation of Roads." *Computer Graphics Forum* 29(2):429–438, 2010 (Eurographics 2010 issue; the PDF header reads "EUROGRAPHICS 2010 / T. Akenine-Möller and M. Zwicker (Guest Editors)").
- **DOI:** 10.1111/j.1467-8659.2009.01612.x
- **URLs opened:**
  - https://perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf: timed out on 2026-11-25 (server unreachable from here).
  - http://web.archive.org/web/20250609201248/https://perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf: Wayback snapshot of that author PDF (10 pages; PDF page 1 = p. 429). Read in full.
  - https://api.archives-ouvertes.fr/search/ (query on the title): HAL record hal-01381447, giving full author first names.
  - https://hal.science/hal-01381447/document: returned an anti-bot challenge, not the PDF.
  - https://api.crossref.org/works/10.1111/j.1467-8659.2009.01612.x: metadata.
- **Access level:** **FT**.

## 2. BibTeX

```bibtex
@article{galin2010procedural,
  author  = {Galin, Eric and Peytavie, Adrien and Mar{\'e}chal, Nicolas and Gu{\'e}rin, Eric},
  title   = {Procedural Generation of Roads},
  journal = {Computer Graphics Forum},
  volume  = {29},
  number  = {2},
  pages   = {429--438},
  year    = {2010},
  doi     = {10.1111/j.1467-8659.2009.01612.x}
}
```

## 3. What it does

The paper generates **one road between a given start and end point** on a terrain. It solves a weighted *anisotropic* shortest-path problem: the cost depends on position, direction and curvature. The cost includes terrain slope along the direction of travel, curvature, water depth and vegetation. Bridges and tunnels are handled consistently as long arcs with their own costs. The discrete path is converted to clothoid splines, split into road, bridge and tunnel parts, carved into the terrain, and meshed.

## 4. How it generates

- **Procedural.** The cost terms are hand-designed "transfer functions" that the user edits interactively. Nothing is fitted to data.
- **Representation:** raster grid search → vector polyline → clothoid spline → 3D mesh.
- **Algorithm (§3–§6):**
  1. **Grid.** Discretise the domain into an n×n grid. Grid points are graph nodes.
  2. **Masks instead of 4- or 8-connectivity.** Each node is implicitly connected to all points (i, j) ∈ [−k, k]² with gcd(i, j) = 1. This fights the "limit-on-direction" artefact.
     - Table 1 gives the angle resolution: 45° for k = 1 down to 9.5° for k = 6.
     - k = 5 (11.3°, 80 segments) is recommended.
  3. **Search.** Run **A\*** with an admissible heuristic: the cost of a straight-line road to the goal.
  4. **Edge cost.** The line integral of c(p, ṗ, p̈) = Σ μᵢ∘κᵢ along the segment, approximated by a finite sum. The characteristics κ are:
     - vegetation density v(p);
     - water height w(p);
     - slope in the direction of travel s(p, ṗ);
     - curvature.
     Each transfer function μ has a threshold κ₀ above which it returns ∞.
  5. **Curvature.** The search runs on a 3D grid of position × orientation (m angle bins; m = 12 in Table 3), so turning cost can be charged.
  6. **Bridges and tunnels.** Extra masks T and B hold long arcs with length in [rᵢ, rₑ].
     - Bridge cost depends on bridge height, slope and the water depth crossed.
     - Tunnel cost depends on depth below the surface, slope and any water above.
     - For speed, only a low-discrepancy random subset S of these arcs is evaluated at each step.
  7. **Post-processing.**
     - Fit piecewise clothoids.
     - Segment the curve by its height above the terrain into tunnel/road/bridge (thresholds h_T, h_B).
     - Excavate and embank within a profile of widths r and R.
     - Instantiate procedural road, bridge and tunnel models.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - terrain heightfield;
  - water bodies (rivers, lakes) with water height;
  - vegetation (tree instances);
  - **start and end points**;
  - transfer-function curves and bridge/tunnel parameters.
- **Outputs:** one road trajectory, labelled road/bridge/tunnel, the modified terrain, and meshes.
- **Terrain:**
  - Anisotropic **slope cost** along the travel direction, with a **hard cut-off** (cost → ∞) above a maximum slope κ₀.
  - Curvature cost.
  - Tunnels when the maximum allowed slope is low.
- **Water:**
  - Shallow water is crossable at a cost (banking).
  - Deep water is infinite cost unless crossed by a costed bridge arc.
- **Numeric values stated:**
  - mask table (k, α, n_k);
  - stochastic example: rᵢ = 50 m, rₑ = 300 m on a 300×300 grid with 10 m spacing; #T = 2728 arcs vs #S = 50 sampled.
  - **No numeric maximum slope or grade** is given.

## 6. Evaluation

- **Qualitative, plus cost and timing tables.**
  - Table 2: effect of mask size on path cost (arbitrary units), time and length, on a 60×60 grid.
  - Table 3: with vs without curvature orientation.
  - Table 4: bridge enabled vs road only.
  - Table 5: exhaustive vs stochastic bridge masks.
  - Figures show switchbacks and tunnels appearing as the slope cost or threshold changes, and bridge vs detour as the water and forest costs change.
- **Not compared:** no real road, no metric of realism, no baseline beyond its own variants, no user study.
- **Runtime:** under 1 s on a 100×100 grid with curvature and bridge/tunnel detection on. Hardware is not stated.

## 7. Quotes for citation

- p. 429, Abstract: "The trajectory of the road minimizes a cost function that takes into account the different parameters of the scene including the slope of the terrain, natural obstacles such as rivers, lakes, mountains and forests."
- p. 432, §4.1: "If the slope is too steep, the transfer function should return infinity so as to prevent paths from climbing very steep slopes which would generate unrealistic roads."
- p. 433, §5.1: "For k = 1, the limited connectivity between grid points results in a shortest path following an unrealistic direct path to the top, ignoring the influence of the slope."
- p. 433, §5.1: "In general, chosing k = 5 proved to be a good compromise between speed and the angle resolution to generate realistic paths." (spelling as in the source)
- p. 434, §5.2: "Therefore, taking the curvature into account is crucial to obtain realistic paths."
- p. 437, §7: "Our method creates realistic trajectories and compares favorably in terms of efficiency and quality to existing editing and sketching techniques [MS09]."
- p. 438, §7: "In contrast, if the maximum authorized slope is very low, it is no longer possible to create a road going up the hills and the algorithm creates a tunnel (right image)."
- p. 438, §7: "Our experiments show that our optimized algorithm can generate realistic trajectories in less than 1 second when using a 100×100 grid with both curvature control and tunnel and bridge detection activated."
- p. 438, §8: "This work is the first step towards a solution to the much more complex and general problem of modeling a complete hierarchical road network connecting cities."

The page numbers are the PDF page plus 428. A script checked every quote against the PyMuPDF text of the PDF.

## 8. Comparability with UrbanMorph

- **Matched inputs.** Yes, for **connector routing between supplied points**.
  - *DEM:* 512 m tile on a 4 m grid gives 128×128 nodes, which is trivially fast.
  - *Water mask:* infinite cost except at permitted crossing cells, or a costed bridge arc.
  - *Entry points / anchors:* become start-end pairs.
  - *Per-class max grade:* becomes the slope cut-off κ₀, with one run per class.
- **Not a network generator.** A network needs all-pairs routing plus merging, which is the Galin 2011 extension.
- **Code.** None found (`docs/M18_BASELINES.md`).
- **Effort.**
  - *Reimplementation (M18_BASELINES estimate):* 20–35 h on `skimage.graph.MCP_Flexible`, plus 4–8 h for the matched protocol.
  - *My note:* the curvature state (position × orientation) is the costly part; dropping it gives a much simpler slope-only variant.
  - *Practical substitute:* `leastcostpath` `"wheeled transport"` with 16–48 neighbours (see `lewis2021_leastcostpath.md`).
- **Fairness.**
  - It optimises slope cost directly, so it will score well on grade metrics and poorly on morphology by construction.
  - Compare it only on anchor-connecting roads, with a matched tuning budget (a sweep of κ₀ and the cost weights).

## 9. Ideas we could adopt for evaluation

- **Terrain-response curves.** The paper shows qualitatively that switchbacks and tunnels appear as the maximum slope drops. We can sweep the per-class grade limit (or the potential weight) and plot two curves against it:
  - the grade distribution;
  - the sinuosity of anchor-to-anchor routes.
  Do the same for the real network on the same tiles (one point) and for the baselines. This turns "terrain adaptiveness" into a dose-response plot.
- **Route efficiency against the least-cost path.** For each pair of anchors, compute the ratio of generated route cost (or length) to the Galin-style least-cost path under the same cost. This separates "obeys terrain" from "takes needless detours".
- **Water crossings.** Report the count and length of bridge crossings per tile against OSM, since the bridge-vs-detour trade-off is explicit here.
- **Grid-artefact guard.** The limit-on-direction effect (§5) biases orientation and circuity statistics for grid paths. Any LCP-based baseline must use large neighbourhoods (k ≥ 3, i.e. 32 or more directions) or post-smoothing before computing bearing entropy or circuity.
