# J. Beneš, Wilkie, Křivánek 2014: Procedural Modelling of Urban Road Networks

## 1. Header

- **Citation:** Jan Beneš, Alexander Wilkie, Jaroslav Křivánek. "Procedural Modelling of Urban Road Networks." *Computer Graphics Forum* 33(6):132–142, 2014 (published online 12 Feb 2014 per Crossref).
- **DOI:** 10.1111/cgf.12283
- **URLs opened:**
  - https://cgg.mff.cuni.cz/~jaroslav/papers/2014-cities/2014-cities-paper.pdf: author preprint, 11 pages, footer "submitted to COMPUTER GRAPHICS Forum (1/2014)". Read in full.
  - https://cgg.mff.cuni.cz/?p=1672: project page. It links only a PDF and the DOI. No code; no supplementary material link was visible.
  - https://api.crossref.org/works/10.1111/cgf.12283: metadata.
- **Access level:** **FT** (preprint; page numbers below are preprint pages 1–11). The supplementary materials (parameter values, symbol list, comparison table) were **not** found or opened.

## 2. BibTeX

```bibtex
@article{benes2014procedural,
  author  = {Bene{\v{s}}, Jan and Wilkie, Alexander and K{\v{r}}iv{\'a}nek, Jaroslav},
  title   = {Procedural Modelling of Urban Road Networks},
  journal = {Computer Graphics Forum},
  volume  = {33},
  number  = {6},
  pages   = {132--142},
  year    = {2014},
  doi     = {10.1111/cgf.12283}
}
```

## 3. What it does

The paper simulates how a city's road network grows over time inside its regional context.

- Major roads are first routed over terrain between unsimulated **neighbouring cities** located beyond the map edge. Their intersections become growth nuclei.
- Minor roads then grow around the nuclei using a time-dependent street pattern.
- A traffic simulation between the neighbouring cities triggers new major roads when capacity runs out. New routes may reuse minor roads, avoid built-up areas, or use waterways and harbours.

## 4. How it generates

- **Procedural simulation.** User-set "historical context" functions and coefficients. Nothing is fitted to data.
- **Representation: vector.** A planar segment graph G = (J, S). Routing uses a lattice pathfinding graph.
- **Algorithm (§3):**
  1. **Pathfinding graph** on a lattice with spacing ℓ = 20 m. Vertices are classed land, water or forbidden.
     - Land edges join vertices within lattice radius r = 8.
     - An edge is **discarded** if it crosses water or forbidden areas, or if its maximum slope E_L,max exceeds a cut-off E_L,cut.
     - Otherwise its weight is LENGTH(e) · max(1, O₁ · E_L,max).
     - Water edges are added, plus harbour edges (land↔water) and bridge edges (as in Galin 2010).
  2. **Neighbouring cities.**
     - Their number c is drawn from a binomial (c_min = 4, c_max = 10, expected 7).
     - They are placed on the map border along c jittered rays from the map centre.
     - City pairs are pruned by angle while keeping the initial network connected, then permuted.
  3. **Initial network.**
     - Each kept pair is connected by Dijkstra. Edge weights along each found path are then multiplied by a reuse coefficient u ∈ (0, 1] to encourage reuse.
     - Paths are converted into planar segments (trade routes).
     - Nuclei: nodes of degree ≥ 2 (as written) plus bridge and harbour nodes.
  4. **Main loop** until t_max:
     - **Grow minor roads** from the first G(t) candidate nodes. Candidates are ranked by distance to the nearest nucleus × (1 − strength), and use the pattern K(t). The minor-road generator is P&M / Weber et al.
     - **Traffic.**
       - Segment capacity is O₂ · WIDTH − T.
       - If a city pair's routes cannot carry t(t), propose two options: (a) a route over existing major roads, or (b) a route through G′. G′ is the pathfinding graph minus the built-up area B, plus existing segments and "stitches".
       - Edge-type weights C₁…C₆ apply, and turns are penalised by P(t, α).
       - Keep the option with the lower O₃·LENGTH + O₄·COST, and widen it.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - a terrain elevation map classed into land, water and forbidden areas;
  - neighbouring cities (auto-generated);
  - control functions t(t), G(t), R(t);
  - the minor-pattern sequence K(t);
  - coefficients O₁–O₄.
- **Outputs:** the road network over time, with major and minor classes and block "age". No buildings.
- **Terrain:**
  - **Hard slope cut-off** on lattice edges: E_L,max > E_L,cut is discarded. The value of E_L,cut is not in the main text; parameter values are said to be in the supplementary materials, which I did not open.
  - **Linear slope penalty:** max(1, O₁ · slope) multiplies edge length.
- **Water:**
  - Land edges over water are discarded.
  - Bridge edges cross water.
  - A water graph with harbours models water transport. Water elevation is constant, with no slope cut-off on water edges.
- Minor-road growth inherits P&M local constraints. No terrain rule for minor roads is described beyond that.

## 6. Evaluation

- **Qualitative only.** Figs. 11–12 show city types: eccentric, normal, coastal, satellite suburbs, a mountainous region (60 m contour quantisation), and a lake-side city with a harbour.
- **Not reported:** a metric, a comparison with a real network, a baseline, or a user study.
- **Runtime:** 8×8 km maps at ℓ = 20 m took 2.5–4.5 min on one core of an i7-2600K (3.4 GHz), with the pathfinding graph built in CUDA on a GTX 480.
- Stability is claimed informally (small parameter changes give small changes in appearance).

## 7. Quotes for citation

- p. 1, Abstract: "The main idea of our paper is that a city cannot be meaningfully simulated without taking its neighbourhood into account."
- p. 2, §3.1: "a terrain elevation map m, classified into land, water, and forbidden areas"
- pp. 3–4, §3.2.2 (the sentence breaks across the page, with the Fig. 3 caption in between): "If an edge runs over water, forbidden areas, or has EL,max greater than a cut-off constant EL,cut, we discard it."
- p. 4, §3.2.2: "This heuristic works well in practice."
- p. 8, §3.5: "In our experience, the system is stable and minor changes in parameter values cause only minor changes to the result's generic appearance."
- p. 8, §4: "The generation times for 8 × 8km maps for a lattice L with a spacing of ℓ= 20m were between 2.5 and 4.5 minutes running on a single core of an Intel i7-2600K 3.4GHz CPU, with the construction of the pathfinding graph G run on a GeForce 480 GTX card."
- p. 8, §4: "For reproducibility, the parameter values for all presented results, together with images of all simulation steps, are available in the supplementary materials."
- p. 9, §4: "This results in a relatively non-deterministic, but plausible major road structure with ring-roads (Figures 11b, 12d, 12e) and by-passes (Figure 11a)."
- p. 10, §4.1: "First, our prototype implementation is not real-time."

A script checked every quote against the PyMuPDF text of the preprint. Subscripts render inline as "EL,max" and "EL,cut".

## 8. Comparability with UrbanMorph

- **Scale.** The simulation is designed for 8×8 km. Our 512 m tile is only about 26×26 lattice cells at 20 m, too small for the traffic and trade dynamics.
- **Usable part: the major-road lattice router.** A hard slope cut-off plus a linear slope penalty plus water bridges is a clean, two-parameter terrain-routing baseline.
  - *Entry points:* the neighbouring-city vertices on the map edge.
  - *Water mask and permitted crossings:* bridge edges.
  - *Classes:* major roads only.
  - *Grade:* our per-class max grade is E_L,cut.
- **Code.** None (project page lists only the PDF and DOI; `docs/M18_BASELINES.md` agrees).
- **Effort (M18_BASELINES estimates):** full reimplementation with the traffic simulation 60–100 h; major-road lattice only 10–15 h.
- **Fair comparisons:**
  - **connector routing between entry points** (major roads);
  - full network growth only on a much larger window than our tile.

## 9. Ideas we could adopt for evaluation

- **A minimal terrain-cost ablation.** The cost LENGTH · max(1, O₁ · slope) with a hard cut-off is about the simplest defensible terrain model. Use it as a reference router, and as a sanity check that our learned terrain potentials beat "length × slope penalty" on realism at equal grade compliance.
- **Junction siting.** Nuclei are placed at intersections. Report the slope distribution at generated junctions vs real junctions on the same DEM, since junctions on steep ground are rare in reality.
- **Path reuse.** The reuse coefficient u controls trunk sharing. Report the share of anchor-to-anchor shortest paths that share their first or last k metres, or edge betweenness concentration, for real vs generated.
- **Stability claim → seed variance.** The paper says small parameter changes give small changes in appearance. We should *quantify* run-to-run variance (metric spread over seeds and particles) for our sampler and for every baseline.
