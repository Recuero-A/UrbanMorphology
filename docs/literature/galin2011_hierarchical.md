# Galin, Peytavie, Guérin, Beneš 2011: Authoring Hierarchical Road Networks

## 1. Header

- **Citation:** Eric Galin, Adrien Peytavie, Eric Guérin, Bedřich Beneš. "Authoring Hierarchical Road Networks." *Computer Graphics Forum* 30(7):2021–2030, 2011.
  - That it is the Pacific Graphics 2011 issue is stated in the M18 matrix and in the reference list of Emilien et al. 2012. I did not verify it on a publisher page.
  - This is the *Bedřich* Beneš paper. Jan Beneš 2014 is a different paper (`benes2014_urbanroads.md`).
- **DOI:** 10.1111/j.1467-8659.2011.02055.x
- **URLs opened:**
  - https://perso.liris.cnrs.fr/eric.guerin/download/pg2011.pdf: timed out on 2026-11-25.
  - https://hal.science/hal-01354487/document: returned an anti-bot challenge.
  - http://web.archive.org/web/20240414021413/https://hal.science/hal-01354487/document: Wayback copy of the HAL author version (11 PDF pages; page 1 is the HAL cover). Read in full.
  - HAL search API (title query): record hal-01354487.
  - https://api.crossref.org/works/10.1111/j.1467-8659.2011.02055.x, https://api.semanticscholar.org/graph/v1/paper/DOI:10.1111/j.1467-8659.2011.02055.x and https://api.openalex.org/works/doi:10.1111/j.1467-8659.2011.02055.x: metadata and open-access locations.
- **Access level:** **FT** (author version; page numbers below are PDF pages of the HAL file, not journal pages).

## 2. BibTeX

```bibtex
@article{galin2011authoring,
  author  = {Galin, Eric and Peytavie, Adrien and Gu{\'e}rin, Eric and Bene{\v{s}}, Bed{\v{r}}ich},
  title   = {Authoring Hierarchical Road Networks},
  journal = {Computer Graphics Forum},
  volume  = {30},
  number  = {7},
  pages   = {2021--2030},
  year    = {2011},
  doi     = {10.1111/j.1467-8659.2011.02055.x}
}
```

## 3. What it does

The method builds a **three-level inter-city road network** (highways, major roads, minor roads) connecting user-placed cities, towns and villages over large terrains. It has three stages:

1. Anisotropic least-cost paths from Galin 2010 between pairs of settlements.
2. A **proximity graph under the terrain-cost (geodesic) metric** keeps only some of those paths; a parameter γ controls density.
3. Nearby path parts are **merged** using a discrete Fréchet distance, which creates junctions (Steiner nodes).

Geometry for roads, bridges, tunnels, crossings and interchanges is then instantiated. City streets are explicitly out of scope.

## 4. How it generates

- **Procedural.** Nothing is fitted to data.
- **Representation: vector.** A graph whose edges store the road paths, with nodes for cities and junctions.
- **Algorithm (§3–§7):**
  1. **Settlements and road types.** Settlements get a size class σ (city 2, town 1, village 0). The road type between i and j is τᵢⱼ = min(σᵢ, σⱼ).
  2. **Level order.** Levels run from highways down to minor roads (§4).
  3. **Control cost.** Compute a control cost h(p) as a sum of compactly supported primitives (points, curves, areas):
     - a penalty near previously built roads;
     - user-painted regions (e.g. parks);
     - city areas that highways must not cross.
     Total cost: c = v + w + s + g + h (vegetation, water height, slope along the road, curvature, control).
  4. **Geodesic paths.** Compute anisotropic least-cost paths between settlement pairs. A heuristic restricts this to the 20 nearest neighbours.
  5. **Proximity filter.** Keep edge (a, b) only if no settlement p lies in its "neighborhood region" Ω_γ(a, b), where d is the path cost.
     - γ = 2 gives a Gabriel graph in the terrain metric.
     - γ → ∞ gives the relative neighbourhood graph.
     - Smaller γ gives denser networks.
     - **Typesetting caveat.** In the author version (PDF p. 8) Ω_γ is printed as {p : d(a,b)^γ < d(a,p)^γ + d(b,p)^γ}, and Ω_∞ as {p : d(a,b) < max(d(a,p), d(b,p))}. The same paragraph describes Ω_∞ as "the intersection of the two balls" of radius d(a,b) around a and b, which is the complement of the printed set. I follow the textual description (the standard Gabriel and RNG definitions); the printed inequality looks reversed.
  6. **Merge.** Merge close portions of paths using the discrete Fréchet ("coupling") distance, inserting Steiner nodes as junctions. Merge rules depend on road type: no merging with toll highways; the merged type is max(τᵢ, τⱼ).
  7. **Waypoints.** Optional persistent and road-specific waypoints.
  8. **Geometry.** Rule-based model selection, then excavation and embankment, then meshes.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - terrain with water and vegetation maps (procedural, painted, or real data);
  - a set of settlements with sizes;
  - optional influence maps, primitives and waypoints;
  - the γ parameter.
- **Outputs:** hierarchical road graph and 3D geometry, including tunnels, bridges and interchanges.
- **Terrain** enters through the Galin 2010 cost terms (slope in the road direction, curvature, water height, vegetation). It defines a non-Euclidean metric used both for routing and for the proximity graph.
- **No numeric slope threshold** is stated.
- **Scale:** the examples are 128×128 km² on a 256×256 grid (Fig. 14) and 200×200 km² on grids of 128–512 (Table 2). That is about 0.4–1.6 km per grid cell, versus our 4 m.

## 6. Evaluation

- **Network statistics.** Table 1: number of roads and network length per level for γ = 1.2, 2.0 and 8.0.
- **Timings.**
  - Table 2: 23.2–625.3 s for grid sizes 128 to 512 on a 200×200 km² terrain.
  - Table 3: 89–147 s on a 256×256 grid.
  - Hardware: Core i7 at 3 GHz with 8 GB RAM.
- **Real-network comparison: visual only.** Corsica real vs synthesised (Fig. 19), generated from 10 cities and 39 towns.
  - A coefficient λ "was found after a few trial and error steps"; λ is not defined elsewhere in the text I read.
  - Mismatches are attributed to land use and infrastructure policy.
  - A few persistent waypoints bring the result close to reality.
- **Not reported:** a metric for the Corsica match, a baseline, or a user study.

## 7. Quotes for citation

- PDF p. 2, Abstract: "We present a procedural method for generating hierarchical road networks connecting cities, towns and villages over large terrains."
- PDF p. 3, §1.1: "While these techniques are effective for city streets, they cannot be generalized to countryside roads whose patterns and layouts are fundamentally different and heavily influenced by the terrain geometry."
- PDF p. 4, §3: "Note that our paper does not address the generation of the city street network."
- PDF p. 4, §3: "Therefore, the proximity graphs based on the Euclidean distance metric do not provide an effective measure to generate realistic network patterns (Figure 5)."
- PDF p. 5, §4: "Having two roads going close to each other is not a casual case, therefore, we merge these parts by evaluating the non-Euclidean Fréchet distance to simplify the geometric graph."
- PDF p. 8, §5.2: "In our experiments, using the 20 nearest neighboring cities proved to be enough to yield the same proximity graph as the one produced with the initial complete graph."
- PDF p. 10, §8: "We carried out several experiments to compare our synthesized road networks with real ones."
- PDF p. 10, §8: "The coefficient λ was found after a few trial and error steps."
- PDF p. 10, §8: "The differences between our results and reality often occur in urbanized regions where some roads were created to connect many small villages and towns along the coast, probably because of the fact that a purely geometric algorithm does not take into account land use policies and infrastructure development."
- PDF p. 10, §8: "Nevertheless, our method can produce large road networks within times ranging from a few seconds to a couple of minutes (Table 3)."

A script checked every quote against the PyMuPDF text of the HAL PDF.

## 8. Comparability with UrbanMorph

- **Scale mismatch.** Inter-city networks on grids of hundreds of metres per cell, versus our 512 m tile.
- **Reusable core: anchors → network.** The pipeline (least-cost paths between anchors, γ-proximity filter in the terrain metric, Fréchet merging into junctions) is a principled "network from supplied points" baseline. It beats a plain union of all-pairs least-cost paths.
  - *Settlements:* our boundary entry points and interior anchors.
  - *Classes:* our road classes.
  - *Grade:* per-class max grade becomes the slope cut-off.
  - *Water:* barrier plus permitted crossings.
- **Code.** None found (`docs/M18_BASELINES.md`).
- **Effort (M18_BASELINES estimate):** +30–50 h on top of a Galin 2010 reimplementation. Using `leastcostpath` LCPs as the path engine would cut that (my estimate: 15–25 h for the proximity filter plus merging).
- **Fair comparison:** **connector routing among supplied points (trunk network)**. Not local-street infill.

## 9. Ideas we could adopt for evaluation

- **Matched-budget comparison.** γ gives a one-knob density control. Sweep it until total network length matches the real tile, then compare the other statistics at matched length. Apply the same idea to any baseline with a density knob, so that realism differences are not just length differences.
- **Route-level similarity.** Use the discrete Fréchet distance between each generated anchor-to-anchor route and the real OSM route between the same anchors. It complements node- and edge-level statistics.
- **Error localisation.** The paper found that mismatches concentrate where land-use drivers dominate. Stratify our real-vs-generated errors by terrain class (flat vs steep) and by distance to water, rather than reporting only means.
- **Junction creation by merging.** Count generated junctions per km of road against real, since merging is the mechanism that creates junctions outside cities here.
