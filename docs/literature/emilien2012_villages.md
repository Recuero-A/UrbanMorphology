# Emilien, Bernhardt, Peytavie, Cani, Galin 2012: Procedural Generation of Villages on Arbitrary Terrains

## 1. Header

- **Citation:** Arnaud Emilien, Adrien Bernhardt, Adrien Peytavie, Marie-Paule Cani, Eric Galin. "Procedural Generation of Villages on Arbitrary Terrains." *The Visual Computer* 28(6–8):809–818, 2012 (published online 18 Apr 2012 per Crossref).
- **DOI:** 10.1007/s00371-012-0699-7
- **URLs opened:**
  - https://perso.liris.cnrs.fr/egalin/Articles/2012-villages.pdf: timed out on 2026-11-25.
  - HAL search API (title query): record hal-00694525 at https://inria.hal.science/hal-00694525. The /document link is behind the same anti-bot challenge as hal.science.
  - http://web.archive.org/web/20250808131845/https://inria.hal.science/hal-00694525/document: Wayback copy of the HAL author manuscript (11 PDF pages; page 1 is the HAL cover; manuscript received 2012-02-17, accepted 2012-03-24). Read in full.
  - https://api.crossref.org/works/10.1007/s00371-012-0699-7, plus the Semantic Scholar and OpenAlex APIs for the same DOI: metadata.
- **Access level:** **FT** (author manuscript; pages below are PDF pages of the HAL file).

## 2. BibTeX

```bibtex
@article{emilien2012procedural,
  author  = {Emilien, Arnaud and Bernhardt, Adrien and Peytavie, Adrien and Cani, Marie-Paule and Galin, Eric},
  title   = {Procedural Generation of Villages on Arbitrary Terrains},
  journal = {The Visual Computer},
  volume  = {28},
  number  = {6--8},
  pages   = {809--818},
  year    = {2012},
  doi     = {10.1007/s00371-012-0699-7}
}
```

## 3. What it does

The paper generates small European-style villages (hamlets, fortified and fishing villages) on arbitrary terrain. Settlement and roads grow together:

- buildings are seeded stochastically where terrain-dependent "interest" is high;
- each new building is immediately connected to the road network by a terrain-aware shortest path that reuses existing roads cheaply and occasionally closes cycles.

Parcels are then grown anisotropically from the roads, and houses are built with an "open shape grammar" that adapts doors and windows to the slope.

## 4. How it generates

- **Procedural and stochastic.** About 150 hand-set parameters per village type, stored in a "building encyclopaedia". Nothing is learned.
- **Representation.**
  - *Roads:* vector node sequences, from grid shortest paths.
  - *Parcels:* grown on a grid (4096×4096), then simplified to polylines.
- **Algorithm (§3–§5):**
  1. **Growth scenario.** A user list of events over time: change village type, or seed n buildings of type B.
  2. **Seeding.** Pick a random position p and check constructibility. Compute the interest I(B):
     - I = 0 if any criterion fᵢ = −1;
     - otherwise I = max(0, Σ wᵢ fᵢ).
     The criteria are:
     - sociability and worship (attraction–repulsion to other buildings);
     - accessibility (preferred distance to the nearest road);
     - **slope** (bell-shaped preference, with prohibited ranges);
     - water distance;
     - fortification;
     - geographical domination (height relative to nearby samples).
     Accept the position with a probability that depends on I; otherwise resample.
  3. **Connect.** Immediately connect the new building to the network or to the boundary connection points Ψ by a shortest path "similar to" Galin 2010.
     - Cost C(R) = Σ wⱼ gⱼ(R) over slope, curvature and water costs, plus a high constant for crossing a building.
     - Existing road segments cost w_ex · C with **w_ex ≪ 1** (road reuse).
  4. **Cycles.** Look for the closest road node within a cone of angle θ ahead of the building. Build a second road there if it is not too close to the first one.
  5. **Parcels.** Road conquest, corner conquest, then anisotropic grid conquest with a slope cost quadratic in the directional gradient; then simplification and footprints.
  6. **Houses.** Open-shape-grammar houses adapted to the slope.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - region Ω with **connection points Ψ on its boundary** (the extremities of the future road network);
  - elevation h(p), water height w(p), vegetation density v(p);
  - village type, growth scenario and encyclopaedia parameters.
- **Outputs:** road network, building seeds, parcels, footprints, 3D houses.
- **Terrain enters in five places:**
  - slope, curvature and water costs in road routing;
  - slope interest for building siting (preferred slope per building type, prohibited outside a range);
  - geographical domination: f(x) = Σ (h(x) − h(p)) / (1 + ‖x − p‖²) over samples within radius r;
  - an anisotropic slope cost for parcel growth;
  - water as a cost and as an attractor (fishing villages).
- **No numeric slope or grade thresholds** are given.

## 6. Evaluation

- **Parcels (quantitative, on one real layout).**
  - Generated parcels were compared with a real village layout with "similar building distributions and road networks" (Fig. 12).
  - Mean number of neighbours: 2.873 vs 2.812 real.
  - Mean number of contour edges: 4.29 vs 4.068 real.
  - The real site is not named in the text I read; no variance or test is given.
- **Roads:** **not quantitatively evaluated.** There is a visual comparison of a real and a generated highland hamlet (Fig. 14).
- **Runtime (Table 1):**
  - skeleton (roads and seeds): 4–7 min;
  - parcels: 7–13 min;
  - geometry: 20–30 s.
  - Hardware is not stated.
- **Limitations:** 150 user parameters per village type, found by trial and error.

## 7. Quotes for citation

- PDF p. 2, §1: "Adaptation to rough terrain has been scarcely studied, and no previous work was conducted, to our knowledge, on the generation sparser settlements." (grammar as in source)
- PDF p. 4, §3: "through a set of connection points Ψ, located at the edges of Ω, and that will serve as extremities for the future road network."
- PDF p. 5, §4.2: "We use a bell-shaped function depending on building type to express preference to a given slope value, and to prohibit construction outside of a given slope range."
- PDF p. 6, §4.3: "To prevent the method from generating a fully ramified road networks, we note that, as in real life, the less costly connection to the existing network should be looked for (road re-use coming for free)."
- PDF p. 6, §4.3: "Real road networks often include cycles providing shortcuts."
- PDF p. 9, §7: "Parcels have similar shapes and the mean number of neighbors (2.873 with our model, 2.812 in the real data) and of contour edges (4.29 with our model, 4.068 in the real data) are similar."
- PDF p. 9, §7: "A complete comparison and validation is beyond the scope of this paper."
- PDF p. 10, §7: "The main limitation of our method is the number of user-set parameters, currently 150 per village type."
- PDF p. 10, §8: "We validated our results through comparison with real layouts and pictures."

A script checked every quote against the PyMuPDF text of the HAL PDF.

## 8. Comparability with UrbanMorph

- **Matched inputs.** Very good.
  - The boundary connection points Ψ are exactly our entry points.
  - The DEM and water mask go in directly.
  - Grade limits could become the slope cost cut-off (our addition).
  - Road classes are not modelled: there is one road type.
- **Different generative driver.** Roads exist only to serve seeded buildings. A comparison needs a building-count budget (e.g. matched to the real building count, or to the real road length, on the tile).
- **Most relevant for steep, sparse sites** (hamlets and hill towns, our Machu Picchu-type benchmarks) rather than dense urban grids.
- **Code.** None found (`docs/M18_BASELINES.md`).
- **Effort (my estimate, labelled).** Road-and-seeding core only, using an existing LCP engine: about 25–40 h. That covers seeding with slope and accessibility interest, connection with a reuse discount, and cone-based cycles. Parcels and houses are not needed.
- **Fair comparison:** **full network growth from boundary entry points**, at a matched budget, on steep sparse tiles.

## 9. Ideas we could adopt for evaluation

- **Siting on gentle land, made explicit.** The paper encodes siting as a slope preference per building type. We can report the slope distribution under generated street length (or junctions) against real streets on the same DEM, and against the slope distribution of all land in the tile (availability). A selection ratio (used/available per slope bin) is easy to read.
- **Road reuse and trunk sharing.** The w_ex ≪ 1 reuse discount produces tree-like trunks. Report the share of network length on the main component's bridges (cut edges), or a "dendricity" measure, for real vs generated.
- **Block-level shape statistics on real layouts.** Their neighbour and contour-edge comparison is a cheap block-level realism check. Our block statistics (Louf–Barthelemy Φ, area) already cover this; adding mean neighbour count per block is trivial.
- **Cycle creation.** Cycles are added deliberately because "real road networks often include cycles". This matches our closed-island and dead-end potentials; report cycle density (circuit rank per km²) for real vs generated.
