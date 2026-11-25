# Nishida, Garcia-Dorado, Aliaga 2016: Example-Driven Procedural Urban Roads

## 1. Header

- **Citation:** Gen Nishida, Ignacio Garcia-Dorado, Daniel G. Aliaga. "Example-Driven Procedural Urban Roads." *Computer Graphics Forum* 35(6):5–17, 2016 (published online 24 Sep 2015 per Crossref).
  - First names are from the M18 matrix and common usage; Crossref lists only initials (G. Nishida, I. Garcia-Dorado, D. G. Aliaga). The given names "Gen" and "Ignacio" are **not verified** on an opened page.
- **DOI:** 10.1111/cgf.12728
- **URLs opened:**
  - https://www.cs.purdue.edu/cgvlab/papers/aliaga/cgf15.pdf: author preprint, 14 PDF pages. Its template header reads "Volume 34 (2015), Number 2 pp. 1–13", which is not the final pagination. Read in full.
  - https://api.crossref.org/works/10.1111/cgf.12728: metadata.
- **Access level:** **FT** (preprint; pages below are preprint PDF pages).
- **Correction to `docs/M18_LITERATURE.md`.** The matrix lists the validation as "Qualitative + statistics comparison (n.v. in detail)". The full text has **no quantitative comparison between generated and example or real networks**. It reports only generation times and the "rate of example-based growth".

## 2. BibTeX

```bibtex
@article{nishida2016example,
  author  = {Nishida, G. and Garcia-Dorado, I. and Aliaga, D. G.},
  title   = {Example-Driven Procedural Urban Roads},
  journal = {Computer Graphics Forum},
  volume  = {35},
  number  = {6},
  pages   = {5--17},
  year    = {2016},
  doi     = {10.1111/cgf.12728}
}
```

## 3. What it does

An interactive tool grows a road network inside user-sketched land-use areas (water, flat land, mountain, park) on a terrain, in the *style* of example OpenStreetMap networks. It uses two kinds of growth:

- **Example-based growth** pastes "patches" cut from the example network (roundabouts, interchange clusters, intersections with their nearby edges).
- **Procedural growth** samples edge length and curvature statistics from the example, and applies terrain and water local constraints. It is used when no patch fits.

Arterials are grown first, then local streets are seeded along them. Warping, blending and interpolation create new styles. Blocks, parcels and buildings follow.

## 4. How it generates

- **Example-based plus procedural.**
  - There is no trained model, but it is data-driven: road geometry is copied from real OSM examples, and growth statistics (mean and variance of edge length and curvature) are measured from them.
  - Rules and constraints are hand-written.
- **Representation: vector.** A geometric graph whose edges are polylines.
- **Algorithm (§3–§4):**
  1. **Sketch.** The user paints land use and terrain height. Each land-use area becomes a target domain Ω_k with one or more example networks.
  2. **Split examples by class.** Arterials are OSM primary, secondary and tertiary (plus links). Local streets are residential and living_street (Table 1).
  3. **Patch extraction.**
     - Hough-detect circles; find tightly clustered vertices.
     - Expand each remaining intersection into a patch by following edges shorter than a threshold.
     - Mark "connector" edges (open in the patch but not in the example) and true dead ends.
  4. **Arterial growth** (breadth-first, FIFO queue of seeds; default one seed at the domain centroid).
     - For each dequeued vertex, choose a patch with probability ∝ exp(−min Hausdorff distance between the patch's connector edges and the vertex's edges) × a legality indicator.
     - Legality: all edges above water level; the terrain slope under the patch is within a threshold; no intersection with existing roads.
     - Paste the patch, removing redundant edges and merging near-coincident vertices.
  5. **Procedural fallback** when no patch is legal. Sample length and curvature from the statistics, with angles near 90°. Local constraints apply:
     - extend to and connect a nearby vertex or edge;
     - discard edges at overly acute angles;
     - a local street under water is discarded;
     - an arterial under water is extended across the river, with a limited reorientation, or discarded if it reaches the ocean;
     - **on steep slope, bend the segment perpendicular to the gradient; if too steep, remove it.**
  6. **Local streets.** Seed them at the example's local-street intersections along example-grown arterials, or at fragment boundaries along procedural arterials, and grow them the same way.
  7. **Optional post-process.** Iteratively remove dangling edges unless they are marked as dead ends in the example.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - land-use and terrain sketch (or an existing area);
  - one or more example OSM networks per domain;
  - optional seeds, warping guidelines and blending parameters.
- **Outputs:** road graph (arterial and local), blocks, parcels, 3D city.
- **Terrain** enters four ways:
  - a slope threshold in the patch legality check;
  - bending perpendicular to the gradient on steep ground, and removal when too steep;
  - the water level (patch rejection, river crossings by arterials only);
  - selection of examples by **average elevation** of the domain (Fig. 13).
- **No numeric thresholds** (slope, angle, merge distance) are given in the text.

## 6. Evaluation

- **Qualitative**, plus two tables:
  - **Table 2, generation time:** e.g. 278 km of road in 49 s; 342 km in 130 s. About 2 min on average for about 300 km on an Intel Core i7.
  - **Table 3, rate of example-based growth:** 91 % (Fig. 10a) down to 44 % (Fig. 13f). It falls on non-flat terrain or with multiple seeds.
- **Fig. 2 (motivation).** Amsterdam and Tel Aviv have similar summary statistics but look different, so "using only statistics cannot fully capture" style.
- **Not reported:** a metric comparing generated and example networks, a baseline comparison, or a user study.
- Largest result: 200 km² with over 3,500 km of roads.

## 7. Quotes for citation

- p. 1, Abstract: "procedural-based growth uses the statistical information of the source example while effectively adapting the roads to the underlying terrain and the already generated roads."
- p. 3, §2: "This implies that using only statistics cannot fully capture the complex details of the road styles."
- p. 7, §4.3.2: "check whether the underlying terrain slope of the patch is within a threshold, and"
- p. 7, §4.3.4: "If the road type is an arterial road and a part of the road segment is below water level, extend the road segment until it reaches the other side of the river."
- p. 7, §4.3.4: "If the underlying terrain slope is steep, the road segment will be bended to be perpendicular to the local terrain gradient. However, if the slope is too steep, the road segment will be removed (Figure 6d)."
- p. 9, §6: "All growing is performed on an Intel Core i7 CPU and takes 2 minutes on average for generating around 300 km of roads (Table 2)."
- p. 10, §6 (Discussion): "Even though the procedural-based growth uses some statistical features from the source examples, most of the interesting details are lost."
- p. 10, §6 (Discussion): "The rate becomes low when the underlying terrain is not flat or the road growth starts from multiple seeds."
- p. 12, §7: "Second, even though our local constraints consider the rivers and the slope of the mountains, there are still some poorly connected roads across rivers with sharp turning directions."

A script checked every quote against the PyMuPDF text of the preprint.

## 8. Comparability with UrbanMorph

- **Matched inputs.** Strong.
  - *Terrain and water:* our DEM tile is the target terrain, and our water mask sets the water level.
  - *Entry points:* become seeds.
  - *Classes:* the arterial → local hierarchy matches our class split.
  - *Grade:* the per-class max grade can serve as the slope threshold (the paper gives none, so that would be our documented choice).
- **Leakage guard.** Example networks must come only from **training cities**, never from the test tile's city. Otherwise patch copying reproduces the answer.
- **Code.** None found (`docs/M18_LITERATURE.md`: n.v.; no repository found in this pass either).
- **Effort (my estimate, labelled).** 40–70 h for a faithful reimplementation:
  - OSM parsing, class split and patch extraction (Hough circles, clustering, expansion);
  - Hausdorff patch selection with legality checks;
  - procedural fallback with terrain and water constraints;
  - local-street seeding;
  - dangling-edge post-process.
- **Fair comparisons:**
  - **full network growth** from entry points;
  - **completion given arterials**: §4.4 seeds local streets along supplied arterials. This is the closest classical analogue of our class-conditioned growth.

## 9. Ideas we could adopt for evaluation

- **Retrieval baseline (non-parametric).** Pick the training-city tile whose terrain statistics (mean elevation, slope histogram, water fraction) are closest to the test tile and paste its network. Fig. 13 does this with average elevation. It is a cheap, strong "copy a real place" baseline for realism, and tests whether our terrain adaptation beats retrieval.
- **Novelty / memorisation check.** Mirror Table 3's "rate of example-based growth": for each generated patch (e.g. ego-graph radius 100 m), measure the distance to the nearest training patch. Report the share that is near-copy vs novel, so that realism is not bought by memorisation.
- **Distributions, not only means.** Fig. 2 shows that two networks can share mean statistics and look different. Report full distributions with distances (Wasserstein, KS), plus structure-aware measures (local patch matching), not only aggregate means.
- **River-crossing quality.** The authors flag "poorly connected roads across rivers with sharp turning directions". Measure the angle between each crossing and the river axis, and the turn angle at bridge ends, for real vs generated.
