# Chen, Esch, Wonka, Müller, Zhang 2008: Interactive Procedural Street Modeling

## 1. Header

- **Citation:** Guoning Chen, Gregory Esch, Peter Wonka, Pascal Müller, Eugene Zhang. "Interactive Procedural Street Modeling." *ACM Transactions on Graphics* 27(3), Article 103, 10 pages, August 2008 (SIGGRAPH 2008).
- **DOI:** 10.1145/1360612.1360702. The SIGGRAPH 2008 papers volume has a second DOI, 10.1145/1399504.1360702 (Crossref).
- **URLs opened:**
  - https://web.engr.oregonstate.edu/~zhange/images/street_sig08.pdf: author PDF, 10 pages. Read in full. Its first page carries the ACM reference format with "Article 103".
  - https://www.sci.utah.edu/~chengu/street_sig08/street_project.htm: project page. It states that source code and a short manual are available and links http://www.sci.utah.edu/~chengu/street/streetmodeling.zip. I did not download the zip.
  - https://api.crossref.org/works/10.1145/1360612.1360702 and https://api.crossref.org/works/10.1145/1399504.1360702: metadata.
- **Access level:** **FT**.

## 2. BibTeX

```bibtex
@article{chen2008interactive,
  author    = {Chen, Guoning and Esch, Gregory and Wonka, Peter and M{\"u}ller, Pascal and Zhang, Eugene},
  title     = {Interactive Procedural Street Modeling},
  journal   = {ACM Transactions on Graphics},
  volume    = {27},
  number    = {3},
  articleno = {103},
  numpages  = {10},
  year      = {2008},
  month     = aug,
  doi       = {10.1145/1360612.1360702}
}
```

## 3. What it does

Street networks are modelled by first designing a 2D **tensor field** and then tracing streets along its two perpendicular eigenvector directions (hyperstreamlines). The user combines several kinds of building block:

- basis fields (grid, radial);
- fields derived from water boundaries and from the **height map**;
- brush strokes, smoothing, and rotation and noise fields.

Major roads are traced first. Minor roads are traced inside the regions the major roads cut out. The resulting graph can then be edited, and 3D geometry is generated.

## 4. How it generates

- **Procedural and interactive.** No learning. The field design is manual, or uses automatic boundary and height-field elements.
- **Representation: vector.** The street graph G = (V, E):
  - nodes are the intersection points of major and minor hyperstreamlines;
  - edges are the streamline pieces between consecutive intersections.
- **Algorithm (§3, §5, §6):**
  1. **Inputs.** Load the water map W, park/forest map F, height map H and population map P, each on a 512×512 grid.
  2. **Build the tensor field.**
     - Each design element defines a global basis field. Elements are blended with Gaussian radial weights (Eq. 3).
     - *Boundary field:* regular elements placed along the water-boundary polyline.
     - *Height-field field:* the tensor whose **minor eigenvector matches ∇H** everywhere, with magnitude |∇H| (§5.1). It follows that the major eigenvector, and hence the major hyperstreamlines, run perpendicular to the gradient, i.e. along contour lines. That last step is my inference from the definition; the paper does not say "contour".
     - Optional extras: Laplacian smoothing, brush regions, and rotation fields R1–R3 including Perlin noise.
  3. **Seed.** Seeds go into a priority queue, weighted by distances to water boundaries, degenerate points and population centres. Narrow passages get higher priority.
  4. **Trace evenly spaced hyperstreamlines** (modified Jobard–Lefer):
     - major and minor tracing alternate, with spacing d_sep (density map P can modulate d_sep);
     - adaptive Runge–Kutta integration;
     - a streamline stops at the domain boundary, a degenerate point, a loop, a maximum length, or when it comes closer than d_sep to another streamline;
     - tracing continues a further d_lookahead to find an intersection;
     - tracing may cross **narrow water** to form a bridge, depending on bridge length and the angle to the coastline.
  5. **Build the graph.** Intersect the major and minor streamlines to obtain G.
  6. **Minor roads.** The major graph plus topographic boundaries split the domain into regions. Each region gets its own (possibly discontinuous) field, and the same tracing is run inside it.
  7. **Editing.** Segment/vertex edits, re-seeding, street displacement, "graph noise" deletions using Halton sampling, and local regeneration inside regions of an existing network (e.g. from Google Maps), reconnected by tracing forward.

## 5. Inputs, outputs and how terrain enters

- **Inputs:** binary water map W, binary park/forest map F, height map H, population-density map P (each a 512×512 grid in their implementation), plus user design elements and seeds.
- **Outputs:** major and minor street graph (attributes such as width and type are stored on nodes and edges) and 3D geometry.
- **Terrain enters only as a direction field.** The height-field tensor biases street *orientation* toward the contours. There is **no hard grade limit and no slope cost.**
- **Water** enters three ways:
  - as boundary-aligned basis fields;
  - as region boundaries for minor roads;
  - as a barrier that only narrow crossings ("bridges") may cross.
- **No numeric thresholds** are given for slope, bridge length or crossing angle.

## 6. Evaluation

- **Qualitative only.**
  - Generated networks are shown for downtown Taipei, Portland (Willamette River), Manhattan, and a fictional city on a stretch of the Benue River.
  - Real map snippets are shown as *inspiration*, not as quantitative targets.
- **Reported:** modelling time. A main layout takes about five minutes, plus 30–60 minutes of fine-tuning.
- **Not reported:** metrics, baselines, user study, or comparison with a real network on the same terrain.

## 7. Quotes for citation

- p. 103:2, §2: "However, the most informative resources are internet based map services, as we try to match street patterns and do not attempt to simulate their formation."
- p. 103:2, §3: "The input to our system includes four maps loaded as images: 1) a binary valued water map W, 2) a binary valued park and forest map F, 3) a height map H, and 4) a population density map P."
- p. 103:4, §5.1: "The natural elevation is an important constraint for most road construction. We observe that roads are often built by taking into account the gradient of the height field."
- p. 103:6, §6.1: "We also allow the tracing to cross relatively narrow water regions to form bridges depending on the required length of the bridge and the angle of the intersection with the coastline."
- p. 103:6, Fig. 9 caption: "Notice that with our approach the street graph has fewer dangling edges."
- p. 103:7, §6.3: "In this case local topography dominates the road layout."
- p. 103:8, §7: "Note that our goal is to generate maps inspired by real world maps, but not to exactly replicate the existing cities."
- p. 103:8, §7: "In our experiments, a city with reasonable complexity can be modeled within five minutes, such as the fictional city in Figure 1, and the cities in Figures 16 and 17 took about five minutes for the main layout, but required an additional thirty to sixty minutes to fine tune the details and to experiment with different designs."
- p. 103:8, §8: "Currently, our system only assumes a single-level spatial resolution, which makes it difficult to modify the tensor field at significantly different scales."

A script checked every quote against the PyMuPDF text of the PDF.

## 8. Comparability with UrbanMorph

- **Matched inputs.** Good fit.
  - *DEM:* our 512 m / 4 m DEM is H.
  - *Water mask:* our mask is W. F is empty.
  - *Density:* P is uniform, or a proxy shared by all methods.
  - *Entry points:* become user seeds (seed weight can be forced).
  - *Classes:* major and minor give two classes.
  - *Grade:* there is no grade limit. Applying our per-class max-grade rule would be our addition.
- **Main fairness threat: the designer-in-the-loop.** A faithful automatic baseline needs a fixed, documented recipe, fixed for all tiles, for example:
  - the height-field field;
  - plus the water-boundary field;
  - plus one global grid element;
  - with decay d and spacing d_sep.
- **Code and licence.**
  - The official C++ source exists: MFC / Visual Studio 2005 GUI, per the project page.
  - Per `docs/M18_BASELINES.md` (not re-checked here) it has no licence file, carries "copyright @2007 Guoning Chen", bundles Numerical Recipes headers, and is Windows-only.
  - The third-party `ProbableTrain/MapGenerator` (LGPL-3.0) has only grid and radial basis fields, with no height-field basis (per M18_BASELINES).
- **Effort (M18_BASELINES estimates):** port the core 40–80 h; Python reimplementation 30–50 h; MapGenerator plus a height-field basis 20–35 h.
- **Fair comparisons:**
  - **completion given arterials**: minor streets traced in regions bounded by supplied major roads, which is the paper's own §6.2 workflow;
  - **full network growth** with the fixed recipe above.
  - Not connector routing.

## 9. Ideas we could adopt for evaluation

- **A "contour oracle" reference.** A tensor field built only from ∇H is a pure terrain-following generator. Its contour-alignment and grade scores give an *upper reference* for how much terrain-following is achievable without realism. Real OSM and our model can then be placed between that reference and a terrain-blind baseline.
- **Contour-alignment metric** that operationalises the paper's construction. For each street segment, measure the angle between its tangent and the local contour direction (the direction perpendicular to ∇H). Weight by length, and only count cells where |∇H| is above a threshold, because the direction is meaningless on flat land. Report this per road class.
- **Dangling-edge count.** The paper's own visual criterion (Fig. 9) is directly a dead-end statistic, which matches our closed-dead-end potentials.
- **Narrow-water bridging.** Report the length distribution of generated water crossings against real ones on the same water mask.
