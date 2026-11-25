# Lewis 2021: Probabilistic Modelling for Incorporating Uncertainty in Least Cost Path Results (leastcostpath methodology)

## 1. Header

- **Citation:** Joseph Lewis. "Probabilistic Modelling for Incorporating Uncertainty in Least Cost Path Results: a Postdictive Roman Road Case Study." *Journal of Archaeological Method and Theory* 28(3):911–924, 2021 (accepted 18 March 2021, published online 27 March 2021). Open access, CC BY.
  - This is the paper the `leastcostpath` CRAN DESCRIPTION cites as its methodological reference ("(Lewis, 2021) <doi:10.1007/s10816-021-09522-w>").
  - Its preprint was recommended by PCI Archaeology (SocArXiv mxas2; recommendation doi 10.24072/pci.archaeo.100005, per the author homepage).
- **DOI:** 10.1007/s10816-021-09522-w
- **URLs opened:**
  - https://link.springer.com/content/pdf/10.1007/s10816-021-09522-w.pdf: returned an HTML bot page, not the PDF.
  - https://www.repository.cam.ac.uk/handle/1810/327575 and https://www.repository.cam.ac.uk/bitstreams/59b08c39-59fd-4fe7-942d-87576a414d0c/download: Cambridge Apollo copy of the publisher PDF (14 pages, printed pp. 911–924). Read in full.
  - https://api.crossref.org/works/10.1007/s10816-021-09522-w and https://api.openalex.org/works/doi:10.1007/s10816-021-09522-w: metadata and OA locations.
  - https://cran.r-project.org/package=leastcostpath (→ https://cran.r-project.org/web/packages/leastcostpath/index.html): version 2.0.13, published 2025-06-09, GPL-3, CRAN DOI 10.32614/CRAN.package.leastcostpath, arm64 macOS binaries listed.
  - https://cran.r-project.org/web/packages/leastcostpath/refman/leastcostpath.html: reference manual (`create_slope_cs`, `PDI_validation`, `buffer_validation`, `add_dem_error`).
  - https://raw.githubusercontent.com/josephlewis/leastcostpath/master/README.md and .../DESCRIPTION: README (case studies, citation "Lewis, J. (2023) leastcostpath…") and DESCRIPTION.
  - https://josephlewis.github.io/: author homepage (publication list).
- **Access level:** **FT**.
- **Other methodological papers by J. Lewis on LCPs:**
  - Lewis 2023 (JAMT 31(3):854–874, multiple cost functions): read; see `lewis2023_costfunctions.md`.
  - SocArXiv preprint "Assessing the Impact of Including Traversal across Slope on Least Cost Path Accuracy: A Roman road case study" (doi 10.31235/osf.io/q5yad, 2020): **metadata only** (Crossref); not read.

## 2. BibTeX

```bibtex
@article{lewis2021probabilistic,
  author  = {Lewis, Joseph},
  title   = {Probabilistic Modelling for Incorporating Uncertainty in Least Cost Path Results: a Postdictive Roman Road Case Study},
  journal = {Journal of Archaeological Method and Theory},
  volume  = {28},
  number  = {3},
  pages   = {911--924},
  year    = {2021},
  doi     = {10.1007/s10816-021-09522-w}
}
```

## 3. What it does

The paper argues that a single least-cost path (LCP) computed on a DEM ignores DEM error. It proposes **Monte Carlo propagation of vertical DEM error**:

1. Add n spatially autocorrelated random error fields to the DEM.
2. Compute n cost surfaces and n LCPs.
3. Report a "probabilistic LCP": the fraction of runs in which each cell is crossed.

The method is implemented in the R package `leastcostpath` (`add_dem_error`). It is demonstrated on the High Street Roman road between the forts of Brougham and Ambleside (Cumbria).

## 4. How it generates

- **Procedural.** A cost function from the literature (Llobera & Sluckin 2007 energy function); no learning.
- **Representation:** raster DEM → conductance matrix over a k-neighbourhood → vector LCP lines; the probabilistic LCP is a raster density.
- **Algorithm (Methodological proposal; Case study):**
  1. **Error fields.** For each of n = 1000 runs, draw a random field from N(0, RMSE) with RMSE = 4 m. Smooth it with a 77×77 mean filter so it is spatially autocorrelated; the window comes from a semivariogram range of 3820.6 m over the 50 m cell size.
  2. **Perturbed DEMs.** Add each field to the DEM, then fill sinks.
  3. **Cost surfaces.** Compute the slope-based cost surface (Llobera–Sluckin) with a **48-neighbourhood** kernel.
  4. **Paths.** Compute the LCP between the two forts in each run. The probabilistic LCP is the crossing count per cell / 1000.

## 5. Inputs, outputs and how terrain enters

- **Inputs:** DEM (OS Terrain 50: 50 m cells, 4 m RMSE), its RMSE, origin and destination, cost function, neighbourhood size.
- **Outputs:** an LCP per run and a probabilistic LCP raster.
- **Terrain:** an anisotropic slope cost (uphill ≠ downhill). Nothing else in this paper; no water or landcover.
- **Package features relevant to us** (reference manual, v2.0.13; not used in this paper):
  - `create_slope_cs(cost_function, neighbours = 4/8/16/32/48, crit_slope = 12, max_slope = NULL, exaggeration)`;
  - the `"wheeled transport"` cost uses `crit_slope`, in percent, with default 12 = "the postulated maximum gradient traversable by ancient transport (Verhagen and Jeneson, 2012)";
  - `max_slope` sets conductance to 0 above a percentage slope;
  - `PDI_validation` and `buffer_validation` compare a path with a known route;
  - `create_FETE_lcps` computes all-pairs LCPs.

## 6. Evaluation

- **Visual only.** A single LCP is compared with the probabilistic LCP and with the known Roman road (Fig. 4).
  - The probabilistic LCP reveals several plausible routes in the south.
  - It shows a persistent deviation from the Roman road in the north, attributed to factors other than energy minimisation (e.g. visibility).
- **Not reported:** a quantitative similarity metric (PDI and buffer validation exist in the package but are not used here), a baseline, or statistics beyond the crossing frequency.
- **Compute note:** the 5 m DEM (5,177,480 cells) needed more than 24 GB of RAM in the then gdistance-based package. From v2.0.0 the package uses terra/sf instead of gdistance (README).

## 7. Quotes for citation

- p. 911, Abstract: "Focusing on vertical error, random error fields are calculated and incorporated into the documented and reproducible LCP modelling process using the R package leastcostpath."
- p. 911, Abstract: "Therefore, it is recommended that probabilistic LCPs become the default approach when modelling movement using input data that contains errors."
- p. 913, Methodological Proposal: "the likelihood of the LCP crossing a raster cell can be communicated by calculating the number of times an LCP crosses a cell divided by the total number of simulations (Fig. 1(E))."
- p. 916, Digital Elevation Model: "Contrast this with the OS Terrain 5 DEM, which has 5,177,480 cells, and required more than 24GB of RAM to calculate the cost surface."
- p. 917, Computing the Cost Surface: "Furthermore, in line with the recommendation by Harris (2000), a 48-neighbourhood kernel was used—that is, each cell in the cost surface is deemed to be adjacent, and therefore connected, to 48 neighbouring cells."
- p. 917, Computing the Cost Surface: "Due to this, the elongation error, which is the difference in length between the optimal straight line and the LCP, is reduced to below 1.4% (Huber & Church, 1985), resulting in more accurate cost accumulation"
- p. 918, Results: "When visually comparing the LCPs (Fig. 4), it is apparent that the single LCP produced without incorporating vertical error (Fig. 4(A)) does not fully capture nor communicate how the vertical error in the DEM, and subsequently the cost surface, propagates to the LCP result."
- p. 919, Discussion: "The lack of similarity, therefore, suggests that the location of the Roman road in the northern section may have been influenced by other factors, for example visibility (e.g. Verhagen & Jeneson, 2012)."

The page numbers are the printed journal pages (PDF page plus 910). A script checked every quote against the PyMuPDF text of the Apollo PDF.

## 8. Comparability with UrbanMorph

- **The package is our recommended terrain-aware connector baseline** (`docs/M18_BASELINES.md`).
- **Matched inputs.** Yes.
  - *DEM:* our 4 m DEM tile as a GeoTIFF; 128×128 = 16,384 cells, trivial in memory.
  - *Anchors:* entry points become origins and destinations (`create_FETE_lcps` for all pairs).
  - *Water:* a barrier via `update_values` / `replace_values`, with permitted crossing cells left open.
  - *Grade:* per-class max grade via `max_slope`; `"wheeled transport"` with `crit_slope`, or the exponential form of Lewis & Crema 2026.
  - *Classes:* one run per class.
- **Licence and platform.** GPL-3, R ≥ 3.4, terra/sf/igraph. CRAN arm64 macOS binaries are listed (not tested).
- **Effort (M18_BASELINES estimate):** 2–4 h as-is; 8–14 h adapted (R↔Python I/O, water barrier, merging overlapping LCPs into a planar graph, class labels).
- **Fair comparison:** **connector routing between supplied points**. It has no local infill or junction logic, and our post-processing (merge, planarise) must be published.

## 9. Ideas we could adopt for evaluation

- **DEM-uncertainty robustness of our terrain metrics.** Perturb the 4 m DEM with autocorrelated noise at the LiDAR product's stated RMSE (and test our street-imprint smoothing). Recompute the grade and contour-alignment metrics for real and generated networks, and report their Monte Carlo spread. A terrain-adaptation difference smaller than this spread is not meaningful.
- **Generator sensitivity to DEM error.** Run our sampler on several perturbed DEMs and report a probabilistic street-density raster (crossing frequency per cell), as the probabilistic LCP does. It shows which streets are terrain-determined (stable) and which are arbitrary.
- **Neighbourhood size matters.** The 48-neighbourhood keeps elongation error below 1.4 %. Use at least 16, preferably 48, neighbours for the LCP baseline so that its grid artefacts do not distort circuity and orientation metrics.
- **Ready-made route-similarity metrics:** `PDI_validation` (area / length, plus a normalised percentage) and `buffer_validation` (share of a path within d metres of the reference), for connector-level comparisons with OSM.
