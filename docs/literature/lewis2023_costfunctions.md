# Lewis 2023: Explaining Known Past Routes, Underdetermination, and the Use of Multiple Cost Functions

## 1. Header

- **Citation:** Joseph Lewis. "Explaining Known Past Routes, Underdetermination, and the Use of Multiple Cost Functions." *Journal of Archaeological Method and Theory* 31(3):854–874 (accepted 2 September 2023; published online 15 September 2023 per Crossref). Open access, CC BY.
  - The issue year of volume 31(3) is **not verified**. Lewis & Crema 2026 cite this paper as "Lewis (2024a)".
- **DOI:** 10.1007/s10816-023-09621-w
- **Why it is here:** a second methodological paper by the `leastcostpath` author. It uses the package (cited as "Lewis, 2023") and is listed first among the package's case studies in the GitHub README.
- **URLs opened:**
  - https://link.springer.com/content/pdf/10.1007/s10816-023-09621-w.pdf: returned an HTML bot page.
  - https://www.repository.cam.ac.uk/bitstreams/b14c5013-798b-4f4a-a59a-8df98cc5decd/download: Cambridge Apollo copy (21 PDF pages, publisher layout without printed page numbers). Read in full for methods, the tactical simulation and the Sardinia case.
  - https://api.crossref.org/works/10.1007/s10816-023-09621-w and https://api.openalex.org/works/doi:10.1007/s10816-023-09621-w: metadata and OA locations.
  - https://raw.githubusercontent.com/josephlewis/leastcostpath/master/README.md: case-study list.
- **Access level:** **FT** (pages below are PDF pages 1–21).

## 2. BibTeX

```bibtex
@article{lewis2023explaining,
  author  = {Lewis, Joseph},
  title   = {Explaining Known Past Routes, Underdetermination, and the Use of Multiple Cost Functions},
  journal = {Journal of Archaeological Method and Theory},
  volume  = {31},
  number  = {3},
  pages   = {854--874},
  year    = {2023},
  doi     = {10.1007/s10816-023-09621-w},
  note    = {Published online 2023-09-15; issue year of vol. 31(3) not verified}
}
```

## 3. What it does

The paper argues that a single slope-based cost function is only one idealisation of a hypothesis such as "people minimise time" or "people minimise energy". It therefore uses **14 cost functions** at once (time-based: Tobler and variants; energy-based: Herzog, Llobera–Sluckin). A "tactical simulation" on synthetic fractal terrains then tests two properties:

- **robustness:** paths from the same hypothesis are similar;
- **underdetermination:** paths from different hypotheses are also similar.

The result is "robust but underdetermined". Probabilistic LCPs (multiple cost functions plus DEM error) are then used to test whether time or energy minimisation explains the 'a Karalibus Sulcos' Roman road in Sardinia. Neither does.

## 4. How it generates

- **Procedural.** Fixed literature cost functions with Dijkstra LCPs in `leastcostpath`. Nothing is fitted.
- **Representation:** raster DEM → vector LCPs.
- **Tactical simulation:**
  1. **Terrains.** Five synthetic DEMs, 1 km × 1 km at 1 m, with fractal dimensions 2.2–2.6 (spectral synthesis in GRASS). Each is scaled to two elevation ranges: 0–5 m (slopes below 30°) and 0–10 m (slopes up to below 50°).
  2. **Paths.** Pick two random points. Compute 14 LCPs (Dijkstra, **4-adjacency**).
  3. **Deviation.** For every pair of LCPs, the "deviation" is the maximum Euclidean distance between them (91 pairs). Repeat 1000 times per landscape and scenario.
- **Sardinia case:**
  - TINITALY 10 m DEM (RMSE 4.3 m), **16-adjacency**;
  - the 14 cost functions, each with 50 unfiltered DEM-error realisations;
  - two probabilistic LCPs (time vs energy), visually compared with the digitised road.

## 5. Inputs, outputs and how terrain enters

- **Inputs:** DEM, origin and destination, cost functions, DEM RMSE.
- **Outputs:** LCPs, pairwise deviations, probabilistic LCPs.
- **Terrain:** slope only. The synthetic terrains are explicitly topography-only "laboratories".
- **Rivers:** deliberately **not modelled** in the Sardinia case, because passability is unknown.
- **Wheeled-vehicle cost:** the Llobera–Sluckin wheeled-vehicle function is **not used**, because its critical gradient must be estimated from the route being explained (see quote).
- **Numeric values:** slope ranges below 30° and below 50°; fractal dimensions 2.2–2.6; DEM 1 m (synthetic) and 10 m (TINITALY).

## 6. Evaluation

- **Synthetic study:** a distribution of maximum Euclidean deviation within and across hypotheses (time vs energy), over 455,000 LCPs per scenario. This gives the "robust but underdetermined" conclusion.
- **Real road:** a qualitative comparison of probabilistic LCPs with the Roman road. "Similarity" is analyst-chosen (footnote 1).
- **Not reported:** baselines beyond the cost-function family, or a user study.
- **Runtime:** not reported.

## 7. Quotes for citation

- p. 1, Abstract: "As a result, least-cost paths calculated using single cost functions are liable to spurious results with limited power for explaining known past routes, and by extension the decision-making processes of past people."
- p. 1, Abstract: "Using the ideas of multiple model idealisation and robustness analysis, and applied via a tactical simulation, this study demonstrates that similar least-cost paths can be produced from slope-based cost functions representing both the same hypothesis and different hypotheses, suggesting that least-cost path results are robust but underdetermined under the tested environmental settings."
- p. 2, footnote 1: "Similarity here is subjectively chosen by the analyst."
- p. 8, Materials and Methods: "Least-cost paths are calculated using the Dijkstra algorithm and a 4-adjacency neighbourhood."
- p. 14, Case Study 2: "the critical gradient parameter within the cost function is not given a predefined value defined by the cost function but instead to be estimated from the past route being explained."
- p. 15, Materials and Methods: "Rather than assigning a constant cost value to all rivers as previously done, e.g. Güimil-Fariña and Parcero-Oubiña (2015), rivers were not incorporated within the analysis."
- p. 15, Results and Discussion: "Thus, the process of road construction for the 'a Karalibus Sulcos' Roman road cannot be credibly attributed to the desire to minimise time or energy when traversing on foot from the western coast to the modern-day town of Decimomannu, or, in short, these hypotheses are rejected as explaining the Roman road." (curly quotes in the source)

A script checked every quote against the PyMuPDF text of the Apollo PDF. The p. 14 quote spans a line-break hyphen ("pre-/defined" printed across lines).

## 8. Comparability with UrbanMorph

- **Not a generator.** It is an LCP methodology. Its relevance is to how we **configure and tune the LCP baseline** and how we **compare routes**.
  - *Ensemble over cost functions.* An LCP baseline built from one cost function is one idealisation. Report it as an ensemble over plausible cost functions and parameters (e.g. `crit_slope` ∈ {9, 12, 15} %, several neighbourhoods), not a single cherry-picked setting.
  - *Neighbourhood.* 4-adjacency (used in the synthetic study) causes grid artefacts. For street-morphology comparisons use 16–48.
- **Code:** `leastcostpath` (GPL-3); see `lewis2021_leastcostpath.md`.
- **Effort:** none beyond the leastcostpath baseline; the ensemble adds about 2–4 h (my estimate).
- **Fair comparison:** connector routing between supplied points.

## 9. Ideas we could adopt for evaluation

- **Underdetermination check for our terrain metrics.** Test whether different generators (or different potential weights) can be *distinguished* by our terrain-adaptation metrics at all. Compute within-method spread (seeds) vs between-method differences, as the paper does with within- and across-hypothesis deviation. If the between-method gap is not larger than the within-method spread, report the metric as non-discriminative.
- **Maximum deviation (discrete Hausdorff-like) between paths** as a simple route-difference statistic, alongside PDI and Fréchet distance.
- **Synthetic-terrain "laboratory".** Run all methods on fractal synthetic DEMs of controlled roughness (fractal dimension 2.2–2.6, slope ranges up to 30° and 50°). Plot terrain-adaptation metrics against roughness. This isolates terrain response from city-specific style, and avoids LiDAR street-imprint leakage entirely.
- **Make the analyst's similarity threshold explicit.** The paper admits similarity is "subjectively chosen". Pre-register our tolerance choices (consolidation radius, matching radius) and report sensitivity to them.
