# Lewis & Crema 2026: Generative inference model reveals political and economic motivations for road construction in past societies

## 1. Header

- **Citation:** Joseph Lewis, Enrico R. Crema. "Generative inference model reveals political and economic motivations for road construction in past societies." *Humanities and Social Sciences Communications* (2026). Article in Press, published 30 October 2026 (received 4 July 2025, accepted 11 September 2026). Open access, CC BY 4.0. Volume and article number are not assigned in Crossref.
- **DOI:** 10.1057/s41599-026-09179-4
- **URLs opened:**
  - https://api.crossref.org/works/10.1057/s41599-026-09179-4: metadata, abstract, licence.
  - https://www.nature.com/articles/s41599-026-09179-4: landing page (abstract, dates, supplementary link, data statement).
  - https://www.nature.com/articles/s41599-026-09179-4.pdf: redirects back to the landing page.
  - https://www.nature.com/articles/s41599-026-09179-4_reference.pdf: **full-text "Article in Press" PDF**, 17 pages. Read in full.
  - https://static-content.springer.com/esm/art%3A10.1057%2Fs41599-026-09179-4/MediaObjects/41599_2026_9179_MOESM1_ESM.docx: supplementary information (captions of Supplementary Figs. 1–6).
  - https://zenodo.org/api/records/20759996: data/code archive "josephlewis/Roman_roadways_structure: Roman_roadways_structure_v9", 2026-06-19, licence CC BY 4.0. It is a supplement to https://github.com/josephlewis/Roman_roadways_structure/tree/v9. I did not open the repository itself.
  - https://josephlewis.github.io/Roman_Roadways_Structure.pdf: an earlier, single-author preprint by J. Lewis, "Process-Explicit Model Reveals Structure of Roman Roads in the Roman Empire" (16 pp.). The same 62 Welsh roads, different wording. Its relationship to the published paper is not stated; I infer it is an earlier version.
  - https://josephlewis.github.io/: author homepage (publication list).
- **Access level:** **FT**.
- **Correction to `docs/M18_LITERATURE.md`:** it marks this paper AO ("nature.com redirected to a login page"). The full text is reachable at the `_reference.pdf` URL above.

## 2. BibTeX

```bibtex
@article{lewis2026generative,
  author  = {Lewis, Joseph and Crema, Enrico R.},
  title   = {Generative inference model reveals political and economic motivations for road construction in past societies},
  journal = {Humanities and Social Sciences Communications},
  year    = {2026},
  doi     = {10.1057/s41599-026-09179-4},
  note    = {Article in Press, published 30 October 2026}
}
```

## 3. What it does

For each of 62 well-evidenced Roman roads in Wales, the paper simulates many least-cost paths between the road's real endpoints. The slope cost is governed by one road-specific parameter b. The b values that reproduce the real road's position are kept by Approximate Bayesian Computation, inside a hierarchical model that partially pools across roads. The estimated slope sensitivity is then related to road sinuosity. The authors interpret the resulting "straightness-gradient trade-off" quadrants as economic vs political motivations for construction.

## 4. How it generates

- **Simulator plus fitted parameters.**
  - The road simulator is procedural: least-cost paths with a fixed functional form.
  - Its parameter b is **inferred from real roads** by likelihood-free ABC in a Bayesian hierarchical model. It is "learned" in the statistical sense, not a neural network.
  - It is **not a network generator**: each road is routed independently between supplied endpoints.
- **Representation:** a raster DEM (OS Terrain 50, 50 m) with an 8-neighbour conductance surface gives vector least-cost paths; the observed roads are vector.
- **Algorithm (Materials and methods):**
  1. **Slope cost.** Influence of slope TCᵢ = exp(−bᵢ · |x + 0.05|), where x is the mathematical slope gradient.
     - The PDF typesets this as "TC_iexp(−b_i ∗ |(x + 0.05)|)"; the relation symbol is not legible.
     - Priors: bᵢ ~ TruncNorm(b̄, σ), b̄ ~ TruncNorm(1, 10), σ ~ Exponential(1).
  2. **Simulate.** Draw 250,000 bᵢ values. For each, build a conductance surface and compute the least-cost path between the road's origin and destination with `leastcostpath`. That is 15.5 million LCPs in total over 62 roads.
  3. **ABC.** For each road, accept the 250 draws (0.1 %) with the lowest Path Deviation Index (PDI) against the real road.
  4. **Translate literature thresholds into b.** Fit b_lim to reported critical slopes with `uniroot`: 15 % → 6.7; 9 % → 11.1; Judea road 18 % → 5.6; escarpment 36 % → 2.8. The critical slope is defined as the slope that maximises the pace height-gain rate.
  5. **Regression.** A Bayesian regression (brms; 10,000 iterations, 4 chains, Rhat < 1.01): sinuosity ~ median b + median terrain ruggedness (1 km buffer) + median road length. Measurement error in b enters via `me()`.

## 5. Inputs, outputs and how terrain enters

- **Inputs:**
  - 50 m DEM (OS Terrain 50);
  - origin and destination of each road;
  - the observed road polylines (62 roads from HER data, filtered to well-evidenced sections).
- **Outputs:** posterior b per road and at population level; accepted simulated roads; sinuosity–b regression.
- **Terrain** enters only through slope gradient in the exponential cost.
- **Water:** not modelled. "Hydrological barriers" are listed as future work.
- **Numeric thresholds stated:**
  - **critical slopes** of 15 % (wheeled vehicle) and 9 % (loaded wheeled vehicle with two mules), after Verhagen & Jeneson 2012;
  - 18 % and 36 % (Judea);
  - b_lim 6.7 / 11.1 / 5.6 / 2.8.
  - The earlier preprint also lists maximum slopes for the Welsh roads of 0.03–0.44, against literature limits of 0.25 (Roman Scotland), 0.20 (wheeled, unassisted) and 0.125 (preferred Roman maximum).

## 6. Evaluation

- **Real roads compared:** yes. All 62 roads are fitted to their real positions.
- **Fit metric (PDI).** Path Deviation Index = area between the two paths / Euclidean distance between their endpoints, read as the average distance between the paths. Median PDI per road ranges from 0.54 to 992.05, overall median 171.65. Units are not stated; presumably metres (my inference).
- **Recovery test.** On a simulated dataset of the same size (n = 62), 59/62 (95.16 %) of true parameters fall inside the 95 % credible interval. The 3 misses have a mean absolute error of 0.06.
- **Results.**
  - Per-road median b: 0.25–35.69. Population b̄ = 6.31 (95 % CI 0.27–35.60); σ = 0.76 (0.03–3.86).
  - 29/62 roads have b ≤ 6.7; their sinuosity is 1.01–1.16.
  - Regression slope β = 0.007 (0.004–0.009), R² = 52.5 % (39.3–61.8 %). A robustness variant gives β = 0.006, R² = 0.508.
- **Not reported:** network-level evaluation, baselines beyond literature thresholds, or a user study.
- **Runtime:** not reported.

## 7. Quotes for citation

- p. 1, Abstract: "The construction of roads in past societies reflects a trade-off between a preference for straightness and the influence of slope gradient."
- p. 4: "A slope gradient of 15% is, for example, suggested to be ascendible/descendible directly by a wheeled vehicle, given the mechanical properties of Roman vehicles (Raepsaet, 2002; Verhagen and Jeneson, 2012)."
- p. 4: "For a critical slope gradient to occur at 15%, a blim value of 6.7 is required."
- p. 5, Hierarchical movement model: "The hierarchical model was deemed robust, with the majority (59 out of 62, 95.16%) of true parameters correctly estimated within the 95% credible interval (Supplementary Figs. 2 and 3)."
- p. 6, Parameter inference: "Posterior distributions for each parameter were estimated by accepting 250 (0.1%) parameter values with the lowest Path Deviation Index value (Jan et al. 2000)."
- p. 6, Parameter inference: "The index can be interpreted as the average distance between the two paths."
- p. 6, Parameter inference: "Median Path Deviation Index across the 62 Roman roads ranges between 0.54 and 992.05, with an overall median of 171.65 (Supplementary Fig. 4)."
- p. 9, Results: "Sinuosity here refers to the length of the road divided by its 'as the crow flies' straight-line distance." (curly quotes in the source)
- p. 10, Discussion: "The first conclusion that can be derived from the results is that slope gradient did not exert a uniform influence on Roman roads in Wales, nor did roads conform to a single unifying preference for straightness."
- p. 14, Discussion: "While the 50 m resolution of the DEM used in this study aims to sufficiently capture the main processes influencing movement choice (Lewis, 2024b), future studies could further benefit from high-resolution LiDAR data."
- p. 15, Discussion: "Refining the cost surface to account for additional factors such as hydrological barriers also remains a priority."

The page numbers are PDF pages of the Article-in-Press file. A script checked every quote against its PyMuPDF text.

## 8. Comparability with UrbanMorph

- **Not a network generator.** It is a route-level inference method. The comparison it would be fair in is **connector routing between supplied points**, and there it is essentially the `leastcostpath` baseline with a calibrated slope cost.
- **Matched inputs.** Yes for routing.
  - *DEM:* our 4 m DEM tile. Note: the 50 m resolution here vs our 4 m.
  - *Endpoints:* entry points or anchors.
  - *Water:* would need a barrier added. The paper has none.
  - *Classes:* there is no class concept. Fit a separate b per class.
- **Code and licence.**
  - Zenodo archive under CC BY 4.0 (record metadata). The licence of the GitHub code itself was not checked.
  - It runs on R and `leastcostpath` (GPL-3).
- **Effort (my estimate, labelled).** 8–16 h to adapt the ABC loop to our tiles and fit b on real OSM streets of training cities. Compute is the main cost: they used 250,000 LCPs per road. Our 128×128 tiles make each LCP cheap, but the number of draws can probably be reduced.
- **Use as a baseline.** Calibrate the LCP connector baseline's slope cost **on real streets of training cities**. That gives the baseline a fair, data-fitted tuning budget instead of a literature default.

## 9. Ideas we could adopt for evaluation

- **Revealed slope preference as a terrain-adaptation metric.** For both generated and real anchor-to-anchor routes on the same tile, estimate the b (or critical slope) that best reproduces each route under an LCP model, via ABC/PDI or a likelihood over a grid of b. Compare the generated vs real distributions of b. This turns "does it treat slope like real streets do?" into one interpretable number per route.
- **Straightness–gradient trade-off plot.** x = slope sensitivity (b or grade percentile), y = sinuosity, controlling for ruggedness and length. Place real, ours and the baselines on the same axes. This is exactly our realism-vs-terrain framing, at route level, with a published precedent.
- **PDI as route similarity.** Area between paths / straight-line distance, plus the normalised PDI (% of straight-line length) implemented in `leastcostpath::PDI_validation`. Use it for generated vs real connectors between the same anchors.
- **Parameter-recovery check.** Simulate networks with known potential weights, then check that our evaluation recovers the ordering or values. It is a cheap validity test of our terrain metrics, mirroring their 59/62 recovery.
- **Literature grade anchors** (9 %, 12 %, 15 %, 20 %, 25 %) as reference lines on our grade CDF plots.
