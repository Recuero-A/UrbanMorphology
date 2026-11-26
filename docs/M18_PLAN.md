# Milestone 18: the strongest defensible paper claim (plan)

**Request (2026-11-25).** The owner proposed this milestone from an external brief, summarised below:

* determine the strongest defensible paper claim;
* reproduce competitive external baselines on matched inputs;
* audit the realism and terrain measures;
* compute comparable distributions for every model;
* test a neural-backbone / procedural-infill hybrid if the evidence supports it;
* end with a submission decision.

Measured findings, interpretations and proposals are labelled as such throughout.

## Rules (unchanged from earlier milestones)

* **Held-out data.** No sealed test split and no declared held-out metro is read without the owner's explicit
  authorisation. The M12 test split is sealed. The M15 held-out metros are listed in
  `configs/m15_benchmark/manifest.json`.
* **History is preserved.** Historical samplers, checkpoints, configurations, protocols and outputs are never
  modified. New code goes in new modules or behind defaults that replay the old results bit for bit.
* **Protocol first.** A protocol with primary comparisons, effect sizes and preservation margins is committed before
  any confirmatory run. Confirmation uses fresh metros that no milestone has inspected (`docs/M18_AUDIT.md` §4 lists
  the inspected ones).
* **Honest reporting.** Negative results are reported. If one dependency blocks an experiment, the exact blocker is
  recorded and independent work continues.
* **Free tools only.** Run IDs hash the specification.

## Phases

| Phase | Content | Output | Status |
|---|---|---|---|
| 1a | Repo audit: claim provenance, method table, discrepancies, inspected-site register, reproduction of the M14 result and its trade-offs | `docs/M18_AUDIT.md`, `outputs/m18/audit_m14/` | done (2026-11-25) |
| 1b | Related work and novelty verdicts | `docs/M18_LITERATURE.md` | done (2026-11-25) |
| 1c | External baseline feasibility (code, licence, inputs, effort) and venues | `docs/M18_BASELINES.md` | done (2026-11-25) |
| 1d | Per-paper literature notes (citation, BibTeX, method, evaluation, verbatim quotes, comparability) and a metric catalogue | `docs/literature/` | in progress |
| 2 | Metric specification and audit: formulas, preprocessing, redundancy, controlled perturbations, density residualisation, the adopted metrics below | `docs/M18_METRICS.md`, `outputs/m18/metrics/` | planned |
| 3a | Real-city reference study: how real streets adapt to terrain (grade, contour following, siting by terrain-slope class, per city) | `outputs/m18/real_terrain/` | first pass done (2026-11-25): 580 real tiles, LiDAR |
| 3 | Distributions for every model: GEN2 first pass done (`outputs/m18/distributions/`); next, all 24 versions on the terrain pack (KAIST, Seoul, Madrid, El Escorial), Machu Picchu, Petra and synthetic surfaces | `outputs/m18/distributions/` | in progress |
| 4 | External baselines on matched inputs: `leastcostpath`, `citygen`, and the extended pool from 1d | `outputs/m18/baselines/` | planned, after 1d |
| 5 | Hybrid pilot: neural backbone + procedural infill | `outputs/m18/hybrid/` | **deprioritised**: not novel (RoadWeaver 2026, Nishida 2016; `docs/M18_LITERATURE.md` C5) |
| 6 | Protocol frozen, then confirmation on fresh metros | `docs/M18_PROTOCOL.md` | needs owner decisions (below) |
| 7 | Reviewer verdict (one page), figure sheets, reproduction commands | `docs/M18_VERDICT.md` | last |

## Experimental design (approved 2026-11-25)

* **Two tasks.**
  * **Task 1, "how" (primary).** Every method receives the same window, DEM, water mask, the real built-up footprint
    (from data, not hand-drawn) and the points where real arterials enter the window. Its streets are compared with
    the real streets inside the same footprint. This removes the advantage of building only where it is easy.
  * **Task 2, "where + how" (secondary).** No footprint is given. Coverage is reported, and comparisons are made at
    matched coverage and street length.
  * Named stress-test sites without a real town (Machu Picchu, Petra, other world-heritage sites) allow only task 2.
    They stay exploratory figures.
* **Confirmation sites.** Real hilly towns where people built streets on steep land, chosen by a rule declared before
  any data are inspected. Each metro has flat and steep windows (stratified by terrain slope) and comparable DEMs.
  * Candidates to screen: Valparaíso, La Paz, Genoa, Naples, Wellington (LiDAR), Medellín, Guanajuato, Chongqing.
  * Every metro of `docs/M18_AUDIT.md` §4 is excluded, which includes Lisbon, Granada and Toledo.
* **Adopted from RoadWeaver (2026).**
  * Largest-component share, reachability, dead-end ratio, cycle ratio.
  * A downstream test: **grade-limited routing success**, the share of random point pairs a vehicle can connect
    without exceeding a maximum grade. Thresholds are justified by design standards and by the real-city study
    (3a).
* **Hard rules already in force.** M\*, NA and their variants forbid ground pieces steeper than the real p99 of US
  training streets: 12.6 / 10.4 / 16.4 / 21.8 % for classes 0–3 (`configs/m12_feasibility.json`). The routing test
  is an evaluation measure, not a generation rule.

## Owner decisions (2026-11-25)

1. **Fresh metros.** The target is 8–12 fresh metros across 3–4 countries.
   * Cover flat and steep terrain and different street patterns, with comparable elevation and road data.
   * Where possible, include flat and hilly neighbourhoods within one metro, so that terrain difficulty is less
     confounded with country.
   * Metros are the unit of geographic generalisation.
   * Every metro in `docs/M18_AUDIT.md` §4 is excluded.
   * Development data are used first to check whether this sample gives useful intervals.
2. **External baselines.** One reproducible terrain-aware procedural method and one recent learned street generator,
   on common sites and compared with real networks.
   * A labelled reimplementation is acceptable if it reproduces a published example or sanity check, documents its
     deviations, and gets a fair tuning budget.
   * M\*, NA-fixed and M\* + repair stay as the internal comparisons.
   * A dataset supplies reference data; it is not a baseline.
3. **Human realism.** A small blinded pilot, secondary to the quantitative evidence.
   * Real held-out maps are mixed with UrbanMorph and baseline outputs, with standardised rendering and randomised
     order.
   * Raters answer separately on layout plausibility and on terrain fit.
   * Agreement and uncertainty are reported.
4. **Required before confirmation.**
   * **NA-SMC's own primary comparison.** NA-SMC, the proposed method, gets its own frozen primary comparison against
     M\*. The reproduced −0.026 contrast is NA-fixed − M\*.
   * **Non-inferiority margins.** NA-SMC gets practical margins for density, junction spacing and block structure:
     an interval that rules out a meaningful increase, not just "no measurable increase".
   * **Shared geographic OD test.** The legacy network-node OD metric is kept to reproduce M14. A shared geographic
     OD test is added, with the same locations for every method: can each location reach a road, and can the pair
     connect? A sparse network must not look good by serving only the nodes it chose to build.
   * **Repair connectors.** They are scored for longitudinal grade and water crossings.
