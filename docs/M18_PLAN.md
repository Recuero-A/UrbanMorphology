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
| 1b | Related work and novelty verdicts; venue comparison | `docs/M18_LITERATURE.md` | in progress |
| 1c | External baseline feasibility (code, licence, inputs, effort) | `docs/M18_BASELINES.md` | in progress |
| 2 | Metric specification and audit: formulas, preprocessing, redundancy, controlled perturbations, density residualisation | `docs/M18_METRICS.md`, `outputs/m18/metrics/` | planned |
| 3 | Distributions for every model on the pinned terrain pack (KAIST, Seoul, Madrid, El Escorial) and synthetic surfaces: grade, blocks, junction spacing, context | `outputs/m18/distributions/` | planned |
| 4 | External baselines: one terrain-aware, one morphology or learned, on matched inputs | `outputs/m18/baselines/` | planned, after 1c |
| 5 | Hybrid pilot: neural backbone + procedural infill, 3 matched conditions (4 if feasible) | `outputs/m18/hybrid/` | planned, development sites only |
| 6 | Protocol frozen, then confirmation on fresh metros | `docs/M18_PROTOCOL.md` | needs owner decisions (below) |
| 7 | Reviewer verdict (one page), figure sheets, reproduction commands | `docs/M18_VERDICT.md` | last |

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
