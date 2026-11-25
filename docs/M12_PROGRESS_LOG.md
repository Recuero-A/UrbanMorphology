# Milestone 12 progress log

Dated entries: decisions, experiments, results and limitations, in the order they happened. Summaries live in
`MILESTONE_12.md`; this log is the audit trail.

## 2026-11-04: Phase 3–5 start (after approval of Phase 2)

**Dataset expansion: Los Angeles and San Diego** (`scripts/m12_select_regions.py`, `m12_build_tiles.py`)
* **Rules** were fixed in the script before any terrain value was seen:
  * M2 urban eligibility (GHSL 2020 ≥ 0.15) inside the city boundary;
  * USGS 3DEP 32 m screening, landform p90 classes;
  * Los Angeles → train, steep first, at most 120 tiles; buffers to the LA val/test tiles (3 km blocks and
    1,536 m); ≥ 1,024 m spacing;
  * San Diego → a held-out city: 3 km blocks assigned at random to `val_t` / `test_t`, 1,536 m between them.
* **Eligible cells:** LA 3,206 (screening: 778 steep, 773 moderate, 1,655 flat); San Diego 1,618 (831 / 542 / 245).
* **Selected:**
  * LA: 120 steep train tiles;
  * San Diego `val_t`: 40 (23 steep, 7 moderate, 10 flat);
  * San Diego `test_t`: 34 (25 / 5 / 4), **not built, not fetched, not read**.
* **Built:** 160 tiles with the unchanged M2 processing; 0 skipped. Index: `data/processed/m12/samples_m12.csv`.
* **Downloads:**
  * GHSL tile R6_C8 (CC BY 4.0, the M2 product) was downloaded to apply the M2 eligibility rule in San Diego;
  * the San Diego city boundary: one Nominatim request, the M2 backup query;
  * OSM came from the cached SoCal extract.
* **San Francisco not added (yet).** San Diego already provides an independent hilly evaluation city, and LA
  supplies 120 steep training tiles. It is reconsidered if coverage stays insufficient after the audit.

**Spain MDT02.** Not downloaded. The CNIG download centre is an interactive web application, and no WCS for MDT02
was found. Its licence was quoted as CC BY 3.0 ES in one source and not verified. Its data type (float or integer)
is unverified, so it was not "demonstrably" better. Spanish street grades stay flagged (terracing, ±3%).

**FABDEM (Japan).** Not used. The licence (CC BY-NC-SA 4.0 per mirrors; one copy says otherwise) would permit
non-commercial research. However, the data come as 10° × 10° archives of several GB for 30 m bare-earth estimates,
and Japanese tiles already contribute water and streets without a DEM. TN trains with a `valid` channel, so
Japanese tiles have valid = 0. An exploratory FABDEM arm is deferred; this is a documented engineering decision.

**CAVW 2024 (doi 10.1002/cav.2267).**
* Authors: Zhou, Li, Zhang, Wang, Qin, Liu; vol. 35(3), 2024.
* OpenAlex lists a "bronze" open-access PDF, but the publisher returned 403 to a normal request. Circumventing that
  (browser impersonation) was refused and is not attempted.
* Its 21 references are mainly image-translation networks (pix2pix, CycleGAN, SPADE, Swin, LPIPS), NTG, Parish &
  Müller, Galin 2011, Nishida 2016 and Emilien 2012. This is consistent with the abstract: terrain probably enters
  through an image-translated density map, with a transformer producing the graph.
* **Graph representation, sequentiality and bridge handling: unknown.** The full text is needed, e.g. through KAIST
  library access.

**Implemented (tested):**
* `terrain/synthesis.py`: seamless, seeded noise; ridges; valleys; basins; 10 scenarios;
* `terrain/hydrology.py`: priority-flood, D8, drainage area, hydraulic-geometry rivers, lakes from deep
  depressions;
* `terrain/water.py`: OSM water and sea from coastline orientation;
* `terrain/road_feasibility.py`, `terrain/road_profiles.py`;
* `graphs/terrain_sequence.py` and `models/terrain_graph_turtle.py`: terrain crops, symmetry alignment, structure
  head;
* `scripts/m12_train.py`, with checkpointing.

**Pre-registration:** `docs/M12_EVALUATION_PROTOCOL.md`, written before any M12 model was trained.

## 2026-11-04 (evening): data, training, evaluation

**Coverage after the expansion** (Phase 2 audit rerun, 32 m landform classes, `outputs/m12/audits/terrain_audit.json`):

| Feature | Before (train / evaluation tiles) | After |
|---|---|---|
| Steep | 30 / 1 | 150 / 24 |
| Valleys | 11 / 1 | 67 / 7 |
| Street grade p90 > 10% | 21 / 1 | 89 / 8 |
| Valley-spanning bridges | 0 | 2 / 1 |

Steep, valley and steep-street coverage is now adequate under the pre-registered rule. Valley-spanning bridges
remain inadequate, so viaducts stay rule-based. **San Francisco was not added:** San Diego already gives an
independent hilly evaluation city.

**Feasibility calibration amendment** (dated, before any model was trained): with 10 m sub-spans, the p99 grade for
motorways came out at 30%, which is DEM artefacts at elevated structures. The window was changed to 30 m. Result:
* grade limits: major 12.6%, mid 10.4%, local 16.4%, minor 21.8%;
* river bridge spans: 1,027 / 276 / 48.5 / 40 m by class;
* minimum crossing angle: 42.6°, from 137 real river bridges.

**Checkpoint and resume on the real training script.** A pilot was stopped with SIGTERM at step 26 and resumed. It
is bit-identical to the uninterrupted run (max |Δw| = 0.0, same validation NLL).

**Training** (`scripts/m12_train_all.py`, 2 jobs at a time on MPS):
* N0_s0: 36.7 min, best epoch 58, decoder units alive 0.59;
* TN_s0: 49.4 min, best epoch 55, decoder units alive 0.62;
* the seed-1 runs and ablations: see `outputs/m12/training/*/summary.json`.

No collapse: every run warmed up for 300 steps.

**Mistake found and fixed.** The first evaluation pass picked up seed-1 checkpoints *while they were still
training*: `best.pt` exists mid-run.
* Seed-1 results from that pass (e.g. S10 0.72 for TN_s1) were invalid.
* They were deleted and recomputed on the final models.
* `seeds_of()` now requires `summary.json`.
* No reported number uses a partial model.

**Pre-registered results** (`outputs/m12/evaluations/stats.json`):

| Hypothesis | Result | Verdict |
|---|---|---|
| H1 | S_T TN 0.44 / 0.43 vs N0 1.40 / 1.38 (−68.5% / −68.9%; bootstrap CIs below 0) | pass |
| H2 | S10 TN 0.312 / 0.319 vs N0 0.328 / 0.333 | pass |
| H3 | TN+R 0.205 / 0.205 vs R 0.411 / 0.426 (−50% / −52%) | pass |
| H4 | HR cuts the termination deficit (0.12 vs 0.20; 0.10 vs 0.17; CIs below 0), but S_T worsens by 0.095 / 0.089 > 0.05 | **fail** |
| H5 | structure-head AUC TN 0.967 / 0.967 (N0 0.882 / 0.880) | pass |

**Observed failure modes**
* R and TN+R leave steep or water-dominated tiles nearly empty: R's premature-termination share is 0.56–0.60.
* TN alone over-places bridges: 3–4 per tile vs real 1.6; 40 per tile on the wide-river scenario.
* TN alone ignores grade limits on the steep-hillside scenario.

**Exploratory HRV (added after the results; not part of H4).** HR never spans a dry valley: the contour turn
always wins, and learned steps are about 30 m long. HRV tries a far-rim extension first. It built 2 viaducts in 50
stress-test landscapes, all rule-placed.
* During this change, the replanning labels were made explicit. Index-based labels would have mislabelled HRV's
  counts.
* HR's order and random draws are unchanged, so its pre-registered results stand.

## 2026-11-05 00:15 — training and development evaluation complete

**Training**
* All runs finished with exit 0: N0, TN, TN-lp × 2 seeds; TN-elev / slope / water / elev+slope × 1 seed.
* TN-lp validation NLL per tile: −28.4 / −30.8 (TN −118.0 / −116.2; N0 3.4 / 6.4). About 75% of TN's likelihood
  gain depended on input detail finer than 16 m.
* Water-only input: NLL +2.6, about N0. The gain is in elevation and slope.

**Leakage-controlled tests** (`stats.json`)
* H1lp passes: −53% / −50%.
* H2lp passes.
* **H3lp fails:** seed 0's CI upper bound is +0.045.
* TN-lp+R premature share: 0.33 / 0.39.
* TN-lp AUC: 0.949 / 0.943.

**Ablations** (one seed; not leakage-controlled): slope alone gives the best S_T (0.344); water alone gives none
(1.417).

**Regime C:** TN-lp makes 110 model-placed bridges in 50 landscapes, against TN's 970.

**Found while writing up**
* The pre-registered circuity term uses connected node pairs only, so starved or fragmented networks are not
  penalised. The definition is not changed. It is reported as a limitation, with street density and dead-end ratio
  as the counterweight.
* Zero-length steps give a 0/0 grade (the RuntimeWarning in every evaluation log) and pass feasibility. They add no
  length, and the same code evaluated every model. The code is not changed.

**Tests:** added a unit test for `lowpass`. 231 pass.

**Docs:** `M12_RESULTS.md` and MILESTONE_12 §13–§17 written.

**Next:** gate 3, the one-time held-out test, awaiting the user's explicit authorisation.
