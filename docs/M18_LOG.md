# Milestone 18 log: every decision and finding, in order

The single place to look up what was decided, found or asked in M18, with a pointer to where the detail lives. Newest
entries go at the end. D = decision, F = finding, Q = open question.

## Where things live

| What | File |
|---|---|
| Phases, rules, design, owner decisions | `docs/M18_PLAN.md` |
| Repo audit, claim provenance, method table, inspected-site register, M14 reproduction | `docs/M18_AUDIT.md`, `outputs/m18/audit_m14/` |
| Related work, novelty verdicts | `docs/M18_LITERATURE.md` |
| Per-paper notes (citation, BibTeX, method, evaluation, verbatim quotes, comparability) | `docs/literature/` (indexes `_index_*.md`) |
| Metric catalogue, real-world street grades | `docs/literature/metrics_catalogue.md`, `docs/literature/street_grades_real_world.md` |
| External baselines and venues | `docs/M18_BASELINES.md` |
| Distributions per model vs real (GEN2) | `outputs/m18/distributions/` |
| Real-city terrain study | `outputs/m18/real_terrain/` |
| Street-erasure validation (leakage control) | `outputs/m18/erasure/` (in progress) |

## 2026-11-25

* **D1. Treat the external brief as M18.** Branch `milestone-18` from `theme-latin-caribbean`, which descends from
  `milestone-17`. Plan in `docs/M18_PLAN.md`.
* **F1. The M14 result reproduces from stored runs.**
  * The H1 contrast is Δ −0.0259 [−0.0363, −0.0154].
  * A precise 10-stream re-estimate gives M\* 0.039 and NA-fixed 0.012.
  * See `docs/M18_AUDIT.md` §5.
* **F2. NA-SMC's connectivity gain has no measurable cost** in density, junction spacing, short links or blocks.
  NA-myopic's has a real cost. Paired block-bootstrap intervals are in `docs/M18_AUDIT.md` §5.
* **F3. The OD metric samples network node pairs, not random points** (a documentation discrepancy). The
  per-tile Monte Carlo SD is 0.03. See `docs/M18_AUDIT.md` §3.
* **F4. Three declared held-out metros are no longer unseen:** Sacramento and Santa Barbara (M14 GEN2) and Seoul
  (M16). See `docs/M18_AUDIT.md` §3–4.
* **D2. Owner decisions on confirmation** (detail in `docs/M18_PLAN.md`):
  * 8–12 fresh metros in 3–4 countries;
  * a labelled reimplementation is acceptable as a baseline;
  * a small blinded realism pilot;
  * NA-SMC gets its own primary comparison with non-inferiority margins;
  * a shared geographic OD test;
  * repair connectors are scored for grade and water crossings.
* **F5. Distributions per model vs real on GEN2** (`outputs/m18/distributions/`).
  * The grade bulk matches. The steep tail does not: about 2× more street length above 20 % for most models.
  * Block compactness fails for every model, because generated blocks are less square.
  * Junctions are too close in most models.
  * NA-SMC and M13-SMC match link length and block area.
* **F6. Venues** (verified on the official pages):
  * Eurographics 2027 full papers: passed.
  * Eurographics short papers: Jan 22, 2027, 4 pages.
  * SIGGRAPH: Jan 28, 2027.
  * GIScience: Jan 29, 2027 (full), Apr 2, 2027 (short).
* **F7. Novelty** (`docs/M18_LITERATURE.md`):
  * C1 (network-aware inference), C2 (terrain-conditioned vector growth with leakage controls) and C3 (the
    realism-terrain evaluation) are narrowly novel.
  * C4 (junction reuse) and C5 (hybrid) are not novel.
  * The closest papers are RoadWeaver 2026 (post-hoc A\* repair, flat) and Fang 2022 (raster inpainting with DEM
    channels).
* **D3. The central experiment is NA-SMC vs post-hoc repair on terrain.** The hybrid (phase 5) is deprioritised.
* **F8. Prior terrain-aware generation exists.** Procedural: Parish & Müller, Chen, Galin, Emilien, Beneš, Nishida.
  Learned raster: Fang, Gu; learned rural vector: Zhou. Most of it is evaluated qualitatively. The gap is a learned,
  vector, urban, terrain-only generator with network control, scored against real networks on the same terrain.
* **D4. The approved design** (`docs/M18_PLAN.md`):
  * task 1 "how": real footprint and entry points given to every method;
  * task 2 "where + how": nothing given, compared at matched coverage;
  * hilly confirmation metros chosen by a declared rule (candidates Valparaíso, La Paz, Genoa, Naples, Wellington,
    Medellín, Guanajuato, Chongqing);
  * RoadWeaver-adopted metrics: largest-component share, reachability, dead-end ratio, cycle ratio, grade-limited
    routing.
  * Wonder sites stay exploratory.
* **F9. Real-city terrain study** (580 LiDAR tiles, 2,147 km of streets; `outputs/m18/real_terrain/`).
  * Real street grades: median 1.0 %, p90 7.3 %, p99 16.2 %.
  * On land steeper than 30 %, streets are about 5× flatter than the land and follow the contours: |cos| falls from
    0.62 to 0.24.
  * Real streets avoid steep land: 15.4 → 4.0 km/km².
* **F10. A hard grade rule is already in force.** M\* and the NA samplers forbid ground pieces steeper than the real
  p99: 12.6 / 10.4 / 16.4 / 21.8 % for classes 0–3 (`configs/m12_feasibility.json`).

## 2026-11-26

* **F11. Learned baselines** (`docs/literature/_index_learned.md`).
  * **Fang 2022:** its figshare code and weights do not run as shipped. The model class is missing, there is no
    terrain-model training code, and CUDA is hard-coded. Usable at most as a patched completion side test.
  * **Gu 2024** (diffusion; code and data under CC BY 4.0; no weights) can be retrained on our tiles: 20–30 h plus
    1–3 days of compute. **This is the best learned baseline candidate.**
  * The earlier `docs/M18_BASELINES.md` verdict ("no clean learned baseline") missed both of these.
* **F12. Grade references** (`docs/literature/street_grades_real_world.md`, every number sourced).
  * Proposed thresholds:
    * ≤ 5 %: comfortable (accessibility);
    * ≤ 10 %: fire access (IFC D103.2) and Spanish exceptional limits;
    * ≤ 15 %: local-street maximum (AASHTO via NYSDOT);
    * > 30 %: implausible;
    * > 35 %: unprecedented (Baldwin St 34.8 %).
  * Only traction physics is a hard limit (about 41–48 % dry, 27 % wet for 2WD cars, per the agent's derivation).
  * The proposed routing thresholds are 10 % (primary) and 15 % (secondary).
* **F13. Boeing 2022 open data** (10,351 urban areas):
  * median urban-area mean grade 2.7 %, p95 5.8 %;
  * share of street length above 15 %: Manizales 18 %, Funchal 17 %, Medellín 13 %, La Paz 8 %.
  * No published baseline exists for street grade vs terrain slope or for contour following. Our real-city study
    (F9) fills that gap.
* **F14. The proposed primary metric set (11)** is in `docs/literature/metrics_catalogue.md`. It is still to be
  reconciled with our metrics in phase 2.
* **D5. Street-erasure leakage control (TNse) gets three validation tests.**
  * **Random-strip control:** erase the same amount of terrain in strips that do not follow streets.
  * **Leakage probe:** a classifier tries to find the former streets on the erased terrain.
  * **Synthetic test:** stamp fake road cuts into natural terrain, erase them, and compare with the known original.
  * Prior art for the erase-and-interpolate recipe: Schmidt et al. 2018 (PLOS ONE), palaeo-DEMs, road embankment
    removal in hydrology. None was found as a leakage control for a learned generator.
* **Q1.** Main venue: Eurographics short (Jan 22) or GIScience full (Jan 29)? Owner to decide.
* **Q2.** Which learned baseline: Gu 2024 retrained, a Fang 2022 completion test, a labelled NTG reimplementation,
  or a combination? Owner to decide.
