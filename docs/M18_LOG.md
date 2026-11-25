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
* **F15. Procedural baselines** (`docs/literature/_index_procedural.md`; 14 notes, quotes checked by script against the
  opened texts).
  * Runnable as-is:
    * `leastcostpath` (GPL-3), for routing between supplied points;
    * `t-mw/citygen` (MIT, Parish & Müller), for full growth, but we must add its water and slope rules.
  * Reimplementable (labelled, hours estimated):
    * Galin 2010: 20–35 h;
    * Galin 2011, network from anchors: +15–50 h;
    * Beneš 2014, major-road lattice: 10–15 h;
    * Chen 2008 tensor field: 30–80 h;
    * Emilien 2012 road core: 25–40 h;
    * Nishida 2016: 40–70 h. It is the strongest classical match for completion given arterials, with examples taken
      from training cities only.
  * No paper states a numeric grade limit except Lewis & Crema 2026 (9 % and 15 % critical slopes).
* **F16. Corrections to `docs/M18_LITERATURE.md`** (from the notes):
  * Nishida 2016 has no quantitative comparison with real networks.
  * Lewis & Crema 2026 is full text, not abstract only, with code and data on Zenodo (CC BY 4.0).
  * Galin 2011's γ-neighbourhood inequality was written the wrong way round.
  * The streetz README defines circuity inverted relative to Birsak's paper.
* **D6. Erasure validation implemented** (`scripts/m18_erasure_validation.py`).
  * **Random strips:** another tile's street corridors, a seeded derangement, so area and shape are matched.
  * **Probe:** a dilated CNN trained on train tiles, scored on development tiles, for raw, low-pass 16 m,
    street-erased, random-erased and landform-only (64 m) inputs.
  * **Synthetic:** 8 natural IGN LiDAR windows with stamped cut-and-fill roads.
  * **Sanity check:** 93–100 % of real carriageway pixels fall inside the recovered erased corridors.
* **F17. Street-erasure validation** (`outputs/m18/erasure/`; development data, 52 development tiles, 473 train tiles,
  8 natural IGN windows).
  * **Random-strip control** (TN backbones, input erased in the street corridors vs another tile's corridors of equal
    area, 35 % each; paired over 3 backbones × 2 seeds):
    * erasing random strips worsens S_T by +0.178 (A) and +0.116 (M\*): information loss and unfamiliar input;
    * erasing the street strips adds a further +0.122 [+0.025, +0.219] (A) and +0.128 [+0.033, +0.222] (M\*):
      genuine street-bed leakage;
    * so roughly half of the TNse drop is leakage and half is lost terrain.
  * **Leakage probe** (a CNN finding carriageway pixels from terrain alone; ROC AUC, whole tile / inside the erased
    corridors):

    | Input | Whole tile | Inside corridors |
    |---|---|---|
    | raw | 0.78 | 0.75 |
    | blur 16 m | 0.66 | 0.56 |
    | street-erased | 0.74 | 0.61 |
    | random-erased | 0.71 | 0.65 |
    | landform 64 m | 0.51 | 0.52 |

    The leakage is real and detectable. The erasure removes most of the signal of where inside a corridor the street
    is, but a probe trained on erased terrain still finds the erased corridors. The M13 Laplacian-ratio criterion was
    too weak.
  * **Synthetic imprint test:**
    * a probe trained on raw tiles finds the stamped roads (AUC 0.74) but not the erased ones (0.53);
    * elevation in the corridors barely recovers (RMSE 1.82 → 1.72 m, median recovery 12 %, worse on 63 % slopes);
    * the 16 m low-pass also distorts natural terrain away from the roads (slope RMSE 9.9 %).
  * **Conclusion.** TNlp and TNse are partial leakage controls, not clean ones. The M14 wording ("robustness to the
    erasure, not independence from all leakage") stays, now with numbers.
* **Q3.** A stronger leakage control for the paper. Options:
  * (a) erase all built-up land (a GHSL mask), not only street corridors;
  * (b) evaluate on terrain without real streets (natural windows, stamped roads);
  * (c) use the 30 m GLO-30 for training and testing, where street beds are below the resolution;
  * (d) an adversarial erasure tuned until the probe fails.

  To discuss.
* **D7. Stronger leakage controls (owner, 2026-11-26): options (b) and (c) of Q3.**
  * **(b)** Terrain adaptation is tested on natural terrain windows that carry no urban street beds.
  * **(c)** The terrain model is retrained on Copernicus GLO-30 (30 m) inputs (a new backbone, TNg) and tested on
    GLO-30, where street beds are below the resolution.
  * GLO-30 is global, so (c) also allows adding hilly training cities outside Spain and the USA, which addresses the
    flat-training bias (F9).
  * Training data today: 685 training tiles from 9 cities; 473 of them (Spain, USA) have LiDAR terrain. Median land
    slope 2.8 %, p90 20.6 %. About 3 MPS-hours per backbone seed (M12).
* **D8. Today's comparison (owner, 2026-11-26):** all internal models and the external baselines (`leastcostpath`,
  `citygen`) on common sites.
* **Pushed to GitHub (2026-11-26).** Branches `milestone-18` (new) and `theme-latin-caribbean`; all commits authored
  by Recuero-A. The README now shows milestones 16–18, the KAIST and Petra towns, and the M18 figures.
* **D9. Natural-window rule (leakage control b), written before any generation**
  (`scripts/m18_natural_terrain.py`, `outputs/m18/natural/windows.json`).
  * Candidates: a 0.08° grid in Cataluña, Madrid and Valencia, visited in seeded order.
  * Each kept window has:
    * at least 2 km of OSM ways within 5 km, so it lies inside the extract;
    * at most 0.5 km of ways in the 1,024 m window;
    * a full IGN DTM;
    * a median tile slope of 5–35 %.
  * Result: 16 kept of 432 visited, with median slopes of 5.5–33 %.
  * **Amendment before generation:** the first pass found windows with 0 km of roads that were simply outside the
    extracts, so the coverage condition was added. Generation runs on these 16 windows.
* **F18. GLO-30 backbone (leakage control c)** (`outputs/m18/glo30/`, `configs/m18_tng.toml`, `scripts/m18_train_tng.py`).
  * **Data.** 525 tiles (train, val, val_t; test excluded) built at 4 m from 13 GLO-30 tiles; sha256 in
    `manifest.json`.
  * **GLO-30 vs LiDAR, per city.** RMSE 2.0–6.1 m. GLO-30 flattens short steep slopes in hilly cities, and its
    buildings make flat cities 2–3× steeper (a surface model).
  * **Training.** TNg = the TN configuration with GLO-30 elevation (unchanged `m12_train.py`). Seeds 0 and 1 are
    training (about 42–65 s per epoch, ending around 02:00 KST on 2026-11-27); seed 2 follows.
  * **Still needed:** `backbones.model_inputs` must learn the "glo30" input before TNg can be evaluated.
  * **Leakage probe on GLO-30 inputs:** AUC 0.59 for the whole tile and 0.53 inside the corridors, against 0.74 / 0.61
    street-erased, 0.66 / 0.56 low-pass and 0.51 / 0.52 landform only. GLO-30 hides the street pattern far better
    than TNse or TNlp. A small residual remains, probably from buildings.
* **F19. External baselines on the site benchmarks** (`docs/SITE_BENCHMARKS.md`, "External baselines";
  `outputs/m18/external/`). Existing rows are unchanged, cell for cell (checked against git HEAD).
  * Machu Picchu: leastcostpath realism 1.86 / TAS 72.8 at 17.8 km/km² (zigzags collapsed: 1.27 / 64.7);
    citygen sparse.
  * Petra: leastcostpath 1.15 / 71.8; citygen 1.37 / 70.3; NA-SMC 1.10 / 57.8.
  * **Implication: on TAS, the explicit slope-cost baselines beat every learned sampler**, and leastcostpath is
    nearly as realistic as NA-SMC by typicality. Our terrain-adaptation advantage is NOT shown. The paper claim must
    rest on connectivity and learned morphology (and fit to real distributions), not on TAS.
* **F20. Terrain response on natural terrain without street beds (leakage control b)** (`outputs/m18/natural/`;
  16 windows, 56 runs, compared with the real-city curves of F9).
  * **Siting is learned, not copied.** TN samplers build about 3 km/km² on land steeper than 30 % and 7–8 on 20–30 %,
    like real cities (4.0, 7.9). The terrain-blind N0 builds about 13–15 everywhere. TN samplers over-build flat land
    (23–26 vs 15).
  * **Steep-grade avoidance is partly learned.** Street length steeper than 15 %: TN samplers 0.13–0.16, N0 0.27.
  * **Contour following is weak for TN.**
    * On 10–30 % slopes TN's contour index stays near random (0.54–0.65 vs real 0.50–0.37); it drops only above 30 %
      (0.35–0.40 vs real 0.24).
    * The grade ÷ slope ratio peaks at 0.66–0.70 on 10–20 % land (real 0.44): TN streets run more directly up
      moderate slopes than real ones.
    * TN's orientation adaptation on city tiles was therefore partly street-bed leakage (consistent with M14 §6).
  * **TNse follows contours almost like the real cities** (0.47 at 10–20 %, 0.21 above 30 %), but has no siting
    preference (S 0.47) and builds little on flat land.
  * **Interpretation (proposal).** Leakage let TN rely on street beds instead of learning slope orientation. A
    backbone trained without readable street beds (TNg, F18) may generalise better to new terrain. TNg is evaluated
    the same way once trained.
  * TAS on these windows: TN samplers 65–68, P 70.1, N0 57.0.
* **D10. TNg evaluation set-up (2026-11-26 02:15).**
  * **Inputs.** GLO-30 versions of the 111 evaluation windows (16 natural + 95 GEN2; `scripts/m18_glo30_eval.py`),
    loaded through `backbones.model_inputs` (dem_variant "glo30"; a missing window raises, never falls back to
    LiDAR). Rules and metrics keep the LiDAR terrain, as for TNlp and TNse.
  * **GEN2 plan** (`configs/m18_plans/gen2_tng.json`). The M14 confirmatory specs of M\*, M\* + repair and NA-SMC on
    TNg s0–s2, seeds 1–3. NA-fixed is left out for time (about 10 s per tile).
  * **Natural windows.** All six samplers on TNg s0–s2, seeds 1–2. TNg reuses TN's soft terrain potentials, which act
    on the true terrain.
  * **Launch.** `scripts/m18_after_tng.sh` starts both once TNg seed 2 finishes training.
* **F21. TNg (GLO-30 backbone) results** (2026-11-26 03:00; `outputs/m18/glo30/gen2_tng.md`, `outputs/m18/natural/`).
  * **Natural windows (no street beds).** TNg follows contours better than TN (C 0.47–0.52 vs 0.38–0.42), like
    TNlp and TNse. It sites less precisely (S 0.57–0.66 vs 0.71–0.77) and has a heavier steep share (0.13–0.21 vs
    0.13–0.16). All of it is still better than the terrain-blind N0 (S 0.53, steep share 0.27). Terrain learning
    without leakage is confirmed.
  * **GEN2 cities** (3 backbones × 3 seeds, M14 estimators).
    * TNg M\* is already less fragmented than TN M\*: unreachability 0.023 vs 0.036, islands 0.18 vs 0.28. Its
      density is real-like (13.8 vs real 13.9 km/km²).
    * **NA-SMC − M\* on TNg is Δ −0.009 [−0.021, +0.009], p 0.31: same direction in 3/3 backbones but not
      significant** (TN: −0.025 [−0.038, −0.010], p 0.001). Islands still fall 0.18 → 0.01.
    * NA-SMC on TNg lowers density (12.1) and worsens S10 (0.38 vs M\* 0.32).
    * S_T on city tiles is much worse for TNg (0.36–0.55 vs TN 0.19–0.38). This is consistent with TN's city-tile
      terrain fit being partly street-bed leakage (F17, F20), since TNg's input cannot show street beds.
  * **Implication for the claim.** The connectivity gain of network-aware inference is backbone-dependent. It is
    strong where the base sampler fragments (TN) and small on a less fragmented backbone (TNg). The paper should
    report both. The defensible statement is "removes fragmentation where it exists, without morphology cost", not
    "always improves connectivity".
* **D11. Terrain-realism plan (owner, 2026-11-26).**
  1. **Real-city reference curves first.** TAS and its G / C / S stay as comparisons.
     * Every network (real, generated, external) is measured after the same physical smoothing.
     * Coverage is reported by slope class.
     * leastcostpath is compared on a routing task with shared anchors. The whole-town comparison stays for the
       generators that build whole towns.
  2. **Then a pilot of terrain potentials.**
     * Fitted on training cities.
     * NA-SMC with the potentials on vs off, paired seeds.
     * Matching the fitted curves on held-out cities shows that the calibration transfers.
     * A broader improvement also needs a gain on an outcome the potentials never target (e.g. the distribution of
       sustained uphill road lengths), with geographic access and coverage preserved.
  3. **Development** uses GEN2 and the two inspected sites. **Settings are frozen before the fresh-city
     confirmation.**
* **F22. Real-city terrain reference curves and the terrain-realism score (D11 step 1)**
  (`outputs/m18/terrain_reference/`).
  * **Reference.** 473 LiDAR training tiles from 6 cities. Every network is smoothed alike (ground polylines resampled
    every 30 m, then 10 m pieces). Per slope class: density, coverage, contour index, grade p50 / p90, shares above
    10 / 15 / 20 %.
  * **Score.** Mean |set − reference| / between-city SD over 5 classes × 5 statistics. The real training cities
    score 0.24 (Los Angeles) to 1.55 (Chicago).
  * **GEN2 city tiles.** Real GEN2 0.47 (fresh cities fall inside the real range); NA-SMC [TN] 0.62; NA-fixed and M\*
    0.75–0.76; repair 1.12; A 1.22; P 1.49; TNg 2.3–2.9. On the same smoothed networks TAS hardly separates anything
    (real 66.2, models 63–68).
  * **Natural windows (leakage-free).**
    * M\* [TNlp] 1.02, P 1.36.
    * NA-SMC, NA-fixed and M13-SMC [TNg] 1.57–1.58; the same samplers [TN] 1.82–2.02.
    * N0 3.26 (worst, a sanity check).
    * **TN's lead on city tiles reverses on honest terrain**, where the leakage-free backbones are closer to real
      cities.
  * Nearly every generator is outside the real-city range on natural terrain, so there is room for terrain
    potentials (step 2).
* **F23. Terrain moment potentials added to the sampler, and a reproducibility finding (D11 step 2 set-up).**
  * **The potential.** `NetConfig.tm_len` / `tm_ci` / `tm_steep` multiply the selection weight by
    exp(−Σ λ_k L_k − Σ μ_k (L·contour index)_k − ν L_steep). L is ground street length per 100 m in slope class k.
    The features are edge-additive, so the factor telescopes like the M14 counts. Off by default.
  * **Replay with the potentials off.** NA-SMC, TN_s0, seed 1, all 95 GEN2 tiles, torch threads = 2 (as
    `m14_experiments.py --threads 2`): **95 of 95 graphs identical** to the stored M14 run.
  * **Thread sensitivity.** The same replay with the default thread count gives only **24 of 95** identical, with
    both the old and the new code. Float sums depend on the thread count, and SMC amplifies the difference.
    **Every M18 generation must pin `torch.set_num_threads(2)`.** The earlier M18 natural-window and random-strip runs
    did not pin threads: they are valid samples and paired within each script run, but not bit-reproducible on
    another thread count.
* **Q1.** Main venue: Eurographics short (Jan 22) or GIScience full (Jan 29)? Owner to decide.
* **Q2.** Which learned baseline: Gu 2024 retrained, a Fang 2022 completion test, a labelled NTG reimplementation,
  or a combination? Owner to decide.
