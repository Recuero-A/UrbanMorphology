# M20 log: terrain-adaptive, realistic hill towns

One ordered log of decisions (D) and findings (F), as `docs/M19_LOG.md`. The plan is `docs/M20_PLAN.md`, the
committee decision of 2026-11-27.

Standing rules:
* development data only;
* the sealed M12 test split, the declared M15 held-out metros and every hold-H and study-H settlement stay unopened
  until D6 is committed;
* every design is written here and reviewed by the committee before its runs;
* commits by Recuero-A without AI attribution.

The committee is five reviewers: procedural graphics, ML generative modelling, urban morphology / GIScience,
methodology and statistics, and games technical direction.

## 2026-11-27

* **D1. The hill-town pool, its split, and the training data for TNg-H.** The first draft was revised after one
  committee round; the changes and rulings follow the decision.
  * **Why this is a design decision.**
    * The model's terrain gap is mainly a data problem (M20 plan §1).
    * Training street length on land steeper than 20 % totals 66 km, 80–86 % of it from Los Angeles.
    * So the terrain of the training data, the number of towns, the selection method and the quantity are fixed
      here, before any town data is downloaded.
  * **Frame.** Downloaded 2026-11-27; it contains no streets.
    * GHSL Degree of Urbanisation, GHS-SMOD E2020 R2023A V2.0, 1 km grid, Mollweide, CC BY 4.0:
      * **towns:** dense urban clusters (DUC), 77,386 polygons, 5,000–50,000 inhabitants, median area 3 km².
        Zip sha256 `cdcb3181b351c2813e2711f097010be10ec77961edda3ca3490e5e8cc7d5f3ad`.
      * **cities:** urban centres (UC), 11,534 polygons, at least 50,000 inhabitants, median area 22 km². Zip sha256
        `2afe3ffc6f2e759036ede137c485f0fa90240b3d895e4e47ab5442981617fd9f`.
      * **villages:** rural clusters, class 13 of the 1 km raster (500–5,000 inhabitants, at least 300 per km²,
        8-connected; definition checked in `GHSL_Data_Package_2023.pdf`). The raster zip is 36 MB.
    * **Country:** Natural Earth admin-0 (50 m, public domain), from the population-weighted centroid. Zip sha256
      `5fed433373581fa648920435f937d95f2d3c0200e067409c6478dcdf1b853139`.
  * **DEM (one recipe for everything).**
    * FABDEM V1-2, the bare-earth GLO-30 with buildings and forest removed (Hawker et al. 2022, ERL 17:024016),
      CC BY-NC-SA 4.0, for research use.
    * 1° tiles come from the per-tile mirror of the official Bristol release. A sample of tiles is checked
      byte-for-byte against the official zips.
    * Slope is computed on the DEM smoothed to 32 m.
    * The same DEM sets eligibility, the metric slope classes, and the inputs of every retrained backbone.
    * The bias is checked against PNOA LiDAR in the Spanish towns.
    * The licence is non-commercial: a model trained on it is for research only. This is stated in the paper.
  * **Exclusions.**
    * Every settlement within 50 km of a place already used:
      * the training cities, including the Japan tiles;
      * the GEN2, M15, M16 and M17 cities;
      * the declared M15 held-out metros;
      * the Petra and Machu Picchu sites.
    * Every settlement intersecting an M18 natural window.
    * The list is built in the script from the configs, with sources.
    * **Declared anchors outside the split:** Wadi Musa and Aguas Calientes, the real towns next to the two demo
      sites. They are development references only.
  * **Strata.**
    * Regions are fixed by country lists in the script:
      * R1 Mediterranean and southern Europe, including the Balkans;
      * R2 Latin America;
      * R3 East Asia;
      * R4 the Middle East, North Africa and the Caucasus;
      * R5 the rest of the world, including Alpine and Central Europe. R5 is used in hold-H and study-H only, as a
        declared secondary test of transfer to unseen regions.
    * Size: town (DUC) or city (UC). Villages form their own stratum, for evaluation only.
    * Slope band: A for 12–20 %, B for 20 % or more. At least a third of dev-H and hold-H must be band B.
  * **Split by spatial block, never by tile or polygon.**
    * Each settlement goes to the 1° × 1° WGS84 block that holds its population centroid.
      * Single-linkage clustering was tested and rejected: at 25 km it chains 23 % of all 88,920 settlements into
        one cluster, and at 10 km it still makes a cluster of 6,716.
    * u = sha256(salt + block id) / 2²⁵⁶:
      * train-H: u < 0.30;
      * dev-H: 0.30 ≤ u < 0.55;
      * hold-H: 0.55 ≤ u < 0.92;
      * study-H: u ≥ 0.92, the perceptual-study stimuli, never tuned on.
    * **Buffer:** a dev-H, hold-H or study-H settlement within 10 km of a settlement of another split is dropped.
    * **Salt:** the outputValue of the NIST randomness beacon pulse at the first full hour at least 30 minutes after
      this decision is pushed. If the beacon cannot be reached, the commit hash is the fallback. The rule, the
      script and the full frame hashes are pushed before that time.
    * Within each split, candidates are ordered by sha256(salt + "order" + settlement id).
    * The ordered lists (id, block, u, order, stratum) and their sha256 are published before any town data is
      downloaded.
  * **The walk.** Candidates are taken in order, round robin over strata. Each must pass two checks.
    * **Eligibility (terrain only).** The median slope inside the polygon, weighted by GHS-BUILT-S E2020 (100 m),
      is at least 12 %.
      * Eligible counts per stratum are published before the walk.
      * A stratum with fewer than 3 × its quota merges its two bands.
      * A quota a stratum cannot fill passes to the next stratum of the same region and size; failing that, the same
        region; failing that, it is reported unfilled.
    * **OSM completeness, independent of street density.** Density is the outcome being measured, so it is not used
      as a gate.
      * At least 90 % of the built-up cells (GHS-BUILT-S built share ≥ 0.2) inside the polygon lie within 100 m of
        any OSM highway way, including steps, footways and paths. Vehicular coverage is an outcome, never a gate:
        real vehicular coverage on land steeper than 30 % is only 0.62, and hill-town homes are often reached by
        stairs.
      * At least 15 km of vehicular ways in total (villages: 4 km).
      * One OSM snapshot for all splits: Overpass `[date:"2026-11-27T00:00:00Z"]`, raw JSON pinned with its sha256.
      * The steps and footway layer is stored for every settlement, with its own completeness score.
    * **At most 2 settlements per 1° block per split.** Further candidates in a full block are skipped and logged.
      Towns in one block share terrain and mapping. D6 clusters inference on the 1° block, by sign-flip or
      wild-cluster bootstrap.
    * A gate failure is final.
    * A CONSORT-style flow per split and stratum and a balance table (slope, area, population, pass rates) are
      reported.
  * **Quotas.**
    * **dev-H:** 24 settlements, pooled across strata (a leave-one-out envelope covers 0.92), plus 6 villages and the
      2 anchors.
    * **hold-H:** at least 40 settlements plus 6 villages, walked only after D6. The final number comes from a
      committed rule on the dev-H pilot variance. The settlement is the analysis unit (unweighted means), with at
      most 8 windows per settlement chosen by hash.
      * Why 40: a 3 km² town gives about one cluster of windows. Twenty towns detect only δ ≥ 0.013–0.016; δ 0.008
        needs 40 towns at 80 % power or 53 at 90 %, at 24 runs per town.
    * **study-H:** 8 settlements plus 4 villages, walked after D6.
    * **train-H:** walk to a cap of 30 settlements. Report the street-km obtained per slope class (below).
  * **Windows.**
    * 512 m windows on a grid aligned to the population centroid.
    * A window is kept if at least 70 % of it lies inside the polygon and its DEM is valid. There is no street
      minimum: "at least 3 km" means 11.4 km/km² and would drop the sparse steep land being studied.
    * Training drops windows with no vehicular edge and reports how many.
  * **Training data and recipes (TNg-H and its controls).**
    * **Three backbones,** all trained from scratch with the unchanged `m12_train.py` TNg configuration, seeds 0–2,
      all with the same input recipe (FABDEM, 32 m slope channel):
      * **TNg-32:** the existing 525 training tiles only. This control separates the input change from the data.
      * **TNg-R:** the existing tiles plus train-H settlements with weighted median slope under 6 %, from the same
        region × size strata, matched to TNg-H on added street-km. This control separates "more data" from "hilly
        data".
      * **TNg-H:** the existing tiles plus the eligible train-H settlements.
    * **Loss weighting, not sampling by outcome.** Training tiles are not resampled by their street content; that
      would be choice-based sampling (Manski & Lerman 1977) and would teach the model to over-build steep land.
      * Each decision's loss is weighted by the slope class of its current node, tempered as the square root of
        the inverse class frequency.
      * Tile repeats are capped.
    * **Region cap:** no region supplies more than 40 % of the steep street-km.
    * **Learning curve:** TNg-H on nested 25 / 50 / 100 % of the added settlements, one seed each, scored by dev-H
      NLL per slope class. NLL still falling at 100 % means more data would help; flat by 50 % means the data are
      enough. The curve is reported.
    * **Vehicular streets only.** Stairs come from the post-hoc hybrid, whose stairs rule is fitted to the real
      steps layer of dev-H settlements that pass the footway completeness check.
    * **Compute:** about 33 MPS-hours.
    * **Freeze rule, by 8 Dec.** TNg-H is frozen only if both conditions hold; otherwise it is dropped.
      1. Its dev-H NLL on 20–30 % and > 30 % land is below that of both TNg-32 and TNg-R, in at least 2 of 3 seeds.
      2. With one identical sampler for all three backbones (NA-SMC with F1 + F4, no terrain potentials), on dev-H:
         * TNg-H is closer than TNg-32 to the real dev-H distribution of the share of length above 15 % grade, and
           of the contour index, on 10–30 % and > 30 % land;
         * its flat-land density (0–5 % and 5–10 %) stays within 1 between-city SD of real.
  * **Sign-off round.**
    * Games and graphics approved. Statistics, urban morphology and ML each raised one blocking objection with an
      exact fix, and all three are adopted above:
      * at most 2 settlements per block;
      * the completeness gate counts any highway way;
      * the TNg-H freeze rule.
    * Notes for D6:
      * Declare the DEM the frozen TNg receives at confirmation: FABDEM as primary, GLO-30 as the sensitivity run.
      * FABDEM is non-commercial, so the game-usable path is the TNg hybrid, not TNg-H. A studio would need to retrain
        on a permissively licensed bare-earth DEM.
  * **Committee changes to the draft, with rulings.**
    * **Statistics.**
      * The density gate selected on the outcome; it is replaced.
      * The salt now comes from a public beacon.
      * Hold-H went from 20 to at least 40, with the settlement as the unit and a CONSORT flow.
      * One OSM snapshot for every split.
    * **Urban morphology.**
      * Eligibility is built-up-weighted on a bare-earth DEM, as the plan already said; the draft had contradicted
        it.
      * The window rule is by land, not streets.
      * Anchors were added, and R5 is for transfer only.
      * Single linkage was rejected (evidence above); spatial blocks plus a buffer replace it.
    * **ML.**
      * Choice-based sampling is replaced by loss weighting.
      * TNg-32 was added as a control.
      * The learning curve and the region cap were added.
      * The fixed km targets were dropped as arbitrary.
    * **Graphics.**
      * Bare-earth inputs (FABDEM instead of a home-made fill).
      * Band B gets at least a third.
      * A study-H split was added.
      * The gate counts vehicular ways only.
      * Not adopted for now: random low-pass augmentation of the inputs. It would be a second input change for every
        backbone, so it is a candidate for a later ablation.
    * **Games.**
      * A village stratum for evaluation.
      * The steps layer is stored now for a later pedestrian class (M21).
      * Vehicular and pedestrian reach are reported separately.
* **D1 pushed 2026-11-26T18:44:00Z (commit 8598f99).** By the rule, the salt is the NIST beacon pulse of
  2026-11-26T20:00:00Z. The draw and the split run automatically at that time (`scripts/m20_pool.py beacon / split`).
* **D2. Terrain-blind repair control on development data (exploratory; the statistics reviewer's H3, M19 F5).
  Design fixed before any run.**
  * **Arm A5b.** Identical to M19 A5 except the router:
    * the cost is step length only, with no grade cap;
    * water and the domain bounds still block.
    * Everything else is unchanged: dead-end eligibility (≥ 15 % landform slope), D2 destinations, the
      route-length limit, connectors ordered by route length, and the commit rules of the sampler.
  * **Units.** The M19 D5 units:
    * 330 tile unit-runs: GEN2 and the natural windows × TNg_s0–s2 × seeds 1–2;
    * Machu Picchu and Petra × TNg_s0 × seeds 1–3.
    * A0 is replayed at its stored thread counts. Every replayed graph must equal the stored A0 graph bit for bit,
      or the run aborts.
  * **Outcomes.**
    1. D5 steep-land access at matched added street length. Both connector curves are read at
       L* = min(A5 total, A5b total), giving A5 − A5b.
    2. The share of connector length steeper than 15 % grade, sampled every ≤ 4 m on the terrain.
    3. Connectors completed.
  * **Reading, declared before the runs.**
    * "Terrain-aware" is supported if both hold:
      * A5 access ≥ A5b access − 0.005 at matched length (non-inferiority);
      * A5's connector share above 15 % is lower.
    * If A5b gives clearly higher access, then the access gain of A5 does not come from terrain awareness. That is
      reported as such.
  * **Statistics.** A block bootstrap over the 30 tile blocks, as in D5. Towns are descriptive. This is development
    data, so the result is exploratory.
* **D2 amendment (statistics review, before any result was computed).**
  * **Measurement bug fixed.** A connector ending in a T-junction or a crossing splits an old street, and the two
    halves got new keys. An edge whose midpoint lies within 0.5 m of an A0 edge is now old street, not connector.
  * **A5b-nocap added as a declared secondary arm.** A5b still commits through the sampler's grade cap (reason
    `grade_cap`), so its steepest steps stop as infeasible. A5b-nocap ignores that cap at commit.
    * Step failures are reported per arm and reason.
    * Unweighted block means are reported as the declared sensitivity; pair-weighted stays primary.
  * **Approved as declared:** L* = min(totals), the −0.005 margin, 2,000 block draws.
  * **The first run was stopped to add the secondary arm** before any measurement. Its 11 partial files are kept in
    `outputs/m20/blind_repair/aborted_run_no_nocap/`, never analysed.
* **F1. Terrain-blind repair control (D2; development data, exploratory)** (`outputs/m20/blind_repair/report.json`;
  330 tile unit-runs, 30 blocks).
  * All 17 A0 replays reproduced the stored M19 graphs bit for bit.
  * **Tiles.**

    | Measure | A5 (terrain-aware) | A5b (terrain-blind) | A5 − A5b [95 % CI] |
    |---|---|---|---|
    | Steep-land access at matched length | | | −0.0015 [−0.0035, −0.0001] |
    | Same, unweighted block means | | | −0.0021 [−0.0043, +0.0000] |
    | Connector length steeper than 15 % | 0.28 [0.25, 0.31] | 0.61 [0.55, 0.69] | −0.34 [−0.39, −0.29] |
    | Street added (km) | 24.7 | 29.5 | |
    | Access at the end (A0 0.480) | 0.492 | 0.497 | |
    | Connectors complete / partial | 311 / 82 | 448 / 2 | |

  * **Verdict by the D2 rule: "terrain-aware" is supported.** At matched length A5 is non-inferior (lower bound
    −0.0035 > −0.005), and its connectors are far gentler.
  * **The honest reading is narrower.**
    * On the D5 access endpoint, which has no grade limit, the blind router does as well or slightly better: the
      point estimate favours it, and it adds more street.
    * So M19's access gain comes from adding connectors, not from terrain awareness.
    * Terrain awareness buys gentle connectors (0.28 against 0.61 of their length steeper than 15 %) at a negligible
      cost in ungraded access.
    * This confirms the committee's point that the endpoint must be grade-limited (M20 plan §4, M6).
  * **A5b-nocap equals A5b exactly.** The sampler's grade cap never stopped a blind step (only 2 steps failed, both
    `ground_in_water`). In this configuration the commit rules do not block steep connectors.
  * **Whole towns (descriptive).**
    * Petra: the blind router reaches more pairs at the end in all 3 seeds (11,564 / 7,961 / 10,316 against
      8,532 / 7,340 / 8,777 of 16,912). Its connectors are 44–62 % steeper than 15 %, against 23–25 % for A5.
    * Machu Picchu: the blind connectors are 89–93 % steeper than 15 %, with almost no access change.
  * **A lead for the hybrid (not yet tested).** A5's router forbids grades above 21.8 % and penalises above 8 %. Yet
    28 % of its committed connector length is steeper than 15 %, and 82 connectors stop partway.
    * A likely cause: commits are 25 m chords along the route, which cut the corners of switchbacks.
    * The hybrid should commit the route's own geometry. This is to be checked when the hybrid is designed.
* **F2. The split is drawn, with the declared fallback salt** (`configs/m20_pool/`).
  * **The NIST beacon stalled before the push.** Its last pulse was 2026-11-26T18:08:00Z (index 2044989), 36 minutes
    before the D1 push at 18:44:00Z. The declared 20:00:00Z pulse returned "Pulse Not Available" (HTTP 404, checked
    20:43Z).
  * **The salt follows D1's fallback rule:** the full hash of the D1 commit,
    `8598f9942c0f891fcd84af034a286d8a08c91eff`. Its content was fixed and pushed before any split was computed, and no
    split was computed with any other salt. `salt.json` records the outage.
  * **Lists, ordered by sha256(salt + "order" + id).**

    | Split | Settlements | 1° blocks |
    |---|---|---|
    | train-H | 11,110 | 947 |
    | dev-H | 32,923 | 961 |
    | hold-H | 158,032 | 3,956 |
    | study-H | 28,590 | 845 |

    * The sha256 of each list is in `split_manifest.json`.
    * 13,687 settlements were dropped by the 10 km buffer. 188,433 are unused: R5 settlements in train or dev, and
      villages in train.
    * No excluded settlement is in any list.
    * Every split has every region R1–R4, in both size classes (and villages outside train).
  * **No town data has been downloaded.** The walk (FABDEM eligibility, then the OSM gate) is next.
    * The pinned OSM snapshot, 2026-11-27T00:00:00Z, is still in the future at the time of the split. OSM queries
      therefore wait until it has passed. The terrain checks can run before.
