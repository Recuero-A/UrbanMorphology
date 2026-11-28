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
* **D3. The metric suite, frozen before any generator is scored.** The draft was revised after review by all five
  members. The chair's rulings are marked "ruling".
  * **Common ground.**
    * **Networks:**
      * real: OSM vehicular ways in the training classes (plus links), with steps, footways and paths as a separate
        pedestrian layer;
      * generated: vehicular only.
    * **Geometry:** native geometry and the common convention (resample ≤ 30 m, merge 8 m). The convention is
      primary, except where a metric says native.
    * **Terrain:** FABDEM averaged to 32 m. Slope classes 0–5 / 5–10 / 10–20 / 20–30 / > 30 %.
    * **Units:**
      * whole settlements (the polygon), the primary unit for hold-H;
      * 512 m windows (tile mode).
    * **Cells:**
      * a (unit, slope class) cell exists if it has at least 0.05 km² of land;
      * a street metric is defined for a cell if the real median street length there is at least 1 km.
      * If real towns have streets in a cell and a generator has under 1 km, the cell gets the worst score (|z| = 5,
        declared). Deleting streets is never skipped.
  * **Metrics, per slope class.**
    * **M1 Siting:**
      * density, km/km² of class land;
      * coverage, the share of class land within 60 m of a street.
    * **M2 Orientation and grade:**
      * contour index, only for classes of 5 % and steeper;
      * grade p90;
      * share of length steeper than 15 %.
    * **M3 Steep structure:**
      * **Runs:** continuous grade above 15 % along strokes (M16 pairing, deflection < 30°, continuing through
        junctions), median and p90.
      * **Hairpins:** a cumulative heading change above 120° within ≤ 40 m of path along a chain of degree-2
        nodes, per km of street on land steeper than 15 %.
      * Before freezing, native and convention hairpin counts are compared on dev-H real switchbacks.
    * **M4 Topology:**
      * dead ends per km;
      * 3-way and 4-way shares;
      * close15 after consolidation at τ 10 m;
      * largest-component share;
      * loops per km², from the cyclomatic number;
      * lens-face share (ruling): the street length on faces whose boundary has exactly two nodes of degree 3 or
        more, with elongation of at least 4. Faces with 2A/P < 20 m and elongation above 6 are a sensitivity.
    * **M5 Blocks:**
      * face area, median and p90;
      * Louf Φ, median;
      * on land steeper than 10 %, the |cos| between each block's long axis and the contour direction.
    * **M7 Tortuosity:** native length ÷ convention length. This catches raster zigzags and chorded switchbacks.
    * **M6 Grade-limited access** (the D5 pairs):
      * the route is the path with the least steep length, ties broken by length (lexicographic Dijkstra);
      * a pair is served if both endpoints join within 60 m, the route's steep share (above 15 %) is at most 10 %,
        and the detour is at most 4.
    * **Game readiness** (secondary, with real towns as reference):
      * **G1 drivable access:** the M6 pairs on the subgraph of pieces whose grade over a 30 m window is at most
        15 %, with a detour of at most 4; a 12 % cap as a sensitivity;
      * **G2:** the share of hairpins tighter than a 12 m centreline radius;
      * **G3:** vertical curvature, the p95 of grade change per 10 m;
      * **pedestrian:** reach with stairs, and reach on ramps of at most 8 % without stairs.
    * **C2ST (secondary diagnostic):**
      * Cells: (window, slope class) with at least 0.02 km² of land. Cells with no streets are kept (density 0,
        other features missing, plus an indicator).
      * Generated cells: the same windows, one seed per window by hash, so real and generated are 1:1.
      * Classifier: L2 logistic regression (C = 1) on standardised M1–M5 features, no tuning; boosted trees as a
        sensitivity.
      * Validation: leave one town out, with each town's real and generated cells in the same fold.
      * Reported: AUC with a CI bootstrapped over towns, and a permutation p from shuffling labels within towns.
      * The 20–30 % and > 30 % classes are pooled if either has fewer than 40 real cells.
      * Also run without M1 (fabric at a given density).
  * **Scoring (ruling: the statistics reviewer's robust z).**
    * z = (t(x) − median of t(real dev-H)) / scale, bootstrapped over towns. The distance is |z|, capped at 5; the
      sign is reported.
      * t(x) = log(x + c_k) for positive metrics, with c_k = 0.1 × the median of the positive real values, declared
        per metric (zero-inflated metrics such as the lens share and the hairpin rate stay defined).
      * scale = max(1.4826 × MAD, (q90 − q10) / 2.563, a declared resolution floor per metric).
    * The real towns' leave-one-out |z| distribution is published as the null, and margins are set relative to it.
    * Composites are equal-weight means of |z| over the declared cells.
  * **Validation gate, passed before freezing.**
    1. **Real towns fit the envelope.**
       * Pooled leave-one-out coverage (|z| ≤ 2) is at least (n − 1)/(n + 1) − 0.05.
       * The leave-one-out composite percentiles are roughly uniform (KS p > 0.05).
       * A town below 0.75 coverage is flagged for a mapping check.
    2. **Adversarial networks fail.** Each one's composite is above the 95th percentile of the real leave-one-out
       composites, and it fails the metric aimed at it:

       | Adversarial network | Must fail |
       |---|---|
       | Single switchback road | M1 or the hairpin rate |
       | leastcostpath bundle | Lens share |
       | Grid running straight up the slope | M2 |
       | Raster zigzag with 4–9 m legs | M7 |
       | Parallel contour streets that never join | Largest-component share or loops |

    3. **Later:** the metrics agree with the human pilot.
    * The suite's code is then hashed and committed. This must happen before any TNg-H generation on dev-H.
  * **Primary endpoints** (hold-H; the unit is the settlement; the contrast is hybrid − A0).
    * **P1a, superiority on M6 steep-land access: hybrid − A0, unconditional.**
      * Density guard: in every slope class, the median across settlements of the hybrid's M1 density z lies within
        ±2. Requiring it of every settlement would fail by chance, since real towns miss about 8 % of cells.
    * **P1b, superiority on M6: hybrid − (A0 + terrain-blind A5b connectors).** Read at matched added street length,
      L* = min(hybrid added km, A5b added km), as in D2. This is the "not just more road" test; against A0 alone a
      matched length would be L* = 0.
    * **P2, non-inferiority on the untargeted realism composite.** Equal weights over these cells:
      * M3 run p90 (10–20 / 20–30 / > 30 %) and the hairpin rate;
      * M4 lens share, dead ends per km and largest-component share;
      * M5 Louf Φ and block–contour alignment;
      * M7.
      * "Untargeted" means no J-based arm was fitted to these. M1 and M2 are reported as targeted secondaries,
        because J was fitted to the F22 density, contour and > 15 % curves (M18 D12/D13).
    * P1a, P1b and P2 form an intersection–union test: the claim holds only if all three reject, each one-sided at
      α = 0.025. The P2 margin is set from the real leave-one-out null before hold-H opens.
    * **Untargeted means untargeted for every method** (ML sign-off):
      * After the D3 freeze, no method is refitted on dev-H M1 or M2.
      * No parameter of any method (the hybrid, the externals, the baselines) may be fitted to a P2 metric on dev-H.
      * Each method's design entry declares the statistics it is fitted to. Any P2 metric used in fitting moves out of
        P2 to "targeted" for every method, declared before hold-H opens.
      * The hybrid's stairs and run rules are therefore fitted to M2-type grade statistics only, never to M3 or M4.
  * **Multiplicity.**
    * Holm correction over a declared family of at most 10 key secondaries: G1, M2 share above 15 %, M1 density,
      tortuosity, the C2ST AUC, and pedestrian reach.
    * Every other cell is descriptive, with CIs and no p-values.
  * **Review record** (all five asked for changes; every change is adopted above):
    * **Statistics:** robust z, the missing-cell rule, gates 1 and 2, the intersection–union test, Holm limited to
      key secondaries, C2ST as secondary. F2's fallback was accepted.
    * **Urban morphology:** the lexicographic M6 route, lens faces, stroke runs, hairpin chains, contour index from
      5 %, the cell minimums.
    * **Graphics:** tortuosity, largest component and loops, M6 at matched length and inside the density envelope,
      targeted adversarial failures, two more adversarial cases.
    * **ML:** the full C2ST specification, freezing before TNg-H generation, the untargeted composite.
    * **Games:** G1–G3 and pedestrian ramps.
    * **Where reviewers proposed different fixes:**
      * Scoring: the robust z replaces the percentile-plus-IQR version, because it stays continuous beyond the real
        range.
      * Bundles: lens faces are primary and the 2A/P version is a sensitivity, because they are defined by topology,
        not by a width threshold.
  * **Sign-off round.**
    * Games and graphics approved. Urban morphology, statistics and ML each raised one blocking objection, and all
      three are adopted above:
      * the zero-inflation-safe robust z;
      * P1 split into P1a (unconditional, with a density guard on the median) and P1b (against terrain-blind repair
        at matched length);
      * untargeted for every method.
    * **Ruling on P1:** statistics proposed the matched contrast against A5b; ML proposed an unconditional contrast
      against A0. Both are kept, inside the intersection–union test.
    * **Notes adopted.**
      * G2 (radius) and G3 (vertical curvature) are computed on native geometry. G3 is flagged as limited by the
        32 m DEM and checked against PNOA LiDAR on the Spanish dev-H towns (games).
      * The leastcostpath adversarial case is the actual M18 Petra seed-1 network (graphics). If it passes gate 2 on
        lens share, the 2A/P < 20 m sensitivity becomes primary. This is decided before the freeze, before any
        generator is scored.
  * **Status:** D3 is approved as a specification. It is frozen (code hashed and committed) only after it is
    implemented and passes the validation gate on the dev-H real towns. That needs the dev-H OSM data, after the
    pinned snapshot.
* **D4. The grade hybrid (the method).** The draft was revised after all five members reviewed it; the chair's
  rulings are marked.
  * **Inputs (ruling: ML).**
    * The frozen TNg receives GLO-30 as trained. J and δ were fitted with GLO-30 (M18 D10/D13), and feeding FABDEM to
      a GLO-30 model would be an untested input shift.
    * FABDEM as model input is a declared sensitivity run. Native FABDEM inputs belong to TNg-32 and TNg-H (D1).
    * Rules, the router and the metrics use FABDEM at 32 m.
  * **Generator and pairing (statistics).** The baseline generator is **A0 + H3**, the frozen J with whole-town
    density calibration. Every arm post-processes the same A0 + H3 sample path, so all contrasts stay paired.
  * **Router used by H1 and H2 (graphics; supersedes urban morphology's ×0.5).**
    * The A5 cost on the 4 m, 16-neighbour grid, with the grade cap at **15 %** (the short-ramp rule below allows
      more on short pieces).
    * Existing vehicular edges cost ×0.2: the Emilien-style reuse discount, so switchbacks reuse contour streets.
    * Cells within 10 m of an existing edge but not on it cost ×10, so the router cannot make braids or lens faces.
    * New routes cross existing streets only at nodes, or at 60° or more.
    * Only the new pieces are committed.
  * **Route geometry before commit (graphics, urban morphology).**
    1. Douglas–Peucker at 8 m, rejecting any simplification that would push a piece over the cap.
    2. Chaikin smoothing to a centreline radius of at least 12 m (the G2 constant).
    3. Grade re-checked. Then the commit, under the sampler's rules (water, crossings, structures).
  * **H1 Connectors.** A5's eligibility, destinations and order, with the router and geometry above.
  * **H2 Steep runs.**
    * A steep run is a stretch along a stroke whose grade over a 30 m window exceeds 15 %.
    * Runs are processed longest first, ties broken by hash, and the condition is re-checked after each change.
    * **A piece above 30 % becomes stairs regardless.** 30 % is the "implausible" line of M18 F12 (urban morphology).
    * **A short ramp (≤ 60 m, at most 25 %) stays vehicular** (games).
    * Every other run, in this order:
      1. **Stairs**, if its ends stay joined within the drivable subgraph (G1: pieces at most 15 %, plus allowed
         short ramps) with a detour of at most 4 (games: the G1 criterion, not M6's 10 % share).
      2. **Switchback**, if the router finds a route between its ends with a detour of at most 4. That route is
         committed and the run becomes stairs.
      3. **Kept**, otherwise: the run stays vehicular and is logged.
    * **Stair placement (graphics).** Run ends are snapped to a junction within 30 m. Vehicular stubs shorter than
      30 m are absorbed. Pieces of 15–25 % are typed as stepped ramps, and pieces above 25 % as stairs with landings.
  * **H3 Whole-town density calibration.**
    * It may be fitted to M1 only.
    * Post-hoc pruning or a change inside growth are both allowed, because every arm shares its output.
    * Its design is written as D4b after the one dev-H look.
  * **Arms.**
    * **Primary:** the hybrid, A0 + H3 + H1 + H2, in whole-town mode, on settlements.
      * P1a is the hybrid − (A0 + H3).
      * P1b is the hybrid − (A0 + H3 + A5b), at matched added length.
    * **Secondary:**
      * H1 + H2 without H3;
      * H1 alone;
      * A5 and A5b;
      * FABDEM-input and GLO-30 sensitivity runs;
      * the 60 m constant at 40 m and 100 m, reported and never selected.
  * **Learned contribution (ML).**
    * The identical H1 + H2 (+ H3) is applied to citygen, leastcostpath, the Emilien-style baseline and the
      terrain-blind N0.
    * The learned contribution is the hybrid on TNg minus the hybrid on each baseline, on P2 and the C2ST at matched
      density.
    * The share of the final vehicular length that comes from the learned model is reported.
  * **Constants, fixed now and fitted to no data:**

    | Constant | Value | Source |
    |---|---|---|
    | Drivable grade | 15 % | M6 / G1 |
    | Stairs regardless | 30 % | M18 F12 |
    | Short ramp | 60 m and 25 % | 60 m is a reviewer's judgement, with sensitivities |
    | Steep share, detour | 10 %, 4 | M6 |
    | Reuse discount, proximity penalty | ×0.2, ×10 within 10 m | Emilien-style |
    | Crossing angle | 60° | |
    | Douglas–Peucker, radius | 8 m, 12 m | G2 |
    | Snap and stub | 30 m | |

    * If the hybrid's M2 z falls below −2 (over-adaptation: real steep land keeps about 15 % of its length above
      15 %, F9), it is reported with its sign and not tuned.
  * **Fitting statistics.**
    * H1 and H2 are fitted to nothing. H3 is fitted to M1.
    * The real steps layer is used only to evaluate: stair density per class, and pedestrian reach. This replaces
      D1's "stairs rule fitted to the real steps layer".
  * **Consequences for D3, declared before any look.**
    * H2 shapes M3 directly (runs and hairpins), so M3 leaves P2 and becomes targeted (ML).
    * **P2 is now:**
      * the M4 lens share, dead ends per km and largest-component share;
      * the M5 Louf Φ and block–contour alignment;
      * M7.
    * **Holm key secondaries (9):**
      * G1;
      * M6 at 12 % and at 10 %;
      * M2 share above 15 %;
      * M1 density;
      * M3 runs (targeted);
      * the C2ST AUC;
      * pedestrian reach;
      * the learned contribution on P2.
  * **Development looks (statistics).**
    * One dev-H look for D4, with 24 runs per settlement (the hold-H count). Its variance is the internal pilot that
      sets hold-H's N.
    * Then at most one amendment, D4b (H3's design, logged bug fixes), then the freeze.
    * Everything else above is frozen now.
    * **P2 stays sealed during development.** P2 cells on dev-H are computed by script and sealed unread until the
      hybrid is frozen. The look reports only M1, M2, M6, G1, the run logs and maps.
  * **Logged per town:** runs turned into stairs, switchbacks, runs kept and switchbacks that failed for lack of
    room; the kept km. "Drivable" is never claimed for a town that keeps any run.
  * **3D (games), for the walkable towns and the study stimuli:**
    * stairs are built as narrow stepped alleys, 2–4 m wide, with landings (MP v2's full-width steps were rejected
      by the owner);
    * bollards at stair ends;
    * a smooth collision ramp for the walker;
    * separate Godot navigation layers for cars and pedestrians;
    * the class-tagged graph exported for traffic AI.
  * **Novelty, as the graphics reviewer framed it.** The claim is the combination:
    * learned fabric;
    * grade typing into a two-class network with a drivability guarantee;
    * a human study against the Emilien-style baseline.
    H1 and H2 alone are Galin-2010-style post-processing.
  * **Sign-off round.** Graphics, games, ML and urban morphology approved. Statistics raised one blocking objection,
    adopted below with the non-blocking notes.
    * **Added length for P1b (statistics, blocking).**
      * Added km = the gross new vehicular km committed by H1 and the H2 switchbacks. Km turned into stairs is not
        subtracted.
      * The hybrid saves a snapshot after each connector or switchback, in its declared order.
      * Both curves are read at L* = min(gross hybrid km, A5b km).
    * **When the grade re-check after smoothing fails** (graphics, urban morphology), in order:
      1. retry with one fewer Chaikin iteration;
      2. re-route with the hairpin vertex pushed 12 m outward;
      3. otherwise reject the route. In H2 the run falls to "kept", logged as a smoothing failure.
    * **Reuse set** (urban morphology): the ×0.2 reuse set excludes the run being replaced and every edge already
      typed as stairs.
    * **Logging** (games): also log the vehicular pieces cut off by stairs forced above 30 %, so any drop in G1 can be
      traced.
    * **Externals at matched density** (ML).
      * citygen, leastcostpath, the Emilien-style baseline and N0 run as declared density sweeps. Each is
        interpolated at the hybrid's own M1 density; H3 is not applied to them.
      * The sweep grids are declared in D4b, before any external is scored on dev-H.
    * **Wording rule, declared now** (ML). "Learned fabric" is dropped from the paper's claim if either:
      * the hybrid on TNg does not beat the hybrid on the Emilien-style baseline on P2 or the C2ST; or
      * under half of the final vehicular length comes from the learned model.
  * **Status:** D4 is approved; the constants and the primary input are frozen. Next:
    * the dev-H data build (after the pinned OSM snapshot);
    * D3's validation gate;
    * the one dev-H look;
    * then D4b (H3 and the sweep grids).
* **D3a. Implementation rulings before the freeze.** The implementer (47bf4b5) reported 10 ambiguities. The rulings
  are by urban morphology (U), statistics (S) and graphics (G), with the chair's ruling where they differed.
  * **Cells (U).**
    * Definedness uses the unit's own real network.
    * The reference median and MAD use only real towns with at least 1 km in the cell, and need at least 8 of them.
      With fewer, 20–30 % and > 30 % are pooled; otherwise the cell is descriptive only.
  * **Transforms (S).**
    * log(x + c) for unbounded positive metrics: density, grade, runs, rates, face areas, dead ends, loops.
    * logit((x + c)/(1 + 2c)) for proportions: coverage, share above 15 %, 3-way and 4-way shares, close15, the
      largest-component and lens shares, G2, the access shares.
    * Identity for cosines, Φ and tortuosity.
    * c is per metric × cell, frozen from the real dev-H towns. Floors come from measurement resolution only.
    * The max(MAD, quantile range, floor) scale is logged as a declared deviation.
  * **Native geometry (G, U).**
    * Native geometry is densified to ≤ 5 m and simplified by Douglas–Peucker at 1 m, identically for real and
      generated networks.
    * It is primary for M3 hairpins, G2 and G3; the convention is the sensitivity.
    * The hairpin window widens to 60 m (U).
  * **M3 runs (chair's ruling).** U asked to align M3 with H2, S asked not to.
    * Runs are measured on a 32 m grade window (the DEM resolution, a measurement floor), at least 32 m long, sign
      kept, along strokes through junctions.
    * M3 is declared targeted and is never used as realism evidence for the hybrid.
  * **M6 (U, S).**
    * The share check applies to the lexicographic route only. M6 is stated as conservative.
    * Also reported, descriptively: served share by the steep endpoint's class.
  * **Loops (U):** bounded faces whose representative point lies on class land, per km² of class land.
  * **Block–contour alignment (U):** only on faces with elongation ≥ 1.5.
  * **Gate 1, leave-one-out (S).** Each pair of towns is scored against a reference that excludes both. The KS p is
    approximate, and is stated as such.
  * **Gate 2 (S, G).**
    * Each adversarial network is run on every dev-H town, and must fail in at least 90 % of them.
    * The leastcostpath bundle, contour-parallel and zigzag cases must also exceed the 95th percentile of the real
      leave-one-out P2 composites.
    * The zigzag's M7 |z| must exceed the maximum of the real leave-one-out values. The composite margin is reported.
    * Nothing is changed to make a case pass; a failure is recorded as a limitation.
  * **C2ST (S).**
    * Checked against a reference solver.
    * At least 1,999 permutations, so it can reject under Holm.
    * The boosted-trees sensitivity is dropped (no scikit-learn in the venv), declared before any generated data is
      scored.
  * **Structures (S):** the bridge and tunnel share per class is reported, and a generator above the real 95th
    percentile is flagged.
* **D3b. Last implementation confirmations (statistics, urban morphology), before any real town is scored.**
  * **Confirmed:**
    * 20–30 % and > 30 % pool into 20 %+ when either has fewer than 8 reference towns;
    * logit metrics use the same c;
    * the zigzag gate compares its M7 |z| with the maximum real leave-one-out value.
    * The synthetic 6/9 zigzag result is a unit test. The gate is decided on real dev-H towns and recorded either way.
  * **Corrected:** the ≥ 1 km reference rule applies to street-shape metrics only.
    * M1 (density, coverage) needs only ≥ 0.05 km² of class land in at least 8 towns.
    * A density of 0 is scored through log(x + c), so deletion on the steepest land is always scored.
  * **Scale floors (both reviewers).** floor = the span on the t scale of one measurement unit at the reference-median
    cell, divided by √12 (the SD of a uniform quantisation error).
    * The units are computed from the real dev-H towns and frozen with c:

      | Metric | One measurement unit |
      |---|---|
      | Density | 0.03 km ÷ median class land |
      | Coverage | 0.001 km² ÷ median class land |
      | Share above 15 %, lens share, largest-component share | 10 m ÷ median cell street length |
      | 3-way share, 4-way share, close15 | 1 ÷ median junction count |
      | Dead ends, hairpin rate | 1 ÷ median street km |
      | Loops | 1 ÷ median land km² |
      | G2 | 1 ÷ median hairpin count |
      | Run p50 / p90 | 10 m |
      | Contour index, Φ | 0.01 |
      | Grade p90, grade change | 0.5 pp |
      | Face area | 64 m² |
      | Block–contour cosine | 0.03 |
      | Tortuosity | 0.003 |
      | Access | 0.001 |

    * The number of floor-bound cells on real dev-H is reported, and the floors are not revised after seeing it.
* **F3. dev-H terrain walk (terrain only, no streets).** `data/processed/m20/walk/dev_terrain.csv`:
  * 1,659 evaluated, 82 eligible, 4 without DEM.

    | Stratum | Evaluated | Eligible | Band B |
    |---|---|---|---|
    | R1 cities | 90 (list exhausted) | 2 | 1 |
    | R1 towns | 162 | 9 | 0 |
    | R2 cities | 215 (exhausted) | 8 | 0 |
    | R2 towns | 150 | 10 | 1 |
    | R3 cities | 488 (exhausted) | 8 | 0 |
    | R3 towns | 90 | 10 | 2 |
    | R4 cities | 224 (exhausted) | 8 | 0 |
    | R4 towns | 96 | 9 | 2 |
    | Villages | 144 | 18 | 2 |

  * Weighted by built-up area, 12 % or more is rare: about 5 % of towns and 1–4 % of cities. Band B (20 % or more)
    is rarer still.
* **D1 implementation note (before any OSM is seen): meeting "at least a third band B" on dev-H.**
  * The terrain walk stopped at 3 × quota eligible per stratum, which found only 6 band-B settlements. The quota is
    8 of 24.
  * **Fix:**
    * Each town stratum's walk continues, in list order, until it holds 6 band-B eligible settlements (3 × its 2
      band-B slots) or its list ends.
    * In the OSM round robin, a town stratum accepts at most 1 band-A settlement of its 3. City strata are band-merged
      (fewer than 3 × quota eligible, D1), so they have no band constraint.
    * The towns alone then give 8 band-B settlements; any band-B city adds to that.
  * If a town stratum runs out, the D1 fallback applies, and the shortfall is reported.
  * Train-H has no band rule.
* **D4a. Implementation rulings before any dev-H run** (graphics G, urban morphology U, games T; code a6155fb).
  * **Smoke run** (M19 natural / TNg_s0 / seed 1; bit-identical A0 replay).
    * H1: 16 connectors complete. Their share of length steeper than 15 % is 0.22, against the stored A5's 0.33.
    * H2: 9 pieces forced to stairs (above 30 %); 1 run turned into stairs and 2 into switchbacks; 5 kept, all
      "failed for room".
    * "Drivable" claimed in 12 of 16 tiles.
  * **Confirmed:**
    * a. A turn cost of 1 m per radian, a fixed constant added to the D4 table (a Galin-2010-style curvature cost;
      G).
    * b. Short ramps searched in two passes: first ≤ 15 %, then ≤ 25 % (T).
    * c. The ×10 band is also charged at step midpoints (G).
    * d. "At nodes" means within 8 m of a junction or dead end (G).
    * e. The 12 m radius is checked on G2 hairpins, with Chaikin at 2–4 iterations; the minimum radius of other
      bends is logged (G).
    * f. Switchbacks are all-or-nothing. Stair snapping goes outward and stops at the first junction, else the stub
      is absorbed. Pedestrian pieces of at most 25 % are stepped ramps (U).
  * **Ruled:**
    1. **Pieces are at most 15 %, except declared short ramps (≤ 60 m, ≤ 25 %).**
       * Back-to-back ramps need at least 30 m at or below 15 % between them; otherwise they merge into one run (T).
       * **A hairpin apex counts as a short ramp only if it is at most 20 % and at most 40 m** (T, chair's ruling;
         stricter than G's 25 %). A vehicle cannot climb 25 % while turning at a 12 m radius.
       * The 3D builder grades apices to at most 12 % by cut and fill (T, G).
    2. **The radius check is binding** (G). A G2 hairpin under 12 m is re-routed once, with the apex pushed 12 m
       outward; if it still fails, it is rejected and logged as `smoothing_fail`. Tight radii are never accepted.
    3. **No loop on a stick** (U). The dead end's own degree-2 chain is excluded from both the reuse set and the
       ×10 band, and every H1 route starts at the dead-end node itself.
    4. **Search box** (U, G). The H2 router pad is max(60 m, 2 × the straight distance), clipped to the town. This
       contains the detour ≤ 4 ellipse; the 60 m box was the binding limit, so this is a bug fix, not tuning. The
       runtime is logged.
* **D4c. Two-pass router (graphics, urban morphology; before any dev-H run).**
  * **The problem.** The D4a smoke run showed that the ×0.2 reuse discount pulls cost-minimising routes onto long
    detours along existing streets. They then violate detour ≤ 4, so the run is kept. One 129 m run reached a detour
    of 5.69.
  * **The rule, for H1 and H2 alike:**
    * Pass 1 uses reuse ×0.2.
    * Pass 2 runs only if pass 1 breaks detour ≤ 4. It uses reuse ×1, with the ×10 band, crossing rules, pad and
      tie-breaks unchanged.
    * If both fail, the run is kept. Nothing further is added.
  * **Measurement:**
    * The detour is measured on the full route length (new pieces plus reused streets) in both passes.
    * The D4a problem-3 rules still hold in pass 2: start at the dead-end node, with its own chain outside the reuse
      set and the band.
  * **Logged per town:**
    * which pass succeeded, and the pass-2 share of switchbacks and connectors;
    * the lens-face share of pass-2 pieces. These lens faces count in P2 as declared and are never filtered.
    * If pass 2 dominates, it is reported that ×0.2 mostly fails the detour rule. ×0.2 is not tuned.
  * **Smoke run after D4c** (M19 natural / TNg_s0 / seed 1; bit-identical replay):
    * H1: 17 connectors complete, 1 of them by pass 2.
    * H2: 4 switchbacks (2 by pass 2); kept runs fell from 5 to 3; no lens face from pass-2 pieces.
    * "Drivable" holds in 13 of 16 tiles, against 12 before.
  * **Two readings, both as declared, with nothing tuned:**
    * For H1, detour ≤ 4 and A5's route-length limit both bind.
    * In some contour-street scenes even reuse ×1 cannot meet detour ≤ 4, because the ×10 band makes leaving a flat
      street costly. Those runs are kept, per D4c.
* **D3c. Scoring the leastcostpath case, and the lens switch (statistics, graphics, urban morphology; before any
  dev-H town is scored).**
  * **The problem.** The leastcostpath case is the actual M18 Petra seed-1 network (D3 sign-off). It exists on one
    site only, so D3a's "fail on ≥ 90 % of dev-H towns" cannot apply.
  * **Measured on Petra only** (FABDEM 32 m on the 2048 m site box; unit = the city limits, 1.47 km²; no dev-H data):

    | | Real Petra (vehicular) | lcp Petra s1 |
    |---|---|---|
    | Street in the limits | 4.49 km | 24.9 km |
    | Lens share (D3 choice 9), every class | 0 | 0 |
    | 2A/P lens share, 10–20 % / 20 %+ | 0 / 0 | 0.019 / 0.001 |
    | Loops per km², 10–20 % | 5.3 | 162 |
    | Faces in the unit | 4 | 115 |
    | Faces with 2 / ≥ 3 junctions | 2 / 2 | 0 / 115 |
    | Faces with 2A/P < 20 m | 25 % | 59 % (2.6 % also elongated > 6) |

  * **Reading (graphics, urban morphology; confirmed by the per-face table).** The convention's 8 m merge puts a
    junction wherever independently laid least-cost paths come close. A braid of three or more strands becomes a mesh
    of faces with at least 3 junctions, never a two-junction lens.
    * Real redundancy, such as a bypass beside an old road or a street split around an island, does make two-junction
      lenses. Lens share is a valid measure of redundancy, but it cannot detect braiding.
    * Terraced contour streets make 4-junction blocks. That is real morphology, rightly not a lens.
  * **Rulings.**
    1. **Signed targeted checks, for all five cases (statistics, graphics; blocking).**
       * The declared direction of each adversarial flaw:

         | Case | Targeted cells and direction |
         |---|---|
         | Switchback | density −, coverage −; hairpin rate + |
         | leastcostpath | 2A/P lens share + |
         | Fall-line grid | contour index −, grade p90 +, share above 15 % + |
         | Raster zigzag | tortuosity + |
         | Contour-parallel | largest-component share −, loops − |

       * A targeted group fails only when its mean sign-oriented z is above the 95th percentile of the real
         leave-one-out sign-oriented group means. The zigzag's M7 must exceed the largest real leave-one-out
         sign-oriented M7.
       * Cells scored by rule (the generator has < 1 km of street, or the value is undefined) are left out of the
         signed means. They are not evidence of the declared flaw, and a group with no scored cell cannot fail. They
         still score |z| = 5 in the composites.
       * The suite and P2 composites stay two-sided (mean |z|). This only tightens the gate.
    2. **The 2A/P switch is triggered now (statistics, graphics).** lcp's lens share is 0, so its signed lens z is
       ≤ 0 against any reference: it cannot fail the topological lens check. That follows from lcp's own values, not
       from dev-H.
       * By the D3 sign-off rule, lens_share_2ap (2A/P < 20 m, elongation > 6) becomes the primary lens metric in
         the lens group, M4, P2 and the C2ST features, before gate 1 runs. c and the floor are frozen with the plan.
       * The topological lens share is the sensitivity, in P2 and in the lens group.
       * The P2 margin comes from the switched null. 2A/P falls under "untargeted for every method" (D3).
    3. **lcp is gated once, as a single-site check (chair's ruling).**
       * The two reviews differed. Statistics would report lcp's 2A/P descriptively only; graphics would gate lcp once
         on the signed 2A/P group. The frozen suite is what must be validated, so lcp is gated on the switched suite.
       * The topological-lens result is reported alongside.
       * No outcome changes anything further: there is no second switch and no third detector. If lcp is not caught,
         that is recorded as a limitation.
       * It is reported as a single-site check (n = 1), a declared deviation from D3a's 90 % rule, never as passing
         gate 2.
    4. **Real Petra as the site control (statistics, graphics; blocking).**
       * Real Petra is scored exactly as lcp: the same field, the same dev-H reference (Petra is not a dev-H unit, so
         no town is left out), and the dev-H plan's cells. There is no Petra-specific pooling.
       * Definedness comes from Petra's real network.
       * A check counts as catching lcp only if real Petra passes it. If real Petra also fails it, the check is
         indeterminate, recorded as a limitation.
       * Reported: the defined cells, every cell's z for both networks, and lcp minus real Petra per group, including
         whether P2 fails lcp through tortuosity or dead ends rather than bundling.
    5. **Mesh diagnostics, descriptive only (urban morphology).**
       * Per class, for every method, with hybrid − A0 CIs. They are outside P2, Holm and the gate.
       * The diagnostics:
         * face area p10 (the median and p90 exist);
         * the share of faces under 0.1 ha;
         * the share of acute junctions (two angularly adjacent incident edges under 30°);
         * the median junction-to-junction length.
       * Their c and floors are frozen with a separate plan at the D3 freeze. The units are 64 m², 1 ÷ median face
         count, 1 ÷ median junction count, and 10 m.
       * Loops and face area are also reported as mesh diagnostics. No P2 change beyond ruling 2.
    6. **Reporting (urban morphology).** Per cell: the number of dev-H towns with lens share > 0 and with 2A/P > 0. 20 m
       is about the depth of a terraced block (a 25 × 150 m block has 2A/P ≈ 21 m).
    7. **Disclosure.** leastcostpath is an M20 external (D4). Its Petra seed-1 output (M18) and the values above were
       seen before the freeze. No external parameter is fitted to them; its dev-H sweep grid is declared in D4b.
  * **The P2 limitation, drafted now:**
    > P2's redundancy component (2A/P < 20 m faces; topological lens faces as a sensitivity) detects divided streets
    > and two-junction bypasses. It does not detect braids of independently laid paths, which the 8 m merge turns into
    > meshes of multi-junction faces. Against such failures P2 has reduced assay sensitivity, so non-inferiority on P2
    > does not exclude braiding. Mesh diagnostics (face-size distribution, acute junctions, junction spacing, loops)
    > are reported descriptively for every method.
* **D3c addendum: the reference set (statistics; before any dev-H town is scored).**
  * **Reading of D1, confirmed.** The D3 reference (the plan, gate 1's n and threshold, gate 2, the leave-one-out
    nulls and the P2 margin) is the accepted dev-H towns and cities only.
    * The 6 dev-H villages ("evaluation only", and gated at 4 km rather than 15, which would widen the envelope) and
      the 2 declared anchors are scored against it descriptively. That means defined-cell counts, z per cell, the
      composite and the share of cells with |z| ≤ 2, with no p-values. They decide nothing.
    * Wadi Musa adjoins the Petra site, so having it in the reference would partly score the D3c Petra control
      against itself.
  * **The anchors** (`scripts/m20_walk.py anchors`): the frame polygon that contains each declared coordinate.
    Terrain and completeness are reported, never used to admit or drop an anchor.
    * Wadi Musa is DUC_27099 (town). Slope 19.1 % (band A); completeness passes, with 70 km of vehicular ways.
    * Aguas Calientes is RC_326905 (village). Slope 20.1 %, with 0 km of vehicular ways but 100 % of its built-up
      cells near a mapped way. It is a car-free town (no road access), so its vehicular cells are undefined and it is
      reported as a pedestrian-only anchor.
  * **Conditions (statistics).**
    1. The unit lists are hash-frozen after the OSM gate, before any metric (`scripts/m20_gate.py townlist` →
       `configs/m20_pool/dev_units.json`), and n is that count. If fewer than 24 towns and cities pass, gate 1 uses
       the actual n. **A town flagged in gate 1 (coverage < 0.75) is never removed** (the declared criterion); its
       mapping check is descriptive, with maps.
    2. A leave-one-block-out sensitivity (the 1° block, at most 2 towns per block) is reported for gate 1 and the
       nulls. It decides nothing.
* **D1 implementation note 2: the quota fallback (statistics; before dev_accepted is final).** In the OSM round robin,
  all 27 eligible dev R3 towns are Chinese, and the first 14 failed the gate, mostly on < 15 km of vehicular ways.
  * **Reading, confirmed.** D1's fallback ("passes to the next stratum of the same region and size; failing that, the
    same region; failing that, it is reported unfilled") refers to the plan's region × size × band strata. After the
    round robin:
    1. **The other band.** An unfilled dev town stratum lifts its band-A limit and revisits, in list order, its
       candidates never tested at the gate.
    2. **The same region, the other size.** Candidates are taken after that stratum has filled its own quota, in list
       order, and never one its own walk took.
    3. Anything still unfilled is reported.
  * **Conditions.**
    * The block cap holds throughout.
    * **On dev-H a band-A fallback is taken only while band B stays at least a third of the towns and cities** (the
      hard D1 constraint). A refusal is logged as `band_B_third`.
    * An accepted fallback keeps its own stratum label and is credited to the unfilled stratum.
    * Cached OSM responses make the re-run reproduce every earlier outcome. Train-H uses the same rule, under its cap
      of 30.
  * **Selection on mapping quality.** The Chinese towns failing on vehicular length are selection on mapping quality.
    This is shown in the CONSORT flow and the balance table. The 15 km threshold is unchanged.
* **D4d. The whole-town round cap and city cores (all five members; before the dev-H look, before dev_units is
  hashed).**
  * **The problem** (dev-look runner, scratch test; no dev-H output exists). The frozen whole-town sampler stops at
    max_rounds = 6000.
    * On an 8192 m domain (Perugia scratch city, 256 start windows) both seeds used all 6000 rounds, and only 1 of
      256 windows ever got a start. A0 was one truncated network: 24 km inside the limits, out of about 140 km in the
      domain, at 26 min per run.
    * Petra (2048 m) needed 1570 rounds, so no M18/M19 site run hit the cap.
    * Most eligible towns need 4096 m (64 windows), which is borderline. Cities need 8192 m, and 3 eligible cities in
      each of R2–R4 exceed 8192 m, which the builder refused.
  * **Rulings.**
    1. **The cap (blocking, ML; declared before the look, so the look and hold-H run the same generator).**
       * max_rounds = 6000 × max(1, n_windows / 16). It is identical for every M18/M19 run.
       * Rounds used and windows started are logged per run.
       * A run that hits the cap is flagged, never dropped: dropping it would select on the outcome.
       * The cap is not revised after the look.
       * One M19 run is replayed bit for bit under the new code.
    2. **City cores (option B; rejected: the cap alone, about 960 CPU-h for hold-H cities, and 512 m tiles, which
       cannot test whole-settlement connectivity or drivability).**
       * Every GHSL urban centre (city) is scored on a 4096 m domain centred on its population-weighted centroid.
       * **The unit is the city core: the polygon ∩ the 3496 m square centred on the population-weighted
         centroid.** It is not the historic centre.
       * The crop is geometry, fixed before any outcome. It applies identically to the real network, the generated
         networks and the reference.
       * Towns (GHSL DUC) and villages stay whole, at their own domain size, under the scaled cap.
       * The claim says "town" and "city core", never "city".
    3. **Size is never a gate (statistics, urban morphology).** The builder's size refusal was an undeclared exclusion
       of the largest cities (often merged or linear-valley ones), so it is removed, and every city takes the core
       rule.
       * The walk never applied a size rule; the OSM gate is on the whole polygon. The accepted lists are therefore
         unchanged, and no re-walk is needed.
    4. **Reported for each core (urban morphology):** its built-up-weighted slope and land km² per slope class, next
       to the polygon's.
       * The polygon's band label is kept, and the core is never re-gated on slope.
       * Town–city contrasts mix size with cropping, so they are never read as morphology. If R3 ends mostly as city
         cores (the D1 fallback), that is stated.
    5. **Fallback units** keep their own stratum label. A descriptive sensitivity without them is reported.
    6. **The runner's choices** (`scripts/m20_devlook.py`):

       | Choice | Ruling |
       |---|---|
       | 1. A0 = J with δ = 0 | Approved. Only that setting runs on dev-H, labelled "A0 (J, δ = 0)", not M19's A0 |
       | 2. grad_at = FABDEM 32 m | Approved; A5 and A5b use the same |
       | 3. The hybrid runs domain-wide | Changed: every log is attributed to the unit and the per-unit counts are primary. P1b's added km is counted inside the unit. "Drivable" is claimed per unit only |
       | 6. What is open (blocking) | Changed: the look opens M1, M2, M6, G1, the run logs and the maps (D4's list). Pedestrian reach and the mesh diagnostics are sealed with P2 |
       | 7. The pilot (blocking, statistics) | Changed; see below |

    7. **The pilot's N formula, committed now (statistics).**
       * The variance of a settlement's mean paired difference is v = σ²_b + σ²_w / R. Here σ²_b is the
         between-settlement variance of town-mean differences, σ²_w the run-to-run variance, and R = 24 runs.
       * N = (1.96 + 1.2816)² × v⁺ / δ², with δ = 0.008 on m6_15 (one-sided α = 0.025, power 0.9).
       * v⁺ is the one-sided 80 % upper confidence limit of v (chi-square on the settlement count), never the observed
         value.
       * The between-block variance (1° blocks) is reported, and N is recomputed from block means as a sensitivity.
         The larger N is used, and N ≥ 40 (D1).
       * If compute binds, R is cut uniformly before N.
       * The look's contrast is H1 + H2 − A0. H3 is designed only after it (D4b), so the pilot is a proxy for P1a.
         The look has no A5b arm, so **P1b goes unpiloted**; this is stated.
  * **Compute under B:** about 190 CPU-h for hold-H (40 × 24 × about 12 min), and less for the look.
* **F4. The walk is complete: who got in, and why (D1; terrain and the OSM completeness gate only, no street metric).**
  Records are in `configs/m20_pool/walk/`: accepted lists, flow and balance tables, and the gzipped terrain and OSM
  logs.
  * **dev-H: 21 towns and cities (target 24), 6 villages, 2 anchors.**
    * Hash-frozen in `configs/m20_pool/dev_units.json`: reference sha256 4e50b7b5…; gate 1 uses n = 21.
    * Band B is 7 of 21, exactly the D1 third.
    * **Unfilled:**
      * R2 towns: 1. The other-band and same-region fallbacks would have taken band A, and were refused 26 times by the
        third-band-B constraint.
      * R3 towns: 2.
    * **Composition:**

      | Region | Cities | Towns |
      |---|---|---|
      | R1 | 2 (the whole eligible list) | 4 (1 credited to R1 cities by the fallback) |
      | R2 | 3 | 2 |
      | R3 | 3 (all Chinese) | 1 (Chinese) |
      | R4 | 3 | 3 |

    * Countries: Italy 6 (towns, cities, villages), Mexico 3, Brazil 2, China 4, and one each from Monaco, Serbia,
      Albania, Guatemala, Turkey, Israel, Syria, Tunisia, Jordan, Lebanon, France and Colombia.
  * **train-H: 26 settlements (cap 30).** Unfilled: R3 cities 2, R3 towns 4.
  * **Selection on mapping quality, as the committee required (statistics).**
    * The OSM gate passed:
      * 4 of 41 Chinese candidates on dev (10 %), and 2 of 19 on train (11 %);
      * against 86 % for Italy, 75 % for Mexico, and 100 % for Brazil, Turkey and most single-candidate countries.
    * Chinese failures are mostly under 15 km of vehicular ways, not coverage. Built-up coverage by any way is usually
      ≥ 0.9.
    * R3 (East Asia) in M20 is therefore mostly Chinese city cores, and the claim must say so. The 15 km threshold
      was not changed.
  * **Balance** (terrain-eligible settlements, medians; "not accepted" includes those never gate-tested):

    | Split | Group | n | Slope | Band B share | Area km² | Population | Vehicular km |
    |---|---|---|---|---|---|---|---|
    | dev | accepted | 28 | 16.8 % | 0.29 | 3.5 | 38,127 | 35.4 |
    | dev | eligible, not accepted | 137 | 15.0 % | 0.13 | 3.0 | 8,615 | 7.5 |
    | train | accepted | 26 | 15.0 % | 0.15 | 7.0 | 41,790 | 54.7 |
    | train | eligible, not accepted | 53 | 15.6 % | 0.25 | 2.0 | 15,151 | 3.6 |

    * The accepted settlements are more populous (and on train larger), as a vehicular-length gate implies.
    * The dev band-B share is higher by design (the band rule).
  * **Anchors.** Wadi Musa passes the gate. Aguas Calientes is car-free (0 km of vehicular ways), so it is a
    pedestrian-only anchor.
  * **Two code fixes during the walk; outcomes unaffected.**
    * Downloads now time out after 120 s with retries (a read had hung for 4.5 h).
    * The skip log is now written after the last gate query. The replay from cache reproduced every earlier outcome
      and both accepted lists byte for byte.
* **F4b. Protocol deviation found and corrected: stale-mirror OSM in the walk (before any street metric).**
  * **The deviation.** The town builder refused a response because the fallback Overpass mirror (overpass.kumi.systems)
    reported a database older than the pinned snapshot: osm_base 2026-05-31 to 2026-07-28. A `[date:]` query to such
    a server returns its own older state, not the snapshot. The check found the same problem in the walk:
    * 31 of the 124 cached gate responses came from the stale mirror;
    * the walk never checked osm_base.
    * D1's "one OSM snapshot for all splits" was therefore violated for those 31 gate outcomes, F4 included.
  * **The correction, declared before re-running.**
    * Every response, in the walk and the builder, must carry osm_base ≥ 2026-11-27T00:00:00Z. Otherwise it is a
      failure and is retried.
    * The 31 stale responses are archived in `data/processed/m20/walk/osm_stale_mirror/`, not deleted.
    * dev, train and the anchors are re-gated from the current server. All other cached responses are reused.
    * The first frozen list is kept as `configs/m20_pool/dev_units_v1_superseded.json` and is superseded by a new
      freeze. Changes in membership are reported. No metric was computed on either list.
    * F4's tables are superseded by F4c.
* **D4d note: an output-identical speed-up (games' suggestion, "profile first"; ML: only speed-ups that leave the
  output identical).**
  * **The profile.** 4000 rounds of A0 on the scratch 4096 m city core took 790 s. Of that, 87 % was one Python loop
    in `GenState.resolve_step`: for every candidate step, each existing edge was tested for membership in the given
    bridge/tunnel set with Python min/max (780 M calls). The cost grows as O(edges) per candidate, so quadratically
    over a run. `junction_reuse._ground_dist` had the same loop.
  * **The fix.** `generate.raised_mask`, a numpy `isin` on (min, max) keys.
    * It gives exactly the same boolean as the loop: checked on 200 random cases.
    * A 2000-round A0 on the scratch core (seed 3) gave the same graph sha256 and the same per-particle node and edge
      counts, in 63.6 s against 147.9 s (2.3×; the gain grows with the network).
    * The stored M19 A0 (Petra, TNg_s0, seed 1) replays bit for bit.
    * 78 sampler, M14, junction, hybrid and dev-look tests pass.
  * `live_edges` (rebuilt each call; the next hot spot) is left as it is. Caching it would need invalidation across
    modules that edit `s.edges` in place.
* **F4c. Re-gate after F4b: membership unchanged.**
  * All 124 walk responses now carry osm_base ≥ the snapshot, and none is stale.
  * The re-gate from the current server gave the same accepted ids for dev-H, train-H and the anchors, and the same
    unfilled quotas: dev R2 town 1, R3 town 2; train R3 city 2, R3 town 4.
  * The new freeze of `configs/m20_pool/dev_units.json` has the same reference sha256 (4e50b7b5…) as the superseded
    one.
  * F4's tables stand. Only the re-queried towns' gate values (vehicular km, coverage) changed slightly. The archived
    records in `configs/m20_pool/walk/` are refreshed.
* **F5. D3 validation gate on dev-H, first and only run (recorded before any deliberation).** 29 units built (21
  reference, 8 descriptive); every OSM response is fresh for the pinned snapshot. Archived in `configs/m20_pool/gate/`
  (report, logs, sha256 of `hilltown.py` 36baff79… and `m20_gate.py` 9d815cc5…).
  * **Gate 1: PASS.** n = 21, pooled LOO coverage 0.917 (threshold 0.859), KS p (approx.) 1.000, no town flagged,
    2 floor-bound cells. Leave-one-block-out: coverage 0.914, KS p 1.000, none flagged.
  * **Gate 2 (≥ 90 % of towns): switchback PASS (1.00), contour-parallel PASS (1.00), fall-line grid FAIL (0.62),
    raster zigzag FAIL (0.29).**

    | Case | Suite composite > p95 | Targeted group | P2 > p95 | M7 > LOO max |
    |---|---|---|---|---|
    | Fall-line grid | 21 / 21 | M2: 13 / 21 | – | – |
    | Raster zigzag | 6 / 21 | M7: 21 / 21 | 17 / 21 | 21 / 21 |

  * leastcostpath (single site, n = 1, deviation): not caught on the suite composite or the lens group; caught on
    P2. Real-Petra control as in D3c.
  * Descriptive units: Wadi Musa composite 0.50 (|z| ≤ 2 on 0.99 of 78 defined cells); Aguas Calientes (car-free)
    4.96 on 6 defined cells; villages 0.94–1.71.
  * Under D3 ("nothing is changed to make a case pass; a failure is recorded as a limitation") these results stand.
    Whether a gate-2 limitation still permits the freeze goes to the committee (D3d).

## 2026-11-28

* **D3d. The gate-2 outcome and the freeze (all five members; after F5, before any generator is scored on dev-H).**
  * **The question.** D3's status line freezes the suite "only after it is implemented and passes the validation
    gate". D3a says "nothing is changed to make a case pass; a failure is recorded as a limitation". In F5, gate 1
    passed. Gate 2 failed for the fall-line grid (0.62) and the raster zigzag (0.29).
  * **What the committee read.**
    * `outputs/m20/gate/report.json`, and `outputs/m20/gate/{real,adv}/*.json`.
    * The builders in `hilltown.py` and `heightfield.Terrain.gradient`.
    * Per-cell z, recomputed in memory with the frozen code (`score`, `fit_reference`, `signed`). It reproduces every
      group mean in the report.
    * Nothing was rebuilt, regenerated or written.
  * **Findings (descriptive).**
    1. **Fall-line grid, the M2 group** (null p95 of the real LOO sign-oriented means: 0.50). "Part" is each family's
       sign-oriented contribution to the group mean.

       | Town | M2 cells | Contour part | Grade p90 part | > 15 % part | Mean | Margin | Land > 20 % |
       |---|---|---|---|---|---|---|---|
       | DUC_17083 | 11 | −0.77 | +0.11 | +0.09 | −0.57 | −1.07 | 0.17 |
       | UC_4315 | 14 | −0.76 | +0.17 | +0.45 | −0.15 | −0.65 | 0.30 |
       | DUC_19296 | 12 | −0.98 | +0.57 | +0.78 | 0.37 | −0.14 | 0.46 |
       | UC_2859 | 14 | −0.76 | +0.45 | +0.69 | 0.39 | −0.11 | 0.31 |
       | UC_1298 | 14 | −0.86 | +0.43 | +0.83 | 0.40 | −0.10 | 0.35 |
       | UC_4263 | 14 | −0.77 | +0.49 | +0.70 | 0.42 | −0.08 | 0.47 |
       | DUC_19289 | 14 | −0.75 | +0.58 | +0.60 | 0.42 | −0.08 | 0.46 |
       | DUC_21321 | 14 | −0.83 | +0.55 | +0.78 | 0.495 | −0.005 | 0.37 |
       | 13 caught | 6–14 | −0.77 to −1.30 | +0.58 to +1.49 | +0.76 to +1.36 | 0.52–1.55 | +0.02 to +1.05 | 0.30–0.96 |

       * **The contour-index cells count against the case in every town.**
         * The grid's contour index is 0.61–0.67 in every town and class. The reference medians run from 0.30
           (> 30 %) to 0.57 (5–10 %).
         * Its z is positive in all 79 contour cells (+0.95 to +4.97).
         * The metric is |cos(piece, fall line)| (M12 protocol §3; `hilltown.py` line 320). It rises towards 1 on the
           fall line, and real towns lower it on steep land.
         * D3c's table declares "contour index −" for this case, which is inverted relative to the metric's own
           definition. `TARGET_SIGN` implements it faithfully, so these cells subtract 0.75–1.30 from every town's
           group mean. (Operator check: M12 protocol lines 43–44, "1 = straight up the slope", and `TARGET_SIGN`
           `"contour_index": -1`; confirmed.)
       * **Geometry.**
         * An orthogonal grid's length-weighted |cos| lies in [0.5, 0.707] at any orientation. It is 0.5 when one
           axis is exactly on the local fall line, because the other half of its length runs along the contour.
         * The observed 0.61–0.67 is close to 2/π (0.637), the value for an orientation unrelated to the local fall
           line. On non-planar hill terrain, the unit-mean gradient frame (D3a choice 17) aligns the grid with the
           local fall line only partly.
         * Under the declared sign, the contour cells could therefore not count for the case on steep land, whatever
           the builder did.
       * **The grade and share cells carry the signal on steep land.**
         * In the 20–30 % and > 30 % cells of every town, z is +1.65 to +5.
         * On 0–5 % and 5–10 % land, z runs from −2.56 to +4.31: a street's grade cannot exceed its land's slope.
       * **The eight misses.**
         * Six miss by at most 0.14; DUC_21321 misses by 0.005.
         * Two miss clearly:
           * DUC_17083: 17 % of its land is above 20 %, and its 0–20 % grade cells sit near 0.
           * UC_4315: the grid's grade p90 on 0–5 % land is below the reference (z −2.56).
       * The suite composite catches the grid in 21/21 towns (smallest margin +0.95).
    2. **Raster zigzag.**
       * The M7 sign-oriented mean is +5 (capped) in all 21 towns, above the largest real LOO value by at least 2.63.
       * The composite is the binding check. M7's 5 cells are among 39–91 scored cells. The zigzag raises each
         town's composite over its own real LOO composite by −0.07 to +0.32 (median +0.26).
       * The six towns where it is caught are exactly the six with the highest real LOO composites (1.05–1.37).
         Every missed town is at 0.99 or below. The composite's verdict reflects the town's baseline more than the
         flaw.
       * P2 catches the zigzag in 17/21 towns; it misses UC_1298, UC_2859, UC_4263 and UC_4315.
    3. **The builders (code reading only).**
       * `Terrain.gradient` returns (east, north), consistent with the pixel centres of `_frame`. The grid's axis is
         therefore on the unit-mean fall line, as choice 17 states.
       * The zigzag's M7 is capped everywhere, so it built what it claims.
       * No builder bug was found. The grid's local alignment has not been measured (diagnostic D-1 below).
  * **Reviews.**
    * **Statistics (S).**
      * The suite is frozen under D3a.
        * D3a is the later and more specific rule. The status line read literally could only be met by changing the
          suite after F5, which D3 itself forbids.
        * The unmet status-line condition is a declared deviation.
      * The sign is a pre-registration error, and it stays in the record.
        * The two corrections computed in this deliberation give different answers: the flipped sign gives 20/21
          (null p95 1.32), and grade cells only give 19/21 (null p95 1.27, a bare 0.90).
        * That is a garden of forking paths. Neither is the gate, and no further variant is computed.
      * The zigzag result means the composite check has little power against single-metric flaws.
    * **Urban morphology (U).**
      * The sign is a drafting error against a definition in force since M12. "Grid running straight up the slope"
        has only one meaning under that definition.
      * The sign-corrected check is what D3 meant, and U asks that it be the primary reading, with F5 reported.
      * The grid is a weak "fall-line" adversary: half its length runs along the contour by construction.
      * Grade is bounded by the class, so gentle classes cannot carry the grid's signal. That is morphology, not a
        defect.
    * **Graphics (G).**
      * The zigzag composite failure is structural. An equal-weight mean over about 90 cells cannot register a
        flaw confined to 5 cells.
      * Supports the freeze.
      * Asks for one measurement of the grid's local alignment before any claim about "fall-line" layouts.
      * Code reading found no frame or sign bug in the builders.
    * **ML (M).**
      * Nothing in the suite changes.
      * M2 is already targeted (J was fitted to it), so the fall-line failure costs little in realism claims.
      * The zigzag failure does cost something: P2's 17/21 means non-inferiority on P2 does not exclude a
        tortuosity artefact in about 1 town in 5.
      * Proposes adding M7 to the Holm family.
    * **Games (T).**
      * Approves the freeze.
      * Zigzag-type artefacts matter for driving, so M7, G2 and G3 should be shown per class next to P2.
      * Proposes dropping the composite criterion from future gates.
  * **Sign-off round.**
    * S, G and T approved the rulings below.
    * U objected that the sign-corrected result should be primary. After ruling (c)3 (prominent disclosure in the
      limitation and the paper), U approved, with a dissent recorded on primacy.
    * M withdrew the Holm proposal after ruling (b)3.
  * **Rulings (chair).**
    * **(a) The freeze proceeds, unchanged, with recorded limitations.**
      1. **What is frozen.** The suite is frozen as run in F5: `hilltown.py` 36baff79…, `m20_gate.py` 9d815cc5…, and
         their import closure, hashed in `configs/m20_pool/gate/d3_freeze_manifest.txt` (16 files, plus the venv
         package versions). `TARGET_SIGN` is not edited. It is used only by the gate, which is never re-run.
      2. **Wording for the log status:** "D3 is frozen as run in F5. Its status-line condition ('passes the
         validation gate') is met for gate 1 and for two of the four multi-town gate-2 cases only. The suite is
         frozen under D3a's limitation rule; this is a declared deviation from the status line."
      3. **Reasons.**
         * The alternatives are to change the suite after seeing F5, or to drop it for a new suite designed after F5.
           Both are worse forms of the same post-hoc freedom.
         * Gate 1 (the real-town envelope) passed.
         * The failures are characterised in detail above.
    * **(b) Claims.**

      | Object | May support | May not support |
      |---|---|---|
      | M2 cells (contour index, grade p90, share > 15 %) | Hybrid − A0 contrasts as targeted secondaries, as declared. The contour index is read by its M12 definition (higher = nearer the fall line); D4's over-adaptation note (M2 z < −2) reads correctly under it | "M2 was validated against a fall-line adversary"; any claim that relies on the M2 group's signed mean |
      | Holm "M2 share above 15 %" | Stays in the family as pre-registered. Its cells moved in the declared direction on steep land in every town (descriptive) | Citing gate 2 as evidence of its sensitivity |
      | Suite composite (not an endpoint) | "Detected multi-metric departures": fall-line grid 21/21, switchback 21/21, contour-parallel 21/21 | Any reading as overall realism, or as excluding single-metric flaws (zigzag 6/21) or braids (leastcostpath not caught) |
      | P2 | Non-inferiority as declared, with D3c's braid limitation and the tortuosity limitation below | "P2 excludes local-geometry artefacts" |
      | M7 | Detects raster zigzags (21/21, smallest margin 2.63 over the real LOO maximum) | — |

      1. Every method's M7, G2 and G3 are reported per class with hybrid − A0 CIs in one table next to P2. M7 stays
         sealed with P2 until the hybrid is frozen (D4).
      2. The paper reports P2 component by component.
      3. The Holm family is unchanged. M's proposal is rejected: changing the family after a gate result changes the
         multiplicity of every other secondary. M7 is reported descriptively.
      4. T's proposal to drop the composite criterion is noted for later milestones only, not for M20.
    * **(c) Diagnostics.**
      1. **Permitted, declared now, before they run.** Each is reported whatever it shows, and none changes the suite,
         the gate record or the freeze:
         * **D-1.** Rebuild the grid on each town's field with the frozen builder (deterministic; no generator, no
           new data). Report per class:
           * the length share within 15° of the local fall line, and within 15° of the local contour;
           * the distribution of the contour index.
         * **D-2.** For the zigzag staircases: the major-leg length distribution (declared U(4, 9) m) and the
           native-to-convention length ratio.
         * **D-3.** For every case and town: the composite increase over the town's own real LOO composite, read from
           the archive.
      2. **If a builder bug is found** (the code does not do what D3, D3a or choice 17 describe):
         * F5 stays the result of record.
         * The bug and the fix are written here before the corrected builder runs.
         * The corrected case runs once, on all 21 towns, with the frozen gate code. It is reported beside F5 as
           "post hoc, after F5 was seen".
         * It cannot turn the gate into a pass or change the frozen metrics, and the freeze does not wait for it.
         * The paper reports both results.
      3. **The sign is a specification error, not a builder bug, and is handled the same way.**
         * Record: an erratum to D3c's table. The declared "contour index −" for the fall-line grid is inverted
           relative to the metric's definition.
         * Disclosed in the limitation text and the paper: the post-hoc 20/21 result (sign flipped) and the 19/21
           result (grade cells only), both computed in this deliberation. Neither is the gate.
         * Not permitted: any further variant, or choosing between the variants.
      4. **Chair's reason against U.** A pre-registration binds as written. Its errors are disclosed, not repaired.
  * **The limitations, drafted now:**
    > The metric suite was frozen after a pre-registered validation gate on 21 development towns. Real towns fit the
    > reference envelope (pooled leave-one-out coverage 0.917; threshold 0.859). Of four adversarial networks built
    > on every town, a single switchback road and unjoined contour streets failed the suite in all 21 towns, as
    > required. Two did not meet the pre-registered criterion (failure in at least 90 % of towns). A grid on the mean
    > fall line failed in 62 % of towns: the suite composite flagged it everywhere, but the targeted orientation and
    > grade check missed eight towns. A raster zigzag failed in 29 %: tortuosity flagged it in every town, but the
    > equal-weight suite composite, in which tortuosity is 5 of up to 91 cells, did not. A single-site least-cost
    > braid was detected only by the untargeted composite P2. Nothing was changed after these results. After the
    > gate we found that the pre-registered direction of the contour index for the fall-line case was inverted
    > relative to the metric's definition. With it corrected, a post-hoc, non-pre-registered re-analysis has the
    > targeted check fail the grid in 20 of 21 towns. P2 detected the zigzag in 17 of 21 towns, so non-inferiority on
    > P2 does not exclude artefacts confined to tortuosity. Tortuosity, hairpin radius and vertical curvature are
    > reported separately for every method.
  * **Status: D3 is frozen as run in F5.** Its status-line condition ('passes the validation gate') is met for gate 1
    and for two of the four multi-town gate-2 cases only. The suite is frozen under D3a's limitation rule; this is a
    declared deviation from the status line. Manifest: `configs/m20_pool/gate/d3_freeze_manifest.txt`. Diagnostics
    D-1 to D-3 run after the freeze and are logged as F6.

* **D4e. Compute for the dev-H look: the round cap, seeds and speed (all five members; before the look).**
  * **Facts (scratch run; `data/processed/m20/scratch_unpinned/devlook/generate.log`).**
    * UC_2859 (Perugia core), 4096 m domain, 64 windows, cap 24,000 rounds, code after d2652a4:

      | Seed | Wall | CPU | Rounds | Windows started | Forward passes |
      |---|---|---|---|---|---|
      | 1 | 15,648 s | 14,839 s | 24,000 (cap) | 32 / 64 | 138,201 |
      | 2 | 15,761 s | 14,952 s | 24,000 (cap) | 35 / 64 | 138,527 |

    * Time per 1000 rounds grows from 19 s to about 1,600 s. Total time grows as about rounds^2.4: ×25.9 from 6,000
      to 24,000 rounds.
    * Peak memory is 0.94 GB per worker.
    * **Every reference unit is on the 4096 m domain.** Each site config's OSM box is about 5.3 km, and only villages
      are on 2048 or 8192 m. So all 21 units have 64 windows and the 24,000-round cap, and capped runs should be
      expected as the norm.
    * **D4d's compute line is superseded.** It assumed about 12 min per run and about 190 CPU-h for hold-H; it is low
      by about 22×.
    * **The cited profile (780 M min/max calls) predates d2652a4.**
      * There is no profile of the current code.
      * Code reading shows costs that grow with the number of edges for every candidate step:
        * `live_edges` rebuilds a Python list of every edge on each call;
        * the crossing and T-junction tests scan every edge;
        * `split_edge` loops over the edges in Python.
      * There is no spatial index.
    * **Disclosure (S).**
      * The scratch town is a dev-H reference unit, and seeds 1–2 are the look's seeds 1–2.
      * Its A0 process logs were seen before the look, and so were its km (domain 538 / 534 km, limits 350 / 298 km).
      * A0 is the frozen J, and nothing of the hybrid or of P2 was seen.
      * The km values are an M1-type outcome. No ruling here or in D4b uses them; the rulings below rest on time,
        rounds and windows only.
    * `scripts/m20_devlook.py` and its test have uncommitted changes. The look runs from a commit whose hash is
      logged.
  * **Budget** (planning values; one laptop, 6 workers, no paid compute).
    * Per run: A0 is 4.36 h (Perugia). The hybrid and measure stages are not yet measured; they are budgeted at
      +25 %. That makes c = 5.45 CPU-h per run.
    * The 6-worker load (5 performance and 10 efficiency cores) is assumed to slow each run by κ = 1.3, giving
      7.09 worker-h per run.
    * S is the further output-identical speed-up.
    * The window from launch (1 Dec 00:00) to 6 Dec 12:00 is 132 h, or 792 worker-h.

    | R | Look runs (21 × R) | CPU-h (S = 1) | Wall, S = 1 | S = 2 | S = 4 | Fits 132 h at |
    |---|---|---|---|---|---|---|
    | 24 | 504 | 2,747 | 596 h (24.8 d) | 298 h | 149 h | S ≥ 4.5 |
    | 16 | 336 | 1,831 | 397 h (16.5 d) | 199 h | 99 h | S ≥ 3.0 |
    | 12 | 252 | 1,373 | 298 h (12.4 d) | 149 h | 74 h | S ≥ 2.3 |
    | 8 | 168 | 916 | 199 h (8.3 d) | 99 h | 50 h | S ≥ 1.5 |
    | 6 | 126 | 687 | 149 h (6.2 d) | 74 h | 37 h | S ≥ 1.13 |
    | 4 | 84 | 458 | 99 h (4.1 d) | 50 h | 25 h | S ≥ 1 |

    * Hold-H, A0 + H1/H2 only (A5b, H3, the externals and the sensitivity arms are extra, budgeted in D4b):

      | N × R | Runs | CPU-h (S = 1) | Wall, S = 1 | Wall, S = 4 |
      |---|---|---|---|---|
      | 40 × 24 | 960 | 5,232 | 47 d | 11.8 d |
      | 40 × 8 | 320 | 1,744 | 15.8 d | 3.9 d |
      | 40 × 4 | 160 | 872 | 7.9 d | 2.0 d |
      | 60 × 8 | 480 | 2,616 | 23.6 d | 5.9 d |

  * **Reviews.**
    * **Games (T).**
      * The quadratic growth comes from scanning every edge for each candidate step. A uniform spatial hash for edges
        and nodes, plus an incrementally maintained live-edge array, should give a large gain late in a run. Profile
        from a late-round state first.
      * As a fallback, T proposes lowering the 64-window cap to 12,000 rounds, which costs 2,309 s against 15,700 s
        in the log, so that R = 24 is kept.
    * **Graphics (G).** Bit-identity hazards:
      * argmin tie order (a candidate subset must keep the global index order);
      * element-wise float results (the subset arithmetic must be the same IEEE operations);
      * edge insertion order;
      * `split_edge`.
      * G wants a differential test of `resolve_step` itself, not just whole-run hashes.
      * The 2000-round check of d2652a4 never reached a late-run state or the node-array guard.
      * A smaller domain gives nothing for city cores (the 3,496 m core needs about 49–64 windows) and changes towns'
        builds. Rejected.
    * **ML (M).**
      * Only output-identical changes.
      * The cap formula, the domain and the window order stay as they are. A unit-first window order, or a lower cap
        for 64-window domains, is a new generator.
      * The verification must include the capped Perugia runs and the M19 Petra replay.
      * The look and hold-H must run the same generator.
    * **Statistics (S).**
      * D4d already says what to do: "if compute binds, R is cut uniformly before N".
      * R is fixed before launch, from scratch timing only. Run time grows with network size, an outcome, so an
        adaptive R would select on it.
      * Hold-H's R equals the look's R, so the pilot estimates the variance hold-H will actually have.
      * Seeds run seed-major, and outcomes are not read until the look completes.
      * S disagrees with T: a lower cap is a generator change, chosen after A0's behaviour on a dev-H town was seen.
    * **Urban morphology (U).**
      * Under the cap, unit land that is never reached has no streets in either arm. Its M6 pairs fail as "not
        joined" in both, so P1a is attenuated towards 0. H1 cannot create fabric where A0 has none, so the contrast
        measures adaptation of an existing fabric, not coverage.
      * Asks for logs of where the unstarted windows fall, and for a pre-declared sensitivity on the pairs A0 reaches.
      * Against T's lower cap: it deepens the truncation.
  * **Sign-off round.**
    * S, M, G and U approved.
    * T approved the speed-up and seed rulings, with a dissent recorded on the 12,000-round fallback (T: "R = 24 at a
      lower cap beats R = 4 at the declared cap"), which the chair rejects in ruling 1.
  * **Rulings (chair).**
    1. **The generator is unchanged.**
       * D4d ruling 1 stands: the cap formula, the 4096 m domain and the window order.
       * Rejected:
         * the lower cap: a generator change specific to 64-window domains, which deepens truncation where M6 is
           measured, when the pre-registered remedy is to cut R;
         * the smaller domain: no gain for cores, and the units would change;
         * unit-first windows: not output-identical.
    2. **Output-identical speed-up: approved, time-boxed to 30 Nov 24:00, all or nothing.** Accepted only if:
       * a. a fresh profile is run from a late-round state;
       * b. a differential test of the new `resolve_step` against the committed d2652a4 version agrees on at least
         10⁵ random candidate steps, drawn from the stored final Perugia states and from states at several rounds.
         The kind, the integers and the bytes of every float array must be equal;
       * c. full replays give identical graph sha256, per-particle node and edge counts, forward passes, rounds used
         and windows started, for:
         * the stored Perugia seeds 1 and 2 (24,000 rounds, capped);
         * the stored M19 Petra A0 (TNg_s0, seed 1);
       * d. every sampler, M14, junction, hybrid and dev-look test passes;
       * e. the change is committed and its hash logged.
       * Any difference discards the whole change.
    3. **R is fixed once, before launch.**
       * R = max{r ∈ {24, 16, 12, 8, 6, 4} : 21 · r · c_m ≤ 6 · W}, where:
         * c_m is the worker-hours per run (A0 + hybrid + measure) measured on the Perugia scratch under the final
           code, while all 6 workers are loaded (the verification replays);
         * W is the hours from launch to 6 Dec 12:00.
       * The minimum is R = 4 (σ²_w then has 63 df).
       * Hold-H uses the same R. It may use more runs, with N unchanged (conservative), never fewer.
    4. **The N formula (D4d ruling 7), with R made explicit.**
       * v(R) = σ²_b + σ²_w / R.
       * v̂ is the variance of the 21 settlement means of the m6_15 difference, each over R runs.
       * v⁺ = 20 v̂ / χ²₀.₂₀,₂₀ = 1.372 v̂.
       * N = (1.96 + 1.2816)² v⁺ / 0.008² = 164,190 v⁺.
       * Block means are the sensitivity; the larger N is used, with N ≥ 40.
       * Reported, not used: σ̂²_b and σ̂²_w separately, and the N that R = 24 would have needed. A smaller R raises N;
         that cost falls on hold-H and is stated.
    5. **Run order and stopping.**
       * Seed-major: seed 1 for all 21 units, then seed 2, and so on.
       * Only process lines (round, seconds) are watched during the look. No outcome field (km, metrics) is read until
         every run is done; the operator filters the log.
       * If time runs out, the look uses complete seed blocks: R′ = the largest r for which all 21 units are done.
       * If R′ < 4, there is no look by 8 Dec. The TNg-H decision is then postponed, or taken explicitly without the
         look, and recorded.
       * The settlement list is never cut, and R never adapts to timing.
       * Descriptive units are generated only after the reference blocks are done, at the same R, if time remains.
         RC_175819 (8192 m, 256 windows) is excluded from the look on compute grounds, which is stated.
    6. **Capped runs: "flagged, never dropped" stands.**
       * Dropping capped runs would drop nearly all of them, and would select on network size.
       * The flag alone is now uninformative. Each run logs:
         * the ids of the windows started;
         * the share of unit land in windows started;
         * the share of the unit's M6 pairs with an endpoint more than 60 m from the A0 network.
       * This logging must leave the outputs unchanged.
       * **Estimand.** "A0 (J, δ = 0) under the D4d round cap; on 64-window domains the cap binds before every window
         is started." P1a is read as the hybrid's effect on capped A0 fabric.
       * **Declared sensitivity.** M6 is computed on the pairs whose two endpoints both join the A0 network within
         60 m. This conditions on the pre-treatment state shared by both arms. It is descriptive in the look, and a
         declared sensitivity of P1a on hold-H, outside the intersection–union test and Holm.
       * The cap is not revised after the look (D4d).
    7. **Hold-H budget.** The table above replaces D4d's compute line. D4b and D6 carry it, with the extra arms.
  * **Status.** The look launches only after:
    * the D3 freeze commit (done with this entry);
    * ruling 2 resolved (accepted, or discarded);
    * c_m measured, and R and its formula inputs logged here.
* **D4e ruling 2 resolved: the output-identical speed-up is accepted (games' change; the commit carrying this entry).**
  * **a. Profile from late rounds (code of d2652a4, cProfile, Perugia UC_2859, no pickled resume: the run itself was
    profiled once it got there).**
    * Seed 1, rounds 12,000–12,300: 137 s (0.46 s per round).
      * `resolve_step` 74 %. Most of it is `live_edges`: a Python list of every edge, then `np.array` of it (55 % of the
        run), on every candidate step.
      * `junction_reuse.reuse` 14 % (`nearest_junction` loops over every node in Python; `_ground_dist` builds the
        whole network for a 15–22 m search).
      * `GenState.copy` 10 % (2.3 copies a round; most of a copy was garbage-collector passes).
    * Seed 2, rounds 10,500–10,600: 35 s; `resolve_step` 78 %, `live_edges` 56 %.
  * **The change** (`graphs/generate.py`, `graphs/junction_reuse.py`; no other file).
    * Each GenState keeps a spatial index beside `edges` and `nodes` (never instead of them):
      * an edge array updated in `connect` and `split_edge`, and per-edge and per-node boxes;
      * a static part sorted on a Morton curve in blocks of 32, plus a tail of up to 4096 new entries;
      * it is copied by `copy`, left out of pickles (rebuilt on first use) and rebuilt whenever its counts disagree
        with `edges` or `n`.
    * `resolve_step` runs its three scans on the candidates the index returns, in global edge (node) order, with the
      same formulas, dtypes and operation order:
      * crossings: the step's box + 64 m;
      * the merge node and the T-junction: q's box of half side MERGE_M + 1 m.
      * A step plus the longest edge over 400 m scans every live edge, as before.
    * `live_edges` reads the edge array. `split_edge` finds the first live edge {a, b} through the index.
    * `nearest_junction` tests the nodes in x's box of half side r + 1 m. `_ground_dist` with a finite limit builds
      the graph from the edges in src's box of half side limit + 1 m.
    * `copy` pauses the garbage collector while it builds the per-edge lists and per-node sets. The bridge/tunnel keys
      of `raised_mask` are cached per state.
  * **Why the output cannot change.**
    * Every filter is conservative: whatever the full scan could select lies in the box.
      * Merge, T-junction and nearest junction: the selected item is closer than the threshold.
      * Paths of at most `limit`: they stay within `limit` of src.
      * A computed crossing: it needs |den| > 1e-9, so rounding moves the crossing point by under 10 m on steps and
        edges of at most 400 m in total, against the 64 m margin.
    * The survivors keep the global order, so `argmin` ties fall on the same edge or node.
    * The float operations on each survivor are element-wise and unchanged.
    * `edges` is built and edited exactly as before: same entries, types and order.
  * **b. Differential test** (the new functions against the d2652a4 code, imported from `git show d2652a4`).
    * 147,200 random candidate steps, all equal in kind, integers and the bytes of every float array. Of these:
      * 108,800 on 34 Perugia states: the 4 particles at rounds 3,000, 6,000, 9,000 and 12,000 (seed 1) and 1,500,
        4,500, 7,500 and 10,500 (seed 2), and both stored final states;
      * 38,400 on 12 synthetic states with bridges and tunnels, edges over 400 m and a full node array (the M16 guard).
    * The steps are of every kind: 65,366 crossing endpoints, 18,118 splits, 17,240 merges, 5,055 T-junctions,
      17,938 new nodes, 23,483 dropped, 8,912 grade-separated and 1,459 clipped.
    * Also equal:
      * 13,170 commit sequences, compared on the full state (edges with their types, adjacency, maps, queue, rules);
      * 630 direct `split_edge` calls;
      * 12,337 `reuse` calls, and 37,011 calls each of `nearest_junction` and `_ground_dist`;
      * `live_edges`, `graph`, `copy` and the pickle bytes.
  * **c. Full replays** (final code; outputs written to scratch only).
    * Perugia seeds 1 and 2, 24,000-round cap, against the stored records:
      * all identical: graph sha256 7a97beef… / 1352f50a…, the graph itself, the chosen particle (3 / 0), forward
        passes 138,201 / 138,527, stats, rounds used 24,000 (cap hit), windows started 32 / 35, nodes, edges, parts
        fingerprint;
      * the bytes of the stored state pickle are identical too.
    * Old code (d2652a4) replays of both seeds, against the new runs: all 4 particles' node and edge counts and
      state signature agree at all 16 checkpoints and at the end. Their records equal the new ones. The old replays took
      16,004 / 16,133 s under the same load.
    * M19 Petra A0 (TNg_s0, seed 1; `m19_arms.job`, as for d2652a4): graphs, forward passes (5,614), stats, mech,
      src_edges and the A5 snapshots and log are all identical.
  * **d.** All 467 tests pass (in the main tree, after the change was applied), among them the sampler, M14, junction, hybrid and dev-look tests.
  * **Speed** (Perugia, seeds 1 / 2, against the stored d2652a4 runs).
    * Whole run: 1,600 / 1,656 s against 15,648 / 15,761 s stored, so 9.8× / 9.5×. Against the old-code replays
      under the same 6-worker load (16,004 / 16,133 s) it is 10.0× / 9.7×.
    * Rounds 23,000–24,000: about 160 s against about 1,620 s, so about 10×.
    * Rounds 0–1,000: about 1.3×.
    * Time still grows with the network, now mostly in `copy` (9 copies a round at 21,000 rounds: the SMC
      resampling and the M14 closing look-ahead) and in the garbage collector. A 4096 m run remains roughly
      quadratic, at a tenth of the cost.
    * Petra A0: 24 s against 37 s.
    * Peak memory per worker is 1.52 GB against 1.29 GB for the old code in the same harness (+18 %).
  * **c_m, A0 part** (ruling 3; final code, 6 workers loaded: the two runs, the two old-code replays and the two hybrid
    jobs of the scratch look).
    * 0.444 and 0.460 worker-h per run (wall = CPU), so about 0.45 worker-h.
    * The hybrid and measure parts are still to be measured under this code. The hybrid's step evaluation goes
      through the same `resolve_step`.
* **D1-impl. Implementation note: the training data and recipes of TNg-32, TNg-H and TNg-R (written before any
  training run; no training result exists yet).** Code: `scripts/m20_tng_data.py` (data), `scripts/m20_train_tng.py`
  with `configs/m20_tng.toml` (training), `scripts/m20_flat.py` (the TNg-R walk). Data in `data/processed/m20/tng/`
  (git-ignored); tables in `outputs/m20/tng/`.
  * **Input recipe, the same for all three backbones.**
    * FABDEM V1-2 is averaged (as the walk and the town build do) onto a 32 m grid covering the tile and its 256 m
      margin (32 × 32 cells). That surface is interpolated bilinearly onto the unchanged 4 m model grid.
    * So the model's own slope channel (unchanged `terrain_sequence.tile_layers`) is the slope of the 32 m DEM, and
      `m12_train.py` is not edited. This is how "32 m slope channel" is implemented; the elevation channel comes from
      the same 32 m surface.
    * One normalisation for all three, from the M12 recipe on the FABDEM training tiles (473 with a full DEM):
      elev_scale 5.494, slope_scale 0.0643 (GLO-30: 6.920 / 0.2009).
    * Water and structure labels of the existing tiles are the M12 arrays.
  * **"The existing 525 training tiles"** is read as the M12 development tiles with a DEM (train 473, val 12,
    val_t 40: the M18 GLO-30 build). Their elevation is now FABDEM (all 525 fully valid).
    * The 212 Japanese training tiles keep their M12 arrays (no DEM), exactly as in TNg. Training lists are therefore
      685 tiles for TNg-32, like TNg.
    * Validation and model selection are unchanged (M12 val + val_t NLL, 59 tiles). dev-H is never used for
      selection.
  * **Training recipe.** `m12_train.py` is run unchanged (as M18 did) with the TN configuration (checked equal to
    `configs/m6_roads.toml`). The wrapper replaces three things:
    * the data loader;
    * `make_samples`, verbatim plus one key `w` (checked identical otherwise on 12 tile × symmetry cases);
    * `pop_nll` multiplied by `w` when present.
  * **Loss weights (D1).**
    * Each training decision gets the weight of the D3 slope class (0–5 / 5–10 / 10–20 / 20–30 / > 30 %, FABDEM 32 m,
      the cell under the current node).
    * w_c = f_c^−½ / Σ f_c^½, where f_c is the share of the backbone's own training nodes with a DEM in class c. The
      mean weight over those decisions is 1.
    * Decisions without a DEM (Japan) get weight 1. Validation NLL is unweighted.
    * TNg-32: shares 0.728 / 0.151 / 0.085 / 0.024 / 0.012, weights 0.65 / 1.43 / 1.91 / 3.61 / 5.02.
    * TNg-H: shares 0.455 / 0.156 / 0.213 / 0.110 / 0.067, weights 0.70 / 1.20 / 1.02 / 1.42 / 1.83.
    * Tile repeats: none. Every tile and window appears once per epoch, so the cap is 1.
  * **TNg-H windows (built).**
    * 26 train-H settlements; the 514 kept windows of the town build (≥ 70 % inside the polygon or, for cities, the
      D4d core; FABDEM and GLO-30 valid).
    * 20 windows without a vehicular edge are dropped (D1), leaving **494**.
    * Window networks are the town build's `graph` (the training build). Structure labels follow the M12 rule, with
      the water crossing tested on the 4 m water raster rather than the polygons (approximation).
  * **Street-km per slope class** (FABDEM 32 m class under the edge midpoint):

    | Set | Tiles / windows | 0–5 | 5–10 | 10–20 | 20–30 | > 30 | no DEM | Total |
    |---|---|---|---|---|---|---|---|---|
    | Existing training tiles (TNg-32) | 685 | 1,303.4 | 266.5 | 149.0 | 40.8 | 21.2 | 1,235.6 | 3,016.5 |
    | Added train-H windows | 494 | 169.4 | 229.2 | 515.5 | 298.8 | 185.3 | 0 | 1,398.2 |

  * **Region cap (≤ 40 % of steep street-km, steep = ≥ 20 %).**
    * Existing tiles are counted as R1 (Spain), R5 (US) and R3 (Japan, no DEM).
    * TNg-H: 546.1 km steep. Shares: R2 0.347, R1 0.303, R4 0.216, R5 0.096, R3 0.038. The cap holds without any
      intervention.
    * TNg-32 alone is 85 % R5 (Los Angeles), as the D1 framing says. It is the unchanged control, so the cap does not
      apply to it.
  * **Learning curve.**
    * Settlements are ordered by sha256("m20-D1-learning-curve" + id). The first ⌈p · 26⌉ are taken.
    * 25 %: 7 settlements, 155 windows, 428.3 km.
    * 50 %: 13 settlements, 246 windows, 702.1 km.
    * 100 %: TNg-H seed 0 itself.
    * Seed 0 for each.
  * **TNg-R selection rule (fixed here, before the flat walk and before any training result).**
    * **Pool:** the train-H list only (`lists/train.csv.gz`, R1–R4), taken in list order (rank) within each
      region × size stratum.
    * **Flatness:** a settlement qualifies if its built-up-weighted median slope (`m20_walk.terrain_one`, FABDEM 32 m)
      is under 6 %.
      * Terrain values already in `train_terrain.csv` are reused.
      * New evaluations use the same function and go to `train_flat_terrain.csv`.
    * **Gate:** the unchanged completeness gate (`m20_walk.osm_gate`) at the pinned snapshot, with the osm_base
      freshness check. A gate failure is final.
    * **Block cap:** at most 2 settlements per 1° block in the train split, counting the train-H settlements.
    * **Build:** the unchanged town build (`m20_towns` osm / site / real, split train), then the same window rules as
      TNg-H (windows without a vehicular edge dropped).
    * **Strata and targets:** the TNg-H strata, each with its TNg-H added street-km as the target:
      * R1 city 236.4, R1 town 234.5;
      * R2 city 412.9, R2 town 87.3;
      * R3 city 62.5;
      * R4 city 184.2, R4 town 180.4;
      * R3 town 0.
    * **Matching:**
      * Strata are visited round robin (sorted). Each takes its next qualifying candidate until its added km reaches
        the target.
      * The last settlement of a stratum contributes only the windows needed, in the order
        sha256("m20-D1-TNgR" + window id), so each stratum overshoots by less than one window.
      * A stratum that runs out of candidates passes its remaining km to the other size of the same region. Failing
        that, the shortfall is reported.
    * The counts actually obtained are appended to this note as built, before any training result is read.
* **F4d. A third Overpass instance for the TNg-R flat walk (implementation, 28 Nov).** overpass-api.de answered
  town-sized queries with HTTP 502 for hours and kumi.systems is stale (osm_base 2026-06-01, rejected by F4b).
  `maps.mail.ru/osm/tools/overpass` is current (osm_base 2026-11-28) and serves attic data: the same
  `[date:"2026-11-27T00:00:00Z"]` highway query around Perugia returned identical way ids (208) from it and from
  overpass-api.de. `scripts/m20_flat.py` inserts it second in the in-process server list (frozen `m20_walk.py`
  unchanged). The F4b freshness check still applies to every response; the walk resumed from its cache (no
  response re-queried, no selection changed). Responses record their endpoint.
