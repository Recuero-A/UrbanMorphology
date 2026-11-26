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
