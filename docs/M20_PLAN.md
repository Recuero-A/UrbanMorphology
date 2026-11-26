# M20 plan: terrain-adaptive, realistic hill towns (committee decision, 2026-11-27)

**Owner's goal:** cityscapes that adapt to terrain and look real, for games and similar uses. The owner delegated the
decisions to the team (M19 Q2).

Five reviewers wrote independent reports and then answered each other in a second round:
* procedural graphics;
* ML generative modelling;
* urban morphology and GIScience;
* methodology and statistics;
* games technical direction.

This file records what they agreed. Where they disagreed, it records the chair's ruling. Findings behind it: M19 F5–F6.

## 1. Diagnosis (unanimous)

* **The learned model's terrain gap is mainly a data problem.**
  * The six training cities with a DEM are BCN, MAD, VLC, LAX, NYC and CHI. Their steep streets come almost entirely
    from Los Angeles.
  * GLO-30 is a surface model, so buildings make flat cities look steep and teach the model that slope does not set
    street direction.
* **What the inference potentials can and cannot do.** They reweight 8 learned proposals; they cannot create gentle
  routes. Raising terrain penalties makes STOP cheapest, so the network adapts by deleting streets (M18 D12: Machu
  Picchu TAS 40.9 → 74.7, connectivity 0.76 → 0.27).
* **Inference machinery is exhausted.** Online routing that keeps the target cannot help (M19 F5); look-ahead hurts.
  Stop adding inference machinery.
* **Visible failures:**
  * streets straight up the slope (33–56 % of length steeper than 15 %);
  * whole towns about 3× too dense on flat land;
  * towns that fall apart at Machu Picchu;
  * no stairs: footways and steps were excluded from training (`src/urbanmorph/config.py:110-116`).
* **The current metrics do not measure "a realistic town on terrain"** (M19 F6). The external baselines win on them
  without looking like towns.

## 2. Decision

* **Method: in sequence, a hybrid first, then a retrained model in parallel.**
  1. **Primary: a post-hoc grade hybrid on the frozen TNg model** (M18 J, then A5 connectors). It is paired on the
     generator's own sample path, so variance is low and confirmation is cheap.
     * **Two street classes**, as real hill towns have:
       * drivable streets: long runs steeper than 15 % (above about 60 m; the exact rule is set on the development
         towns) are rerouted, arterials into switchbacks;
       * pedestrian links: short steep links become stairs or ramps when a drivable route exists within a declared
         detour bound.
     * **Whole-town density calibration** toward real per-slope-class density.
     * **Not a hard grade cap inside the sampler.** That is the same mechanism as STOP and deletes streets.
  2. **Secondary: TNg-H, a retrain on hilly cities.**
     * Training towns come from the train-H split (§3).
     * The slope channel is computed at ≥ 32 m, which limits the surface-model error. Sampling is stratified by
       slope.
     * Steps and footways become a separate street class if the data allow.
     * About 9 MPS-hours of training, run in the background.
     * It is frozen or dropped by **8 Dec**. It enters the confirmation as a secondary arm only if frozen in time.
  3. **The strongest baseline, not our method:** an Emilien 2012 / Galin 2011-style procedural generator.
     * Interest-based seeding, the M19 router with a reuse discount (merged paths), loop closure.
     * A rule-only fabric ablation: procedural blocks through the same grade layer.
     * **Pre-declared rule:** if the procedural hybrid beats ours in the human study at matched density, the paper
       reports that, and the hybrid procedural method becomes the method paper.
* **Venue: two papers. No Eurographics short paper on 22 Jan.** There is no visual win yet, and graphics reviewers
  would set it beside Galin 2010 and Emilien 2012.
  * **Paper 1 (GIScience, 29 Jan 2027; deadline as recorded in M18, re-check the call):** measurement, benchmark and
    analysis:
    * real slope-conditional street curves from a hill-town corpus;
    * the validated metric suite;
    * every generator scored fairly at matched density;
    * the analysis findings: adaptation by deletion, target-preserving routing cannot help, repair after growth.
    * Development results are labelled exploratory.
  * **Paper 2 (method; IEEE CoG 2027 or Pacific Graphics 2027, both dates unverified):** the hybrid, the perceptual
    study, a game-readiness table and the fresh-geography confirmation.
* **Confirmation: after the method change, once.** Fresh geography is not spent on A5 alone.
  * Arms: A0, A5, terrain-blind repair at matched added km, the hybrid, and TNg-H if frozen.
  * The tuned externals run on the same blocks.
  * The sealed M12 test split and the declared M15 held-out metros stay untouched.

## 3. Data: one seeded pool of hilly cities, split by rule before any download

* **Sampling frame:** urban centres from a public global list (GHSL urban centres, or an equivalent declared list).
* **Eligibility uses terrain only:** median built-up slope on GLO-30 ≥ 12 %.
* **Exclusions:** within 50 km of any training, development, used or declared held-out metro.
* **Stratification:** at least 4 regions × 2 slope bands.
* **Split by whole town**, not by tile, because tiles of one town are spatially dependent (Roberts et al. 2017):
  * **train-H:** 8–10 towns, for TNg-H;
  * **dev-H:** at least 19 towns. These form the reference corpus and serve for envelope validation and tuning every
    method. Fewer than 19 needs parametric envelopes: min–max coverage of a new town is (n − 1)/(n + 1).
  * **hold-H:** sealed, for the confirmation. Size from the power analysis (§5).
* **Before any street is looked at,** commit:
  * the list hash;
  * the RNG seed, taken from the commit;
  * an OSM completeness gate.
* **Data per town:**
  * OSM vehicular ways, with steps, footways and paths as a separate layer;
  * GLO-30;
  * PNOA LiDAR for the Spanish towns, to measure the GLO-30 bias.
* **Named candidates, for checking the frame only, not a selection:** Cuenca, Toledo, Ronda, Granada Albaicín,
  Liguria / Umbria / Amalfi towns, Lisbon, Porto, Funchal, Busan, Valparaíso, Guanajuato, Cusco, La Paz, Wadi Musa,
  Aguas Calientes, Ollantaytambo. The rule decides membership, and a listed town may fail it.

## 4. Metric suite (frozen before any generator is scored)

* **Stratification:** by slope class (0–5 / 5–10 / 10–20 / 20–30 / > 30 %), measured on both the native geometry
  and the 30 m resampled geometry.
* **Scoring:** each metric is scored as a percentile within the real dev-H distribution, bootstrapped over towns.
  Never higher-is-better, and never divided by a few-city SD.
* **The suite:**
  * **M1 Siting:** density and coverage per class.
  * **M2 Orientation and grade:** W1 distance of the contour index; grade p90; share of length above 15 %.
  * **M3 Steep structure:** lengths of runs above 15 %; hairpin rate (turns over 120° with legs of at least 30 m,
    per km of street on land steeper than 15 %).
  * **M4 Topology:** dead ends per km; 3-way and 4-way share; close15 after consolidation at τ 10 m; bundle share
    (length on faces narrower than 8 m).
  * **M5 Blocks:** block area; Louf Φ; alignment of the block's long axis with the contours.
  * **M6 Grade-limited access:** a pair counts as served only if at most 10 % of the route is steeper than 15 %.
  * **C2ST per slope class:** a classifier two-sample test (Lopez-Paz & Oquab 2017).
* **Validation gate, passed before freezing:**
  * every real dev-H town lies inside the envelope, checked leave-one-town-out;
  * each adversarial case fails at least one metric: a single switchback road, a leastcostpath bundle, a grid
    running straight up the slope;
  * the metrics agree with human judgment in the perceptual pilot.
* **Primary endpoints:** one per claim, every other metric with Holm correction.
  * **Method paper:** M6 grade-limited steep-land access (superiority).
  * **Realism (non-inferiority):** the composite of M2 share above 15 % on 10–30 % land, plus M4 bundle share, scored
    against the real envelope.
  * The margins are set in D6 from dev-H.
* **TAS and typicality:** descriptive only.

## 5. Statistics and fairness (frozen in D6, committed by ~10 Dec, before hold-H opens)

* **Hashes:** code commit, checkpoints, the J config, pooled δ, router and hybrid parameters, metric code, thread
  counts, library versions.
* **Estimator:** unweighted block means, at most 6 blocks per city, sign-flip or wild-cluster inference, B = 10,000,
  fixed seed. A city-cluster bootstrap is the sensitivity check.
* **Power:**
  * paired post-hoc arms: δ 0.008, 12 runs per block, 90 % power → about 75 blocks;
  * unpaired arms (a retrained model): at least 24 runs per block (20× the run variance);
  * the variance of the grade-limited endpoint is unknown. It comes from a blinded internal pilot on dev-H with
    the frozen method.
* **Fair externals:**
  * the same logged tuning budget for every method (for example 20 development configurations);
  * density sweeps and Pareto fronts;
  * the same evaluation points, blocks and graph convention (30 m resample, 8 m merge);
  * both tile and whole-town mode.
* **Stopping rules:**
  * one look;
  * no cities added;
  * a crash is re-run with the same seed; a failed run counts as empty;
  * results are reported whatever they show;
  * no hold-H map is viewed before the analysis is locked.
* **Graveyard table:** every variant tried across M12–M20, reported in the paper's supplement.

## 6. Human evaluation (method paper)

* **Format:** a preregistered 2AFC study (OSF), built with jsPsych on GitHub Pages; free.
* **Stimuli:**
  * at least 24 pairs (8 dev-H towns × 3 seeds), rendered with identical buildings, camera and light;
  * aerial stills plus 10 s street-level clips;
  * real towns on the same terrain as anchors.
* **Arms:** real, frozen learned, hybrid, tuned externals and the Emilien-style baseline.
* **Raters:** about 40 players plus 10–15 developers, 48 trials each; volunteers only (no paid services).
* **Analysis:** a logistic mixed model with crossed random effects for raters and towns (Westfall, Kenny & Judd
  2014). "As real as real" is tested as equivalence with TOST (Lakens 2017), for example P(real chosen) ≤ 0.60.
* **Owner action:** ask KAIST about IRB review or exemption this week. If it comes late, run an expert-only pilot
  (5–8 developers).
* **Game-readiness:**
  * drivable reachability under the grade limit;
  * navmesh bake;
  * seconds per town;
  * determinism across runs;
  * triangle and frame budgets.

## 7. Timeline (today 2026-11-27)

| When | Work |
|---|---|
| 27 Nov – 3 Dec | Commit the pool rule and the split; download dev-H and train-H; code M1–M6; run the terrain-blind repair control on M19 development data; start TNg-H data |
| 4 – 10 Dec | Validate and freeze the metrics; hybrid and density calibration on dev-H; tune the externals and the Emilien-style baseline; train TNg-H (freeze or drop by 8 Dec); commit D6 by 10 Dec |
| 11 – 24 Dec | Open hold-H once and run the confirmation; build the perceptual study; renders |
| Jan 2027 | Write paper 1 (GIScience, 29 Jan); run the perceptual study |
| Feb – Mar 2027 | Paper 2 (method): CoG or Pacific Graphics |

## 8. Stop list

* No more inference machinery (A1–A4 type).
* TAS is not a headline number.
* Petra and Machu Picchu are not evidence of city realism; they are demos.
* No 3D art work until the 2D result is settled, except the matched renders for the study.
* No more literature notes.

## 9. Risks and hedges

* **Hilly-city budget against the calendar:** 30–38 towns to check. Hedge: one seeded pool, split by rule up front;
  the hybrid on TNg is the primary arm, so the confirmation does not depend on TNg-H.
* **The procedural baseline may win.** Hedge: the pre-declared rule in §2; that outcome is reported honestly.
* **Real envelopes from few towns.** Hedge: at least 19 towns, percentiles, the LiDAR bias check, and the
  completeness gate.
* **OSM steps are mapped unevenly.** Hedge: report stairs metrics only for towns that pass the completeness gate.
* **IRB delay.** Hedge: the expert-only pilot.
