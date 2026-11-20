# Scripts

Every script is a thin command-line entry point. Reusable code lives in `src/urbanmorph/` (see `docs/OVERVIEW.md`).
Script names start with the milestone that introduced them; historical scripts are kept unchanged at their original
paths because the milestone documents and reproducibility trails cite those exact commands.

## The conference pipeline (M12 to M15)

| Step | Command | What it does |
|---|---|---|
| 1. Data | `m12_build_tiles.py`, `m12_fetch_dem.py`, `m12_prepare_data.py` | 512 m tiles: street graphs, DEM, water (M12) |
| 2. Train | `m12_train_all.py` | the TerrainGraphTurtle backbones TN / TNlp / TNse x 3 seeds (`checkpoints/m12/`) |
| 3. Calibrate | `m13_calibrate_potentials.py`, `m14_calibrate.py` | terrain potentials (M13) and network-potential weights (M14), training tiles only |
| 4. M13 experiments | `m13_run_all.py` (or `m13_experiments.py --plan ...`) then `m13_report.py` | M* and the other M13 samplers |
| 5. M14 experiments | `m14_plans.py <plan>`, `m14_experiments.py --plan ...`, `m14_report.py`, `m14_figures.py` | network-aware samplers (NA-*) |
| 6. M15 benchmark | `m15_build_benchmark.py --stage all` | real-city pilot manifest, GLO-30, graphs, strata |
| 7. M15 experiments | `m15_plans.py eval / real`, `m15_run.py --plan ...`, `m15_report.py` | style guidance on M* / NA-fixed, synthetic + real pilot |
| Audit | `m15_replay_baseline.py` | replays the archived M14 baselines bit for bit |

Held-out data: the sealed test splits are read only through the authorisation gate in
`src/urbanmorph/data/m12_index.py` (`test_authorised`); `m12_final_test.sh` is the one-time test evaluation and needs
explicit authorisation (it has not been run). The M15 held-out metros are declared in `configs/m15_benchmark/manifest.json` and are not built.

## Milestone 15 (style guidance, pilot)

| Script | Purpose |
|---|---|
| `m15_build_benchmark.py` | versioned site manifest and preprocessing (docs/M15_PROTOCOL.md §5). |
| `m15_dev_select.py` | apply the pre-written style-strength selection rule (configs/m15_style.json "selection_rule") to the development sweep (configs/m15_plans/dev_tune.json) and freeze the strengths. |
| `m15_plans.py` | Milestone 15 run plans, generated from the frozen protocol (docs/M15_PROTOCOL.md §2). |
| `m15_replay_baseline.py` | deterministic replay of the archived M14 baselines. |
| `m15_run.py` | style guidance on top of the existing samplers (docs/M15_PROTOCOL.md). |

## Milestone 14 (network-aware inference)

| Script | Purpose |
|---|---|
| `m14_analyze.py` | Milestone 14 analysis library (docs/M14_EXPERIMENTAL_PROTOCOL.md §5-§7). |
| `m14_audit.py` | is connectivity / coverage actually a problem of the M13 generators? (stored runs only) |
| `m14_build_gen2.py` | never used for training, selection or evaluation before M14. |
| `m14_calibrate.py` | moment-matching calibration of the network-potential weights on TRAINING tiles (protocol §3). |
| `m14_demo3d.py` | Milestone 14 3D demonstration cities (downstream illustration; not part of any road-generation result). |
| `m14_density_calibrate.py` | per method, the stop-logit bias that brings the mean street density to the real density on DEV-A (TN_s0, inference seeds 1-2), by bisection on [-1.5, 1.5] (8 steps, stop when /ratio - 1/ <= 0.05). Development data only. Methods: A, M*, NA-myopic, NA-SMC (NA-fixed and NA-adaptive are not density-matched: cost; documented). |
| `m14_experiments.py` | generation + metrics, resumable. |
| `m14_figures.py` | Milestone 14 figures from stored runs (never generates). |
| `m14_gallery_all.py` | M14 side-by-side gallery of every method on GEN2 (one typical tile per city). |
| `m14_plans.py` | Milestone 14 run plans, generated from the frozen configuration (docs/M14_EXPERIMENTAL_PROTOCOL.md). |
| `m14_report.py` | tables, hypothesis tests and figures from the stored runs (never generates). |
| `m14_street_erase_gen2.py` | street-erased DEMs (leakage control, TNse backbone input) for the GEN2 tiles, with the factor frozen on training tiles in M13 (4.0). Same erasure as scripts/m13_street_erase.py (corridor mask, biharmonic fill, transplanted micro-relief, per-tile seeded); output data/processed/m13/terrain_se/f4/<sid>.npz (the path the TNse runner reads). Model input only: rules and metrics keep the original DEM. |
| `m14_style_adapt.py` | does a generator adapt its street-pattern style to the place it is asked to fill? |
| `m14_toy.py` | exact enumeration of the distributions in docs/M14_MATHEMATICAL_FORMULATION.md. |
| `m14_triggers.py` | Milestone 14 trigger selection and validation on DEV-A (docs/M14_EXPERIMENTAL_PROTOCOL.md §4). |

## Milestone 13 (terrain-constrained sampling)

| Script | Purpose |
|---|---|
| `m13_analyze.py` | per-run summaries and paired block-bootstrap contrasts (docs/M13_EVALUATION_PROTOCOL.md §6). |
| `m13_build_gen_cities.py` | Girona, Manresa (Spain) and San Francisco (USA). |
| `m13_calibrate_potentials.py` | calibrate the sampler potentials from TRAINING tiles only (docs/M13_METHOD.md §3.3). |
| `m13_confirm.py` | Milestone 13 confirmatory experiments, built from the frozen selection (outputs/m13/selection.json). |
| `m13_elevation_maps.py` | aerial elevation maps of the demo cities (hypsometric tint + hillshade + contours, streets on top). |
| `m13_experiments.py` | generation + per-sample metrics, resumable (docs/M13_EVALUATION_PROTOCOL.md). |
| `m13_global_vs_windowed.py` | global vs windowed generation. |
| `m13_report.py` | hypothesis tests, tables, figures and outputs/m13/FINAL_REPORT.md, all from saved runs. |
| `m13_run_all.py` | Milestone 13 unattended experiment runner (independent of any interactive session). |
| `m13_scoreboard.py` | M* vs A scoreboard and a random map gallery, from the STORED Milestone 13 runs (no new generation). |
| `m13_select.py` | Milestone 13 staged selection on DEV-A (docs/M13_EVALUATION_PROTOCOL.md §7), applied automatically. |
| `m13_street_erase.py` | street-erased DEMs for every development tile (train, val, val_t; never test). |
| `m13_three_tiles.py` | Three real-terrain tiles from the unseen GEN cities: real streets vs M* vs A, with per-tile scores. |
| `m13_trace.py` | one traced M* run on the steepest DEV-B tile, for the method figures (figs 2-6 of m13_report.py). |
| `m13_demo_cities.sh` | the selected method M* and the strongest baseline (A, standard) on IDENTICAL terrain and seed; first sample (no cherry-picking); M13 downstream fixes on. |

## Milestone 12 (terrain-aware generator: data, training, evaluation)

| Script | Purpose |
|---|---|
| `m12_audit_terrain.py` | how much water, coastline and crossing infrastructure do our tiles contain? |
| `m12_big_landscape.py` | a multi-tile terrain-aware city on ONE continuous procedural landscape (spec §21). |
| `m12_build_city3d.py` | an explorable terrain-aware city: terrain -> streets (a trained model + policy) -> BF parcels and buildings -> M11 3D scene -> terrain draping, bridges, water -> Blender (GLB, renders) -> Godot check. |
| `m12_build_structures.py` | OSM street ways with vertical metadata (bridge / tunnel / layer / ford), one table per city. |
| `m12_build_tiles.py` | build the expansion tiles of plan_m12.csv with the unchanged M2 processing. |
| `m12_calibrate_feasibility.py` | calibrate the feasibility thresholds from TRAINING tiles only. |
| `m12_city_plans.py` | 2D plans of the 3D cities (parcels drawn, as required for final city figures) and a contact sheet. |
| `m12_evaluate.py` | pre-registered evaluation (docs/M12_EVALUATION_PROTOCOL.md). Development tiles only. |
| `m12_fetch_dem.py` | fetch bare-earth elevation for the TRAIN + VALIDATION tiles of Spain and the US. |
| `m12_figures.py` | Milestone 12 result figures (development tiles only). |
| `m12_graph_levels_effect.py` | how does the 3D-aware street graph change the GraphTurtle training and validation data? |
| `m12_hills_compare.py` | the same streets model on gentle and on high hills (3 x 3 landscapes, TN_s0 + HR). |
| `m12_lit_citations.py` | forward citation chasing (OpenAlex) from seed papers, to complement the keyword search. |
| `m12_lit_search.py` | reproducible literature search on terrain-aware street-network generation. |
| `m12_prepare_data.py` | terrain + water arrays and structure labels for every development tile. |
| `m12_select_regions.py` | choose terrain-diverse expansion tiles in Los Angeles and San Diego objectively. |
| `m12_temperature_sweep.py` | sensitivity of conditional generation to the sampling temperature T. |
| `m12_terrain_audit.py` | terrain diversity of the existing train + validation tiles (test never read). |
| `m12_train.py` | train N0 (terrain-blind) and TN (terrain-conditioned) GraphTurtles on the 3D-aware graphs. |
| `m12_train_all.py` | run the training plan, two jobs at a time, each resumable (``--resume latest``). |
| `m12_final_test.sh` | the ONE-TIME final test evaluation. Requires explicit user authorisation: M12_TEST_AUTHORISED=yes bash scripts/m12_final_test.sh Everything it uses is frozen BEFORE it runs: checkpoints (outputs/m12/training/*/summary.json), thresholds |
| `m12_showcase.sh` | build, render (Blender), export (GLB) and check (Godot). Usage: bash scripts/m12_showcase.sh deep_valley seed 0 + HRV was chosen to SHOW a viaduct (a search over seeds; viaducts are rule-placed and occur |

## Milestone 11 (3D pipeline)

| Script | Purpose |
|---|---|
| `m11_big_city.py` | an n x n tile city (default 3 x 3 = 1,536 m) generated from scratch, then built in 3D. |
| `m11_build_city.py` | one tile -> city.json (urbanmorph-city v0.1) -> meshes.npz + validation.json. |
| `m11_calibrate_heights.py` | calibrate the within-block spread of synthetic floors on TRAINING tiles (validation untouched). |
| `m11_evaluate.py` | evaluation (docs/MILESTONE_11.md §5). No criteria were pre-registered for M11 3D quality; every number here is descriptive and reported as measured. |
| `m11_godot.py` | Godot 4 import / collision check (headless) and frame-rate benchmark (windowed) per tile. |
| `m11_height_audit.py` | can building heights and roof shapes come from data we already have? |
| `m11_height_priors.py` | empirical floor-count priors for synthetic building heights (TRAIN tiles only). |
| `m11_pipeline.py` | reproduce every 3D output with one command (build -> Blender -> Godot -> evaluation). |
| `m11_temperature_streets.py` | GraphTurtle N street samples at several temperatures, same seed and batch. |

## Milestone 10 (hierarchical streets and blocks)

| Script | Purpose |
|---|---|
| `m10_block_data.py` | BlockGen training / evaluation windows (docs/M10_BLOCK_GENERATION.md §4). |
| `m10_block_roundtrip.py` | representation ceiling of BlockGen's binary building canvas (diagnostic, no criteria). |
| `m10_dataset_audit.py` | does the street dataset support hierarchical / conditioned generators? |
| `m10_drift.py` | exposure-bias measure D2 (scripts/m10_street_diagnosis.py) for the unconditioned street models. |
| `m10_eval_blocks.py` | evaluate BlockGen as pre-registered (docs/M10_BLOCK_GENERATION.md §5.2-5.3). |
| `m10_eval_streets.py` | evaluate the street generators exactly as pre-registered (docs/M10_EVALUATION_PROTOCOL.md). |
| `m10_generate_cities.py` | generate new complete tiles with the recommended pipeline (no evaluation, no criteria). |
| `m10_integrate.py` | complete generated tiles. |
| `m10_landuse_audit.py` | what is the unbuilt land of real blocks? Tests the M9b rule "no street access -> green". |
| `m10_street_diagnosis.py` | why are generated streets irregular? Controlled diagnosis of GraphTurtle (M6 roads v2). |
| `m10_train_blockgen.py` | train BlockGen (docs/M10_BLOCK_GENERATION.md §4). |
| `m10_train_roads.py` | train the street-generator variants of docs/M10_EVALUATION_PROTOCOL.md §3. |

## Milestone 9 (tidy subdivision)

| Script | Purpose |
|---|---|
| `m9_checks.py` | two cheap checks before writing the plan. |
| `m9_evaluate.py` | C (M8) vs T (tidy subdivision + templates) vs TZ (T + green zoning). |
| `m9_records.py` | training records for BlockNet, zoning and template choice, from OBSERVED parcels. |
| `m9_subdivision_analysis.py` | how are real blocks subdivided? (OBSERVED parcels, TRAIN tiles, Spain + US.) |
| `m9_train.py` | train the four BlockNets (type, side, zone, shape) on OBSERVED train parcels. |

## Milestone 8 (parcels)

| Script | Purpose |
|---|---|
| `m8_analysis.py` | Milestone 8, Phase 1 (research and diagnosis; no training). |
| `m8_evaluate.py` | Milestone 8 evaluation of the no-parcel experiments (A, B1, B2, S) on the real validation street networks. |
| `m8_evaluate_parcels.py` | Milestone 8 evaluation of the parcel experiments (C, D) against S, on the validation tiles that have OBSERVED parcels (Spain 6, US 4; docs/MILESTONE_8.md §5). |
| `m8_fetch_parcels.py` | fetch observed cadastral parcels (raw data stays in data/raw/parcels/, git-ignored). |
| `m8_parcel_records.py` | per-parcel records of the OBSERVED parcels, with the context used by C and D. |
| `m8_parcels.py` | clip OBSERVED parcels to our train / val tiles and measure them (layout/parcels.py rules). |
| `m8_shape_library.py` | fit the real-footprint shape library on TRAIN buildings. |
| `m8_shapes_density.py` | (1) can a small library of real footprint shapes replace the sampled shapes? (2) is parcel building coverage spatially autocorrelated (dense parcels next to dense parcels)? |
| `m8_train_b1.py` | train the edge-parameterised frontage model (FrontageNetEdge). |
| `m8_train_b2.py` | neighbour-aware fallback placer (no parcels). |
| `m8_train_d.py` | train the parcel mixture-density network on OBSERVED train parcels. |
| `m8_train_s.py` | train the frontage model and the fallback placer with the real-shape library. |

## Milestone 7 (constraints, frontage)

| Script | Purpose |
|---|---|
| `m7_evaluate.py` | strict constraints + frontage model vs Milestone 6, on real validation street networks. |
| `m7_train_frontage.py` | train the frontage model on REAL street networks and buildings. |

## Milestone 6 (hybrid neural-procedural)

| Script | Purpose |
|---|---|
| `m6_build_objects.py` | encode the buildings and green spaces of every TRAIN and VAL tile (test tiles are not read). |
| `m6_build_roads.py` | street graphs WITH street classes for train and val tiles (test tiles are not read). |
| `m6_evaluate.py` | roads v2, buildings + green on real roads (P1 vs P2), full pipeline vs raster baselines. |
| `m6_train_placer.py` | train the object placer on REAL street networks. |
| `m6_train_roads.py` | train the street-graph generator v2 (turtle frame + street classes). |

## Milestone 5 (NTG-inspired street graphs)

| Script | Purpose |
|---|---|
| `m5_build_graphs.py` | extract the planar street graph of every TRAIN and VAL tile (test tiles are not read). |
| `m5_evaluate_graph.py` | evaluate the NTG-inspired street-graph generator against real validation graphs. |
| `m5_train_graph.py` | train the NTG-inspired street-graph generator. |

## Milestone 4 (PixelCNN prior)

| Script | Purpose |
|---|---|
| `m4_evaluate.py` | compare priors on the SAME frozen C2 decoder with the SAME evaluation. |
| `m4_train_prior.py` | train the Gated PixelCNN prior on codes of the FROZEN C2 encoder. |

## Road-topology experiment (soft-clDice)

| Script | Purpose |
|---|---|
| `topo_evaluate.py` | Road-topology experiment: identical evaluation of C2 (lambda = 0) and C2 + lambda * soft-clDice. |
| `topo_experiments.py` | Road-topology experiment: C2 + lambda * soft-clDice (roads only). |
| `topo_prior_diagnosis.py` | Road-topology experiment, part 7: does the factorised N(0, I) prior cause discontinuities between latent cells? |
| `topo_report.py` | Road-topology experiment: aggregate evaluations into tables and figures (outputs/topology/). |

## Milestone 3b (VAE data and loss controls)

| Script | Purpose |
|---|---|
| `m3b_build_expansion.py` | process the expansion tiles into TRAIN-only samples (offline). |
| `m3b_data_control.py` | data-size control for the spatial VAE (B trained on the ORIGINAL 97 training tiles). |
| `m3b_evaluate.py` | identical validation-only evaluation for every experiment. |
| `m3b_expansion_plan.py` | plan additional TRAINING tiles from cached data (no network). |
| `m3b_experiments.py` | run the controlled experiments sequentially, then evaluate all of them. |
| `m3b_seed_check.py` | repeat selected experiments with another seed to estimate run-to-run variation. |

## Milestone 3 (VAE baseline)

| Script | Purpose |
|---|---|
| `m3_dataset_audit.py` | independent audit of the M2 dataset before training. |
| `m3_evaluate.py` | evaluate the VAE against the AE and trivial baselines, and make figures. |
| `m3_sweep.py` | small validation-only sweep (latent size x beta) plus AE baselines. |
| `m3_train.py` | train the VAE (or the deterministic AE baseline) from a config. |

## Milestone 2 (multi-city dataset)

| Script | Purpose |
|---|---|
| `m2_analysis.py` | exploratory comparison of Spain, Peru and South Korea. |
| `m2_build_dataset.py` | cached raw OSM -> quality-checked, split dataset (offline). |
| `m2_download.py` | build the raw OSM cache for planned tiles (resumable). |
| `m2_explorer.py` | Interactive dataset explorer (Matplotlib only, runs locally). |
| `m2_plan_sampling.py` | build the sampling plan (no OSM requests). |

## Milestone 1 (single-tile prototype)

| Script | Purpose |
|---|---|
| `build_raster.py` | Step 2: cached raw OSM data -> data/processed/<sample_id>.npy (+ .json, _vectors.gpkg). |
| `compare_milestones.py` | Generate one new tile FROM SCRATCH with the model of every milestone (M3 to M8), side by side. |
| `download_osm.py` | Step 1: download roads, buildings and green spaces for the sample area. |
| `full_city.py` | One full 512 m tile generated from scratch, drawn large: roads v2 (M6) -> M8 S, and the same streets -> M8 C (generated parcels drawn) for comparison. |
| `generate_tz.py` | New cities fully from scratch: street graph (M6 roads v2) -> TZ (M9 tidy parcels + zoning + templates). |
| `visualize.py` | Step 3: render figures for a processed sample into outputs/figures/. |
