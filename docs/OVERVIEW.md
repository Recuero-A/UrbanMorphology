# UrbanMorph: start here

This page is the map of the project for a reviewer: what the system is, the names used everywhere, where the code
and results are, and how to reproduce them. Details are in the per-milestone documents it links to.

## 1. What the system is

UrbanMorph generates neighbourhood street networks (512 m × 512 m tiles) that respond to terrain and water, and can
turn them into 3D scenes.

* **The generator** is a learned autoregressive vector road-graph model in the style of Neural Turtle Graphics (Chu
  et al. 2019), called **TerrainGraphTurtle**.
  * A frontier queue is expanded about 26 m at a time.
  * A GRU encodes the path; a CNN encodes elevation, slope and water around the current node.
  * Heads predict stop, a mixture-density step, the road class, and bridge or tunnel.
* **The trained network is never changed after Milestone 12.** Every later milestone changes only how it is sampled.

## 2. The research line used for the paper (M12 to M15)

| Milestone | Question | Outcome (development data) | Documents |
|---|---|---|---|
| M12 | Can a terrain-conditioned generator fit terrain better than a terrain-blind one? | Yes, but part of the gain is street traces in urban DEMs; leakage controls TNlp / TNse | `docs/MILESTONE_12.md`, `docs/M12_RESULTS.md` |
| M13 | Do terrain constraints and look-ahead at sampling time help? | Hard rules remove infeasible crossings; realism gains vanish under leakage and density controls (negative) | `docs/MILESTONE_13.md`, `outputs/m13/FINAL_REPORT.md` |
| M14 | Do network-level potentials fix fragmentation? | Yes: connectivity improves about 70 % on four unseen cities, diversity kept; adaptive look-ahead no better than random | `docs/M14_*.md`, `outputs/m14/REPORT.md`, `docs/M14_REVIEWER_SUMMARY.md` |
| M15 | Can optional style guidance give a requested morphology on terrain, and do M14's gains survive? | Pilot in progress | `docs/M15_*.md`, `outputs/m15/REPORT.md` |

Milestones 1–11 built the dataset, earlier raster and graph models, parcels, buildings and the 3D pipeline. They are
listed in `README.md`.

The sealed test splits of M12 have **not** been evaluated. The M15 held-out metros are declared but not built.

## 3. Names used everywhere

**Sampler configurations.** One registry, `src/urbanmorph/experiments/methods.py`. Every name is the same trained
backbone with a different inference procedure:

| Name | What it is |
|---|---|
| A | Ancestral sampling: one candidate, no rules |
| M\* | The M13 terrain-aware baseline: 8 candidates per step, hard feasibility rules, soft terrain potentials |
| M\*+repair | M\* then post-hoc connection repair |
| P | Procedural (rule-based) terrain-aware generator |
| M13-SMC | M\* proposal inside sequential Monte Carlo (4 particles): the effect of SMC alone |
| NA-myopic | M\* + network potentials (closed dead ends and islands) |
| NA-fixed | NA-myopic + 2-step look-ahead at every decision |
| NA-adaptive | NA-myopic + look-ahead where a trigger fires |
| NA-random | Same as NA-adaptive, but the trigger fires at random at the same rate (its control) |
| NA-SMC | NA-myopic inside SMC |
| + style | M15: optional style guidance on any of the above (grid, organic, radial, cul-de-sac, mixtures) |
| TRACER | M15 external baseline: Chen-inspired cross-field street tracer |

**Backbones.** `TN_s{0,1,2}` (raw DEM), `TNlp_s*` (DEM smoothed at 16 m) and `TNse_s*` (street beds erased from
the DEM). The last two are leakage controls; checkpoints are in `checkpoints/m12/<name>/best.pt`.

**Data sets:**

| Set | Contents | Use |
|---|---|---|
| train / val / val_t / test | M12 spatial splits (3 km blocks, 1 km buffers) | test is sealed |
| DEV-A, DEV-B | Development tiles split by block | M13–M14 tuning (DEV-A) and confirmation (DEV-B) |
| GEN | Girona, Manresa, San Francisco | M13 generalisation |
| GEN2 | Alcoi, Tarragona, Santa Barbara, Sacramento | M14 primary confirmation |
| syn_dev, syn_eval | Analytic synthetic terrains (flat, incline, ridge, valley, river) | M15; dev for tuning only |
| real_dev | 120 windows in 6 development metros (Spain, South Korea, Peru) | M15 real-city pilot, GLO-30 elevation |

**Metrics** (definitions and units):

| Family | Where defined |
|---|---|
| S10, S_T | `docs/M13_EVALUATION_PROTOCOL.md` |
| OD unreachability, islands, coverage | `docs/M14_EXPERIMENTAL_PROTOCOL.md` |
| Style (R4, junction shares, sinuosity, radial / tangential, dead ends, loops, block areas, circuity bands), grade exposure, precision / recall / density / coverage | `docs/M15_METRICS.md` |

## 4. Code map

| Concept | Module |
|---|---|
| Street graph type, M2/M12 graph metrics (incl. S10 inputs) | `src/urbanmorph/graphs/road_graph.py` |
| Generation state and the shared graph operations (snapping, splitting, merging) | `src/urbanmorph/graphs/generate.py` (`GenState`) |
| The trained network | `src/urbanmorph/models/terrain_graph_turtle.py` |
| Model inputs (terrain crops, turtle frame) | `src/urbanmorph/graphs/terrain_sequence.py`, `sequence.py` |
| M13 sampler family (A, M\*, SMC, beam, look-ahead) | `src/urbanmorph/graphs/tc_sampler.py` |
| Hard rules and terrain potentials | `src/urbanmorph/graphs/tc_potentials.py` |
| M14 network-aware sampler (NA-\*) | `src/urbanmorph/graphs/m14_sampler.py` |
| M15 style guidance and grade ceiling | `src/urbanmorph/graphs/style.py` |
| Procedural baseline P, post-hoc repair | `src/urbanmorph/graphs/procedural_terrain.py`, `repair.py` |
| M15 tracer baseline | `src/urbanmorph/graphs/field_tracer.py` |
| Metrics by milestone | `terrain_metrics.py` (S_T), `m13_metrics.py`, `m14_metrics.py`, `style_metrics.py` (M15), `src/urbanmorph/metrics/fidelity.py` |
| Fixtures (analytic terrains, hand-built graphs) | `src/urbanmorph/graphs/fixtures.py` |
| Experiment infrastructure (backbones, method registry, graph I/O, sites) | `src/urbanmorph/experiments/` |
| Terrain and water data | `src/urbanmorph/terrain/` |
| 3D scenes | `src/urbanmorph/city3d/` and `blender/` |

* **Entry points.** Command-line entry points are in `scripts/`, indexed in `scripts/README.md`, with the M12–M15
  pipeline at the top.
* **Configs.** Frozen configurations and run plans are in `configs/`:
  * `m13_potentials/`;
  * `m14_lambda.json`, `m14_frozen.json`;
  * `m15_style.json`;
  * `m15_plans/`;
  * `m15_benchmark/` (the versioned site manifest).
* **Tests.** `tests/`, run with `python -m pytest -q`. They cover the graph operations, sampler invariants
  (M14 telescoping; an inactive style factor when style is off, with the full style-off = baseline check done by the
  replay audit), metrics on analytic fixtures, and the split fingerprints.

**Naming caveat.** Some library modules are named after the milestone that added them (`m13_metrics`, `m14_sampler`,
`tc_*` = "terrain-constrained"). The table above maps them to concepts. They were not renamed, because stored run
files and documents cite them.

## 5. Rules that every experiment follows

* Development data only, unless the project owner explicitly authorises a held-out evaluation.
* Protocols are written and committed before confirmatory runs. Amendments are appended with date and reason, and
  negative results are reported.
* Every run's specification is hashed into its file name. Runs are resumable per seed and never overwritten by a
  different specification.
* The M14 baselines replay bit for bit after the M15 changes (`outputs/m15/audit/`).

## 6. Reproduction

* **Environment:** Python 3.12 virtual environment (`README.md` › Installation).
* **Paths:** run scripts from the repository root.
* **Large inputs** (raw OSM extracts, DEMs, checkpoints, per-run JSON) are not in git. The build scripts recreate them
  from pinned sources; the manifests and logs record versions and failures.
* **Per milestone:** the reproducibility section of each results document lists the exact commands (e.g.
  `docs/M14_REPRODUCIBILITY.md`, `docs/M15_PROTOCOL.md` §7).
