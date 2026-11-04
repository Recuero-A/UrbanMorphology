# Milestone 12: terrain-aware Neural Turtle Graphics

**Status (2026-11-04): Phase 1 (research and audit) complete. Waiting at approval gate 1.**
* No DEM or other data has been downloaded.
* No model has been trained.
* The test split has not been read.

Detail documents:
* `M12_LITERATURE_REVIEW.md`
* `M12_DATA_PROTOCOL.md`
* `M12_TERRAIN_BRIEF.md` (the summary of M1–M11 written before this milestone)

---

## 1. Phase 1: what was done

| Item | Result | Where |
|---|---|---|
| Repository inspection | Integration points identified (§2) | — |
| Literature (17 works; 9 read in full or in part) | Comparison table, qualified novelty assessment | `M12_LITERATURE_REVIEW.md` |
| Terrain and water audit, 565 train + 19 val tiles, OSM only | Water, coastline, bridges, tunnels, crossings, tag coverage | `scripts/m12_audit_terrain.py`, `outputs/m12/audits/` |
| Elevation-source investigation (no download) | DTM candidates per country with datum and licence status | `M12_DATA_PROTOCOL.md` §2 |
| Checkpointing and automatic resume (requested 2026-11-04) | Implemented and tested before any training | `src/urbanmorph/models/checkpoint.py`, `tests/test_checkpoint.py` |
| Dependencies | `rasterio` 1.5.2 and `pyproj` present. `scipy` absent (only needed for convenience in hydrology; NumPy suffices). No large application needed | — |

## 2. Repository inspection: where terrain enters

| Component | Current behaviour | M12 change (planned; baseline untouched) |
|---|---|---|
| `graphs/sequence.py` `crops_turtle` | 3-channel 48 × 48 crop (streets, open nodes, outside), rotated to the heading | Add terrain channels sampled on the **same rotated grid**. The gradient vector is rotated by −heading |
| `models/graph_gen.py` `GraphTurtle` | `Conv2d(3, …)` input; scalars; GRU decoder; 10-component mixture, stop, class | New `TerrainGraphTurtle` subclass with `3 + k` input channels and terrain scalars. `GraphTurtle` keeps its exact behaviour and checkpoints |
| `graphs/generate.py` | Geometric rules after each step: clip, split at crossings, merge 8 m, T-junction, drop short | A feasibility hook *before* the rules (rules-only R) and a bounded replanning loop (HR). The baseline path is unchanged when the hook is absent |
| `graphs/road_graph.py` `build_graph` | **Planar**: every crossing within 8 m is merged | A 3D-aware variant keeps `bridge`, `tunnel` and `layer`, so grade-separated crossings are not junctions. 76% of bridge length crosses non-water (audit) |
| `scripts/m11_big_city.py` | Windowed growth with `given` / `frontier` | Each window samples crops from one city-wide terrain field |
| `city3d/*`, Blender, Godot | Flat z = 0 | Phase 8 |

## 3. Scientific findings of Phase 1

1. **Terrain-conditioned learned street generation already exists, for raster models** (DeepStreet 2020; Fang 2022;
   Gu 2024).
   * Adding elevation channels is therefore not a contribution.
   * Naively added terrain channels helped little in two independent studies (Fang 2022, Model 1; Gu 2024).
   * The defensible contribution space is: sequential graph decisions with feasibility feedback, explicit
     infrastructure actions, and 3D-consistent graphs. This holds only after a systematic search.
2. **Our data contain water mainly in Japan.**
   * 81 training tiles have ≥ 1% water, 57 of them in Tokyo, Osaka and Nagoya; 23 tiles are coastal.
   * **Validation has water in only 2–4 of its 19 tiles**, too few for any water or bridge claim.
3. **OSM crossing labels are usable.** About 90% of street–water crossings are bridge-tagged, and untagged ones will
   be "unknown".
4. **Relief cannot be assessed yet.** The tiles were sampled from dense, mostly flat built-up areas. Whether they
   contain moderate or steep slopes is unknown until a DTM is fetched. This is the main risk to the milestone.
5. **Checkpointing.**
   * Interrupted and resumed training equals uninterrupted training. Parameters match to 1e-7 (CPU; exact in
     practice) for interruptions mid-chunk, at a chunk end and at an epoch end, and for two interruptions in a
     row.
   * Disabling RNG restoration makes these tests fail, so they detect a broken resume.

## 4. Checkpoint and resume contract (all M12 training scripts)

`Checkpointer` in `src/urbanmorph/models/checkpoint.py`.

**When it saves**
* every 300 s (`--ckpt-interval`) at the next safe point (after an optimiser step);
* at the end of every epoch;
* on Ctrl+C or SIGTERM. A second signal aborts immediately.

**What it saves:** model, optimiser, optional scheduler and gradient scaler, user state (epoch, global step,
position in the epoch, the data-generator state at the start of the current chunk, best score, history), the RNG
states (Python, NumPy, torch CPU and MPS / CUDA, named generators), and the config with its SHA-256.

**Crash safety**
* Each save goes to a temporary file, is fsynced, then `os.replace`d. The old `latest.pt` becomes `previous.pt`.
* A corrupt or truncated `latest.pt` falls back to `previous.pt`.
* **Resuming with a different configuration is refused.**

**Logs.** `RunLog` is append-only JSON lines. A resume writes a `resume` event and continues; nothing is
overwritten.

**Usage:** `--resume latest` or `--resume PATH`.

## 5. Unresolved risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Training tiles may be too flat | TN cannot learn slope behaviour, whatever its design | Measure first (gate 2). Expand with terrain-diverse cities chosen by measured diversity |
| 19 validation tiles, 2–4 with water | No statistical power for terrain or water metrics | New held-out regions (whole cities) for terrain evaluation; the existing test split untouched |
| Terrain signal is weak in learned models (literature) | TN ≈ N0 | Pre-registered ablations, a shuffled-terrain control, and R as the honest comparison |
| Learned replanning has no obvious supervision | Cannot be trained without fabricated labels | Heuristic replanning first. The learned variant only with a ranking objective whose assumptions are documented (spec §10.5) |
| A planar graph loses grade separation | False junctions near overpasses | A 3D-aware graph build (Phase 2), measured before and after |
| Mixed vertical datums | Wrong elevations across countries | Per-country orthometric conversion; slopes only across countries |
| DEM5A survey-result terms (Japan), Korean DEM unknown | Licence or availability gaps | Verify before download. Exclude regions without a compatible licence |

## 6. Estimated next-stage compute (estimates; the pilot will measure them)

| Stage | Estimate |
|---|---|
| DEM fetch for train + val boxes of the nine cities (about 584 tiles × (512 + 2·256 m)² ≈ 610 km²) | Hours of wall time at service rate limits; volume small at 4–5 m. **The exact size and time need the provider indices: verify** |
| Terrain sample building (DEM sampling, water rasters, 3D-aware graphs) | Under 1 h CPU for the current tiles |
| TN training, current data | About 80–100 min per seed on MPS (N took 79 min; extra crop channels add a little CNN cost) |
| Full plan: N0 retrain + TN (2 seeds each) + 4 single-seed channel ablations, on 2× data | About 10 runs × about 3 h ≈ 30 MPS-hours sequential (2–3 runs can share the GPU) |

## 7. Decisions requiring approval (gate 1)

1. **DEM download for the nine existing cities**, train + validation bounding boxes only (+256 m margin):
   * IGN MDT05 (Spain), USGS 3DEP 1 m (US), GSI DEM5A with DEM10B fallback (Japan);
   * through bbox or tile services, after the licence checks marked *(verify)* in `M12_DATA_PROTOCOL.md`.
2. **Then the terrain diversity audit:** elevation range, slope quantiles, street-grade distributions, measured
   per city and split.
3. **Expansion choice deferred until 2.** Are Seoul and Busan acceptable in principle as street-only cities, given
   M2 excluded them for building completeness? Are new OSM downloads acceptable (San Francisco, Rio de Janeiro)?
4. **The systematic literature search** (Scopus or Google Scholar style queries), recorded in
   `M12_LITERATURE_REVIEW.md` §6, before any novelty statement.
5. **Optional:** install `scipy` (small pip package) for hydrology convenience. NumPy-only is possible.
