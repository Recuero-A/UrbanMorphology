# M18 D7 (c): GLO-30 terrain inputs and the TNg backbone

Built 2026-11-26 (KST). TNg is the M12 TN backbone with Copernicus GLO-30 (30 m DSM) terrain inputs instead of
the LiDAR DTMs. It is a leakage control.

**No blocker.** The build, the smoke test and the launch all succeeded. See "Open points" for what evaluation still
needs.

## 1. Data: `data/processed/m18/terrain_glo30/` (git-ignored)

Script: `scripts/m18_build_glo30_terrain.py`. Log: `build.log`. Provenance: `manifest.json`.

**Scope.** Exactly the development tiles that have LiDAR terrain today (`valid.any()` in the M12 npz): 525 tiles.

| City | train | val | val_t | GLO-30 cells |
|---|---|---|---|---|
| ESP_BCN | 39 | 2 | – | N41E002 |
| ESP_MAD | 70 | 2 | – | N40W004 |
| ESP_VLC | 31 | 2 | – | N39W001 |
| USA_CHI | 71 | 2 | – | N41W088, N42W088 |
| USA_LAX | 191 | 2 | – | N33W119, N34W119 |
| USA_NYC | 71 | 2 | – | N40W075, N40W074 |
| USA_SAN | – | – | 40 | N32W118, N32W117, N33W118, N33W117 |
| **Total** | **473** | **12** | **40** | 13 cells |

* **val_t is included** because TN's model selection uses val + val_t.
* **Test splits are never touched.** They are removed by `load_index_m12(include_test=False)` before any request,
  and the script refuses to run when `M12_INCLUDE_TEST=yes`.
* **Source cells.** 13 cells from `https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif` (public, no account).
  * 10 were downloaded into `data/raw/dem/glo30_src/`.
  * 3 were reused read-only from earlier caches (N41E002 and N39W001 from `glo30_src`, N40W004 from
    `data/raw/m16/glo30`).
  * The sha256 of every cell is in `manifest.json`.
* **Per tile, how the array was made.**
  * The city's cells are mosaicked on their native grid, with bounds snapped to pixel edges.
  * They are then reprojected with bilinear resampling to the tile's own M12 grid: tile UTM CRS, 4 m, tile plus a
    256 m margin, 256 × 256. This is the same transform as `terrain.data.tile_terrain`.
  * `elev` and `valid` come from GLO-30. Every tile is 100 % valid.
  * `water` and `struct_label` are copied unchanged from the M12 npz. Keys and dtypes are identical.
* **`norm.json`.** This is the M12 recipe (`m12_prepare_data.py`: same seed, same tile order) applied to the 473
  GLO-30 training tiles.
  * Re-run on the LiDAR arrays, the recipe reproduces the M12 `norm.json` exactly.
  * GLO-30 values: elev_scale **6.92** (LiDAR 6.81), slope_scale **0.201** (LiDAR 0.129).
* **Alignment check.** On 24 high-relief tiles, the best-fit shift between GLO-30 and LiDAR is mostly 0–8 m. It is
  at most 16 m per axis, in two Madrid tiles. There is no systematic offset.

## 2. Audit: GLO-30 vs LiDAR on the same grid

Measured inside the 512 m tile and given as the median over tiles. Full data: `audit_tiles.csv`, `audit_cities.csv`.

| City | n | bias m | RMSE m | RMSE de-biased m | relief p95–p5 LiDAR / GLO m | slope p90 4 m LiDAR / GLO % | \|Δslope\| 4 m % | slope p90 32 m LiDAR / GLO % | \|Δslope\| 32 m % | r(elev) |
|---|---|---|---|---|---|---|---|---|---|---|
| ESP_BCN | 41 | +4.6 | 6.1 | 3.2 | 11.1 / 15.5 | 10.0 / 23.8 | 9.8 | 5.2 / 13.5 | 3.8 | 0.64 |
| ESP_MAD | 72 | +2.8 | 3.8 | 2.6 | 15.1 / 18.4 | 14.8 / 18.8 | 7.0 | 8.2 / 11.6 | 3.0 | 0.88 |
| ESP_VLC | 33 | +3.1 | 4.9 | 3.5 | 3.0 / 11.6 | 8.0 / 21.1 | 8.9 | 2.5 / 12.3 | 4.6 | 0.35 |
| USA_CHI | 73 | +1.6 | 2.5 | 1.7 | 2.0 / 6.1 | 4.8 / 13.5 | 5.9 | 1.8 / 6.6 | 2.8 | 0.23 |
| USA_LAX | 193 | +1.7 | 2.4 | 1.7 | 37.7 / 37.8 | 38.8 / 26.4 | 7.9 | 23.5 / 21.8 | 2.2 | 0.99 |
| USA_NYC | 73 | +3.3 | 4.3 | 2.5 | 8.3 / 12.3 | 10.0 / 17.3 | 7.2 | 4.7 / 9.5 | 3.2 | 0.63 |
| USA_SAN | 40 | +1.3 | 2.0 | 1.5 | 25.1 / 23.9 | 24.9 / 18.9 | 6.4 | 16.7 / 15.5 | 1.9 | 0.98 |
| **All** | 525 | +2.0 | 2.9 | 2.0 | 11.7 / 15.8 | 13.1 / 20.0 | 7.3 | 7.3 / 12.4 | 2.7 | 0.88 |

**Reading.**
* **Bias.** GLO-30 sits 1–5 m above LiDAR. The causes are the surface model (buildings and trees) and the different
  datums (EGM2008 vs REDNAP / NAVD88).
* **Hilly cities (LA, San Diego).** GLO-30 follows the landform (r 0.98–0.99, 32 m slopes within about 2 points)
  but flattens short steep slopes (4 m p90 slope: 39 → 26 %).
* **Flat cities (Chicago, Valencia, Barcelona, NYC).** GLO-30 relief is mostly surface clutter (r 0.23–0.64), and
  its slopes are 2–3× the LiDAR slopes.
* **Caveat for D7.** GLO-30 removes the street beds, but as a DSM it can show building blocks as bumps and streets
  as troughs (docs/M16_TERRAIN_SOURCES.md §5.3).
  * TNg is therefore not leakage-free by construction.
  * Recommended: run the D6 leakage probe (`scripts/m18_erasure_validation.py`) on GLO-30 inputs before using TNg
    as the clean control.

## 3. TNg training

**Files.**
* Config: `configs/m18_tng.toml`. Its [model], [train] and [sample] sections are a copy of `configs/m6_roads.toml`,
  and the wrapper checks that they are equal.
* Wrapper: `scripts/m18_train_tng.py`. It calls `scripts/m12_train.py`'s own `main()` unchanged.

**What is the same as TN.**
* Model: 1,399,878 parameters.
* Channels: elev, slope, water (+ valid).
* Recipe: 60 epochs, patience 10, warmup 300, the same seeds.
* Data: the same graphs and splits (685 train, 19 val + 40 val_t).
* Tiles without a DEM: the 219 Japanese tiles keep their unchanged M12 arrays (valid = 0).

**What differs from TN.**
* The elevation input comes from GLO-30, which also changes the slope channel.
* `norm.json` is the GLO-30 one, stored in each checkpoint.
* The run config records `dem_variant = "glo30"`. It is part of the config hash, so a resume with other inputs is
  refused.

**Outputs.**
* Checkpoints: `checkpoints/m12/TNg_s{seed}/` (new directories).
* Logs and summary: `outputs/m18/glo30/training/TNg_s{seed}/`.
* Console: `train_TNg_s{seed}.log`.

**Smoke test.** One pilot epoch on 64 / 16 tiles, written to the session scratchpad: 25 steps, loss finite, the
checkpoint written, and the resume accepted.

**Commands** (run from the repo root). Each one starts the run, or resumes it after any interruption:

```
nohup .venv/bin/python scripts/m18_train_tng.py --seed 0 --resume latest >> outputs/m18/glo30/train_TNg_s0.log 2>&1 &
nohup .venv/bin/python scripts/m18_train_tng.py --seed 1 --resume latest >> outputs/m18/glo30/train_TNg_s1.log 2>&1 &
nohup .venv/bin/python scripts/m18_train_tng.py --seed 2 --resume latest >> outputs/m18/glo30/train_TNg_s2.log 2>&1 &
```

Stop a run cleanly with `kill <PID>` (SIGTERM). It saves a checkpoint before exiting.

**Launched 2026-11-26.**

| Seed | Started (KST) | PID |
|---|---|---|
| s0 | 00:44:44 | 54385 |
| s1 | 00:46:18 | 54760 |

**Speed.**
* One run alone: 42.5 s per epoch, 228 steps per epoch (TN_s0 took 48 s).
* Two runs in parallel: 64–65 s per epoch each, which is about 30 % more total throughput.
* Memory: about 2 GB RSS per run.

**Expected finish.** With all 60 epochs (TN ran all 60), both runs end at about 01:50 KST on 2026-11-26 (16:50 UTC on
11-25). Allow up to 02:00 KST, because the `m18_natural_terrain.py` job shares the machine.

**Seed 2.** It has not been launched. Run it alone after s0 and s1: about 43 min, so roughly 02:35 KST if started at
01:50.

## 4. Open points (not blockers; nothing existing was modified)

* **Evaluation code.** `urbanmorph.experiments.backbones.model_inputs` reads any `dem_variant` as an M13
  street-erased folder (`split("_")[1]`).
  * For TNg (`"glo30"`) it raises an IndexError. This is loud rather than silent.
  * Before evaluating TNg, it must be extended to feed `data/processed/m18/terrain_glo30/<sid>.npz`, or the GLO-30
    arrays of the evaluation sites.
* **Testing on GLO-30 (D7) needs GLO-30 arrays for the evaluation sites.**
  * The development tiles are here.
  * The M15 real windows already have `terrain_glo30/`.
  * The sealed test split was not built.
