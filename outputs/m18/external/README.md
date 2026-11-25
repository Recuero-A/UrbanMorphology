# M18 external baselines on the terrain sites (Machu Picchu, Petra)

This report covers two external street generators added to the site benchmark (`docs/SITE_BENCHMARKS.md`). Both get
exactly the inputs of the 24 internal versions:

* the 2048 m domain and the 4 m GLO-30 grid;
* the water layer;
* the fixed city limits;
* seeds 1-3.

Neither method gets supplied roads or anchors. Each uses only the DEM, the water layer and the city-limits mask, plus
seeded internal choices. Both are scored by the unchanged score code of `scripts/site_benchmark.py`: typicality
realism, TAS = 100 (G + C + S) / 3, and the statistics table. Milestone `external`. This is exploratory, like the
benchmark itself.

| Label in the tables | What it is |
|---|---|
| **leastcostpath (ext)** | leastcostpath (Lewis) routed between seeded anchors; merge by us |
| **citygen P&M (ext)** | t-mw/citygen run headless, with only the Parish & Müller 2001 local constraints added |

## Results (means of seeds 1-3 ± seed SD; full tables in `outputs/sites/<site>/README.md`)

| Site | Version | Realism (lower = better) | TAS | G | C | S | km/km² | Coverage 60 m | Largest piece |
|---|---|---|---|---|---|---|---|---|---|
| Machu Picchu | leastcostpath (ext) | 1.86 ± 0.09 | 72.8 ± 0.2 | 0.73 | 0.79 | 0.67 | 17.8 | 66 % | 97 % |
| Machu Picchu | citygen P&M (ext), **sparse** | 4.32 ± 1.03 | 75.3 ± 4.4 | 0.92 | 0.88 | 0.45 | 2.9 | 11 % | 100 % |
| Machu Picchu | NA-SMC | 1.45 ± 0.28 | 56.4 ± 1.1 | 0.45 | 0.64 | 0.61 | 19.1 | 90 % | 56 % |
| Machu Picchu | NA-fixed | 2.04 ± 0.26 | 64.1 ± 0.4 | 0.54 | 0.67 | 0.71 | 7.6 | 66 % | 26 % |
| Petra | leastcostpath (ext) | 1.15 ± 0.08 | 71.8 ± 0.4 | 0.91 | 0.64 | 0.60 | 18.4 | 75 % | 100 % |
| Petra | citygen P&M (ext) | 1.37 ± 0.08 | 70.3 ± 1.4 | 0.87 | 0.55 | 0.69 | 11.4 | 54 % | 100 % |
| Petra | NA-SMC | 1.10 ± 0.05 | 57.8 ± 1.3 | 0.67 | 0.41 | 0.66 | 25.1 | 96 % | 87 % |
| Petra | NA-fixed | 1.41 ± 0.19 | 68.3 ± 3.3 | 0.79 | 0.40 | 0.85 | 11.5 | 71 % | 89 % |

`M* + repair` has no runs at the two sites. It exists only in the GEN2 distributions (`outputs/m18/distributions/`),
so it is not in this table.

Rank by realism among the 26 versions: leastcostpath is 12th at Machu Picchu and 4th at Petra; citygen is 26th
(last, sparse) at Machu Picchu and 11th at Petra.

**Per seed** (realism / TAS / km/km²):

| Site | Version | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| Machu Picchu | leastcostpath | 1.95 / 72.7 / 17.3 | 1.78 / 72.9 / 18.0 | 1.85 / 73.0 / 18.2 |
| Machu Picchu | citygen | 4.32 / 71.1 / 0.2 | 3.29 / 79.9 / 8.0 | 5.35 / 74.8 / 0.5 |
| Petra | leastcostpath | 1.20 / 72.0 / 19.2 | 1.06 / 72.0 / 18.1 | 1.19 / 71.3 / 17.9 |
| Petra | citygen | 1.36 / 71.8 / 12.4 | 1.30 / 69.0 / 9.5 | 1.46 / 69.9 / 12.3 |

The Machu Picchu citygen realism comes from only 3, 6 and 4 of the 8 windows, because the scorer skips windows
without streets.

**Read with these caveats:**

1. **leastcostpath's TAS at Machu Picchu is partly a raster artefact.** The cost has no turning penalty, so on land
   steeper than the 20 % cut the paths climb by zigzags of 4-9 m legs: 44 % of the vertex turns in seed 1 exceed
   120°. Each leg is under 20 %, so G and C are high, but the zigzags are not buildable switchbacks. This is the
   optimal zigzag of Llobera & Sluckin (2007) under such a cost.
   * We re-built the same merged graph with the repository's polyline-to-graph convention (`build_graph`: resample
     ≤ 30 m, merge 8 m, as used for the real OSM ways). This collapses the zigzags.
   * Result: Machu Picchu TAS 64.7 and realism 1.27 (10.7 km/km²); Petra TAS 72.3 and realism 1.10.
   * This is in memory only (`sensitivity.json`), not in the site tables.
2. **citygen's TAS at Machu Picchu describes very few streets.** With a 20 % grade rule, P&M growth dies on the
   ridge in two of three seeds (0.25 and 0.80 km). Seed 2 reached the density target.
3. **Both baselines are "terrain-rule" methods by construction**, so G is high: every DEM step is under 20 %.
   * On the raw DEM, 0 % of citygen's length has a step ≥ 20 %.
   * For leastcostpath it is 1.4-3.0 %, from our merge connectors and the bilinear sampling (`runs.json`).
   * At Machu Picchu, leastcostpath has 9 m of street on mapped water: knight moves clip the corners of water cells.

## Baseline 1: leastcostpath (Lewis) routed between seeded anchors; merge by us

**Versions.**
* leastcostpath **2.0.13** (CRAN binary for R 4.3, arm64; GPL-3), installed with `install.packages` into the user
  library `~/Library/R/arm64/4.3/library`.
* terra 1.8-54, sf 1.0-21, igraph 2.1.4, Matrix 1.5-4.1.
* R 4.3.1. The packages were built under R 4.3.3, which only produces a warning.

**Code.**
* R part: `scripts/m18_ext_leastcostpath.R`.
* Python glue: `scripts/m18_external.py` (`run_lcp`, `lcp_anchors`, `lcp_merge`).
* Intermediates: `data/processed/sites/<site>/external/ext_leastcostpath_s<seed>/`: `dem.tif`, `water.tif`,
  `anchors.csv`, `paths.csv`, `r.log`, `stats.json`.

**Method.**
1. **Anchors.** Farthest-point sampling of K = 12 cell centres.
   * Candidate cells are inside the city limits, not water, with a DEM slope below the city's median DEM slope.
   * The slope uses central differences on the raw 4 m grid. The thresholds are 73.7 % at Machu Picchu and 28.4 %
     at Petra.
   * The first anchor is drawn with `numpy.random.default_rng(seed)`. The rest are deterministic.
2. **Cost surface.**
   * `create_slope_cs(dem, cost_function = "wheeled transport", neighbours = 16, crit_slope = 12, max_slope = 20)`.
   * The package default `crit_slope` is 12 %, so the same value is used.
   * Note: the cost object's `criticalSlope` metadata field records `max_slope`. This is a cosmetic bug in
     `create_slope_cs`; the cost function receives `crit_slope = 12`.
   * `max_slope = 20` sets conductance to 0 when |slope| ≥ 20 %. It is the same 20 % as for citygen.
   * Justification (`docs/literature/street_grades_real_world.md`):
     * 15 % is the AASHTO/NYSDOT local-street maximum;
     * 20 % sits between that and the "> 30 % implausible" level;
     * real streets have p99 16.2 % (F9), and the internal hard rule allows up to 21.8 % for class 3 (F10);
     * Lewis & Crema 2026 use critical slopes of 9 and 15 %.
3. **Barriers.**
   * Water cells get conductance 0 through the package's `update_values` (FUN = j × 0). This changes nothing here,
     because the city limits already exclude water.
   * **Routing is restricted to the city limits:** the DEM is NA outside them, and the package drops NA cells from
     the adjacency. This is our choice, the counterpart of "keep growth inside the limits" for citygen.
4. **Paths.**
   * `create_FETE_lcps(cs, anchors)` routes all ordered pairs, 132 per run.
   * Anchors that cannot be entered from any neighbour are dropped. None was dropped.
   * Unreachable pairs come back empty: the passable land (steps < 20 %) splits into components.

   | Site | Seed | Paths / ordered pairs | Connected unordered pairs |
   |---|---|---|---|
   | Machu Picchu | 1 | 92 / 132 | 46 / 66 |
   | Machu Picchu | 2 | 110 / 132 | 55 / 66 |
   | Machu Picchu | 3 | 92 / 132 | 46 / 66 |
   | Petra | 1-3 | 110 / 132 | 55 / 66 |

5. **Merge (ours, Python/shapely).**
   1. Paths are added longest first.
   2. A vertex within **8 m** of the network built so far is a duplicate. Each run of such vertices is dropped, and the
      path is joined to the network at the network vertex nearest to the run's ends (the snap).
   3. `unary_union` nodes every crossing and dissolves overlapping pieces.
   4. Repeated edges are dropped.
   5. Class 3, structure ground.

   Raw path length 135-314 km per run → 24.8-28.3 km after the merge.

   **Deviation.** The first implementation used plain GEOS `snap(…, 8 m)` followed by a union. On the zigzags it
   joined vertices of parallel zigzags into a mesh, with node degrees up to 11. It was replaced before any scoring.

## Baseline 2: citygen P&M (headless t-mw/citygen with P&M 2001 local constraints)

**Source.**
* `third_party/citygen` holds t-mw/citygen at commit **aec49d2efaa59a1eebffaacc322459c5219ab9a6** (2026-03-17, "Upgrade
  to vite v8"). The tarball was fetched from codeload.github.com without git (sha256 `3b286b32…585c81`).
* MIT licence; `LICENSE` is kept.
* The upstream files are unmodified. `npm ci` installed the pinned lockfile: esbuild 0.27.3, seedrandom 2.4.4,
  noisejs 2.1.0, @timohausmann/quadtree-js 1.2.6.
* Node v22.13.1, npm 10.9.2.

**Code.**
* `scripts/m18_citygen/generate_pm.ts` is a copy of the upstream `generate.ts`, with every change marked `M18`.
* `scripts/m18_citygen/runner.ts` is the headless CLI. No browser and no pixi: it imports upstream `segment`, `math`,
  `config` and `heatmap`.
* The bundle `scripts/m18_citygen/build/runner.cjs` is built with esbuild and is git-ignored.
* Python glue: `run_citygen` in `scripts/m18_external.py`.

**Method.**
* **RNG.** `seedrandom(String(seed), {global: true})`, as upstream. Citygen's own simplex population heatmap is kept
  unchanged, seeded from the same stream, with the city-limits centroid at its origin (0, 0). No population data
  is an input of any method.
* **Scale.** 1 citygen unit = 62.87 / 300 = **0.2096 m**.
  * Upstream at default settings has a median junction-to-junction link of exactly 300 units, its
    `DEFAULT_SEGMENT_LENGTH` (seeds 1-3, measured with `m18_audit_m14.links`).
  * The real median link is 62.87 m (`outputs/m18/distributions/summary.json`, real GEN2 tiles).
  * So a street segment is 62.9 m, a highway segment 83.8 m, and the snap radius 10.5 m.
  * The generator runs in its native units, and metres are applied only at input and output.
* **Start.** The city-limits centroid, with the upstream axiom (a highway pair, due east), when that segment is legal.
  Otherwise the first legal direction is used in 5° steps. At both sites the axiom stayed at the centroid
  (Machu Picchu: 150°; Petra: 90°).
* **Stop.**
  * The summed segment length reaches the target: the median street density of the 24 internal versions × the
    city area. Machu Picchu: 8.35 km/km² × 1.427 km² = 11.9 km. Petra: 12.39 km/km² × 1.467 km² = 18.2 km.
  * Or the upstream segment budget of 2000 is reached (never), or the queue empties.

  | Site | Seed | Outcome |
  |---|---|---|
  | Machu Picchu | 1 | queue empty, 0.25 km |
  | Machu Picchu | 2 | target reached |
  | Machu Picchu | 3 | queue empty, 0.80 km |
  | Petra | 1 | target reached |
  | Petra | 2 | queue empty, 14.0 km |
  | Petra | 3 | target reached |

  Exact duplicate segments in citygen's output (up to 0.54 km) are dropped in the conversion, so the stored density
  is slightly below the target.
* **localConstraints().** Only the P&M 2001 constraints are added (§3.3, PDF p. 305, quoted from the ETH author
  copy):

  > "check if the road segment ends inside or crosses an illegal area. … If the first check determines that the road
  > end is inside water, a park or another illegal area, the system tries to readjust the values for the road segment
  > in the following ways. Prune the segment length up to a certain factor so that it fits inside the legal area of the
  > starting point. Rotate the segment within a maximal angle until it is completely inside the legal area. This allows
  > the creation of roads that follow a coastline or a park boundary."

  P&M list elevation among the local constraints ("local: land/water/park boundaries, elevation, crossing of
  streets", p. 303, §3.2.1). They give no number, so the 20 % grade is ours. Illegal means any of:
  * water;
  * outside the city limits;
  * a DEM step of 4 m or less along the segment with a grade ≥ 20 %. The elevation is bilinear on the raw 4 m grid,
    the same rule as baseline 1.

  The readjustments, in P&M's order:
  1. prune to the last legal point if that keeps at least **50 %** of the length;
  2. else rotate up to **90°** in 5° steps (the sign order is seeded) until the whole segment is legal;
  3. else FAILED (the segment is rejected).

  The rest of `localConstraints()` is upstream: crossings, snapping to crossings, extending to roads. Its only
  change: an end moved by a snap or extension must also be legal.

  Not used:
  * P&M's highway bridge rule, because segments entering water are rejected or shortened;
  * the San Francisco rule, a global goal that P&M leave without a formula.
* **Classes.** Highway → class 1, street → class 2. Structure ground.

**Parameter choices.** The paper gives no prune factor and no maximal angle, so these are ours.
* The first setting was 30° (citygen's own `MINIMUM_INTERSECTION_DEVIATION`).
* A growth-only check showed that at 30° the queue empties early: Petra grows 0.9-7.3 km of its 18.2 km target, and
  Machu Picchu 0.2-0.9 km of 11.9 km.
* 90° is the angle P&M's coastline-following needs for a road that meets the boundary head-on, so 90° was adopted.
  The choice was made on growth only, before any realism or TAS of citygen was computed.
* For transparency, `sensitivity.json` scores the 30° variant:
  * Machu Picchu: realism 4.43, TAS 71.0, 0.3 km/km²;
  * Petra: realism 2.14, TAS 73.5, 3.0 km/km².

## Sanity checks (`sanity/`, `python scripts/m18_external.py sanity`)

* **leastcostpath reproduces its package example.** The verbatim `?create_FETE_lcps` example runs: Sicily 1 km DEM,
  tobler, 4 neighbours, 4 locations of which 3 and 4 coincide.
  * It returns the 6 paths the package's own message implies (12 ordered pairs − 6 "duplicate locations").
  * Path lengths are 298, 122, 122, 298, 176 and 176 km (1→2 and 2→1 have the same length; 1→3 = 1→4 and
    2→3 = 2→4 because locations 3 and 4 coincide).
  * See `sanity/fig_sanity.png` (left).
* **The citygen runner reproduces upstream.** At default settings, the headless upstream `generateCity(seed)` gives
  2000 / 2001 / 2000 segments for seeds 1-3.
  * Our derived generator, with terrain off and the default axiom, outputs **identical** segment lists, bit for bit.
  * The layout matches the demo: curving highways, near-grid blocks and stubs. Compare `sanity/fig_sanity.png`
    (right) with `third_party/citygen/screen.jpg`.
  * This also gives the scale calibration: median link 300.0 units.
* **The existing rows are unchanged.** After the official `score` stage, all 26 old rows of `summary.csv` and all 74
  old rows of `per_seed.csv`, at both sites, are identical as text to the pre-run backups (every column). Their
  relative order is the same, `summary.json` rows and windows are identical, and `missing_runs` is 0.
* **The cross-site comparison is unchanged.** `stage_compare` now keeps only the declared 24 internal versions
  (milestone ≠ external). It was not re-run, and `outputs/sites/cross_site.json` is untouched.

## Runtime and hardware

| Step | Time |
|---|---|
| leastcostpath, per run | 11.0-13.4 s (R 6.4-8.3 s inside R; merge 1.5-2.6 s) |
| citygen, per run | 0.8-1.0 s (Node 0.4-0.6 s) |
| Generation of all 12 external runs | about 30 s, with 6 workers |
| `score`, whole site | 19 s (Machu Picchu) and 12 s (Petra), with 12 workers |
| `figures`, whole site | 25 s |

Hardware: Apple M5 Pro (15 cores), 48 GiB RAM, macOS 26.6.2. Python 3.12.2 (shapely 2.2.0, numpy 2.5.4, scipy 1.18.1,
rasterio 1.5.2).

## How to reproduce

```
Rscript -e 'install.packages("leastcostpath", lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org")'
(cd third_party/citygen && npm ci)
.venv/bin/python scripts/m18_external.py build
.venv/bin/python scripts/site_benchmark.py generate --site machu_picchu --only ext_leastcostpath,ext_citygen_PM
.venv/bin/python scripts/site_benchmark.py generate --site petra --only ext_leastcostpath,ext_citygen_PM
.venv/bin/python scripts/site_benchmark.py score --site <site>; .venv/bin/python scripts/site_benchmark.py figures --site <site>
.venv/bin/python scripts/m18_external.py sanity | report | sensitivity
```

## Files

* `runs.json`: per-run diagnostics (anchors, paths, stop reason, prune/rotate/fail counts, population share, runtime,
  and the share of length with raw-DEM steps ≥ 20 %).
* `sensitivity.json`: the two in-memory variants above.
* `sanity/`: the package-example paths, the citygen demo outputs, `sanity.json` and `fig_sanity.png`.
* Site outputs regenerated by the official stages: `outputs/sites/<site>/{summary.csv, per_seed.csv, summary.json,
  README.md, fig_maps_seed1.png, fig_realism_vs_terrain.png, maps/}`.
  * The `maps/` file names carry the realism rank, and the two new rows shifted some ranks. The superseded files
    with old rank prefixes are still there (39 at Machu Picchu, 63 at Petra; git-ignored, local only). They were not
    deleted.

## Changes to `scripts/site_benchmark.py`

The changes are additive. The labels, keys and milestone are `EXTERNAL`, `EXT_KEY` and `"external"`. Beyond those:
* **Jobs.** `jobs()` includes the two externals.
* **Generation.** `generate_one` dispatches the externals to `m18_external.generate`. The cache format is unchanged.
* **Figures.** The externals get categorical slot 5 (`#e87ba4`, validated with slots 1-4) and a diamond marker.
* **Layout.** Two label nudges at Machu Picchu.
* **README.** One line explains "(ext)".
* **Comparison.** `stage_compare` excludes `external`.

No score function was touched.
