# Development Log

## Milestone 1 — Real urban data → ML-ready layout (2026-10-30)

### What was implemented

- `src/urbanmorph/data/osm.py`: downloads the street graph, road-area polygons, buildings
  and green spaces for a fixed centre (41.3910 N, 2.1650 E) and radius, with retries and
  backoff over two Overpass endpoints. Raw responses are cached under
  `data/raw/barcelona_eixample_001/` (GraphML + GeoJSON + `download_info.json` with query,
  timestamp, versions, attribution). Later steps never touch the network.
- `preprocessing/`: projection to EPSG:25831, a 512 m square around the projected centre,
  road-width estimation and buffering, clipping, and a pixel-centre rasterizer.
- `data/dataset.py`: save / load / validate (shape, dtype, binary values, metadata keys,
  geometric consistency).
- `visualization/layout_plot.py`: per-channel maps, combined map, vector-vs-raster comparison.
- 29 tests (synthetic geometry tests + real-data tests that use only the local cache).

Result for `barcelona_eixample_001`: roads 19.7%, buildings 56.9%, green 1.25% of pixels;
overlaps road∩building 3 px, road∩green 5 px, building∩green 83 px; 3,721 px (22.7%)
unclassified (sidewalks, courtyards, plazas).

### Technical decisions

| Decision | Why |
|---|---|
| Centre point + fixed size, not a lat/lon bbox | A metric square is the actual sample unit; a lat/lon box is not square in metres. Download radius = half-diagonal + 100 m so buffers of roads just outside the square are included. |
| EPSG:25831 (ETRS89 / UTM 31N) | Official metric CRS for Catalonia; < 0.1% scale distortion here. |
| Own rasterizer (Shapely 2 `intersects_xy`) instead of `rasterio` | One dependency fewer, and the whole algorithm is ~10 readable lines. Same rule as rasterio's default (`all_touched=False`). |
| Pixel-centre rule, boundary counts as inside | Deterministic and unbiased: area is preserved on average. `all_touched` would systematically inflate thin features. |
| Road classes: motorway…residential, living_street, pedestrian | Barcelona maps sidewalks as separate `footway` ways; including them would double-count. Service roads and paths are excluded. |
| Width = max(`width` tag, `lanes` × 3.25 m) → class default | Tags conflict (`width=5, lanes=4` on Carrer de València); lanes give a physical lower bound. Final sources: 55 lanes, 33 width tag, 42 class default, 7 area polygons. |
| Road areas (`area=yes`) downloaded as polygons | In the graph their outline becomes edges, which buffered into hollow rings. |
| Tunnels excluded from roads | They occupy no surface. |
| Channels independent (multi-label), no priority | Lossless with respect to the vectors; overlaps are counted in metadata, and an exclusive label map can be derived later. Combined figure paint order: green < buildings < roads. |
| Green = vegetated land regardless of access | Not all green is `leisure=park`: interior-block gardens are `leisure=garden`, verges are `landuse=grass`. Points (e.g. single trees) are dropped, since only polygons have area. |
| Raw data kept in EPSG:4326, as received | Preserves the original source. Processed, clipped vectors are saved separately (`*_vectors.gpkg`). |
| Plot palette validated for colour-vision deficiency | The first attempt (terracotta vs. green) was indistinguishable under deuteranopia (ΔE 0.8). Now blue / amber / dark green. |

### Problems encountered

1. **Editable install invisible to Python.** The project lives on an iCloud-synced Desktop.
   iCloud sets the macOS `hidden` flag on files inside `.venv/`, and Python ≥ 3.12 ignores
   hidden `.pth` files, so `import urbanmorph` failed. `chflags nohidden` is undone within
   seconds. Fix: scripts insert `src/` into `sys.path`, and pytest uses `pythonpath = ["src"]`.
   Moving the project out of iCloud is the cleaner long-term fix (it also avoids syncing
   thousands of venv files).
2. **Hollow rings at pedestrian squares**: fixed with the road-area layer (see above).
3. **Thin, dashed streets**: real 3–3.6 m lanes (Consell de Cent superilla, Passeig de Gràcia
   side lanes) at 45° to the grid miss pixel centres. Contradictory width tags made more
   streets too thin than necessary, which was fixed with the lanes lower bound. The rest is
   inherent to 4 m pixels and is documented as a limitation.
4. **Slow Overpass responses**: the first download took ~3.8 min. Caching means this happens once.
5. **Numeric columns stringified in the vector export**: fixed so only object columns are converted.

### Known limitations

- Road widths are estimates of the carriageway, not measured street widths; sidewalks are
  not included. The road channel therefore under-represents public street space
  (Eixample streets are ~20 m façade to façade).
- Thin features (< 4 m) may be fragmented or lost; connectivity is not guaranteed in the raster.
- OSM completeness: green spaces in particular are under-mapped (private interior gardens).
  Nothing is inferred or fabricated.
- One sample only; no dataset split or statistics across neighbourhoods yet.
- The sample grid is aligned to UTM north, not to the street grid (Eixample is rotated ~45°),
  which is fine for learning but means orientation varies between cities.
- Building heights are not used (kept in raw data for the 3D milestone).
- Re-downloading later gives *current* OSM data, so the cache (or an OSM snapshot date) is
  what makes results reproducible.

### Next milestone (recommended)

**Milestone 2 — A small, documented multi-city dataset.**
1. Literature review (generative urban layouts, street-network generation, evaluation metrics).
2. Tile several neighbourhoods (Eixample, Gràcia, Ciutat Vella, plus contrasting cities) into
   hundreds of 512 m samples with a sliding window; store per-sample metadata.
3. Compute the conditioning variables per sample (building coverage, road coverage, green
   coverage, dominant street orientation, intersection density) and plot their distributions.
4. Define train/val/test splits **by area** (no spatial leakage between neighbouring tiles).
5. Only then: a first baseline model (e.g. a small conditional VAE) in Milestone 3.

---

## Milestone 2: Comparative urban-morphology pilot (2026-10-30)

### What was implemented

| Component | Module / script |
|---|---|
| City boundaries (Nominatim), per-city UTM CRS | `sampling/study_area.py` |
| GHSL GHS-BUILT-S R2023A tiles (1975/2000/2020 + non-residential 2020), per-tile built-up fraction | `sampling/ghsl.py` |
| Snapped 512 m candidate grid, main-agglomeration filter | `sampling/grid.py` |
| Strata (density × era of urbanisation), stratified + spatially separated selection with reserves | `sampling/strata.py`, `sampling/select.py`, `scripts/m2_plan_sampling.py` |
| OSM from dated Geofabrik extracts via osmium (Overpass path kept) | `data/osm_extract.py`, `scripts/m2_download.py` |
| Metrics: coverage, building statistics, intersection density, orientation | `metrics/morphology.py` |
| Quality flags (sample- and dataset-level), per-city reports | `quality/checks.py`, `scripts/m2_build_dataset.py` |
| Geographic splits with blocks + buffer, city hold-out | `splits/geographic.py` |
| Consolidated index (`samples.csv`), full per-sample JSON metadata | `data/m2_dataset.py` |
| Exploratory comparison figures + summary table | `visualization/compare.py`, `scripts/m2_analysis.py` |
| Matplotlib dataset explorer (interactive + headless) | `scripts/m2_explorer.py` |
| Literature review | `docs/LITERATURE_REVIEW.md` |

**Result:** 135 tiles (15 per city × 9 cities), all usable (no error flags). Splits are train 96,
val 21, test 18, with every city present in every split. No tile was excluded by the buffer,
because the 1.5 km selection spacing already guarantees more than 1 km between tile edges.

### Sampling methodology (and why)

1. **Study area = the main contiguous built-up agglomeration inside the administrative unit.**
   The units differ by country (Spanish *municipios*, Peruvian *provincias*, Korean metropolitan
   cities), and the large ones contain detached towns. Eligible cells have GHSL 2020 built-up ≥ 15%
   and lie fully inside the boundary. Only the largest cluster of eligible cells is kept, with gaps
   of up to 2 cells (1,024 m) bridged. This dropped 0–104 cells per city.
2. **Strata are defined by GHSL, never by OSM.** Stratifying on OSM building density would confuse
   "dense" with "well mapped". Two definitions are used:
   * *density*: GHSL 2020 built-up fraction, cut at **absolute** thresholds **0.272 / 0.373**. These
     are the pooled terciles of eligible cells, with each city weighted equally, frozen in the
     config before any OSM download.
   * *era of urbanisation*: the first GHSL epoch (1975, 2000, 2020) at which the tile reaches 15%
     built-up. This is **not** building age, and "by1975" lumps medieval cores together with
     1970s estates.
   * *non-residential share* (GHSL NRES 2020) is recorded as a covariate, not used as a stratum
     (15 tiles cannot fill 27 strata).
3. **Allocation:** one tile per non-empty stratum, the rest proportional to stratum size. Each tile
   records `p_inclusion = n_h/N_h` and `design_weight = 1/p`, so analyses can be weighted.
4. **Selection:** seeded random order within each stratum (seed `20261030` plus a CRC of the city
   code), round-robin over strata, with tile centres at least 1,536 m apart. Reserves are drawn
   the same way.
5. **OSM completeness is not used in selection.** Quality is assessed after download. Unusable
   tiles would be replaced by same-stratum reserves, and the replacement is logged (none was needed).

### Important technical decisions

| Decision | Why |
|---|---|
| Geofabrik extracts + `osmium` instead of Overpass | Overpass was overloaded (502 "at capacity", connections refused on 3 mirrors) and asks automated users to look elsewhere. Extracts put zero API load on it, use one snapshot for every tile (2026-10-28T21:21:02Z), cost about 2 s per tile, and scale to ~900 tiles. |
| Replicate OSMnx's graph construction step by step | Buffered extract → unsimplified graph → truncate → simplify → truncate to the bbox, matching `graph_from_point`. **Verified:** the Milestone 1 Eixample tile and a Cusco tile give *pixel-identical* rasters and identical feature counts from both sources. |
| One combined polygon request per tile | Halves API load on the Overpass path. Layers are split locally with the same tag rules. |
| Grid snapped to multiples of 512 m in UTM | Tiles are either identical or disjoint, so overlap checks are exact and IDs (`PER_LIM_e561_n16895`) are stable. |
| Intersections: ≥ 3 streets, consolidated at a 10 m tolerance | Counted on the graph, not on pixels. Dual carriageways and complex junctions are merged (nodes < 20 m apart). Only nodes inside the tile are counted, and the download margin keeps their degree correct. |
| Orientation from undirected, length-weighted segments | Two-way streets are counted once. Bearings are folded to 0–180°, with an 18-bin entropy, Boeing's φ, a doubled-angle dominant axis and a quadrupled-angle grid orientation (see CONCEPTS). |
| Building statistics by the centroid rule, using full footprints | Each building belongs to exactly one tile; areas are not truncated at tile edges. |
| An empty green channel is `info`, not an error | It may be real or unmapped, and OSM cannot distinguish the two. |
| CSV, not Parquet | Avoids adding `pyarrow`. |

### Problems encountered

1. **Wrong administrative units.** "Arequipa, Peru" and "Cusco, Peru" returned whole *regions*
   (63–72k km²), so the queries now ask for the provinces. The original Madrid query returned nothing.
2. **Overpass overload.** The first tile took 247 s, then all mirrors refused connections. I
   stopped the run and switched to extracts (with your approval).
3. **Order-dependent road widths (a bug inherited from Milestone 1).** OSMnx merges ways into
   list-valued attributes (`lanes = ['2','3']`) whose order differed between sources, and Milestone
   1 took the first element. Rules now use the maximum plausible value and the widest class, and
   `is_tunnel` requires every part to be a tunnel. The Milestone 1 sample was rebuilt (road
   coverage 19.73% → 19.75%).
4. **List tags as Python-repr strings** (`"['2', '3']"`) are now parsed with an `ast.literal_eval`
   fallback. `;`-separated multi-values are handled too.
5. **A tile without any streets** (Madrid-Barajas airport) crashed graph building. Empty graphs are
   now stored and flagged `no_roads`.
6. **Matplotlib backend.** `layout_plot.py` forced the non-GUI `Agg` backend on import, which would
   have made the explorer window blank. Scripts now choose the backend instead.
7. **The OSMnx HTTP cache landed in `./cache`** during boundary fetches. It was moved to `data/raw/boundaries/http_cache`.

### Data-quality findings (pilot)

* **OSM building completeness is extremely uneven.** Median OSM-to-GHSL building-coverage ratio:
  Barcelona 0.79, Madrid 0.74, Valencia 0.86; Seoul 0.37, Daejeon 0.20, Busan 0.16;
  Cusco 0.40, Arequipa 0.22, **Lima 0.02**. Share of tiles flagged `buildings_incomplete_suspected`:
  13–20% in Spain, 67–87% in Korea, 73–93% in Peru. Lima's streets are mapped but its buildings
  mostly are not (see `outputs/m2/explorer_example.png`). This agrees with Herfort et al. (2023).
* **False positives of that flag:** port and industrial land. GHSL counts container yards as
  built-up surface (Port of Valencia: GHSL 0.77, NRES share 0.96, OSM 0.003). The NRES share
  identifies these tiles.
* **Infrastructure tiles are legitimately in the sample.** For example, Madrid-Barajas airport
  (no public streets; one ~38,000 m² terminal) is legitimate urban land, not an error.
* **Historic cores break the road-width defaults.** In Valencia's Ciutat Vella, 6–8 m class
  defaults exceed the 3–4 m streets, and the road buffers overlap buildings (the only
  `excessive_channel_overlap` flag).
* **Empty green channel:** 26 of 135 tiles. These are kept and flagged `info`.

### Exploratory comparison (descriptive only)

From `outputs/m2/analysis/summary_by_city.md` (city medians, n = 15 per city):
* **Road coverage** is similar across all nine cities (0.10–0.16), and road mapping looks
  complete everywhere. This is the most comparable channel.
* **Intersection density** has overlapping ranges: city medians are 50–126 per km², and the
  within-city spread is larger than the between-country differences.
* **Per-tile street-grid order φ** has medians of 0.44 (Cusco) to 0.81 (Daejeon). **Pooled per
  city**, φ is low everywhere (0.03–0.24), because each city mixes differently oriented grids.
  "Grid-like" is a neighbourhood property, not a city property.
* **Building metrics are dominated by mapping completeness.** For example, Korea's large median
  footprints (490–650 m²) may partly reflect that mainly large buildings are mapped.
  **No cross-country conclusion about buildings should be drawn from this pilot.**

No significance tests were run: tiles are spatially clustered, there are only three cities per
country, and the strata are unequally sampled.

### Known limitations

* 15 tiles per city is a pilot. Some strata hold a single tile, which carries a large design weight.
* `p_inclusion` ignores the distance constraint, which slightly favours sparse areas.
* GHSL starts in 1975 and is itself a model (Landsat/Sentinel-2), with errors at 100 m.
* Administrative units differ by country; the agglomeration filter reduces but doesn't remove this.
* Bearings are relative to UTM grid north (≤ ~1.3° from true north here).
* A 1.5 km spacing caps the tiles per city: Cusco's ~175 eligible cells allow roughly 40.
  **About 300 tiles per country will need more cities or a smaller spacing** (with buffer-based exclusion).
* Three cities per country is too few to separate "country" from "city" effects.

### Before training a model (recommended)

1. **Fix building completeness first.** Either (a) add an external footprint layer (e.g. Overture
   Maps buildings, which combine OSM with ML-derived footprints, or national cadastres, checking
   licence compatibility), or (b) restrict training and evaluation to tiles with an OSM/GHSL ratio
   ≥ 0.5 and document the resulting selection bias. (b) would currently drop most Peruvian and
   Korean tiles.
2. Replace road class defaults with a calibrated width model, or use `area:highway` polygons where they exist.
3. Scale up to ~6–10 cities per country with the same extract pipeline before any country-level claim.
4. Pre-register the evaluation: per-metric errors on leave-one-city-out folds, separated from country classification.

---

## Milestone 2, v2: Spain, United States, Japan (2026-10-30)

### Why the change
The v1 pilot showed that OSM building coverage in Peru and South Korea was only 2–40% of the
satellite (GHSL) built-up estimate, against 74–86% in Spain. Building-based comparisons would have
measured mapping effort, not urban form. At the project lead's direction, Peru and South Korea
were replaced by the **United States** and **Japan**. v1 is archived, not deleted:
`configs/m2_pilot_v1_peru_korea.toml`, `data/processed/m2_v1_peru_korea/`, `outputs/m2_v1_peru_korea/`.

### What changed
* **Cities:** New York City, Chicago, Los Angeles; Tokyo (23 wards, OSM relation R19631009,
  queried by ID because name searches were ambiguous), Osaka, Nagoya. Spain is unchanged.
* **City-eligibility criterion (new):** a study city must have a median OSM/GHSL building ratio
  ≥ 0.5 over its tiles, otherwise it is replaced by a configured backup city (San Diego, Yokohama,
  Kyoto, Kobe; same extracts). All nine passed, so no backup was used:

  | | | | |
  |---|---|---|---|
  | Barcelona 0.94 | Madrid 0.77 | Valencia 0.94 | |
  | New York 0.63 | Chicago 0.72 | Los Angeles 0.68 | |
  | Tokyo 0.73 | Osaka 0.76 | Nagoya 0.68 | |

* **Agglomeration rule:** v1 kept only the largest cluster of built-up cells. That dropped
  **Staten Island** (357 cells; separated from Brooklyn by the 1.6 km Narrows) and **San Pedro /
  the Port of LA** (184 cells; joined to LA only by a narrow strip). v2 keeps every cluster with at
  least 5% of the largest cluster's cells. The 5% was chosen *after* inspecting cluster sizes and is
  documented as post hoc.
* **Density thresholds recalibrated** for the new city set: 0.311 / 0.393 (v1: 0.272 / 0.373).
  As a result Spain's tiles were redrawn too.
* **Single snapshot:** all 9 extracts (~3.4 GB in total) have replication timestamp
  2026-10-28T21:21:02Z, and all MD5 checksums were verified.
* **Code:** country and city order in figures and the explorer now come from the config, not
  hard-coded lists. Multi-word names are labelled properly ("Los Angeles"). Boundary queries accept
  OSM IDs. `city_completeness.csv` is written by the build.

### Results (135 tiles, all usable; train 97 / val 19 / test 19)
* **Quality:** 18 tiles (13%) are flagged `buildings_incomplete_suspected` (v1: 54%). Of the 6
  tiles with ratio < 0.2, **4 are container ports** (Barcelona, Valencia, Port of LA, Osaka), where
  GHSL counts container yards as built-up, and **2 are genuine OSM gaps in residential Edogawa
  (eastern Tokyo)**. 22 tiles have an empty green channel (flagged `info`); this is most common in
  Los Angeles (47%), where private yards are not mapped as green space.
* **City medians** (`outputs/m2/analysis/summary_by_city.md`):
  * building coverage is now similar across countries: 0.24–0.29 in Spain, 0.20–0.24 in the US, 0.24–0.28 in Japan;
  * intersection density: Japan 111–217 per km², Spain 61–114, US 38–76;
  * grid order φ: Chicago 0.99 and New York 0.94 (strict grids), Los Angeles 0.65, Spain 0.55–0.76, Japan 0.54–0.77;
  * median building footprint: Japan 64–79 m², US 103–184 m², Spain 187–397 m². Part of this
    gap may reflect *how* buildings are delineated in each source (Spanish cadastre imports versus
    per-house footprints in Japan), not only real building size.

### New limitation found: road width defaults are country-biased
77% of Japanese road segments get a class-default width (Spain 42%, US 66%), and Japanese OSM tags
most local streets `unclassified` (41% of length) or `residential` (28%). The 7–8 m defaults exceed
the ~4 m width of many Japanese local streets, so **Japan's higher road coverage (0.14–0.21) is
partly an artefact**. Road coverage is therefore *not* yet comparable across countries.
Graph-based metrics (intersection density, orientation) are unaffected.

### Before training a model (updated)
1. **Calibrate road widths per country** against measured data (e.g. municipal planimetric street
   widths in NYC, Japan GSI road-edge data, Spanish cadastre street polygons), or switch the road
   channel to measured `area:highway` / street-space polygons where available.
2. Handle ports and industrial land explicitly (e.g. a GHSL non-residential-share covariate or stratum).
3. Scale up: 6–10 cities per country from the same extracts. More US states and Japanese regions
   are available from Geofabrik.
4. Report clearly that the study population is *well-mapped cities*, and treat extending it to
   under-mapped regions (with completed building layers) as separate future work.

---

## Milestone 3: First generative model, a convolutional VAE baseline (2026-10-30)

Full write-up: [docs/MILESTONE_3.md](docs/MILESTONE_3.md).

### What was done
* Milestone 2 frozen on branch `milestone-2` (commit `1898a9b`); Milestone 3 work is on `milestone-3`.
* An independent dataset audit (`scripts/m3_dataset_audit.py`) found no structural problems.
  Splits are ≥ 1,024 m apart and there are no near-duplicates (max pairwise building IoU 0.415).
  **The dataset was not changed.**
* PyTorch 2.15 (MPS + CPU fallback) added as the `ml` extra. A 3.67M-parameter convolutional VAE
  (z = 32) and its deterministic AE twin. Losses, training loop with KL warm-up, early stopping,
  checkpoints (git-ignored), histories, evaluation, figures, and 14 new tests (81 in total).
* Decisions taken with the project lead: 8-fold rotate/flip augmentation (training only), and
  `sqrt(neg/pos)` positive weighting.

### Problems encountered
1. **The short run looked broken** (road IoU 0.00). An overfitting diagnostic showed the network
   memorises 8 tiles perfectly (IoU 1.00), so the code was not at fault; the problem is generalising from 97 tiles.
2. **The VAE prior samples were wrong** (0.4% roads, 22.6% green). The diagnosis is a prior /
   aggregate-posterior mismatch: posterior means spread with std ≈ 3.3 against the prior's 1. A
   control that samples from a Gaussian fitted to the training codes (no retraining) fixed the
   coverage but not the roads.
3. **The connectivity metric can be fooled by blobs.** One blob counts as one connected road
   component. This is documented, and a skeleton-based metric is a next step.
4. Loss spikes during training (Adam instability), visible in the loss curves.

### Results (validation only; the test split is unused)
* Selected VAE: z = 32, β = 0.5 (lowest validation −ELBO at β = 1).
* VAE IoU: roads 0.046, buildings 0.252, green 0.413. AE IoU: 0.079 / 0.176 / 0.511. The trivial
  baseline gets 0 / 0 for roads and buildings. The models beat the trivial baseline on buildings
  and green, but **not on road likelihood** (road BCE 0.45 against 0.40).
* Train ≈ validation loss everywhere, so the models are underfit, not overfit.
* There is no memorisation, and the latent space interpolates smoothly. The latent space encodes
  green coverage (R² 0.86), but not street structure, and cities are not separable (k-NN at chance).
* **Conclusion:** the baseline does not generate realistic neighbourhoods. This is the reference
  point that future models must beat on the same metrics.
* Reproducibility: a repeat of the selected run with the same seed gave bit-identical results
  (MPS, torch 2.15). CPU determinism is covered by a unit test.

### Next
More training tiles (documented M3b), a spatial or hierarchical latent, a better prior (larger β or
two-stage), a road topology metric/loss, and then a diffusion model. The test split is used once,
only for the final chosen model.
