# Milestone 12 data protocol: elevation, water and crossings

**Status:** Phase 1 (2026-11-04).
* **Nothing has been downloaded.** Every DEM row below is a candidate that needs approval.
* Facts marked **(verify)** come from secondary pages and must be checked against the provider's documentation
  before use.

## 1. Rules

1. **Roads, water polygons, waterways, coastlines, and bridge / tunnel / layer tags** come from the cached OSM
   clips, same snapshot as the tiles: 2026-10-28T21:21:02Z.
2. **Ground elevation** comes from bare-earth DTMs only.
   * OSM `ele` and `incline` tags are never used as a terrain model; `incline` covers < 1% of street length (§3).
   * A DSM (Copernicus GLO-30, SRTM) is used only if no DTM exists, and must then be labelled a surface model.
3. **No silent substitution.** Every tile records its DEM product, native resolution, vertical datum, horizontal
   CRS, acquisition date and licence. A tile whose source differs from its city's default is flagged.
4. **Common frame.** Everything is reprojected to the tile's existing UTM CRS (EPSG:326xx).
   * Elevations are converted to one orthometric reference per country, and the conversion (geoid model) is
     recorded.
   * Cross-country comparisons use slopes and relative elevations only. These are unaffected by a constant datum
     offset.
5. **Resampling.** The DEM is resampled to the 4 m grid of the model (bilinear for elevation; slope is computed
   after resampling *and* at native resolution for the sensitivity check). It is never described as 4 m data.
6. **Splits.**
   * The 19 test tiles are never fetched, read or summarised.
   * DEM requests use train and validation bounding boxes only, plus a 256 m margin so that crops near the edge
     have terrain.
7. **Licences.** Raw DEM files stay in `data/raw/dem/` (git-ignored). Only derived statistics and figures are
   committed. Attribution strings go into every output's metadata.

## 2. Candidate elevation sources

| Region | Product | Type | Native res. | Vertical datum | Horizontal | Licence | Access without bulk download |
|---|---|---|---|---|---|---|---|
| Spain (BCN, MAD, VLC) | IGN **MDT05** (from PNOA LiDAR, 1st coverage); MDT02 (2nd coverage, 2 m) | DTM | 5 m (2 m) | Orthometric; geoid EGM08-REDNAP **(verify)** | ETRS89 / UTM 30 extended (Peninsula) | **CC BY 4.0** (scne.es) per INSPIRE metadata | IGN WCS service (`spaignwcs_mdt`) for bbox requests **(verify limits)** |
| US (NYC, CHI, LAX) | USGS **3DEP** 1 m DEM (lidar); 1/3 arc-second seamless as fallback | DTM | 1 m (~10 m) | NAVD88 | NAD83 / UTM | Public domain (credit USGS) | 3DEP ImageServer export by bbox **(verify)**. Note: 1 m tiles are seamless within, not across, collection projects |
| Japan (TYO, OSA, NGO) | GSI 基盤地図情報 **DEM5A** (airborne laser, urban areas); DEM10B (from contours, national) | DTM | 5 m (10 m) | T.P. (Tokyo Bay mean sea level) **(verify)** | JGD2011 / JGD2024 geodetic | GSI content terms (PDL 1.0 / government standard terms), stated compatible with CC BY 4.0; **survey results under the Survey Act may need approval: verify for DEM5A** | GSI elevation tiles (標高タイル: DEM5A/5B/10B) by tile index **(verify)** |
| Korea (Busan, Seoul; expansion only) | National DEM: product and licence **unknown** | ? | ? | ? | ? | ? | Not investigated yet |
| Global fallback | Copernicus **GLO-30** | **DSM** (includes buildings and trees) | 30 m | EGM2008 | WGS84 | Free with attribution (© DLR / Airbus) | — |
| Global fallback | **FABDEM** (GLO-30 with forests and buildings removed by ML) | Bare earth (estimated) | 30 m | EGM2008 | WGS84 | **CC BY-NC-SA 4.0** per mirrors; one copy states a different non-commercial licence **(verify)**. Research use only | — |

**Recommendation.** Use the three national DTMs for the nine existing cities, fetched only for train + validation
bounding boxes through bbox or tile services. Use 30 m products only as a declared sensitivity arm.

## 3. Phase 1 audit (OSM only; `scripts/m12_audit_terrain.py` → `outputs/m12/audits/`)

565 train and 19 validation tiles. Test tiles were never read.

| | Train | Validation |
|---|---|---|
| Tiles with ≥ 1% water (polygons; sea excluded) | 81 | 2 |
| Tiles with ≥ 10% water | 28 | 2 |
| Tiles with a river polygon / a lake / coastline | 84 / 54 / 23 | 2 / 0 / 1 |
| Tiles with any street bridge / a bridge over water | 185 / 64 | 9 / 3 |
| Street–water crossings: bridge-tagged / tunnel / untagged | 269 / 0 / 30 | 32 / 0 / 0 |
| Bridge share of street length; share of bridge length over water | 2.75%; 24% | 5.1%; 50% |
| Tunnel share of street length | 0.93% | 0.80% |
| `incline` tag, 90th percentile of per-tile street-length share | 0.0% | 0.0% |

* **Crossing labels are mostly present.**
  * About 90% of street–water crossings (269 of 299) carry a bridge tag.
  * Untagged crossings are excluded or kept as "unknown", never counted as ground-level.
  * Waterways mapped underground (culverts, negative `layer`) are excluded. Before this filter, Barcelona showed
    242 spurious "crossings".
* **Water is concentrated in Japan.**
  * Tokyo, Osaka and Nagoya hold 57 of the 81 wet training tiles.
  * Spain has essentially none (Barcelona 1, Madrid 1, Valencia 4).
* **76% of bridge length crosses something other than water** (roads, railways, gaps). This gives OSM labels for
  grade separation and viaduct-like structures.
* **The current street graph is planar.** `road_graph.build_graph` merges every crossing within `MERGE_M` = 8 m, so
  these overpasses currently become false junctions. A 3D-consistent graph needs `bridge`, `tunnel` and `layer`
  kept through graph building (Phase 2).
* **Validation is too small for terrain or water claims.** Only 2–4 of the 19 validation tiles contain any water,
  and their relief is unknown until a DEM is available. Terrain-stratified validation therefore needs new held-out
  regions (§4).
* **Not measurable without a DEM:** elevation range, median and 90th-percentile slope, street-grade distributions.
  These are the first Phase 2 measurements after approval.

## 4. Expansion (proposal only; needs approval)

**Selection rule**
* Rank candidate areas by measured terrain diversity: relief, the slope distribution of built-up cells, water share
  and crossings.
* Compute this from the DEM and OSM *before* choosing; do not choose by reputation.
* Hold out whole cities for out-of-city evaluation.

| Candidate | OSM available locally | DEM route | Note |
|---|---|---|---|
| Kobe, Kyoto | Kansai extract (cached) | GSI | Same country as the training data; Kobe is a backup city in the M2 config |
| Yokohama | Kantō extract (cached) | GSI | Backup city in the M2 config |
| San Diego | SoCal extract (cached) | 3DEP | Backup city in the M2 config |
| Seoul, Busan | City clips (cached) | Korea: unknown | Excluded in M2 for building completeness; only streets are needed here |
| San Francisco | **Not cached** (NorCal extract = new download) | 3DEP | Needs an OSM download |
| Rio de Janeiro | **Not cached** (Brazil extract = new download) | Unknown (FABDEM fallback) | Needs OSM and DEM downloads |

The new evaluation regions must be at least 1,536 m from every training tile, the existing gap, and the existing
test split stays untouched.
