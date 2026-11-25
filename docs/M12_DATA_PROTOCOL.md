# Milestone 12 data protocol: elevation, water and crossings

**Status:** Phase 2 (2026-11-04).
* **Downloaded** (approved): DEMs for the train + validation tiles of **Spain and the US** (§5).
* **Japan is NOT downloaded:** its licence condition is unclear (§5.1).
* No other data was downloaded, and the test split was never requested or read.
* Facts marked **(verify)** in §2 are superseded by §5 where §5 says so.

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


---

## 5. Phase 2: what was downloaded and how (2026-11-04)

### 5.1 Licence and access verification

| Source | Verified | Decision |
|---|---|---|
| **Spain, IGN MDT** via the INSPIRE WCS `servicios.idee.es/wcs-inspire/mdt` | The capabilities document itself says "AccessConstraints: CC BY 4.0 scne.es", "Fees: no se aplican condiciones". Coverage `Elevacion25830_5`: 5 m grid, EPSG:25830, origin at a pixel centre (−19450, 4865680). Heights orthometric (REDNAP; geoid EGM08-REDNAP per IGN) | **Downloaded** |
| **US, USGS 3DEP** Bare Earth dynamic ImageServer | Service JSON: mosaic `ByAttribute` on field `Best` (best available source); per-source catalogue fields include vertical datum (NAVD 88 in every record we stored), acquisition date and resolution. USGS information is public domain | **Downloaded** |
| **Japan, GSI** 基盤地図情報 DEM / 標高タイル | Site terms: PDL 1.0 with attribution, and processing must be stated. **But** survey results under the Survey Act need approval for copying or use, and the terms do not say whether research figures derived from the DEM count (prefectural guidance suggests deriving new maps can require 使用承認) | **Not downloaded. Needs your decision**, e.g. check the GSI 利用手続ナビ / ask GSI, or use FABDEM (CC BY-NC-SA, 30 m) for Japan as a declared lower-quality arm |

### 5.2 Procedure (`scripts/m12_fetch_dem.py`; resumable, one request per tile, polite rate)

* **Boxes:** each train + validation tile plus a 256 m margin. **Test tiles are filtered out before any request.**
* **Spain**
  * Requested as **ArcGIS ASCII** (millimetre precision). The WCS `image/tiff` output is int16, i.e. 1 m steps.
  * The subset is **snapped to native pixel edges** (2.5 m modulo 5). An unaligned subset is silently *resampled*
    by the server (a pilot returned dy = 5.00045 m). The script now rejects any response that is not exactly 5 m
    on the native grid.
  * Stored as float32 GeoTIFF in EPSG:25830.
* **US:** requested at 1 m in the tile's UTM CRS, bilinear, F32, no-data −9999. The catalogue records of every
  source raster intersecting the box are stored as provenance.
* **Result:** 146 Spanish tiles (5 m) and 219 US tiles (1 m), **no missing data in any tile**. About 1 GB in
  `data/raw/dem/` (git-ignored). Provenance JSON per tile.

### 5.3 Data-quality findings

1. **Terracing in the Spanish mosaic.** A large share of the 5 m pixels hold whole-metre values:

   | City | Median share of whole-metre pixels per tile |
   |---|---|
   | Valencia | 56% |
   | Barcelona | 24% |
   | Madrid | 15% |

   * In those areas the surface is terraced (1 m steps): zero slope, then 20% jumps.
   * Street grades over a 30 m edge are uncertain by about ±1/30 = 3% there.
   * US 3DEP shows no such quantisation (0% whole-metre values).
   * The likely cause is that parts of the WCS mosaic come from integer-metre sources. **(verify** with IGN; the
     newer LiDAR products MDT02 / 2nd coverage would avoid it, but they are a separate download needing approval.)
2. **Micro-relief at 4 m.** A bare-earth DTM in a city still contains embankments, underpasses, retaining walls and
   ground interpolated under buildings. The 4 m slope therefore overstates landform steepness: Chicago tiles with
   2–3 m relief reach a p90 slope of 5–8%. The audit adds a **landform slope at 32 m** (amendment in
   `scripts/m12_terrain_audit.py`).
3. **Vertical datums differ by country** (REDNAP orthometric vs NAVD 88). They were not converted, because only
   slopes and relative elevations are compared across countries.

### 5.4 Candidate expansion regions: availability (nothing downloaded)

| Region | Elevation | Roads (OSM) | Assessment |
|---|---|---|---|
| **San Francisco** | 3DEP catalogue for the city box: lidar 1 m (2010 SF topobathy, 2010 Golden Gate, 2018 wildfire lidar), NAVD 88, public domain. Same pipeline as the US cities | **Not cached** (NorCal extract = new download). Barrington-Leigh & Millard-Ball 2017: OSM > 95% complete in most North American countries (US not named individually) | Excellent DEM; needs an OSM download |
| **Seoul, Busan** | NGII national DEM through 국토정보플랫폼 / data.go.kr: **login required**; resolution and licence **not verified** | Cached clips. From M2 v1 tiles, median street length per tile is Seoul 3.9 km, Busan 3.2 km (our cities: 3.2–6.7 km). **OSM building coverage / GHSL is low** (Seoul 0.11, Busan 0.06 vs ≥ 0.5 required in M2). Road completeness has **no independent reference yet** | Unclear DEM access and licence, unverified road completeness |
| **Rio de Janeiro** | IBGE RJ25 elevation model: **photogrammetric surface model** (includes forest and buildings, ±5 m accuracy). No open bare-earth DTM found. FABDEM (30 m, non-commercial) as fallback | **Not cached** (Brazil extract = new download). BL2017 names Brazil among countries where OSM was **not** yet complete in 2016 | Weakest candidate |
| Kobe, Kyoto, Yokohama | GSI (same licence question as Japan) | Cached (Kansai / Kantō extracts) | Depends on the GSI decision |
| **San Diego; hillside parts of Los Angeles** | 3DEP (same pipeline) | **Cached** (SoCal extract) | Good: hilly, with canyons and canyon bridges, no new OSM download. Needs approval as a dataset expansion |
