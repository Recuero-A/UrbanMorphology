# M16: terrain sources for the real-terrain benchmark (KAIST, Seoul, Madrid, Guadarrama foothills)

Status: sources checked and downloaded on 2026-11-20 UTC (2026-11-21 KST). Everything is under `data/raw/m16/`.
That folder is git-ignored by the `data/raw/*` rule. Each raster has a provenance JSON next to it.
"not verified" means the value could not be confirmed from a primary source during this check.

## 0. Summary

| Site | Proposed extent | Terrain source chosen | Grid | Surface type | Licence |
|---|---|---|---|---|---|
| KAIST Daejeon main campus | OSM way 73679850 (1.139 km²). Domain: 3 × 3 windows of 512 m (1536 m) | Copernicus GLO-30 (fallback) | 1″ (~31 m N–S × ~25 m E–W) | **DSM** | Copernicus DEM licence (free, attribution) |
| Seoul (Haebangchon, south slope of Namsan) | 512 m window centred at 37.5425 N, 126.9875 E | Copernicus GLO-30 (fallback) | 1″ | **DSM** | as above |
| Madrid (Lavapiés / Embajadores) | 512 m window centred at 40.4087 N, 3.7020 W | IGN MDT05 (WCS) | 5 m | DTM (LiDAR ground class) | CC BY 4.0 (scne.es) |
| Guadarrama foothills (San Lorenzo de El Escorial, upper town) | 512 m window centred at 40.5928 N, 4.1490 W | IGN MDT05 (WCS) | 5 m | DTM | CC BY 4.0 (scne.es) |

**Korea has no open high-resolution DTM in this benchmark.** The NGII "공개DEM" requires a logged-in account (§2). Both
Korean sites therefore rely on the 30 m GLO-30 **surface** model. Read §5 before comparing Korean and Spanish results.

## 1. KAIST Daejeon main campus

**Map PDF.** `data/raw/m16/kaist/Campus_Map2025.pdf` is downloaded from https://www.kaist.ac.kr/Upload/map/Campus_Map2025.pdf.
* File details: 2,573,652 B, sha256 in `Campus_Map2025.provenance.json`.
* PDF metadata: title "20250305 카이스트맵 - campusmap-명칭수정", created 2025-03-05.
* It is the **Daejeon main campus**: E/W/N building codes, main gate (정문) on the south side, east gate (동문), the 어은동산 and 구성동산 hills, and the stadium.
* It does **not** show the Munji, Hwaam or Seoul campuses.
* It is an illustrated oblique drawing. It has no coordinates and no scale bar. The north arrow points to the upper right, so the map is rotated about 35°.
* It cannot define a boundary by itself. It was used only as a visual check.

**Boundary (OSM).** Overpass (overpass-api.de) returned "502, at capacity" at access time. The polygon was taken from the live
OSM API instead.
* Found via Nominatim ("KAIST Daejeon"; first hit):
  * **way 73679850**, `amenity=university`, `name=한국과학기술원`, address 291 Daehak-ro, Yuseong-gu;
  * version 54, timestamp 2026-06-14T19:27:12Z, changeset 184108233;
  * it is not a member of any relation.
* Excluded: way 469034109 (한국과학기술원 문지캠퍼스, Munji) and the Hwaam dormitory landuse.
* Files:
  * `data/raw/m16/kaist/osm_way73679850_full.json` (raw API response);
  * `kaist_main_campus_osm.geojson` (EPSG:4326, with the OSM id, version and timestamp);
  * `kaist_main_campus_osm.provenance.json`.

| Quantity | Value |
|---|---|
| bbox (lon/lat) | 127.3551293, 36.3641422 – 127.3700799, 36.3761098 |
| centroid | 36.370653 N, 127.361915 E |
| bbox centre | 36.370164 N, 127.362599 E |
| area (geodesic) | **1.139 km²**; perimeter 4.52 km |
| extent (UTM 52N, EPSG:32652) | 1340 m × 1331 m; bbox 352441.8, 4025585.7 – 353781.6, 4026916.7 |
| GLO-30 surface height inside the campus | 32.6 – 88.5 m (median 51.5 m). This is a DSM, so the wooded hills include the canopy |

**Match with the PDF.** The OSM outline was rotated −35° (a plot made outside the repo) and compared with the PDF. Both show:
* the straight southern edge with the main gate;
* the east side along Gwahak-ro (과학로);
* the western bulge (W dormitories, 어은동산);
* the northern apex.

The match is visual only, because the PDF is not georeferenced.

The campus boundary follows **roads, not the stream**:
* Daehak-ro (대학로) on the south and south-east, about 10 m from the polygon edge;
* Gwahak-ro on the east and north-east.

The streams are outside the campus:
* Gapcheon (갑천) is ~117 m beyond Daehak-ro to the south-east;
* Yuseongcheon (유성천) is ~113 m to the south;
* Tandongcheon (탄동천) is ~47 m to the north-north-east.

**Suggested modelling extent.**
* **Domain:** 1536 m × 1536 m, i.e. 3 × 3 windows of the standard 512 m.
  * Centred on the bbox centre (UTM 52N 353111.7, 4026251.2).
  * Domain bbox UTM 352343.7, 4025483.2 – 353879.7, 4027019.2; lon/lat 127.353895, 36.363126 – 127.371301, 36.377202.
  * Justification: the campus is 1.34 km across, so 1536 m is the smallest multiple of 512 m that contains it. That leaves a margin of about 100 m on each side, which captures the bounding roads (Daehak-ro, Gwahak-ro) and the Gapcheon bank.
* **DEM request:** the domain plus a 256 m margin, the same `MARGIN_M` as `scripts/m12_fetch_dem.py`. That gives lon/lat 127.350994, 36.360779 – 127.374202, 36.379548.
  * The margin keeps slope and filter kernels valid at the domain edge.
  * At 30 m it adds about 8 GLO-30 pixels on each side.
* **Leakage check:** the domain is ≥ 1.1 km from the nearest M15 Daejeon development window (KOR_DJN_e685_n7865), so there is no overlap.

**Terrain.** The only source available is `data/raw/m16/glo30/Copernicus_DSM_COG_10_N36_00_E127_00_DEM.tif` (DSM). The NGII
public DEM index lists sheet **36710 "대전"** (years 2014, 2021–2025), but downloading it needs a login (§2).

## 2. Korea elevation sources (KAIST and Seoul)

**data.go.kr 15059920 "국토교통부 국토지리정보원_DEM"** (fetched without login):
* registered 2024-09-24, modified 2025-06-17;
* format "IMG";
* cost: free; reuse scope "이용허락범위 제한 없음" (no restriction on permitted use);
* department: 국토조사과.

The page holds no file. It points to `http://map.ngii.go.kr/ms/map/NlipMap.do?tabGb=total` and states:
"다운로드를 위해서는 로그인이 필요합니다" (login is required to download). It also requires NGII's large-file transfer software.

The page does not state the resolution, CRS or vertical datum: **not verified**.

**map.ngii.go.kr (국토정보플랫폼), 공개DEM:**
* **The tile index is public.** `POST /ms/map/demQuery.do` with an EPSG:5179 bbox returned:
  * the 1:50,000 sheets **36710 대전** and **37608 서울**;
  * years 2014, 2021, 2022, 2023, 2024 and 2025.
* **Download needs a login.** The site's `download.js` calls `/mn/loginCheck.do` before `/pd/shbtManage/shbtInsertTritVidoDem.do`. The request also carries birth date (`brthdy`), purpose (`purchsPurps`) and detail fields.
* Result: this requires personal registration, so we **stopped** here as instructed. Nothing was downloaded.
* The grid spacing (often quoted as 5 m) is **not verified** from any NGII document reached.
* The NGII aerial-survey DEM tiles (1:5000 sheet metadata, data.go.kr 15067632) are distributed through the same application system. They are **not open downloads**.

**Other open Korean DEMs.** A data.go.kr search for "수치표고모델" and "DEM" (file datasets) found:
* no directly downloadable nationwide high-resolution DEM;
* a national-parks DEM (15003715), which does not cover our windows;
* metadata tables only.

**Fallback used: Copernicus GLO-30** (DSM, 30 m, EGM2008). It has free access with no account. It is in the repo since M15.

Not used: **FABDEM**, a 30 m bare-earth estimate made from GLO-30. Its licence is non-commercial (CC BY-NC-SA 4.0 per mirrors, flagged "verify" in `docs/M12_DATA_PROTOCOL.md`) and it ships in 10° archives. It is a candidate for a DSM-vs-bare-earth sensitivity check on the Korean sites. It was not downloaded or checked here.

## 3. Spain: IGN MDT05

**What the earlier milestones used.** M12–M15 obtained MDT05 **not** from the CNIG download centre but from the public IGN
INSPIRE WCS. The script is `scripts/m12_fetch_dem.py`:
* service `https://servicios.idee.es/wcs-inspire/mdt`, coverage `Elevacion25830_5`;
* request: GetCoverage in ArcGrid format, with the subset snapped to the native 5 m pixel edges (origin −19450, 4865680) so that no silent resampling occurs.

M16 reuses exactly that method. GetCapabilities states:
* `AccessConstraints`: "CC BY 4.0 scne.es";
* `Fees`: "No se aplican condiciones" (no conditions apply).

The service needs no login.

**CNIG product page (MDT05 – 1ª cobertura):**
* grid: 5 m, interpolated from the ground class of PNOA LiDAR (1st coverage);
* CRS: ETRS89, UTM in its own zone (zone 30 extended for zones 29 and 31);
* heights: **orthometric**;
* download unit: MTN50 sheets, as COG;
* licence: the IGN use licence, "compatible con CC-BY 4.0";
* attribution: "MDT05-cob1 2008-2015 CC-BY 4.0 scne.es".

Download from the CNIG centre is gated by **reCAPTCHA v3** (`preAutorizarDescarga`) and an anti-bot form. It needs no login, but it is not scriptable without bypassing that gate, so it was **not used**.

The EEA CIS² record (https://cis2.eea.europa.eu/data/912/) gives:
* the same product: "Digital Terrain Model Spain (5 m)", IGN, bare-earth from PNOA LiDAR;
* temporal coverage 2008–2015;
* data policy "Freely accessible";
* a link to the CNIG page.

**Caveat: which coverage the WCS serves is unknown.** It is a mosaic, and which LiDAR coverage feeds each pixel is not exposed (not verified). MDT02 (2 m) is not offered by this WCS, which tops out at 5 m (`Elevacion25830_5`).

**MTN sheets.** These come from IGN WMS `cuadriculas`, GetFeatureInfo at the window corners and centre plus the 256 m margin, grid version 20260303:
* Madrid window: **MTN50 0559 "Madrid"** (MTN25 0559-3 Madrid Suroeste);
* San Lorenzo de El Escorial window: **MTN50 0533 "San Lorenzo de El Escorial"** (MTN25 0533-1);
* the Cercedilla alternative: **MTN50 0508 "Cercedilla"** (0508-3).

None of the windows crosses a sheet edge.

**Downloaded windows.** Each is the 512 m window plus a 256 m margin, as EPSG:25830 float32 GeoTIFF at 5 m (≈210 × 210 px, 80–130 kB). The JSON next to each holds the request URL, sha256 and access time.
* `data/raw/m16/madrid/MAD_LAVAPIES_mdt05.tif`: **chosen**. sha256 6ac84e0f…8a20.
* `data/raw/m16/madrid/MAD_CHAMBERI_mdt05.tif`: alternative and DSM-vs-DTM control.
* `data/raw/m16/guadarrama/GUA_ESCORIAL_NW_mdt05.tif`: **chosen**. sha256 ddcf6468…5730.
* Other screened candidates: GUA_ESCORIAL, GUA_ESCORIAL_N, GUA_CERCEDILLA (+ _W, _S), GUA_GUADARRAMA.

**Terracing artefact.** M12 had already flagged it ("partly 1 m-terraced"). Share of pixels on flat, integer-valued plateaus in the downloaded rasters:

| Raster | Flat integer plateaus |
|---|---|
| Chamberí | 25 % |
| Lavapiés | 10 % |
| Guadarrama town | 15 % |
| Escorial NW | 4 % |
| Cercedilla | 0.4–1 % |

Grades from MDT05 in flat central Madrid therefore come in 1 m steps. Measure them over spans of ≥ 10 m, as M15 did, or smooth lightly and label the smoothing.

## 4. Window proposals and justification

Method (indicative, `data/raw/m16/window_screening.json`):
* streets: OSM API 0.6 per window, drivable/residential/pedestrian streets clipped to the 512 m UTM window, bridges and tunnels excluded;
* grade: |Δz|/Δs sampled every 10 m on MDT05 (Spain) and every 30 m on GLO-30.

| Window | Centre (lat, lon) | Street km | MDT05 share ≥10 % / ≥15 % | GLO-30 share ≥10 % / ≥15 % | Relief in window (DEM) |
|---|---|---|---|---|---|
| **MAD_LAVAPIES** | 40.4087, −3.7020 | 7.7 | 9 % / 0.5 % | 17 % / 1.8 % | 614–650 m |
| MAD_CHAMBERI | 40.4335, −3.7020 | 6.0 | 0 % / 0 % | 26 % / 5.1 % | 672–685 m |
| **GUA_ESCORIAL_NW** | 40.5928, −4.1490 | 10.7 | **32 % / 12 %** | 33 % / 13 % | 1026–1114 m |
| GUA_ESCORIAL_N | 40.5945, −4.1455 | 8.4 | 37 % / 15 % | 36 % / 15 % | 1007–1121 m |
| GUA_ESCORIAL | 40.5905, −4.1470 | 8.5 | 15 % / 4 % | 21 % / 9 % | 993–1070 m |
| GUA_CERCEDILLA | 40.7395, −4.0575 | 6.6 | 22 % / 6 % | 26 % / 7 % | 1141–1241 m |
| GUA_CERCEDILLA_W | 40.7400, −4.0610 | 5.3 | 31 % / 10 % | 31 % / 14 % | 1155–1278 m |
| GUA_GUADARRAMA | 40.6725, −4.0890 | 9.7 | 7 % / 2 % | 6 % / 1 % | 949–979 m |
| **SEL_HAEBANGCHON** | 37.5425, 126.9875 | 10.5 | – | 39 % / 23 % (DSM) | 38–99 m (DSM) |
| SEL_HUAM | 37.5485, 126.9785 | 8.3 | – | 19 % / 3 % (DSM) | 23–82 m |
| SEL_BUKCHON | 37.5820, 126.9845 | 6.5 | – | 24 % / 8 % (DSM) | 39–78 m |
| SEL_IHWA | 37.5785, 127.0065 | 6.5 | – | 18 % / 7 % (DSM) | 32–47 m |

Window bounding boxes (lon/lat, 512 m in UTM):

| Window | UTM zone | bbox (lon/lat) |
|---|---|---|
| Lavapiés | 30N | −3.704993, 40.406375 – −3.699007, 40.411024 |
| Escorial NW | 30N | −4.151985, 40.590464 – −4.146015, 40.595136 |
| Haebangchon | 52N | 126.984666, 37.540145 – 126.990334, 37.544855 |

Full UTM bounds are in `window_screening.json`.

**Madrid: Lavapiés / Embajadores.**
* It is dense central fabric with an irregular historic street pattern.
* It has real but moderate relief: 36 m, falling southward (614–650 m in the window).
* 9 % of its street length is at ≥ 10 % grade on MDT05, which makes it a "mixed" site.
* Chamberí is flatter (0 % ≥ 10 %). It is kept only as a DSM-vs-DTM control.
* Not held out: the window does not overlap any M12 Madrid training tile, but the nearest one (ESP_MAD_e859_n8735) is only 269 m away. Madrid is an M12 **training city** (70 tiles), so this site tests in-distribution transfer, not a held-out city.

**Foothill: San Lorenzo de El Escorial, upper town (MTN50 0533).**
* It is a compact, dense town built on the lower slope of Monte Abantos, centred north-west of the monastery.
* The monastery block lies just south of the window.
* 10.7 km of streets, of which **32 % are at ≥ 10 % grade** and 12 % at ≥ 15 %. Relief is 88 m (MDT05 DTM).
* Its MDT05 has little terracing (4 %), and GLO-30 agrees there (33 %).
* Cercedilla has comparable slopes but half the street length, so its network is sparser and more dispersed.
* Guadarrama town is too flat (7 %).
* Collado Villalba was not screened. It lies on the GLO-30 tile edge at 4° W and is mostly flatter (not verified).
* **Selection note:** this window was picked *using street grades*, as the task asked. It is a deliberate stress-test site, not a random draw, and it should be reported as such.

**Seoul: Haebangchon (Yongsan 2-ga-dong), south slope of Namsan.**
* It is dense small-lot housing on a steep hillside, with the steepest screened streets (39 % ≥ 10 % on GLO-30).
* Bukchon and Huam-dong are milder alternatives. Ihwa has steep alleys but little relief at 30 m.
* **Caution 1 (DSM):** these grades come from a 30 m DSM. Chamberí shows how a DSM inflates street grades in dense blocks: 0 % ≥ 10 % on the DTM against 26 % on GLO-30. So the Seoul ranking is indicative only.
* **Caution 2 (held out):** M15 (`docs/M15_PROTOCOL.md` §5) declared **Seoul a reserved held-out metro**, "not built or evaluated without authorisation". This screening read OSM streets and GLO-30 for four Seoul windows. That is terrain and street data only; no generator was run. The project owner should confirm that Seoul is intentionally released from held-out status for M16.

## 5. Caveats that must travel with the benchmark

1. **Label the surface type on every result.**
   * GLO-30 is a **DSM**: buildings, trees and bridges are included.
   * MDT05 is a **DTM** (LiDAR ground class).
   * KAIST and Seoul run on GLO-30, and Madrid and Escorial on MDT05, so cross-site comparisons mix surface types. Report a GLO-30 run for the Spanish sites as well (both sources are downloaded) so that a like-for-like comparison exists.
2. **Resampling cannot restore detail.**
   * Bilinear resampling of GLO-30 to the 2 m model grid gives a smooth 30 m surface sampled at 2 m. It contains no 2 m information.
   * Slopes computed on it are slopes of a 30 m surface and are underestimated on short steep streets.
   * In dense blocks the DSM also adds building-driven bumps.
   * Never describe these grids as "2 m terrain" without the source resolution.
3. **Street imprint in urban DTMs.**
   * LiDAR-derived DTMs in cities interpolate the ground class, and that class is dominated by streets and courtyards. The DTM therefore follows street beds, cuttings and retaining walls.
   * A generator conditioned on such a DTM can "see" the real street layout in the terrain. This is the leakage risk already flagged in M15.
   * MDT05 at 5 m carries a weaker imprint than 1–2 m products, but not zero.
   * GLO-30 has the opposite problem: building blocks raise the surface and streets show as troughs.
   * Report a street-erase or smoothing control (as in `scripts/m13_street_erase.py`) for the Spanish sites.
4. **MDT05 terracing** (§3). The WCS mosaic does not expose which coverage, or year, feeds each pixel.
5. **Vertical datums.**
   * GLO-30: EGM2008.
   * MDT05: Spanish orthometric heights; the geoid model is not stated by the WCS (M12 recorded EGM08-REDNAP as "verify").
   * The two differ by at most decimetres to a metre in the windows (not verified). This matters only for absolute heights, not grades.

## 6. Attribution text

* **MDT05:** "MDT05 CC-BY 4.0 scne.es". For the 1st coverage the CNIG form is "MDT05-cob1 2008-2015 CC-BY 4.0 scne.es".
* **Copernicus GLO-30:** "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
  * The years differ between the licence PDF and the Copernicus Data Space page. Re-check against https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf before publication.
  * The AWS bucket files carry no version tag (version not verified).
* **OSM:** "© OpenStreetMap contributors, ODbL 1.0".
* **KAIST campus map PDF:** no licence stated. Reference use only; do not redistribute.

## 7. What failed

| Attempt | Result |
|---|---|
| Overpass API (overpass-api.de) | 502 "at capacity". A mirror answered later with stale data (base 2026-07-15). The live OSM API 0.6 was used instead |
| NGII 공개DEM (data.go.kr 15059920 → map.ngii.go.kr) | Login plus a personal-data form required. Stopped. Resolution not verified |
| CNIG centro de descargas MDT05 sheet download | reCAPTCHA v3 gate. Not bypassed. The public IGN WCS was used instead (same product family, CC BY 4.0) |
| Copernicus licence page (spacedata.copernicus.eu) | Connection refused. Licence wording taken from the Copernicus Data Space pages via search (see §6) |

## 8. Files (`data/raw/m16/`, git-ignored)

* **`glo30/`**: 4 GLO-30 tiles, each with a `.provenance.json`:

  | Tile | Covers | Size |
  |---|---|---|
  | N36E127 | KAIST | 49.6 MB |
  | N37E126 | Seoul | 29.9 MB |
  | N40W004 | Madrid | 40.9 MB |
  | N40W005 | Guadarrama | 40.4 MB |

  Total ≈ 161 MB.
* **`kaist/`**: `Campus_Map2025.pdf` (2.6 MB), `osm_way73679850_full.json`, `kaist_main_campus_osm.geojson`, provenance JSONs, `SOURCES.json`.
* **`madrid/`**: `MAD_LAVAPIES_mdt05.tif`, `MAD_CHAMBERI_mdt05.tif`, each with a provenance JSON.
* **`guadarrama/`**: 7 MDT05 candidate windows, each with a provenance JSON. `GUA_ESCORIAL_NW` is the chosen one.
* **`seoul/`**: `SOURCES.json` only. The terrain is the GLO-30 tile.
* **`window_screening.json`**: the street-grade screening table of §4.
