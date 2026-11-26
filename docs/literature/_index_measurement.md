# Index: measurement literature (M18 phase, 2026-11-25)

**Access levels:**
- **FT**: full text read;
- **AO**: abstract only;
- **MD**: metadata only.

Each note lists the URLs opened, verified BibTeX, verbatim quotes with pages, and the uses we could make of it. Cross-cutting outputs:
- [metrics_catalogue.md](metrics_catalogue.md): 36 metrics and a recommended primary set;
- [street_grades_real_world.md](street_grades_real_world.md): standards, limits, records, distributions and proposed thresholds.

## 1. Papers with notes

| Paper | Venue | DOI | Access | Note | What we take from it |
|---|---|---|---|---|---|
| Boeing 2019, "Urban spatial order" | Appl. Netw. Sci. 4:67 | 10.1007/s41109-019-0189-1 | FT (preprint) | [boeing2019_orientation.md](boeing2019_orientation.md) | H_o, φ conventions; edge-sinuosity "circuity"; 100-city reference table; hilly cities most circuitous |
| Boeing 2022, "Street network models and indicators for every urban area" | Geogr. Anal. 54(3):519–535 (online 2021) | 10.1111/gean.12281 | FT + open data | [boeing2022_global.md](boeing2022_global.md) | **grade_mean/grade_median** definitions (node-to-node, unweighted); subregion grades 2.0–4.5%; our computed global and per-city grade distributions |
| Boeing 2025, "Topological graph simplification … intersection miscount" | Trans. GIS 29(3):e70037 | 10.1111/tgis.70037 | FT (arXiv v4) | [boeing2025_consolidation.md](boeing2025_consolidation.md) | Topological consolidation; median 14% overcount; Seoul 8% vs Sydney 26% |
| Louf & Barthelemy 2014, "A typology of street patterns" | J. R. Soc. Interface 11:20140924 | 10.1098/rsif.2014.0924 | FT (arXiv v1) | [louf2014_blocks.md](louf2014_blocks.md) | Φ = A/A_circle; P(Φ\|A) fingerprint and distance; 10³–10⁵ m² bins |
| Barrington-Leigh & Millard-Ball 2015, "A century of sprawl in the US" | PNAS 112(27):8244–8249 | 10.1073/pnas.1504033112 | FT (PMC) | [barringtonleigh2019_sndi.md](barringtonleigh2019_sndi.md) | Degree, dead-end and 4+ shares; US trend |
| Barrington-Leigh & Millard-Ball 2019, "A global assessment of street-network sprawl" | PLOS ONE 14(11):e0223078 | 10.1371/journal.pone.0223078 | FT + S1 | same | **SNDi**; dendricity (bridge/non-cycle); banded log circuity; 20 m merging; paths included |
| Barrington-Leigh & Millard-Ball 2020, "Global trends toward urban street-network sprawl" | PNAS 117(4):1941–1950 | 10.1073/pnas.1905232116 | FT (PMC) | same | SNDi trends; footpath and service-road caveat |
| Marshall 2005, *Streets and Patterns* | Spon/Routledge (book) | 10.4324/9780203589397 | MD (TOC + publisher text) | [marshall2005_streets.md](marshall2005_streets.md) | Vocabulary only; typology definitions **n.v.** (no legitimate copy) |
| Birsak, Kelly, Para, Wonka 2022 + `twak/streetz` | arXiv 2209.00281 | 10.48550/arXiv.2209.00281 | FT + code | [birsak2022_streetz.md](birsak2022_streetz.md) | Segment circuity; transport ratio (**drops unreachable pairs**); block aspect and rectangularness; land-area densities |
| Belli & Kipf 2019, "Image-conditioned graph generation for road network extraction" | NeurIPS 2019 GRL workshop; arXiv 1910.14388 | 10.48550/arXiv.1910.14388 | FT + code | [belli2019_streetmover.md](belli2019_streetmover.md) | StreetMover (Sinkhorn OT on point clouds), for paired comparisons only |
| Yang, Jiang, Chao, Li, Liu 2021, "Influence of relief degree of land surface on street network complexity in China" | IJGI 10(10):705 (corr. 11(1):52) | 10.3390/ijgi10100705 | FT (publisher CDN) | [yang2021_relief.md](yang2021_relief.md) | RDLS vs circuity r = 0.612, dead ends 0.444, P4W −0.499 (391 units) |
| Zhou, Bao, Liu 2022, "Mapping urban forms worldwide" | IJGI 11(7):370 | 10.3390/ijgi11070370 | FT (partly) | [zhou2022_urbanforms.md](zhou2022_urbanforms.md) | Terrain-variation PC (20.6%) with grade_mean loading 0.841; grade vs orientation 0.35–0.38 |
| Gu, Wang, Lyu, Zhang, Ge 2025, "Exploring the impacts of topography on characteristics of urban road network" | Cities 160:105848 | 10.1016/j.cities.2025.105848 | **MD** | [gu2025_topography.md](gu2025_topography.md) | Likely the "Gu et al. topography" paper M18 could not find; contents n.v. |
| Li et al. 2026, RoadWeaver | arXiv 2608.11580 | 10.48550/arXiv.2608.11580 | FT (I re-read §IV-C) | [li2026_roadweaver.md](li2026_roadweaver.md) (pre-existing note, not written or edited here) | LCC, reachability, dead-end ratio, cycle ratio; definitions confirmed on paper p. 5 |

## 2. Also found, no separate note

- **Barthelemy & Boeing 2026, "Geometry of Urban Order"**, arXiv 2609.13029. **AO** (abstract via the arXiv page). "Bearings communities" across 1,025 cities. The abstract reports mean domain coherence 0.964, varying 1.4% between cities, and says global orientation order is set mainly by the number of grains. Relevant to the orientation metric: per-tile H_o in a 1 km window mostly counts grains.
- **Najem 2017**, arXiv 1704.01119. **FT** (skimmed). Few cities. It asserts that hilly cities (Portland, Beirut, San Francisco) have *lower* street-orientation entropy because of "landform constraints". This is the opposite sign to Yang 2021 and to our Dataverse correlation (+0.43). It uses a small sample and does not measure slope at street level.
- **Standards, codes and records** used for the grade note are listed in [street_grades_real_world.md](street_grades_real_world.md) and not repeated here:
  - NYSDOT HDM ch. 2; Japan Road Structure Ordinance; Spain 3.1-IC; DMRB CD 109; Manual for Streets;
  - 2021 Seattle Fire Code App. D; 2010 ADA Standards; UN R13-H (via GS 4013H); EU 1230/2012;
  - GA DOAS bus specification; Guinness 2019/2020; Pittsburgh Post-Gazette 2005; Wikipedia (secondary).

## 3. Search for empirical studies of street adaptation to terrain slope

**Question.** Is there a paper that measures, for real cities, (a) street grade against the underlying terrain slope, (b) contour following (alignment of street bearing with terrain aspect or fall line), or (c) hillside street orientation?

**Searches run (2026-11-25):**
- "street network orientation contour lines terrain slope empirical …"
- "angle between street direction and terrain aspect slope direction … contour"
- "streets gentler than terrain: comparison of street gradient with underlying terrain slope …"
- "Exploring the impacts of topography …" (title search)
- plus Crossref/OpenAlex lookups.

**Result: no paper found that measures (a) or (b) directly.**

| Found | Level | What it does | Gap |
|---|---|---|---|
| Yang et al. 2021 | city aggregate | RDLS vs circuity, dead ends, P4W, entropy | no street-level grade or alignment |
| Zhou et al. 2022 | urban-area aggregate | grade_mean in PCA; weak link to orientation | same |
| Gu et al. 2025 | n.v. | 10 topographic metrics vs road traits (search snippet) | full text not accessed |
| Boeing 2019 | city | "due largely to topography" explanation | p. 17 calls for exactly this study |
| Boeing 2022 data | edge | street grades only | no terrain slope beside the street |
| Barrington-Leigh & Millard-Ball 2019 | 1 km cell | PCA2 = circuity beyond topology, "steep topography"; Hiroshima example | qualitative |
| Najem 2017 | city | hilly cities have lower entropy | small n, no slope measure |
| Graz wheelchair OSM study; Salt Lake City grade imputation (search results) | edge | DEM vs OSM incline tags; OSM grade estimation | accessibility and data focus; not opened |

**Implication.** TAS's contour term (C) and siting term (S) have **no published real-city baseline**. That agrees with the M18 note. It is also a potential contribution: the Boeing Dataverse edge lists plus a DEM slope raster would give the first multi-city distribution of street grade / terrain slope. The edge geometries are included, but the DEM is not. A real-city contour index could be computed this way.

## 4. Corrections and additions to M18_LITERATURE §3 (for the main thread; this agent did not edit M18 files)

1. **Boeing "2021" Geographical Analysis** is vol. 54(3), July **2022** issue, published online 9 Mar 2021.
   - Its grades are **node-to-node per simplified edge** on ASTER (about 30 m) or SRTM (about 90 m).
   - The published indicators are **unweighted** edge means and medians.
   - The open data are now v3.1 (2026-05-01): 10,351 urban areas, not 8,914.
2. **Boeing 2025's global 14%** used a **per-urban-area tolerance from Atlas of Urban Expansion road widths**, not a fixed 10 m. Our own check on Dataverse v3.1 (10 m topological) gives a median of 14.6%.
3. **SNDi paradigm scores disagree** between the PLOS main text (−1.01 / 0.66 / 7.9) and the S1 table (−0.76 / 0.91 / 8.11).
4. **Yang et al. 2021 has a correction** (IJGI 11:52, 2022). It covers Figure 2C only; conclusions are unchanged.
5. **"Gu et al. on topography"** is most likely Gu, Wang, Lyu, Zhang & Ge 2025, *Cities* 160:105848 (MD only).
6. **Our S_T circuity and streetz's transport ratio both drop unreachable pairs.** M13's `circuity_all` (cap at 3.0) is the safer convention.
