# Boeing 2022: Street network models and indicators for every urban area in the world

## 1. Header

- **Citation:** Geoff Boeing. "Street Network Models and Indicators for Every Urban Area in the World." *Geographical Analysis* 54(3): 519–535.
  - Published online 9 Mar 2021; issue dated July 2022 (Crossref).
  - Usually cited as 2021 (online) or 2022 (issue). Here it is "Boeing 2022".
- **DOI:** 10.1111/gean.12281
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/2009.09106 (v3, 9 Mar 2021; its footnote says it is the preprint of the Geographical Analysis article; read in full)
  - https://doi.org/10.1111/gean.12281 (BibTeX); https://api.crossref.org/works/10.1111/gean.12281 (dates)
  - Dataverse, indicators: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/ZTFPTB, plus file 11058506 (`indicators.tab`), downloaded
  - Dataverse, metadata: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/WMPPF9, plus files 11058507–11058509 (field definitions)
  - Dataverse, node/edge lists: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DC7U0A; 10 country zips downloaded (BOL, CHL, COL, ECU, ITA, NZL, PRT, KOR, USA, VEN)
  - https://raw.githubusercontent.com/gboeing/osmnx/main/osmnx/elevation.py (`add_edge_grades` source)
- **Access level:** **FT** (preprint) plus the open data (CC0).
  - Page numbers are preprint pages; PDF page = printed page.
  - The published Wiley version was not opened.

## 2. BibTeX (Crossref via doi.org)

```bibtex
@article{boeing2022street,
  title   = {Street Network Models and Indicators for Every Urban Area in the World},
  author  = {Boeing, Geoff},
  journal = {Geographical Analysis},
  volume  = {54},
  number  = {3},
  pages   = {519--535},
  year    = {2022},
  doi     = {10.1111/gean.12281}
}
@misc{boeing_dataverse_indicators,
  title     = {Global Urban Street Networks Indicators},
  author    = {Boeing, Geoff},
  publisher = {Harvard Dataverse},
  year      = {2020},
  note      = {Version 3.1, released 2026-05-01, CC0},
  doi       = {10.7910/DVN/ZTFPTB}
}
```

## 3. What it measures

The paper builds drivable OSMnx street graphs for every GHSL urban centre (8,914 urban areas, 178 countries). It attaches node elevations and edge grades, and computes a table of urban-area indicators:
- circuity and straightness;
- elevation statistics;
- **grade_mean and grade_median**;
- intersection counts (raw, geometrically cleaned at 10 m, topologically cleaned at 10 m);
- k_avg;
- segment lengths;
- orientation entropy and order;
- node-type shares;
- clustering;
- PageRank.

## 4. Exact metric definitions

### Elevation and grade (the reason for this note)

- **Node elevation** (pp. 6–7) comes from ASTER v2 (1 arc-second, "roughly 30 meters") or SRTM (3 arc-second, "roughly 90 meters", CGIAR-processed). Per node, the value closer to the Google Maps Elevation API value is kept; Google is used for validation only.
- **Edge grade.** OSMnx `add_edge_grades` (source code read): **grade = (elev(v) − elev(u)) / edge length**. This is rise over run, using the length along the edge, and `grade_abs` is its absolute value.
  - The grade is a **node-to-node average over a whole simplified edge** (intersection to intersection).
  - The profile between the two end nodes is never sampled.
  - Short edges amplify DEM noise.
- **Urban-area indicators** (Table 1, p. 8):
  - `grade_mean` = "Mean absolute street grade (incline)";
  - `grade_median` = "Median absolute street grade (incline)".
  - My check: recomputing from the v3.1 edge lists reproduces both to 4 decimals. They are **unweighted means/medians of grade_abs over undirected edges**, not length-weighted (see section 6).
- **No share-above-threshold, percentile or length-weighted grade indicator is published.**

### Other indicators

- **circuity:** "the graph's ratio of street lengths to straight-line distances between adjacent nodes" (edge sinuosity, p. 7).
- **intersect_count:** nodes with >2 incident edges (undirected).
- **intersect_count_clean:** intersections merged geometrically within 10 m buffers.
- **intersect_count_clean_topo:** merged within 10 m topologically (p. 9).
- **Orientation entropy and order** follow Boeing 2019.

## 5. Data

- GHSL Urban Centre Database 2019a. Areas are kept if marked "true positive", if they have ≥1 km² built-up area, and if they contain ≥3 drivable nodes.
- OSMnx drivable network, "excluding service roads like alleyways or parking lot circulation" (p. 5). All components are retained.
- 160 M raw nodes simplify to 37 M nodes and 53 M edges.
- **Current open data differ from the paper.** Dataverse v3.1 (released 2026-05-01) holds **10,351** urban areas, 189 country codes and UN SDG regions. The paper has 8,914 areas, 178 countries and GHSL subregions. The newer workflow's boundary source and DEM are not documented in the paper. The node metadata still says "from ASTER or SRTM".

## 6. Key empirical numbers

### Reported in the paper

- Population-weighted mean `grade_mean` by world subregion, as a percent (Table 2, p. 11):
  - highest: Australia/New Zealand **4.47**, Southern Europe **4.09**, Central America **4.06**, South America **3.85**;
  - lowest: South-Central Asia **1.97**, Eastern Asia **2.14**;
  - Northern America **2.50**.
- Elevation validation across 37 M nodes (Table 3, p. 14):
  - ASTER−SRTM mean difference −0.19 m (SD 7.83 m);
  - the median node's selected elevation differs from Google by 0.32 m.
- Circuity percent (Table 2): from 3.66 (Northern Africa) to 8.59 (Southern Africa).
- Intersection density: 39 to 284 per km² of built-up area.

### Computed by us from the open data (v3.1; not in the paper; script in the session scratchpad)

Distribution of `grade_mean` (unweighted mean |grade| per urban area) across 10,351 urban areas:

| Statistic | Value |
|---|---|
| p5 / p25 / **median** / p75 / p95 / p99 | 1.14% / 1.94% / **2.69%** / 3.68% / 5.83% / 7.84% |
| max | 18.8% (Taxco, Mexico), then Aizawl, India 16.1% |
| share of urban areas with grade_mean > 5% / > 8% / > 10% | 9.1% / 0.93% / 0.19% |
| `grade_median` p50 / p95 | 1.67% / 3.75% |
| population-weighted mean of grade_mean | 3.41% |

Spearman correlations across urban areas, at the urban-area scale:
- grade_mean vs elev_iqr: 0.63;
- grade_mean vs orientation_entropy: 0.43;
- grade_mean vs circuity: 0.19;
- grade_mean vs prop_deadend: −0.02.

**Per-edge grades in selected urban areas.** These are length-weighted over undirected edges from the v3.1 edge lists. An "urban area" is the whole GHSL urban centre, so the San Francisco area includes the flat Bay Area.

| Urban area | km | p50 | p90 | >5% | >10% | >15% | >20% | >15% (edges ≥ 50 m) |
|---|---|---|---|---|---|---|---|---|
| Manizales (COL) | 657 | 7.1% | 19.6% | 61.6% | 34.1% | 18.3% | 9.6% | 14.8% |
| Funchal (PRT) | 483 | 7.4% | 18.7% | 64.7% | 36.8% | 17.1% | 7.8% | 15.6% |
| Medellín (COL) | 3,128 | 5.3% | 16.6% | 51.1% | 26.8% | 12.7% | 5.9% | 11.5% |
| La Paz (BOL) | 6,010 | 4.2% | 13.8% | 43.4% | 18.4% | 8.2% | 4.0% | 6.9% |
| Valparaíso (CHL) | 2,200 | 3.9% | 13.1% | 40.3% | 17.0% | 7.1% | 2.9% | 6.0% |
| Quito (ECU) | 5,481 | 4.2% | 13.0% | 43.3% | 17.0% | 6.9% | 3.0% | 5.5% |
| Caracas (VEN) | 3,020 | 4.3% | 12.7% | 44.2% | 17.2% | 6.8% | 2.9% | 5.8% |
| Wellington (NZL) | 448 | 3.5% | 11.6% | 38.8% | 14.7% | 4.7% | 1.8% | 4.0% |
| Genoa (ITA) | 1,043 | 3.8% | 11.2% | 39.2% | 12.8% | 4.4% | 1.7% | 2.8% |
| Lisbon (PRT) | 5,457 | 3.0% | 10.3% | 32.0% | 10.5% | 3.9% | 1.7% | 1.8% |
| Dunedin (NZL) | 408 | 3.3% | 11.5% | 37.4% | 14.4% | 3.8% | 1.2% | 3.3% |
| Busan (KOR) | 2,963 | 2.0% | 9.8% | 24.7% | 9.6% | 3.5% | 1.2% | 2.8% |
| Pittsburgh (USA) | 3,047 | 3.1% | 9.6% | 31.8% | 8.9% | 2.2% | 0.6% | 1.8% |
| Seoul (KOR) | 23,110 | 1.7% | 7.5% | 18.4% | 5.6% | 2.0% | 0.8% | 1.1% |
| Santa Barbara (USA) | 415 | 2.0% | 7.4% | 21.2% | 3.5% | 0.7% | 0.3% | 0.5% |
| San Francisco (USA, whole urban centre) | 18,735 | 1.3% | 6.0% | 13.1% | 3.4% | 0.9% | 0.3% | 0.6% |
| Chicago (USA, flat) | 21,502 | 0.9% | 3.0% | 2.6% | 0.4% | 0.2% | 0.1% | 0.1% |
| Sacramento (USA, flat) | 8,180 | 0.8% | 2.8% | 2.4% | 0.2% | 0.1% | 0.0% | 0.0% |

Reading the table:
- Flat Chicago and Sacramento still show about 2.5% of length above 5% and 0.1–0.2% above 15%. That is the **DEM-noise and structure floor** of this pipeline: overpasses and bridges get ground elevations, and ASTER/SRTM carry errors.
- Excluding edges shorter than 50 m lowers the >15% share by about 9–54% (relative) in the hilly cities above: Funchal −9%, Lisbon −54%, Seoul −45%.

## 7. Verbatim quotes

- p. 5: "is parameterized to retrieve all public drivable streets, excluding service roads like alleyways or parking lot circulation."
- p. 7: "It selects for each node whichever of the ASTER or SRTM elevation values has the least absolute difference from the corresponding Google value, as a validation reference point. Finally it calculates the grade (i.e., incline) of each edge"
- p. 7: "The grade_mean and grade_median fields represent the calculated mean and median street grade absolute values."
- p. 7: "The intersect_count_clean indicator is calculated by merging intersections within 10 meter buffers of each other geometrically (i.e., 10 meter Euclidean radii) before counting them."
- p. 12: "The weighted-average urban areas in Australia/New Zealand (4.5%) and Southern Europe (4.1%) have the highest average street grades, indicating cities built on hillier terrain, whereas Eastern (2.1%) and South-Central Asia (2.0%) have the lowest, indicating cities built on flatter land."
- OSMnx source (`elevation.py`, `add_edge_grades` docstring): "Vectorized function to calculate the directed grade (i.e., rise over run) for each edge in the graph"

## 8. How UrbanMorph could use it

1. **The only global open reference for street grades, used with care.** Its grades are node-to-node averages over whole edges on 30–90 m DEMs.
   - UrbanMorph measures 10 m sub-spans on a 4 m DSM proxy (σ = 16 m low-pass).
   - Our share-steeper-than-15% will therefore read **higher** than Boeing-style grades on the same street.
   - To compare, either recompute our real tiles node-to-node on a resampled 30 m DEM, or recompute the reference cities on our pipeline. **Do not compare absolute shares across pipelines.**
2. **Realistic target band for G (gentle grades).**
   - Even the hilliest large urban areas (Manizales, Funchal) keep at least about 81% of street length at or below 15% (Boeing pipeline).
   - Typical hilly cities (Lisbon, Wellington, Genoa, Pittsburgh) keep about 95–98%.
   - A generator producing much more length above 15% than its own real reference tiles is unrealistic.
3. **No contour or terrain-slope variable.** The dataset has street grades but no terrain slope under each street, so **it cannot give a reference for the contour index** (street grade / terrain slope). That needs our own pairing of street grade with DEM slope.
4. **Consolidation.** Its 10 m topological intersection count gives a standard to match when reporting junction density. See [boeing2025_consolidation.md](boeing2025_consolidation.md).
5. **Not verified:** the DEM used in Dataverse v3.x; whether grade_mean in v3.1 is computed exactly as in the 2022 paper (the reproduction above matches the edge lists, not the paper's methods).
