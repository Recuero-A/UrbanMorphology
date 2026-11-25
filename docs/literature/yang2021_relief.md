# Yang et al. 2021: relief degree of land surface and street-network complexity in China

## 1. Header

- **Citation:** Nai Yang, Le Jiang, Yi Chao, Yang Li, Pengcheng Liu. "Influence of Relief Degree of Land Surface on Street Network Complexity in China." *ISPRS International Journal of Geo-Information* 10(10): 705 (Oct 2021). Open access, CC BY.
- **DOI:** 10.3390/ijgi10100705
- **Correction:** *ISPRS Int. J. Geo-Inf.* 11(1): 52 (2022), DOI 10.3390/ijgi11010052. It replaces Figure 2C only; "the scientific conclusions are unaffected".
- **URLs opened (2026-11-25):**
  - https://www.mdpi.com/2220-9964/10/10/705 and `/htm`, `/pdf`: **403**, both by curl and WebFetch
  - https://api.openalex.org/works/doi:10.3390/ijgi10100705 (abstract, OA status)
  - https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/ijgi10100705
  - https://mdpi-res.com/d_attachment/ijgi/ijgi-10-00705/article_deploy/ijgi-10-00705.pdf (publisher CDN copy of the article PDF; **read in full**; Eq. 1 checked on a rendered image)
  - https://mdpi-res.com/d_attachment/ijgi/ijgi-11-00052/article_deploy/ijgi-11-00052.pdf (correction, read)
- **Access level:** **FT**. Page numbers are journal pages ("x of 20").

## 2. BibTeX (Crossref via doi.org)

```bibtex
@article{yang2021influence,
  title   = {Influence of Relief Degree of Land Surface on Street Network Complexity in China},
  author  = {Yang, Nai and Jiang, Le and Chao, Yi and Li, Yang and Liu, Pengcheng},
  journal = {ISPRS International Journal of Geo-Information},
  volume  = {10}, number = {10}, pages = {705}, year = {2021},
  doi     = {10.3390/ijgi10100705}
}
```

## 3. What it measures

The paper correlates a regional terrain index (RDLS) with city-level OSMnx street-complexity metrics across 391 Chinese administrative units. Both sides are **city aggregates**; nothing is measured per street.

## 4. Exact metric definitions

- **RDLS** (Eq. 1, p. 3): RDLS = ALT/1000 + (Max(H) − Min(H)) × (1 − P(A)/A) / 500.
  - ALT is the mean altitude; Max(H) and Min(H) are the extreme altitudes; P(A) is the flat area; A is the total area.
  - How "flat area" is thresholded is **not stated** in the text I read: **not verified**.
- **Grain:** average street length in the undirected graph (LS, m).
- **Connectedness:** "average streets per node" (DS). Also the proportions of dead ends (PDE), 3-way (P3W) and 4-way (P4W) nodes.
- **Circuity** (Eq. 2, p. 4): C = D_net / D_gc for each of **50,000 simulated random routes** (node pairs) per city. The city value is the average. This is a **route circuity**, unlike Boeing's edge-level ς.
- **Orientation entropy** (Eq. 3, p. 5): Boeing 2019 convention (36 × 10° bins, −5° shift, nats, unweighted).
- **Statistic:** Pearson r between RDLS and each metric, with significance tested in SPSS (Table 3, p. 10).

## 5. Data

- 30 m DEM from the Geospatial Data Cloud (gscloud.cn).
- OSM via OSMnx, 391 units: 4 municipalities, 2 SARs, 293 prefecture-level cities, 30 autonomous prefectures and others, including 3 cities and 13 counties in Taiwan (p. 3).
- The network type (drive, walk or all) is not stated in the methods: **not verified**.

## 6. Key empirical numbers

- RDLS ranges from 0 to 17.7 at municipal level (p. 6).
- Pearson r with RDLS (Table 3, p. 10):

| Metric | r with RDLS |
|---|---|
| circuity C | **0.612** |
| average streets per node DS | −0.549 |
| P4W | −0.499 |
| average street length LS | 0.493 |
| dead-end share PDE | 0.444 |
| orientation entropy Ho | 0.331 |
| P3W | −0.012 (not significant) |

- Example cities (Table 1, p. 6):
  - Chengde (RDLS 4.72): C 1.15, PDE 0.28, P4W 0.07;
  - Beijing (3.4): C 1.09;
  - Tianjin (0.25): C 1.05.

## 7. Verbatim quotes

- p. 3 (Eq. 1 definitions): "where RDLS is the relief degree of land surface of a region, ALT is the average altitude within the region, Max(H) is the maximum altitude within the region, Min(H) is the minimum altitude within the region, P(A) is the flat area of the region, and A is the total area of the region."
- p. 4: "In our study, 50,000 pairs of nodes with starting and ending points were simulated as random routes for each city and network type."
- p. 9: "RDLS exhibits the strongest correlation with street network circuity, with a correlation coefficient of 0.612."
- p. 10: "The correlation between RDLS and dead-end proportion is 0.444, demonstrating that more dead-end nodes exist in places with greater topographic relief."
- p. 10: "By contrast, four-way-intersection proportion demonstrates a larger and negative correlation with RDLS (−0.499)."

## 8. How UrbanMorph could use it

- **A directional prior for terrain-conditioned realism, not a numeric target.** On rougher terrain real networks are more circuitous, have more dead ends, fewer 4-way junctions and longer streets.
  - A generator that becomes *more* grid-like or less circuitous on steeper tiles than on flat ones goes against this.
  - Test it as a within-model slope response: regress tile metrics on tile relief for real and for generated tiles, and compare the slopes. That is close to what S_T does.
- **Ecological level.** The correlations are across cities with RDLS over whole administrative units, so they say nothing about street-level adaptation (grade versus slope, contour following).
- **Dead ends rise with relief in real cities.** A blanket "fewer dead ends is better" target is wrong on steep land. Compare dead-end ratios with real tiles of similar relief (see M18 measurement notes).
