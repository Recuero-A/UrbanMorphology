# Barrington-Leigh & Millard-Ball 2015, 2019, 2020: street-network sprawl and the SNDi

One note covers three papers, listed in the index as three rows. The 2019 PLOS ONE paper (with its S1 file) defines the SNDi; the other two apply it.

## 1. Header

| | 2015 | 2019 | 2020 |
|---|---|---|---|
| Citation | C. Barrington-Leigh, A. Millard-Ball. "A century of sprawl in the United States." *PNAS* 112(27): 8244–8249 | C. Barrington-Leigh, A. Millard-Ball. "A global assessment of street-network sprawl." *PLOS ONE* 14(11): e0223078 (26 Nov 2019; editor H. Kato) | C. Barrington-Leigh, A. Millard-Ball. "Global trends toward urban street-network sprawl." *PNAS* 117(4): 1941–1950 |
| DOI | 10.1073/pnas.1504033112 | 10.1371/journal.pone.0223078 | 10.1073/pnas.1905232116 |
| Opened | https://pmc.ncbi.nlm.nih.gov/articles/PMC4500277/ (full text HTML) | https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0223078&type=printable (PDF, full), and S1 File https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0223078.s001 (read sections A, B.1, the paradigm table and the examples) | https://pmc.ncbi.nlm.nih.gov/articles/PMC6994978/ (full text HTML) |
| Access | **FT** (SI Appendix not opened) | **FT** + S1 | **FT** (SI Appendix not opened) |

Page numbers: PLOS pages are "x / 16"; S1 pages are the PDF pages of the S1 file. PMC HTML has no pages, so sections are cited.

## 2. BibTeX (Crossref via doi.org)

```bibtex
@article{barringtonleigh2015century,
  title   = {A century of sprawl in the United States},
  author  = {Barrington-Leigh, Christopher and Millard-Ball, Adam},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {112}, number = {27}, pages = {8244--8249}, year = {2015},
  doi     = {10.1073/pnas.1504033112}
}
@article{barringtonleigh2019global,
  title   = {A global assessment of street-network sprawl},
  author  = {Barrington-Leigh, Christopher and Millard-Ball, Adam},
  journal = {PLOS ONE},
  volume  = {14}, number = {11}, pages = {e0223078}, year = {2019},
  doi     = {10.1371/journal.pone.0223078}
}
@article{barringtonleigh2020global,
  title   = {Global trends toward urban street-network sprawl},
  author  = {Barrington-Leigh, Christopher and Millard-Ball, Adam},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {117}, number = {4}, pages = {1941--1950}, year = {2020},
  doi     = {10.1073/pnas.1905232116}
}
```

## 3. What they measure

**Street-network sprawl, meaning low connectivity.**
- **2015:** a US time series from 1920 to 2012 of three nodal measures, built from TIGER/Line, census and parcel data.
- **2019:** a global, roughly 1 km² resolution, composite **SNDi** plus 8 k-means "empirical types", built from OSM.
- **2020:** SNDi trends of new construction from 1975 to 2014 worldwide, dated with GHSL built-up epochs.

## 4. Exact metric definitions

### 2015 (PNAS, section "Introduction")

- (i) mean nodal degree;
- (ii) proportion of dead ends (degree 1);
- (iii) proportion of nodes of degree ≥ 4.
- **Edge degree** (Fig. 3) is the mean degree of an edge's two end nodes, from 2.0 to 4.0.

### 2019 SNDi (PLOS main text §2.2, S1 §A)

- **Nodal degree:** mean degree, share of dead ends, share of degree ≥ 4 (S1 Table A also lists the share of degree < 4).
- **Dendricity.** Each edge is classified as exactly one of:
  - **bridge**: deletion increases the component count; dead ends are excluded by the authors' convention;
  - **dead-end**: one endpoint has degree 1;
  - **self-loop**;
  - **cycle basis**: anything else.
  - The aggregates are the fraction of bridges, and the fraction "non-cycle" (bridges + dead ends + self-loops, Parthasarathi's "treeness"). Each is computed **by edge count and by length**.
- **Circuity:** LC_{r1,r2} = log Σ S_AB − log Σ R_AB over node pairs with r₁ < R_AB ≤ r₂.
  - S is network distance and R is straight-line distance.
  - Six bands: 0–0.5, 0.5–1, 1–1.5, 1.5–2, 2–2.5 and 2.5–3 km (S1 §A.3).
- **Sinuosity:** Σ edge length / Σ end-to-end distance (S1 §A.5).
- **Nodal sparsity:** log of a node-weighted density (S1 §A.4). It is not in the 13-variable PCA table.
- **SNDi** is the first principal component of 13 aggregate measures. The loadings are all positive (Table 1, p. 4); PCA1 explains 37%, eigenvalue 4.9. Loadings include:
  - −degree 0.35;
  - frac dead-ends 0.35;
  - log circuity 0.11–0.30 by band;
  - frac non-cycle (N edges) 0.38;
  - frac bridges (length) 0.22;
  - log sinuosity 0.16.
- **Preprocessing** (p. 5):
  - edges are split at intersections and duplicates dropped;
  - **intersections within 20 m are merged**;
  - **degree-2 nodes are annealed**.
- **Network scope** (S1 §B.1): every OSM `highway=*` except `track` is used **to compute the graph properties**. Then `service`, `pedestrian`, `path`, `cycleway`, `footway` and `steps` are **excluded when aggregating**. Footpaths therefore raise the connectivity of nearby car streets.
- **Urban only:** edges are classified as urban by imputation (LandScan, Atlas of Urban Expansion, GHSL), on a grid of about 1 km.

## 5. Data

- 2015: US TIGER/Line 1991–2013 vintages, ACS, and 23 M parcels (226 counties for the parcel series).
- 2019: global OSM (about 46 M km of streets; about 120 M intersections, about 171 M edges).
- 2020: the same, plus GHSL epochs (306 m aggregation) and Atlas of Urban Expansion boundaries (200 cities).

## 6. Key empirical numbers

- **2015 (US).** Mean nodal degree of new streets fell to about 2.60 at the 1994 sprawl peak, then rose to about 2.83 by 2012. The New Urbanist Stapleton development has a mean of about 3.47.
- **2019 paradigms.** Grid, medieval and culs-de-sac paradigms score SNDi **−1.01 / 0.66 / 7.9** in the main text (Fig. 2 caption, p. 5). The S1 table (S1 p. 20) gives **−0.76 / 0.91 / 8.11**. **The two versions disagree; cite either with care.** S1 component values:

| Paradigm | Degree | Dead-end share | Non-cycle share (N) | Circuity 0–0.5 km | Circuity 0.5–1 km |
|---|---|---|---|---|---|
| Grid | 3.99 | 0.00 | 0.00 | 1.24 | 1.27 |
| Medieval | 3.05 | 0.00 | 0.00 | 1.35 | 1.33 |
| Culs-de-sac | 2.19 | 0.45 | 0.48 | 2.34 | 1.94 |

- **2019, countries and cities** (Tables 3–4, pp. 11–12):
  - least sprawling countries: Argentina SNDi 0.6 (degree 3.4), Japan 1.3, Korea 1.4;
  - most sprawling: Sri Lanka 5.5, Philippines 5.4;
  - cities: Bangkok 7.3; Caracas **5.3** (degree 2.8; log circuity 0.5–1 km = 0.3); Hong Kong 0.9; Buenos Aires 0.3; Valledupar 0.2.
- **2020.** Bolivia, Argentina, Peru and Uruguay have recent SNDi ≤ 2.0. In the United States, network sprawl rose until the mid-1990s and has declined since.

## 7. Verbatim quotes

- 2019, p. 3: "Dendricity, i.e. whether a network is tree-like in that there is only one possible route to reach certain nodes;. Specifically, we calculate the fractions of edges that are classified as a network bridge (i.e., a link between two otherwise-disconnected network components), a dead-end, a self loop, or a part of the cycle basis;"
- 2019, p. 4: "high PCA2 can be interpreted as a place where circuity is much greater than would be expected from nodal degree, dendricity, and sinuosity alone, such as a place where connected neighborhoods are separated by a river or steep topography."
- 2019, p. 5: "including splitting roads into edges at intersections, dropping duplicates, merging intersections within 20m of each other, and annealing degree-two intersections into single edges"
- 2019 S1, p. 6: "Contrary to formal network terminology, our measure of bridges excludes dead-ends."
- 2019 S1, p. 10: "Thus we include cycleways, footpaths, service roads, and similar ways when computing graph properties and our measures of dendricity, nodal degree and circuity, but exclude them when computing the aggregate measures."
- 2019 S1, p. 22 (Japan example, Hiroshima): "Here, ravines separate adjacent residential development, and streets sometimes dead end as the terrain slopes up more steeply."
- 2015, Introduction: "Sprawl is characterized by a low nodal degree of intersections, a high proportion of dead ends, and a low proportion of nodes of degree four or more, all of which imply a street network with limited connectivity."
- 2020, Discussion: "those estimates excluded pedestrian and bicycle paths and service roads, which we incorporate in our measure of SNDi because they are demonstrably important to connectivity in some places."

## 8. How UrbanMorph could use it

1. **Adopt the dendricity measures: bridge share and non-cycle share, by count and by length.** They are what our D (dead ends) and I (islands) miss:
   - a tree-like branch with no dead end and no island (a loop on a stick) scores perfectly on D and I but is non-cycle;
   - non-cycle share is close to (1 − cycle ratio) but **edge-based**, unlike RoadWeaver's node-based cycle ratio.
2. **Route circuity in distance bands, not one band.** Our S_T circuity uses one 200–400 m band and drops unreachable pairs. SNDi's banded log circuity (sum ratio, not median) is more stable. Within a 1 km window only the 0–0.5 km band and part of the 0.5–1 km band fit.
3. **Do not compare absolute SNDi or dead-end rates.** SNDi uses 20 m merging, footpaths in the graph, an urban-only 1 km grid and a PCA fitted on the whole world. Report our component measures instead, and compute real references from our own real tiles processed identically.
4. **Topography.** SNDi treats terrain as a confound. Its PCA2 flags "steep topography" places where circuity exceeds what topology predicts. The Hiroshima example says streets dead-end as slope steepens. Both are qualitative support for the hypothesis that slope raises circuity and dead ends. Neither is a measurement of slope effects.
5. **Scoring unreachable OD pairs.** SNDi circuity is defined only over connected pairs (sums of network distances). Our OD unreachability is a separate measure; keep it separate and say so.
