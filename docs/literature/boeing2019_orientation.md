# Boeing 2019: Urban spatial order (orientation, configuration, entropy)

## 1. Header

- **Citation:** Geoff Boeing. "Urban spatial order: street network orientation, configuration, and entropy." *Applied Network Science* 4, article 67 (2019). Published online 23 Aug 2019.
- **DOI:** 10.1007/s41109-019-0189-1
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/1808.00600 (author preprint dated "August 2019". It states it is a preprint of the journal article. Read in full.)
  - https://doi.org/10.1007/s41109-019-0189-1 (BibTeX by content negotiation)
  - https://api.crossref.org/works/10.1007/s41109-019-0189-1 (article number 67, online date)
  - https://appliednetsci.springeropen.com/counter/pdf/10.1007/s41109-019-0189-1.pdf (returned an HTML bot page, so the journal PDF was **not** read)
- **Access level:** **FT** (author preprint). Page numbers below are **preprint** pages; the journal pagination is not verified.

## 2. BibTeX (from Crossref via doi.org; article number from the Crossref API)

```bibtex
@article{boeing2019urban,
  title   = {Urban spatial order: street network orientation, configuration, and entropy},
  author  = {Boeing, Geoff},
  journal = {Applied Network Science},
  volume  = {4},
  number  = {1},
  pages   = {67},
  year    = {2019},
  doi     = {10.1007/s41109-019-0189-1}
}
```

## 3. What it measures

The paper studies city-scale street orientation order and network configuration in 100 cities. It computes:
- orientation entropy, unweighted (simplified graph) and length-weighted (unsimplified graph);
- a normalised orientation-order indicator φ (new in this paper);
- median segment length, average node degree, dead-end share and four-way share;
- average circuity.

Cities are then clustered on (k̄, φ, l̃, ς). Terrain is not measured; it is only invoked to explain results.

## 4. Exact metric definitions (preprint pp. 5–6)

- **Bearings.** On the simplified graph, each edge gives its u→v compass bearing **and its reciprocal**. On the unsimplified graph, bearings of raw OSM straight segments are length-weighted. Self-loops are ignored.
- **Binning.** 36 bins of 10°, **shifted by −5°** so that 0°, 90°, … sit at bin centres.
- **Orientation entropy.** H_o = −Σ_{i=1}^{36} P(o_i) ln P(o_i), in nats (Eq. 1). H_w is the same over length-weighted bearings (Eq. 2).
  - H_max = ln 36 = 3.584.
  - Two-bin minimum = 0.693.
  - "Perfect grid" (four equal bins) H_g = 1.386.
- **Orientation order** (Eq. 3): **φ = 1 − ((H_o − H_g)/(H_max − H_g))²**.
  - φ = 0 for a uniform distribution and φ = 1 for a single perfect grid.
  - The square is said to "linearize" the scale.
- **Average circuity** (Eq. 4): **ς = L_net / L_gc**.
  - L_net is the sum of all edge lengths.
  - L_gc is the sum of great-circle distances between all pairs of *adjacent* nodes.
  - This is an **edge-level sinuosity**, not a route circuity between OD pairs.
- **Other indicators** (simplified, unweighted graph):
  - median segment length l̃ (m);
  - average node degree k̄;
  - dead-end node share P_de;
  - four-way node share P_4w.

## 5. Data

- 100 cities on all continents except Antarctica. The sampling "emulates Louf and Barthelemy's (2014)".
- Boundaries are each city's closest approximation to a "municipality"; Manhattan is used instead of all of New York City.
- Streets are downloaded from OSM with OSMnx (Boeing 2017).
- Over 4.8 million nodes and 3.3 million edges in total (preprint p. 2).
- The network type (drive or all) is not stated in the preprint text I read: **not verified**.

## 6. Key empirical numbers (real-world references)

From Table 1 (pp. 7–9) and Table 2 (p. 10):

| Quantity | Value (city) |
|---|---|
| Lowest H_o / highest φ | Chicago: H_o 2.083, φ 0.899 |
| Highest H_o / lowest φ | Charlotte: H_o 3.582, φ 0.002 |
| Highest circuity ς | Caracas 1.148, Hong Kong 1.137, Sarajevo 1.133 |
| Lowest circuity ς | Detroit 1.012, Buenos Aires 1.011, Chicago 1.016 |
| P_de range | 0.027 (Buenos Aires, Manhattan) to 0.395 (Helsinki) |
| k̄ range | 2.348 (Helsinki) to 3.548 (Buenos Aires) |
| l̃ range | 23.2 m (Venice) to 233.0 m (Shanghai) |
| Hilly cities of interest | Pittsburgh φ 0.018, ς 1.054; San Francisco φ 0.278, ς 1.033; Lisbon φ 0.023, ς 1.068; Seoul φ 0.009, ς 1.048, P_de 0.101; Rio de Janeiro ς 1.055 |
| Regional means (Table 2) | US/Canada: φ 0.427, ς 1.043, P_de 0.116, P_4w 0.334. Europe: φ 0.033, ς 1.061, P_de 0.172, P_4w 0.172 |

Correlations across the 100 cities (p. 15):
- r(φ, ς) = −0.432;
- r(φ, P_de) = −0.376;
- r(φ, k̄) = 0.518;
- r(φ, P_4w) = 0.634;
- r(ς, k̄) = −0.672;
- r(ς, P_4w) = −0.689.

## 7. Verbatim quotes

The Greek capital Eta of the PDF is written H here.

- p. 5: "To avoid extreme bin-edge effects around common values like 0° and 90°, we shift each bin by -5° so that these values sit at the centers of their bins rather than at their edges."
- p. 6: "While Hw is biased by the city's shape (due to length-weighting), Ho is not."
- p. 6: "This perfect grid entropy, Hg, would equal 1.386 nats."
- p. 6: "where Lnet represents the sum of all edge lengths in the graph and Lgc represents the sum of all great-circle distances between all pairs of adjacent nodes."
- p. 7: "We find that Ho and Hw are very strongly correlated (Pearson product-moment correlation coefficient r > 0.99, p < 0.001) and thus provide essentially redundant statistical information about these networks."
- p. 9: "while Caracas, Hong Kong, and Sarajevo have the most circuitous networks (13.3%-14.8% more circuitous than straight-line distances) due largely to topography."
- p. 17: "For instance, topography likely constrains griddedness and influences circuity and orientation entropy: a study of urban elevation change and hilliness in conjunction with entropy and circuity would help clarify these relationships."

## 8. How UrbanMorph could use it

- **Keep φ and H_o in S10 and fix the conventions.** Use 36 bins, the −5° shift, reciprocal bearings and natural log. Report the unweighted simplified H_o as primary (unbiased by tile shape); H_w is redundant (r > 0.99 at city scale).
- **Scale caveat.** Boeing's values are for whole municipalities. Our 1 km windows hold far fewer edges, so per-tile entropy is biased low and noisy. The plug-in entropy estimator is biased downward for small samples (my statistical note, not from the paper). Compare tiles only with real tiles of the same size, never with Table 1 values.
- **Name the circuity.** Boeing's ς is edge sinuosity (adjacent nodes). Our S_T circuity is a median *route* ratio over reachable node pairs 200–400 m apart (`terrain_metrics.circuity`). Report both under distinct names: "sinuosity" and "route circuity".
- **Terrain.** The paper gives no slope measurement. It explicitly calls for a study of "hilliness in conjunction with entropy and circuity" (p. 17), so it is not a baseline for TAS. Its hilly-city circuity values are city-wide edge sinuosity, not comparable with our window route circuity.
