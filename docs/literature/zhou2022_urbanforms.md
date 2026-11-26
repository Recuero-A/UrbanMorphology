# Zhou, Bao & Liu 2022: mapping urban forms worldwide (8,910 networks, 25 indicators)

## 1. Header

- **Citation:** Qi Zhou, Junya Bao, Helin Liu. "Mapping Urban Forms Worldwide: An Analysis of 8910 Street Networks and 25 Indicators." *ISPRS International Journal of Geo-Information* 11(7): 370 (30 Jun 2022). Open access, CC BY.
- **DOI:** 10.3390/ijgi11070370
- **URLs opened (2026-11-25):**
  - https://api.crossref.org/works?query.bibliographic=... (located the DOI)
  - https://doi.org/10.3390/ijgi11070370 (BibTeX)
  - https://mdpi-res.com/d_attachment/ijgi/ijgi-11-00370/article_deploy/ijgi-11-00370.pdf (publisher CDN PDF; read the abstract, sections 2, 4.1 and 4.2)
- **Access level:** **FT** (partly read: sections 4.3 onward were skimmed only).

## 2. BibTeX (Crossref via doi.org)

```bibtex
@article{zhou2022mapping,
  title   = {Mapping Urban Forms Worldwide: An Analysis of 8910 Street Networks and 25 Indicators},
  author  = {Zhou, Qi and Bao, Junya and Liu, Helin},
  journal = {ISPRS International Journal of Geo-Information},
  volume  = {11}, number = {7}, pages = {370}, year = {2022},
  doi     = {10.3390/ijgi11070370}
}
```

## 3. What it measures

The paper applies correlation analysis, PCA and hierarchical clustering to Boeing's global indicator dataset (8,910 urban areas, 25 indicators, including `grade_mean` and `grade_median`). The resulting clusters are compared with a 1 km global terrain dataset and land-cover data.

## 4. Definitions

- Indicators are as in Boeing 2022; this paper defines no new grade metric.
- The PCA keeps components with eigenvalue > 1.

## 5. Key numbers

- Six PCs explain about 86% of variance (p. 7). PC2 "Terrain-variation" explains 20.55%.
  - It loads on elev_iqr 0.881, elev_range 0.850, elev_std 0.914, **grade_mean 0.841** and **grade_median 0.817** (Table 3, p. 7).
  - PC6 "Altitude" explains 4.40% (elev_mean 0.950).
- Correlations: grade_mean with grade_median 0.97 (p. 6). The terrain indicators correlate "moderate[ly] (e.g., 0.35−0.38)" with orientation entropy and order (p. 6).
- The seven clusters include "varied-terrain" and "high-altitude" (abstract).

## 6. Verbatim quotes

- Abstract: "six principal components (i.e., size, terrain-variation, regularity, long-street, circuity and altitude) were found."
- p. 6: "It should also be noted that the terrain-related indicators (grade_mean and grade_median) have a moderate correlation (e.g., 0.35−0.38) with orientation_entropy and orientation_order, which indicates the relationship between urban form and geographical factors."

## 7. How UrbanMorph could use it

- **Evidence that terrain is a main independent axis of real street form at city scale.** Grade and elevation spread form their own PC, separate from regularity and circuity. Realism should therefore be judged **conditional on terrain**: compare generated hilly tiles with real hilly tiles, not with a pooled reference.
- **Weak link to orientation.** The terrain-orientation link is weak to moderate (|r| ≈ 0.35–0.38). It agrees with our own Spearman 0.43 (grade_mean vs orientation entropy) on Dataverse v3.1 ([boeing2022_global.md](boeing2022_global.md)).
- **Not a street-level adaptation measurement.**
