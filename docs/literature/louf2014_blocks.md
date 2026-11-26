# Louf & Barthelemy 2014: A typology of street patterns (block area and shape)

## 1. Header

- **Citation:** Rémi Louf, Marc Barthelemy. "A typology of street patterns." *Journal of the Royal Society Interface* 11(101): 20140924 (Dec 2014).
- **DOI:** 10.1098/rsif.2014.0924
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/1410.2094 (v1, 8 Oct 2014; read in full; Eq. 4–5 checked on a rendered page image)
  - https://doi.org/10.1098/rsif.2014.0924 (BibTeX)
- **Access level:** **FT** (arXiv v1). The published version was not opened, so differences between v1 and the journal text are **not verified**. Page numbers are arXiv v1 pages.

## 2. BibTeX (Crossref via doi.org)

```bibtex
@article{louf2014typology,
  title   = {A typology of street patterns},
  author  = {Louf, R{\'e}mi and Barthelemy, Marc},
  journal = {Journal of The Royal Society Interface},
  volume  = {11},
  number  = {101},
  pages   = {20140924},
  year    = {2014},
  doi     = {10.1098/rsif.2014.0924}
}
```

## 3. What it measures

The paper classifies cities by their **blocks**, the faces of the planar street graph, rather than by their streets. Each block has an area A and a shape factor Φ. A city's "fingerprint" is the conditional distribution P(Φ | A). Cities are then hierarchically clustered on a distance between fingerprints.

## 4. Exact metric definitions

- **Block:** "the smallest area delimited by roads" (p. 2), a cell of the planar graph.
  - Blocks ignore dead ends: the Fig. 1 caption says "the block representation does not take dead-ends into account."
  - Faces "that have no real-world counterpart but appear due to the particular way data are encoded in OpenStreetMap" were removed. The removal procedure is **not specified**.
- **Area distribution:** fat tail P(A) ~ A^(−τ) with τ ≈ 2 (Eq. 1). The authors say it does not distinguish cities.
- **Shape factor:** **Φ = A / A_C**, where A_C is the area of the **circumscribed circle** (Eq. 2). Φ < 1; smaller means more anisotropic.
  - My reference values (geometry, not from the paper): circle 1; square 2/π ≈ 0.64; 2:1 rectangle ≈ 0.51; 4:1 rectangle ≈ 0.30; equilateral triangle ≈ 0.41.
  - Manhattan's reported peaks at Φ ≈ 0.3 and ≈ 0.5 are consistent with about 4:1 and 2:1 rectangles. This is my reading.
- **Fingerprint:** P(Φ | A) over logarithmic area bins, with P(Φ) = Σ_A P(Φ | A) P(A) (Eq. 3).
- **Distance:**
  - Blocks are kept with A in [10³, 10⁵] m², in two bins: α₁ = [10³, 10⁴] and α₂ = [10⁴, 10⁵] m².
  - f_α(Φ) = number of cells in bin α with form factor Φ / **total number of cells in the city**.
  - d_α(a, b) = ∫₀¹ |f_α^a(Φ) − f_α^b(Φ)|^n dΦ, with n = 1 and n = 2 tested and conclusions "robust" (Eq. 4).
  - D(a, b) = Σ_α d_α(a, b)² (Eq. 5, as printed in v1).
  - Then classical hierarchical clustering.

## 5. Data

- 131 cities on all continents except Antarctica.
- OSM street networks "restricted to the city center using the Global Administrative Areas database (or databases provided by the countries administration)" (p. 2).

## 6. Key empirical numbers

- Four groups at an intermediate dendrogram level (Fig. 2 caption, p. 3):

| Group | Share of cities | Example(s) | Character |
|---|---|---|---|
| 1 | 0.8% | Buenos Aires only | medium blocks in bin α₂, square and regular rectangles |
| 2 | 20.9% | Athens | dominant small blocks, broad shapes |
| 3 | 77.5% | New Orleans | various shapes, slight predominance of medium areas |
| 4 | 0.8% | Mogadishu | small square blocks |

- All North American cities except Vancouver are in group 3, as are all European cities except Athens (p. 5).
- Seven US cities classify with the European subgroup: Indianapolis, Portland, Pittsburgh, Cincinnati, Baltimore, Washington, Boston (p. 5).
- Blocks of 10³–10⁵ m² dominate cell counts (p. 4–5).
- **No per-city table of mean A or Φ is given**, so there are no numeric block-size references beyond the bin ranges.

## 7. Verbatim quotes

- p. 2 (Fig. 1 caption): "Note that the block representation does not take dead-ends into account."
- p. 2: "A block can usually be defined without ambiguity as being the smallest area delimited by roads"
- p. 2: "the form (or shape) factor Φ, defined in the Geography literature in [31] as the ratio between the area of the block and the area of the circumscribed circle C"
- p. 2: "it does not allow to distinguish cities from one another." (on the power-law P(A))
- p. 3: "if two cities have blocks of the same shape in the same proportion but with totally different areas, they will look different. We thus need to combine the information about both the shape and the area."
- p. 5: "we will neglect very small blocks (of area < 10³m²) and very large ones (of area > 10⁵m²)."

## 8. How UrbanMorph could use it

1. **Add Φ next to Polsby–Popper (PP = 4πA/P²).**
   - PP falls with boundary roughness, so polyline-digitised curves lower it.
   - Φ depends only on the farthest vertices.
   - Generated curvy streets made of short segments will lower PP but barely change Φ.
   - Report both. Compare the **conditional** distribution P(Φ | A-bin) with real tiles: W1 per area bin, or the d_α of Eq. 4 with n = 1.
2. **Window clipping.** Faces cut by the 1 km window are not blocks. Drop faces touching the window boundary, for both real and generated graphs. This biases toward small blocks, and the bias is the same for both.
3. **Dead-end blindness.** Block metrics cannot see culs-de-sac (stated by the authors). Always pair them with dead-end ratio and dendricity (see [barringtonleigh2019_sndi.md](barringtonleigh2019_sndi.md)).
4. **OSM face artefacts.** Dual carriageways and roundabouts create faces with no real counterpart. The authors removed them but did not say how. Filter them, or consolidate first ([boeing2025_consolidation.md](boeing2025_consolidation.md)), before building the real reference distribution.
5. **Area range.** The 10³–10⁵ m² range matches residential/urban blocks and fits inside a 1 km window. It is a sensible default range for our block-area W1.
