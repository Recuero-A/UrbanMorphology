# Feng, Li, Li, Li, Shen 2025: CaRoLS, Condition-adaptive multi-level road layout synthesis

## 1. Header

- **Citation:** Tian Feng, Long Li, Weitao Li, Bo Li, Junao Shen. "CaRoLS: Condition-adaptive multi-level road layout synthesis." *Computers & Graphics* 133:104451.
  - Issue dated December 2025 (Crossref); online 2025-10-04 (OpenAlex).
  - Elsevier PII S0097849325002924.
  - Funders per Crossref: Zhejiang Province Natural Science Foundation (LTGS23F020001); NSFC (62202421).
- **DOI:** 10.1016/j.cag.2025.104451
- **URLs opened:**
  - https://api.crossref.org/works/10.1016/j.cag.2025.104451: metadata and PII, no abstract.
  - https://api.openalex.org/works/doi:10.1016/j.cag.2025.104451: closed access, no abstract.
  - https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.cag.2025.104451: abstract null.
  - https://www.sciencedirect.com/science/article/pii/S0097849325002924 and .../abs/pii/S0097849325002924: HTTP 403.
  - https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c251t-2e62a9868f58940423dd9bd90f8f8236ec651678ea02e66a671965401c7a4a523, a library catalogue that search engines show with an abstract: HTTP 404 when fetched.
  - https://researchr.org/publication/FengLLLS25/bibtex: connection refused.
  - Third-party reimplementation, **not the paper**: https://github.com/jalengg/groundwork (README; `docs/carols_spec_audit.md`; `docs/findings.md`). See `jalengg2026_groundwork.md`.
  - The same group's earlier paper, for context: https://arxiv.org/abs/2305.08186 and https://arxiv.org/pdf/2305.08186 (read; mostly in Chinese).
  - GitHub repository search for "CaRoLS": no official code.
- **Access level:** **MD** (metadata only). No abstract or full text was opened from a primary or publisher page.
  - Search-engine snippets describe a VAE latent plus diffusion, optional conditioning on "natural and socio-economic information", and an Australian metropolitan dataset. These are **not verified** and are not used as facts below.

## 2. BibTeX

```bibtex
@article{feng2025carols,
  author  = {Feng, Tian and Li, Long and Li, Weitao and Li, Bo and Shen, Junao},
  title   = {{CaRoLS}: Condition-adaptive multi-level road layout synthesis},
  journal = {Computers \& Graphics},
  year    = {2025},
  volume  = {133},
  pages   = {104451},
  doi     = {10.1016/j.cag.2025.104451}
}
```

## 3. What it does

**Not verified from the paper.** From the title, it synthesises road layouts with several road levels (hierarchy) and adapts to conditions.

The third-party `groundwork` repository, which claims to reimplement it, describes a two-stage latent diffusion:
1. A VAE compresses a one-hot multi-level road raster (512 × 512 × k) to a 64 × 64 × 4 latent.
2. A conditional diffusion U-Net with "Condition-aware Decoder Blocks" generates latents, unconditionally or from a condition image.

These descriptions come from a third party and are **not verified** against the paper.

## 4. Learned or procedural, and HOW

- **Learned** (title and venue). Model type, data and tile size are **not verified** from a primary source.
- **Third-party claims** (`groundwork/docs/carols_spec_audit.md` and `findings.md`, written by someone with apparent access to the paper):
  - VAE loss is "0.0001 · KL + focal".
  - DDPM ε-prediction.
  - The conditioning image is encoded by "3x3 conv + ReLU + BatchNorm".
  - Local Details Enhancement (LDE) and Global Context Integration (GCI) inside each decoder block.
  - A dataset of about 2,584 Australian tiles, 512 × 512 at 5 m, with Brisbane held out against Sydney and Melbourne.
  - Post-processing by "dilation → skeletonization → small-component pruning" (README, "per CaRoLS Section 3.3").
  - **All not verified.**
- **Raster or vector:** the output is probably a raster road layout per level (third-party description); **not verified**.

**Group context (verified, from arXiv 2305.08186 v2, the same senior author Tian Feng):**
- The precursor uses a cGAN conditioned on elevation, population density and land use.
- Data: OSM roads for Sydney, Melbourne and Brisbane, rasterised to 512 × 512 tiles at 5 m with 3-px roads; 2,586 pairs, split 70/30.
- Graph extraction: threshold 127, dilation, skeletonisation, 8-neighbour vertex detection, triangle removal and redundant-vertex removal.
- Metrics: connectivity index, intersection count, total length, metric reach (500 m), transport convenience (Euclid/shortest path, 100 OD pairs > 200 m apart).
- Whether CaRoLS reuses this dataset is **not verified**.

## 5. Inputs/outputs and how terrain enters

**Not verified.**
- The third-party audit says the paper's condition image contains "land use, elevation, population density". The audit also mentions an "existing roads" channel, which the reimplementation dropped as a leak.
- If correct, elevation enters as a raster channel. Slope, water or a DEM resolution are **not verified**.

## 6. Evaluation

**Not verified.** No metrics, baselines, splits, statistics or runtime could be read.

## 7. Verbatim quotes

None. No primary text was accessible. Quoting third-party summaries as the paper's words would be misleading.

## 8. Comparability with UrbanMorph

- **Official code, weights, licence:** **none found** (GitHub search; no link in metadata).
- **Third-party code:** https://github.com/jalengg/groundwork (MIT, unpublished prototype, no weights, self-reported poor outputs). See `jalengg2026_groundwork.md`.
- **macOS:** n/a for the official method.
- **Inputs we could supply, if the third-party description is right:** a DEM channel, a water channel (if any), land use, and a boundary implied by the tile.
- **Fair comparison:** unconditional or condition-only generation from terrain, i.e. the same setting as UrbanMorph's terrain-only growth.
- **Retraining:** required in any case.
- **Effort (estimate):** reimplementation 80–150 h plus training (consistent with `docs/M18_BASELINES.md`). Starting from the `groundwork` code instead: 15–30 h human time plus multi-day training, with unknown fidelity to the paper. **Not recommended** as a primary baseline until the full text or the authors' code is available.

## 9. Ideas we could adopt

1. **Multi-level evaluation:** report realism metrics per road class (arterial vs local), not only pooled. Hierarchy is the paper's titular concern.
2. **Group metric set** (verified in the precursor arXiv 2305.08186): connectivity index (average degree of nodes with degree ≠ 2), intersection count, total length, metric reach at 500 m, transport convenience. These are cheap network statistics we already partly compute. Add metric reach if it is missing.
3. **Condition-leak check:** the third-party finding that an "existing roads" condition channel turned generation into segmentation is a direct warning for any DEM channel that carries street imprints.
