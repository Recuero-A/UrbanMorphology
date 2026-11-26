# Dong, Li, Li, Li, Shen, Feng 2025: DRoLaS, Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts

## 1. Header

- **Citation:** Shenao Dong, Weitao Li, Bo Li, Long Li, Junao Shen, Tian Feng. "DRoLaS: Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts." In *Proceedings of the 2025 International Conference on Multimedia Retrieval (ICMR '25)*, Chicago, IL, USA, pp. 237–245. ACM, published 30 Jun 2025.
- **DOI:** 10.1145/3731715.3733316
- **Identification:** "DRoLaS" is this ICMR 2025 paper, from the same Zhejiang group (senior author Tian Feng) as CaRoLS.
- **URLs opened:**
  - https://api.openalex.org/works/doi:10.1145/3731715.3733316: abstract, pages, closed access.
  - https://api.crossref.org/works/10.1145/3731715.3733316: event, pages, authors; no abstract.
  - https://dl.acm.org/doi/10.1145/3731715.3733316: HTTP 403.
  - GitHub repository search for "DRoLaS": no matching code.
  - Third-party mentions in https://github.com/jalengg/groundwork (`README.md`, `docs/findings.md`), not the paper.
- **Access level:** **AO** (abstract only, via OpenAlex).

## 2. BibTeX

```bibtex
@inproceedings{dong2025drolas,
  author    = {Dong, Shenao and Li, Weitao and Li, Bo and Li, Long and Shen, Junao and Feng, Tian},
  title     = {{DRoLaS}: Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts},
  booktitle = {Proceedings of the 2025 International Conference on Multimedia Retrieval},
  series    = {ICMR '25},
  year      = {2025},
  pages     = {237--245},
  publisher = {ACM},
  doi       = {10.1145/3731715.3733316}
}
```

## 3. What it does

A diffusion model synthesises **hierarchical** (multi-class) road layouts under input conditions, coarse to fine. A "local adaptation module" and a "global perception module" provide conditional control. A class-weighted denoising loss addresses road-class imbalance, and a "connectivity refinement strategy" improves the generated layouts (all from the abstract).

## 4. Learned or procedural, and HOW

- **Learned:** diffusion (abstract).
- Latent vs pixel space, network details and the coarse-to-fine mechanism: **not verified**.
- **Training data:** a "self-constructed dataset". Cities, number of samples and tile size are **not verified**.
  - The third-party `groundwork/docs/findings.md` cites "DRoLaS approx." class pixel shares (background about 88–90%) and refers to Brisbane suburbs in "the reference papers". Its README cites "DRoLaS Eq. 9" with class weights `[1.0, 1.2, 1.4, 1.4, 1.4]`.
  - These are third-party, **not verified**.
- **Raster or vector:** probably raster class maps; **not verified**. Whether the "connectivity refinement strategy" acts on pixels or on an extracted graph, and whether it is post-hoc, is **not verified**.

## 5. Inputs/outputs and how terrain enters

**Not verified.** The abstract names only "diverse input conditions". The same group's precursor (arXiv 2305.08186) used elevation, population density and land use; whether DRoLaS uses a DEM or water is **not verified**.

## 6. Evaluation

**Not verified** beyond the abstract, which says experiments on a self-constructed dataset show layouts that "reflect real-world structures and hierarchical patterns while effectively responding to input conditions". Metrics, baselines, splits, statistics and runtime are not available.

## 7. Verbatim quotes (abstract, via OpenAlex; no page numbers)

- Abstract: "While deep learning-based methods have demonstrated potential in road layout synthesis, they struggle to adapt to diverse input conditions and capture the hierarchical nature of real-world road networks."
- Abstract: "Our method incorporates a local adaptation module and a global perception module to enable effective conditional control."
- Abstract: "Meanwhile, we address data imbalance regarding road classes through a class-weighted denoising loss, and introduce a connectivity refinement strategy to enhance the quality of generated layouts."
- Abstract: "Experimental results on a self-constructed dataset demonstrate that our DRoLaS successfully synthesizes high-quality road layouts that reflect real-world structures and hierarchical patterns while effectively responding to input conditions."

## 8. Comparability with UrbanMorph

- **Code, weights, licence:** **none found**.
- **macOS:** n/a.
- **Inputs we could supply:** unknown. If it follows the group's precursor, a DEM channel plus land use.
- **Fair comparison:** condition-driven generation (no surrounding streets, as far as the abstract says), i.e. terrain-only growth.
- **Retraining:** required.
- **Effort (estimate):** reimplementation 80–150 h plus training. **Not recommended.**
- **Novelty relevance:** the "connectivity refinement strategy" is a second counterexample, after RoadWeaver, to "nobody targets connectivity". Whether it is post-hoc repair or inference-time guidance is **not verified**, and that distinction matters for UrbanMorph's claim of inference-time network potentials.

## 9. Ideas we could adopt

1. **Report with and without any post-hoc connectivity repair.** If a baseline (DRoLaS-style or RoadWeaver-style) repairs connectivity after generation, report its metrics both raw and repaired. Do the same for UrbanMorph with its potentials switched off and on, so that generator quality and repair are separable.
2. **Class-imbalance-aware metrics:** report realism per road class, weighted so that rare arterials are not swamped by local streets.
