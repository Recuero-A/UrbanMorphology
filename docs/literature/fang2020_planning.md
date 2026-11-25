# Fang, Jin, Yang 2020: planning-guided GAN street infill (arXiv 2010.04536; JUT 2022)

## 1. Header

- **Citation (preprint read):** Zhou Fang, Ying Jin, Tianren Yang. "Incorporating planning intelligence into deep learning: A planning support tool for street network design." arXiv:2010.04536, v1, 9 Oct 2020 (only version, per the arXiv API).
- **Journal version (not read):** same authors, "Incorporating Planning Intelligence into Deep Learning: A Planning Support Tool for Street Network Design", *Journal of Urban Technology* 29(2):99–114. Online 22 Dec 2021, issue 3 Apr 2022. DOI 10.1080/10630732.2021.2001713 (Crossref). Differences between the preprint and the journal version are not verified.
- **Precursor:** Fang, Yang, Jin 2020, "DeepStreet: A deep learning powered urban street network generation module", arXiv:2010.04365 (abstract read via the arXiv API). Barcelona; context-only CNN infill. Fang et al. 2022 (p. 2039) says that DeepStreet attached elevation and aspect channels and failed in hilly areas.
- **DOI:** 10.48550/arXiv.2010.04536 (preprint); 10.1080/10630732.2021.2001713 (journal).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/2010.04536 (full text including the supplementary sections, 39 PDF pages)
  - https://export.arxiv.org/api/query?id_list=2010.04536 and ?id_list=2010.04365
  - https://api.crossref.org/works/10.1080/10630732.2021.2001713
  - https://api.github.com/search/repositories?q=DeepStreet+street+network (0 results)
- **Access level:** **FT** for the arXiv preprint, **MD** for the journal version.
- **Page numbers** below are PDF pages of the arXiv file. The footer "n / 18" equals PDF page n + 1. Tables are on pp. 20–21 and the supplement on pp. 27–39.

## 2. BibTeX

```bibtex
@article{fang2022planning,
  author  = {Fang, Zhou and Jin, Ying and Yang, Tianren},
  title   = {Incorporating Planning Intelligence into Deep Learning: A Planning Support Tool for Street Network Design},
  journal = {Journal of Urban Technology},
  volume  = {29},
  number  = {2},
  pages   = {99--114},
  year    = {2022},
  doi     = {10.1080/10630732.2021.2001713}
}

@misc{fang2020planning_arxiv,
  author        = {Fang, Zhou and Jin, Ying and Yang, Tianren},
  title         = {Incorporating planning intelligence into deep learning: A planning support tool for street network design},
  year          = {2020},
  eprint        = {2010.04536},
  archivePrefix = {arXiv}
}
```

## 3. What it does

A conditional GAN fills a masked region of a 512 m × 512 m street raster. It is conditioned on the surrounding streets, SRTM elevation and slope aspect, and optionally on planning guidance: kept junction nodes and hand-labelled street-pattern types. Three nested models test whether adding junctions (Model 2) and pattern types (Model 3) improves the infill. Training is on Prague and testing on Barcelona. The paper frames the result as an interactive planning-support tool, not as a terrain study.

## 4. Learned or procedural, and how

- **Learned, conditional GAN, raster.**
- **Generator:** a fully convolutional encoder-decoder.
  - Stride convolutions only, no pooling, 256 → 64 px.
  - Dilated convolutions in the middle.
  - The guidance channels are concatenated repeatedly in the decoder (supp. B, p. 31).
- **Discriminator:** 11 conv-SpectralNorm-LeakyReLU layers plus fully connected layers, producing a real/fake probability (p. 32).
- **Loss:** a masked MSE plus a standard adversarial loss weighted by α (supp. C, pp. 33–34). Adam optimiser.
- **Training data:**
  - **Prague:** "46,856 unrepeated multi-channel patches", 256 × 256 px at 2 m/px (512 m), cropped at random from a 6 × 10 km map (p. 8).
  - **Test, Barcelona:** "26,872 unrepeated patches" from a 6 × 10 km region (p. 8).
  - Roads: OSM, with hierarchy (motorway, primary, secondary, tertiary, residential).
- **Training cost:** "a single NVIDIA Tesla P100 GPU for 100 epochs ... roughly 221.3 h (9.2 d)" per experiment (p. 10).
- **Raster to graph:** **none.** "All reported results are direct outputs from the trained model without any post-processing" (p. 11). No graph is extracted.

## 5. Inputs/outputs and how terrain enters

- **Generator input** (Table S1, p. 33):
  - x, a 256×256×5 tensor: street network, elevation and aspect (the "fundamental context"), plus a mask channel and a masked noise channel;
  - x_g, 256×256×2: a masked junction-node channel and a masked pattern-type annotation channel.
- **Terrain:**
  - **SRTM elevation plus slope aspect only**, as grey-scale rasters at 2 m/px after projection to the local CRS (supp. A, p. 27). There is **no slope-magnitude channel and no water channel.**
  - The terrain channels are simply concatenated to the input, with no dedicated module.
  - The successor paper (Fang 2022, p. 2039) says this design "could not effectively extract topographic features".
- **Guidance:**
  - Model 2 keeps a random 10–90 % of the real junctions inside the generation region during training and testing; the sensitivity runs keep 30, 60 and 90 % (Table 2 note, p. 21).
  - Model 3 adds pattern-type labels (linear development, gated compound, medieval, irregular grid, orthogonal grid; p. 8).
- **Mask:** a random binary mask defines the generation region (supp. A, p. 28). Its shape and size distribution are not stated (not verified). Figure 6 shows test samples with context coverage up to and below 40 %.
- **Output:** a single-channel 256 × 256 street raster.

## 6. Evaluation

- **Split: geographic hold-out by city**, Prague for training and Barcelona for testing. "guarantee that our trained model has never been exposed to the data used for testing" (p. 8).
- **Metric:** mean l1 and l2 pixel error inside the generation region on the whole test set (Table 2, p. 21):

  | Model | Mean l1 | Mean l2 |
  |---|---|---|
  | Model 1 (context + terrain only) | 6.55 % | 5.94 % |
  | Model 2 (+ junctions, 10–90 % kept) | 5.58 % | 5.13 % |
  | Model 2, 30 % kept | 5.90 % | 5.52 % |
  | Model 2, 60 % kept | 5.56 % | 5.09 % |
  | Model 2, 90 % kept | 5.21 % | 4.67 % |
  | Model 3 (+ pattern types) | 5.37 % | 4.96 % |

- **Context-coverage analysis:** 2-D histograms of l2 error against context coverage (Figure 6, pp. 12–13). Error rises sharply below 40 % coverage.
- **Qualitative:** Barcelona examples (Figs. 3–5) and user-guided edits (Figs. 7–8).
- **Not reported:**
  - **No terrain-specific evaluation.** No flat vs hilly split, and no slope or grade analysis.
  - No network or graph metrics.
  - No user study. The realism claim is the authors' own judgement (see the p. 11 quote).
  - No uncertainty, seeds or repeated runs.
  - No external baselines.
- **Runtime:** "0.21 s per data pair" on a P100 GPU and "3.16 s on Intel Xeon CPU @ 4.00GHz" per 256 × 256 sample (p. 10).

## 7. Verbatim quotes

- p. 2 (abstract): "We propose a novel, complementary use of deep neural networks and planning guidance to automate street network generation that can be context-aware, example-based and user-guided."
- p. 8: "available digital elevation data was sourced from the shuttle radar topography mission"
- p. 8: "we use Prague data to train the model and Barcelona data to test. More specifically, we randomly cropped 46,856 unrepeated multi-channel patches in a size of 256 × 256 pixels (2 × 2 m per pixel and 512 × 512 m per patch) within a prepared 6 × 10 km multi-channel city map of Prague"
- p. 10: "trained with Python v3.7.1, PyTorch v1.1.0, CUDA v10.0 and CUDNN v7.3.1 on a single NVIDIA Tesla P100 GPU for 100 epochs on each experiment."
- p. 10: "the model ran at 0.21 s per data pair on the same GPU used for training and 3.16 s on Intel Xeon CPU @ 4.00GHz"
- p. 11: "All reported results are direct outputs from the trained model without any post-processing."
- p. 11: "although all the results are different from the ground truths, the results themselves are natural and realistic (difficult for people to discern whether the road network is an actual one or a generated one)"
- p. 12: "Highly uncertain performance should be expected when the coverage of context is less than 40%."
- p. 27 (supp. A): "project these data to the local geo-coordinate system, raster, scale and process them into three single channel raster images: one binary images (street network) and two grey-scale raster images (digital elevation and aspect of slope), where each pixel represents an area of 2×2 meters."
- p. 28 (supp. A): "we randomly crop multi-channel patches of size 256×256, from the prepared multi-channel maps and randomly generate binary masks for the cropped patches to define the context region and the region to generate street network."

## 8. Comparability with UrbanMorph

- **Code, weights and licence:**
  - **None found.** The arXiv text has no code or data link.
  - A GitHub search for "DeepStreet street network" returned 0 repositories.
  - The journal version's data-availability statement was not read (T&F not accessed).
  - The figshare record 10.6084/m9.figshare.19606960 belongs to the **2022 IJGIS paper**, not to this one. That archive's `networks.py` contains a one-stage `InpaintSANet` used as "Model 1, following Fang et al. (2021)" (see `fang2022_topography.md`). It is not this paper's generator and was not verified to be.
- **macOS without CUDA:** not applicable, as there is no code.
- **Inputs we could supply:**
  - SRTM-like elevation and aspect: yes, from our DEM;
  - surrounding streets: yes;
  - junctions and pattern labels: only from real OSM, which is privileged;
  - water: no channel exists.
- **Fair comparison:** completion only (a masked region within a 512 m tile, given the surrounding streets). Model 1 is the only variant without oracle guidance.
- **Retraining:** a full reimplementation would be needed.
- **Effort (estimate):** 40–70 h to reimplement and train. **Not recommended**, because Fang 2022 supersedes it on terrain (more terrain channels, a GE module, released weights).

## 9. Ideas we could adopt

- **Report performance against how much context is given.** Their Figure 6 plots l2 error against context coverage. For UrbanMorph completion or hybrid experiments, plot the realism and terrain metrics against the fraction of the tile pre-seeded with real streets or anchors.
- **Dose-response for guidance.** The Model 2 sensitivity runs at 30, 60 and 90 % of junctions kept show how an oracle hint inflates scores. We can use the same design to quantify how much our anchors or permitted crossings help, and to keep "oracle" and "non-oracle" results apart.
- **City-level hold-out across countries** (Prague → Barcelona), a stronger form of geographic hold-out than a within-city split.
- **Caution for our own write-up:** avoid "difficult for people to discern" realism claims without a user study or a distributional metric.
