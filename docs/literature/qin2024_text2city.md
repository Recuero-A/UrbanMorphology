# Qin, Zhao, Sheng, Lau 2024: Text2City, One-Stage Text-Driven Urban Layout Regeneration

**NEW: not in the M18 list.** It is a learned road-plus-building layout generator from 2024 with released code. The M18 literature matrix lists it only by title ("Text2City ... not assessed").

## 1. Header

- **Citation:** Yiming Qin, Nanxuan Zhao, Bin Sheng, Rynson W. H. Lau. "Text2City: One-Stage Text-Driven Urban Layout Regeneration." *Proceedings of the AAAI Conference on Artificial Intelligence* 38(5):4578–4586 (AAAI-24), published 24 Mar 2024. Open access ("diamond" per OpenAlex).
- **DOI:** 10.1609/aaai.v38i5.28257
- **URLs opened:**
  - https://api.openalex.org/works?search=Text2City%20One-Stage%20Text-Driven%20Urban%20Layout%20Regeneration: metadata and the OA PDF URL.
  - https://ojs.aaai.org/index.php/AAAI/article/download/28257/28507: main paper PDF, read in full.
  - https://github.com/LittleQBerry/Text2City (WebFetch: file list, licence sidebar, commits).
  - https://raw.githubusercontent.com/LittleQBerry/Text2City/HEAD/README.md
  - https://raw.githubusercontent.com/LittleQBerry/Text2City/HEAD/LICENSE (and LICENSE.md, LICENSE.txt): all 404.
- **Access level:** **FT** for the main paper. The supplementary material, which holds the dataset and metric details, was **not read**.

## 2. BibTeX

```bibtex
@inproceedings{qin2024text2city,
  author    = {Qin, Yiming and Zhao, Nanxuan and Sheng, Bin and Lau, Rynson W. H.},
  title     = {{Text2City}: One-Stage Text-Driven Urban Layout Regeneration},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
  volume    = {38},
  number    = {5},
  pages     = {4578--4586},
  doi       = {10.1609/aaai.v38i5.28257}
}
```

## 3. What it does

Given a target region inside a 1 km² urban layout map, a text description of the desired layout, and the surrounding context (existing roads, buildings and functions), Text2City jointly regenerates the road layer and the building layer of the target region. It uses guided DDPMs, with prior knowledge exchanged between the road and building streams. The surrounding context is blended in at every denoising step, in the style of inpainting.

## 4. Learned or procedural, and HOW

- **Learned:** conditioned **diffusion** (DDPM, built on OpenAI guided-diffusion).
  - Guidance from a CLIP model fine-tuned on layout–text pairs: attribute gradients (Eq. 5–7).
  - An "interactive & enhanced guidance" (IEG) module exchanges road/building priors.
  - Constraints at attribute level (CLIP cosine), geometry level (isoperimetric quotient of building footprints) and pixel level (range loss).
- **Training data:**
  - 147K regions covering most of Greater London, from OSM.
  - Rendered as layered images (region/function, road, building). The road layer is drawn "based on the road hierarchy" with class-specific widths and colours.
  - Each layout map covers 1 km².
  - Image resolution: the README uses the 256 × 256 guided-diffusion checkpoint, which suggests 256 px per 1 km (about 3.9 m/px). This is my inference, **not verified**.
  - Paris and Shanghai are used only for fine-tuning demonstrations.
- **Raster.** The paper does not vectorise as part of the method: "Additionally, the urban layout could be vectorized like (Chu et al. 2019)" (p. 4581). It does say layouts "need to be vectorized before computing urban layout metrics" (p. 4582); the procedure is in the supplement (not read).

## 5. Inputs/outputs and how terrain enters

- **Inputs:**
  - The target region mask R.
  - Text, split into road and building descriptions.
  - The surrounding context S = (S_r, S_b): road and building layouts outside R, blended in at each step (Eq. 8).
  - Region functions (land use) are part of the rendered layout.
- **Output:** a road raster (hierarchy by colour/width) plus a building raster for the target region, composited with the context.
- **Terrain:** **none.** No DEM, slope or water input is described. Water appears only as rendered context, if at all; not verified.

## 6. Evaluation

- **Data and split:** London regions. The train/test split and any geographic hold-out are **not verified** in the main text.
- **Metrics (p. 4582):**
  - **SSIM, FID, WD** (Wasserstein distance) for quality and diversity.
  - **RLS** (road layout similarity, after AlHalawani et al. 2014) and **BLS** (building layout similarity, after Chen et al. 2021), both computed on vectorised layouts. The exact definitions are in the supplement (not read).
- **Baselines:**
  - Procedural: CityEngine, IPSM (Chen 2008), UrbanBrush (manual tuning by the authors).
  - Learned image methods: TDANet, StyleCLIP, BD (Blended Diffusion), SDv2.
- **Results:** Table 1 (Ours) SSIM 0.89, FID 77.71, WD 0.04, RLS 19.37, BLS 0.24. Ablations are in Table 3.
- **User study:** 20 participants, 15 questions, ranking 4 methods on Rationality, Harmony (with the surroundings) and Match (with the text). Ranking-weighted scores with variance are in Table 2.
- **Statistics:** no CIs or seeds beyond the user-study variance.
- **Runtime:** about 1.5 min per regeneration on an RTX 2080 Ti (p. 4582).

## 7. Verbatim quotes

- p. 4579, Related Work: "However, traditional methods generate urban layouts in a two-stage manner which do not consider the interaction between roads and buildings, and always start from scratch ignoring the surrounding context."
- p. 4580, Our Dataset: "(1) 147K regions covering most of the Greater London, UK."
- p. 4581, Text2City: "Note that blending the M̃rt−1 , rt−1 and Srt−1 produces a result outside the current manifold, which is corrected in the next diffusion step that projects the result to the t − 2 step manifold, thereby ensuring coherence." The symbols are as pdftotext produced them; the original's subscripts and superscripts were flattened.
- p. 4581: "Additionally, the urban layout could be vectorized like (Chu et al. 2019), extending its representation beyond the current resolution."
- p. 4582, Experiments: "Note that the regenerated urban layouts need to be vectorized before computing urban layout metrics."
- p. 4582, Implementation Details: "For our experiments, each urban layout map covers 1km2 and the central region with a single function is taken as the target region. On an RTX 2080ti, it takes about one and a half minutes to complete a text-driven urban layout regeneration using our method."
- p. 4584, Limitations: "One limitation is the long inference time due to denoising and joint optimization calculation."

## 8. Comparability with UrbanMorph

- **Code:** https://github.com/LittleQBerry/Text2City.
  - 3 commits; last push 2024-06-27 per the GitHub search API.
  - Python 3.8, torch 1.12.0, opencv 4.6.0, numpy 1.22.3.
  - Files: `guided_diffusion/`, `utils/`, `arguments.py`, `augmentations.py`, `create_dataset.py`, `losses.py`, `main.py`, `net.py`.
  - **No licence file**, so all rights are reserved by default and permission is needed.
- **Weights:** **no Text2City weights or fine-tuned CLIP weights are linked.** Only the OpenAI base checkpoint `256x256_diffusion_uncond.pt` is referenced.
- **Dataset:** on Google Drive (link in the README; not opened).
- **macOS without CUDA:** guided-diffusion is plain PyTorch and probably runs on CPU/MPS after device edits; **not verified**. At 1.5 min per sample on a 2080 Ti, expect much slower on CPU (estimate).
- **Inputs we could supply:**
  - Surrounding streets: yes, from OSM.
  - Target-region mask: yes.
  - Text: we would have to template it.
  - **DEM and water: not accepted.** Adding them would mean new conditioning channels plus retraining, i.e. our own design.
- **Fair comparison:** **completion given surrounding streets** (inpainting), not unconditional growth. It would be fair only in an UrbanMorph completion experiment, and it ignores terrain, so it is a terrain-blind completion reference.
- **Retraining:** required (London-trained; no weights; different resolution).
- **Effort (estimate):** 30–50 h human time (data rendering in their format, retraining both the CLIP fine-tune and the DDPM, vectorisation) plus multi-day compute. **Low priority:** terrain-blind, raster, unlicensed.

## 9. Ideas we could adopt

1. **User-study design for completion:** rank methods on *Rationality*, *Harmony with surroundings* and (for us) *Terrain plausibility*. Use 20 or more participants, ranked comparisons, and report variance. A "harmony with surroundings" criterion fits any completion experiment.
2. **RLS-style vector similarity** (AlHalawani et al. 2014) as a complement to our network-statistics distances. To be checked in that paper before adoption.
3. **Text-free stratification:** their per-region function labels suggest stratifying our realism metrics by land use of the target area (residential vs mixed), so that terrain effects are not confounded with function.
