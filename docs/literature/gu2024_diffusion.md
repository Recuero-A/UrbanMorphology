# Gu, Zhang, Lyu, Ge 2024: Generating Urban Road Networks with Conditional Diffusion Models

## 1. Header

- **Citation:** Xiaoyan Gu, Mengmeng Zhang, Jinxin Lyu, Quansheng Ge. "Generating Urban Road Networks with Conditional Diffusion Models." *ISPRS International Journal of Geo-Information* 13(6):203, 2024. Received 6 Feb 2024, accepted 14 Jun 2024, published 16 Jun 2024. Licence: CC BY 4.0.
- **DOI:** 10.3390/ijgi13060203
- **Code and data:** figshare "diffusion-road-dev", DOI 10.6084/m9.figshare.24618942.v1, CC BY 4.0, published 2024-06-11. Inspected concretely; see section 8.
- **URLs opened:**
  - https://www.mdpi.com/2220-9964/13/6/203 and https://www.mdpi.com/2220-9964/13/6/203/htm: HTTP 403.
  - https://www.mdpi.com/2220-9964/13/6/203/pdf?version=1718518711: HTTP 403.
  - https://mdpi-res.com/d_attachment/ijgi/ijgi-13-00203/article_deploy/ijgi-13-00203.pdf?version=1718518711: the published PDF, read in full (19 pages).
  - https://api.openalex.org/works/doi:10.3390/ijgi13060203 (OA status and PDF URL).
  - https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/ijgi13060203 (abstract).
  - https://api.figshare.com/v2/articles/24618942 (record metadata, file list, licence).
  - https://ndownloader.figshare.com/files/44420231 (the 367.5 MB `diffusion-road-dev.7z`). I listed the whole archive and read the 9 code files. The image data were not opened.
  - Public record page (from the API, not opened separately): https://figshare.com/articles/dataset/diffusion-road-dev/24618942
- **Access level:** **FT** (full text read). Supplementary tables S1–S9 were not read.

## 2. BibTeX

```bibtex
@article{gu2024generating,
  author  = {Gu, Xiaoyan and Zhang, Mengmeng and Lyu, Jinxin and Ge, Quansheng},
  title   = {Generating Urban Road Networks with Conditional Diffusion Models},
  journal = {ISPRS International Journal of Geo-Information},
  year    = {2024},
  volume  = {13},
  number  = {6},
  pages   = {203},
  doi     = {10.3390/ijgi13060203}
}
@misc{gu2024diffusionroaddev,
  author    = {Gu, Mono},
  title     = {diffusion-road-dev},
  publisher = {figshare},
  year      = {2024},
  doi       = {10.6084/m9.figshare.24618942.v1},
  note      = {Dataset and Python code, CC BY 4.0}
}
```

The figshare author name is "Mono Gu" exactly as the record gives it.

## 3. What it does

The model is a conditional DDPM that paints a binary road raster for a 1 km × 1 km tile, at 128 × 128 px (7.81 m/px). It is conditioned on four "context factors" stacked as image channels: land use (9 one-hot class images), elevation, slope, and road-intersection pixels. The generated raster is thresholded, thinned and vectorised into a road network. The model is trained and tested separately for each of five US cities: Chicago, Los Angeles, New York, Phoenix and Washington. The paper also checks whether water, "mountains" and vegetation suppress generated road length the way they suppress real road length.

## 4. Learned or procedural, and HOW

- **Learned.** Conditional **diffusion**: DDPM (Ho et al.), adapted to conditioning following Saharia et al. (SR3).
  - The denoiser is a U-Net with self-attention.
  - In the released code, the context channels are **concatenated with the noisy road image at the U-Net input** (`models.py`: `input_channels = channels + ctx_channels`; `torch.cat((context, x), dim=1)`).
  - T = 1000 steps, linear β schedule 1e-4 to 0.02 (code), batch size 32.
  - `config.py` sets 50 epochs, and `main.py` uses Adam with lr 1e-3.
  - The paper gives an L2 noise-prediction loss (Eq. 9), but `main.py` trains with `loss_type="huber"`. This is a paper/code discrepancy.
- **Training data:**
  - OSM roads of six types: motorway, trunk, primary, secondary, tertiary, residential.
  - Esri Sentinel-2 10 m LULC with 9 classes.
  - SRTM 30 m DEM, with slope derived from it.
  - Per city, 10,000 random 1 km squares, keeping squares with more than 2 km of road.
  - In the released archive, each city has exactly 10,000 patches. Train/test: Chicago 7,893/2,107; Los Angeles 7,951/2,049; New York 7,824/2,176; Phoenix 8,039/1,961; Washington 7,896/2,104. These counts come from my listing of the archive.
  - **One model per city.** There is no cross-city training.
- **Raster, not vector.**
  - Pixels become a graph in three steps (Section 3.3): binarise at 127, thin with Zhan's "hybrid line thinning" (Auto-Carto 11, 1993), then vectorise the thinned lines and build road graphs.
  - **The thinning and vectorisation code is not in the figshare archive**, and neither is the DAD/DARL network-metric code. I grepped all 9 `.py` files for thin, skeleton, vector, degree and length.

## 5. Inputs/outputs and how terrain enters

- **Inputs:** 1 × 128 × 128 images per factor (Table 1).
  - Land use: 9 binary images, one per class (water, trees, flooded vegetation, crops, built area, bare ground, snow/ice, clouds, rangeland).
  - Elevation: SRTM 30 m, bilinearly resampled to 7.81 m.
  - Slope.
  - Intersections: a binary image of "Nodes with a degree greater than 1" of the *target* OSM network.
  - Model 5 therefore has 12 context channels: landuse0–8, dem, slope, intersects.
- **Terrain:**
  - Elevation and slope are raw channels.
  - Each is **min–max normalised to 0–255 per city** (Eq. 1), so absolute elevation is comparable within a city but not across cities.
  - Water enters only as one of the 9 land-use one-hot channels. There is no explicit water mask and no bridge handling.
- **Output:** a grey-scale road image, then a binary road raster, then vector lines. There are no road classes and no widths.
- **No surrounding streets:** the authors stress this. However, in Model 5 the **intersection channel puts the target network's junction locations into the condition**. This is my reading of Section 3.1: degree > 1 covers every non-dead-end node. The task is therefore closer to "reconstruct the streets given their junctions and the terrain" than to unconditional generation.

## 6. Evaluation

- **Datasets and split:**
  - Five US cities, a separate model each.
  - Within each city, an 80/20 split by a "spatially non-overlapping" procedure (Section 4.1): pick a random square, mark it and all squares intersecting it as test, and repeat until the test set exceeds 20%.
  - **No geographic hold-out across cities.**
  - My inference from the procedure as written (not checked on the data): only squares intersecting a *seed* square are moved to test. A training square can therefore still overlap a non-seed test square, so some train/test overlap may remain.
- **Image metrics (Section 3.4.1):**
  - **FID.**
  - **F1:** pixels are the samples, F1 is computed per image and averaged.
  - **IoU:** overlap area over union area of road pixels, per image, averaged.
- **Network metrics (Section 3.4.2):**
  - **DAD:** average degree of the generated networks minus that of the real networks in the test regions; 0 is best.
  - **DARL:** mean total road length (km) per generated network minus the same for the real networks; 0 is best.
- **Baseline and ablations:**
  - The only baseline is an unconditional diffusion model with no context.
  - Ablations drop one factor each: Model 1 has no land use, Model 2 no elevation, Model 3 no slope, Model 4 no intersections. Model 5 uses all four.
  - Ranking: 6 to 1 points per row of Table 3, summed over 25 rows (Fig. 8).
- **Results (Table 3):**
  - Model 5, Chicago: FID 34.79, F1 74.62%, IoU 61.53%.
  - Model 5, Phoenix, which the paper calls "inland and mountainous": FID 102.54, F1 49.98%, IoU 34.56%.
  - Model 4 (no intersections) has F1 of only 15.68–20.51% across cities. The unconditional baseline has F1 of 3.70–25.98%.
- **Terrain analysis (Section 4.3):**
  - Linear regressions of road length per patch on water, "mountain" and vegetation area, for generated and real patches.
  - "Mountains" are extracted from the DEM by mathematical morphology (Sathyamoorthy 2006).
  - Coefficients and R² are compared. All p < 0.01.
  - Water coefficient: −6.8124 generated vs −6.4778 real.
  - Mountain: −2.5283 generated vs −2.8406 real. R² is 0.0648 for Table S6 (generated) and 0.0796 for Table S7 (real), in that order, and the text says the generated model "achieved a lower R2".
  - Vegetation: −2.6512 generated vs −4.1866 real; R² 0.0773 vs 0.2147.
- **Statistics and uncertainty:**
  - No seeds, no repeated runs and no confidence intervals on Table 3.
  - The only p-values are from the regressions.
  - Model selection is "the one that achieved the lowest loss" (Section 4.2).
- **Runtime:** about 2 h training and 3 h testing per city, on 2 × Xeon Gold 6230 and 4 × A100 (Section 4.1).
- **Released FID code (`assess.py`):**
  - The mean is taken as `np.mean(act1)` with no axis, i.e. a scalar rather than a per-feature mean.
  - Features are the outputs of torchvision `inception_v3` (pretrained) called in eval mode, which are the 1000-class logits, not pool3 features.
  - This is a non-standard FID. Whether the published FIDs used this script is **not verified**.

## 7. Verbatim quotes

- p. 2, Section 1: "Therefore, we selected four factors to represent these aspects: land use, elevation, slope (underlying terrain), and road intersections (human design)."
- p. 3, Section 2.2: "Unlike previous image-inpainting-based works [13,28], we do not need the surrounding road information but only include the geospatial context to generate the road networks for new areas."
- p. 5, Section 3.1: "Nodes with a degree greater than 1 were extracted from the road networks as intersections."
- p. 5, Section 3.1: "The DEM was obtained from the Shuttle Radar Topography Mission (SRTM), which has a spatial resolution of 30 m. The slope data were calculated based on the elevations."
- p. 5, Section 3.1: "We randomly generated 10,000 squares with a size of 1 km × 1 km within each city. To remove areas where the road density is too low, we only retained the squares with a road length that is larger than 2 km."
- p. 5, Section 3.1, Eq. 1: "where V is the raw elevation or slope, Vnorm is the normalized value, and Vmin and Vmax are the minimum and maximum values of a city, respectively."
- p. 7, Section 3.3: "Then, we used the thin algorithm proposed by Zhan [47] to reduce the width of the linear features of the images. Lastly, we performed vectorization for the thinned images to generate road vector lines."
- p. 9, Section 4.1: "Because the training and testing sets were spatially non-overlapping, our testing could be performed completely on the untouched spatial area."
- p. 10, Section 4.2: "For each city, we trained diffusion models and chose the one that achieved the lowest loss as the final road network generator."
- p. 10, Section 4.2: "Another reason is that curved road courses are harder to predict compared with straight lines, as our training datasets in the five cities contain more straight roads than curved roads."
- p. 11, Section 4.3: "It is worth noting that our model does not seem to be effective at predicting the bridge in case 4."
- p. 11, Section 4.3: "In all four cases, the roads generated by our model were mainly located in gently sloping areas, consistent with the real situation."
- p. 13, Section 4.3: "When compared to the real cases, the regression model using generated results decreased the coefficient and improved the R2, indicating that our model enhanced the obstructive effect of water on road network generation."
- p. 14, Section 4.4: "Model 4 is the worst-performing model except for the baseline, implying that intersections are the most important guide for the road generation task."
- p. 16, Section 5: "There are two limitations in our research at present. The first is that we generate a local road network within a grid with the size of 1 km × 1 km each time. The second is that the resolution of the generated image is limited to 128 × 128 pixels."
- p. 17, Data Availability: "The codes and data that support this work are available at the figure share link: https://doi.org/10.6084/m9.figshare.24618942.v1"

## 8. Comparability with UrbanMorph

**What the figshare record contains** (API plus a full listing of the archive):
- **One file:** `diffusion-road-dev.7z`, 367,503,897 bytes, MD5 8c064c8b597ee80de6ab52be49483ae2 (matches the API).
- **Licence:** CC BY 4.0 on the record. That the OSM-derived content is relicensed this way is not verified against ODbL.
- **Code** (9 files, about 27 KB): `main.py`, `models.py` (U-Net), `utils.py`, `data_loader.py`, `assess.py` (FID, F1, IoU only), `config.py`, `logger.py`, `run.sh` (`python main.py --task train|test`), `requirements.txt`.
- **Framework:** PyTorch.
  - `requirements.txt`: torch==1.13.1, torchvision==0.14.1, datasets==2.12.0, einops==0.7.0, numpy==1.24.3, scipy==1.9.3, scikit_learn==1.1.3.
- **Data:**
  - `dataset/<city>/{road,dem,slope,intersects,landuse,landuse0..landuse8}/{train,test}/*.png`: 14 folders × 10,000 PNGs per city.
  - `road-network/<city>/edges_prj.*, nodes_prj.*`: projected OSM shapefiles for the 5 cities.
- **Trained weights: none.**
  - No `.pt`, `.pth` or `.ckpt` file is in the archive.
  - `config.py` refers to `chicago_weight_1700657571_4.pt` (epoch 4) for Model 5 only; the other cities' entries are empty and models 1–4 are commented out.
- **Missing:** the thinning/vectorisation post-processing and the DAD/DARL code.

**Runs on macOS without CUDA?**
- Probably, with edits; not tested.
- `Diffusion.__init__` hard-codes `device="cuda"`. `utils.device()` falls back to CPU but is not used by `main.py`.
- torch 1.13.1 has macOS arm64 builds, and its MPS backend was early-stage. Pin `pin_memory`/`num_workers` as needed.
- Sampling needs 1000 U-Net passes per batch at 128², with base width 128 and multipliers (1, 2, 4). Expect minutes per tile on CPU (my estimate).

**Inputs we could supply:**
- 128 px × 4 m = **512 m, exactly our tile**. Our DEM and slope map directly, though their normalisation is per-city min–max.
- Water would go in a land-use channel. For the other 8 LULC classes, use Esri 10 m LULC, or retrain on a reduced channel set.
- Intersections: the full-junction channel is an **oracle** that UrbanMorph does not receive. We could supply only our boundary anchors as junction pixels, but that needs retraining because the training distribution differs.
- No surrounding streets are used.

**Fair comparison:**
- **"Model 4" configuration (land use + DEM + slope, no intersections):** this is the matched terrain-only, unconditional-growth setting.
- **"Model 5" with ground-truth junctions:** report it only as a privileged upper reference, clearly labelled.
- **Anchors as junctions:** a "Model 5-anchors" variant, retrained, would be closest to our anchor protocol. It is our own design.

**Retraining:** required. There are no weights, the region and DEM differ (SRTM 30 m vs our 4 m LiDAR), and street imprints in a 4 m DEM create a leakage risk that their 30 m DEM mostly avoids.

**Effort (estimate):**
- Port device handling to MPS/CPU: 2–4 h.
- Tile export of our 512 m tiles to their PNG layout: 6–10 h.
- Our own threshold → skeletonise (skimage) → graph extraction → simplification: 6–10 h.
- Hook into our metrics: 3–5 h.
- **Human time: about 20–30 h.**
- **Compute:** they report about 2 h per city on 4 × A100. On Apple Silicon (MPS), expect roughly 1–3 days per configuration; this is an estimate, untested.

**Verdict:** the most runnable *terrain-conditioned learned* generator found (code, data and CC BY), but raster, per-city, without weights, and with an oracle junction channel in its best configuration.

## 9. Ideas we could adopt

1. **Obstruction regression** as a cheap terrain-adaptation statistic.
   - Per tile, regress generated road length (or density) on the area of water and of steep terrain (slope > threshold), and do the same for real OSM.
   - Compare slope coefficients and R² with bootstrap CIs. This is a coarse complement to our grade distribution and contour-following metrics.
2. **Drop-one-factor ablation** (DEM, slope, water) under the same metrics. It shows which terrain channel actually drives adaptation.
3. **Leakage lesson.** Their "intersections are the most important" finding shows how much any target-derived conditioning inflates pixel scores: F1 is 16–21% without junctions and 44–75% with them.
   - Our evaluation should never feed target-network junctions.
   - DEM-imprint leakage needs the same suspicion.
4. **Stratified reporting** by pattern (regular, irregular, suburban) and by terrain type, as their qualitative Section 4.2 does but made quantitative.
5. **Spatial block split.** Copy the idea, but enforce a buffer so that no training tile overlaps any test tile, which their seed-plus-neighbours rule does not guarantee.
6. **Do not reuse their FID script.** If image FID is reported at all, use a standard pool3 implementation.
