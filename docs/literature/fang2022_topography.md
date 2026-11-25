# Fang et al. 2022: topography-aware GAN street generation (IJGIS)

## 1. Header

- **Citation:** Zhou Fang, Jiaxin Qi, Lubin Fan, Jianqiang Huang, Ying Jin, Tianren Yang. "A topography-aware approach to the automatic generation of urban road networks." *International Journal of Geographical Information Science* 36(10):2035–2059, 2022. Published online 1 Jun 2022; print issue 3 Oct 2022 (Crossref). Received 11 Aug 2021, accepted 28 Apr 2022.
- **DOI:** 10.1080/13658816.2022.2072849
- **Code and data DOI:** 10.6084/m9.figshare.19606960 (v1), titled "Data and Code for IJGIS paper".
- **URLs opened (2026-11-25):**
  - https://lbfan.oss-cn-hangzhou.aliyuncs.com/personal_webpage/publications/2022-ijis-auto_road_network.pdf (author-hosted final PDF with T&F cover page; read in full)
  - https://api.crossref.org/works/10.1080/13658816.2022.2072849 (metadata)
  - https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2072849 and https://www.tandfonline.com/doi/suppl/10.1080/13658816.2022.2072849 (both **403**; Supplemental Materials A–D **not read**)
  - https://api.figshare.com/v2/articles/search (DOI lookup) and https://api.figshare.com/v2/articles/19606960 (record metadata)
  - https://doi.org/10.6084/m9.figshare.19606960 → https://figshare.com/articles/software/Data_and_Code_for_IJGIS_paper/19606960
  - https://ndownloader.figshare.com/files/34826926 (`Final_Code.zip`). **Byte-range reads only:**
    - the zip central directory (full file listing);
    - the text and code files named in §8;
    - one example sample `.npz` (array shapes only);
    - the GeoTIFF headers of five base maps;
    - the first ~0.6 MB of `T3_pretrined_modelG.pth`, scanned for parameter names with a regex. It was not unpickled or loaded.
  - Nothing was executed.
- **Access level:** **FT** for the main paper. The supplement is not read. The released archive was inspected as described above.
- **Page numbers** below are journal pages (2035–2059). PDF page *n* = journal page 2033 + *n*, because PDF page 1 is the T&F cover.

## 2. BibTeX

```bibtex
@article{fang2022topography,
  author  = {Fang, Zhou and Qi, Jiaxin and Fan, Lubin and Huang, Jianqiang and Jin, Ying and Yang, Tianren},
  title   = {A topography-aware approach to the automatic generation of urban road networks},
  journal = {International Journal of Geographical Information Science},
  volume  = {36},
  number  = {10},
  pages   = {2035--2059},
  year    = {2022},
  doi     = {10.1080/13658816.2022.2072849}
}

@misc{fang2022topography_code,
  author    = {Fang, Zhou and Yang, Tianren},
  title     = {Data and Code for {IJGIS} paper},
  publisher = {figshare},
  year      = {2022},
  doi       = {10.6084/m9.figshare.19606960.v1},
  note      = {Software. CC BY 4.0}
}
```

## 3. What it does

It is a GAN image-inpainting model that fills a rectangular hole in a 512 m × 512 m street raster. It is conditioned on four things: the surrounding streets, junction "guidance" heat maps, and seven DEM-derived channels (elevation, slope, aspect, and hillshades from four azimuths). The paper's contribution is a "geo-extractor" (GE) gate/attention module plus "geo-merging" (GM) bypaths. Together they let the network use terrain where context streets are sparse, as in hilly fringes. It is the closest learned precedent for DEM-conditioned *urban* street generation at UrbanMorph's tile size. But it is **raster completion with privileged junctions**, not unconditional growth.

## 4. Learned or procedural, and how

- **Learned, GAN, raster.** The base is Yu et al. 2019 gated-convolution inpainting (DeepFill v2 style) with an SN-PatchGAN discriminator and hinge loss (pp. 2042–2046).
- **Three stages (Model 3):**
  1. **GE module:** an HRNet-style multi-resolution network over the 7 terrain channels. It outputs a gate map G_topo and per-channel attention A_topo, and the processed terrain is In_topo = Raw_topo · A_topo · G_topo (Algorithm 1, p. 2046).
  2. **Coarse stage:** a dilated-convolution path over the masked streets, guidance and mask, plus a GM bypath over terrain and guidance. The GM bypath also predicts a **vertices heat map**.
  3. **Refinement:** dilated-convolution, contextual-attention and GM paths feed one decoder.
- **Loss:**
  - L_recon = a·L1(vertices) + b·L1(coarse) + c·L1(refined) (Eq. 1);
  - hinge adversarial loss (Eqs. 2–3);
  - the values of a, b, c, k and l are not given in the main text (not verified).
- **Baselines in the paper:**
  - Model 1: one-stage inpainting from Fang et al. 2021, with the 7 terrain channels simply concatenated;
  - Model 2: two-stage Yu et al. 2019;
  - Model 3: the full method (Table 1, p. 2047).
- **Training data:**
  - OSM roads (motorway down to residential) rasterised with a uniform width;
  - junction points of degree 1, 3, 4 and 5 and all vertices rasterised as Gaussian heat maps (σ = 10 px);
  - 512 m crops centred at random points within 100 m cells that have more than 10 % road coverage, downsampled to **256 × 256 px (2 m/px)**;
  - **about 56,000 samples** from Florence, Perugia, Rome and Siena (pp. 2041–2042).
- **Training setup:** "Python v3.6.9, PyTorch v1.3.0, CUDA v10.0" on P100 GPUs, 50 epochs, batch 64, 741 iterations per epoch (p. 2047).
- **Raster to graph:** for the APEMC/APEMI metric only, "we first convert the output image representations of street networks to graphs" (p. 2050). The method is **not described** in the main text and the code is **not in the archive** (see §8). The generator's own output is a raster.

## 5. Inputs/outputs and how terrain enters

- **Input (released code, `GeoStreet_v1.py`; one sample checked):** an `IN` array of shape 256×256×10, stored as int8 and scaled to about [−1, 1]. Channels:
  - 0: road raster (context; the hole is masked);
  - 1–3: elevation, slope, aspect;
  - 4: node map "ND";
  - 5–8: four hillshades;
  - 9: junction guidance.
- **Targets:** `GT_256` (streets) and `GT_64_JC` (vertices/junctions).
- **Terrain:**
  - The raw DEM is "30 by 30 m" (p. 2041), interpolated to 2, 8 and 32 m for the three GE branches.
  - The crop script normalises elevation as (z − city minimum)/400, slope/80, aspect/360 and hillshade/255.
  - The base maps are float32 GeoTIFFs at about 2 m in "Monte Mario (Rome) / Italy zone 1".
  - Terrain reaches the generator **only** through the GE gate and attention and the GM bypaths.
  - The code also derives a binary slope mask (blurred slope > 0.15 after rescaling) that multiplies the junction ground truth used for vertex supervision (`GeoStreet_v1.py`). This is not described in the paper.
- **Water: not an input.** "polyline-based hydrologic features such as waterways and drainage channels are not included in the model training" (p. 2056).
- **Masks:**
  - Paper: 48–64 px rectangles (p. 2042).
  - Released code: `mask_gen` draws 64–96 px rectangles whose centre lies within ±32 px of the tile centre. The hole covers about 6–14 % of the tile (computed from 64²/256² to 96²/256²).
- **Junction guidance inside the hole:** the evaluation code builds `inp_guid = inp_RN * msk4vt + inp_guid_ss * (1 - msk4vt)`, so inside the hole the model receives the **real junction heat map**. The paper says the same: "With inputting ground truth junction locations and surrounding street networks" (p. 2048).
- **Output:** a 256 × 256 street raster (coarse and refined) plus a vertices heat map.

## 6. Evaluation

- **Split: geographic hold-out by city.** "samples extracted from Florence and Perugia are used to train the models, while samples extracted from Rome and Siena are used for validation and testing" (p. 2047). Each side has one flat city and one hilly city. Validation and test are the same two cities (no separate validation city).
- **Test size:** "Our evaluation used 9,033 samples" (p. 2047).
- **Metrics:**
  1. **Pixel MAE / MSE** of the output against the ground truth on the test set (Table 2, p. 2050):

     | Model | MAE | MSE |
     |---|---|---|
     | Model 1 | 0.1248 | 0.2453 |
     | Model 2 | 0.1236 | 0.2447 |
     | Model 3 | 0.1199 | 0.2345 |

     The value range of the images is not stated in the paper. In the code the images are in [−1, 1].
  2. **Space-syntax network metrics** after an image-to-graph conversion (pp. 2050–2051):
     - metric choice (MC) and metric integration (MI) per segment;
     - length-weighted LWMC_k = Σ_i l_{k,i}·MC_{k,i} / Σ_i l_{k,i}, and LWMI likewise (Eqs. 4–5);
     - absolute percentage errors APEMC_k = |LWMĈ_k − LWMC_k| / LWMC_k, and APEMI likewise (Eqs. 6–7; the absolute-value bars are inferred from the name, since the PDF text extraction lost the symbols);
     - the MC/MI radius and the conversion method are not stated in the main text (not verified).
  3. **Flat vs hilly gap:** the share of outputs with APE > 100 %, Rome vs Siena.
     - Model 1: Siena has 17.02 points more than Rome on APEMC (21.93 % vs 4.91 %) and 8.00 points more on APEMI (10.30 % vs 2.30 %).
     - Model 3 narrows the gaps to 3.56 and 0.38 points (p. 2051).
     - Figure 10 shows violin plots.
  4. **Interpretability:**
     - per-sample mean gate (InRatio_topo) and per-channel weights W_k (Eq. 8) on a dense "GEanalysis set" of 6,800 Rome and Siena samples (note 3, p. 2057);
     - finding: slope is weighted most (24–29 %), aspect least (about 5 %), and the weight on slope "is found to be greatly amplified ... when the average slope percent is higher than 9%" (p. 2054).
- **Not reported:** uncertainty, seeds, repeated runs and confidence intervals (none reported); a user study (none); generation runtime (not reported).
- **Baselines:** only internal (Models 1 and 2). No procedural or other learned baselines.

## 7. Verbatim quotes

- p. 2035 (abstract): "Existing deep-learning tools for road network generation have limited applications in flat urban areas due to their overreliance on the geometric and spatial configurations of street networks and inadequate considerations of topographic information."
- p. 2036: "Note that street network generation in this study specifically refers to the prediction of missing segments in a street configuration within a pre-defined region."
- p. 2036: "the model is not designed to accurately reproduce all the road segments (as some of them are arbitrarily designed)"
- p. 2041: "The resolution of the raw DEM data is 30 by 30 m."
- p. 2041: "The width of streets was consistently set as 10 pix (representing 10 m)". This conflicts with the stated 2 m/px; it is reported as written.
- p. 2042: "a center location of the missing area is randomly assigned to generate a rectangular mask (with the width and height being a random number from 48 to 64 pix) within a street network sample of 256 by 256 pix."
- p. 2044: "However, a street network in a hilly urban fringe area is always sparse, thus lacking enough surrounding street network patterns that can be borrowed to infill the missing area."
- p. 2047: "the samples extracted from Florence and Perugia are used to train the models, while samples extracted from Rome and Siena are used for validation and testing."
- p. 2048: "With inputting ground truth junction locations and surrounding street networks, all three models successfully connected the junction nodes using straight-line segments and adapted the generated network to the surrounding network."
- p. 2048: "The performance of all three models is not affected by the topographic inputs (i.e. the seven types of topographic information)." This refers to flat areas.
- p. 2049: "learning-based street network generation lacks good quantitative evaluation metrics, due to there being many possible solutions that differ from the original street network"
- p. 2049: "we provide junction nodes and seven types of topographic information that serve as constraints in street network generation. As a result, the predicted street network is expected to be identical to the original one."
- p. 2050: "we first convert the output image representations of street networks to graphs, where the street network junction nodes and road segments are represented by graph vertices and graph edges, respectively."
- p. 2056: "The training datasets only include two Italian cities (i.e. Florence and Perugia) where the road networks have been developed through the centuries."
- p. 2056: "Unsurprisingly, the current model will poorly predict their street networks." This refers to hilly cities with orthogonal grids, e.g. San Francisco.
- p. 2056: "polyline-based hydrologic features such as waterways and drainage channels are not included in the model training."
- p. 2057: "The data and codes that support the findings of this study are available at the following link: https://doi.org/10.6084/m9.figshare.19606960.v1."

## 8. Comparability with UrbanMorph

### Figshare verdict (resolves the M18_LITERATURE vs M18_BASELINES conflict)

**The record:**
- figshare 19606960 v1, type "software", published 2022-04-16, licence **CC BY 4.0** (API), authors Zhou Fang and Tianren Yang;
- **one file**, `Final_Code.zip`, 946,829,630 bytes, MD5 7e9b39b789488a2a00b9f703beabf96d;
- 554 zip entries, including macOS `._` resource forks.

**Contents:**
- `Section3/`: data generation.
  - Full-city base maps for FL, PR, RM and SN: EM (elevation), SM (slope), AM (aspect), HS0/90/180/270, RN/ExRN (roads), ND, JM, JIDM, SampM. Float32 GeoTIFFs at about 2 m; Florence is 3779 × 2463 px.
  - OSM-derived road and node GeoPackages.
  - Coarse `*_EM_Large.tif` DEMs for Perugia and Siena. Perugia's header shows about 30 m pixels; Siena's header was not read.
  - Six crop scripts (`Crop_Samples_v1_GeoStreet_*.py`) that regenerate the samples.
- `Section5/`: training and evaluation.
  - `train.py`, `evaluation.py`, `setting.yaml`;
  - `lib/` (config, dataset, `networks.py`, models T1–T3, train/validate functions);
  - **pretrained weights for all three models:** `T1/T2/T3_pretrined_modelG.pth` and `..._modelD.pth`. G is 24 MB for T1 and T2 and 69 MB for T3.
- `Section6/`: GE-weight extraction (`get_weights.py`) with T3 weights.
- **Data shipped:** only **10 example samples per city per section** (40 + 20 `.npz` files plus PNG previews), not the ~56k set. The README asks the user to regenerate the rest from the base maps.

**Framework:**
- **PyTorch**, with yacs, tensorboardX, OpenCV, pandas, torchvision and PIL.
- No requirements file. The paper states PyTorch 1.3.0 and CUDA 10.0.
- The checkpoints use the **legacy (pre-1.6) non-zip `torch.save` format**: the file starts with a pickle protocol-2 magic number. Parameter names carry the `module.` prefix from `nn.DataParallel`.

**Gaps that stop a faithful run of the topography-aware Model 3 (T3):**
1. `baseline_modelT3.py` instantiates `InpaintSANet_ZF()`, which **is not defined** in `networks.py` (only `InpaintSANet` and `InpaintSANet_v2` exist). The T3 checkpoint's `endec_model.*` sub-module names (coarse_net, conv_topo, conv_topo_vt_pred, coarse_topobranch_upsample_net, refine_conv_net_topobranch, …) match `InpaintSANet_v2` exactly, so an alias probably works. Tensor shapes were not checked.
2. `function_baseline_T3.py` imports `from .evaluation import decode_preds, compute_nme`. **That module is missing** from the archive; the names come from an HRNet face-landmark template.
3. `function_baseline_T3.py` has **no `train()` function**, so `python train.py --add_note T3` cannot train Model 3. T1 and T2 have `train()`.
4. T3's `validate()` saves images and a pixel-mask log. It **computes neither MAE/MSE nor APEMC/APEMI**. **No image-to-graph or space-syntax code** is in the archive.
5. The full-dataset lists (`URBANNET_train.csv`, `Final_Testlist_Filenames.csv`) are not shipped, and crop points and masks are random. **The exact 9,033-sample test set cannot be rebuilt** (not verified whether the seeds would reproduce it).
6. The mask size in the code (64–96 px) differs from the paper (48–64 px).

**Conclusion:**
- **M18_LITERATURE is right that data and code exist under CC BY 4.0, and weights exist too.**
- **M18_BASELINES's narrower claim still holds:** no learned *vector* generator with code and weights accepts a DEM.
- Its sentence "Learned baseline: none is clean" should be qualified. Fang 2022 is a learned, terrain-conditioned, **raster, completion-only** model with weights. The paper's own Model 3 is not fully runnable as released (gaps 1–4).
- Licence: the record is CC BY 4.0. The data and weights derive from OSM (ODbL). Whether redistributed derivatives need ODbL attribution or share-alike is not verified.

### macOS without CUDA

**Not as shipped.**
- `.cuda()`, `torch.cuda.FloatTensor` and `nn.DataParallel(...).cuda()` are hard-coded in `train.py`, `evaluation.py` and every `function_baseline_T*.py`.
- `torch.load` has no `map_location`.
- Deprecated APIs: `MSELoss(size_average=True)`, and `vgg16(pretrained=True)` in the unused `P_net`.

After patching, CPU inference should work, since the layers are standard conv, norm and interpolation (estimate; not tested). MPS is likely but not verified. Loading the legacy-format checkpoints with current PyTorch (`weights_only` default) is not verified.

### Inputs we could supply

| Input | Can we supply it? |
|---|---|
| DEM window | **Yes.** Resample 4 m to 2 m over a 512 m tile, which gives exactly 256 px. Apply their normalisations. To match the training distribution, also run a variant that first degrades our DEM to about 30 m, as their training DEM was. |
| Slope, aspect, 4 hillshades | **Yes** (derived). The hillshade sun altitude is not verified. |
| Surrounding streets | Yes, from OSM, rasterised the same way. |
| Junction guidance inside the hole | **Only from the real OSM network.** That is privileged information: the model sees the true junction positions. We could zero it, but that is out of distribution because the model was trained with it. |
| Water | **No input exists.** |
| Boundary | Not applicable. |

### Which comparison is fair

- **Only completion:** a 128–192 m hole near the centre of a 512 m tile, given everything else.
- It is **not** a fair baseline for UrbanMorph's unconditional growth over a tile from anchors.
- A fair protocol is a "hole-infill" task:
  - mask the same hole in real OSM;
  - give UrbanMorph the surrounding graph as fixed context and let it grow only inside the hole;
  - give Fang the same context, with the junction channel either privileged (labelled as an oracle) or zeroed;
  - skeletonise Fang's raster with one fixed published pipeline;
  - score both with the same graph and terrain metrics inside the hole.
- Domain shift: the weights are trained on two Italian cities.

### Retraining

- Needed for a fair test on our regions.
- It requires writing the missing T3 `train()` from T1/T2 and Eqs. 1–3, with unknown loss weights a, b, c, k and l.
- Compute: the paper used 8 P100 GPUs (the "×" is lost in text extraction) for 50 epochs of 741 iterations at batch 64. That is about 47.4k samples per epoch, my arithmetic. Apple Silicon time is not verified.

### Effort (estimates)

| Task | Hours |
|---|---|
| Patch CUDA to CPU/MPS and `map_location`, alias `InpaintSANet_ZF`, stub the missing import, write an inference script | 4–7 |
| Build 10-channel inputs from our DEM and OSM with their normalisations (read the 49 KB crop script) | 6–10 |
| Raster-to-graph conversion (skeletonise, prune, simplify) plus the hole-infill harness on our side | 6–10 |
| **Total, pretrained weights, completion-only** | **about 16–27** |
| Retraining on our cities (write T3 `train()`, regenerate the dataset, tune, plus days of compute) | **+30–50** |

## 9. Ideas we could adopt

- **City-level geographic hold-out, with one flat and one hilly city on each side.** Report flat-vs-hilly *gaps* rather than averages alone. Their headline result is a shrinking gap, 17.02 → 3.56 points.
- **Network-level error on reconstructed graphs.** Length-weighted space-syntax choice and integration, with an absolute percentage error against the real network, as a realism/accessibility check. We would add a fixed radius and state how unreachable pairs are handled.
- **"Share of samples with error > 100 %" as a tail metric.** It is more informative than the mean on hard tiles.
- **Terrain-reliance diagnostics.** Their gate (InRatio_topo) against slope found a threshold near 9 % slope (5.14°). We could similarly plot our terrain-potential influence, or our grade distribution, against local slope, and check whether a threshold appears.
- **Junction-oracle leakage control.** Their completion conditions on ground-truth junctions. Any comparison should separate "oracle junction" from "no-junction" settings, and our own evaluation must not feed real junctions.
- **DEM resolution matching.** They trained on 30 m DEMs. When comparing against them, or against Boeing-style grades, degrade our 4 m LiDAR DEM to 30 m so that terrain effects are compared like for like.
