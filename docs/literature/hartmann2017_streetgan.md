# Hartmann et al. 2017: StreetGAN

## 1. Header

- **Citation:** Stefan Hartmann, Michael Weinmann, Raoul Wessel, Reinhard Klein (University of Bonn). "StreetGAN: Towards Road Network Synthesis with Generative Adversarial Networks." In *WSCG 2017: Full Papers Proceedings, 25th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision* (in co-operation with the EUROGRAPHICS Association). Computer Science Research Notes CSRN 2701, pp. 133–142. Václav Skala – UNION Agency, 2017.
- **DOI:** none found. The repository handle is http://hdl.handle.net/11025/29554.
- **URLs opened (2026-11-25):**
  - https://dspace.zcu.cz/items/5fb1aa2d-caf0-470c-af2f-7546b9b5beea (University of West Bohemia repository; "openAccess", "Peer-reviewed", "publishedVersion")
  - https://dspace.zcu.cz/items/5fb1aa2d-caf0-470c-af2f-7546b9b5beea/full (full Dublin Core metadata, including "p. 133-142")
  - https://dspace.zcu.cz/bitstreams/8fe6a37a-3b57-451e-b2f8-c7d886de64be/download (full-text PDF, 10 pages)
  - https://otik.zcu.cz/bitstream/11025/29554/1/Hartmann.pdf (old link from search results; host did not resolve)
  - GitHub search API for "StreetGAN"; https://api.github.com/repos/cgaueb/streetGAN plus its README (an unrelated EG 2024 street-lighting paper)
  - https://api.github.com/repos/zalandoresearch/spatial_gan plus its README (the SGAN code StreetGAN builds on)
- **Access level:** **FT.** Page numbers are proceedings pages; PDF page 1 = p.133.

## 2. BibTeX

```bibtex
@inproceedings{hartmann2017streetgan,
  author    = {Hartmann, Stefan and Weinmann, Michael and Wessel, Raoul and Klein, Reinhard},
  title     = {{StreetGAN}: Towards Road Network Synthesis with Generative Adversarial Networks},
  booktitle = {WSCG 2017: Full Papers Proceedings: 25th International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision},
  editor    = {Skala, V{\'a}clav},
  series    = {Computer Science Research Notes},
  number    = {CSRN 2701},
  pages     = {133--142},
  publisher = {V{\'a}clav Skala -- UNION Agency},
  year      = {2017},
  note      = {Handle: http://hdl.handle.net/11025/29554}
}
```

There is an ISBN discrepancy. The repository metadata says 978-80-86943-44-2 and the PDF footer says 978-80-86943-49-7, so the ISBN is omitted above.

## 3. What it does

StreetGAN is the first GAN-based road-network synthesiser, according to the authors and to later surveys (e.g. RoBus §2.1). One OSM road patch is rasterised to a binary image. A Spatial GAN (SGAN, Jetchev et al. 2016) is trained on random crops of that single image. Fully convolutional generation from a spatial noise tensor then produces road images of arbitrary size. A morphological and skeleton post-processing step turns those images into a vector road graph that can be used in CityEngine.

## 4. Learned or procedural, and how

- **Learned:** a GAN, specifically a fully convolutional SGAN (single-channel adaptation of the Jetchev et al. code), trained **per example**.
- **One model per style:**
  - one training image per model, with random patches of **321 × 321 px**;
  - styles tested: synthetic irregular, Cellino San Marco (an Italian village), Berlin, synthetic suburban, and Portland (failure case).
  - The synthetic examples come from a news graphic (ref. [21], Misra 2017).
- **Raster:**
  - OSM highway classes motorway, primary, secondary, tertiary, residential, living_street and pedestrian, all treated as **one street level**;
  - 3 × 3 m per pixel; lines drawn 15 px wide with OpenCV (§4.2).
- **Pixels to graph** (§4.4):
  1. threshold at 127;
  2. erosion, then 5 dilations (3×3 cross);
  3. Zhang–Suen thinning;
  4. one node per skeleton pixel, with edges to the 8-neighbourhood;
  5. block cleanup: find minimal cycles and remove 0.5 px² artefact blocks by deleting the √2 edge;
  6. 5 passes of [1/4, 1/2, 1/4] smoothing on chains between non-degree-2 nodes;
  7. Douglas–Peucker simplification with a 3 m tolerance;
  8. removal of dead-end chains shorter than 25 m.
- **Training:**
  - 100 iterations per epoch, batch 64, about 90 s per epoch on a TitanX (Pascal);
  - at least 100 epochs; **the checkpoint is chosen by visual inspection**; about 3 h offline (§6).

## 5. Inputs/outputs and terrain

- **Input at generation:** a uniform spatial noise tensor only, e.g. z ∈ ℝ^{25×25×100} → 769 × 769 px. There is no conditioning, boundary or context streets.
- **Output:** a grey-scale road image, then a vector road graph after post-processing.
- **Terrain:** **none.** Terrain, land use and density maps are named as future conditioning layers (§7). The authors list the lack of control with respect to "landuse or terrain" as a limitation (§5.3).

## 6. Evaluation

- **Data:**
  - four case-study networks (Fig. 3a–d) plus Portland (Fig. 3e, failure);
  - **two synthesised samples per network**;
  - there is no train/test split and no hold-out: the model is trained on the example it is compared with.
- **Metrics:**
  - visual comparison;
  - histograms of **city-block area** (m²), **compactness** (block area / minimal bounding-box area) and **aspect ratio** (shorter / longer side of the minimal bounding box). Observed vs synthesised distributions are shown as bar charts (Figs. 5–8).
  - **No distance between the distributions is computed**, and there is no statistical test or uncertainty.
- **Baselines:** only qualitatively, against texture synthesis (Portilla & Simoncelli; Kwatra et al.; Gatys et al.; Xie et al. Generative ConvNet; §5.4, Fig. 11).
- **Runtime** (§6, p.141):
  - generation 0.08 s for a 769² sample on a TitanX;
  - post-processing on one i7-5820K core: graph construction 1.5 s, block computation 1.6 s, simplification 2.0 s, dead-end removal 0.02 s.
- **Reported failures:**
  - highway ramps and close parallel lanes produce blob artefacts;
  - large-scale curved roads exceed the patch context;
  - **far more dead ends than the originals**;
  - multiple street levels could not be learned.

## 7. Verbatim quotes (line-break hyphens removed)

- p.133, Abstract: "In a pre-processing step, we first convert a given representation of a road network patch into a binary image where pixel intensities encode the presence or absence of streets." … "In a post-processing step, we extract a graph-based representation from the generated images."
- p.136, §3: "So far, when the GAN is trained using neural networks, no well-founded theory about the determination of the success of training procedure of a GAN can be found in literature. Therefore, it is necessary to visually check generated samples"
- p.136, §4.2: "Next, we scale the road network that each pixel in the rastered representation represents an area of 3 × 3 meters. Finally, we raster the road segments as lines with a width of 15 pixels using the line rasterization routine of OpenCV [3]"
- p.137, §4.3: "Please note that we use only a single image, that provides patches for the training procedure, but using multiple different images would be possible and would increase the examples in the training set."
- p.137, §4.4: "We allow to remove nodes that deviate up to 3 meters from a straight line. The last step removes short dead-end street chains with an total length of less than 25 meters."
- p.138, §5.2: "These considered measures include the cityblock area, the compactness, i.e. the ratio between block area and its minimal bounding box, and the city block aspect ratio, i.e. the ratio between the shorter and the longer side of the minimal bounding box."
- p.140, §5.3: "Deadend roads. All the synthesized examples contain much more dead-ends when compared with the number of dead-ends present in their corresponding original road network. This might be due to the patch-based training procedure. Each patch that is used for training typically contains virtual dead-end street segments that abruptly end at the patch boundary."
- p.140, §6: "We trained all the models for at least 100 epochs and decided from a visual examination of samples taken from various epochs which model to choose."
- p.141, §7: "Second, currently we consider all highway categories as part of the same street level. We did not succeed in learning models for different street levels"
- p.141, §7: "In real road networks the road courses are specifically planned to fulfil specific requirements regarding landuse or terrain." … "As our approach is a very first step towards using GANs for road network generation, we did not incorporate such external constraints."
- pp.141–142, §7: "First, we would like to add attribute layers e.g. density maps, landuse maps, terrain maps etc. in order to condition the learning process."
- p.142, §7: "The post-processing needs to be extended to reproduce large-scale structures so that a fair comparison to existing example-based or procedural algorithms for road network generation can be given."

## 8. Comparability with UrbanMorph

- **Code and weights:**
  - **No StreetGAN code or weights found.** A GitHub search for "StreetGAN" returned only unrelated repositories (`cgaueb/streetGAN` is an EG 2024 street-light placement paper, MIT). The paper gives no link.
  - The base SGAN code is https://github.com/zalandoresearch/spatial_gan (MIT). It is marked "deprecated", runs on Theano/Lasagne, and is superseded by `zalandoresearch/psgan`.
- **macOS without CUDA:** the original stack (Theano 0.9-dev) is obsolete. A reimplementation in PyTorch would run on CPU or MPS (not tested).
- **Inputs we could supply:** none of ours (no DEM, water, boundary or context). We could only train one model per reference site from that site's own OSM raster, which leaks the target network.
- **Fair comparison:** unconditional texture-like generation only. It is not comparable on terrain adaptation and not suitable as a baseline under our matched-input protocol.
- **Retraining:** required (per example).
- **Effort (estimate):** reimplementing SGAN plus the post-processing in PyTorch would take **15–25 h**, for a baseline that is weak and out of date. **Not recommended.** Cite it as historical context only.

## 9. Ideas we could adopt

- **Separate boundary-truncation dead ends from true dead ends.** StreetGAN's main artefact is dead ends caused by patch boundaries. In UrbanMorph's dead-end and island metrics, exclude degree-1 nodes within a buffer of the 512 m tile edge, or count them separately, for both generated and real OSM networks.
- **Block-shape distributions** (area, compactness = area / minimum-bounding-rectangle area, aspect ratio). These are cheap complements to the network statistics. Unlike StreetGAN, report a distance between distributions (e.g. Wasserstein or KS) with uncertainty, not only histograms.
- **Disclose every post-processing threshold:** dilation count, 25 m dead-end pruning, 3 m Douglas–Peucker tolerance. Pruning short dead ends changes dead-end rates, so apply identical cleaning to real and generated graphs.
- **Avoid checkpoint selection by eye.** It is a reproducibility weakness that reviewers will notice; select on a held-out metric instead.
