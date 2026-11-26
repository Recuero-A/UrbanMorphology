# Chu et al. 2019: Neural Turtle Graphics (NTG)

## 1. Header

- **Citation:** Hang Chu, Daiqing Li, David Acuna, Amlan Kar, Maria Shugrina, Xinkai Wei, Ming-Yu Liu, Antonio Torralba, Sanja Fidler. "Neural Turtle Graphics for Modeling City Road Layouts." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019 (oral), pp. 4521–4529. arXiv:1910.02055 (v1, 4 Oct 2019).
- **DOI:** 10.1109/ICCV.2019.00462 (resolved through the Crossref API).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/pdf/1910.02055 (full text, 9 pages, v1)
  - https://raw.githubusercontent.com/nv-tlabs/NTG/master/docs/resources/supp.pdf (official supplementary appendix, 5 pages)
  - https://api.github.com/repos/nv-tlabs/NTG and its git tree; https://raw.githubusercontent.com/nv-tlabs/NTG/master/README.md; `docs/index.html` (a redirect)
  - https://nv-tlabs.github.io/NTG/ → https://research.nvidia.com/labs/toronto-ai/NTG/ (project page)
  - https://openaccess.thecvf.com/content_ICCV_2019/html/Chu_Neural_Turtle_Graphics_for_Modeling_City_Road_Layouts_ICCV_2019_paper.html (BibTeX)
  - https://api.crossref.org/works/10.1109/ICCV.2019.00462
  - https://api.github.com/repos/Cewein/Neural-Turtle-Graphics and its git tree; raw `readme.md`, `requirements.txt`, `config/config.yaml`, `main.py`, `src/model.py`, `src/generator.py`, `src/train.py`, `src/data.py`
- **Access level:** **FT** (main paper and supplement read in full). Page numbers below are arXiv v1 PDF pages (1–9) and supplement pages (S1–S5).

## 2. BibTeX

```bibtex
@inproceedings{chu2019ntg,
  author    = {Chu, Hang and Li, Daiqing and Acuna, David and Kar, Amlan and Shugrina, Maria and Wei, Xinkai and Liu, Ming-Yu and Torralba, Antonio and Fidler, Sanja},
  title     = {Neural Turtle Graphics for Modeling City Road Layouts},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4521--4529},
  year      = {2019},
  month     = oct,
  doi       = {10.1109/ICCV.2019.00462},
  eprint    = {1910.02055},
  archivePrefix = {arXiv}
}
```

## 3. What it does

NTG is a sequential generative model of planar spatial graphs, applied to city road layouts. A queue of unvisited nodes is expanded one node at a time. An RNN encoder summarises the node's incoming paths, and an RNN decoder emits that node's outgoing neighbours as relative offsets. The same model is used for city generation, with optional city-style conditioning and user sketches. It is also used as a topological prior for aerial road parsing on SpaceNet, and for "Sat2Sim" completion of parsed graphs.

## 4. Learned or procedural, and how

- **Learned, autoregressive, vector.** This is UrbanMorph's base architecture.
- **Encoder:** a bidirectional GRU over each incoming path's motion vectors (Δx, Δy). The last hidden states are summed over K sampled random-walk paths of at most L nodes, which makes the code order-invariant (§3.2, p.3).
- **Decoder:** a GRU that emits outgoing nodes sorted counter-clockwise, plus a binary "generate another node" flag. An edge type (minor or major road) is optional (§3.2, §3.5).
- **Optional attribute vector `h_attr`:** e.g. a city-ID embedding for style.
- **Coordinates:** discrete 1 m bins for city generation, with Δx and Δy limited to [−100, 100] m and predicted independently. There are 500 GRU hidden units (§3.4, p.4).
- **Training:**
  - teacher forcing with cross-entropy per output node;
  - Adam, learning rate 1e-3, weight decay 1e-4, gradient clip 1.0 (§3.5);
  - incoming paths are sampled "that have similar second last node", to mimic generation (supp. §2, p.S2).
- **Inference rules:**
  - **Snapping:** a new node within 5 m of an existing node becomes an edge to that node and is not queued, which creates cycles (§3.2, §3.5).
  - **Limits:** maximum degree, maximum node density and minimum angle are taken from the training set and enforced.
  - **Planarity:** enforced by inserting nodes at edge crossings (supp. §2).
- **Training data, "RoadNet":**
  - OSM from **17 cities** (13 countries); the most densely annotated 10 km² region of each city;
  - 233.6k nodes, 262.1k edges, 170.0 km², 7,410.7 km of road (Table 1, p.4);
  - no tiling into fixed windows is described for NTG itself.
  - Cities named in the figures: Istanbul, Mexico City, New York (main paper), and Cambridge, Budapest, Berlin, Berkeley, Barcelona, Aachen, Toronto, Tokyo, San Francisco, Prague ("Prag"), Paris, London (supp. Figs. 3–4). Brussels and Toronto appear in Fig. 8.
- **Raster to graph:** not needed for NTG, whose output is native vector. In the baselines, PGGAN images are converted "by thresholding and thinning" (p.5).

## 5. Inputs/outputs and terrain

- **Inputs:**
  - a root node with a few edges: a random root (NTG-vanilla) or a "pre-stored high connectivity root" (NTG-enhance);
  - an optional city-ID embedding;
  - optional user-sketched local topology, matched to pre-stored node templates;
  - for parsing, CNN road likelihoods.
  - For Sat2Sim completion, all "single-connection nodes in the parsed graph" are pushed to the queue (§4.3, p.8).
- **Output:** a planar vector graph in metres, optionally with major/minor edge labels.
- **Terrain:** **none.** No DEM, slope or water input. Terrain appears only in related work (procedural methods [18, 8]).

## 6. Evaluation

- **Generation data and split:**
  - Metrics are computed per city against that city's ground-truth map (FID "for each city").
  - **No train/test split or geographic hold-out for generation is described** (not found in the paper or supplement).
- **Generation metrics** (§4.1.1, pp.4–5):
  - **Perceptual FID:** for every node, render the graph in a 300 m neighbourhood. Features come from InceptionV3 fine-tuned to classify city ID on RoadNet renders (90.27 % accuracy). FIDs are reported at four layers: maxpool1, maxpool2, pre-aux and fc.
  - **Urban Planning features** (after AlHalawani et al. 2014):
    1. node density within 100, 200 and 300 m;
    2. connectivity (node degrees);
    3. reach (total accessible road length within 100, 200 and 300 m);
    4. transportation convenience (Euclidean / Dijkstra shortest-path distance for node pairs > 500 m apart).
    - The reported score is the Fréchet distance between normal distributions fitted to the concatenated feature vectors of real and generated maps.
  - **Diversity:** the percentage of road in one graph lying outside the 10 m vicinity of the other graph, both ways (Chamfer-like). It is computed against all ground-truth maps and the average lowest value is reported. Extremely low values flag memorisation.
  - **"Rating":** averages with scales {10,10,10,20} (perceptual) and {60,30,50,20} (urban planning).
- **Generation baselines:**
  - GraphRNN-2D: GraphRNN plus RoadTracer-style rendering, with a crossing mask;
  - PGGAN at 256×256, thinned to graphs;
  - CityEngine with an exhaustive attribute search: 5,000 km² ("5k") or 10,000 km² ("10k") of generated maps, keeping the top 10 km² per city.
- **Headline results (Table 2, p.5).** Each FID column carries its own power-of-ten scale in the table header.

  | Variant | FID mp1 | FID mp2 | FID pa | FID fc | Perceptual rating | Urban-planning rating | Diversity |
  |---|---|---|---|---|---|---|---|
  | NTG-enhance | 1.52 | 1.34 | 2.83 | 6.76 | 77.3 | 92.4 | 42.09 |
  | CityEngine-10k | 2.55 | 2.56 | 8.23 | 14.17 | 48.9 | 52.1 | 46.00 |
  | PGGAN | — | — | — | — | — | — | 5.95 (memorises) |

- **Statistics and uncertainty:** none reported (no confidence intervals, no seeds).
- **Ablations (supp. Table 1):**
  - The FID effect of removing max-density, max-degree, planarity and min-angle is reported; removing planarity hurts most.
  - The effect of K and L on reconstruction error and inference time is shown in Fig. 6, as a figure only.
- **Runtime:** only Fig. 6 (seconds per km² vs K, L); no numbers in the text.
- **Parsing (SpaceNet):**
  - split 4-1-1 per city;
  - metric APLS on RDP-simplified graphs subdivided to ≤30 m edges, plus IoU/F1;
  - **one-city hold-out, 4-fold** (Table 4);
  - NTG-I reaches APLS 74.97 (Table 3).

## 7. Verbatim quotes (line-break hyphens removed)

- p.2, §2: "Several approaches followed this line of work, exploiting user-created tensor fields [11], domain splitting [40], constraints stemming from the terrain [18, 8], and blending of retrieved examplars [6, 31, 16]."
- p.3, §3.2: "At generation time we check overlap between the new node and existing graph with a 5m threshold to produce loops."
- p.4, §3.4: "For city generation, we use discrete ∆x, ∆y with resolution of 1m for both encoder and decoder, where x points to east and y points to north. Here, ∆x and ∆y are limited to [−100 : 100], indicating that the largest offset in either direction is 100m."
- p.4, §3.5: "we do not add the new node to the queue. Instead, an edge is included to connect the current node to the existing node. This enables the generation of cycles in the graph."
- p.4, §3.5: "We also find the maximum node degree, maximum node density, and minimum angle between two edges in the training set, and ensure our generation does not exceed these limits."
- p.4, §4.1: "OSM, being crowd-sourced, often has incomplete markers in underpopulated areas. To alleviate this, we manually select the most densely annotated 10km2 region within each city."
- p.4, §4.1.1: "To ensure a meaningful FID, we adapt InceptionV3, which has originally been trained on natural images, to road drawings, by finetuning it to predict city ids on our dataset."
- pp.4–5, §4.1.1: "Transportation convenience as the ratio of the euclidean distance over the Dijkstra shortest path for node pairs that are more than 500m away. We also compute the Fréchet distance between normal distributions computed from the concatenation of the Urban Planning features of real and generated maps."
- p.5, §4.1.1: "We measure the ability to create novel cities by computing the overlap between a real and generated city as the percentage of road in one graph falling outside the 10m vicinity of the road in the other graph, and vice versa."
- p.6, §4.1.2: "Training with longer and more paths does not necessarily improve perceptual quality, since generation starts from a single root node without long paths."
- p.S2, supp. §2: "We enforce planarity by adding nodes when two edges intersect." / "We notice the max-density and max-degree constraints are useful in preventing over-generation."
- p.S2, supp. §4: "Our data and code are available at https:// nv-tlabs.github.io/NTG." (This was not true when checked; see §8.)

## 8. Comparability with UrbanMorph

- **Official code and weights:** **none.**
  - https://github.com/nv-tlabs/NTG (no licence; last push 2023-10-23) holds only a README and `docs/` with the project page, paper, supplement, figures and video.
  - The project page links only arXiv, the supplement and figures, despite the supplement's claim.
  - The RoadNet dataset is not released (RoBus Table 2 also lists NTG data as "Closed").
- **Unofficial code:** https://github.com/Cewein/Neural-Turtle-Graphics.
  - **No licence file** (GitHub API: `license: null`), so all rights are reserved. 17 commits, 2025-04-29 to 2025-05-08.
  - Dependencies: PyTorch ≥1.8, networkx, osmnx, numpy, pyyaml, tqdm, matplotlib, pandas.
  - Weights: `example/NeuralTurtleGraphics_model_balmain.pth` (35 MB). The training OSM file `data/aussie/map_balmain.osm` suggests Balmain, Sydney.
  - **The config cannot load the weights as-is.** It points to `example\\NeuralTurtleGraphics_model_blamain.pth`, which has a Windows backslash and a misspelt "blamain".
  - Differences from the paper:
    - max displacement 200 (paper 100) and hidden size 512 (paper 500);
    - **no city-ID attribute and no max-density constraint**;
    - planarity by rejecting crossing edges instead of inserting nodes;
    - degree and angle limits from the 98th and 2nd percentiles.
  - The README misstates the venue as "CVPR 2020".
- **macOS without CUDA:**
  - Unofficial code: **yes on CPU.** `device: auto` chooses CUDA if available, else CPU, with no MPS path; GRUs are cheap on CPU.
  - No CUDA-only operations were seen in the code read.
- **Inputs we could supply:**
  - seed nodes or anchors (root), or degree-1 boundary nodes of the surrounding streets (Sat2Sim-style completion).
  - It **cannot take a DEM, slope, water or a boundary**; a boundary or water mask could only be imposed as a post-hoc rejection rule, which would be our addition.
- **Fair comparison type:**
  1. **Unconditional growth from the same anchors.** This is an ablation of UrbanMorph's base without terrain; reviewers will read it that way.
  2. **Completion given surrounding streets:** push the degree-1 nodes of the context into the queue, as in the paper's §4.3.
- **Retraining:** required in both cases (no official weights; the unofficial weights cover one Sydney suburb).
- **Effort (estimates):**
  - Our own "TerrainGraphTurtle with terrain channels zeroed, same sampler": **2–6 h** if our code exposes the switch. This is the most faithful NTG-style baseline and is licence-free.
  - Cewein as-is: **2–4 h** (path fix, CPU run).
  - Cewein retrained on our tiles with anchors and boundary clipping: **15–25 h**, and only with the author's permission (no licence).

## 9. Ideas we could adopt

- **Fréchet distance over concatenated network features** (density, degree, reach and convenience at several radii) as one summary realism number. Fit it per terrain class so that hilly and flat tiles are not pooled.
- **Domain-adapted FID:** fine-tune a small CNN to classify *terrain class or city* from road renders before computing FID. Report the classifier's accuracy, as NTG does (90.27 %), to show that the features are meaningful.
- **Chamfer-like novelty at 10 m** between a generated tile and the real OSM network on the *same* terrain. This directly measures copying or DEM-imprint leakage, which matters more for us than for NTG because we condition on the real site.
- **Transport convenience with a distance threshold** (NTG > 500 m; RoBus > 300 m). State how unreachable pairs are scored; NTG does not.
- **Constraint ablation table** (degree, density, angle, planarity), mirroring supp. Table 1, for our hard feasibility rules.
- **APLS** (with the 5 m buffer and 30 m resampling) for the completion setting, where a ground-truth network exists.
