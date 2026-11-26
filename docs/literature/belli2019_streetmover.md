# Belli & Kipf 2019: StreetMover distance

## 1. Header

- **Citation:** Davide Belli, Thomas Kipf. "Image-Conditioned Graph Generation for Road Network Extraction." arXiv:1910.14388 [cs.LG], 31 Oct 2019. "Presented at NeurIPS 2019 Workshop on Graph Representation Learning" (arXiv comments).
- **DOI:** 10.48550/arXiv.1910.14388 (arXiv DataCite).
- **URLs opened (2026-11-25):**
  - https://arxiv.org/abs/1910.14388; https://arxiv.org/pdf/1910.14388 (read in full, including Appendix A.1–A.5); https://arxiv.org/bibtex/1910.14388
  - https://api.github.com/repos/davide-belli/toulouse-road-network-dataset/git/trees/HEAD?recursive=1
  - https://api.github.com/repos/davide-belli/generative-graph-transformer/git/trees/HEAD?recursive=1
  - https://raw.githubusercontent.com/davide-belli/generative-graph-transformer/master/metrics/streetmover_distance.py
  - same repo: `main.py`, `utils/utils.py`, `metrics/statistics.py` (searched for the Sinkhorn parameters)
- **Access level:** **FT** (paper) plus the code read.

## 2. BibTeX (arXiv)

```bibtex
@misc{belli2019imageconditioned,
  title         = {Image-Conditioned Graph Generation for Road Network Extraction},
  author        = {Belli, Davide and Kipf, Thomas},
  year          = {2019},
  eprint        = {1910.14388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {NeurIPS 2019 Workshop on Graph Representation Learning}
}
```

## 3. What it measures

**StreetMover distance (SMD)** is a geometric distance between a predicted road graph and a ground-truth road graph of the *same* tile, based on optimal transport between point clouds. It is a **paired reconstruction** metric for extraction (image → graph). It is not a distributional realism metric.

## 4. Exact definition (paper §3.2, p. 3; code)

1. Sample a fixed number of **equidistant points along all edges** of each graph.
   - Code: `n_points = 100`, spaced at (total length)/(n − 1 + 0.01).
   - An empty graph is replaced by 100 points at the origin.
2. Compute the entropy-regularised OT cost between the two clouds with **Sinkhorn iterations**.
   - Code: `SinkhornDistance(eps=0.00001, max_iter=10)`; the implementation is credited to dfdazac/wassdistance.
   - Cost matrix: |x − y|^p summed over coordinates with **p = 2, i.e. squared Euclidean**.
   - Uniform weights.
3. SMD is the transport cost.
   - Units are squared normalised tile coordinates. The scale depends on how coordinates are normalised; tiles are about 110 m.

## 5. Data

- **Toulouse Road Network dataset** (OSM): 111,034 tiles of 0.001° (about 110 m) square.
- Split: 80,357 train / 11,679 validation / 18,998 test.
- Preprocessing: merge nodes closer than 0.00005°; merge almost-straight consecutive edges; drop graphs with |V| ≤ 3 and the top 5% most cluttered (|V| ≥ 10, |E| ≥ 16).

## 6. Key numbers (model results, not real-world references)

Table 1 (p. 4), StreetMover on the test set:

| Model | SMD |
|---|---|
| MLP | 0.1380 ± 0.0050 |
| RNN | 0.0289 ± 0.0003 |
| GraphRNN | 0.0245 ± 0.0004 |
| GGT | **0.0158 ± 0.0006** |

## 7. Verbatim quotes

- p. 3: "Road networks are first represented as point clouds by sampling a fixed number of equidistant points over the edges of the graphs. Then, the StreetMover distance is computed as the optimal cost of moving the predicted proposal point cloud to the ground-truth target point cloud."
- p. 3: "The main benefits of the StreetMover distance are its interpretability, scalability, and invariance with respect to node permutation graph translations, and rotations."
- Appendix A.3 (p. 8): "Pixel-based metrics like IoU, PSNR and SSI [35] computed on the image representation of the ground-truth and predicted graphs are not good candidates."

## 8. How UrbanMorph could use it, and its limits

- **Where it fits:** paired comparisons only, i.e. a generated network for a site against the *real* network of the same site (our "fidelity to the real city" diagnostics). It is not for unconditional realism, where S10/typicality/energy distance are the right tools.
- **Topology-blind.** SMD sees geometry only, so a gap or a missing junction connection costs almost nothing. Pair it with OD reachability, or with APLS-style routing comparisons (Etten et al., cited in A.3).
- **Invariance claim.** OT between absolute coordinates is invariant to node permutation and to the same rigid transform applied to *both* graphs. It is not invariant to shifting one graph relative to the other. That is desirable for site-registered comparisons, but the paper's wording ("invariance with respect to node permutation graph translations, and rotations") overstates it. This is my reading.
- **Length-normalised sampling.** A fixed point count means total-length differences are not penalised: a sparse and a dense network spread the same mass. Report a length ratio next to it.
- **For 1 km tiles,** use a fixed spacing (e.g. one point per 5 m), unbalanced OT or a total-length term, and metres as units.
