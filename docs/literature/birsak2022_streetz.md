# Birsak et al. 2022: evaluation metrics and the `twak/streetz` code

## 1. Header

- **Citation:** Michael Birsak, Tom Kelly, Wamiq Para, Peter Wonka. "Large-Scale Auto-Regressive Modeling Of Street Networks." arXiv:2209.00281v1 [cs.LG], 1 Sep 2022.
  - The PDF uses an ACM template with a placeholder DOI ("https://doi.org/XXXXXXX.XXXXXXX").
  - No peer-reviewed venue was found: **not verified**.
- **Code:** https://github.com/twak/streetz, Apache-2.0. Description "street graph stats and viz". Created 2022-05-12; last push 2023-05-15. The README says it is the "source for evaluation from the publication".
- **URLs opened (2026-11-25):**
  - https://arxiv.org/abs/2209.00281; https://arxiv.org/pdf/2209.00281 (main text and Appendix D read); https://arxiv.org/bibtex/2209.00281
  - https://api.github.com/repos/twak/streetz; https://api.github.com/repos/twak/streetz/git/trees/HEAD?recursive=1
  - https://raw.githubusercontent.com/twak/streetz/master/ followed by each of: `README.md`, `stats.py`, `stats_graph.py`, `stats_segs.py`, `stats_blocks.py`, `utils.py`, `main.py`
- **Access level:** **FT** (paper) plus the code read.

## 2. BibTeX (arXiv)

```bibtex
@misc{birsak2022largescale,
  title         = {Large-Scale Auto-Regressive Modeling Of Street Networks},
  author        = {Birsak, Michael and Kelly, Tom and Para, Wamiq and Wonka, Peter},
  year          = {2022},
  eprint        = {2209.00281},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## 3. What it measures

A set of per-city graph statistics compares generated networks with OSM ground truth and CityEngine output (Fig. 9). The statistics are mostly histograms plus table means. The paper takes no terrain input: "we do not consider terrain as conditioning information in the current iteration" (p. 5).

## 4. Exact metric definitions (paper Appendix D, p. 10, and code)

- **Segment:** "a sequence of edges in the graph without branches, connected by vertices with valencies only of 2" (paper). The README says segments "can bend 90 degrees, as long as they don't go through junctions".
- **Segment circuity** = segment length / Euclidean distance between its first and last vertices. This is **edge sinuosity**.
  - Code (`stats_segs.segment_circuity`): the table mean is over all non-loop segments.
  - The histogram (range 1–2, 32 bins) **excludes segments with ratio < 1.02**: "Segment Circuity (> 1.02)" in Fig. 9.
  - The README wording ("ratio of the euclidean distance ... and its length") is the inverse of the code; the **code computes length / euclid ≥ 1**.
- **Transport ratio** = shortest-path length / Euclidean distance between two random vertices.
  - The paper says "We sample its value over n = 500 randomly chosen vertex pairs" (`SAMPLES = 500` in `stats_graph.py`).
  - **The code silently skips pairs with no path** (`except nx.exception.NetworkXNoPath: return`). The statistic is therefore **conditional on connectivity**, and disconnected graphs are not penalised.
  - Pairs are uniform over vertices and not distance-banded, so long pairs dominate.
- **Block:** "every enclosed region in the graph without any internal edges". Holes are not processed; planarity is assumed.
  - **Block aspect ratio** = short/long side of the minimum-area bounding rectangle, found by trying each convex-hull edge orientation.
  - **Rectangularness** = block area / that rectangle's area.
  - Block area and perimeter histograms are also reported.
- **Node degree histogram**, **mean node degree** = 2|E|/|V|, and number of dead ends.
- **Edge angle:** an 18-bin length-weighted histogram over [0, π) of edge atan2. Its table value is the **argmax angle**, not an entropy.
- **Densities:** edges, vertices, street length (km/km²) and blocks, all normalised by **land area** from a land/water mask.
- **PageRank** (k = 0.85 in the paper, topology only) and **PageRank-on-edges** (edge graph, k = 0.95, initialised proportional to length).
- **Betweenness:** sampled; present in the code but commented out of the default metric list.

## 5. Data

OSM street graphs. Fig. 9 shows Chicago, San Francisco and other US cities. The `npz/` folder holds generated graphs for Chicago, LA, NY, SF and Washington. Ground-truth values are shown in figures only, with **no numeric table extracted**.

## 6. Key empirical numbers

None extracted as numbers. The Fig. 9 histograms are visual. **Not verified.**

## 7. Verbatim quotes

- Appendix D (p. 10): "circuity measures the ratio between the length of the segment (the sum of the length of the edges in the segment) against the euclidean distance between the first and last vertex of the segment. It describes the local curvature of the streets."
- Appendix D (p. 10): "The transport ratio [8] measures the ratio between the walking distance (along the graph) and the euclidean between two vertices. We sample its value over n = 500 randomly chosen vertex pairs in the graph."
- p. 5: "Further, we do not consider terrain as conditioning information in the current iteration of"
- README: "transport ratio is calculated between two vertices and is the ratio of the shortest path length to euclidean distance. It is slow to compute the true value, so we sample random vertex pairs."

## 8. How UrbanMorph could use it

1. **Borrow block aspect and rectangularness.** They complement Polsby–Popper and Louf's Φ, and are cheap (minimum-area rectangle).
2. **Do not adopt the transport ratio as written.** Dropping unreachable pairs rewards fragmented graphs. If we report it:
   - compute it over same-land pairs within distance bands;
   - report the unreachable share next to it, or score unreachable pairs as a capped penalty (e.g. ratio = 3, the top of the streetz histogram).
   - The same flaw applies to our S_T circuity. See [metrics_catalogue.md](metrics_catalogue.md).
3. **Normalise densities by land area,** as streetz does with its water mask. On steep sites also consider "buildable land" (slope below a threshold). Otherwise generated and real densities differ simply because cliffs are counted.
4. **Code quality note.** The vertex sampler uses `random.randint(0, len(vertices))`, whose upper bound is inclusive in Python, so it can index past the end. The 1.02 histogram cut-off also hides straight segments. **Reimplement; do not call the code directly.**
