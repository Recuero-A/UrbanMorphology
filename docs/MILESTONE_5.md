# Milestone 5: Generating roads as graphs (NTG-inspired)

## Answer in brief

* **Connectivity is solved by construction.**
  * Generated tiles: 0 floating fragments, 1 road piece (median).
  * M4's VAE + learned prior: 4 fragments and 10 pieces.
* **Street structure is not realistic.** Junctions run 3.8× the real density, loops 6× and street
  length 1.5×, and orientation order is near random (φ 0.15 vs 0.59).
* **Pre-registered decision: NOT met** (C1, C3 and C4 pass; C2 structure fails, both seeds).
* **Diagnosis:** steps are predicted in absolute (dx, dy) coordinates. The network must relearn
  "perpendicular to the street I came from" for every possible heading, so junction angles
  deviate a median of 14.8° from 90°/180° (real: 3.4°) and nodes branch twice as often.
* **A second bias:** 88–91 % of generated tiles have ONE component, against 42 % of real ones.
  The start head stops too early.
* **Verdict:** a suitable **backbone** (connectivity, no memorisation, diversity kept), but it needs
  turtle-frame steps (as in NTG itself) before it can carry buildings (Milestone 6 plan).


## 1. Why a graph generator

Milestones 3–4 generated roads as **pixels**. Connectivity was therefore only a statistical
outcome:

* soft-clDice did not fix it;
* a learned latent prior roughly halved it (M4: 4 floating fragments per tile against 0 real).

M4 also showed that the bottleneck is the generative model, not the decoder. Here the
representation changes instead: roads are generated as a **graph** of nodes and street segments.
A street can only start from an existing node, so every component is connected **by construction**.
How many components there are, and where, is still learned, not imposed.

## 2. Related work (verified)

| Work | What it does | What we take |
|---|---|---|
| Neural Turtle Graphics (Chu et al., ICCV 2019) | Grows city road graphs from a node queue. An RNN encodes a node's incoming paths; an RNN decoder emits new outgoing steps. New nodes close to existing ones are merged. | Queue-based growth, step-by-step decoding, merging (loop closing) |
| GraphRNN (You et al., ICML 2018) | Autoregressive graph generation in BFS order; evaluation by MMD of graph statistics | BFS ordering; comparing distributions of graph statistics |
| RoadTracer (Bastani et al., CVPR 2018) | Iterative graph construction for road extraction from aerial images | Evidence that step-by-step graph growth yields connected networks |

**Not novel:** this is a small re-implementation of NTG's idea on our tiles, not a new method.

## 3. Data: street graphs (`src/urbanmorph/graphs/road_graph.py`)

Source: the `road_centerlines` layer of every tile. These are the OSMnx edges clipped to the
512 m window, the same vectors the road raster came from. No new downloads.

| Step | Rule | Why |
|---|---|---|
| Resample | every line at equal spacing ≤ 30 m; endpoints kept | bounded step length; curves follow as chords |
| Snap | greedy, to node centres ≥ 8 m apart; endpoints first | merges two-way duplicates, consolidates junctions (cf. M2's 10 m) |
| Edges | consecutive points of each line | streets that cross WITHOUT a shared OSM node (bridges) stay unjoined |
| Border | node < 1 m from the tile edge | street cut by the window: legitimate end |

Size: 565 training and 19 validation graphs. A median tile has 168 nodes (max 400) and
2 components. **Test tiles are not read.** Check against M2's vector metrics on validation
tiles: street length and φ agree (e.g. φ 0.58 vs 0.58; 26.2 vs 26.3 km/km²). Junction counts
are somewhat higher because 8 m snapping joins very close parallel ways. Real and generated
graphs go through the same definitions, so this does not bias the comparison.

## 4. Model (`src/urbanmorph/models/graph_gen.py`, `src/urbanmorph/graphs/`)

**Construction sequence (BFS, `sequence.py`):**

```
for each component (largest first):  START at a node  ->  queue = [start]
    while queue: pop u
        for each neighbour w with edge (u,w) not yet drawn (counter-clockwise from east):
            emit displacement d = w - u           (new w is created and queued)
        emit STOP
STOP (no more components)
```

**What the network sees when popping node u** (identical in training and sampling, unit-tested
by replaying real graphs through the sampler):

| Input | Shape | Meaning |
|---|---|---|
| crop | 3 × 48 × 48 (4 m pixels = 192 m) | roads drawn so far; "open" nodes still in the queue; outside-the-tile |
| arms | ≤ 8 × (dx, dy, length) / 30 m | u's already-drawn streets (NTG's incoming paths) |
| scalars | 8 | position in tile, sub-pixel offset, border flag, #arms, #nodes, step index |

**Network (1.38 M params):**

* **Context:** a CNN on the crop (48→24→12→6, then 256-d), a set-MLP over the arms (sum, so
  their order does not matter) and the scalars, merged by an MLP into a 256-d context.
* **Step decoder:** a GRU (hidden 256). At step k its input is [previous displacement,
  first-step flag, context]. Output: a STOP logit and a mixture of 10 bivariate Gaussians
  (with correlation) for d / 30 m.
* **Start head:** a CNN over the whole 128 × 128 canvas.
  * Output 1: a logit for "no more components".
  * Output 2: a categorical over 32 × 32 cells (16 m), uniform inside a cell, giving the
    location of the next component's first node.

**Loss:** the exact negative log-likelihood of the construction sequence (all factors are
normalised densities or probabilities). Augmentation per epoch: a random symmetry of the square
and random start nodes.

**Sampling (`generate.py`)** turns each sampled step u→u+d into geometry. These are fixed rules,
not learned:

1. clip at the tile edge;
2. **crossing** an existing street creates a junction there;
3. landing within 8 m of a node **joins** it (NTG's merge);
4. landing within 8 m of a street makes a **T-junction**;
5. a step shorter than 8 m is dropped; otherwise a new node is created and queued.

Rules 2 and 4 never occur in training sequences: there, every step lands exactly on a node.
This is a train/sample mismatch.

## 5. Evaluation protocol and decision criteria (fixed before sampling)

**Samples:** 256 generated tiles per training seed (seeds 0 and 1), sampling seed 123,
temperature 1.

**References:**

* **Primary:** real validation graphs (19).
* **Additional:** real training graphs (565).

**Metric families:**

* **Graph metrics** (`graph_metrics`):
  * junctions/km²;
  * dead-end ratio;
  * loops/km² (cycle rank E − V + C);
  * Boeing φ;
  * mean link length;
  * street length/km²;
  * components and interior (floating) components.
* **Raster topology:** M3/M4's `full_road_profile`, computed after rasterising **every set with
  one 7 m width** (the median OSM width). Raster differences then come from the network, not
  from road widths. M4's numbers used real widths, so they are context only.
* **Diversity:** 1 − pairwise road IoU.
* **Memorisation:** nearest training tile under the 8 symmetries (road IoU).

**Success, if it holds for BOTH seeds:**

| | Criterion |
|---|---|
| C1 fragmentation | median floating fragments ≤ 1 **and** median road pieces within ±1 of real |
| C2 structure | W1/IQR to real ≤ 1 for ≥ 5 of the 6 graph metrics |
| C3 diversity | ≥ real diversity − 0.05 |
| C4 memorisation | median nearest-train IoU ≤ real validation's + 0.10 |

## 6. Results (`outputs/m5/results.json`; 256 generated tiles per seed; medians)

**Training:** best epoch 28 / 29 (seeds 0 / 1), early stopped at 36 / 37, about 9 min per seed on
MPS. Validation NLL fluctuates strongly from epoch to epoch (constant learning rate 1e-3; a
schedule is a cheap improvement). **Sampling:** 256 tiles in about 50 s on CPU.

| | Road pieces (raster) | Floating fragments | Components (graph) | Junctions/km² | Dead-end ratio | Loops/km² | φ | Link length (m) | Street km/km² |
|---|---|---|---|---|---|---|---|---|---|
| Real validation (19) | 2 | 0 | 2 | 76 | 0.056 | 31 | 0.59 | 81 | 17.4 |
| Real training (565) | 2 | 0 | 2 | 88 | 0.048 | 31 | 0.76 | 84 | 17.0 |
| **G seed 0** | 1 | 0 | 1 | 328 | 0.078 | 181 | 0.15 | 44 | 25.3 |
| **G seed 1** | 1 | 0 | 1 | 313 | 0.062 | 183 | 0.16 | 45 | 25.8 |
| M4 PixelCNN prior (context; real widths) | 10 | 4 | — | 175 (raster) | 0.36 (raster) | 23 (raster) | 0.63 (raster) | — | — |

**W1/IQR to real validation, by graph metric:**

| Seed | Junctions | Dead ends | Loops | φ | Link length | Street length |
|---|---|---|---|---|---|---|
| 0 | 1.72 | 0.32 | 2.46 | 1.14 | 0.93 | 0.56 |
| 1 | 1.70 | 0.31 | 2.57 | 1.14 | 0.95 | 0.61 |

**Other measures:**
* **Diversity:** 0.925 / 0.924 against 0.943 real.
* **Memorisation:** nearest-train road IoU 0.138 / 0.139, against 0.137 for real validation tiles.
  No copying.

**Decision:**

| Criterion | Seed 0 | Seed 1 |
|---|---|---|
| C1 fragmentation | ✅ | ✅ |
| C2 structure (≥ 5 of 6 within 1 IQR) | ❌ 3 / 6 | ❌ 3 / 6 |
| C3 diversity | ✅ | ✅ |
| C4 memorisation | ✅ | ✅ |
| **Success** | **no** | **no** |

**Per country (real training graphs; the generator is unconditional, so generated tiles have no
country):**

| | Junctions/km² | Loops/km² | φ | Link length (m) | Components |
|---|---|---|---|---|---|
| Japan | 174 | 82 | 0.65 | 66 | 1 |
| Spain | 118 | 42 | 0.59 | 77 | 2 |
| US | 50 | 15 | 0.96 | 111 | 2 |

Generated junction density (≈ 320/km²) is above even Japan's. The model does not reproduce any
single country's structure.

Figures: `outputs/m5/samples_graphs.png`, `outputs/m5/distributions_graphs.png`,
`outputs/m5/diagnosis/first_look.png`.

## 7. Diagnosis

| Test | Real | Generated (seed 0) | Reading |
|---|---|---|---|
| How often each sampling rule fired | — | new node 80 %, joined a node 10 %, split a street (cross/T) 6 %, dropped 4 % | rules are not the main source of junctions |
| Turn at pass-through nodes | 82 % < 5° | 70 % < 5° | straight runs are roughly fine |
| Junction gaps: deviation from 90°/180° | median 3.4°, 69 % < 10° | median 14.8°, 40 % < 10° | **branches leave at wrong angles** |
| Node degrees 1 / 2 / 3 / 4+ | 1.8 / 79 / 12 / 7 % | 2.8 / 62 / 24 / 11 % | **branches twice as often** |
| Teacher-forced next step at real pass-through nodes | — | median angular error 2.1°, but 22 % > 20° | the absolute-frame mixture often puts mass on a perpendicular turn |
| Temperature 0.5 instead of 1 | — | φ 0.17, junctions 313 | not a sampling-noise problem |
| Tiles with one component | 42 % | 88–91 % | start head under-predicts additional components |

**Cause (hypothesis, consistent with all tests):** the parameterisation.
* A mixture over absolute (dx, dy) must represent "continue / turn 90° left / turn 90° right"
  separately for every incoming heading. With 565 tiles it cannot do so sharply.
* Small angular errors at each branch rotate the street directions of the whole tile, so φ
  collapses towards random.
* Skewed branches then cross other streets and close many small loops.

NTG predicts steps relative to the incoming path. That makes the model **equivariant to
rotation by construction**: the same local configuration gives the same relative prediction at
any heading. It does not prefer 90°; it makes the angle *relative*, so the data can show
whichever angles occur.

**Proposed fixes (not implemented; part of the Milestone 6 plan):**
1. steps as (turn angle, length) in the frame of the incoming arm: a mixture of von Mises ×
   log-normal;
2. the crop rotated to the incoming heading;
3. a learning-rate schedule;
4. a component-count check of the start head (compare per-tile component distributions).

## 8. Limitations

* 4 m crops; one road width for rasterisation.
* No street hierarchy (class / width not modelled).
* Crossing and T-junction rules are not seen in training.
* An unconditional model, so no per-country generation.
* Two seeds; 19 validation tiles.

## 9. Reproduce

```
python scripts/m5_build_graphs.py
python scripts/m5_train_graph.py --seed 0 && python scripts/m5_train_graph.py --seed 1
python scripts/m5_evaluate_graph.py
pytest tests/test_graph_gen.py
```

## References

* Chu, H., Li, D., Acuna, D., Kar, A., Shugrina, M., Wei, X., Liu, M.-Y., Torralba, A., Fidler, S.
  (2019). Neural Turtle Graphics for Modeling City Road Layouts. *ICCV 2019*.
* You, J., Ying, R., Ren, X., Hamilton, W., Leskovec, J. (2018). GraphRNN: Generating Realistic
  Graphs with Deep Auto-regressive Models. *ICML 2018*.
* Bastani, F. et al. (2018). RoadTracer: Automatic Extraction of Road Networks from Aerial Images.
  *CVPR 2018*.

