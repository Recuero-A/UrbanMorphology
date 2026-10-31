# Milestone 10 — Street generator architectures

The baselines A–E, why each one exists, and how it is built. The pre-registered criteria are in
`M10_EVALUATION_PROTOCOL.md` §3, and the results in `MILESTONE_10.md` §3.

## 1. Diagnosis that motivates the designs

Source: `scripts/m10_street_diagnosis.py` → `outputs/m10/diagnosis/`. The model is GraphTurtle (M6 roads v2),
256 tiles per seed, compared with the 19 real validation tiles.

| | Real validation | A, seed 0 / 1 |
|---|---|---|
| Orientation φ (median) | 0.59 | 0.10 / 0.11 |
| Junction angle deviation (median) | 4.2° | 15.3° / 13.4° |
| Intersections per km² | 76 | 378 / 385 |
| Mean link length | 81 m | 43 / 40 m |
| Street length per km² | 17 km | 31 / 29 km |
| Loops per km² | 31 | 196 / 227 |

**1. Exposure bias exists and grows with sequence length.**
* The measure is the "snap deviation" of raw sampled turn angles: the distance to the nearest of 0°, ±90° and 180°,
  measured before any geometric rule.
* Teacher-forced (given the real prefix), it stays at 0.8–1.5° at every pop index.
* Free-running (given the model's own prefix), it grows from 2.8° (pops 0–10) to 3.1°, 4.8° and 7.9° (pops > 120).
  The real value is 0.3–0.6°.
* Junction deviation also rises with creation order: from 17.0° in the first quartile to 21.6° in the last
  (seed 0).

**2. About half of all junctions are made by the geometric rules, not by the network.**
* Merge into an existing node: 28–33% of junctions. Crossing split: 10–12%. T-junction: 11–13%.
* All three have a median deviation of 23–26°, against 11–15° for junctions where the network itself branched.
* These rules are not in the training sequences — the train/generate mismatch documented in M5.

**3. Lowering the temperature fixes angles but not structure.**

| T | φ | Junction deviation | Intersections per km² |
|---|---|---|---|
| 1.0 | 0.11 | 15.8° | 374 |
| 0.5 | 0.21 | 9.2° | 378 |
| 0.25 | 0.31 | 6.9° | 402 |

Density is unaffected, so over-density is not sampling noise.

**4. Over-density is the largest error.**
* Generated tiles have 5× the intersections and 6× the loops of real tiles.
* The metric with the worst W1/IQR is loops (2.7 / 3.3), then intersections (2.1 / 2.2).

## 2. Architectures

### 2.1 A — GraphTurtle (reproduced)

* The M6 checkpoints `roads_v2_s0`/`s1`, unchanged: 1.38M parameters.
* An NTG-like breadth-first queue; per pop, a CNN over a turtle-frame crop plus a GRU emitting a mixture of
  bivariate Gaussians, a stop flag and a street class.
* **Reproduction check.** `scripts/m10_eval_streets.py` gives S10 = 1.230 / 1.225, the same as D1 of the
  diagnosis.

### 2.2 B — city-conditioned GraphTurtle

* **Question:** does telling the model the city fix its geography? The cities' φ medians run from 0.57
  (Tokyo, Madrid) to 0.99 (Chicago).
* **Design.**
  * A 9-way one-hot city vector is concatenated to the step-context input.
  * It is added as a per-channel shift of the start-head features and appended to the start-head pooled features.
  * Model option `n_cond`, 1.385M parameters.
* **Sampling.** The city is drawn with the training-tile proportions, so there is no validation leakage.
* This is the "geographic conditioning comparison" asked for in the M10 specification. B is compared with A,
  which has no conditioning, on exactly the same protocol.

### 2.3 N — denoising context noise (exposure-bias mitigation)

* **The idea.** The context (canvas, arms, headings) is built from a graph whose non-border nodes are jittered
  by N(0, 2.5² m²), while every target points to the clean position of the next node. The model therefore sees
  imperfect prefixes in training and learns to steer back.
* **Why not the standard fixes.**
  * Scheduled sampling or DAgger would need an "expert action" for states off the data, and for a street graph
    that action is undefined.
  * Noise injection with clean targets is a simple, well-known alternative.
* **No novelty is claimed.**
* **The 2.5 m σ** matches the measured free-running turn error: about 5° on a 30 m step gives about 2.6 m
  lateral error.
* Same architecture as A (`make_trace(..., clean_nodes=...)`).

### 2.4 C — hierarchical, major roads → local streets

* **Stage 1 (C1).** GraphTurtle trained on the major + mid subgraph of each tile (OSM classes ≤ mid).
  * The audit found that this network forms through-corridors in every city (median corridor share 1.00).
  * It is 9–16% (major) + 15–24% (mid) of street length.
* **Stage 2 (C2).** GraphTurtle trained to emit only local and minor streets, with the major + mid network
  pre-drawn on the canvas from step 0 (`make_trace(..., given=...)`).
  * A local component that touches the major network starts at one of its nodes, heading along a random
    major edge into that node. The turtle "arrives along the arterial", so local streets are framed relative
    to the arterial direction.
  * Major nodes are popped when a local street first reaches them.
* **Generation.**
  * C1 samples the major network, which is given to C2.
  * A C2 start point within 16 m (one start cell) of an unreached major node starts there, as in training.
* **Hierarchy levels.** This is two of the four levels in the specification: major roads → local streets.
  * The "neighbourhood latent" is approximated by the city vector in B.
  * "Refinement" is left out until the two-stage result is known.
* **C-oracle** runs C2 on the real validation major networks. It measures stage 2 alone (conditional,
  exploratory).

### 2.5 D — lightweight diffusion: not run for streets in M10

* **Reason.** The literature that works uses large pretrained raster models (SD 1.5 plus ControlNet, about
  105 GPU-hours) or datasets 20× ours. The one vector diffusion road paper (CaRoLS) could not be read.
* **The raster obstacle.** A raster street model needs vectorisation (skeleton → graph), which reintroduces a
  rule-based step like the one the diagnosis blames. No skeletonisation library is installed, and adding one
  was out of scope.
* **What was done instead.** The small conditional DDPM was built for *blocks* (`M10_BLOCK_GENERATION.md` §4),
  where the output is a building mask and vectorisation is trivial.
* **If street diffusion is wanted next:** the same U-Net on 4 m road rasters, plus skeletonisation. About
  2–3 h of training on this machine.

### 2.6 E — procedural reference

* A perturbed grid whose block size comes from a randomly chosen **training** tile of a city drawn by training
  proportions (`src/urbanmorph/graphs/procedural.py`).
* Random orientation, 4% node jitter, 8% segment dropout, and classes from line index.
* It is built through `build_graph`, so it has the same representation as the real data.
* **It is not tuned to validation.** It shows what a trivial grid scores on the metrics.

## 3. Recommendation (before results)

Order of priority:
1. **C** — the hierarchy addresses over-density directly, because it separates the sparse arterial skeleton
   from local fill.
2. **N** — exposure bias.
3. **B** — geography.

**Not recommended:** D (see §2.5), or any large model. The data is 565 tiles.

The post-results recommendation is in `MILESTONE_10.md` §6.
