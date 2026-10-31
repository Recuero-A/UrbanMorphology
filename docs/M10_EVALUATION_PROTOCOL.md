# Milestone 10 — Evaluation protocol and pre-registration

Written 2026-10-31, **before** any M10 model was trained. Any later change to a criterion is recorded
in §6 with its date and reason. Results never change a criterion silently.

## 1. Units, splits, references

* **The unit of analysis is the tile** (512 × 512 m).
  * Every statistic is computed per tile first, then compared across tiles.
  * Pooled junctions, buildings or steps are never treated as independent samples. They come from the
    same tile and are spatially dependent.
* **Splits.** The M2 splits, unchanged.

  | Split | Tiles | Role in M10 |
  |---|---|---|
  | Train | 565 | Training |
  | Validation | 19 | Development reference |
  | Test | — | **Never read in M10**: no geometry, no statistics. |

  * Validation and test tiles are at least 1,536 m from every training tile (`outputs/m10/audit/dataset_audit.json`).
  * Validation tiles still neighbour each other within a city. The bootstrap therefore resamples whole
    tiles, and its interval is reported as approximate.
* **Unconditional evaluation.** Generators receive **no** validation or test statistics.
  * Conditioning on city or country: the city is drawn from the training tile proportions.
  * **"Conditional" evaluations** that do give a model real validation structure (for example, local
    streets generated on the real validation arterials) are run separately. They are labelled
    *conditional* and never mixed into unconditional scores.
* **Repeated use of validation.** Every M10 street experiment is scored on the same 19 validation tiles,
  so they are a development set, not a confirmation set. The final confirmatory number belongs to the
  untouched test split, in a later, approved step.

## 2. Street metrics

### 2.1 Validity

Measured as the share of tiles where each holds:
* the generator produced at least one edge;
* no two edges cross without a shared node (planarity; the sampler is planar by construction);
* the sampler did not hit MAX_NODES.

### 2.2 Realism

Ten per-tile metrics from `graph_metrics` (`src/urbanmorph/graphs/road_graph.py`):
* orientation φ;
* orientation entropy;
* junction angle deviation;
* intersections per km²;
* dead-end ratio;
* mean link length;
* street length per km²;
* loops per km²;
* share of degree-4 junctions;
* components.

For each metric:

**W1/IQR = Wasserstein-1 distance between the generated per-tile values and the validation per-tile
values, divided by the validation IQR.** It is unit-free; 0 means identical distributions.

**Primary score S10 = the mean of W1/IQR over the 10 metrics.** It uses 256 generated tiles per model
seed, with sampling seed 123 as in M6.

### 2.3 Paired tile bootstrap

* Validation tiles are resampled with replacement, 1,000 times.
* On each resample, S10 is computed for the candidate and for baseline A against the same resampled
  reference.
* ΔS = S10(candidate) − S10(A), reported with a 95% percentile interval.

### 2.4 Diagnostics

Reported, but never pass/fail criteria:
* share of junctions by mechanism (branch, merge, crossing, T) and their median deviation;
* snap deviation of raw turn angles by pop index: the free-running vs teacher-forced gap, i.e.
  exposure bias;
* rule counts per tile.

### 2.5 Diversity

* The per-tile spread of φ and of intersections per km² (IQR of generated tiles ÷ IQR of validation tiles).
* Spearman ρ between per-city generated medians and per-city **training** medians, for city-conditioned models.

### 2.6 Efficiency

* Parameter count.
* Training wall-clock time (M5 Pro, MPS).
* Sampling seconds per tile (CPU).

## 3. Pre-registered street experiments

All models train on the 565 training tiles with **2 training seeds (0, 1)**. Each seed generates 256 tiles.

| ID | Model | Status |
|---|---|---|
| A | GraphTurtle M6 roads v2, existing checkpoints, unchanged | Reference |
| E | Procedural perturbed grid with parameters from **training** tiles only (`M10_STREET_ARCHITECTURES.md` §3) | Reference (no pass/fail) |
| B | A + city conditioning: a one-hot city vector into the step context and the start head | Candidate |
| N | A trained with **denoising context noise**: node positions in the context are jittered (σ = 2.5 m, border nodes fixed), while targets point to the clean next nodes. This is the exposure-bias mitigation. | Candidate |
| C | Hierarchical, two stages. Stage 1 generates the major and mid network. Stage 2 generates local and minor streets with the stage-1 network pre-drawn and attachment starts. Scored unconditionally (stage 2 on stage-1 samples). | Candidate |
| CN | C trained with N's noise. **Run only if N passes.** | Candidate |
| C-oracle | Stage 2 of C on the **real** validation major and mid networks. | Conditional, exploratory (no pass/fail) |
| D | Lightweight raster diffusion | Pilot only; see `M10_STREET_ARCHITECTURES.md` §2.4 |

**Success criteria for a candidate (B, N, C, CN).** All four must hold for **both** seeds:

* **P1** The 95% paired-bootstrap interval of ΔS lies entirely below 0.
* **P2** S10 ≤ 0.8 × S10(A) of the same seed: at least a 20% improvement. A's measured S10 is 1.23 for both seeds.
* **P3** The median over tiles of junction angle deviation is ≤ 8°. Real validation is 4.2°; A is 13–15°.
* **P4** No single metric's W1/IQR is worse than A's by more than 0.25.

**B, additionally (conditioning works):** Spearman ρ ≥ 0.7 between the per-city median φ of B's samples
and the per-city median φ of the training tiles.

**Exploratory (no criteria):**
* temperature 0.5 for every model;
* C-oracle;
* any combination not listed above.

## 4. Block-level experiments

These are pre-registered in `M10_BLOCK_GENERATION.md` §5 before they run, with the same rules:
tiles as units, a paired bootstrap against the procedural M9 reference on the same real blocks,
2 seeds, and no test data.

## 5. Reporting rules

* Every number in the docs comes from a JSON file in `outputs/m10/`.
* Failed criteria are reported as failures. Nothing is re-run with a new seed to get a pass.
* Figures draw the street graph, and for block experiments the parcels and buildings as well.

## 6. Changes after pre-registration

None yet.
