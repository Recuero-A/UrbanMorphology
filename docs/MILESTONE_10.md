# Milestone 10: learning coherent urban structure through hierarchical generative modeling

**Status (2026-11-01).** Every criterion was applied as pre-registered in `M10_EVALUATION_PROTOCOL.md` and
`M10_BLOCK_GENERATION.md` §5. The one documented deviation, a training fix, is protocol §6.1.

* **Streets: no pre-registered candidate passes.** All fail P3 (median junction deviation ≤ 8°).
  * **N**, denoising context noise: S10 halves, from 1.23 to 0.50–0.51, and passes P1, P2 and P4.
  * **C_w**, hierarchical: S10 0.65–0.85, also passes P1, P2 and P4.
  * **B**, city conditioning: does not change the geometry.
  * **Exploratory, the best model found:** N at sampling temperature 0.5 scores S10 0.30–0.33 with a junction
    deviation of 5.1–5.5°. It beats the procedural grid E (0.64–0.67), but has no pass/fail status.
* **Training defect found and fixed.** The M6 training recipe can collapse: a dying ReLU in the step decoder
  leaves the stop head constant. This happened in 5 of 8 first runs. A 300-step warmup fixes it (§3.2).
* **Blocks: BF and BlockGen both fail their pre-registered criteria.**
  * BF (the data-based procedural fix) cuts green coverage from 0.40 to 0.05–0.07. It fails F2 (Spanish built
    share 0.47, observed 0.93) and, for seed 0, F3.
  * BlockGen (4.2M-parameter conditional DDPM) is valid (G2) but fails G1 and G3: about 97 buildings per tile
    against 382 real, and over-filled blocks. Half of its count gap is the binary-mask representation.
  * BlockGen **inpainting passes I1** for both seeds (hidden-side coverage error roughly halves); I2 fails for
    seed 1.
* **Integration (exploratory):** complete tiles from N at T = 0.5 streets, with BF parcels and buildings,
  reach building S_B 0.49 against the real validation tiles. BlockGen on the same streets: 0.92.

Detail documents:
* `M10_LITERATURE_REVIEW.md`
* `M10_STREET_ARCHITECTURES.md`
* `M10_BLOCK_GENERATION.md`
* `M10_EVALUATION_PROTOCOL.md`

Every number below comes from a JSON file in `outputs/m10/`. Test tiles were never read.

## 1. Phase 1: research and repository inspection

### 1.1 Literature (8 papers; `M10_LITERATURE_REVIEW.md`)

* **Read in full:** 5 of the 8 papers.
* **Not fully read:**
  * CaRoLS and GenBuilding-DDPM were inaccessible;
  * the WSDM 2026 paper was available as an abstract only.

Conclusions:
1. **Streets.** None of the papers read in full generates streets as an arterial-then-local hierarchy with a
   small vector model. The diffusion approaches that work use pretrained SD 1.5 (about 105 GPU-hours) or 20×
   more data than we have. **No novelty is claimed:** three papers were not fully read, and this was not a
   systematic search.
2. **Buildings.** The established approach is to condition on the block shape and place buildings on real
   streets (GlobalMapper; npj 2026). BlockGen follows it with a small raster DDPM.
3. **Exposure bias** is not treated in NTG, the closest street model.

### 1.2 What the repository inspection found

| Finding | Where | Consequence |
|---|---|---|
| About half of all generated junctions come from geometric rules (merge 28–33%, crossing 10–12%, T 11–13%). They have a median deviation of 23–26°, against 11–15° for junctions the network itself branched. These rules are not in the training sequences. | `graphs/generate.py`; `outputs/m10/diagnosis/` | Train/generate mismatch. The hierarchical model (C) was designed to reduce it. |
| Exposure bias: the free-running turn error grows from 2.8° to 7.9° over a tile, while teacher-forced stays at 0.8–1.5°. | diagnosis D2, `drift.json` | The denoising variant (N) was designed for this. |
| Over-density: 5× the intersections and 6× the loops of real tiles. Temperature does not change it. | diagnosis D1, D5 | The largest single error. |
| The M9b "no street access → green" rule contradicts the data. Land without frontage is 74% (Spain) or 94% (US) built, and only 1–3% green. | `outputs/m10/landuse/` | Fixed procedurally in BF. |
| **The M6 training recipe can collapse** (dying ReLU in the step decoder's output MLP). Both M6 seeds survived it by chance. | found in M10, §3.2 | Fixed with warmup. Every run now logs `decoder_units_alive`. |
| Street classes "flicker" along a street. The class is sampled per step, so a major road shows up as short dark dashes. | `samples_grid.png` (A, B_w, N rows) | Not measured by S10. Listed as a next experiment. |

## 2. Dataset audit and feasibility (no downloads)

Source: `outputs/m10/audit/dataset_audit.json`.

**Current data**
* Size: 9 cities in 3 countries, 565 training tiles, 307 training spatial blocks (3 km), and 6,733 closed
  street blocks in train + validation.
* Validation and test tiles are at least 1,536 m from every training tile.
* Storage: 0.45 MB of vectors per tile.

**What limits the models**
* **Streets.** The effective sample size is the number of spatial blocks (about 300), not the number of tiles.
  City conditioning (B) has only 2 validation tiles per city to show an effect against.
* **Buildings.** About 15.5k training blocks and 17.6k windows (`data/processed/m10/`, git-ignored). That is
  enough for a 4.2M-parameter DDPM (§4), but 10–20× fewer blocks than GlobalMapper used.

**Expansion options**

These are estimates only. Nothing was processed or downloaded.

| Option | Source | Gain | Cost (estimated) | Constraint |
|---|---|---|---|---|
| E1: more cities inside the **already cached** Geofabrik extracts (Cataluña, Madrid, Valencia, SoCal, Japan Kantō, Kansai and Chūbu, South Korea; 7.7 GB on disk) | local, no download | about 25 cities, 3,000–4,500 tiles, about 400–600 spatial blocks (5–7×) | ≈ 2–4 h CPU (clip 1–2 min per city; about 1–2 s per tile); +2 GB processed | Needs approval (dataset expansion). Same split rules; existing test split untouched; new held-out cities for leave-one-city-out. |
| E2: Spanish cadastral **land-use codes** (separate Catastro product) | new download | separates vacant, courtyard and private-yard land (land-use audit §2.3) | ≈ 0.5–1 GB, < 1 h | Catastro licence v1.0: transformed use only, raw data never committed. Needs approval. |
| E3: US parcels for Chicago | Cook County | completes US parcels | small | **Licence unspecified: excluded** (M8). |
| E4: Overture / OSM building heights (for 3D, as in npj 2026) | new download | heights | several GB | Not needed for 2D. Deferred. |

**Recommendation:** do E1 first. It is free, local, and addresses the main limit, which is independent street
samples.

## 3. Streets (Phases 2–3)

`scripts/m10_eval_streets.py` → `outputs/m10/streets/eval.json`, `samples_grid.png`, `s10.png`.

Setup: 256 tiles per seed, sampling seed 123, compared with the 19 validation tiles. S10 is the mean of
W1/IQR over 10 metrics (lower is better).

### 3.1 Pre-registered results

| Model | S10, seed 0 / 1 | ΔS vs A, 95% CI (s0) | Junction deviation, median (real 4.2°) | Intersections per km² (real 76) | φ (real 0.59) | Verdict |
|---|---|---|---|---|---|---|
| A (M6 roads v2) | 1.230 / 1.225 | — | 15.3° / 13.4° | 378 / 385 | 0.10 | reference |
| E (procedural grid) | 0.672 / 0.644 | [−0.73, −0.38] | 3.4° / 3.5° | 132 / 141 | 0.93 | reference (fails P4: deg-4 share 0.72, real 0.31) |
| **N** (context noise) | **0.512 / 0.501** | [−0.74, −0.62] | 10.5° / 11.4° | 145 / 134 | 0.39 / 0.35 | **FAIL** (P3) |
| **C_w** (hierarchical) | 0.850 / 0.648 | [−0.40, −0.34] | 13.8° / 13.1° | 238 / 137 | 0.14 / 0.21 | **FAIL** (P3), vs A and vs A_w |
| **B_w** (city one-hot) | 1.274 / 1.161 | [+0.03, +0.05] | 14.8° / 13.8° | 378 / 347 | 0.10 | **FAIL** (P2, P3; seed 0 also P1 and ρ 0.60 < 0.7) |
| A_w (A + warmup, control) | 1.404 / 1.394 | [+0.17, +0.18] | 15.8° / 16.0° | 435 / 423 | 0.09 | control |
| B, C (first runs) | B seed 1 and both C seeds collapsed (§3.2) | — | — | — | — | training failures, kept on record |

Reading the table:
* **Warmup alone does not improve A.** A_w is worse than A, so the gains of C_w over A are not warmup gains.
  C_w also passes P1, P2 and P4 against A_w.
* **N removes the growth of exposure bias** (`outputs/m10/diagnosis/drift.json`; median snap deviation of raw
  turn angles, free-running):

  | Pop index | 0–10 | 10–40 | 40–120 | > 120 |
  |---|---|---|---|---|
  | A | 2.8° | 3.1° | 4.8° | 7.9° |
  | A_w | 2.6° | 3.3° | 5.0° | 8.6° |
  | N | 9.7° | 8.4° | 8.3° | 9.7° |

  * N's error no longer grows with sequence length.
  * Its single-step distribution is wider: teacher-forced error is 5.8–7.2°, against 0.8–1.5° for A, because
    it is trained on noisy contexts.
  * Result: structure improves (density, loops, link length, street length W1/IQR 0.24–0.50, against 0.84–3.3 for A),
    but junctions are not square enough for P3.
* **N's junctions:** merge junctions fall from 23.7° to 14.1° deviation, and crossing splits from 10% to 4% of
  junctions. Steps that recover their heading cross existing streets less often.
* **C_w fixes density but not angles.**
  * Intersections fall to 137–238 per km².
  * Junction deviation stays at 13–14°: local streets drawn by stage 2 are as irregular as A's
    (`samples_grid.png`, C_w rows).
  * Stage 1 alone is fine: major networks are sparse corridors, as in the data.
  * C-oracle_w (stage 2 on the *real* major networks; conditional) scores S10 0.72 / 0.50, again with 13° junctions.
    The hierarchy's weak part is the local-street stage, not the arterial stage.
* **B: the city vector barely changes the geometry.**
  * All cities get φ ≈ 0.10, while real cities range from 0.57 to 0.99.
  * Spearman ρ against training per-city φ is 0.60 / 0.92 (threshold 0.7), computed on near-identical medians.
  * The model ignores the condition in favour of the local canvas.

### 3.2 The training collapse (protocol §6.1)

* **Affected runs:** B seed 1, C1 seeds 0 and 1, C2 seeds 0 and 1 (`scripts/m10_train_roads.py`, the unchanged
  M6 loop).
* **Symptom:**
  * only 4–19% of the step decoder's output-MLP units are alive;
  * the stop logit is constant (P(stop) = 0.50 at every step);
  * samples have a median of 5–12 edges.
* **Cause:** dying ReLU in the first ~10 Adam steps at learning rate 1e-3.
* **Fix:** a 300-step linear warmup. With it, all 8 retrained runs keep 60–73% of units alive and have a working
  stop head (stop-logit SD 2.4–3.1). Their summaries record `decoder_units_alive`.

### 3.3 Exploratory (no criteria)

| Model | S10, seed 0 / 1 | Junction deviation | Intersections per km² | φ | Note |
|---|---|---|---|---|---|
| **N, T = 0.5** | **0.330 / 0.300** | 5.1° / 5.5° | 103 / 90 | 0.74 / 0.72 | Best model found. Every metric's W1/IQR ≤ 0.5 except dead-ends (0.74–0.77). |
| A, T = 0.5 | 1.189 / 1.481 | 9.7° / 11.5° | 378 / 515 | 0.19 / 0.11 | Temperature does not fix A's density. |
| C-oracle_w (conditional) | 0.721 / 0.496 | 12.9° / 13.1° | 250 / 118 | 0.26 / 0.41 | Stage 2 on real arterials. |
| CN_w (C + N's noise) | 0.460 / 0.496 | 11.3° / 11.4° | 124 / 128 | 0.32 / 0.31 | About equal to N. Not pre-registered (N failed P3), so no pass/fail. |
| **CN_w, T = 0.5** | **0.298 / 0.320** | 6.3° / 6.4° | 95 / 114 | 0.60 / 0.60 | About equal to N at T = 0.5. |
| C_w, T = 0.5 | 0.890 / 0.637 | 10.5° / 10.2° | 303 / 151 | 0.18 / 0.22 | |
| B_w, T = 0.5 | 1.310 / 1.117 | 10.5° / 9.6° | 429 / 382 | 0.16 / 0.17 | |

* **Noise, not the hierarchy, carries the improvement.** CN_w ≈ N at both temperatures, and C_w without noise is
  far behind.
* **Why N at T = 0.5 works where A at T = 0.5 does not.** N's errors do not compound, so narrowing its wide
  step distribution recovers square turns without the free-running drift. A at low temperature still drifts.
* **Caveat:** T = 0.5 is a sampling choice made on the same 19 validation tiles. It must be confirmed on the
  test split before it counts.

## 4. Blocks (Phase 4; detail in `M10_BLOCK_GENERATION.md` §6)

| Experiment | Result | Verdict |
|---|---|---|
| Land-use audit | Land without frontage is 74% (Spain) or 94% (US) built, and 1–3% green. Unbuilt block interiors are mostly untagged (yards, courtyards, parking), not green. | Categories now: public green, private green, unbuilt other, built |
| **BF** (procedural, P(green \| no frontage) from data) | Green 0.05–0.07 (B: 0.40; real 0.013). Buildings 277–292 per tile (B: 114–132; real 382). Built share Spain 0.47, US 0.82–0.85 (observed 0.93, 0.92). | **FAIL** (F2; F3 for s0) |
| **BlockGen G** (DDPM, real streets) | S_B 1.03–1.07 (BF 0.61–0.64). 96–98 buildings per tile. Coverage 0.45–0.50 (real 0.29). No road overlap. | **FAIL** (G1, G3) |
| **BlockGen I** (inpainting) | Hidden-side coverage error 0.13–0.17, against 0.29–0.38 unconditional (CI below 0 for both seeds). Count error 3 vs 4 and 3 vs 3. | **I1 pass**; I2 fails for s1 |
| Representation ceiling (diagnostic) | Real buildings round-tripped through the canvas: 462 → 235 per tile, rectangularity 0.97 → 0.82 | Binary mask merges attached buildings |

Figures: `outputs/m10/procedural_fix/zoom_B_BF.png` (streets, parcels, buildings), `outputs/m10/blocks/blockgen_vs_real.png`
and `inpainting.png`.

BlockGen produces no parcels, so its figures draw streets and buildings only. The integration figure draws BF parcels.

## 5. Integration (Phase 5, exploratory; `scripts/m10_integrate.py` → `outputs/m10/integration/`)

**Setup**
* Streets: the first 32 cached tiles of **N at T = 0.5**, seed 0, the best street model.
* Cities are drawn with the training proportions.
* Blocks: BlockGen on every tile, and BF on the 21 Spanish and US tiles (parcel models exist only there).
* Reference: building metrics against all 19 real validation tiles.

| | Buildings per tile (real 462) | Coverage (real 0.27) | Rectangularity (real 0.97) | S_B |
|---|---|---|---|---|
| N streets + **BF** | 315 | 0.20 | 0.95 | **0.49** |
| N streets + BlockGen | 157 | 0.46 | 0.76 | 0.92 |

**Caveats**
* BF's 0.49 is not comparable with its 0.61–0.64 on real streets: those are different tiles (10 M9 tiles
  against all 19), and Japan has no BF.
* One of the 32 street samples is nearly empty, and BF zones the resulting tile-sized block as green
  (`integration_N_T05.png`, last row). Empty or very sparse street samples need a guard before generating
  blocks.

**Result:** generated streets → procedural parcels → buildings runs end to end, with no test data and no
validation statistics as inputs.

## 6. Architecture recommendation

1. **Streets: GraphTurtle + denoising context noise (N), sampled at T ≈ 0.5.**
   * Best realism found: S10 0.30–0.33, against 1.23 for A and 0.64–0.67 for the procedural grid.
   * N failed P3 at T = 1, and T = 0.5 was chosen on the same validation tiles. **Before any claim, confirm
     N at T = 0.5 on the untouched test split**, as one pre-registered comparison against A and E.
2. **Hierarchy: keep it as an interface, not as the source of realism.**
   * CN_w ≈ N, so the hierarchy adds nothing measurable once noise is used.
   * It is still useful for user control, for example "draw the arterials and fill the rest", which C-oracle
     shows works (S10 0.50–0.72).
3. **City conditioning as a one-hot (B) is not enough.**
   * The model ignores it, so all cities get φ ≈ 0.10.
   * Next attempt: a condition the model cannot ignore, such as a target orientation histogram or block-size
     prior from the training city, applied in the *start head and step decoder* together with N.
4. **Blocks: BF (procedural, learned zoning) remains the best unconditional option.**
   * BlockGen is worth keeping for **inpainting and editing** (I1 passes).
   * Before it can replace BF it needs:
     * an instance-aware output (a boundary channel, or per-building masks), which removes the −49% ceiling;
     * an open-space or land-use condition.
5. **Training:** every GraphTurtle run must use warmup and log `decoder_units_alive`, so that a collapsed run
   is flagged before it is evaluated.

## 7. Compute cost (M5 Pro, MPS; up to 6 jobs sharing the GPU)

| Item | Wall-clock time |
|---|---|
| Street run, full graph (A_w, B, N), 60 epochs | 4,500–5,200 s each (≈ 2,100–2,300 s alone, as in M6) |
| Street run, stage 1 / stage 2 | 1,200–1,900 s / 1,600–3,700 s each |
| Street runs total | 20 runs, 18.0 h of summed wall-clock (run 4 at a time) (B, N, C1, C2, A_w, B_w, C1_w, C2_w, CN_w; 2 seeds) |
| BlockGen, 20k steps × 32 | 20,200–20,400 s each, 2 seeds |
| Street sampling, 256 tiles on CPU | 36–125 s per model-seed (0.14–0.49 s per tile) |
| BlockGen sampling | 4–6 s per tile (MPS) |
| Block evaluation (G and I) | 23 min; BF evaluation 2 min (`procedural_fix_log.txt`) |

Parameters:
* GraphTurtle: 1.38M (C = 2 × 1.38M).
* BlockGen: 4.16M.

Disk: checkpoints and `data/processed/m10/` are git-ignored.

## 8. Prioritised next experiments

1. **Test-split confirmation (needs approval):** N at T = 0.5 against A and E, 2 seeds, criteria fixed in
   advance. This is the only route to a claim.
2. **Street-class consistency.** Sample one class per street, not per step, by carrying the class across a
   chain of degree-2 nodes. This fixes the "flickering" arterials, which S10 does not measure.
3. **Instance-aware BlockGen.** Add a building-boundary channel (3-class output: background, interior,
   boundary). Measure the round-trip ceiling first (it should get close to 462), then retrain. About 6 h per
   seed.
4. **BlockGen open-space conditioning.** Feed BF's learned zoning decision (green or building block) as an
   input channel. This tests whether over-filling disappears.
5. **Dataset expansion E1** (cached extracts, about 2–4 h CPU, needs approval). It directly targets the
   ≈ 300-block limit, and allows leave-one-city-out tests for conditioning.
6. **Conditioning that cannot be ignored:** a per-city orientation or block-size prior inside N. Re-test the
   ρ ≥ 0.7 criterion.
7. **BF built share:** the Spanish built share is 0.47 because `not_fitted` lots get no building. Fit
   templates to small or irregular interior lots, or merge them. Pre-register F2 again.

## 9. Files

| Purpose | File |
|---|---|
| Street diagnosis, drift | `scripts/m10_street_diagnosis.py`, `scripts/m10_drift.py` |
| Street variants (training, warmup, health check) | `scripts/m10_train_roads.py`; `graphs/sequence.py` (`given`, `clean_nodes`), `graphs/generate.py` (`given`, `cond`), `models/graph_gen.py` (`n_cond`) |
| Procedural grid E | `src/urbanmorph/graphs/procedural.py` |
| Street evaluation | `scripts/m10_eval_streets.py` → `outputs/m10/streets/` |
| Land-use audit, BF | `scripts/m10_landuse_audit.py`; `layout/tidy_parcels.py` (`green_fix`); `scripts/m9_evaluate.py --m10` |
| BlockGen | `layout/block_canvas.py`, `models/block_diffusion.py`, `layout/block_gen.py`, `scripts/m10_block_data.py`, `scripts/m10_train_blockgen.py`, `scripts/m10_eval_blocks.py`, `scripts/m10_block_roundtrip.py` |
| Integration | `scripts/m10_integrate.py` |
| Tests | `tests/test_m10.py` |
