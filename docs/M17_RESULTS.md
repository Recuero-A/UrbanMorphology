# Milestone 17 results: excess close junctions in sequential street generation

**Sources:**
* `outputs/m17/junction_audit.md`: reproduction and representation;
* `outputs/m17/diagnosis.md`: development diagnosis;
* `outputs/m17/fixdev.md`: development fixes and the frozen radius rule;
* `outputs/m17/holdout.md`: confirmatory holdout;
* figures in `outputs/m17/figures/`.

Protocol `docs/M17_PROTOCOL.md` (amendment 1 and frozen values before the holdout); method `docs/M17_METHOD.md`;
metrics `docs/M17_METRICS.md`; literature `docs/M17_LITERATURE.md`.

## 1. Reproduction and measurement

* **The earlier exploratory table reproduces** with the formal metrics when window means of REBUILT graphs are used:
  * training 0.121;
  * real pilot 0.179;
  * NA-SMC (KAIST) 0.441;
  * organic 0.774.
* **Pooled and window summaries differ and are both labelled.** Pooled close15 and pooled p10 are consistent
  (training 0.164 and 12.0 m). Window-mean p10 (35 m) is not comparable.
* **Representation matters.**
  * Real graphs carry an 8 m greedy snap, so real junctions are almost never < 8 m apart.
  * RAW generated graphs read 0.01–0.04 higher than the same graphs REBUILT.
  * All primary results use REBUILT; RAW is reported alongside.

## 2. Diagnosis (development: terrain-pack windows, 10 seeds)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 insertion geometry | **Supported** | T-snaps 23–36 % and crossing splits 18–30 % of close pairs; both edits never occur in training sequences. u-branch births 23–41 %. Reuse feasible for 52–88 % of split births |
| H2 network reward | **Partly, via density** | close15 rises with the NA weight (NA-SMC 0.120 → 0.175). Chosen closures are only 0.1–0.2 % of edits, so the mechanism is the dead-end stop penalty, which raises junction density |
| H3 density / style | **Mostly density** | At matched density: no excess for A / M\* / M13-SMC / NA-fixed / NA-random; +0.03–0.05 for NA-myopic / adaptive / SMC; organic +0.05 at a realistic density; radial none |

## 3. Confirmatory holdout

**Design.** 9 fresh metros never used before: Gwangju, Ulsan, Jeonju, Cheongju, Piura, Chiclayo, Huancayo, Castellón,
Figueres. 6 windows each, seeds 101–103. The real close15 of these windows averages 0.098.

**Primary outcome.** Excess = generated − same-window real close15 (REBUILT). Δ is relative to the original.
Bootstrap over metros → windows → seeds.

| Sampler | Original excess | F1 reuse r = 15 m: Δ [95% CI] | F0 post-hoc τ = 15 m: Δ | F0 τ = 9 m (secondary): Δ | Spacing ablation r = 15: Δ |
|---|---|---|---|---|---|
| M\* | +0.086 | −0.033 [−0.073, +0.009] · margins ok | −0.162, overcorrects (38 % of windows), **block margin FAIL** | −0.023 [−0.033, −0.014] · ok | −0.081 · ok |
| NA-fixed | +0.097 | −0.038 [−0.090, +0.012] · ok | −0.184, overcorrects, **FAIL** | −0.021 · ok | −0.116 · **FAIL** (blocks) |
| NA-SMC | +0.111 | −0.040 [−0.084, +0.004] · ok | −0.193, overcorrects, **FAIL** | −0.021 · ok | −0.110 · **FAIL** (blocks) |
| M\* + organic | +0.518 | **−0.153 [−0.187, −0.117]** · ok | −0.555, **FAIL** | −0.074 · ok | −0.343 · **FAIL** (length, blocks) |
| M\* + radial | +0.378 | **−0.100 [−0.137, −0.062]** · ok | −0.431, **FAIL** | −0.070 · ok | −0.269 · **FAIL** (blocks) |

**Other results:**
* **Preservation.** F1 met every margin for every sampler: OD reachability, coverage, street length, steep share,
  block size. Cost: the same K = 8 candidate budget, equal or fewer candidate evaluations, the same seconds per tile.
* **Mechanism.** F1 reduces the share of close pairs from T-snaps and crossing splits. The remaining excess is
  dominated by u-branch births (M\*: 32 % → 42 % of the remaining pairs), which F1 does not address.
* **Post-hoc cleanup at the same radius removes all excess but overcorrects.**
  * 34–40 % of windows fall more than 0.05 below the real value.
  * It enlarges blocks beyond the margin (median log area +0.25 to +0.64): it erases legitimate short blocks.
  * At a conservative 9 m it preserves morphology but removes less than F1 for every sampler.

## 4. Claims table (manuscript-ready; exact wording supported by the evidence)

| Claim | Status | Evidence |
|---|---|---|
| Sequential generation with crossing-split and T-snap insertion rules creates close junctions that the training data does not contain | Supported | Diagnosis H1; rules absent from teacher-forced sequences |
| Most of the excess close-junction share of the base learned samplers is explained by junction density | Supported (development) | Matched-density comparison; no excess for A / M\* / NA-fixed |
| The NA dead-end potential raises close junctions mainly by raising density, not through closure rewards | Supported (development) | Weight sweep; closures 0.1–0.2 % of edits; F2 had no effect |
| Online topology-aware junction reuse reduces excess close junctions without violating preservation margins | Supported for styled generation (organic −0.15, radial −0.10, CIs exclude 0); **directionally only** for base samplers (−0.03 to −0.04, CIs include 0) | Holdout |
| Online reuse outperforms post-hoc cleanup | **Qualified.** Better than cleanup at a morphology-safe radius (9 m); cleanup at the same radius removes more but overcorrects and erases short blocks | Holdout |
| Online reuse removes the excess | **Not supported.** About 35–40 % of the excess is removed for base samplers and ~30 % for styled ones; the rest comes from u-branch births | Holdout mechanism |

## 5. Limitations

* r = 15 m is the edge of the development sweep. A larger radius was not tested.
* GLO-30 is a surface model. The holdout windows were chosen by street length only.
* 3 seeds per window; 9 metros give coarse CIs.
* Real references use the training road definition (no service roads). Parking-lot clusters are therefore excluded by
  construction.
* u-branch births remain: a fix would need placement-aware proposals or a learned spacing signal (future work).
* Junction reuse is not new as an idea (Parish & Müller, NTG, CitySprout, RoadTracer). The contribution is the
  attribution, the matched diagnosis, and the paired online-vs-post-hoc holdout evaluation.

## 6. Reproduce

```bash
python scripts/m17_junction_audit.py                       # reproduction table
python scripts/m17_plans.py diagnose && bash scripts/m17_launch.sh diagnose 10 && python scripts/m17_diagnose.py
python scripts/m17_plans.py fixdev && bash scripts/m17_launch.sh fixdev 10 && python scripts/m17_fixdev_report.py
python scripts/m17_build_holdout.py                        # fresh metros (configs/m17_holdout.json)
python scripts/m17_plans.py holdout && bash scripts/m17_launch.sh holdout 12 && python scripts/m17_holdout_report.py
python scripts/m17_maps.py                                 # before / after maps with junction-event overlays
pytest tests/test_m17_junctions.py
```

## 7. Amendment 2: junction-degree prior (EXPLORATORY on the holdout; `outputs/m17/degdev.md`, `outputs/m17/degholdout.md`)

* **Problem.** Generated networks have too many 5+-way junctions: 2.3–3.7 % for the base samplers and 7–12 % with
  styles, against 0.7 % in the real holdout windows. F1 and F0 do not lower this; they slightly raise it, because
  merging close junctions creates high-degree nodes.
* **F4.** A soft prior from the real degree distribution (≈ −3.5 nats for a 5th branch).
* **Development.** F4 brings every sampler to the real 5+-way level. No λ met all frozen margins: NA-SMC and organic
  get larger blocks, and organic loses street length. No λ was selected; λ = 1 was run on the holdout as an
  exploratory comparison.

**Holdout, 5+-way share and close-junction excess:**

| Sampler | Original | F1 + F4 (λ 1) | Margins |
|---|---|---|---|
| M\* | 3.0 %, +0.085 | **0.6 %, +0.056** | ok |
| NA-fixed | 2.3 %, +0.094 | **0.7 %, +0.040** | ok |
| NA-SMC | 3.7 %, +0.110 | **0.7 %, +0.035** | block area +0.34 (fail) |
| M\* + organic | 11.8 %, +0.518 | **2.8 %, +0.316** | street km, block area (fail) |
| M\* + radial | 6.9 %, +0.378 | **3.2 %, +0.271** | ok |

* **Reading.** F1 + F4 is the best combination for both problems. Base samplers reach the real 5+-way level and
  lose 35–70 % of their close-junction excess.
* **Cost and caveat.** NA-SMC and organic become sparser with larger blocks. The holdout was reused, so these
  numbers are exploratory.

## 8. KAIST campus demo: original vs improved (F1 + F4, λ 1) — EXPLORATORY (`outputs/m16/kaist/m17_kaist.md`)

**Setup.**
* 1536 m campus on the GLO-30 DSM, with the same real arterials as Milestone 16; seeds 1–3; 7 samplers.
* Scored per 512 m window:
  * close15 (REBUILT);
  * pooled 5+-way share;
  * junction density;
  * realism typicality against the 156 real neighbourhoods of `scripts/m16_realism.py` (1 = as typical as a real
    neighbourhood).
* Real KAIST scores close15 0.098, 5+-way 0.5 %, typicality 1.59.

| Sampler | close15 orig → F1+F4 | 5+-way orig → F1+F4 | Typicality orig → F1+F4 | Street km | Unreachable across windows |
|---|---|---|---|---|---|
| A | 0.196 → 0.179 | 2.0 % → **4.2 %** | 1.25 → 1.22 | 39.4 → 37.7 | 0.13 → 0.17 |
| M\* | 0.191 → 0.182 | 2.3 % → **0.3 %** | 1.23 → 1.17 | 35.2 → 35.8 | 0.12 → 0.11 |
| M13-SMC | 0.108 → **0.157** | 1.3 % → 0.7 % | 1.40 → **1.53** | 29.3 → 26.4 | 0.37 → 0.21 |
| NA-fixed | 0.254 → **0.176** | 2.0 % → **0.7 %** | 0.99 → 1.05 | 45.5 → 42.2 | 0.05 → **0.23** |
| NA-SMC | 0.441 → **0.243** | 5.9 % → **0.5 %** | 1.11 → **0.93** | 76.2 → 61.2 | 0.05 → 0.08 |
| M\* + organic | 0.774 → 0.606 | 14.2 % → 5.1 % | 2.52 → 2.29 | 166.5 → 158.6 | 0.00 → 0.00 |
| M\* + radial | 0.708 → 0.625 | 9.1 % → 4.6 % | 2.47 → 2.27 | 157.2 → 158.0 | 0.00 → 0.00 |

**Reading (one campus, three seeds, no confidence intervals: a demonstration, not evidence).**
* **NA-SMC improves on every measure.** It is the most typical generated campus after the fix (0.93).
* **M\* and NA-fixed lose most of their excess 5+-way and close junctions.**
  * NA-fixed's typicality is about the same (0.99 → 1.05).
  * NA-fixed's cross-window reachability gets worse (0.05 → 0.23 unreachable).
* **A gets worse on 5+-way (2.0 → 4.2 %).** With K = 1 there is nothing for the degree prior to reweight, and the
  reuse merges raise node degree. This matches §7: F4 needs candidates.
* **M13-SMC gets worse on close15 (0.108 → 0.157) and typicality (1.40 → 1.53).**
* **The style samplers improve but stay far from real.** Their density problem (§2 H3) is not a junction-placement
  problem.

**Figures.**
* `outputs/m17/figures/fig_m17_kaist_maps.png`: 2D maps, one row per sampler: real | original | improved, with 5+
  junctions marked.
* `fig_m17_kaist_3d_blender_aerial.jpg` and `fig_m17_kaist_3d_godot_oblique.jpg`: the 3D cities, original | improved.

**Reproduce.**
* `python scripts/m17_kaist.py --one KEY SEED` for each sampler and seed, then `--assemble`.
* 3D: `scripts/m16_city3d_big.py --method "<label> [improved]" --graphs outputs/m16/kaist/m17_graphs.pkl --tag kaist_m17`.
