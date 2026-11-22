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
