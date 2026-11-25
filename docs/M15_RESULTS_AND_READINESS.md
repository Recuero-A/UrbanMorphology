# Milestone 15 results and readiness (PILOT)

* **Sources.** Every number is from `outputs/m15/REPORT.md` (`scripts/m15_report.py`, stored runs only).
* **Protocol.** `docs/M15_PROTOCOL.md`, frozen before evaluation. Amendment 1 adds detail to the real-city pilot.
  Amendment 2 adds a fixed starting point for synthetic sites; it was made after partial results.
* **Data.** Development data only. No held-out metro or sealed test split was read. All results are **pilot**
  results.

## 1. What was tested

**Style guidance** is an optional, selection-normalised factor on candidate headings: grid, organic, radial, plus
cul-de-sac termination and loop controls. It was added on top of the unchanged M13 / M14 samplers. Core comparison:
M\* and NA-fixed, each with style off and on, on the same terrains and seeds.

**Comparisons:**
* the other samplers with style off: A, NA-myopic, NA-SMC, M13-SMC, the procedural P;
* morphology-only (terrain potentials off);
* a hard grade ceiling;
* a Chen-inspired tracer;
* a mismatched-terrain negative control;
* temperature, tested separately;
* a second backbone.

**Real-city pilot:** 120 development windows in 6 metros (Girona, Manresa, Busan, Daejeon, Arequipa, Cusco) with
GLO-30 elevation.

## 2. Pre-registered outcomes

**O1. Style control on flat terrain** (paired on − off; smallest effects ±0.05 R4 or share, −5 loops / km², +0.3
dead ends / km):

| Style | Descriptor | M\* | NA-fixed | Verdict |
|---|---|---|---|---|
| grid | R4 | +0.18 [+0.05, +0.35] | +0.23 [+0.13, +0.33] | Met |
| radial | radial + tangential share | +0.23 [+0.10, +0.35] | +0.20 [+0.11, +0.28] | Met |
| organic | R4 vs grid style | −0.27 [−0.41, −0.14] | −0.32 [−0.43, −0.22] | Met |
| cul-de-sac | loops / km² | −6.9 [−37.8, +27.5] | −27.1 [−58.7, 0.0] | Not shown |
| cul-de-sac | dead ends / km | +0.60 [+0.01, +1.15] | +0.05 [−0.38, +0.56] | Mixed |

The controllability check agrees. Grid field alignment rises from 0.73 to 0.93 (M\*) and from 0.45 to 0.90
(NA-fixed) on flat terrain.

**O2. Retention on terrain** (effect on the five non-flat conditions ÷ flat effect; non-inferior if the lower bound is
≥ 0.5): **not demonstrated for any style or sampler.**

| Style | M\* | NA-fixed |
|---|---|---|
| grid | 0.59 [0.17, 2.36] | 0.13 [−0.17, 0.51] |
| radial | 0.68 [0.36, 1.56] | 0.83 [0.49, 1.54] |
| organic | 0.67 [0.31, 1.41] | 0.18 [−0.06, 0.46] |

* The radial style keeps most of its effect.
* With NA-fixed, the grid and organic effects largely vanish on terrain: its look-ahead favours terrain-following
  continuations over the declared field.
* With the cul-de-sac controls, the flat effect is too uncertain to form a ratio.

**O3. Do NA-fixed's connectivity gains survive style?** Not met as pre-registered.
* The synthetic sites offer no gain to survive: with style off, NA-fixed − M\* = +0.014 [−0.011, +0.054].
* Pooled with style on, the contrast is −0.015 [−0.038, +0.008].
* On the real Task B windows (secondary), the gain does survive: style off −0.010 [−0.019, −0.001]; declared grid
  −0.011 [−0.015, −0.007].

**O4. Terrain exposure** (share of ground street steeper than 15 %, the four sloped conditions):

* **(a) Style on − off, margin +0.02:**
  * non-inferior for M\* grid, radial and cul-de-sac, and for NA-fixed radial and cul-de-sac;
  * not shown for M\* organic and NA-fixed grid / organic (upper bounds 0.029–0.040).
* **(b) Against the tracer (pre-registered 6 km budget):** the samplers have 0.10–0.14 lower exposure, but they build
  only 0.26–0.34 × the tracer's length. This is flagged "fewer roads" and does **not** count.
* **(b) Against the negative control** (terrain rotated 90°): exposure is 0.12–0.15 lower at 0.86–1.15 × the length.
  Terrain conditioning, not length, drives the exposure.
* **(c) Hard ceiling (real p99):**
  * it removes almost all of the > 15 % share (NA-fixed grid 0.058 → 0.008);
  * the share between 10 % and 15 % is unchanged (0.245 vs 0.238), so the gain is compliance.
  * Tighter ceilings break the network: at 12 % OD unreachability is 0.121 and at 8 % it is 0.184, against 0.020
    without a ceiling, and length falls to 1.27 and 0.99 km.

## 3. Secondary and exploratory results

* **Temperature** (separately; 0.5 / 0.8 / 1.0):
  * no consistent change in rejected-candidate share (0.075–0.095) or diversity (Jaccard 0.85–0.90);
  * grid R4 on flat falls slightly with temperature (M\*: 0.91 → 0.87 → 0.85).
* **Un-normalised style factor (ablation):** shorter networks (1.73 vs 1.88 km) and a weaker grid (flat R4 0.87 vs
  0.91). This confirms the termination coupling the normalisation removes.
* **Second backbone (TN_s1):** the grid effect replicates (M\* +0.17, NA-fixed +0.19 R4).
* **EXPLORATORY, length-matched tracer (2 km budget, added after results were seen):** both samplers with style have
  0.10–0.16 lower exposure than the tracer, at 0.74–0.95 × its length. This is the comparison the claim rule needs,
  but it was not pre-registered.
* **Real-city pilot, Task A (style off).** Precision 0.97–0.98 for the learned samplers. Coverage, as a fraction of
  the real-vs-real benchmark (target 0.8):

  | Sampler | Coverage | Fraction of benchmark |
  |---|---|---|
  | A | 0.83 | 0.85 |
  | M\* | 0.81 | 0.83 |
  | NA-fixed | 0.83 | 0.85 |
  | NA-SMC | 0.93 | 0.95 |
  | procedural P | 0.38 | 0.39 |

  Only P fails the target.
* **NA-fixed vs M\* on real windows:** OD unreachability lower in 4 of 6 metros. Metro mean −0.016, leave-one-metro-out
  [−0.020, −0.012]. Busan is the outlier for both samplers (OD about 0.14–0.16; coastal windows).
* **Terrain on real windows:** on hilly windows generated streets are slightly steeper than real (0.15–0.17 vs 0.135
  share above 15 %, 30 m spans on a DSM).
* **Task B (declared grid on real grid-like windows):** R4 rises (+0.11), but no non-conditioned feature (junction
  shares, block area, circuity) moves closer to the real network.
* **Elevation source (Spain):** MDT05 instead of GLO-30 makes generated networks about 0.7–0.9 km longer. The DEM
  source matters, and GLO-30 is a DSM.
* **Sample size:** with the observed metro-level SD (0.014), a confirmatory benchmark needs about **17 metros** to
  detect a 0.01 connectivity effect (α 0.05, power 0.8).

## 4. Claim rule (protocol §6)

| Condition | Status |
|---|---|
| Lower terrain exposure than the strongest matched baseline at useful coverage | **Not met as pre-registered.** Against the tracer at the pre-registered budget the samplers build too few roads. At matched length (exploratory) they are clearly better. The hard ceiling has lower exposure, through compliance |
| Morphology retained within the margin | **Not met.** No style passes O2. Radial comes closest (NA-fixed 0.83 [0.49, 1.54]) |
| Held-out distributions with adequate diversity and coverage | **Not assessed.** Held-out metros are reserved |

**Revised claim (what the evidence supports).**
1. Inference-time style guidance controls street morphology on flat terrain (grid, radial, organic), on two
   backbones.
2. On terrain, the learned prior and the look-ahead pull streets back toward terrain-following layouts. Control
   breaks down most for NA-fixed with grid and organic fields, and least for radial fields.
3. Terrain exposure is driven by terrain conditioning (negative control), not by the style.
4. NA-fixed's connectivity gain persists with style on real windows but cannot be shown on synthetic sites, where
   neither sampler fragments much.
5. Hard grade ceilings trade exposure for broken connectivity.

## 5. Readiness

* **Status: pilot.** Not ready for a confirmatory claim or a main-track submission.

**What would be needed:**
1. Authorisation to build and evaluate the declared held-out metros (Spain: Lleida, Alicante; South Korea: Seoul,
   Daegu; Peru: Lima, Trujillo; USA as an external country).
2. About 17 metros for a confirmatory connectivity effect.
3. A pre-registered, length-matched tracer comparison (the exploratory result suggests it would pass).
4. A decision on retention. Options:
   * a stronger or terrain-aware style weighting (κ was at the edge of its pre-declared range for organic and
     radial);
   * style applied inside the look-ahead roll-outs with its own weight;
   * accepting a terrain-dependent compromise and reporting where it occurs.
5. A fix for the start head, which ends tiles early on smooth terrain (amendment 2).
6. Bare-earth elevation for Korea and Peru, or a DSM-robust model input.

**Deliverables:**
* the runnable command `bash scripts/m15_all.sh`;
* frozen configurations (`configs/m15_style.json`, `configs/m15_plans/`);
* the versioned site manifest (`configs/m15_benchmark/`) and preprocessing (`scripts/m15_build_benchmark.py`);
* metric definitions with units (`docs/M15_METRICS.md`);
* raw per-run results (`outputs/m15/runs/`, git-ignored, regenerable);
* matched comparison figures (`outputs/m15/figures/`);
* this report.
