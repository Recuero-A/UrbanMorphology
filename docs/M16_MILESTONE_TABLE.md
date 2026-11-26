# M16: milestone comparison table (M12–M15) for matched replays

Purpose: list, with file:line evidence, every setting that differs between how M12–M15 runs and demos were made, to
diagnose whether "terrain adaptation" regressed. Read from branches `milestone-12` … `milestone-16` (2026-11-21).
"not verified" = stated in docs or commit messages but not re-run or re-read in code here.

**Code identity across branches** (`git diff --stat`): `models/`, `terrain/` (synthesis, feasibility), `city3d/`,
`terrain_generate.py`, `terrain_metrics.py`, `procedural_terrain.py`, `configs/m12_feasibility.json` are unchanged
M12 → M16. `tc_sampler.py` is unchanged M13 → M14; M15 adds style / `grade_cap` / `fixed_start` only (62 lines,
defaults off; style-off replay 95/95 graphs identical, `docs/M15_AUDIT.md` §5). `generate.py`, `sequence.py`,
`terrain_sequence.py` changed M12 → M13 by a resolve/commit split and domain-size parameters (claimed bit-identical,
commit 0253808; not re-run here). All checkpoints are the M12 ones (`checkpoints/m12/*/best.pt`, dated 4–6 Nov).

## 1. Milestones at a glance

| | M12 | M13 | M14 | M15 |
|---|---|---|---|---|
| Final commit (results) | 1d7d471 (7a67a2f results; 671b4e5 training) | d4d73c3 (5105b68 report; e3cd5de M* frozen) | a25a202 (6d13b6e results; 6eccee5 prereg) | c2d2ce9 = `milestone-16` HEAD (b5b8b47 protocol; 0e36759 amendment 2) |
| Backbones used | N0, TN, TNlp (s0, s1); ablations TNelev/slope/water/elevslope (s0) | TN, TNlp, TNse, N0 × s0–s2 (s2 + TNse trained in M13, `configs/m13_full.toml`) | TN_s0–s2 primary; TNlp, TNse for H5 | TN_s0 primary; TN_s1 replication |
| Sampler under test | `generate.sample_graphs` + `TerrainPolicy` none / R / HR / HRV | `tc_sampler` A (standard) vs M* (energy K8 λ0.5); B–G, E+LA | M* vs NA-myopic / fixed / adaptive / random / SMC, M13-SMC, M*+repair, P | M*, NA-fixed (± style); A, NA-myopic, NA-SMC, M13-SMC, P, TRACER |
| Evaluation sets | Regime A: 52 dev tiles (12 val + 40 San Diego val_t); B: 256 training-terrain tiles, seed 123; C: 10 synthetic scenarios × 5 | DEV-A 25 tiles / 13 blocks (selection), DEV-B 27 / 14 (confirm), GEN 100 (Girona 30, Manresa 30, SF 40); X5 synthetic 1536 m | GEN2 95 (Alcoi 20, Tarragona 25, Santa Barbara 25, Sacramento 25) primary; DEV-B, GEN (re-run after contamination, `configs/m14_plans/gen_rerun_*`); train_cal 48; shifts `@x2`, `@rough` | syn_dev (6, tuning), syn_eval (6), syn_eval_mismatch; real_dev 120 windows / 6 metros (GLO-30), real_dev_gridB 64, real_dev_spain_mdt05 60 |
| Inference seeds | 1000 + training seed, 4 samples per tile (`m12_evaluate.py:41,127`) | 1–4 (`outputs/m13/plans/*.json`) | 1–3; 1–2 for H5, dm, ablations, shifts | syn 101–110; real 101–103; dev tuning 1–3 |
| Device | MPS if available (`m12_evaluate.py`, not verified) | MPS if available (`m13_experiments.py:223`) | CPU, 2 threads (`M14_EXPERIMENTAL_PROTOCOL.md:213`) | CPU forced (`m15_run.py:154`) |

## 2. Backbones and terrain inputs

| Item | Value | Evidence |
|---|---|---|
| Architecture | TerrainGraphTurtle, hidden 256, 10-component mixture, 4 classes, structure head; noise_m 2.5, 60 epochs, patience 10 | checkpoint `config` |
| Training data | `graphs3d.pkl` train: 685 tiles (BCN 39, MAD 70, VLC 31, LAX 191 incl. 120 steep, NYC 71, CHI 71, Japan 212 without DEM: `valid` = 0); 473 with a full DEM | counted from pickle |
| TN | channels elev, slope, water (+ valid); raw DEM | checkpoint `terrain`, `lowpass_m` None/0 |
| TNlp | same channels; model input Gaussian σ = 16 m (`lowpass_m` 16.0); rules / metrics on raw DEM | `terrain_sequence.py:41-50`, `backbones.py:36-37` |
| TNse | `dem_variant` se_f4: street beds erased (f = 4) | `backbones.py:33-35` |
| DEM (training, M12–M14 eval) | DTM: IGN MDT05 (5 m, REDNAP) Spain; USGS 3DEP 1 m (NAVD88) US; GEN / GEN2 same sources | `M12_DATA_PROTOCOL.md:36-37`, `m14_build_gen2.py:9` |
| DEM (M15 real) | Copernicus GLO-30 **DSM** (30 m, EGM2008) bilinear to 4 m; MDT05 only as Spain sensitivity | `m15_build_benchmark.py:14-17`, `sites.py:90-95` |
| DEM (M15 synthetic) | analytic fixtures: flat 50/80 m, plane, Gaussian ridge / valley, straight river (water kind 1 only) | `fixtures.py:36-55`, `sites.py:25-36` |
| DEM (all 3D demos) | procedural `terrain.synthesis.generate` (fBm, ridges, valleys, hydrology: river / lake / sea, water surface) | `m12_build_city3d.py:54`, `m14_demo3d.py:50-52` |
| Grid | 512 m tile, 4 m pixels, 256 m margin → 256 × 256; 48 × 48 rotated crops; start canvas 128 × 128 | `terrain_sequence.py:3-16,26` |
| Normalisation | elev = (z − z_node) / 6.809 m clipped ±3; slope = \|∇z\| / 0.1289 clipped [0, 4]; start canvas elev − tile median; shared by all backbones | `terrain_sequence.py:58-61,100,117`; `data/processed/m12/terrain/norm.json` |

## 3. Sampler settings

| Setting | M12 (eval + demos) | M13 (M*) | M14 (NA-*) | M15 |
|---|---|---|---|---|
| Proposal | 1 step per arm position, arm of ≤ 7 steps sampled first, decoder fed the **sampled** step (`generate.py:405-431`) | K = 8 iid candidates from one decoder call; decoder fed the **committed** step (`tc_sampler.py:163-186,555`) | same, K = 8 (`m14_sampler.py:327-332`) | same; style factor added per candidate, normalised per decision (`tc_sampler.py:270-285`) |
| Temperature | displacement T = 0.5 (`m12_evaluate.py:41`; demos `m12_build_city3d.py:172`, big `m12_big_landscape.py` T 0.5); exploratory sweep 0.3–1.0 | T = 0.5 (mixture scales only; stop / class / structure untempered, `tc_sampler.py:141-150`) | T = 0.5 | 0.5; secondary arms 0.8, 1.0 (eval plan) |
| Hard rules | R / HR: per-class **grade max** 0.126 / 0.104 / 0.164 / 0.218 (30 m sub-span), bridge span ≤ 1027 / 276 / 48.5 / 40 m, angle ≥ 42.6°, lake / sea ≤ 30 m, deck grade ≤ class max (`road_feasibility.py:129-150`, `configs/m12_feasibility.json`); none: no rules | no grade rule; water / ends-in-water, structure support, sea bridge banned, bridge span ≤ 1781 m (all-class max), tunnel ≤ 35.6 m, angle ≥ 38.9° (`tc_potentials.py:172-210`, `configs/m13_potentials/TN.json`) | as M13 | as M13; optional `grade_cap` = M12 grade max (`tc_sampler.py:247-249`, `style.py:73-79`) |
| Soft terrain | none (HR replans instead) | λ_ratio 0.5 × [log r_grade(class, 32 m slope bin, 30 m grade) + log r_align], \|log r\| ≤ log 20; λ_infra 1 (`tc_potentials.py:182-213`) | as M13 | as M13 (off in "morphology-only" arm) |
| Replanning / search | HR: 4 redraws, halve, contour turn, far bank; ≤ 8 evals (`terrain_generate.py:35,121-147`) | H = 0 (E+LA H = 1 not selected); SMC "G" = 8 particles | NA-fixed H = 2, rollout_m 4; NA-adaptive trigger deg1 (fires 0.43); NA-SMC 4 particles; M13-SMC **4** particles (`methods.py:61-69`) | as M14; style κ grid 4, organic 8, radial 8 nats per 45° (`configs/m15_style.json`) |
| Stop | Bernoulli(sigmoid(stop_logit)), independent of terrain | P(stop) = p_stop / Z, Z = p_stop + (1 − p_stop)·mean w (`tc_sampler.py:358-368`) | P(stop) = p_stop w_stop / Z, λ_dead 1.814, λ_island 0 (`m14_sampler.py:12-13`, `configs/m14_lambda.json`) | as M13 / M14; cul-de-sac stop_bias +0.25 |
| stop_bias | 0 | 0; density-matched A +0.25, M* −0.125 | 0; dm (GEN2) A +0.375, M* −0.1875, NA-myopic +0.375, NA-SMC −0.375 | 0 (cul-de-sac 0.25) |
| Budgets | 1500 rounds, ≤ 7 steps per pop, 1000 nodes, 40 components (`generate.py:45-46,292`; `sequence.py:58`) | same (`tc_sampler.py:464`) | same | same; TRACER 6 km (2 km exploratory), step 26 m, d_sep 78 m |
| Start | learned start head; R / HR reject starts in water (`terrain_generate.py:102-107`) | learned; hard methods reject water starts, A accepts them (`tc_sampler.py:453-460`) | learned | **syn eval: first start fixed at tile centre** (amendment 2; `tc_sampler.py:424-428`, `m15_run.py:69-70`); syn_dev tuning and real pilot: learned |
| Potentials file | — | per backbone (`TN`, `TNlp`, `TNse`, `N0`) | `TN` unless spec names one (`m14_experiments.py:139`) | always `TN` (`m15_run.py:77`) |
| Model input variant | low-pass applied for TNlp (`m12_evaluate.py:71-72`) | `model_inputs` (lp / se) | same | low-pass only; se ignored (`m15_run.py:94`) |

## 4. 3D demo pipelines

| Demo | Script / args | Sampler | Terrain, seed | Selection |
|---|---|---|---|---|
| M12 single tiles | `m12_showcase.sh:10-20` → `m12_build_city3d.py` | TN_s0 + **HR** (also HRV, R, none, P) | flat_city1, narrow_river2, wide_river3, deep_valley0, coastal_city1, steep_hillside1 (R), SD tile; seed 2041 | **densest of 4 samples** (`m12_build_city3d.py:171-173`) |
| M12 big cities | `m12_big_landscape.py` (`m12_showcase.sh:24`) | TN_s0 + HR; alpine: TNlp + HR with low-pass (`:65`) | rolling_hills4, high_hills4, alpine_valley1; 3 × 3 windows, 128 m overlap, given streets + frontier continuation; seed 2050 + k | one sample per window |
| M13 demos | `m13_demo_cities.sh:11-15` | `--sampler energy` (M*) vs standard, TN_s0, no crop variant (`m12_build_city3d.py:161`) | flat_city1, rolling_hills4, steep_hillside1, narrow_river2, mountain_pass0, coastal_city1; seed 2041 | first sample (`:162`) |
| M13 GEN demos | `m13_three_tiles.py` + `--gen-tile --graph` | stored M* / A runs | GEN tiles (MDT05 / 3DEP), seed 1 | stored run |
| M14 demos | `m14_demo3d.py:44,123-125` | P, A, M*, M*+repair, M13-SMC, NA-myopic, NA-fixed, NA-SMC; TN_s0; CPU (`:55`) | mountain_ridges0, deep_valley0, mountain_pass0 (+ flat_city1 at T 0.8 / 1.0); **seed 1** | first sample (`:70-73`) |

All demos: same downstream (BF buildings, slope filter 0.35, `--m13-fixes` in M13/M14). No (scenario, seed) pair is shared
between M12 and M14 demos.

## 5. Metrics: same name, different definition

| Name | M12 | M13 | M14 | M15 |
|---|---|---|---|---|
| `grade_p90` | length-weighted p90 of per-edge **mean (endpoint) grade** (`terrain_metrics.py:53`); stored as `m12_grade_p90` in M13 / M14 | `grade_p90_m13`: p90 of per-edge **max 30 m sliding** sub-span grade (`m13_metrics.py:141`, `tc_potentials.py:89-91`) | both kept | p90 over **consecutive 10 m** sub-spans (syn, MDT05) or **30 m** (GLO-30), weighted by sub-span length (`style_metrics.py:192-230`; `m15_run.py:124`) |
| `share_gt10` | length share of ground **edges** whose max 10 m sliding sub-span > 10 % (whole edge counted) (`terrain_metrics.py:54`, `heightfield.py:84-95`) | as M12 (inside S_T) | as M12 | length share of **sub-spans** > 10 %; new `share_gt15` = M15 O4 primary |
| S_T | mean W1/IQR over grade_p90, share_gt10, contour index, steep-density ratio, circuity vs real tiles **of the same set**; 52 dev tiles, 4 samples averaged per tile | same formula; reference DEV-B / GEN; per-tile mean over 4 seeds (`m13_analyze.py:58-81`) | same; 3 seeds; GEN2 reference | **not computed** |
| Empty graph in S_T | S_T keys NaN → tile silently dropped (`terrain_metrics.py:41-43`, `m12_evaluate.py:244`) | same | same | n/a (`empty` reported) |
| S10 | unconditional (regime B, 256 tiles, vs val) | `S10_cond`: conditional, vs the same tiles | `S10_cond` (report column labelled "S10") | not used |
| Connectivity | — | `giant_share`, `unreachable_share`: **open** window (`m13_metrics.py:74-112`) | `closed_giant`, `od_unreach(_land)`: window **closed** by a virtual outside node; empty = 1 (`m14_metrics.py:95-142`) | M14 function; rng `default_rng([s, 15])` after `od_circuity` (`m15_run.py:119-125`) vs M14 `1000 + s` after M13 metrics |
| Dead ends | S10 dead-end ratio (`graph_metrics`) | `dead_end_ratio_m13`: interior deg-1 / nodes, stored border flags | same | `deadend_per_km`, border recomputed geometrically |
| Coverage | — | — | `coverage_60`: buildable (dry, slope < 0.30) land within 60 m | also PRDC **coverage** (Naeem 2020, k = 5) in the real pilot: different quantity |
| Density | street km/km² / real | `density_ratio` | same | PRDC **density** in the real pilot: different quantity |
| Premature | street length < 50 % of the real tile | same (`m13_analyze.py:82-90`) | same | none on synthetic sites (no real network) |

## 6. Differences that could change terrain behaviour

| # | Difference | M12 / M13 | M14 / M15 | Evidence |
|---|---|---|---|---|
| 1 | Grade feasibility | M12 eval TN+R / TN+HR and **all M12 demos**: hard per-class grade max + replanning (contour turn) | never hard (soft only, λ_ratio 0.5); M15 `grade_cap` only in control arms. M13 M* already soft-only | `road_feasibility.py:142`, `terrain_generate.py:121-147`, `tc_potentials.py:15,179-187`, `methods.py:61` |
| 2 | Demo selection | M12: densest of 4 samples | M13 / M14: first sample | `m12_build_city3d.py:171-173` vs `:162`, `m14_demo3d.py:70-73` |
| 3 | Demo terrain and seed | M12 seed 2041 / 2050; gentler showcase scenarios; big cities windowed 3 × 3 with continuation | M14 seed 1; mountain_ridges, deep_valley, mountain_pass; single tile; CPU | `m12_showcase.sh:10-24`, `m14_demo3d.py:55,123-125`, `m11_big_city.py:91-93` |
| 4 | Arm sampling loop | M12: whole arm sampled before any rule, decoder fed sampled steps | M13+: per-step commit, decoder fed committed steps (DEV-B S_T: A0 legacy 0.429 vs A 0.420) | `generate.py:405-436`, `tc_sampler.py:555`, `outputs/m13/FINAL_REPORT.md:61-62` |
| 5 | Stop coupling | M12: stop independent of terrain; M13: low candidate weights raise P(stop) (shorter arms instead of turns) | M14 adds w_stop with λ_dead 1.81 (joins rewarded, dead-end stops penalised), competing with the terrain ratio (≤ 0.5·2·ln 20 ≈ 3 nats) | `tc_sampler.py:358-368`, `m14_sampler.py:12-13`, `configs/m14_lambda.json` |
| 6 | Style strength (style on) | — | κ 4–8 nats at 45° (local / minor streets, gain 1) vs terrain soft weight ≤ ≈ 3 nats | `style.py:124-131`, `configs/m15_style.json` |
| 7 | Bridge rules | M12: per-class span p99, angle ≥ 42.6°, lake / sea ≤ 30 m | M13+: one span limit 1781 m, angle ≥ 38.9°, sea banned, lakes allowed. M15 protocol §2 says "span ≤ class p99, angle ≥ real p5": **does not match the code** | `road_feasibility.py:129-139`, `tc_potentials.py:190-200`, `TN.json` `span_max`, `M15_PROTOCOL.md:43-44` |
| 8 | Elevation source | DTM 1–5 m with street imprint (training, DEV, GEN, GEN2) | M15 real: GLO-30 DSM 30 m (smoother, includes buildings); MDT05 networks 0.7–0.9 km longer | `sites.py:90-91`, `m15_build_benchmark.py:14-17`, `M15_RESULTS_AND_READINESS.md:107-108` |
| 9 | Synthetic terrain family | M12 regime C / all demos: procedural `synthesis` (TN alone already violated grade limits there, `M12_RESULTS.md:289-292`) | M15: analytic fixtures (planes, Gaussian ridge / valley, straight river); M* builds 0.5–2.8 km per tile | `fixtures.py:36-55`, `M15_AUDIT.md` §6.3 |
| 10 | Start rule | learned start head everywhere | M15 syn eval: first start at tile centre; style strengths tuned with the learned start | `tc_sampler.py:424-428`, `m15_run.py:69-70`, `configs/m15_plans/dev_tune.json` |
| 11 | Geography | Spain / US (MDT05, 3DEP) | M15 real adds Korea, Peru (no training terrain there) | `M15_PROTOCOL.md:132-138` |
| 12 | Potentials calibration | TN_s0 standard sampler, T 0.5, 160 train tiles × 2 (v2) | reused unchanged for TN_s1 / s2 and for T 0.8 / 1.0 arms | `TN.json` `source`, `m14_experiments.py:139`, `m15_run.py:77` |
| 13 | Model-input variant in demos | TNlp big city: low-pass applied | `m12_build_city3d.py` sampler / legacy paths pass no `crop_tiles` (a TNlp single-tile demo would see raw DEM); `m15_run.py` ignores TNse | `m12_build_city3d.py:161,171`, `m15_run.py:94` |
| 14 | Evaluator | S_T family (M12 definitions), 4 samples or seeds | M15: no S_T; new grade definitions (§5); 10 vs 30 m spans | §5 above |
| 15 | P baseline density | per-tile random draw from training pool | median of the pool (M15, M14 demo); M12 demo fixed 120 | `m12_evaluate.py:69`, `m14_experiments.py:148`, `m15_run.py:87`, `m14_demo3d.py:67`, `m12_build_city3d.py:166` |
| 16 | Device / seeds | MPS; seeds 1000 + s, 1–4, 2041 / 2050 | CPU; seeds 1–3, 1, 101–110 | §1 |
| 17 | Name reuse | "SMC" (M13 G) = 8 particles | "M13-SMC" = 4 particles | `outputs/m13/plans/tn.json`, `methods.py:69` |

## 7. Not verified here

* Bit-identity of the M12 legacy loop after the M13 resolve/commit refactor (commit message only; DEV-B A0 runs exist).
* Device of the M12 regime A/B/C runs; whether MPS vs CPU changes samples systematically or only by noise.
* That `data/processed/m15/real/terrain_mdt05` equals the M13 GEN terrain arrays, and that M15 water rasters equal
  the M12/M13 ones (same functions are imported, files not compared).
* That every M14 confirmatory run used CPU (protocol states it; the audit replay of GEN2 M* / NA-fixed on CPU
  reproduced 95/95 graphs, which supports it for those runs).

## 8. Matched replays this table enables

1. **Rule effect on demos:** same scenario and seed (e.g. deep_valley0, mountain_pass0, s2041 and s1) with HR, M*,
   NA-fixed, first sample and densest-of-4: separates items 1–3 from any code change.
2. **Code path:** M12 TN `none` (legacy) vs M13 A (standard) on DEV-B, same seeds: item 4.
3. **Elevation source:** Girona / Manresa windows with MDT05 (M13 GEN arrays) vs GLO-30, M* and NA-fixed, seeds
   101–103, scored with both `grade_p90_m13` and M15 `grade_exposure`: items 8 and 14.
4. **Network potentials:** M* vs NA-myopic vs NA-fixed on the M12 procedural scenarios with the M13 metrics: item 5.
