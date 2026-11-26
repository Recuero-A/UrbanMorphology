# Milestone 18 audit: what we already know, and how solid it is

Phase 1a of `docs/M18_PLAN.md`. Sources: the milestone documents, the stored runs, the sampler and metric code. The
reproduction in §5 re-reads stored graphs only (`scripts/m18_audit_m14.py` → `outputs/m18/audit_m14/`). No new
generation was run, and no held-out or sealed data was read.

## 1. Where each claim comes from

| Claim | Milestone | Kind of evidence | Data | Caveats on record |
|---|---|---|---|---|
| Terrain conditioning improves terrain fit (S_T −69 % vs terrain-blind); the low-pass control keeps about three quarters | M12 | **Pre-registered** | `val` (12 tiles), `val_t` (40 San Diego tiles) | 4 dated protocol changes, 2 after results were seen (`docs/M12_EVALUATION_PROTOCOL.md` §8); H3lp and H4 fail |
| Hard rules remove unbridged water crossings; sampling-time realism gains vanish under leakage and density controls | M13 | **Pre-registered**, negative | DEV-A / DEV-B, GEN | — |
| Network potentials cut same-land OD unreachability about 70 % (H1); diversity kept (H4); survives leakage controls (H5); adaptive look-ahead no better than random (H3 fails) | M14 | **Pre-registered, confirmatory** | GEN2 (4 fresh cities), with DEV-B and GEN secondary | 3 analysis bugs fixed after confirmation, none changed a verdict (`docs/M14_FAILURE_ANALYSIS.md` §4); reproduced in §5 |
| Learned samplers adapt orientation style per tile (φ ρ 0.68–0.77), partly through leakage | M14 | **Exploratory, post hoc** | GEN2, GEN, DEV-B | Not pre-registered |
| Style control works on flat terrain but is not retained on terrain | M15 | **Pilot** (pre-registered outcomes) | `syn_dev` / `syn_eval`, 120 `real_dev` windows | Amendment 2 (fixed start) made after partial results |
| Style proposals and topology rules | M16 | Development, confirmatory (synthetic), recurring named sites, exploratory: each result labelled | `syn_conf`, the terrain pack | Real-terrain confirmation not run |
| Close-junction excess is mostly density; online reuse (F1) helps styled generation | M17 | **Confirmatory holdout** for F1 | 9 fresh metros | F4 (degree prior) is exploratory: the holdout was reused |
| Realism vs terrain-adaptation trade-off; TAS | Site benchmarks | **Exploratory**; definitions committed before scoring | Machu Picchu, Petra | TAS is our own composite; 2 sites, 3 seeds |

## 2. Methods that exist

Every learned method uses the M12 TerrainGraphTurtle checkpoints (`checkpoints/m12/<backbone>/best.pt`): TN, TNlp and
TNse, seeds 0–2 each, plus N0 (terrain-blind) and single-channel ablations (TNelev, TNslope, TNwater, TNelevslope).

The registry is `src/urbanmorph/experiments/methods.py`.

* **Network-potential weights.** `configs/m14_lambda.json` gives λ_D = 1.8143 and λ_I = 0.
* **Frozen NA settings.** `configs/m14_frozen.json` (the NA-adaptive trigger, the NA-random firing rate).
* **Style settings.** `configs/m15_style.json` and `configs/m16_frozen.json`.
* **Junction fixes.** `configs/m17_degree_prior.json`, with reuse radius 15 m.

Cost below is measured on the GEN2 confirmatory runs (CPU, wall seconds per 512 m tile under parallel load).

| Method | Milestone | Inputs | Preset / settings | GEN2 run IDs (TN_s0–s2) | s / tile |
|---|---|---|---|---|---|
| A | M13 | DEM, slope, water | `standard`, K = 1, no rules | 1013dd2a, 41dadf76, 844f7aca | 0.37 |
| M\* | M13 | same | `energy`, K = 8, λ ratio 0.5, hard rules, terrain potentials | 95650dd8, ba2c47a6, def2c357 | 0.64 |
| M\* + repair | M14 | same | M\* then A\*-style connection repair | 7254d498, 762c72a3, 91bc9469 | 0.64 + repair |
| M13-SMC | M13 | same | `smc`, 4 particles, no network potentials | 633059f4, 7080cfcd, e408f757 | 2.19 |
| NA-myopic | M14 | same | `na_myopic`, λ as above | 2a83b152, 88fa0bcf, f4696be1 | 0.86 |
| NA-adaptive | M14 | same | `na_adaptive`, frozen trigger | 0c609376, 2335d9e7, ef4b695c | 5.35 |
| NA-random | M14 | same | `na_adaptive`, random trigger at the same rate | 36bf1202, bdfdcbd9, d2621909 | 4.91 |
| NA-fixed | M14 | same | `na_fixed`, 2-step roll-outs everywhere | 4630cb1c, dc9f9632, de14bfa4 | 9.75 |
| NA-SMC | M14 | same | `na_smc`, 4 particles | 68d58335, 72f838d9, ec61e91b | 2.74 |
| P | M12 | DEM, water | procedural, density drawn from training tiles | e46383af | 0.02 |
| M12-HR | M12 / M16 replay | DEM, slope, water | the M12 generation loop with hard rules and replanning (`legacy`, policy HR) | not run on GEN2 | — |
| + style (grid, organic, radial, cul-de-sac) | M15–M16 | + a style field | reranking (M15) and style proposals (M16) on M\* or NA-fixed | site benchmarks, KAIST | — |
| F1 + F4 | M17 | same | junction reuse 15 m + degree prior λ 1 | M17 holdout, KAIST, site benchmarks | ≈ base |
| TRACER | M15 | DEM, a style field | Chen-inspired cross-field tracer (`graphs/field_tracer.py`) | M15, site benchmarks | — |

TN, TNlp and TNse stay the terrain-input controls for every new comparison.

## 3. Discrepancies between reports and implementation

1. **What "OD unreachability" samples.** The documents call it the share of random *point* pairs on the same dry
   land with no network path. The code (`graphs/m14_metrics.tile_metrics_m14`) does something different:
   * it samples pairs of the network's **own nodes**: 30 sources × up to 7 targets, 150–500 m apart, about 200 pairs;
   * it counts a pair as unreachable when it is on the same dry-land region but in different closed components;
   * the denominator is all pairs drawn;
   * an empty network scores 1.

   Consequences:
   * the measure weights dense parts of a network more;
   * it cannot see land the network does not reach (coverage is reported separately);
   * one estimate is noisy, with a per-tile SD of about 0.03 (§5).

   The documents should say "node pairs". The M14 effect is unaffected (§5).
2. **Different random streams for real and generated graphs.**
   * Generated graphs were scored with one stream per inference seed, `default_rng(1000 + seed)`, shared across tiles
     and with the M13 metrics.
   * Real graphs were scored with a fresh `default_rng(0)` per tile.

   This adds noise to real-vs-generated comparisons, but no bias.
3. **Two declared held-out metros are no longer unseen.**
   * Sacramento and Santa Barbara are declared M15 held-out metros, but they were M14's GEN2 primary set.
   * Seoul is a declared M15 held-out metro, but one Seoul window is an M16 recurring site.

   The M15 held-out pool therefore needs replacements for these three before any confirmatory real benchmark.
4. **Two aggregations of H1.** M14's estimator gives Δ = −0.026. A paired tile-level estimate averaged per block (§5)
   gives −0.021. Both intervals exclude 0; the aggregations differ.
5. **close15 representation.** M17's primary results use rebuilt graphs (8 m snap, as for real data). Raw generated
   graphs read 0.01–0.04 higher. The trade-off table in §5 uses raw graphs and labels them so.

## 4. Sites and sets that have already influenced decisions

None of these can serve as fresh confirmation data in M18.

| Set | Places | How it was used |
|---|---|---|
| Training | Barcelona, Madrid, Valencia, New York, Chicago, Los Angeles, Tokyo, Osaka, Nagoya | Training; 48 training tiles for λ calibration |
| M12 val / val_t | Training cities' validation blocks; San Diego | M12 evaluation |
| DEV-A, DEV-B | Training cities' validation blocks | M13–M14 tuning and secondary confirmation |
| GEN | Girona, Manresa, San Francisco | M13 generalisation; M14 secondary |
| GEN2 | Alcoi, Tarragona, Santa Barbara, Sacramento | M14 primary confirmation |
| real_dev | Girona, Manresa, Busan, Daejeon, Arequipa, Cusco | M15 pilot |
| M16 style references | 36 windows in 31 cities (`configs/m16_style_refs/manifest.csv`, e.g. Barcelona, Kyoto, Paris, Palmanova, Canberra) | Style thresholds and blinded sorting |
| Terrain pack | KAIST (Daejeon), Seoul Haebangchon, Madrid Lavapiés, San Lorenzo de El Escorial | M16–M17 recurring sites and demos |
| M17 holdout | Gwangju, Ulsan, Jeonju, Cheongju, Piura, Chiclayo, Huancayo, Castellón, Figueres | F1 confirmation, then reused for F4 |
| Site benchmarks | Machu Picchu, Petra | Exploratory comparison |
| Synthetic | `syn_dev`, `syn_eval`, `syn_conf` | M15–M16 |
| **Not yet read** | M12 test split (sealed); M15 held-out Lleida, Alicante, Daegu, Lima, Trujillo | Need the owner's authorisation |

## 5. Reproduction of the strongest result, and what it costs

**Measured.** All from the stored GEN2 graphs of the M14 confirmatory runs: 3 TN backbones × 3 inference seeds × 95
tiles.

* **The primary outcome reproduces.**
  * Re-scoring 1,710 stored graphs (M\* and NA-fixed) with M14's exact random stream reproduces the stored
    unreachability on 1,707 graphs. The other 3 differ by at most 0.0099, about 2 of 203 pairs; the cause is not
    identified.
  * The means agree to 4 decimals: M\* 0.0363, NA-fixed 0.0109.
  * The H1 contrast recomputes to Δ = −0.0259 [−0.0363, −0.0154], p < 10⁻⁴, over 34 blocks, as in the M14 report.
* **It survives a precise re-estimate.** Averaging 10 independent pair samples per tile gives M\* 0.0389 and NA-fixed
  0.0116.
* **Each 200-pair estimate is noisy.** The SD across samples has a median of 0.028 and a p90 of 0.046 on tiles with
  any unreachable pair. Tile-level values should not be read alone.
* **The gain does not come from clutter for NA-SMC.** The table below gives paired differences from M\* (matched
  backbone, seed and tile), averaged per 3 km block, with block-bootstrap 95 % intervals. Real GEN2 values for
  reference: 13.9 km/km², 77 junctions/km², 27 interior blocks/km², a 37 % share of blocks under 2,000 m², and
  close15 0.145 (built graphs).

| Contrast | OD unreach. | km/km² | junctions/km² | close15 (raw) | links < 15 m | blocks/km² | blocks < 2,000 m² |
|---|---|---|---|---|---|---|---|
| NA-SMC − M\* | −0.019 [−0.029, −0.007] | −0.39 [−0.88, +0.10] | −4.7 [−9.5, +0.1] | +0.002 [−0.008, +0.012] | −0.005 [−0.010, +0.001] | +2.0 [−0.9, +5.0] | −0.052 [−0.101, +0.009] |
| NA-fixed − M\* | −0.021 [−0.028, −0.014] | +0.98 [+0.39, +1.57] | +4.1 [−1.3, +9.9] | +0.007 [−0.005, +0.019] | +0.002 [−0.002, +0.007] | +5.2 [+2.0, +8.7] | −0.045 [−0.096, +0.011] |
| NA-myopic − M\* | −0.020 [−0.027, −0.013] | +2.70 [+2.04, +3.32] | +23.2 [+17.4, +29.4] | +0.036 [+0.022, +0.051] | +0.010 [+0.005, +0.015] | +15.3 [+11.6, +19.2] | −0.017 [−0.040, +0.006] |
| M\* + repair − M\* | −0.025 [−0.034, −0.017] | +0.34 [+0.26, +0.42] | +4.9 [+4.0, +5.9] | +0.001 [−0.001, +0.005] | −0.001 [−0.002, −0.001] | +0.3 [+0.2, +0.5] | +0.007 [+0.003, +0.011] |

**Interpretation.**

* **NA-SMC.** Its connectivity gain comes with no measurable rise in density, junctions, close junctions, short links
  or block fragmentation. Its block-size distribution moves toward the real one, though that interval includes 0.
* **NA-fixed.** It adds a little street length and about 5 more blocks per km².
* **NA-myopic.** It pays for its gain with density, junction count and close junctions.
* **Every generator.** All make smaller and more numerous blocks than real GEN2: median 4.9–6.9k m² against 8.3k m²
  (`outputs/m18/audit_m14/tradeoffs.md`). This shared deficit is unrelated to the network potentials.
* **Repair.** It gives the largest unreachability reduction at almost no cost in these statistics. Its known cost is
  elsewhere: straight connectors that ignore terrain (M14: detour 1.59, unbridged links). These measures do not show
  it, which is a measurement gap for phase 2.

**Narrowed claim (proposal).** "Network-aware inference (NA-SMC) removes the generator's fragmentation deficit
without measurable cost in density, junction spacing, link length or block structure, on fresh cities. A post-hoc
repair does the same for connectivity alone, but ignores terrain."
