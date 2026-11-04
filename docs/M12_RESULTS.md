# Milestone 12 results: terrain-aware Neural Turtle Graphics (development data only)

**Status.** Every number below comes from DEVELOPMENT data:
* `val`: the 12 M2 validation tiles with a DEM;
* `val_t`: 40 San Diego tiles, a city never used for training.

The held-out `test` / `test_t` splits are untouched (`test_t` is not even built). The one-time test evaluation
(`scripts/m12_final_test.sh`) has **not** been run; it needs explicit authorisation.

Criteria, metrics and thresholds are those of `docs/M12_EVALUATION_PROTOCOL.md`, pre-registered before training. Its
§8 lists the four dated changes, two of them made after results were seen (marked where they matter).

Sources:
* `outputs/m12/evaluations/stats.json`;
* per-tile rows in `outputs/m12/evaluations/{A,B,C,H5}/`;
* figures in `outputs/m12/figures/`.

## 1. Summary

1. **Terrain inputs change the generated streets strongly, but part of the gain is leakage.**
   * TN lowers the terrain score S_T by 69% relative to the terrain-blind N0, for both seeds (H1 passes).
   * However, urban bare-earth DEMs still contain the street beds, so TN can partly *read* existing streets.
   * The leakage control **TN-lp** (terrain input low-passed at σ = 16 m) removes about 75% of TN's likelihood gain.
     It still improves S_T by 50–53% over N0 for both seeds: **H1lp passes**. Terrain adaptation survives the
     control, at about three quarters of the uncontrolled effect (S_T 0.66 / 0.69 vs TN 0.44 / 0.43).
   * **H3lp fails.** TN-lp+R beats R on both seeds (−21% / −33%), but seed 0's interval includes 0 (upper +0.045).
     The terrain-inputs-beyond-rules effect is not established once leakage is controlled.
2. **Rules alone (R) avoid violations by not building.**
   * R: 60% of tiles end with less than half the real street length; dead-end ratio 0.58 (real 0.16).
   * TN+R: best S_T (0.205); it still leaves 12–14% of tiles premature.
3. **Pre-registered heuristic replanning (H4) fails its criterion.**
   * Under leakage control, TN-lp+R leaves 33–39% of tiles premature, against 14% for TN+R. Part of TN+R's
     completeness came from reading the street beds.
   * HR cuts the termination deficit relative to TN+R (0.12 vs 0.20; 0.10 vs 0.17; CI below 0).
   * But it worsens S_T by 0.095 / 0.089, above the 0.05 allowance, mostly through circuity.
4. **The structure head learned bridge decisions where labels exist**: AUC 0.967 for both seeds (H5); TN-lp 0.949 / 0.943.
   * Left free, TN over-bridges: 3–4 bridges per tile vs 1.6 real; 42–60 per tile on synthetic rivers.
   * TN-lp does not over-bridge synthetic rivers (1–6 per landscape). We have no tested explanation for the
     difference.
   * Viaducts are **never** model-placed: there are 0–2 real dry-valley spans to learn from. Every viaduct in this
     milestone is rule-placed (exploratory HRV), and HRV over-builds them.
5. **Realism is not lost** (H2, H2lp): S10 TN 0.312 / 0.319 and TN-lp 0.305 / 0.314 vs N0 0.328 / 0.333
   (unconditional).
   * Rule-filtered generation does lose realism: TN+R S10 0.51 / 0.57.
6. **3D:** 11 cities (one of them 3 × 3 tiles) on real and procedural terrain:
   * every building watertight, 0 floating;
   * Godot collision rays 63–64 of 64, walker on the floor in every scene;
   * bridges and viaducts as decks with piers.

No novelty claim is made (literature review §7.4).

## 2. Data and models (what was actually trained)

**Data**
* Training: 685 tiles (565 M2 / M3b + 120 steep Los Angeles tiles), graphs rebuilt as 3D-aware (bridges, tunnels
  and layers are not junctions).
* Development: 59 tiles (19 `val` + 40 `val_t`). Regime A uses the 52 with a DEM.
* DEMs: IGN MDT05 (Spain) and USGS 3DEP 1 m (US).
* Japan: no DEM (GSI licence unresolved), so it trains on water only.

**Training recipe**
* All runs share the recipe of `scripts/m12_train.py`: 60 epochs, 300-step warmup, MPS, 2 jobs at a time.
* Checkpoints every 5 min and per epoch.
* Model selection on validation NLL.

| Run | Terrain input | Seeds | Best val NLL / tile (s0; s1) | Best epoch | Decoder units alive | Minutes |
|---|---|---|---|---|---|---|
| N0 | none | 2 | 3.43; 6.45 | 58; 60 | 0.59; 0.54 | 36.7; 36.8 |
| TN | elev, slope, water | 2 | −118.0; −116.2 | 55; 54 | 0.62; 0.61 | 49.4; 48.9 |
| TN-elev | elev | 1 | −101.2 | 55 | 0.63 | 44.9 |
| TN-slope | slope | 1 | −109.7 | 55 | 0.63 | 43.4 |
| TN-water | water | 1 | +2.55 | 58 | 0.59 | 40.5 |
| TN-elev+slope | elev, slope | 1 | −113.5 | 55 | 0.61 | 41.4 |
| TN-lp | elev, slope, water, σ = 16 m low-pass | 2 | −28.4; −30.8 | 55; 58 | 0.59; 0.64 | 48.6; 48.7 |

**The NLL column is itself evidence about leakage.**
* Every run with elevation or slope gains about 105–120 nats per tile over N0.
* Water alone gains about 1.
* A development tile has about 150 labelled steps (8,756 on 59 tiles), so the gain is about 0.8 nats per step.
  That is consistent with the model locating streets from their imprint in the DEM (§4).
* Low-passing the input at 16 m (TN-lp) leaves a gain of only about 32–37 nats per tile: three quarters of TN's
  likelihood advantage depended on sub-16 m detail.
* N0's NLL differs by 3 nats between seeds, so differences of a few nats are noise.

**Not trained**
* **TN+LR (learned replanning).** We found no defensible objective: training on rejected proposals would need
  counterfactual labels the data does not contain.
* MDT02 and FABDEM were not used.
* San Francisco was not added: Los Angeles + San Diego already met the coverage targets of the Phase 2 audit.

## 3. Pre-registered hypotheses

Each hypothesis must hold for both seeds.

| | Test | Seed 0 | Seed 1 | Verdict |
|---|---|---|---|---|
| H1 | S_T: TN vs N0 (≥ 20% better, CI < 0) | 0.442 vs 1.402 (−68.5%; CI [−1.45, −0.63]) | 0.430 vs 1.380 (−68.9%; CI [−1.43, −0.63]) | pass, **confounded** (§4) |
| H2 | S10: TN ≤ N0 + 0.10 (regime B) | 0.312 vs 0.328 | 0.319 vs 0.333 | pass |
| H3 | S_T: TN+R vs R (≥ 10% better, CI < 0) | 0.205 vs 0.411 (−50%; CI [−0.40, −0.10]) | 0.205 vs 0.426 (−52%; CI [−0.43, −0.13]) | pass, **confounded** (§4) |
| H4 | TN+HR premature deficit < TN+R (CI < 0) and ΔS_T ≤ 0.05 | deficit 0.124 vs 0.196 (CI [−0.113, −0.036]); ΔS_T **+0.095** | 0.103 vs 0.169 (CI [−0.110, −0.023]); ΔS_T **+0.089** | **fail** |
| H5 | structure head bridge AUC ≥ 0.8 | 0.967 (N0 0.883) | 0.967 (N0 0.880) | pass |
| H1lp | S_T: TN-lp vs N0 (≥ 20% better, CI < 0) | 0.658 vs 1.402 (−53.1%; CI [−1.12, −0.49]) | 0.691 vs 1.380 (−49.9%; CI [−1.07, −0.44]) | **pass** |
| H2lp | S10: TN-lp ≤ N0 + 0.10 | 0.305 vs 0.328 | 0.314 vs 0.333 | **pass** |
| H3lp | S_T: TN-lp+R vs R (≥ 10% better, CI < 0) | 0.323 vs 0.411 (−21.3%; CI [−0.27, **+0.045**]) | 0.286 vs 0.426 (−32.8%; CI [−0.29, −0.02]) | **fail** (seed 0) |

H4 was not re-tested with TN-lp, because the protocol did not pre-register an H4lp. TN-lp+HR's numbers are in §5,
descriptively.

**About H5**
* 8,756 labelled development steps, 214 of them bridges.
* Recall at 0.5: TN 0.42 / 0.36; N0 0.16 / 0.19.
* N0's AUC of 0.88 shows how much the street context alone predicts. Bridges sit on major roads, in straight runs;
  terrain adds the rest.

## 4. Leakage: the street imprint in the DEM, and the TN-lp control

**Discovery (after the regime A results)**
* The example figure showed TN reproducing real street curves on held-out terrain.
* On real street pixels, the DEM's curvature (|Laplacian|) is 0.53× its off-street value (median over 52 tiles;
  `outputs/m12/audits/dem_street_imprint.json`).
* A bare-earth DEM of a city is graded, so street beds are visible in it.

**Consequences**
* TN is partly conditioned on the answer.
* H1 and H3 are therefore marked confounded. This was protocol amendment 2, made after the results and dated.
* Regime C (procedural terrain) is leakage-free by construction.

**Control.** TN-lp has the same architecture and recipe, but its terrain input is Gaussian low-passed at σ = 16 m
(4 px), which removes street-width features. Rules and metrics still use the unfiltered DEM.

**Result**

| | TN | TN-lp | N0 |
|---|---|---|---|
| Val NLL / tile (s0; s1) | −118.0; −116.2 | −28.4; −30.8 | 3.4; 6.4 |
| S_T regime A (s0; s1) | 0.442; 0.430 | 0.658; 0.691 | 1.402; 1.380 |
| S_T flat / moderate / steep (s0) | 0.36 / 0.48 / 0.40 | 0.66 / 1.10 / 0.49 | 1.17 / 1.61 / 1.31 |
| Structure-head AUC (s0; s1) | 0.967; 0.967 | 0.949; 0.943 | 0.883; 0.880 |
| Bridges per tile, regime A (s0) | 4.42 | 2.40 | 1.85 (real 1.56) |
| S10 regime B (s0; s1) | 0.312; 0.319 | 0.305; 0.314 | 0.328; 0.333 |

**Interpretation**
1. **The street imprint was real and large.** Most of TN's likelihood gain came from detail finer than 16 m.
   * TN's advantage over TN-lp is largest on flat tiles (0.36 vs 0.66), where the street beds are almost the only
     relief.
   * The example figure shows it directly: TN's streets trace the street beds in the hillshade, TN-lp's do not.
2. **A genuine terrain effect remains.**
   * TN-lp improves on N0 on every landform class, by 53% / 50% overall.
   * The largest per-metric gains are grade_p90 (0.22 vs N0 1.40) and share_gt10 (1.36 vs 3.06): TN-lp avoids
     steep streets.
3. **TN-lp is not a clean bound.**
   * Wide roads (more than about 30 m) may still leave a smoothed imprint, which would bias TN-lp UP.
   * The filter also removes real terrain detail under 16 m, which biases it DOWN.
   * The true terrain-only effect is therefore not bracketed by TN-lp and TN.
4. **The ablations are not leakage-controlled** (§8). Slope alone does best of all (S_T 0.344); the slope channel
   shows street cuts most sharply.

## 5. Regime A: conditional generation on real development terrain (52 tiles)

Four samples per tile per seed, T = 0.5. Lower S_T is better.

| Model (seed 0) | S_T | S10 cond. | Unbridged water m/tile | Bridges/tile | Street km/km² | Dead-end ratio | Premature share |
|---|---|---|---|---|---|---|---|
| real | — | — | 1.1 | 1.56 | 13.6 | 0.16 | — |
| N0 | 1.402 | 0.508 | 105.4 | 1.85 | 15.9 | 0.22 | 0.02 |
| R | 0.411 | 0.635 | 0.0 | 0.15 | 6.7 | 0.58 | **0.60** |
| P (procedural) | 0.770 | 1.163 | 1.9 | 0.15 | 15.0 | 0.16 | 0.10 |
| TN | 0.442 | 0.461 | 7.6 | **4.42** | 15.5 | 0.21 | 0.04 |
| TN+R | **0.205** | **0.328** | 0.0 | 0.27 | 12.1 | 0.35 | 0.14 |
| TN+HR | 0.299 | 0.388 | 0.06 | 0.43 | 13.5 | 0.28 | 0.08 |
| TN+HRV (exploratory) | 0.360 | 0.424 | 0.08 | 1.61 | 14.8 | 0.27 | 0.04 |
| TN-elev | 0.499 | 0.479 | 80.4 | 3.06 | 15.4 | 0.25 | 0.00 |
| TN-slope | 0.344 | 0.376 | 36.4 | 3.96 | — | — | 0.02 |
| TN-water | 1.417 | 0.504 | 13.8 | 1.59 | — | — | 0.04 |
| TN-elev+slope | 0.410 | 0.426 | 55.8 | 4.29 | — | — | 0.02 |
| **TN-lp** (leakage control) | 0.658 | 0.414 | 3.3 | 2.40 | — | — | 0.04 |
| **TN-lp+R** | 0.323 | 0.469 | 0.12 | 0.18 | — | — | **0.33** |
| **TN-lp+HR** | 0.359 | 0.397 | 0.18 | 0.36 | — | — | 0.12 |

"—": not tabulated for these runs (per-tile rows in `evaluations/A/`).

**TN-lp, seed 1:** 0.691 / 0.286 / 0.395 S_T for TN-lp, TN-lp+R and TN-lp+HR; premature 0.06 / 0.39 / 0.12.

Seed 1 is within 0.04 S_T of seed 0 for every model (`stats.json`).

**Per-metric W1/IQR (seed 0)**

| Model | grade_p90 | share_gt10 | contour_index | steep_density | circuity |
|---|---|---|---|---|---|
| N0 | 1.40 | 3.06 | 0.74 | 0.89 | 0.93 |
| R | 0.29 | 0.78 | 0.43 | 0.27 | 0.28 |
| P | 0.36 | 1.88 | 0.43 | 0.77 | 0.42 |
| TN | 0.13 | 0.62 | 0.09 | 0.38 | 0.98 |
| TN+R | 0.11 | 0.26 | 0.15 | 0.09 | 0.41 |
| TN+HR | 0.11 | 0.29 | 0.10 | 0.23 | 0.77 |
| TN-lp | 0.22 | 1.36 | 0.31 | 0.34 | 1.06 |
| TN-lp+R | 0.22 | 0.48 | 0.16 | 0.33 | 0.42 |
| TN-lp+HR | 0.16 | 0.63 | 0.19 | 0.22 | 0.59 |

**Reading the tables**
* **N0 ignores terrain.** Twice the real share of steep streets, and 105 m of unbridged water per tile.
* **R looks good on S_T because it builds little.**
  * Its mean termination deficit is 0.54: on average it builds 54% less street length than the real tile.
  * The pre-registered circuity term only uses CONNECTED node pairs (`terrain_metrics.circuity`), so a fragmented,
    starved network is not penalised by it.
  * The density and dead-end columns show the starvation.
  * Per the task specification, R is **not** a success.
* **TN+R is the best compromise measured.** But its premature share (0.14) and dead-end ratio (0.35) are still
  well above real.
* **TN+HR trades realism for completeness.** Contour turns and halved steps fill the tile. They also lengthen
  routes: circuity 0.77 vs 0.41 for TN+R, which is most of the H4 failure.
* **P (least-cost paths)** is connected and complete, but its streets do not look like streets (S10 1.16). It
  places many steep streets because its cost is a soft penalty.

**By landform class (S_T, seed 0)**

| Class | n | N0 | R | P | TN | TN+R | TN+HR |
|---|---|---|---|---|---|---|---|
| flat | 17 | 1.17 | 0.84 | 0.59 | 0.36 | 0.32 | 0.37 |
| moderate | 11 | 1.61 | 0.71 | 1.13 | 0.48 | 0.52 | 0.77 |
| steep | 24 | 1.31 | 0.48 | 0.69 | 0.40 | 0.21 | 0.25 |

TN-lp / TN-lp+R / TN-lp+HR (seed 0): flat 0.66 / 0.40 / 0.28; moderate 1.10 / 0.48 / 0.73; steep 0.49 / 0.52 / 0.32.

* TN beats N0 even on flat tiles. That is consistent with leakage: on flat land there is little terrain signal, but
  the street imprint remains. TN-lp loses most on flat tiles (0.66 vs TN 0.36), which supports this reading.
* Moderate tiles (n = 11) vary most between seeds: TN+HR scores 0.77 on seed 0 and 0.43 on seed 1.

## 6. Regime B: unconditional realism (S10, 256 tiles vs the 19 val tiles)

| Model | seed 0 | seed 1 |
|---|---|---|
| N0 | 0.328 | 0.333 |
| TN | 0.312 | 0.319 |
| TN+R | 0.512 | 0.568 |
| P | 1.069 | — |
| TN-lp | 0.305 | 0.314 |

For comparison, M10's best model (N at T = 0.5) scored S10 0.30–0.33 on the planar graphs. The 3D-aware graphs
shift the validation reference by 0.055 (MILESTONE_12 §9), so the two are close but not identical in definition.

## 7. Regime C: procedural stress tests (10 scenarios × 5 seeds; exploratory)

Synthetic terrain, no street imprint. Mean per landscape, seed 0 models.

**Street length (km/km²)**

| Scenario | N0 | R | P | TN | TN+R | TN+HR | TN+HRV |
|---|---|---|---|---|---|---|---|
| flat_city | 10.5 | 14.4 | 14.3 | 13.0 | 7.2 | 9.0 | 6.9 |
| rolling_hills | 12.0 | 8.0 | 14.4 | 12.1 | 6.8 | 8.8 | 13.7 |
| steep_hillside | 15.9 | **1.8** | 13.3 | 7.2 | **1.4** | 9.4 | 7.6 |
| mountain_pass | 13.8 | 4.3 | 6.3 | 9.0 | 4.2 | 12.4 | 8.9 |
| mountain_ridges | 15.6 | 4.4 | 7.3 | 14.4 | 13.8 | 12.8 | 11.6 |
| deep_valley | 19.0 | 2.0 | 3.9 | 6.2 | 5.5 | 6.5 | 7.7 |
| narrow_river | 13.9 | 5.9 | 11.0 | 9.6 | 7.3 | 7.5 | 7.1 |
| wide_river | 16.5 | 8.2 | 9.2 | 11.4 | 8.5 | 10.7 | 7.8 |
| coastal_city | 12.0 | 6.4 | 8.3 | 9.7 | 5.9 | 8.9 | 8.8 |
| large_lake | 13.5 | 0.0 | 0.0 | 1.7 | 0.0 | 0.0 | 0.0 |

**Infeasible ground edges per landscape** (grade above the class limit, or ending in or obstructed by water;
`road_feasibility.evaluate`)

| Scenario | N0 | P | TN | TN+HR |
|---|---|---|---|---|
| steep_hillside | 64.0 | 23.2 | 41.2 | 2.0 |
| mountain_pass | 47.6 | 9.4 | 29.2 | 0.6 |
| deep_valley | 84.2 | 4.4 | 9.0 | 0.0 |
| large_lake | 136.4 | 0.0 | 0.2 | 0.0 |
| others | 0–41 | 0–1.6 | 0–7 | 0 |

R and TN+R have 0 everywhere, by construction.

**Residual HR violations.** HR's remaining violations (≤ 2 per landscape) are probably edges that the generator
created AFTER the check, by snapping a step end to a nearby node or splitting an existing edge. This is not
instrumented.

**Bridges per landscape**

| Scenario | N0 | P | TN | TN+R | TN+HR | TN+HRV |
|---|---|---|---|---|---|---|
| narrow_river | 2.0 | 2.2 | **42.0** | 1.6 | 1.2 | 1.4 |
| wide_river | 1.8 | 2.2 | **59.6** | 2.0 | 3.0 | 1.2 |
| coastal_city | 0.6 | 0.0 | **45.0** | 0.6 | 1.6 | 2.0 |
| rolling_hills | 0.2 | 0.4 | 19.0 | 0.4 | 1.6 | 1.6 |
| large_lake | 1.4 | 0.0 | 16.8 | 0.0 | 0.0 | 0.0 |

**Findings**
* **TN alone over-bridges.** The structure head says "bridge" for almost every step near synthetic water; synthetic
  rivers look unlike anything in training.
* **TN alone violates grade limits on steep scenarios.** The real training data holds few streets above 15% for the
  model to imitate.
* **R and TN+R starve the network** on steep_hillside (1.4–1.8 km/km²) and deep_valley.
* **TN+HR keeps 6.5–12.4 km/km² with almost no violations, but never builds a viaduct.**
* **TN+HRV** builds viaducts by rule (184 rule-placed viaduct steps in regime A, 8 in regime C) and over-builds on
  deep_valley0 (16 decks in one city).
* **TN-lp under the same stress tests** (seed 0):
  * 8.5–18.2 km/km², except large_lake (0);
  * grade violations like TN's (steep_hillside 50.6, mountain_pass 32.0);
  * far fewer bridges (1.2–6.2 on the river scenarios vs 42–60);
  * 110 model-placed bridges over all 50 landscapes, vs TN's 970.
* **TN-lp+HR** keeps 5.9–17.9 km/km² with at most 1 violating edge per landscape and 0–1.2 bridges.
  * It leaves less street than TN+HR on wide_river (5.9 vs 10.7) and flat_city (6.4 vs 9.0).
* **large_lake is degenerate.** The lake fills most of the tile, so every feasible model builds nearly nothing.
  The scenario is reported, not interpreted.

## 8. Ablations (one seed each; descriptive)

| Input | Val NLL / tile | S_T | grade_p90 | share_gt10 | contour | steep density | circuity | Unbridged water m/tile | Bridges/tile |
|---|---|---|---|---|---|---|---|---|---|
| none (N0) | 3.4 | 1.402 | 1.40 | 3.06 | 0.74 | 0.89 | 0.93 | 105.4 | 1.85 |
| water | 2.6 | 1.417 | 1.45 | 3.36 | 0.64 | 1.04 | 0.60 | 13.8 | 1.59 |
| elev | −101.2 | 0.499 | 0.16 | 0.89 | 0.12 | 0.40 | 0.92 | 80.4 | 3.06 |
| slope | −109.7 | **0.344** | 0.09 | 0.45 | 0.09 | 0.27 | 0.82 | 36.4 | 3.96 |
| elev + slope | −113.5 | 0.410 | 0.11 | 0.56 | 0.10 | 0.37 | 0.91 | 55.8 | 4.29 |
| elev + slope + water (TN) | −118.0 | 0.442 | 0.13 | 0.62 | 0.09 | 0.38 | 0.98 | 7.6 | 4.42 |

**Findings**
* **All terrain adaptation comes from elevation and slope.** Water alone leaves S_T at N0's level (1.42 vs 1.40).
* **Water is what removes unbridged crossings.** 106 → 14 m per tile with water alone; 7.6 m with all three
  inputs. Without water, elev / slope models still cross 36–80 m per tile.
* **Slope alone has the best S_T.** Adding elevation or water does not help on these 52 tiles: 0.344 → 0.410 →
  0.442.
  * The 0.07–0.10 differences exceed the overall seed-to-seed spread of the 2-seed models (≤ 0.04).
  * But each ablation has one seed and no interval, and per class the seed spread reaches 0.34 (§5).
  * The ranking is therefore suggestive only.
* **These runs use the full-resolution input, so they inherit the leakage of §4.** In particular, slope is the
  channel in which street cuts are sharpest. Leakage-controlled ablations were not run.

## 9. 3D cities (`outputs/m12/cities/`, plans in `outputs/m12/figures/cities.png`)

| City | Mode | Buildings (watertight / floating) | Bridges / viaducts | Removed by buildability (water / slope) | Godot rays, walker |
|---|---|---|---|---|---|
| flat_city1 | TN+HR | 345 (345 / 0) | 1 / 0 | 0 / 0 | 64/64, on floor |
| USA_SAN_e958_n7125 (real San Diego terrain) | TN+HR | 220 (220 / 0) | 0 / 0 | 0 / 36 | 64/64, on floor |
| narrow_river2 | TN+HR | 54 (54 / 0) | 3 / 0 | 65 / 0 | 64/64, on floor |
| wide_river3 | TN+HR | 161 (161 / 0) | 1 / 0 | 108 / 0 | 64/64, on floor |
| wide_river3 | TN, no rules | 198 (198 / 0) | **54** model-placed / 0 | 110 / 0 | 64/64, on floor |
| wide_river3 | P | 260 (260 / 0) | 0 / 0 | 137 / 0 | 64/64, on floor |
| deep_valley0 | TN+HR | 227 (227 / 0) | 0 / 0 | 0 / 83 | 64/64, on floor |
| deep_valley0 | TN+HRV | 159 (159 / 0) | 0 / **16** rule-placed | 0 / 78 | 63/64, on floor |
| coastal_city1 | TN+HR | 163 (163 / 0) | 2 / 0 | 109 / 0 | 64/64, on floor |
| steep_hillside1 | R (failure example) | 165 (165 / 0) | 0 / 0 | 0 / 0 | 64/64, on floor |
| rolling_hills4, 3 × 3 tiles | TN+HR | 1,664 (1,664 / 0) | 32 decks (9 model, 18 rule) | — | 64/64, on floor |

**Failure examples, on purpose**
* wide_river3 with no rules: over-bridging.
* deep_valley0 with HRV: viaduct over-building.
* steep_hillside1 with R: a nearly empty street network.

**Known 3D limitations**
* Buildings on slopes stand on the lowest ground under their footprint (no terracing or retaining walls).
* Bridge decks are flat slabs on piers.
* There are no tunnels in 3D: tunnel steps are recorded but not modelled.

## 10. Limitations and threats to validity

1. **Leakage (§4).** Real-terrain claims rest on the TN-lp control.
   * σ = 16 m was chosen before TN-lp was trained (not tuned), with one value only.
   * The ablations and H4 were not repeated under the control.
2. **Coverage.** Valleys (67 train / 7 eval tiles), dry valley spans (2 / 1) and water tiles (few) support only
   descriptive claims. Viaducts are entirely rule-based.
3. **One held-out city.** `val_t` is San Diego only. Generalisation to other countries' terrain is untested beyond
   the 12 `val` tiles.
4. **Spanish DEM artefacts.** Terracing and micro-relief at 5 m. Grades use a 30 m window and landform slope a 32 m
   block average (dated amendments) to suppress them.
5. **Metric definitions.**
   * Circuity ignores disconnected pairs (§5).
   * S_T compares grade distributions, not the safety of individual streets.
6. **Seeds.** Two per primary model and one per ablation. Seed 1 differs by up to 0.34 S_T on the 11 moderate
   tiles.
7. **Synthetic terrain** is not evidence of real-world generalisation.
   * `large_lake` is degenerate.
   * The valley-axis bug was fixed and all of regime C rerun (amendment 3).
8. **Zero-length steps.** The feasibility grade is 0/0 for a step with coincident ends, and such steps pass. They
   add no street length, and the same code evaluated every model.
9. **Two changes were made after seeing results**, both dated in protocol §8: the leakage control (amendment 2)
   and HRV (amendment 4, exploratory). No criterion was changed.

## 11. Reproduction (development results)

```
.venv/bin/python scripts/m12_train_all.py                       # all runs, resumable (--resume latest)
.venv/bin/python scripts/m12_evaluate.py --regime A             # + B, C, H5, stats
.venv/bin/python scripts/m12_figures.py
.venv/bin/python scripts/m12_city_plans.py
bash scripts/m12_showcase.sh                                     # 3D cities, Blender, Godot check
```

The one-time held-out test is not run:

```
M12_TEST_AUTHORISED=yes bash scripts/m12_final_test.sh
```

It refuses to run without the variable set.
