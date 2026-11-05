# Milestone 13 progress log

Dated entries; mistakes and course changes included.

## 2026-11-05 / 06: setup, literature, sampler

**Start**
* Branch `milestone-13` from `milestone-12` at `1d7d471`; 231 tests passed.
* Environment: M5 Pro, 48 GB, 567 GB free.

**Literature review** (`docs/M13_LITERATURE_REVIEW.md`)
* Read in full: NTG, Birsak, Anagnostidis, Galin 2010, Emilien 2012, Lew 2023, Zhao 2024, Lu 2022, Loula 2025.
* Abstract-only: Gu 2024 (MDPI returned 403 to our client), Fang 2022 (repository unreachable), Zhou 2024 (CAVW,
  paywalled).
* No first claims. The candidate contribution is the application of properly weighted constrained SMC with Monte
  Carlo look-ahead to road graphs, plus the evaluation protocol.

**Generator refactor**
* `GenState.apply_step` split into a pure `resolve_step` and `commit`.
* Bit-identical outputs for 6 policies (N0, TN, TN+R, TN+HR, TN+HRV, TNlp+HR) on 6 tiles, verified by hashes.

**Sampler** (`graphs/tc_sampler.py`)
* K-candidate proposal, hard constraints and calibrated density-ratio potentials, look-ahead twists, SMC, beam,
  schedulers; state-consistent commitment.
* Course change: beam search compared log p(stop) with log *densities* of continuous displacements; it would almost
  never stop. The stop decision is now sampled.

**Potentials** (`configs/m13_potentials/`)
* Calibrated on 473 training tiles with a DEM: 68,113 real ground edges; proposals from 160 tiles × 2.
* Structures are sparse: 107 / 40 / 17 / 4 bridge runs by class, only 12 crossing angles. Infrastructure potentials
  are therefore weak and exploratory.

**Street-erased DEM** (`scripts/m13_street_erase.py`)
* Corridor factor chosen on training tiles by the pre-stated rule (on/off-street |Laplacian| ratio closest to 1):
  **f = 4.0**, ratio 1.04 (original 0.93), mask 40 % of the tile, slope p90 shift −0.001.
* The training-tile original ratio (0.93) is weaker than M12's 0.53 on development tiles. The on-street mask
  definitions differ (M13: the carriageway buffer).

**Runner** (`scripts/m13_run_all.py`)
* Started 2026-11-05 15:16 UTC with `caffeinate`.
* Queue: TN-se × 3 seeds, then seed 2 of TN, TN-lp and N0.

**Pilot** (8 tiles, TN_s0; bug and runtime check only)
* Every preset runs.
* Beam produced 4.7 km/km² and top-k 11.5, against about 19 for the sampling methods. Beam and likelihood-ranked
  selection starve the network in the pilot. The confirmatory runs will show whether this holds.

## 2026-11-06 01:00 KST: course change before selection

**Calibration mismatch (my implementation error)**
* The first potential calibration used the M12 *legacy* loop for the proposal distribution. `M13_METHOD.md` §2.2
  defines the proposal as the *state-consistent* sampler, and the two differ in density (pilot: 18.2 vs 21.5 km/km²).
* Found while writing the TN-se calibration job, before any selection metric was examined: only timing lines of the
  first selection run had been seen.
* Fix:
  * the selection was stopped and its DEV-A runs deleted;
  * the potentials were recalibrated with the state-consistent sampler (`version: v2`);
  * run ids carry `pot_version: v2`, so stale runs cannot be reused;
  * the selection was restarted.

**Runner restarted with the extended plan** (15 jobs: calibration, selection, confirmatory tiers, GEN, ablations,
X5). TNse_s0 was interrupted by SIGTERM and resumed from step 6417, the first real-world resume.

**Generalisation data**
* Girona 35 eligible → 30 drawn; Manresa 30 → 30; built 60 / 60.
* San Francisco from the NorCal extract (655 MB, ODbL) is building; DEM fetch and preparation follow.

**Global (tile-free) sampler** (`graphs/tc_global.py`): works on a 1536 m domain. The legacy hashes are still
identical after the domain-size generalisation of `sequence.py`, `generate.py` and `terrain_sequence.py`.

## 2026-11-06 01:25 KST: selection bug (λ not applied)

* The first full selection finished, and its table showed identical results for λ_ratio 0.5, 1 and 2.
* Cause: the sampler did not copy its λ weights into the potentials object.
* Fixed with a regression test. The whole selection is being rerun as `pot_version v3` (protocol amendment 1); the
  invalid file is archived in `outputs/m13/superseded/`.
* What that (λ = 1) run nonetheless showed, kept for the record and not used for any decision: every
  potential-weighted method had a lower S_T than standard sampling (0.570), and beam had density / real 0.61 and a
  premature share of 0.32: it starves.

## 2026-11-06 02:10 KST: selection v3 (frozen)

**Frozen choices**
* M* = **E, one-step energy-guided sampling, K = 8, λ_ratio = 0.5**. On DEV-A: S_T 0.338 vs 0.570 for A, density /
  real 0.93, premature 0.12.
* E* = M*. H* = 1, the only eligible look-ahead depth: H = 2 and H = 4 failed the premature criterion, H = 8 the
  density criterion.

**Not eligible, recorded as observed**
* SMC + look-ahead (N = 4, H = 1) had the lowest S_T (0.286), but density / real 0.83 is outside ±15 %.
* Beam: density 0.62, premature 0.32.
* SMC without look-ahead: density 0.78–0.79.

The particle methods trade density for terrain score on DEV-A.

**Other progress**
* GEN built: 100 tiles with a DEM (San Francisco 40, Girona 30, Manresa 30), 24 blocks. DEM sidecars were rebuilt
  from the fetch log after a missing-sidecar error.
* TN-se potentials calibrated.
* The confirmatory TN tier started early for the trained backbones. `m13_confirm` now skips untrained backbones and
  writes its done marker only when all runs exist.
* Runner restarted with the 18-job plan; TNse_s2 resumed at step 6549.

## 2026-11-05 18:40 UTC: early TN tier complete

* TN_s2 finished training during the early tier, so all three TN training seeds have DEV-B runs (inference seeds 1–4).
  The runner's `confirm_tn` job re-checks them and writes the done marker.
* Descriptive sanity check only (`m13_analyze.py --summaries`; no configuration or criterion changed). Confirmatory
  statistics come from `m13_report.py`.
  * M* (energy K8 λ0.5) vs A (standard), S_T per training seed: 0.327 / 0.338 / 0.398 vs 0.437 / 0.384 / 0.438.
  * M* + repair raises giant share (0.90 → 0.99) but also S_T (0.33 → 0.63–0.77): the router's connectors fit
    terrain poorly. This is reported as a trade-off, not hidden.
  * Lower S_T at reduced density (top-k, SMC, beam: density ratio 0.56–0.94) is what the density-matched tier tests.
* Runner: training TNlp_s2; about 23 h estimated remaining, within the 36 h budget.

## 2026-11-05 19:50 UTC: leakage tier complete (descriptive)

* All seed-2 backbones are trained, and `confirm_tn` and `confirm_leak` are done. Next: `confirm_dm`.
* Mean S_T on DEV-B, M* vs A, per training seed:

  | Backbone | M* | A |
  |---|---|---|
  | TN-lp | 0.596 / 0.626 / 0.456 | 0.612 / 0.624 / 0.648 |
  | TN-se | 0.579 / 0.592 / 0.617 | 0.688 / 0.733 / 0.707 |

* Every leakage-controlled backbone sits around 0.6–0.7, against about 0.4 for TN. Much of TN's terrain fit therefore
  likely comes from street beds in the DEM.
* The sampler's gain is consistent on TN-se but small or absent on TN-lp in two of three seeds. Confirmatory tests
  follow in the report.

## 2026-11-05 20:35 UTC: confirm_gen failure and fix

* **Failure:** `confirm_gen` failed with exit 1 at the first TN-se run on GEN tiles. The GEN cities had no street-erased
  DEMs: `m13_street_erase.py` only read the M12 index.
* **Fix:**
  * Added `--gen` to `m13_street_erase.py`. It erases the 100 GEN tiles with the **frozen** factor 4.0, chosen on
    training tiles; there is no re-selection, and the script asserts factor 4.0.
  * Added a runner job `erase_gen` (`configs/m13_full.toml`); `confirm_gen` now depends on it.
  * Check on one tile: 29 % of pixels changed, mean |dz| 0.79 m, same arrays as the original.
* **Recovery:**
  * TN and TN-lp GEN runs completed before the failure and are kept; they do not depend on erasure.
  * The tier was restarted by hand (log `outputs/m13/logs/confirm_gen_retry.log`) in parallel with `confirm_ablate`.
    Both are inference only; no training job runs.
  * When the runner reaches `report`, it is blocked by the failed `confirm_gen`. A runner `resume` afterwards finds
    the done marker and runs `report`.
