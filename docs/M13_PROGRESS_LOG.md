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
