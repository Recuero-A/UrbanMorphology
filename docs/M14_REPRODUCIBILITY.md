# Milestone 14 reproducibility

## Environment

* **Python:** the project `.venv` (Python 3.12), with PyTorch on CPU. MPS gives the same samplers but different floating‑point streams.
* **Hardware:** Apple M5 Pro, 15 cores, 48 GB. Runs used CPU workers with 2 threads each, up to 8 in parallel.
* **Checkpoints:** the frozen M12 checkpoints in `checkpoints/m12/{TN,TNlp,TNse}_s{0,1,2}/best.pt` (not in git; regenerable with the M12 training scripts). M14 trains nothing.
* **Data:**
  * M12 tiles (`data/processed/m12/`);
  * M13 GEN (`data/processed/m13/gen/`) and the street‑erased DEMs;
  * M14 GEN2 (`data/processed/m14/gen2/`, built by `scripts/m14_build_gen2.py` from cached Geofabrik extracts, the GHSL 2023A rasters, IGN MDT05 (CC BY 4.0) and USGS 3DEP (public domain)).

  None of these are in git.
* **Never read:** the sealed test splits.

## One command per stage

```bash
P=.venv/bin/python
$P scripts/m14_audit.py                                   # phase-1 audit (stored M13 runs)
$P scripts/m14_toy.py                                     # exact toy (C1-C6)
$P scripts/m14_build_gen2.py --stage all                  # GEN2 data (network access; ~15 min)
$P scripts/m14_calibrate.py                               # lambda -> configs/m14_lambda.json (~1.5 h)
$P scripts/m14_plans.py voc && $P scripts/m14_experiments.py --plan configs/m14_plans/voc.json --device cpu
$P scripts/m14_triggers.py select                         # -> configs/m14_frozen.json
$P scripts/m14_plans.py pilot && $P scripts/m14_experiments.py --plan configs/m14_plans/pilot.json --device cpu
$P scripts/m14_triggers.py validate
for n in confirm h5 ablate shift; do $P scripts/m14_plans.py $n; done
# the queue used (priority order): configs/m14_plans/queue.json; run N workers in parallel:
for k in 0 1 2 3 4 5; do $P scripts/m14_experiments.py --plan configs/m14_plans/queue.json --device cpu --worker $k/6 & done
$P scripts/m14_report.py                                  # -> outputs/m14/REPORT.md, report.json
$P scripts/m14_figures.py                                 # -> outputs/m14/figures/
$P -m pytest tests/test_m14.py tests/test_tc_sampler.py   # 16 tests
```

## Determinism

* **Run identity.** A run is identified by the SHA‑1 of its spec (`m14_experiments.run_id`): tiles, backbone, preset, overrides, seeds and flags. Re‑running a plan skips the seeds that are already stored.
* **Seeding.** Each inference seed seeds the torch generator (candidates, selections, roll‑outs) and the numpy generator (start positions). Metric randomness (OD pair sampling) uses `default_rng(1000 + seed)`.
* **Repeatability.** The same seed on the same device reproduces a graph exactly. This is tested for λ = 0 against M13.
* **Random trigger.** `random:p` uses its own generator, seeded with 7 per sampler instance.
* **Thread‑count dependence.** CPU results can differ at the floating‑point level across thread counts. Every stored run records its own seconds and peak RSS, but not its thread count: all M14 runs used 2 or 3 threads.

## What is stored

| Path | Content | In git |
|---|---|---|
| `outputs/m14/runs/*.json` | Spec, per‑tile per‑seed metrics (M12, M13, M14), sampler statistics, graphs, diversity | no (≈ 1 MB each; regenerable) |
| `outputs/m14/voc/*.json` | Per‑decision VoC records (trigger selection) | no |
| `outputs/m14/real/*.json` | Metrics of the real graphs of each tile set | yes |
| `configs/m14_lambda.json`, `configs/m14_frozen.json`, `configs/m14_plans/*.json` | Frozen configuration and plans | yes |
| `outputs/m14/{audit,toy,triggers,calibration,literature,figures}/`, `REPORT.md`, `report.json` | Results | yes |

## Pre‑registration trail

All commits are on branch `milestone-14`, in this order:

1. **6eccee5**: protocol, formulation and code, before any confirmatory run.
2. **628e89c**: λ calibration and amendment 1.
3. **7499e5b**: trigger frozen (`configs/m14_frozen.json`).
4. **6995562**: pilot and amendment 2.
5. **6d13b6e**: confirmatory results, report, figures, density control and the results documents.

Confirmatory runs started after amendment 2. Git history was not rewritten.

## Incidents during the confirmatory runs

* **Missing GEN2 street‑erased DEMs.** The TNse runs of H5 need street‑erased DEMs, which the GEN2 build did not produce. Queue worker 4 crashed on the first of them (`FileNotFoundError`; log kept as `outputs/m14/logs/queue_4_crashed_no_se_dem.log`). The DEMs were then built with `scripts/m14_street_erase_gen2.py` (the M13 f4 procedure, unchanged; 95 tiles written to `data/processed/m13/terrain_se/f4/`) and the worker was restarted. No run had stored results for these specs before the crash, so nothing was overwritten.
* **GEN contamination.** An import‑time patch in `scripts/m14_build_gen2.py` made workers that had loaded GEN2 read GEN2 tiles when asked for GEN (details: `M14_FAILURE_ANALYSIS.md` §4.1). The patch now runs only in the build script's `main()`. The 28 affected GEN runs and `outputs/m14/real/gen.json` were moved to `outputs/m14/invalid_gen_contaminated/` (kept, with a README) and GEN was re‑run from the same specs, split by longest‑processing‑time into `configs/m14_plans/gen_rerun_{0..6}.json`. `real/gen.json` is rebuilt by the runner on first use.
* **Run lookup and seed subsets.** `m14_analyze.find` now prefers the run with the most seeds when a configuration is stored twice (§4.4), and runs used for 2‑seed analyses (ablations, H2, shifts, H5, density matching, dose figure) are cut to seeds 1–2 (§4.6).
* **Density‑matched plan.** The first `dm` plan left out NA‑SMC because it used the fast method set. The missing runs were added as `configs/m14_plans/dm_smc.json` (3 runs) and the generator was fixed. The stop biases come from DEV‑A only (`scripts/m14_density_calibrate.py`, then `--collect` to write `configs/m14_density.json`) and were not re‑fitted on GEN2.

```bash
$P scripts/m14_street_erase_gen2.py                      # GEN2 street-erased DEMs (TNse)
for l in A "M*" NA-myopic NA-SMC; do $P scripts/m14_density_calibrate.py "$l" & done; wait; $P scripts/m14_density_calibrate.py --collect  # DEV-A stop biases
$P scripts/m14_plans.py dm && $P scripts/m14_experiments.py --plan configs/m14_plans/dm.json --device cpu
$P scripts/m14_experiments.py --plan configs/m14_plans/dm_smc.json --device cpu
for k in 0 1 2 3 4 5 6; do $P scripts/m14_experiments.py --plan configs/m14_plans/gen_rerun_$k.json --device cpu --threads 2 & done; wait   # GEN re-run (~70 min)
```
