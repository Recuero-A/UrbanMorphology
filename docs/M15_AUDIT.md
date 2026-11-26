# Milestone 15 audit: the existing generator, before any M15 change

## 1. Checkout and starting point

* **Repository:** `/Users/antoniorecuero/Desktop/UrbanismProject`, populated (no dependency missing).
* **Starting commit:** `a25a202`, the tip of `milestone-14`. M15 work is on the branch `milestone-15`.
* **Environment:** Python 3.12 virtual environment (`.venv`), PyTorch on CPU (Apple M5 Pro, 15 cores).
* **Tests at the start:** 247 passing (`python -m pytest -q`). After the M15 additions: 262 passing.

## 2. Where things are

| Item | Location |
|---|---|
| Generator (network) | `src/urbanmorph/models/terrain_graph_turtle.py`; checkpoints `checkpoints/m12/{TN,TNlp,TNse}_s{0,1,2}/best.pt` (not in git) |
| Training configuration | Inside each checkpoint. TN_s0: terrain channels elevation, slope, water; positional noise `noise_m` 2.5 m; warm-up 300 steps; lr 1e-3 → 5e-5; 60 epochs, patience 10 |
| Model input conventions | `src/urbanmorph/graphs/terrain_sequence.py`. Elevation enters as (z − z at the node) / scale, clipped to ±3, so absolute altitude does not matter |
| Samplers and presets | `src/urbanmorph/graphs/tc_sampler.py` (`PRESETS`), `src/urbanmorph/graphs/m14_sampler.py` (`NET_PRESETS`); named configurations in `src/urbanmorph/experiments/methods.py` |
| Hard rules and potentials | `src/urbanmorph/graphs/tc_potentials.py`; calibrations in `configs/m13_potentials/`, `configs/m12_feasibility.json`, `configs/m14_lambda.json` |
| Prior runs | `outputs/m13/runs/`, `outputs/m14/runs/` (per-run JSON with specification, per-tile metrics and graphs; git-ignored, regenerable) |
| Reports | `outputs/m13/FINAL_REPORT.md`, `outputs/m14/REPORT.md` |
| Validation and analysis scripts | `scripts/m13_analyze.py`, `scripts/m14_analyze.py`, `scripts/m14_report.py`; M15 replay `scripts/m15_replay_baseline.py` |

## 3. Entry points and commands

| Purpose | Command |
|---|---|
| Generate with any M13 / M14 configuration | `python scripts/m14_experiments.py --plan configs/m14_plans/<plan>.json --device cpu --threads 2` (one run = backbone × preset × tile set × seeds) |
| M14 report from stored runs | `python scripts/m14_report.py`; `python scripts/m14_figures.py` |
| M15 | `python scripts/m15_run.py --plan configs/m15_plans/<plan>.json`; plans from `python scripts/m15_plans.py eval|real` |
| Programmatic | `urbanmorph.graphs.tc_sampler.sample_tc(...)` and `urbanmorph.graphs.m14_sampler.sample_net(...)` with a `SamplerConfig` / `NetConfig` from `preset(...)` / `net_preset(...)` |

## 4. Exact meaning of the existing metrics (M12–M14)

| Metric | Definition | Defined in |
|---|---|---|
| **S10_cond** | Street-network realism: mean over 10 statistics of the Wasserstein-1 distance between generated and real per-tile values, each divided by the real interquartile range. The 10 statistics: street length per km², intersections per km², dead-end ratio, components, loops per km², mean link length, 4-way share of junctions, junction-angle deviation, orientation entropy, orientation order φ. Lower is better; real-vs-real median on GEN2 = 0.293 | `docs/M13_EVALUATION_PROTOCOL.md` P4; inputs `road_graph.graph_metrics` (`s10_*` keys) |
| **S_T** | Terrain-response gap, the same construction over grade p90, share of streets above 10 % grade, contour index, steep-density ratio and circuity | `src/urbanmorph/graphs/terrain_metrics.py` |
| **OD unreachability (same land)** | Share of random point pairs, 150–500 m apart on the same dry-land region, with no network path. An empty network counts as 1 | `src/urbanmorph/graphs/m14_metrics.py` |
| **Islands** | Closed components of ≥ 40 m of street touching no border node | `src/urbanmorph/graphs/m14_metrics.py` |
| **Coverage 60 m** | Buildable land within 60 m of a street | `src/urbanmorph/graphs/m14_metrics.py` |
| **Jaccard diversity** | Mean pairwise Jaccard distance of 8 m occupancy grids of samples on the same tile | `src/urbanmorph/graphs/m13_metrics.py` |
| **Energy distance** | On the M13 joint feature keys | `src/urbanmorph/graphs/m13_metrics.py` |

## 5. Baseline reproduction and archive

* **Replay.** `scripts/m15_replay_baseline.py` re-runs the archived M14 baselines: M\* and NA-fixed, GEN2, TN_s0,
  inference seed 1, CPU, 2 threads, all 95 tiles batched as in M14. Each graph is compared with the stored one (node
  coordinates to 0.01 m, edges, road classes, structures).

  | Code | M\* | NA-fixed | Record |
  |---|---|---|---|
  | Before any M15 change | 95/95 identical | 95/95 identical | `outputs/m15/audit/replay.json` |
  | After the style-guidance code, style off | 95/95 identical | 95/95 identical | `outputs/m15/audit/replay_after_style_code.json` |

  The second row is the check that style "off" leaves every existing sampler unchanged.
* **Archive.** `outputs/m15/audit/baseline_archive.json` holds, for M\* and NA-fixed on GEN2 / TN_s0:
  * the run specifications;
  * a SHA-1 per seed of the stored graphs;
  * the per-seed mean OD unreachability, islands and street density.

  The full per-run JSON stays in `outputs/m14/runs/` (git-ignored, regenerable).

## 6. Findings of the audit that shaped M15

1. **Hidden cross-script dependencies.** Shared helpers (backbone loading, graph I/O, atomic writes, the method list)
   lived in milestone scripts and were imported across scripts; the M14 GEN contamination came from this pattern.
   They now live in `src/urbanmorph/experiments/`, with the old names re-exported. The method list is byte-identical
   (checked), and stored graphs round-trip exactly.
2. **Termination coupling.** The K-candidate samplers decide stopping with Z = p_stop + (1 − p_stop) · mean(w). Any
   extra factor ≤ 1 lowers Z and so raises the stop probability. A naive style factor would therefore make styled
   networks shorter and "succeed" by building fewer roads. M15 normalises the style factor within each decision, and
   keeps the un-normalised version as an ablation.
3. **Synthetic terrain produces small networks.** On smooth analytic terrain, M\* builds about 0.5–2.8 km per tile,
   less than on real tiles. The 2.5 m training noise is one hypothesis; it is not tested here. Achieved length is
   therefore reported with every result.
4. **Structures.** The learned samplers can build bridges under the M13 eligibility rule, so the tracer baseline uses
   the same rule (`field_tracer.bridge`) rather than stopping at water.
