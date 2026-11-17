# Milestone 14 implementation

## Code map (all new; no M12 or M13 file was changed except as listed at the end)

| File | Role |
|---|---|
| `src/urbanmorph/graphs/m14_sampler.py` | `NetSampler(TCSampler)`. It adds telescoping network potentials (dead ends, islands), the network weight on STOP, network-aware roll-outs, an adaptive trigger, VoC logging and the KL trust region. `NetConfig` extends the M13 `SamplerConfig`. The presets are `na_myopic`, `na_fixed`, `na_adaptive` and `na_smc`. |
| `src/urbanmorph/graphs/m14_metrics.py` | Closed-window connectivity: closed giant, islands, OD reachability split into land vs water, detour, coverage of buildable land, street metres per connectivity. |
| `scripts/m14_audit.py` | Phase-1 audit on the stored M13 runs. |
| `scripts/m14_toy.py` | Exact-enumeration toy (claims C1–C6). |
| `scripts/m14_build_gen2.py` | GEN2 cities, built with the unchanged M13 GEN pipeline and a new city list. |
| `scripts/m14_calibrate.py` | λ by generalized iterative scaling on training tiles. |
| `scripts/m14_experiments.py` | Resumable runner. It writes one JSON per run and rewrites it atomically after every inference seed, and supports `--worker k/n` and `--device cpu`. |
| `scripts/m14_plans.py` | Every plan is generated from the frozen configuration. |
| `scripts/m14_triggers.py` | Trigger selection and validation (protocol §4). |
| `scripts/m14_analyze.py` | Statistics: block sign-flip test, block bootstrap, set-level contrasts, noise floor, density adjustment, trigger tables. |
| `scripts/m14_report.py`, `scripts/m14_figures.py` | Report and figures, built from stored runs only. |
| `tests/test_m14.py` | Unit tests (listed below). |

## How the sampler is attached to the M13 loop

The M13 decision loop (`TCSampler.run`) calls `candidates` → `lookahead` (if H > 0) → `select` for every active expansion. `NetSampler` overrides those three methods only, so every other behaviour is unchanged:
* the batching across tiles;
* the start head;
* the geometric resolution;
* the state-consistent GRU update;
* the SMC resampling;
* the twist bookkeeping.

* **`candidates`** calls the M13 version (geometry and terrain potentials on the committed edge). It then adds each continuation's network log-weight (`lw_net`) and returns a `CandList` that also carries the STOP option's network log-weight (`stop_lw`). On the MAX_T-th arm of an expansion, each continuation also carries the weight of closing u, because the loop ends the expansion without a STOP decision. This keeps Proposition 1 exact.
* **`lookahead`** groups the candidates by decision and computes the context features. If the trigger fires, it rolls out each feasible open-ended candidate on a copy of the state, H steps of the new road with M = 4 candidates per step. During the roll-out the node being expanded is kept open (`_m14_extra_open`), and the network weights and the STOP weight apply. It then sets `log_psi` to the sum of the log local normaliser estimates.
* **`select`** applies the selection with the STOP weight: P(stop) = p_s w_s / Z. It also applies the optional KL trust region (β by bisection) and, when logging, records VoC = TV(selection with the twist, selection without it).

## Audit items from the M14 brief

* **The GRU sees only committed actions.** The M13 loop feeds back `rotate(q_committed − u)`. M13 tests this (`test_decoder_is_fed_the_committed_edge`), and M14 does not touch that path.
* **Snapping and intersections.** Candidates are scored on `resolve_step` / `resolve_structure` output, which is exactly what `commit` inserts (M13 test `test_scored_geometry_is_inserted_geometry`). The network potentials read the same resolution: `kind`, `v`, `edge`.
* **Road termination** can happen five ways, and each is counted:
  * a STOP decision (network-weighted);
  * MAX_T arms (closure weight added; `maxT_last_arm`);
  * all candidates infeasible (`forced_stop`);
  * the node budget (1,000) or component budget (40);
  * `max_rounds` (1,500).
* **Equal opportunity across methods.** All methods share the start head, the node, component and round budgets, K, MAX_T and the temperature. Planning changes which candidate is chosen and whether u stops. It never adds candidates or extends budgets.

## Tests (`pytest tests/test_m14.py tests/test_tc_sampler.py`: 16 pass)

| Test | Checks |
|---|---|
| `test_network_weights_telescope_to_final_counts` (5 seeds) | Σ of applied network log-weights = −λ_D D(x) − λ_I I(x) of the final graph, to machine precision |
| `test_lambda_zero_reproduces_m13_energy` | With λ = 0 and no roll-outs, NetSampler gives the identical graph to the M13 `energy` sampler (same seed) |
| `test_trust_region_bound` | KL(q_β ‖ q_0) ≤ ε on 50 random decisions; hard zeros stay zero |
| `test_island_and_dead_end_bookkeeping` | Closed-island and dead-end events on a hand-built state, including the border case |
| `test_closed_window_metrics` | Island detection; OD pairs split into land and water; the empty-network convention |

## Compute

* **Hardware:** Apple M5 Pro, 15 cores, 48 GB RAM. Inference runs on CPU workers with 2 threads each. CPU timing is close to MPS: on DEV-A, `na_myopic` takes 0.39 vs 0.56 s per tile on MPS, and `na_fixed` 4.5 vs 3.7 s.
* **Concurrency:** up to 8 CPU workers run at once (7 for the GEN re‑run), and no training job runs at the same time.
* **Cost accounting:** per-run seconds include contention between parallel workers. Cost comparisons in the results use runs made in the same plan under the same load, and the comparison is approximate. Forward passes and roll-out steps are recorded as machine-independent cost measures.

## Changes outside new files

* `.gitignore`: tracks `outputs/m14/` except per-run JSON, logs and VoC logs.
* No M12 or M13 source file was modified on this branch. `scripts/m12_build_city3d.py` gained the `--gen-tile` and `--graph` options in the M13 exploratory commit (d4d73c3) before the branch was created.
