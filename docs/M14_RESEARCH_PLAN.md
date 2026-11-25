# Milestone 14 research plan: generalisable, network‑aware urban generation

Branch `milestone-14`, created from `milestone-13` at d4d73c3. M12 and M13 code, results, checkpoints, protocols and manuscripts are untouched: M14 adds new files only, plus the GEN2 data directory.

## Phase 1 audit: what M13 left open

1. **Mathematical.** The single‑chain M\* samples the locally normalised distribution, not the tilted target π. Only SMC is properly weighted for it. In M14 this is formalised (formulation §4) and checked exactly on a toy problem (`outputs/m14/toy/`).
2. **Network coherence, measured correctly.** M13's `giant_share` treats the 512 m window as closed, which makes real tiles look *less* connected than generated ones (DEV‑B real 0.78 vs generated 0.86). After closing the window (border nodes are connected outside), real tiles are almost fully connected, and the M13 generators show three deficits:
   * 4–6× more isolated islands (DEV‑A: real 0.00 vs 0.18–0.26 per tile);
   * about 2× the real interior dead‑end ratio;
   * 3–8× more unreachable same‑land OD pairs.

   Neither terrain constraints nor M13 look‑ahead fixed this (`outputs/m14/audit/`).
3. **Density does not adapt to the tile.** Generated density follows real density only weakly: Spearman 0.33–0.60, slope 0.2–0.5. This is not targeted in M14, because density depends on information the model does not receive (land use, history). It is reported as a failure mode, and density stays a controlled confounder.
4. **Other M13 limitations** stay open and are listed in the final review:
   * no external learned baseline;
   * terrain thresholds computed from small samples;
   * the structure potentials are inactive;
   * DEV‑only evidence.

## Research question and approach

**Question.** Can network‑level objectives, applied at inference time with limited and *adaptively allocated* look‑ahead, remove the network‑coherence deficit without degrading realism or diversity, on geography never seen?

**Approach.**
* **Objectives.** Two telescoping count potentials, closed dead ends and closed islands. Their weights are fitted by moment matching to real *training‑tile* rates (I‑projection), not hand‑tuned.
* **Samplers.** Every sampler's induced distribution is characterised: myopic, fixed roll‑out planning, adaptive roll‑out planning, properly weighted SMC, and a KL trust region.
* **Adaptive trigger.** The trigger is chosen on DEV‑A by measured value of computation, and must beat random allocation at the same budget.
* **Baselines.** A, M\*, M13‑SMC, the procedural baseline P, and post‑hoc procedural repair.
* **Confirmation.** On a fresh city set (GEN2), plus DEV‑B and GEN, three training seeds, leakage controls (TNlp, TNse), density control, terrain strata and synthetic shifts.

## Phases and status

| Phase | Content | Output |
|---|---|---|
| 1 Audit | Closed‑window connectivity; density adaptation | `scripts/m14_audit.py`, `outputs/m14/audit/` |
| 2 Literature | Two verified notes; comparison matrix | `docs/M14_LITERATURE_AND_NOVELTY.md` |
| 3 Mathematics | Formulation; exact toy | `docs/M14_MATHEMATICAL_FORMULATION.md`, `scripts/m14_toy.py` |
| 4 Minimal implementation | `NetSampler`, network metrics, tests | `src/urbanmorph/graphs/m14_{sampler,metrics}.py`, `tests/test_m14.py` |
| 5 Pilot | λ calibration (training tiles); DEV‑A pilot | `scripts/m14_calibrate.py`, `configs/m14_lambda.json` |
| 6 Adaptive | VoC logging; trigger selection on DEV‑A | `scripts/m14_triggers.py`, `configs/m14_frozen.json` |
| 7–8 Comparison, generalisation | GEN2 (primary), DEV‑B, GEN; H5; shifts; density matching | `scripts/m14_experiments.py`, `configs/m14_plans/` |
| 9 Statistics, ablations | Block sign‑flip and bootstrap; noise floor; strata | `scripts/m14_analyze.py`, `scripts/m14_report.py` |
| 10 Documentation | Results, manuscript, final review | `docs/M14_*.md` |

The pre‑registration is `docs/M14_EXPERIMENTAL_PROTOCOL.md`.
