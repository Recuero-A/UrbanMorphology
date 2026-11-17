# Milestone 14 final review (strict‑reviewer assessment)

This is written as a skeptical conference reviewer would write it, from the evidence in `outputs/m14/REPORT.md` and the M14 documents only. The sealed test splits were not read. Every confirmatory number is on development or generalization data.

## Answers to the fifteen questions

**1. Is the research question genuinely important?**
* **Moderately, yes.** Fragmentation is a real, measured deficit of learned road generators. M13's generator had 4–6× the islands and 2× the dead ends of real networks once the tile window is handled correctly, and published learned generators either ignore it or repair it post hoc.
* **Importance is limited by scope.** The question is about one generator family at 512 m tiles. Whether network‑level coherence at this scale matters for downstream use (simulation, games, planning sketches) is argued, not shown.

**2. Is the proposed method novel relative to the closest prior work?**
* **Only narrowly, and conditionally.**
* **The machinery is established.** Count potentials, I‑projection weights, twisted and look‑ahead SMC, and KL trust regions all exist (Ganchev 2010; Lin 2013; Guarniero 2017; Zhao 2024; Peters 2010).
* **The candidate novelty is their combination.** Exactly telescoping, moment‑matched *network* potentials applied at inference to a learned terrain‑conditioned vector road generator. We found no work doing this among those read.
* **That search is incomplete.** Zhou et al. 2024 (abstract only) and two 2025 diffusion works (metadata only) could not be read, and no Scopus or Web of Science search was possible.
* **The second candidate contribution failed.** Adaptive allocation by VoC did not beat random.
* **Novelty: small.**

**3. Is the mathematical formulation correct?**
* **Yes, as far as it claims.**
  * Proposition 1 (telescoping) is exact, including closures without STOP, and is tested to machine precision.
  * Proposition 2 is the standard I‑projection result, cited through Ganchev et al., because Csiszár 1975 was not accessed.
  * The formulation states that the fitted λ matches moments *under the implemented sampler*, not the exact tilt.
  * Each sampler's target is stated honestly: only SMC is asymptotically exact.
  * The exact toy confirms the qualitative claims (label bias, truncated twists, non‑monotone K, trigger placement).
* **Weak point: the toy is tiny** (T = 8, 3 actions), and the real samplers' induced distributions are characterised empirically, not bounded.

**4. Does adaptive planning outperform fixed planning?**
* **No.**
  * The pre‑registered H3 fails on all three sets: S10 is +0.085 to +0.116 worse than fixed, outside the +0.02 margin.
  * The adaptive trigger is indistinguishable from random allocation at the same rate (p = 0.66, 0.49, 0.81).
  * It is cheaper (0.55×) and equally connected, but so is random allocation.
* **The pilot had already failed this check** before confirmation, and this was recorded (Amendment 2).

**5. Does it improve connectivity?**
* **Yes, strongly and consistently.**
  * Same‑land OD unreachability falls by 0.026 on GEN2 (CI [−0.036, −0.015], p < 10⁻⁴), about 70 % of the baseline.
  * It also falls on DEV‑B (−0.021) and GEN (−0.017).
  * The sign is negative in 9/9 set × seed cells, 7/7 cities and 7/7 terrain strata.
  * Islands fall 3–14× and dead ends reach the real level.
* **Caveat.** On GEN2 the real networks are less connected than the generated ones, so "more connected" is not "more real" there.

**6. Does it preserve realism?**
* **Depends on the sampler.**
  * NA‑fixed: S10 +0.000 vs M\* (GEN2), −0.095 (GEN).
  * NA‑SMC: −0.037 (GEN2), −0.089 (GEN).
  * Both have the lowest energy distance of all methods.
  * The single‑chain variants (myopic, adaptive) worsen S10 by 0.12–0.15 on GEN2 and DEV‑B (GEN: +0.05 and −0.01) and inflate density to about 1.3×.
* **S10 is density‑sensitive.** Under density matching the ordering reverses, so realism claims beyond "no degradation for NA‑fixed and NA‑SMC" are not supported. All differences are within the real‑vs‑real noise floor.

**7. Does it preserve diversity?**
* **Yes.** H4 is met on all sets (Jaccard ratio 0.974–0.994).
* The only variation removed is fragmentation, which is the target.

**8. Does it generalize to unseen terrain?**
* **To unseen cities in the same two countries: yes, for connectivity.**
  * GEN2 is four fresh cities chosen before any data were seen.
  * The effect holds in every city and stratum, including mountain and river terrain.
* **Synthetic terrain shifts are robustness probes only.** Under rough terrain, NA‑myopic held, M\* collapsed and NA‑adaptive degraded.
* **Limits.**
  * Seven cities, Spain and the USA only.
  * Density calibration does not transfer between city sets.
  * λ does not transfer exactly across backbones (19–27 % off).
  * The sealed test has not been run.

**9. Do improvements survive density matching?**
* **For connectivity, yes:**
  * density‑adjusted intercepts are −0.021 to −0.025, all CIs excluding 0;
  * DEV‑A‑matched contrasts are −0.019 to −0.026 (p ≤ 0.013);
  * the sign agrees across all three analyses.
* **Two qualifications.**
  * The matching did not achieve matched density on GEN2 (1.01–1.30).
  * Realism comparisons do *not* survive matching.

**10. Do improvements survive terrain‑leakage controls?**
* **Yes.** The CI excludes 0 in 3/3 seeds for both TNlp and TNse, and both are significant after Holm correction.
* **Caveat.** One TNlp seed has sign‑flip p = 0.072.
* **What this shows.** The street‑erased DEM is a model‑based reconstruction, so this shows robustness to that erasure, not independence from all leakage.

**11. Are results statistically convincing?**
* **For H1, H4 and H5: yes.**
  * The analysis is pre‑registered, with geographic blocks as the unit, sign‑flip randomisation and Holm correction.
  * The effects are consistent across seeds, cities and strata, and the p‑values are at the attainable floor.
* **Weaknesses.**
  * DEV‑B has only 13 blocks.
  * Training seeds share data.
  * Set‑level S10 tests have coarse p resolution.
  * Three analysis bugs were fixed after confirmation. All moved the analysis *toward* the pre‑registered specification and none changed a verdict, but a reviewer will notice them.
  * Matched‑density runs use 2 inference seeds.

**12. Are the improvements worth the computational cost?**

| Variant | Cost vs M\* | Assessment |
|---|---|---|
| NA‑myopic | 1.3× | Most of the connectivity gain, at a density/realism cost |
| NA‑SMC | 4.3× | Connectivity plus better realism; **the best trade‑off** |
| NA‑fixed | 15× | Not worth it over NA‑SMC |
| NA‑adaptive | 8× | Not worth it: no benefit over random allocation or over NA‑SMC |

All variants run in seconds per 512 m tile on CPU.

**13. Are the external baselines sufficiently strong?**
* **No.**
  * The baselines are the project's own: A, M\*, M13‑SMC, a procedural generator and post‑hoc A\* repair (RoadWeaver‑style).
  * No external *learned* generator was run. Fang 2022 and Gu 2024 are raster models that need retraining and take ground‑truth junctions as input; NTG has no code.
  * The repair baseline is the right simple alternative, and the network‑aware samplers beat it on realism, not on raw connectivity.
* **A reviewer would ask for at least one external learned model.**

**14. Can the experiments be reproduced?**
* **Largely yes.**
  * One command per stage.
  * Frozen configs and plans in git.
  * Per‑run JSON with specs, per‑tile metrics and graphs.
  * Seeds recorded, and duplicate runs reproduce identical rows.
  * 16 unit tests pass.
* **Not in git:** checkpoints and data (regenerable with earlier milestone scripts).
* **Timing depends on parallel load.** Forward passes and roll‑out steps are recorded as load‑independent costs.

**15. What would a strict reviewer reject?**
1. **Novelty.** "Inference‑time potentials + SMC is standard; the network counts are simple; the combination is an application." With adaptive planning failing, the methodological contribution is thin.
2. **No external learned baseline**, and no evaluation on a sealed test.
3. **Realism.** Realism is measured by one composite distance (S10) whose between‑method differences sit inside the real‑vs‑real noise floor and reverse under density matching.
4. **Over‑connection on the primary set.** The headline metric moves *away* from real on GEN2.
5. **Generalization scope.** Seven cities in two countries.
6. **Post‑confirmation analysis fixes** (even though documented).
7. **No user or downstream study** showing that the connectivity gain matters.

## Separate assessments (1 = poor, 5 = excellent)

| Dimension | Score | Justification |
|---|---|---|
| Methodological novelty | **2 / 5** | The application of established inference to network‑level road objectives is new among works read, but conditional on unread literature. The adaptive‑allocation idea failed. |
| Technical correctness | **4 / 5** | The propositions are proved and tested, sampler targets are stated honestly, and the toy confirms the claims. The calibration is moment matching under the implemented sampler, not exact I‑projection. One residual constraint violation is unexplained. |
| Experimental rigor | **4 / 5** | Pre‑registered, block‑level, leakage‑ and density‑controlled, with negative results reported. It loses a point for the missing external learned baseline, the sealed test not yet run, 13 blocks on DEV‑B, and post‑hoc bug fixes. |
| Generalization | **3 / 5** | A strong and consistent connectivity effect on fresh cities and three backbones. But the evidence is narrow geographically, density and λ do not transfer, there is a start‑head failure on GEN, and there is no sealed‑test result. |
| Graphics relevance | **2 / 5** | It improves the network coherence of generated street layouts, which matters for any downstream use. But there is no 3D or interactive demonstration in M14, no user study, galleries not density‑matched, and 512 m tiles only. |
| Overall publication readiness | **2 / 5** | It is a solid workshop paper as is. It is not ready for a main‑track graphics, vision or ML conference. |

## Venue recommendation

**Now.** A **workshop** on probabilistic inference for generative models or on geospatial / urban AI. The SMC and twisted‑inference framing and the negative adaptive‑allocation result fit the first; the pre‑registered fresh‑city protocol and the connectivity metrics fit the second.
* Examples are workshops co‑located with NeurIPS, ICML or ICLR on structured probabilistic inference and generative modelling, or ACM SIGSPATIAL workshops.
* The exact venue depends on the year's call for papers, which we have not checked.

**After the following, a full paper becomes plausible** at a GIScience journal (IJGIS, *Computers, Environment and Urban Systems*) or as a Eurographics / Pacific Graphics short paper:
1. Authorised evaluation on the sealed test.
2. One external learned baseline.
3. Reading the remaining literature (Zhou 2024, DRoLaS, CaRoLS) and a database search.
4. Per‑backbone, per‑sampler λ calibration.
5. A realism evaluation that is robust to density (for example, per‑statistic comparisons at matched density).

**Main‑track** SIGGRAPH, CVPR or ICCV would additionally need a substantially stronger method contribution than the evidence currently supports.

## Completion against the M14 criteria

| Criterion | Status |
|---|---|
| Methods implemented and tested | Done (`NetSampler`, metrics; 16 tests pass) |
| Mathematical claims verified | Done (Proposition 1 tested; exact toy C1–C6) |
| Primary development experiments finished | Done (GEN2, DEV‑B, GEN; 3 backbones × 3 seeds) |
| Strongest appropriate baselines evaluated or documented as unavailable | Partly: internal and procedural baselines plus repair were run; external learned baselines are documented as unavailable within budget |
| Realism, connectivity, diversity, feasibility evaluated | Done |
| Generalization on authorised development data | Done (GEN2 primary, GEN, DEV‑B, H5, shifts, strata) |
| Statistics and ablations | Done |
| Failure cases documented | Done (`M14_FAILURE_ANALYSIS.md`) |
| Reproducible | Done, except for data and checkpoints, which are not in git |
| Manuscript and final review | Done (`M14_PAPER_DRAFT.md`, this file) |
| Sealed test | **Untouched**; it needs explicit authorisation |

**Not done (out of scope or budget):**
* 3D (Blender or Godot) demonstrations of M14 outputs;
* density‑matched galleries;
* an external learned baseline;
* sealed‑test evaluation.

**Before any submission,** check the venue's policy on disclosure of tool use in research and writing.
