# M15 post-hoc critique (dated 2026-11-21)

* **Status.** POST-HOC: written after every M15 result was seen. It sits beside the frozen protocol
  (`docs/M15_PROTOCOL.md`) and changes no pre-registered verdict in `docs/M15_RESULTS_AND_READINESS.md`; it changes
  how those verdicts should be read.
* **Numbers.** From `scripts/m16_posthoc_m15.py`, which reads the stored M15 confirmatory runs only
  (`outputs/m16/posthoc_m15.md`).

## 1. A correction to numbers given in conversation

Flat-site means quoted in an informal review on 2026-11-21, and repeated in the M16 brief, partly came from the wrong
stored run files:
* the quoted values were M\* grid with road length 5.3 → 3.5 km, loops 74 → 31 / km², four-way share 0.33 → 0.27,
  and radial with radial-or-tangential share 0.41 → 0.76;
* the "style off" file was picked by a filename glob, which matches four M\* off runs: the superseded learned-start
  run and the temperature 0.8 / 1.0 runs.

The M15 report and figures select runs by their exact specification and are not affected. The corrected paired
values (style on − off, same seed, 10 seeds on the one flat site, seed-bootstrap 95 % CI, within one site only):

| Outcome | M\* off | M\* grid − off | NA-fixed off | NA-fixed grid − off |
|---|---|---|---|---|
| R4 (orthogonal order; optimised by the rule) | 0.73 | +0.18 [+0.04, +0.33] | 0.67 | +0.22 [+0.13, +0.33] |
| street km | 3.68 | +0.49 [−0.82, +1.93] | 4.55 | −1.06 [−2.67, +0.45] |
| loops / km² | 36.2 | 0.0 [−23.7, +26.7] | 54.6 | −30.1 [−64.5, −3.1] |
| four-way junction share | 0.38 | +0.01 [−0.12, +0.18] (n 9) | 0.36 | +0.07 [+0.01, +0.13] (n 9) |
| closed blocks | 8.0 | +1.0 [−3.9, +6.8] | 13.2 | −8.1 [−16.0, −1.6] |
| median block area (m²) | 7 385 | +503 [−7 681, +7 871] (n 7) | 2 961 | +8 795 [+3 060, +15 889] (n 7) |

Radial guidance under M\* raised the radial-or-tangential share from **0.53** (not 0.41) to 0.76: +0.23
[+0.10, +0.35].

**What the corrected numbers show:**
* Under M\*, grid guidance moved heading statistics only. No network outcome changed detectably: length, loops,
  junction mix and blocks were all unchanged.
* Under NA-fixed, grid guidance gave fewer loops and fewer, larger blocks, and a small rise in four-way junctions.
* The "length and loops fall, four-way share falls" trade-off in the brief does not hold for M\*.

The conclusion of the brief still stands. **Guidance steers headings; recognisable street layouts need independent
evidence**, and the corrected numbers support it more strongly: at the network level, grid guidance under M\* is
indistinguishable from no guidance.

## 2. How the O1 verdicts should be read

* **The descriptors partly measure the rule itself.** The grid rule rewards headings near θ0 mod 90° and O1 scored
  it with R4; the radial rule rewards radial or tangential headings and O1 scored the radial share. These are
  controllability checks, not evidence of a recognisable pattern. The organic descriptor was "R4 below the grid
  style". Against style off, organic R4 does not change detectably (M\* −0.08 [−0.28, +0.10]).
* **The smallest-effect thresholds were not calibrated to anything visible.** ±0.05 R4 and ±0.05 share were not
  derived from real between-style differences or within-style variability. Seed-to-seed spread of R4 on the one flat
  site is larger than the effect, so single-seed maps show no visible difference (`fig_style_terrain_matrix_*`).
* **O1 rests on one flat site × 10 seeds.** The CIs are within-site and do not generalise across sites.
* **Recognisability was never measured.** There was no blinded sorting, no classifier and no reference distribution
  of real grid or organic networks for the synthetic outcomes.

## 3. Which real-reference comparisons actually ran in M15

| Comparison | Ran? | Styles | Result (report section) |
|---|---|---|---|
| Task A: generated vs real feature distributions (PRDC, features outside every objective), real-vs-real benchmark | yes | off only | learned samplers 0.83–0.95 of the real-vs-real coverage; P 0.39 (B1) |
| Per-metro street length and OD reachability, real vs generated | yes | off only | B2 |
| Grade exposure, real vs generated, by GLO-30 slope stratum | yes | off only | B3 |
| Task B: declared grid on real windows with R4 ≥ 0.5, compared with the real network on non-conditioned features | yes | grid only | no feature (junction shares, block area, circuity) moved closer to real (B4) |
| MDT05 vs GLO-30 elevation | yes | off only | DEM source changes length (B5) |
| Any real reference for organic, radial or cul-de-sac guidance | **no** | – | – |
| Synthetic styled outputs vs real grid / organic reference distributions | **no** | – | – |
| Blinded recognition (people or classifier) | **no** | – | – |

The M14 style-adaptation analysis (`outputs/m14/style/STYLE_ADAPT.md`) compares real and generated orientation
order per city, but it concerns unguided adaptation, not M15 guidance.

## 4. Cul-de-sac guidance and stopping: reconciling the descriptions

The documents describe the cul-de-sac controls in three different ways:
* the protocol (§ strengths): "style changes which continuation is chosen, not whether an arm stops; termination is a
  separate control";
* the method document: "**Local** termination (cul-de-sac). A bias added to the stop logit";
* `graphs/style.py`: "added to every arm's stop logit".

What the code does (`graphs/tc_sampler.py`, `graphs/m14_sampler.py`):

1. **`stop_bias` (+0.25) is global, not local.** It is added to the stop logit of every arm (`tempered()`), before the
   road class is sampled. Major and collector roads stop earlier too. "Local termination" in `docs/M15_METHOD.md` is
   therefore inaccurate; `graphs/style.py` is correct.
2. **The loop-closure penalty sits inside the normalised style factor.** It moves probability from loop-closing to
   non-closing continuations; it does not raise the stop probability directly.
3. **"Leaves Z essentially unchanged" is approximate.** The normaliser makes the mean style weight over *feasible*
   candidates equal to 1. The stop probability depends on p_stop and the mean of the *product* of terrain, network and
   style weights over all K candidates. Z is unchanged only when the style weights are uncorrelated with the others.
   * Measured on the flat site, the per-decision stop rate changes by at most 0.01 for any style (grid −0.00 / +0.01,
     cul-de-sac +0.01 for M\* / NA-fixed).
   * On terrain, where style and terrain weights can correlate, this was not measured.
4. **For NA samplers, cul-de-sacs conflict with the dead-end potential.** `net_stop_lw` gives STOP at an interior
   degree-1 node −λ_D = −1.81 nats, i.e. a factor 0.16 on the stop weight. The cul-de-sac bias is +0.25 on the logit.
   The network potential therefore actively removes the dead ends that cul-de-sac guidance asks for.
   * Consistent with this, NA-fixed cul-de-sac did not raise dead ends: +0.05 [−0.38, +0.56] per km, against
     +0.60 [+0.00, +1.14] under M\*.
   * The protocol did not anticipate the conflict. M16 tests it in its own ablation.

## 5. Consequences for M16

* Style claims need outcomes the rule does not optimise (block area and shape, junction mix, continuity, routes) and
  a blinded recognisability test. Thresholds come from real between-style and within-style variation.
* Cul-de-sac termination must be class- and context-specific (local branches off a collector), and its interaction
  with the dead-end potential needs its own ablation.
* Ad-hoc analyses must select runs by exact specification, as the report does: `scripts/m16_posthoc_m15.py` now
  refuses an ambiguous match.
