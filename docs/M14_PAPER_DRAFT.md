# Network-Level Potentials for Inference-Time Control of a Learned Terrain-Conditioned Road Generator

*Draft manuscript (Milestone 14).*
* Results are taken only from `outputs/m14/REPORT.md` (development and generalization data).
* No result comes from the held-out test splits, which have not been read.
* The novelty statement is conditional (§2); the conditions are listed in `docs/M14_LITERATURE_AND_NOVELTY.md` §4.

## Abstract

Learned autoregressive road-graph generators place each street from local context. Their output networks are fragmented: compared with real streets, a terrain-conditioned generator of the Neural Turtle Graphics family produces about twice the interior dead ends and two to six times the isolated components. We ask whether network-level objectives applied at inference time, without retraining, can remove this deficit on cities never seen.

**Method.** We add two count potentials to the sampling target: closed interior dead ends and closed islands. We prove that their per-step increments telescope exactly to the final counts, and we set their weights by moment matching to real *training* statistics, which makes the exact target the I-projection of the learned prior onto those statistics. We characterise which distribution each sampler actually draws: single-chain selection with a myopic weight, truncated roll-out twists (everywhere or adaptively triggered), and properly weighted SMC.

**Evaluation.** The evaluation was pre-registered. The primary set is four fresh cities (95 tiles, 34 geographic blocks) with block-level randomisation tests, and it adds two leakage-controlled backbones and three density analyses.

**Results.**
* **Connectivity.** Network-aware sampling reduces same-land unreachable origin–destination pairs by 0.026 (from 0.037; 95 % CI [−0.036, −0.015], p < 10⁻⁴), with no S10 realism cost, and brings the dead-end ratio to the real level.
* **Replication.** The effect replicates on two secondary sets, in all seven cities, in every terrain stratum, for all three training seeds, and with both leakage-controlled backbones (Holm-significant). Its sign is the same under density adjustment and density matching.
* **Best variant.** The SMC variant is also more realistic than the constrained baseline (S10 −0.037, p = 0.026) at 4.3× its cost.
* **Negative result.** Allocating look-ahead adaptively, by a trigger chosen to capture value of computation, was no better than random allocation at the same rate on any set. It also failed its realism non-inferiority margin.
* **Over-connection.** On the primary set, real networks are themselves less connected than all generators, so lower unreachability is not uniformly closer to real.

## 1. Introduction

**The problem.** Generative models of street networks are trained to reproduce local geometry. They are evaluated mostly on distributional statistics of that geometry. Network function is a global property: whether a street leads anywhere, and whether two nearby points are connected. An autoregressive generator decides each step from a local window and a recurrent state; nothing at sampling time rewards a road end that will connect later or penalises a fragment that never will.

**Our earlier work.** In Milestone 13 we found that inference-time terrain constraints made generated streets buildable but did not change fragmentation. After the window boundary is handled correctly (border nodes connect outside the tile), real tiles are almost fully connected. The constrained generator has 4–6× the isolated components and about 2× the interior dead ends (audit, `outputs/m14/audit/`).

**This paper's question.** Does adding network-level objectives at inference time close this gap on unseen geography? How should such objectives be weighted? Is limited look-ahead needed, and if so, where should it be spent?

**Contributions,** stated with their limits:
1. **Telescoping network potentials** for a learned, terrain-conditioned vector road generator. They give an exact per-step decomposition of the final dead-end and island counts (Proposition 1) and are applied without retraining.
2. **Moment-matched weights.** The two weights are fitted to real training-tile statistics by generalized iterative scaling. Under the exact target, this is the I-projection of the learned prior (Proposition 2). In practice it is moment matching under the implemented sampler, which we state explicitly.
3. **A characterisation of what each sampler targets,** verified on an exact toy problem. This includes the finding that, in a single chain, *where* look-ahead is spent changes the distribution, in either direction.
4. **A pre-registered generalization study** on four fresh cities, with block-level inference, leakage controls and density controls. It includes a negative result for adaptive look-ahead allocation.

**What we do not claim.** We claim no novelty for the inference machinery (SMC, twisted SMC, look-ahead, KL regularisation), all of which are established (§2). The novelty of contribution 1 is conditional on literature we could not access (§2).

## 2. Related work

**Learned road generators.**
* **Neural Turtle Graphics** (Chu et al., ICCV 2019) generates road graphs autoregressively with local validity checks. It has no network-level objective at inference. Our generator is an NTG-family reimplementation, not a reproduction.
* **Raster models.** Fang et al. (IJGIS 2022) condition a GAN on elevation, slope and aspect and measure connectivity, but do not optimise it. Gu et al. (IJGI 2024) use a conditional diffusion model with terrain and land-use inputs; it is trained per city and takes ground-truth intersections as input.
* **Birsak et al. (arXiv 2022)** generate large street networks autoregressively and delete small fragments post hoc.
* **RoadWeaver (Li et al., arXiv 2026)** reconnects dangling ends after generation by A\* search and keeps the largest component. That is the post-hoc repair we use as a baseline.
* **Zhou et al. (CAVW 2024)** propose a transformer-based vectorised road generator with terrain inputs. We could access only its abstract, so we cannot rule out overlap with contribution 1.

**Procedural generation.** Parish & Müller (SIGGRAPH 2001) and Chen et al. (TOG 2008) use rules and local look-ahead to snap roads to intersections. Galin et al. (CGF 2010, 2011) compute terrain-aware roads by anisotropic shortest paths. These methods connect by construction but have no learned prior.

**Search over graph construction.** Anagnostidis et al. (ICCV 2023) use MCTS with topology rewards to *extract* road graphs from images against ground truth. Our setting is generation, with no ground truth at inference.

**Inference foundations.**
* Proper weighting of nested proposals: Naesseth et al. (ICML 2015).
* Look-ahead SMC, including adaptive look-ahead depth: Lin, Chen & Liu (Stat. Sci. 2013).
* Twisted and controlled SMC: Whiteley & Lee (AoS 2014), Guarniero et al. (JASA 2017), Heng et al. (AoS 2020), Zhao et al. (ICML 2024).
* SMC steering of generative models: Lew et al. (arXiv 2023).
* The minimum-KL distribution under expectation constraints: Ganchev et al. (JMLR 2010, Prop. 2.1; the original I-projection result is Csiszár 1975, which we did not access).
* Per-update KL bounds: Peters et al. (AAAI 2010); Schulman et al. (ICML 2015).
* Allocating computation by its value: Russell & Wefald (1991); Hay et al. (UAI 2012).

**Few-cluster inference.** Cameron, Gelbach & Miller (REStat 2008); Canay, Romano & Shaikh (Econometrica 2017).

## 3. Problem formulation

**The generator.** The frozen generator p_θ (TerrainGraphTurtle, trained in Milestone 12) builds a graph x by expanding queued nodes. At each decision it proposes a continuation, or STOP, conditioned on a terrain window (elevation and water) and a GRU state. Milestone 13 added hard feasibility constraints 1_F (water crossings only as bridges, grade limits) and density-ratio terrain potentials φ_r.

**Network counts.** An interior node is *open* while it can still gain edges. For a state s:
* D(s) is the number of interior nodes of degree 1 that are not open (closed dead ends);
* I(s) is the number of components of at least 40 m with no border node and no open node (closed islands).

**Network factor.** G_t = exp(−λ_D ΔD_t − λ_I ΔI_t).

**Proposition 1 (telescoping).** For every complete trajectory, Π_t G_t = exp(−λ_D D(x) − λ_I I(x)), provided every transition that closes a node (including closure by reaching the arm limit without STOP) carries its increment. This is checked to machine precision in the unit tests.

**Target.** π_λ(τ) ∝ p_θ(τ) Π_t 1_F exp(λ_r Σ φ_r − λ_D D(x) − λ_I I(x)). This is an energy-reweighted prior, not a posterior.

**Proposition 2 (I-projection).** Among distributions with E[D] = d\* and E[I] = i\*, π_λ minimises KL to the feasible, terrain-tilted prior, for the λ solving the moment equations.

**Fitting λ.** We fit λ by generalized iterative scaling to the real training-tile rates (48 training tiles; d\* = 2.27 dead ends and i\* = 0.104 islands per tile). This gives λ_D = 1.81 and λ_I = 0: with the dead-end potential active, islands are already below the real rate. The moments are estimated with the implemented sampler, so the minimum-KL property holds only for the exact tilt.

## 4. Samplers and what they target

| Sampler | Mechanism | Distribution |
|---|---|---|
| M\* (Milestone 13) | K = 8 candidates, hard constraints, terrain potentials | locally normalised (heuristic) |
| NA-myopic | M\* + network factor on each continuation and on STOP | locally normalised (heuristic) |
| NA-fixed | + H = 2-step roll-out twist at every decision (M = 4 candidates per roll-out step) | truncated twist (heuristic) |
| NA-adaptive | twist only where a trigger fires | heuristic; trigger changes the distribution |
| NA-SMC | 4 particles, properly weighted K-candidate proposal | asymptotically π_λ |
| M\*+repair | M\*, then A\* reconnection of dead ends | post hoc |

**Exact toy (T = 8, 3 actions, full enumeration).**
* **C1, label bias.** Local selection alone is 1.12 nats from π.
* **C2, truncated twists.** They converge to π as H grows (0.25, 0.08, 0.014 nats at H = 1, 2, 3).
* **C3, finite K.** More candidates do not monotonically approach π.
* **C4, trigger placement.** With look-ahead at 41 % of decisions, a plausible "near a constraint" trigger is *worse* than random placement (0.77 vs 0.62 nats). An oracle value-of-computation trigger is much better (0.27).

**The adaptive trigger.**
* **Pre-registration.** A trigger must be chosen by measured value of computation (VoC) on development data and must beat random allocation.
* **VoC.** The VoC of a decision is the total-variation distance between selection with and without the twist.
* **Choice.** On DEV-A (25 tiles), the trigger "candidate leaves a degree-1 node" (`deg1`) captured 78 % of VoC at 43 % of decisions and was frozen.
* **Pilot.** It did not beat random allocation (Amendment 2). As pre-registered, it was still evaluated.

**KL trust region (exploratory).** It bounds the per-decision KL to the base selection, which gives a trajectory-level bound by the chain rule. It was not needed at the calibrated λ.

## 5. Experimental setup

**Data.** No tile from any set is in the training split. The sealed test splits were not used.

| Set | Role | Content |
|---|---|---|
| GEN2 | primary | Alcoi, Tarragona, Santa Barbara, Sacramento; 95 tiles of 512 m, 34 blocks of 3 km. Chosen by known geography before any data were seen, and built with the unchanged Milestone 13 pipeline. |
| DEV-B | secondary | 27 tiles, 13 blocks |
| GEN | secondary | Girona, Manresa, San Francisco; 100 tiles, 24 blocks |

**Models.** Three training seeds of the terrain-conditioned backbone (TN). For leakage control, two further backbone families:
* TNlp: low-pass DEM;
* TNse: DEM with street imprints erased by a model-based procedure.

**Outcomes.** All were pre-registered.
* **Primary:** same-land OD unreachability, the share of street-node pairs 150–500 m apart on the same dry-land region with no path on the window-closed graph (empty network = 1).
* **Realism:** S10, the mean normalised Wasserstein distance to real tiles over 10 network statistics.
* **Diversity:** Jaccard distance between samples.
* **Feasibility:** unbridged crossings.
* **Cost:** seconds per tile-sample on CPU.

**Statistics.**
* The geographic block is the unit.
* Per-tile metrics are tested by block sign-flip randomisation, with block-bootstrap CIs.
* Set-level metrics use a joint block bootstrap and within-block label randomisation.
* Holm correction is applied over the four primary GEN2 tests.
* Real-vs-real S10 between block halves gives a noise floor.

**Density control.**
* (i) Per-tile differences are regressed on log-density differences, and the intercept is reported.
* (ii) Stop-logit biases are matched on DEV-A and applied to GEN2.

The conclusion requires sign agreement.

## 6. Results

### 6.1 Main comparison (GEN2)

| Method | OD unreach. | Islands / tile | Dead-end ratio | S10 | Density / real | Jaccard | s / tile |
|---|---|---|---|---|---|---|---|
| real | 0.047 | 0.13 | 0.023 | — | 1.00 | — | — |
| A (ancestral) | 0.035 | 0.28 | 0.052 | 0.433 | 1.20 | 0.772 | 0.37 |
| M\* | 0.037 | 0.28 | 0.054 | 0.369 | 1.12 | 0.772 | 0.64 |
| M\*+repair | 0.006 | 0.02 | 0.044 | 0.424 | 1.15 | 0.773 | 0.64 |
| M13-SMC | 0.048 | 0.23 | 0.051 | 0.309 | 0.90 | 0.771 | 2.19 |
| NA-myopic | 0.010 | 0.08 | 0.022 | 0.501 | 1.34 | 0.764 | 0.86 |
| NA-adaptive | 0.012 | 0.09 | 0.022 | 0.486 | 1.32 | 0.752 | 5.35 |
| NA-fixed | 0.011 | 0.07 | 0.019 | 0.370 | 1.21 | 0.758 | 9.75 |
| NA-SMC | 0.012 | 0.02 | 0.014 | 0.332 | 1.11 | 0.758 | 2.74 |

**H1 (NA-fixed − M\*).** Δ = −0.0259 [−0.0363, −0.0154], p < 10⁻⁴.
* Per training seed: −0.019, −0.033, −0.026.
* S10 Δ = +0.000; the realism guard holds.

**Secondary sets.**
* DEV-B: Δ = −0.021, p = 0.0005.
* GEN: Δ = −0.017, p < 10⁻⁴, where NA-fixed also lowers S10 (−0.095, p = 0.002).

**Per city.** The effect has the same sign in all seven cities. It is largest in the mountain town (Alcoi: 0.062 → 0.011 on non-empty tiles).

**SMC.** NA-SMC reaches the same connectivity with lower S10 than M\* on GEN2 (−0.037, p = 0.026) and GEN (−0.089, p = 0.002), at 4.3× the cost of M\*.

**Post-hoc repair** connects slightly more (0.006) but raises the terrain-realism gap S_T by 0.17. It creates detours (1.59 vs 1.44) and 0.035 unbridged crossings per tile.

### 6.2 Leakage controls (H5)

NA-adaptive − M\* is negative with a CI excluding zero for all three seeds of both control families:
* TNlp: −0.017, −0.025, −0.013;
* TNse: −0.039, −0.041, −0.029.

Holm-adjusted, H1, H5-TNse and H5-TNlp are significant. For TNlp_s0 the bootstrap CI excludes zero while the sign-flip p is 0.072.

### 6.3 Density

The single-chain network samplers increase street density (1.29–1.34 vs 1.12). The connectivity effect survives both density analyses:
* density-adjusted intercepts: −0.021 to −0.025;
* DEV-A-matched contrasts: NA-SMC − M\* −0.026 (p = 0.013), NA-myopic − M\* −0.019 (p = 0.0006).

The matching biases did not transfer to GEN2 (matched densities 1.01–1.30), and S10 comparisons reverse under matching. We therefore claim density-robustness for connectivity only.

### 6.4 Adaptive look-ahead (H3): negative

| Set | OD vs fixed (margin +0.010) | S10 vs fixed (margin +0.02) | Cost | vs random |
|---|---|---|---|---|
| GEN2 | +0.002 [−0.002, +0.005] | **+0.116 [+0.050, +0.189]** | 0.55× | +0.001, p = 0.66 |
| DEV-B | −0.004 | +0.094 | 0.56× | p = 0.49 |
| GEN | −0.000 | +0.085 | 0.55× | p = 0.81 |

* **The adaptive sampler is cheaper and as connected, but less realistic** than fixed look-ahead.
* **Its trigger is indistinguishable from random placement.**
* **Cheaper triggers do as well.** In descriptive ablations, triggers that fire on 1–16 % of decisions (near water, near islands) matched `deg1` at 2–6× lower cost.
* **Our reading** (not tested): the VoC proxy measures local change in the next selection, and most such changes lead to equivalent networks. Connectivity is mostly decided by the STOP weight, which the myopic sampler already has.

### 6.5 Planning strength (H2)

* **Dose.** Raising λ beyond the calibrated value drives dead ends below the real rate and raises S10 (DEV-B: 0.411 at m = 0, 0.532 at m = 1, 0.732 at m = 4).
* **Verdict.** The rise is in the predicted direction but below the pre-registered real-vs-real noise floor (0.422), so H2 is not supported.

### 6.6 Strata and shifts

* **Strata.** The baseline deficit grows with terrain difficulty (non-empty OD 0.015 flat → 0.078 mountainous). The network potentials remove most of it in every stratum.
* **Synthetic shifts.** Under synthetic rough terrain, M\* collapses (premature termination 0.41, density 0.56), while NA-myopic stays connected (OD 0.017). This is a robustness probe, not evidence of real-world generalization.

## 7. Limitations

* **Over-connection.** On GEN2, real networks are less connected than any generator's (OD 0.047; Tarragona 0.128), likely because of motorways, rail, gated estates and mapping gaps. Lower unreachability is closer to real on DEV-B and GEN but not on GEN2. The network potentials match dead-end and island *rates*, not the reasons real networks are fragmented.
* **Density.** Network-aware single-chain sampling inflates density. Density calibration does not transfer across city sets, and all generators over-build in Tarragona (1.4–1.7×).
* **λ is sampler- and backbone-specific.**
  * Applied to the other two backbones, λ gives 19–27 % more dead ends than its target.
  * Under SMC, dead ends fall below real.
  * A deployment would recalibrate per backbone and sampler (about 1.5 CPU-hours).
* **Start-head failures.** On five GEN tiles, the frozen start head proposes no start for any sampler, which no inference method can repair.
* **No external learned baseline.** The raster models with code (Fang 2022, Gu 2024) need retraining and take ground-truth junctions as input. Repair is our only external-style baseline.
* **Scale.** The evaluation covers seven cities in two countries, with 13–34 blocks per set. DEV-B allows a smallest p of 0.00024.
* **Post-hoc analysis fixes.** Three analysis bugs were found and fixed after the confirmatory runs (a tile-set contamination, a duplicate-run lookup, mixed seed subsets). Each brings the analysis closer to the pre-registered specification, none changed a verdict, and all are documented.
* **Residual hard-constraint violations.** These are 1 per 285–300 tile-samples in three runs, unexplained, and likely sampling aliasing.
* **Novelty.** The novelty statement is conditional: Zhou et al. 2024 and two 2025 diffusion works could not be read in full, and no Scopus or Web of Science search was possible.

## 8. Conclusion

* **Main result.** Two count potentials with exactly decomposable increments, weighted by moment matching to real training statistics, remove most of the fragmentation of a learned terrain-conditioned road generator on unseen cities. They do so without retraining and without a realism or diversity cost when the sampler corrects for its own bias (full roll-outs or SMC).
* **Where the gain comes from.** The gain does not come from spending look-ahead cleverly: adaptive allocation was no better than random.
* **Cheapest effective form.** It is the STOP weight inside a properly weighted sampler.
* **Remaining gap.** The remaining gap to real networks is not fragmentation but the *structure* of real disconnection and density, which requires inputs (land use, infrastructure) the generator does not have.

## Reproducibility

* **Code, configuration and protocol.** `docs/M14_REPRODUCIBILITY.md` lists one command per stage. The pre-registration is `docs/M14_EXPERIMENTAL_PROTOCOL.md`, with two dated amendments made before any confirmatory run. Per-run JSON (spec, per-tile metrics and graphs) is regenerable from the frozen checkpoints.

## References

All entries were verified against Crossref, OpenAlex, arXiv or the publisher. Access levels are in `docs/M14_LITERATURE_AND_NOVELTY.md`.

**Road generation and procedural modelling**
* Anagnostidis, S., Lucchi, A., Hofmann, T. Mastering Spatial Graph Prediction of Road Networks. ICCV 2023. doi:10.1109/ICCV51070.2023.00498.
* Birsak, M., Kelly, T., Para, W., Wonka, P. Large-Scale Auto-Regressive Modeling Of Street Networks. arXiv:2209.00281, 2022.
* Chen, G., Esch, G., Wonka, P., Müller, P., Zhang, E. Interactive Procedural Street Modeling. ACM TOG 27(3):103, 2008. doi:10.1145/1360612.1360702.
* Chu, H., Li, D., Acuna, D., Kar, A., Shugrina, M., Wei, X., Liu, M.-Y., Torralba, A., Fidler, S. Neural Turtle Graphics for Modeling City Road Layouts. ICCV 2019, 4521–4529. doi:10.1109/ICCV.2019.00462.
* Fang, Z., Qi, J., Fan, L., Huang, J., Jin, Y., Yang, T. A topography-aware approach to the automatic generation of urban road networks. IJGIS 36(10):2035–2059, 2022. doi:10.1080/13658816.2022.2072849.
* Galin, E., Peytavie, A., Maréchal, N., Guérin, E. Procedural Generation of Roads. CGF 29(2):429–438, 2010. doi:10.1111/j.1467-8659.2009.01612.x.
* Galin, E., Peytavie, A., Guérin, E., Beneš, B. Authoring Hierarchical Road Networks. CGF 30(7):2021–2030, 2011. doi:10.1111/j.1467-8659.2011.02055.x.
* Gu, X., Zhang, M., Lyu, J., Ge, Q. Generating Urban Road Networks with Conditional Diffusion Models. ISPRS IJGI 13(6):203, 2024. doi:10.3390/ijgi13060203.
* Li et al. RoadWeaver. arXiv:2608.11580, 2026.
* Parish, Y. I. H., Müller, P. Procedural Modeling of Cities. SIGGRAPH 2001, 301–308. doi:10.1145/383259.383292.
* Zhou, P., Li, C., Zhang, J., Wang, C., Qin, H., Liu, L. A novel transformer-based graph generation model for vectorized road design. CAVW 35(3):e2267, 2024. doi:10.1002/cav.2267. *(abstract only)*

**Inference, planning and statistics**
* Cameron, A. C., Gelbach, J. B., Miller, D. L. Bootstrap-Based Improvements for Inference with Clustered Errors. REStat 90(3):414–427, 2008. doi:10.1162/rest.90.3.414.
* Canay, I. A., Romano, J. P., Shaikh, A. M. Randomization Tests Under an Approximate Symmetry Assumption. Econometrica 85(3):1013–1030, 2017. doi:10.3982/ECTA13081.
* Csiszár, I. I-Divergence Geometry of Probability Distributions and Minimization Problems. Ann. Probab. 3(1):146–158, 1975. doi:10.1214/aop/1176996454. *(not accessed; cited via Ganchev et al.)*
* Ganchev, K., Graça, J., Gillenwater, J., Taskar, B. Posterior Regularization for Structured Latent Variable Models. JMLR 11:2001–2049, 2010.
* Guarniero, P., Johansen, A. M., Lee, A. The iterated auxiliary particle filter. JASA 112(520):1636–1647, 2017. doi:10.1080/01621459.2016.1222291.
* Hay, N., Russell, S., Tolpin, D., Shimony, S. E. Selecting Computations: Theory and Applications. UAI 2012. arXiv:1207.5879.
* Heng, J., Bishop, A. N., Deligiannidis, G., Doucet, A. Controlled sequential Monte Carlo. Ann. Statist. 48(5):2904–2929, 2020. doi:10.1214/19-AOS1914.
* Holm, S. A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Statist. 6(2):65–70, 1979.
* Lew, A. K., Zhi-Xuan, T., Grand, G., Mansinghka, V. K. Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs. arXiv:2306.03081, 2023.
* Lin, M., Chen, R., Liu, J. S. Lookahead Strategies for Sequential Monte Carlo. Stat. Sci. 28(1):69–94, 2013. doi:10.1214/12-STS401.
* Naesseth, C. A., Lindsten, F., Schön, T. B. Nested Sequential Monte Carlo Methods. ICML 2015, PMLR 37:1292–1301.
* Peters, J., Mülling, K., Altün, Y. Relative Entropy Policy Search. AAAI 2010, 1607–1612. doi:10.1609/aaai.v24i1.7727.
* Russell, S., Wefald, E. Principles of metareasoning. Artificial Intelligence 49(1–3):361–395, 1991. doi:10.1016/0004-3702(91)90015-C. *(bibliographic record only)*
* Schulman, J., Levine, S., Abbeel, P., Jordan, M., Moritz, P. Trust Region Policy Optimization. ICML 2015, PMLR 37:1889–1897.
* Whiteley, N., Lee, A. Twisted particle filters. Ann. Statist. 42(1):115–141, 2014. doi:10.1214/13-AOS1167.
* Zhao, S., Brekelmans, R., Makhzani, A., Grosse, R. Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo. ICML 2024, PMLR 235:60704–60748.
