# Milestone 14 literature review and novelty assessment

Compiled on 2026-11-17, before any M14 confirmatory result. The full per-work notes are in two files:
* `outputs/m14/literature/road_generation.md`: road and street generation, 19 works;
* `outputs/m14/literature/inference_planning.md`: inference, planning and statistics, 17 groups.

Both files record citations verified against Crossref, OpenAlex or arXiv, the access level actually achieved, and section pointers for every claim we rely on. The M13 review (`docs/M13_LITERATURE_REVIEW.md`) still applies. This document records what changed in M14 and what the proposed contribution can and cannot claim.

**Access levels:**
* **F**: we read the full text.
* **S**: full text, skimmed.
* **A**: abstract or metadata only.
* **N**: not accessible.

Nothing marked A or N is used to support a technical claim.

## 1. Comparison matrix: generators

| Work | Access | Output | Learned prior | Terrain input | Inference-time search / constraints | Network-level objective at inference | Diversity evaluated | Unseen cities | Code |
|---|---|---|---|---|---|---|---|---|---|
| Chu et al. 2019, NTG (ICCV) | F | vector graph | yes (GRU encoder/decoder, queue) | none | local hard checks (merge 5 m, max degree, density, min angle); prior × CNN likelihood for parsing | no | yes (novelty vs. ground truth) | parsing only | no official code |
| Birsak et al. 2022 (arXiv) | F | index field → traced graph | yes (VQ-VAE + transformer) | water only (terrain explicitly excluded) | none; small fragments deleted post hoc | no | qualitative | qualitative (Istanbul) | none found |
| Fang et al. 2022 (IJGIS) | F | raster inpainting | yes (GAN) | elevation, slope, aspect, hillshade (no water) | none | no (connectivity only measured) | no | **yes** (Florence, Perugia → Rome, Siena) | figshare |
| Gu et al. 2024 (IJGI) | F | raster → vectorised | yes (conditional DDPM) | elevation, slope, land use; ground-truth intersections as input | none | no | no | no (one model per city, spatial split) | figshare |
| Zhou et al. 2024 (CAVW) | **A** | vector graph via density map | yes (image translation + transformer) | elevation, water, vegetation | **unknown** | **unknown** | unknown | unknown | none found |
| Yang, Zhu & Feng 2023/25 | F | raster → graph | yes (cGAN) | elevation, population, land use | none | no | dropout only | no | not found |
| GraphWalker (IJCAI 2025) | S | graph from walks | yes (latent diffusion) | none (GPS trajectories) | none | no | — | yes | GitHub |
| RoadWeaver (arXiv 2026) | S | vector graph | yes (VQ-VAE + masked transformer) | none | **post-hoc A\* reconnection** of dangling ends; keeps the largest component | post hoc only | — | — | not checked |
| DRoLaS, CaRoLS (2025) | **N** (metadata only) | ? | diffusion | ? | ? | ? | ? | ? | ? |
| Parish & Müller 2001 | A + secondary | vector | procedural (L-system) | water, density | local rules | rule-based | n/a | n/a | n/a |
| Chen et al. 2008 (TOG) | F | vector (tensor field) | procedural | water, park, height, density | **look-ahead to find intersections, reducing dangling roads** | rule-based | via noise | n/a | — |
| Galin et al. 2010 / 2011 (CGF) | F (skimmed) | single road / hierarchical network | procedural optimisation | slope, water, vegetation | anisotropic shortest path / path merging | connected by construction | parameters | n/a | n/a |
| Anagnostidis et al. 2023 (ICCV) | F (arXiv v1) | vector graph (extraction) | yes (autoregressive transformer) | none (satellite image) | **MCTS / MuZero look-ahead** | **yes: topology rewards vs. ground truth** | — | no | none |
| **UrbanMorph M13** | — | vector graph | yes (TerrainGraphTurtle) | elevation, water (+ low-pass / street-erased controls) | hard feasibility, density-ratio potentials, K-candidate, SMC, roll-out look-ahead | no | yes | yes (GEN) | this repo |
| **UrbanMorph M14** | — | same | same (frozen) | same | + telescoping network potentials; validated adaptive roll-outs; KL trust region | **yes: closed dead ends and islands, moment-matched to real** | yes | yes (GEN, GEN2) | this repo |

## 2. Comparison matrix: inference and planning foundations

| Topic | Works (access) | What we take from them |
|---|---|---|
| Proper weighting, nested proposals | Naesseth et al. 2015 (F), Def. 1 §3.1, §3.2 | Our K-candidate SMC weight (mean candidate weight) is properly weighted. The definition has no free constant. |
| Look-ahead SMC | Lin, Chen & Liu 2013 (F), §5, Prop. 7 | Look-ahead depth can be chosen adaptively (heuristic criterion). Extra look-ahead can *hurt* at fixed sample size. |
| Twisted / controlled SMC | Guarniero et al. 2017, Prop. 2 (F); Heng et al. 2020, Prop. 1 (F); Whiteley & Lee 2014 (F); Zhao et al. 2024 (F) | Optimal twist = conditional expectation of future potentials. Any positive twist leaves the target unchanged when corrected. The tilted target is a locally normalised policy *with* the twist (Zhao, Prop. 3.2). |
| SMC steering | Lew et al. 2023 (F; ICML 2023 workshop + arXiv); Loula et al. 2025 (M13, F) | Constrained generation as inference. |
| Local vs. global normalisation | Lafferty 2001, Andor 2016 (F) | Analogy only. Their results concern conditional models (Andor, footnote 6). |
| I-projection / KL-regularised optimum | Ganchev et al. 2010, Prop. 2.1 (F); Csiszár 1975 (**N**); Levine 2018 (F); Korbak et al. 2022 (F); Todorov 2006 (F) | Minimum-KL distribution under expectation constraints is an exponential tilt. The dual gradient is the moment mismatch (justifies the calibration). |
| KL trust region | Peters et al. 2010 REPS (F); Schulman et al. 2015 TRPO (F) | REPS bounds KL(new ‖ old) ≤ ε, the closest precedent to our per-decision bound. |
| Planning at decoding time | PPL-MCTS (S), Leblond et al. 2021 (S), Lu et al. 2022 (F), Williams et al. 2017 MPPI (F) | Value-guided / roll-out decoding is established. We do not claim it. |
| Value of computation | Hay et al. 2012 (S); Russell & Wefald 1991 (bibliographic records only) | Principle for allocating computation. Our trigger rule is our own construction and is validated empirically. |
| Few-cluster inference | Cameron, Gelbach & Miller 2008 (F); Canay, Romano & Shaikh (F, 2014 working-paper version) | Few-cluster bootstraps over-reject. Sign-change randomisation is valid under approximate symmetry. |
| Self-normalised IS bias | Owen 2013, §2.7 eqs. 2.32–2.33, §9.2 (F) | O(1/K) bias of K-candidate selection. |

## 3. What M14 can claim (and what it cannot)

| Candidate claim | Closest prior art | Status |
|---|---|---|
| Terrain-aware neural road generator | Fang 2022, Gu 2024, Zhou 2024 | **Not claimed.** |
| Search or look-ahead for road graphs | Anagnostidis 2023, Chen 2008, Galin 2010 | **Not claimed.** |
| Connectivity repair of generated networks | RoadWeaver 2026, Birsak 2022, M13 repair | **Not claimed.** Used as a baseline. |
| SMC / twisted SMC / KL-regularised inference | Zhao 2024, Lew 2023, Guarniero 2017, Levine 2018 | **Not claimed.** Applied. |
| **Inference-time, network-level objectives (closed dead ends, isolated components) on a learned terrain-conditioned vector road generator: telescoping potentials, moment-matched to real statistics by I-projection, with the induced distribution of each sampler characterised (exact vs. heuristic)** | None found among works read. Anagnostidis optimises topology against ground truth for extraction, not design. RoadWeaver repairs post hoc. Zhou 2024 is unread. | **Candidate contribution.** Conditional on reading Zhou 2024 and DRoLaS / CaRoLS in full and on a library database search (Scopus / Web of Science, not available to us). |
| **Adaptive allocation of look-ahead computation by a context trigger validated against the measured value of computation and against random allocation at equal budget** | Lin 2013 §5 (adaptive depth, heuristic); Hay 2012 (VoC principle) | **Candidate contribution, small.** It is meaningful only if the experiments show the trigger beats random allocation. |
| Window-closed connectivity metrics that separate water-separated pairs from same-land failures | Fang 2022 and Gu 2024 measure connectivity and degree; NTG measures reach | **Candidate contribution (evaluation)**, modest. |
| Leakage-controlled, density-controlled, fresh-city generalisation protocol | M13 (ours) | Continuity with M13, not new. |

## 4. Consequences for the M14 design

1. **No novelty claim for the inference machinery itself.** Every sampler is an instance of established methods. The text says "we apply and characterise".
2. **The connectivity baseline is post-hoc repair** (RoadWeaver-style A\* reconnection = our M13 `repair`). It is included because it is the obvious simpler alternative.
3. **The adaptive trigger must beat random allocation.** Lin et al. (Prop. 7) and our toy (C4) both show more or better-placed look-ahead is not automatically better. This is pre-registered (protocol §4).
4. **Before any publication claim:**
   * obtain and read Zhou 2024 (KAIST library), DRoLaS and CaRoLS;
   * run a Scopus / Web of Science search;
   * read Csiszár 1975.

   Until then the novelty statement stays conditional.
5. **External learned baselines.**
   * Fang 2022 and Gu 2024 release code, but they are raster models that take ground-truth junctions as input (an information advantage our generator does not get), and Gu trains one model per city.
   * Reproducing them on our data would need their preprocessing plus retraining (several GPU-days for Gu's DDPM). This is outside the M14 compute budget and is documented as a limitation.
   * NTG has no released code. Our GraphTurtle is an NTG-family reimplementation, not a reproduction.
