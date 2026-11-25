# Milestone 14 failure analysis

Every number is from `outputs/m14/REPORT.md` (generated from stored runs) unless another file is named. Failures are grouped by kind:
1. Pre‑registered failure flags.
2. Scientific failures: hypotheses or claims that did not hold.
3. Behavioural failure modes of the generators.
4. Engineering incidents during the milestone.

## 1. Pre‑registered failure flags (protocol §9)

A method is flagged on a set if any of the following holds:
* premature share > A's + 0.10;
* density ratio outside [0.6, 1.6];
* any hard‑constraint violation.

| Set | Method | Flag | Value |
|---|---|---|---|
| GEN2 | M13‑SMC | premature | 0.123 vs A 0.018 |
| GEN2 | M\*+repair | hard constraint | 0.035 unbridged river crossings / tile. The A\* repair adds straight links across rivers. |
| GEN2 | M\* | hard constraint | 1 unbridged crossing in 285 tile‑samples (TN_s2, seed 2, `ESP_ALY_e1405_n8371`) |
| GEN2 | NA‑random | hard constraint | 1 unbridged crossing in 285 tile‑samples (TN_s1, seed 3, `ESP_TGN_e705_n8896`) |
| DEV‑B | M\*+repair | hard constraint | 0.008 unbridged crossings / tile (repair links) |
| GEN | M\*+repair | hard constraint | 0.017 unbridged crossings / tile (repair links) |
| GEN | NA‑adaptive | hard constraint | 1 unbridged crossing in 300 tile‑samples (TN_s2, seed 2, `ESP_GRO_e945_n9074`) |
| DEV‑B@rough | M\* | premature, density | premature 0.41 vs A 0.07; density 0.56 |
| DEV‑B@rough | NA‑adaptive | premature | 0.26 vs A 0.07 |

No method is flagged for premature termination or density on DEV‑B or GEN.

**Hard‑constraint residuals.**
* **Which runs.** The three single crossings (GEN2 M\*, GEN2 NA‑random, GEN NA‑adaptive) come from samplers that enforce the M13 hard constraint at every step.
* **Diagnosis (partial).** For the M\* case, the offending crossing is in the narrowest river‑width bin. On the same graph, the M12 water‑length metric measures **0 m** of unbridged water; the same holds for the GEN NA‑adaptive case. Both the hard constraint and the M13 metric detect water by sampling points along the edge at a fixed step (`tc_potentials` edge features). The likely cause is therefore sampling aliasing on a single narrow river cell: the committed edge and the final edge are sampled at different points, for example after a later split at a T‑junction. This is consistent with the evidence but not fully verified.
* **Status.** It is rare (1 crossing in 285–300 tile‑samples for each of the three runs; 0 in every other constrained run on every set) and is listed as an open issue.

## 2. Scientific failures

### 2.1 Adaptive planning did not work (H3, and the pilot validation)

The milestone's second candidate contribution, look‑ahead allocated by a VoC‑selected trigger, is **not supported**. The evidence:

* **DEV‑A pilot** (pre‑registered validation, Amendment 2): NA‑adaptive vs NA‑random at the same firing rate, `od_unreach_land` 0.102 vs 0.095. The pilot failed.
* **GEN2 confirmatory**, NA‑adaptive − NA‑fixed:
  * OD +0.0017 [−0.0020, +0.0045], non‑inferior;
  * S10 +0.116 [+0.050, +0.189], fails the +0.02 margin;
  * cost 5.35 vs 9.75 s, which passes.

  H3 is therefore **not met**.
* **GEN2 against random allocation.** NA‑adaptive − NA‑random: +0.0008 [−0.0030, +0.0038], p = 0.66, **not supported**.
* **DEV‑B and GEN (secondary)** give the same pattern:
  * vs fixed: S10 +0.094 (DEV‑B) and +0.085 (GEN), both failing the margin; OD non‑inferior on both;
  * vs random: p = 0.49 (DEV‑B) and 0.81 (GEN).
* **Ablation (DEV‑A, DEV‑B).** Cheaper triggers (`water` at 1 % of decisions, `island` at 16 %) matched or beat `deg1` on connectivity at 2–6× lower cost. This was not tested formally (`M14_ABLATIONS.md` §5).

**Why (interpretation, not tested).**
* The trigger was chosen to capture value of computation, defined as the total‑variation change of the next selection caused by the roll‑out twist.
* That proxy is local: many decisions where the twist changes the selection lead to near‑equivalent networks. The final connectivity is mostly decided by the stop weight, which the myopic sampler already has.
* Toy C4 had already shown that a context trigger can fall well short of the oracle allocation: KL 0.77 vs oracle 0.27 vs random 0.62. The pre‑registration made this risk explicit.

### 2.2 Planning strength vs realism (H2): direction as predicted, magnitude below the noise floor

* **Direction.** On DEV‑B, S10 rises from 0.411 (m = 0) to 0.732 (m = 4), non‑decreasing in 3 of 4 steps (it dips to 0.351 at m = 0.5).
* **Magnitude.** The rise (+0.320) is below the pre‑registered DEV‑B real‑vs‑real median (0.422).
* **Verdict.** H2 is **not supported** under the pre‑registered rule.
* **Caveat.** The noise floor is computed from half‑size real sets and is therefore conservative (an upper bound for full‑size sets). We do not reinterpret it after the fact.

### 2.3 Over‑connection relative to real streets

The primary outcome is pre‑registered as "lower is better". Against real networks, the network‑aware methods **over‑connect** on GEN2:

| GEN2 | Real | M\* | NA‑fixed | NA‑SMC |
|---|---|---|---|---|
| OD unreach. (land) | 0.047 | 0.037 | 0.011 | 0.012 |
| Islands / tile | 0.13 | 0.28 | 0.07 | 0.02 |
| Dead‑end ratio | 0.023 | 0.054 | 0.019 | 0.014 |

* **Per city.** Real Tarragona has same‑land unreachability of 0.128 on non‑empty tiles. Every generator produces 0.000–0.028 there.
* **Plausible causes in the real data:**
  * separated carriageways and motorway corridors;
  * gated or private estates;
  * rail lines and ports;
  * missing OSM links.

  Those causes are not modelled. Part of the real disconnection is therefore desirable structure and part is data artefact; we cannot separate the two.
* **Not on every set.** On GEN, real same‑land unreachability is lower (0.003 on non‑empty tiles), and the NA methods move *towards* it (0.002–0.007) from M\* (0.023). DEV‑B is similar: real 0.011, M\* 0.028, NA‑fixed 0.007. Over‑connection relative to real is therefore mainly a GEN2 (Tarragona) phenomenon.
* **Consequence.** The H1 effect is a connectivity improvement, **not** a move toward real connectivity on every axis.
  * Dead‑end ratio: the calibrated NA methods move to the real level (NA‑fixed 0.019 vs real 0.023).
  * OD unreachability: they move past it.
  * NA‑SMC goes below real on dead ends too, because the calibration matched the moment under `na_fixed`, not under SMC (§3.3).

### 2.4 Density control did not transfer from DEV‑A to GEN2

* **What was done.** Protocol §7(1) calibrates a stop‑logit bias per method on DEV‑A (density 1 ± 0.05) and re‑uses it on GEN2.
* **What happened.** On GEN2 the matched densities were 1.01 (A), 1.22 (M\*), 1.22 (NA‑myopic) and 1.30 (NA‑SMC). For NA‑SMC, the matched run is *denser* than the unmatched one (1.11).
* **Conclusion.** The bias is set‑specific. "Density‑matched" in this milestone means matched on DEV‑A only.
* **The connectivity conclusion is unaffected.** It agrees in sign under all three analyses (`M14_STATISTICAL_ANALYSIS.md` §5).

### 2.5 λ does not transfer exactly across training seeds

λ was moment‑matched with backbone TN_s0. Applied unchanged to TN_s1 and TN_s2 on the same training tiles:

| Backbone | Dead ends / tile | Target |
|---|---|---|
| TN_s1 | 2.89 | 2.27 |
| TN_s2 | 2.71 | 2.27 |
| TN_s0 (calibration) | 2.15 | 2.27 |

* **Size of the miss.** It is 19–27 %. Islands are within the target range: 0.094 and 0.052 vs 0.104.
* **Interpretation.** A deployed method would need per‑backbone calibration (about 1.5 h on CPU).
* **Effect on the results.** The confirmatory effects hold for all three backbones, but the "moment‑matched" description is exact only for TN_s0.

## 3. Behavioural failure modes

### 3.1 Density inflation of the single‑chain network samplers

The myopic stop weight discourages STOP at interior road ends. With one chain and no correction, the generator keeps building.

| Method | GEN2 density / real | DEV‑B density / real | GEN density / real |
|---|---|---|---|
| NA‑myopic | 1.34 | 1.33 | 1.23 |
| NA‑adaptive | 1.32 | 1.27 | 1.16 |
| NA‑random | 1.29 | 1.27 | 1.16 |
| NA‑fixed | 1.21 | 1.22 | 1.06 |
| NA‑SMC | 1.11 | 1.12 | 1.05 |
| M\* | 1.12 | 1.14 | 1.04 |

All values are within the pre‑registered [0.6, 1.6] band, but the H1 density guard (|density − 1| ≤ 0.25) would fail for the single‑chain variants:
* **Roll‑outs** let the sampler see that extending a road end also costs future dead ends.
* **SMC** re‑weights whole partial networks.
* **Both** remove most of the inflation.

The density‑adjusted contrasts (`M14_STATISTICAL_ANALYSIS.md` §5) show the connectivity gain is not *explained* by the extra density.

### 3.2 City‑level density bias (all generators)

* **Tarragona.** Every generator is 1.4–1.7× the real density, including A (1.56). NA‑myopic reaches 1.72, above the failure band at city level.
* **Alcoi and Sacramento.** Generators are within 0.8–1.2.
* **GEN.** Manresa shows the same pattern in milder form: A 1.21, NA‑myopic 1.44. Girona and San Francisco are within 0.98–1.20.
* **Cause.** This is the M13 density‑adaptation deficit (research plan, audit point 3). Density depends on land use and history, which the model does not receive. M14 did not target it.

### 3.3 The calibrated λ is sampler‑specific

λ was fitted with the `na_fixed` sampler. The formulation (§4) shows that the single chain, the K‑candidate selection and SMC target different distributions, so the same λ produces different moments under different samplers:
* **NA‑SMC** reaches dead‑end ratio 0.014 vs real 0.023 on GEN2.
* **NA‑fixed** reaches 0.019.

Moment‑matching *under the sampler that will be used* would be the consistent procedure.

### 3.4 Empty tiles

* **GEN2.** All methods have 0.4–0.8 % empty tile‑samples. NA‑SMC and M13‑SMC have twice the rate of A. Their empty samples record committed decisions, so the chosen particle stopped early, which is consistent with SMC resampling towards particles that stopped early.
* **GEN.** 1.4–1.6 % of tile‑samples are empty for **every** method. Almost all are on five tiles: one San Francisco tile with 23 % sea (54 of 81 non‑P tile‑samples empty) and four flat Girona tiles (2–23 of 81). The real network on `ESP_GRO_e946_n9075` is dense (25.7 km/km²). These samples record no decision at all: the frozen M12 start head proposed no start, so no inference‑time method can act. This is a backbone failure, independent of M14. Because an empty tile counts as `od_unreach_land` = 1, it contributes about 0.015 of the GEN NA methods' 0.019–0.022; on non‑empty tiles they reach 0.002–0.007 vs real 0.003.
* **DEV‑A.** The same 2 of 25 tiles are empty for every method, including P. Because an empty tile counts as `od_unreach_land` = 1, these tiles dominate the DEV‑A primary outcome. The non‑empty variant was added as a secondary outcome before confirmation (Amendment 2).

### 3.5 Synthetic rough terrain

`devB@rough` adds synthetic high‑frequency relief; it is a robustness probe, not real geography.
* **M\*** collapses: premature 0.41, density 0.56, OD 0.265.
* **NA‑adaptive** degrades: OD 0.080, premature 0.26.
* **NA‑myopic** stays robust: OD 0.017, density 1.05.

The terrain hard constraints reject many steps on rough terrain, and the M13 energy sampler then stops early. The network stop weight counteracts the early stopping, but the roll‑outs of NA‑adaptive, which simulate under the same constraints, see mostly infeasible futures. This is an interpretation; it was not tested.

### 3.6 Visual failure cases

`outputs/m14/figures/fig_gallery_{gen2,devB,gen}.png`: the last row of each figure is the tile where NA‑adaptive has the highest same‑land unreachability in the shown sample (not cherry‑picked, rule fixed in code). On GEN, the last row is `ESP_GRO_e946_n9075`, the start‑head failure of §3.4: empty for every generator. The GEN2 case (`ESP_ALY_e1404_n8370`, Alcoi) is a steep valley side:
* NA‑adaptive leaves an isolated fragment on the slope (OD 0.36).
* The real network there is a sparse set of switchbacks.

## 4. Engineering incidents

### 4.1 GEN results were contaminated by GEN2 tiles (found and fixed before reporting)

* **Cause.** `scripts/m14_build_gen2.py` re‑pointed `m13_build_gen_cities.GEN` to the GEN2 directory *at import time*. The experiment runner imports it lazily to load GEN2. Any worker that had loaded a GEN2 run before a GEN run then read GEN2 tiles under the label `gen`, and `outputs/m14/real/gen.json` was built the same way.
* **Detection.** The GEN and GEN2 panels of `fig_tradeoff.png` were identical.
* **Fix:**
  * The patch now runs only inside the build script's `main()`.
  * The 28 contaminated runs and `real/gen.json` were moved, not deleted, to `outputs/m14/invalid_gen_contaminated/` (with a README).
  * GEN was re‑run with the same specs.
* **Scope check.** No other tile set goes through that code path: DEV‑A, DEV‑B and `train_cal` use other loaders, and GEN2 reads its directory directly. All GEN numbers in the M14 documents come from the re‑run, which was checked to contain 28 runs × 3 seeds on the 100 GEN tiles (Girona 30, Manresa 30, San Francisco 40).

### 4.2 Missing street‑erased DEMs for GEN2

* **Problem.** The TNse runs of H5 need street‑erased DEMs, which the GEN2 build did not produce. One queue worker crashed (`FileNotFoundError`).
* **Fix.** The DEMs were built with the unchanged M13 procedure (`scripts/m14_street_erase_gen2.py`) and the worker was restarted. No results were affected.

### 4.3 NA‑SMC missing from the first density‑matched plan

The plan generator used the fast method set. The 3 missing runs were added (`configs/m14_plans/dm_smc.json`) and the generator was fixed.

### 4.4 Duplicate configuration resolved to the smaller run

* **Bug.** The frozen NA‑adaptive configuration on DEV‑B with TN_s0 exists twice: as the 3‑seed confirm run and as the 2‑seed ablation run "trigger = `deg1`". The run lookup (`m14_analyze.find`) returned the last file by name, which was the 2‑seed run. The 3‑seed completeness check then dropped it.
* **Effect.** The draft report's DEV‑B NA‑adaptive row and DEV‑B H3 tests used TN_s1 and TN_s2 only. With all three backbones, DEV‑B NA‑adaptive OD is 0.003 and H3‑vs‑random p is 0.49 (draft: 0.74); the verdicts did not change.
* **Fix.** The lookup now prefers the run with the most seeds. This was the only duplicated configuration among the stored runs (checked by script), and all reported numbers come from the corrected lookup.

### 4.5 Gallery failure‑row selection

The first version picked the failure tile only among tiles not already chosen for the terrain rows. On DEV‑B this produced a "failure" row with OD 0. The rule now picks the worst tile first.

### 4.6 Ablation rows mixed 2‑ and 3‑seed runs

* **Bug.** On DEV‑B, five ablation configurations (frozen NA‑adaptive, λ × 0 = M\*, λ × 1 = NA‑myopic, random trigger, NA‑fixed) are also confirm runs with 3 seeds. The ablation table, H2 and the dose figure used those runs with all 3 seeds, while every other variant used seeds 1–2. The protocol specifies seeds 1–2 for H2.
* **Effect.** H2's S10 at m = 0 was 0.473 (3 seeds) instead of 0.411 (seeds 1–2), so the reported rise was +0.258 instead of +0.320. The verdict (below the 0.422 floor) is unchanged. Other DEV‑B ablation rows moved by ≤ 0.002 in OD and ≤ 0.04 in S10.
* **Fix.** Runs used for ablations, H2, shifts, H5, density matching and the dose figure are cut to seeds 1–2. Confirm runs and their 2‑seed duplicates are identical row for row on the shared seeds (checked), so the cut is the same sample the ablation run would have produced.
