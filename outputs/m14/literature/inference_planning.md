# Literature verification: inference-time control, SMC, tilting, planning, few-cluster inference

Verified on 2026-11-17 with WebSearch and WebFetch, plus PDF text extraction (`pdftotext`) of the full texts where they were accessible.

**Access levels**
- **FULL**: I read the full text myself and grepped or read the relevant sections.
- **META**: I checked the bibliographic record (publisher, ACL Anthology BibTeX, PMLR, DBLP or the conference page) and the abstract only.
- **N/A**: the text could not be accessed.

"Not verified" means exactly that. Nothing below is filled in from memory without a label.

**Verdicts:** yes / partly / no / not verified.

---

## Summary table

| # | Work | Venue verified | Access | Claim supported |
|---|------|---------------|--------|-----------------|
| 1 | Naesseth, Lindsten & Schön 2015, NSMC | ICML 2015, PMLR 37 | FULL | **partly** (the definition has no free constant *c*; see notes) |
| 2 | Lin, Chen & Liu 2013, Lookahead SMC | Stat. Sci. 28(1) | FULL (arXiv = journal version) | **yes**, including adaptive lookahead (Sec. 5) |
| 3a | Whiteley & Lee 2014, Twisted PF | Ann. Statist. 42(1) | FULL | **partly** (optimal twist is an eigenfunction in an asymptotic-in-time setting) |
| 3b | Guarniero, Johansen & Lee 2017, iAPF | JASA 112(520) | FULL (arXiv v2) | **yes** (Prop. 1 invariance; Prop. 2 optimal ψ* = conditional expectation of future potentials) |
| 3c | Heng, Bishop, Deligiannidis & Doucet 2020, cSMC | Ann. Statist. 48(5) | FULL (arXiv v3) | **yes** (eq. 16 invariance at T; Prop. 1 optimal policy) |
| 4 | Zhao, Brekelmans, Makhzani & Grosse 2024 | ICML 2024, PMLR 235 | FULL (arXiv v1) | **yes** (Prop. 3.2) |
| 5 | Lew, Zhi-Xuan, Grand & Mansinghka 2023 | **Workshop only**: ICML 2023 Workshop "Sampling and Optimization in Discrete Space" (oral); arXiv 2306.03081 | FULL | venue confirmed as a workshop, not the main conference |
| 6a | Lafferty, McCallum & Pereira 2001, CRF | ICML 2001, pp. 282–289 | FULL | **yes** for label bias (Sec. 2) |
| 6b | Andor et al. 2016 | ACL 2016, pp. 2442–2452 | FULL | **partly**: Thm 3.1 covers *conditional* models; footnote 6 says joint models differ (see notes) |
| 7a | Csiszár 1975 | Ann. Probab. 3(1):146–158 | META only | **not verified** from the 1975 text; the exponential-form result is confirmed via Csiszár's own lecture note and Ganchev et al. |
| 7b | Ganchev et al. 2010, PR | JMLR 11:2001–2049 | FULL | **yes** (Prop. 2.1, eq. 5–6) |
| 7c | Levine 2018 tutorial | arXiv 1805.00909 | FULL | **yes** (eq. 4–5; Sec. 2.3 backward messages) |
| 7d | Korbak, Perez & Buckley 2022 | Findings of EMNLP 2022 | FULL | **yes** (eq. 5) |
| 7e | Todorov 2006/2007 (LMDP, NIPS 19); Todorov 2009 (PNAS) | NIPS 19, pp. 1369–1376; PNAS 106(28) | FULL (NIPS) / META (PNAS) | **yes** (NIPS eq. 14) |
| 8a | Schulman et al. 2015, TRPO | ICML 2015, PMLR 37:1889–1897 | FULL | **yes** (eq. 12–13; KL(old‖new)) |
| 8b | Peters, Mülling & Altün 2010, REPS | AAAI-10, pp. 1607–1612 | FULL | **yes** (eq. 4; closed form π ∝ q·exp(δ/η)) |
| 9a | Chaffin, Claveau & Kijak 2022, PPL-MCTS | NAACL 2022, pp. 2953–2967 | FULL (skimmed) | confirmed |
| 9b | Leblond et al. 2021 | EMNLP 2021, pp. 8410–8434 | FULL (skimmed) | confirmed |
| 9c | Browne et al. 2012, MCTS survey | IEEE TCIAIG 4(1):1–43 | META | confirmed (bibliography only) |
| 9d | Williams et al. 2017, IT-MPC | ICRA 2017, pp. 1714–1721 | FULL | confirmed (eq. 14); **author list has 7 names** |
| 10a | Russell & Wefald 1991 (AIJ; MIT Press book) | AIJ 49(1–3):361–395; MIT Press 1991 | META | bibliography confirmed; VOC concept confirmed only via secondary sources |
| 10b | Hay, Russell, Tolpin & Shimony 2012 | UAI 2012 (arXiv 1207.5879) | FULL (skimmed) | confirmed |
| 10c | Graves 2016, ACT | arXiv 1603.08983 (preprint only) | META | confirmed |
| 11 | FUDGE; Grid BS; Constrained BS; NeuroLogic A*esque; GNMT | see below | FULL (FUDGE, GNMT, NL-A*) / META (others) | confirmed |
| 12 | GraphRNN; GRAN; DiGress; ConStruct; PRODIGY | ICML'18; NeurIPS'19; ICLR'23; NeurIPS'24; ICML'24 | META | confirmed; **PRODIGY's ICML title is "Diffuse, Sample, Project…"** |
| 13 | Friedman & Dieng, Vendi Score | TMLR 07/2023 | FULL (skimmed) | confirmed |
| 14a | Cameron, Gelbach & Miller 2008 | REStat 90(3):414–427 | FULL | **yes** (Table 2; Sec. VII) |
| 14b | Canay, Romano & Shaikh 2017 | Econometrica 85(3):1013–1030 | FULL (2014 working paper) / META (journal) | **yes** |
| 14c | Holm 1979 | Scand. J. Statist. 6(2):65–70 | META | confirmed |
| 15 | Owen 2013, *Monte Carlo theory, methods and examples* | online book | FULL (Ch. 2, 9) | **yes**: the O(1/n) bias is in **§2.7 eq. (2.32)–(2.33)**, not Ch. 9 |
| 16a | Douc, Cappé & Moulines 2005 | ISPA 2005, pp. 64–69 | FULL | confirmed; **systematic resampling is not guaranteed to beat multinomial** |
| 16b | Doucet & Johansen 2009/2011 | Oxford Handbook of Nonlinear Filtering (OUP 2011) | FULL | confirmed (Sec. 3.4) |
| 17a | Koh et al. 2021, WILDS | ICML 2021, PMLR 139:5637–5664 | FULL (skimmed) | confirmed (FMoW/PovertyMap = geographic shift) |
| 17b | Gulrajani & Lopez-Paz 2021 | ICLR 2021 | META | confirmed |

---

## 1. Nested SMC / proper weighting

**Citation.** Christian A. Naesseth, Fredrik Lindsten, Thomas B. Schön. "Nested Sequential Monte Carlo Methods." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:1292–1301, 2015. Editors: F. Bach and D. Blei. Links: https://proceedings.mlr.press/v37/naesseth15.html. Extended version: arXiv:1502.02536 (v3, Sep 2015).

**Access.** FULL. I read both the ICML PDF and the arXiv v3. Definition 1 and Section 3 are numbered the same in both.

**What the paper says.**
- **Sec. 3 ("Proper Weighting and Nested Importance Sampling"), Sec. 3.1, Definition 1 (Properly weighted sample), quoted verbatim:** "A (random) pair (X, W) is properly weighted for an unnormalised distribution p if W ≥ 0 and E[f(X)W] = p(f) := ∫ f(x)p(x)dx for all measurable functions f."
  - The paper then notes that taking f ≡ 1 shows the average weight is an unbiased estimate of Z_p.
- **Sec. 3.1, assumption (A1) and Algorithm 1 (Nested IS).** If Ẑ_q is a non-negative unbiased estimate of the proposal's normaliser, and (X, Ẑ_q) is properly weighted for q, then W = Ẑ_q π(X)/q(X) gives a valid IS. This is proved with an extended-target argument and is called an "exact approximation".
- **Sec. 3.2 (Modularity).** The paper runs IS with N samples, sets Ẑ_π = mean of the weights, and then draws X^B with P(B = i) ∝ W^i. The result is E[f(X^B) Ẑ_π] = π̄(f) Z_π, so (X^B, Ẑ_π) is properly weighted.
  - **This is exactly the "sample K candidates, select ∝ weight, carry weight = mean candidate weight" construction.**
- **Sec. 4.1, Algorithm 2 (fully adapted SMC).** The resampling weight at step k is the local normaliser Z_{q_k}(x_{1:k−1}) = ∫ π_k(x_{1:k})/π_{k−1}(x_{1:k−1}) dx_k.
- **Sec. 4.2, Algorithm 3 (nested SMC).** This replaces Z_{q_k} with an unbiased estimate Ẑ_{q_k} from an inner sampler.

**Verdict: partly / essentially yes.**
- **What does not match:** the claim's exact form "E[w f(x)] = c ∫ f γ" with an arbitrary constant c is **not** Naesseth et al.'s Definition 1, which has no free c (it is stated for the unnormalised p).
  - A version with an arbitrary constant is often attributed to Liu (2001), *Monte Carlo Strategies in Scientific Computing*. I did **not** verify that source.
  - Either quote Definition 1 verbatim, or say "up to a constant (cf. Liu 2001)" and mark the Liu reference as needing a check.
- **What does match:** the second half of the claim ("nested/approximate proposals stay valid when the weight equals an unbiased estimate of the local normaliser") is supported by Sec. 3.1–3.2 and Algorithm 3.

## 2. Lookahead strategies for SMC

**Citation.** Ming Lin, Rong Chen, Jun S. Liu. "Lookahead Strategies for Sequential Monte Carlo." *Statistical Science* 28(1):69–94, 2013. DOI: 10.1214/12-STS401. arXiv:1302.5206. The arXiv PDF carries the journal typesetting.

**Access.** FULL.

**What the paper says.**
- **Sec. 3 (Principle of lookahead).**
  - The ∆-step lookahead target π*_t(x_t) = π_{t+∆}(x_t) (eq. 3).
  - Proposition 1: the error term II(∆) decreases in ∆.
  - Future information can be used in three ways: (i) the proposal, (ii) the weights, (iii) the resampling priority scores.
- **Sec. 4** covers:
  - 4.1 Lookahead weighting.
  - 4.2 Exact lookahead sampling (the "delayed-sample" method of Chen, Wang & Liu 2000).
  - 4.3 Block sampling.
  - 4.4 Pilot lookahead sampling, i.e. Monte Carlo roll-out "pilots".
  - 4.5 Deterministic piloting.
  - 4.6 Multilevel pilot.
  - 4.7 Resampling with lookahead and piloting.
  - 4.8 Combined methods.
- **Sec. 5 (ADAPTIVE LOOKAHEAD).**
  - "Many systems have structures with different local complexity. In these systems, it may be beneficial to have different lookahead schemes based on local [information]."
  - Proposition 7 shows that with a fixed Monte Carlo size, one extra lookahead step can make the estimator worse.
  - The authors then give a heuristic criterion (Fig. 4) for choosing the lookahead step adaptively.
  - Their tables report the "average lookahead steps in the adaptive lookahead method".

**Verdict: yes.**
- Look-ahead weighting, sampling and resampling are all covered.
- On "optimal lookahead": more lookahead is always better asymptotically (Prop. 1), but not at a fixed sample size (Prop. 7).
- They **do** discuss adaptively choosing the depth (Sec. 5, a heuristic criterion). This is the closest SMC precedent for "look-ahead only when a context trigger fires".

## 3. Twisted / controlled SMC

### 3a. Whiteley & Lee 2014

**Citation.** Nick Whiteley, Anthony Lee. "Twisted particle filters." *Annals of Statistics* 42(1):115–141, 2014. DOI: 10.1214/13-AOS1167. arXiv:1210.0220.

**Access.** FULL.

**What the paper says.**
- **Sec. 1, eq. (1.5)–(1.6).** The alternative sampling laws keep the "lack of bias" of the Z_n estimate when the weights are corrected by the functionals φ_p.
- **Theorem 1 / eq. (1.12).** The optimal (zero variance-growth-rate) particle transition reweights M by a function h^ω that is a *generalized eigenfunction*.
  - For the bootstrap PF, h^ω is interpreted as the limit of a Radon–Nikodym derivative after further conditioning on future observations.
- **Sec. 3 (Twisted particle algorithms), eq. (3.1).** This defines the twisted kernel M̃(x, dx′) ∝ M(x, dx′) ψ̄(x′) for any positive bounded ψ.

**Verdict: partly.**
- **Supported:** the invariance part (twist plus weight correction leaves the estimate unbiased).
- **Different framing:** the "optimal twist = conditional expectation of future potentials" is stated here in an asymptotic-in-time, eigenfunction framing, not as a finite-horizon conditional expectation.
- For the finite-horizon statement, cite 3b, 3c or 4.

### 3b. Guarniero, Johansen & Lee 2017

**Citation.** Pieralberto Guarniero, Adam M. Johansen, Anthony Lee. "The iterated auxiliary particle filter." *Journal of the American Statistical Association* 112(520):1636–1647, 2017. DOI: 10.1080/01621459.2016.1222291. The DOI and pages were confirmed by bibliographic search (Bristol research portal). arXiv:1511.06286.

**Access.** FULL (arXiv v2).

**What the paper says.**
- **Sec. 2, Proposition 1 (invariance).** For any bounded, continuous, positive ψ, the twisted model (eq. 5–6) has Z_ψ = L, i.e. the same marginal likelihood.
- **Proposition 2 (optimal twist).** ψ*_t(x_t) = g(x_t, y_t) · E[∏_{p=t+1}^{T} g(X_p, y_p) | X_t = x_t].
  - That is, the optimal twist is the conditional expectation of the future potentials (the backward information filter).
  - It gives Ẑ = L with probability 1.

**Verdict: yes.** Both parts of the claim are in Sec. 2.

### 3c. Heng, Bishop, Deligiannidis & Doucet 2020

**Citation.** Jeremy Heng, Adrian N. Bishop, George Deligiannidis, Arnaud Doucet. "Controlled sequential Monte Carlo." *Annals of Statistics* 48(5):2904–2929, 2020. DOI: 10.1214/19-AOS1914. The DOI and pages were confirmed via the Project Euclid record found in search. arXiv:1708.08396.

**Access.** FULL (arXiv v3, Oct 2019).

**What the paper says.**
- **Sec. 3 (twisted Feynman–Kac models).**
  - Eq. (9)–(10): the backward information filter ψ*_t(x_t) = P(y_{t:T} | x_t), with the recursion ψ*_t = g_t · f_{t+1}(ψ*_{t+1}).
  - Eq. (15)–(16): the twisted intermediate marginals generally differ from the original ones, but "they coincide at time T", so Z^ψ_T = Z.
- **Sec. 4.1, Proposition 1.** The optimal policy φ* from the backward recursion (19) makes the twisted path measure equal to the target, and makes Ẑ = Z almost surely.
  - The paper also states that φ* solves a KL optimal-control problem (Supplement, Sec. E).

**Verdict: yes.**

## 4. Twisted SMC for language models

**Citation.** Stephen Zhao, Rob Brekelmans, Alireza Makhzani, Roger Grosse. "Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo." *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:60704–60748, 2024. https://proceedings.mlr.press/v235/zhao24c.html. arXiv:2404.17546. The ICML virtual site lists it as an oral.

**Access.** FULL (arXiv v1).

**Key pointer.** **Prop. 3.2 (Optimal Twists)**, eq. (11)–(13):
- ψ*_t(s_{1:t}) ∝ Σ_{s_{t+1:T}} p_0(s_{t+1:T} | s_{1:t}) φ(s_{1:T}), i.e. the expected terminal potential under the base model.
- It satisfies the recursion ψ*_t ∝ Σ_{s_{t+1}} p_0(s_{t+1} | s_{1:t}) ψ*_{t+1}.
- It recovers the target marginals σ(s_{1:t}).

The paper also notes that greedy decoding with respect to the optimal twists does not sample from the target (Sec. 6).

**Verdict: confirmed.**

## 5. SMC steering of LLMs

**Citation.** Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka. "Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs." arXiv:2306.03081 (v1 5 Jun 2023; v2 26 Nov 2023).

**Venue.** It was an **oral at the ICML 2023 Workshop "Sampling and Optimization in Discrete Space"** (https://icml.cc/virtual/2023/29326). The PDF calls itself "this workshop abstract".
- The arXiv record has no journal-ref.
- I found **no main-conference publication**.
- Cite it as an arXiv preprint or ICML 2023 workshop paper.
- Note: the second author's name is "Tan Zhi-Xuan".

**Access.** FULL.

## 6. Label bias / local vs. global normalisation

### 6a. Lafferty, McCallum & Pereira 2001

**Citation.** John Lafferty, Andrew McCallum, Fernando C. N. Pereira. "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data." *Proceedings of the 18th International Conference on Machine Learning (ICML 2001)*, pp. 282–289.

**Access.** FULL (a course-hosted copy of the paper).

**What the paper says.**
- **Sec. 2 ("The Label Bias Problem").** In per-state normalised models (MEMMs), "states with a single outgoing transition effectively ignore their observations".
- The proper solution "require[s] models that account for whole state sequences at once … score mass will not be conserved".
- Sec. 5.1 gives the experiment.

**Verdict: yes** for the label-bias concept.

### 6b. Andor et al. 2016

**Citation.** Daniel Andor, Chris Alberti, David Weiss, Aliaksei Severyn, Alessandro Presta, Kuzman Ganchev, Slav Petrov, Michael Collins. "Globally Normalized Transition-Based Neural Networks." *Proceedings of ACL 2016 (Vol. 1: Long Papers)*, pp. 2442–2452. DOI: 10.18653/v1/P16-1231.

**Access.** FULL.

**What the paper says.**
- **Sec. 3 ("The Label Bias Problem"), Theorem 3.1:** P_L ⊊ P_G.
  - Globally normalised models are strictly more expressive than locally normalised ones.
  - This holds for *conditional* models p(d_{1:n} | x_{1:n}) where the score at step i sees only x_{1:i}.
  - The original proof is by Smith & Johnson (2007).
- **Footnote 6:** "Equivalence (or non-equivalence) results for joint models … are quite different", e.g. weighted CFGs and PCFGs are equally expressive.

**Verdict: partly. Important caveat.**
- Our generator is a *joint* model over the action sequence (possibly conditioned on a fully observed context).
- By the chain rule, the tilted target π ∝ p_θ · 1[feasible] · exp(Σλφ) **is** representable by a locally normalised policy, namely the *twisted* conditionals π(x_t | x_{<t}) ∝ p_θ(x_t | x_{<t}) · ψ*_t.
- So "locally normalised models cannot represent" is **not** the right statement for us. The correct statement is:
  - the *myopic* local sampler q(x_t | x_{<t}) ∝ p_θ(x_t | x_{<t}) exp(λφ_t) (no future normaliser) generally ≠ π;
  - π(x) / q(x) ∝ ∏_t Z_t(x_{<t}), where Z_t is the local normaliser.
- That is exactly why the SMC weights equal the local normalisers (Naesseth Sec. 4.1) and why the optimal twist is needed (Zhao Prop. 3.2; Levine Sec. 2.3; Todorov eq. 14).
- **Recommendation:** use CRF/Andor only as *analogy* ("label-bias-style"), and cite the twisted-SMC and control-as-inference results for the actual discrepancy.

## 7. Exponential tilting / I-projection / KL-regularised optimum

### 7a. Csiszár 1975

**Citation.** I. Csiszár. "I-Divergence Geometry of Probability Distributions and Minimization Problems." *Annals of Probability* 3(1):146–158, 1975. DOI: 10.1214/aop/1176996454.

**Access.** META. The Project Euclid PDF and the mirrors I tried did not return the paper's text.

**Claim (min KL(q‖p) s.t. E_q[f] = c ⇒ q ∝ p·exp(λf)).** **Not verified from the 1975 text**, and no theorem number was checked.
- **Supporting evidence I did read:**
  - Csiszár's own lecture note, "Iterative Algorithms with an Information Geometry Background" (https://static.renyi.hu/~csiszar/infgeom.pdf), finite case:
    - Lemma 1: log q − log r is orthogonal to the affine family.
    - Lemma 3: the I-projection onto A = {p : Ap = b} lies in the exponential family log r = log q + ϑA + c, with the Pythagorean identity.
  - Csiszár's survey "Information Theoretic Methods in Probability and Statistics" cites "(Csiszár 1975)" for uniqueness of the I-projection and the Pythagorean inequality (eq. 1.6). It says equality holds for finitely many linear constraints.
- **Recommendation:** for the exponential form, cite Csiszár 1975 together with Ganchev et al. 2010 Prop. 2.1, which states it explicitly. Avoid giving a Csiszár theorem number unless someone checks the paper.

### 7b. Ganchev et al. 2010

**Citation.** Kuzman Ganchev, João Graça, Jennifer Gillenwater, Ben Taskar. "Posterior Regularization for Structured Latent Variable Models." *JMLR* 11:2001–2049, 2010.

**Access.** FULL.

**What the paper says.** **Sec. 2.4, Proposition 2.1 (eq. 4–6):**
- The problem min_q KL(q(Y) ‖ p_θ(Y | X)) s.t. E_q[φ] − b ≤ ξ, ‖ξ‖_β ≤ ε has the unique primal solution q*(Y) = p_θ(Y | X) exp{−λ*·φ} / Z(λ*).
- The dual is max_{λ≥0} −b·λ − log Z(λ) − ε‖λ‖_{β*}.
- Its gradient is the moment mismatch, so dual ascent on λ is moment matching (cf. Sec. 2.6 example: "gradient of the dual objective is 1 − E_q[φ]").

**Verdict: yes.** This is the best directly verifiable source for KL-calibrated λ by moment matching. Mind the sign convention (−λ with inequality constraints; λ free for equality constraints).

### 7c. Levine 2018

**Citation.** Sergey Levine. "Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review." arXiv:1805.00909, 2018.

**Access.** FULL.

**What the paper says.**
- **Sec. 2.2, eq. (4).** p(τ | o_{1:T}) ∝ p(τ) exp(Σ_t r).
- **Eq. (5)** (deterministic dynamics): p(τ | o_{1:T}) ∝ **1[p(τ) ≠ 0] exp(Σ_t r(s_t, a_t))**. This has exactly the "feasibility indicator × exponentiated potential" structure of our target.
- **Sec. 2.3.** The optimal policy is p(a_t | s_t, O_{t:T}) ∝ β_t(s_t, a_t) = p(O_{t:T} | s_t, a_t), the backward messages. A non-uniform action prior can be folded into the reward.
- **Sec. 3.1, eq. (11).** −KL(p̂ ‖ p) = Σ E[r + H(π)].

**Verdict: yes.**

### 7d. Korbak, Perez & Buckley 2022

**Citation.** Tomasz Korbak, Ethan Perez, Christopher L. Buckley. "RL with KL penalties is better viewed as Bayesian inference." *Findings of the Association for Computational Linguistics: EMNLP 2022*, pp. 1083–1091. DOI: 10.18653/v1/2022.findings-emnlp.77.

**Access.** FULL (arXiv v2).

**What the paper says.** **Sec. 4, eq. (5):** π*_{KL-RL}(x) = (1/Z) π_0(x) exp(r(x)/β). The appendix shows equivalence to KL minimisation toward this target.

**Verdict: yes** (the temperature parametrisation is r/β).

### 7e. Todorov

**Citations.**
- E. Todorov. "Linearly-solvable Markov decision problems." *Advances in Neural Information Processing Systems 19* (NIPS 2006; proceedings published 2007), pp. 1369–1376.
- E. Todorov. "Efficient computation of optimal actions." *PNAS* 106(28):11478–11483, 2009. DOI: 10.1073/pnas.0710743106.

**Access.** FULL for NIPS. META for PNAS (the PDF download failed).

**What the paper says (NIPS).**
- The control cost is KL(p_i(u) ‖ p_i(0)).
- **Eq. (14):** the optimally controlled transitions are p_ij exp(−v(j)) / Σ_k p_ik exp(−v(k)), i.e. the passive dynamics tilted by the exponentiated **value-to-go**, not by the immediate cost.
- Eq. (17)–(18): the linear Bellman equation in z = exp(−v).

**Verdict: yes.** This supports the "optimal local tilt needs the future (twist)" argument.

## 8. Trust-region / KL-constrained updates

### 8a. TRPO

**Citation.** John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, Philipp Moritz. "Trust Region Policy Optimization." *ICML 2015*, PMLR 37:1889–1897. The pages were confirmed on the PMLR page.

**Access.** FULL.

**What the paper says.** **Sec. 4, eq. (12)–(13):**
- The objective is minimised subject to the max-KL constraint, or in practice the average-KL constraint E_s[D_KL(π_old(·|s) ‖ π_θ(·|s))] ≤ δ.

**Verdict: yes.** Note the KL direction is **old ‖ new**.

### 8b. REPS

**Citation.** Jan Peters, Katharina Mülling, Yasemin Altün. "Relative Entropy Policy Search." *Proceedings of the 24th AAAI Conference on Artificial Intelligence (AAAI-10)*, pp. 1607–1612, 2010. DOI: 10.1609/aaai.v24i1.7727.

**Access.** FULL (AAAI PDF).

**What the paper says.**
- **Eq. (4):** information-loss bound D(p_π ‖ q) ≤ ε, with KL(new ‖ observed/old).
- **Table 1:** closed-form solution π(a|s) ∝ q(s,a) exp(δ_θ(s,a)/η), where η is the dual variable of the ε constraint.

**Verdict: yes.**
- REPS's direction (new ‖ old) matches a per-decision trust region KL(q_t ‖ p_θ(·|x_{<t})) ≤ ε.
- It yields an exponential tilt whose λ_t = 1/η is set by the dual. This is the best precedent for "KL trust region per decision".

## 9. MCTS / planning for decoding; MPC

- **9a. PPL-MCTS.** Antoine Chaffin, Vincent Claveau, Ewa Kijak. "PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided MCTS Decoding." *NAACL-HLT 2022*, pp. 2953–2967. DOI: 10.18653/v1/2022.naacl-main.215.
  - Access: FULL (skimmed).
  - Content: PUCT with the LM prior, discriminator-scored roll-outs (Sec. 3). Also re-ranking baselines.
- **9b. Leblond et al.** Rémi Leblond, Jean-Baptiste Alayrac, Laurent Sifre, Miruna Pislar, Jean-Baptiste Lespiau, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals. "Machine Translation Decoding beyond Beam Search." *EMNLP 2021*, pp. 8410–8434. DOI: 10.18653/v1/2021.emnlp-main.662.
  - Access: FULL (skimmed).
  - Content: value-network-guided decoding and MCTS. Finds the best algorithm depends on the target metric.
- **9c. Browne et al.** Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, Simon Colton. "A Survey of Monte Carlo Tree Search Methods." *IEEE Trans. Computational Intelligence and AI in Games* 4(1):1–43, 2012. DOI: 10.1109/TCIAIG.2012.2186810.
  - Access: META.
- **9d. Williams et al.** Grady Williams, **Nolan Wagener**, Brian Goldfain, Paul Drews, James M. Rehg, **Byron Boots**, Evangelos A. Theodorou. "Information Theoretic MPC for Model-Based Reinforcement Learning." *ICRA 2017*, pp. 1714–1721. DOI: 10.1109/ICRA.2017.7989202.
  - Access: FULL.
  - **Sec. III, eq. (14):** optimal distribution q*(V) ∝ exp(−S(V)/λ) p(V). The control minimises KL(Q* ‖ Q). Sec. IV uses this in receding-horizon MPC.
  - The author list has 7 names; "Williams et al." is fine.

## 10. Value of computation / metareasoning / adaptive computation

- **10a. Russell & Wefald.**
  - Stuart Russell, Eric Wefald. "Principles of metareasoning." *Artificial Intelligence* 49(1–3):361–395, 1991. DOI: 10.1016/0004-3702(91)90015-C.
  - Book: S. Russell, E. Wefald. *Do the Right Thing: Studies in Limited Rationality.* MIT Press, 1991. ISBN 0-262-18144-4.
  - Access: META.
  - The VOC concept (a computation's value = expected improvement in decision quality minus its cost) is confirmed via Russell's MITECS entry "Metareasoning" (FULL), which cites Russell & Wefald (1989) and Horvitz (1989). I did not read the AIJ text.
- **10b. Hay et al.** Nicholas Hay, Stuart Russell, David Tolpin, Solomon Eyal Shimony. "Selecting Computations: Theory and Applications." *UAI 2012*. arXiv:1207.5879.
  - Access: FULL (skimmed).
  - Formal metalevel decision problems (Def. 3); the myopic policy and its stopping properties; bounds on the number of computations.
  - It addresses selecting computations for action choice and Go, **not** SMC. Use it as conceptual justification only.
- **10c. Graves.** Alex Graves. "Adaptive Computation Time for Recurrent Neural Networks." arXiv:1603.08983, 2016 (v6 Feb 2017). **Preprint only**; I found no venue.
  - Access: META.

**Verdict on "adaptive look-ahead justified as VOC": partly.**
- The VOC sources support the principle.
- The SMC-specific precedent is Lin, Chen & Liu Sec. 5 (adaptive lookahead).
- None of these sources tests a "context trigger" rule. That rule is our own design choice.

## 11. Constrained / controlled decoding

- **FUDGE.** Kevin Yang, Dan Klein. "FUDGE: Controlled Text Generation With Future Discriminators." *NAACL-HLT 2021*, pp. 3511–3535. DOI: 10.18653/v1/2021.naacl-main.276.
  - Access: FULL (skimmed).
  - **Sec. 3:** P(x_i | x_{1:i−1}, a) ∝ P(a | x_{1:i}) P(x_i | x_{1:i−1}). The future discriminator is effectively a learned twist.
- **Grid beam search.** Chris Hokamp, Qun Liu. "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search." *ACL 2017 (Vol. 1)*, pp. 1535–1546. DOI: 10.18653/v1/P17-1141.
  - Access: META.
- **Constrained beam search.** Peter Anderson, Basura Fernando, Mark Johnson, Stephen Gould. "Guided Open Vocabulary Image Captioning with Constrained Beam Search." *EMNLP 2017*, pp. 936–945. DOI: 10.18653/v1/D17-1098.
  - Access: META.
- **NeuroLogic A*esque.** Ximing Lu, Sean Welleck, Peter West, Liwei Jiang, Jungo Kasai, Daniel Khashabi, Ronan Le Bras, Lianhui Qin, Youngjae Yu, Rowan Zellers, Noah A. Smith, Yejin Choi. "NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics." *NAACL-HLT 2022*, pp. 780–799. DOI: 10.18653/v1/2022.naacl-main.57.
  - Access: FULL (skimmed).
  - Lookahead heuristics estimate the future constraint satisfaction or cost.
- **GNMT.** Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, et al. (31 authors). "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv:1609.08144, 2016.
  - Access: FULL.
  - **Sec. 7 (Decoder):** the length normalisation lp(Y) (and coverage penalty) is needed because beam search with summed log-probabilities "will favor shorter results". α is tuned (0.6–0.7).

## 12. Graph generation (venues confirmed)

- **GraphRNN.** Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec. "GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models." *ICML 2018*, PMLR 80:5708–5717.
  - Node-by-node autoregressive generation with a graph-level RNN and an edge-level RNN under BFS ordering.
- **GRAN.** Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Will Hamilton, David K. Duvenaud, Raquel Urtasun, Richard Zemel. "Efficient Graph Generation with Graph Recurrent Attention Networks." *NeurIPS 2019* (Advances in Neural Information Processing Systems 32). arXiv:1910.00760.
  - Autoregressive generation of blocks of nodes and edges with attention-based GNNs.
  - The page range 4255–4265 appears only in a third-party record and is **not verified**.
- **DiGress.** Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard. "DiGress: Discrete Denoising diffusion for graph generation." *ICLR 2023* (poster). arXiv:2209.14734.
  - Discrete diffusion with marginal-preserving noise and a graph transformer denoiser. Supports graph-level conditioning.
- **ConStruct.** Manuel Madeira, Clément Vignac, Dorina Thanou, Pascal Frossard. "Generative Modelling of Structurally Constrained Graphs." *NeurIPS 2024*. arXiv:2406.17341.
  - The DOI 10.52202/079017-4360 comes from a third-party record and is not checked against the publisher.
  - Hard structural constraints (e.g. planarity, acyclicity) are enforced in graph diffusion via an edge-absorbing noise model and a projector.
- **PRODIGY.** Kartik Sharma, Srijan Kumar, Rakshit Trivedi. **"Diffuse, Sample, Project: Plug-And-Play Controllable Graph Generation."** *ICML 2024*, PMLR 235:44545–44564.
  - Method name: PRODIGY (PROjected DIffusion for controlled Graph Generation). Projects each diffusion step onto the constraint set; works with any pretrained model.
  - **Title caveat:** an earlier version, "Plug-and-Play Controllable Graph Generation with Diffusion Models", was at the ICML 2023 SPIGM workshop. Use the ICML 2024 title.

Access for all of item 12: META (PMLR, NeurIPS or ICLR pages, and arXiv abstracts).

## 13. Diversity

**Citation.** Dan Friedman, Adji Bousso Dieng. "The Vendi Score: A Diversity Evaluation Metric for Machine Learning." *Transactions on Machine Learning Research*, 07/2023. OpenReview id g97OHbQyk1. arXiv:2210.02410.

**Access.** FULL (skimmed). The arXiv header says "Published in Transactions on Machine Learning Research (07/2023)". The NSF copy says 06/2023.

**Content.** The score is the exponential of the Shannon entropy of the eigenvalues of K/n, interpreted as the "effective number" of dissimilar elements.

## 14. Few-cluster inference

### 14a. Cameron, Gelbach & Miller 2008

**Citation.** A. Colin Cameron, Jonah B. Gelbach, Douglas L. Miller. "Bootstrap-Based Improvements for Inference with Clustered Errors." *Review of Economics and Statistics* 90(3):414–427, 2008. DOI: 10.1162/rest.90.3.414.

**Access.** FULL (a JSTOR copy of the article).

**What the paper says.**
- **Abstract:** standard cluster-robust (CRVE) tests "can over-reject … with few (five to thirty) clusters".
- **Table 2** (rejection rates at nominal 0.05, G = 5 → 30):

  | Method | Rejection rates |
  |---|---|
  | CRVE | 0.195, 0.132, …, 0.069 |
  | Pairs cluster bootstrap-se | 0.152, …, 0.072 |
  | **Wild cluster bootstrap-t** | 0.054, 0.062, 0.056, 0.045, 0.060, 0.045 |

- **Sec. VII (Conclusion):** with the wild cluster bootstrap-t, rejection rates are "extremely close to the theoretical values, even with as few as six clusters".

**Verdict: yes.** "Cluster bootstrap over-rejects" is true for the pairs cluster bootstrap-se. The pairs bootstrap-t is better but still over-rejects at small G.

### 14b. Canay, Romano & Shaikh 2017

**Citation.** Ivan A. Canay, Joseph P. Romano, Azeem M. Shaikh. "Randomization Tests Under an Approximate Symmetry Assumption." *Econometrica* 85(3):1013–1030, 2017. DOI: 10.3982/ECTA13081.

**Access.** FULL for the working paper (Dec 2014, Princeton seminar copy). META for the journal version. Section numbers may differ in print.

**What the paper says.**
- **Sec. 2.1:** the group of sign changes G = {−1, 1}^q.
- **Sec. 3, Theorem 3.1:** asymptotic validity under approximate symmetry.
- **Sec. 4:** applications with a fixed number q of clusters, each with many observations. Clusters may be heterogeneous and may depend on each other.
- The test is computed from q cluster-level estimates.
- They contrast it with the t-test results of Ibragimov–Müller and Bakirov–Székely, which require α ≤ 8.3% (q ≥ 2) or α ≤ 10% (2 ≤ q ≤ 14).

**Verdict: yes.**
- **My own arithmetic note (not from the paper):** with q clusters the sign-change group has only 2^q elements, so the smallest attainable p-value is limited (e.g. q = 5 gives 32 sign patterns). Report the attainable p-value resolution.

### 14c. Holm 1979

**Citation.** Sture Holm. "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6(2):65–70, 1979. JSTOR 4615733.

**Access.** META.

## 15. Self-normalised IS bias O(1/K)

**Citation.** Art B. Owen. *Monte Carlo theory, methods and examples.* 2013, online book: https://artowen.su.domains/mc/.

**Access.** FULL (Ch. 2 and Ch. 9 PDFs).

**What the book says.**
- **§9.2 (Self-normalized importance sampling):**
  - eq. (9.6)–(9.7) define the estimator as a ratio of means;
  - Theorem 9.2 gives consistency;
  - eq. (9.8) gives the delta-method variance;
  - §9.2 refers back to ratio estimation.
- **The O(1/n) bias is in §2.7 "Estimating ratios", eq. (2.32)–(2.33):** E(θ̂ − θ) ≈ (1/(nµ_x²))(θσ_x² − ρσ_xσ_y). The text says: "Because the bias is O(1/n) while the root mean squared error is of order 1/√n …".

**Verdict: yes.** Cite "Owen (2013), §2.7 and §9.2", not Ch. 9 alone.

## 16. Resampling

### 16a. Douc, Cappé & Moulines 2005

**Citation.** Randal Douc, Olivier Cappé, Eric Moulines. "Comparison of Resampling Schemes for Particle Filtering." *Proc. 4th Int. Symposium on Image and Signal Processing and Analysis (ISPA 2005)*, pp. 64–69. DOI: 10.1109/ISPA.2005.195385. arXiv:cs/0507025.

**Access.** FULL.

**What the paper says.**
- Residual and stratified resampling have lower conditional variance than multinomial.
- **There is a counter-example showing that this does not hold for systematic resampling.**
- In practice the schemes give comparable results, and systematic is "often preferred because it is the simplest method to implement".

**Caveat.** Do **not** claim that systematic resampling provably reduces variance.

### 16b. Doucet & Johansen

**Citation.** Arnaud Doucet, Adam M. Johansen. "A Tutorial on Particle Filtering and Smoothing: Fifteen years later." In D. Crisan & B. Rozovskii (eds.), *The Oxford Handbook of Nonlinear Filtering*, Oxford University Press, 2011.
- The tutorial version is v1.0 (Apr 2008) / v1.1 (Dec 2008); it is often cited as 2009.
- The page range was **not verified**.

**Access.** FULL.

**What the tutorial says.**
- **Sec. 3.4 (Resampling):** systematic resampling "is the most widely-used … outperforms other resampling schemes in most scenarios (although this is not guaranteed in general [Douc et al.])".
- **Sec. 4:** "we recommend only resampling when the ESS is below a threshold and employing the systematic resampling scheme".

## 17. Distribution shift / domain generalisation

- **17a. WILDS.** Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M. Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, Percy Liang. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." *ICML 2021*, PMLR 139:5637–5664.
  - Access: FULL (skimmed).
  - FMoW-WILDS has shifts over time and across regions, and is evaluated by worst-region accuracy. PovertyMap-WILDS shifts across countries. These are suitable citations for geographic shift.
- **17b. Gulrajani & Lopez-Paz.** Ishaan Gulrajani, David Lopez-Paz. "In Search of Lost Domain Generalization." *ICLR 2021*. arXiv:2007.01434.
  - Access: META.
  - Introduces DomainBed. Finds that ERM is within about 1 point of every DG algorithm under careful tuning, and that model selection is part of the method.

---

## Items not verified (do not state as fact without checking)

1. **Csiszár 1975:** theorem number and exact statement of the exponential-family form. The full text was not accessible.
2. **GRAN NeurIPS page range** (4255–4265 appears only in a third-party record).
3. **Doucet & Johansen Oxford Handbook page range.**
4. **ConStruct NeurIPS DOI** (10.52202/079017-4360 comes from a third-party record).
5. **Liu (2001)** as the source of "proper weighting up to a constant c".
6. **Russell & Wefald 1991 AIJ:** content not read; bibliographic data only.
