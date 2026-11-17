# Milestone 14: mathematical formulation

Status: written before the confirmatory experiments (2026-11-17). It formalises the M13 samplers that M14 keeps, adds network‑level potentials and adaptive look‑ahead, and states for every sampler which distribution it actually draws from. The exact toy experiments are in §8 (`scripts/m14_toy.py`). The code lives in `src/urbanmorph/graphs/m14_sampler.py`, and the tests in `tests/test_m14.py`.

## 1. Objects

| Symbol | Meaning |
|---|---|
| T | Terrain of the tile and its 256 m margin: 4 m elevation and water kind (river / lake / sea). The model sees the original, low‑passed or street‑erased elevation, depending on the backbone (TN / TNlp / TNse). Rules and metrics always use the original. |
| s_t = (G_t, Q_t, u_t, k_t, h_t) | Generation state. G_t is the current graph: nodes, edges, classes, structures. Q_t is the FIFO queue of **open** nodes. u_t is the node being expanded, k_t its arm counter, and h_t the GRU state of that expansion. |
| a_t | Action. Either STOP (end the expansion of u_t) or a continuation (d, c, σ): a displacement d ∈ ℝ², a class c ∈ {0..3} and a structure σ ∈ {ground, bridge, tunnel}. Start decisions use the start head: either end the tile, or place a new start node in a 16 m cell. |
| R(s, a) | Deterministic geometric resolution of a continuation (M5 rules, M13 `resolve_step` / `resolve_structure`). It clips to the tile, splits at the first crossing, merges within 8 m (node, then T‑junction), and drops steps shorter than 8 m. It returns the **committed** edge. |
| s_{t+1} = f(s_t, a_t) | Transition. It inserts R(s_t, a_t), queues a new node, and feeds the GRU the committed displacement. State consistency (tested in M13 and M14): the decoder never sees a sampled‑but‑uncommitted action. After MAX_T = 7 arms the expansion of u ends without a STOP decision. |
| p_θ(a_t \| s_t) | Learned prior: the TerrainGraphTurtle at temperature 0.5 on the displacement mixture (as M12/M13). |
| τ = (a_1, …, a_n) | Action trajectory; n is random, and generation ends when the queue is empty and the start head ends the tile. |
| x = G(τ) | Final graph, a deterministic function of τ. Many trajectories map to the same graph, so p_θ(x) = Σ_{τ: G(τ)=x} p_θ(τ). **Every distribution below is defined over trajectories.** Metrics are functions of x only. |

## 2. Terrain feasibility and terrain potentials (unchanged from M13)

* **Hard feasibility.** The constraint 1_F(s, a) ∈ {0, 1} is evaluated on the committed edge. A step is infeasible if it is unrepresentable, is a ground edge over water, ends in water, is an unsupported structure, crosses the sea, exceeds the maximum span, or has a degenerate crossing angle. The thresholds are M13's; the audit's criticisms of them stay open and are listed in `docs/M14_FINAL_REVIEW.md`.
* **Soft terrain potential.** φ_r(s, a) = log r̂_grade + log r̂_align. These are density ratios between real training statistics and the model's own proposals, calibrated on training tiles, with λ_r = 0.5 as frozen in M13.
* **Local factor.** G^terr_t = 1_F · exp(λ_r φ_r).

## 3. Network potentials (new in M14)

### Definitions

An interior node is a node that is not on the tile border (BORDER_M = 1 m). A node is **open** in s if it is in Q or is the node u currently being expanded; open nodes can still gain edges.

* **D(s): closed interior dead ends.** The number of interior nodes of degree 1 that are not open.
* **I(s): closed islands.** The number of components that meet three conditions: at least 40 m of street, no border node, and no open node.

When generation ends there are no open nodes. D(x) and I(x) are then the final graph's interior dead ends and its isolated components of at least 40 m. These are the quantities the M14 audit found to be 2× and 4–6× the real rates (`outputs/m14/audit/`).

### Network factor

G^net_t = exp(−λ_D [D(s_{t+1}) − D(s_t)] − λ_I [I(s_{t+1}) − I(s_t)]).

**Proposition 1 (telescoping).** For every complete trajectory τ,

  Π_t G^net_t = exp(−λ_D D(x) − λ_I I(x)).

*Proof.* The product of exp(−λ ΔS) over consecutive states is exp(−λ [S(s_final) − S(s_0)]). Here S(s_0) = 0 (empty graph) and S(s_final) = S(x), since no node is open at the end. ∎

The proof needs the increments to be accounted for on **every** transition, and in particular on the transitions that close a node without a STOP decision. Our implementation handles them as follows:

* **Events that change D or I.**
  * STOP at u closes u. This gives D += 1 if deg u = 1 and u is interior, and I += 1 if this closes a component of at least 40 m that has no border node.
  * A continuation that joins a closed dead end gives D −= 1.
  * A continuation that joins a different, closed island gives I −= 1.
  * Every other event leaves D and I unchanged: a new node is open, a split at a crossing or T‑junction changes no degree‑1 node, and a merge into an open component closes nothing.
* **Closure without STOP.** On the MAX_T‑th arm, the factor of each continuation includes the closure of u (`m14_sampler.NetSampler.candidates`). A run that ends by `max_rounds` with nodes still queued would break the identity; this is counted in the statistics and did not occur in the tests.

`tests/test_m14.py::test_network_weights_telescope_to_final_counts` checks Proposition 1 to machine precision on five random‑model runs.

### Target

π_λ(τ) ∝ p_θ(τ) · Π_t 1_F · exp(λ_r Σ_t φ_r − λ_D D(x) − λ_I I(x)).

This is an **energy‑reweighted prior**, not a calibrated posterior; nothing is observed. The only probabilistic claim attached to it is Proposition 2.

### Proposition 2 (I‑projection; Csiszár 1975, full text not accessed; the result as used here is stated in Ganchev et al. 2010, Prop. 2.1, eqs. 5–6)

Let p_F be the prior restricted to feasible trajectories and tilted by the terrain potentials. Among all distributions q on trajectories with E_q[D] = d* and E_q[I] = i*, the one that minimises KL(q ‖ p_F) is q ∝ p_F exp(−λ_D D − λ_I I), for the λ that solves the two moment equations (when a solution exists in the interior).

*Sketch.* This is the standard Lagrangian argument for minimum‑KL distributions under linear expectation constraints. The literature file `outputs/m14/literature/inference_planning.md` records the verified source.

**Consequence: why λ is not arbitrary.** We choose λ so that the generated dead‑end and island rates match the real rates on **training** tiles (d*, i* = real means). Under the exact target, π_λ is then the distribution that matches real network statistics while deviating least, in KL, from what the network learned. This is our formal reading of "preserve the learned prior".

**Fitting.** The moment equations are solved by generalized iterative scaling (multiplicative moment matching; `scripts/m14_calibrate.py`): λ_k ← λ_k + log((E_gen S_k + ε) / (S*_k + ε)), with ε = 0.05 and λ clipped to [0, 6].

**Caveat.** E_gen is estimated with the *implemented* sampler, which does not sample π_λ (§4). The fitted λ therefore matches the moments under the implemented sampler, and the minimum‑KL property holds only for the exact tilt. We state this wherever the calibration is used.

**Not a weighted sum of arbitrary objectives.** The network objective has two terms, each a count of a failure type the audit measured. Their weights are fitted by moment matching, not hand‑tuned. Other candidate objectives (coverage, circuity, reachability) are *evaluated*, not optimised, to avoid metric gaming.

## 4. Samplers and the distributions they actually draw from

Let w_t(a) = G^terr_t(a) · G^net_t(a), the product of the local factors (including the STOP option).

### 4.1 Local K‑candidate selection (M\*, and the M14 `na_myopic` / `na_fixed` / `na_adaptive` variants)

At each decision the sampler works in three steps:

1. Draw K continuations a^{1..K} iid from p_θ(· | s, not STOP).
2. Choose STOP with probability p_s w_s / Z, where Z = p_s w_s + (1 − p_s) · (1/K) Σ_k w(a^k), and otherwise choose a^k with probability ∝ w(a^k).
3. Commit the choice.

As K → ∞, the selection converges to the locally normalised distribution:

  q_loc(a | s) = p_θ(a | s) w(a) / Z(s),  where Z(s) = Σ_a p_θ(a | s) w(a).

Over trajectories this gives

  q_loc(τ) = Π_t p_θ(a_t | s_t) w_t / Z(s_t) = π_λ(τ) · Z_π / Π_t Z(s_t).

So **q_loc = π_λ if and only if Π_t Z(s_t) is constant over trajectories.** It is not constant in general: a state close to water, or to closing an island, has a smaller Z. This is the discrepancy flagged by the M13 audit. The tilted target *can* be written as a locally normalised policy, but only with conditionals that include the future normaliser, i.e. the optimal twist (Zhao et al. 2024, Prop. 3.2; Levine 2018, §2.3; Todorov 2006, NIPS 19). The myopic sampler omits that factor. The "label bias" results of Lafferty et al. 2001 and Andor et al. 2016 concern *conditional* models, so we use them only as an analogy (Andor et al., footnote 6, note that joint models differ).

For finite K, the selection is a self‑normalised importance resampling step. Its single‑step distribution q_K interpolates between the prior and q_loc: K = 1 is the prior plus a feasibility check, and q_K → q_loc as K → ∞ (toy C3).

**What we claim for every single‑chain sampler: nothing exact.** It is a heuristic sampler whose induced distribution we measure.

### 4.2 Twisted local selection (look‑ahead)

Let ψ*(s) = Σ_{future} Π_{t′>t} p_θ w be the expected future product of local factors. Selecting with w(a) · ψ*(f(s, a)) gives q = π_λ exactly (toy C2: KL = 0 to machine precision).

We estimate ψ by **Monte Carlo roll‑outs** (the M13 roll‑out, extended), applied to each feasible continuation candidate:

1. Commit the candidate on a copy of the state.
2. Continue the new road for up to H steps with M = 4 candidates per step, under the same local selection, including STOP and both network factors. The node being expanded, u, is kept open in the roll‑out.
3. Compute ψ̂ = Π_h Ẑ_h, where Ẑ_h is the local normaliser estimate at roll‑out step h.

Three approximations separate ψ̂ from ψ*:

* the truncation at horizon H;
* the roll‑out policy only extends the new road, not other frontier nodes;
* log ψ̂ is the log of a Monte Carlo estimate.

**Claim: heuristic approximation of the twisted selection.** It is exact only in the limit of infinite roll‑outs over the full future, which we do not implement.

### 4.3 Sequential Monte Carlo (`smc`, `na_smc`)

N particles evolve under the K‑candidate proposal. Each weight is multiplied by Ẑ_t = p_s w_s + (1 − p_s) mean_k w(a^k), an unbiased estimate of Z(s_t) given s_t. Particles are resampled (systematic) when ESS < N/2, and one particle is returned ∝ its final weight.

With the proposal and weight above, each (selected action, Ẑ) pair is properly weighted for the local factor (Naesseth et al. 2015: Definition 1, §3.1, E[f(X) W] = ∫ f γ for the unnormalised γ; and §3.2, which shows that drawing K candidates, selecting one ∝ weight and carrying the mean weight is properly weighted). The weighted particle system then targets π_λ, and the returned graph is distributed as π_λ as N → ∞ (standard SMC consistency; Doucet & Johansen 2009).

The confirmatory `na_smc` uses **no** look‑ahead (horizon 0), so no random twist enters the weights. **Claim: asymptotically exact for π_λ. At N = 4 it is biased; we report ESS.**

### 4.4 Adaptive look‑ahead

Roll‑outs are run at a decision only if a trigger c(s) ∈ {0, 1} fires, so the twist used is ψ̂^{c(s)}. Inside a properly weighted SMC, the twist only changes variance and cost, not the target (toy C5).

Inside the single‑chain sampler, a trigger changes the induced distribution. The toy shows this can go either way: a plausible trigger ("close to a constraint") was **worse** than random allocation at the same budget, while an oracle trigger based on the value of computation was much better (toy C4). Hence:

* **Value of computation (VoC) at a decision.** VoC = TV(selection distribution with the twist, selection distribution without it). Its total over a run is the selection change the roll‑outs buy.
* **Trigger validation (protocol §4).** Triggers are built from cheap context features: the island state of u's component, deg u = 1, water within 48 m, landform slope ≥ 0.15, feasible‑candidate share < 0.75, and the spread of candidate log‑weights. They are scored on DEV‑A by the share of total VoC they capture against the share of decisions on which they fire, and compared with a random trigger at the same budget. A trigger is used only if it beats random allocation on DEV‑A.
* **Metareasoning motivation.** Allocating computation by its expected value follows Russell & Wefald 1991 (book, not read in full) and Hay et al. 2012 (UAI). In SMC, Lin, Chen & Liu 2013, §5, choose the look‑ahead depth adaptively with a heuristic criterion, and their Prop. 7 shows that extra look‑ahead can hurt at a fixed sample size. A context trigger validated by measured VoC is, to our knowledge, our own construction.

**Claim: heuristic, with a measured VoC. No guarantee.**

### 4.5 Per‑decision KL trust region (diversity preservation, exploratory)

The selection distribution is q_β ∝ (p_s w_s^β, (1 − p_s) w(a^k)^β / K), with β ∈ [0, 1] the largest value such that KL(q_β ‖ q_0) ≤ ε. Here q_0 is β = 0, with hard zeros kept: the prior restricted by the hard constraints.

**Proposition 3.** By the chain rule of KL (cf. the per‑update KL bound of REPS, Peters et al. 2010), the trajectory‑level divergence obeys KL(q_traj ‖ q_0,traj) = Σ_t E_q[KL(q_t ‖ q_0,t)] ≤ ε · E_q[number of decisions].

So ε bounds how far the sampler drifts from the feasibility‑restricted prior, in nats per decision. β adapts the planning strength automatically: it is weak where the potentials disagree sharply with the prior, and full where they agree. **The bound is relative to q_0, the K‑candidate filter, not to p_θ.**

## 5. Termination

Generation ends when the queue is empty and the start head ends the tile. It can also end through the node budget (1,000 nodes), the component budget (40), or `max_rounds` = 1,500. Both of the following are counted per tile in the statistics:

* `maxT_last_arm`: expansions closed after MAX_T arms (accounted for in G^net);
* `forced_stop`: stops forced when every candidate is infeasible.

Premature termination is evaluated as in M13: the share of tiles with less than half the real density, and empty networks.

## 6. What the planner can and cannot change

* **What it can do.** The network factors change *which* sampled continuation is committed, and whether u stops. Through those choices they also change where future starts are placed.
* **What it cannot do.** They cannot move a start node, and cannot create a street the prior assigns zero density to.
* **Connections are not forced.** A missing connection that needs a bridge is not rewarded unless the bridge is feasible. Islands separated by water are counted by I, but the metric `od_unreach_water` does not count them as failures.

## 7. Claims summary

| Sampler | Distribution claimed |
|---|---|
| A (`standard`) | p_θ restricted to the committed geometry (M13), with no terrain or network factors |
| M\*, `na_myopic` | Local selection (§4.1); heuristic; induced distribution measured |
| `na_fixed`, `na_adaptive` | Local selection with an estimated, truncated twist (§4.2, §4.4); heuristic |
| `na_smc` | Asymptotically π_λ (§4.3); finite‑N bias reported via ESS |
| `na_tr` (trust region) | Heuristic, with the per‑decision KL bound of Proposition 3 |

## 8. Exact toy experiments (`scripts/m14_toy.py`, `outputs/m14/toy/`)

**Setup.** The toy has T = 8 steps, A = 3 actions, a random Markov prior and random local potentials. A terminal hard constraint plays the role of "the road must reach the network", and a telescoping dead‑end count is weighted by λ = 2. All 6,561 trajectories are enumerated.

| Claim | Result |
|---|---|
| C1 label bias | KL(q_loc ‖ π) = 1.12 nats |
| C2 twist | Exact ψ*: KL = 0. H‑step truncated twist: KL 0.25 (H = 1), 0.082 (H = 2), 0.014 (H = 3), 0.0011 (H = 4), 0 (H = 8) |
| C3 finite K | KL(q_K ‖ π) = 0.64 (K = 1), 0.43 (2), 0.42 (4), 0.45 (8). Not monotone. Local feasibility forcing (K → ∞) is *further* from π (1.12) than plain rejection |
| C4 adaptive look‑ahead (H = 2, budget 41 % of decisions) | No look‑ahead 1.12; random allocation 0.62 (p10–p90: 0.29–0.92); plausible "near constraint" trigger **0.77**; oracle‑VoC trigger 0.27; all decisions 0.08 |
| C5 SMC | N = 8, no twist: 0.115. N = 8, adaptive twist: 0.097. N = 8, twist at every step: 0.034. N = 32, no twist: 0.029. Monte Carlo floor (π itself sampled 20,000 times): 0.025 |
| C6 frontier | For the same expected dead‑end count, the exact tilt has the smallest KL from the feasible prior; K = 8 selection is in between, and the local K → ∞ sampler is far off (`fig_toy.png`). Local forcing alone *raises* the dead‑end count (0.59 vs 0.37 under the exact feasible prior) |

These toy results motivate three design rules:

1. measure, never assume, the distribution induced by a single‑chain sampler;
2. validate any adaptive trigger against random allocation;
3. include a properly weighted SMC variant as the reference for the target.
