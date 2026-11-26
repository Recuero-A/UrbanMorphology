# Milestone 13 method: TerrainConstrainedGraphTurtle

**Terrain-constrained probabilistic inference for an autoregressive road-graph model.**

The trained network (the M12 TerrainGraphTurtle: N0, TN, TN-lp, TN-se) is not changed. Everything here is
inference. Code: `src/urbanmorph/graphs/tc_sampler.py`, `tc_potentials.py`, `generate.py`
(`resolve_step` / `commit`).

## 1. The base model as a sequential decision process

**State.** s = (partial graph G, queue of open nodes, current node u, decoder state h, previous committed arm).

**Decisions.**
* Node level: when the queue is empty, the start head proposes a start cell or ends the tile (unchanged from M12).
* Arm level: when node u is popped, the decoder emits outgoing arms one at a time.

**Arm action.** a = STOP, or a continuation (d, c, σ):
* d: a displacement in the turtle frame (30 m units);
* c ∈ {major, mid, local, minor}: the street class;
* σ ∈ {ground, bridge, tunnel}: the structure.

**Probability of an arm action.** Given the decoder output y = GRU(h, input), the heads are conditionally independent:

  p_θ(STOP | s) = σ(ℓ_stop)

  p_θ(d, c, σ | s) = (1 − σ(ℓ_stop)) · Σ_j π_j N(d; μ_j, T²Σ_j) · Cat(c; softmax ℓ_c) · Cat(σ; softmax ℓ_σ)

* T = 0.5 is the M12 reference displacement temperature. It acts **only** on the mixture spread.
* `fulltemp` (method C) additionally divides the stop, class and structure logits and the mixture logits by
  T_full, and sets the spread to T_full. That is a different mechanism and is reported separately.

**Transition.** s' = Φ(s, a), deterministic.
* The geometric rules of `GenState.resolve_step` (ground): clip to the tile; turn a crossing into a junction, except
  over or under structures; merge with a node within 8 m; T-junction with a street within 8 m; drop steps shorter
  than 8 m.
* For bridges and tunnels, `resolve_structure`: pass over or under, land on a ground node within 8 m or on a new node.
* Φ defines the **committed edge** e(s, a) = (u, q), where q can differ from u + d.

**State consistency (M13 requirement).**
* The decoder's next input is the committed displacement R(−heading)(q − u)/30 m and the committed class, not the
  sampled d.
* An action that inserts nothing (dropped) is never fed to the decoder.
* Tests: `tests/test_tc_sampler.py::test_decoder_is_fed_the_committed_edge` and `::test_scored_geometry_is_inserted_geometry`.
* The M12 loop (`legacy`, control A0) fed the sampled d and applied the steps after decoding all arms. It is kept,
  unchanged and bit-identical, for comparison.

## 2. Target distribution

The target over whole generation histories a_{1:T}, with s_{t+1} = Φ(s_t, a_t), is

  π(a_{1:T}) ∝ Π_t p_θ(a_t | s_t) · G(s_t, a_t),   G(s, a) = 1[hard(s, a)] · exp(Σ_k λ_k φ_k(e(s, a)))

STOP has G = 1. **π is not a calibrated posterior.** It is the base model reweighted by explicit, documented
factors. The weights λ are hyperparameters chosen on development data (DEV-A only).

### 2.1 Hard constraints 1[hard]

Hard constraints are genuine impossibilities or representation restrictions, evaluated on the **committed** edge
(`tc_potentials.py`):
* the step is not representable;
* a ground edge over water;
* an edge ending in water;
* an unsupported structure (a bridge over neither water nor a street; a tunnel with less than 3 m of cover over
  half its length);
* a bridge over sea;
* a structure longer than the longest real one of its kind in training;
* a water crossing at an angle smaller than the smallest real one.

**Steep grades are not hard.** Streets are built on steep land with switchbacks and earthwork.

### 2.2 Soft log-potentials φ_k, calibrated on training data

**Terrain response of ground edges:** density ratios of the real to the proposal distribution of edge features:

  φ_grade = log P_real(g | c, b) − log P_prop(g | c, b),   φ_align = log P_real(α | b) − log P_prop(α | b)

* g: the maximum 30 m sub-span grade of the committed edge.
* b: the landform-slope bin (32 m) at its midpoint (0–5, 5–10, 10–15, 15–25, > 25 %).
* α: the |cos| between the edge and the terrain gradient (land > 5 %).
* P_real: the 3D-aware graphs of the 473 training tiles with a DEM.
* P_prop: the same backbone's own unconstrained generation (state-consistent policy `none`, T = 0.5) on 160 of
  those tiles, × 2 samples.
* Histograms are add-one smoothed and clipped to |log r| ≤ log 20. Cells with fewer than 30 edges get r = 1.

**Why ratios, not ad-hoc energies.** A potential exp(−E) with E ≥ 0 multiplies every edge by a factor ≤ 1, so π
would prefer fewer and shorter streets: the M12 starvation failure. A density ratio satisfies E_prop[r] = 1, so
reweighting corrects the feature marginal without a systematic pull toward stopping. This holds per step and
approximately: features are evaluated on committed edges of the state-dependent proposal.

**Structures** (secondary; sparse data):

  φ_span = log min(1, S(span)/S(median)),   φ_angle = log min(1, F(angle)/F(median))

* S: the real span survival function (class-specific; pooled when a class has fewer than 30 runs).
* F: the real crossing-angle CDF.
* Long and oblique structures cost more, so infrastructure is not free.

**Geometry** (optional, λ_geom = 0 by default): φ_junc = log min(1, F_J(φ)/F_J(median)), for the smallest junction
angle a merge, T or crossing creates.

**Connectivity** has no local potential. It enters only through look-ahead (§4).

## 3. One decision: the K-candidate proposal

For a continuation decision at state s:
1. Draw K iid candidates a^k ~ p_θ(· | s, continue). This costs one decoder call; the heads are sampled K times.
2. Resolve each candidate by Φ and compute w^k = G(s, a^k), times the twist ψ̂^k when look-ahead is on.
3. Estimate the local normaliser, Rao-Blackwellised over the stop decision:

   Ẑ(s) = p_stop + (1 − p_stop) · (1/K) Σ_k w^k

4. STOP with probability p_stop / Ẑ; otherwise continue with candidate k with probability ∝ w^k.

**Validity.**
* This is the importance-resampling step of nested or random-weight SMC.
* The selected action, with weight Ẑ, is properly weighted for the local target p_θ · G (Naesseth et al. 2015): it
  is unbiased for the normaliser, and approaches exact sampling as K → ∞.
* **Do not select ∝ p_θ · w among candidates drawn from p_θ.** That would target p_θ² · G.

**When all K candidates are infeasible,** Ẑ = p_stop: the arm stops, and the particle weight records it. In a
single chain this is the only approximation.

Methods (all share the proposal and the same committed geometry):

| Method | Candidate weights w | Choice | Particles |
|---|---|---|---|
| A0 `legacy` | — | M12 loop | 1 |
| A `standard` | none (K = 1) | sample | 1 |
| B `topk` | — | keep the k most likely of K by log p_θ; sample ∝ p_θ among them | 1 |
| C `fulltemp` | — | K = 1, all heads tempered | 1 |
| D `filter` | 1[hard] | Rao-Blackwellised, ∝ w | 1 |
| E `energy` | 1[hard] · e^{λφ} | Rao-Blackwellised, ∝ w | 1 |
| E+LA | × ψ̂^β | as E | 1 |
| F `beam` | 1[hard] · e^{λφ} | stop **sampled**; continuation = argmax log p_θ + log w; keep the top B/2 particles by **length-normalised** score and duplicate them | B |
| G `smc` | as E | as E; particle weight × Ẑ; systematic resampling when ESS < 0.5 N | N |
| G+LA | as E+LA | approximate twisted SMC | N |

**Beam.** Comparing log p(STOP) with continuation scores that contain log *densities* of the continuous
displacement would almost never stop (mode-seeking degeneracy). The beam therefore samples the stop decision, and
scores are normalised by the number of decisions.

**Final output.**
* SMC: one particle drawn ∝ its weight. This is an approximate sample from π, consistent as N → ∞; for finite N we
  claim no exactness.
* Beam: the best normalised score.

## 4. Multi-step look-ahead: estimated twists

**Purpose.** A candidate can be locally fine but lead into terrain where the road cannot continue: the M12 HR
failure.

**Roll-out.** For each candidate that opens a new road end v:
1. Copy the state and commit the candidate.
2. Roll out the continuation for up to H steps. At each step, take the first arm decision at the current end, with
   M candidates: Ẑ_h = p_stop + (1 − p_stop) · mean_m w_m.
3. Continue with a candidate drawn ∝ w. The roll-out ends early if the road stops by model choice, reconnects (a
   merge, crossing or T-junction), leaves the tile, or cannot continue (w ≡ 0).
4. The estimated twist is ψ̂ = Π_h Ẑ_h. It is an estimate of the future feasibility mass along the continuation.

**Properties.**
* Under ratio potentials ψ̂ ≈ 1 on average where the terrain is passable, and ψ̂ ≪ 1 where every continuation is
  hard-infeasible: a terminal cost for forced dead ends.
* Stopping by model choice is not penalised. Reconnection ends the roll-out with no penalty.
* Normalised scores are not needed: ψ̂ is a product of normalisers, each ≤ 1 only through infeasibility. It does
  not shrink because a path is long.

**Selection** uses w · ψ̂^β.

**In SMC,** when node v is later popped, the particle weight is divided by ψ̂(v)^β, so the twists cancel in the
final target.
* Because ψ̂ is a random estimate and roll-outs follow only the first arm, **G+LA is an approximate twisted SMC**
  (Lin et al. 2013; Zhao et al. 2024). It is not exact. E+LA is a look-ahead heuristic within the same scoring
  framework.

**Scope.** Look-ahead is local road-path planning: one road continuation, H steps. It does not plan the global
network and guarantees no connectivity. Global aspects are handled separately: frontier scheduling (§5),
tile-free generation (§6), and procedural repair as a labelled baseline (§7).

## 5. Frontier scheduling

The queue order changes which roads grow first:

| Scheduler | Next node |
|---|---|
| `fifo` | first in, first out (M12) |
| `class` | the most major incident class |
| `confidence` | the highest log p of the step that created the node |
| `connect` | the smallest distance to a node of another component |
| `stochastic` | among the first L = 4 queue entries, sampled ∝ exp(−class): bounded, so major roads are not deterministic |

## 6. Tile-free (global) generation

**Model constraint.** The turtle context is local (192 m crops). Only the start head (a 128 × 128 canvas) and
clipping are tied to 512 m.

**Global mode (§ protocol, experiment X5):**
* one GenState over the whole area, one terrain field;
* the start head is applied to sliding 512 m windows;
* clipping happens only at the outer boundary.

The position features are tile-relative in training. In global mode each node uses its position within the 512 m
cell containing it, which is a documented approximation. Global mode is compared with M11/M12 windowed generation on
the same terrain.

## 7. Procedural repair baseline (not learned)

`repair`: after generation, dead ends and small components are connected to the nearest other component within
150 m by a terrain least-cost path (8 m grid, slope cost, water obstacle; the P-baseline router). It is reported
separately, as procedural repair. The added length, intersections and bridges are counted.

## 8. Calibration and selection data

| Item | Data |
|---|---|
| Potentials | training tiles only |
| Joint-metric normalisation | real training tiles only |
| Sampler hyperparameters (K, λ, H, M, N, B, scheduler) and the density-matching stop bias | DEV-A only |
| Confirmation | DEV-B, then the generalisation cities, once |

The original `test` / `test_t` splits are not read.
