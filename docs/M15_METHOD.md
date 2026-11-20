# Milestone 15 method: style guidance on the existing samplers

## 1. What is unchanged

* The trained TerrainGraphTurtle (M12), the M13 sampler family and the M14 network-aware samplers are used as they
  are (`docs/OVERVIEW.md` §3).
* At each decision a sampler draws K = 8 candidate continuations a_k from the model. Each candidate is resolved by the
  same geometric insertion rules (snap, merge, split, clip), checked against the hard rules, and given a weight:
  * **M\***: w_k = G_terrain(a_k);
  * **NA-\***: w_k = G_terrain(a_k) · G_net(a_k).
* The road continues with candidate k ∝ w_k (optionally twisted by look-ahead roll-outs, or inside SMC), or stops
  with probability p_stop / Z, where Z = p_stop + (1 − p_stop) · mean_k w_k.

## 2. The style factor

M15 multiplies each feasible candidate's weight by a style factor W_style(a_k) (`src/urbanmorph/graphs/style.py`).

**Heading term.** A candidate edge a → q has axial bearing θ, measured at its midpoint m:

> log W_head = g(c) · κ · Σ_j π_j · (cos 4(θ − θ_j(m)) − 1) / 2

* The term is in [−κ, 0]: a street on the field direction or perpendicular to it keeps its weight; one at 45° loses
  κ nats.
* **Fields θ_j:**
  * **grid**: a constant θ0;
  * **radial**: the bearing from a declared centre (the cross field favours spokes and rings);
  * **organic**: θ0 + A · Σ_{i=1}^{6} b_i sin(2π f_i · m + φ_i), with wavelengths in [λ, 2λ], unit-RMS amplitudes
    and random directions and phases from a generator keyed by (site, inference seed).
* **Mixture weights π_j** come from `mix:grid=0.5+organic=0.5`.
* **Class gain g(c):** 0.25 major, 0.5 mid, 1 local, 1 minor. This guides local streets fully and the connecting
  backbone less, using the road-class head of the model.

**Loop term (cul-de-sac).** log W_loop = −g(c) · λ_loop when the candidate joins its own component, i.e. closes a
cycle (merge, T-junction or crossing).

**Local termination (cul-de-sac).** A bias added to the stop logit. It is separate from the heading and loop terms
and can be combined with any heading field.

**Selection normalisation.** Within each decision, the log style factors of the feasible candidates are shifted by
their log-mean, so the mean of W_style over the K candidates is 1. Style then redistributes probability among
continuations but leaves Z, and so the stop probability, essentially unchanged. Without this, every factor ≤ 1 shrinks
Z and styled networks get shorter (`docs/M15_AUDIT.md` §6.2). The un-normalised factor is an ablation.

**Grade ceiling (optional).** A ground candidate whose maximum 30 m sub-span grade exceeds a cap is infeasible.
* `real_p99` uses the class-specific 99th percentile of real streets (`configs/m12_feasibility.json`): major 12.6 %,
  mid 10.4 %, local 16.4 %, minor 21.8 %.
* It is off by default: M13 kept grades soft, because real streets are built on steep land.

## 3. What each sampler targets once style is on

The normalised factor depends on the decision's own candidate set, so it is not a fixed tilt of the trajectory
distribution. No sampler, SMC included, is exact for a fixed target once style is on; all are heuristics whose induced
distribution is measured. With style off, every sampler is unchanged: the replay audit gives 95/95 identical graphs
for M\* and NA-fixed.

## 4. Strengths

Strengths were frozen by a rule written before the development sweep (`configs/m15_style.json`):
* **grid** κ = 4; **organic** κ = 8; **radial** κ = 8;
* **cul-de-sac**: stop bias +0.25, loop penalty 2.

For organic and radial the controllability gain was still rising at κ = 8, the edge of the pre-declared grid
(`outputs/m15/dev_select.md`). Larger κ was not explored, to keep the rule as written.

## 5. External baseline: cross-field tracer

`src/urbanmorph/graphs/field_tracer.py`. Streamlines follow the same style field from a start near the tile centre.
* Each step is 26 m and follows the cross-field direction closest to the current heading.
* A streamline ends at the tile edge, at a hard rule, or on joining existing street.
* Every d_sep = 78 m it seeds two perpendicular streamlines. Tracing stops at a 6 km length budget or when no seed is
  left.
* It uses the same graph operations, hard rules, bridge eligibility and grade ceiling as the samplers.
* It is inspired by the streamline tracing of Chen et al. (2008); it is not their implementation. It has no learned
  prior and no soft terrain preference.
