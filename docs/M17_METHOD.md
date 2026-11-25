# Milestone 17 method: where close junctions come from, and the fixes

* **Code.**
  * `graphs/junction_events.py`: the event log, off by default.
  * `graphs/junction_reuse.py`: F0, F1, F2 and the spacing ablation, all off by default.
  * The hooks in `graphs/tc_sampler.evaluate_candidate` and `graphs/m14_sampler.net_candidate_lw`.
* **Defaults.** With every option off the samplers are unchanged: identical graphs with `event_log` on and off;
  bit-identity replay of the archived M14 runs re-checked at the end of M17.

## 1. How the generator creates junctions

At each step, a proposed move u → u + d is turned into geometry by fixed insertion rules (M5, `GenState.resolve_step`):

1. **Clip** to the tile.
2. **Crossing split:** if the move crosses an existing street, it stops there and the street is split.
3. **Merge:** if it ends within 8 m of an existing node, it joins it.
4. **T-snap:** if it ends within 8 m of a street, it joins with a T.
5. Otherwise it creates a **new node**.

**Training never contains rules 2 and 4.** Teacher-forced sequences are breadth-first walks over planar real graphs
(`graphs/sequence.py`), so every target lands exactly on a real node; merges, i.e. loop closings, do occur. The model
is therefore never trained on the consequences of cutting or snapping onto a street.

## 2. Diagnosis

Development, terrain-pack windows, seeds 1–10; `outputs/m17/diagnosis.md`.

* **Close pairs (< 15 m) by the edit that created the later junction:**
  * T-snaps 23–36 %;
  * crossing splits 18–30 %;
  * u-branch (a node already near a junction gains a branch) 23–41 %;
  * merges 5–13 %.
* **Feasibility:** reuse of the nearby junction was feasible for 52–88 % of split births. For about half of all close
  births, a feasible candidate without a close birth existed.
* **NA network term:**
  * close junctions rise with its weight (NA-SMC rebuilt close15: 0.120 at weight 0, 0.175 at weight 1 and 2);
  * chosen closures are only 0.1–0.2 % of edits;
  * so the effect runs through the dead-end STOP penalty (more continuation, more junctions per km²: 97 → 146), not
    through closure rewards.
* **Matched density** (excess over real windows of the same junction density):
  * none for A / M\* / M13-SMC / NA-fixed / NA-random;
  * +0.03 to +0.05 for NA-myopic / NA-adaptive / NA-SMC;
  * organic still +0.05 at a realistic density;
  * radial's excess disappears.

## 3. Fixes

| Fix | Rule | Status |
|---|---|---|
| **F1 topology-aware junction reuse** | A move that would create a junction (split node, or a degree-2 node reached by a merge) within r of an existing junction j is replaced by u → j when: j is on the same street near the contact (an endpoint of the split edge, or reachable from it at grade within r; for merges within 1.5 r); u → j resolves to exactly j; ≥ 8 m; no duplicate edge or self-loop; ≥ 30° from every branch at j. The hard rules then score it like any candidate. Otherwise the original edit stays | adopted; r = 15 m by the frozen rule |
| **Spacing ablation** | The same moves are infeasible instead of redirected | ablation |
| **F0 post-hoc consolidation** | Topology-aware clusters (pairwise ≤ τ, at-grade path ≤ 1.5 τ) collapsed to their centroid; edges re-attached | cleanup baseline |
| **F2 closure discount** | The dead-end closure reward counts only if the closure shortens the at-grade u–v distance by ≥ 25 % | not supported by the diagnosis; development check only (no effect) |
| **F3 training on edit-consistent sequences** | — | not run: training graphs already carry an 8 m snap, and the gap comes from generation-only edits |

* **u-branch births are not addressed by F1.** That would need the earlier placement of u to change; left as future
  work.

## 4. Relation to prior work (`docs/M17_LITERATURE.md`)

Junction snapping during generation is old:
* Parish & Müller 2001: local constraints.
* NTG 2019: a new node within 5 m joins the existing node.
* CitySprout: intersect radius.
* RoadTracer: merge only if graph distance ≥ 6 steps, i.e. topology-aware.

Post-hoc topological consolidation is standard (Boeing 2025; OSMnx `rebuild_graph=True`).

**What M16–M17 add is narrow:**
* the edit-level attribution of close junctions in a learned sequential generator (rules absent from training);
* the separation of geometry, network-reward and density causes by matched interventions;
* a paired holdout comparison of online reuse against a post-hoc cleanup, at matched compute.
