# Milestone 17 junction metrics: definitions, conventions and checks

* **Code:** `src/urbanmorph/graphs/junctions.py`.
* **Tests:** `tests/test_m17_junctions.py`, 9 fixtures with known answers including an OSMnx 2.1.1 cross-check.
* **Audit:** `scripts/m17_junction_audit.py` → `outputs/m17/junction_audit.md`.

## 1. Representation

| Item | Convention |
|---|---|
| Coordinates | Projected metres (UTM); 512 m tiles, x east, y north |
| Road definition | Training definition: OSM `highway` in the public-street list of `terrain/water.STREET_VALUES`; **service roads, footways and parking aisles excluded** (`configs/m16_pack` also stores `__vehicular` and `__pedestrian` variants, used only where stated) |
| Real graphs | OSM lines → `build_graph`: resampled at ≤ 30 m, greedy snap of points to centres ≥ 8 m apart (endpoints first; not transitive). Real junctions are therefore almost never < 8 m apart |
| Generated graphs | **RAW**: as produced. **REBUILT**: every edge passed through the same `build_graph` as the real data. Reported side by side |
| Grade separation | Edges carry a structure code (0 ground, 1 bridge, 2 tunnel). Overpasses share no node, in OSM-derived and in generated graphs |
| Duplicates | Nodes < 5 cm apart are one node; self-loops and repeated edges are dropped |

## 2. Junction

* A node with **≥ 3 unique neighbours** (incident street segments after de-duplication). Shape points and bends
  (degree 2) are excluded.
* Boeing (2025) counts intersections as nodes of degree > 1 after simplification, so 2-way joins of two named
  streets also count; ours is stricter (≥ 3).
* OSMnx's `street_count` counts physical segments (parallel edges each count).

## 3. Spacing statistics

* **Focal junctions:** ≥ 40 m inside the window (minus sampling). Their nearest neighbour is searched among all
  junctions of the window, or of the buffered domain when one exists. Every distance up to 40 m is therefore exact.
* **close_τ:** share of focal junctions with another junction at **pairwise** distance < τ.
  * **close15** = τ 15 m.
  * **nn p10:** 10th percentile of focal nearest-neighbour distances.
* **Summaries:** **pooled** (all focal junctions of a set) and **window** (mean of per-window values over windows with
  ≥ 2 focal junctions). The number of windows with < 2 focal junctions is always reported.
  * Pooled close15 and pooled p10 come from the same distances: close15 > 0.10 implies p10 < 15 m.
  * Window-mean p10 is not bound by that relation and can be much larger (training: 35 m window, 12 m pooled).
* **Sweep:** τ ∈ {5, 7.5, 10, 12.5, 15, 20, 25, 30, 40} m, always reported.

## 4. Topology-aware consolidation (consol10)

* **Link rule:** two junctions are linked when pairwise distance ≤ τ (10 m) **and** an at-grade path (structure 0
  only) of length ≤ 1.5 τ joins them.
* **Clusters:** connected components of the links (single linkage; chains allowed; the largest cluster diameter is
  reported).
* **consol10:** focal junctions ÷ clusters containing a focal junction.
* **Relation to OSMnx:** OSMnx `consolidate_intersections(tolerance=r)` buffers each node by r, so nodes < 2r apart
  merge. Our τ = 2r. With `rebuild_graph=True` OSMnx splits clusters into weakly connected induced subgraphs;
  Boeing 2025 §3.2.2 calls this topological consolidation. Our path condition is a stricter, length-bounded version
  of the same idea. The test suite checks the buffer semantics on a divided carriageway (6.5 m buffers merge the 12 m
  pair; 5.5 m do not).

## 5. Fixtures (known answers)

| Fixture | Result |
|---|---|
| True close T pair (8 m) | close15 = 1; merged at τ 10 |
| Chamfered corner (10 m square) | 4 junctions → 1 cluster (diameter 14.1 m) |
| Divided carriageway (12 m) | kept at τ 10, merged at τ 12 |
| Short block (30 m) | never close; never merged at τ ≤ 15 |
| Overpass, 4 m in plan, 300 m at grade | counted by the geometric share, **never merged** |
| Chain of 5 junctions every 8 m | 1 cluster, diameter 32 m |
| Duplicated nodes | one junction |
| Boundary clipping (< 40 m from the edge) | not focal; the window is reported as having < 2 focal junctions |
