"""Milestone 17 fixes for close junctions (docs/M17_METHOD.md); every option is off by default.

F1 - topology-aware junction reuse (``SamplerConfig.junction_reuse_m`` = r > 0). A resolved ground move that would
create a junction (the split node of a crossing / T-junction, or an existing degree-2 node reached by a merge) closer
than r to an existing junction j is replaced by the move u -> j, resolved by the same insertion rules, when ALL hold:
* j is on the same street near the intended contact: for a split, j is an endpoint of the split edge or reachable from
  one along at-grade edges within r; for a merge at v, j is reachable from v within 1.5 r (otherwise the two junctions
  serve distinct connections and the original edit stays);
* the move u -> j resolves to exactly j (merge or crossing endpoint, so no other street is cut on the way), is at least
  8 m long, j is not u and not already joined to u (no duplicate edge or self-loop);
* the new branch arrives at j at least ``TURN_DEG`` (30 degrees) away from every existing branch of j;
* the caller's hard rules then score the replaced edge (terrain, water, structures) as for any candidate.
Otherwise the original edit is kept. The fixed-spacing ablation (``junction_min_m`` = r) instead makes such a move
infeasible (reason "spacing").

F0 - post-hoc topology-aware consolidation (``consolidate_graph``): the clusters of graphs/junctions.consolidate are
collapsed to their centroid; edges are re-attached, self-loops and duplicates dropped. A cleanup pass baseline.

F2 - closure discount (NetConfig.closure_gain > 0): the M14 dead-end closure reward counts only when the closure
shortens the at-grade network distance between its two ends by at least that fraction (no reward for a short
redundant connector next to an existing route); a heuristic, not part of the M14 potentials' derivation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from urbanmorph.graphs.road_graph import RoadGraph

TURN_DEG = 30.0
MIN_EDGE_M = 8.0
CLOSING = ("merge_node", "crossing_endpoint")
SPLITS = ("crossing_split", "t_junction")


def _ground_dist(s, src: int, limit: float) -> np.ndarray:
    """At-grade network distances from node src (state s), up to limit."""
    e = s.live_edges()
    if not len(e):
        return np.full(s.n, np.inf)
    keep = np.array([(min(a, b), max(a, b)) not in s.raised for a, b in e], bool) if s.raised else np.ones(len(e), bool)
    e = e[keep]
    L = np.hypot(*(s.nodes[e[:, 0]] - s.nodes[e[:, 1]]).T)
    A = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(s.n, s.n)).tocsr()
    return dijkstra(A, indices=src, limit=limit)


def contact(s, u: int, o: dict):
    """(point that becomes a junction, its anchor nodes for the same-street test) of a resolved ground move, or None."""
    if o["kind"] in SPLITS:
        return np.asarray(o["q"], float), list(o["edge"])
    if o["kind"] in CLOSING and o.get("v", -1) >= 0 and len(s.adj[o["v"]]) == 2:
        return s.nodes[o["v"]].copy(), [int(o["v"])]
    return None


def nearest_junction(s, x: np.ndarray, r: float, exclude=()):
    J = [k for k in range(s.n) if len(s.adj[k]) >= 3 and k not in exclude]
    if not J:
        return None
    d = np.hypot(*(s.nodes[J] - x).T)
    k = int(np.argmin(d))
    return (J[k], float(d[k])) if d[k] < r else None


def reuse(s, u: int, o: dict, r: float):
    """The replacement resolution (dict) of F1, or None to keep ``o``."""
    c = contact(s, u, o)
    if c is None:
        return None
    x, anchors = c
    nj = nearest_junction(s, x, r, exclude=(u,))
    if nj is None:
        return None
    j, _ = nj
    if j in s.adj[u]:
        return None
    reach = r if o["kind"] in SPLITS else 1.5 * r
    if j not in anchors:
        ok = False
        for a in anchors:
            if _ground_dist(s, a, reach)[j] <= reach:
                ok = True
                break
        if not ok:
            return None                                   # distinct connection: keep the original edit
    dd = s.nodes[j] - s.nodes[u]
    if math.hypot(*dd) < MIN_EDGE_M:
        return None
    o2 = s.resolve_step(u, dd)
    if o2["kind"] not in CLOSING or int(o2.get("v", -1)) != j:
        return None
    back = math.atan2(*(s.nodes[u] - s.nodes[j])[::-1])
    for w in s.adj[j]:
        b = math.atan2(*(s.nodes[w] - s.nodes[j])[::-1])
        if abs((back - b + math.pi) % (2 * math.pi) - math.pi) < math.radians(TURN_DEG):
            return None
    o2 = dict(o2)
    o2["reused_from"] = o["kind"]
    return o2


def too_close(s, u: int, o: dict, r: float) -> bool:
    """Fixed-spacing ablation: does the move create a junction within r of an existing junction?"""
    c = contact(s, u, o)
    return c is not None and nearest_junction(s, c[0], r, exclude=(u,)) is not None


def closure_gain(s, u: int, v: int, length: float) -> float:
    """Fractional shortening of the at-grade network distance between u and v by a new edge of ``length``."""
    d = _ground_dist(s, u, np.inf)[v]
    if not np.isfinite(d) or d <= 0:
        return 1.0
    return max(0.0, (d - length) / d)


def consolidate_graph(g: RoadGraph, tau: float) -> RoadGraph:
    """F0: collapse topology-aware junction clusters (graphs/junctions.consolidate) to their centroids."""
    from urbanmorph.graphs.junctions import clean, consolidate
    groups = consolidate(g, tau, window=(-1e9, -1e9, 1e9, 1e9))["groups"]
    nodes, e, st = clean(g)
    if not groups:
        return g
    rep = np.arange(len(nodes))
    P = nodes.copy()
    for m in groups:
        c = nodes[m].mean(0)
        for k in m:
            rep[k] = m[0]
        P[m[0]] = c
    e2 = rep[e]
    keep = e2[:, 0] != e2[:, 1]
    e2, st2 = np.sort(e2[keep], 1), st[keep]
    _, first = np.unique(e2, axis=0, return_index=True)
    first = np.sort(first)
    e2, st2 = e2[first], st2[first]
    used = np.unique(e2.ravel())
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    cls = np.full(len(e2), 2)
    return RoadGraph(P[used], remap[e2], np.zeros(len(used), bool), edge_class=cls, edge_struct=st2)
