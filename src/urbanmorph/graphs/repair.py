"""Milestone 13: procedural reconnection REPAIR (a labelled baseline, not learned generation; docs/M13_METHOD.md §7).

After generation, every component that is not the giant one is joined to the nearest node of another component
within MAX_GAP_M by a terrain least-cost path:
* the M12 procedural router (graphs/procedural_terrain.grid_graph): 8 m grid, step cost length * (1 + 4 (g / g_max)^2),
  grade above 1.5 g_max forbidden;
* paths through water are rejected: repair adds NO bridges, by design;
* the path is Chaikin-smoothed, resampled about every 24 m, and added as local (class 2) ground edges.
At most MAX_JOINS joins per tile. Statistics returned: joins, added length, added intersections, attempts, failures.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from urbanmorph.graphs.procedural_terrain import RES, chaikin, grid_graph
from urbanmorph.graphs.road_graph import TILE_M, RoadGraph, is_border

MAX_GAP_M, MAX_JOINS, STEP_M = 150.0, 20, 24.0


def _components(g: RoadGraph):
    n = len(g.nodes)
    if not len(g.edges):
        return np.arange(n), np.zeros(n)
    L = np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T)
    A = coo_matrix((np.ones(len(g.edges)), (g.edges[:, 0], g.edges[:, 1])), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    size = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=k)
    return lab, size


def repair(g: RoadGraph, t, th) -> tuple[RoadGraph, dict]:
    stats = {"joins": 0, "added_m": 0.0, "attempts": 0, "failures": 0, "added_bridges": 0}
    if len(g.edges) == 0 or len(g.nodes) < 2:
        return g, {**stats, "added_intersections": 0}
    W, n, wet = grid_graph(t, th)
    nodes = [p for p in g.nodes]
    edges = [tuple(e) for e in g.edges]
    cls = list(np.asarray(g.classes()) if g.edge_class is not None else np.full(len(edges), 2))
    st = list(np.asarray(g.structs()))
    lv = list(np.asarray(g.levels()))
    deg0 = int((g.degree() >= 3).sum())
    tried = set()
    for _ in range(MAX_JOINS):
        G = RoadGraph(np.array(nodes), np.array(edges, int).reshape(-1, 2), is_border(np.array(nodes)))
        lab, size = _components(G)
        if len(size) <= 1:
            break
        giant = int(np.argmax(size))
        best = None
        for c in np.argsort(size):
            if c == giant or (c in tried):
                continue
            a_idx = np.flatnonzero(lab == c)
            b_idx = np.flatnonzero(lab != c)
            if not len(a_idx) or not len(b_idx):
                continue
            D = np.hypot(*(G.nodes[a_idx][:, None] - G.nodes[b_idx][None]).transpose(2, 0, 1))
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] <= MAX_GAP_M:
                best = (c, int(a_idx[i]), int(b_idx[j]))
                break
            tried.add(c)
        if best is None:
            break
        c, a, b = best
        stats["attempts"] += 1
        cell = lambda p: int(np.clip((TILE_M - p[1]) // RES, 0, n - 1)) * n + int(np.clip(p[0] // RES, 0, n - 1))
        s0, s1 = cell(G.nodes[a]), cell(G.nodes[b])
        dist, pred = dijkstra(W, indices=s0, return_predecessors=True)
        if not np.isfinite(dist[s1]):
            stats["failures"] += 1
            tried.add(c)
            continue
        path, k = [], s1
        while k != s0 and k >= 0:
            path.append(k)
            k = pred[k]
        path.append(s0)
        path = path[::-1]
        rc = np.array([divmod(k, n) for k in path])
        if wet[rc[:, 0], rc[:, 1]].any():
            stats["failures"] += 1
            tried.add(c)
            continue
        pts = np.c_[(rc[:, 1] + 0.5) * RES, TILE_M - (rc[:, 0] + 0.5) * RES]
        pts = np.vstack([G.nodes[a], pts[1:-1], G.nodes[b]]) if len(pts) > 2 else np.vstack([G.nodes[a], G.nodes[b]])
        pts = chaikin(pts)
        Ltot = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        m = max(1, int(round(Ltot[-1] / STEP_M)))
        keep = np.interp(np.linspace(0, Ltot[-1], m + 1), Ltot, np.arange(len(pts)))
        res = np.array([pts[int(np.floor(x))] + (x - np.floor(x)) * (pts[min(int(np.floor(x)) + 1, len(pts) - 1)] - pts[int(np.floor(x))]) for x in keep])
        prev = a
        for p in res[1:-1]:
            nodes.append(p)
            v = len(nodes) - 1
            edges.append((min(prev, v), max(prev, v)))
            cls.append(2), st.append(0), lv.append(0)
            prev = v
        edges.append((min(prev, b), max(prev, b)))
        cls.append(2), st.append(0), lv.append(0)
        stats["joins"] += 1
        stats["added_m"] += float(Ltot[-1])
    out = RoadGraph(np.array(nodes), np.array(edges, int).reshape(-1, 2), is_border(np.array(nodes)), np.array(cls), np.array(lv), np.array(st))
    stats["added_intersections"] = int((out.degree() >= 3).sum()) - deg0
    return out, stats
