"""Baseline P (Milestone 12): classical terrain-aware procedural streets, no learning.

After Galin et al. (2010, 2011) - streets as least-cost paths whose cost grows with slope and crossing water - with a
simple network layout:
1. junction seeds: Poisson-disk points on dry land with data, radius 0.9 / sqrt(density); the density is drawn from
   the TRAINING distribution of real junctions per km^2 (the baseline never sees the evaluated tile's streets);
2. layout: the Gabriel graph of the seeds (a standard proximity graph, mean degree about 4);
3. routing: Dijkstra on an 8 m grid (8-neighbour); step cost = length * (1 + ALPHA * (grade / g_max)^2); grade above
   1.5 * g_max is impassable; water costs BETA per metre and a crossing longer than the class span limit is rejected;
4. paths are smoothed (Chaikin, 2 passes); pieces over water become BRIDGE edges.
Curvature is not penalised inside the search (that would need direction states); the smoothing stands in for it.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay

from urbanmorph.graphs.levels import BRIDGE, GROUND
from urbanmorph.graphs.road_graph import TILE_M, build_graph
from urbanmorph.terrain.heightfield import Terrain
from urbanmorph.terrain.road_feasibility import Thresholds, water_kind_at

ALPHA, BETA, RES, CLS = 4.0, 6.0, 8.0, 2


def poisson_disk(rng, radius, ok_fn, tries=30, size=TILE_M):
    pts = []
    cand = rng.uniform(0, size, (int(size * size / radius ** 2 * 3), 2))
    for p in cand:
        if ok_fn(p) and all(np.hypot(*(p - q)) >= radius for q in pts):
            pts.append(p)
    return np.array(pts).reshape(-1, 2)


def gabriel(pts):
    if len(pts) < 3:
        return [(0, 1)] if len(pts) == 2 else []
    tri = Delaunay(pts)
    edges = {tuple(sorted((s[i], s[j]))) for s in tri.simplices for i, j in ((0, 1), (1, 2), (0, 2))}
    out = []
    for a, b in edges:
        m, r2 = (pts[a] + pts[b]) / 2, np.sum((pts[a] - pts[b]) ** 2) / 4
        d2 = np.sum((pts - m) ** 2, 1)
        d2[[a, b]] = np.inf
        if (d2 >= r2).all():
            out.append((a, b))
    return out


def grid_graph(t: Terrain, th: Thresholds):
    n = int(TILE_M / RES)
    xc = (np.arange(n) + 0.5) * RES
    X, Y = np.meshgrid(xc, TILE_M - xc)
    z = t.sample(X.ravel(), Y.ravel()).reshape(n, n)
    wet = water_kind_at(t, X.ravel(), Y.ravel()).reshape(n, n) > 0
    gmax = th.grade_max[CLS]
    rows, cols, w = [], [], []
    idx = np.arange(n * n).reshape(n, n)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1, c0, c1 = max(0, -dr), n - max(0, dr), max(0, -dc), n - max(0, dc)
        a = idx[r0:r1, c0:c1].ravel()
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel()
        L = RES * np.hypot(dr, dc)
        za, zb = z.ravel()[a], z.ravel()[b]
        g = np.abs(zb - za) / L
        cost = L * (1 + ALPHA * (g / gmax) ** 2)
        ww = wet.ravel()[a] | wet.ravel()[b]
        cost = np.where(ww, L * BETA, cost)
        bad = (~ww & (g > 1.5 * gmax)) | ~np.isfinite(cost)
        rows += [a[~bad], b[~bad]]
        cols += [b[~bad], a[~bad]]
        w += [cost[~bad], cost[~bad]]
    W = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)).tocsr()
    return W, n, wet


def chaikin(p, passes=2):
    """Corner cutting; the end points are kept."""
    for _ in range(passes):
        if len(p) < 3:
            return p
        a, b = p[:-1], p[1:]
        mid = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2)
        p = np.vstack([p[:1], mid[1:-1], p[-1:]])
    return p


def sample_P(t: Terrain, rng, density_per_km2: float, th: Thresholds | None = None):
    th = th or Thresholds()
    W, n, wet = grid_graph(t, th)

    def ok(p):
        z = t.sample(p[:1], p[1:2])
        return bool(np.isfinite(z).all() and water_kind_at(t, p[:1], p[1:2])[0] == 0)

    radius = 0.9 / np.sqrt(max(density_per_km2, 1.0) / 1e6)
    pts = poisson_disk(rng, radius, ok)
    if len(pts) < 2:
        return build_graph([], classes=[], levels=[], structs=[])
    cell = lambda p: int(min(n - 1, (TILE_M - p[1]) // RES)) * n + int(min(n - 1, p[0] // RES))
    cells = np.array([cell(p) for p in pts])
    D, pred = dijkstra(W, indices=cells, return_predecessors=True)
    lines, lev, st = [], [], []
    for a, b in gabriel(pts):
        if not np.isfinite(D[a, cells[b]]):
            continue
        path, v = [], cells[b]
        while v != cells[a] and v >= 0:
            path.append(v)
            v = pred[a, v]
        path.append(cells[a])
        rc = np.array([divmod(v, n) for v in path[::-1]])
        xy = np.c_[(rc[:, 1] + 0.5) * RES, TILE_M - (rc[:, 0] + 0.5) * RES]
        xy[0], xy[-1] = pts[a], pts[b]
        if np.hypot(*(xy[-1] - xy[0])) and np.sum(np.hypot(*np.diff(xy, axis=0).T)) / np.hypot(*(xy[-1] - xy[0])) > 1.8:
            continue                                            # implausible detour: not built
        w = wet.ravel()[np.array(path[::-1])]
        if w.sum() * RES > th.bridge_span_max[CLS]:
            continue
        xy = chaikin(xy)
        wk = water_kind_at(t, xy[:, 0], xy[:, 1]) > 0
        # split into ground / bridge pieces
        start = 0                                               # runs of wet / dry points; pieces share end points
        for i in range(1, len(xy) + 1):
            if i == len(xy) or wk[i] != wk[start]:
                # bridges run dry bank to dry bank; ground pieces stop at the last dry point
                piece = xy[max(start - 1, 0):min(i + 1, len(xy))] if wk[start] else xy[start:i]
                if len(piece) >= 2:
                    lines.append(piece)
                    lev.append(1 if wk[start] else 0)
                    st.append(BRIDGE if wk[start] else GROUND)
                start = i
    return build_graph(lines, classes=[CLS] * len(lines), levels=lev, structs=st)
