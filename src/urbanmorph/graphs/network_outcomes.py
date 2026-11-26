"""Milestone 16 independent network outcomes (docs/M16_METRICS.md): measures that no style rule optimises directly.

The style rules act on headings, lengths, junction placement and termination; these outcomes describe the resulting
network. All take a RoadGraph in tile metres (512 m tiles); faces and dead ends touching the tile edge are censored
as in graphs/style_metrics.py.

* Blocks (Louf & Barthelemy 2014): interior faces; ``block_logA_med`` / ``block_logA_sd`` (natural log of m2), shape
  factor Phi = area / area of the minimum bounding circle (``block_phi_med``; a square is 2/pi = 0.64), and
  ``block_rect_share``: faces that simplify (3 m) to 4 corners each within 90 +- 15 degrees.
* Junction angles: at interior junctions (degree >= 3), the angles between bearing-consecutive incident edges;
  ``jangle_right_share`` within 90 +- 15 degrees, ``jangle_sd_deg``.
* Strokes (natural streets; Porta et al. 2006 / Jiang 2009 style): at every node incident edges are paired greedily
  by smallest deflection below 30 degrees and chained. ``stroke_len_lw`` (length-weighted mean stroke length, m),
  ``stroke_top10_share`` (length share of the longest 10 % of strokes: hierarchy), ``stroke_xjunc_lw``
  (length-weighted mean number of junctions a stroke passes THROUGH: multi-block continuity) and
  ``through_share`` (junctions where some stroke continues straight through).
* Terminal branches: the 2-core (degree-1 nodes pruned repeatedly; tile-edge nodes never pruned) is the looped
  backbone. ``tree_len_share`` (street length outside it), ``term_len_med`` (median network distance from an interior
  dead end to the backbone, m), ``term_branch_med`` (median junctions on that path), ``term_attached_share`` (dead
  ends whose tree reaches a backbone at all).
* Radial structure about ``centre``: strokes whose length-weighted |cos(bearing - radial)| >= cos 20 are spokes;
  ``spoke_span_lw`` (length-weighted radial extent / 256 m), ``spokes_long`` (spokes spanning >= 0.4 x 256 m);
  ``ring_cover_max`` (largest angular coverage, 0-1, of ring-direction edges in any 48 m radius band from 48 m);
  ``sector_circ`` / ``sector_unreach`` (circuity and unreachable share of node pairs in different quadrants).
* ``phi_boeing``: Boeing's orientation order, 1 - ((H - ln 4) / (ln 36 - ln 4))^2 with H the entropy of the
  length-weighted bidirectional bearing histogram (36 bins).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph, is_border

DEFLECT_DEG = 30.0
RIGHT_TOL = 15.0
R_TILE = TILE_M / 2


def _clean(g: RoadGraph) -> np.ndarray:
    if len(g.edges) == 0:
        return np.zeros((0, 2), int)
    e = np.unique(np.sort(np.asarray(g.edges, int), axis=1), axis=0)
    return e[e[:, 0] != e[:, 1]]


def _adj(g: RoadGraph, e: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(g.n)]
    for a, b in e:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _bearing(p, q) -> float:
    return math.atan2(q[1] - p[1], q[0] - p[0])


# ---------------------------------------------------------------- blocks
def blocks(g: RoadGraph, margin: float = 2.0) -> dict:
    import shapely
    from shapely.ops import polygonize, unary_union
    out = {"block_logA_med": np.nan, "block_logA_sd": np.nan, "block_phi_med": np.nan, "block_rect_share": np.nan}
    e = _clean(g)
    if len(e) < 3:
        return out
    faces = [f for f in polygonize(unary_union(shapely.linestrings(np.stack([g.nodes[e[:, 0]], g.nodes[e[:, 1]]], 1))))
             if min(f.bounds[:2]) > margin and max(f.bounds[2:]) < TILE_M - margin and f.area > 1.0]
    if not faces:
        return out
    A = np.array([f.area for f in faces])
    phi = np.array([f.area / (math.pi * shapely.minimum_bounding_radius(f) ** 2) for f in faces])
    rect = []
    for f in faces:
        c = np.asarray(f.simplify(3.0).exterior.coords)[:-1]
        if len(c) != 4:
            rect.append(False)
            continue
        ang = []
        for k in range(4):
            u, v = c[k - 1] - c[k], c[(k + 1) % 4] - c[k]
            ang.append(math.degrees(math.acos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1))))
        rect.append(all(abs(a - 90) <= RIGHT_TOL for a in ang))
    out.update(block_logA_med=float(np.median(np.log(A))), block_logA_sd=float(np.std(np.log(A))) if len(A) > 1 else np.nan,
               block_phi_med=float(np.median(phi)), block_rect_share=float(np.mean(rect)))
    return out


# ---------------------------------------------------------------- junction angles
def junction_angles(g: RoadGraph) -> dict:
    e = _clean(g)
    adj = _adj(g, e)
    inner = ~is_border(g.nodes) if g.n else np.zeros(0, bool)
    angs = []
    for v in range(g.n):
        if len(adj[v]) < 3 or not inner[v]:
            continue
        b = sorted(_bearing(g.nodes[v], g.nodes[w]) for w in adj[v])
        d = np.diff(np.r_[b, b[0] + 2 * math.pi])
        angs += list(np.degrees(d))
    if not angs:
        return {"jangle_right_share": np.nan, "jangle_sd_deg": np.nan}
    a = np.array(angs)
    return {"jangle_right_share": float(np.mean(np.abs(a - 90) <= RIGHT_TOL)), "jangle_sd_deg": float(np.std(a))}


# ---------------------------------------------------------------- strokes
def strokes(g: RoadGraph):
    """List of strokes, each a node path; and the set of junctions some stroke passes through."""
    e = _clean(g)
    adj = _adj(g, e)
    P = g.nodes
    pair: dict = {}                                   # (v, w) -> u: the stroke entering v from w leaves to u
    through = set()
    for v in range(g.n):
        nb = adj[v]
        if len(nb) < 2:
            continue
        cand = []
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, b = P[nb[i]] - P[v], P[nb[j]] - P[v]
                c = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
                dev = 180.0 - math.degrees(math.acos(np.clip(c, -1, 1)))
                if dev < DEFLECT_DEG:
                    cand.append((dev, nb[i], nb[j]))
        used = set()
        for dev, a, b in sorted(cand):
            if a in used or b in used:
                continue
            used |= {a, b}
            pair[(v, a)], pair[(v, b)] = b, a
            if len(nb) >= 3:
                through.add(v)
    seen, out = set(), []
    for a, b in e:
        if (a, b) in seen:
            continue
        path = [a, b]                                  # extend forward from b, then backward from a
        prev, cur = a, b
        while (cur, prev) in pair and (cur, pair[(cur, prev)]) not in seen and len(path) <= g.n:
            nxt = pair[(cur, prev)]
            seen |= {(cur, nxt), (nxt, cur)}
            path.append(nxt)
            prev, cur = cur, nxt
        prev, cur = b, a
        back = []
        while (cur, prev) in pair and (cur, pair[(cur, prev)]) not in seen and len(back) <= g.n:
            nxt = pair[(cur, prev)]
            seen |= {(cur, nxt), (nxt, cur)}
            back.append(nxt)
            prev, cur = cur, nxt
        seen |= {(a, b), (b, a)}
        out.append(back[::-1] + path)
    return out, through


def _plen(P, path) -> float:
    return float(np.linalg.norm(np.diff(P[path], axis=0), axis=1).sum())


def stroke_stats(g: RoadGraph, st=None) -> dict:
    out = {"stroke_len_lw": np.nan, "stroke_top10_share": np.nan, "stroke_xjunc_lw": np.nan, "through_share": np.nan}
    if len(_clean(g)) == 0:
        return out
    S, through = strokes(g) if st is None else st
    P = g.nodes
    deg = np.bincount(_clean(g).ravel(), minlength=g.n)
    inner = ~is_border(P)
    L = np.array([_plen(P, s) for s in S])
    X = np.array([sum(1 for v in s[1:-1] if deg[v] >= 3) for s in S])
    o = np.sort(L)[::-1]
    k = max(1, int(math.ceil(0.1 * len(o))))
    junc = np.flatnonzero((deg >= 3) & inner)
    out.update(stroke_len_lw=float((L * L).sum() / L.sum()), stroke_top10_share=float(o[:k].sum() / o.sum()),
               stroke_xjunc_lw=float((L * X).sum() / L.sum()),
               through_share=float(np.mean([v in through for v in junc])) if len(junc) else np.nan)
    return out


# ---------------------------------------------------------------- terminal branches
def terminal(g: RoadGraph) -> dict:
    out = {"tree_len_share": np.nan, "term_len_med": np.nan, "term_branch_med": np.nan, "term_attached_share": np.nan}
    e = _clean(g)
    if len(e) == 0:
        return out
    adj = _adj(g, e)
    P = g.nodes
    border = is_border(P)
    deg = np.array([len(a) for a in adj])
    d = deg.copy()
    alive = deg > 0
    stack = [v for v in range(g.n) if d[v] == 1 and not border[v]]
    while stack:
        v = stack.pop()
        if not alive[v] or d[v] > 1:
            continue
        alive[v] = False
        for w in adj[v]:
            if alive[w]:
                d[w] -= 1
                if d[w] <= 1 and not border[w]:
                    stack.append(w)
    core = alive & (d >= 2) | (alive & border)
    Lall = np.hypot(*(P[e[:, 0]] - P[e[:, 1]]).T)
    in_core = core[e[:, 0]] & core[e[:, 1]]
    out["tree_len_share"] = float(Lall[~in_core].sum() / Lall.sum())
    dead = np.flatnonzero((deg == 1) & ~border)
    if len(dead) == 0:
        return out
    A = coo_matrix((np.r_[Lall, Lall], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(g.n, g.n)).tocsr()
    D, pred = dijkstra(A, indices=dead, return_predecessors=True)
    cidx = np.flatnonzero(core)
    lens, branches, att = [], [], []
    for i, v in enumerate(dead):
        if len(cidx) == 0 or not np.isfinite(D[i, cidx]).any():
            att.append(False)
            continue
        t = cidx[np.argmin(D[i, cidx])]
        att.append(True)
        lens.append(float(D[i, t]))
        n, x = 0, pred[i, t]
        while x >= 0 and x != v:
            n += int(deg[x] >= 3)
            x = pred[i, x]
        branches.append(n)
    out["term_attached_share"] = float(np.mean(att))
    if lens:
        out["term_len_med"] = float(np.median(lens))
        out["term_branch_med"] = float(np.median(branches))
    return out


# ---------------------------------------------------------------- radial structure
def radial_structure(g: RoadGraph, centre, rng=None, st=None, n_pairs: int = 300) -> dict:
    out = {"spoke_span_lw": np.nan, "spokes_long": 0, "ring_cover_max": np.nan, "sector_circ": np.nan,
           "sector_unreach": np.nan}
    e = _clean(g)
    if len(e) == 0 or centre is None:
        return out
    c = np.asarray(centre, float)
    P = g.nodes
    S, _ = strokes(g) if st is None else st
    lim = math.cos(math.radians(20.0))
    spans, Ls = [], []
    for s in S:
        q = P[s]
        seg = np.diff(q, axis=0)
        L = np.linalg.norm(seg, axis=1)
        mid = (q[1:] + q[:-1]) / 2
        rv = mid - c
        rn = np.linalg.norm(rv, axis=1)
        ok = rn > 1e-6
        if L.sum() == 0 or not ok.any():
            continue
        cosr = np.abs((seg[ok] * rv[ok]).sum(1)) / np.maximum(L[ok] * rn[ok], 1e-9)
        if (L[ok] * cosr).sum() / L[ok].sum() >= lim:
            r = np.linalg.norm(q - c, axis=1)
            spans.append((r.max() - r.min()) / R_TILE)
            Ls.append(L.sum())
    if Ls:
        out["spoke_span_lw"] = float(np.dot(spans, Ls) / np.sum(Ls))
        out["spokes_long"] = int(sum(x >= 0.4 for x in spans))
    a, b = P[e[:, 0]], P[e[:, 1]]
    m = (a + b) / 2
    rv = m - c
    rn = np.linalg.norm(rv, axis=1)
    seg = b - a
    L = np.linalg.norm(seg, axis=1)
    sin_r = np.sqrt(np.clip(1 - ((seg * rv).sum(1) / np.maximum(L * rn, 1e-9)) ** 2, 0, 1))
    tang = (sin_r >= lim) & (rn >= 48.0)
    best = 0.0
    for r0 in np.arange(48.0, R_TILE, 48.0):
        k = tang & (rn >= r0) & (rn < r0 + 48.0)
        if not k.any():
            continue
        cov = np.zeros(360, bool)
        for p, q in zip(a[k], b[k]):
            t0, t1 = (math.degrees(math.atan2(*(x - c)[::-1])) % 360 for x in (p, q))
            lo, hi = sorted((t0, t1))
            if hi - lo > 180:
                cov[int(hi):] = True
                cov[:int(lo) + 1] = True
            else:
                cov[int(lo):int(hi) + 1] = True
        best = max(best, float(cov.mean()))
    out["ring_cover_max"] = best
    rng = np.random.default_rng(0) if rng is None else rng
    used = np.flatnonzero(np.bincount(e.ravel(), minlength=g.n) > 0)
    if len(used) >= 2:
        A = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(g.n, g.n)).tocsr()
        quad = (P[:, 0] >= c[0]).astype(int) + 2 * (P[:, 1] >= c[1]).astype(int)
        i = rng.choice(used, 4 * n_pairs)
        j = rng.choice(used, 4 * n_pairs)
        eu = np.hypot(*(P[i] - P[j]).T)
        k = np.flatnonzero((quad[i] != quad[j]) & (eu >= 100.0))[:n_pairs]
        if len(k) >= 10:
            src, inv = np.unique(i[k], return_inverse=True)
            nd = dijkstra(A, indices=src)[inv, j[k]]
            fin = np.isfinite(nd)
            out["sector_unreach"] = float(1 - fin.mean())
            out["sector_circ"] = float(np.median(nd[fin] / eu[k][fin])) if fin.any() else np.nan
    return out


def phi_boeing(g: RoadGraph, bins: int = 36) -> float:
    e = _clean(g)
    if len(e) == 0:
        return np.nan
    d = g.nodes[e[:, 1]] - g.nodes[e[:, 0]]
    L = np.hypot(*d.T)
    th = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    th = np.r_[th, th + 180.0]                          # bidirectional
    h = np.histogram((th + 5.0) % 360.0, bins=bins, range=(0, 360), weights=np.r_[L, L])[0]
    p = h[h > 0] / h.sum()
    H = float(-(p * np.log(p)).sum())
    return float(1 - ((H - math.log(4)) / (math.log(bins) - math.log(4))) ** 2)


def outcome_row(g: RoadGraph, centre=None, rng=None) -> dict:
    st = strokes(g) if len(_clean(g)) else ([], set())
    c = (TILE_M / 2, TILE_M / 2) if centre is None else centre
    return {**blocks(g), **junction_angles(g), **stroke_stats(g, st), **terminal(g),
            **radial_structure(g, c, rng, st), "phi_boeing": phi_boeing(g)}
