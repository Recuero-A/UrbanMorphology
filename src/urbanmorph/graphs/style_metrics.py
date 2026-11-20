"""Milestone 15 street-style and terrain-exposure metrics (docs/M15_METRICS.md gives definitions and units).

All metrics take a RoadGraph in tile metres (projected, 512 m tiles). Border nodes are recomputed geometrically
(``is_border``) for real and generated graphs alike, so dead ends created by the tile boundary are censored the same way.
Edge = one stored graph edge (~26 m generated, resampled real); link = a junction-to-junction chain of edges.

Style diagnostics (joint descriptors; bearing entropy alone confounds radial and organic layouts):
* grid      ``orth_order`` R4 = |sum l e^{4 i theta}| / sum l (1 = two perpendicular directions only, rotation-invariant;
            0 = uniform), with ``share_4way`` (4-way junctions among interior junctions);
* organic   ``sinuosity`` (length-weighted link length / chord, links with chord >= 30 m), ``turn_med_deg`` (median
            heading change at interior degree-2 vertices), with R4 low and ``meshedness`` (connectivity);
* radial    ``radial_share`` / ``tangential_share``: length shares within +-20 degrees of the radial / ring direction
            about a centre (declared, or ``estimate_centre`` by one fixed rule); 0.222 each under uniform bearings;
* cul-de-sac ``deadend_per_km`` (interior dead ends per km of street) and ``loops_per_km2`` (cyclomatic number
            E - V + C per km2).
Terrain exposure (``grade_exposure``): ground edges are sampled every 2 m on the TRUE elevation raster (bilinear, the
raster is 4 m); directional grade over consecutive 10 m sub-spans; length-weighted p90 and the length shares above 10 %
and 15 %. Bridges and tunnels are excluded; sub-spans touching no-data are excluded and their share is reported.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph, is_border

AREA_KM2 = (TILE_M / 1000.0) ** 2
BAND_DEG = 20.0
CHANCE_BAND = 2 * BAND_DEG / 180.0


def _lengths_angles(g: RoadGraph):
    if len(g.edges) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 2))
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    d = b - a
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0]), (a + b) / 2


def _degree(g: RoadGraph) -> np.ndarray:
    return np.bincount(g.edges.ravel(), minlength=g.n) if len(g.edges) else np.zeros(g.n, int)


def street_km(g: RoadGraph) -> float:
    return float(_lengths_angles(g)[0].sum() / 1000.0)


def orth_order(g: RoadGraph) -> float:
    L, th, _ = _lengths_angles(g)
    if L.sum() == 0:
        return np.nan
    return float(abs((L * np.exp(4j * th)).sum()) / L.sum())


def junction_shares(g: RoadGraph) -> dict:
    deg = _degree(g)
    inner = ~is_border(g.nodes) if g.n else np.zeros(0, bool)
    j = deg[(deg >= 3) & inner]
    if len(j) == 0:
        return {"share_3way": np.nan, "share_4way": np.nan, "share_5plus": np.nan, "junctions": 0}
    return {"share_3way": float((j == 3).mean()), "share_4way": float((j == 4).mean()),
            "share_5plus": float((j >= 5).mean()), "junctions": int(len(j))}


def winding(g: RoadGraph, min_chord: float = 30.0) -> dict:
    out = {"sinuosity": np.nan, "turn_med_deg": np.nan}
    if len(g.edges) == 0:
        return out
    adj = [[] for _ in range(g.n)]
    for a, b in g.edges:
        adj[a].append(b), adj[b].append(a)
    deg = np.array([len(x) for x in adj])
    P = g.nodes
    turns = []
    for v in np.flatnonzero(deg == 2):
        d1, d2 = P[v] - P[adj[v][0]], P[adj[v][1]] - P[v]
        c = float(d1 @ d2) / max(float(np.linalg.norm(d1) * np.linalg.norm(d2)), 1e-9)
        turns.append(math.degrees(math.acos(min(1.0, max(-1.0, c)))))
    seen, num, den = set(), 0.0, 0.0
    for u in np.flatnonzero(deg != 2):
        for w in adj[u]:
            if (u, w) in seen:
                continue
            path, prev, cur = [u, w], u, w
            seen.add((u, w))
            while deg[cur] == 2:
                prev, cur = cur, (adj[cur][0] if adj[cur][0] != prev else adj[cur][1])
                path.append(cur)
            seen.add((cur, prev))
            pts = P[path]
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            chord = float(np.linalg.norm(pts[-1] - pts[0]))
            if chord >= min_chord:
                num, den = num + length * length / chord, den + length
    if den:
        out["sinuosity"] = num / den
    if turns:
        out["turn_med_deg"] = float(np.median(turns))
    return out


def radial_tangential(g: RoadGraph, centre) -> dict:
    L, th, m = _lengths_angles(g)
    keep = np.hypot(*(m - np.asarray(centre)).T) >= 30.0 if len(L) else np.zeros(0, bool)   # bearing undefined at centre
    L, th, m = L[keep], th[keep], m[keep]
    if L.sum() == 0:
        return {"radial_share": np.nan, "tangential_share": np.nan}
    rad = np.arctan2(m[:, 1] - centre[1], m[:, 0] - centre[0])
    c = np.abs(np.cos(th - rad))
    lim = math.cos(math.radians(BAND_DEG))
    return {"radial_share": float(L[c >= lim].sum() / L.sum()),
            "tangential_share": float(L[np.sqrt(1 - c ** 2) >= lim].sum() / L.sum())}


def estimate_centre(g: RoadGraph, step: float = 32.0, margin: float = 64.0) -> tuple:
    """The grid point (``step`` m, ``margin`` m from the edge) maximising radial + tangential share: one fixed rule
    applied identically to real and generated graphs."""
    best, arg = -1.0, (TILE_M / 2, TILE_M / 2)
    for x in np.arange(margin, TILE_M - margin + 1e-9, step):
        for y in np.arange(margin, TILE_M - margin + 1e-9, step):
            r = radial_tangential(g, (x, y))
            v = r["radial_share"] + r["tangential_share"]
            if np.isfinite(v) and v > best:
                best, arg = v, (float(x), float(y))
    return arg


def culdesac(g: RoadGraph) -> dict:
    km = street_km(g)
    if g.n == 0 or km == 0:
        return {"deadend_per_km": np.nan, "loops_per_km2": np.nan, "meshedness": np.nan}
    deg = _degree(g)
    inner = ~is_border(g.nodes)
    dead = int(((deg == 1) & inner).sum())
    used = deg > 0
    A = coo_matrix((np.ones(len(g.edges)), (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n))
    ncomp = connected_components(A, directed=False)[0] - int((~used).sum())
    V, E = int(used.sum()), len(np.unique(np.sort(g.edges, axis=1), axis=0))
    cyc = max(E - V + ncomp, 0)
    return {"deadend_per_km": dead / km, "loops_per_km2": cyc / AREA_KM2,
            "meshedness": float(cyc / (2 * V - 5)) if V > 2 else np.nan}


def block_areas(g: RoadGraph, margin: float = 2.0) -> np.ndarray:
    """Areas (m2) of the interior faces of the planar street graph; faces touching the tile boundary are censored."""
    import shapely
    from shapely.ops import polygonize, unary_union
    if len(g.edges) < 3:
        return np.zeros(0)
    lines = shapely.linestrings(np.stack([g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]], axis=1))
    faces = list(polygonize(unary_union(lines)))
    out = []
    for f in faces:
        x0, y0, x1, y1 = f.bounds
        if min(x0, y0) > margin and max(x1, y1) < TILE_M - margin:
            out.append(f.area)
    return np.array(out, float)


def od_circuity(g: RoadGraph, rng, bands=((100, 200), (200, 350), (350, 500)), n_pairs: int = 300) -> dict:
    """Network / Euclidean distance between random node pairs in fixed Euclidean bands (median per band, connected
    pairs only); the share of unreachable pairs is reported separately, never folded into circuity."""
    out = {}
    if g.n < 2 or len(g.edges) == 0:
        for lo, hi in bands:
            out[f"circuity_{lo}_{hi}"], out[f"unreach_{lo}_{hi}"] = np.nan, np.nan
        return out
    L = np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T)
    A = coo_matrix((np.r_[L, L], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])),
                   shape=(g.n, g.n)).tocsr()
    used = np.flatnonzero(_degree(g) > 0)
    for lo, hi in bands:
        i = rng.choice(used, min(n_pairs * 4, 4000))
        j = rng.choice(used, len(i))
        e = np.hypot(*(g.nodes[i] - g.nodes[j]).T)
        k = np.flatnonzero((e >= lo) & (e < hi))[:n_pairs]
        if len(k) < 10:
            out[f"circuity_{lo}_{hi}"], out[f"unreach_{lo}_{hi}"] = np.nan, np.nan
            continue
        src, inv = np.unique(i[k], return_inverse=True)
        D = dijkstra(A, indices=src)
        nd = D[inv, j[k]]
        fin = np.isfinite(nd)
        out[f"circuity_{lo}_{hi}"] = float(np.median(nd[fin] / e[k][fin])) if fin.any() else np.nan
        out[f"unreach_{lo}_{hi}"] = float(1 - fin.mean())
    return out


def grade_exposure(g: RoadGraph, t, step: float = 2.0, span: float = 10.0) -> dict:
    """Directional grade of ground edges on the true raster ``t`` (heightfield.Terrain)."""
    out = {"grade_p90": np.nan, "share_gt10": np.nan, "share_gt15": np.nan, "grade_nodata_share": np.nan,
           "ground_km": 0.0}
    if len(g.edges) == 0:
        return out
    st = np.asarray(g.structs())
    k = max(1, int(round(span / step)))
    grades, weights, nodata, total = [], [], 0.0, 0.0
    for (ia, ib), s in zip(g.edges, st):
        if s != 0:
            continue
        a, b = g.nodes[ia], g.nodes[ib]
        length = float(np.hypot(*(b - a)))
        if length < 1.0:
            continue
        n = max(1, int(math.ceil(length / step)))
        f = np.linspace(0, 1, n + 1)
        z = t.sample(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        ds = length / n
        idx = np.arange(0, n + 1, k)
        if idx[-1] != n:
            idx = np.r_[idx, n]
        for p, q in zip(idx[:-1], idx[1:]):
            seg = (q - p) * ds
            total += seg
            if not (np.isfinite(z[p]) and np.isfinite(z[q])):
                nodata += seg
                continue
            grades.append(abs(z[q] - z[p]) / seg)
            weights.append(seg)
    if not grades:
        return {**out, "grade_nodata_share": 1.0 if total else np.nan}
    G, W = np.array(grades), np.array(weights)
    o = np.argsort(G)
    cw = np.cumsum(W[o]) / W.sum()
    return {"grade_p90": float(G[o][np.searchsorted(cw, 0.9)]), "share_gt10": float(W[G > 0.10].sum() / W.sum()),
            "share_gt15": float(W[G > 0.15].sum() / W.sum()), "grade_nodata_share": nodata / total,
            "ground_km": total / 1000.0}


def field_alignment(g: RoadGraph, sf) -> float:
    """CONTROLLABILITY check (the generator's own objective, not independent evidence): length-weighted mean of
    (1 + cos 4 (theta - theta_field)) / 2 over edges, against the requested style field ``sf`` (graphs/style.StyleField);
    1 = every street on the cross field, 0.5 = expected under uniform bearings."""
    L, th, m = _lengths_angles(g)
    if L.sum() == 0 or not sf.weights:
        return np.nan
    a = np.array([sum(w * (1 + math.cos(4 * (t_ - sf.direction(k, p)))) / 2 for k, w in sf.weights.items())
                  for t_, p in zip(th, m)])
    return float((L * a).sum() / L.sum())


def morphology_row(g: RoadGraph, t=None, centre=None, rng=None, grade_span: float = 10.0) -> dict:
    """Every M15 per-graph metric in one flat dict (``centre`` None: estimated by ``estimate_centre``; ``grade_span``
    the grade sub-span, metres: 10 on the 4 m synthetic rasters, 30 on GLO-30, the source resolution)."""
    rng = np.random.default_rng(0) if rng is None else rng
    c = estimate_centre(g) if centre is None else centre
    row = {"street_km": street_km(g), "orth_order": orth_order(g), **junction_shares(g), **winding(g),
           **{f"{k}": v for k, v in radial_tangential(g, c).items()}, "centre_x": c[0], "centre_y": c[1],
           **culdesac(g), **od_circuity(g, rng)}
    ba = block_areas(g)
    row.update(blocks=int(len(ba)), block_area_med=float(np.median(ba)) if len(ba) else np.nan,
               block_area_p90=float(np.percentile(ba, 90)) if len(ba) else np.nan)
    if t is not None:
        row.update(grade_exposure(g, t, span=grade_span))
    return row
