"""Milestone 14 connectivity and coverage metrics (docs/M14_EXPERIMENTAL_PROTOCOL.md §5). New names only; the M13
metrics (graphs/m13_metrics.py) are unchanged and still computed.

A 512 m tile is a WINDOW onto a larger network: a real street that leaves the window is connected outside it. The M13
``giant_share`` and ``unreachable_share`` treat the window as closed, so real tiles look fragmented (DEV-B real giant
share 0.78) and a generator that never leaves the window looks better connected than reality. M14 therefore closes the
window: every node on the tile border joins one virtual "outside" node (a street cut by the window is assumed to
continue). Under this convention a component is a genuine fragment only if it never reaches the border.

Per tile (generated or real graph, ORIGINAL terrain):
* ``closed_giant``      length share of the largest component after closing the window;
* ``islands``           components (>= 40 m of street) that do not touch the border and are not the closed giant;
* ``island_share``      their length share;
* OD reachability: random street-point pairs 150-500 m apart (as M13), on the CLOSED graph.
  ``od_unreach``        share of pairs with no path (also not via the outside);
  ``od_unreach_land``   share of pairs with no path whose two points lie on the SAME dry-land region (4 m raster,
                        8-connected; generation failures: a path was possible without any water crossing);
  ``od_unreach_water``  share of pairs with no path whose points lie on DIFFERENT dry-land regions (separation that
                        needs a bridge; not counted as a failure);
                        An empty network counts as od_unreach = od_unreach_land = 1 (premature termination);
  ``od_detour``         median network / straight distance of connected pairs, path through the outside excluded
                        (the outside has no length, so such paths are not measurable);
* ``coverage_60``       share of buildable land (dry, landform slope < BUILDABLE_SLOPE) within 60 m of a street;
* ``street_m_per_conn`` street length (m) per unit of closed-giant share: road length spent per connectivity.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from urbanmorph.graphs.road_graph import TILE_M, is_border

BUILDABLE_SLOPE = 0.30      # landform slope above which land is not counted as buildable (assumption, see protocol)
COVER_M = 60.0
RES = 4.0
N = int(TILE_M / RES)


def _lengths(g):
    return np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T)


def closed_components(g):
    """Component labels with the virtual outside node (index n). Returns (labels over n + 1, edge lengths)."""
    n = len(g.nodes)
    L = _lengths(g)
    b = np.flatnonzero(is_border(g.nodes))
    rows = np.r_[g.edges[:, 0], g.edges[:, 1], b, np.full(len(b), n)]
    cols = np.r_[g.edges[:, 1], g.edges[:, 0], np.full(len(b), n), b]
    w = np.r_[L, L, np.full(2 * len(b), 1e-6)]
    A = coo_matrix((w, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    _, lab = connected_components(A, directed=False)
    return lab, L, A


def dry_regions(tf):
    """8-connected dry-land regions of the tile (4 m raster, row 0 = north). 0 = water."""
    wk = getattr(tf.t, "water_kind", None)
    dry = np.ones((N, N), bool) if wk is None else (wk[64:192, 64:192] == 0)
    lab, _ = ndimage.label(dry, structure=np.ones((3, 3)))
    return lab


def _rc(p):
    p = np.atleast_2d(p)
    return (np.clip(((TILE_M - p[:, 1]) // RES).astype(int), 0, N - 1),
            np.clip((p[:, 0] // RES).astype(int), 0, N - 1))


def street_raster(g):
    M = np.zeros((N, N), bool)
    for x0, y0, x1, y1 in (g.segments() if len(g.edges) else []):
        k = max(2, int(np.hypot(x1 - x0, y1 - y0) / (RES / 2)) + 1)
        r, c = _rc(np.c_[np.linspace(x0, x1, k), np.linspace(y0, y1, k)])
        M[r, c] = True
    return M


def coverage(g, tf, radius=COVER_M):
    wk = getattr(tf.t, "water_kind", None)
    dry = np.ones((N, N), bool) if wk is None else (wk[64:192, 64:192] == 0)
    slope = np.hypot(tf.gx, tf.gy)[64:192, 64:192]
    ok = dry & (slope < BUILDABLE_SLOPE)
    if not ok.any():
        return np.nan
    S = street_raster(g)
    if not S.any():
        return 0.0
    dist = ndimage.distance_transform_edt(~S) * RES
    return float((dist[ok] <= radius).mean())


def tile_metrics_m14(g, tf, rng, n_pairs=200, dmin=150.0, dmax=500.0) -> dict:
    out = {"closed_giant": 0.0, "islands": 0, "island_share": 0.0, "od_unreach": np.nan, "od_unreach_land": np.nan,
           "od_unreach_water": np.nan, "od_detour": np.nan, "coverage_60": coverage(g, tf),
           "street_m_per_conn": np.nan}
    if len(g.edges) == 0:                                   # no network at all: a generation failure on land
        out.update(od_unreach=1.0, od_unreach_land=1.0, od_unreach_water=0.0)
        return out
    n = len(g.nodes)
    lab, L, A = closed_components(g)
    k = lab.max() + 1
    cl = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=k)
    gi = int(cl.argmax())
    out["closed_giant"] = float(cl[gi] / L.sum())
    touches = np.zeros(k, bool)
    touches[lab[n]] = True
    isl = (cl >= 40) & ~touches & (np.arange(k) != gi)
    out["islands"] = int(isl.sum())
    out["island_share"] = float(cl[isl].sum() / L.sum())
    out["street_m_per_conn"] = float(L.sum() / max(out["closed_giant"], 1e-9))
    # OD pairs on street nodes; reachability on the closed graph, detour on the open graph
    reg = dry_regions(tf)
    rr, cc = _rc(g.nodes)
    nreg = reg[rr, cc]
    open_A = A[:n, :n]
    src = rng.choice(n, size=min(30, n), replace=False)
    Dopen = dijkstra(open_A, indices=src)
    det, un, un_land, un_water = [], 0, 0, 0
    tot = 0
    for i, s in enumerate(src):
        e = np.hypot(*(g.nodes - g.nodes[s]).T)
        cand = np.flatnonzero((e >= dmin) & (e <= dmax))
        if not len(cand):
            continue
        for j in rng.choice(cand, size=min(7, len(cand)), replace=False):
            tot += 1
            if lab[s] != lab[j]:
                un += 1
                same = nreg[s] > 0 and nreg[s] == nreg[j]
                un_land += int(same)
                un_water += int(not same)
            elif np.isfinite(Dopen[i, j]):
                det.append(Dopen[i, j] / e[j])
        if tot >= n_pairs:
            break
    if tot:
        out.update(od_unreach=un / tot, od_unreach_land=un_land / tot, od_unreach_water=un_water / tot,
                   od_detour=float(np.median(det)) if det else np.nan)
    return out
