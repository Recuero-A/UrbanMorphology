"""Per-tile terrain, water, infrastructure and failure metrics (Milestone 12; definitions in
docs/M12_EVALUATION_PROTOCOL.md §3). The same functions score real and generated graphs on the same terrain.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from urbanmorph.graphs.levels import BRIDGE, TUNNEL
from urbanmorph.graphs.road_graph import TILE_KM2, TILE_M
from urbanmorph.terrain.heightfield import Terrain, grade_stats
from urbanmorph.terrain.road_feasibility import water_kind_at

S_T_KEYS = ["grade_p90", "share_gt10", "contour_index", "steep_density_ratio", "circuity"]


def landform_gradient(t: Terrain, block_m: float = 32.0):
    """(gx, gy) of the terrain block-averaged to ``block_m``, sampled back on the terrain grid."""
    k = int(round(block_m / t.res))
    H = t.elev.shape[0] // k * k
    z = t.elev[:H, :H]
    c = np.nanmean(z.reshape(H // k, k, H // k, k), axis=(1, 3))
    gy, gx = np.gradient(c, block_m)
    up = lambda a: np.repeat(np.repeat(a, k, 0), k, 1)
    return up(gx), -up(gy)


def _weighted_quantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(cw, q), len(o) - 1)])


def tile_metrics(g, t: Terrain, rng=None, river_len_km: float | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    out = {k: np.nan for k in S_T_KEYS}
    st, lv = g.structs(), g.levels()
    if not len(g.edges):                                        # an empty graph: same keys, nothing built
        return {**out, "street_km_per_km2": 0.0, "unbridged_water_m": 0.0, "bridged_water_m": 0.0, "bridges": 0,
                "empty_share": 1.0}
    seg = g.segments()
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    ground = (st != BRIDGE) & (st != TUNNEL) & (lv >= 0)
    mean_g, max_g = np.full(len(L), np.nan), np.full(len(L), np.nan)
    for k in np.flatnonzero(ground):
        s = grade_stats(*t.profile(seg[k, :2], seg[k, 2:]))
        mean_g[k], max_g[k] = s["mean_grade"], s["max_grade"]
    ok = ground & np.isfinite(mean_g)
    if ok.any() and L[ok].sum() > 0:
        out["grade_p90"] = _weighted_quantile(mean_g[ok], L[ok], 0.9)
        out["share_gt10"] = float(L[ok & (max_g > 0.10)].sum() / L[ok].sum())
    gx, gy = landform_gradient(t)
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    r, c = t.rc(mid[:, 0], mid[:, 1])
    r = np.clip(np.round(r).astype(int), 0, gx.shape[0] - 1)
    c = np.clip(np.round(c).astype(int), 0, gx.shape[1] - 1)
    G = np.c_[gx[r, c], gy[r, c]]
    gm = np.hypot(*G.T)
    dvec = (seg[:, 2:] - seg[:, :2]) / np.maximum(L, 1e-9)[:, None]
    sel = ok & np.isfinite(gm) & (gm > 0.05)
    if sel.sum() >= 3:
        out["contour_index"] = float(np.average(np.abs((dvec[sel] * G[sel]).sum(1)) / gm[sel], weights=L[sel]))
    # street density on steep land relative to the tile
    gm_grid = np.hypot(gx, gy)
    tile = t.window(0, 0, TILE_M, TILE_M)
    steep_cells = (gm_grid > 0.10) & tile
    if steep_cells.sum() >= 0.10 * tile.sum():
        steep_len = L[np.isfinite(gm) & (gm > 0.10)].sum()
        dens_tile = L.sum() / TILE_KM2
        dens_steep = steep_len / (steep_cells.sum() * t.res ** 2 / 1e6)
        out["steep_density_ratio"] = float(dens_steep / dens_tile) if dens_tile > 0 else np.nan
    out["circuity"] = circuity(g, L, rng)
    out["street_km_per_km2"] = float(L.sum() / 1000 / TILE_KM2)
    # water and infrastructure
    p = np.linspace(0, 1, 9)[:, None, None]
    pts = (seg[None, :, :2] + p * (seg[None, :, 2:] - seg[None, :, :2]))
    wk = water_kind_at(t, pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(9, -1)
    wet = (wk > 0).any(0)
    out["unbridged_water_m"] = float(L[wet & ground].sum())
    out["bridged_water_m"] = float(L[wet & (st == BRIDGE)].sum())
    out["bridges"] = int((st == BRIDGE).sum())
    if river_len_km:
        out["unbridged_per_river_km"] = out["unbridged_water_m"] / river_len_km
    # empty area: share of 64 m cells farther than 100 m from any street
    cx = np.arange(32, TILE_M, 64.0)
    X, Y = np.meshgrid(cx, cx)
    lines = shapely.linestrings(seg.reshape(-1, 2, 2))
    tree = shapely.STRtree(lines)
    P = shapely.points(np.c_[X.ravel(), Y.ravel()])
    idx, dist = tree.query_nearest(P, return_distance=True, all_matches=False)
    out["empty_share"] = float((dist > 100).mean())
    return out


def circuity(g, L, rng, n_pairs: int = 50, dmin: float = 200.0, dmax: float = 400.0) -> float:
    if g.n < 4:
        return np.nan
    a, b = g.edges[:, 0], g.edges[:, 1]
    W = coo_matrix((np.r_[L, L], (np.r_[a, b], np.r_[b, a])), shape=(g.n, g.n)).tocsr()
    src = rng.choice(g.n, size=min(20, g.n), replace=False)
    D = dijkstra(W, indices=src)
    ratios = []
    for i, s in enumerate(src):
        e = np.hypot(*(g.nodes - g.nodes[s]).T)
        cand = np.flatnonzero((e >= dmin) & (e <= dmax) & np.isfinite(D[i]))
        if len(cand):
            j = rng.choice(cand, size=min(3, len(cand)), replace=False)
            ratios += list(D[i, j] / e[j])
        if len(ratios) >= n_pairs:
            break
    return float(np.median(ratios)) if ratios else np.nan
