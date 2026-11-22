"""Milestone 17 junction metrics: functional junctions, close-junction share, nearest-neighbour spacing and
topology-aware consolidation (docs/M17_METRICS.md). Real and generated graphs go through exactly this code.

Definitions (fixed before any M17 measurement):
* **Representation.** A RoadGraph in projected metres. Nodes closer than ``DUP_M`` (0.05 m) are one node (duplicate
  points); self-loops and repeated edges are dropped. Edges carry a structure code (0 ground, 1 bridge, 2 tunnel).
* **Junction.** A node with at least 3 unique neighbours (incident street branches), any structure code. Shape points
  and bends have degree 2 and are never junctions. A grade-separated crossing shares no node (OSM overpasses and the
  generator's bridges / tunnels pass over or under without a node), so it is never a junction.
* **Edge correction (minus sampling).** Nearest-neighbour statistics are computed for FOCAL junctions only: junctions
  at least ``EDGE_M`` (40 m) from the evaluation window's edge. Their neighbours may be any junction in the window, so
  every distance up to 40 m is exact. When a buffered graph is available (generated domains), the same focal rule is
  applied inside the evaluation window and neighbours come from the whole buffered graph.
* **Close-junction share at tau.** Share of focal junctions whose nearest other junction is closer than tau
  (strictly, pairwise Euclidean distance). ``close15`` = tau 15 m. ``nn_p10`` = 10th percentile of the focal
  nearest-neighbour distances. A sweep over ``SWEEP_M`` is reported.
* **Topology-aware consolidation at tau.** Two junctions are linked when their Euclidean distance is at most tau
  (pairwise distance; NOT a buffer radius: OSMnx's ``tolerance`` is a per-node buffer radius, so its tolerance r groups
  nodes up to 2r apart) AND an at-grade path (structure 0 edges only) joins them with length at most 1.5 tau. Clusters
  are the connected components of these links (single linkage: chains are allowed; the largest cluster diameter is
  reported). Consolidation ratio = focal junctions / clusters that contain a focal junction (1 = nothing to merge).
  The path condition keeps junctions apart that are close only in plan (overpasses, parallel streets with no short
  link).
* **Summaries.** ``pooled`` statistics use all focal junctions of a set together; ``window`` statistics are computed per
  window and averaged over windows with at least ``MIN_FOCAL`` focal junctions. Windows with fewer are counted and
  reported, never silently dropped.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph

DUP_M = 0.05
EDGE_M = 40.0
MIN_FOCAL = 2
SWEEP_M = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0)


def clean(g: RoadGraph):
    """(nodes, edges (E, 2), struct (E,)) with duplicate nodes merged, self-loops and repeated edges dropped."""
    if g.n == 0 or len(g.edges) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, int)
    key = np.round(g.nodes / DUP_M).astype(np.int64)
    _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    nodes = g.nodes[rep]
    e = inv[np.asarray(g.edges, int)]
    st = np.asarray(g.structs(), int) if hasattr(g, "structs") else np.zeros(len(e), int)
    keep = e[:, 0] != e[:, 1]
    e, st = np.sort(e[keep], 1), st[keep]
    _, first = np.unique(e, axis=0, return_index=True)
    first = np.sort(first)
    return nodes, e[first], st[first]


def junction_nodes(nodes, edges) -> np.ndarray:
    deg = np.bincount(edges.ravel(), minlength=len(nodes)) if len(edges) else np.zeros(len(nodes), int)
    return np.flatnonzero(deg >= 3)


def focal_mask(P: np.ndarray, window=(0.0, 0.0, TILE_M, TILE_M), edge_m: float = EDGE_M) -> np.ndarray:
    x0, y0, x1, y1 = window
    return (P[:, 0] >= x0 + edge_m) & (P[:, 0] <= x1 - edge_m) & (P[:, 1] >= y0 + edge_m) & (P[:, 1] <= y1 - edge_m)


def nn_distances(g: RoadGraph, window=(0.0, 0.0, TILE_M, TILE_M)) -> np.ndarray:
    """Nearest-other-junction distance of every focal junction of ``window`` (neighbours: all junctions of g)."""
    nodes, e, _ = clean(g)
    J = nodes[junction_nodes(nodes, e)]
    if len(J) < 2:
        return np.zeros(0)
    foc = focal_mask(J, window)
    if not foc.any():
        return np.zeros(0)
    d, _ = cKDTree(J).query(J[foc], 2)
    return d[:, 1]


def consolidate(g: RoadGraph, tau: float, window=(0.0, 0.0, TILE_M, TILE_M)) -> dict:
    """Topology-aware single-linkage consolidation of junctions (module docstring)."""
    nodes, e, st = clean(g)
    jid = junction_nodes(nodes, e)
    out = {"junctions": int(len(jid)), "focal": 0, "clusters": 0, "ratio": np.nan, "max_diameter": 0.0, "groups": []}
    if len(jid) == 0:
        return out
    J = nodes[jid]
    foc = focal_mask(J, window)
    out["focal"] = int(foc.sum())
    par = np.arange(len(jid))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    pairs = cKDTree(J).query_pairs(tau, output_type="ndarray") if len(J) > 1 else np.zeros((0, 2), int)
    if len(pairs):
        ground = e[st == 0]
        L = np.hypot(*(nodes[ground[:, 0]] - nodes[ground[:, 1]]).T)
        A = coo_matrix((np.r_[L, L], (np.r_[ground[:, 0], ground[:, 1]], np.r_[ground[:, 1], ground[:, 0]])),
                       shape=(len(nodes), len(nodes))).tocsr()
        src = np.unique(pairs[:, 0])
        D = dijkstra(A, indices=jid[src], limit=1.5 * tau)
        row = {s: k for k, s in enumerate(src)}
        for a, b in pairs:
            if D[row[a], jid[b]] <= 1.5 * tau:
                par[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(jid))])
    groups = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(i)
    big = [m for m in groups.values() if len(m) > 1]
    out["max_diameter"] = float(max((np.max(np.hypot(*(J[m][:, None] - J[m][None]).transpose(2, 0, 1))) for m in big),
                                    default=0.0))
    out["groups"] = [[int(jid[i]) for i in m] for m in big]
    focal_roots = {roots[i] for i in np.flatnonzero(foc)}
    out["clusters"] = len(focal_roots)
    out["ratio"] = float(foc.sum() / len(focal_roots)) if focal_roots else np.nan
    return out


def junction_row(g: RoadGraph, window=(0.0, 0.0, TILE_M, TILE_M), tau_close: float = 15.0, tau_cons: float = 10.0) -> dict:
    """Window-level junction statistics (focal junctions only)."""
    nn = nn_distances(g, window)
    c = consolidate(g, tau_cons, window)
    area_km2 = max((window[2] - window[0] - 2 * EDGE_M) * (window[3] - window[1] - 2 * EDGE_M), 1.0) / 1e6
    row = {"focal_junctions": int(len(nn)) if len(nn) else c["focal"], "focal_density_km2": c["focal"] / area_km2,
           "close15": float(np.mean(nn < tau_close)) if len(nn) >= MIN_FOCAL else np.nan,
           "nn_p10": float(np.percentile(nn, 10)) if len(nn) >= MIN_FOCAL else np.nan,
           "nn_median": float(np.median(nn)) if len(nn) >= MIN_FOCAL else np.nan,
           "consol10_ratio": c["ratio"], "consol10_max_diam": c["max_diameter"]}
    for t in SWEEP_M:
        row[f"close_{t:g}"] = float(np.mean(nn < t)) if len(nn) >= MIN_FOCAL else np.nan
    row["_nn"] = nn.tolist()
    return row


def summarise(rows: list[dict]) -> dict:
    """Pooled and window-averaged summaries of window rows (with denominators)."""
    ok = [r for r in rows if r is not None and np.isfinite(r.get("close15", np.nan))]
    nn = np.concatenate([np.asarray(r["_nn"]) for r in ok]) if ok else np.zeros(0)
    out = {"windows": len(rows), "windows_scored": len(ok),
           "windows_lt2_focal": sum(1 for r in rows if r is None or not np.isfinite(r.get("close15", np.nan))),
           "focal_junctions": int(len(nn)),
           "pooled_close15": float(np.mean(nn < 15.0)) if len(nn) else np.nan,
           "pooled_nn_p10": float(np.percentile(nn, 10)) if len(nn) else np.nan,
           "window_close15": float(np.mean([r["close15"] for r in ok])) if ok else np.nan,
           "window_nn_p10": float(np.mean([r["nn_p10"] for r in ok])) if ok else np.nan,
           "window_consol10": float(np.nanmean([r["consol10_ratio"] for r in ok])) if ok else np.nan,
           "focal_density_km2": float(np.mean([r["focal_density_km2"] for r in rows if r is not None])) if rows else np.nan}
    for t in SWEEP_M:
        out[f"pooled_close_{t:g}"] = float(np.mean(nn < t)) if len(nn) else np.nan
    return out
