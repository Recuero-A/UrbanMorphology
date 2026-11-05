"""Milestone 13 evaluation metrics (docs/M13_EVALUATION_PROTOCOL.md §4). Historical S10 and S_T are unchanged and
computed by scripts/m12_evaluate.py functions; everything here has a NEW name.

Per tile (generated or real graph, ORIGINAL DEM):
* terrain response, per landform-slope bin (tc_potentials.SLOPE_BINS, 64 m cells): street length density,
  intersection density, length-weighted median and p90 of the 30 m sub-span grade, mean contour alignment, share of
  major + mid length;
* connectivity: giant-component length share, components (>= 40 m), dead-end ratio, pairwise accessibility (share of
  random street-point pairs 150-500 m apart that are connected), detour ratio of connected pairs (median),
  ``circuity_all`` = median detour with unreachable pairs set to DETOUR_UNREACHABLE (never silently excluded);
* infrastructure: bridges, bridge length, bridges per km of street and per km of river, river crossings by river-width
  class (crossed by bridge / unbridged), spacing of consecutive river crossings along the river axis, spans,
  unsupported structures (bridges over neither water nor a street);
* earthwork proxy: cut / fill depth between the terrain profile and a 30 m moving-average design profile, times the
  class carriageway width, per km of ground street (m^3 / km). A geometric proxy, not a construction estimate:
  flat cross-section, vertical sides, no batters, no mass haul; bridges and tunnels excluded.
Across samples of one tile: diversity (pairwise Jaccard distance of 8 m edge-occupancy rasters, L1 distance of
length-weighted orientation histograms). Across tiles: joint realism (energy distance, MMD) on per-tile feature
vectors normalised by REAL TRAINING-tile statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from urbanmorph.graphs.tc_potentials import SLOPE_BINS, TerrainField, edge_features, slope_bin

WIDTH = np.array([9.75, 10.0, 8.0, 6.0])
DETOUR_UNREACHABLE = 3.0
CELL_M = 64.0
WIDTH_BINS = np.r_[0.0, 20.0, 50.0, 100.0, np.inf]
JOINT_KEYS = ("street_km_per_km2", "intersections_per_km2", "dead_end_ratio", "giant_share", "grade_p90_m13",
              "contour_align", "circuity_all", "bridges_per_km", "slope_mean_ctx", "water_share_ctx")


def _cell_slope(tf: TerrainField):
    """Landform slope per 64 m cell of the 512 m tile (8 x 8)."""
    s = np.hypot(tf.gx, tf.gy)
    k = int(CELL_M / tf.t.res)
    inner = s[64:192, 64:192]                                       # the tile, without the 256 m margin
    return inner.reshape(8, k, 8, k).mean((1, 3))


def edge_table(g, tf: TerrainField):
    seg = g.segments() if len(g.edges) else np.zeros((0, 4))
    cls = g.classes() if len(g.edges) else np.zeros(0, int)
    st = g.structs() if len(g.edges) else np.zeros(0, int)
    rows = []
    for (x0, y0, x1, y1), c, s in zip(seg, cls, st):
        f = edge_features(tf, np.array([x0, y0]), np.array([x1, y1]), struct=int(s))
        rows.append((f, int(c), int(s)))
    return seg, cls, st, rows


def earthwork_per_km(seg, cls, st, tf: TerrainField, window_m=30.0, step=2.0):
    vol, length = 0.0, 0.0
    if not tf.has_dem:
        return np.nan
    for (x0, y0, x1, y1), c, s in zip(seg, cls, st):
        if s != 0:
            continue
        d, z = tf.t.profile(np.array([x0, y0]), np.array([x1, y1]), step)
        if len(z) < 3 or not np.isfinite(z).all():
            continue
        k = max(1, int(round(window_m / step)))
        pad = np.pad(z, k // 2, mode="edge")
        design = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(z)]
        vol += float(np.abs(z - design).sum() * step * WIDTH[int(c)])
        length += float(d[-1])
    return vol / max(length / 1000, 1e-9) if length > 0 else np.nan


def connectivity(g, rng, n_pairs=200, dmin=150.0, dmax=500.0):
    out = {"giant_share": np.nan, "components_m13": 0, "dead_end_ratio_m13": np.nan, "pair_access": np.nan,
           "detour_connected": np.nan, "circuity_all": np.nan, "unreachable_share": np.nan}
    if len(g.edges) == 0:
        out.update(giant_share=0.0, pair_access=0.0, circuity_all=DETOUR_UNREACHABLE, unreachable_share=1.0)
        return out
    n = len(g.nodes)
    L = np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T)
    A = coo_matrix((np.r_[L, L], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])), shape=(n, n)).tocsr()
    k, lab = connected_components(A, directed=False)
    cl = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=k)
    out["giant_share"] = float(cl.max() / L.sum())
    out["components_m13"] = int((cl >= 40).sum())
    deg = np.bincount(g.edges.ravel(), minlength=n)
    inner = ~g.border if g.border is not None and len(g.border) == n else np.ones(n, bool)
    out["dead_end_ratio_m13"] = float(((deg == 1) & inner).sum() / max((deg > 0).sum(), 1))
    src = rng.choice(n, size=min(30, n), replace=False)
    D = dijkstra(A, indices=src)
    det, unreach = [], 0
    for i, s in enumerate(src):
        e = np.hypot(*(g.nodes - g.nodes[s]).T)
        cand = np.flatnonzero((e >= dmin) & (e <= dmax))
        if not len(cand):
            continue
        j = rng.choice(cand, size=min(7, len(cand)), replace=False)
        for jj in j:
            if np.isfinite(D[i, jj]):
                det.append(D[i, jj] / e[jj])
            else:
                unreach += 1
        if len(det) + unreach >= n_pairs:
            break
    tot = len(det) + unreach
    if tot:
        out["pair_access"] = len(det) / tot
        out["unreachable_share"] = unreach / tot
        out["detour_connected"] = float(np.median(det)) if det else np.nan
        out["circuity_all"] = float(np.median(np.r_[np.minimum(det, DETOUR_UNREACHABLE), np.full(unreach, DETOUR_UNREACHABLE)]))
    return out


def river_axis(tf: TerrainField):
    """Principal axis (unit vector) and centre of the river pixels inside the tile, or None."""
    w = tf.t.water_kind[64:192, 64:192] == 1 if getattr(tf.t, "water_kind", None) is not None else None
    if w is None or w.sum() < 20:
        return None
    r, c = np.nonzero(w)
    xy = np.c_[c * 4.0 + 2, 512 - (r * 4.0 + 2)]
    m = xy.mean(0)
    u = np.linalg.svd(xy - m, full_matrices=False)[2][0]
    return m, u, float(w.sum() * 16.0)


def tile_metrics_m13(g, tf: TerrainField, rng) -> dict:
    seg, cls, st, rows = edge_table(g, tf)
    area_km2 = 0.262144
    L = np.array([f["length"] for f, _, _ in rows]) if rows else np.zeros(0)
    out = {"street_km_per_km2": float(L.sum() / 1000 / area_km2)}
    deg = g.degree() if len(g.edges) else np.zeros(0)
    out["intersections_per_km2"] = float((deg >= 3).sum() / area_km2)
    ground = [(f, c) for f, c, s in rows if s == 0 and np.isfinite(f["grade"])]
    if ground:
        gv = np.array([f["grade"] for f, _ in ground])
        gw = np.array([f["length"] for f, _ in ground])
        o = np.argsort(gv)
        cw = np.cumsum(gw[o]) / gw.sum()
        out["grade_p50_m13"] = float(gv[o][np.searchsorted(cw, 0.5)])
        out["grade_p90_m13"] = float(gv[o][min(np.searchsorted(cw, 0.9), len(o) - 1)])
        al = [(f["align"], f["length"]) for f, _ in ground if np.isfinite(f["align"]) and f["slope"] >= 0.05]
        out["contour_align"] = float(np.average([a for a, _ in al], weights=[w for _, w in al])) if al else np.nan
    else:
        out.update(grade_p50_m13=np.nan, grade_p90_m13=np.nan, contour_align=np.nan)
    # terrain response per slope bin (64 m cells)
    cs = _cell_slope(tf) if tf.has_dem else None
    nb = len(SLOPE_BINS) - 1
    resp = {k: np.full(nb, np.nan) for k in ("len_density", "int_density", "grade_med", "align_mean", "major_share")}
    cell_area = np.zeros(nb)
    if cs is not None:
        cb = np.clip(np.searchsorted(SLOPE_BINS, cs, side="right") - 1, 0, nb - 1)
        for b in range(nb):
            cell_area[b] = (cb == b).sum() * (CELL_M / 1000) ** 2
        mids = 0.5 * (seg[:, :2] + seg[:, 2:]) if len(seg) else np.zeros((0, 2))
        ci = np.clip((mids[:, 0] // CELL_M).astype(int), 0, 7) if len(mids) else np.zeros(0, int)
        ri = np.clip(((512 - mids[:, 1]) // CELL_M).astype(int), 0, 7) if len(mids) else np.zeros(0, int)
        eb = cb[ri, ci] if len(mids) else np.zeros(0, int)
        nodes3 = g.nodes[deg >= 3] if len(deg) else np.zeros((0, 2))
        nb_ = cb[np.clip(((512 - nodes3[:, 1]) // CELL_M).astype(int), 0, 7), np.clip((nodes3[:, 0] // CELL_M).astype(int), 0, 7)] if len(nodes3) else np.zeros(0, int)
        for b in range(nb):
            if cell_area[b] <= 0:
                continue
            m = eb == b
            resp["len_density"][b] = float(L[m].sum() / 1000 / cell_area[b]) if len(L) else 0.0
            resp["int_density"][b] = float((nb_ == b).sum() / cell_area[b])
            gr = [(rows[k][0]["grade"], rows[k][0]["length"]) for k in np.flatnonzero(m) if rows[k][2] == 0 and np.isfinite(rows[k][0]["grade"])]
            if gr:
                v, w = np.array(gr).T
                o = np.argsort(v)
                resp["grade_med"][b] = float(v[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)])
            al = [(rows[k][0]["align"], rows[k][0]["length"]) for k in np.flatnonzero(m) if rows[k][2] == 0 and np.isfinite(rows[k][0]["align"])]
            if al:
                v, w = np.array(al).T
                resp["align_mean"][b] = float(np.average(v, weights=w))
            if m.any():
                resp["major_share"][b] = float(L[m][cls[m] <= 1].sum() / max(L[m].sum(), 1e-9))
    out["response"] = {k: v.tolist() for k, v in resp.items()}
    out["response_cell_km2"] = cell_area.tolist()
    out.update(connectivity(g, rng))
    # infrastructure
    br = [(f, c) for f, c, s in rows if s == 1]
    out["bridges"] = len(br)
    out["bridge_m"] = float(sum(f["length"] for f, _ in br))
    out["bridges_per_km"] = len(br) / max(L.sum() / 1000, 1e-9) if len(L) else 0.0
    out["unsupported_structures"] = int(sum(1 for f, _ in br if f["wet_length"] == 0 and not f.get("over_street", False)
                                            and not (np.isfinite(f["clearance"]) and f["clearance"] >= 10)))
    out["viaducts"] = int(sum(1 for f, _ in br if f["wet_length"] == 0 and np.isfinite(f["clearance"]) and f["clearance"] >= 10))
    out["spans"] = [float(f["wet_length"] or f["length"]) for f, _ in br]
    ax = river_axis(tf)
    cross = []
    for f, c, s in rows:
        if 1 in f["water_kinds"] and np.isfinite(f["river_width"]):
            cross.append((s, f["river_width"]))
    out["river_crossings_bridged"] = int(sum(1 for s, _ in cross if s == 1))
    out["river_crossings_unbridged"] = int(sum(1 for s, _ in cross if s == 0))
    wb = np.zeros((2, len(WIDTH_BINS) - 1))
    for s, w in cross:
        wb[int(s == 1), int(np.clip(np.searchsorted(WIDTH_BINS, w, side="right") - 1, 0, len(WIDTH_BINS) - 2))] += 1
    out["crossings_by_width"] = wb.tolist()
    if ax is not None:
        m, u, area = ax
        river_km = (area / max(np.nanmedian([w for _, w in cross]) if cross else 30.0, 1.0)) / 1000
        out["river_km"] = river_km
        out["bridges_per_river_km"] = out["river_crossings_bridged"] / max(river_km, 1e-9)
        pts = [0.5 * (seg[k, :2] + seg[k, 2:]) for k, (f, c, s) in enumerate(rows) if s == 1 and 1 in f["water_kinds"]]
        proj = np.sort([(p - m) @ u for p in pts])
        out["crossing_spacing"] = np.diff(proj).tolist() if len(proj) > 1 else []
    else:
        out.update(river_km=0.0, bridges_per_river_km=np.nan, crossing_spacing=[])
    out["earthwork_m3_per_km"] = earthwork_per_km(seg, cls, st, tf)
    # terrain context for joint features (identical for real and generated on the same tile)
    out["slope_mean_ctx"] = float(np.mean(cs)) if cs is not None else np.nan
    out["water_share_ctx"] = float((tf.t.water_kind[64:192, 64:192] > 0).mean()) if getattr(tf.t, "water_kind", None) is not None else 0.0
    out["dead_end_ratio"] = out["dead_end_ratio_m13"]
    return out


def occupancy(g, res=8.0):
    from urbanmorph.graphs.sequence import segment_pixels  # noqa: F401  (4 m; we use our own 8 m grid)
    n = int(512 / res)
    M = np.zeros((n, n), bool)
    for x0, y0, x1, y1 in (g.segments() if len(g.edges) else []):
        k = max(2, int(np.hypot(x1 - x0, y1 - y0) / (res / 2)) + 1)
        xs, ys = np.linspace(x0, x1, k), np.linspace(y0, y1, k)
        M[np.clip(((512 - ys) // res).astype(int), 0, n - 1), np.clip((xs // res).astype(int), 0, n - 1)] = True
    return M


def orientation_hist(g, bins=18):
    if not len(g.edges):
        return np.zeros(bins)
    s = g.segments()
    ang = np.mod(np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])), 180)
    w = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    h = np.histogram(ang, bins=bins, range=(0, 180), weights=w)[0]
    return h / max(h.sum(), 1e-9)


def diversity(graphs) -> dict:
    """Pairwise diversity among samples generated on the SAME tile."""
    occ = [occupancy(g) for g in graphs]
    oh = [orientation_hist(g) for g in graphs]
    jac, ori = [], []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            u = (occ[i] | occ[j]).sum()
            jac.append(1 - (occ[i] & occ[j]).sum() / u if u else 0.0)
            ori.append(float(np.abs(oh[i] - oh[j]).sum()))
    return {"jaccard_distance": float(np.mean(jac)) if jac else np.nan, "orientation_l1": float(np.mean(ori)) if ori else np.nan}


def feature_matrix(rows, norm: dict, keys=JOINT_KEYS):
    X = np.array([[r.get(k, np.nan) for k in keys] for r in rows], float)
    mu, sd = np.array([norm["mean"][k] for k in keys]), np.array([norm["std"][k] for k in keys])
    X = (X - mu) / np.where(sd > 0, sd, 1)
    return np.nan_to_num(X, nan=0.0)


def energy_distance(X, Y):
    d = lambda A, B: np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def mmd_rbf(X, Y, bandwidth):
    k = lambda A, B: np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (2 * bandwidth ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def response_distance(gen_rows, real_rows, key):
    """Mean over slope bins of |mean_gen - mean_real| / pooled-real IQR, each bin weighted equally (flat land cannot
    dominate). Bins with no real cells are skipped."""
    nb = len(SLOPE_BINS) - 1
    G = np.array([r["response"][key] for r in gen_rows], float)
    R = np.array([r["response"][key] for r in real_rows], float)
    allr = R[np.isfinite(R)]
    iqr = np.subtract(*np.percentile(allr, [75, 25])) if len(allr) > 4 else 1.0
    out = []
    for b in range(nb):
        g, r = G[:, b], R[:, b]
        ok = np.isfinite(r) & np.isfinite(g)
        if ok.sum() >= 3:
            out.append(abs(np.mean(g[ok]) - np.mean(r[ok])) / max(iqr, 1e-9))
    return float(np.mean(out)) if out else np.nan
