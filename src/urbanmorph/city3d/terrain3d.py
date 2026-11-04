"""Put a flat Milestone 11 scene onto terrain (Milestone 12, spec §17-§19). Geometry changes are recorded.

``apply_terrain(groups, materials, city, terrain, graph)`` returns new groups, materials and a report:
* ``ground``: the flat base plane becomes a terrain mesh (the 4 m grid of the tile), material "ground";
* ``water``: flat quads on water cells at the water surface (``terrain.water_surface`` where known, else the DEM +
  0.2 m), material "water";
* flat layers (roads, sidewalks, markings, green): every triangle is cut by the 4 m grid (pieces of a triangle and a
  cell are convex, so they are fanned; neighbouring triangles get the same split points: no cracks), then lifted:
  - streets, sidewalks, markings: the ROAD SURFACE = the terrain smoothed over ~8 m and never below the terrain
    (a small fill; a road does not copy every DEM bump). The fill is reported (mean / p99 / max);
  - green: the terrain + 0.02 m;
  - the street surface under a bridge deck is removed where no ground street passes (no road drawn on a river);
* buildings: every face of building i is raised by its base height = the LOWEST terrain under its footprint, so no
  building floats (on a slope it is embedded uphill: a plinth / stepped foundation, spec §19);
* trees and streetlights: raised to the road surface;
* ``bridges`` and ``piers``: for every run of bridge edges, a deck slab (0.8 m thick, street width) following a
  straight profile between the road surface at its abutments, lifted to clear water by 3 m and ground by 1 m, with
  1.2 m piers every 30 m where the deck is > 2.5 m above the ground. Grade-separated crossings keep the street below
  at ground level.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy import ndimage

from urbanmorph.city3d.materials import MaterialTable
from urbanmorph.city3d.scene import corner_uvs
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.terrain.road_feasibility import water_kind_at
from urbanmorph.terrain.road_profiles import deck_profile

FLAT_ROAD = ("roads", "sidewalks", "markings")
CELL = 4.0
DECK_T, PIER_W, PIER_STEP = 0.8, 1.2, 30.0


def _mat(mats: list, name: str) -> int:
    p = MaterialTable()
    p.get(name)
    mats.append(p.params[0])
    return len(mats) - 1


class Surfaces:
    """Height fields over the tile: raw terrain, road surface (smoothed, >= terrain)."""

    def __init__(self, t):
        self.t = t
        z = np.where(np.isfinite(t.elev), t.elev, np.nanmean(t.elev))
        self.smooth = np.maximum(ndimage.gaussian_filter(z, 2.0), z)
        self.raw = z

    def _sample(self, a, x, y):
        r, c = self.t.rc(x, y)
        r0, c0 = np.clip(np.floor(r).astype(int), 0, a.shape[0] - 2), np.clip(np.floor(c).astype(int), 0, a.shape[1] - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        return ((1 - fr) * (1 - fc) * a[r0, c0] + (1 - fr) * fc * a[r0, c0 + 1] + fr * (1 - fc) * a[r0 + 1, c0]
                + fr * fc * a[r0 + 1, c0 + 1])

    def road(self, x, y):
        return self._sample(self.smooth, x, y)

    def ground(self, x, y):
        return self._sample(self.raw, x, y)


def grid_cut(V, F, mat, obj):
    """Split each triangle by the CELL grid into convex pieces, fanned. Returns V, F, mat, obj."""
    if not len(F):
        return V, F, mat, obj
    tris = shapely.polygons(V[F][:, :, :2])
    lo = np.floor(V[F][:, :, :2].min(1) / CELL).astype(int)
    hi = np.floor(V[F][:, :, :2].max(1) / CELL).astype(int)
    outV, outF, outM, outO = [], [], [], []
    n = 0
    for k in range(len(F)):
        z0 = float(V[F[k], 2].mean())
        if (hi[k] == lo[k]).all():
            pieces = [tris[k]]
        else:
            xs, ys = np.arange(lo[k, 0], hi[k, 0] + 1), np.arange(lo[k, 1], hi[k, 1] + 1)
            X, Y = np.meshgrid(xs, ys)
            boxes = shapely.box(X.ravel() * CELL, Y.ravel() * CELL, (X.ravel() + 1) * CELL, (Y.ravel() + 1) * CELL)
            pieces = [p for p in shapely.intersection(tris[k], boxes) if p.area > 1e-6]
        for p in pieces:
            for q in shapely.get_parts(p):
                if q.geom_type != "Polygon" or q.area < 1e-6:
                    continue
                ring = np.asarray(q.exterior.coords)[:-1]
                if shapely.Polygon(ring).exterior.is_ccw is False:
                    ring = ring[::-1]
                m = len(ring)
                outV.append(np.c_[ring, np.full(m, z0)])
                outF.append(np.c_[np.zeros(m - 2, int), np.arange(1, m - 1), np.arange(2, m)] + n)
                outM.append(np.full(m - 2, mat[k]))
                outO.append(np.full(m - 2, obj[k]))
                n += m
    if not outV:
        return V[:0], F[:0], mat[:0], obj[:0]
    return (np.vstack(outV).astype(np.float32), np.vstack(outF).astype(np.int32), np.concatenate(outM).astype(np.int16),
            np.concatenate(outO).astype(np.int32))


def grid_mesh(z, x0, y1, res, mask):
    """Quads (two triangles) for every cell with ``mask``; heights ``z`` at cell corners from the cell grid."""
    H, W = z.shape
    zc = np.pad(z, 1, mode="edge")
    corner = (zc[:-1, :-1] + zc[1:, :-1] + zc[:-1, 1:] + zc[1:, 1:]) / 4          # (H+1, W+1)
    R, C = np.nonzero(mask)
    ids = {}
    V, F = [], []

    def vid(r, c):
        k = (r, c)
        if k not in ids:
            ids[k] = len(V)
            V.append((x0 + c * res, y1 - r * res, corner[r, c]))
        return ids[k]
    for r, c in zip(R, C):
        a, b, d, e = vid(r, c), vid(r, c + 1), vid(r + 1, c + 1), vid(r + 1, c)
        F += [(a, e, d), (a, d, b)]
    return np.array(V, np.float32).reshape(-1, 3), np.array(F, np.int32).reshape(-1, 3)


def bridge_runs(g):
    st = g.structs()
    bk = np.flatnonzero(st == BRIDGE)
    adj = {}
    for k in bk:
        a, b = g.edges[k]
        adj.setdefault(int(a), []).append((int(b), k))
        adj.setdefault(int(b), []).append((int(a), k))
    seen, runs = set(), []
    for start in adj:
        if len(adj[start]) == 1 and start not in seen:     # walk chains from an end
            chain, prev, u = [start], -1, start
            seen.add(start)
            while True:
                nxt = [(w, k) for w, k in adj[u] if w != prev and k not in seen]
                if not nxt:
                    break
                w, k = nxt[0]
                seen.add(k)
                chain.append(w)
                prev, u = u, w
                if len(adj[u]) != 2:
                    break
            if len(chain) >= 2:
                runs.append(chain)
    return runs


def box_along(p, q, half_w, z_top_p, z_top_q, thick):
    d = q - p
    L = np.hypot(*d)
    if L < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    n = np.array([-d[1], d[0]]) / L * half_w
    top = [np.r_[p + n, z_top_p], np.r_[p - n, z_top_p], np.r_[q - n, z_top_q], np.r_[q + n, z_top_q]]
    bot = [v - np.r_[0, 0, thick] for v in top]
    V = np.array(top + bot, np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)
    return V, F


def apply_terrain(groups: dict, mats: list, city: dict, t, g, class_widths=(14.0, 10.0, 7.0, 5.0)):
    S = Surfaces(t)
    mats = list(mats)
    out, rep = {}, {"fill_m": None, "buildings": {}, "bridges": 0, "piers": 0}
    m_water, m_bridge, m_ground = _mat(mats, "water"), _mat(mats, "bridge_concrete"), None
    # bridge corridors minus ground corridors: where the street surface is replaced by a deck
    st = g.structs() if len(g.edges) else np.zeros(0, int)
    seg = g.segments() if len(g.edges) else np.zeros((0, 4))
    cls = g.classes() if len(g.edges) else np.zeros(0, int)
    lines = lambda m: [shapely.LineString(s.reshape(2, 2)).buffer(class_widths[int(c)] / 2 + 1.0)
                       for s, c in zip(seg[m], cls[m])]
    bcor = shapely.union_all(lines(st == BRIDGE)) if (st == BRIDGE).any() else shapely.Polygon()
    gcor = shapely.union_all(lines(st != BRIDGE)) if (st != BRIDGE).any() else shapely.Polygon()
    under = bcor.difference(gcor) if not bcor.is_empty else bcor
    fills = []
    for name, a in groups.items():
        if name.startswith("inst_") or name.startswith("proto_") or "F" not in a:
            out[name] = a
            continue
        V, F, mat, obj = a["V"].astype(float), a["F"], a["mat"], a["obj"]
        if name == "ground":
            m_ground = int(mat[0]) if len(mat) else 0
            continue
        if name in FLAT_ROAD or name == "green":
            V, F, mat, obj = grid_cut(V, F, mat, obj)
            V = V.astype(float)
            if name in FLAT_ROAD and not under.is_empty and len(F):
                cen = V[F].mean(1)
                keep = ~shapely.contains_xy(under, cen[:, 0], cen[:, 1])
                F, mat, obj = F[keep], mat[keep], obj[keep]
            zf = S.road if name in FLAT_ROAD else S.ground
            dz = zf(V[:, 0], V[:, 1])
            if name == "roads":
                fills.append(dz - S.ground(V[:, 0], V[:, 1]))
            V[:, 2] += dz + (0.08 if name in FLAT_ROAD else 0.05)         # clear the cell-averaged terrain mesh
        elif len(obj) and (obj >= 0).any():                      # building parts: raise per building
            base = rep["buildings"]
            for i in np.unique(obj[obj >= 0]):
                if i not in base:
                    fp = shapely.Polygon(np.asarray(city["buildings"][i]["footprint"][0])[:, :2])
                    pts = np.vstack([np.asarray(fp.exterior.coords), np.asarray(fp.representative_point().coords)])
                    base[int(i)] = float(np.min(S.ground(pts[:, 0], pts[:, 1])))
            vb = np.full(len(V), np.nan)
            for f, o in zip(F, obj):
                if o >= 0:
                    vb[f] = base[int(o)]
            V[:, 2] += np.nan_to_num(vb, nan=0.0)
        out[name] = {"V": V.astype(np.float32), "F": F.astype(np.int32), "mat": mat.astype(np.int16), "obj": obj.astype(np.int32),
                     "UV": corner_uvs(V, F.astype(np.int32))}
    # terrain and water meshes over the tile
    tile = t.window(0, 0, TILE_M, TILE_M)
    rows, cols = np.nonzero(tile)
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    x0 = t.x0 + cols.min() * t.res
    y1 = t.y1 - rows.min() * t.res
    zt = S.raw[sl]
    V, F = grid_mesh(zt, x0, y1, t.res, np.ones(zt.shape, bool))
    # a skirt: vertical walls from the terrain edge down to the surround plane (no visible gap at the tile edge)
    zs = float(np.nanmin(zt)) - 1.0
    edge = [v for v in range(len(V)) if V[v, 0] in (V[:, 0].min(), V[:, 0].max()) or V[v, 1] in (V[:, 1].min(), V[:, 1].max())]
    cx, cy = V[:, 0].mean(), V[:, 1].mean()
    edge = sorted(edge, key=lambda v: np.arctan2(V[v, 1] - cy, V[v, 0] - cx))
    SV, SF = [V], [F]
    n0 = len(V)
    low = V[edge].copy()
    low[:, 2] = zs
    SV.append(low)
    for i in range(len(edge)):
        j = (i + 1) % len(edge)
        a, b, c, e = edge[i], edge[j], n0 + j, n0 + i
        SF.append(np.array([[a, e, c], [a, c, b]], np.int32))
    V, F = np.vstack(SV).astype(np.float32), np.vstack(SF).astype(np.int32)
    out["ground"] = {"V": V, "F": F, "mat": np.full(len(F), m_ground or 0, np.int16), "obj": np.full(len(F), -1, np.int32),
                     "UV": corner_uvs(V.astype(float), F)}
    rep["surround_z"] = zs
    wk = getattr(t, "water_kind", None)
    if wk is not None and (wk[sl] > 0).any():
        surf = getattr(t, "water_surface", None)
        zw = np.where(np.isfinite(surf[sl]), surf[sl], zt + 0.2) if surf is not None else zt + 0.2
        V, F = grid_mesh(ndimage.minimum_filter(zw, 3) + 0.05, x0, y1, t.res, wk[sl] > 0)
        out["water"] = {"V": V, "F": F, "mat": np.full(len(F), m_water, np.int16), "obj": np.full(len(F), -1, np.int32),
                        "UV": corner_uvs(V.astype(float), F)}
    # bridges: decks and piers
    BV, BF, PV, PF = [], [], [], []
    eidx = {(int(min(a, b)), int(max(a, b))): k for k, (a, b) in enumerate(g.edges)} if len(g.edges) else {}
    nb = npier = 0
    for chain in bridge_runs(g) if len(g.edges) else []:
        P = g.nodes[chain]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        ground = S.ground(P[:, 0], P[:, 1])
        wet = water_kind_at(t, P[:, 0], P[:, 1]) > 0
        clear = np.where(wet, 3.0, 1.0)
        za, zb = S.road(*P[0][:, None])[0], S.road(*P[-1][:, None])[0]
        deck = deck_profile(d, ground + clear - 1.0, za, zb, clearance=1.0)
        ek = eidx.get((min(chain[0], chain[1]), max(chain[0], chain[1])))
        hw = class_widths[int(cls[ek]) if ek is not None else 2] / 2
        for i in range(len(P) - 1):
            v, f = box_along(P[i], P[i + 1], hw, deck[i] + 0.05, deck[i + 1] + 0.05, DECK_T)
            BF.append(f + sum(len(x) for x in BV))
            BV.append(v)
        nb += 1
        s = np.arange(PIER_STEP / 2, d[-1], PIER_STEP)
        for si in s:
            xy = np.array([np.interp(si, d, P[:, 0]), np.interp(si, d, P[:, 1])])
            zd = float(np.interp(si, d, deck)) - DECK_T
            zg = float(S.ground(xy[:1], xy[1:2])[0])
            if zd - zg > 2.5:
                c = xy
                v = np.array([[c[0] + dx, c[1] + dy, z] for z in (zg - 0.5, zd) for dx, dy in
                              ((-PIER_W / 2, -PIER_W / 2), (PIER_W / 2, -PIER_W / 2), (PIER_W / 2, PIER_W / 2), (-PIER_W / 2, PIER_W / 2))])
                f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                              [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
                PF.append(f + sum(len(x) for x in PV))
                PV.append(v)
                npier += 1
    for name, VV, FF in (("bridges", BV, BF), ("piers", PV, PF)):
        if VV:
            V, F = np.vstack(VV).astype(np.float32), np.vstack(FF).astype(np.int32)
            out[name] = {"V": V, "F": F, "mat": np.full(len(F), m_bridge, np.int16), "obj": np.full(len(F), -1, np.int32),
                         "UV": corner_uvs(V.astype(float), F)}
    # instances
    for name in list(out):
        if name.startswith("inst_"):
            T = out[name]["T"].copy()
            T[:, 2] += S.road(T[:, 0], T[:, 1])
            out[name] = {"T": T.astype(np.float32)}
    f = np.concatenate(fills) if fills else np.zeros(1)
    rep.update(fill_m={"mean": float(f.mean()), "p99": float(np.percentile(f, 99)), "max": float(f.max())},
               bridges=nb, piers=npier, building_base_m={"min": float(min(rep["buildings"].values(), default=0)),
                                                       "max": float(max(rep["buildings"].values(), default=0))})
    rep["buildings"] = len(rep["buildings"])
    return out, mats, rep
