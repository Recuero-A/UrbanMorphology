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
  building floats (on a slope it is embedded uphill: a plinth / stepped foundation, spec §19); with ``seat="street"``
  (steep hill towns, themes) the base is the street level at the front, with a terrace cut and a stone plinth
  (street_seats);
* an engineered road surface ``terrain.road_z`` (finite where defined), when present, replaces the smoothed one;
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
        rz = getattr(t, "road_z", None)                      # an engineered road surface (level across the street)
        if rz is not None:
            self.smooth = np.where(np.isfinite(rz), rz, self.smooth)
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


FRONT_M = 12.0             # street_levels: a building farther than this from every street fronts a lane or a terrace


def _outline(b, hull=None):
    """A building's ground outline: its footprint, or the hull of its own geometry when that reaches well beyond it
    (churches and towers built over a lot)."""
    fp = shapely.make_valid(shapely.Polygon(np.asarray(b["footprint"][0])[:, :2]))
    fp = max(getattr(fp, "geoms", [fp]), key=lambda p: p.area) if fp.geom_type != "Polygon" else fp
    if hull is not None and fp.geom_type == "Polygon" and hull.area > 1.15 * fp.area:
        return hull
    return fp


def building_hulls(groups: dict) -> dict:
    """{building index: convex hull (xy) of its own geometry} over the building groups."""
    pts = {}
    for name, a in groups.items():
        if name.startswith(("inst_", "proto_")) or "F" not in a or name in FLAT_ROAD or name in ("green", "ground"):
            continue
        obj = np.asarray(a["obj"])
        sel = obj >= 0
        if not sel.any():
            continue
        F, o = a["F"][sel], obj[sel]
        order = np.argsort(o, kind="stable")
        F, o = F[order], o[order]
        cuts = np.flatnonzero(np.diff(o)) + 1
        for Fc, oc in zip(np.split(F, cuts), np.split(o, cuts)):
            pts.setdefault(int(oc[0]), []).append(np.asarray(a["V"])[np.unique(Fc)][:, :2])
    return {i: shapely.MultiPoint(np.vstack(p)).convex_hull for i, p in pts.items()}


def street_levels(S, city, seg, hulls: dict | None = None) -> list:
    """Per building: (street level, lowest ground, highest ground under its outline), or None. The street level is the
    road surface at the street point nearest to the outline; a building more than FRONT_M from every street (on a
    lane, a plaza edge or the town edge) stands on its own terrace at the median ground under it.
    ``seg``: (n, 4) ground street segments of the tile; ``hulls``: building_hulls (outlines of churches and towers)."""
    lines = shapely.linestrings(seg.reshape(-1, 2, 2)) if len(seg) else None
    tree = shapely.STRtree(lines) if lines is not None else None
    out = []
    for i, b in enumerate(city["buildings"]):
        fp = _outline(b, (hulls or {}).get(i))
        if fp.geom_type != "Polygon" or fp.area < 1.0:
            out.append(None)
            continue
        ring = np.asarray(fp.exterior.coords)[:-1]
        x0, y0, x1, y1 = fp.bounds
        gx, gy = np.meshgrid(np.arange(x0 + 1.0, x1, 2.0), np.arange(y0 + 1.0, y1, 2.0))
        inner = np.c_[gx.ravel(), gy.ravel()]
        inner = inner[shapely.contains_xy(fp, inner[:, 0], inner[:, 1])]
        pts = np.vstack([ring, inner])
        gz = S.ground(pts[:, 0], pts[:, 1])
        zs = float(np.median(gz))
        if tree is not None:
            sl = shapely.shortest_line(lines[tree.nearest(fp)], fp)
            if sl.length <= FRONT_M:
                q = np.asarray(sl.coords[0])
                zs = float(S.road(q[:1], q[1:2])[0])
        out.append((zs, float(gz.min()), float(gz.max())))
    return out


def _quads(P0, P1, lo, hi, normal):
    """Vertical quads between points P0[k] -> P1[k] (xy) from lo[k] to hi[k], each turned to face ``normal[k]``."""
    k = len(P0)
    V = np.vstack([np.c_[P0, lo], np.c_[P1, lo], np.c_[P1, hi], np.c_[P0, hi]])
    a, b, c, d = np.arange(k), np.arange(k) + k, np.arange(k) + 2 * k, np.arange(k) + 3 * k
    e = P1 - P0
    out = np.c_[e[:, 1], -e[:, 0]]                            # the right side of P0 -> P1
    flip = (out * normal).sum(1) < 0
    F = np.vstack([np.where(flip[:, None], np.c_[a, c, b], np.c_[a, b, c]),
                   np.where(flip[:, None], np.c_[a, d, c], np.c_[a, c, d])])
    return V, F


def street_seats(S, t, city, seg, base: dict, mats: list, plinth: str, hulls: dict | None = None) -> dict | None:
    """seat="street" (steep hill towns): every building's street floor is at its street's level (street_levels).
    A house below its street (downhill) continues down the slope by the storeys the theme added (``below_m``, set by
    the house builder from ``below_floors``); any other building stands at the street level clipped to the ground
    under its outline. The terrain under each outline (+ 0.5 m) is cut down to the street level (uphill: a terrace dug
    into the slope); what remains between the base and the lowest ground is a stone plinth. Fills ``base`` in place;
    returns the plinth group (or None)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry.polygon import orient
    tr = from_origin(t.x0, t.y1, t.res, t.res)
    m_pl = _mat(mats, plinth)
    PV, PF, n = [], [], 0
    for i, (b, lv) in enumerate(zip(city["buildings"], street_levels(S, city, seg, hulls))):
        if lv is None:
            continue
        zs, gmin, gmax = lv
        below = float(b.get("below_m", 0.0))
        b0 = zs - below if below > 0 else float(np.clip(zs, gmin, gmax))
        base[i] = b0
        fp = _outline(b, (hulls or {}).get(i))
        cut = rasterize([(fp.buffer(0.5), 1)], out_shape=S.raw.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        S.raw[cut] = np.minimum(S.raw[cut], zs if below > 0 else b0)
        if b0 - gmin > 0.3:                                    # the stone plinth, sides only (the house covers its top)
            r = np.asarray(orient(fp, 1.0).exterior.coords)[:-1]
            k = len(r)
            V = np.vstack([np.c_[r, np.full(k, gmin - 0.5)], np.c_[r, np.full(k, b0 + 0.02)]])
            a = np.arange(k)
            c = (a + 1) % k
            PF.append(np.vstack([np.c_[a, c, c + k], np.c_[a, c + k, a + k]]) + n)
            PV.append(V)
            n += len(V)
    if not PV:
        return None
    V, F = np.vstack(PV).astype(np.float32), np.vstack(PF).astype(np.int32)
    return {"V": V, "F": F, "mat": np.full(len(F), m_pl, np.int16), "obj": np.full(len(F), -1, np.int32),
            "UV": corner_uvs(V.astype(float), F)}


def terrace_geometry(S, t, polys, step: float, mats: list, top: str, wall: str, res: float = 1.0):
    """Inca-style terraces (andenes) on the open land of a steep town: every polygon is cut into bands of ``step`` m
    of ground height (marching on a ``res`` m grid), each band a level platform at its lower bound; vertical stone
    walls rise between bands and along a polygon's edge where the land beside it (a street, a house terrace) is
    higher or lower. The heightfield under the terraces is lowered out of sight. Returns {group: arrays}."""
    from rasterio.features import rasterize, shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    from shapely.geometry.polygon import orient
    from urbanmorph.city3d.format import polygons as parts
    from urbanmorph.city3d.mesh import surface
    m_top, m_wall = _mat(mats, top), _mat(mats, wall)
    rz = getattr(t, "road_z", None)
    raw0 = S.raw.copy()                                        # the levels, before the heightfield is lowered

    def outside_z(P):                                          # the street surface on a street, else the ground
        z = S.ground(P[:, 0], P[:, 1])
        if rz is not None:
            r, c = t.rc(P[:, 0], P[:, 1])
            r = np.clip(np.round(r).astype(int), 0, rz.shape[0] - 1)
            c = np.clip(np.round(c).astype(int), 0, rz.shape[1] - 1)
            z = np.where(np.isfinite(rz[r, c]), S.road(P[:, 0], P[:, 1]), z)
        return z
    TV, TF, WV, WF, nt, nw = [], [], [], [], 0, 0
    low, edge_pts = [], []
    for P in polys:
        if P.area < 25.0:
            continue
        x0, y0, x1, y1 = P.bounds
        nx, ny = int(np.ceil((x1 - x0) / res)) + 2, int(np.ceil((y1 - y0) / res)) + 2
        tr = from_origin(x0 - res, y1 + res, res, res)
        inside = rasterize([(P, 1)], out_shape=(ny, nx), transform=tr, fill=0, dtype="uint8").astype(bool)
        if inside.sum() < 16:
            continue
        XX, YY = np.meshgrid(x0 - res + (np.arange(nx) + 0.5) * res, y1 + res - (np.arange(ny) + 0.5) * res)
        lev = np.floor(S.ground(XX.ravel(), YY.ravel()).reshape(ny, nx) / step).astype(np.int32)
        bands = {}
        for geom, v in shapes(lev, mask=inside, transform=tr):
            bands.setdefault(int(v), []).append(shapely.make_valid(shape(geom)))
        for v, gs in bands.items():
            B = shapely.make_valid(shapely.union_all(gs).simplify(0.45)).intersection(P)
            zin = v * step
            Vt, Ft = surface(B, zin + 0.03)
            if len(Ft):
                TV.append(Vt)
                TF.append(Ft + nt)
                nt += len(Vt)
            for poly in parts(B):
                poly = orient(poly, 1.0)
                for ring in [poly.exterior, *poly.interiors]:
                    xy = np.asarray(shapely.segmentize(ring, 1.0).coords)
                    p0, p1 = xy[:-1], xy[1:]
                    d = p1 - p0
                    Ld = np.hypot(d[:, 0], d[:, 1])
                    ok = Ld > 0.05
                    p0, p1, d, Ld = p0[ok], p1[ok], d[ok], Ld[ok]
                    nrm = np.c_[d[:, 1], -d[:, 0]] / Ld[:, None]           # away from the band
                    outp = (p0 + p1) / 2 + nrm * 0.6
                    inP = shapely.contains_xy(P, outp[:, 0], outp[:, 1])
                    r_ = np.clip(((y1 + res - outp[:, 1]) / res).astype(int), 0, ny - 1)
                    c_ = np.clip(((outp[:, 0] - x0 + res) / res).astype(int), 0, nx - 1)
                    zout = np.where(inP, lev[r_, c_] * step, outside_z(outp))
                    up = zout > zin + 0.3                                   # higher land beside: a wall facing in
                    dn = ~inP & (zout < zin - 0.3)                          # a lower street or terrace: facing out
                    lip = zout < zin - 0.3                                  # the platform's outer edge: plants on it
                    if lip.any():
                        edge_pts.append(np.c_[((p0 + p1) / 2 - nrm * 0.9)[lip], np.full(int(lip.sum()), zin + 0.03)])
                    for sel, lo, hi, nn in ((up, np.full(len(zout), zin - 0.3), zout, -nrm),
                                            (dn, zout - 0.3, np.full(len(zout), zin + 0.03), nrm)):
                        if sel.any():
                            V, F = _quads(p0[sel], p1[sel], lo[sel], hi[sel], nn[sel])
                            WV.append(V)
                            WF.append(F + nw)
                            nw += len(V)
        low.append(P)
    out = {}
    if TV:
        V, F = np.vstack(TV).astype(np.float32), np.vstack(TF).astype(np.int32)
        out["terrace_tops"] = {"V": V, "F": F, "mat": np.full(len(F), m_top, np.int16), "obj": np.full(len(F), -1, np.int32),
                               "UV": corner_uvs(V.astype(float), F)}
    if WV:
        V, F = np.vstack(WV).astype(np.float32), np.vstack(WF).astype(np.int32)
        out["terrace_walls"] = {"V": V, "F": F, "mat": np.full(len(F), m_wall, np.int16), "obj": np.full(len(F), -1, np.int32),
                                "UV": corner_uvs(V.astype(float), F)}
    area = shapely.union_all(low) if low else shapely.Polygon()
    if low:                                                    # the heightfield out of sight under the platforms
        trg = from_origin(t.x0, t.y1, t.res, t.res)
        cells = rasterize([(area, 1)], out_shape=S.raw.shape, transform=trg, fill=0, dtype="uint8").astype(bool)
        floor_ = np.floor(ndimage.minimum_filter(S.raw, 3) / step) * step - 0.5
        S.raw[cells] = np.minimum(S.raw[cells], floor_[cells])

    if edge_pts:                                               # planting along the terrace edges, about every 12 m
        E = np.vstack(edge_pts)
        rng = np.random.default_rng(int(abs(t.x0 * 7 + t.y1 * 3)) % (2 ** 31))
        E = E[rng.uniform(size=len(E)) < 0.08]
        kind = rng.choice(["inst_shrub", "inst_bougainvillea", "inst_banana", "inst_flowers"], size=len(E), p=[0.5, 0.2, 0.2, 0.1])
        out["_terrace_plants"] = {k: np.c_[E[kind == k], rng.uniform(0.8, 1.3, (kind == k).sum()),
                                           rng.uniform(0, 2 * np.pi, (kind == k).sum())].astype(np.float32)
                                  for k in np.unique(kind)}

    def level_at(x, y):                                        # the platform level at (x, y), NaN off the terraces
        r, c = t.rc(x, y)
        r0, c0 = np.clip(np.floor(r).astype(int), 0, raw0.shape[0] - 2), np.clip(np.floor(c).astype(int), 0, raw0.shape[1] - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        z = ((1 - fr) * (1 - fc) * raw0[r0, c0] + (1 - fr) * fc * raw0[r0, c0 + 1] + fr * (1 - fc) * raw0[r0 + 1, c0]
             + fr * fc * raw0[r0 + 1, c0 + 1])
        return np.where(shapely.contains_xy(area, x, y), np.floor(z / step) * step + 0.03, np.nan)
    return out, level_at


def stair_flights(S, g, class_widths, grade: float, mats: list, material: str, walk: float = 1.6, rise: float = 0.2,
                  end_m: float = 1.5):
    """Street edges steeper than ``grade`` (on the road surface) become stone stairs over the carriageway and its
    sidewalks: steps of about ``rise`` m, with end_m m landings at junctions and street ends. Returns (zone polygon
    where the street surface is dropped, stairs group or None)."""
    seg = g.segments() if len(g.edges) else np.zeros((0, 4))
    st = np.asarray(g.structs()) if len(g.edges) else np.zeros(0, int)
    cls = g.classes() if len(g.edges) else np.zeros(0, int)
    deg = np.bincount(g.edges.ravel(), minlength=len(g.nodes)) if len(g.edges) else np.zeros(0, int)
    zones, VV, FF, n = [], [], [], 0
    for k in np.flatnonzero(st == 0):
        p, q = seg[k, :2].astype(float), seg[k, 2:].astype(float)
        ea, eb = (end_m if deg[g.edges[k, 0]] != 2 else 0.0), (end_m if deg[g.edges[k, 1]] != 2 else 0.0)   # landings at junctions only
        L = float(np.hypot(*(q - p)))
        if L < ea + eb + 1.0:
            continue
        za, zb = float(S.road(p[:1], p[1:])[0]), float(S.road(q[:1], q[1:])[0])
        if abs(zb - za) / L <= grade:
            continue
        if za > zb:
            p, q, ea, eb = q, p, eb, ea
        u = (q - p) / L
        nr = np.array([-u[1], u[0]])
        a, b = p + u * ea, q - u * eb
        z0, z1 = float(S.road(a[:1], a[1:])[0]), float(S.road(b[:1], b[1:])[0])
        m = max(2, int(np.ceil((z1 - z0) / rise)))
        run, h = (L - ea - eb) / m, (z1 - z0) / m
        w = class_widths[int(cls[k])] / 2 + walk
        i = np.arange(m)
        s0, s1 = i * run, (i + 1) * run + 0.02
        top, bot = z0 + (i + 1) * h + 0.06, z0 + i * h - 0.35
        c00, c01 = a + np.outer(s0, u) - nr * w, a + np.outer(s0, u) + nr * w
        c10, c11 = a + np.outer(s1, u) - nr * w, a + np.outer(s1, u) + nr * w
        parts = [_quads(c00, c01, bot, top, np.tile(-u, (m, 1))),          # risers face downhill
                 _quads(c00, c10, bot, top, np.tile(-nr, (m, 1))),         # sides
                 _quads(c01, c11, bot, top, np.tile(nr, (m, 1))),
                 _quads(c10[-1:], c11[-1:], bot[-1:], top[-1:], u[None])]  # the back of the last step
        tv = np.vstack([np.c_[c00, top], np.c_[c10, top], np.c_[c11, top], np.c_[c01, top]])
        A, B_, C, D = i, i + m, i + 2 * m, i + 3 * m
        parts.append((tv, np.vstack([np.c_[A, B_, C], np.c_[A, C, D]])))   # treads, facing up
        for V, F in parts:
            VV.append(V)
            FF.append(F + n)
            n += len(V)
        zones.append(shapely.LineString([a, b]).buffer(w, cap_style="flat"))
    if not VV:
        return shapely.Polygon(), None
    V, F = np.vstack(VV).astype(np.float32), np.vstack(FF).astype(np.int32)
    m_st = _mat(mats, material)
    return shapely.union_all(zones), {"V": V, "F": F, "mat": np.full(len(F), m_st, np.int16), "obj": np.full(len(F), -1, np.int32),
                                      "UV": corner_uvs(V.astype(float), F)}


def apply_terrain(groups: dict, mats: list, city: dict, t, g, class_widths=(14.0, 10.0, 7.0, 5.0), seat: str = "min",
                  plinth: str = "bridge_concrete", terraces: dict | None = None, stairs: dict | None = None):
    """``seat``: "min" (default: every building on the lowest ground under it) or "street" (street_seats: terraces
    and plinths for steep hill towns; ``plinth`` = the plinth material). Steep towns also take ``terraces``
    {"step", "top", "wall"} (terrace_geometry on the polygons ``city["terraces"]``) and ``stairs`` {"grade",
    "material", "walk"} (stair_flights)."""
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
    terrace_level, terrace_plants = None, {}
    if seat == "street" and city.get("buildings"):            # bases, terraces and plinths before anything is lifted
        pl = street_seats(S, t, city, seg[st != BRIDGE] if len(seg) else seg, rep["buildings"], mats, plinth,
                          building_hulls(groups))
        if pl is not None:
            out["plinths"] = pl
    if terraces and city.get("terraces"):
        from urbanmorph.city3d.format import polygon
        polys = [shapely.make_valid(polygon(r)) for r in city["terraces"]]
        tg, terrace_level = terrace_geometry(S, t, [p for p in polys if p.area > 0], terraces["step"], mats, terraces["top"],
                                             terraces["wall"])
        terrace_plants = tg.pop("_terrace_plants", {})
        out.update(tg)
        rep["terrace_faces"] = int(sum(len(out[k]["F"]) for k in ("terrace_tops", "terrace_walls") if k in out))
    if stairs:
        zone, grp = stair_flights(S, g, class_widths, stairs["grade"], mats, stairs["material"], stairs.get("walk", 1.6))
        if grp is not None:
            out["stairs"] = grp
            under = zone if under.is_empty else under.union(zone)
            rep["stair_faces"] = int(len(grp["F"]))
    for name, a in groups.items():
        if name.startswith("inst_") or name.startswith("proto_") or "F" not in a:
            out[name] = a
            continue
        V, F, mat, obj = a["V"].astype(float), a["F"], a["mat"], a["obj"]
        uv = None                                                # recomputed unless the faces are only lifted
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
            uv = a.get("UV")                                     # building parts keep their own UVs (roof slopes, cut-outs)
        out[name] = {"V": V.astype(np.float32), "F": F.astype(np.int32), "mat": mat.astype(np.int16), "obj": obj.astype(np.int32),
                     "UV": uv if uv is not None and len(uv) == len(F) else corner_uvs(V, F.astype(np.int32))}
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
            if wk is not None and len(T):                          # no plant or lamp on water
                T = T[water_kind_at(t, T[:, 0], T[:, 1]) == 0]
            T[:, 2] += S.road(T[:, 0], T[:, 1])
            if terrace_level is not None and len(T):            # on the terraces: no forest trees; plants on the platform
                lv = terrace_level(T[:, 0], T[:, 1])
                on = np.isfinite(lv)
                if any(k in name for k in ("canopy", "jungle_tree")):
                    T = T[~on]
                else:
                    T[on, 2] = lv[on] + (T[on, 2] - S.road(T[on, 0], T[on, 1]))
            out[name] = {"T": T.astype(np.float32)}
    for name, P in terrace_plants.items():                     # terrace planting, where the theme has the prototype
        if f"proto_{name[5:]}" in out and len(P):
            old_T = out.get(name, {"T": np.zeros((0, 5), np.float32)})["T"]
            if old_T.shape[1] == P.shape[1]:
                out[name] = {"T": np.vstack([old_T, P]).astype(np.float32)}
                rep["terrace_plants"] = rep.get("terrace_plants", 0) + len(P)
    f = np.concatenate(fills) if fills else np.zeros(1)
    f = f if f.size else np.zeros(1)                     # M16: a tile whose fills are all empty
    rep.update(fill_m={"mean": float(f.mean()), "p99": float(np.percentile(f, 99)), "max": float(f.max())},
               bridges=nb, piers=npier, building_base_m={"min": float(min(rep["buildings"].values(), default=0)),
                                                       "max": float(max(rep["buildings"].values(), default=0))})
    t.building_base = dict(rep["buildings"])               # for later dressing (roof-terrace plants)
    rep["buildings"] = len(rep["buildings"])
    return out, mats, rep
