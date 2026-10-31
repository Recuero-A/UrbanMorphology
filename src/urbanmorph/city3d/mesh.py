"""Watertight building solids and flat surfaces from footprint polygons (docs/M11_PROCEDURAL_BUILDINGS.md §1-2).

Conventions: x east, y north, z up, metres; triangles counter-clockwise seen from outside (outward normals).
A building solid = bottom cap + one quad per ring edge (holes included, so courtyards get walls) + roof.
Each face carries a part label: 0 bottom, 1 wall, 2 roof, 3 gable (wall-coloured roof-level wall).
Caps use shapely's constrained Delaunay triangulation, which keeps concave outlines and holes and adds no
vertices, so cap and wall vertices coincide exactly.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.format import polygons

BOTTOM, WALL, ROOF, GABLE = 0, 1, 2, 3
SIMPLIFY_M = 0.05
MIN_AREA_M2 = 10.0
RECT_TOL = 0.05            # pitched roofs only where |footprint XOR its min. rotated rectangle| <= 5 % of its area


def clean_footprint(poly) -> shapely.Polygon | None:
    """Valid, oriented, simplified single polygon (largest part), or None if degenerate."""
    parts = polygons(shapely.make_valid(poly))
    if not parts:
        return None
    p = max(parts, key=lambda q: q.area).simplify(SIMPLIFY_M, preserve_topology=True)
    p = shapely.Polygon(p.exterior, [h for h in p.interiors if shapely.Polygon(h).area >= 1.0])
    if not p.is_valid:
        p = max(polygons(shapely.make_valid(p)), key=lambda q: q.area, default=None)
    if p is None or p.area < MIN_AREA_M2:
        return None
    return shapely.orient_polygons(p)


def rect_footprint(poly) -> shapely.Polygon | None:
    """The footprint's minimum rotated rectangle if the footprint is (nearly) that rectangle and has no holes;
    vertices CCW starting with a LONG edge."""
    if poly.interiors:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = shapely.minimum_rotated_rectangle(poly)
    if r.geom_type != "Polygon" or r.area <= 0 or poly.symmetric_difference(r).area > RECT_TOL * poly.area:
        return None
    xy = np.asarray(shapely.orient_polygons(r).exterior.coords)[:4]
    if np.linalg.norm(xy[1] - xy[0]) < np.linalg.norm(xy[2] - xy[1]):
        xy = np.roll(xy, -1, axis=0)
    return shapely.Polygon(xy)


def triangulate(poly) -> tuple[np.ndarray, np.ndarray]:
    """(xy (n, 2), tris (k, 3)) indexing the polygon's own ring vertices, CCW seen from +z."""
    xy, key = [], {}
    for ring in [poly.exterior, *poly.interiors]:
        for x, y in np.asarray(ring.coords)[:-1]:
            k = (round(x, 6), round(y, 6))
            if k not in key:
                key[k] = len(xy)
                xy.append((x, y))
    tris = []
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        c = np.asarray(t.exterior.coords)[:3]
        idx = [key.get((round(x, 6), round(y, 6))) for x, y in c]
        if None in idx:
            raise ValueError("triangulation introduced a vertex")
        a, b, d = (c[1] - c[0]), (c[2] - c[0]), None
        cross = a[0] * b[1] - a[1] * b[0]
        if abs(cross) < 1e-9:
            continue
        tris.append(idx if cross > 0 else idx[::-1])
    return np.asarray(xy, float), np.asarray(tris, np.int64).reshape(-1, 3)


class Solid:
    """Triangle soup with shared vertices: V (n, 3), F (m, 3), part (m,)."""

    def __init__(self):
        self.V, self.F, self.part = [], [], []

    def v(self, p) -> int:
        self.V.append(tuple(float(x) for x in p))
        return len(self.V) - 1

    def tri(self, a, b, c, part):
        self.F.append((a, b, c))
        self.part.append(part)

    def quad(self, a, b, c, d, part):
        self.tri(a, b, c, part)
        self.tri(a, c, d, part)

    def arrays(self):
        return np.asarray(self.V, float).reshape(-1, 3), np.asarray(self.F, np.int64).reshape(-1, 3), np.asarray(self.part, np.int8)


def _walls(s: Solid, poly, bot: dict, top: dict) -> None:
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)[:-1]
        for i in range(len(c)):
            a, b = c[i], c[(i + 1) % len(c)]
            ka, kb = (round(a[0], 6), round(a[1], 6)), (round(b[0], 6), round(b[1], 6))
            s.quad(bot[ka], bot[kb], top[kb], top[ka], WALL)


def prism(poly, z0: float, z1: float) -> Solid:
    """Extruded footprint with a flat roof at z1 (any simple polygon, holes allowed)."""
    xy, tris = triangulate(poly)
    s = Solid()
    bi = [s.v((x, y, z0)) for x, y in xy]
    ti = [s.v((x, y, z1)) for x, y in xy]
    for a, b, c in tris:
        s.tri(bi[a], bi[c], bi[b], BOTTOM)
        s.tri(ti[a], ti[b], ti[c], ROOF)
    keys = [(round(x, 6), round(y, 6)) for x, y in xy]
    _walls(s, poly, dict(zip(keys, bi)), dict(zip(keys, ti)))
    return s


def pitched(rect, z0: float, eave: float, rise: float, kind: str) -> Solid:
    """Gable or hip roof on a rectangle (from rect_footprint: CCW, edge 0-1 long). Hip on a near-square = pyramid."""
    p = np.asarray(rect.exterior.coords)[:4]
    s = Solid()
    b = [s.v((x, y, z0)) for x, y in p]
    t = [s.v((x, y, eave)) for x, y in p]
    s.quad(b[0], b[3], b[2], b[1], BOTTOM)
    for i in range(4):
        j = (i + 1) % 4
        s.quad(b[i], b[j], t[j], t[i], WALL)
    u = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
    L, W = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
    m0, m1 = (p[0] + p[3]) / 2, (p[1] + p[2]) / 2
    zr = eave + rise
    if kind == "gable":
        r0, r1 = s.v((*m0, zr)), s.v((*m1, zr))
        s.tri(t[1], t[2], r1, GABLE)
        s.tri(t[3], t[0], r0, GABLE)
    elif L - W > 0.5:
        r0, r1 = s.v((*(m0 + u * W / 2), zr)), s.v((*(m1 - u * W / 2), zr))
        s.tri(t[1], t[2], r1, ROOF)
        s.tri(t[3], t[0], r0, ROOF)
    else:                                                       # pyramid
        apex = s.v((*p.mean(0), zr))
        for i in range(4):
            s.tri(t[i], t[(i + 1) % 4], apex, ROOF)
        return s
    s.quad(t[0], t[1], r1, r0, ROOF)
    s.quad(t[2], t[3], r0, r1, ROOF)
    return s


def building_solid(b: dict, z0: float = 0.0) -> tuple[Solid, bool]:
    """(solid, roof_as_specified) for one city-format building, rebuilt from its footprint rings only.
    A pitched roof whose footprint is no longer a near-rectangle (coordinate rounding) falls back to flat."""
    poly = shapely.orient_polygons(shapely.Polygon(b["footprint"][0], b["footprint"][1:]))
    if b["roof"]["type"] in ("gable", "hip"):
        rect = rect_footprint(poly)
        if rect is not None:
            return pitched(rect, z0, b["eave_m"], b["roof"]["rise_m"], b["roof"]["type"]), True
        return prism(poly, z0, b["eave_m"]), False
    return prism(poly, z0, b["eave_m"]), True


def surface(geom, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Upward-facing triangulated surface of all polygon parts of ``geom`` at height z."""
    V, F, n = [], [], 0
    for p in polygons(geom):
        p = shapely.orient_polygons(p.simplify(SIMPLIFY_M, preserve_topology=True))
        if p.is_empty or p.area < 0.5 or not p.is_valid:
            continue
        xy, tris = triangulate(p)
        V.append(np.c_[xy, np.full(len(xy), z)])
        F.append(tris + n)
        n += len(xy)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(V), np.vstack(F)
