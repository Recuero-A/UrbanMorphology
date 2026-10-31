"""Environment layer: sidewalks, road markings, crossings, trees, streetlights (docs/M11_PROCEDURAL_BUILDINGS.md §5).

Everything respects the layout: sidewalks only on free land next to roads (never under buildings, slivers
narrower than 2 x SLIVER_M removed); markings only on the road surface; trees and lights only on sidewalks or
public green, clear of buildings. All placement is deterministic (rng from the city seed).
Heights: sidewalk slab top at CURB_M (a closed solid from -0.02), markings at MARK_Z above the road.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from urbanmorph.city3d.format import polygon, polygons
from urbanmorph.city3d.mesh import prism

SIDEWALK_M = {"major": 3.0, "mid": 2.5, "local": 2.0, "minor": 0.0}   # minor / service streets: no sidewalk
CURB_M = 0.15
SLIVER_M = 0.6
MARK_Z = 0.03
DASH, GAP, MARK_W = 3.0, 6.0, 0.15
TREE_SPACING_STREET, TREE_SPACING_PARK = 9.0, 9.0
CANOPY_R = 2.2
LIGHT_SPACING = 28.0


def sidewalks(city, road, buildings_u) -> shapely.Geometry:
    parts = []
    for s in city["streets"]:
        xy = np.asarray(s["centerline"])
        for (a, b), w, c in zip(zip(xy[:-1], xy[1:]), s["segment_widths_m"], s["segment_classes"]):
            if SIDEWALK_M[c] > 0:
                parts.append(LineString([a, b]).buffer(w / 2 + SIDEWALK_M[c], cap_style="flat"))
    if not parts:
        return shapely.Polygon()
    tile = shapely.box(0, 0, 512, 512)
    sw = shapely.union_all(parts).difference(road).difference(buildings_u).intersection(tile)
    sw = sw.buffer(-SLIVER_M, join_style="mitre").buffer(SLIVER_M, join_style="mitre")
    return sw.difference(road).difference(buildings_u).intersection(tile)          # the opening can re-grow slightly


def sidewalk_solids(sw):
    out = []
    for p in polygons(sw):
        p = shapely.orient_polygons(p.simplify(0.05, preserve_topology=True))
        if p.area < 2.0 or not p.is_valid:
            continue
        try:
            V, F, _ = prism(p, -0.02, CURB_M).arrays()
        except ValueError:
            continue
        out.append((V, F))
    return out


def _junction_nodes(city):
    return [np.asarray(n["xy"]) for n in city["nodes"] if n["degree"] >= 3 and not n["border"]]


def markings(city, road):
    """Dashed centrelines on streets >= 8 m wide (away from junctions) and zebra crossings at junction arms."""
    V, F = [], []

    def rect(c, u, n, hl, hw):
        k = len(V)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            q = c + u * hl * sx + n * hw * sy
            V.append([q[0], q[1], MARK_Z])
        F.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])

    junctions = _junction_nodes(city)
    J = np.asarray(junctions) if junctions else np.zeros((0, 2))
    n_dash = n_cross = 0
    for s in city["streets"]:
        xy = np.asarray(s["centerline"], float)
        line = LineString(xy)
        w = max(s["segment_widths_m"])
        if w >= 8.0 and line.length > 20:
            for d in np.arange(GAP, line.length - DASH, DASH + GAP):
                c0, c1 = line.interpolate(d), line.interpolate(d + DASH)
                c = np.array([(c0.x + c1.x) / 2, (c0.y + c1.y) / 2])
                if len(J) and np.min(np.linalg.norm(J - c, axis=1)) < w + 4:
                    continue
                u = np.array([c1.x - c0.x, c1.y - c0.y])
                if np.linalg.norm(u) < 1e-6 or not road.buffer(-0.3).contains(Point(c)):
                    continue
                u /= np.linalg.norm(u)
                rect(c, u, np.array([-u[1], u[0]]), DASH / 2, MARK_W / 2)
                n_dash += 1
        # zebra crossing on each end that is a junction
        for end, other in ((0, 1), (-1, -2)):
            p, q = xy[end], xy[other]
            if not len(J) or np.min(np.linalg.norm(J - p, axis=1)) > 0.5:
                continue
            u = (q - p) / (np.linalg.norm(q - p) + 1e-9)
            nrm = np.array([-u[1], u[0]])
            c = p + u * (w / 2 + 3.0)
            if line.length < w / 2 + 6.0:
                continue
            for k in np.arange(-w / 2 + 0.5, w / 2 - 0.4, 1.0):
                cc = c + nrm * (k + 0.25)
                if road.buffer(-0.2).contains(Point(cc)):
                    rect(cc, nrm, u, 0.25, 1.5)              # stripe 0.5 m across the street axis, 3 m along it
            n_cross += 1
    return np.asarray(V, float).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3), {"dashes": n_dash, "crossings": n_cross}


def poisson(poly, spacing, rng, max_pts=4000):
    """Dart-throwing minimum-distance sampling inside a polygon."""
    if poly.is_empty:
        return []
    x0, y0, x1, y1 = poly.bounds
    n_try = int(poly.area / spacing ** 2 * 8) + 10
    cand = rng.uniform([x0, y0], [x1, y1], size=(n_try, 2))
    keep = []
    inside = shapely.contains_xy(poly, cand[:, 0], cand[:, 1])
    for p in cand[inside]:
        if all(np.hypot(*(p - q)) >= spacing for q in keep):
            keep.append(p)
            if len(keep) >= max_pts:
                break
    return keep


def trees_and_lights(city, road, sw, buildings_u, rng):
    """(tree placements [x, y, z, scale, yaw], light placements [x, y, z, yaw]) and counts."""
    trees, lights = [], []
    clear = buildings_u.buffer(CANOPY_R + 0.3)
    sw_core = sw.buffer(-0.6)
    for s in city["streets"]:
        if s["class"] not in ("major", "mid", "local"):
            continue
        xy = np.asarray(s["centerline"], float)
        line = LineString(xy)
        w = max(s["segment_widths_m"]) / 2
        for side in (1, -1):
            off = line.offset_curve(side * (w + SIDEWALK_M[s["class"]] * 0.55))
            if off.is_empty:
                continue
            for d in np.arange(TREE_SPACING_STREET / 2, off.length, TREE_SPACING_STREET):
                p = off.interpolate(d)
                if sw_core.contains(p) and not clear.contains(p) and not road.contains(p):
                    trees.append([p.x, p.y, CURB_M, float(rng.uniform(0.8, 1.2)), float(rng.uniform(0, 2 * np.pi))])
            loff = line.offset_curve(side * (w + 0.45))
            if loff.is_empty:
                continue
            start = LIGHT_SPACING / 2 if side == 1 else LIGHT_SPACING
            for d in np.arange(start, loff.length, LIGHT_SPACING * 2):
                p = loff.interpolate(d)
                if sw.contains(p) and not buildings_u.buffer(0.5).contains(p):
                    q = line.interpolate(line.project(p))
                    yaw = float(np.arctan2(q.y - p.y, q.x - p.x))           # arm points over the road
                    lights.append([p.x, p.y, CURB_M, yaw])
    green = shapely.union_all([polygon(g["polygon"]) for g in city["landuse"] if g["class"] == "public_green"] or [shapely.Polygon()])
    park = green.difference(road).difference(buildings_u.buffer(CANOPY_R)).buffer(-1.5)
    for p in poisson(park, TREE_SPACING_PARK, rng):
        trees.append([p[0], p[1], 0.0, float(rng.uniform(0.8, 1.3)), float(rng.uniform(0, 2 * np.pi))])
    # thin street trees that ended up closer than the minimum spacing to an earlier tree (corners, both offsets)
    kept = []
    for t in trees:
        if all(np.hypot(t[0] - k[0], t[1] - k[1]) >= 5.0 for k in kept):
            kept.append(t)
    return np.asarray(kept, float).reshape(-1, 5), np.asarray(lights, float).reshape(-1, 4)


def _cylinder(r, z0, z1, k=6):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    poly = shapely.Polygon(np.c_[r * np.cos(a), r * np.sin(a)])
    V, F, _ = prism(shapely.orient_polygons(poly), z0, z1).arrays()
    return V, F


def _blob(r, zc):
    """Low-poly closed crown: an octahedron subdivided once and pushed to the sphere."""
    V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    mid = {}

    def m(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            p = V[i] + V[j]
            V.append(p / np.linalg.norm(p))
            mid[k] = len(V) - 1
        return mid[k]
    for _ in range(2):
        nf = []
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        F, mid = nf, {}
    V = np.asarray(V) * [r, r, r * 0.85] + [0, 0, zc]
    return V, np.asarray(F, np.int64)


def prototypes():
    """name -> (V, F, part) with part 0/1 selecting the proto's two materials."""
    tv, tf = _cylinder(0.15, 0.0, 3.0)
    cv, cf = _blob(CANOPY_R, 4.6)
    tree = (np.vstack([tv, cv]), np.vstack([tf, cf + len(tv)]), np.r_[np.zeros(len(tf), int), np.ones(len(cf), int)])
    pv, pf = _cylinder(0.08, 0.0, 6.0)
    arm = np.array([[0, -0.04, 5.85], [1.4, -0.04, 5.85], [1.4, 0.04, 5.85], [0, 0.04, 5.85],
                    [0, -0.04, 5.95], [1.4, -0.04, 5.95], [1.4, 0.04, 5.95], [0, 0.04, 5.95]])
    head = arm.copy()
    head[:, 0] = np.where(head[:, 0] > 0.5, 1.6, 1.1)
    head[:, 1] *= 4.0
    head[:, 2] = np.where(head[:, 2] > 5.9, 5.85, 5.7)
    bf = np.array([[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    light = (np.vstack([pv, arm, head]), np.vstack([pf, bf + len(pv), bf + len(pv) + 8]),
             np.r_[np.zeros(len(pf) + 12, int), np.ones(12, int)])
    return {"tree": tree, "streetlight": light}
