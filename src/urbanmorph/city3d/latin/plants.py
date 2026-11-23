"""Tropical vegetation and street furniture of the Latin Caribbean theme (DESIGN.md §6; sizes from RESEARCH.md §4).

Low-poly prototypes, instanced (one mesh per species, a transform per plant; Godot turns them into chunked
MultiMeshes): fronds and leaves are textured cut-out strips (alpha mask), crowns are lumpy textured blobs.
Budget: <= ~450 triangles per palm, ~150 per shrub, ~10 per flower card.

    prototypes() -> {name: (V, F, part, UV, [material keys])}
    place(city, road, sw, buildings_u, rng) -> {name: (k, 5) x, y, z, scale, yaw}  (lamps: (k, 4) x, y, z, yaw)
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString

from urbanmorph.city3d import environment as env
from urbanmorph.city3d.format import polygon


class Proto:
    def __init__(self, mats):
        self.mats = list(mats)
        self.V, self.F, self.part, self.UV = [], [], [], []

    def add(self, mat, V, F, UV):
        n = sum(len(v) for v in self.V)
        self.V.append(np.asarray(V, float))
        self.F.append(np.asarray(F, np.int64) + n)
        self.part.append(np.full(len(F), self.mats.index(mat), np.int8))
        self.UV.append(np.asarray(UV, np.float32).reshape(-1, 3, 2))

    def out(self):
        return np.vstack(self.V), np.vstack(self.F), np.concatenate(self.part), np.concatenate(self.UV), self.mats


def tube(P, R, k=6, circ=0.9):
    """Tube along polyline P (m, 3) with radii R (m,): V, F, UV (u around in metres / circ, v = arc length)."""
    P = np.asarray(P, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(t, ref)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(t, a)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ang = np.linspace(0, 2 * np.pi, k + 1)
    V, F, UV = [], [], []
    for i in range(len(P) - 1):
        for j in range(k):
            q = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            base = len(V)
            V += [P[ii] + R[ii] * (np.cos(ang[jj]) * a[ii] + np.sin(ang[jj]) * b[ii]) for ii, jj in q]
            cc = np.array([[ang[jj] / (2 * np.pi) * circ, s[ii]] for ii, jj in q])
            F += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            UV += [cc[[0, 1, 2]], cc[[0, 2, 3]]]
    return V, F, UV


def blob(c, r, rng, level=1, lump=0.18):
    """Lumpy closed crown (subdivided octahedron with radial noise); UV = planar metres (leaf texture)."""
    V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(level):
        mid, nf = {}, []

        def m(i, j):
            k = (min(i, j), max(i, j))
            if k not in mid:
                p = V[i] + V[j]
                V.append(p / np.linalg.norm(p))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, d in F:
            ab, bd, da = m(a, b), m(b, d), m(d, a)
            nf += [(a, ab, da), (ab, b, bd), (da, bd, d), (ab, bd, da)]
        F = nf
    V = np.asarray(V) * (1 + lump * rng.uniform(-1, 1, (len(V), 1))) * np.asarray(r, float) + np.asarray(c, float)
    F = np.asarray(F)
    P = V[F]
    n = np.abs(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))
    ax = n.argmax(1)                        # box projection on the dominant axis: no stretched stripes
    UV = np.where((ax == 2)[:, None, None], P[:, :, [0, 1]], np.where((ax == 1)[:, None, None], P[:, :, [0, 2]], P[:, :, [1, 2]]))
    return V, F, UV                         # shared vertices: smooth shading rounds the crown


def frond(base, yaw, elev, L, w, droop, fold=0.22, seg=5):
    """Pinnate frond: two strips folded along the rachis, drooping; UV u 0..1 across, v 0..1 along."""
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    s = np.linspace(0, 1, seg + 1)
    cen = base + np.outer(s * L * np.cos(elev), d) + np.outer(s * L * np.sin(elev) - droop * (s * L) ** 2, [0, 0, 1])
    wid = w * np.sin(np.pi * np.clip(s, 0.04, 1) ** 0.75) * 0.5
    V, F, UV = [], [], []
    for sgn, u0 in ((-1, 0.0), (1, 1.0)):
        edge = cen + sgn * np.outer(wid, side) - np.outer(fold * wid, [0, 0, 1])
        for i in range(seg):
            b = len(V)
            V += [cen[i], cen[i + 1], edge[i + 1], edge[i]]
            q = np.array([[0.5, s[i]], [0.5, s[i + 1]], [u0, s[i + 1]], [u0, s[i]]])
            F += [[b, b + 1, b + 2], [b, b + 2, b + 3]] if sgn > 0 else [[b, b + 2, b + 1], [b, b + 3, b + 2]]
            UV += [q[[0, 1, 2]], q[[0, 2, 3]]] if sgn > 0 else [q[[0, 2, 1]], q[[0, 3, 2]]]
    return V, F, UV


def coconut_palm(rng, H=12.0, lean=0.12):
    p = Proto(["trunk_coconut", "frond", "frond_dry", "coconut"])
    s = np.linspace(0, 1, 8)
    P = np.c_[lean * H * s ** 2, np.zeros_like(s), H * s]
    V, F, UV = tube(P, 0.24 - 0.09 * s, 6)
    p.add("trunk_coconut", V, F, UV)
    top = P[-1]
    for k in range(14):
        yaw = 2 * np.pi * k / 14 + rng.uniform(-0.2, 0.2)
        elev = np.radians(rng.uniform(-5, 45) if k % 3 else rng.uniform(20, 55))
        V, F, UV = frond(top, yaw, elev, rng.uniform(4.2, 5.4), rng.uniform(1.4, 1.8), rng.uniform(0.06, 0.1))
        p.add("frond", V, F, UV)
    for k in range(2):
        V, F, UV = frond(top - [0, 0, 0.3], rng.uniform(0, 2 * np.pi), np.radians(-70), 3.4, 1.0, 0.0)
        p.add("frond_dry", V, F, UV)
    for k in range(5):
        a = rng.uniform(0, 2 * np.pi)
        V, F, UV = blob(top + [0.35 * np.cos(a), 0.35 * np.sin(a), -0.45], 0.15, rng, level=0, lump=0.0)
        p.add("coconut", V, F, UV)
    return p.out()


def royal_palm(rng, H=16.0):
    p = Proto(["trunk_royal", "crownshaft", "frond"])
    s = np.linspace(0, 1, 9)
    r = 0.3 - 0.06 * s + 0.08 * np.exp(-((s - 0.35) / 0.18) ** 2)                 # the swelling of Roystonea
    P = np.c_[np.zeros_like(s), np.zeros_like(s), (H - 2.4) * s]
    V, F, UV = tube(P, r, 7)
    p.add("trunk_royal", V, F, UV)
    V, F, UV = tube(np.array([[0, 0, H - 2.45], [0, 0, H - 1.2], [0, 0, H]]), np.array([0.27, 0.29, 0.2]), 7)
    p.add("crownshaft", V, F, UV)
    top = np.array([0, 0, H])
    for k in range(18):
        yaw = 2 * np.pi * k / 18 + rng.uniform(-0.15, 0.15)
        elev = np.radians(rng.uniform(55, 80) if k % 3 == 0 else rng.uniform(5, 45))
        V, F, UV = frond(top, yaw, elev, rng.uniform(4.6, 5.6), rng.uniform(1.6, 2.0), rng.uniform(0.07, 0.11))
        p.add("frond", V, F, UV)
    return p.out()


def fan_palm(rng, H=7.0):
    p = Proto(["trunk_coconut", "fan_leaf"])
    s = np.linspace(0, 1, 5)
    V, F, UV = tube(np.c_[np.zeros_like(s), np.zeros_like(s), H * s], 0.24 - 0.04 * s, 6)
    p.add("trunk_coconut", V, F, UV)
    top = np.array([0, 0, H])
    for k in range(18):
        yaw = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(-20, 75))
        L = rng.uniform(1.6, 2.1)
        d = np.array([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
        base = top + d * 0.9
        side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.cross(side, d)
        up = up if up[2] >= 0 else -up
        q = [base - side * L / 2, base + side * L / 2, base + side * L / 2 + d * L + up * 0.2, base - side * L / 2 + d * L + up * 0.2]
        p.add("fan_leaf", q, [[0, 1, 2], [0, 2, 3]], np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]]))
    return p.out()


def banana(rng):
    p = Proto(["crownshaft", "banana_leaf"])
    for k in range(3):
        c = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0])
        h = rng.uniform(2.0, 3.0)
        V, F, UV = tube(np.array([c, c + [0, 0, h]]), np.array([0.13, 0.1]), 6)
        p.add("crownshaft", V, F, UV)
        for j in range(7):
            yaw = rng.uniform(0, 2 * np.pi)
            V, F, UV = frond(c + [0, 0, h], yaw, np.radians(rng.uniform(15, 65)), rng.uniform(1.8, 2.4), 0.65, 0.18,
                             fold=0.05, seg=4)
            p.add("banana_leaf", V, F, UV)
    return p.out()


def shrub(rng, mat="bougainvillea", scale=1.0):
    p = Proto([mat])
    for k in range(3):
        c = [rng.uniform(-0.6, 0.6) * scale, rng.uniform(-0.6, 0.6) * scale, (0.8 + 0.3 * k) * scale]
        V, F, UV = blob(c, np.array([1.0, 1.0, 0.85]) * rng.uniform(0.7, 1.0) * scale, rng, level=1, lump=0.22)
        p.add(mat, V, F, UV)
    return p.out()


def broadleaf(rng, mat="leaves", H=9.0, spread=4.0, flat=0.6, n=4, fine=False):
    """Jungle / shade tree (and the flamboyant with mat='flamboyant', flat crown wider than tall)."""
    p = Proto(["bark", mat])
    V, F, UV = tube(np.array([[0, 0, 0], [0.2, 0, H * 0.5], [0, 0.1, H * 0.7]]), np.array([0.32, 0.22, 0.16]), 6)
    p.add("bark", V, F, UV)
    for k in range(n):                      # a cluster of lumps: a central dome and a ring of smaller ones
        a = 2 * np.pi * k / max(n - 1, 1) + rng.uniform(-0.4, 0.4)
        d = spread * 0.38 * (k > 0)
        rr = spread * (0.5 if k == 0 else 0.36) * rng.uniform(0.85, 1.1)
        c = [d * np.cos(a), d * np.sin(a), H * (0.78 if k == 0 else 0.62 + 0.14 * rng.uniform())]
        V, F, UV = blob(c, np.array([rr, rr, rr * flat]), rng, level=(3 if k == 0 else 2) if fine else (2 if k == 0 else 1),
                        lump=0.1)
        p.add(mat, V, F, UV)
    return p.out()


def canopy(rng, mat="leaves", n=5):
    """A patch of rainforest canopy (~11 m across): a few short trunks under overlapping lumps at 7-13 m."""
    p = Proto(["bark", mat])
    for k in range(3):
        c = rng.uniform(-2.5, 2.5, 2)
        V, F, UV = tube(np.array([[c[0], c[1], 0], [c[0] + 0.3, c[1], 6.0]]), np.array([0.3, 0.2]), 5)
        p.add("bark", V, F, UV)
    for k in range(n):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 4.0)
        rr = rng.uniform(3.0, 4.6)
        V, F, UV = blob([d * np.cos(a), d * np.sin(a), rng.uniform(7.5, 12.5)], np.array([rr, rr, rr * 0.75]), rng, level=1, lump=0.15)
        p.add(mat, V, F, UV)
    return p.out()


def flower_card(rng):
    p = Proto(["flower_card"])
    for k in range(3):
        a = np.pi * k / 3
        d = np.array([np.cos(a), np.sin(a), 0]) * 0.65
        q = [-d, d, d + [0, 0, 1.3], -d + [0, 0, 1.3]]
        p.add("flower_card", q, [[0, 1, 2], [0, 2, 3]], np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]]))
    return p.out()


def lamp():
    """Colonial iron lamp post with a lantern."""
    p = Proto(["iron", "lamp"])
    V, F, UV = tube(np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 3.4]]), np.array([0.14, 0.07, 0.05]), 8)
    p.add("iron", V, F, UV)
    z0, z1 = 3.45, 4.1
    lo, hi = 0.12, 0.2
    ring = lambda r, z: [[r * np.cos(a), r * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4]
    Vl = ring(lo, z0) + ring(hi, z1)
    Fl = [[i, (i + 1) % 4, 4 + (i + 1) % 4] for i in range(4)] + [[i, 4 + (i + 1) % 4, 4 + i] for i in range(4)]
    p.add("lamp", Vl, Fl, np.zeros((len(Fl), 3, 2)))
    Vc = ring(hi + 0.05, z1) + [[0, 0, z1 + 0.28]]
    p.add("iron", Vc, [[i, (i + 1) % 4, 4] for i in range(4)], np.zeros((4, 3, 2)))
    return p.out()


def fountain():
    """Plaza fountain: octagonal stone basin with water, pedestal, upper bowl and finial."""
    p = Proto(["stone", "water", "trim"])
    ring = lambda r, z, k=8: [[r * np.cos(a), r * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 8]

    def band(mat, r0, r1, z0, z1, k=8, cap=None):
        V = ring(r0, z0, k) + ring(r1, z1, k)
        F = [[i, (i + 1) % k, k + (i + 1) % k] for i in range(k)] + [[i, k + (i + 1) % k, k + i] for i in range(k)]
        if cap is not None:
            V += [[0, 0, cap]]
            F += [[k + i, k + (i + 1) % k, 2 * k] for i in range(k)]
        P = np.asarray(V)[np.asarray(F)]
        p.add(mat, V, F, np.stack([np.arctan2(P[:, :, 1], P[:, :, 0]) * max(r0, r1), P[:, :, 2]], -1))
    band("stone", 3.2, 3.2, 0.0, 0.7)
    band("stone", 3.2, 3.45, 0.7, 0.8)
    band("stone", 2.9, 2.9, 0.8, 0.2)
    band("water", 2.9, 0.0, 0.55, 0.55)
    band("stone", 0.45, 0.35, 0.2, 1.7)
    band("trim", 0.35, 1.4, 1.7, 2.1)
    band("water", 1.25, 0.0, 2.0, 2.0)
    band("trim", 0.25, 0.12, 2.0, 2.9, cap=3.2)
    return p.out()


def prototypes(seed: int = 7) -> dict:
    r = np.random.default_rng(seed)
    return {
        "palm_coconut": coconut_palm(r, 12.0, 0.10), "palm_coconut_lean": coconut_palm(r, 11.0, 0.32),
        "palm_royal": royal_palm(r, 14.0), "palm_fan": fan_palm(r, 7.0), "banana": banana(r),
        "bougainvillea": shrub(r, "bougainvillea", 1.0), "shrub": shrub(r, "leaves_dark", 0.9),
        "flamboyant": broadleaf(r, "flamboyant", H=7.0, spread=8.0, flat=0.45, n=6, fine=True),
        "jungle_tree": broadleaf(r, "leaves", H=9.0, spread=6.5, flat=0.8, n=6),
        "jungle_tree_dark": broadleaf(r, "leaves_dark", H=12.0, spread=7.5, flat=0.75, n=5),
        "flowers": flower_card(r), "lamp": lamp(), "fountain": fountain(),
        "canopy": canopy(r, "leaves"), "canopy_dark": canopy(r, "leaves_dark"),
    }


# ----------------------------------------------------------------------------------------------- placement
PLAZA_MAX_M2 = 6000.0           # public green smaller than this is paved as a plaza (with palms), larger is a park
STREET_PALM_M = 13.0
JUNGLE_CLEAR_M = 20.0           # no jungle closer than this to a building or a street
JUNGLE_SPACING_M = 8.0


def _poisson(poly, spacing, rng, cap=6000):
    """Jittered-grid sampling (O(n); points >= ~0.4 x spacing apart) inside a polygon."""
    if poly.is_empty:
        return []
    x0, y0, x1, y1 = poly.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    P = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(0.0, 1.0, (gx.size, 2)) * spacing
    P = P[shapely.contains_xy(poly, P[:, 0], P[:, 1])]
    return list(P[rng.permutation(len(P))][:cap])


def place(city, road, sw, buildings_u, rng, tile=512.0, opts=None) -> tuple[dict, shapely.Geometry]:
    """Instances per prototype, the plaza area (paved) and the free land (jungle). opts {"jungle": "dense"}: a closed
    rainforest canopy (patches every ~8.5 m, emergent palms) right up to the town edge."""
    dense = (opts or {}).get("jungle") == "dense"
    out = {k: [] for k in prototypes()}
    add = lambda k, x, y, z=0.0, s=1.0: out[k].append([x, y, z, s * float(rng.uniform(0.85, 1.15)), float(rng.uniform(0, 2 * np.pi))])
    lights = []
    clear = buildings_u.buffer(2.6)
    sw_core = sw.buffer(-0.5)
    for st in city["streets"]:
        if st["class"] not in ("major", "mid", "local"):
            continue
        line = LineString(np.asarray(st["centerline"], float))
        w = max(st["segment_widths_m"]) / 2
        for side in (1, -1):
            off = line.offset_curve(side * (w + env.SIDEWALK_M[st["class"]] * 0.55))
            if not off.is_empty and st["class"] in ("major", "mid"):
                for d in np.arange(STREET_PALM_M / 2 + (side < 0) * STREET_PALM_M / 2, off.length, STREET_PALM_M):
                    p = off.interpolate(d)
                    if sw_core.contains(p) and not clear.contains(p) and not road.contains(p):
                        add("palm_royal" if st["class"] == "major" else "palm_coconut", p.x, p.y, env.CURB_M)
            loff = line.offset_curve(side * (w + 0.45))
            if loff.is_empty:
                continue
            for d in np.arange(env.LIGHT_SPACING / 2 if side == 1 else env.LIGHT_SPACING, loff.length, env.LIGHT_SPACING * 1.4):
                p = loff.interpolate(d)
                if sw.contains(p) and not buildings_u.buffer(0.5).contains(p):
                    lights.append([p.x, p.y, env.CURB_M, 0.0])
    plaza = shapely.Polygon()
    for g in city["landuse"]:
        if g["class"] not in ("public_green", "private_green"):
            continue
        P = shapely.make_valid(polygon(g["polygon"])).difference(road).difference(buildings_u)
        if P.is_empty:
            continue
        if g["class"] == "public_green" and P.area < PLAZA_MAX_M2:
            plaza = plaza.union(P)
            rim = P.buffer(-2.0)
            if not rim.is_empty:
                ring = rim.boundary
                for d in np.arange(0, ring.length, 9.0):
                    q = ring.interpolate(d)
                    add("palm_royal", q.x, q.y)
            centre = P.buffer(-6.0)
            c = shapely.Point()
            if not centre.is_empty and P.area > 900:
                c = centre.representative_point()
                out["fountain"].append([c.x, c.y, 0.0, 1.0, float(rng.uniform(0, 2 * np.pi))])
            core = P.buffer(-9.0).difference(c.buffer(10.0)) if not c.is_empty else P.buffer(-9.0)
            for q in _poisson(core, 14.0, rng)[:4]:
                add("flamboyant", q[0], q[1])
            for q in _poisson(P.buffer(-1.0).difference(P.buffer(-3.0)), 6.0, rng)[:24]:
                add("flowers" if rng.uniform() < 0.6 else "bougainvillea", q[0], q[1], s=0.8)
        elif g["class"] == "public_green":
            for q in _poisson(P.buffer(-1.5), 10.0, rng):
                k = rng.choice(["palm_coconut", "jungle_tree", "flamboyant", "palm_fan", "palm_royal"], p=[0.3, 0.3, 0.12, 0.18, 0.1])
                add(str(k), q[0], q[1])
            for q in _poisson(P.buffer(-1.0), 7.0, rng)[:200]:
                add("flowers" if rng.uniform() < 0.5 else "shrub", q[0], q[1], s=0.9)
        elif g.get("source") == "theme:garden":                      # skyscraper district gardens: tall palms
            core = P.difference(buildings_u.buffer(2.0))
            for q in _poisson(core, 8.0, rng):
                k = rng.choice(["palm_royal", "palm_coconut", "flamboyant", "palm_fan", "banana"], p=[0.3, 0.3, 0.14, 0.14, 0.12])
                add(str(k), q[0], q[1], s=float(rng.uniform(0.95, 1.25)))
            for q in _poisson(P.difference(buildings_u.buffer(0.8)), 4.5, rng)[:300]:
                add("bougainvillea" if rng.uniform() < 0.5 else "flowers", q[0], q[1], s=0.9)
        else:
            core = P.difference(buildings_u.buffer(1.2))
            for q in _poisson(core, 6.5, rng):
                k = rng.choice(["banana", "bougainvillea", "palm_fan", "palm_coconut", "flowers", "shrub"],
                               p=[0.26, 0.22, 0.13, 0.12, 0.17, 0.10])
                add(str(k), q[0], q[1])
    # jungle on land far from the town (water is removed later, on the terrain)
    tilebox = shapely.box(0, 0, tile, tile)
    built = (shapely.union_all([buildings_u.buffer(12.0), road.buffer(5.0), sw.buffer(2.0)]) if dense else
             shapely.union_all([buildings_u, road, sw]).buffer(JUNGLE_CLEAR_M))
    free = tilebox.difference(built)
    for x in (city["landuse"] if dense else []):
        if x["class"] in ("public_green", "private_green"):
            free = free.difference(shapely.make_valid(polygon(x["polygon"])).buffer(3.0))
    if dense:
        for q in _poisson(free, 8.5, rng, cap=12000):
            add("canopy" if rng.uniform() < 0.55 else "canopy_dark", q[0], q[1], s=float(rng.uniform(0.85, 1.3)))
        for q in _poisson(free, 22.0, rng, cap=3000):
            k = rng.choice(["palm_coconut", "palm_royal", "flamboyant", "palm_fan"], p=[0.5, 0.2, 0.1, 0.2])
            add(str(k), q[0], q[1], s=float(rng.uniform(1.0, 1.3)))
    for q in ([] if dense else _poisson(free, JUNGLE_SPACING_M, rng, cap=8000)):
        k = rng.choice(["jungle_tree", "jungle_tree_dark", "palm_coconut", "palm_coconut_lean", "palm_fan", "banana"],
                       p=[0.38, 0.3, 0.13, 0.05, 0.08, 0.06])
        add(str(k), q[0], q[1], s=float(rng.uniform(0.7, 1.4)))
    for q in ([] if dense else _poisson(free, 10.0, rng, cap=6000)):  # understorey
        add("shrub", q[0], q[1], s=float(rng.uniform(0.9, 1.7)))
    # open land inside the town (lots without buildings, verges): scattered palms, bananas and bushes
    open_ = tilebox.difference(free.buffer(4.0)).difference(shapely.union_all([buildings_u.buffer(3.0), road.buffer(1.0), sw]))
    for x in city["landuse"]:
        if x["class"] in ("public_green", "private_green"):
            open_ = open_.difference(shapely.make_valid(polygon(x["polygon"])))
    for q in _poisson(open_, 13.0, rng, cap=4000):
        if rng.uniform() < 0.45:
            k = rng.choice(["palm_coconut", "banana", "shrub", "bougainvillea", "palm_fan", "flamboyant", "jungle_tree"],
                           p=[0.24, 0.16, 0.2, 0.12, 0.12, 0.06, 0.1])
            add(str(k), q[0], q[1])
    inst = {k: np.asarray(v, float).reshape(-1, 5) for k, v in out.items() if k != "lamp"}
    inst["lamp"] = np.asarray(lights, float).reshape(-1, 4)
    return inst, plaza, free
