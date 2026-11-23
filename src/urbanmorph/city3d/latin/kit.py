"""Geometry kit of the Latin Caribbean theme: primitives, tile roofs and the colonial house (DESIGN.md §4).

Every element is a triangle soup keyed by material ('wall:amarillo', 'trimc:blanco', 'roof:teja', ...), with UVs
per corner in metres (images repeat every ``tile_m``) or in texture units for cut-outs (balusters, railings).
Wall elements live in a wall frame (origin o, along-wall unit u, outward normal n): x along the wall, d in front of
it, z up. Ornament is relief (boxes and panels in front of the wall), never a boolean cut, so walls stay simple and
cheap; small parts drop their hidden back faces.

Roofs: hip / gable shells with eave overhang, fascia and ridge caps on near-rectangular footprints; a 'skirt' roof
(a tiled band sloping out from a flat clay-paved azotea, mitred at every corner) on any other footprint without holes;
flat roofs keep a parapet with a coping. Tile textures follow the slope (u along the eave, v up the slope).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.mesh import pitched, prism, triangulate

Z_B = -0.05                     # buildings start below the base plane (scene.Z_BUILDING)


class Soup:
    """Triangle soups per material key."""

    def __init__(self):
        self.parts = {}

    def add(self, key, V, F, UV=None):
        from urbanmorph.city3d.scene import corner_uvs
        V = np.asarray(V, float).reshape(-1, 3)
        F = np.asarray(F, np.int64).reshape(-1, 3)
        if not len(F):
            return
        UV = corner_uvs(V, F) if UV is None else np.asarray(UV, np.float32).reshape(-1, 3, 2)
        self.parts.setdefault(key, []).append((V, F, UV))

    def arrays(self, key):
        lst = self.parts.get(key, [])
        n, F = 0, []
        for V, f, _ in lst:
            F.append(f + n)
            n += len(V)
        return (np.vstack([x[0] for x in lst]), np.vstack(F), np.concatenate([x[2] for x in lst])) if lst else None

    def triangles(self) -> int:
        return int(sum(len(f) for lst in self.parts.values() for _, f, _ in lst))


# ----------------------------------------------------------------------------------------------- primitives
def wp(o, u, n, x, d, z):
    return [o[0] + u[0] * x + n[0] * d, o[1] + u[1] * x + n[1] * d, z]


def wquad(S, key, o, u, n, x0, x1, z0, z1, d, uv=None):
    """Panel in front of the wall at depth d. uv='panel': (x - x0, z - z0) in metres (cut-out panels)."""
    V = [wp(o, u, n, x0, d, z0), wp(o, u, n, x1, d, z0), wp(o, u, n, x1, d, z1), wp(o, u, n, x0, d, z1)]
    UV = None
    if uv == "panel":
        c = np.array([[0, 0], [x1 - x0, 0], [x1 - x0, z1 - z0], [0, z1 - z0]], float)
        UV = c[[[0, 1, 2], [0, 2, 3]]]
    S.add(key, V, [[0, 1, 2], [0, 2, 3]], UV)


def wside(S, key, o, u, n, x, d0, d1, z0, z1, flip=False):
    """Panel perpendicular to the wall at along-wall position x (balcony and grille sides), UV (d, z - z0)."""
    V = [wp(o, u, n, x, d0, z0), wp(o, u, n, x, d1, z0), wp(o, u, n, x, d1, z1), wp(o, u, n, x, d0, z1)]
    F = [[0, 2, 1], [0, 3, 2]] if not flip else [[0, 1, 2], [0, 2, 3]]
    c = np.array([[0, 0], [d1 - d0, 0], [d1 - d0, z1 - z0], [0, z1 - z0]], float)
    S.add(key, V, F, c[np.asarray(F)])


BOX_F = [[4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
BACK_F = [[0, 3, 2], [0, 2, 1]]


def wbox(S, key, o, u, n, x0, x1, z0, z1, d0, d1, back=False):
    """Box in the wall frame; the face against the wall is dropped unless ``back``."""
    V = [wp(o, u, n, x, d, z) for d in (d0, d1) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    S.add(key, V, BOX_F + (BACK_F if back else []))


SLAB_F = [[4, 5, 6], [4, 6, 7], [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]]


def wslab(S, key, o, u, n, x0, x1, z0, z1, d0, d1):
    """Moulding: front, top and bottom faces only (6 triangles; the ends are too small to see)."""
    V = [wp(o, u, n, x, d, z) for d in (d0, d1) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    S.add(key, V, SLAB_F)


def beam(S, key, p, q, w, h):
    """Closed box from 3D point p to q, width w (horizontal, across) and height h (vertical), centred on the line.
    Not for vertical lines (no horizontal across direction): use cylinder(k=4)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    a = q - p
    side = np.array([-a[1], a[0], 0.0])
    side = side / (np.linalg.norm(side) + 1e-12) * w / 2
    up = np.array([0, 0, h / 2])
    V = [p - side - up, q - side - up, q - side + up, p - side + up, p + side - up, q + side - up, q + side + up, p + side + up]
    S.add(key, V, [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [3, 2, 6], [3, 6, 7], [0, 5, 1], [0, 4, 5], [0, 3, 7], [0, 7, 4],
                   [1, 5, 6], [1, 6, 2]])


def arch_outline(x0, x1, zs, k=10):
    """Points of a semicircular arch from the right spring (x1, zs) over the top to (x0, zs)."""
    r = (x1 - x0) / 2
    t = np.linspace(0, np.pi, k + 1)
    return np.c_[(x0 + x1) / 2 + r * np.cos(t), zs + r * np.sin(t)]


def warch(S, key, o, u, n, x0, x1, z0, zs, d, k=10):
    """Arched panel (rectangle up to the spring line zs, semicircle above), as a fan from the bottom centre."""
    P = np.r_[[[x0, z0], [x1, z0]], arch_outline(x0, x1, zs, k)]
    c = np.array([(x0 + x1) / 2, (z0 + zs) / 2])
    V = [wp(o, u, n, c[0], d, c[1])] + [wp(o, u, n, x, d, z) for x, z in P]
    m = len(P)
    S.add(key, V, [[0, 1 + i, 1 + (i + 1) % m] for i in range(m)])


def warch_frame(S, key, o, u, n, x0, x1, z0, zs, t, d0, d1, k=10, keystone=True):
    """Moulded surround of an arched opening: jambs (boxes), the arch band (front + outer faces), a keystone."""
    wbox(S, key, o, u, n, x0 - t, x0, z0, zs, d0, d1)
    wbox(S, key, o, u, n, x1, x1 + t, z0, zs, d0, d1)
    inner = arch_outline(x0, x1, zs, k)
    r = (x1 - x0) / 2
    cx = (x0 + x1) / 2
    ang = np.linspace(0, np.pi, k + 1)
    outer = np.c_[cx + (r + t) * np.cos(ang), zs + (r + t) * np.sin(ang)]
    V, F = [], []
    for i in range(k):
        b = len(V)
        V += [wp(o, u, n, *inner[i][:1], d1, inner[i][1]), wp(o, u, n, *outer[i][:1], d1, outer[i][1]),
              wp(o, u, n, *outer[i + 1][:1], d1, outer[i + 1][1]), wp(o, u, n, *inner[i + 1][:1], d1, inner[i + 1][1]),
              wp(o, u, n, *outer[i][:1], d0, outer[i][1]), wp(o, u, n, *outer[i + 1][:1], d0, outer[i + 1][1])]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3], [b + 1, b + 4, b + 5], [b + 1, b + 5, b + 2]]
    S.add(key, V, F)
    if keystone:
        wbox(S, key, o, u, n, cx - t * 0.45, cx + t * 0.45, zs + r - 0.02, zs + r + t * 1.25, d0, d1 + 0.04)


def wdisc(S, key, o, u, n, cx, cz, r, d, k=16):
    V = [wp(o, u, n, cx, d, cz)] + [wp(o, u, n, cx + r * np.cos(a), d, cz + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, k, endpoint=False)]
    S.add(key, V, [[0, 1 + i, 1 + (i + 1) % k] for i in range(k)])


def wring(S, key, o, u, n, cx, cz, r0, r1, d, k=16):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = [wp(o, u, n, cx + r * np.cos(t), d, cz + r * np.sin(t)) for t in a for r in (r0, r1)]
    F = []
    for i in range(k):
        j = (i + 1) % k
        F += [[2 * i, 2 * i + 1, 2 * j + 1], [2 * i, 2 * j + 1, 2 * j]]
    S.add(key, V, F)


def slope_uvs(V, F) -> np.ndarray:
    """Per-corner UVs of sloped faces in metres: u along the eave (contour), v up the slope (true length)."""
    P = np.asarray(V, float)[np.asarray(F)]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    h = np.linalg.norm(nrm[:, :2], axis=1, keepdims=True)
    d = np.where(h > 1e-6, nrm[:, :2] / np.maximum(h, 1e-12), np.array([[0.0, 1.0]]))    # downhill
    t = np.c_[-d[:, 1], d[:, 0]]
    nz = np.maximum(np.abs(nrm[:, 2:3]), 0.2)
    u = np.einsum("fcj,fj->fc", P[:, :, :2], t)
    v = -np.einsum("fcj,fj->fc", P[:, :, :2], d) / nz
    return np.stack([u, v], -1).astype(np.float32)


def sloped(S, key, V, F):
    S.add(key, V, F, slope_uvs(V, F))


def cylinder(S, key, c, r0, r1, z0, z1, k=8, cap=False, uv_around=None):
    """Tapered open cylinder (frustum); UV u around (metres, or uv_around repeats), v = z."""
    a = np.linspace(0, 2 * np.pi, k + 1)
    V, F, UV = [], [], []
    for i in range(k):
        q = [(r0, a[i], z0), (r0, a[i + 1], z0), (r1, a[i + 1], z1), (r1, a[i], z1)]
        b = len(V)
        V += [[c[0] + r * np.cos(t), c[1] + r * np.sin(t), z] for r, t, z in q]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        circ = 2 * np.pi * max(r0, r1) if uv_around is None else uv_around
        uu = [i / k * circ, (i + 1) / k * circ]
        cc = np.array([[uu[0], z0], [uu[1], z0], [uu[1], z1], [uu[0], z1]])
        UV += [cc[[0, 1, 2]], cc[[0, 2, 3]]]
    if cap:
        b = len(V)
        V += [[c[0], c[1], z1]] + [[c[0] + r1 * np.cos(t), c[1] + r1 * np.sin(t), z1] for t in a[:-1]]
        for i in range(k):
            F.append([b, b + 1 + i, b + 1 + (i + 1) % k])
            UV.append(np.zeros((3, 2)))
    S.add(key, V, F, np.asarray(UV))


def dome(S, key, c, r, z0, k=16, m=7, rise=1.0, ribs_key=None, ribs=8, rib_w=0.12):
    """Dome (quarter-ellipse profile, height rise * r) over a k-gon drum; UV u along the circumference at the
    springing, v along the meridian (metres). Optional raised ribs along ``ribs`` meridians."""
    phi = np.linspace(0, np.pi / 2, m + 1)
    th = np.linspace(0, 2 * np.pi, k + 1)
    R = r * np.cos(phi)
    Z = z0 + rise * r * np.sin(phi)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(R), np.diff(Z)))]
    V, F, UV = [], [], []
    for j in range(m):
        for i in range(k):
            q = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            b = len(V)
            V += [[c[0] + R[jj] * np.cos(th[ii]), c[1] + R[jj] * np.sin(th[ii]), Z[jj]] for jj, ii in q]
            cc = np.array([[th[ii] * r, arc[jj]] for jj, ii in q])
            if j == m - 1:                                       # top ring: a triangle to the crown
                F.append([b, b + 1, b + 2])
                UV.append(cc[[0, 1, 2]])
            else:
                F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
                UV += [cc[[0, 1, 2]], cc[[0, 2, 3]]]
    S.add(key, V, F, np.asarray(UV))
    if ribs_key:
        for i in range(ribs):
            t = th[i * k // ribs]
            for j in range(m - 1):
                p = [c[0] + (R[j] + 0.03) * np.cos(t), c[1] + (R[j] + 0.03) * np.sin(t), Z[j]]
                q = [c[0] + (R[j + 1] + 0.03) * np.cos(t), c[1] + (R[j + 1] + 0.03) * np.sin(t), Z[j + 1]]
                beam(S, ribs_key, p, q, rib_w, rib_w)
    return float(Z[-1])


def cross(S, key, c, z, h=1.6, t=0.12):
    """Cross on a square post (beam() is horizontal-only: its width vector vanishes on a vertical line)."""
    cylinder(S, key, c, t / np.sqrt(2), t / np.sqrt(2), z, z + h, k=4, cap=True)
    beam(S, key, [c[0] - h * 0.3, c[1], z + h * 0.68], [c[0] + h * 0.3, c[1], z + h * 0.68], 0.8 * t, 0.8 * t)


# ----------------------------------------------------------------------------------------------- roofs
def offset_ring(xy, dist):
    """Mitred offset of a CCW ring (no closing vertex): dist > 0 outward."""
    e = np.roll(xy, -1, 0) - xy
    L = np.linalg.norm(e, axis=1, keepdims=True)
    nrm = np.c_[e[:, 1], -e[:, 0]] / np.maximum(L, 1e-9)
    prev = np.roll(nrm, 1, 0)
    denom = 1 + (prev * nrm).sum(1, keepdims=True)
    return xy + dist * (prev + nrm) / np.maximum(denom, 1e-6), nrm


def hip_shell(S, roof_key, fascia_key, rect, eave, rise, kind, overhang=0.5):
    """Tile shell of a hip / gable / pyramid roof on a rectangle (CCW, edge 0-1 long), with overhang."""
    p = np.asarray(rect.exterior.coords)[:4]
    L, W = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])
    ex, ey = (p[1] - p[0]) / L, (p[3] - p[0]) / W
    s = rise / (W / 2)
    o = overhang
    ze, zr = eave - o * s, eave + rise
    P = lambda x, y, z: [*(p[0] + ex * x + ey * y), z]
    A, B, C, D = P(-o, -o, ze), P(L + o, -o, ze), P(L + o, W + o, ze), P(-o, W + o, ze)
    if kind == "gable":
        R0, R1 = P(-o, W / 2, zr), P(L + o, W / 2, zr)
        sloped(S, roof_key, [A, B, R1, R0, C, D], [[0, 1, 2], [0, 2, 3], [4, 5, 3], [4, 3, 2]])
        beam(S, roof_key, R0, R1, 0.3, 0.16)
        for a, b in ((A, R0), (R0, D), (B, R1), (R1, C)):                     # rake boards
            beam(S, fascia_key, a, b, 0.08, 0.22)
    elif L - W > 0.5:
        R0, R1 = P(W / 2, W / 2, zr), P(L - W / 2, W / 2, zr)
        sloped(S, roof_key, [A, B, R1, R0, C, D], [[0, 1, 2], [0, 2, 3], [4, 5, 3], [4, 3, 2], [1, 4, 2], [5, 0, 3]])
        beam(S, roof_key, R0, R1, 0.3, 0.16)
        for a, b in ((A, R0), (D, R0), (B, R1), (C, R1)):
            beam(S, roof_key, a, b, 0.26, 0.14)
    else:
        X = P(L / 2, W / 2, zr)
        sloped(S, roof_key, [A, B, C, D, X], [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        for a in (A, B, C, D):
            beam(S, roof_key, a, X, 0.26, 0.14)
    for a, b in ((A, B), (C, D)) + (((B, C), (D, A)) if kind != "gable" else ()):
        a2, b2 = np.array(a) - [0, 0, 0.16], np.array(b) - [0, 0, 0.16]
        V = [a, b, b2, a2]
        S.add(fascia_key, V, [[0, 2, 1], [0, 3, 2]])


def skirt_shell(S, roof_key, flat_key, fascia_key, poly, eave, w, slope, overhang=0.4) -> bool:
    """A tiled band sloping out from a flat azotea, on any footprint without holes. False if no valid band."""
    if poly.interiors:
        return False
    q = shapely.orient_polygons(poly.simplify(0.25, preserve_topology=True))
    if q.geom_type != "Polygon" or q.is_empty:
        return False
    xy = np.asarray(q.exterior.coords)[:-1]
    for ww in (w, 0.7 * w, 0.45 * w, 0.28 * w):
        inner, nrm = offset_ring(xy, -ww)
        outer, _ = offset_ring(xy, overhang)
        if np.abs(inner - xy).max() > 3.5 * ww or np.abs(outer - xy).max() > 3.5 * overhang:
            continue
        ip = shapely.Polygon(inner)
        e0, e1 = np.roll(xy, -1, 0) - xy, np.roll(inner, -1, 0) - inner
        if not ip.is_valid or ip.area < 1.0 or (np.einsum("ij,ij->i", e0, e1) <= 0).any() or not shapely.Polygon(outer).is_valid:
            continue
        zi, zo = eave + ww * slope, eave - overhang * slope
        k = len(xy)
        V = np.r_[np.c_[outer, np.full(k, zo)], np.c_[inner, np.full(k, zi)]]
        F = []
        for i in range(k):
            j = (i + 1) % k
            F += [[i, j, k + j], [i, k + j, k + i]]
        sloped(S, roof_key, V, F)
        for i in range(k):                                           # hips at the mitred corners
            beam(S, roof_key, V[i], V[k + i], 0.24, 0.12)
        ipo = shapely.orient_polygons(ip)
        txy, tris = triangulate(ipo)
        S.add(flat_key, np.c_[txy, np.full(len(txy), zi + 0.01)], tris)
        Vf, Ff = [], []
        for i in range(k):                                           # fascia board under the eave edge
            j = (i + 1) % k
            b = len(Vf)
            Vf += [V[i], V[j], V[j] - [0, 0, 0.14], V[i] - [0, 0, 0.14]]
            Ff += [[b, b + 2, b + 1], [b, b + 3, b + 2]]
        S.add(fascia_key, Vf, Ff)
        return True
    return False


def parapet(S, wall_key, coping_key, poly, eave, h=0.9):
    inner = poly.buffer(-0.25, join_style="mitre")
    ring = poly.difference(inner) if not inner.is_empty else None
    if ring is None or ring.is_empty or ring.area < 1.0:
        return
    from urbanmorph.city3d.format import polygons
    for p in polygons(ring):
        try:
            V, F, _ = prism(shapely.orient_polygons(p), eave, eave + h).arrays()
        except ValueError:
            continue
        S.add(wall_key, V, F)
    cap = poly.buffer(0.06, join_style="mitre").difference(poly.buffer(-0.32, join_style="mitre"))
    for p in polygons(cap):
        try:
            V, F, _ = prism(shapely.orient_polygons(p), eave + h, eave + h + 0.1).arrays()
        except ValueError:
            continue
        S.add(coping_key, V, F)


def walls_and_roof(S, b, st, poly, rect):
    """Wall solid + roof for one house; returns the solid (V, F) used for validation."""
    wall = f"wall:{st['wall']}"
    eave = st["eave"]
    if st["roof_kind"] in ("hip", "gable") and rect is not None:
        sol = pitched(rect, Z_B, eave, st["rise"], st["roof_kind"])
        V, F, part = sol.arrays()
        S.add(wall, V, F[part != 2])
        hip_shell(S, f"roof:{st['tiles']}", f"wood:{st['wood']}", rect, eave, st["rise"], st["roof_kind"], st["overhang"])
        return V, F
    sol = prism(poly, Z_B, eave)
    V, F, part = sol.arrays()
    if st["roof_kind"] == "skirt" and skirt_shell(S, f"roof:{st['tiles']}", "roof_flat", f"wood:{st['wood']}", poly, eave,
                                                  st["skirt_w"], st["slope"], st["overhang"]):
        S.add(wall, V, F[part != 2])
        return V, F
    st["roof_kind"] = "flat"
    S.add(wall, V, F[part != 2])
    S.add("roof_flat", V, F[part == 2])
    parapet(S, wall, f"trimc:{st['trim']}", poly, eave)
    return V, F


# ----------------------------------------------------------------------------------------------- house facade
CORNER = 0.8                    # pilaster + margin at each run end
PIL_W = 0.45


def levels(st) -> list[tuple[float, float]]:
    return [(0.0, st["gf"])] + [(st["gf"] + i * st["up"], st["gf"] + (i + 1) * st["up"]) for i in range(st["floors"] - 1)]


def window(S, st, o, u, n, x0, x1, z0, z1, shutters: bool, room: float):
    trim = f"trimc:{st['trim']}"
    wquad(S, "glass", o, u, n, x0, x1, z0, z1, 0.02)
    wquad(S, trim, o, u, n, x0 - 0.16, x0, z0, z1, 0.05)                            # flat surround
    wquad(S, trim, o, u, n, x1, x1 + 0.16, z0, z1, 0.05)
    wslab(S, trim, o, u, n, x0 - 0.28, x1 + 0.28, z1, z1 + 0.22, 0.0, 0.13)         # lintel moulding
    wslab(S, trim, o, u, n, x0 - 0.12, x1 + 0.12, z0 - 0.09, z0, 0.0, 0.11)         # sill
    if shutters and room >= (x1 - x0) * 2 + 0.5:
        w = (x1 - x0) / 2
        wquad(S, f"shutter:{st['wood']}", o, u, n, x0 - 0.16 - w, x0 - 0.16, z0, z1, 0.03)
        wquad(S, f"shutter:{st['wood']}", o, u, n, x1 + 0.16, x1 + 0.16 + w, z0, z1, 0.03)
    elif shutters:                                                                    # closed louvred leaves
        wquad(S, f"shutter:{st['wood']}", o, u, n, x0, x1, z0 + (z1 - z0) * 0.45, z1, 0.035)


def door(S, st, o, u, n, x0, x1, z0, z1, arched: bool):
    trim = f"trimc:{st['trim']}"
    key = f"door:{st['door_wood']}"
    if arched and z1 - (x1 - x0) / 2 > z0 + 1.6:
        zs = z1 - (x1 - x0) / 2
        warch(S, key, o, u, n, x0, x1, z0, zs, 0.03)
        warch_frame(S, trim, o, u, n, x0, x1, z0, zs, 0.22, 0.0, 0.09)
    else:
        wquad(S, key, o, u, n, x0, x1, z0, z1, 0.03)
        wbox(S, trim, o, u, n, x0 - 0.22, x0, z0, z1, 0.0, 0.09)
        wbox(S, trim, o, u, n, x1, x1 + 0.22, z0, z1, 0.0, 0.09)
        wbox(S, trim, o, u, n, x0 - 0.34, x1 + 0.34, z1, z1 + 0.3, 0.0, 0.15)
    wbox(S, "stone", o, u, n, x0 - 0.3, x1 + 0.3, Z_B, z0 + 0.14, 0.0, 0.45)        # threshold step


def reja(S, st, o, u, n, x0, x1, z0, z1):
    """Cartagena projecting window grille: ledge, cap, turned-baluster front and sides."""
    wd = f"wood:{st['wood']}"
    bal = f"baluster:{st['wood']}"
    D = 0.32
    wbox(S, wd, o, u, n, x0 - 0.15, x1 + 0.15, z0 - 0.14, z0, 0.0, D + 0.05)
    wbox(S, wd, o, u, n, x0 - 0.2, x1 + 0.2, z1, z1 + 0.12, 0.0, D + 0.1)
    wquad(S, bal, o, u, n, x0 - 0.1, x1 + 0.1, z0, z1, D, uv="panel")
    wside(S, bal, o, u, n, x0 - 0.1, 0.0, D, z0, z1)
    wside(S, bal, o, u, n, x1 + 0.1, 0.0, D, z0, z1, flip=True)
    if st.get("reja_roof"):
        V = [wp(o, u, n, x0 - 0.3, 0.0, z1 + 0.5), wp(o, u, n, x1 + 0.3, 0.0, z1 + 0.5),
             wp(o, u, n, x1 + 0.3, D + 0.3, z1 + 0.14), wp(o, u, n, x0 - 0.3, D + 0.3, z1 + 0.14)]
        sloped(S, f"roof:{st['tiles']}", V, [[0, 2, 1], [0, 3, 2]])


def wood_balcony(S, st, o, u, n, x0, x1, z0, z1):
    """Cartagena balcony: slab on corbels, turned balusters, posts and a small tile roof (tejaroz)."""
    wd = f"wood:{st['wood']}"
    bal = f"baluster:{st['wood']}"
    D = 0.95
    wbox(S, wd, o, u, n, x0, x1, z0 - 0.18, z0, 0.0, D)
    for xc in np.arange(x0 + 0.35, x1 - 0.2, 0.9):
        wbox(S, wd, o, u, n, xc - 0.08, xc + 0.08, z0 - 0.55, z0 - 0.18, 0.0, D - 0.12)
    wquad(S, bal, o, u, n, x0, x1, z0, z0 + 1.0, D - 0.04, uv="panel")
    wside(S, bal, o, u, n, x0 + 0.02, 0.0, D - 0.04, z0, z0 + 1.0)
    wside(S, bal, o, u, n, x1 - 0.02, 0.0, D - 0.04, z0, z0 + 1.0, flip=True)
    wbox(S, wd, o, u, n, x0, x1, z0 + 1.0, z0 + 1.08, D - 0.1, D)
    top = z1 - 0.3
    k = max(2, int(round((x1 - x0) / 2.2)) + 1)
    for xp in np.linspace(x0 + 0.06, x1 - 0.06, k):
        wbox(S, wd, o, u, n, xp - 0.06, xp + 0.06, z0, top, D - 0.12, D)
    V = [wp(o, u, n, x0 - 0.2, 0.0, z1 + 0.05), wp(o, u, n, x1 + 0.2, 0.0, z1 + 0.05),
         wp(o, u, n, x1 + 0.2, D + 0.3, top - 0.05), wp(o, u, n, x0 - 0.2, D + 0.3, top - 0.05)]
    sloped(S, f"roof:{st['tiles']}", V, [[0, 2, 1], [0, 3, 2]])
    Vu = [np.array(v) - [0, 0, 0.05] for v in V]
    S.add(wd, Vu, [[0, 1, 2], [0, 2, 3]])


def iron_balcony(S, st, o, u, n, x0, x1, z0):
    trim = f"trimc:{st['trim']}"
    D = 0.62
    wbox(S, trim, o, u, n, x0, x1, z0 - 0.14, z0 + 0.04, 0.0, D)
    wbox(S, trim, o, u, n, x0 + 0.1, x1 - 0.1, z0 - 0.3, z0 - 0.14, 0.0, D - 0.15)
    wquad(S, "iron_railing", o, u, n, x0 + 0.02, x1 - 0.02, z0 + 0.04, z0 + 1.0, D - 0.03, uv="panel")
    wside(S, "iron_railing", o, u, n, x0 + 0.03, 0.0, D - 0.03, z0 + 0.04, z0 + 1.0)
    wside(S, "iron_railing", o, u, n, x1 - 0.03, 0.0, D - 0.03, z0 + 0.04, z0 + 1.0, flip=True)
    wbox(S, "iron", o, u, n, x0, x1, z0 + 1.0, z0 + 1.05, D - 0.06, D)


def house(S, b, st, walls, runs_of, street_of, door_wall):
    """Facade of one house on its visible wall runs (party walls get nothing)."""
    trim = f"trimc:{st['trim']}"
    eave = st["eave"]
    lv = levels(st)
    count = {"windows": 0, "doors": 0, "balconies": 0}
    door_done = False
    for k, (a, e, n) in enumerate(walls):
        L = float(np.hypot(*(e - a)))
        u = (e - a) / L
        street = street_of[k]
        lift = 0.012 * (k % 3)
        for r0, r1 in runs_of[k]:
            if r1 - r0 < 1.2:
                continue
            if st["plinth"]:
                wquad(S, f"plinth:{st['plinth']}", a, u, n, r0, r1, Z_B, st["plinth_h"], 0.025)
                wslab(S, trim, a, u, n, r0, r1, st["plinth_h"] + lift, st["plinth_h"] + 0.07 + lift, 0.0, 0.05)
            if st["pilasters"]:
                if r0 < 0.05:
                    wbox(S, trim, a, u, n, r0, r0 + PIL_W, Z_B, eave, 0.0, 0.07)
                if r1 > L - 0.05:
                    wbox(S, trim, a, u, n, r1 - PIL_W, r1, Z_B, eave, 0.0, 0.07)
            if st["cornice"]:
                wslab(S, trim, a, u, n, r0, r1, eave - 0.42 + lift, eave - 0.24 + lift, 0.0, 0.12)
                wslab(S, trim, a, u, n, r0, r1, eave - 0.24 + lift, eave + lift, 0.0, 0.24)
            if st["string_course"]:
                for z0, _ in lv[1:]:
                    wslab(S, trim, a, u, n, r0, r1, z0 - 0.1 + lift, z0 + 0.08 + lift, 0.0, 0.06)
            c0, c1 = r0 + CORNER, r1 - CORNER
            if c1 - c0 < 1.4:
                continue
            nb = max(1, int(round((c1 - c0) / st["bay"])))
            bw = (c1 - c0) / nb
            door_bay = nb // 2 if (k == door_wall and not door_done and bw >= 1.9) else -1
            wood_bal_floor = 1 if (st["balcony"] == "wood" and street and len(lv) > 1) else -1
            for fi, (z0, z1) in enumerate(lv):
                fh = z1 - z0
                if fi == wood_bal_floor and nb >= 1:
                    xb0, xb1 = (c0 + 0.1, c1 - 0.1) if nb <= 4 else (c0 + bw * (nb // 2 - 1) + 0.1, c0 + bw * (nb // 2 + 1) - 0.1)
                    wood_balcony(S, st, a, u, n, xb0, xb1, z0, z1)
                    count["balconies"] += 1
                for j in range(nb):
                    cx = c0 + (j + 0.5) * bw
                    if fi == 0 and j == door_bay:
                        dw = min(1.6, bw - 0.7)
                        door(S, st, a, u, n, cx - dw / 2, cx + dw / 2, 0.12, min(3.1, fh - 0.55), st["arched_door"])
                        door_done = True
                        count["doors"] += 1
                        continue
                    ww = float(np.clip(bw * 0.42, 0.85, 1.25))
                    room = bw - ww - 0.2
                    if fi == 0:
                        zz0, zz1 = 0.7, min(0.7 + 2.0, fh - 0.55)
                        if st["reja"] and street:
                            wquad(S, "glass", a, u, n, cx - ww / 2, cx + ww / 2, zz0, zz1, 0.02)
                            reja(S, st, a, u, n, cx - ww / 2, cx + ww / 2, zz0, zz1)
                        else:
                            window(S, st, a, u, n, cx - ww / 2, cx + ww / 2, zz0, zz1, st["shutters"], room)
                        count["windows"] += 1
                        continue
                    french = street and (fi == wood_bal_floor or (st["balcony"] == "iron" and fi == 1))
                    if french:
                        zz0, zz1 = z0 + 0.05, z0 + min(fh - 0.6, 2.6)
                        window(S, st, a, u, n, cx - ww / 2, cx + ww / 2, zz0, zz1, st["shutters"] and fi != wood_bal_floor, room)
                        if st["balcony"] == "iron" and (st["iron_all"] or j % 2 == 0 or nb == 1):
                            iron_balcony(S, st, a, u, n, cx - ww / 2 - 0.3, cx + ww / 2 + 0.3, z0)
                            count["balconies"] += 1
                    else:
                        zz0 = z0 + 0.9
                        zz1 = min(z1 - 0.55, zz0 + 1.75)
                        if zz1 - zz0 < 0.8:
                            continue
                        window(S, st, a, u, n, cx - ww / 2, cx + ww / 2, zz0, zz1, st["shutters"], room)
                    count["windows"] += 1
    return count
