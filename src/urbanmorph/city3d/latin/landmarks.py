"""Landmark towers of the Latin theme (DESIGN.md §5.3; user references 2026-11-23 in architectural features/thin and
architectural features/cohesive): three designs, each in two forms.

* coral  - Coral Lantern: gold-yellow walls, coral strips, arches and tympana with white relief, cream trim, a coral
           ribbed dome with gabled dormers, a colonnaded lantern and a coral spire with a needle;
* eagle  - Eagle Belfry: ivory walls framed by green marble columns, peach mouldings and scroll pediments, belfries
           with bronze bells and round green balconies, a green balustrade, a stepped peach dome, a bronze eagle with
           raised wings;
* golden - Golden Clock Tower: gold walls with cream pilasters, a clock (teal dial, gilt numerals), a belfry of three
           gilt bells, an orange ribbed dome with cream relief and teal ovals, a colonnaded lantern, a gilt finial.

Forms: 'thin' - a slim shaft under a grand crown wider than the shaft, on an arcaded podium (the "slim shaft, grand
crown" sheets); 'cohesive' - one width from the podium to the dome, the ornament a tier of storeys repeated up the
shaft (the "slender ornate tropical tower" sheets). Square plan, built centred in a local frame (X along ``axis``, Y to
its left, Z up; the front faces -Y), then rotated onto the site. Ornament is relief and cut-out panels (filigree,
shells, balusters), as in the rest of the kit. Colours come from a scheme (role -> material key), so a house colour can
repaint any tower: landmark(..., colours={"wall": "wall:rosa"}).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.latin.church import BELL, balustrade, box_walls, lathe, ngon, terrace, volute, walls4
from urbanmorph.city3d.latin.kit import Soup, Z_B, beam, cylinder, dome, wbox, warch, warch_frame, wdisc, wp, wquad, wring, wside

FL, GF = 3.6, 6.0                  # storey and ground-floor heights (m)
DESIGNS, FORMS = ("coral", "eagle", "golden"), ("thin", "cohesive")
WIDTH = {("coral", "thin"): 17.0, ("coral", "cohesive"): 18.0, ("eagle", "thin"): 16.0, ("eagle", "cohesive"): 18.0,
         ("golden", "thin"): 17.0, ("golden", "cohesive"): 22.0}          # shaft side (m), from the sheets' proportions
# role -> material key; colours sampled from the reference sheets (palette.LANDMARK, DOMES, MARBLES)
SCHEMES = {
    "coral": {"wall": "wall:amarillo", "accent": "wall:coral", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:crema",
              "dome": "dome:cupula_coral", "spire": "wall:coral", "glass": "glass_teal", "metal": "gold",
              "rail": "baluster:blanco", "orn": "filigree:blanco", "frame": "lm:crema", "lit": "glass_lit"},
    "eagle": {"wall": "lm:marfil", "accent": "lm:melocoton", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:melocoton",
              "dome": "dome:cupula_melocoton", "glass": "glass_teal", "metal": "bronze", "rail": "baluster:verde",
              "column": "marble:verde", "green": "lm:verde", "frame": "lm:verde", "lit": "glass_lit"},
    "golden": {"wall": "lm:oro", "accent": "lm:rojo_naranja", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:crema",
               "dome": "dome:cupula_dorada", "glass": "glass_warm", "metal": "gold", "rail": "baluster:blanco",
               "teal": "lm:teal", "frame": "lm:teal", "orn": "filigree:crema", "shell": "shell:crema", "lit": "glass_lit"},
}
CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


# ----------------------------------------------------------------------------------------------- helpers
def faces(h):
    return walls4(-h, h, -h, h)


def slab(S, key, h, z0, z1):
    box_walls(S, key, -h, h, -h, h, z0, z1)


def mould(S, steps, z):
    """Stacked square slabs [(key, half side, height)], bottom up; returns the top."""
    for key, h, t in steps:
        slab(S, key, h, z, z + t)
        z += t
    return z


def oriented(S, key, V, F, ref, UV=None):
    """Triangles turned to face away from the point ref (curved pieces built point by point)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64).reshape(-1, 3)
    P = V[F]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    flip = np.einsum("ij,ij->i", nrm, P.mean(1) - np.asarray(ref, float)) < 0
    F[flip] = F[flip][:, ::-1]
    if UV is not None:
        UV = np.asarray(UV, float).copy()
        UV[flip] = UV[flip][:, ::-1]
    S.add(key, V, F, UV)


def wpoly(S, key, o, u, n, P, d, uv01=False, rep=1):
    """Flat panel in front of the wall: polygon P of (x, z), counter-clockwise, star-shaped about its vertex mean.
    uv01: UV 0-1 over the bounding box (images drawn for the panel: filigree, shells), repeated rep times along x."""
    P = np.asarray(P, float)
    c = P.mean(0)
    V = [wp(o, u, n, c[0], d, c[1])] + [wp(o, u, n, x, d, z) for x, z in P]
    m = len(P)
    F = np.array([[0, 1 + i, 1 + (i + 1) % m] for i in range(m)])
    UV = None
    if uv01:
        lo, hi = P.min(0), P.max(0)
        q = (np.r_[[c], P] - lo) / np.maximum(hi - lo, 1e-6) * [rep, 1]
        UV = q[F]
    S.add(key, V, F, UV)


def wband(S, key, o, u, n, A, B, d0, d1):
    """Moulded band between polylines A (inner) and B (outer) of (x, z), both running counter-clockwise round the
    opening: front face at d1, outer edge from d0 to d1."""
    V, F = [], []
    for i in range(len(A) - 1):
        b = len(V)
        V += [wp(o, u, n, A[i][0], d1, A[i][1]), wp(o, u, n, B[i][0], d1, B[i][1]), wp(o, u, n, B[i + 1][0], d1, B[i + 1][1]),
              wp(o, u, n, A[i + 1][0], d1, A[i + 1][1]), wp(o, u, n, B[i][0], d0, B[i][1]), wp(o, u, n, B[i + 1][0], d0, B[i + 1][1])]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3], [b + 1, b + 4, b + 5], [b + 1, b + 5, b + 2]]
    S.add(key, V, F)


def pointed(x0, x1, zs, f=0.6, k=8, zk=1.0, grow=0.0):
    """Pointed arch from the right spring over the apex to the left spring: two arcs of radius f * span centred on the
    springing line (grow: offset outward about the same centres); heights scaled by zk."""
    s = x1 - x0
    R, xc = f * s, (x0 + x1) / 2
    Rg = R + grow
    ta = np.arccos((xc - (x1 - R)) / Rg)
    right = [(x1 - R + Rg * np.cos(t), zs + zk * Rg * np.sin(t)) for t in np.linspace(0, ta, k + 1)]
    left = [(x0 + R + Rg * np.cos(t), zs + zk * Rg * np.sin(t)) for t in np.linspace(np.pi - ta, np.pi, k + 1)[1:]]
    return np.array(right + left)


def segment(x0, x1, zs, rise, k=10, grow=0.0):
    """Segmental arch through (x0, zs), (x1, zs) rising by ``rise``, right to left (grow: concentric offset)."""
    s = x1 - x0
    R = (s * s / 4 + rise * rise) / (2 * rise)
    xc, zc = (x0 + x1) / 2, zs + rise - R
    t1 = np.arctan2(zs - zc, x1 - xc)
    return np.array([(xc + (R + grow) * np.cos(t), zc + (R + grow) * np.sin(t)) for t in np.linspace(t1, np.pi - t1, k + 1)])


def clip_above(P, zc):
    """Part of a convex polygon (x, z) above z = zc."""
    p = shapely.Polygon(P).intersection(shapely.box(-1e4, zc, 1e4, 1e4))
    if p.is_empty or p.geom_type != "Polygon":
        return None
    return np.asarray(shapely.orient_polygons(p).exterior.coords)[:-1]


def ball(S, key, c, z, r, k=8):
    lathe(S, key, c, [(0.0, 0.0), (0.71 * r, 0.29 * r), (r, r), (0.71 * r, 1.71 * r), (0.0, 2 * r)], z, k=k)


def column(S, key, cap, base, c, r, z0, z1, k=10, cap_h=0.55, flare=1.45):
    """Round column: square plinth, shaft, flared capital (Corinthian bell when tall) with an abacus."""
    box_walls(S, base, c[0] - 1.3 * r, c[0] + 1.3 * r, c[1] - 1.3 * r, c[1] + 1.3 * r, z0, z0 + 0.3)
    cylinder(S, key, c, r, 0.92 * r, z0 + 0.3, z1 - cap_h, k=k)
    capital(S, cap, c, r, z1, cap_h, flare, k)


def capital(S, key, c, r, z1, h=0.55, flare=1.45, k=10):
    cylinder(S, key, c, 0.95 * r, flare * r, z1 - h, z1 - 0.18, k=k)
    box_walls(S, key, c[0] - (flare + 0.05) * r, c[0] + (flare + 0.05) * r, c[1] - (flare + 0.05) * r, c[1] + (flare + 0.05) * r,
              z1 - 0.18, z1)


def obelisk(S, key, x, y, z, s=0.35):
    """Slim pinnacle: pedestal, cap, four-sided needle, ball."""
    box_walls(S, key, x - s, x + s, y - s, y + s, z, z + 1.4 * s)
    box_walls(S, key, x - 1.2 * s, x + 1.2 * s, y - 1.2 * s, y + 1.2 * s, z + 1.4 * s, z + 1.65 * s)
    cylinder(S, key, (x, y), 0.8 * s, 0.05, z + 1.65 * s, z + 5.5 * s, k=4)
    ball(S, key, (x, y), z + 5.2 * s, 0.32 * s, k=6)


ONION = [(0.3, 0.0), (0.55, 0.12), (0.62, 0.35), (0.5, 0.6), (0.22, 0.82), (0.1, 0.95), (0.16, 1.05), (0.0, 1.25)]


def onion(S, key, ped, x, y, z, s=0.5):
    """Gilt finial (acorn on a ball) on a small pedestal."""
    box_walls(S, ped, x - 0.55 * s, x + 0.55 * s, y - 0.55 * s, y + 0.55 * s, z, z + 0.7 * s)
    lathe(S, key, (x, y), [(r * s, h * s) for r, h in ONION], z + 0.7 * s, k=8)
    cylinder(S, key, (x, y), 0.05 * s, 0.01, z + 1.9 * s, z + 2.8 * s, k=4)


def cypress(S, x, y, z, h):
    """Clipped cypress in a terracotta pot (setbacks of the thin towers)."""
    cylinder(S, "plinth:terracota", (x, y), 0.1 * h, 0.14 * h, z, z + 0.22 * h, k=8, cap=True)
    lathe(S, "leaves_dark", (x, y), [(0.12 * h, 0.0), (0.17 * h, 0.15 * h), (0.15 * h, 0.4 * h), (0.09 * h, 0.62 * h),
                                     (0.0, 0.8 * h)], z + 0.2 * h, k=8)


def potted(S, x, y, z, rng):
    """Potted plant on a balcony."""
    h = float(rng.uniform(0.6, 1.0))
    cylinder(S, "plinth:terracota", (x, y), 0.14, 0.18, z, z + 0.3, k=6, cap=True)
    lathe(S, "leaves", (x, y), [(0.12, 0.0), (0.32 * h, 0.25 * h), (0.28 * h, 0.5 * h), (0.0, 0.75 * h)], z + 0.28, k=6)


def bell(S, key, c, z_top, r):
    h = 1.1 * r
    lathe(S, key, c, [(rr * r, hh * h) for rr, hh in BELL], z_top - h, k=10)


def win(S, sc, o, u, n, xc, z0, w, h, frame=None):
    """Window: surround, glass, optional mullion and transom (frame key), sill."""
    T = sc["trim"]
    wquad(S, T, o, u, n, xc - w / 2 - 0.2, xc + w / 2 + 0.2, z0 - 0.12, z0 + h + 0.2, 0.03)
    wquad(S, sc["glass"], o, u, n, xc - w / 2, xc + w / 2, z0, z0 + h, 0.06)
    if frame:
        wbox(S, frame, o, u, n, xc - 0.05, xc + 0.05, z0, z0 + h, 0.06, 0.1)
        wbox(S, frame, o, u, n, xc - w / 2, xc + w / 2, z0 + 0.66 * h - 0.04, z0 + 0.66 * h + 0.04, 0.06, 0.1)
    wbox(S, T, o, u, n, xc - w / 2 - 0.28, xc + w / 2 + 0.28, z0 - 0.3, z0 - 0.12, 0.0, 0.2)


def bay_balcony(S, sc, o, u, n, x0, x1, z, D=0.5, h=1.0, rng=None, plant=0.0):
    """Balcony across a bay: slab, cut-out balusters (front and sides), coping; sometimes a potted plant."""
    T, R = sc["trim"], sc["rail"]
    wbox(S, T, o, u, n, x0, x1, z - 0.18, z, 0.0, D)
    wquad(S, R, o, u, n, x0 + 0.03, x1 - 0.03, z, z + h, D - 0.05, uv="panel")
    wside(S, R, o, u, n, x0 + 0.04, 0.0, D - 0.05, z, z + h)
    wside(S, R, o, u, n, x1 - 0.04, 0.0, D - 0.05, z, z + h, flip=True)
    wbox(S, T, o, u, n, x0, x1, z + h, z + h + 0.1, D - 0.12, D + 0.02)
    if rng is not None and rng.uniform() < plant:
        p = wp(o, u, n, float(rng.uniform(x0 + 0.4, x1 - 0.4)), D * 0.45, 0.0)
        potted(S, p[0], p[1], z, rng)


def round_balcony(S, sc, o, u, n, xc, z, r, k=8):
    """Semicircular balcony (Eagle Belfry): slab, curved cut-out balustrade, coping."""
    T, R, G = sc["trim"], sc["rail"], sc.get("green", sc["trim"])
    rim = [(xc + r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, np.pi, k + 1)]
    for zz, order in ((z, 1), (z - 0.25, 0)):              # (u, n) turns clockwise: [0, i + 2, i + 1] faces up
        V = [wp(o, u, n, xc, 0.0, zz)] + [wp(o, u, n, x, d, zz) for x, d in rim]
        S.add(T, V, [[0, i + 2, i + 1] if order else [0, i + 1, i + 2] for i in range(k)])
    ref = wp(o, u, n, xc, 0.0, z)
    V, F, UV = [], [], []
    s = 0.0
    for i in range(k):
        (xa, da), (xb, db) = rim[i], rim[i + 1]
        b = len(V)
        V += [wp(o, u, n, xa, da, z - 0.25), wp(o, u, n, xb, db, z - 0.25), wp(o, u, n, xb, db, z), wp(o, u, n, xa, da, z)]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    oriented(S, T, V, F, ref)
    V, F = [], []
    rr = r - 0.06
    pts = [(xc + rr * np.cos(a), rr * np.sin(a)) for a in np.linspace(0, np.pi, k + 1)]
    for i in range(k):
        (xa, da), (xb, db) = pts[i], pts[i + 1]
        L = float(np.hypot(xb - xa, db - da))
        b = len(V)
        V += [wp(o, u, n, xa, da, z), wp(o, u, n, xb, db, z), wp(o, u, n, xb, db, z + 1.0), wp(o, u, n, xa, da, z + 1.0)]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        c = np.array([[s, 0], [s + L, 0], [s + L, 1.0], [s, 1.0]])
        UV += [c[[0, 1, 2]], c[[0, 2, 3]]]
        s += L
        beam(S, G, wp(o, u, n, xa, da, z + 1.05), wp(o, u, n, xb, db, z + 1.05), 0.16, 0.1)
    oriented(S, R, V, F, ref, np.array(UV))


def corbels(S, key, h, z, D, step=1.4):
    """Stepped brackets under a projecting cornice, on every face of the square of half side h, tops at z."""
    for o, u, n, L in faces(h):
        wbox(S, key, o, u, n, -0.05, L + 0.05, z - 1.8, z - 1.55, 0.0, 0.2)
        m = max(2, int(round(L / step)))
        for j in range(m + 1):
            x = min(max(L * j / m, 0.25), L - 0.25)
            for z0, z1, dd in ((z - 1.5, z - 1.0, 0.35), (z - 1.0, z - 0.5, 0.65), (z - 0.5, z, 1.0)):
                wbox(S, key, o, u, n, x - 0.22, x + 0.22, z0, z1, 0.0, dd * D)


def clock(S, o, u, n, xc, zc, r, d, trim):
    """Clock face (image) in a gilt ring and a moulded surround."""
    k = 24
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = [wp(o, u, n, xc, d, zc)] + [wp(o, u, n, xc + r * np.cos(t), d, zc + r * np.sin(t)) for t in a]
    F = np.array([[0, 1 + i, 1 + (i + 1) % k] for i in range(k)])
    q = np.r_[[[0.5, 0.5]], np.c_[0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)]]
    S.add("clock", V, F, q[F])
    wring(S, "gold", o, u, n, xc, zc, 0.97 * r, 1.08 * r, d + 0.02, k=k)
    wring(S, trim, o, u, n, xc, zc, 1.08 * r, 1.3 * r, d - 0.02, k=k)


def oval(S, key, ring, o, u, n, xc, zc, rx, rz, d, k=16):
    """Oval window or medallion with a moulded ring."""
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = [wp(o, u, n, xc, d, zc)] + [wp(o, u, n, xc + rx * np.cos(t), d, zc + rz * np.sin(t)) for t in a]
    S.add(key, V, [[0, 1 + i, 1 + (i + 1) % k] for i in range(k)])
    V, F = [], []
    for i in range(k):
        j = (i + 1) % k
        b = len(V)
        V += [wp(o, u, n, xc + s * rx * np.cos(t), d + 0.03, zc + s * rz * np.sin(t)) for t, s in
              ((a[i], 1.0), (a[i], 1.3), (a[j], 1.3), (a[j], 1.0))]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    S.add(ring, V, F)


def dome_oval(S, key, ring, rd, rise, z0, theta, frac, rx, rz, k=14):
    """Oval window on a dome (quarter-ellipse profile about the origin): at azimuth theta, at height frac of the rise."""
    phi = np.arcsin(frac)
    P = np.array([rd * np.cos(phi) * np.cos(theta), rd * np.cos(phi) * np.sin(theta), z0 + rise * rd * frac])
    N = np.array([np.cos(phi) * np.cos(theta) / rd, np.cos(phi) * np.sin(theta) / rd, np.sin(phi) / (rise * rd)])
    N /= np.linalg.norm(N)
    T1 = np.array([-np.sin(theta), np.cos(theta), 0.0])
    T2 = np.cross(N, T1)
    T2 *= np.sign(T2[2]) or 1.0
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c = P + 0.12 * N
    V = [c] + [c + T1 * rx * np.cos(t) + T2 * rz * np.sin(t) for t in a]
    oriented(S, key, V, [[0, 1 + i, 1 + (i + 1) % k] for i in range(k)], P - N)
    V, F = [], []
    for i in range(k):
        j = (i + 1) % k
        b = len(V)
        V += [P + 0.08 * N + T1 * s * rx * np.cos(t) + T2 * s * rz * np.sin(t) for t, s in
              ((a[i], 1.0), (a[i], 1.35), (a[j], 1.35), (a[j], 1.0))]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    oriented(S, ring, V, F, P - N)


def eagle(S, key, c, z, s):
    """Bronze eagle with raised wings, facing -Y, on a ball (s: wingspan, m)."""
    x, y = c
    ball(S, key, c, z, 0.07 * s)
    z += 0.12 * s
    lathe(S, key, c, [(0.0, 0.0), (0.07 * s, 0.03 * s), (0.1 * s, 0.13 * s), (0.09 * s, 0.24 * s), (0.06 * s, 0.31 * s),
                      (0.045 * s, 0.34 * s), (0.0, 0.35 * s)], z, k=8)
    ball(S, key, (x, y - 0.03 * s), z + 0.32 * s, 0.05 * s, k=8)                       # head
    tip = [x, y - 0.13 * s, z + 0.35 * s]                                                # beak
    V = [tip, [x - 0.025 * s, y - 0.06 * s, z + 0.37 * s], [x + 0.025 * s, y - 0.06 * s, z + 0.37 * s],
         [x, y - 0.06 * s, z + 0.32 * s]]
    oriented(S, key, V, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], [x, y - 0.07 * s, z + 0.35 * s])
    wing = np.array([(0.05, 0.0, 0.26), (0.18, 0.02, 0.44), (0.32, 0.04, 0.62), (0.45, 0.06, 0.8), (0.5, 0.07, 0.76),
                     (0.48, 0.07, 0.68), (0.52, 0.08, 0.64), (0.48, 0.08, 0.57), (0.5, 0.08, 0.52), (0.44, 0.08, 0.47),
                     (0.44, 0.08, 0.4), (0.36, 0.08, 0.36), (0.33, 0.07, 0.28), (0.24, 0.06, 0.22), (0.12, 0.04, 0.14)])
    for sx in (-1, 1):
        P = np.c_[x + sx * wing[:, 0] * s, y + wing[:, 1] * s, z + wing[:, 2] * s]
        root = np.array([x + sx * 0.08 * s, y + 0.03 * s, z + 0.24 * s])
        V = np.r_[[root], P]
        F = [[0, i + 1, i + 2] for i in range(len(P) - 1)]
        S.add(key, V, F)
        S.add(key, V, [f[::-1] for f in F])
    T = np.array([[x - 0.05 * s, y + 0.07 * s, z + 0.06 * s], [x + 0.05 * s, y + 0.07 * s, z + 0.06 * s],
                  [x + 0.1 * s, y + 0.2 * s, z - 0.08 * s], [x - 0.1 * s, y + 0.2 * s, z - 0.08 * s]])      # tail
    S.add(key, T, [[0, 1, 2], [0, 2, 3]])
    S.add(key, T, [[0, 2, 1], [0, 3, 2]])


def lantern(S, sc, r, z, h, roof="cone"):
    """Open lantern on the dome: base ring, eight columns round a dark core, cornice, then a coral cone ('cone') or a
    gilt cupola ('gold') with a ball and a needle or finial; returns the top."""
    T = sc["trim"]
    c = (0.0, 0.0)
    cylinder(S, T, c, r + 0.3, r + 0.3, z, z + 0.4, k=16, cap=True)
    cylinder(S, "void", c, 0.6 * r, 0.6 * r, z + 0.4, z + 0.4 + h, k=8)
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8:
        column(S, T, T, T, (r * np.cos(a), r * np.sin(a)), max(0.12, 0.09 * r), z + 0.4, z + 0.4 + h, k=6)
    z += 0.4 + h
    cylinder(S, T, c, r + 0.25, r + 0.25, z, z + 0.45, k=16, cap=True)
    z += 0.45
    if roof == "cone":
        cylinder(S, sc["spire"], c, 0.8 * r, 0.8 * r, z, z + 0.55 * r, k=8)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            p = (0.82 * r * np.cos(a + np.pi / 8), 0.82 * r * np.sin(a + np.pi / 8))
            cylinder(S, "void", p, 0.12 * r, 0.12 * r, z + 0.12 * r, z + 0.4 * r, k=4)
        z += 0.55 * r
        cylinder(S, T, c, 0.85 * r, 0.85 * r, z, z + 0.25, k=8, cap=True)
        cylinder(S, sc["spire"], c, 0.8 * r, 0.06, z + 0.25, z + 0.25 + 2.3 * r, k=8)
        z += 0.25 + 2.3 * r
        ball(S, sc["metal"], c, z - 0.3, 0.22 + 0.05 * r)
        cylinder(S, "iron", c, 0.06, 0.01, z, z + 1.8 * r + 1.5, k=4)
        return z + 1.8 * r + 1.5
    top = dome(S, sc["metal"], c, r + 0.15, z, k=12, m=4, rise=1.0)
    lathe(S, sc["metal"], c, [(0.12 * r, 0.0), (0.3 * r, 0.25 * r), (0.32 * r, 0.5 * r), (0.12 * r, 0.75 * r),
                              (0.22 * r, 0.95 * r), (0.0, 1.3 * r)], top - 0.1, k=8)
    cylinder(S, sc["metal"], c, 0.05 * r + 0.03, 0.01, top + 1.2 * r, top + 2.6 * r, k=4)
    return top + 2.6 * r


def dormer(S, sc, o, u, n, wd, z, h, depth, body=None, gable=0.62, glass=True):
    """Gabled dormer standing out of a dome (front plane d = 0, x in [0, wd]): a body reaching back into the dome, an
    arched window with a small balcony, a pointed gable with a white relief tympanum and a finial."""
    T = sc["trim"]
    body = body or sc["wall"]
    wbox(S, body, o, u, n, 0.0, wd, z, z + h, -depth, 0.0)
    if glass:
        a0, a1 = 0.28 * wd, 0.72 * wd
        zs = z + h - 0.35 - (a1 - a0) / 2
        warch(S, sc["glass"], o, u, n, a0, a1, z + 0.5, zs, 0.04, k=8)
        warch_frame(S, T, o, u, n, a0, a1, z + 0.5, zs, 0.16, 0.0, 0.12, k=8)
        bay_balcony(S, sc, o, u, n, a0 - 0.25, a1 + 0.25, z + 0.5, D=0.4, h=0.8)
    wbox(S, T, o, u, n, -0.15, wd + 0.15, z + h - 0.2, z + h + 0.1, 0.0, 0.25)
    inner = pointed(0.0, wd, z + h + 0.1, f=0.62, k=6, zk=gable / 0.62)
    outer = pointed(0.0, wd, z + h + 0.1, f=0.62, k=6, zk=gable / 0.62, grow=0.3)
    wpoly(S, sc["accent"] if glass else body, o, u, n, inner, 0.0)
    if "orn" in sc:
        wpoly(S, sc["orn"], o, u, n, inner, 0.04, uv01=True)
    wband(S, T, o, u, n, inner, outer, -depth, 0.15)
    p = wp(o, u, n, wd / 2, 0.0, 0.0)
    obelisk(S, T, p[0], p[1], float(outer[:, 1].max()) - 0.1, 0.16 + 0.02 * wd)


def dome_dormers(S, sc, rd, z, wd, h, gable=0.62, body=None, glass=True, which=(0, 1, 2, 3)):
    """Dormers on the dome facing the four faces (front = face 0, -Y)."""
    for i, (o, u, n, L) in enumerate(faces(rd)):
        if i not in which:
            continue
        o2 = np.asarray(n) * (rd + 0.25) - np.asarray(u) * (wd / 2)
        dormer(S, sc, o2, u, n, wd, z, h, rd * 0.9, body, gable, glass)
        for x0, sgn in ((0.0, -1), (wd, 1)):                      # scrolls on both sides
            a = wp(o2, u, n, x0 + sgn * 0.3 * wd, -0.35 * wd, z + 0.15)
            b = wp(o2, u, n, x0, -0.2, z + 0.65 * h)
            volute(S, sc["accent"] if glass else sc["trim"], a, b, w=0.12 * wd + 0.1)


def podium(S, sc, h, z1, big=False, roundels=True, n_bays=3):
    """Arcaded ground storeys on the square of half side h up to z1: tall arched shopfronts (warm glass, mullion and
    transom), surrounds, piers, roundels in the spandrels, a frieze and a cornice; returns the cornice top."""
    T, A, B = sc["trim"], sc["accent"], sc["base"]
    box_walls(S, B, -h, h, -h, h, Z_B, z1)
    zf = z1 - 1.4
    for o, u, n, L in faces(h):
        wbox(S, "stone", o, u, n, -0.05, L + 0.05, Z_B, 0.6, 0.0, 0.14)
        wts = np.array([1.0, 1.35, 1.0] if big else [1.0] * n_bays)
        pier = max(1.0, 0.08 * L)
        xs = pier + np.r_[0, np.cumsum(wts)] / wts.sum() * (L - 2 * pier)
        for j in range(len(wts)):
            a0, a1 = xs[j] + 0.45, xs[j + 1] - 0.45
            r = (a1 - a0) / 2
            zs = zf - 0.55 - r
            warch(S, sc["lit"], o, u, n, a0, a1, 0.15, zs, 0.05, k=10)
            warch_frame(S, T, o, u, n, a0, a1, 0.15, zs, 0.32, 0.0, 0.16, k=10)
            wbox(S, T, o, u, n, a0, a1, zs - 0.12, zs + 0.04, 0.05, 0.1)
            wbox(S, T, o, u, n, a0 + r - 0.06, a0 + r + 0.06, 0.15, zs, 0.05, 0.1)
        for j, x in enumerate(xs):
            if 0 < j < len(xs) - 1:
                wbox(S, T, o, u, n, x - 0.3, x + 0.3, 0.6, zf, 0.0, 0.12)
            if roundels:
                zc = zf - 0.55 - 0.15 * (xs[1] - xs[0])
                wring(S, T, o, u, n, x, zc, 0.42, 0.58, 0.08, k=14)
                wdisc(S, A, o, u, n, x, zc, 0.42, 0.06, k=14)
        wbox(S, sc["band"], o, u, n, -0.1, L + 0.1, zf, zf + 0.3, 0.0, 0.22)
    return mould(S, [(T, h + 0.15, 0.5), (sc["band"], h + 0.35, 0.3), (T, h + 0.45, 0.25)], z1 - 0.5)


def piers(S, key, o, u, n, L, z0, z1, f=0.09, d=0.25):
    wbox(S, key, o, u, n, 0.0, f * L, z0, z1, 0.0, d)
    wbox(S, key, o, u, n, L - f * L, L, z0, z1, 0.0, d)


def pedestal(S, sc, q, z, nf=2):
    """Short stage between the podium and a thin shaft: walls, cream corner piers, three windows a storey."""
    T = sc["trim"]
    box_walls(S, sc["wall"], -q, q, -q, q, z, z + nf * FL)
    for o, u, n, L in faces(q):
        piers(S, T, o, u, n, L, z, z + nf * FL)
        for f in range(nf):
            for j in range(3):
                win(S, sc, o, u, n, L * (0.25 + 0.25 * j), z + f * FL + 0.9, 1.3, 2.0, sc.get("frame"))
    return mould(S, [(T, q + 0.1, 0.35), (T, q + 0.3, 0.3)], z + nf * FL)


# ----------------------------------------------------------------------------------------------- Coral Lantern
def coral_dome(S, sc, rd, z, rise, rng):
    """Coral ribbed dome on a cream ring, four gabled dormers, the lantern and spire; returns the top."""
    T = sc["trim"]
    cylinder(S, T, (0, 0), rd + 0.4, rd + 0.4, z, z + 0.6, k=24, cap=True)
    cylinder(S, sc["accent"], (0, 0), rd + 0.15, rd + 0.15, z + 0.6, z + 0.9, k=24)
    zd = z + 0.9
    top = dome(S, sc["dome"], (0, 0), rd, zd, k=24, m=9, rise=rise, ribs_key=T, ribs=8, rib_w=0.2 + 0.012 * rd)
    wd = 0.44 * rd
    dome_dormers(S, sc, rd, zd, wd, 0.34 * rd)
    rl = 0.3 * rd
    return lantern(S, sc, rl, top - 0.25, 1.2 * rl, "cone")


def coral_thin(S, sc, w, H, rng):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw, P = w / 2, 0.7 * w
    z = podium(S, sc, P, GF + FL)
    q = 0.6 * w
    terrace(S, shapely.box(-P - 0.3, -P - 0.3, P + 0.3, P + 0.3), shapely.box(-q, -q, q, q), z + 0.02)
    balustrade(S, faces(P + 0.3), z)
    for sx, sy in CORNERS:
        cypress(S, sx * (P - 0.6), sy * (P - 0.6), z, 3.4)
    z = pedestal(S, sc, q, z)
    terrace(S, shapely.box(-q - 0.3, -q - 0.3, q + 0.3, q + 0.3), shapely.box(-hw, -hw, hw, hw), z + 0.02)
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.4), sy * (q - 0.4), z, 2.6)
    # shaft: corner piers, a coral centre strip with a balcony every storey, a column of windows each side
    crown = 2.95 * w + 12.0
    n = max(8, round((H - z - crown) / FL))
    zs1 = z + n * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z, zs1 + 2.0)
    for o, u, n_, L in faces(hw):
        piers(S, T, o, u, n_, L, z, zs1 + 2.0, 0.09, 0.2)
        x0, x1 = 0.38 * L, 0.62 * L
        wquad(S, A, o, u, n_, x0, x1, z, zs1 + 0.3, 0.02)
        wbox(S, T, o, u, n_, x0 - 0.35, x0, z, zs1 + 0.3, 0.0, 0.3)
        wbox(S, T, o, u, n_, x1, x1 + 0.35, z, zs1 + 0.3, 0.0, 0.3)
        for f in range(n):
            zf = z + f * FL
            for xc in (0.225 * L, 0.775 * L):
                win(S, sc, o, u, n_, xc, zf + 0.8, min(1.5, 0.11 * L), 2.2, T)
            if f < n - 2:
                wquad(S, T, o, u, n_, 0.5 * L - 0.95, 0.5 * L + 0.95, zf + 0.1, zf + 2.95, 0.035)
                wquad(S, sc["glass"], o, u, n_, 0.5 * L - 0.75, 0.5 * L + 0.75, zf + 0.25, zf + 2.75, 0.05)
                bay_balcony(S, sc, o, u, n_, x0 + 0.05, x1 - 0.05, zf + 0.25, D=0.6, h=0.95)
        zt = z + (n - 2) * FL + 0.4
        zs = zs1 - 0.5 - (x1 - x0 - 0.8) / 2
        warch(S, sc["glass"], o, u, n_, x0 + 0.4, x1 - 0.4, zt, zs, 0.05, k=10)
        warch_frame(S, T, o, u, n_, x0 + 0.4, x1 - 0.4, zt, zs, 0.3, 0.0, 0.18, k=10)
        bay_balcony(S, sc, o, u, n_, x0 + 0.05, x1 - 0.05, zt, D=0.6, h=0.95)
    # grand crown: bracketed cornice, an arcaded stage, an oculus stage, corner turrets, the dome
    z = zs1 + 2.0
    corbels(S, T, hw, z, 0.72 * w - hw)
    z = mould(S, [(T, 0.72 * w, 0.35), (T, 0.75 * w, 0.45)], z)
    c1, h1 = 0.66 * w, 0.62 * w
    box_walls(S, Wl, -c1, c1, -c1, c1, z, z + h1)
    for o, u, n_, L in faces(c1):
        piers(S, T, o, u, n_, L, z, z + h1, 0.1, 0.3)
        for a0, a1, tall in ((0.15 * L, 0.33 * L, 0.74), (0.38 * L, 0.62 * L, 0.86), (0.67 * L, 0.85 * L, 0.74)):
            zs = z + tall * h1 - (a1 - a0) / 2
            warch(S, A, o, u, n_, a0 - 0.45, a1 + 0.45, z + 0.6, zs, 0.02, k=10)
            warch(S, sc["glass"], o, u, n_, a0, a1, z + 1.4, zs, 0.05, k=10)
            warch_frame(S, T, o, u, n_, a0, a1, z + 1.4, zs, 0.22, 0.0, 0.14, k=10)
            bay_balcony(S, sc, o, u, n_, a0 - 0.3, a1 + 0.3, z + 1.4, D=0.55, h=0.95)
        for x in (0.355 * L, 0.645 * L):
            wbox(S, T, o, u, n_, x - 0.25, x + 0.25, z, z + h1, 0.0, 0.22)
    z = mould(S, [(T, c1 + 0.2, 0.4), (T, c1 + 0.4, 0.35)], z + h1)
    for sx, sy in CORNERS:
        obelisk(S, T, sx * (c1 + 0.05), sy * (c1 + 0.05), z, 0.45)
    c2, h2 = 0.6 * w, 0.45 * w
    box_walls(S, Wl, -c2, c2, -c2, c2, z, z + h2)
    for o, u, n_, L in faces(c2):
        piers(S, T, o, u, n_, L, z, z + h2, 0.1, 0.3)
        for xa, xb in ((0.14 * L, 0.34 * L), (0.66 * L, 0.86 * L)):
            wquad(S, A, o, u, n_, xa, xb, z + 0.5, z + h2 - 0.6, 0.02)
            xc, zc = (xa + xb) / 2, z + 0.5 * h2
            s_ = 0.22 * (xb - xa)
            wpoly(S, T, o, u, n_, [(xc, zc - 1.5 * s_), (xc + s_, zc - 0.4 * s_), (xc + s_, zc + s_), (xc - s_, zc + s_),
                                    (xc - s_, zc - 0.4 * s_)], 0.06)
            wdisc(S, A, o, u, n_, xc, zc, 0.45 * s_, 0.09, k=10)
        r = 0.12 * L
        wring(S, T, o, u, n_, 0.5 * L, z + 0.5 * h2, r, 1.3 * r, 0.1, k=20)
        wdisc(S, A, o, u, n_, 0.5 * L, z + 0.5 * h2, r, 0.06, k=20)
        wring(S, T, o, u, n_, 0.5 * L, z + 0.5 * h2, 0.5 * r, 0.65 * r, 0.09, k=16)
    z = mould(S, [(T, c2 + 0.25, 0.4), (T, c2 + 0.45, 0.4)], z + h2)
    for sx, sy in CORNERS:
        x, y = sx * (c2 - 0.5), sy * (c2 - 0.5)
        box_walls(S, Wl, x - 0.6, x + 0.6, y - 0.6, y + 0.6, z, z + 0.3 * w)
        box_walls(S, T, x - 0.75, x + 0.75, y - 0.75, y + 0.75, z + 0.3 * w, z + 0.3 * w + 0.3)
        obelisk(S, T, x, y, z + 0.3 * w + 0.3, 0.35)
    terrace(S, shapely.box(-c2 - 0.45, -c2 - 0.45, c2 + 0.45, c2 + 0.45), ngon((0, 0), 0.52 * w + 0.4, k=24), z + 0.02)
    return coral_dome(S, sc, 0.52 * w, z, 1.2, rng), n + 6, {"crown_z": zs1}


def coral_tier(S, sc, hw, z, nf, rng, gable=True):
    """One tier of the cohesive Coral Lantern on every face: cornice, cream corner pilasters with coral strips, a triple
    arcade of coral arches (nf storeys of windows and balconies), a pointed gable with a white relief tympanum and
    an oculus; returns the tier top."""
    T, A = sc["trim"], sc["accent"]
    top = z
    for o, u, n, L in faces(hw):
        xa0, xa1 = 0.17 * L, 0.83 * L
        bay = (xa1 - xa0) / 3
        zf0 = z + 0.65
        zs = zf0 + nf * FL - 0.4
        wbox(S, T, o, u, n, -0.15, L + 0.15, z, z + 0.35, 0.0, 0.3)
        for j in range(3):
            a0, a1 = xa0 + j * bay + 0.3, xa0 + (j + 1) * bay - 0.3
            warch(S, A, o, u, n, a0, a1, zf0 - 0.3, zs, 0.03, k=10)
            warch_frame(S, A, o, u, n, a0, a1, zf0 - 0.3, zs, 0.32, 0.0, 0.14, k=10, keystone=False)
            warch_frame(S, T, o, u, n, a0 + 0.1, a1 - 0.1, zf0 - 0.3, zs, 0.1, 0.0, 0.17, k=10, keystone=False)
            warch(S, sc["lit"], o, u, n, a0 + 0.3, a1 - 0.3, zs - 0.25, zs, 0.05, k=8)
            xc, gw = (a0 + a1) / 2, (a1 - a0) - 0.55
            for f in range(nf):
                zf = zf0 + f * FL
                wquad(S, sc["lit"], o, u, n, xc - gw / 2, xc + gw / 2, zf + 0.15, zf + 2.6, 0.06)
                wbox(S, T, o, u, n, xc - 0.06, xc + 0.06, zf + 0.15, zf + 2.6, 0.06, 0.11)
                bay_balcony(S, sc, o, u, n, a0 + 0.05, a1 - 0.05, zf + 0.1, D=0.45, h=0.95, rng=rng, plant=0.45)
        for j in range(4):
            p = wp(o, u, n, xa0 + j * bay, 0.3, 0.0)
            column(S, T, T, T, (p[0], p[1]), 0.2, zf0 - 0.3, zs + 0.2, k=8)
        zt = zs + bay / 2 + 0.6
        if gable:
            gx0, gx1 = xa0 - 0.2, xa1 + 0.2
            zg = zs + 0.15
            inner = pointed(gx0, gx1, zg, f=0.57, k=8, zk=0.74)
            outer = pointed(gx0, gx1, zg, f=0.57, k=8, zk=0.74, grow=0.4)
            wpoly(S, A, o, u, n, inner, 0.02)
            vis = clip_above(inner, zs + bay / 2 + 0.25)
            if vis is not None:
                wpoly(S, sc["orn"], o, u, n, vis, 0.05, uv01=True)
            wband(S, T, o, u, n, inner, outer, 0.0, 0.25)
            ap = float(outer[:, 1].max())
            p = wp(o, u, n, L / 2, 0.15, 0.0)
            obelisk(S, T, p[0], p[1], ap - 0.25, 0.2)
            zo = ap + 1.5
            wring(S, T, o, u, n, L / 2, zo, 0.75, 1.0, 0.08, k=18)
            wdisc(S, A, o, u, n, L / 2, zo, 0.75, 0.06, k=18)
            zt = zo + 1.3
        top = max(top, zt)
    for o, u, n, L in faces(hw):                                  # corner pilasters, coral strips, capitals
        piers(S, T, o, u, n, L, z + 0.35, top, 0.075, 0.3)
        for xa, xb in ((0.075 * L, 0.115 * L), (0.885 * L, 0.925 * L)):
            wquad(S, A, o, u, n, xa, xb, z + 0.35, top, 0.02)
        for xa, xb in ((-0.15, 0.075 * L + 0.15), (0.925 * L - 0.15, L + 0.15)):
            wbox(S, T, o, u, n, xa, xb, top - 0.5, top, 0.0, 0.42)
    return top


def coral_cohesive(S, sc, w, H, rng):
    T, Wl = sc["trim"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.35, GF + 0.6 * FL)
    z0 = z
    top_h = 2.3 * w + 14.0                                        # top tier, dome, lantern, spire
    Ht = coral_tier(Soup(), sc, hw, 0.0, 3, np.random.default_rng(0))
    n_t = max(2, round((H - z - top_h) / Ht))
    for i in range(n_t):
        z = coral_tier(S, sc, hw, z, 3, rng)
    zc = z
    z = coral_tier(S, sc, hw, z, 2, rng, gable=False)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    z = mould(S, [(T, hw + 0.25, 0.4), (T, hw + 0.45, 0.35)], z)
    rd = 0.47 * w
    for sx, sy in CORNERS:
        x, y = sx * (hw - 0.5), sy * (hw - 0.5)
        cylinder(S, T, (x, y), 0.8, 0.8, z, z + 0.4 * w, k=8, cap=True)
        for a in np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 8:
            u_, n_ = np.array([-np.sin(a), np.cos(a)]), np.array([np.cos(a), np.sin(a)])
            warch(S, "void", np.array([x, y]) + 0.74 * n_ - 0.2 * u_, u_, n_, 0.0, 0.4, z + 0.22 * w, z + 0.32 * w, 0.02, k=6)
        cylinder(S, sc["spire"], (x, y), 0.95, 0.05, z + 0.4 * w, z + 0.4 * w + 0.2 * w, k=8)
        ball(S, sc["metal"], (x, y), z + 0.6 * w - 0.2, 0.18)
    terrace(S, shapely.box(-hw - 0.45, -hw - 0.45, hw + 0.45, hw + 0.45), ngon((0, 0), rd + 0.4, k=24), z + 0.02)
    return coral_dome(S, sc, rd, z, 1.15, rng), 1 + 3 * n_t + 2, {"crown_z": zc, "tier": [zc - 2 * Ht, zc - Ht]}


# ----------------------------------------------------------------------------------------------- Eagle Belfry
def hood(S, key, o, u, n, x0, x1, z, rise, t=0.35, d=0.4):
    """Broken scroll pediment: a segmental band over an opening, ending in scroll curls."""
    inner, outer = segment(x0, x1, z, rise, 10), segment(x0, x1, z, rise, 10, grow=t)
    wband(S, key, o, u, n, inner, outer, 0.0, d)
    for x in (x0, x1):
        wring(S, key, o, u, n, x, z - 0.15, 0.1, 0.42, d, k=12)
        wdisc(S, key, o, u, n, x, z - 0.15, 0.12, d - 0.05, k=8)


def corner_columns(S, sc, h, z0, z1, rc, rings=()):
    """Green marble columns, one each side of every corner of the square of half side h, with peach Corinthian
    capitals; at the given heights a capital and a base ring (the column runs on through every tier)."""
    C, A = sc["column"], sc["accent"]
    for o, u, n, L in faces(h):
        for x in (rc + 0.25, L - rc - 0.25):
            p = wp(o, u, n, x, rc + 0.05, 0.0)
            column(S, C, A, A, (p[0], p[1]), rc, z0, z1, k=12, cap_h=1.0, flare=1.7)
            for zr in rings:
                capital(S, A, (p[0], p[1]), rc, zr + 0.1, 0.9, 1.6, k=12)
                cylinder(S, A, (p[0], p[1]), 1.25 * rc, 1.25 * rc, zr + 0.1, zr + 0.4, k=12, cap=False)


def eagle_belfry(S, sc, b, z, h, cols=True):
    """Belfry stage of the Eagle Belfry on the square of half side b: ivory walls, paired marble columns with peach
    Corinthian capitals at the corners, cream pilasters, a tall arched opening framed in green and cream with a bronze
    bell and a round green balcony, two oculi under small hoods, a broad scroll pediment, a layered peach and cream
    entablature; returns its top."""
    T, A, C = sc["trim"], sc["accent"], sc["column"]
    box_walls(S, sc["wall"], -b, b, -b, b, z, z + h)
    for o, u, n, L in faces(b):
        if cols:
            rc = 0.022 * L + 0.2
            for x in (rc + 0.2, 3.3 * rc + 0.3, L - rc - 0.2, L - 3.3 * rc - 0.3):
                p = wp(o, u, n, x, rc + 0.05, 0.0)
                column(S, C, A, A, (p[0], p[1]), rc, z, z + h - 1.0, k=10, cap_h=1.0, flare=1.75)
        aw = 0.25 * L
        a0, a1 = L / 2 - aw / 2, L / 2 + aw / 2
        zb = z + 0.2 * h
        zs = z + h - 3.4 - aw / 2
        for x in (a0 - 0.95, a1 + 0.95):
            wbox(S, T, o, u, n, x - 0.3, x + 0.3, z + 0.3, zs + 0.3, 0.0, 0.18)
            capital(S, A, wp(o, u, n, x, 0.0, 0.0)[:2], 0.25, zs + 0.6, 0.3, 1.4, k=4)
        warch(S, "void", o, u, n, a0, a1, zb, zs, 0.04, k=10)
        warch_frame(S, C, o, u, n, a0, a1, zb, zs, 0.4, 0.0, 0.18, k=10, keystone=False)
        warch_frame(S, T, o, u, n, a0 - 0.4, a1 + 0.4, zb, zs, 0.25, 0.0, 0.12, k=10, keystone=False)
        wbox(S, A, o, u, n, L / 2 - 0.35, L / 2 + 0.35, zs + aw / 2 + 0.1, zs + aw / 2 + 1.0, 0.0, 0.35)
        cb = wp(o, u, n, L / 2, 0.4, 0.0)
        rb = min(0.36 * aw, 1.9)
        bell(S, sc["metal"], (cb[0], cb[1]), zs + 0.1, rb)
        beam(S, "wood:madera_oscura", wp(o, u, n, a0 + 0.1, 0.4, zs + 0.25), wp(o, u, n, a1 - 0.1, 0.4, zs + 0.25), 0.25, 0.3)
        round_balcony(S, sc, o, u, n, L / 2, zb, 0.5 * aw + 0.35)
        wbox(S, A, o, u, n, a0 - 0.6, a1 + 0.6, zb - 0.6, zb - 0.25, 0.0, 0.3)
        for x in (0.27 * L, 0.73 * L):
            zo = z + 0.55 * h
            wdisc(S, "void", o, u, n, x, zo, 0.045 * L, 0.04, k=12)
            wring(S, A, o, u, n, x, zo, 0.045 * L, 0.045 * L + 0.3, 0.08, k=12)
            hood(S, A, o, u, n, x - 0.045 * L - 0.5, x + 0.045 * L + 0.5, zo + 0.045 * L + 0.55, 0.5, t=0.25, d=0.3)
        hood(S, A, o, u, n, a0 - 1.3, a1 + 1.3, zs + aw / 2 + 1.2, 1.3, t=0.45, d=0.5)
        hood(S, T, o, u, n, a0 - 0.8, a1 + 0.8, zs + aw / 2 + 0.75, 1.0, t=0.18, d=0.3)
        wbox(S, A, o, u, n, -0.15, L + 0.15, z + h - 0.95, z + h - 0.6, 0.0, 0.35)
        wbox(S, T, o, u, n, -0.25, L + 0.25, z + h - 0.6, z + h - 0.3, 0.0, 0.5)
        wbox(S, A, o, u, n, -0.4, L + 0.4, z + h - 0.3, z + h, 0.0, 0.65)
    return z + h


def ball_balustrade(S, sc, h, z, posts=3):
    """Green balustrade with ball-topped posts at the corners and between."""
    R, G = sc["rail"], sc["green"]
    for o, u, n, L in faces(h):
        wquad(S, R, o, u, n, 0.0, L, z, z + 1.1, -0.1, uv="panel")
        wbox(S, G, o, u, n, 0.0, L, z + 1.1, z + 1.25, -0.25, 0.05)
        for j in range(posts):
            p = wp(o, u, n, L * j / posts, -0.1, 0.0)
            box_walls(S, G, p[0] - 0.25, p[0] + 0.25, p[1] - 0.25, p[1] + 0.25, z, z + 1.4)
            ball(S, G, (p[0], p[1]), z + 1.4, 0.3)


def stepped_dome(S, sc, r, z, steps=5):
    """Stepped dome on a rounded profile: tiers of a peach band and a sloping peach skirt, a cap with slit windows;
    returns the top."""
    A, D, T = sc["accent"], sc["dome"], sc["trim"]
    ph = np.linspace(0.0, 1.2, steps + 1)
    rs, zs = r * np.cos(ph), z + 0.95 * r * np.sin(ph)
    for i in range(steps):
        zb = zs[i] + 0.06 * r
        cylinder(S, A, (0, 0), rs[i] + 0.06, rs[i] + 0.06, zs[i], zb, k=16)
        cylinder(S, D, (0, 0), rs[i], 1.05 * rs[i + 1], zb, max(zs[i + 1], zb + 0.3), k=16, cap=True)
    z = zs[-1]
    rs = np.r_[rs, rs[-1]]
    cylinder(S, D, (0, 0), rs[-1], rs[-1], z, z + 0.22 * r, k=8)
    for a in np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 8:
        u_, n_ = np.array([-np.sin(a), np.cos(a)]), np.array([np.cos(a), np.sin(a)])
        wquad(S, "void", rs[-1] * np.cos(np.pi / 8) * n_ - 0.25 * u_, u_, n_, 0.0, 0.5, z + 0.05 * r, z + 0.16 * r, 0.02)
    z += 0.22 * r
    cylinder(S, T, (0, 0), rs[-1] + 0.15, rs[-1] + 0.15, z, z + 0.3, k=8, cap=True)
    return z + 0.3


def eagle_thin(S, sc, w, H, rng):
    T, A, Wl, B = sc["trim"], sc["accent"], sc["wall"], sc["base"]
    hw, P = w / 2, 0.62 * w
    z = podium(S, sc, P, GF + FL, big=True, roundels=False)
    q = 0.58 * w
    box_walls(S, B, -q, q, -q, q, z, z + FL)
    for o, u, n, L in faces(q):
        for j in range(6):
            win(S, sc, o, u, n, L * (0.14 + 0.144 * j), z + 0.9, 1.0, 1.8, sc["frame"])
    corner_columns(S, sc, P, Z_B, z - 0.55, 0.5)
    corner_columns(S, sc, q, z, z + FL, 0.45)
    z = mould(S, [(A, q + 0.2, 0.3), (T, q + 0.4, 0.4)], z + FL)
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.3), sy * (q - 0.3), z, 2.2)
    # shaft: a window grid of 1 | 4 | 1 between marble strips, paired marble columns at the corners
    crown = 2.75 * w + 9.0
    n = max(10, round((H - z - crown) / FL))
    zs1 = z + n * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z, zs1 + 1.2)
    for o, u, n_, L in faces(hw):
        for x in (0.2 * L, 0.8 * L):
            wbox(S, sc["column"], o, u, n_, x - 0.3, x + 0.3, z, zs1 + 1.2, 0.0, 0.15)
        for f in range(n):
            zf = z + f * FL
            for xc in (0.125 * L, 0.29 * L, 0.43 * L, 0.57 * L, 0.71 * L, 0.875 * L):
                win(S, sc, o, u, n_, xc, zf + 0.85, min(1.1, 0.07 * L), 2.0, sc["frame"])
                wquad(S, A, o, u, n_, xc - 0.45, xc + 0.45, zf + 0.12, zf + 0.42, 0.04)
    corner_columns(S, sc, hw, z, zs1 + 1.2, 0.45)
    # crown: peach brackets and cornice, two belfries, a green balustrade, the stepped dome, the eagle
    z = zs1 + 1.2
    corbels(S, A, hw, z, 0.74 * w - hw)
    z = mould(S, [(T, 0.7 * w, 0.3), (A, 0.74 * w, 0.35), (T, 0.77 * w, 0.4)], z)
    b0 = 0.66 * w
    box_walls(S, Wl, -b0, b0, -b0, b0, z, z + 2.6)
    for o, u, n_, L in faces(b0):
        for j in range(7):
            win(S, sc, o, u, n_, L * (0.14 + 0.12 * j), z + 0.6, 0.9, 1.3, sc["frame"])
    z = mould(S, [(A, b0 + 0.15, 0.3), (T, b0 + 0.3, 0.3)], z + 2.6)
    z = eagle_belfry(S, sc, 0.68 * w, z, 1.0 * w)
    z = mould(S, [(A, 0.7 * w, 0.35), (T, 0.73 * w, 0.4)], z)
    for sx, sy in CORNERS:
        volute(S, A, (sx * 0.71 * w, sy * 0.71 * w, z), (sx * 0.58 * w, sy * 0.58 * w, z + 0.25 * w), w=0.5)
    z = eagle_belfry(S, sc, 0.56 * w, z, 0.8 * w)
    z = mould(S, [(A, 0.58 * w, 0.35), (T, 0.62 * w, 0.4)], z)
    ball_balustrade(S, sc, 0.6 * w, z)
    top = stepped_dome(S, sc, 0.43 * w, z)
    eagle(S, sc["metal"], (0.0, 0.0), top, 0.45 * w)
    return top + 0.45 * w * 0.9, n + 5, {"crown_z": zs1}


def eagle_tier(S, sc, hw, z, rng):
    """One tier of the cohesive Eagle Belfry on every face: a layered entablature, two storeys of a 3-window grid
    between cream pilasters, an arched window framed in green marble with a round balcony, two oculi, a broad peach
    scroll pediment over a cream one; returns the tier top."""
    T, A, C = sc["trim"], sc["accent"], sc["column"]
    L_ = 2 * hw
    aw = 0.24 * L_
    zA = z + 0.75 + 2 * FL + 0.35
    zs = zA + 2.4
    top = zs + aw / 2 + 0.6 + 1.5 + 0.45 + 0.35
    for o, u, n, L in faces(hw):
        wbox(S, A, o, u, n, -0.15, L + 0.15, z, z + 0.3, 0.0, 0.45)
        wbox(S, T, o, u, n, -0.1, L + 0.1, z + 0.3, z + 0.55, 0.0, 0.3)
        wbox(S, A, o, u, n, 0.0, L, z + 0.55, z + 0.75, 0.0, 0.18)
        for x in (0.25 * L, 0.42 * L, 0.58 * L, 0.75 * L):
            wbox(S, T, o, u, n, x - 0.2, x + 0.2, z + 0.75, z + 0.75 + 2 * FL, 0.0, 0.14)
        for f in range(2):
            for xc in (0.335 * L, 0.5 * L, 0.665 * L):
                zf = z + 0.75 + f * FL
                win(S, {**sc, "glass": sc["lit"]}, o, u, n, xc, zf + 0.7, min(1.7, 0.1 * L), 2.3, sc["frame"])
                wquad(S, A, o, u, n, xc - 0.5, xc + 0.5, zf + 0.06, zf + 0.3, 0.04)
        a0, a1 = L / 2 - aw / 2, L / 2 + aw / 2
        warch(S, sc["lit"], o, u, n, a0, a1, zA, zs, 0.05, k=10)
        wbox(S, sc["frame"], o, u, n, L / 2 - 0.06, L / 2 + 0.06, zA, zs + aw / 2, 0.05, 0.11)
        wbox(S, sc["frame"], o, u, n, a0, a1, zs - 0.05, zs + 0.05, 0.05, 0.11)
        warch_frame(S, C, o, u, n, a0, a1, zA, zs, 0.4, 0.0, 0.18, k=10, keystone=False)
        warch_frame(S, T, o, u, n, a0 - 0.4, a1 + 0.4, zA, zs, 0.22, 0.0, 0.12, k=10, keystone=False)
        wbox(S, A, o, u, n, L / 2 - 0.3, L / 2 + 0.3, zs + aw / 2 + 0.15, zs + aw / 2 + 0.9, 0.0, 0.35)
        round_balcony(S, sc, o, u, n, L / 2, zA, 0.5 * aw + 0.3)
        for x in (0.25 * L, 0.75 * L):
            wdisc(S, sc["glass"], o, u, n, x, zs + 0.3, 0.42, 0.05, k=12)
            wring(S, A, o, u, n, x, zs + 0.3, 0.42, 0.7, 0.08, k=12)
        hood(S, A, o, u, n, 0.14 * L, 0.86 * L, zs + aw / 2 + 0.6 + 0.35, 1.5, t=0.5, d=0.5)
        hood(S, T, o, u, n, 0.22 * L, 0.78 * L, zs + aw / 2 + 0.55, 1.1, t=0.2, d=0.3)
    return top


def eagle_cohesive(S, sc, w, H, rng):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.3, GF + 0.5 * FL, big=True, roundels=False)
    z0 = z
    top_h = 1.85 * w + 6.0
    probe = eagle_tier(Soup(), sc, hw, 0.0, rng)
    n_t = max(2, round((H - z - top_h) / probe))
    rings = []
    for i in range(n_t):
        z = eagle_tier(S, sc, hw, z, rng)
        rings.append(z)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    zc = z
    z = eagle_belfry(S, sc, hw, z, 0.85 * w, cols=False)
    corner_columns(S, sc, hw, z0, z - 0.6, 0.5, rings)
    z = mould(S, [(A, hw + 0.4, 0.35), (T, hw + 0.7, 0.45)], z)
    ball_balustrade(S, sc, hw + 0.55, z)
    top = stepped_dome(S, sc, 0.44 * w, z)
    eagle(S, sc["metal"], (0.0, 0.0), top, 0.5 * w)
    return top + 0.5 * w * 0.9, 1 + 3 * n_t + 2, {"crown_z": zc, "tier": [zc - 2 * probe, zc - probe]}


# ----------------------------------------------------------------------------------------------- Golden Clock Tower
def golden_belfry(S, sc, b, z, h):
    """Belfry of three arches a face (the centre one taller) with gilt bells over a balustrade, cream piers and
    columns, warm orange interior; returns its top."""
    T, A = sc["trim"], sc["accent"]
    box_walls(S, T, -b, b, -b, b, z, z + h)
    for o, u, n, L in faces(b):
        piers(S, T, o, u, n, L, z, z + h, 0.1, 0.3)
        for a0, a1, f in ((0.13 * L, 0.33 * L, 0.86), (0.37 * L, 0.63 * L, 0.95), (0.67 * L, 0.87 * L, 0.86)):
            zb = z + 1.2
            zs = z + f * h - 0.5 - (a1 - a0) / 2
            warch(S, A, o, u, n, a0, a1, zb, zs, 0.04, k=10)
            warch_frame(S, T, o, u, n, a0, a1, zb, zs, 0.25, 0.0, 0.16, k=10)
            cb = wp(o, u, n, (a0 + a1) / 2, 0.35, 0.0)
            bell(S, sc["metal"], (cb[0], cb[1]), zs + 0.3, min(0.3 * (a1 - a0), 1.6))
            bay_balcony(S, sc, o, u, n, a0 - 0.1, a1 + 0.1, zb, D=0.45)
        for x in (0.35 * L, 0.65 * L):
            p = wp(o, u, n, x, 0.3, 0.0)
            column(S, T, T, T, (p[0], p[1]), 0.25, z + 0.9, z + h - 0.4, k=8)
    return z + h


def golden_dome(S, sc, rd, z, rise, clocks, rng):
    """Orange ribbed dome: cream ring, relief dormers (the front one with a clock when ``clocks``), teal ovals, the
    lantern with a gilt cupola and finial; returns the top."""
    T = sc["trim"]
    cylinder(S, T, (0, 0), rd + 0.4, rd + 0.4, z, z + 0.6, k=24, cap=True)
    zd = z + 0.6
    top = dome(S, sc["dome"], (0, 0), rd, zd, k=24, m=10, rise=rise, ribs_key=T, ribs=12, rib_w=0.18 + 0.01 * rd)
    wd = 0.5 * rd
    for i, (o, u, n, L) in enumerate(faces(rd)):
        o2 = np.asarray(n) * (rd + 0.2) - np.asarray(u) * (wd / 2)
        if clocks and i == 0:
            wc = 0.92 * rd
            o3 = np.asarray(n) * (rd + 0.3) - np.asarray(u) * (wc / 2)
            wbox(S, T, o3, u, n, 0.0, wc, zd, zd + 0.92 * rd, -rd * 0.9, 0.0)
            clock(S, o3, u, n, wc / 2, zd + 0.48 * rd, 0.36 * rd, 0.08, T)
            for sgn in (-1, 1):
                wring(S, T, o3, u, n, wc / 2 + sgn * 0.5 * rd, zd + 0.14 * rd, 0.05 * rd, 0.13 * rd, 0.06, k=12)
            inner = pointed(0.0, wc, zd + 0.92 * rd, f=0.62, k=6, zk=0.45)
            outer = pointed(0.0, wc, zd + 0.92 * rd, f=0.62, k=6, zk=0.45, grow=0.3)
            o2 = o3
            wpoly(S, sc["dome"], o2, u, n, inner, 0.0)
            wpoly(S, sc["orn"], o2, u, n, inner, 0.04, uv01=True)
            wband(S, T, o2, u, n, inner, outer, -rd * 0.9, 0.15)
        else:
            dormer(S, sc, o2, u, n, wd, zd, 0.25 * rd, rd * 0.9, body=sc["dome"], gable=0.9, glass=False)
            oval(S, sc["teal"], T, o2, u, n, wd / 2, zd + 0.14 * rd, 0.07 * rd, 0.1 * rd, 0.05, k=14)
        th = np.arctan2(n[1], n[0])
        for dt in (-0.42, 0.42):
            dome_oval(S, sc["teal"], T, rd, rise, zd, th + dt, 0.42, 0.07 * rd + 0.1, 0.11 * rd + 0.1)
    rl = 0.3 * rd
    return lantern(S, sc, rl, top - 0.3, 1.25 * rl, "gold")


def golden_thin(S, sc, w, H, rng):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw, P = w / 2, 0.66 * w
    z = podium(S, sc, P, GF + FL)
    q = 0.58 * w
    terrace(S, shapely.box(-P - 0.3, -P - 0.3, P + 0.3, P + 0.3), shapely.box(-q, -q, q, q), z + 0.02)
    balustrade(S, faces(P + 0.3), z)
    for sx, sy in CORNERS:
        cypress(S, sx * (P - 0.6), sy * (P - 0.6), z, 3.2)
    z = pedestal(S, sc, q, z)
    terrace(S, shapely.box(-q - 0.3, -q - 0.3, q + 0.3, q + 0.3), shapely.box(-hw, -hw, hw, hw), z + 0.02)
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.45), sy * (q - 0.45), z, 2.4)
    # shaft: cream corner piers and pilasters, three bays of stacked windows in teal frames
    crown = 2.95 * w + 10.0
    n = max(8, round((H - z - crown) / FL))
    zs1 = z + n * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z, zs1 + 1.6)
    for o, u, n_, L in faces(hw):
        piers(S, T, o, u, n_, L, z, zs1 + 1.6, 0.08, 0.25)
        for xa, xb in ((0.03 * L, 0.05 * L), (0.95 * L, 0.97 * L)):
            wquad(S, Wl, o, u, n_, xa, xb, z + 0.4, zs1 + 1.2, 0.26)
        for x in (0.36 * L, 0.64 * L):
            wbox(S, T, o, u, n_, x - 0.025 * L, x + 0.025 * L, z, zs1 + 1.6, 0.0, 0.22)
        for xc, bw in ((0.2175 * L, 0.235 * L), (0.5 * L, 0.23 * L), (0.7825 * L, 0.235 * L)):
            gw = 0.55 * bw
            wquad(S, sc["lit"], o, u, n_, xc - gw / 2, xc + gw / 2, z + 0.5, zs1 + 0.6, 0.05)
            for sgn in (-1, 1):
                wbox(S, T, o, u, n_, xc + sgn * gw / 2 - (0.25 if sgn < 0 else 0.0), xc + sgn * gw / 2 + (0.25 if sgn > 0 else 0.0),
                     z + 0.3, zs1 + 0.8, 0.0, 0.14)
            wbox(S, sc["frame"], o, u, n_, xc - 0.06, xc + 0.06, z + 0.5, zs1 + 0.6, 0.05, 0.1)
            for f in range(n + 1):
                zf = z + 0.5 + f * FL
                wbox(S, sc["frame"], o, u, n_, xc - gw / 2, xc + gw / 2, zf - 0.12, zf + 0.12, 0.05, 0.11)
    # grand crown: bracketed cornice with a balustrade, cypresses and urns, the clock stage, the belfry, the dome
    z = zs1 + 1.6
    corbels(S, T, hw, z, 0.86 * w - hw)
    z = mould(S, [(T, 0.83 * w, 0.3), (T, 0.88 * w, 0.5)], z)
    balustrade(S, faces(0.88 * w - 0.05), z)
    for sx, sy in CORNERS:
        cypress(S, sx * 0.76 * w, sy * 0.76 * w, z, 3.0)
    k1, h1 = 0.66 * w, 0.62 * w
    box_walls(S, Wl, -k1, k1, -k1, k1, z, z + h1)
    for o, u, n_, L in faces(k1):
        piers(S, T, o, u, n_, L, z, z + h1, 0.17, 0.3)
        for xa, xb in ((0.04 * L, 0.13 * L), (0.87 * L, 0.96 * L)):
            wquad(S, Wl, o, u, n_, xa, xb, z + 0.8, z + 0.45 * h1, 0.31)
            wquad(S, Wl, o, u, n_, xa, xb, z + 0.55 * h1, z + h1 - 0.9, 0.31)
        wquad(S, A, o, u, n_, 0.17 * L, 0.83 * L, z + 0.3, z + h1 - 0.5, 0.02)
        r = 0.2 * L
        clock(S, o, u, n_, 0.5 * L, z + 0.52 * h1, r, 0.14, T)
        for xa, xb in ((0.19 * L, 0.32 * L), (0.68 * L, 0.81 * L)):
            wpoly(S, sc["orn"], o, u, n_, [(xa, z + 0.12 * h1), (xb, z + 0.12 * h1), (xb, z + 0.82 * h1), (xa, z + 0.82 * h1)],
                  0.06, uv01=True)
    z = mould(S, [(T, k1 + 0.2, 0.35), (T, k1 + 0.45, 0.4)], z + h1)
    for sx, sy in CORNERS:
        onion(S, sc["metal"], T, sx * (k1 + 0.1), sy * (k1 + 0.1), z, 0.55)
    k2 = 0.6 * w
    z = golden_belfry(S, sc, k2, z, 0.56 * w)
    z = mould(S, [(T, k2 + 0.2, 0.35), (T, k2 + 0.45, 0.4)], z)
    for sx, sy in CORNERS:
        onion(S, sc["metal"], T, sx * (k2 + 0.1), sy * (k2 + 0.1), z, 0.6)
    terrace(S, shapely.box(-k2 - 0.45, -k2 - 0.45, k2 + 0.45, k2 + 0.45), ngon((0, 0), 0.52 * w + 0.4, k=24), z + 0.02)
    return golden_dome(S, sc, 0.52 * w, z, 1.05, False, rng), n + 6, {"crown_z": zs1}


def golden_tier(S, sc, hw, z, rng):
    """One tier of the cohesive Golden Clock Tower on every face: three storeys of three bays (warm windows in teal
    frames, cream lattice balconies), cream pilasters, a shell hood with a relief tympanum, teal ovals on the side
    strips; returns the tier top."""
    T = sc["trim"]
    top = z
    for o, u, n, L in faces(hw):
        xa0, xa1 = 0.17 * L, 0.83 * L
        bay = (xa1 - xa0) / 3
        wbox(S, T, o, u, n, -0.1, L + 0.1, z, z + 0.3, 0.0, 0.3)
        for j in range(3):
            xc = xa0 + (j + 0.5) * bay
            gw = 0.62 * bay
            for f in range(3):
                zf = z + 0.3 + f * FL
                wquad(S, T, o, u, n, xc - gw / 2 - 0.2, xc + gw / 2 + 0.2, zf + 0.2, zf + 3.1, 0.03)
                wquad(S, sc["lit"], o, u, n, xc - gw / 2, xc + gw / 2, zf + 0.35, zf + 2.95, 0.06)
                wbox(S, sc["frame"], o, u, n, xc - 0.05, xc + 0.05, zf + 0.35, zf + 2.95, 0.06, 0.1)
                wbox(S, sc["frame"], o, u, n, xc - gw / 2, xc + gw / 2, zf + 2.1, zf + 2.18, 0.06, 0.1)
                bay_balcony(S, sc, o, u, n, xc - gw / 2 - 0.35, xc + gw / 2 + 0.35, zf + 0.3, D=0.4, h=0.9, rng=rng, plant=0.5)
        for j in range(4):
            x = xa0 + j * bay
            wbox(S, T, o, u, n, x - 0.18, x + 0.18, z + 0.3, z + 0.3 + 3 * FL, 0.0, 0.2)
        zh = z + 0.3 + 3 * FL + 0.25
        rise = 0.075 * L
        inner = segment(xa0, xa1, zh, rise, 12)
        outer = segment(xa0, xa1, zh, rise, 12, grow=0.35)
        wpoly(S, sc["orn"], o, u, n, inner, 0.04, uv01=True, rep=5)
        wband(S, T, o, u, n, inner, outer, 0.0, 0.3)
        for x in (xa0, xa1):
            wring(S, T, o, u, n, x, zh - 0.1, 0.1, 0.4, 0.3, k=12)
        rs = 0.09 * L
        zsh = zh + rise + 0.25
        half = [(L / 2 + rs * np.cos(a), zsh + rs * np.sin(a)) for a in np.linspace(0, np.pi, 13)]
        wpoly(S, sc["shell"], o, u, n, half, 0.32, uv01=True)
        for xc in (0.13 * L, 0.87 * L):
            oval(S, sc["teal"], T, o, u, n, xc, zh, 0.42, 0.65, 0.05)
            oval(S, sc["teal"], T, o, u, n, xc, z + 0.3 + 1.2 * FL, 0.32, 0.5, 0.05)
        top = max(top, zsh + rs + 0.35)
    for o, u, n, L in faces(hw):
        piers(S, T, o, u, n, L, z, top, 0.09, 0.25)
        for x in (0.045 * L, 0.955 * L):
            p = wp(o, u, n, x, 0.42, 0.0)
            cylinder(S, T, (p[0], p[1]), 0.26, 0.26, z, top, k=8)
        for xa, xb in ((-0.12, 0.09 * L + 0.12), (0.91 * L - 0.12, L + 0.12)):
            wbox(S, T, o, u, n, xa, xb, top - 0.45, top, 0.0, 0.55)
    return top


def golden_cohesive(S, sc, w, H, rng):
    T, Wl = sc["trim"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.3, GF + 0.5 * FL, roundels=False)
    z0 = z
    top_h = 2.25 * w + 8.0
    Ht = golden_tier(Soup(), sc, hw, 0.0, np.random.default_rng(0))
    n_t = max(2, round((H - z - top_h) / Ht))
    for i in range(n_t):
        z = golden_tier(S, sc, hw, z, rng)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    zc = z
    z = golden_belfry(S, sc, hw, z, 0.5 * w)
    z = mould(S, [(T, hw + 0.25, 0.35), (T, hw + 0.5, 0.4)], z)
    for sx, sy in CORNERS:
        onion(S, sc["metal"], T, sx * (hw + 0.05), sy * (hw + 0.05), z, 0.6)
    terrace(S, shapely.box(-hw - 0.5, -hw - 0.5, hw + 0.5, hw + 0.5), ngon((0, 0), 0.47 * w + 0.4, k=24), z + 0.02)
    return golden_dome(S, sc, 0.47 * w, z, 1.4, True, rng), 1 + 3 * n_t + 1, {"crown_z": zc, "tier": [zc - 2 * Ht, zc - Ht]}


BUILD = {("coral", "thin"): coral_thin, ("coral", "cohesive"): coral_cohesive, ("eagle", "thin"): eagle_thin,
         ("eagle", "cohesive"): eagle_cohesive, ("golden", "thin"): golden_thin, ("golden", "cohesive"): golden_cohesive}


def scheme(design: str, form: str = "thin", **roles) -> dict:
    sc = dict(SCHEMES[design])
    if (design, form) == ("golden", "cohesive"):
        sc["wall"] = "lm:oro_naranja"
    sc.update(roles)
    return sc


def landmark(design: str, form: str, center, axis, H: float, rng, W: float | None = None,
             colours: dict | None = None) -> tuple[Soup, dict]:
    """Landmark tower ``design`` ('coral' | 'eagle' | 'golden') in ``form`` ('thin' | 'cohesive') centred on
    ``center``, its front (local -Y) turned by the unit ``axis`` (local +X), about H m tall; W = shaft side (default
    from the reference proportions); colours: role -> material key overrides (see SCHEMES)."""
    W = float(W or WIDTH[(design, form)])
    sc = scheme(design, form, **(colours or {}))
    state = rng.bit_generator.state

    def build(h):
        rng.bit_generator.state = state
        S = Soup()
        r = BUILD[(design, form)](S, sc, W, h, rng)
        return S, max(float(V[:, 2].max()) for lst in S.parts.values() for V, _, _ in lst), r[1], r[2]

    tops = {h: build(h)[1] for h in float(H) + 4.5 * np.arange(-4, 5)}     # calibrate: storeys and tiers are discrete
    S, top, floors, marks = build(min(tops, key=lambda h: (abs(tops[h] - H), h)))
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    R = np.array([[ax[0], -ax[1]], [ax[1], ax[0]]])
    out = Soup()
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for key, lst in S.parts.items():
        for V, F, UV in lst:
            lo, hi = np.minimum(lo, V[:, :2].min(0)), np.maximum(hi, V[:, :2].max(0))
            V2 = V.copy()
            V2[:, :2] = V[:, :2] @ R.T + np.asarray(center, float)
            out.parts.setdefault(key, []).append((V2, F, UV))
    return out, {"design": design, "form": form, "height_m": float(top), "floors": int(floors), "shaft_m": W,
                 "half_extent_m": float(max(-lo.min(), hi.max())), "colours": sc, "triangles": out.triangles(), **marks}
