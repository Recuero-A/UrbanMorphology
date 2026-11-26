"""Relief kit of the Latin theme (DESIGN.md §5.3-5.4): real 3D mouldings and ornament for landmark towers and churches.

Everything has depth: profiles swept along paths (cornices, string courses, bases, architraves, archivolts, raking
cornices, dome ribs), 2D shapes extruded out of a wall (scrolls, cartouches, tracery, shells, notched bands), lathe
solids (columns, balusters, urns, bulbs, bells, finials), and elements composed of them (pilasters and columns with
bases and capitals, windows with architraves, sills and pediments, consoles, balustrades, pinnacles).

A profile is a list of (a, b) points running from the base surface out and back: a = distance out of the surface,
b = position along the profile's second axis (z for horizontal sweeps; across the path for sweeps in a wall), the
solid lying between the profile and the surface. Faces are oriented from the profile tangent, so every primitive
faces outward whatever the handedness of its frame. Sizes are metres.
"""

from __future__ import annotations

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import LineString, Point

from urbanmorph.city3d.format import polygons
from urbanmorph.city3d.latin.kit import Soup, cylinder, dome, wbox, wp
from urbanmorph.city3d.mesh import triangulate

ZV = np.array([0.0, 0.0, 1.0])

# unit profiles (a, b): scaled by (depth, height) when used
CORNICE = [(0, 0), (0.1, 0), (0.1, 0.07), (0.2, 0.1), (0.28, 0.2), (0.36, 0.23), (0.36, 0.3), (0.86, 0.33), (0.94, 0.36),
           (0.94, 0.56), (1.0, 0.6), (1.0, 0.66), (0.93, 0.72), (0.88, 0.82), (0.93, 0.92), (0.96, 0.95), (0.96, 1.0), (0, 1.0)]
STRING = [(0, 0), (0.45, 0), (0.45, 0.22), (1.0, 0.42), (1.0, 0.72), (0.55, 0.85), (0.55, 1.0), (0, 1.0)]
BASE = [(0, 0), (1.0, 0), (1.0, 0.3), (0.85, 0.36), (0.9, 0.48), (0.72, 0.58), (0.55, 0.64), (0.6, 0.78), (0.42, 0.86),
        (0.42, 1.0), (0, 1.0)]
CAPITAL = [(0, 0), (0.25, 0), (0.25, 0.12), (0.4, 0.2), (0.55, 0.45), (0.75, 0.62), (0.75, 0.72), (1.0, 0.76), (1.0, 1.0),
           (0, 1.0)]
ARCHITRAVE = [(0, 0), (0.55, 0), (0.62, 0.12), (0.62, 0.3), (0.8, 0.36), (0.8, 0.62), (1.0, 0.7), (1.0, 0.86), (0.7, 0.93),
              (0.7, 1.0), (0, 1.0)]
ARCH_S = [(0, 0), (0.7, 0), (0.8, 0.35), (1.0, 0.55), (1.0, 1.0), (0, 1.0)]          # light architrave (repeated windows)
SILL = [(0, 0), (0.55, 0), (0.55, 0.25), (1.0, 0.45), (1.0, 0.85), (0.9, 1.0), (0, 1.0)]
RAKE = [(0, 0), (0.3, 0), (0.3, 0.12), (0.75, 0.22), (1.0, 0.45), (1.0, 0.75), (0.85, 1.0), (0, 1.0)]
RIB = [(0, -0.5), (0.6, -0.5), (0.9, -0.35), (1.0, -0.12), (1.0, 0.12), (0.9, 0.35), (0.6, 0.5), (0, 0.5)]
RAIL = [(0, 0), (1.0, 0), (1.0, 0.45), (0.85, 0.6), (0.85, 0.8), (0.6, 1.0), (0, 1.0)]
BALUSTER = [(0.0, 0.0), (0.55, 0.0), (0.55, 0.08), (0.35, 0.14), (0.42, 0.2), (0.62, 0.36), (0.6, 0.48), (0.3, 0.62),
            (0.22, 0.72), (0.3, 0.8), (0.42, 0.86), (0.45, 0.92), (0.5, 0.92), (0.5, 1.0), (0.0, 1.0)]
URN = [(0.0, 0.0), (0.42, 0.0), (0.42, 0.08), (0.24, 0.16), (0.2, 0.24), (0.5, 0.36), (0.62, 0.5), (0.56, 0.66),
       (0.34, 0.76), (0.4, 0.8), (0.18, 0.86), (0.12, 0.92), (0.2, 0.96), (0.0, 1.0)]
BULB = [(0.62, 0.0), (0.7, 0.06), (0.92, 0.2), (1.0, 0.34), (0.94, 0.48), (0.72, 0.62), (0.42, 0.74), (0.24, 0.84),
        (0.16, 0.92), (0.0, 1.0)]


def scale(prof, da, db, a0=0.0, b0=0.0):
    return [(a0 + a * da, b0 + b * db) for a, b in prof]


def closed(prof):
    """A free-standing section (rails, plinths): the profile closed back to its first point."""
    return list(prof) + [prof[0]]


# ----------------------------------------------------------------------------------------------- cores
def add_oriented(S, key, T, N, UV=None):
    """Add triangles T (k, 3, 3), each turned to face its expected normal N (k, 3)."""
    T = np.asarray(T, float).reshape(-1, 3, 3)
    if not len(T):
        return
    g = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = np.einsum("ij,ij->i", g, np.broadcast_to(np.asarray(N, float), g.shape)) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    if UV is not None:
        UV = np.asarray(UV, float).reshape(-1, 3, 2).copy()
        UV[flip] = UV[flip][:, [0, 2, 1]]
    S.add(key, T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), UV)


def sweep_core(S, key, P, A, B, prof, closed=False, caps=False):
    """Profile (a, b) along the path P: vertex = P + A a + B b (A, B per path vertex, mitre-scaled by the caller)."""
    P, A, B = (np.asarray(x, float) for x in (P, A, B))
    prof = np.asarray(prof, float)
    N, M = len(P), len(prof)
    if N < 2 or M < 2:
        return
    V = P[:, None] + A[:, None] * prof[None, :, :1] + B[:, None] * prof[None, :, 1:]
    I = np.arange(N if closed else N - 1)
    I2 = (I + 1) % N
    J = np.arange(M - 1)
    t = np.diff(prof, axis=0)
    q = np.stack([V[I][:, J], V[I2][:, J], V[I2][:, J + 1], V[I][:, J + 1]], 2)          # (s, m, 4, 3)
    ne = t[None, :, 1:2] * ((A[I] + A[I2]) / 2)[:, None] - t[None, :, 0:1] * ((B[I] + B[I2]) / 2)[:, None]
    PP = np.r_[P, P[:1]] if closed else P
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PP, axis=0), axis=1))]
    pl = np.r_[0, np.cumsum(np.hypot(t[:, 0], t[:, 1]))]
    u0, u1 = L[I][:, None] * np.ones(len(J)), L[I + 1][:, None] * np.ones(len(J))
    v0, v1 = np.ones(len(I))[:, None] * pl[J], np.ones(len(I))[:, None] * pl[J + 1]
    uv = np.stack([np.stack([u0, v0], -1), np.stack([u1, v0], -1), np.stack([u1, v1], -1), np.stack([u0, v1], -1)], 2)
    keep = (np.abs(t).sum(1) > 1e-9)[None, :] * np.ones((len(I), 1), bool)
    q, ne, uv = q[keep], ne[keep], uv[keep]
    T = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    add_oriented(S, key, T, np.concatenate([ne, ne]), np.concatenate([uv[:, [0, 1, 2]], uv[:, [0, 2, 3]]]))
    if caps and not closed:
        poly = shapely.Polygon(prof).buffer(0)
        for p in polygons(poly):
            try:
                xy, tris = triangulate(shapely.orient_polygons(p))
            except ValueError:
                continue
            for i, s in ((0, -1.0), (N - 1, 1.0)):
                tang = P[min(i + 1, N - 1)] - P[max(i - 1, 0)]
                Vc = P[i] + A[i] * xy[:, :1] + B[i] * xy[:, 1:]
                add_oriented(S, key, Vc[tris], s * tang)


def mitre2(path, closed):
    """Unit right-hand normals of a 2D polyline, mitred at its vertices (offset by d keeps distance d from each edge)."""
    p = np.asarray(path, float)
    e = (np.roll(p, -1, 0) - p) if closed else np.diff(p, axis=0)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    r = np.c_[e[:, 1], -e[:, 0]]
    if closed:
        a, b = np.roll(r, 1, 0), r
    else:
        a, b = np.r_[r[:1], r], np.r_[r, r[-1:]]
    m = a + b
    m /= np.maximum(1.0 + np.einsum("ij,ij->i", a, b), 0.25)[:, None]
    return m


def sweep(S, key, path, z, prof, closed=False, caps=False):
    """Horizontal sweep: a plan path (x, y) at height z, profile (out to the right of the path, dz); a closed CCW
    path runs a moulding round a building (cornices, string courses, plinths)."""
    p = np.asarray(path, float)
    m = mitre2(p, closed)
    P = np.c_[p, np.full(len(p), z)]
    sweep_core(S, key, P, np.c_[m, np.zeros(len(p))], np.tile(ZV, (len(p), 1)), prof, closed, caps)


def rect(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ring(S, key, h, z, prof, x0=None):
    """Moulding round the square of half side h (or the rectangle x0 = (x0, x1, y0, y1))."""
    sweep(S, key, rect(*(x0 or (-h, h, -h, h))), z, prof, closed=True)


def wsweep(S, key, o, u, n, path, prof, closed=False, caps=False):
    """Sweep in a wall: path (x, z) in the wall plane, profile (out of the wall, offset to the right of the path);
    a counter-clockwise path round an opening runs its architrave, an arch outline its archivolt."""
    p = np.asarray(path, float)
    m = mitre2(p, closed)
    u3 = np.array([u[0], u[1], 0.0])
    n3 = np.array([n[0], n[1], 0.0])
    P = np.array([wp(o, u, n, x, 0.0, z) for x, z in p])
    B = m[:, :1] * u3 + m[:, 1:] * ZV
    sweep_core(S, key, P, np.tile(n3, (len(p), 1)), B, prof, closed, caps)


def sweep3d(S, key, P, Nrm, prof, closed=False):
    """Sweep along a 3D path with given outward normals (dome ribs): profile (out along the normal, across)."""
    P, Nrm = np.asarray(P, float), np.asarray(Nrm, float)
    T = np.gradient(P, axis=0)
    B = np.cross(T, Nrm)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    sweep_core(S, key, P, Nrm, B, prof, closed)


def pextrude(S, key, origin, ax, ay, az, geom, z0, z1, back=False):
    """Extrude a 2D shape (coordinates in the plane spanned by ax, ay) from z0 to z1 along az: front cap at z1
    facing +az, side walls, optional back cap."""
    O, ax, ay, az = (np.asarray(v, float) for v in (origin, ax, ay, az))
    for p in polygons(geom):
        p = shapely.orient_polygons(p.simplify(0.004))
        if p.is_empty or p.area < 1e-5 or p.geom_type != "Polygon":
            continue
        try:
            xy, tris = triangulate(p)
        except ValueError:
            p = p.buffer(0.001).buffer(-0.001)
            if p.geom_type != "Polygon":
                continue
            try:
                xy, tris = triangulate(shapely.orient_polygons(p))
            except ValueError:
                continue
        F3 = O + ax * xy[:, :1] + ay * xy[:, 1:]
        add_oriented(S, key, (F3 + az * z1)[tris], az)
        if back:
            add_oriented(S, key, (F3 + az * z0)[tris], -az)
        T, Nn = [], []
        for r in [p.exterior, *p.interiors]:
            c = np.asarray(r.coords)
            for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
                a, b = O + ax * x0 + ay * y0, O + ax * x1 + ay * y1
                q = [a + az * z0, b + az * z0, b + az * z1, a + az * z1]
                nn = ax * (y1 - y0) - ay * (x1 - x0)
                T += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
                Nn += [nn, nn]
        add_oriented(S, key, T, Nn)


def wextrude(S, key, o, u, n, geom, d0, d1):
    """Extrude a 2D shape drawn in wall coordinates (x along the wall, z up) out of the wall, from d0 to d1."""
    pextrude(S, key, (o[0], o[1], 0.0), (u[0], u[1], 0.0), ZV, (n[0], n[1], 0.0), geom, d0, d1)


def lathe(S, key, c, prof, z0=0.0, k=10):
    """Solid of revolution: prof = [(radius, height)] bottom to top."""
    for (ra, za), (rb, zb) in zip(prof[:-1], prof[1:]):
        if abs(zb - za) > 1e-6 or abs(rb - ra) > 1e-6:
            if zb >= za:
                cylinder(S, key, c, max(ra, 1e-3), max(rb, 1e-3), z0 + za, z0 + zb, k=k)
            else:
                cylinder(S, key, c, max(rb, 1e-3), max(ra, 1e-3), z0 + zb, z0 + za, k=k)
    r_top, z_top = prof[-1]
    if r_top > 1e-3:
        cylinder(S, key, c, r_top, r_top, z0 + z_top, z0 + z_top, k=k, cap=True)


# ----------------------------------------------------------------------------------------------- 2D ornament shapes
def stroke(pts, w):
    return LineString(pts).buffer(w / 2, quad_segs=3)


def spiral_pts(cx, cz, r0, r1, a0, turns, k=40, sgn=1):
    t = np.linspace(0, turns * 2 * np.pi, k)
    r = r0 + (r1 - r0) * t / max(t[-1], 1e-9)
    return np.c_[cx + r * np.cos(a0 + sgn * t), cz + r * np.sin(a0 + sgn * t)]


def scroll(cx, cz, r, w, sgn=1, a0=-np.pi / 2, turns=1.15, tail=None):
    """C-scroll: a spiral of outer radius r curling inward (sgn = turning sense), with an optional tail to (x, z)."""
    pts = spiral_pts(cx, cz, r, 0.25 * r, a0, turns, sgn=sgn)
    g = stroke(pts, w).union(Point(pts[-1]).buffer(0.9 * w, quad_segs=3))
    if tail is not None:
        g = g.union(stroke([pts[0], tail], w))
    return g


def leaf(x0, z0, x1, z1, wmax):
    """Pointed leaf from (x0, z0) to (x1, z1), widest a third of the way."""
    d = np.array([x1 - x0, z1 - z0], float)
    L = np.linalg.norm(d)
    d /= max(L, 1e-9)
    nrm = np.array([-d[1], d[0]])
    s = np.linspace(0, 1, 9)
    w = wmax * np.sin(np.pi * s ** 0.8) * (1 - 0.3 * s)
    left = [np.array([x0, z0]) + d * L * t + nrm * ww for t, ww in zip(s, w)]
    right = [np.array([x0, z0]) + d * L * t - nrm * ww for t, ww in zip(s, w)][::-1]
    return shapely.Polygon(left + right).buffer(0)


def lobe(x0, z0, x1, z1, r):
    """Rounded teardrop lobe from (x0, z0) to (x1, z1), widest (r) at 60 %."""
    pts = [((1 - t) * x0 + t * x1, (1 - t) * z0 + t * z1, rr) for t, rr in ((0.0, 0.25 * r), (0.6, r), (0.95, 0.45 * r))]
    return shapely.union_all([Point(x, z).buffer(rr, quad_segs=4) for x, z, rr in pts]).convex_hull


def palmette(xc, zb, h, w):
    """Acanthus palmette: seven rounded lobes fanning from a small arc, the middle one tallest, the outer ones curling
    down, on a bud."""
    g = Point(xc, zb + 0.06 * h).buffer(0.09 * w, quad_segs=4)
    for i, a in enumerate(np.radians([-75, -50, -25, 0, 25, 50, 75])):
        L = h * (0.95 - 0.14 * abs(i - 3))
        b = (xc + np.sin(a) * 0.06 * w, zb + 0.06 * h + np.cos(a) * 0.04 * h)
        tip = (b[0] + np.sin(a) * L * 0.62, b[1] + np.cos(a) * L)
        g = g.union(lobe(b[0], b[1], tip[0], tip[1], (0.085 if i != 3 else 0.11) * w))
        if abs(i - 3) == 3:
            sg = 1 if i > 3 else -1
            g = g.union(stroke(spiral_pts(tip[0] + sg * 0.02 * w, tip[1] - 0.06 * h, 0.06 * w, 0.015 * w, np.pi / 2, 0.9, 12, -sg),
                               0.035 * w))
    return g


def rosette(xc, zc, r, petals=8):
    g = Point(xc, zc).buffer(0.32 * r, quad_segs=3)
    for k in range(petals):
        a = 2 * np.pi * k / petals
        g = g.union(Point(xc + 0.62 * r * np.cos(a), zc + 0.62 * r * np.sin(a)).buffer(0.36 * r, quad_segs=3))
    return g


def bezier(p0, p1, p2, p3, k=16):
    t = np.linspace(0, 1, k)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def vine(pts, w, leaf_len, every=3, side0=1):
    """Stem along pts with leaves alternating on both sides."""
    g = stroke(pts, w)
    side = side0
    for i in range(2, len(pts) - 1, every):
        d = pts[i + 1] - pts[i - 1]
        d = d / max(np.linalg.norm(d), 1e-9)
        nrm = np.array([-d[1], d[0]]) * side
        tip = pts[i] + (0.75 * nrm + 0.65 * d) * leaf_len
        g = g.union(leaf(*pts[i], *tip, 0.28 * leaf_len))
        side = -side
    return g


def filigree(x0, x1, zb, zt, w=None):
    """Carved baroque lace for a panel [x0, x1] x [zb, zt]: a central palmette on a stem, mirrored vine scrolls with
    acanthus leaves curling into spirals that end in rosettes, lower S-scrolls, a base rosette; drawn at the panel's
    own proportions (one scale for both axes, never stretched)."""
    W, H = x1 - x0, zt - zb
    xc = (x0 + x1) / 2
    s = min(W / 2, H)
    w = w or 0.07 * s
    g = palmette(xc, zb + 0.38 * H, 0.58 * H, 0.5 * min(W, 1.3 * H))
    g = g.union(stroke([(xc, zb + 0.02 * H), (xc, zb + 0.45 * H)], 1.2 * w))
    g = g.union(rosette(xc, zb + 0.1 * H, 0.09 * s, 8))
    for sg in (-1, 1):
        ex, ez = xc + sg * min(0.4 * W, 0.95 * s), zb + 0.5 * H
        stem = bezier((xc + sg * 0.03 * W, zb + 0.18 * H), (xc + sg * 0.25 * W, zb + 0.05 * H), (ex, zb + 0.15 * H), (ex, ez), 14)
        spiral = spiral_pts(ex - sg * 0.13 * s, ez, 0.13 * s, 0.03 * s, 0.0 if sg > 0 else np.pi, 1.1, k=20, sgn=sg)
        g = g.union(vine(np.r_[stem, spiral[1:]], w, 0.16 * s, every=3, side0=sg))
        g = g.union(rosette(ex - sg * 0.13 * s, ez, 0.06 * s, 6))
        low = bezier((xc + sg * 0.1 * W, zb + 0.06 * H), (xc + sg * 0.3 * W, zb), (xc + sg * 0.45 * W, zb + 0.02 * H),
                     (xc + sg * min(0.47 * W, 1.05 * s), zb + 0.12 * H), 10)
        g = g.union(stroke(low, 0.8 * w)).union(Point(low[-1]).buffer(1.1 * w, quad_segs=3))
        up = bezier((xc + sg * 0.05 * W, zb + 0.55 * H), (xc + sg * 0.12 * W, zb + 0.72 * H), (xc + sg * 0.3 * W, zb + 0.75 * H),
                    (xc + sg * 0.28 * W, zb + 0.62 * H), 10)
        g = g.union(vine(up, 0.8 * w, 0.1 * s, every=3, side0=-sg))
        tx, tz = xc + sg * min(0.33 * W, 0.78 * s), zb + 0.82 * H              # upper corner: a C-scroll pair
        g = g.union(scroll(tx, tz, 0.09 * s, 0.75 * w, sgn=-sg, a0=np.pi / 2, turns=1.1))
        g = g.union(leaf(tx, tz + 0.06 * s, tx + sg * 0.14 * s, tz + 0.2 * s, 0.05 * s))
    return g.intersection(shapely.box(x0, zb, x1, zt))


def wrap_dome(S, src, rd, rise, z0, theta, lift=0.06):
    """Copy the parts of ``src`` (built in the local wall frame o = (0, 0), u = (1, 0), n = (0, -1): x along the
    dome, -y out of it, z up) onto the dome surface at azimuth theta: x becomes arc length round the dome at each
    height, -y the height above its surface (ornament lying on a dome, between its ribs)."""
    for key, lst in src.parts.items():
        for V, F, UV in lst:
            V = np.asarray(V, float)
            zz = np.clip((V[:, 2] - z0) / (rise * rd), 0.0, 0.995)
            Rz = rd * np.sqrt(1 - zz ** 2)
            a = theta + V[:, 0] / np.maximum(Rz, 0.5)
            rr = Rz + lift - V[:, 1]
            W = np.c_[rr * np.cos(a), rr * np.sin(a), V[:, 2]]
            S.parts.setdefault(key, []).append((W, F, UV))


# ----------------------------------------------------------------------------------------------- elements
def pilaster(S, key, o, u, n, x0, x1, z0, z1, d=0.35, cap_key=None, base_h=0.6, cap_h=0.7, base=True, cap=True):
    """Pilaster [x0, x1] standing d out of the wall, with a moulded base and capital wrapping its three faces."""
    hb = base_h if base else 0.0
    hc = cap_h if cap else 0.0
    wbox(S, key, o, u, n, x0, x1, z0 + hb, z1 - hc, 0.0, d)
    U = [wp(o, u, n, x0, 0.0, 0)[:2], wp(o, u, n, x0, d, 0)[:2], wp(o, u, n, x1, d, 0)[:2], wp(o, u, n, x1, 0.0, 0)[:2]]
    if base:
        sweep(S, key, U, z0, scale(BASE, 0.18, base_h))
    if cap:
        sweep(S, cap_key or key, U, z1 - cap_h, scale(CAPITAL, 0.22, cap_h))


def column(S, key, c, r, z0, z1, cap_key=None, base_key=None, order="corinthian", k=12):
    """Free or engaged column: square plinth, Attic base, shaft with entasis, Tuscan or Corinthian capital
    (bell with two rows of leaf tips, abacus, corner volutes)."""
    ck, bk = cap_key or key, base_key or key
    x, y = c
    pb = 1.35 * r
    box = lambda kk, s, za, zb: wbox(S, kk, (x - s, y - s), (1.0, 0.0), (0.0, -1.0), 0.0, 2 * s, za, zb, -2 * s, 0.0, back=True)
    box(bk, pb, z0, z0 + 0.35 * r)
    lathe(S, bk, c, [(1.3 * r, 0.0), (1.3 * r, 0.12 * r), (1.12 * r, 0.3 * r), (1.02 * r, 0.42 * r), (1.15 * r, 0.55 * r),
                     (1.0 * r, 0.7 * r)], z0 + 0.35 * r, k=k)
    zs, ch = z0 + 1.05 * r, (2.2 * r if order == "corinthian" else 1.0 * r)
    H = z1 - ch - zs
    lathe(S, key, c, [(r, 0.0), (0.99 * r, 0.33 * H), (0.93 * r, 0.75 * H), (0.88 * r, H)], zs, k=k)
    za = z1 - ch
    if order == "corinthian":
        lathe(S, ck, c, [(0.95 * r, 0.0), (1.02 * r, 0.12 * ch), (0.98 * r, 0.2 * ch), (1.12 * r, 0.45 * ch), (1.32 * r, 0.75 * ch),
                         (1.42 * r, 0.82 * ch)], za, k=k)
        for row, (zf, L, ww) in enumerate(((0.05, 0.42, 0.22), (0.3, 0.42, 0.2))):     # leaf tips curling out
            for j in range(8):
                a = 2 * np.pi * (j + 0.5 * row) / 8
                p0 = np.array([x + 0.98 * r * np.cos(a), y + 0.98 * r * np.sin(a), za + zf * ch])
                p1 = p0 + np.array([0.3 * r * np.cos(a), 0.3 * r * np.sin(a), L * ch])
                side = np.array([-np.sin(a), np.cos(a), 0.0]) * ww * r
                add_oriented(S, ck, [[p0 - side, p0 + side, p1]], [np.cos(a), np.sin(a), 0.0])
        box(ck, 1.5 * r, z1 - 0.18 * ch, z1)
        for a in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):        # corner volutes on the diagonals
            nv = np.array([np.cos(a), np.sin(a)])
            uv = np.array([-nv[1], nv[0]])
            o = np.array([x, y]) + nv * 1.45 * r
            wextrude(S, ck, o, uv, nv, scroll(0.0, z1 - 0.34 * ch, 0.22 * r, 0.08 * r, sgn=1, a0=np.pi / 2, turns=1.2),
                     -0.25 * r, 0.05 * r)
    else:
        lathe(S, ck, c, [(0.9 * r, 0.0), (1.05 * r, 0.1 * ch), (1.05 * r, 0.25 * ch), (0.95 * r, 0.32 * ch), (1.25 * r, 0.6 * ch)], za, k=k)
        box(ck, 1.3 * r, z1 - 0.38 * ch, z1)


def baluster_run(S, key, o, u, n, x0, x1, z, h=1.0, d=0.0, pitch=0.3, rail_key=None, posts=True, k=6):
    """Real balustrade along a wall frame line (d out of it): plinth, turned balusters, moulded handrail, end posts."""
    rk = rail_key or key
    L = x1 - x0
    if L < 0.3:
        return
    path = [wp(o, u, n, x0, d, 0)[:2], wp(o, u, n, x1, d, 0)[:2]]
    w = 0.32 * min(1.0, h)
    off = lambda pr: [(a - w / 2, b) for a, b in pr]
    sweep(S, rk, path, z, closed(scale([(0, 0), (1, 0), (1, 1), (0, 1)], w, 0.16 * h, a0=-w / 2)), caps=True)
    sweep(S, rk, path, z + 0.86 * h, closed(scale(RAIL, w * 1.05, 0.16 * h, a0=-w * 0.52)), caps=True)
    m = max(1, int(L / pitch))
    for i in range(m):
        xb = x0 + (i + 0.5) * L / m
        p = wp(o, u, n, xb, d, 0.0)
        lathe(S, key, (p[0], p[1]), scale(BALUSTER, 0.17 * min(1.0, h), 0.7 * h), z + 0.16 * h, k=k)
    if posts:
        for xb in (x0, x1):
            p = wp(o, u, n, xb, d, 0.0)
            s = 0.24 * min(1.0, h) + 0.04
            wbox(S, rk, (p[0] - s, p[1] - s), (1.0, 0.0), (0.0, -1.0), 0.0, 2 * s, z, z + 1.02 * h, -2 * s, 0.0, back=True)


def buttress(S, key, o, u, n, xc, w, z0, h, D):
    """Volute buttress ('ear'): an S-curve from the wall at z0 + h sweeping out and down to D at z0, ending in a
    scroll (stage corners of crowns, dormer sides)."""
    t = np.linspace(0, 1, 16)
    curve = [(D * (1 - np.cos(np.pi * s_) ) / 2 * 0.85, h * (1 - s_)) for s_ in t]
    shape = shapely.Polygon([(0.0, h)] + curve[1:] + [(0.85 * D, 0.0), (0.0, 0.0)]).buffer(0)
    shape = shape.union(scroll(0.86 * D, 0.22 * h, 0.2 * h, 0.1 * h, sgn=1, a0=np.pi, turns=1.0))
    pextrude(S, key, wp(o, u, n, xc, 0.0, z0), (n[0], n[1], 0.0), ZV, (u[0], u[1], 0.0), shape, -w / 2, w / 2, back=True)


def console(S, key, o, u, n, xc, w, z_top, h, D):
    """Scroll console (modillion / bracket) under a cornice: an S-curved profile, big scroll at the wall below."""
    t = np.linspace(0, 1, 14)
    top = [(D, 0.0)]
    curve = [(0.18 * D + 0.82 * D * (1 - s) ** 1.1, -h * (0.12 + 0.7 * s ** 1.3)) for s in t]
    shape = shapely.Polygon([(0.0, 0.0)] + top + curve + [(0.0, -h * 0.82)]).buffer(0)
    shape = shape.union(scroll(0.24 * D, -0.78 * h, 0.2 * h, 0.12 * h, sgn=-1, a0=0.0, turns=1.0))
    shape = shape.union(scroll(0.86 * D, -0.14 * h, 0.12 * h, 0.08 * h, sgn=1, a0=np.pi, turns=1.0))
    origin = wp(o, u, n, xc, 0.0, z_top)
    pextrude(S, key, origin, (n[0], n[1], 0.0), ZV, (u[0], u[1], 0.0), shape, -w / 2, w / 2, back=True)


def window(S, sc, o, u, n, x0, x1, z0, z1, frame=None, glass=None, arch=False, depth=0.3, w=0.3, sill=True,
           hood=None, mullion=None, keystone=False, prof=None):
    """Window with a moulded architrave (rectangular or round-headed), glass set back in it, a sill, an optional
    hood ('cornice' | 'triangle' | 'segment'), mullion and transom, keystone."""
    F, G = frame or sc["trim"], glass or sc["glass"]
    xc = (x0 + x1) / 2
    if arch:
        zs = z1 - (x1 - x0) / 2
        a = np.linspace(0, np.pi, 11)
        path = [(x0, z0), (x1, z0)] + [(xc + (x1 - x0) / 2 * np.cos(t), zs + (x1 - x0) / 2 * np.sin(t)) for t in a]
        path = [(x0, z0), (x1, z0), (x1, zs)] + path[3:]
        glass_panel(S, G, o, u, n, shapely.Polygon(path))
    else:
        path = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        wquad_glass(S, G, o, u, n, x0, x1, z0, z1)
    wsweep(S, F, o, u, n, path, scale(prof or ARCHITRAVE, depth, w), closed=True)
    if mullion:
        wbox(S, mullion, o, u, n, xc - 0.05, xc + 0.05, z0, z1 - (0 if not arch else 0.0), 0.0, 0.1)
        zt = z0 + 0.68 * (z1 - z0) if not arch else z1 - (x1 - x0) / 2
        wbox(S, mullion, o, u, n, x0, x1, zt - 0.05, zt + 0.05, 0.0, 0.1)
    if sill:
        sweep(S, F, [wp(o, u, n, x0 - w - 0.12, 0, 0)[:2], wp(o, u, n, x1 + w + 0.12, 0, 0)[:2]], z0 - w - 0.1,
              scale(SILL, depth + 0.14, 0.75 * w), caps=True)
    if keystone:
        wextrude(S, F, o, u, n, shapely.Polygon([(xc - 0.22, z1 - 0.05), (xc + 0.22, z1 - 0.05), (xc + 0.3, z1 + w + 0.12),
                                                 (xc - 0.3, z1 + w + 0.12)]), 0.0, depth + 0.08)
    if hood:
        pediment(S, F, o, u, n, x0 - w - 0.15, x1 + w + 0.15, z1 + w + 0.02, hood, depth + 0.1)


def wquad_glass(S, G, o, u, n, x0, x1, z0, z1):
    glass_panel(S, G, o, u, n, shapely.box(x0, z0, x1, z1))


LIT_CELLS = (0, 0, 1, 0, 2, 1, 0, 3, 2, 0, 1)            # interiors atlas: lit, lit with a plant, half-lit, dark


def glass_panel(S, G, o, u, n, poly, d=0.02):
    """Glass over an opening (shapely polygon in wall coordinates), UV 0-1 over its bounding box so a window image
    maps once; the interiors atlas (glass_lit) gets one of its four cells, chosen from the window's position."""
    for p in polygons(poly):
        try:
            xy, tris = triangulate(shapely.orient_polygons(p.simplify(0.003)))
        except ValueError:
            continue
        lo, hi = xy.min(0), xy.max(0)
        uv = (xy - lo) / np.maximum(hi - lo, 1e-6)
        if G == "glass_lit":
            k = LIT_CELLS[int(abs(hash((round(float(lo[0]), 1), round(float(lo[1]), 1), round(float(o[0]), 1),
                                        round(float(o[1]), 1))))) % len(LIT_CELLS)]
            uv[:, 0] = (k + 0.04 + 0.92 * uv[:, 0]) / 4
        V = np.array([wp(o, u, n, x, d, z) for x, z in xy])
        add_oriented(S, G, V[tris], np.array([n[0], n[1], 0.0]), uv[tris])


def pediment(S, key, o, u, n, x0, x1, z, kind="triangle", depth=0.45, rise=None, h=0.42, band=0.36):
    """Hood over an opening: 'cornice' (straight), 'triangle' or 'segment' pediment (raking cornice and a base
    cornice), 'broken' (segment open at the top, ending in scrolls)."""
    W = x1 - x0
    xc = (x0 + x1) / 2
    base = [wp(o, u, n, x0, 0, 0)[:2], wp(o, u, n, x1, 0, 0)[:2]]
    sweep(S, key, base, z, scale(STRING, depth, h), caps=True)
    if kind == "cornice":
        sweep(S, key, base, z + h, scale(CORNICE, depth * 0.9, 0.5 * h + 0.1), caps=True)
        return
    rise = rise or (0.28 * W if kind == "triangle" else 0.2 * W)
    zb = z + h
    if kind == "triangle":
        path = [(x0, zb), (xc, zb + rise), (x1, zb)]
    else:
        R = (W * W / 4 + rise * rise) / (2 * rise)
        zc = zb + rise - R
        t1 = np.arctan2(zb - zc, x1 - xc)
        tt = np.linspace(np.pi - t1, t1, 13)
        path = [(xc + R * np.cos(t), zc + R * np.sin(t)) for t in tt]
        if kind == "broken":
            gap = 0.16 * W
            left = [p for p in path if p[0] < xc - gap / 2]
            right = [p for p in path if p[0] > xc + gap / 2]
            for side, pts in ((-1, left), (1, right)):
                if len(pts) >= 2:
                    wsweep(S, key, o, u, n, pts, scale(RAKE, depth, band), caps=True)
                    ex, ez = pts[-1] if side < 0 else pts[0]
                    rs = max(0.28, 0.9 * band)
                    wextrude(S, key, o, u, n, scroll(ex, ez - 0.4 * rs, rs, 0.45 * rs, sgn=side, a0=np.pi / 2, turns=1.1), 0.0, depth)
                    sx, sz = pts[0] if side < 0 else pts[-1]                        # scroll at the foot too
                    wextrude(S, key, o, u, n, scroll(sx, sz - 0.3 * rs, 0.75 * rs, 0.35 * rs, sgn=-side, a0=-np.pi / 2, turns=1.0),
                             0.0, 0.8 * depth)
            return
    wsweep(S, key, o, u, n, path, scale(RAKE, depth, band), caps=True)
    return path


def oculus(S, key, glass, o, u, n, xc, zc, rx, rz, w=0.3, depth=0.3, lugs=True, k=24):
    """Oval or round window: glass, a moulded frame round it, four lugs (keystones) at the cardinal points."""
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    path = np.c_[xc + rx * np.cos(a), zc + rz * np.sin(a)]
    glass_panel(S, glass, o, u, n, shapely.Polygon(path))
    wsweep(S, key, o, u, n, path, scale(ARCHITRAVE, depth, w), closed=True)
    if lugs:
        for t in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            cx, cz = xc + (rx + w * 0.5) * np.cos(t), zc + (rz + w * 0.5) * np.sin(t)
            g = shapely.box(-0.16, -w * 0.75, 0.16, w * 0.75)
            g = shapely.affinity.rotate(g, np.degrees(t) - 90, origin=(0, 0))
            wextrude(S, key, o, u, n, shapely.affinity.translate(g, cx, cz), 0.0, depth + 0.06)


def clock(S, o, u, n, xc, zc, r, trim, d=0.15, k=32, face="clock"):
    """Clock: dial (image) set in a bold moulded bezel, a gilt inner ring."""
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = [wp(o, u, n, xc, d, zc)] + [wp(o, u, n, xc + r * np.cos(t), d, zc + r * np.sin(t)) for t in a]
    F = np.array([[0, 1 + i, 1 + (i + 1) % k] for i in range(k)])
    q = np.r_[[[0.5, 0.5]], np.c_[0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)]]
    n3 = np.array([n[0], n[1], 0.0])
    add_oriented(S, face, np.asarray(V)[F], n3, q[F])
    path = np.c_[xc + r * np.cos(a), zc + r * np.sin(a)]
    wsweep(S, "gold", o, u, n, path, [(d - 0.02, 0.0), (d + 0.1, 0.0), (d + 0.12, 0.06 * r), (d - 0.02, 0.08 * r)], closed=True)
    wsweep(S, trim, o, u, n, np.c_[xc + 1.08 * r * np.cos(a), zc + 1.08 * r * np.sin(a)],
           scale([(0, 0), (0.7, 0), (1.0, 0.25), (1.0, 0.55), (0.8, 0.75), (0.85, 1.0), (0, 1.0)], d + 0.35, 0.2 * r), closed=True)


def shell(S, key, o, u, n, xc, zb, r, depth=0.3, ribs=11):
    """Scallop shell: fluted ribs fanning up from a hinge, a scalloped rim, deeper in the middle."""
    for i in range(ribs):
        a0, a1 = np.pi * i / ribs + 0.02, np.pi * (i + 1) / ribs - 0.02
        am = (a0 + a1) / 2
        g = shapely.Polygon([(xc + 0.18 * r * np.cos(a0), zb + 0.18 * r * np.sin(a0)),
                             (xc + r * np.cos(a0), zb + r * np.sin(a0)),
                             (xc + 1.06 * r * np.cos(am), zb + 1.06 * r * np.sin(am)),
                             (xc + r * np.cos(a1), zb + r * np.sin(a1)),
                             (xc + 0.18 * r * np.cos(a1), zb + 0.18 * r * np.sin(a1))])
        wextrude(S, key, o, u, n, g, 0.0, depth * (0.6 + 0.4 * np.sin(am)))
    wextrude(S, key, o, u, n, Point(xc, zb).buffer(0.2 * r, quad_segs=4).intersection(shapely.box(xc - r, zb, xc + r, zb + r)),
             0.0, depth * 1.15)


def bell(S, key, c, z_top, r, yoke="wood:madera_oscura"):
    from urbanmorph.city3d.latin.church import BELL
    h = 1.1 * r
    lathe(S, key, c, [(rr * r, hh * h) for rr, hh in BELL], z_top - h, k=12)


def urn(S, key, c, z, h, k=10):
    lathe(S, key, c, scale(URN, 0.5 * h, h), z, k=k)


def ball(S, key, c, z, r, k=10):
    lathe(S, key, c, [(0.0, 0.0), (0.5 * r, 0.13 * r), (0.87 * r, 0.5 * r), (r, r), (0.87 * r, 1.5 * r), (0.5 * r, 1.87 * r),
                      (0.0, 2 * r)], z, k=k)


def obelisk(S, key, x, y, z, s, ball_key=None):
    """Pinnacle: moulded pedestal, tapering four-sided needle, ball."""
    c = (x, y)
    sq = lambda a, za, zb: wbox(S, key, (x - a, y - a), (1.0, 0.0), (0.0, -1.0), 0.0, 2 * a, za, zb, -2 * a, 0.0, back=True)
    sq(1.2 * s, z, z + 0.3 * s)
    sq(s, z + 0.3 * s, z + 1.5 * s)
    ring(S, key, 0, z + 1.5 * s, scale(STRING, 0.22 * s, 0.3 * s), (x - s, x + s, y - s, y + s))
    cylinder(S, key, c, 0.85 * s * np.sqrt(2), 0.12 * s, z + 1.8 * s, z + 6.0 * s, k=4)
    ball(S, ball_key or key, c, z + 5.8 * s, 0.35 * s, k=8)


def onion(S, key, ped, x, y, z, s):
    """Gilt finial: pedestal, bulb, neck, small bulb, spike."""
    sq = lambda a, za, zb, kk: wbox(S, kk, (x - a, y - a), (1.0, 0.0), (0.0, -1.0), 0.0, 2 * a, za, zb, -2 * a, 0.0, back=True)
    sq(0.6 * s, z, z + 0.25 * s, ped)
    sq(0.48 * s, z + 0.25 * s, z + 0.8 * s, ped)
    lathe(S, key, (x, y), [(0.3 * s, 0.0), (0.5 * s, 0.15 * s), (0.62 * s, 0.4 * s), (0.5 * s, 0.65 * s), (0.2 * s, 0.85 * s),
                           (0.1 * s, 0.95 * s), (0.18 * s, 1.05 * s), (0.24 * s, 1.15 * s), (0.1 * s, 1.3 * s), (0.04 * s, 1.4 * s),
                           (0.03 * s, 2.0 * s), (0.0, 2.1 * s)], z + 0.8 * s, k=10)


def bulb_stack(S, key, c, z, r, tiers=3, k=12):
    """Stacked bulbs (Bahia cupola and pinnacles): ogee bulbs shrinking upward with necks between; returns the top."""
    for i in range(tiers):
        rr = r * (1.0 - 0.28 * i)
        h = 1.15 * rr
        lathe(S, key, c, [(rr * a, h * b) for a, b in BULB], z, k=k)
        z += h
        lathe(S, key, c, [(0.18 * rr, 0.0), (0.3 * rr, 0.08 * rr), (0.3 * rr, 0.16 * rr), (0.2 * rr, 0.22 * rr)], z, k=k)
        z += 0.22 * rr
    return z


def ribbed_dome(S, key, rib_key, r, z0, rise, ribs=8, k=32, m=12, rib=(0.35, 0.3), phase=0.0):
    """Dome (quarter-ellipse profile about the origin) with raised moulded ribs (at phase + 2 pi i / ribs); returns
    the crown height."""
    top = dome(S, key, (0.0, 0.0), r, z0, k=k, m=m, rise=rise)
    phi = np.linspace(0, np.pi / 2 * 0.93, 14)
    for i in range(ribs):
        t = phase + 2 * np.pi * i / ribs
        P = np.c_[r * np.cos(phi) * np.cos(t), r * np.cos(phi) * np.sin(t), z0 + rise * r * np.sin(phi)]
        N = np.c_[np.cos(phi) * np.cos(t) / r, np.cos(phi) * np.sin(t) / r, np.sin(phi) / (rise * r)]
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        sweep3d(S, rib_key, P - 0.02 * N, N, scale(RIB, rib[1], rib[0]))
    return top


def statue(S, key, c, z, h):
    """Robed standing figure on its plinth (gable apex, pedestals)."""
    lathe(S, key, c, [(0.0, 0.0), (0.2 * h, 0.0), (0.2 * h, 0.06 * h), (0.15 * h, 0.08 * h), (0.16 * h, 0.35 * h),
                      (0.13 * h, 0.55 * h), (0.15 * h, 0.72 * h), (0.11 * h, 0.8 * h), (0.05 * h, 0.83 * h), (0.075 * h, 0.9 * h),
                      (0.06 * h, 0.97 * h), (0.0, 1.0 * h)], z, k=10)
    for s in (-1, 1):                                            # arms folded against the robe
        cylinder(S, key, (c[0] + s * 0.13 * h, c[1] - 0.03 * h), 0.035 * h, 0.03 * h, z + 0.5 * h, z + 0.74 * h, k=6)
