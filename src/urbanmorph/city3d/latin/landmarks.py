"""Landmark towers of the Latin theme (DESIGN.md §5.3; user references 2026-11-23 in architectural features/thin and
architectural features/cohesive): three designs, each in two forms, built with the relief kit (latin.relief) so that
piers, pilasters, columns, cornices, consoles, architraves, balustrades and carved ornament have real depth.

* coral  - Coral Lantern: gold-yellow walls, a coral strip of balconies between deep cream pilasters; a grand crown
           on scroll consoles (arcaded stage with balustraded balconies, oculus stage with cartouches and volute
           ears), corner turrets, a coral ribbed dome with aedicule dormers, a two-tier lantern, a coral spire;
* eagle  - Eagle Belfry: ivory walls framed by green marble columns with peach Corinthian capitals; a chamfered crown
           of two belfries (bronze bells, round green balconies on corbels, marble archivolts, oculi, swan-neck
           pediments, layered peach and cream entablatures), a green balustrade, a stepped peach dome, a bronze eagle;
* golden - Golden Clock Tower: gold walls between deep cream piers and pilasters; a bracketed cornice with a
           balustrade, cypresses and urns; a clock stage (teal dial in a bold bezel, carved C-scrolls), a belfry of
           three gilt bells between columns, an orange ribbed dome with lace dormers and teal oculi, a colonnaded
           lantern with a gilt cupola.

Forms: 'thin' - a shaft under a crown about 1.2 x wider, on an arcaded podium (the user chose shafts close to the
crown width, like the city's skyscrapers); 'cohesive' - one width from the podium to the dome, a tier of storeys
repeated up the shaft. Square plan, built centred in a local frame (X along ``axis``, Y to its left, Z up; the front
faces -Y), then rotated onto the site. Colours come from a scheme (role -> material key), so any house colour can
repaint a tower: landmark(..., colours={"wall": "wall:rosa"}).
"""

from __future__ import annotations

import numpy as np
import shapely
import shapely.affinity

from urbanmorph.city3d.latin import relief as R
from urbanmorph.city3d.latin.church import box_walls, terrace, walls4
from urbanmorph.city3d.latin.kit import Soup, Z_B, beam, cylinder, wbox, wp, wquad, wside
from urbanmorph.city3d.mesh import prism

FL, GF = 3.6, 6.0                  # storey and ground-floor heights (m)
DESIGNS, FORMS = ("coral", "eagle", "golden"), ("thin", "cohesive")
WIDTH = {("coral", "thin"): 23.0, ("coral", "cohesive"): 20.0, ("eagle", "thin"): 22.0, ("eagle", "cohesive"): 20.0,
         ("golden", "thin"): 23.0, ("golden", "cohesive"): 22.0}          # shaft side (m): 1 : 6.5 (thin) to 1 : 7.5
SCHEMES = {
    "coral": {"wall": "lm:amarillo", "accent": "lm:coral", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:crema",
              "dome": "dome:cupula_coral", "spire": "lm:coral", "glass": "glass_teal", "lit": "glass_lit", "metal": "gold",
              "rail": "baluster:blanco", "rail3d": "lm:crema", "frame": "lm:crema", "orn": "lm:crema", "dark": "void"},
    "eagle": {"wall": "lm:marfil", "accent": "lm:melocoton", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:melocoton",
              "dome": "dome:cupula_melocoton", "glass": "glass_teal", "lit": "glass_lit", "metal": "bronze",
              "rail": "baluster:verde", "rail3d": "lm:verde", "column": "marble:verde", "green": "lm:verde",
              "frame": "lm:verde", "orn": "lm:melocoton", "dark": "void"},
    "golden": {"wall": "lm:oro", "accent": "lm:rojo_naranja", "trim": "lm:crema", "base": "lm:marfil", "band": "lm:crema",
               "dome": "dome:cupula_dorada", "glass": "glass_lit", "lit": "glass_lit", "metal": "gold",
               "rail": "baluster:blanco", "rail3d": "lm:crema", "teal": "lm:teal", "frame": "lm:teal", "orn": "lm:crema",
               "dark": "lm:granate"},
}
CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


# ----------------------------------------------------------------------------------------------- helpers
def faces(h):
    return walls4(-h, h, -h, h)


def poly_faces(P):
    """(o, u, n, length) of the faces of a CCW polygon (list of (x, y))."""
    P = np.asarray(P, float)
    out = []
    for i in range(len(P)):
        a, e = P[i], P[(i + 1) % len(P)]
        L = float(np.linalg.norm(e - a))
        u = (e - a) / L
        out.append((a, u, np.array([u[1], -u[0]]), L))
    return out


def chamfer(h, c):
    """Square of half side h with its corners cut by c (CCW, front face first)."""
    return [(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c)]


def circle(r, k=32):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def sq(h):
    return R.rect(-h, h, -h, h)


def solid(S, key, P, z0, z1):
    """Prism on a CCW polygon (walls and top)."""
    V, F, part = prism(shapely.orient_polygons(shapely.Polygon(P)), z0, z1).arrays()
    S.add(key, V, F[part != 0])


def mould(S, key, P, z, prof):
    """Moulding swept round the polygon P at height z; returns its top."""
    R.sweep(S, key, P, z, prof, closed=True)
    return z + max(b for _, b in prof)


def piers(S, key, fs, f, z0, z1, d, cap_key=None, cap=False, base=False):
    """Corner piers on every face: from the corner (covering it) to f x the face length, d out of the wall."""
    for o, u, n, L in fs:
        R.pilaster(S, key, o, u, n, -d, f * L, z0, z1, d, cap_key, base=base, cap=cap)


def bands(S, key, o, u, n, x0, x1, zs, d, t=0.35):
    for z in zs:
        wbox(S, key, o, u, n, x0, x1, z, z + t, 0.0, d)


def cypress(S, x, y, z, h):
    cylinder(S, "plinth:terracota", (x, y), 0.1 * h, 0.14 * h, z, z + 0.22 * h, k=8, cap=True)
    R.lathe(S, "leaves_dark", (x, y), [(0.12 * h, 0.0), (0.17 * h, 0.15 * h), (0.15 * h, 0.4 * h), (0.09 * h, 0.62 * h),
                                       (0.0, 0.8 * h)], z + 0.2 * h, k=8)


def potted(S, x, y, z, rng):
    h = float(rng.uniform(0.6, 1.0))
    cylinder(S, "plinth:terracota", (x, y), 0.14, 0.18, z, z + 0.3, k=6, cap=True)
    R.lathe(S, "leaves", (x, y), [(0.12, 0.0), (0.32 * h, 0.25 * h), (0.28 * h, 0.5 * h), (0.0, 0.75 * h)], z + 0.28, k=6)


def balcony(S, sc, o, u, n, x0, x1, z, D=0.7, h=1.0, real=False, rng=None, plant=0.0):
    """Balcony across [x0, x1]: moulded slab, a balustrade (turned balusters when ``real``, else a cut-out panel)
    with a handrail, sometimes a potted plant."""
    T = sc["trim"]
    path = [wp(o, u, n, x0, 0, 0)[:2], wp(o, u, n, x1, 0, 0)[:2]]
    R.sweep(S, T, path, z - 0.3, R.scale([(0, 0), (0.55, 0), (0.6, 0.12), (1.0, 0.45), (1.0, 1.0), (0, 1.0)], D, 0.3), caps=True)
    if real:
        R.baluster_run(S, sc["rail3d"], o, u, n, x0 + 0.2, x1 - 0.2, z, h, D - 0.2, rail_key=T, posts=True)
    else:
        wquad(S, sc["rail"], o, u, n, x0 + 0.05, x1 - 0.05, z, z + h, D - 0.1, uv="panel")
        wside(S, sc["rail"], o, u, n, x0 + 0.06, 0.0, D - 0.1, z, z + h)
        wside(S, sc["rail"], o, u, n, x1 - 0.06, 0.0, D - 0.1, z, z + h, flip=True)
        R.sweep(S, T, path, z + h, R.closed(R.scale(R.RAIL, 0.2, 0.14, a0=D - 0.2)), caps=True)
    if rng is not None and rng.uniform() < plant:
        p = wp(o, u, n, float(rng.uniform(x0 + 0.4, x1 - 0.4)), 0.45 * D, 0.0)
        potted(S, p[0], p[1], z, rng)


def roundel(S, ring_key, disc_key, o, u, n, x, z, r, d=0.12):
    R.wextrude(S, disc_key, o, u, n, shapely.Point(x, z).buffer(r, quad_segs=6), 0.0, d)
    R.wsweep(S, ring_key, o, u, n, circle(r, 20) + [x, z], R.scale(R.ARCHITRAVE, d + 0.18, 0.18 + 0.12 * r), closed=True)


def cartouche(S, key, o, u, n, xc, zc, w, h, d=0.25, inner=None):
    """Baroque cartouche: a moulded frame with a lobed outline round a recessed oval, C-scrolls curling out at the
    sides, a palmette crest, a pendant leaf."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rx, rz = 0.34 * w, 0.4 * h
    outer = shapely.Polygon(np.c_[xc + rx * np.cos(t) * (1 - 0.1 * np.cos(4 * t)), zc + rz * np.sin(t) * (1 + 0.06 * np.cos(2 * t))])
    hole = shapely.affinity.scale(shapely.Point(xc, zc).buffer(1.0, quad_segs=10), 0.22 * w, 0.27 * h)
    g = outer.difference(hole)
    s = 0.08 * min(w, h)
    for sg in (-1, 1):
        g = g.union(R.scroll(xc + sg * 0.4 * w, zc + 0.12 * h, 0.11 * w, s, sgn=sg, a0=np.pi / 2, turns=1.1,
                             tail=(xc + sg * 0.3 * w, zc - 0.25 * h)))
    g = g.union(R.palmette(xc, zc + 0.36 * h, 0.26 * h, 0.4 * w)).union(R.lobe(xc, zc - 0.38 * h, xc, zc - 0.58 * h, 0.07 * w))
    R.wextrude(S, key, o, u, n, g, 0.0, d)
    if inner:
        R.wextrude(S, inner, o, u, n, hole, 0.0, 0.6 * d)


def consoles(S, key, fs, z, D, h, step=1.9, w=0.6):
    """Scroll consoles under a cornice on every face, tops at z."""
    for o, u, n, L in fs:
        m = max(2, int(round(L / step)))
        for j in range(m + 1):
            R.console(S, key, o, u, n, min(max(L * j / m, w / 2 + 0.1), L - w / 2 - 0.1), w, z, h, D)


def pointed(x0, x1, zs, f, zk, k=8):
    """Pointed arch from the left spring over the apex to the right spring (arcs of radius f x span, heights x zk)."""
    s = x1 - x0
    Rr, xc = f * s, (x0 + x1) / 2
    ta = np.arccos((xc - (x1 - Rr)) / Rr)
    left = [(x0 + Rr + Rr * np.cos(t), zs + zk * Rr * np.sin(t)) for t in np.linspace(np.pi, np.pi - ta, k + 1)]
    right = [(x1 - Rr + Rr * np.cos(t), zs + zk * Rr * np.sin(t)) for t in np.linspace(ta, 0, k + 1)[1:]]
    return left + right


def segment(x0, x1, zs, rise, k):
    """Segmental arch from the right spring over the top to the left spring."""
    s = x1 - x0
    Rr = (s * s / 4 + rise * rise) / (2 * rise)
    xc, zc = (x0 + x1) / 2, zs + rise - Rr
    t1 = np.arctan2(zs - zc, x1 - xc)
    return [(xc + Rr * np.cos(t), zc + Rr * np.sin(t)) for t in np.linspace(t1, np.pi - t1, k + 1)]


def arch_outline(a0, a1, zb, zs, k=12):
    """Round-headed opening: right jamb up, over the arch, left jamb down (counter-clockwise round the opening)."""
    r, xc = (a1 - a0) / 2, (a0 + a1) / 2
    return [(a1, zb), (a1, zs)] + [(xc + r * np.cos(t), zs + r * np.sin(t)) for t in np.linspace(0, np.pi, k + 1)[1:-1]] + \
        [(a0, zs), (a0, zb)]


def arch_poly(a0, a1, zb, zs):
    r = (a1 - a0) / 2
    return shapely.box(a0, zb, a1, zs).union(shapely.Point((a0 + a1) / 2, zs).buffer(r, quad_segs=10))


def podium(S, sc, h, z1, big=False, roundels=True):
    """Arcaded ground storeys on the square of half side h up to z1: stone plinth, tall round-arched shopfronts (warm
    glass, mullion, transom, keystone) in moulded architraves, pilasters between them, roundels in the spandrels, a
    string course and a cornice; returns the cornice top."""
    T, A, B = sc["trim"], sc["accent"], sc["base"]
    box_walls(S, B, -h, h, -h, h, Z_B, z1)
    mould(S, "stone", sq(h), Z_B, R.scale(R.BASE, 0.3, 0.9))
    zf = z1 - 1.8
    for o, u, n, L in faces(h):
        wts = np.array([1.0, 1.35, 1.0] if big else [1.0, 1.0, 1.0])
        pier = max(1.5, 0.085 * L)
        xs = pier + np.r_[0, np.cumsum(wts)] / wts.sum() * (L - 2 * pier)
        for j in range(3):
            a0, a1 = xs[j] + 0.65, xs[j + 1] - 0.65
            R.window(S, sc, o, u, n, a0, a1, 0.9, zf - 0.6, glass=sc["lit"], arch=True, depth=0.42, w=0.42, sill=False,
                     mullion=T, keystone=True)
        for x in xs[1:-1]:
            R.pilaster(S, T, o, u, n, x - 0.4, x + 0.4, 0.9, zf, 0.35)
        if roundels:
            r = min(0.55, 0.06 * (xs[1] - xs[0]))
            for x in xs:
                roundel(S, T, A, o, u, n, x, zf - 0.75 - r, r)
    piers(S, T, faces(h), 0.07, 0.9, zf, 0.45)
    mould(S, sc["band"], sq(h), zf, R.scale(R.STRING, 0.3, 0.55))
    return mould(S, T, sq(h), z1 - 1.25, R.scale(R.CORNICE, 0.95, 1.25))


def podium_roof(S, sc, P, z, inner, cyp=3.2, urns=True):
    """Roof terrace of a podium: pavers, a real balustrade at the edge with urns at the corners, cypresses."""
    T = sc["trim"]
    terrace(S, shapely.box(-P - 0.5, -P - 0.5, P + 0.5, P + 0.5), shapely.box(-inner, -inner, inner, inner), z + 0.02)
    for o, u, n, L in faces(P + 0.3):
        R.baluster_run(S, sc["rail3d"], o, u, n, 0.4, L - 0.4, z, 1.0, -0.25, rail_key=T)
    for sx, sy in CORNERS:
        if urns:
            R.urn(S, T, (sx * (P + 0.05), sy * (P + 0.05)), z + 1.05, 1.3)
        if cyp:
            cypress(S, sx * (P - 1.4), sy * (P - 1.4), z, cyp)


def pedestal(S, sc, q, z, nf=2):
    """Short stage between the podium and a thin shaft: walls, deep cream corner piers, three windows a storey."""
    T = sc["trim"]
    box_walls(S, sc["wall"], -q, q, -q, q, z, z + nf * FL)
    for o, u, n, L in faces(q):
        for f in range(nf):
            for j in range(3):
                xc = L * (0.27 + 0.23 * j)
                R.window(S, sc, o, u, n, xc - 0.75, xc + 0.75, z + f * FL + 0.9, z + f * FL + 3.0, depth=0.3, w=0.28,
                         mullion=sc.get("frame"), hood="cornice" if f == 1 else None)
    piers(S, T, faces(q), 0.09, z, z + nf * FL, 0.5)
    return mould(S, T, sq(q), z + nf * FL, R.scale(R.CORNICE, 0.7, 0.9))


def drum_dome(S, sc, rd, z, rise, ribs=8, rib=(0.7, 0.35), wall=None):
    """Drum (base moulding, wall, string) and a ribbed dome; returns (dome springing, crown top)."""
    T = sc["trim"]
    z = mould(S, T, circle(rd + 0.35, 40), z, R.scale(R.BASE, 0.35, 0.8))
    cylinder(S, wall or sc["accent"], (0, 0), rd + 0.35, rd + 0.35, z, z + 0.9, k=40)
    z = mould(S, T, circle(rd + 0.35, 40), z + 0.9, R.scale(R.STRING, 0.3, 0.45))
    top = R.ribbed_dome(S, sc["dome"], T, rd, z, rise, ribs=ribs, rib=rib, k=40, m=14, phase=np.pi / ribs)
    return z, top


# ----------------------------------------------------------------------------------------------- Coral Lantern
def aedicule(S, sc, o, u, n, wd, z, h, depth):
    """Dome dormer of the Coral Lantern (front plane d = 0, x in [0, wd]): a balustraded balcony, coral pilasters with
    cream capitals and an arched window in a moulded frame, an entablature, a pointed pediment whose coral tympanum
    holds carved cream lace, a finial, coral volute wings with cream scrolls at its foot."""
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    wbox(S, Wl, o, u, n, 0.0, wd, z, z + h, -depth, 0.0)
    pw = 0.14 * wd
    for x0 in (0.0, wd - pw):
        R.pilaster(S, A, o, u, n, x0, x0 + pw, z + 1.3, z + h - 0.9, 0.35, cap_key=T, base_h=0.4, cap_h=0.55)
    R.window(S, sc, o, u, n, 0.27 * wd, 0.73 * wd, z + 1.4, z + h - 1.2, arch=True, depth=0.35, w=0.3, sill=False, keystone=True,
             mullion=T)
    balcony(S, sc, o, u, n, -0.2, wd + 0.2, z + 1.3, D=1.0, h=1.0, real=True)
    base = [wp(o, u, n, -0.35, 0, 0)[:2], wp(o, u, n, wd + 0.35, 0, 0)[:2]]
    R.sweep(S, T, base, z + h - 0.9, R.scale(R.STRING, 0.45, 0.45), caps=True)
    R.sweep(S, T, base, z + h - 0.45, R.scale(R.CORNICE, 0.6, 0.6), caps=True)
    zp = z + h + 0.15
    rise = 0.62 * wd
    tri = shapely.Polygon([(-0.2, zp), (wd + 0.2, zp), (wd / 2, zp + rise)])
    R.wextrude(S, A, o, u, n, tri, -depth, 0.02)
    inner = tri.buffer(-0.5)
    if not inner.is_empty:
        b = inner.bounds
        R.wextrude(S, sc["orn"], o, u, n, R.filigree(b[0], b[2], b[1], b[3]).intersection(inner), 0.0, 0.18)
    R.pediment(S, T, o, u, n, -0.4, wd + 0.4, zp - 0.4, "triangle", depth=0.55, rise=rise + 0.3, h=0.4)
    p = wp(o, u, n, wd / 2, 0.0, 0.0)
    R.obelisk(S, T, p[0], p[1], zp + rise + 0.45, 0.24)
    for x, sg in ((0.0, -1), (wd, 1)):                             # volute wings at the foot of the aedicule
        nn = np.asarray(u) * sg
        uu = np.array([-nn[1], nn[0]])
        R.buttress(S, A, np.asarray(wp(o, u, n, x, -0.45, 0)[:2]), uu, nn, 0.0, 0.8, z, 0.55 * h, 0.32 * wd)
        g = R.scroll(0.0, z + 0.28 * h, 0.09 * wd, 0.035 * wd, sgn=-sg, a0=np.pi / 2, turns=1.3,
                     tail=(sg * 0.1 * wd, z + 0.05 * h))
        R.wextrude(S, T, np.asarray(wp(o, u, n, x + sg * 0.14 * wd, 0.0, 0)[:2]), np.asarray(u), np.asarray(n), g, -0.1, 0.06)


def lantern_coral(S, sc, r, z):
    """Two-tier lantern: an octagonal coral drum with arched lights and cream angle pilasters, an open colonnade of
    cream columns under coral arches with a balustrade, a coral drum, a banded coral spire, a gilt ball, an iron
    needle; returns the top."""
    T, A = sc["trim"], sc["accent"]
    oc = circle(r, 8)
    z = mould(S, T, oc * 1.08, z, R.scale(R.BASE, 0.35, 0.6))
    solid(S, A, oc, z, z + 0.9 * r)
    for o, u, n, L in poly_faces(oc):
        R.window(S, sc, o, u, n, 0.28 * L, 0.72 * L, z + 0.2 * r, z + 0.75 * r, arch=True, depth=0.2, w=0.14, sill=False)
        R.pilaster(S, T, o, u, n, -0.1, 0.12 * L, z, z + 0.9 * r, 0.15, base=False, cap=False)
    z = mould(S, T, oc, z + 0.9 * r, R.scale(R.CORNICE, 0.45, 0.55))
    for o, u, n, L in poly_faces(circle(0.95 * r, 8)):
        R.baluster_run(S, sc["rail3d"], o, u, n, 0.1, L - 0.1, z, 0.7, -0.15, rail_key=T, posts=False, pitch=0.25, k=5)
    cylinder(S, "void", (0, 0), 0.55 * r, 0.55 * r, z, z + 1.2 * r, k=8)
    for x, y in circle(0.8 * r, 8):
        R.column(S, T, (x, y), max(0.14, 0.08 * r), z, z + 0.9 * r, order="tuscan", k=8)
    for o, u, n, L in poly_faces(circle(0.86 * r, 8)):
        g = shapely.box(0, z + 0.9 * r, L, z + 1.2 * r).difference(shapely.Point(L / 2, z + 0.9 * r).buffer(0.4 * L, quad_segs=6))
        R.wextrude(S, A, o, u, n, g, -0.2, 0.08)
    z = mould(S, T, circle(0.9 * r, 8), z + 1.2 * r, R.scale(R.CORNICE, 0.35, 0.45))
    solid(S, A, circle(0.72 * r, 8), z, z + 0.45 * r)
    z = mould(S, T, circle(0.72 * r, 8), z + 0.45 * r, R.scale(R.STRING, 0.2, 0.3))
    H = 2.7 * r
    cylinder(S, sc["spire"], (0, 0), 0.75 * r, 0.06, z, z + H, k=8)
    for f in (0.15, 0.4, 0.65):
        mould(S, T, circle(0.75 * r * (1 - f) + 0.06 * f + 0.02, 8), z + f * H, R.scale(R.STRING, 0.12, 0.28))
    z += H
    R.ball(S, sc["metal"], (0, 0), z - 0.25, 0.28 + 0.04 * r)
    cylinder(S, "iron", (0, 0), 0.08, 0.01, z, z + 1.8 * r + 2.0, k=4)
    return z + 1.8 * r + 2.0


def coral_crown_dome(S, sc, w, z, rd, rise, turrets):
    """Slender banded corner turrets with obelisks (when ``turrets`` > 0: their centre offset), the drum and ribbed
    coral dome, four tall aedicule dormers between the ribs, the two-tier lantern; returns the top."""
    T, Wl = sc["trim"], sc["wall"]
    zd, top = drum_dome(S, sc, rd, z, rise, ribs=8, rib=(0.06 * rd + 0.35, 0.35))
    for sx, sy in CORNERS if turrets else ():
        x, y = sx * turrets, sy * turrets
        s = 0.028 * w + 0.2
        th = 0.55 * rd * rise + (zd - z)
        P = R.rect(x - s, x + s, y - s, y + s)
        box_walls(S, Wl, x - s, x + s, y - s, y + s, z, z + th)
        mould(S, T, P, z, R.scale(R.BASE, 0.2, 0.5))
        for f in (0.3, 0.55, 0.8):
            mould(S, T, P, z + f * th, R.scale(R.STRING, 0.15, 0.4))
        zt = mould(S, T, P, z + th, R.scale(R.CORNICE, 0.3, 0.55))
        R.obelisk(S, T, x, y, zt, 0.42 * s + 0.12)
    wd = 0.64 * rd
    for o, u, n, L in faces(rd):
        o2 = np.asarray(n) * (rd + 0.4) - np.asarray(u) * (wd / 2)
        aedicule(S, sc, o2, u, n, wd, z - 0.2, 0.5 * rd * rise, 0.85 * rd)
    return lantern_coral(S, sc, 0.33 * rd, top - 0.3)


def coral_thin(S, sc, w, H, rng, nf=None):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw, P = w / 2, 0.62 * w
    z = podium(S, sc, P, GF + FL)
    q = 0.56 * w
    podium_roof(S, sc, P, z, q)
    z = pedestal(S, sc, q, z)
    terrace(S, shapely.box(-q - 0.6, -q - 0.6, q + 0.6, q + 0.6), shapely.box(-hw, -hw, hw, hw), z + 0.02)
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.3), sy * (q - 0.3), z, 2.8)
    # shaft: deep cream piers; a column of windows each side; a coral centre strip of balconies between pilasters
    if nf is None:
        nf = max(8, round((H - z - (2.55 * w + 14.0)) / FL))
    z0, zs1 = z, z + nf * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, zs1 + 2.6)
    for o, u, n, L in faces(hw):
        x0, x1 = 0.37 * L, 0.63 * L
        R.wextrude(S, A, o, u, n, shapely.box(x0, z0, x1, zs1 + 0.4), 0.0, 0.04)
        for xa, xb in ((x0 - 0.75, x0), (x1, x1 + 0.75)):
            R.pilaster(S, T, o, u, n, xa, xb, z0, zs1 + 0.4, 0.6, base=False, cap=False)
        for f in range(nf):
            zf = z0 + f * FL
            for xc in (0.215 * L, 0.785 * L):
                R.window(S, sc, o, u, n, xc - 1.05, xc + 1.05, zf + 0.85, zf + 3.05, depth=0.32, w=0.3, mullion=T,
                         hood="cornice" if f % 3 == 2 else None, prof=R.ARCH_S)
            if f < nf - 2:
                R.window(S, sc, o, u, n, 0.5 * L - 0.9, 0.5 * L + 0.9, zf + 0.3, zf + 3.0, depth=0.25, w=0.25, sill=False,
                         mullion=T, prof=R.ARCH_S)
                balcony(S, sc, o, u, n, x0 + 0.15, x1 - 0.15, zf + 0.3, D=0.75, h=0.95)
        zt = z0 + (nf - 2) * FL + 0.4
        R.window(S, sc, o, u, n, x0 + 0.6, x1 - 0.6, zt, zs1 - 0.2, arch=True, depth=0.35, w=0.32, sill=False, mullion=T,
                 keystone=True)
        balcony(S, sc, o, u, n, x0 + 0.15, x1 - 0.15, zt, D=0.8, h=0.95, real=True)
    piers(S, T, faces(hw), 0.075, z0, zs1 + 2.6, 0.6)
    for zb in np.arange(z0 + 6 * FL, zs1, 6 * FL):                # string courses every six storeys
        mould(S, T, sq(hw + 0.6), zb, R.scale(R.STRING, 0.18, 0.4))
    # grand crown on scroll consoles
    zc = zs1 + 2.6
    D = 0.62 * w - hw
    mould(S, T, sq(hw + 0.6), zc - 2.9, R.scale(R.STRING, 0.2, 0.35))
    consoles(S, T, faces(hw + 0.6), zc, D - 0.6, 2.4, step=2.0, w=0.65)
    z = mould(S, T, sq(hw), zc, R.scale(R.CORNICE, D + 0.4, 1.4))
    c1, h1 = 0.58 * w, 0.4 * w
    for sx, sy in CORNERS:
        R.obelisk(S, T, sx * (c1 + 0.8), sy * (c1 + 0.8), z, 0.5)
    box_walls(S, Wl, -c1, c1, -c1, c1, z, z + h1)
    mould(S, T, sq(c1), z, R.scale(R.BASE, 0.25, 0.6))
    for o, u, n, L in faces(c1):
        for a0, a1, f in ((0.15 * L, 0.31 * L, 0.72), (0.37 * L, 0.63 * L, 0.86), (0.69 * L, 0.85 * L, 0.72)):
            zt = z + f * h1
            r = (a1 - a0) / 2
            R.wextrude(S, A, o, u, n, arch_poly(a0 - 0.6, a1 + 0.6, z + 0.7, zt - r).difference(arch_poly(a0, a1, z + 1.6, zt - r)),
                       0.0, 0.05)
            R.window(S, sc, o, u, n, a0, a1, z + 1.6, zt, glass=sc["glass"], arch=True, depth=0.38, w=0.32, sill=False, mullion=T,
                     keystone=True)
            balcony(S, sc, o, u, n, a0 - 0.45, a1 + 0.45, z + 1.6, D=0.95, h=1.0, real=True)
        for x in (0.34 * L, 0.66 * L):
            R.pilaster(S, Wl, o, u, n, x - 0.45, x + 0.45, z, z + h1 - 1.5, 0.45, cap_key=T)
        bands(S, T, o, u, n, -0.8, 0.1 * L, (z + 0.33 * h1, z + 0.66 * h1), 0.85)
        bands(S, T, o, u, n, 0.9 * L, L + 0.8, (z + 0.33 * h1, z + 0.66 * h1), 0.85)
    piers(S, Wl, faces(c1), 0.1, z, z + h1 - 1.5, 0.8, cap_key=T, cap=True)
    z = mould(S, T, sq(c1), z + h1 - 1.5, R.scale(R.STRING, 0.3, 0.5))
    z = mould(S, T, sq(c1), z, R.scale(R.CORNICE, 1.0, 1.0))
    c2, h2 = 0.52 * w, 0.3 * w
    for sx, sy in CORNERS:
        R.obelisk(S, T, sx * (c1 + 0.4), sy * (c1 + 0.4), z, 0.42)
    box_walls(S, Wl, -c2, c2, -c2, c2, z, z + h2)
    for o, u, n, L in faces(c2):
        xc, zc2 = 0.5 * L, z + 0.5 * h2
        r = 0.12 * L
        for x in (xc - r - 0.9, xc + r + 0.3):
            R.pilaster(S, T, o, u, n, x, x + 0.6, z + 0.3, z + h2 - 0.9, 0.3, base=False)
        R.oculus(S, T, A, o, u, n, xc, zc2, r, r, w=0.45, depth=0.4)
        for xa, xb in ((0.12 * L, 0.33 * L), (0.67 * L, 0.88 * L)):
            R.wextrude(S, A, o, u, n, shapely.box(xa, z + 0.6, xb, z + h2 - 1.1), 0.0, 0.05)
            R.wsweep(S, T, o, u, n, R.rect(xa, xb, z + 0.6, z + h2 - 1.1), R.scale(R.ARCHITRAVE, 0.2, 0.2), closed=True)
            cartouche(S, T, o, u, n, (xa + xb) / 2, z + 0.52 * h2, 0.7 * (xb - xa), 0.6 * h2, d=0.3, inner=A)
        for x in (1.0, L - 1.0):                                   # volute ears down to the stage below
            R.buttress(S, T, o, u, n, x, 0.7, z, 0.5 * h2, c1 - c2 + 0.5)
    piers(S, Wl, faces(c2), 0.08, z, z + h2 - 0.9, 0.6, cap_key=T, cap=True)
    z = mould(S, T, sq(c2), z + h2 - 0.9, R.scale(R.CORNICE, 0.9, 0.9))
    for sx, sy in CORNERS:
        R.obelisk(S, T, sx * (c2 + 0.25), sy * (c2 + 0.25), z, 0.38)
    rd = 0.44 * w
    terrace(S, shapely.box(-c2 - 0.8, -c2 - 0.8, c2 + 0.8, c2 + 0.8), shapely.Polygon(circle(rd + 0.6, 32)), z + 0.02)
    top = coral_crown_dome(S, sc, w, z, rd, 1.25, c2 - 0.035 * w - 0.6)
    return top, nf + 6, FL, {"crown_z": zs1, "n": nf}


def coral_tier(S, sc, hw, z, nf, rng, gable=True):
    """One tier of the cohesive Coral Lantern on every face: a string course, a triple arcade of coral archivolts on
    cream Corinthian colonnettes (nf storeys of lit windows and balconies with plants), a pointed gable with a coral
    tympanum and carved cream relief, a roundel; deep cream corner pilasters and coral strips; returns the top."""
    T, A = sc["trim"], sc["accent"]
    L = 2 * hw
    xa0, xa1 = 0.18 * L, 0.82 * L
    bay = (xa1 - xa0) / 3
    zf0 = z + 0.9
    zs = zf0 + nf * FL - 0.5
    top = zs + bay / 2 + 1.2
    gx0, gx1 = xa0 - 0.35, xa1 + 0.35
    if gable:
        apex = max(p[1] for p in pointed(gx0, gx1, zs + 0.25, 0.6, 0.95))
        top = apex + 0.6 + 3.2
    for o, u, n, L_ in faces(hw):
        for j in range(3):
            a0, a1 = xa0 + j * bay + 0.38, xa0 + (j + 1) * bay - 0.38
            r = (a1 - a0) / 2
            R.wextrude(S, A, o, u, n, arch_poly(a0, a1, zf0 - 0.3, zs), -0.3, -0.03)
            outline = arch_outline(a0, a1, zf0 - 0.3, zs)
            R.wsweep(S, A, o, u, n, outline, R.scale(R.ARCHITRAVE, 0.4, 0.42))
            R.wsweep(S, T, o, u, n, outline[1:-1], R.scale(R.STRING, 0.46, 0.16, b0=0.42))
            R.window(S, sc, o, u, n, a0 + 0.35, a1 - 0.35, zs - 0.3, zs + r - 0.3, arch=True, glass=sc["lit"], depth=0.12, w=0.08,
                     sill=False)
            for f in range(nf):
                zf = zf0 + f * FL
                R.window(S, sc, o, u, n, a0 + 0.3, a1 - 0.3, zf + 0.2, zf + 2.7, glass=sc["lit"], depth=0.15, w=0.12, sill=False,
                         mullion=T, prof=R.ARCH_S)
                balcony(S, sc, o, u, n, a0 + 0.05, a1 - 0.05, zf + 0.15, D=0.6, h=0.95, rng=rng, plant=0.45)
        for j in range(4):
            p = wp(o, u, n, xa0 + j * bay, 0.35, 0.0)
            R.column(S, T, (p[0], p[1]), 0.26, zf0 - 0.3, zs + 0.15, order="corinthian", k=10)
        if gable:
            inner = pointed(gx0, gx1, zs + 0.25, 0.6, 0.95)
            poly = shapely.Polygon(inner)
            R.wextrude(S, A, o, u, n, poly, 0.0, 0.04)
            vis = poly.intersection(shapely.box(-1e3, zs + bay / 2 + 0.6, 1e3, 1e3)).buffer(-0.35)
            if not vis.is_empty:
                b = vis.bounds
                R.wextrude(S, sc["orn"], o, u, n, R.filigree(b[0], b[2], b[1], b[3]).intersection(vis), 0.0, 0.2)
            R.wsweep(S, T, o, u, n, inner, R.scale(R.RAKE, 0.45, 0.5), caps=True)
            p = wp(o, u, n, L_ / 2, 0.2, 0.0)
            R.obelisk(S, T, p[0], p[1], apex + 0.25, 0.28)
            roundel(S, T, A, o, u, n, L_ / 2, apex + 0.6 + 1.35, 0.85)
    for o, u, n, L_ in faces(hw):
        for xa, xb in ((0.075 * L_, 0.115 * L_), (0.885 * L_, 0.925 * L_)):
            R.wextrude(S, A, o, u, n, shapely.box(xa, z, xb, top), 0.0, 0.05)
    piers(S, T, faces(hw), 0.075, z + 0.35, top - 0.45, 0.45, cap=True)
    mould(S, T, sq(hw), z, R.scale(R.STRING, 0.5, 0.4))
    mould(S, T, sq(hw + 0.45), top - 0.45, R.scale(R.STRING, 0.25, 0.45))
    return top


def coral_cohesive(S, sc, w, H, rng, nf=None):
    T, Wl = sc["trim"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.5, GF + 0.6 * FL)
    z0 = z
    Ht = coral_tier(Soup(), sc, hw, 0.0, 3, np.random.default_rng(0))
    if nf is None:
        nf = max(2, round((H - z - (2.2 * w + 16.0)) / Ht))
    for i in range(nf):
        z = coral_tier(S, sc, hw, z, 3, rng)
    zc = z
    z = coral_tier(S, sc, hw, z, 2, rng, gable=False)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    z = mould(S, T, sq(hw), z, R.scale(R.CORNICE, 1.0, 1.1))
    rd = 0.44 * w
    for sx, sy in CORNERS:                                         # slim octagonal corner turrets with coral spirelets
        c = np.array([sx * (hw - 0.9), sy * (hw - 0.9)])
        oc = circle(0.95, 8) + c
        solid(S, T, oc, z, z + 0.42 * w)
        for o, u, n, L in poly_faces(oc)[::2]:
            R.window(S, sc, o, u, n, 0.15 * L, 0.85 * L, z + 0.24 * w, z + 0.36 * w, arch=True, glass="void", depth=0.08,
                     w=0.06, sill=False)
        zt = mould(S, T, oc, z + 0.42 * w, R.scale(R.CORNICE, 0.25, 0.45))
        cylinder(S, sc["spire"], c, 1.0, 0.05, zt, zt + 0.22 * w, k=8)
        R.ball(S, sc["metal"], c, zt + 0.22 * w - 0.1, 0.2)
    terrace(S, shapely.box(-hw - 1.0, -hw - 1.0, hw + 1.0, hw + 1.0), shapely.Polygon(circle(rd + 0.6, 32)), z + 0.02)
    top = coral_crown_dome(S, sc, w, z, rd, 1.25, 0)
    return top, 1 + 3 * nf + 2, Ht, {"crown_z": zc, "tier": [zc - 2 * Ht, zc - Ht], "n": nf}


# ----------------------------------------------------------------------------------------------- Eagle Belfry
def eagle_columns(S, sc, fs, z0, z1, rc, rings=()):
    """Green marble columns at both ends of every face (paired round each corner), peach Corinthian capitals; at each
    ring height a capital and base collar (the column runs on through the tiers)."""
    C, A = sc["column"], sc["accent"]
    for o, u, n, L in fs:
        for x in (rc + 0.3, L - rc - 0.3):
            p = wp(o, u, n, x, rc + 0.1, 0.0)
            R.column(S, C, (p[0], p[1]), rc, z0, z1, cap_key=A, base_key=A, k=12)
            for zr in rings:
                R.lathe(S, A, (p[0], p[1]), [(1.0 * rc, -1.6 * rc), (1.15 * rc, -1.1 * rc), (1.4 * rc, -0.3 * rc), (1.55 * rc, 0.0),
                                              (1.2 * rc, 0.2 * rc), (1.3 * rc, 0.45 * rc), (1.0 * rc, 0.6 * rc)], zr, k=12)


def round_balcony(S, sc, o, u, n, xc, z, r, k=12):
    """Semicircular balcony on a curved peach corbel: moulded slab, turned green balusters, a green handrail."""
    T, G, A = sc["trim"], sc.get("green", sc["trim"]), sc["accent"]
    rim = [(xc + r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, np.pi, k + 1)]
    path = [wp(o, u, n, x, d, 0)[:2] for x, d in rim][::-1]          # counter-clockwise: the profile faces out
    R.sweep(S, T, path, z - 0.32, R.closed(R.scale([(0, 0), (0.5, 0), (0.55, 0.15), (1.0, 0.5), (1.0, 1.0), (0, 1.0)], 0.3, 0.32,
                                                   a0=-0.3)), caps=True)
    top = [wp(o, u, n, xc, 0.0, z)] + [wp(o, u, n, x, d, z) for x, d in rim]
    R.add_oriented(S, T, [[top[0], top[i + 1], top[i + 2]] for i in range(k)], [0.0, 0.0, 1.0])
    prof = [(0.0, 0.0), (0.25 * r, -0.05), (0.55 * r, -0.25), (0.8 * r, -0.6), (0.9 * r, -1.0), (0.6 * r, -1.25), (0.2 * r, -1.45),
            (0.0, -1.5)]
    u3, n3 = np.array([u[0], u[1], 0.0]), np.array([n[0], n[1], 0.0])
    Tri, Nrm = [], []
    for i in range(k):
        a0, a1 = np.pi * i / k, np.pi * (i + 1) / k
        rad = np.cos((a0 + a1) / 2) * u3 + np.sin((a0 + a1) / 2) * n3
        for (ra, za), (rb, zb) in zip(prof[:-1], prof[1:]):
            P = [wp(o, u, n, xc + ra * np.cos(a0), ra * np.sin(a0), z - 0.32 + za), wp(o, u, n, xc + ra * np.cos(a1), ra * np.sin(a1), z - 0.32 + za),
                 wp(o, u, n, xc + rb * np.cos(a1), rb * np.sin(a1), z - 0.32 + zb), wp(o, u, n, xc + rb * np.cos(a0), rb * np.sin(a0), z - 0.32 + zb)]
            nn = rad * (za - zb) + np.array([0.0, 0.0, 1.0]) * (rb - ra)
            Tri += [[P[0], P[1], P[2]], [P[0], P[2], P[3]]]
            Nrm += [nn, nn]
    R.add_oriented(S, A, Tri, Nrm)
    rr = r - 0.22
    for i in range(k):
        a = np.pi * (i + 0.5) / k
        p = wp(o, u, n, xc + rr * np.cos(a), rr * np.sin(a), 0.0)
        R.lathe(S, sc["rail3d"], (p[0], p[1]), R.scale(R.BALUSTER, 0.16, 0.78), z, k=6)
    rail = [wp(o, u, n, xc + rr * np.cos(a), rr * np.sin(a), 0)[:2] for a in np.linspace(0, np.pi, k + 1)][::-1]
    R.sweep(S, G, rail, z + 0.78, R.closed(R.scale(R.RAIL, 0.24, 0.16, a0=-0.12)), caps=True)


def panel(S, key, o, u, n, x0, x1, z0, z1, d=0.12, w=0.16):
    """Moulded rectangular panel frame on a wall (raised fillet and ovolo)."""
    R.wsweep(S, key, o, u, n, R.rect(x0, x1, z0, z1), R.scale([(0, 0), (0.6, 0), (1.0, 0.35), (1.0, 0.65), (0.6, 1.0), (0, 1.0)], d, w),
             closed=True)



def entablature(S, sc, P, z, h=2.6, grow=1.0):
    """Layered entablature round the polygon P: peach architrave, cream frieze, big cream cornice, peach crown
    moulding projecting beyond it; returns its top."""
    T, A = sc["trim"], sc["accent"]
    z = mould(S, A, P, z, R.scale(R.STRING, 0.4, 0.22 * h))
    z += 0.22 * h
    z = mould(S, T, P, z, R.scale(R.CORNICE, 1.15 * grow, 0.38 * h))
    return mould(S, A, [(x * (1 + 0.0), y) for x, y in P], z, R.scale([(0, 0), (1.25, 0), (1.32, 0.3), (1.22, 0.7), (1.0, 1.0), (0, 1.0)],
                                                                       grow, 0.18 * h))



def eagle_bay(S, sc, o, u, n, L, z, h, aw, glass=None, xc=None):
    """Arched bay of the Eagle Belfry: green marble archivolt with a cream outer moulding, peach keystone and imposts,
    a bronze bell (or glass with green glazing bars), round green balcony on a peach corbel, cream pilasters with
    peach capitals, peach-framed panels, two oculi under small hoods, a broad peach swan-neck pediment."""
    T, A, C = sc["trim"], sc["accent"], sc["column"]
    xc = L / 2 if xc is None else xc
    a0, a1 = xc - aw / 2, xc + aw / 2
    zb = z + 0.18 * h
    zs = z + h - 3.2 - aw / 2
    outline = arch_outline(a0, a1, zb, zs, 14)
    if glass:
        R.glass_panel(S, glass, o, u, n, arch_poly(a0, a1, zb, zs))
        wbox(S, sc["frame"], o, u, n, xc - 0.06, xc + 0.06, zb, zs + aw / 2, 0.0, 0.1)
        wbox(S, sc["frame"], o, u, n, a0, a1, zs - 0.06, zs + 0.06, 0.0, 0.1)
    else:
        R.wextrude(S, "void", o, u, n, arch_poly(a0, a1, zb, zs), -0.05, 0.02)
        cb = wp(o, u, n, xc, -0.3, 0.0)
        R.bell(S, sc["metal"], (cb[0], cb[1]), zs + 0.2, min(0.34 * aw, 1.9))
        beam(S, "wood:madera_oscura", wp(o, u, n, a0 + 0.1, -0.3, zs + 0.35), wp(o, u, n, a1 - 0.1, -0.3, zs + 0.35), 0.25, 0.3)
    R.wsweep(S, C, o, u, n, outline, R.scale(R.ARCHITRAVE, 0.5, 0.55))
    R.wsweep(S, T, o, u, n, outline[1:-1], R.scale(R.STRING, 0.32, 0.28, b0=0.55))
    for x in (a0 - 0.3, a1 + 0.3):                                  # imposts
        wbox(S, A, o, u, n, x - 0.45, x + 0.45, zs - 0.35, zs, 0.0, 0.55)
    R.wextrude(S, A, o, u, n, shapely.Polygon([(xc - 0.32, zs + aw / 2 + 0.1), (xc + 0.32, zs + aw / 2 + 0.1), (xc + 0.45, zs + aw / 2 + 1.1),
                                               (xc - 0.45, zs + aw / 2 + 1.1)]), 0.0, 0.7)
    round_balcony(S, sc, o, u, n, xc, zb, 0.5 * aw + 0.4)
    for x in (a0 - 1.55, a1 + 1.55):
        if glass:
            R.pilaster(S, T, o, u, n, x - 0.4, x + 0.4, z + 0.3, z + h - 1.0, 0.3, cap_key=A)
        else:                                                      # crown bays: engaged marble columns on pedestals
            p = wp(o, u, n, x, 0.5, 0.0)
            box_walls(S, A, p[0] - 0.7, p[0] + 0.7, p[1] - 0.7, p[1] + 0.7, z, z + 1.2)
            R.column(S, C, (p[0], p[1]), 0.45, z + 1.2, z + h - 0.9, cap_key=A, base_key=A, k=12)
    for x0, x1 in ((0.9, a0 - 2.15), (a1 + 2.15, L - 0.9)):
        if x1 - x0 > 1.2:
            xm = (x0 + x1) / 2
            zo = z + 0.6 * h
            panel(S, A, o, u, n, x0 + 0.2, x1 - 0.2, z + 0.6, zo - 1.5, d=0.12, w=0.18)
            r = min(0.6, 0.3 * (x1 - x0))
            R.oculus(S, A, "void", o, u, n, xm, zo, r, r, w=0.32, depth=0.38, lugs=False)
            R.pediment(S, A, o, u, n, xm - r - 0.55, xm + r + 0.55, zo + r + 0.45, "broken", depth=0.5, rise=0.55, h=0.2, band=0.35)
            panel(S, A, o, u, n, x0 + 0.2, x1 - 0.2, zo + r + 1.3, z + h - 1.4, d=0.12, w=0.18)
    R.pediment(S, A, o, u, n, a0 - 2.2, a1 + 2.2, zs + aw / 2 + 1.35, "broken", depth=1.0, rise=1.7, h=0.4, band=0.75)
    R.pediment(S, T, o, u, n, a0 - 1.5, a1 + 1.5, zs + aw / 2 + 1.0, "segment", depth=0.5, rise=1.1, h=0.25, band=0.3)


def stepped_dome(S, sc, r, z, steps=6):
    """Stepped peach dome on a tall rounded profile: tiers of a moulded peach band and a sloping skirt, a cap with
    small windows and a cornice; returns the top."""
    A, D, T = sc["accent"], sc["dome"], sc["trim"]
    ph = np.linspace(0.0, 1.2, steps + 1)
    rs, zs = r * np.cos(ph), z + 1.15 * r * np.sin(ph)
    for i in range(steps):
        zb = mould(S, A, circle(rs[i], 32), zs[i], R.scale(R.STRING, 0.2, 0.08 * r))
        cylinder(S, D, (0, 0), rs[i], 1.03 * rs[i + 1], zb, max(zs[i + 1], zb + 0.3), k=32, cap=True)
    z = zs[-1]
    rc = rs[-1]
    cylinder(S, D, (0, 0), rc, rc, z, z + 0.3 * r, k=8)
    for o, u, n, L in poly_faces(circle(rc, 8))[::2]:
        R.window(S, {"trim": T, "glass": "void"}, o, u, n, 0.25 * L, 0.75 * L, z + 0.07 * r, z + 0.22 * r, depth=0.12, w=0.1,
                 sill=False)
    return mould(S, T, circle(rc, 8), z + 0.3 * r, R.scale(R.CORNICE, 0.35, 0.55))


def ball_balustrade(S, sc, P, z):
    """Green balustrade round the polygon P: turned balusters, handrail, posts with green balls at every vertex."""
    G = sc["green"]
    for o, u, n, L in poly_faces(P):
        R.baluster_run(S, sc["rail3d"], o, u, n, 0.35, L - 0.35, z, 1.1, -0.2, rail_key=G, posts=False)
    for x, y in P:
        box_walls(S, G, x - 0.32, x + 0.32, y - 0.32, y + 0.32, z, z + 1.45)
        R.ball(S, G, (x, y), z + 1.45, 0.34)


def eagle(S, key, c, z, s):
    """Bronze eagle with raised wings ending in fingered primaries, facing -Y, on a ball (s: wingspan, m)."""
    x, y = c
    R.ball(S, key, c, z, 0.07 * s)
    z += 0.12 * s
    R.lathe(S, key, (x, y - 0.01 * s), [(0.0, 0.0), (0.07 * s, 0.03 * s), (0.1 * s, 0.13 * s), (0.09 * s, 0.24 * s),
                                         (0.06 * s, 0.31 * s), (0.045 * s, 0.34 * s), (0.0, 0.35 * s)], z, k=10)
    R.ball(S, key, (x, y - 0.04 * s), z + 0.31 * s, 0.05 * s)
    V = np.array([[x, y - 0.14 * s, z + 0.35 * s], [x - 0.025 * s, y - 0.07 * s, z + 0.37 * s], [x + 0.025 * s, y - 0.07 * s, z + 0.37 * s],
                  [x, y - 0.07 * s, z + 0.32 * s]])
    tris = np.array([V[[0, 1, 2]], V[[0, 2, 3]], V[[0, 3, 1]]])
    R.add_oriented(S, key, tris, tris.mean(1) - np.array([x, y - 0.08 * s, z + 0.35 * s]))
    base = shapely.Polygon([(0.04, 0.22), (0.16, 0.42), (0.3, 0.62), (0.4, 0.76), (0.46, 0.7), (0.42, 0.42), (0.3, 0.3), (0.16, 0.16),
                            (0.06, 0.12)])
    fingers = [R.lobe(0.36 + 0.02 * i, 0.68 - 0.09 * i, 0.5 + 0.03 * i - 0.01 * i * i, 0.86 - 0.1 * i, 0.03) for i in range(5)]
    g = shapely.affinity.scale(shapely.union_all([base] + fingers).buffer(0.005), s, s, origin=(0, 0))
    for sx in (-1, 1):
        ax = np.array([sx * np.cos(np.radians(32)), np.sin(np.radians(32)), 0.0])     # wings swept back
        R.pextrude(S, key, (x, y + 0.03 * s, z), ax, R.ZV, np.cross(ax, R.ZV), g, -0.015 * s, 0.015 * s, back=True)
    tail = shapely.affinity.scale(shapely.Polygon([(-0.05, 0.06), (0.05, 0.06), (0.11, -0.1), (0.0, -0.12), (-0.11, -0.1)]), s, s,
                                  origin=(0, 0))
    R.pextrude(S, key, (x, y + 0.1 * s, z), np.array([1.0, 0, 0]), np.array([0, 0.6, 0.8]), np.array([0, -0.8, 0.6]), tail,
               -0.012 * s, 0.012 * s, back=True)


def eagle_crown(S, sc, w, z, b1, h1, b2, h2):
    """Crown of the Eagle Belfry: two chamfered belfry stages (arched bays on the main faces; paired marble columns on
    the chamfers), each under a layered entablature with swan-neck pediments and urns rising over the chamfers, volute
    buttresses from the upper stage down onto the lower cornice, the green balustrade, the stepped dome, the eagle."""
    T, A, C, Wl = sc["trim"], sc["accent"], sc["column"], sc["wall"]
    prev = None
    for b, h, aw in ((b1, h1, 0.26 * 2 * b1), (b2, h2, 0.27 * 2 * b2)):
        ch = 0.17 * b
        Pl = chamfer(b, ch)
        solid(S, Wl, Pl, z, z + h)
        mould(S, A, Pl, z, R.scale(R.BASE, 0.35, 0.8))
        fs = poly_faces(Pl)
        for i, (o, u, n, L) in enumerate(fs):
            if i % 2 == 0:
                eagle_bay(S, sc, o, u, n, L, z, h - 1.0, aw)
            else:
                for x in (0.26 * L, 0.74 * L):
                    p = wp(o, u, n, x, 0.65, 0.0)
                    box_walls(S, A, p[0] - 0.8, p[0] + 0.8, p[1] - 0.8, p[1] + 0.8, z, z + 1.4)
                    mould(S, T, R.rect(p[0] - 0.8, p[0] + 0.8, p[1] - 0.8, p[1] + 0.8), z + 1.4, R.scale(R.STRING, 0.12, 0.25))
                    R.column(S, C, (p[0], p[1]), 0.52, z + 1.65, z + h - 1.0, cap_key=A, base_key=A, k=12)
                R.wextrude(S, "void", o, u, n, arch_poly(0.42 * L, 0.58 * L, z + 0.4 * h, z + 0.72 * h), -0.05, 0.02)
                R.wsweep(S, A, o, u, n, arch_outline(0.42 * L, 0.58 * L, z + 0.4 * h, z + 0.72 * h), R.scale(R.ARCHITRAVE, 0.3, 0.25))
                if prev is not None:                               # volute buttress down onto the stage below
                    R.buttress(S, A, o, u, n, L / 2, 1.0, z, 0.32 * h, prev - b + 0.6)
        zt = entablature(S, sc, Pl, z + h - 1.0, h=3.2, grow=1.5)
        for i, (o, u, n, L) in enumerate(poly_faces(chamfer(b + 1.6, ch + 0.65))):
            if i % 2:
                R.pediment(S, A, o, u, n, -0.4, L + 0.4, zt, "broken", depth=1.3, rise=0.9 * L, h=0.4, band=0.8)
                p = wp(o, u, n, L / 2, -0.4, 0.0)
                R.urn(S, T, (p[0], p[1]), zt + 0.35, 1.6)
        z = zt
        prev = b
    ball_balustrade(S, sc, chamfer(b2 + 0.5, 0.17 * b2 + 0.25), z)
    top = stepped_dome(S, sc, 0.8 * b2, z)
    eagle(S, sc["metal"], (0.0, 0.0), top, 0.42 * w)
    return top + 0.42 * w * 0.9


def eagle_thin(S, sc, w, H, rng, nf=None):
    T, A, Wl, B = sc["trim"], sc["accent"], sc["wall"], sc["base"]
    hw, P = w / 2, 0.6 * w
    z = podium(S, sc, P, GF + FL, big=True, roundels=False)
    eagle_columns(S, sc, faces(P), Z_B + 0.9, z - 1.25, 0.5)
    q = 0.56 * w
    box_walls(S, B, -q, q, -q, q, z, z + FL)
    for o, u, n, L in faces(q):
        for j in range(7):
            xc = L * (0.15 + 0.117 * j)
            R.window(S, sc, o, u, n, xc - 0.55, xc + 0.55, z + 1.0, z + 2.6, depth=0.25, w=0.22, mullion=sc["frame"], sill=False,
                     prof=R.ARCH_S)
    z = mould(S, A, sq(q), z + FL, R.scale(R.STRING, 0.3, 0.4))
    z = mould(S, T, sq(q), z, R.scale(R.CORNICE, 0.6, 0.7))
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.3), sy * (q - 0.3), z, 2.2)
    if nf is None:
        nf = max(10, round((H - z - (2.3 * w + 10.0)) / FL))
    z0, zs1 = z, z + nf * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, zs1 + 1.4)
    for o, u, n, L in faces(hw):
        for x in (0.2 * L, 0.8 * L):
            R.pilaster(S, sc["column"], o, u, n, x - 0.35, x + 0.35, z0, zs1 + 1.4, 0.3, base=False, cap=False)
        for f in range(nf):
            zf = z0 + f * FL
            for xc in (0.12 * L, 0.29 * L, 0.43 * L, 0.57 * L, 0.71 * L, 0.88 * L):
                R.window(S, sc, o, u, n, xc - 0.65, xc + 0.65, zf + 0.9, zf + 3.0, glass=sc["lit"], depth=0.26, w=0.22,
                         mullion=sc["frame"], prof=R.ARCH_S)
                R.wextrude(S, A, o, u, n, shapely.box(xc - 0.5, zf + 0.12, xc + 0.5, zf + 0.42), 0.0, 0.08)
    eagle_columns(S, sc, faces(hw), z0, zs1 + 1.4, 0.55)
    z = zs1 + 1.4
    D = 0.6 * w - hw
    mould(S, A, sq(hw + 0.7), z - 2.7, R.scale(R.STRING, 0.25, 0.35))
    consoles(S, A, faces(hw + 0.7), z, D - 0.7, 2.2, step=1.9, w=0.6)
    z = mould(S, T, sq(hw), z, R.scale(R.CORNICE, D + 0.3, 1.1))
    z = mould(S, A, sq(hw + D), z, R.scale(R.STRING, 0.2, 0.3))
    b0 = 0.58 * w
    box_walls(S, Wl, -b0, b0, -b0, b0, z, z + 2.8)
    for o, u, n, L in faces(b0):
        for j in range(8):
            xc = L * (0.12 + 0.108 * j)
            R.window(S, sc, o, u, n, xc - 0.5, xc + 0.5, z + 0.8, z + 2.1, depth=0.2, w=0.18, mullion=sc["frame"], sill=False,
                     prof=R.ARCH_S)
    z = mould(S, A, sq(b0), z + 2.8, R.scale(R.CORNICE, 0.5, 0.5))
    top = eagle_crown(S, sc, w, z, 0.6 * w, 0.62 * w, 0.48 * w, 0.5 * w)
    return top, nf + 6, FL, {"crown_z": zs1, "n": nf}


def eagle_tier(S, sc, hw, z, rng):
    """One tier of the cohesive Eagle Belfry on every face: a peach and cream band, two storeys of a 3-window grid
    (cream architraves, green glazing bars, peach aprons, small segmental hoods) between cream pilasters with peach
    capitals, an arched bay framed in green marble with a round balcony, framed panels and oculi, a broad swan-neck
    pediment; returns the tier top."""
    T, A = sc["trim"], sc["accent"]
    L = 2 * hw
    aw = 0.24 * L
    h = 2 * FL + 0.9 + 0.62 * L
    for o, u, n, L_ in faces(hw):
        for f in range(2):
            zf = z + 0.9 + f * FL
            for xc in (0.34 * L, 0.5 * L, 0.66 * L):
                R.window(S, sc, o, u, n, xc - 0.8, xc + 0.8, zf + 0.75, zf + 2.95, glass=sc["lit"], depth=0.32, w=0.26,
                         mullion=sc["frame"], prof=R.ARCH_S, hood="segment" if f == 1 else None)
                R.wextrude(S, A, o, u, n, shapely.box(xc - 0.6, zf + 0.12, xc + 0.6, zf + 0.38), 0.0, 0.1)
            for x in (0.25 * L, 0.42 * L, 0.58 * L, 0.75 * L):
                R.pilaster(S, T, o, u, n, x - 0.22, x + 0.22, zf, zf + FL, 0.2, cap_key=A, base=False, cap_h=0.4)
        eagle_bay(S, sc, o, u, n, L, z + 0.9 + 2 * FL, h - 0.9 - 2 * FL, aw, glass=sc["lit"])
        R.pediment(S, A, o, u, n, 0.18 * L, 0.82 * L, z + 0.9 + 2 * FL - 0.55, "broken", depth=0.9, rise=0.9, h=0.35, band=0.55)
    mould(S, A, sq(hw), z, R.scale(R.STRING, 0.45, 0.4))
    mould(S, T, sq(hw), z + 0.4, R.scale(R.STRING, 0.3, 0.35))
    return z + h


def eagle_cohesive(S, sc, w, H, rng, nf=None):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.5, GF + 0.5 * FL, big=True, roundels=False)
    eagle_columns(S, sc, faces(hw + 0.5), Z_B + 0.9, z - 1.25, 0.5)
    z0 = z
    Ht = eagle_tier(Soup(), sc, hw, 0.0, rng)
    if nf is None:
        nf = max(2, round((H - z - (1.8 * w + 8.0)) / Ht))
    rings = []
    for i in range(nf):
        z = eagle_tier(S, sc, hw, z, rng)
        rings.append(z)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    zc = z
    eagle_columns(S, sc, faces(hw), z0, z, 0.6, rings[:-1])
    z = mould(S, A, sq(hw), z, R.scale(R.STRING, 0.5, 0.4))
    z = mould(S, T, sq(hw), z, R.scale(R.CORNICE, 0.9, 0.8))
    solid(S, Wl, sq(hw), z, z + 0.62 * w)
    for o, u, n, L in faces(hw):
        eagle_bay(S, sc, o, u, n, L, z, 0.62 * w, 0.3 * L)
    eagle_columns(S, sc, faces(hw), z, z + 0.62 * w, 0.6)
    z = mould(S, A, sq(hw), z + 0.62 * w, R.scale(R.STRING, 0.5, 0.4))
    z = mould(S, T, sq(hw), z, R.scale(R.CORNICE, 1.2, 1.0))
    ball_balustrade(S, sc, sq(hw + 0.6), z)
    top = stepped_dome(S, sc, 0.42 * w, z)
    eagle(S, sc["metal"], (0.0, 0.0), top, 0.45 * w)
    tier = [rings[-3], rings[-2]] if len(rings) > 2 else [z0, rings[0]]
    return top + 0.45 * w * 0.9, 1 + 3 * nf + 2, Ht, {"crown_z": zc, "tier": tier, "n": nf}


# ----------------------------------------------------------------------------------------------- Golden Clock Tower
def lace_dormer(S, sc, o, u, n, wd, z, h, depth, clock_r=None):
    """Tall pointed frame standing out of the dome (front plane d = 0, x in [0, wd]), filled with carved cream lace,
    or holding the clock under a lace crest; a gilt finial."""
    T = sc["trim"]
    pts = [(0.0, z), (0.0, z + 0.55 * h)] + pointed(0.0, wd, z + 0.55 * h, 0.62, (0.45 * h) / (0.61 * wd))[1:-1] + \
        [(wd, z + 0.55 * h), (wd, z)]
    poly = shapely.Polygon(pts).buffer(0)
    R.wextrude(S, sc["dome"], o, u, n, poly, -depth, 0.0)
    R.wsweep(S, T, o, u, n, pts[::-1], R.scale(R.ARCHITRAVE, 0.35, 0.3))
    inner = poly.buffer(-0.45)
    b = inner.bounds
    if clock_r:
        R.clock(S, o, u, n, wd / 2, z + 0.42 * h, clock_r, T)
        lace = R.filigree(b[0], b[2], z + 0.42 * h + clock_r + 0.4, b[3]).intersection(inner)
        for sg in (-1, 1):
            lace = lace.union(R.scroll(wd / 2 + sg * (clock_r + 0.35), z + 0.2 * h, 0.12 * wd, 0.05 * wd, sgn=-sg, a0=np.pi / 2,
                                       turns=1.1).intersection(inner))
    else:
        lace = R.filigree(b[0], b[2], b[1], b[3]).intersection(inner)
        lace = lace.union(shapely.Polygon(inner.exterior).difference(inner.buffer(-0.25)))
    R.wextrude(S, sc["orn"], o, u, n, lace, 0.0, 0.2)
    p = wp(o, u, n, wd / 2, 0.0, 0.0)
    R.onion(S, sc["metal"], T, p[0], p[1], float(max(q[1] for q in pts)) - 0.1, 0.35)


def dome_oculus(S, sc, rd, rise, z0, theta, frac, rx, rz, k=18):
    """Oval window on the dome surface in a moulded cream frame."""
    phi = np.arcsin(frac)
    P = np.array([rd * np.cos(phi) * np.cos(theta), rd * np.cos(phi) * np.sin(theta), z0 + rise * rd * frac])
    N = np.array([np.cos(phi) * np.cos(theta) / rd, np.cos(phi) * np.sin(theta) / rd, np.sin(phi) / (rise * rd)])
    N /= np.linalg.norm(N)
    T1 = np.array([-np.sin(theta), np.cos(theta), 0.0])
    T2 = np.cross(N, T1)
    T2 *= np.sign(T2[2]) or 1.0
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pts = np.array([P + 0.05 * N + T1 * rx * np.cos(t) + T2 * rz * np.sin(t) for t in a])
    R.add_oriented(S, sc["teal"], [[P + 0.08 * N, pts[i], pts[(i + 1) % k]] for i in range(k)], N)
    R.sweep3d(S, sc["trim"], np.r_[pts, pts[:1]], np.tile(N, (k + 1, 1)), R.scale(R.RIB, 0.35, 0.4))


def lantern_gold(S, sc, r, z):
    """Colonnaded lantern: base ring with a balustrade, eight cream columns round a lit glass core, cornice, a gilt
    ribbed cupola, a gilt onion finial and spike; returns the top."""
    T = sc["trim"]
    z = mould(S, T, circle(r + 0.5, 24), z, R.scale(R.BASE, 0.35, 0.6))
    for o, u, n, L in poly_faces(circle(r + 0.4, 12)):
        R.baluster_run(S, sc["rail3d"], o, u, n, 0.05, L - 0.05, z, 0.7, -0.15, rail_key=T, posts=False, pitch=0.24, k=5)
    cylinder(S, sc["glass"], (0, 0), 0.62 * r, 0.62 * r, z, z + 1.3 * r, k=12)
    for x, y in circle(0.85 * r, 8):
        R.column(S, T, (x, y), max(0.14, 0.09 * r), z, z + 1.3 * r, order="corinthian", k=8)
    z = mould(S, T, circle(r, 24), z + 1.3 * r, R.scale(R.CORNICE, 0.4, 0.55))
    top = R.ribbed_dome(S, sc["metal"], sc["metal"], 0.95 * r, z, 1.0, ribs=8, rib=(0.15, 0.1), k=24, m=8)
    R.onion(S, sc["metal"], sc["metal"], 0.0, 0.0, top - 0.1, 0.45 * r)
    return top + 0.45 * r * 2.8


def dome_lace(S, sc, rd, rise, zd, theta, wd, h, clock_r=None):
    """Carved cream lace lying on the dome between two ribs (a pointed frame filled with lace), or a clock in a
    moulded cartouche under a lace crest; built flat, then wrapped onto the dome."""
    T = sc["trim"]
    L = Soup()
    o, u, n = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0])
    x0, x1 = -wd / 2, wd / 2
    pts = [(x0, zd + 0.4), (x0, zd + 0.5 * h)] + pointed(x0, x1, zd + 0.5 * h, 0.62, (0.5 * h) / (0.61 * wd))[1:-1] + \
        [(x1, zd + 0.5 * h), (x1, zd + 0.4)]
    poly = shapely.Polygon(pts).buffer(0)
    R.wsweep(L, T, o, u, n, pts[::-1], R.scale(R.ARCHITRAVE, 0.3, 0.28))
    inner = poly.buffer(-0.35)
    if clock_r:
        zc = zd + 0.34 * h
        R.clock(L, o, u, n, 0.0, zc, clock_r, T, d=0.12)
        crest = inner.intersection(shapely.box(-1e3, zc + clock_r + 0.55, 1e3, 1e3))
        if not crest.is_empty:
            b = crest.bounds
            R.wextrude(L, sc["orn"], o, u, n, R.filigree(b[0], b[2], b[1], b[3]).intersection(crest), 0.0, 0.2)
        for sg in (-1, 1):
            g = R.scroll(sg * (clock_r + 0.45), zc - 0.2 * clock_r, 0.32 * clock_r, 0.11 * clock_r, sgn=-sg, a0=np.pi / 2, turns=1.2,
                         tail=(sg * 0.3 * clock_r, zc - 1.25 * clock_r))
            R.wextrude(L, T, o, u, n, g.intersection(inner), 0.0, 0.3)
    else:
        b = inner.bounds
        lace = R.filigree(b[0], b[2], b[1], b[3]).intersection(inner)
        R.wextrude(L, sc["orn"], o, u, n, lace, 0.0, 0.2)
        R.oculus(L, T, sc["teal"], o, u, n, 0.0, zd + 0.22 * h, 0.16 * wd, 0.24 * wd, w=0.22, depth=0.25, lugs=False)
    R.wrap_dome(S, L, rd, rise, zd, theta, lift=0.04)
    top = max(q[1] for q in pts)
    zz = np.clip((top - zd) / (rise * rd), 0.0, 0.99)
    rr = rd * np.sqrt(1 - zz ** 2) + 0.1
    R.onion(S, sc["metal"], T, rr * np.cos(theta), rr * np.sin(theta), top - 0.3, 0.32)



def golden_dome(S, sc, rd, z, rise, clocks):
    """Orange ribbed dome: carved lace between the ribs over each face (over the front a clock in a lace cartouche
    when ``clocks``), teal oculi in moulded frames between, the colonnaded lantern; returns the top."""
    zd, top = drum_dome(S, sc, rd, z, rise, ribs=12, rib=(0.045 * rd + 0.35, 0.35), wall=sc["trim"])
    for i, (o, u, n, L) in enumerate(faces(rd)):
        th = np.arctan2(n[1], n[0])
        if clocks and i == 0:
            dome_lace(S, sc, rd, rise, zd, th, 1.05 * rd, 0.92 * rd * rise, clock_r=0.36 * rd)
        else:
            dome_lace(S, sc, rd, rise, zd, th, 0.42 * rd, 0.82 * rd * rise)
        for dt in (-0.52, 0.52):
            dome_oculus(S, sc, rd, rise, zd, th + dt, 0.36, 0.07 * rd + 0.15, 0.11 * rd + 0.2)
    return lantern_gold(S, sc, 0.28 * rd, top - 0.3)


def golden_belfry(S, sc, b, z, h):
    """Belfry of three arches a face (the centre one taller): gilt bells on yokes against a dark interior, moulded
    archivolts, cream Corinthian columns, balustrades, corner piers, a broken pediment with a cartouche over the
    centre; returns its top."""
    T = sc["trim"]
    box_walls(S, T, -b, b, -b, b, z, z + h)
    for o, u, n, L in faces(b):
        for a0, a1, f in ((0.14 * L, 0.32 * L, 0.84), (0.38 * L, 0.62 * L, 0.94), (0.68 * L, 0.86 * L, 0.84)):
            zb, zt = z + 1.4, z + f * h - 0.6
            r = (a1 - a0) / 2
            R.wextrude(S, sc["dark"], o, u, n, arch_poly(a0, a1, zb, zt - r), -0.6, 0.01)
            R.wsweep(S, T, o, u, n, arch_outline(a0, a1, zb, zt - r), R.scale(R.ARCHITRAVE, 0.4, 0.36))
            cb = wp(o, u, n, (a0 + a1) / 2, -0.35, 0.0)
            R.bell(S, sc["metal"], (cb[0], cb[1]), zt - r + 0.2, min(0.34 * (a1 - a0), 1.7))
            beam(S, "wood:madera_oscura", wp(o, u, n, a0 + 0.1, -0.35, zt - r + 0.35), wp(o, u, n, a1 - 0.1, -0.35, zt - r + 0.35),
                 0.22, 0.28)
            balcony(S, sc, o, u, n, a0 - 0.2, a1 + 0.2, zb, D=0.6, h=0.9, real=True)
        for x in (0.35 * L, 0.65 * L):
            p = wp(o, u, n, x, 0.35, 0.0)
            R.column(S, T, (p[0], p[1]), 0.3, z + 0.9, z + h - 1.0, order="corinthian", k=10)
        R.pediment(S, T, o, u, n, 0.3 * L, 0.7 * L, z + h - 0.2, "broken", depth=0.7, rise=0.12 * L, h=0.3)
        cartouche(S, T, o, u, n, 0.5 * L, z + h + 0.9, 1.6, 1.8, d=0.35)
    piers(S, T, faces(b), 0.1, z, z + h - 0.9, 0.7, cap=True)
    return mould(S, T, sq(b), z + h - 0.9, R.scale(R.CORNICE, 0.9, 0.9))


def golden_thin(S, sc, w, H, rng, nf=None):
    T, A, Wl = sc["trim"], sc["accent"], sc["wall"]
    hw, P = w / 2, 0.62 * w
    z = podium(S, sc, P, GF + FL)
    q = 0.56 * w
    podium_roof(S, sc, P, z, q)
    z = pedestal(S, sc, q, z)
    terrace(S, shapely.box(-q - 0.6, -q - 0.6, q + 0.6, q + 0.6), shapely.box(-hw, -hw, hw, hw), z + 0.02)
    for sx, sy in CORNERS:
        cypress(S, sx * (q - 0.3), sy * (q - 0.3), z, 2.6)
        R.onion(S, sc["metal"], T, sx * (q + 0.15), sy * (q + 0.15), z, 0.5)
    if nf is None:
        nf = max(8, round((H - z - (2.6 * w + 10.0)) / FL))
    z0, zs1 = z, z + nf * FL
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, zs1 + 2.0)
    for o, u, n, L in faces(hw):
        for x in (0.355 * L, 0.645 * L):
            R.pilaster(S, T, o, u, n, x - 0.5, x + 0.5, z0, zs1 + 2.0, 0.45, base=False, cap=False)
            R.wextrude(S, Wl, o, u, n, shapely.box(x - 0.18, z0 + 0.5, x + 0.18, zs1 + 1.6), 0.45, 0.5)
        for xc, bw in ((0.215 * L, 0.2 * L), (0.5 * L, 0.21 * L), (0.785 * L, 0.2 * L)):
            gw = 0.62 * bw
            for f in range(nf):
                zf = z0 + f * FL
                R.window(S, sc, o, u, n, xc - gw / 2, xc + gw / 2, zf + 0.75, zf + 3.25, depth=0.3, w=0.26, mullion=sc["frame"],
                         sill=False, prof=R.ARCH_S)
                R.wextrude(S, T, o, u, n, shapely.box(xc - gw / 2 + 0.1, zf + 0.12, xc + gw / 2 - 0.1, zf + 0.32), 0.0, 0.12)
    piers(S, T, faces(hw), 0.085, z0, zs1 + 2.0, 0.65)
    for o, u, n, L in faces(hw):                                   # gold inset lines on the corner piers
        for xa in (0.025 * L, L - 0.06 * L):
            R.wextrude(S, Wl, o, u, n, shapely.box(xa, z0 + 0.6, xa + 0.035 * L, zs1 + 1.5), 0.65, 0.7)
    # crown: consoles, a cornice with a balustrade, cypresses and urns; the clock stage; the belfry; the dome
    z = zs1 + 2.0
    D = 0.68 * w - hw
    mould(S, T, sq(hw + 0.65), z - 2.6, R.scale(R.STRING, 0.25, 0.35))
    consoles(S, T, faces(hw + 0.65), z, D - 0.65, 2.3, step=1.8, w=0.6)
    z = mould(S, T, sq(hw), z, R.scale(R.CORNICE, D + 0.3, 1.3))
    edge = hw + D + 0.1
    for o, u, n, L in faces(edge):
        R.baluster_run(S, sc["rail3d"], o, u, n, 0.5, L - 0.5, z, 1.05, -0.3, rail_key=T)
    k1, h1 = 0.56 * w, 0.5 * w
    for sx, sy in CORNERS:
        cypress(S, sx * (edge - 1.6), sy * (edge - 1.6), z, 3.4)
        R.urn(S, sc["metal"], (sx * (edge - 0.3), sy * (edge - 0.3)), z + 1.1, 1.2)
    box_walls(S, A, -k1, k1, -k1, k1, z, z + h1)
    mould(S, T, sq(k1), z, R.scale(R.BASE, 0.3, 0.7))
    for o, u, n, L in faces(k1):
        r = 0.18 * L
        xc, zc = 0.5 * L, z + 0.52 * h1
        R.clock(S, o, u, n, xc, zc, r, T, d=0.25)
        for sg in (-1, 1):                                         # carved C-scrolls flanking the clock
            g = R.scroll(xc + sg * (r + 1.0), zc + 0.35 * r, 0.42 * r, 0.12 * r, sgn=-sg, a0=np.pi / 2, turns=1.2,
                         tail=(xc + sg * (r + 0.3), zc - 0.9 * r))
            g = g.union(R.scroll(xc + sg * (r + 0.9), zc - 0.75 * r, 0.26 * r, 0.1 * r, sgn=sg, a0=-np.pi / 2, turns=1.0))
            g = g.union(R.leaf(xc + sg * (r + 0.3), zc + 0.9 * r, xc + sg * (r + 1.6), zc + 1.3 * r, 0.12 * r))
            R.wextrude(S, T, o, u, n, g, 0.0, 0.35)
        for xa, xb in ((0.03 * L, 0.17 * L), (0.83 * L, 0.97 * L)):
            R.wextrude(S, Wl, o, u, n, shapely.box(xa + 0.3, z + 1.2, xb - 0.3, z + h1 - 1.6), 0.8, 0.85)
            R.wextrude(S, T, o, u, n, R.filigree(xa + 0.45, xb - 0.45, z + 0.45 * h1, z + h1 - 1.9), 0.85, 1.0)
        balcony(S, sc, o, u, n, 0.2 * L, 0.8 * L, z + 0.8, D=0.7, h=0.9, real=True)
    piers(S, T, faces(k1), 0.17, z, z + h1 - 0.9, 0.8, cap=True)
    z = mould(S, T, sq(k1), z + h1 - 0.9, R.scale(R.CORNICE, 1.0, 0.9))
    for sx, sy in CORNERS:
        R.onion(S, sc["metal"], T, sx * (k1 - 0.4), sy * (k1 - 0.4), z, 0.75)
    k2 = 0.52 * w
    z = golden_belfry(S, sc, k2, z, 0.46 * w)
    for sx, sy in CORNERS:
        R.onion(S, sc["metal"], T, sx * (k2 - 0.3), sy * (k2 - 0.3), z, 0.8)
    rd = 0.45 * w
    terrace(S, shapely.box(-k2 - 0.9, -k2 - 0.9, k2 + 0.9, k2 + 0.9), shapely.Polygon(circle(rd + 0.6, 32)), z + 0.02)
    return golden_dome(S, sc, rd, z, 1.08, False), nf + 6, FL, {"crown_z": zs1, "n": nf}


def golden_tier(S, sc, hw, z, rng):
    """One tier of the cohesive Golden Clock Tower on every face: three storeys x three bays of lit windows (teal
    mullions, cream architraves) with balconies, cream pilasters, a deep moulded segmental hood with scroll ends, a
    scallop shell and carved lace (one panel per bay), framed teal ovals on the side strips, clustered cream corner
    columns; returns the tier top."""
    T = sc["trim"]
    L = 2 * hw
    xa0, xa1 = 0.17 * L, 0.83 * L
    bay = (xa1 - xa0) / 3
    zh = z + 0.4 + 3 * FL + 0.35
    rise = 0.09 * L
    top = zh + rise + 0.7 + 0.1 * L + 0.6
    for o, u, n, L_ in faces(hw):
        for j in range(3):
            xc = xa0 + (j + 0.5) * bay
            gw = 0.6 * bay
            for f in range(3):
                zf = z + 0.4 + f * FL
                R.window(S, sc, o, u, n, xc - gw / 2, xc + gw / 2, zf + 0.35, zf + 3.0, depth=0.32, w=0.26, mullion=sc["frame"],
                         sill=False, prof=R.ARCH_S)
                balcony(S, sc, o, u, n, xc - gw / 2 - 0.5, xc + gw / 2 + 0.5, zf + 0.3, D=0.65, h=0.9, rng=rng, plant=0.5)
        for j in range(4):
            x = xa0 + j * bay
            R.pilaster(S, T, o, u, n, x - 0.28, x + 0.28, z + 0.4, zh - 0.2, 0.32, cap_key=T, cap_h=0.5)
        arc = segment(xa0 - 0.3, xa1 + 0.3, zh, rise, 18)
        poly = shapely.Polygon(arc).buffer(0)
        for j in range(3):                                         # carved lace, one panel per bay (never stretched)
            cell = poly.intersection(shapely.box(xa0 + j * bay + 0.25, zh + 0.1, xa0 + (j + 1) * bay - 0.25, zh + rise)).buffer(-0.12)
            if not cell.is_empty:
                b = cell.bounds
                R.wextrude(S, sc["orn"], o, u, n, R.filigree(b[0], b[2], b[1], b[3]).intersection(cell), 0.0, 0.18)
        R.wsweep(S, T, o, u, n, arc, R.scale(R.ARCHITRAVE, 0.65, 0.6))
        R.wsweep(S, T, o, u, n, arc[1:-1], R.scale(R.STRING, 0.4, 0.22, b0=0.6))
        for x, sg in ((xa0 - 0.3, 1), (xa1 + 0.3, -1)):
            R.wextrude(S, T, o, u, n, R.scroll(x, zh - 0.15, 0.5, 0.18, sgn=sg, a0=np.pi / 2, turns=1.2), 0.0, 0.6)
        R.shell(S, T, o, u, n, L_ / 2, zh + rise + 0.55, 0.1 * L_, depth=0.55)
        for xc in (0.13 * L_, 0.87 * L_):
            R.oculus(S, T, sc["teal"], o, u, n, xc, zh, 0.42, 0.62, w=0.24, depth=0.32, lugs=False)
            R.oculus(S, T, sc["teal"], o, u, n, xc, z + 0.4 + 1.2 * FL, 0.32, 0.5, w=0.22, depth=0.3, lugs=False)
    for o, u, n, L_ in faces(hw):
        for x in (0.04 * L_, 0.96 * L_):
            p = wp(o, u, n, x, 0.62, 0.0)
            R.column(S, T, (p[0], p[1]), 0.34, z, top - 0.5, order="corinthian", k=10)
    piers(S, T, faces(hw), 0.085, z, top - 0.5, 0.35)
    mould(S, T, sq(hw), z, R.scale(R.STRING, 0.35, 0.4))
    mould(S, T, sq(hw + 0.35), top - 0.5, R.scale(R.STRING, 0.45, 0.5))
    return top


def golden_cohesive(S, sc, w, H, rng, nf=None):
    T, Wl = sc["trim"], sc["wall"]
    hw = w / 2
    z = podium(S, sc, hw + 0.5, GF + 0.5 * FL, roundels=False)
    z0 = z
    Ht = golden_tier(Soup(), sc, hw, 0.0, np.random.default_rng(0))
    if nf is None:
        nf = max(2, round((H - z - (2.3 * w + 10.0)) / Ht))
    for i in range(nf):
        z = golden_tier(S, sc, hw, z, rng)
    box_walls(S, Wl, -hw, hw, -hw, hw, z0, z)
    zc = z
    z = golden_belfry(S, sc, hw, z, 0.42 * w)
    for sx, sy in CORNERS:
        R.onion(S, sc["metal"], T, sx * (hw - 0.3), sy * (hw - 0.3), z, 0.85)
    rd = 0.47 * w
    terrace(S, shapely.box(-hw - 0.9, -hw - 0.9, hw + 0.9, hw + 0.9), shapely.Polygon(circle(rd + 0.6, 32)), z + 0.02)
    top = golden_dome(S, sc, rd, z, 1.55, True)
    return top, 1 + 3 * nf + 1, Ht, {"crown_z": zc, "tier": [zc - 2 * Ht, zc - Ht], "n": nf}


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
    from WIDTH); colours: role -> material key overrides (see SCHEMES)."""
    W = float(W or WIDTH[(design, form)])
    sc = scheme(design, form, **(colours or {}))
    state = rng.bit_generator.state

    def build(nf=None):
        rng.bit_generator.state = state
        S = Soup()
        _, floors, unit, marks = BUILD[(design, form)](S, sc, W, float(H), rng, nf)
        top = max(float(V[:, 2].max()) for lst in S.parts.values() for V, _, _ in lst)
        return S, top, floors, unit, marks

    S, top, floors, unit, marks = build()                          # calibrate: whole storeys or tiers
    shift = round((float(H) - top) / unit)
    if shift:
        alt = build(max(2, marks["n"] + shift))
        if abs(alt[1] - H) < abs(top - H):
            S, top, floors, unit, marks = alt
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    Rm = np.array([[ax[0], -ax[1]], [ax[1], ax[0]]])
    out = Soup()
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for key, lst in S.parts.items():
        for V, F, UV in lst:
            lo, hi = np.minimum(lo, V[:, :2].min(0)), np.maximum(hi, V[:, :2].max(0))
            V2 = V.copy()
            V2[:, :2] = V[:, :2] @ Rm.T + np.asarray(center, float)
            out.parts.setdefault(key, []).append((V2, F, UV))
    marks = {k: v for k, v in marks.items() if k != "n"}
    return out, {"design": design, "form": form, "height_m": float(top), "floors": int(floors), "shaft_m": W,
                 "half_extent_m": float(max(-lo.min(), hi.max())), "colours": sc, "triangles": out.triangles(), **marks}
