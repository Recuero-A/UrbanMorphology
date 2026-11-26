"""Parametric Spanish-colonial church (DESIGN.md §5): the domed parish church of IMG_3376 and the Cartagena / Caribbean
colonial type (San Pedro Claver, Santo Domingo, Popayán, Antigua, Old San Juan).

Plan (local frame: X from the facade plane into the church, Y to the left, Z up; built in this frame, then rotated
onto the parcel): a nave with a tile gable roof; a baroque facade rising above the roof (pilasters, entablature,
curved shoulders, pediment, arched portal with columns, oculus or choir window, niche, pinnacles, cross); one or two
bell towers in front (shaft, cornice, one or two belfry stages with arched openings, cupola or tile pyramid, lantern,
cross); a crossing with a drum, ribbed dome and lantern; optional transepts and a semicircular apse.
Everything scales with the parcel; ornament keeps human sizes (doors, steps, mouldings).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.latin.kit import (Soup, Z_B, beam, cross, cylinder, dome, hip_shell, iron_balcony, sloped, wbox,
                                         warch, warch_frame, wdisc, wp, wquad, wring)
from urbanmorph.city3d.mesh import prism, surface


def frame(n):
    n = np.asarray(n, float)
    return np.array([-n[1], n[0]]), n


def box_walls(S, key, x0, x1, y0, y1, z0, z1):
    V, F, part = prism(shapely.orient_polygons(shapely.box(x0, y0, x1, y1)), z0, z1).arrays()
    S.add(key, V, F[part != 0])


def walls4(x0, x1, y0, y1):
    """(o, u, n, length) of the four walls of an axis-aligned box, CCW."""
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    out = []
    for i in range(4):
        a, e = np.array(c[i], float), np.array(c[(i + 1) % 4], float)
        L = np.linalg.norm(e - a)
        u = (e - a) / L
        out.append((a, u, np.array([u[1], -u[0]]), L))
    return out


def pinnacle(S, key, x, y, z, s=0.35):
    box_walls(S, key, x - s, x + s, y - s, y + s, z, z + 2 * s)
    cylinder(S, key, (x, y), s * 0.8, 0.02, z + 2 * s, z + 5 * s, k=4)


def tower(S, st, x0, y0, T, h1, rng):
    """Bell tower on the square [x0, x0 + T] x [y0, y0 + T] from the ground to its cross."""
    wall, trim = st["wall"], "trimc:blanco"
    box_walls(S, wall, x0, x0 + T, y0, y0 + T, Z_B, h1)
    for o, u, n, L in walls4(x0, x0 + T, y0, y0 + T):
        wbox(S, trim, o, u, n, 0.0, 0.4, Z_B, h1, 0.0, 0.08)
        wbox(S, trim, o, u, n, L - 0.4, L, Z_B, h1, 0.0, 0.08)
        for zc in (h1 * 0.45, h1 * 0.75):
            warch(S, "void", o, u, n, L / 2 - 0.3, L / 2 + 0.3, zc - 0.7, zc + 0.4, 0.03, k=6)
    z = h1
    box_walls(S, trim, x0 - 0.25, x0 + T + 0.25, y0 - 0.25, y0 + T + 0.25, z, z + 0.45)
    z += 0.45
    side = T
    stages = 2 if rng.uniform() < 0.5 else 1
    for k in range(stages):
        inset = (T - side) / 2
        a0, b0 = x0 + inset, y0 + inset
        h = 1.25 * side if k == 0 else 1.0 * side
        box_walls(S, wall, a0, a0 + side, b0, b0 + side, z, z + h)
        ow = 0.42 * side
        for o, u, n, L in walls4(a0, a0 + side, b0, b0 + side):
            zs = z + h - 0.55 - ow / 2
            warch(S, "void", o, u, n, L / 2 - ow / 2, L / 2 + ow / 2, z + 0.6, zs, 0.03, k=8)
            warch_frame(S, trim, o, u, n, L / 2 - ow / 2, L / 2 + ow / 2, z + 0.6, zs, 0.18, 0.0, 0.09, k=8)
            wbox(S, trim, o, u, n, L / 2 - ow / 2 - 0.25, L / 2 + ow / 2 + 0.25, z + 0.45, z + 0.6, 0.0, 0.15)
            wbox(S, trim, o, u, n, 0.0, 0.32, z, z + h, 0.0, 0.08)
            wbox(S, trim, o, u, n, L - 0.32, L, z, z + h, 0.0, 0.08)
        z += h
        box_walls(S, trim, a0 - 0.2, a0 + side + 0.2, b0 - 0.2, b0 + side + 0.2, z, z + 0.35)
        z += 0.35
        for px, py in ((a0, b0), (a0 + side, b0), (a0 + side, b0 + side), (a0, b0 + side)):
            pinnacle(S, trim, px, py, z, 0.18 + 0.03 * side)
        side *= 0.72
    c = (x0 + T / 2, y0 + T / 2)
    if st["tower_cap"] == "pyramid":
        r = side / 2 + 0.3
        sloped(S, st["roof"], [[c[0] - r, c[1] - r, z], [c[0] + r, c[1] - r, z], [c[0] + r, c[1] + r, z],
                               [c[0] - r, c[1] + r, z], [c[0], c[1], z + 1.4 * r]], [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        top = z + 1.4 * r
    else:
        r = side / 2
        cylinder(S, wall, c, r, r, z, z + 0.7 * r, k=8)
        cylinder(S, trim, c, r + 0.12, r + 0.12, z + 0.7 * r, z + 0.7 * r + 0.2, k=8, cap=True)
        top = dome(S, st["dome"], c, r, z + 0.7 * r + 0.2, k=12, m=5, rise=1.15 if st["tower_cap"] == "bulb" else 0.95,
                   ribs_key=trim if st["ribs"] else None, ribs=8, rib_w=0.08)
        cylinder(S, trim, c, 0.28 * r, 0.28 * r, top - 0.05, top + 0.55 * r, k=8, cap=True)
        top += 0.55 * r
    cross(S, "iron", c, top, 1.2 + 0.1 * T)
    return top


def _faces(c, r, k=8, rot=np.pi / 8):
    """(o, u, n, length) of the faces of a regular k-gon (circumradius r, CCW, first vertex at angle rot)."""
    a = rot + 2 * np.pi * np.arange(k + 1) / k
    P = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    out = []
    for i in range(k):
        L = float(np.linalg.norm(P[i + 1] - P[i]))
        u = (P[i + 1] - P[i]) / L
        out.append((P[i], u, np.array([u[1], -u[0]]), L))
    return out


def ngon(c, r, k=8, rot=np.pi / 8):
    a = rot + 2 * np.pi * np.arange(k) / k
    return shapely.Polygon(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def ngon_walls(S, key, c, r, z0, z1, k=8, rot=np.pi / 8):
    for o, u, n, L in _faces(c, r, k, rot):
        wquad(S, key, o, u, n, 0.0, L, z0, z1, 0.0)


def terrace(S, outer, inner, z):
    """Flat terrace (clay pavers) between two footprints at height z."""
    V, F = surface(outer.difference(inner), z)
    if len(F):
        S.add("roof_flat", V, F)


def balustrade(S, faces, z, h=1.1):
    """Cut-out balusters with a coping, just inside the given outer faces."""
    for o, u, n, L in faces:
        wquad(S, "baluster:blanco", o, u, n, 0.0, L, z, z + h, -0.12, uv="panel")
        wbox(S, "trimc:blanco", o, u, n, 0.0, L, z + h, z + h + 0.14, -0.3, 0.0)


def lathe(S, key, c, prof, z0=0.0, k=10):
    """Surface of revolution about the vertical through c: prof = [(radius, height)] from bottom to top."""
    for (ra, za), (rb, zb) in zip(prof[:-1], prof[1:]):
        if zb > za:
            cylinder(S, key, c, max(ra, 1e-3), max(rb, 1e-3), z0 + za, z0 + zb, k=k)


URN = [(0.55, 0.0), (0.55, 0.22), (0.3, 0.3), (0.62, 0.55), (0.7, 0.75), (0.5, 0.95), (0.2, 1.05), (0.28, 1.2), (0.0, 1.45)]
BELL = [(1.0, 0.0), (0.93, 0.1), (0.72, 0.28), (0.62, 0.5), (0.58, 0.75), (0.5, 0.95), (0.0, 1.05)]


def urn(S, key, x, y, z, s=0.5):
    """Baroque urn / ball finial on a small pedestal (corners of balustrades and stages)."""
    box_walls(S, key, x - 0.6 * s, x + 0.6 * s, y - 0.6 * s, y + 0.6 * s, z, z + 0.5 * s)
    lathe(S, key, (x, y), [(r * s, h * s) for r, h in URN], z + 0.5 * s, k=8)


def bell(S, c, z_top, r):
    """Bronze bell hanging from a wooden yoke (mouth radius r, top at z_top)."""
    h = 1.1 * r
    lathe(S, "bronze", c, [(rr * r, hh * h) for rr, hh in BELL], z_top - h, k=10)


def volute(S, key, a, b, w=0.45, k=6):
    """Baroque scroll from a (on the lower terrace) curving up to b (against the stage above)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, np.pi / 2, k + 1)
    P = [np.r_[a[:2] + (b[:2] - a[:2]) * np.sin(s), a[2] + (b[2] - a[2]) * (1 - np.cos(s))] for s in t]
    for p, q in zip(P[:-1], P[1:]):
        beam(S, key, p, q, w, w)


def belfry(S, wall, trim, c, s, z, h, n_a):
    """Square belfry (cuerpo de campanas) of side s: corner and paired pilasters, n_a arched openings per face, each
    with a bell, an entablature."""
    x0, y0 = c[0] - s / 2, c[1] - s / 2
    box_walls(S, wall, x0, x0 + s, y0, y0 + s, z, z + h)
    pw = 0.09 * s
    gap = 0.11 * s
    aw = (s - 2 * pw - (n_a + 1) * gap) / n_a
    zs = z + h - 1.0 - aw / 2 - 0.3                                   # arches at most 2.3 x as tall as wide
    za = max(z + 0.12 * h, zs + aw / 2 - 2.3 * aw)
    for o, u, n, L in walls4(x0, x0 + s, y0, y0 + s):
        wbox(S, trim, o, u, n, 0.0, pw, z, z + h, 0.0, 0.2)
        wbox(S, trim, o, u, n, L - pw, L, z, z + h, 0.0, 0.2)
        wbox(S, trim, o, u, n, -0.15, L + 0.15, z + h - 0.8, z + h - 0.45, 0.0, 0.3)
        wbox(S, trim, o, u, n, -0.3, L + 0.3, z + h - 0.45, z + h, 0.0, 0.45)
        wbox(S, trim, o, u, n, pw, L - pw, za - 0.45, za - 0.1, 0.0, 0.3)
        for j in range(n_a):
            a0 = pw + gap + j * (aw + gap)
            warch(S, "void", o, u, n, a0, a0 + aw, za, zs, 0.03, k=10)
            warch_frame(S, trim, o, u, n, a0, a0 + aw, za, zs, 0.24, 0.0, 0.14, k=10)
            cb = wp(o, u, n, a0 + aw / 2, 0.25, 0.0)
            rb = min(0.36 * aw, 2.2)
            bell(S, (cb[0], cb[1]), zs + 0.1, rb)
            beam(S, "wood:madera_oscura", wp(o, u, n, a0 + 0.05, 0.25, zs + 0.2), wp(o, u, n, a0 + aw - 0.05, 0.25, zs + 0.2),
                 0.25, 0.3)
            for xp in (a0 - gap / 2 - 0.22, a0 + aw + gap / 2 + 0.22) if n_a > 1 else ():
                if pw < xp < L - pw:
                    wbox(S, trim, o, u, n, xp - 0.18, xp + 0.18, z, z + h - 0.8, 0.0, 0.16)


def giant_tower(S, st, x0, y0, T, H, rng):
    """Giant Spanish-colonial bell tower on the square [x0, x0 + T] x [y0, y0 + T], about H m to the top of its cross
    (DESIGN.md §5.1). Square in plan to the cupola, as in Cartagena, Havana, Lima and Puebla, and in the user's red
    reference tower:
    * a plain shaft (stone plinth, cream quoins, string courses, a few small arched windows and niches);
    * a clock stage (a clock on every face, pilasters);
    * two belfries, slightly set back, with paired pilasters and a bronze bell in every arch; a balustrade with corner
      urns at the foot of the first, baroque scrolls and urns at the foot of the second;
    * an attic, a drum with oculi, a ribbed cupola (orange by default), a lantern, a gilt ball and an iron cross.
    Returns (top, plants): plants = [(prototype, x, y, z, scale)] for the balcony planters of the first belfry."""
    wall, trim = st["wall"], st.get("trim", "trimc:crema")
    c = (x0 + T / 2, y0 + T / 2)
    s1, s2 = 0.86 * T, 0.7 * T
    rd = 0.44 * s2
    cross_h = 2.0 + 0.012 * H
    nominal = [0.75 * T, 1.15 * s1, 1.1 * s2]                       # clock stage, belfry 1, belfry 2
    fixed = (4 * 0.7 + 0.1 * s2 + 0.35 * rd + 0.4 + 1.0 * rd + 0.45 * rd + 0.9 * (0.3 * rd + 0.12) + 0.6 + cross_h)
    f = float(np.clip((0.5 * H - fixed) / sum(nominal), 0.75, 1.35))  # stages ~ half the height, the shaft the rest
    hc, h1, h2 = (f * x for x in nominal)
    hs = max(1.5 * T, H - fixed - hc - h1 - h2)
    plants = []
    # ---- shaft
    box_walls(S, wall, x0, x0 + T, y0, y0 + T, Z_B, hs)
    qw = 0.08 * T
    ph = 1.4 * T
    n_p = max(1, int(round((hs - 2.0) / ph)))
    zb = np.linspace(2.0, hs, n_p + 1)
    for o, u, n, L in walls4(x0, x0 + T, y0, y0 + T):
        wbox(S, "stone", o, u, n, -0.1, L + 0.1, Z_B, 2.0, 0.0, 0.2)
        wbox(S, trim, o, u, n, 0.0, qw, Z_B, hs, 0.0, 0.16)
        wbox(S, trim, o, u, n, L - qw, L, Z_B, hs, 0.0, 0.16)
        for z in zb[1:-1]:
            wbox(S, trim, o, u, n, qw, L - qw, z - 0.3, z + 0.3, 0.0, 0.22)
        for k, (za, zc) in enumerate(zip(zb[:-1], zb[1:])):
            hp, ww = zc - za, 0.1 * T
            if k % 2 == 0:                                            # small arched windows
                warch(S, "glass", o, u, n, L / 2 - ww / 2, L / 2 + ww / 2, za + 0.4 * hp, za + 0.7 * hp, 0.03, k=8)
                warch_frame(S, trim, o, u, n, L / 2 - ww / 2, L / 2 + ww / 2, za + 0.4 * hp, za + 0.7 * hp, 0.25, 0.0, 0.12, k=8)
            else:                                                     # a niche
                warch(S, "void", o, u, n, L / 2 - 0.7 * ww, L / 2 + 0.7 * ww, za + 0.3 * hp, za + 0.62 * hp, 0.03, k=8)
                warch_frame(S, trim, o, u, n, L / 2 - 0.7 * ww, L / 2 + 0.7 * ww, za + 0.3 * hp, za + 0.62 * hp, 0.3, 0.0, 0.15,
                            k=8, keystone=False)
                wbox(S, trim, o, u, n, L / 2 - 0.7 * ww - 0.4, L / 2 + 0.7 * ww + 0.4, za + 0.3 * hp - 0.3, za + 0.3 * hp, 0.0, 0.35)
    z = hs
    sq = lambda s_, z0, z1, key: box_walls(S, key, c[0] - s_ / 2, c[0] + s_ / 2, c[1] - s_ / 2, c[1] + s_ / 2, z0, z1)
    # ---- clock stage (same side as the shaft)
    sq(T + 0.5, z, z + 0.7, trim)
    z += 0.7
    sq(T, z, z + hc, wall)
    for o, u, n, L in walls4(c[0] - T / 2, c[0] + T / 2, c[1] - T / 2, c[1] + T / 2):
        wbox(S, trim, o, u, n, 0.0, qw, z, z + hc, 0.0, 0.16)
        wbox(S, trim, o, u, n, L - qw, L, z, z + hc, 0.0, 0.16)
        rc = min(0.3 * T, 0.36 * hc)
        zm = z + hc / 2
        wdisc(S, trim, o, u, n, L / 2, zm, rc, 0.06, k=20)
        wring(S, "gold", o, u, n, L / 2, zm, rc, rc + 0.35, 0.09, k=20)
        wring(S, trim, o, u, n, L / 2, zm, rc + 0.35, rc + 0.8, 0.12, k=20)
        beam(S, "iron", wp(o, u, n, L / 2, 0.12, zm), wp(o, u, n, L / 2, 0.12, zm + 0.75 * rc), 0.14, 0.14)
        beam(S, "iron", wp(o, u, n, L / 2, 0.12, zm), wp(o, u, n, L / 2 + 0.5 * rc, 0.12, zm), 0.14, 0.14)
    z += hc
    # ---- belfry 1 on a balustraded balcony with corner urns and planters
    co = 0.45
    sq(T + 2 * co, z, z + 0.7, trim)
    outer = shapely.box(c[0] - T / 2 - co, c[1] - T / 2 - co, c[0] + T / 2 + co, c[1] + T / 2 + co)
    z += 0.7
    terrace(S, outer, shapely.box(c[0] - s1 / 2, c[1] - s1 / 2, c[0] + s1 / 2, c[1] + s1 / 2), z)
    balustrade(S, walls4(c[0] - T / 2 - co, c[0] + T / 2 + co, c[1] - T / 2 - co, c[1] + T / 2 + co), z)
    q = T / 2 + co - 0.55
    for sx in (-1, 1):
        for sy in (-1, 1):
            urn(S, trim, c[0] + sx * q, c[1] + sy * q, z + 1.24, 0.4 + 0.02 * T)
            if (T + 2 * co - s1) / 2 >= 1.0:
                plants.append(("bougainvillea", c[0] + sx * q * 0.55, c[1] + sy * q, z, 0.55))
    belfry(S, wall, trim, c, s1, z, h1, 3 if s1 >= 13 else 2)
    z += h1
    # ---- belfry 2 with scrolls and urns at its foot
    sq(s1 + 2 * co, z, z + 0.7, trim)
    z += 0.7
    terrace(S, shapely.box(c[0] - s1 / 2 - co, c[1] - s1 / 2 - co, c[0] + s1 / 2 + co, c[1] + s1 / 2 + co),
            shapely.box(c[0] - s2 / 2, c[1] - s2 / 2, c[0] + s2 / 2, c[1] + s2 / 2), z)
    for sx in (-1, 1):
        for sy in (-1, 1):
            a = (c[0] + sx * (s1 / 2 + co - 0.4), c[1] + sy * (s1 / 2 + co - 0.4), z)
            b = (c[0] + sx * (s2 / 2 + 0.2), c[1] + sy * (s2 / 2 + 0.2), z + 0.4 * h2)
            volute(S, trim, a, b, 0.35 + 0.02 * T)
            urn(S, trim, a[0], a[1], z, 0.35 + 0.02 * T)
    belfry(S, wall, trim, c, s2, z, h2, 2 if s2 >= 9 else 1)
    z += h2
    # ---- attic, drum, cupola, lantern, ball, cross
    sq(s2 + 2 * co, z, z + 0.7, trim)
    z += 0.7
    terrace(S, shapely.box(c[0] - s2 / 2 - co, c[1] - s2 / 2 - co, c[0] + s2 / 2 + co, c[1] + s2 / 2 + co), ngon(c, rd + 0.3, 16, 0.0), z)
    for sx in (-1, 1):
        for sy in (-1, 1):
            urn(S, trim, c[0] + sx * (s2 / 2 + co - 0.45), c[1] + sy * (s2 / 2 + co - 0.45), z, 0.3 + 0.015 * T)
    sq(2 * rd + 0.6, z, z + 0.1 * s2, trim)
    z += 0.1 * s2
    cylinder(S, wall, c, rd, rd, z, z + 0.35 * rd, k=16)
    for i, (o, u, n, L) in enumerate(_faces(c, rd + 0.01, 16, 0.0)):
        if i % 4 == 0:
            wdisc(S, "glass", o, u, n, L / 2, z + 0.18 * rd, 0.1 * rd, 0.03, k=12)
            wring(S, trim, o, u, n, L / 2, z + 0.18 * rd, 0.1 * rd, 0.1 * rd + 0.25, 0.06, k=12)
    z += 0.35 * rd
    cylinder(S, trim, c, rd + 0.35, rd + 0.35, z, z + 0.4, k=16, cap=True)
    z += 0.4
    top = dome(S, st["dome"], c, rd, z, k=16, m=7, rise=1.0, ribs_key=trim, ribs=8, rib_w=0.12 + 0.02 * rd)
    rl = 0.3 * rd
    cylinder(S, trim, c, rl, rl, top - 0.1, top + 0.45 * rd, k=8, cap=True)
    for o, u, n, L in _faces(c, rl + 0.01, rot=0.0):
        warch(S, "void", o, u, n, 0.22 * L, 0.78 * L, top + 0.08 * rd, top + 0.3 * rd, 0.01, k=6)
    zl = dome(S, st["dome"], c, rl + 0.12, top + 0.45 * rd, k=8, m=3, rise=0.9)
    lathe(S, "gold", c, [(0.0, 0.0), (0.3, 0.12), (0.42, 0.3), (0.3, 0.5), (0.0, 0.6)], zl, k=8)
    zc = zl + 0.55
    cross(S, "iron", c, zc, cross_h, t=0.12 + 0.06 * cross_h)                # thick enough to read from the street
    beam(S, "iron", [c[0], c[1] - 0.3 * cross_h, zc + 0.68 * cross_h], [c[0], c[1] + 0.3 * cross_h, zc + 0.68 * cross_h],
         0.1 + 0.05 * cross_h, 0.1 + 0.05 * cross_h)
    return zc + cross_h, plants


def facade(S, st, y0, y1, H, rise, rng):
    """Facade slab on the plane X = 0 spanning Y in [y0, y1] (towers excluded), with its ornament."""
    wall, trim = st["wall"], "trimc:blanco"
    Wf = y1 - y0
    o, u, n = np.array([0.0, y1]), np.array([0.0, -1.0]), np.array([-1.0, 0.0])
    h1 = H + 0.6
    top = H + rise + 2.4
    sh = 0.22 * Wf                                       # shoulders: the upper body is the central 56 %
    t = np.linspace(0, np.pi / 2, 7)
    right = [(Wf - sh + sh * (1 - np.cos(a)) * 0.0 - sh * np.sin(a) * 0.0, 0) for a in t]
    del right
    cx = Wf / 2
    bw = Wf / 2 - sh
    curve_r = [(cx + bw + (sh * 0.9) * np.cos(a), h1 + (top - 1.6 - h1) * np.sin(a)) for a in t]      # quarter ellipse
    outline = [(0.0, Z_B), (Wf, Z_B), (Wf, h1)] + [(x, z) for x, z in curve_r[::-1][:-1]][::-1][::-1]
    outline = [(0.0, Z_B), (Wf, Z_B), (Wf, h1)]
    outline += [(cx + bw + sh * 0.9 * np.cos(a), h1 + (top - 1.6 - h1) * np.sin(a)) for a in t[1:]]
    outline += [(cx, top)]
    outline += [(cx - bw - sh * 0.9 * np.cos(a), h1 + (top - 1.6 - h1) * np.sin(a)) for a in t[::-1][:-1]]
    outline += [(0.0, h1)]
    poly = shapely.orient_polygons(shapely.Polygon(outline).buffer(0))
    V, F, _ = prism(poly, -0.75, 0.0).arrays()                     # local (x, z, d) -> world
    W_ = np.array([[o[0] + u[0] * x + n[0] * d, o[1] + u[1] * x + n[1] * d, z] for x, z, d in V])
    S.add(wall, W_, F)
    # ornament
    for xp in (0.06 * Wf, 0.27 * Wf, 0.73 * Wf, 0.94 * Wf):
        wbox(S, trim, o, u, n, xp - 0.3, xp + 0.3, Z_B, h1 - 0.6, 0.0, 0.22)
        wbox(S, trim, o, u, n, xp - 0.42, xp + 0.42, Z_B, 0.9, 0.0, 0.32)                   # pedestals
    wbox(S, trim, o, u, n, -0.1, Wf + 0.1, h1 - 0.75, h1 - 0.35, 0.0, 0.3)                # entablature
    wbox(S, trim, o, u, n, -0.2, Wf + 0.2, h1 - 0.35, h1, 0.0, 0.42)
    for xp in (cx - bw * 0.55, cx + bw * 0.55):
        wbox(S, trim, o, u, n, xp - 0.25, xp + 0.25, h1, top - 2.2, 0.0, 0.16)
    pts = [(cx + bw + sh * 0.9 * np.cos(a), h1 + (top - 1.6 - h1) * np.sin(a)) for a in t] + [(cx, top)]
    pts_l = [(cx - bw - sh * 0.9 * np.cos(a), h1 + (top - 1.6 - h1) * np.sin(a)) for a in t] + [(cx, top)]
    for P in (pts, pts_l):
        for (xa, za), (xb, zb) in zip(P[:-1], P[1:]):
            beam(S, trim, [o[0] + u[0] * xa - 0.12, o[1] + u[1] * xa, za], [o[0] + u[0] * xb - 0.12, o[1] + u[1] * xb, zb], 0.32, 0.3)
    # portal
    dw = float(np.clip(0.22 * Wf, 2.4, 3.6))
    dh = float(np.clip(0.42 * H, 4.4, 6.4))
    zs = dh - dw / 2
    warch(S, "door:madera_oscura", o, u, n, cx - dw / 2, cx + dw / 2, 0.15, zs, 0.04)
    warch_frame(S, trim, o, u, n, cx - dw / 2, cx + dw / 2, 0.15, zs, 0.4, 0.0, 0.18)
    for s in (-1, 1):
        xc = cx + s * (dw / 2 + 0.95)
        cylinder(S, trim, (o[0] - 0.45, o[1] - xc), 0.26, 0.24, 0.15, dh + 0.6, k=8)
    wbox(S, trim, o, u, n, cx - dw / 2 - 1.4, cx + dw / 2 + 1.4, dh + 0.6, dh + 1.05, 0.0, 0.7)
    V = [[o[0] - 0.7, o[1] - (cx - dw / 2 - 1.4), dh + 1.05], [o[0] - 0.7, o[1] - (cx + dw / 2 + 1.4), dh + 1.05],
         [o[0] - 0.7, o[1] - cx, dh + 2.0]]
    S.add(trim, V, [[0, 1, 2]])
    # choir window or oculus, and a niche in the upper body
    zo = (dh + 2.2 + h1 - 0.8) / 2
    if st["oculus"]:
        r = float(np.clip(0.07 * Wf, 0.7, 1.2))
        wdisc(S, "glass", o, u, n, cx, zo, r, 0.03)
        wring(S, trim, o, u, n, cx, zo, r, r + 0.35, 0.12)
    else:
        warch(S, "glass", o, u, n, cx - 0.65, cx + 0.65, zo - 1.1, zo + 0.5, 0.03, k=8)
        warch_frame(S, trim, o, u, n, cx - 0.65, cx + 0.65, zo - 1.1, zo + 0.5, 0.25, 0.0, 0.12, k=8)
    zn = (h1 + top) / 2 - 0.6
    warch(S, "void", o, u, n, cx - 0.55, cx + 0.55, zn - 1.0, zn + 0.5, 0.03, k=8)
    warch_frame(S, trim, o, u, n, cx - 0.55, cx + 0.55, zn - 1.0, zn + 0.5, 0.18, 0.0, 0.1, k=8, keystone=False)
    if st["espadana"]:                                    # a bell gable instead of towers
        for xb in (cx - 1.3, cx + 1.3):
            warch(S, "void", o, u, n, xb - 0.5, xb + 0.5, top - 3.6, top - 2.6, 0.03, k=8)
    for xp in (0.25, Wf - 0.25):
        pinnacle(S, trim, o[0] - 0.4, o[1] - xp, h1, 0.3)
    cross(S, "iron", (o[0] - 0.35, o[1] - cx), top + 0.2, 2.0)
    for k, d in enumerate((2.6, 1.8, 1.0)):                                                # steps
        wbox(S, "stone", o, u, n, cx - dw / 2 - 2.2 + 0.3 * k, cx + dw / 2 + 2.2 - 0.3 * k, Z_B, 0.15 * (k + 1), 0.0, d)
    return top


def side_walls(S, st, x0, x1, y0, y1, H):
    trim = f"trimc:blanco"
    for o, u, n, L in walls4(x0, x1, y0, y1):
        if abs(n[1]) < 0.5:                                       # front / back walls: cornice only
            wbox(S, trim, o, u, n, 0.0, L, H - 0.6, H, 0.0, 0.3)
            continue
        wbox(S, trim, o, u, n, 0.0, L, H - 0.6, H - 0.3, 0.0, 0.18)
        wbox(S, trim, o, u, n, 0.0, L, H - 0.3, H, 0.0, 0.34)
        k = max(2, int(L // 4.6))
        xs = np.linspace(0.0, L, k + 1)
        for xp in xs:
            wbox(S, trim, o, u, n, max(0.0, xp - 0.3), min(L, xp + 0.3), Z_B, H - 0.6, 0.0, 0.14)
        for xa, xb in zip(xs[:-1], xs[1:]):
            c = (xa + xb) / 2
            warch(S, "glass", o, u, n, c - 0.6, c + 0.6, H * 0.5, H * 0.78, 0.03, k=8)
            warch_frame(S, trim, o, u, n, c - 0.6, c + 0.6, H * 0.5, H * 0.78, 0.2, 0.0, 0.08, k=8)
        wbox(S, trim, o, u, n, 0.0, L, Z_B, 0.9, 0.0, 0.08)


def church(center, axis, Lr: float, Wr: float, rng, colour: str = "coral", dome_kind: str = "cupula_teja",
           giant: float | None = None) -> tuple[Soup, dict]:
    """Church on the rectangle (centre, unit axis from the facade to the back, length Lr, width Wr). ``giant``: one
    giant tower of about that height (m) on the left of the facade instead of the ordinary bell towers."""
    S = Soup()
    st = {"wall": f"church:{colour}", "roof": "roof:teja", "dome": f"dome:{dome_kind}", "ribs": rng.uniform() < 0.8,
          "tower_cap": str(rng.choice(["dome", "dome", "bulb", "pyramid"])), "oculus": rng.uniform() < 0.6, "espadana": False}
    Lr = min(Lr, 56.0)                                                 # RESEARCH.md §3.2: nave 8-14 (16) m x 25-55 m
    T = float(np.clip(0.24 * min(Wr, 24.0), 3.6, 6.0))
    nt = 2 if (Wr >= 20 and rng.uniform() < 0.45) else (1 if Wr >= 13 else 0)
    if giant:                                                          # stockier than a minaret: up to 0.55 x the width
        T, nt = max(5.0, float(min(np.clip(0.55 * Wr, 8.0, 20.0), Wr - 9.0))), 1
    W = min(Wr - nt * T, 16.0)
    Wr = W + nt * T
    if W < 8.0:
        nt, W = 0, Wr
    st["espadana"] = nt == 0
    yc = 0.0 if nt != 1 else -T / 2                                    # one tower on the left (+Y)
    y0, y1 = yc - W / 2, yc + W / 2
    H = float(np.clip(0.85 * W, 9.0, 15.0))
    rise = W / 2 * np.tan(np.radians(28))
    apse = Lr >= 26 and rng.uniform() < 0.65
    ra = 0.42 * W
    Ln = Lr - ra if apse else Lr
    has_dome = Lr >= 22 and rng.uniform() < 0.9
    # nave
    box_walls(S, st["wall"], 0.75, Ln, y0, y1, Z_B, H)
    side_walls(S, st, 0.75, Ln, y0, y1, H)
    rect = shapely.Polygon([(0.75, y0), (Ln, y0), (Ln, y1), (0.75, y1)])
    hip_shell(S, st["roof"], "trimc:blanco", rect, H, rise, "gable", overhang=0.35)
    top = facade(S, st, y0, y1, H, rise, rng)
    heights = [top]
    plants = []
    if nt and giant:
        top_t, plants = giant_tower(S, st, 0.0, y1, T, float(giant), rng)
        heights.append(top_t)
    elif nt:
        h1 = H + 0.6
        for ty in ([y1] if nt == 1 else [y1, y0 - T]):
            heights.append(tower(S, st, 0.0, ty, T, h1, rng))
    if apse:
        cylinder(S, st["wall"], (Ln, yc), ra, ra, Z_B, H * 0.92, k=16)
        dome(S, st["roof"], (Ln, yc), ra + 0.2, H * 0.92, k=16, m=4, rise=0.55)
    if has_dome:
        xc = 0.75 + (Ln - 0.75) * 0.68
        if nt != 1 and rng.uniform() < 0.6:                            # transepts
            e = min(0.45 * W, (Wr - W) / 2 + (0.0 if nt else 0.0))
            e = max(e, 0.35 * W) if nt == 0 else e
            if e >= 2.5:
                for s in (-1, 1):
                    ya, yb = (y1, y1 + e) if s > 0 else (y0 - e, y0)
                    box_walls(S, st["wall"], xc - 0.4 * W, xc + 0.4 * W, ya, yb, Z_B, H)
                    side_walls(S, st, xc - 0.4 * W, xc + 0.4 * W, ya, yb, H)
                    tr = shapely.Polygon([(xc - 0.4 * W, ya), (xc + 0.4 * W, ya), (xc + 0.4 * W, yb), (xc - 0.4 * W, yb)])
                    hip_shell(S, st["roof"], "trimc:blanco", tr, H, 0.4 * W * np.tan(np.radians(28)), "gable", overhang=0.3)
        hc = H + rise + 0.8
        box_walls(S, st["wall"], xc - W / 2 + 0.3, xc + W / 2 - 0.3, y0 + 0.3, y1 - 0.3, H, hc)
        box_walls(S, "trimc:blanco", xc - W / 2 + 0.1, xc + W / 2 - 0.1, y0 + 0.1, y1 - 0.1, hc, hc + 0.4)
        Rd = 0.4 * W
        hd = max(2.8, 0.32 * W)
        c = (xc, yc)
        cylinder(S, st["wall"], c, Rd, Rd, hc + 0.4, hc + 0.4 + hd, k=16)
        for i in range(16):                                            # drum windows and pilasters
            a0, a1 = 2 * np.pi * i / 16, 2 * np.pi * (i + 1) / 16
            pa, pb = np.array([np.cos(a0), np.sin(a0)]) * Rd, np.array([np.cos(a1), np.sin(a1)]) * Rd
            L = np.linalg.norm(pb - pa)
            uu = (pb - pa) / L
            nn = np.array([uu[1], -uu[0]])
            oo = np.array(c) + pa
            if i % 2 == 0:
                warch(S, "glass", oo, uu, nn, L / 2 - 0.35, L / 2 + 0.35, hc + 0.9, hc + 0.4 + hd - 0.9, 0.03, k=6)
            wbox(S, "trimc:blanco", oo, uu, nn, -0.12, 0.12, hc + 0.4, hc + 0.4 + hd, 0.0, 0.12)
        z = hc + 0.4 + hd
        cylinder(S, "trimc:blanco", c, Rd + 0.3, Rd + 0.3, z, z + 0.4, k=16, cap=True)
        top_d = dome(S, st["dome"], c, Rd + 0.05, z + 0.4, k=16, m=7, rise=1.05, ribs_key="trimc:blanco" if st["ribs"] else None,
                     ribs=8, rib_w=0.16)
        rl = 0.2 * Rd
        cylinder(S, "trimc:blanco", c, rl, rl, top_d - 0.1, top_d + 1.4 * rl, k=8)
        for i in range(8):
            a0, a1 = 2 * np.pi * i / 8, 2 * np.pi * (i + 1) / 8
            pa, pb = np.array([np.cos(a0), np.sin(a0)]) * (rl + 0.01), np.array([np.cos(a1), np.sin(a1)]) * (rl + 0.01)
            L = np.linalg.norm(pb - pa)
            uu = (pb - pa) / L
            wquad(S, "void", np.array(c) + pa, uu, np.array([uu[1], -uu[0]]), 0.15 * L, 0.85 * L, top_d + 0.15 * rl,
                  top_d + 1.15 * rl, 0.01)
        tl = dome(S, st["dome"], c, rl + 0.1, top_d + 1.4 * rl, k=8, m=3, rise=0.9)
        cross(S, "iron", c, tl, 1.8)
        heights.append(tl + 1.8)
    # onto the parcel: local X along ``axis`` from the facade, Y to its left
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    ay = np.array([-ax[1], ax[0]])
    origin = np.asarray(center, float) - ax * Lr / 2
    R = np.array([[ax[0], ay[0]], [ax[1], ay[1]]])
    out = Soup()
    for key, lst in S.parts.items():
        for V, F, UV in lst:
            W_ = V.copy()
            W_[:, :2] = V[:, :2] @ R.T + origin
            out.parts.setdefault(key, []).append((W_, F, UV))
    info = {"towers": nt, "dome": has_dome, "apse": apse, "height_m": float(max(heights)), "nave_m": [float(Ln), float(W)],
            "colour": colour, "dome_kind": dome_kind, "tower_cap": st["tower_cap"]}
    if giant:
        tc = np.array([T / 2, y1 + T / 2]) @ R.T + origin
        info.update(giant_m=float(max(heights)), tower_side_m=float(T), tower_xy=tc.tolist(),
                    plants=[(k, *(np.array([x, y]) @ R.T + origin).tolist(), float(z), float(sc)) for k, x, y, z, sc in plants])
    return out, info
