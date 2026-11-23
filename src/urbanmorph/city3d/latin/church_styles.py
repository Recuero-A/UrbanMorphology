"""Two church styles from the user's photos (DESIGN.md §5.4; themes/latin_caribbean/references/church_*.png), built
with the relief kit (latin.relief):

* gothic - a neo-Gothic / Romanesque-revival church painted cobalt blue with white trim: twin square towers (white
           colonnettes, louvred twin lancets, a clock, Lombard arched bands under the cornice) under octagonal slate
           spires with gabled lucarnes and four open corner pinnacles; a round-arched portal; a rose window with cut
           tracery in a moulded archivolt; a gable with a white stepped corbel band, a medallion and an apex statue;
           pinnacle turrets at the gable feet;
* bahia  - the Pelourinho baroque type (Nossa Senhora do Rosário dos Pretos, Salvador): aqua walls with projecting
           sandstone pilasters, cornices and window frames; three doors and three balconied windows in eared frames
           with curved hoods; oval oculi in carved cartouches; belfries with bronze bells; a curved frontispiece with
           volutes, an oculus and a cross; towers crowned by tiled stacked-bulb cupolas and four bulb pinnacles with
           terracotta flames.

Built in a local frame (facade along X at Y = 0 facing -Y, the nave running to +Y, Z up), then turned so that local
+Y follows ``axis`` (into the church) and moved to ``center`` (the middle of the facade line).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.latin import relief as R
from urbanmorph.city3d.latin.church import box_walls, walls4
from urbanmorph.city3d.latin.kit import Soup, Z_B, beam, cylinder, hip_shell, wbox, wp, wquad, wside
from urbanmorph.city3d.latin.landmarks import arch_outline, arch_poly, circle, mould, poly_faces, solid

STYLES = ("gothic", "bahia")
SCHEMES = {
    "gothic": {"wall": "lm:azul_iglesia", "trim": "lm:blanco", "spire": "dome:cupula_pizarra", "stone": "stone", "glass": "void",
               "door": "wood:madera_oscura", "metal": "bronze", "roof": "roof:teja", "face": "clock_white"},
    # Bahia colours (user request 2026-11-23: more cohesive with the city; the grey edges darker, more vibrant and
    # tropical): house-palette turquoise walls, deep golden ochre stone, glazed terracotta tiles on the bulbs as on the
    # city's domes. The photo's own colours: wall lm:aguamarina, trim lm:arenisca, bulbs dome:cupula_gris, pin lm:piedra_gris.
    "bahia": {"wall": "wall:turquesa", "trim": "lm:ocre_vivo", "dark": "lm:arenisca_oscura", "bulb": "dome:cupula_teja",
              "pin": "lm:ocre_vivo", "flame": "lm:llama", "glass": "glass_teal", "door": "wood:madera_oscura", "metal": "bronze",
              "roof": "roof:teja", "iron": "iron_railing"},
}
# the colours of the user's photos (the studio sheets compare the models with them); SCHEMES holds the city's
PHOTO_COLOURS = {"gothic": {}, "bahia": {"wall": "lm:aguamarina", "trim": "lm:arenisca", "bulb": "dome:cupula_gris",
                                         "pin": "lm:piedra_gris"}}
FRONT = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]))     # wall frame of the facade (x, z)


def front(x0, y):
    """Wall frame of a facade plane at Y = y facing -Y, with x measured from X = x0."""
    return np.array([x0, y]), np.array([1.0, 0.0]), np.array([0.0, -1.0])


def box_faces(x0, x1, y0, y1):
    return walls4(x0, x1, y0, y1)


# ----------------------------------------------------------------------------------------------- Gothic
def lancet_louvre(S, sc, o, u, n, x0, x1, z0, zs):
    """Round-headed opening with white louvre slats in a moulded white frame."""
    T = sc["trim"]
    R.wextrude(S, "void", o, u, n, arch_poly(x0, x1, z0, zs), -0.25, 0.01)
    for z in np.arange(z0 + 0.2, zs + (x1 - x0) / 2 - 0.15, 0.32):
        r = (x1 - x0) / 2
        dx = r if z <= zs else np.sqrt(max(r * r - (z - zs) ** 2, 0.0))
        xc = (x0 + x1) / 2
        if dx > 0.06:
            wbox(S, T, o, u, n, xc - dx, xc + dx, z, z + 0.07, -0.22, -0.02)
    R.wsweep(S, T, o, u, n, arch_outline(x0, x1, z0, zs), R.scale(R.ARCH_S, 0.25, 0.22))


def lombard(S, key, o, u, n, x0, x1, z, h=0.7, w=0.8, d=0.22):
    """Lombard band: a row of small round arches hanging under a cornice."""
    m = max(1, int(round((x1 - x0) / w)))
    ww = (x1 - x0) / m
    g = shapely.box(x0, z - h, x1, z)
    for i in range(m):
        a0, a1 = x0 + i * ww + 0.12, x0 + (i + 1) * ww - 0.12
        g = g.difference(arch_poly(a0, a1, z - h - 0.1, z - h + 0.05 + 0.0).union(
            shapely.Point((a0 + a1) / 2, z - h + 0.05).buffer((a1 - a0) / 2, quad_segs=6)))
    R.wextrude(S, key, o, u, n, g, 0.0, d)


def colonnette(S, key, o, u, n, x, z0, z1, r=0.2):
    p = wp(o, u, n, x, r, 0.0)
    R.column(S, key, (p[0], p[1]), r, z0, z1, order="tuscan", k=8)


def gothic_pinnacle(S, sc, c, z, s):
    """Small open octagonal turret with lancets and a spirelet, a ball."""
    T = sc["trim"]
    oc = circle(s, 8) @ np.array([[np.cos(np.pi / 8), np.sin(np.pi / 8)], [-np.sin(np.pi / 8), np.cos(np.pi / 8)]]) + c
    solid(S, T, oc, z, z + 0.35 * s)
    cylinder(S, "void", c, 0.6 * s, 0.6 * s, z + 0.35 * s, z + 2.4 * s, k=8)
    for x, y in circle(0.92 * s, 8) + c:
        cylinder(S, T, (x, y), 0.1 * s, 0.1 * s, z + 0.35 * s, z + 2.4 * s, k=4)
    zt = mould(S, T, oc, z + 2.4 * s, R.scale(R.STRING, 0.12 * s, 0.35 * s))
    cylinder(S, sc["spire"], c, 1.02 * s, 0.03, zt, zt + 3.2 * s, k=8)
    R.ball(S, T, c, zt + 3.1 * s, 0.14 * s)


def spire(S, sc, c, z, r, H):
    """Octagonal slate spire with white bands, four gabled lucarnes on the cardinal faces, a ball and a cross."""
    T = sc["trim"]
    ang = np.pi / 8 + np.arange(8) * np.pi / 4
    base = np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]
    apex = np.array([c[0], c[1], z + H])
    tris = [[np.r_[base[i], z], np.r_[base[(i + 1) % 8], z], apex] for i in range(8)]
    R.add_oriented(S, sc["spire"], tris, [np.r_[np.mean([base[i], base[(i + 1) % 8]], 0) - c, 0.25 * r] for i in range(8)],
                   np.array([[[0, 0], [2 * r * np.sin(np.pi / 8), 0], [r * np.sin(np.pi / 8), H]]] * 8))
    for f in (0.3, 0.6):
        rr = r * (1 - f)
        mould(S, T, np.c_[c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang)], z + f * H, R.scale(R.STRING, 0.08, 0.25))
    for k in range(4):                                             # lucarnes on the cardinal faces
        a = k * np.pi / 2 - np.pi / 2
        n_ = np.array([np.cos(a), np.sin(a)])
        u_ = np.array([-n_[1], n_[0]])                              # wall frame: n = (u_y, -u_x)
        o = c + n_ * (r * np.cos(np.pi / 8) + 0.35) - u_ * 0.75
        wbox(S, sc["spire"], o, u_, n_, 0.0, 1.5, z, z + 1.6, -1.2, 0.0)
        R.wextrude(S, "void", o, u_, n_, arch_poly(0.45, 1.05, z + 0.3, z + 1.2), -0.05, 0.01)
        R.wsweep(S, T, o, u_, n_, arch_outline(0.45, 1.05, z + 0.3, z + 1.2), R.scale(R.ARCH_S, 0.15, 0.12))
        tri = shapely.Polygon([(-0.1, z + 1.6), (1.6, z + 1.6), (0.75, z + 2.6)])
        R.wextrude(S, sc["spire"], o, u_, n_, tri, -1.2, 0.0)
        R.wsweep(S, T, o, u_, n_, [(-0.15, z + 1.6), (0.75, z + 2.65), (1.65, z + 1.6)], R.scale(R.RAKE, 0.15, 0.15), caps=True)
        p = wp(o, u_, n_, 0.75, 0.0, 0.0)
        cylinder(S, T, (p[0], p[1]), 0.08, 0.01, z + 2.55, z + 3.3, k=4)
    R.ball(S, T, c, z + H - 0.3, 0.3)
    cylinder(S, "iron", c, 0.06, 0.06, z + H + 0.2, z + H + 2.4, k=4)
    beam(S, "iron", [c[0] - 0.55, c[1], z + H + 1.7], [c[0] + 0.55, c[1], z + H + 1.7], 0.1, 0.1)


def gothic_tower(S, sc, x0, x1, y0, y1, H1):
    """Square tower (x0..x1, y0..y1) up to the spire base H1: white corner colonnettes, string courses, louvred twin
    lancets, the clock, Lombard bands, the spire with lucarnes and corner pinnacles."""
    T, Wl = sc["trim"], sc["wall"]
    box_walls(S, Wl, x0, x1, y0, y1, Z_B, H1)
    Tw = x1 - x0
    zs = (0.35 * H1, 0.66 * H1)
    for o, u, n, L in box_faces(x0, x1, y0, y1):
        for x in (0.32, L - 0.32):
            colonnette(S, T, o, u, n, x, 0.8, H1 - 1.0, 0.24)
        # middle stage: long twin louvred lancets
        lancet_louvre(S, sc, o, u, n, 0.24 * L, 0.46 * L, zs[0] + 1.4, zs[1] - 1.6 - 0.11 * L)
        lancet_louvre(S, sc, o, u, n, 0.54 * L, 0.76 * L, zs[0] + 1.4, zs[1] - 1.6 - 0.11 * L)
        # belfry stage: clock over twin lights
        R.clock(S, o, u, n, 0.5 * L, H1 - 2.9, 0.2 * L, T, d=0.2, face=sc["face"])
        lancet_louvre(S, sc, o, u, n, 0.3 * L, 0.47 * L, zs[1] + 0.8, H1 - 4.4 - 0.085 * L)
        lancet_louvre(S, sc, o, u, n, 0.53 * L, 0.7 * L, zs[1] + 0.8, H1 - 4.4 - 0.085 * L)
        lombard(S, T, o, u, n, 0.6, L - 0.6, H1 - 0.6, h=0.6, w=0.75)
    P = R.rect(x0, x1, y0, y1)
    for z in zs:
        mould(S, T, P, z - 0.5, R.scale(R.STRING, 0.3, 0.5))
    mould(S, T, P, Z_B, R.scale(R.BASE, 0.35, 0.9))
    z = mould(S, T, P, H1 - 0.6, R.scale(R.CORNICE, 0.6, 0.8))
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        gothic_pinnacle(S, sc, c + np.array([sx, sy]) * (Tw / 2 - 0.45), z, 0.42)
    solid(S, Wl, circle(0.4 * Tw, 8) @ np.array([[np.cos(np.pi / 8), np.sin(np.pi / 8)], [-np.sin(np.pi / 8), np.cos(np.pi / 8)]]) + c, z,
          z + 0.6)
    spire(S, sc, c, z + 0.6, 0.4 * Tw, 2.5 * Tw)


def rose(S, sc, o, u, n, xc, zc, r):
    """Rose window: white tracery cut by a rosette, a ring of twelve petals and twelve six-lobed flowers, in a moulded
    white ring; dark behind."""
    T = sc["trim"]
    R.wextrude(S, "void", o, u, n, shapely.Point(xc, zc).buffer(r, quad_segs=16), -0.05, 0.03)
    g = shapely.Point(xc, zc).buffer(r, quad_segs=16)
    g = g.difference(R.rosette(xc, zc, 0.2 * r, 8))
    for k in range(12):
        a = 2 * np.pi * k / 12
        g = g.difference(R.lobe(xc + 0.28 * r * np.cos(a), zc + 0.28 * r * np.sin(a), xc + 0.58 * r * np.cos(a), zc + 0.58 * r * np.sin(a),
                                0.07 * r))
        b = a + np.pi / 12
        g = g.difference(R.rosette(xc + 0.8 * r * np.cos(b), zc + 0.8 * r * np.sin(b), 0.1 * r, 6))
    R.wextrude(S, T, o, u, n, g, 0.03, 0.2)
    R.wsweep(S, T, o, u, n, circle(r, 40) + [xc, zc], R.scale(R.ARCHITRAVE, 0.4, 0.45), closed=True)


def stepped_gable(S, sc, o, u, n, x0, x1, zb, rise):
    """Gable: a white raking cornice, under it a white band cut by round-topped notches stepping down the slopes."""
    T = sc["trim"]
    xc = (x0 + x1) / 2
    tri = shapely.Polygon([(x0, zb), (x1, zb), (xc, zb + rise)])
    band = tri.difference(shapely.affinity.translate(tri, 0, -1.6).intersection(tri.buffer(-1.0)))
    step = 1.25
    for i in range(1, int((x1 - x0) / 2 / step) + 1):
        for sg in (-1, 1):
            x = xc + sg * (i - 0.5) * step
            zt = zb + rise * (1 - abs(x - xc) / ((x1 - x0) / 2)) - 0.45
            band = band.difference(arch_poly(x - 0.32, x + 0.32, zt - 2.2, zt - 0.32))
    R.wextrude(S, T, o, u, n, band.intersection(tri), 0.0, 0.22)
    R.wsweep(S, T, o, u, n, [(x0 - 0.3, zb), (xc, zb + rise + 0.35), (x1 + 0.3, zb)], R.scale(R.RAKE, 0.5, 0.45), caps=True)


def gothic_church(center, axis, L, W, rng, colours=None) -> tuple[Soup, dict]:
    sc = {**SCHEMES["gothic"], **(colours or {})}
    T, Wl = sc["trim"], sc["wall"]
    S = Soup()
    Tw = 0.21 * W
    xa, xb = -W / 2 + Tw, W / 2 - Tw                               # the centre bay between the towers
    sh = float(np.clip(W / 26.0, 0.82, 1.1))                       # narrow sites: lower towers and nave
    H1, Hn = 28.0 * sh, 19.0 * sh
    # nave: walls, buttresses, round-headed windows, tile gable roof
    box_walls(S, Wl, xa, xb, 0.0, L, Z_B, Hn)
    for o, u, n, Lf in box_faces(xa, xb, 0.0, L)[1::2]:
        for y in np.arange(5.0, Lf - 2.0, 5.5):
            R.window(S, sc, o, u, n, y - 0.8, y + 0.8, 5.0, 12.5, glass="glass_teal", arch=True, depth=0.3, w=0.3, sill=True)
            wbox(S, T, o, u, n, y + 2.0, y + 2.8, Z_B, Hn - 1.5, 0.0, 0.7)
    mould(S, T, [(xa, 0.0), (xb, 0.0), (xb, L), (xa, L)], Hn - 0.9, R.scale(R.CORNICE, 0.55, 0.9))
    hip_shell(S, sc["roof"], T, shapely.box(xa - 0.4, 0.5, xb + 0.4, L + 0.4), Hn, 0.46 * (xb - xa), "gable", overhang=0.4)
    # facade of the centre bay
    o, u, n = front(xa, 0.0)
    Wc = xb - xa
    box_walls(S, Wl, xa, xb, -0.6, 0.0, Z_B, Hn)
    o, u, n = front(xa, -0.6)
    mould(S, T, [(xa, -0.6), (xb, -0.6), (xb, 0.0), (xa, 0.0)], Z_B, R.scale(R.BASE, 0.35, 0.9))
    R.wextrude(S, sc["door"], o, u, n, arch_poly(Wc / 2 - 1.8, Wc / 2 + 1.8, 0.0, 4.8), -0.05, 0.04)
    for k, g in enumerate((0.0, 0.45, 0.9)):                       # stepped archivolt of the portal
        R.wsweep(S, T, o, u, n, arch_outline(Wc / 2 - 1.8 - g, Wc / 2 + 1.8 + g, 0.0, 4.8), R.scale(R.ARCHITRAVE, 0.45 - 0.1 * k, 0.4))
    zstr = 0.49 * Hn
    mould(S, T, [(xa, -0.6), (xb, -0.6), (xb, 0.0), (xa, 0.0)], zstr, R.scale(R.STRING, 0.4, 0.6))
    # rose window in a round-arched recess on white pilasters
    rr = min(0.27 * Wc, (0.51 * Hn - 1.2) / 2)                     # the rose fits between the string course and cornice
    zc = min(max(0.716 * Hn, zstr + 0.3 + rr), Hn - 0.9 - rr)
    for x in (Wc / 2 - rr - 1.3, Wc / 2 + rr + 0.9):
        R.pilaster(S, T, o, u, n, x, x + 0.4, zstr + 0.6, zc, 0.3, base_h=0.4, cap_h=0.45)
    R.wsweep(S, T, o, u, n, arch_outline(Wc / 2 - rr - 0.9, Wc / 2 + rr + 0.9, zc, zc, 16)[1:-1], R.scale(R.ARCHITRAVE, 0.5, 0.55))
    rose(S, sc, o, u, n, Wc / 2, zc, rr)
    for x in (Wc / 2 - rr - 1.9, Wc / 2 + rr + 1.9):
        R.wsweep(S, T, o, u, n, circle(0.55, 20) + [x, Hn - 1.6], R.scale(R.ARCHITRAVE, 0.25, 0.18), closed=True)
    mould(S, T, [(xa, -0.6), (xb, -0.6), (xb, 0.0), (xa, 0.0)], Hn - 0.6, R.scale(R.CORNICE, 0.45, 0.6))
    # gable, medallion, statue, pinnacle turrets
    rise = 0.56 * Wc
    tri = shapely.Polygon([(0.0, Hn), (Wc, Hn), (Wc / 2, Hn + rise)])
    R.wextrude(S, Wl, o, u, n, tri, -0.6, 0.0)
    stepped_gable(S, sc, o, u, n, 0.0, Wc, Hn, rise)
    R.wextrude(S, T, o, u, n, shapely.Point(Wc / 2, Hn + 0.38 * rise).buffer(1.25, quad_segs=10), 0.0, 0.12)
    R.wextrude(S, T, o, u, n, R.rosette(Wc / 2, Hn + 0.38 * rise, 0.8, 8), 0.12, 0.25)
    R.wsweep(S, T, o, u, n, circle(1.25, 24) + [Wc / 2, Hn + 0.38 * rise], R.scale(R.ARCHITRAVE, 0.32, 0.3), closed=True)
    p = wp(o, u, n, Wc / 2, 0.2, 0.0)
    box_walls(S, T, p[0] - 0.6, p[0] + 0.6, p[1] - 0.6, p[1] + 0.6, Hn + rise - 0.2, Hn + rise + 0.8)
    R.statue(S, "bronze", (p[0], p[1]), Hn + rise + 0.8, 2.6)
    for x in (xa + 0.6, xb - 0.6):
        c = np.array([x, -0.3])
        solid(S, Wl, R.rect(x - 0.7, x + 0.7, -1.0, 0.4), Hn - 0.1, Hn + 1.4)
        gothic_pinnacle(S, sc, c, Hn + 1.4, 0.55)
    # towers
    for x0 in (-W / 2, W / 2 - Tw):
        gothic_tower(S, sc, x0, x0 + Tw, -1.2, Tw - 1.2, H1)
        o, u, n = front(x0, -1.2)
        R.wsweep(S, T, o, u, n, arch_outline(0.22 * Tw, 0.78 * Tw, 2.0, 7.9 - 0.28 * Tw), R.scale(R.ARCHITRAVE, 0.35, 0.35))
        for a0, a1 in ((0.27 * Tw, 0.48 * Tw), (0.52 * Tw, 0.73 * Tw)):
            R.window(S, sc, o, u, n, a0, a1, 2.3, 7.2, glass="glass_teal", arch=True, depth=0.2, w=0.14, sill=False)
    return S, {"style": "gothic", "height_m": float(H1 + 0.6 + 0.6 + 1.95 * Tw + 2.4), "W": W, "L": L}


# ----------------------------------------------------------------------------------------------- Bahia baroque
def baroque_frame(S, sc, o, u, n, x0, x1, z0, z1, hood="segment", apron=True, door=False):
    """Eared stone frame (architrave with lugs at the top corners), a curved or broken hood, a scrolled apron."""
    T = sc["trim"]
    if door:                                                       # jambs and head only (no frame under a door)
        R.wsweep(S, T, o, u, n, [(x1, z0), (x1, z1), (x0, z1), (x0, z0)], R.scale(R.ARCHITRAVE, 0.35, 0.38), caps=True)
    else:
        R.wsweep(S, T, o, u, n, R.rect(x0, x1, z0, z1), R.scale(R.ARCHITRAVE, 0.35, 0.38), closed=True)
    for x in (x0 - 0.38, x1 + 0.38):
        wbox(S, T, o, u, n, x - 0.22, x + 0.22, z1 - 0.2, z1 + 0.38, 0.0, 0.36)
    R.pediment(S, T, o, u, n, x0 - 0.7, x1 + 0.7, z1 + 0.42, hood, depth=0.5, rise=0.25 * (x1 - x0) + 0.3, h=0.3, band=0.4)
    xc = (x0 + x1) / 2
    R.wextrude(S, T, o, u, n, R.palmette(xc, z1 + 0.55, 0.9, 0.9), 0.0, 0.3)
    if apron and not door:
        g = R.scroll(xc - 0.45, z0 - 0.75, 0.3, 0.12, sgn=1, a0=np.pi / 2, turns=1.0, tail=(xc, z0 - 0.45)).union(
            R.scroll(xc + 0.45, z0 - 0.75, 0.3, 0.12, sgn=-1, a0=np.pi / 2, turns=1.0, tail=(xc, z0 - 0.45)))
        R.wextrude(S, T, o, u, n, g, 0.0, 0.25)


def oculus_cartouche(S, sc, o, u, n, xc, zc, rx, rz):
    """Oval oculus in a carved stone cartouche: moulded frame, side C-scrolls, a scrolled crest, a pendant."""
    T = sc["trim"]
    R.oculus(S, T, sc["glass"], o, u, n, xc, zc, rx, rz, w=0.3, depth=0.35, lugs=False)
    g = shapely.Polygon()
    for sg in (-1, 1):
        g = g.union(R.scroll(xc + sg * (rx + 0.45), zc + 0.1, 0.32, 0.13, sgn=sg, a0=np.pi / 2, turns=1.1, tail=(xc + sg * (rx + 0.2), zc - rz)))
        g = g.union(R.scroll(xc + sg * 0.42, zc + rz + 0.55, 0.22, 0.1, sgn=-sg, a0=-np.pi / 2, turns=1.0))
    g = g.union(R.palmette(xc, zc + rz + 0.35, 0.8, 0.9)).union(R.lobe(xc, zc - rz - 0.2, xc, zc - rz - 0.8, 0.12))
    R.wextrude(S, T, o, u, n, g, 0.0, 0.3)


def bahia_pinnacle(S, sc, c, z, s):
    """Stone pedestal, stacked grey bulbs, a terracotta flame."""
    box_walls(S, sc["trim"], c[0] - 0.55 * s, c[0] + 0.55 * s, c[1] - 0.55 * s, c[1] + 0.55 * s, z, z + 0.9 * s)
    mould(S, sc["trim"], R.rect(c[0] - 0.55 * s, c[0] + 0.55 * s, c[1] - 0.55 * s, c[1] + 0.55 * s), z + 0.9 * s, R.scale(R.CORNICE, 0.12 * s, 0.2 * s))
    zt = R.bulb_stack(S, sc["pin"], c, z + 1.1 * s, 0.42 * s, tiers=3, k=10)
    R.lathe(S, sc["flame"], c, [(0.12 * s, 0.0), (0.2 * s, 0.15 * s), (0.17 * s, 0.4 * s), (0.08 * s, 0.65 * s), (0.0, 0.85 * s)], zt, k=8)


def bahia_cupola(S, sc, c, z, r):
    """Tiled stacked cupola of the Pelourinho towers: a wide squat ogee bulb with ribs, a neck, a middle bulb, a small
    top bulb; returns the top."""
    T = sc["trim"]
    prof = [(0.75, 0.0), (0.95, 0.1), (1.0, 0.22), (0.92, 0.38), (0.66, 0.55), (0.38, 0.68), (0.22, 0.78), (0.0, 0.85)]
    h = 0.95 * r
    R.lathe(S, sc["bulb"], c, [(a * r, b * h / 0.85) for a, b in prof], z, k=24)
    for i in range(8):                                             # ribs down the big bulb
        t = 2 * np.pi * (i + 0.5) / 8
        P = np.array([[c[0] + a * r * np.cos(t), c[1] + a * r * np.sin(t), z + b * h / 0.85] for a, b in prof[:-2]])
        N = np.array([[np.cos(t), np.sin(t), 0.6] for _ in prof[:-2]])
        R.sweep3d(S, T, P, N / np.linalg.norm(N, axis=1, keepdims=True), R.scale(R.RIB, 0.12, 0.22))
    z += h * 0.93
    R.lathe(S, T, c, [(0.2 * r, 0.0), (0.26 * r, 0.1 * r), (0.2 * r, 0.25 * r)], z, k=12)
    z += 0.25 * r
    z = R.bulb_stack(S, sc["bulb"], c, z, 0.45 * r, tiers=2, k=16)
    return z


def bahia_tower(S, sc, x0, x1, y0, y1, Hb, Ht):
    """Tower: aqua walls between wide stone pilasters, an oval oculus in a cartouche, a stone entablature, a belfry
    with a bell in a round arch between paired pilasters, a big cornice, four bulb pinnacles, a tiled stacked-bulb
    cupola on a drum."""
    T, Wl = sc["trim"], sc["wall"]
    box_walls(S, Wl, x0, x1, y0, y1, Z_B, Ht)
    Tw = x1 - x0
    for o, u, n, L in box_faces(x0, x1, y0, y1):
        for xa, xb in ((-0.55, 0.16 * L), (0.84 * L, L + 0.55)):
            R.pilaster(S, T, o, u, n, xa, xb, Z_B, Hb - 1.2, 0.55, base_h=0.9, cap_h=0.8)
            R.pilaster(S, T, o, u, n, xa, xb, Hb + 0.2, Ht - 1.4, 0.55, base_h=0.6, cap_h=0.8)
        oculus_cartouche(S, sc, o, u, n, 0.5 * L, 0.68 * Hb, 0.13 * L, 0.1 * L)
        R.window(S, sc, o, u, n, 0.4 * L, 0.6 * L, 0.28 * Hb, 0.28 * Hb + 2.4, depth=0.35, w=0.3)
        a0, a1 = 0.3 * L, 0.7 * L
        zs = Ht - 2.6 - (a1 - a0) / 2
        R.wextrude(S, "void", o, u, n, arch_poly(a0, a1, Hb + 1.0, zs), -0.4, 0.01)
        R.wsweep(S, T, o, u, n, arch_outline(a0, a1, Hb + 1.0, zs), R.scale(R.ARCHITRAVE, 0.4, 0.45))
        for x in (a0 - 0.45, a1 + 0.45):
            wbox(S, T, o, u, n, x - 0.4, x + 0.4, zs - 0.35, zs, 0.0, 0.5)
        cb = wp(o, u, n, 0.5 * L, -0.35, 0.0)
        R.bell(S, sc["metal"], (cb[0], cb[1]), zs + 0.1, 0.32 * (a1 - a0))
        beam(S, "wood:madera_oscura", wp(o, u, n, a0 + 0.1, -0.35, zs + 0.25), wp(o, u, n, a1 - 0.1, -0.35, zs + 0.25), 0.22, 0.25)
    P = R.rect(x0, x1, y0, y1)
    mould(S, T, P, Z_B, R.scale(R.BASE, 0.6, 1.0))
    mould(S, T, P, Hb - 1.2, R.scale(R.STRING, 0.55, 0.6))
    mould(S, T, P, Hb - 0.6, R.scale(R.CORNICE, 0.75, 0.8))
    z = mould(S, T, P, Ht - 1.4, R.scale(R.CORNICE, 1.0, 1.4))
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        bahia_pinnacle(S, sc, c + np.array([sx, sy]) * (Tw / 2 + 0.2), z, min(1.55, 0.25 * Tw))
    dr = 0.3 * Tw
    solid(S, T, R.rect(c[0] - dr - 0.3, c[0] + dr + 0.3, c[1] - dr - 0.3, c[1] + dr + 0.3), z, z + 1.6)
    for o, u, n, L in box_faces(c[0] - dr - 0.3, c[0] + dr + 0.3, c[1] - dr - 0.3, c[1] + dr + 0.3):
        R.wextrude(S, "void", o, u, n, arch_poly(0.38 * L, 0.62 * L, z + 0.3, z + 1.0), -0.05, 0.01)
    z = mould(S, T, R.rect(c[0] - dr - 0.3, c[0] + dr + 0.3, c[1] - dr - 0.3, c[1] + dr + 0.3), z + 1.6, R.scale(R.CORNICE, 0.3, 0.45))
    zt = bahia_cupola(S, sc, c, z, 0.46 * Tw)
    cylinder(S, "iron", c, 0.05, 0.03, zt, zt + 1.8, k=4)
    beam(S, "iron", [c[0] - 0.4, c[1], zt + 1.2], [c[0] + 0.4, c[1], zt + 1.2], 0.08, 0.08)
    return zt + 1.8


def bahia_church(center, axis, L, W, rng, colours=None) -> tuple[Soup, dict]:
    sc = {**SCHEMES["bahia"], **(colours or {})}
    T, Wl = sc["trim"], sc["wall"]
    S = Soup()
    Tw = 0.23 * W
    xa, xb = -W / 2 + Tw, W / 2 - Tw
    Wc = xb - xa
    sh = float(np.clip(W / 27.0, 0.82, 1.1))                       # narrow sites: lower towers and body
    Hb, Ht, Hn = 16.5 * sh, 26.0 * sh, 15.0 * sh
    box_walls(S, Wl, xa, xb, 0.0, L, Z_B, Hn)
    for o, u, n, Lf in box_faces(xa, xb, 0.0, L)[1::2]:
        for y in np.arange(5.0, Lf - 2.0, 6.0):
            R.window(S, sc, o, u, n, y - 0.8, y + 0.8, 6.5, 10.0, depth=0.3, w=0.3, hood="segment")
            R.pilaster(S, T, o, u, n, y + 2.4, y + 3.3, Z_B, Hn - 1.0, 0.4, cap_h=0.6)
    mould(S, T, [(xa, 0.0), (xb, 0.0), (xb, L), (xa, L)], Hn - 1.0, R.scale(R.CORNICE, 0.7, 1.0))
    hip_shell(S, sc["roof"], T, shapely.box(xa - 0.4, 3.0, xb + 0.4, L + 0.4), Hn, 0.3 * Wc, "gable", overhang=0.5)
    # the central body: three doors, three balconied windows, stone pilasters, entablature
    box_walls(S, Wl, xa, xb, -0.5, 0.0, Z_B, Hn)
    o, u, n = front(xa, -0.5)
    for i, xc in enumerate((0.2 * Wc, 0.5 * Wc, 0.8 * Wc)):
        dw = 1.5 if i != 1 else 1.9
        dh = (4.4 if i != 1 else 5.4) * min(sh, 1.0)
        R.wextrude(S, sc["door"], o, u, n, shapely.box(xc - dw / 2, 0.0, xc + dw / 2, dh), -0.4, -0.05)
        baroque_frame(S, sc, o, u, n, xc - dw / 2, xc + dw / 2, 0.0, dh, hood="segment" if i != 1 else "broken", door=True)
        zw0, zw1 = 8.2 * sh, 11.4 * sh
        R.window(S, sc, o, u, n, xc - 0.75, xc + 0.75, zw0, zw1, depth=0.25, w=0.2, sill=False, mullion="wood:madera_oscura")
        baroque_frame(S, sc, o, u, n, xc - 0.75, xc + 0.75, zw0, zw1, hood="segment")
        path = [wp(o, u, n, xc - 1.3, 0, 0)[:2], wp(o, u, n, xc + 1.3, 0, 0)[:2]]
        R.sweep(S, T, path, zw0 - 0.3, R.scale(R.SILL, 0.8, 0.3), caps=True)
        wquad(S, sc["iron"], o, u, n, xc - 1.2, xc + 1.2, zw0, zw0 + 1.0, 0.75, uv="panel")
        wside(S, sc["iron"], o, u, n, xc - 1.18, 0.0, 0.75, zw0, zw0 + 1.0)
        wside(S, sc["iron"], o, u, n, xc + 1.18, 0.0, 0.75, zw0, zw0 + 1.0, flip=True)
    for x in (0.0, 0.35 * Wc, 0.65 * Wc, Wc):
        R.pilaster(S, T, o, u, n, x - 0.45, x + 0.45, Z_B, Hn - 1.4, 0.45, base_h=0.9, cap_h=0.7)
    P = [(xa, -0.5), (xb, -0.5), (xb, 0.0), (xa, 0.0)]
    mould(S, T, P, Z_B, R.scale(R.BASE, 0.5, 1.0))
    mould(S, T, P, 6.9 * sh, R.scale(R.STRING, 0.45, 0.5))
    z = mould(S, T, P, Hn - 1.4, R.scale(R.CORNICE, 0.9, 1.4))
    # curved frontispiece: S-curved shoulders rising to an arched top, volutes, oculus cartouche, urns, a cross
    t = np.linspace(0, 1, 14)
    left = [(0.18 * Wc + 0.2 * Wc * s_, z + 3.4 * (1 - np.cos(np.pi * s_)) / 2) for s_ in t]
    arc = [(Wc / 2 + 0.12 * Wc * np.cos(a), z + 3.4 + 0.9 + 0.12 * Wc * np.sin(a) * 0.9) for a in np.linspace(np.pi, 0, 13)]
    outline = [(0.12 * Wc, z)] + left + [(0.38 * Wc, z + 4.3)] + arc + [(0.62 * Wc, z + 4.3)] + \
        [(Wc - x, zz) for x, zz in left[::-1]] + [(0.88 * Wc, z)]
    gable = shapely.Polygon(outline).buffer(0)
    R.wextrude(S, Wl, o, u, n, gable, -0.5, 0.0)
    R.wsweep(S, T, o, u, n, outline[1:-1], R.scale(R.RAKE, 0.55, 0.45), caps=True)
    for sg, x in ((1, 0.12 * Wc), (-1, 0.88 * Wc)):
        R.wextrude(S, T, o, u, n, R.scroll(x, z + 0.9, 0.85, 0.3, sgn=sg, a0=np.pi / 2, turns=1.2), 0.0, 0.55)
        p = wp(o, u, n, x, 0.3, 0.0)
        R.urn(S, T, (p[0], p[1]), z, 1.6)
    oculus_cartouche(S, sc, o, u, n, Wc / 2, z + 2.6, 0.9, 0.7)
    p = wp(o, u, n, Wc / 2, 0.0, 0.0)
    ztop = z + 3.4 + 0.9 + 0.11 * Wc
    box_walls(S, T, p[0] - 0.35, p[0] + 0.35, p[1] - 0.35, p[1] + 0.35, ztop - 0.2, ztop + 0.5)
    cylinder(S, T, (p[0], p[1]), 0.1, 0.1, ztop + 0.5, ztop + 2.4, k=4)
    beam(S, T, [p[0] - 0.6, p[1], ztop + 1.8], [p[0] + 0.6, p[1], ztop + 1.8], 0.16, 0.16)
    tops = []
    for x0 in (-W / 2, W / 2 - Tw):
        tops.append(bahia_tower(S, sc, x0, x0 + Tw, -1.0, Tw - 1.0, Hb, Ht))
    return S, {"style": "bahia", "height_m": float(max(tops)), "W": W, "L": L}


BUILD = {"gothic": gothic_church, "bahia": bahia_church}


def church_style(style: str, center, axis, L: float, W: float, rng, colours=None) -> tuple[Soup, dict]:
    """Church ``style`` ('gothic' | 'bahia') with its facade centred on ``center``, the nave running along the unit
    ``axis`` for L m, facade width W m (towers included); colours: role -> material key overrides."""
    S, info = BUILD[style](center, axis, float(L), float(W), rng, colours)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    Rm = np.array([[ax[1], ax[0]], [-ax[0], ax[1]]])                 # local +Y -> axis, local +X -> axis turned clockwise
    out = Soup()
    for key, lst in S.parts.items():
        for V, F, UV in lst:
            V2 = V.copy()
            V2[:, :2] = V[:, :2] @ Rm.T + np.asarray(center, float)
            out.parts.setdefault(key, []).append((V2, F, UV))
    info["triangles"] = out.triangles()
    return out, info
