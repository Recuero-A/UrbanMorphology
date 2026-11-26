"""Latin skyscraper (DESIGN.md §5.2; user reference 2026-11-23): the vermilion tower with cream trim and an orange dome
of Spanish-colonial revival / Beaux-Arts Latin American downtowns, on a rectangle (centre, unit axis, W x D).

Bottom to top, in its local frame (X along ``axis``, Y to its left, Z up; built centred, then rotated):
* podium (cream, three storeys): stone plinth, a ground-floor arcade (portal) with warm shop windows, two floors of
  windows with surrounds and iron balconies, a cornice and a balustraded roof terrace around the shaft;
* shaft (red), set back from the podium: cream quoins, a pilaster between window bays, and stages of 3-5 storeys
  separated by projecting cornices with wrap-around balustraded balconies. Stages alternate rectangular windows (cream
  surround, sill, lintel) and two-storey round-arched windows; windows are warm glass (lit interiors at sunset);
* loggia: one or two storeys of tall arched openings, an entablature, a balustrade with corner urns;
* crown: an octagonal drum with small arched windows, an orange ribbed dome, a lantern, a gilt finial.
About 10-25 k triangles for 40-130 m.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.latin.church import _faces, balustrade, box_walls, ngon, ngon_walls, terrace, urn, walls4
from urbanmorph.city3d.latin.kit import Soup, Z_B, cylinder, dome, iron_balcony, wbox, wquad, warch, warch_frame, wslab

GF, FL = 5.0, 3.7                  # ground floor (arcade) and upper storey heights (m)
TRIM, CREAM, GLASS = "trimc:crema", "podium", "glass_warm"


def window(S, o, u, n, xc, z0, w, h):
    """Rectangular window: cream surround, warm glass, sill and lintel."""
    wquad(S, TRIM, o, u, n, xc - w / 2 - 0.22, xc + w / 2 + 0.22, z0 - 0.2, z0 + h + 0.22, 0.03)
    wquad(S, GLASS, o, u, n, xc - w / 2, xc + w / 2, z0, z0 + h, 0.07)
    wbox(S, TRIM, o, u, n, xc - w / 2 - 0.32, xc + w / 2 + 0.32, z0 - 0.32, z0 - 0.12, 0.0, 0.22)
    wslab(S, TRIM, o, u, n, xc - w / 2 - 0.36, xc + w / 2 + 0.36, z0 + h + 0.22, z0 + h + 0.45, 0.0, 0.25)


def arched(S, o, u, n, x0, x1, z0, zs, key=GLASS, t=0.24):
    warch(S, key, o, u, n, x0, x1, z0, zs, 0.04, k=8)
    warch_frame(S, TRIM, o, u, n, x0, x1, z0, zs, t, 0.0, 0.12, k=8)


def wrap_balcony(S, faces, z, depth=0.8):
    """Projecting cornice slab with a balustrade all around the tower (the stage breaks of the reference)."""
    for o, u, n, L in faces:
        wbox(S, TRIM, o, u, n, -depth, L + depth, z - 0.35, z, 0.0, depth)
        wslab(S, TRIM, o, u, n, -depth, L + depth, z - 0.6, z - 0.35, 0.0, 0.4)
        wquad(S, "baluster:blanco", o, u, n, -depth + 0.05, L + depth - 0.05, z, z + 1.05, depth - 0.06, uv="panel")
        wbox(S, TRIM, o, u, n, -depth, L + depth, z + 1.05, z + 1.18, depth - 0.2, depth)


def skyscraper(center, axis, W: float, D: float, H: float, rng, wall: str = "tower:rojo",
               dome_kind: str = "cupula_naranja") -> tuple[Soup, dict]:
    """Latin skyscraper on the rectangle (centre, unit axis along W), about H m to the top of its finial."""
    S = Soup()
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
    # ---- podium
    zp = GF + 2 * FL
    box_walls(S, CREAM, x0, x1, y0, y1, Z_B, zp)
    for o, u, n, L in walls4(x0, x1, y0, y1):
        wbox(S, "stone", o, u, n, -0.05, L + 0.05, Z_B, 0.7, 0.0, 0.15)
        nb = max(2, int(round(L / 4.6)))
        bw = L / nb
        for j in range(nb):
            a0, a1 = j * bw + 0.6, (j + 1) * bw - 0.6
            arched(S, o, u, n, a0, a1, 0.15, GF - 0.75 - (a1 - a0) / 2, t=0.3)
            if j:
                wbox(S, TRIM, o, u, n, j * bw - 0.25, j * bw + 0.25, 0.7, GF - 0.4, 0.0, 0.12)
            for f in range(2):
                z = GF + f * FL + 0.8
                window(S, o, u, n, (j + 0.5) * bw, z, min(1.3, 0.4 * bw), 2.1)
                if f == 0:
                    iron_balcony(S, {"trim": "crema"}, o, u, n, (j + 0.5) * bw - 0.95, (j + 0.5) * bw + 0.95, z)
        wbox(S, TRIM, o, u, n, -0.1, L + 0.1, GF - 0.4, GF, 0.0, 0.3)
        wbox(S, TRIM, o, u, n, -0.25, L + 0.25, zp - 0.5, zp, 0.0, 0.45)
    # ---- shaft
    si = 1.4 if min(W, D) >= 14 else 0.8
    X0, X1, Y0, Y1 = x0 + si, x1 - si, y0 + si, y1 - si
    terrace(S, shapely.box(x0 - 0.25, y0 - 0.25, x1 + 0.25, y1 + 0.25), shapely.box(X0, Y0, X1, Y1), zp)
    balustrade(S, walls4(x0 - 0.25, x1 + 0.25, y0 - 0.25, y1 + 0.25), zp)
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        urn(S, TRIM, px + 0.3 * np.sign(-px), py + 0.3 * np.sign(-py), zp + 1.24, 0.42)
    Ws, Ds = X1 - X0, Y1 - Y0
    rd = 0.43 * min(Ws, Ds)
    crown = 0.6 + 1.3 + 0.7 * rd + 0.4 + 0.95 * rd + 0.5 * rd + 0.9 * (0.28 * rd + 0.1) + 1.6
    big = H - zp - crown >= 8 * FL
    n_log = 2 if big else 1
    n_sh = max(2, int(round((H - zp - crown - n_log * FL) / (FL + 0.12))))
    stages, left = [], n_sh
    while left > 0:
        k = int(min(left, rng.integers(3, 6)))
        if left - k in (1, 2):
            k = left
        stages.append(k)
        left -= k
    faces = walls4(X0, X1, Y0, Y1)
    z = zp
    box_walls(S, wall, X0, X1, Y0, Y1, zp, zp + n_sh * FL + 0.12 * n_sh + n_log * FL + 0.6)
    for si_, nf in enumerate(stages):
        kind = "arch" if (si_ % 2 == 1 and nf >= 2) else "rect"
        h = nf * FL
        for o, u, n, L in faces:
            qw = 0.85
            nb = max(2, int(round((L - 2 * qw) / 5.2)))
            bw = (L - 2 * qw) / nb
            wbox(S, TRIM, o, u, n, 0.0, qw, z, z + h, 0.0, 0.14)
            wbox(S, TRIM, o, u, n, L - qw, L, z, z + h, 0.0, 0.14)
            for j in range(1, nb):
                xp = qw + j * bw
                wbox(S, TRIM, o, u, n, xp - 0.17, xp + 0.17, z, z + h, 0.0, 0.08)
            for j in range(nb):
                xc = qw + (j + 0.5) * bw
                ww = min(2.6, 0.5 * bw)
                if kind == "rect":                                    # the top storey of a stage is round-arched
                    for f in range(nf - 1):
                        window(S, o, u, n, xc, z + f * FL + 0.8, ww, 2.3)
                    zb = z + (nf - 1) * FL + 0.6
                    arched(S, o, u, n, xc - ww / 2, xc + ww / 2, zb, zb + FL - 0.9 - ww / 2)
                else:
                    for f in range(0, nf - 1, 2):
                        zb = z + f * FL + 0.8
                        arched(S, o, u, n, xc - ww / 2, xc + ww / 2, zb, zb + 2 * FL - 1.3 - ww / 2)
                    if nf % 2:
                        window(S, o, u, n, xc, z + (nf - 1) * FL + 0.8, ww, 2.3)
        z += h
        wrap_balcony(S, faces, z + 0.12 * nf)
        z += 0.12 * nf
    # ---- loggia
    hl = n_log * FL
    for o, u, n, L in faces:
        nb = max(2, int(round(L / 4.6)))
        bw = L / nb
        for j in range(nb):
            a0, a1 = j * bw + 0.55, (j + 1) * bw - 0.55
            arched(S, o, u, n, a0, a1, z + 0.5, z + hl - 0.6 - (a1 - a0) / 2, key="void", t=0.3)
        wbox(S, TRIM, o, u, n, 0.0, 0.85, z, z + hl, 0.0, 0.18)
        wbox(S, TRIM, o, u, n, L - 0.85, L, z, z + hl, 0.0, 0.18)
        wbox(S, TRIM, o, u, n, -0.3, L + 0.3, z + hl, z + hl + 0.6, 0.0, 0.5)
    z += hl + 0.6
    terrace(S, shapely.box(X0 - 0.3, Y0 - 0.3, X1 + 0.3, Y1 + 0.3), ngon(np.zeros(2), rd + 0.25), z)
    balustrade(S, walls4(X0 - 0.3, X1 + 0.3, Y0 - 0.3, Y1 + 0.3), z, h=1.15)
    for px, py in ((X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)):
        urn(S, TRIM, px + 0.25 * np.sign(-px), py + 0.25 * np.sign(-py), z + 1.29, 0.5)
    # ---- crown: drum, dome, lantern, finial
    c = (0.0, 0.0)
    hd = 0.7 * rd
    ngon_walls(S, CREAM, c, rd, z, z + hd)
    for o, u, n, L in _faces(c, rd):
        arched(S, o, u, n, 0.3 * L, 0.7 * L, z + 0.25 * hd, z + 0.8 * hd - 0.2 * L, key="glass_warm", t=0.15)
        wbox(S, TRIM, o, u, n, 0.0, 0.08 * L, z, z + hd, 0.0, 0.1)
    z += hd
    cylinder(S, TRIM, c, rd + 0.3, rd + 0.3, z, z + 0.4, k=16, cap=True)
    z += 0.4
    top = dome(S, f"dome:{dome_kind}", c, rd, z, k=16, m=7, rise=0.95, ribs_key=TRIM, ribs=8, rib_w=0.1 + 0.02 * rd)
    rl = 0.28 * rd
    cylinder(S, CREAM, c, rl, rl, top - 0.1, top + 0.5 * rd, k=8, cap=True)
    for o, u, n, L in _faces(c, rl + 0.01, rot=0.0):
        warch(S, "void", o, u, n, 0.22 * L, 0.78 * L, top + 0.08 * rd, top + 0.34 * rd, 0.01, k=6)
    zl = dome(S, f"dome:{dome_kind}", c, rl + 0.1, top + 0.5 * rd, k=8, m=3, rise=0.9)
    cylinder(S, "gold", c, 0.12, 0.02, zl, zl + 1.6, k=6)
    dome(S, "gold", c, 0.28, zl + 0.3, k=8, m=3, rise=1.0)
    # onto the site
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    R = np.array([[ax[0], -ax[1]], [ax[1], ax[0]]])
    out = Soup()
    for key, lst in S.parts.items():
        for V, F, UV in lst:
            V2 = V.copy()
            V2[:, :2] = V[:, :2] @ R.T + np.asarray(center, float)
            out.parts.setdefault(key, []).append((V2, F, UV))
    return out, {"height_m": float(zl + 1.6), "floors": int(3 + n_sh + n_log), "stages": stages, "dome": dome_kind,
                 "wall": wall, "W": float(W), "D": float(D)}
