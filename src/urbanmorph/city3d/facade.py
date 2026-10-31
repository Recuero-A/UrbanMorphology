"""Hierarchical procedural façades in the spirit of split / CGA shape grammars (Wonka et al. 2003; Müller et al.
2006); docs/M11_PROCEDURAL_BUILDINGS.md §3.

Rules (applied to every wall of the building's actual solid, so pitched-roof buildings use their rectangle):
  Wall      -> occlusion(neighbours) -> visible runs        (party walls touching another building get nothing)
  Run       -> split(y){ ground | floor* }                   (ground floor taller for shop typologies; equal upper floors)
  Floor     -> repeat(x){ bay } with CORNER_M margins        (bay count = floor(run / target bay width), equal bays)
  Bay       -> window | door | shop window [+ balcony]       (centred in the bay; sill and head relative to the floor)
  Wall top  -> cornice band;  flat roof -> parapet ring
Openings are thin panels OFFSET_M in front of the wall (no boolean cuts), so walls stay watertight and the
4 cm gap is well above Eevee's depth precision at the camera distances used. Every opening is checked against
its wall rectangle and party-wall runs (metrics in ``facade_metrics``).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.mesh import RECT_TOL, RECT_TOL_DETACHED, rect_footprint

CORNER_M = 0.6           # no opening closer than this to a wall end (corner handling)
OFFSET_M = 0.04          # opening panels in front of the wall plane
FRAME_M = 0.08           # frame strip width around the glass
MIN_RUN_M = 1.6          # visible runs shorter than this get no openings
SILL_M = 0.9             # window sill above floor level
DOOR_W, DOOR_H = 1.2, 2.3
OCCLUDE_M = 0.45         # a wall point is a party wall if another footprint lies within this distance outside it
STEP_M = 0.25            # occlusion sampling step along the wall
BALCONY_D = 0.6          # balcony depth


class Parts:
    """Triangle soup per element kind; F indexes the kind's own V."""

    def __init__(self):
        self.V, self.F = {}, {}

    def add(self, kind, V, F):
        v = self.V.setdefault(kind, [])
        f = self.F.setdefault(kind, [])
        n = sum(len(x) for x in v)
        v.append(np.asarray(V, float))
        f.append(np.asarray(F, np.int64) + n)

    def arrays(self, kind):
        if kind not in self.V:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        return np.vstack(self.V[kind]), np.vstack(self.F[kind])


def quad(o, u, n, x0, x1, z0, z1, off):
    """Rectangle in a wall plane: origin o (2,), along-wall unit u, outward normal n; facing outward."""
    p = lambda x, z: [o[0] + u[0] * x + n[0] * off, o[1] + u[1] * x + n[1] * off, z]
    return [p(x0, z0), p(x1, z0), p(x1, z1), p(x0, z1)], [[0, 1, 2], [0, 2, 3]]


def box(o, u, n, x0, x1, z0, z1, d0, d1):
    """Closed box spanning along-wall [x0, x1], height [z0, z1], depth [d0, d1] in front of the wall."""
    V = []
    for d in (d0, d1):
        for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
            V.append([o[0] + u[0] * x + n[0] * d, o[1] + u[1] * x + n[1] * d, z])
    # 0-3 at depth d0 (wall side), 4-7 at d1 (outside); outward-facing triangles
    F = [[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
    return V, F


def walls_of(b: dict) -> tuple[shapely.Polygon, list]:
    """(solid footprint, [(a, b, outward normal)]) for the walls the mesher actually builds."""
    poly = shapely.orient_polygons(shapely.Polygon(b["footprint"][0], b["footprint"][1:]))
    if b["roof"]["type"] in ("gable", "hip"):
        r = rect_footprint(poly, RECT_TOL_DETACHED if b["roof"].get("footprint_regularised") else RECT_TOL)
        if r is not None:
            poly = r
    out = []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)[:-1]
        for i in range(len(c)):
            a, e = c[i], c[(i + 1) % len(c)]
            d = e - a
            L = np.hypot(*d)
            if L < 1e-6:
                continue
            out.append((a, e, np.array([d[1], -d[0]]) / L))          # right of a->b = outward (CCW exterior, CW holes)
    return poly, out


def visible_runs(a, e, n, neighbours) -> list[tuple[float, float]]:
    """Along-wall intervals (m) not facing a neighbouring footprint within OCCLUDE_M."""
    L = float(np.hypot(*(e - a)))
    if neighbours is None:
        return [(0.0, L)]
    t = np.arange(STEP_M / 2, L, STEP_M)
    u = (e - a) / L
    pts = a + np.outer(t, u) + n * OCCLUDE_M
    occ = shapely.contains_xy(neighbours, pts[:, 0], pts[:, 1])
    runs, start = [], None
    for ti, o in zip(t, occ):
        if not o and start is None:
            start = ti - STEP_M / 2
        if o and start is not None:
            runs.append((max(0.0, start), ti - STEP_M / 2))
            start = None
    if start is not None:
        runs.append((max(0.0, start), L))
    return runs


def floor_levels(eave: float, floors: int, shop: bool, storey: float) -> list[tuple[float, float]]:
    """[(z0, z1)] per floor. Ground floor 1.3 x storey (max 4.2 m) for shop typologies, upper floors equal."""
    floors = max(1, min(int(floors), int(eave // 2.4)))      # no storey lower than 2.4 m (inconsistent OSM height/levels)
    if floors == 1:
        return [(0.0, eave)]
    gf = min(4.2, 1.3 * storey) if shop else storey
    up = (eave - gf) / (floors - 1)
    if up < 2.4:                                               # not enough height: equal floors
        h = eave / floors
        return [(i * h, (i + 1) * h) for i in range(floors)]
    return [(0.0, gf)] + [(gf + i * up, gf + (i + 1) * up) for i in range(floors - 1)]


def facade(b: dict, neighbours, road, storey: float, parts: Parts, metrics: dict) -> None:
    """Add this building's openings, balconies, cornice and parapet to ``parts``; update ``metrics``."""
    st = b["style"]
    poly, walls = walls_of(b)
    eave = b["eave_m"]
    shop = b.get("typology") in ("midrise", "highrise") and st.get("ground_floor") == "shop"
    levels = floor_levels(eave, b["floors"], shop, storey)
    metrics["floors_capped_for_facade"] += len(levels) < b["floors"]
    bay_w = st.get("bay_width_m", 3.6)
    win_w = st.get("window_width_ratio", 0.5)
    win_h = st.get("window_height_ratio", 0.55)
    door_done = False
    # street-facing walls first (a door goes on the longest one), then the rest
    info = []
    for a, e, n in walls:
        L = float(np.hypot(*(e - a)))
        mid = (a + e) / 2
        street = bool(road is not None and not road.is_empty and road.distance(shapely.Point(mid + n * 3.0)) < 4.0)
        runs = visible_runs(a, e, n, neighbours)
        usable = max((r1 - r0 - 2 * CORNER_M for r0, r1 in runs), default=0.0)
        info.append((a, e, n, L, street, runs, usable))
    cand = [k for k in range(len(info)) if info[k][6] >= max(MIN_RUN_M, DOOR_W + 0.2)]
    door_wall = max(cand, key=lambda k: (info[k][4], info[k][6])) if cand else None
    m = metrics
    m["buildings_without_door_space"] += door_wall is None
    for k, (a, e, n, L, street, runs, _) in enumerate(info):
        u = (e - a) / L
        m["walls"] += 1
        m["party_wall_m"] += L - sum(r1 - r0 for r0, r1 in runs)
        for r0, r1 in runs:
            r0c, r1c = r0 + CORNER_M, r1 - CORNER_M
            if r1c - r0c < MIN_RUN_M:
                continue
            nb = max(1, int(round((r1c - r0c) / bay_w)))              # bays within ±33 % of the target width
            bw = (r1c - r0c) / nb
            m["bay_widths"].append(bw)
            ww = float(np.clip(bw * win_w, 0.7, 2.2))
            door_bay = nb // 2 if (k == door_wall and not door_done and bw >= DOOR_W + 0.2) else -1
            for fi, (z0, z1) in enumerate(levels):
                fh = z1 - z0
                for j in range(nb):
                    cx = r0c + (j + 0.5) * bw
                    if fi == 0 and j == door_bay:
                        x0, x1, zz0, zz1, kind = cx - DOOR_W / 2, cx + DOOR_W / 2, z0, z0 + min(DOOR_H, fh - 0.3), "door"
                        door_done = True
                    elif fi == 0 and shop and street:
                        sw = min(bw - 0.6, 3.5)
                        x0, x1, zz0, zz1, kind = cx - sw / 2, cx + sw / 2, z0 + 0.4, z0 + fh * 0.8, "shop"
                    else:
                        wh = float(np.clip(fh * win_h, 0.9, 2.3))
                        sill = min(SILL_M, fh - wh - 0.3)
                        x0, x1, zz0, zz1, kind = cx - ww / 2, cx + ww / 2, z0 + sill, z0 + sill + wh, "window"
                    if x1 - x0 < 0.5 or zz1 - zz0 < 0.5:
                        continue
                    # containment checks (by construction; counted to prove it)
                    if x0 < r0 + CORNER_M - 1e-6 or x1 > r1 - CORNER_M + 1e-6 or zz0 < -1e-6 or zz1 > eave + 1e-6:
                        m["openings_outside_wall"] += 1
                    m[kind + "s"] += 1
                    if kind == "door":
                        m["door_bottom_z"].append(zz0)
                        m["door_front_xy"].append((a + u * (x0 + x1) / 2 + n * 1.0).tolist())
                        V, F = quad(a, u, n, x0, x1, zz0, zz1, OFFSET_M)
                        parts.add("doors", V, F)
                        continue
                    V, F = quad(a, u, n, x0 + FRAME_M, x1 - FRAME_M, zz0 + FRAME_M, zz1 - FRAME_M, OFFSET_M)
                    parts.add("glass", V, F)
                    for fx0, fx1, fz0, fz1 in ((x0, x1, zz0, zz0 + FRAME_M), (x0, x1, zz1 - FRAME_M, zz1),
                                               (x0, x0 + FRAME_M, zz0 + FRAME_M, zz1 - FRAME_M), (x1 - FRAME_M, x1, zz0 + FRAME_M, zz1 - FRAME_M)):
                        V, F = quad(a, u, n, fx0, fx1, fz0, fz1, OFFSET_M)
                        parts.add("frames", V, F)
                    pattern = st.get("balcony_pattern", "all")
                    wants = (pattern == "all" or (pattern == "alternate" and j % 2 == 0) or (pattern == "first_floor" and fi == 1))
                    if kind == "window" and st.get("balconies") and street and fi >= 1 and wants:
                        bx0, bx1 = x0 - 0.25, x1 + 0.25
                        corners = np.array([a + u * x + n * d for x in (bx0, bx1) for d in (0.1, BALCONY_D)])
                        if neighbours is not None and shapely.contains_xy(neighbours, corners[:, 0], corners[:, 1]).any():
                            m["balconies_skipped_clash"] += 1
                            continue
                        V, F = box(a, u, n, bx0, bx1, z0 - 0.02, z0 + 0.13, 0.0, BALCONY_D)          # slab
                        parts.add("balconies", V, F)
                        for rx0, rx1, rz0, rz1, rd0 in ((bx0, bx1, z0 + 1.0, z0 + 1.05, BALCONY_D - 0.05),   # top rail
                                                        (bx0, bx1, z0 + 0.5, z0 + 0.53, BALCONY_D - 0.04),   # mid rail
                                                        (bx0, bx0 + 0.05, z0 + 0.13, z0 + 1.0, BALCONY_D - 0.05),
                                                        (bx1 - 0.05, bx1, z0 + 0.13, z0 + 1.0, BALCONY_D - 0.05)):
                            V, F = box(a, u, n, rx0, rx1, rz0, rz1, rd0, BALCONY_D)
                            parts.add("railings", V, F)
                        m["balconies"] += 1
            if st.get("cornice"):
                lift = 0.01 * (k % 3)                         # corner pieces of adjacent walls never coplanar
                V, F = box(a, u, n, r0, r1, eave - 0.35 + lift, eave + lift, 0.0, 0.22)
                parts.add("trim", V, F)
    if b["roof"]["type"] == "flat" or poly is None:
        inner = poly.buffer(-0.25, join_style="mitre")
        ring = poly.difference(inner) if not inner.is_empty else None
        if ring is not None and not ring.is_empty and ring.area > 1.0:
            from urbanmorph.city3d.mesh import prism
            from urbanmorph.city3d.format import polygons
            for p in polygons(ring):
                try:
                    V, F, _ = prism(shapely.orient_polygons(p), eave, eave + 0.9).arrays()
                except ValueError:
                    continue
                parts.add("parapet", V, F)
                m["parapets"] += 1


def new_metrics() -> dict:
    return {"walls": 0, "party_wall_m": 0.0, "windows": 0, "doors": 0, "shops": 0, "balconies": 0, "parapets": 0,
            "openings_outside_wall": 0, "bay_widths": [], "door_bottom_z": [], "buildings_without_door_space": 0,
            "balconies_skipped_clash": 0, "door_front_xy": [], "floors_capped_for_facade": 0}
