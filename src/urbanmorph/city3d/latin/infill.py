"""Dense colonial infill for the Latin Caribbean theme (DESIGN.md §4.1): every street block of the village becomes
attached row houses on narrow lots, as in the reference towns, instead of the sparse Milestone 10 layout.

Per block (city["blocks"], the land between road surfaces), after a 1.6 m colonial sidewalk:
* lots: Voronoi cells of seeds placed every 7-12 m (one frontage per block) along the line half a lot depth inside
  the block edge, clipped to the perimeter band of depth 11-15 m. Cell boundaries are perpendicular to the frontage,
  so lots are row-house strips and corner lots take the corners. A block too small for a band is cut the same way
  across its whole depth (houses with two street fronts); a block under 45 m2 holds one house.
* patio: what is left inside the band becomes a private patio garden.
* plaza sites (given in world coordinates): their block becomes a paved plaza (public green) with a church at its
  head, facade toward the plaza.
Buildings are attached row houses (attachment recomputed); heights and styles are drawn later by the theme.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.format import polygon, polygons, rings

WALK_M = 1.6                    # colonial sidewalk; wider than the 2 x 0.6 m sliver filter of environment.sidewalks
MIN_LOT_M2 = 25.0


def _clean(g, min_area=MIN_LOT_M2):
    parts = [p for p in polygons(shapely.make_valid(g)) if p.area >= min_area]
    if not parts:
        return None
    p = max(parts, key=lambda q: q.area).simplify(0.25, preserve_topology=True)
    return shapely.orient_polygons(shapely.Polygon(p.exterior)) if p.is_valid and p.area >= min_area else None


def lots_of(P, rng):
    """(lots, patio) for one block polygon (already inset by the sidewalk)."""
    if P.area < 45.0:
        p = _clean(P)
        return ([p] if p is not None else []), None
    f = float(rng.uniform(7.0, 12.0))
    d = float(rng.uniform(11.0, 15.0))
    core = P.buffer(-d, join_style="mitre")
    band = P.difference(core) if not core.is_empty else P
    line = P.buffer(-min(d / 2, 0.45 * np.sqrt(P.area / np.pi)), join_style="mitre").boundary
    if line.is_empty or line.length < f:                                   # thin block: cut along its long axis
        R = shapely.minimum_rotated_rectangle(P)
        xy = np.asarray(R.exterior.coords)[:4]
        e = [np.linalg.norm(xy[(k + 1) % 4] - xy[k]) for k in range(4)]
        k = int(np.argmax(e))
        a, b = xy[k], xy[(k + 1) % 4]
        mid = (xy[(k + 2) % 4] + xy[(k + 3) % 4]) / 2 - (a + b) / 2
        n = max(1, int(round(e[k] / f)))
        seeds = [a + (b - a) * (j + 0.5) / n + mid / 2 for j in range(n)]
    else:
        n = max(2, int(round(line.length / f)))
        seeds = [np.asarray(line.interpolate((j + rng.uniform(0.35, 0.65)) * line.length / n).coords[0]) for j in range(n)]
    if len(seeds) == 1:
        p = _clean(band)
        return ([p] if p is not None else []), None
    cells = shapely.voronoi_polygons(shapely.MultiPoint(seeds), extend_to=P.buffer(5.0))
    lots = [p for c in shapely.get_parts(cells) if (p := _clean(c.intersection(band))) is not None]
    patio = _clean(core, 40.0) if not core.is_empty else None
    return lots, patio


def _building(i, block_id, p, floors, extra=None):
    b = {"id": f"V{i}", "block": block_id, "parcels": [], "footprint": rings(p), "floors": floors, "floors_source": "theme",
         "height_m": 3.6 * floors, "height_source": "synthetic", "typology": "attached_low", "eave_m": 3.6 * floors,
         "roof": {"type": "flat", "pitch_deg": 0.0, "rise_m": 0.0, "source": "synthetic"},
         "style": {"attached_to": [], "source": "synthetic", "facade_family": "plaster", "palette": i % 6, "bay_width_m": 3.4,
                   "window_width_ratio": 0.5, "window_height_ratio": 0.55, "balconies": False, "balcony_pattern": "all",
                   "cornice": True, "ground_floor": "residential"}}
    b.update(extra or {})
    return b


def colonial_infill(city: dict, rng, inside=None, sites=(), offset=(0.0, 0.0)) -> dict:
    """In place: replace buildings and green inside the village blocks with dense row houses, patios, plazas and
    their churches. ``inside(x, y)``: tile-local point in the village; ``sites``: [{"plaza": rings, "church": {...}}] in
    world coordinates; ``offset``: the tile's world origin. Returns counts."""
    ox, oy = offset
    loc = lambda xy: np.asarray(xy, float) - [ox, oy]
    plazas = [(shapely.Polygon(loc(s["plaza"][0])), s) for s in sites]
    blocks = [(b["id"], shapely.make_valid(polygon(b["polygon"]))) for b in city["blocks"]]
    target = []
    for bid, B in blocks:
        rp = B.representative_point()
        if B.area >= 20 and (inside is None or inside(rp.x, rp.y)):
            target.append((bid, B))
    zone = shapely.union_all([B for _, B in target]) if target else shapely.Polygon()
    keep_b = [b for b in city["buildings"] if not zone.contains(shapely.Polygon(b["footprint"][0]).representative_point())]
    keep_l = [g for g in city["landuse"] if not zone.contains(shapely.make_valid(polygon(g["polygon"])).representative_point())]
    new_b, new_l = [], []
    rep = {"blocks": len(target), "houses": 0, "patios": 0, "plazas": 0, "churches": 0}
    for bid, B in target:
        P = B.buffer(-WALK_M, join_style="mitre")
        if P.is_empty:
            continue
        site = next((s for pz, s in plazas if pz.intersection(B).area > 0.5 * pz.area), None)
        if site is not None:
            c = site["church"]
            ctr = loc(c["center"])
            ax = np.asarray(c["axis"], float)
            ay = np.array([-ax[1], ax[0]])
            corners = [ctr + sx * ax * c["L"] / 2 + sy * ay * c["W"] / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            R = shapely.orient_polygons(shapely.Polygon(corners))
            new_b.append(_building(len(new_b), bid, R, 2, {"church_site": {"center": ctr.tolist(), "axis": ax.tolist(),
                                                                        "L": c["L"], "W": c["W"]}}))
            pz = _clean(P.difference(R.buffer(2.5)), 200.0)
            if pz is not None:
                new_l.append({"class": "public_green", "source": "theme:plaza", "id": f"plaza_{bid}", "polygon": rings(pz)})
            rep["plazas"] += 1
            rep["churches"] += 1
            used = [R] + ([pz] if pz is not None else [])
        else:
            used = []
            for part in polygons(P):
                lots, patio = lots_of(part, rng)
                for p in lots:
                    new_b.append(_building(len(new_b), bid, p, 3))
                used += lots
                if patio is not None:
                    new_l.append({"class": "private_green", "source": "theme:patio", "id": f"patio_{bid}_{len(new_l)}",
                                  "polygon": rings(patio)})
                    rep["patios"] += 1
                    used.append(patio)
        rest = B.difference(shapely.union_all(used)) if used else B     # sidewalk edge and leftovers: paved
        for k, q in enumerate(polygons(shapely.make_valid(rest))):
            if q.area >= 2.0:
                new_l.append({"class": "paved", "source": "theme", "id": f"paved_{bid}_{k}", "polygon": rings(q)})
    rep["houses"] = sum("church_site" not in b for b in new_b)
    allb = keep_b + new_b
    polys = [shapely.Polygon(b["footprint"][0]) for b in allb]
    tree = shapely.STRtree(polys)
    for i, (b, p) in enumerate(zip(allb, polys)):
        b["style"]["attached_to"] = [allb[j]["id"] for j in tree.query(p.buffer(0.3)) if j != i and p.distance(polys[j]) < 0.3]
    city["buildings"] = allb
    city["landuse"] = keep_l + new_l
    return rep


def plaza_sites(g, mask, L, class_widths, n=3, spacing=280.0, tile=512.0):
    """Global plaza + church sites: large compact blocks deep inside the village, inside one tile, >= spacing apart.
    The church takes one end of the block, its facade toward the block centre (axis = facade -> back); the rest is
    the plaza."""
    from scipy import ndimage
    from urbanmorph.city3d.latin.build import fit_rect
    lines = [shapely.LineString(g.nodes[e]) for e in g.edges]
    road = shapely.union_all([ln.buffer(class_widths[int(c)] / 2) for ln, c in zip(lines, g.classes())])
    blocks = [p for p in polygons(shapely.box(0, 0, L, L).difference(road)) if 1500 <= p.area <= 7000]
    depth = ndimage.distance_transform_edt(mask) * 4.0
    rc = lambda x, y: (min(max(int((L + 256 - y) / 4), 0), mask.shape[0] - 1), min(max(int((x + 256) / 4), 0), mask.shape[1] - 1))
    cand = []
    for B in blocks:
        x0, y0, x1, y1 = B.bounds
        if int(x0 // tile) != int(x1 // tile) or int(y0 // tile) != int(y1 // tile):
            continue
        if 4 * np.pi * B.area / B.length ** 2 < 0.45:
            continue
        c = B.representative_point()
        if not mask[rc(c.x, c.y)]:
            continue
        R = fit_rect(B.buffer(-2.0))
        if R is None:
            continue
        xy = np.asarray(R.exterior.coords)[:4]
        e = [np.linalg.norm(xy[(k + 1) % 4] - xy[k]) for k in range(4)]
        k = int(np.argmax(e))
        LR, WR = e[k], min(e)
        if LR < 34 or WR < 14:
            continue
        u = (xy[(k + 1) % 4] - xy[k]) / LR
        cand.append((depth[rc(c.x, c.y)] + 0.002 * B.area, B, R, u, LR, WR))
    cand.sort(key=lambda t: -t[0])
    out = []
    for score, B, R, u, LR, WR in cand:
        cR = np.asarray(R.centroid.coords)[0]
        if any(np.hypot(*(cR - np.asarray(o["church"]["center"]))) < spacing for o in out):
            continue
        Lc, Wc = float(min(44.0, 0.55 * LR)), float(min(WR, 22.0))
        ctr = cR + u * (LR / 2 - Lc / 2)
        out.append({"plaza": rings(B), "church": {"center": ctr.tolist(), "axis": u.tolist(), "L": Lc, "W": Wc}})
        if len(out) >= n:
            break
    return out
