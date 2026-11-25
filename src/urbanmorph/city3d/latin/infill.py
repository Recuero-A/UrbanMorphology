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
* skyscraper district (``district``, optional; DESIGN.md §5.2): blocks whose representative point lies in the district
  get wide, deep lots (frontage 18-26 m, depth 18-24 m) and each lot one Latin skyscraper on the largest rectangle
  that fits it (1 m clear of its neighbours, sides 11-30 m), taller toward the district centre and at most 7.5 x its
  narrower side; lots too small for a tower and the block cores become gardens.
* lanes (``lanes_m``, optional): a block much larger than a colonial manzana is first cut by straight paved lanes
  (callejones, LANE_M wide, about every ``lanes_m`` along both axes of its minimum rectangle, joining the streets
  around it), and each sub-block gets its own ring of lots and patio (no 9 ha lawns inside big generated blocks).
Buildings are attached row houses (attachment recomputed); heights and styles are drawn later by the theme.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d.format import polygon, polygons, rings

WALK_M = 1.6                    # colonial sidewalk; wider than the 2 x 0.6 m sliver filter of environment.sidewalks
MIN_LOT_M2 = 25.0
LANE_M = 4.0


def _clean(g, min_area=MIN_LOT_M2):
    parts = [p for p in polygons(shapely.make_valid(g)) if p.area >= min_area]
    if not parts:
        return None
    p = max(parts, key=lambda q: q.area).simplify(0.25, preserve_topology=True)
    return shapely.orient_polygons(shapely.Polygon(p.exterior)) if p.is_valid and p.area >= min_area else None


def lots_of(P, rng, front=(7.0, 12.0), depth=(11.0, 15.0)):
    """(lots, patio) for one block polygon (already inset by the sidewalk); frontage and band depth ranges in m."""
    if P.area < 45.0:
        p = _clean(P)
        return ([p] if p is not None else []), None
    f = float(rng.uniform(*front))
    d = float(rng.uniform(*depth))
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


def split_block(P, rng, spacing):
    """Sub-blocks of P cut by straight lanes about every ``spacing`` m along both axes of its minimum rectangle."""
    if P.area <= 1.6 * spacing ** 2:
        return [P]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.asarray(shapely.minimum_rotated_rectangle(P).exterior.coords)[:4]
    cuts = []
    for e, f in ((xy[1] - xy[0], xy[3] - xy[0]), (xy[3] - xy[0], xy[1] - xy[0])):
        L = float(np.linalg.norm(e))
        n = int(round(L / spacing))
        for k in range(1, n):
            q = xy[0] + e * (k + rng.uniform(-0.12, 0.12)) / n
            cuts.append(shapely.LineString([q - f * 0.05, q + f * 1.05]).buffer(LANE_M / 2, cap_style="flat"))
    if not cuts:
        return [P]
    return [q for q in polygons(shapely.make_valid(P.difference(shapely.union_all(cuts)))) if q.area >= 45.0]


def _building(i, block_id, p, floors, extra=None):
    b = {"id": f"V{i}", "block": block_id, "parcels": [], "footprint": rings(p), "floors": floors, "floors_source": "theme",
         "height_m": 3.6 * floors, "height_source": "synthetic", "typology": "attached_low", "eave_m": 3.6 * floors,
         "roof": {"type": "flat", "pitch_deg": 0.0, "rise_m": 0.0, "source": "synthetic"},
         "style": {"attached_to": [], "source": "synthetic", "facade_family": "plaster", "palette": i % 6, "bay_width_m": 3.4,
                   "window_width_ratio": 0.5, "window_height_ratio": 0.55, "balconies": False, "balcony_pattern": "all",
                   "cornice": True, "ground_floor": "residential"}}
    b.update(extra or {})
    return b


def tower_site(p, rng, centre, district):
    """Skyscraper spec for one district lot (world centre ``centre`` of the tile origin), or None."""
    from urbanmorph.city3d.latin import palette as pal
    from urbanmorph.city3d.latin.build import fit_rect
    R = fit_rect(p)
    if R is None:
        return None
    xy = np.asarray(R.exterior.coords)[:4]
    e = [np.linalg.norm(xy[(k + 1) % 4] - xy[k]) for k in range(4)]
    k = int(np.argmax(e))
    ax = (xy[(k + 1) % 4] - xy[k]) / e[k]
    W, D = min(e[k] - 2.0, 30.0), min(e[(k + 1) % 4] - 2.0, 30.0)
    c = np.asarray(R.centroid.coords)[0]
    ay = np.array([-ax[1], ax[0]])
    for _ in range(6):                                              # fit_rect allows 5 % outside: shrink until inside
        Q = shapely.Polygon([c + sx * ax * W / 2 + sy * ay * D / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        if p.buffer(0.01).contains(Q):
            break
        W, D = 0.94 * W, 0.94 * D
    else:
        return None
    if min(W, D) < 11.0:
        return None
    lo, hi = district.get("h", (45.0, 130.0))
    t = max(0.0, 1.0 - float(np.hypot(*(c + centre - np.asarray(district["center"])))) / district["radius"])
    H = float(np.clip((lo + (hi - lo) * t ** 0.8) * rng.uniform(0.8, 1.12), 0.8 * lo, hi))
    H = min(H, 7.5 * min(W, D))
    names, w = [n for n, _, _ in pal.TOWER_WALLS], np.array([x for _, _, x in pal.TOWER_WALLS])
    return {"center": c.tolist(), "axis": ax.tolist(), "W": float(W), "D": float(D), "H": H,
            "wall": names[int(rng.choice(len(names), p=w / w.sum()))],
            "dome": str(rng.choice(["cupula_naranja", "cupula_teja", "cupula_azul"], p=[0.65, 0.2, 0.15]))}


def colonial_infill(city: dict, rng, inside=None, sites=(), offset=(0.0, 0.0), lanes_m=None, district=None, clip=None,
                    lot_depth=None) -> dict:
    """In place: replace buildings and green inside the village blocks with dense row houses, patios, plazas and
    their churches. ``inside(x, y)``: tile-local point in the village; ``sites``: [{"plaza": rings, "church": {...}}] in
    world coordinates; ``offset``: the tile's world origin; ``lanes_m``: cut big blocks by lanes (split_block);
    ``district``: {"polygon": rings, "center", "radius", "h": (lo, hi)} in world coordinates, a skyscraper district, or
    a list of them (each tower spec records its district's index); ``lot_depth``: (lo, hi) m of the row-house band
    (default 11-15 m; steep towns: shallower);
    ``clip``: the village as a tile-local polygon: every block is cut to it (instead of the representative-point test),
    so the village part of a block that reaches into the jungle is built too (machu_picchu.py).
    Returns counts."""
    ox, oy = offset
    loc = lambda xy: np.asarray(xy, float) - [ox, oy]
    plazas = [(shapely.Polygon(loc(s["plaza"][0])), s) for s in sites]
    blocks = [(b["id"], shapely.make_valid(polygon(b["polygon"]))) for b in city["blocks"]]
    target = []
    for bid, B in blocks:
        if clip is not None:
            Bc = shapely.make_valid(B.intersection(clip))
            if Bc.area >= 20:
                target.append((bid, Bc))
            continue
        rp = B.representative_point()
        if B.area >= 20 and (inside is None or inside(rp.x, rp.y)):
            target.append((bid, B))
    zone = shapely.union_all([B for _, B in target]) if target else shapely.Polygon()
    keep_b = [b for b in city["buildings"] if not zone.contains(shapely.Polygon(b["footprint"][0]).representative_point())]
    keep_l = [g for g in city["landuse"] if not zone.contains(shapely.make_valid(polygon(g["polygon"])).representative_point())]
    new_b, new_l = [], []
    rep = {"blocks": len(target), "houses": 0, "patios": 0, "plazas": 0, "churches": 0, "skyscrapers": 0, "gardens": 0}
    dists = district if isinstance(district, list) else ([district] if district else [])
    dpolys = [shapely.Polygon(loc(d_["polygon"][0])) for d_ in dists]
    if clip is not None and dpolys:                        # clipped (big) blocks: their part in a district builds towers
        split = []
        for bid, B in target:
            for dp in dpolys:
                inner = shapely.make_valid(B.intersection(dp))
                if inner.area >= 200.0 and B.area - inner.area >= 50.0:
                    split.append((bid, inner))
                    B = shapely.make_valid(B.difference(dp))
            split.append((bid, B))
        target = [(bid, B) for bid, B in split if B.area >= 20]
    house_lots = (lambda part: lots_of(part, rng, depth=tuple(lot_depth))) if lot_depth else (lambda part: lots_of(part, rng))
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
            cs = {"center": ctr.tolist(), "axis": ax.tolist(), "L": c["L"], "W": c["W"]}
            if c.get("tower_h"):
                cs["tower_h"] = float(c["tower_h"])                # a giant tower (church.giant_tower)
            if c.get("style"):
                cs["style"] = c["style"]                           # a church from the user's photos (church_styles)
            new_b.append(_building(len(new_b), bid, R, 2, {"church_site": cs}))
            # a "sub_block" site (machu_picchu.py: big contour blocks) is a plaza rectangle inside the block; the rest
            # of the block gets row houses, which then also face the plaza
            PZ = P.intersection(shapely.Polygon(loc(site["plaza"][0]))) if site.get("sub_block") else P
            pz = _clean(PZ.difference(R.buffer(2.5)), 200.0)
            if pz is not None:
                new_l.append({"class": "public_green", "source": "theme:plaza", "id": f"plaza_{bid}", "polygon": rings(pz)})
            rep["plazas"] += 1
            rep["churches"] += 1
            used = [R] + ([pz] if pz is not None else [])
            if site.get("sub_block"):
                rest = shapely.make_valid(P.difference(PZ.buffer(0.5, join_style="mitre")))
                for part in [q for p_ in polygons(rest) for q in (split_block(p_, rng, lanes_m) if lanes_m else [p_])]:
                    if part.area < MIN_LOT_M2:
                        continue
                    lots, patio = house_lots(part)
                    for p in lots:
                        new_b.append(_building(len(new_b), bid, p, 3))
                    used += lots
                    if patio is not None:
                        new_l.append({"class": "private_green", "source": "theme:patio", "id": f"patio_{bid}_{len(new_l)}",
                                      "polygon": rings(patio)})
                        rep["patios"] += 1
                        used.append(patio)
        elif any(dp.contains(B.representative_point()) for dp in dpolys):
            dk = next(k for k, dp in enumerate(dpolys) if dp.contains(B.representative_point()))
            district = dists[dk]
            if district.get("sites"):                          # explicit tower sites (a cluster): towers, the rest gardens
                used = []
                for sp in district["sites"]:
                    ctr = loc(sp["center"])
                    if not B.contains(shapely.Point(ctr)):
                        continue
                    ax_ = np.asarray(sp["axis"], float)
                    ay_ = np.array([-ax_[1], ax_[0]])
                    R = shapely.orient_polygons(shapely.Polygon([ctr + sx * ax_ * sp["W"] / 2 + sy * ay_ * sp["D"] / 2
                                                                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
                    if any(R.intersects(u_) for u_ in used):
                        continue
                    sk = {**sp, "center": ctr.tolist(), "district": dk}
                    new_b.append(_building(len(new_b), bid, R, int(sp["H"] // 3.8), {"skyscraper": sk, "typology": "tower"}))
                    rep["skyscrapers"] += 1
                    used.append(R)
                gard = _clean(P.difference(shapely.union_all(used).buffer(1.0)) if used else P, 25.0)
                if gard is not None:
                    new_l.append({"class": "private_green", "source": "theme:garden", "id": f"garden_{bid}_{len(new_l)}",
                                  "polygon": rings(gard)})
                    rep["gardens"] += 1
                    used.append(gard)
                rest = B.difference(shapely.union_all(used)) if used else B
                for k_, q in enumerate(polygons(shapely.make_valid(rest))):
                    if q.area >= 2.0:
                        new_l.append({"class": "paved", "source": "theme", "id": f"paved_{bid}_{k_}", "polygon": rings(q)})
                continue
            used = []
            parts = [q for part in polygons(P) for q in (split_block(part, rng, 1.5 * lanes_m) if lanes_m else [part])]
            for part in parts:
                lots, core = lots_of(part, rng, front=tuple(district.get("lot_front", (18.0, 26.0))),
                                     depth=tuple(district.get("lot_depth", (18.0, 24.0))))
                for p in lots:
                    sk = tower_site(p, rng, np.array([ox, oy]), district)
                    if sk is not None and len(dists) > 1:
                        sk["district"] = dk
                    if sk is None and district.get("fill") == "houses":     # a dense downtown: a colonial block, no garden
                        new_b.append(_building(len(new_b), bid, p, 4))
                        used.append(p)
                        continue
                    if sk is None:
                        new_l.append({"class": "private_green", "source": "theme:garden", "id": f"garden_{bid}_{len(new_l)}",
                                      "polygon": rings(p)})
                        rep["gardens"] += 1
                        used.append(p)
                        continue
                    ax_, ay_ = np.asarray(sk["axis"]), np.array([-sk["axis"][1], sk["axis"][0]])
                    ctr = np.asarray(sk["center"])
                    R = shapely.orient_polygons(shapely.Polygon([ctr + sx * ax_ * sk["W"] / 2 + sy * ay_ * sk["D"] / 2
                                                                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
                    nfl = int(sk["H"] // 3.8)
                    new_b.append(_building(len(new_b), bid, R, nfl, {"skyscraper": sk, "typology": "tower"}))
                    rep["skyscrapers"] += 1
                    used.append(R)
                if core is not None:
                    new_l.append({"class": "private_green", "source": "theme:garden", "id": f"court_{bid}_{len(new_l)}",
                                  "polygon": rings(core)})
                    rep["gardens"] += 1
                    used.append(core)
        else:
            used = []
            parts = [q for part in polygons(P) for q in (split_block(part, rng, lanes_m) if lanes_m else [part])]
            for part in parts:
                lots, patio = house_lots(part)
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
    rep["houses"] = sum("church_site" not in b and "skyscraper" not in b for b in new_b)
    allb = keep_b + new_b
    polys = [shapely.Polygon(b["footprint"][0]) for b in allb]
    tree = shapely.STRtree(polys)
    for i, (b, p) in enumerate(zip(allb, polys)):
        b["style"]["attached_to"] = [allb[j]["id"] for j in tree.query(p.buffer(0.3)) if j != i and p.distance(polys[j]) < 0.3]
    city["buildings"] = allb
    city["landuse"] = keep_l + new_l
    return rep


def plaza_sites(g, mask, L, class_widths, n=3, spacing=280.0, tile=512.0, max_w=22.0):
    """Global plaza + church sites: large compact blocks deep inside the village, inside one tile, >= spacing apart.
    The church takes one end of the block, its facade toward the block centre (axis = facade -> back); the rest is
    the plaza. ``max_w``: widest church (m; wider leaves room for a giant tower beside the nave)."""
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
        Lc, Wc = float(min(44.0, 0.55 * LR)), float(min(WR, max_w))
        ctr = cR + u * (LR / 2 - Lc / 2)
        out.append({"plaza": rings(B), "church": {"center": ctr.tolist(), "axis": u.tolist(), "L": Lc, "W": Wc}})
        if len(out) >= n:
            break
    return out
