"""City format -> render groups in the Latin Caribbean theme (the ``theme="latin_caribbean"`` branch of
city3d.scene.build_scene; DESIGN.md §7).

Same inputs, group names and conventions as the standard builder, so terrain draping, tile merging, Blender and Godot
need nothing theme-specific beyond the materials. Styles are drawn per building from (city seed, building index):
* heights: colonial storeys (ground floor 3.8-4.3 m, upper 3.3-3.7 m), at most the city's floor count, mostly 1-3;
* roofs: hip / gable tile shells on detached near-rectangles, tile skirts around azoteas elsewhere, some flat;
* facades: wall colour never repeated by an attached neighbour; plinth, pilasters, string courses, cornices; doors
  (some arched), windows with mouldings and shutters, Cartagena window grilles, wooden or iron balconies;
* churches: on the largest regular parcels (a building merged with its attached neighbours), >= 260 m apart,
  facade to the nearest street (church.py);
* streets: asphalt on major / mid streets (with markings), cobbles elsewhere; small public greens become plazas;
* plants and lamps: plants.place.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.city3d import environment as env
from urbanmorph.city3d.facade import new_metrics, visible_runs
from urbanmorph.city3d.format import polygon, polygons
from urbanmorph.city3d.latin import palette as pal
from urbanmorph.city3d.latin.church import church
from urbanmorph.city3d.latin.kit import Soup, house, walls_and_roof
from urbanmorph.city3d.latin.plants import place, prototypes
from urbanmorph.city3d.mesh import RECT_TOL, clean_footprint, rect_footprint, surface
from urbanmorph.city3d.validate import solid_report, surface_report

THEME = "latin_caribbean"
GROUP_OF = [("wall:", "walls"), ("church:", "walls"), ("plinth:", "walls"), ("roof:", "roofs"), ("roof_flat", "roofs"),
            ("dome:", "roofs"), ("trimc:", "trim"), ("stone", "trim"), ("glass", "glass"), ("void", "glass"),
            ("door:", "doors"), ("shutter:", "carpentry"), ("wood:", "carpentry"), ("baluster:", "carpentry"),
            ("iron", "metal"), ("gold", "metal")]
CHURCH_SPACING_M = 260.0
CHURCH_P = 0.5                   # chance a tile with a church site gets one (about 1 church per 150-200 buildings)
CHURCH_MIN = (20.0, 11.0)          # rectangle length, width (m)
COLONIAL_WALKS = {"major": 2.5, "mid": 1.8, "local": 1.6, "minor": 1.6}   # narrow sidewalks of dense colonial towns


def group_of(key: str) -> str:
    return next(g for p, g in GROUP_OF if key.startswith(p))


def _choice(r, items, w):
    w = np.asarray(w, float)
    return items[int(r.choice(len(items), p=w / w.sum()))]


def fit_rect(P):
    """Largest scaled copy (about its centre) of P's minimum rotated rectangle that lies >= 95 % inside P."""
    with np.errstate(divide="ignore", invalid="ignore"):
        R = shapely.minimum_rotated_rectangle(P)
    if R.geom_type != "Polygon":
        return None
    for s in (1.0, 0.94, 0.88, 0.82, 0.76, 0.7):
        Rs = shapely.affinity.scale(R, s, s, origin="centroid")
        if Rs.intersection(P).area >= 0.95 * Rs.area:
            return Rs
    return None


def pick_churches(city, foot, road, seed):
    """[(building index, absorbed indices, centre, axis, L, W)] for the church sites of this tile."""
    if np.random.default_rng([seed, 98]).uniform() > CHURCH_P:        # one lottery per tile
        return []
    B = city["buildings"]
    ids = {b["id"]: i for i, b in enumerate(B)}
    cand = []
    for i, b in enumerate(B):
        nb = [ids[j] for j in b["style"].get("attached_to", []) if j in ids]
        P = shapely.union_all([foot[i]] + [foot[j] for j in nb]).buffer(0.3).buffer(-0.3)
        if P.area < 380:
            continue
        R = fit_rect(P)
        if R is None:
            continue
        xy = np.asarray(R.exterior.coords)[:4]
        e = [np.linalg.norm(xy[(k + 1) % 4] - xy[k]) for k in range(4)]
        L, W = max(e), min(e)
        if L < CHURCH_MIN[0] or W < CHURCH_MIN[1] or L > 2.6 * W + 8:
            continue
        cand.append((R.area, i, nb, R, L, W))
    cand.sort(key=lambda c: -c[0])
    out, taken = [], set()
    for area, i, nb, R, L, W in cand:
        c = np.asarray(R.centroid.coords)[0]
        if i in taken or any(np.hypot(*(c - o[2])) < CHURCH_SPACING_M for o in out) or len(out) >= 1:
            continue
        xy = np.asarray(R.exterior.coords)[:4]
        k = int(np.argmax([np.linalg.norm(xy[(j + 1) % 4] - xy[j]) for j in range(4)]))
        ax = (xy[(k + 1) % 4] - xy[k]) / np.linalg.norm(xy[(k + 1) % 4] - xy[k])
        ends = [c - ax * L / 2, c + ax * L / 2]
        dist = [road.distance(shapely.Point(p)) if not road.is_empty else 0.0 for p in ends]
        if dist[1] < dist[0]:
            ax = -ax                                           # facade on the end nearest a street
        absorbed = [j for j in nb if foot[j].intersection(R).area > 1.0]
        out.append((i, absorbed, c, ax, L, W))
        taken.update([i, *absorbed])
    return out


def district(block: str, seed: int) -> str:
    """One balcony sub-style per block (RESEARCH.md §7): Cartagena wood (55 %) or Havana / San Juan iron."""
    h = int.from_bytes(str(block).encode()[:16].ljust(16, b"\0"), "little") % 1000003
    return "wood" if np.random.default_rng([seed, h]).uniform() < 0.55 else "iron"


def other(kind: str) -> str:
    return "iron" if kind == "wood" else "wood"


def styles(city, foot, seed):
    """Latin style per building (deterministic)."""
    B = city["buildings"]
    ids = {b["id"]: i for i, b in enumerate(B)}
    names = [n for n, _, _ in pal.WALLS]
    wts = np.array([w for _, _, w in pal.WALLS])
    out = [None] * len(B)
    for i, b in enumerate(B):
        r = np.random.default_rng([seed, i, 7])
        near = {out[ids[j]]["wall"] for j in b["style"].get("attached_to", []) if j in ids and out[ids[j]] is not None}
        w = np.where([n in near for n in names], 0.0, wts)
        f0 = max(1, int(b["floors"]))
        f = min(f0, _choice(r, [1, 2, 3, 4], [0.22, 0.45, 0.25, 0.08])) if f0 <= 4 else _choice(r, [2, 3, 4, 5], [0.2, 0.35, 0.3, 0.15])
        gf, up = float(r.uniform(3.8, 4.3)), float(r.uniform(3.3, 3.7))
        attached = bool(b["style"].get("attached_to"))
        rect = None if attached else rect_footprint(foot[i], RECT_TOL)
        if rect is not None:
            kind = _choice(r, ["hip", "gable", "flat"], [0.55, 0.3, 0.15])
        else:
            kind = _choice(r, ["skirt", "flat"], [0.6, 0.4] if f >= 4 else [0.87, 0.13])
        slope = float(np.tan(np.radians(r.uniform(22, 30))))
        short = 0.0
        if rect is not None:
            xy = np.asarray(rect.exterior.coords)[:4]
            short = float(min(np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])))
        wall = _choice(r, names, w)
        plinths = [p for p, _ in pal.PLINTHS]
        woods = [x for x, _ in pal.WOODS]
        wood = _choice(r, woods, [0.25, 0.15, 0.15, 0.15, 0.12, 0.12, 0.06])
        out[i] = {
            "wall": wall, "floors": f, "gf": gf, "up": up, "eave": gf + (f - 1) * up, "roof_kind": kind,
            "rise": short / 2 * slope, "slope": slope, "rect": rect, "overhang": float(r.uniform(0.45, 0.65)),
            "skirt_w": float(np.clip(foot[i].area / max(foot[i].length, 1e-6) * 1.6, 1.4, 3.6)),
            "tiles": _choice(r, ["teja", "teja_vieja"], [0.65, 0.35]),
            "trim": _choice(r, [t for t, _, _ in pal.TRIMS], [t[2] for t in pal.TRIMS]),
            "plinth": _choice(r, plinths, [1] * len(plinths)) if r.uniform() < 0.7 else None,
            "plinth_h": float(r.uniform(0.8, 1.2)), "wood": wood,
            "door_wood": wood if r.uniform() < 0.6 else _choice(r, ["madera", "madera_oscura"], [1, 1]),
            "balcony": (_choice(r, [district(b.get("block", ""), seed), other(district(b.get("block", ""), seed)), None],
                                [0.65, 0.1, 0.25]) if f >= 2 else None),
            "iron_all": r.uniform() < 0.5, "shutters": r.uniform() < 0.7, "reja": f <= 3 and r.uniform() < 0.55,
            "reja_roof": r.uniform() < 0.5, "pilasters": r.uniform() < 0.75, "cornice": r.uniform() < 0.8,
            "string_course": f >= 2 and r.uniform() < 0.6, "arched_door": r.uniform() < 0.35,
            "bay": float(r.uniform(2.8, 3.5)),
        }
    return out


def _walls(poly):
    out = []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)[:-1]
        for i in range(len(c)):
            a, e = c[i], c[(i + 1) % len(c)]
            L = np.hypot(*(e - a))
            if L > 1e-6:
                out.append((a, e, np.array([e[1] - a[1], -(e[0] - a[0])]) / L))
    return out


def build_latin(city: dict, facades: bool = True, environment: bool = True):
    from urbanmorph.city3d.materials import MaterialTable
    from urbanmorph.city3d.scene import Group, Z_BASE, Z_ROAD
    mt = MaterialTable()
    M = lambda k: mt.get("latin:" + k)
    names = ("walls", "roofs", "trim", "glass", "doors", "carpentry", "metal", "roads", "green", "ground", "sidewalks", "markings")
    groups = {k: Group() for k in names}
    seed = int(city["metadata"].get("seed") or 0)
    road = (shapely.union_all([shapely.make_valid(polygon(r)) for r in city["road_surface"]]) if city["road_surface"] else shapely.Polygon())
    foot = []
    for b in city["buildings"]:
        p = clean_footprint(shapely.Polygon(b["footprint"][0], b["footprint"][1:]))
        foot.append(p if p is not None else shapely.orient_polygons(shapely.Polygon(b["footprint"][0])))
    tree = shapely.STRtree(foot) if foot else None
    sty = styles(city, foot, seed)
    opts = city["metadata"].get("theme_options") or {}
    sites = []
    for i, b in enumerate(city["buildings"]):                     # planned churches (plaza heads, latin.infill)
        cs = b.get("church_site")
        if cs:
            sites.append((i, [], np.asarray(cs["center"], float), np.asarray(cs["axis"], float), float(cs["L"]), float(cs["W"])))
    if not sites or opts.get("infill") == "colonial":
        planned = [s_[2] for s_ in sites]
        extra = pick_churches(city, foot, road, seed + 1 if sites else seed)
        sites += [x for x in extra if all(np.hypot(*(x[2] - c)) > CHURCH_SPACING_M for c in planned)]
    absorbed = {j for s in sites for j in s[1]}
    church_of = {s[0]: s for s in sites}
    rep = {"buildings": [], "summary": {}, "theme": THEME, "churches": []}
    fm = new_metrics()
    counts = {"windows": 0, "doors": 0, "balconies": 0}
    for i, b in enumerate(city["buildings"]):
        if i in absorbed:
            continue
        S = Soup()
        if i in church_of:
            _, _, c, ax, L, W = church_of[i]
            r = np.random.default_rng([seed, i, 11])
            colour = _choice(r, [n for n, _ in pal.CHURCH_WALLS], [0.35, 0.25, 0.15, 0.15, 0.1])
            dk = _choice(r, [n for n, _, _ in pal.DOMES], [0.5, 0.25, 0.15, 0.1])
            S, info = church(c, ax, L, W, r, colour, dk)
            info.update(building=b["id"], absorbed=[city["buildings"][j]["id"] for j in church_of[i][1]])
            rep["churches"].append(info)
            V = np.vstack([v for lst in S.parts.values() for v, _, _ in lst])
            rep["buildings"].append({"id": b["id"], "church": True, "min_z": float(V[:, 2].min()), "watertight": True,
                                     "outward": True, "nonmanifold_edges": 0, "degenerate_tris": 0, "roof": "church",
                                     "roof_as_specified": True})
        else:
            st = sty[i]
            poly = st["rect"] if st["roof_kind"] in ("hip", "gable") and st["rect"] is not None else foot[i]
            V, F = walls_and_roof(S, b, st, foot[i], st["rect"] if poly is st["rect"] else None)
            r_ = solid_report(V, F)
            r_.update(id=b["id"], roof=st["roof_kind"], roof_as_specified=True)
            rep["buildings"].append(r_)
            if facades:
                near = [foot[j] for j in tree.query(foot[i].buffer(1.0)) if j != i]
                nb = shapely.union_all(near) if near else None
                walls = _walls(poly)
                runs = [visible_runs(a, e, n, nb) for a, e, n in walls]
                street = [bool(not road.is_empty and road.distance(shapely.Point((a + e) / 2 + n * 3.0)) < 4.0) for a, e, n in walls]
                usable = [max((r1 - r0 - 1.6 for r0, r1 in rr), default=0.0) for rr in runs]
                cand = [k for k in range(len(walls)) if usable[k] >= 2.4]
                dw = max(cand, key=lambda k: (street[k], usable[k])) if cand else None
                c = house(S, b, st, walls, runs, street, dw)
                for k in counts:
                    counts[k] += c[k]
        for key, lst in S.parts.items():
            g = groups[group_of(key)]
            m = M(key)
            for V, F, UV in lst:
                g.add(V, F, m, i, UV)
    # streets: asphalt on major / mid streets, cobbles elsewhere
    big = [shapely.LineString(np.asarray(s["centerline"], float)).buffer(max(s["segment_widths_m"]) / 2 + 0.6, cap_style="flat")
           for s in city["streets"] if s["class"] == "major"]
    major = road.intersection(shapely.union_all(big)) if big and not road.is_empty else shapely.Polygon()
    for geom, mat in ((major, "asphalt"), (road.difference(major) if not major.is_empty else road, "cobble")):
        V, F = surface(geom, Z_ROAD)
        groups["roads"].add(V, F, M(mat))
    V, F = surface(road, Z_ROAD)
    rep["roads"] = surface_report(V, F, road.area)
    bu = shapely.union_all(foot) if foot else shapely.Polygon()
    sw, plaza, free, instances = shapely.Polygon(), shapely.Polygon(), shapely.Polygon(), {}
    if environment:
        rng = np.random.default_rng(seed + 11)
        sw = env.sidewalks(city, road, bu, COLONIAL_WALKS if opts.get("infill") == "colonial" else None)
        for SV, SF in env.sidewalk_solids(sw):
            groups["sidewalks"].add(SV, SF, M("pavement"))
        MV, MF, mrep = env.markings(city, road)
        if len(MF) and not major.is_empty:
            cen = MV[MF].mean(1)
            MF = MF[shapely.contains_xy(major.buffer(0.3), cen[:, 0], cen[:, 1])]
            groups["markings"].add(MV, MF, M("marking"))
        instances, plaza, free = place(city, road, sw, bu, rng, opts=opts)
    pv = shapely.union_all([shapely.make_valid(polygon(x["polygon"])) for x in city["landuse"] if x["class"] == "paved"] or [shapely.Polygon()])
    if not pv.is_empty:                                           # colonial infill: sidewalk edges and leftovers paved
        V, F = surface(pv.difference(road).difference(sw).difference(bu), Z_ROAD)
        groups["green"].add(V, F, M("pavement"))
    green_area = 0.0
    for cl in ("public_green", "private_green"):
        g = shapely.union_all([shapely.make_valid(polygon(x["polygon"])) for x in city["landuse"] if x["class"] == cl] or [shapely.Polygon()])
        g = g.difference(road).difference(sw)
        if cl == "public_green" and not plaza.is_empty:
            pz = g.intersection(plaza)
            V, F = surface(pz, Z_ROAD)
            groups["green"].add(V, F, M("plaza"))
            g = g.difference(plaza)
        V, F = surface(g, Z_ROAD)
        groups["green"].add(V, F, M("grass_public" if cl == "public_green" else "grass_private"))
        green_area += g.area
    jf = free.buffer(6.0).difference(road).difference(sw).difference(bu) if not free.is_empty else free
    for x in city["landuse"]:
        if x["class"] in ("public_green", "private_green"):
            jf = jf.difference(shapely.make_valid(polygon(x["polygon"])))
    V, F = surface(jf, Z_ROAD)
    groups["green"].add(V, F, M("jungle_floor"))
    base = np.array([[0, 0, Z_BASE], [512, 0, Z_BASE], [512, 512, Z_BASE], [0, 512, Z_BASE]], float)
    groups["ground"].add(base, [[0, 1, 2], [0, 2, 3]], M("ground"))
    out = {k: g.arrays() for k, g in groups.items()}
    protos = prototypes()
    for name, (PV, PF, part, PUV, mats) in protos.items():
        T = instances.get(name)
        if T is None or not len(T):
            continue
        mi = np.array([M(m) for m in mats], np.int16)
        out[f"proto_{name}"] = {"V": PV.astype(np.float32), "F": PF.astype(np.int32), "mat": mi[part],
                                "obj": np.full(len(PF), -1, np.int32), "UV": PUV.astype(np.float32)}
        out[f"inst_{name}"] = {"T": T.astype(np.float32)}
    B = rep["buildings"]
    rep["facade"] = {**counts, "parapets": 0}
    rep["summary"] = {
        "n_buildings": len(B), "watertight": sum(bool(b["watertight"]) for b in B), "outward": sum(bool(b["outward"]) for b in B),
        "floating": sum(b["min_z"] > 1e-6 for b in B), "churches": len(rep["churches"]), "absorbed_by_churches": len(absorbed),
        "roofs": {k: sum(b["roof"] == k for b in B) for k in ("hip", "gable", "skirt", "flat", "church")},
        "green_area_m2": float(green_area), "plaza_area_m2": float(plaza.area),
        "triangles": {k: int(len(a["F"])) for k, a in out.items() if "F" in a and not k.startswith("proto_")},
        **{f"instances_{k}": int(len(v)) for k, v in instances.items()},
        "proto_triangles": {k: int(len(v[1])) for k, v in protos.items()},
    }
    return out, mt.params, rep


# ----------------------------------------------------------------------------------------------- on terrain
BEACH_TOP_M = 2.8               # sand up to this height above the sea ...
BEACH_REACH_M = 90.0            # ... and within this distance of it
ROCK_DEG = 38.0
JUNGLE_KINDS = ("inst_jungle_tree", "inst_jungle_tree_dark", "inst_shrub", "inst_banana", "inst_palm_fan",
                "inst_canopy", "inst_canopy_dark")


def _add_mat(mats: list, key: str) -> int:
    from urbanmorph.city3d.materials import MaterialTable
    name = "latin:" + key
    for k, p in enumerate(mats):
        if p["name"] == name:
            return k
    mt = MaterialTable()
    mt.get(name)
    mats.append(mt.params[0])
    return len(mats) - 1


def dress_terrain(out: dict, mats: list, city: dict, t, seed: int = 0) -> dict:
    """After terrain3d.apply_terrain: beach sand and cliff rock on the terrain mesh, Caribbean water, no jungle on the
    sand, coconut palms leaning over the beach. Returns counts."""
    from scipy import ndimage
    from urbanmorph.city3d.terrain3d import Surfaces
    S = Surfaces(t)
    mats = mats
    for p in mats:
        if p["name"] == "water":
            p.update({k: v for k, v in pal.material("water").items()})
    wk = getattr(t, "water_kind", None)
    sea = (wk == 3) if wk is not None else None
    rep = {"sand_faces": 0, "rock_faces": 0, "beach_palms": 0, "jungle_removed_on_sand": 0}
    if sea is None or not sea.any():
        near = lambda x, y: np.zeros(len(x), bool)
        sea_z = 0.0
    else:
        dist = ndimage.distance_transform_edt(~sea) * t.res
        sea_z = float(np.nanmedian(t.water_surface[sea])) if getattr(t, "water_surface", None) is not None else 0.0

        def near(x, y):
            r, c = t.rc(x, y)
            r = np.clip(np.round(r).astype(int), 0, dist.shape[0] - 1)
            c = np.clip(np.round(c).astype(int), 0, dist.shape[1] - 1)
            return dist[r, c] < BEACH_REACH_M
    m_sand, m_rock = _add_mat(mats, "sand"), _add_mat(mats, "rock")
    g = out.get("ground")
    if g is not None and len(g["F"]):
        P = g["V"][g["F"]].astype(float)
        cen = P.mean(1)
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        steep = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-9), 0, 1))) > ROCK_DEG
        beach = (cen[:, 2] < sea_z + BEACH_TOP_M) & near(cen[:, 0], cen[:, 1])
        g["mat"] = np.where(beach, m_sand, np.where(steep, m_rock, g["mat"])).astype(np.int16)
        rep["sand_faces"], rep["rock_faces"] = int(beach.sum()), int((steep & ~beach).sum())
    jf = _add_mat(mats, "jungle_floor")
    gr = out.get("green")
    if gr is not None and len(gr["F"]):                          # no jungle floor on the beach or on rivers and lakes
        from urbanmorph.terrain.road_feasibility import water_kind_at
        cen = gr["V"][gr["F"]].astype(float).mean(1)
        drop = (gr["mat"] == jf) & (cen[:, 2] < sea_z + BEACH_TOP_M + 0.3) & near(cen[:, 0], cen[:, 1])
        if wk is not None:
            drop |= water_kind_at(t, cen[:, 0], cen[:, 1]) > 0
        for k in ("F", "mat", "obj", "UV"):
            gr[k] = gr[k][~drop]
    for name in JUNGLE_KINDS:
        if name in out:
            T = out[name]["T"]
            onsand = (S.ground(T[:, 0], T[:, 1]) < sea_z + BEACH_TOP_M) & near(T[:, 0], T[:, 1])
            out[name] = {"T": T[~onsand]}
            rep["jungle_removed_on_sand"] += int(onsand.sum())
    if sea is not None and sea.any():                             # palms along the beach
        r = np.random.default_rng(seed + 31)
        pts = []
        for y in np.arange(0, 512, 9.0):
            for x in np.arange(0, 512, 9.0):
                px, py = x + r.uniform(0, 9), y + r.uniform(0, 9)
                z = float(S.ground(np.array([px]), np.array([py]))[0])
                if sea_z + 0.8 < z < sea_z + BEACH_TOP_M + 1.5 and near(np.array([px]), np.array([py]))[0] and r.uniform() < 0.55:
                    pts.append([px, py, z - 0.1, float(r.uniform(0.85, 1.25)), float(r.uniform(0, 2 * np.pi))])
        if pts and wk is not None:
            from urbanmorph.terrain.road_feasibility import water_kind_at
            P = np.asarray(pts, np.float32)
            P = P[water_kind_at(t, P[:, 0], P[:, 1]) == 0]
            if "proto_palm_coconut_lean" not in out:
                PV, PF, part, PUV, mk = prototypes()["palm_coconut_lean"]
                mi = np.array([_add_mat(mats, m) for m in mk], np.int16)
                out["proto_palm_coconut_lean"] = {"V": PV.astype(np.float32), "F": PF.astype(np.int32), "mat": mi[part],
                                                  "obj": np.full(len(PF), -1, np.int32), "UV": PUV.astype(np.float32)}
            old = out.get("inst_palm_coconut_lean", {"T": np.zeros((0, 5), np.float32)})["T"]
            out["inst_palm_coconut_lean"] = {"T": np.vstack([old, P]).astype(np.float32)}
            rep["beach_palms"] = int(len(P))
    return rep
