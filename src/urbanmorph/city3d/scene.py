"""City format -> batched render meshes + validation report (docs/M11_3D_ARCHITECTURE.md §4).

Vertical layout (no coplanar overlaps, hence no z-fighting, and no cracks):
  base plane  z = -0.02  whole tile, 'ground' material (shows wherever nothing else is drawn)
  roads       z =  0.00  one triangulated surface (the union of all street pieces)
  green       z =  0.00  public / private green minus roads (never overlaps roads)
  buildings   from z = -0.05 (sunk into the base plane) to the eave, plus roof
Groups: one mesh per material class, each face with a material index and a building index (-1 = none).
"""

from __future__ import annotations

import json

import numpy as np
import shapely

from urbanmorph.city3d import environment as env
from urbanmorph.city3d.facade import Parts, facade, new_metrics, walls_of
from urbanmorph.city3d.format import polygon, polygons
from urbanmorph.city3d.materials import ROOF, MaterialTable
from urbanmorph.city3d.mesh import BOTTOM, GABLE, ROOF as ROOF_PART, WALL, building_solid, surface
from urbanmorph.city3d.validate import overlap_pairs, solid_report, surface_report

Z_BASE, Z_ROAD, Z_BUILDING = -0.02, 0.0, -0.05


def corner_uvs(V, F) -> np.ndarray:
    """Per-corner UVs in metres (m, 3, 2): near-horizontal faces use (x, y); walls use (distance along the wall's
    horizontal direction, z), so procedural textures keep real-world scale on every face."""
    if not len(F):
        return np.zeros((0, 3, 2), np.float32)
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    flat = np.abs(n[:, 2]) > 0.7
    d = np.c_[-n[:, 1], n[:, 0]]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    u_wall = np.einsum("fcj,fj->fc", P[:, :, :2], d)
    uv = np.where(flat[:, None, None], P[:, :, :2], np.stack([u_wall, P[:, :, 2]], -1))
    return uv.astype(np.float32)


class Group:
    def __init__(self):
        self.V, self.F, self.mat, self.obj, self.n = [], [], [], [], 0

    def add(self, V, F, mat, obj=-1):
        if not len(F):
            return
        self.V.append(np.asarray(V, np.float32))
        self.F.append(np.asarray(F, np.int64) + self.n)
        self.mat.append(np.broadcast_to(np.asarray(mat, np.int16), (len(F),)).copy())
        self.obj.append(np.full(len(F), obj, np.int32))
        self.n += len(V)

    def arrays(self) -> dict:
        if not self.F:
            return {"V": np.zeros((0, 3), np.float32), "F": np.zeros((0, 3), np.int32), "mat": np.zeros(0, np.int16),
                    "obj": np.zeros(0, np.int32), "UV": np.zeros((0, 3, 2), np.float32)}
        V, F = np.vstack(self.V), np.vstack(self.F).astype(np.int32)
        return {"V": V, "F": F, "mat": np.concatenate(self.mat), "obj": np.concatenate(self.obj), "UV": corner_uvs(V.astype(float), F)}


FACADE_GROUPS = {"glass": "glass", "frames": "window_frame", "doors": "door", "balconies": "balcony", "railings": "metal", "trim": "trim"}


PROTO_MATS = {"tree": ("tree_trunk", "tree_leaves"), "streetlight": ("metal", "lamp")}


def build_scene(city: dict, facades: bool = True, environment: bool = True) -> tuple[dict, list, dict]:
    """(groups {name: arrays}, material params, validation report)."""
    mt = MaterialTable()
    groups = {k: Group() for k in ("walls", "roofs", "roads", "green", "ground", "sidewalks", "markings", *FACADE_GROUPS)}
    road = shapely.union_all([polygon(r) for r in city["road_surface"]]) if city["road_surface"] else shapely.Polygon()
    foot = [walls_of(b)[0] for b in city["buildings"]]          # the footprint the solid actually stands on
    tree = shapely.STRtree(foot) if foot else None
    fm = new_metrics()
    storey = (city["metadata"].get("storey_height_m") or {}).get("value", 3.0)
    parapet_reports = []
    rep = {"buildings": [], "summary": {}}
    for i, b in enumerate(city["buildings"]):
        s, as_spec = building_solid(b, Z_BUILDING)
        V, F, part = s.arrays()
        r = solid_report(V, F)
        r.update(id=b["id"], roof=b["roof"]["type"], roof_as_specified=as_spec)
        rep["buildings"].append(r)
        st = b["style"]
        wall = mt.get(f"facade:{st['facade_family']}:{st['palette']}")
        fam = "tile" if b["roof"]["type"] != "flat" and as_spec else "flat"
        roof = mt.get(f"roof:{fam}:{(st['palette'] + i) % len(ROOF[fam])}")
        w = np.isin(part, (WALL, GABLE, BOTTOM))
        groups["walls"].add(V, F[w], wall, i)
        groups["roofs"].add(V, F[part == ROOF_PART], roof, i)
        if facades:
            near = [foot[j] for j in tree.query(foot[i].buffer(1.0)) if j != i]
            nb = shapely.union_all(near) if near else None
            parts = Parts()
            d0 = fm["doors"]
            facade(b, nb, road, storey, parts, fm)
            r["doors"] = fm["doors"] - d0
            for kind, mat in FACADE_GROUPS.items():
                PV, PF = parts.arrays(kind)
                groups[kind].add(PV, PF, mt.get(mat), i)
            PV, PF = parts.arrays("parapet")
            if len(PF):
                parapet_reports.append(solid_report(PV, PF))
                groups["walls"].add(PV, PF, wall, i)
    V, F = surface(road, Z_ROAD)
    groups["roads"].add(V, F, mt.get("asphalt"))
    rep["roads"] = surface_report(V, F, road.area)
    rep["roads"]["parts"] = len(polygons(road))
    bu = shapely.union_all(foot) if foot else shapely.Polygon()
    sw = shapely.Polygon()
    instances = {}
    if environment:
        rng = np.random.default_rng((city["metadata"].get("seed") or 0) + 11)
        sw = env.sidewalks(city, road, bu)
        sw_rep = []
        for SV, SF in env.sidewalk_solids(sw):
            sw_rep.append(solid_report(SV, SF))
            groups["sidewalks"].add(SV, SF, mt.get("pavement"))
        MV, MF, mrep = env.markings(city, road)
        groups["markings"].add(MV, MF, mt.get("marking"))
        trees, lights = env.trees_and_lights(city, road, sw, bu, rng)
        instances = {"tree": trees, "streetlight": lights}
        rad = {"tree": 0.15, "streetlight": 0.08}
        clash = {k: int(sum(bu.distance(shapely.Point(x[0], x[1])) < rad[k] for x in v)) for k, v in instances.items()} if foot else {}
        on_road = {k: int(sum(road.contains(shapely.Point(x[0], x[1])) for x in v)) for k, v in instances.items()}
        canopy = int(sum(bu.distance(shapely.Point(x[0], x[1])) < env.CANOPY_R * x[3] * 0.8 for x in trees)) if foot else 0
        rep["environment"] = {"sidewalk_area_m2": float(sw.area), "sidewalk_solids": len(sw_rep),
                              "sidewalk_watertight": int(sum(r["watertight"] and r["outward"] for r in sw_rep)),
                              "sidewalk_on_buildings_m2": float(sw.intersection(bu).area), "sidewalk_on_road_m2": float(sw.intersection(road).area),
                              **mrep, "trees": len(trees), "streetlights": len(lights),
                              "trunk_or_pole_inside_building": clash, "props_on_road": on_road,
                              "canopy_within_80pct_radius_of_building": canopy}
    green_area = 0.0
    for cl in ("public_green", "private_green"):
        g = shapely.union_all([polygon(x["polygon"]) for x in city["landuse"] if x["class"] == cl] or [shapely.Polygon()])
        g = g.difference(road).difference(sw)
        V, F = surface(g, Z_ROAD)
        groups["green"].add(V, F, mt.get("grass_public" if cl == "public_green" else "grass_private"))
        green_area += g.area
    base = np.array([[0, 0, Z_BASE], [512, 0, Z_BASE], [512, 512, Z_BASE], [0, 512, Z_BASE]], float)
    groups["ground"].add(base, [[0, 1, 2], [0, 2, 3]], mt.get("ground"))
    # ---- summary
    B = rep["buildings"]
    ov = overlap_pairs(foot)
    bw = np.asarray(fm.pop("bay_widths"))
    doors = np.asarray(fm.pop("door_front_xy")).reshape(-1, 2)
    if len(doors):
        on_sw = shapely.contains_xy(sw, doors[:, 0], doors[:, 1]) if not sw.is_empty else np.zeros(len(doors), bool)
        on_rd = shapely.contains_xy(road, doors[:, 0], doors[:, 1]) if not road.is_empty else np.zeros(len(doors), bool)
        fm["doors_facing_sidewalk"] = int(on_sw.sum())
        fm["doors_opening_onto_road"] = int((on_rd & ~on_sw).sum())
    dz = np.asarray(fm.pop("door_bottom_z"))
    rep["facade"] = {**fm, "bay_width_m_quantiles": [float(x) for x in np.quantile(bw, [0.1, 0.5, 0.9])] if len(bw) else None,
                     "door_bottom_z_max_abs": float(np.abs(dz).max()) if len(dz) else None,
                     "buildings_with_door": int(sum(b.get("doors", 0) > 0 for b in B)), "buildings_with_multiple_doors": int(sum(b.get("doors", 0) > 1 for b in B)),
                     "parapets_watertight": int(sum(p["watertight"] and p["outward"] for p in parapet_reports)), "parapet_solids": len(parapet_reports)}
    on_road = [f.intersection(road).area for f in foot] if not road.is_empty else [0.0] * len(foot)
    rep["summary"] = {
        "n_buildings": len(B), "watertight": sum(b["watertight"] for b in B), "outward": sum(b["outward"] for b in B),
        "nonmanifold_edges": sum(b["nonmanifold_edges"] for b in B), "degenerate_tris": sum(b["degenerate_tris"] for b in B),
        "floating": sum(b["min_z"] > 1e-6 for b in B), "roof_fallback_to_flat": sum(not b["roof_as_specified"] for b in B),
        "missing_roof": int((np.bincount(groups["roofs"].arrays()["obj"], minlength=len(B)) == 0).sum()) if B else 0,
        "footprint_overlap_pairs_gt_0.5m2": len(ov), "footprint_overlap_area_m2": float(sum(a for *_, a in ov)),
        "buildings_on_road_gt_1m2": int(sum(a > 1.0 for a in on_road)), "building_area_on_road_m2": float(sum(on_road)),
        "road_coverage": rep["roads"]["coverage"], "road_down_facing": rep["roads"]["down_facing"], "green_area_m2": float(green_area),
        "triangles": {k: int(len(g.arrays()["F"])) for k, g in groups.items()},
        "height_sources": {s: sum(b["height_source"] == s for b in city["buildings"]) for s in ("observed", "derived", "synthetic")},
        "roofs": {t: sum(b["roof"]["type"] == t for b in city["buildings"]) for t in ("flat", "gable", "hip")},
        "height_m_quantiles": [float(x) for x in np.quantile([b["height_m"] for b in city["buildings"]], [0.1, 0.5, 0.9])] if B else None,
    }
    out = {k: g.arrays() for k, g in groups.items()}
    for name, (PV, PF, part) in env.prototypes().items():
        if not len(instances.get(name, [])):
            continue
        m0, m1 = (mt.get(x) for x in PROTO_MATS[name])
        out[f"proto_{name}"] = {"V": PV.astype(np.float32), "F": PF.astype(np.int32), "mat": np.where(part == 0, m0, m1).astype(np.int16),
                                "obj": np.full(len(PF), -1, np.int32), "UV": corner_uvs(PV, PF.astype(np.int32))}
        out[f"inst_{name}"] = {"T": instances[name].astype(np.float32)}
        rep["summary"]["triangles"][f"proto_{name}"] = int(len(PF))
        rep["summary"]["instances_" + name] = int(len(instances[name]))
    return out, mt.params, rep


def save_scene(path, groups: dict, materials: list) -> None:
    flat = {f"{g}__{k}": v for g, a in groups.items() for k, v in a.items()}
    np.savez_compressed(path, materials=np.array(json.dumps(materials)), **flat)
