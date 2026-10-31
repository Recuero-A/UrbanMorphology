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


def build_scene(city: dict) -> tuple[dict, list, dict]:
    """(groups {name: arrays}, material params, validation report)."""
    mt = MaterialTable()
    groups = {k: Group() for k in ("walls", "roofs", "roads", "green", "ground")}
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
    road = shapely.union_all([polygon(r) for r in city["road_surface"]]) if city["road_surface"] else shapely.Polygon()
    V, F = surface(road, Z_ROAD)
    groups["roads"].add(V, F, mt.get("asphalt"))
    rep["roads"] = surface_report(V, F, road.area)
    rep["roads"]["parts"] = len(polygons(road))
    green_area = 0.0
    for cl in ("public_green", "private_green"):
        g = shapely.union_all([polygon(x["polygon"]) for x in city["landuse"] if x["class"] == cl] or [shapely.Polygon()])
        g = g.difference(road)
        V, F = surface(g, Z_ROAD)
        groups["green"].add(V, F, mt.get("grass_public" if cl == "public_green" else "grass_private"))
        green_area += g.area
    base = np.array([[0, 0, Z_BASE], [512, 0, Z_BASE], [512, 512, Z_BASE], [0, 512, Z_BASE]], float)
    groups["ground"].add(base, [[0, 1, 2], [0, 2, 3]], mt.get("ground"))
    # ---- summary
    B = rep["buildings"]
    foot = [polygon(b["footprint"]) for b in city["buildings"]]
    ov = overlap_pairs(foot)
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
    return {k: g.arrays() for k, g in groups.items()}, mt.params, rep


def save_scene(path, groups: dict, materials: list) -> None:
    flat = {f"{g}__{k}": v for g, a in groups.items() for k, v in a.items()}
    np.savez_compressed(path, materials=np.array(json.dumps(materials)), **flat)
