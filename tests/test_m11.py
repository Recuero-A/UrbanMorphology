"""Milestone 11: structured city format, footprint triangulation and extrusion, roofs, mesh validity, heights,
materials, scene assembly, determinism, and (if Blender is installed) the GLB coordinate conversion."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import shapely

from test_m10 import classed_grid
from urbanmorph.city3d.assemble import street_surface, streets_from_graph
from urbanmorph.city3d.cameras import cameras
from urbanmorph.city3d.format import new_city, polygon, rings, validate_city
from urbanmorph.city3d.heights import QS, assign_heights
from urbanmorph.city3d.materials import MaterialTable, srgb
from urbanmorph.city3d.mesh import clean_footprint, pitched, prism, rect_footprint, triangulate
from urbanmorph.city3d.scene import build_scene, corner_uvs, save_scene
from urbanmorph.city3d.validate import solid_report

BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
L_SHAPE = shapely.Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
U_SHAPE = shapely.Polygon([(0, 0), (12, 0), (12, 10), (8, 10), (8, 4), (4, 4), (4, 10), (0, 10)])
COURT = shapely.Polygon([(0, 0), (30, 0), (30, 30), (0, 30)], [[(10, 10), (10, 20), (20, 20), (20, 10)]])


@pytest.mark.parametrize("poly", [shapely.box(0, 0, 20, 10), L_SHAPE, U_SHAPE, COURT])
def test_extrusion_is_watertight_outward_with_exact_volume(poly):
    V, F, part = prism(clean_footprint(poly), 0.0, 9.0).arrays()
    r = solid_report(V, F)
    assert r["watertight"] and r["outward"] and r["nonmanifold_edges"] == 0 and r["degenerate_tris"] == 0
    assert abs(r["volume_m3"] - poly.area * 9.0) < 1e-6
    assert set(np.unique(part)) == {0, 1, 2}


def test_triangulation_keeps_holes_and_concavity_without_new_vertices():
    for poly in (L_SHAPE, U_SHAPE, COURT):
        xy, tris = triangulate(shapely.orient_polygons(poly))
        assert len(xy) == sum(len(r.coords) - 1 for r in [poly.exterior, *poly.interiors])
        a = xy[tris]
        cross = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0])
        assert (cross > 0).all()                                    # counter-clockwise seen from above
        assert abs(cross.sum() / 2 - poly.area) < 1e-6


@pytest.mark.parametrize("kind,roof_volume", [("gable", 10 * 2.5 / 2 * 20), ("hip", 10 * 2.5 / 2 * 10 + 10 * 10 * 2.5 / 3)])
def test_pitched_roofs_close_the_solid(kind, roof_volume):
    rect = rect_footprint(shapely.box(0, 0, 20, 10))
    V, F, _ = pitched(rect, 0.0, 6.0, 2.5, kind).arrays()
    r = solid_report(V, F)
    assert r["watertight"] and r["outward"]
    assert abs(r["volume_m3"] - (200 * 6 + roof_volume)) < 1e-6
    assert abs(V[:, 2].max() - 8.5) < 1e-9                          # ridge at eave + rise: no floating roof


def test_hip_on_square_is_a_pyramid():
    V, F, _ = pitched(rect_footprint(shapely.box(0, 0, 10, 10.2)), 0.0, 6.0, 3.0, "hip").arrays()
    r = solid_report(V, F)
    assert r["watertight"] and abs(r["volume_m3"] - (102 * 6 + 102 * 3 / 3)) < 1e-6


def test_rect_footprint_accepts_rotated_rectangles_and_rejects_l_shapes():
    rot = shapely.affinity.rotate(shapely.box(0, 0, 8, 20), 33)
    r = rect_footprint(rot)
    xy = np.asarray(r.exterior.coords)
    assert np.linalg.norm(xy[1] - xy[0]) >= np.linalg.norm(xy[2] - xy[1])     # long edge first
    assert rect_footprint(L_SHAPE) is None and rect_footprint(COURT) is None


def test_clean_footprint_drops_slivers_and_repairs_bowties():
    assert clean_footprint(shapely.box(0, 0, 2, 2)) is None                   # 4 m2 < 10 m2
    bow = shapely.Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    p = clean_footprint(bow)
    assert p is not None and p.is_valid


def test_rings_orientation_and_round_trip():
    rs = rings(COURT)
    assert shapely.LinearRing(rs[0]).is_ccw and not shapely.LinearRing(rs[1]).is_ccw
    assert polygon(rs).equals(COURT)


def test_streets_cover_every_graph_edge_once_with_class_widths():
    g = classed_grid()
    streets, nodes = streets_from_graph(g, [10.0, 8.0, 6.0, 4.0])
    assert sum(len(s["centerline"]) - 1 for s in streets) == len(g.edges)
    assert all(len(s["segment_widths_m"]) == len(s["centerline"]) - 1 for s in streets)
    ids = {n["id"] for n in nodes}
    assert all(set(s["nodes"]) <= ids for s in streets)
    surf = street_surface(streets)
    assert surf.is_valid and surf.area > 0


def tiny_city(seed=0) -> dict:
    """Four buildings (rectangle, gable rectangle, L, courtyard), one street, one green."""
    c = new_city("tiny", "generated", seed=seed)
    c["nodes"] = [{"id": "n0", "xy": [0, 100], "degree": 1, "border": True}, {"id": "n1", "xy": [512, 100], "degree": 1, "border": True}]
    c["streets"] = [{"id": "s0", "centerline": [[0, 100], [512, 100]], "segment_widths_m": [10.0], "width_source": "class_default",
                     "class": "local", "segment_classes": ["local"], "nodes": ["n0", "n1"]}]
    c["road_surface"] = [rings(street_surface(c["streets"]))]
    foot = [shapely.box(10, 120, 30, 130), shapely.box(40, 120, 60, 130), shapely.affinity.translate(L_SHAPE, 80, 120),
            shapely.affinity.translate(COURT, 120, 120)]
    roof = [("flat", 0), ("gable", 2.5), ("flat", 0), ("flat", 0)]
    for i, (p, (t, rise)) in enumerate(zip(foot, roof)):
        c["buildings"].append({"id": f"B{i}", "footprint": rings(p), "parcels": [], "height_m": 9.0 + rise, "eave_m": 9.0,
                               "height_source": "synthetic", "floors": 3, "floors_source": "synthetic",
                               "roof": {"type": t, "pitch_deg": 26.6 if rise else 0.0, "rise_m": rise, "source": "synthetic"},
                               "style": {"facade_family": "plaster", "palette": i % 3, "source": "synthetic"}})
    c["landuse"] = [{"id": "g0", "polygon": rings(shapely.box(200, 200, 260, 240)), "class": "public_green", "source": "generated"}]
    return c


def test_city_validation_flags_invented_geography():
    c = tiny_city()
    assert validate_city(c) == []
    c["metadata"]["crs"] = "EPSG:25831"
    assert "generated city has geographic coordinates" in validate_city(c)


def test_scene_is_valid_and_deterministic(tmp_path):
    g1, m1, r1 = build_scene(tiny_city())
    g2, m2, r2 = build_scene(tiny_city())
    s = r1["summary"]
    assert s["watertight"] == s["n_buildings"] == 4 and s["floating"] == 0 and s["missing_roof"] == 0
    assert s["buildings_on_road_gt_1m2"] == 0 and s["roof_fallback_to_flat"] == 0
    assert abs(s["road_coverage"] - 1) < 1e-6 and s["road_down_facing"] == 0
    for k in g1:
        for a in g1[k]:
            assert np.array_equal(g1[k][a], g2[k][a])
    assert m1 == m2
    save_scene(tmp_path / "m.npz", g1, m1)                         # export consistency: npz round trip
    d = np.load(tmp_path / "m.npz")
    assert np.array_equal(d["walls__F"], g1["walls"]["F"]) and json.loads(str(d["materials"])) == m1


def test_material_assignment_is_by_name_and_linear():
    t = MaterialTable()
    a, b = t.get("facade:brick:1"), t.get("asphalt")
    assert t.get("facade:brick:1") == a and a != b
    assert srgb("#ffffff") == [1.0, 1.0, 1.0] and abs(srgb("#808080")[0] - 0.2159) < 1e-3


def test_wall_uvs_are_metric():
    V, F, _ = prism(shapely.box(0, 0, 4, 2), 0.0, 3.0).arrays()
    uv = corner_uvs(V, F)
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    wall = np.abs(n[:, 2]) < 1e-9
    assert np.allclose(uv[wall][..., 1], P[wall][..., 2])              # v = height in metres
    assert np.allclose(uv[~wall], P[~wall][..., :2])                   # caps: v = (x, y)


def test_heights_priority_and_block_correlation():
    pri = {"priors": {"k": {str(b): {"n": 100, "q": list(np.linspace(1, 9, len(QS)))} for b in range(5)}}}
    mk = lambda blk, h=np.nan, lv=np.nan: {"_poly": shapely.box(0, 0, 10, 10), "block": blk, "_obs_height": h, "_obs_levels": lv}
    bs = [mk("a", h=12.0), mk("a", lv=4), mk("b")] + [mk("c") for _ in range(6)] + [mk("d") for _ in range(6)]
    assign_heights(bs, pri, "k", 3.0, np.random.default_rng(0))
    assert bs[0]["height_source"] == "observed" and bs[0]["height_m"] == 12.0
    assert bs[1]["height_source"] == "derived" and bs[1]["height_m"] == 12.0
    assert bs[2]["height_source"] == "synthetic"
    c = [b["floors"] for b in bs[3:9]]
    d = [b["floors"] for b in bs[9:]]
    assert max(c) - min(c) <= 3 and max(d) - min(d) <= 3                  # neighbours in a block are correlated
    bs2 = [mk("a", h=12.0), mk("a", lv=4), mk("b")] + [mk("c") for _ in range(6)] + [mk("d") for _ in range(6)]
    assign_heights(bs2, pri, "k", 3.0, np.random.default_rng(0))
    assert [b["floors"] for b in bs] == [b["floors"] for b in bs2]        # deterministic for a seed


def glb_json(path: Path) -> dict:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n])


@pytest.mark.skipif(not BLENDER.exists(), reason="Blender not installed")
def test_glb_export_converts_z_up_metres_to_y_up(tmp_path):
    c = tiny_city()
    c["buildings"] = c["buildings"][:1]                              # one 20 x 10 m box, eave 9 m, at x 10-30, y 120-130
    groups, mats, _ = build_scene(c)
    save_scene(tmp_path / "meshes.npz", groups, mats)
    (tmp_path / "cameras.json").write_text(json.dumps(cameras(c)))
    script = Path(__file__).resolve().parents[1] / "blender" / "m11_scene.py"
    subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script), "--", "--tile", str(tmp_path),
                    "--views", "aerial", "--res", "64", "--samples", "1"], check=True, capture_output=True, timeout=300)
    g = glb_json(tmp_path / "scene.glb")
    walls = next(m for m in g["meshes"] if m["name"] == "walls")
    acc = [g["accessors"][p["attributes"]["POSITION"]] for p in walls["primitives"]]
    lo = np.min([a["min"] for a in acc], axis=0)
    hi = np.max([a["max"] for a in acc], axis=0)
    # glTF = (x, z, -y): x 10..30, height -0.05..9 + 0.9 m parapet, -y -130..-120
    assert np.allclose(lo, [10, -0.05, -130], atol=1e-3) and np.allclose(hi, [30, 9.9, -120], atol=1e-3)
    assert "surround" not in [m["name"] for m in g["meshes"]]


def _facade_building(poly, floors=4, eave=12.0, **style):
    st = {"facade_family": "plaster", "palette": 0, "bay_width_m": 3.5, "window_width_ratio": 0.5, "window_height_ratio": 0.55,
          "balconies": True, "balcony_pattern": "all", "cornice": True, "ground_floor": "shop", "source": "synthetic", **style}
    return {"id": "X", "footprint": rings(poly), "floors": floors, "eave_m": eave, "height_m": eave, "typology": "midrise",
            "roof": {"type": "flat", "rise_m": 0.0, "pitch_deg": 0.0, "source": "synthetic"}, "style": st}


def test_party_wall_gets_no_openings_and_doors_are_at_ground_level():
    from urbanmorph.city3d.facade import Parts, facade, new_metrics
    left, right = shapely.box(0, 0, 20, 12), shapely.box(20, 0, 40, 12)          # attached along x = 20
    road = shapely.box(-10, -10, 50, -0.01)                                      # street along the south side
    parts, m = Parts(), new_metrics()
    facade(_facade_building(left), right, road, 3.0, parts, m)
    V, _ = parts.arrays("glass")
    assert len(V) and not np.any(np.isclose(V[:, 0], 20.04, atol=1e-6))        # nothing on the shared wall plane
    assert m["party_wall_m"] >= 12.0 - 2 * 0.25 and m["openings_outside_wall"] == 0
    assert m["doors"] == 1 and m["door_bottom_z"] == [0.0]
    assert m["balconies"] > 0 and m["balconies_skipped_clash"] == 0


def test_floor_levels_are_consistent():
    from urbanmorph.city3d.facade import floor_levels
    lv = floor_levels(15.9, 5, True, 3.0)
    assert np.allclose(lv[0], (0.0, 3.9)) and np.allclose(np.diff([z1 - z0 for z0, z1 in lv[1:]]), 0)
    assert abs(lv[-1][1] - 15.9) < 1e-9 and all(a[1] == b[0] for a, b in zip(lv, lv[1:]))


def test_environment_respects_roads_and_buildings():
    _, _, rep = build_scene(tiny_city())
    e = rep["environment"]
    assert e["sidewalk_on_road_m2"] < 1e-6 and e["sidewalk_on_buildings_m2"] < 1e-6
    assert e["sidewalk_watertight"] == e["sidewalk_solids"] > 0
    assert sum(e["trunk_or_pole_inside_building"].values()) == 0 and sum(e["props_on_road"].values()) == 0
