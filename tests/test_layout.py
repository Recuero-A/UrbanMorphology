"""Milestone 6: parameterised footprints, symmetries, placer likelihood and the validity rules."""

import math

import numpy as np
import shapely
import torch
from shapely import affinity
from shapely.geometry import Polygon, box

from urbanmorph.graphs.road_graph import build_graph
from urbanmorph.layout.objects import (building_metrics, decode, encode, fold_axial, polygon_pixels,
                                       representation_iou, transform_polygon)
from urbanmorph.layout.place import LayoutState, StatisticalPlacer, place_neural
from urbanmorph.layout.placer_data import encode_tile, tile_samples, transform_objects, transform_pixels
from urbanmorph.models.placer import ObjectPlacer
from urbanmorph.preprocessing.projection import SquareBounds
from urbanmorph.preprocessing.rasterize import rasterize_geometries
from test_graph_gen import grid_lines

CW = [9.75, 10.0, 8.0, 6.0]


def test_rectangle_is_exact_and_l_shape_is_close():
    r = affinity.rotate(box(100, 100, 140, 120), 33)
    assert representation_iou(r) > 0.999
    L = affinity.translate(Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)]), 200, 200)
    assert representation_iou(L) > 0.8
    e = encode(r)
    assert e["length"] >= e["width"] and abs(fold_axial(e["theta"] - math.radians(33))) < 1e-6


def test_pixels_match_rasteriser_and_symmetries_commute():
    p = affinity.rotate(box(150.3, 200.7, 190.1, 230.2), 17)
    ref = rasterize_geometries([p], SquareBounds(0, 0, 512, 512), 128).ravel().nonzero()[0]
    assert np.array_equal(np.sort(polygon_pixels(p)), ref)
    for k in range(8):
        a = np.sort(polygon_pixels(transform_polygon(p, k)))
        b = np.sort(transform_pixels(polygon_pixels(p), k))
        assert np.array_equal(a, b), k


def _tile():
    lines = grid_lines()
    g = build_graph(lines, classes=[2] * len(lines))
    polys = [affinity.rotate(box(120 + 30 * i, 120, 140 + 30 * i, 132), 5 * i) for i in range(3)]
    polys += [Polygon([(130, 150), (160, 150), (160, 160), (140, 160), (140, 180), (130, 180)])]
    polys += [box(220, 220, 280, 280)]
    from urbanmorph.graphs.road_graph import rasterize_graph
    return {"graph": g, "road_raster": rasterize_graph(g, class_widths=CW).astype(bool),
            "objects": encode_tile(polys[:4], polys[4:])}, polys


def test_object_symmetry_matches_encoding_the_transformed_polygons():
    tile, polys = _tile()
    for k in range(8):
        o = transform_objects(tile["objects"], k)
        for i, p in enumerate(polys):
            e = encode(transform_polygon(p, k))
            assert np.allclose(o["anchor"][i], e["anchor"], atol=1e-3)
            if e["length"] > 1.01 * e["width"]:          # a square's "longer side" is ambiguous
                assert abs(fold_axial(o["theta"][i] - e["theta"])) < 1e-4
            q = decode(o["anchor"][i], o["theta"][i], o["length"][i], o["width"][i], o["ratios"][i])
            assert q.intersection(transform_polygon(p, k)).area / q.union(transform_polygon(p, k)).area > 0.8


def test_canonical_frame_keeps_the_polygon():
    """Folding the orientation relative to the street must not rotate asymmetric shapes."""
    from urbanmorph.layout.placer_data import canonical, street_context
    tile, polys = _tile()
    o = tile["objects"]
    axis, _, _ = street_context(o["anchor"], tile["graph"])
    phi, ratios = canonical(o["theta"], o["ratios"], axis)
    assert (np.abs(phi) <= np.pi / 2 + 1e-6).all()
    for i, p in enumerate(polys):
        q = decode(o["anchor"][i], axis[i] + phi[i], o["length"][i], o["width"][i], ratios[i])
        assert q.intersection(p).area / q.union(p).area > 0.8, i


def test_tile_samples_and_placer_likelihood():
    torch.manual_seed(0)
    tile, _ = _tile()
    b = tile_samples(tile, 3, np.random.default_rng(0))
    assert len(b["stop"]) == 4 + 1 + 1 + 1 and b["stop"].sum() == 2          # 4 buildings, STOP, 1 green, STOP
    assert b["canvas"][0, 1].sum() == 0 and b["canvas"][4, 1].sum() > 0     # canvas grows with the steps
    batch = {k: torch.from_numpy(v) for k, v in b.items()}
    model = ObjectPlacer(width=16, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = sum(v.mean() for v in model.nll(batch).values()).item()
    for _ in range(30):
        loss = sum(v.mean() for v in model.nll(batch).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item()) and loss.item() < first


def test_axial_von_mises_density_integrates_to_one():
    phi = torch.linspace(-math.pi / 2, math.pi / 2, 20001)[:-1]
    logits, mu, kappa = torch.tensor([[0.3, -0.2]]), torch.tensor([[0.5, -2.0]]), torch.tensor([[4.0, 1.0]])
    lp = ObjectPlacer.angle_log_prob(phi, logits.expand(len(phi), 2), mu.expand(len(phi), 2), kappa.expand(len(phi), 2))
    assert abs(lp.exp().sum().item() * (math.pi / len(phi)) - 1) < 1e-3


def test_rules_reject_road_and_overlap_and_trim_green():
    tile, _ = _tile()
    s = LayoutState(tile["graph"], CW)
    assert not s.try_add(box(100, 95, 112, 120), 0)                       # sits on a street
    assert s.try_add(box(150, 150, 170, 170), 0)
    assert not s.try_add(box(155, 155, 175, 175), 0)                      # overlaps the first building
    assert s.try_add(box(170, 150, 185, 170), 0)                          # touching is allowed
    assert s.try_add(box(140, 140, 200, 200), 1)                          # green: trimmed around buildings
    assert s.green[0].intersection(shapely.union_all(s.buildings)).area < 1e-6
    assert s.rules["rejected_on_road"] == 1 and s.rules["rejected_overlap"] == 1 and s.rules["green_trimmed"] == 1


def test_both_placers_produce_valid_layouts():
    tile, _ = _tile()
    torch.manual_seed(0)
    for states in (place_neural(ObjectPlacer(width=16, hidden=64), [tile["graph"]] * 2, CW, seed=0,
                                max_buildings=15, max_green=3, max_fails=5),
                   StatisticalPlacer([tile]).place([tile["graph"]], CW, seed=0)):
        for s in states:
            assert s.done
            for p in s.buildings + s.green:
                assert p.is_valid and not p.is_empty
            if s.buildings:
                m = building_metrics(s.buildings, s.graph.segments())
                assert m["building_count"] == len(s.buildings)


# ------------------------------------------------------------------ Milestone 7: strict rules
def test_strict_rules_trim_to_road_edge_and_reject_acute_and_overlap():
    tile, _ = _tile()
    s = LayoutState(tile["graph"], CW, strict=True)
    # a 20 x 14 m building whose first 2 m lie on the road surface of the street at x = 106 (8 m wide)
    assert s.try_add(box(108, 150, 128, 164), 0)
    assert s.buildings[0].intersection(s.surface).area < 1e-6 and s.rules["building_trimmed"] == 1
    assert not s.try_add(box(100, 180, 115, 190), 0)                      # mostly on the road: rejected
    assert not s.try_add(Polygon([(220, 230), (260, 230), (220, 236)]), 0)  # needle-like triangle: acute
    assert s.rules["rejected_acute"] == 1
    assert s.try_add(box(150, 150, 170, 170), 0)
    assert not s.try_add(box(169, 150, 189, 170), 0)                      # 5 % overlap: rejected when strict
    assert s.try_add(box(170, 150, 190, 170), 0)                          # touching: allowed


def test_frontage_assignment_orders_a_row_of_houses():
    from urbanmorph.layout.frontage import assign, pattern_metrics
    tile, _ = _tile()
    row = [box(120 + 18 * i, 115, 132 + 18 * i, 125) for i in range(4)]        # along the street y = 106, left side
    fr, is_front = assign(row, tile["graph"])
    assert is_front.all()
    f = next(f for f in fr if f["members"])
    assert f["members"] == [0, 1, 2, 3] or f["members"] == [3, 2, 1, 0]
    assert np.allclose(np.diff(f["s"]), 18.0, atol=0.5) and np.allclose(f["d"], 14.0, atol=0.5)
    m = pattern_metrics(row, tile["graph"])
    assert m["nn_aligned_5deg"] == 1.0 and m["setback_std_m"] < 0.1 and m["spacing_cv"] < 0.05 and m["acute_share"] == 0


def test_frontage_samples_and_model_and_sampler():
    from urbanmorph.layout.frontage_data import crops_for, tile_frontage_samples
    from urbanmorph.layout.place import place_frontage
    from urbanmorph.models.frontage_net import FrontageNet
    tile, _ = _tile()
    smp = tile_frontage_samples(tile)
    n_front = int((1 - smp["stop"]).sum())
    assert n_front >= 3 and smp["stop"].sum() >= 1
    b = {k: torch.from_numpy(np.asarray(v)) for k, v in smp.items() if k not in ("bmap", "roads", "centre", "angle", "t")}
    b["crops"] = torch.from_numpy(crops_for(smp))
    assert b["crops"].shape[1:] == (5, 32, 32)
    torch.manual_seed(0)
    model = FrontageNet(hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = sum(v.mean() for v in model.nll(b).values()).item()
    for _ in range(30):
        loss = sum(v.mean() for v in model.nll(b).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item()) and loss.item() < first
    st = place_frontage(model, [tile["graph"]], CW, seed=0)[0]
    for p in st.buildings:
        assert p.is_valid and p.intersection(st.surface).area < 1e-6


def test_relative_frontage_targets_are_residuals_and_sampler_inverts_them():
    from urbanmorph.layout.frontage_data import tile_frontage_samples
    tile, _ = _tile()
    a = tile_frontage_samples(tile)
    r = tile_frontage_samples(tile, relative=True)
    first = r["scalars"][:, -12 + 5] == 0                                  # no previous building
    obj = r["stop"] == 0
    assert np.allclose(a["gapd"][obj & first], r["gapd"][obj & first])
    # a residual plus the previous building's absolute value gives the absolute target back
    prev_logd = r["scalars"][:, -12 + 1] * 3
    sel = obj & ~first
    assert np.allclose(r["gapd"][sel, 1] + prev_logd[sel], a["gapd"][sel, 1], atol=1e-5)


def test_strict_trim_removes_even_a_tiny_overlap_of_a_large_building():
    tile, _ = _tile()
    s = LayoutState(tile["graph"], CW, strict=True)
    assert s.try_add(box(109.5, 150, 200, 200), 0)                      # 4525 m2, 0.5 m x 50 m on the road surface
    assert s.buildings[0].intersection(s.surface).area < 1e-6


# ------------------------------------------------------------------ Milestone 8: blocks
def test_blocks_from_grid_are_closed_inside_and_open_at_tile_edge():
    from urbanmorph.layout.blocks import block_of, extract_blocks
    tile, _ = _tile()                                   # 4 x 4 streets 100 m apart, inside the tile, 8 m wide
    blocks = extract_blocks(tile["graph"], CW)
    closed = [b for b in blocks if not b["open"]]
    assert len(closed) == 9                             # 3 x 3 enclosed blocks
    assert all(abs(b["area_m2"] - 92 ** 2) < 30 for b in closed)   # 100 m spacing minus one 8 m road
    assert sum(b["open"] for b in blocks) == 1          # everything outside the grid: one open ring
    pts = np.array([[156.0, 156.0], [106.0, 156.0]])    # block centre, and a point on a road
    ids = block_of(pts, blocks)
    assert ids[0] >= 0 and ids[1] == -1


def test_blocks_without_closed_cycles():
    from urbanmorph.layout.blocks import extract_blocks
    g = build_graph([np.array([[0.0, 256.0], [512.0, 256.0]])], classes=[2])   # one street across the tile
    blocks = extract_blocks(g, CW)
    assert len(blocks) == 2 and all(b["open"] for b in blocks)


# ------------------------------------------------------------------ Milestone 8 B1: edge parameterisation
def test_edge_gaps_signed_and_placement_by_edges():
    from shapely.geometry import LineString
    from urbanmorph.layout.frontage_edge import extent_along, gap_to_z, place_by_edges, z_to_gap
    assert np.allclose(z_to_gap(gap_to_z(np.array([-3.0, 0.0, 0.2, 12.0]))), [-3.0, 0.0, 0.2, 12.0])
    line = LineString([(0, 0), (200, 0)])
    house = box(0, 0, 10, 8)
    p = place_by_edges(house, line, 50.0, 6.0)              # extent starts at 50 m, front 6 m left of the street
    a, b = extent_along(p, line)
    assert abs(a - 50.0) < 1e-6 and abs(b - 60.0) < 1e-6 and abs(p.distance(line) - 6.0) < 1e-6 and p.centroid.y > 0
    q = place_by_edges(house, line, b + 0.0, 6.0)           # attached: zero gap, shared wall, no overlap
    assert q.intersection(p).area < 1e-9 and q.distance(p) < 1e-9


def test_edge_samples_and_model():
    from urbanmorph.layout.frontage_edge import crops_for, place_frontage_edge, tile_edge_samples
    from urbanmorph.models.frontage_net import FrontageNetEdge
    tile, _ = _tile()
    smp = tile_edge_samples(tile)
    obj = smp["stop"] == 0
    assert obj.sum() >= 3 and set(np.unique(smp["rect"])) <= {0.0, 1.0}
    b = {k: torch.from_numpy(np.asarray(smp[k])) for k in ("scalars", "stop", "gapd", "phi", "size", "shape", "rect")}
    b["crops"] = torch.from_numpy(crops_for(smp))
    torch.manual_seed(0)
    m = FrontageNetEdge(64)
    terms = m.nll(b)
    assert all(torch.isfinite(v).all() for v in terms.values())
    st = place_frontage_edge(m, [tile["graph"]], CW, seed=0)[0]
    assert st.rules["frontage_proposed"] >= st.rules["frontage_accepted"]
    for p in st.buildings:
        assert p.is_valid and p.intersection(st.surface).area < 1e-6


# ------------------------------------------------------------------ Milestone 8: parcel measurements
def test_parcel_frontage_depth_setbacks_and_coverage():
    from urbanmorph.layout.parcels import parcel_table
    # three 20 m x 40 m parcels side by side; their south edge (y = 100) faces the street
    parcels = [box(100 + 20 * i, 100, 120 + 20 * i, 140) for i in range(3)]
    house = box(124, 105, 136, 125)                        # in the middle parcel: front 5, sides 4, rear 15
    street = np.array([[0.0, 95.0, 512.0, 95.0]])          # street centreline 5 m south of the parcels
    rows, summ = parcel_table(parcels, [house], street)
    mid = rows[1]
    assert abs(mid["frontage_m"] - 20.0) < 1.5 and abs(mid["depth_m"] - 40.0) < 1.0
    assert mid["n_buildings"] == 1 and abs(mid["coverage"] - 240 / 800) < 1e-6
    assert abs(mid["front_setback_m"] - 5.0) < 0.6 and abs(mid["side_setback_m"] - 4.0) < 0.6 and abs(mid["rear_setback_m"] - 15.0) < 0.6
    assert rows[0]["n_buildings"] == 0 and summ["buildings_spanning_parcels"] == 0
    rows2, summ2 = parcel_table(parcels, [box(115, 105, 125, 120)], street)   # straddles two parcels
    assert summ2["buildings_spanning_parcels"] == 1


def test_b2_samples_preplace_frontage_and_neighbour_features():
    from urbanmorph.layout.placer_data import K_NB, neighbour_features, tile_samples_b2
    tile, polys = _tile()
    n_b = int((tile["objects"]["type"] == 0).sum())
    tile["front_mask"] = np.array([True] * (n_b - 1) + [False])       # all but the last building are frontage
    b = tile_samples_b2(tile, 0, np.random.default_rng(0))
    assert len(b["stop"]) == 1 + 1 + 1 + 1                             # 1 interior, STOP, 1 green, STOP
    assert b["canvas"][0, 1].sum() > 0                                  # frontage buildings already on the canvas
    assert b["nb"].shape[1:] == (K_NB, 7) and b["nb_mask"][0].sum() >= 1
    f, m = neighbour_features(np.zeros(2), 0.0, np.array([[10.0, 0.0], [500.0, 0.0]]), np.array([0.3, 0.0]),
                              np.zeros(2), np.zeros(2))
    assert m.sum() == 1 and np.allclose(f[0, :2], [10 / 30, 0.0])    # only the neighbour within 60 m
    model = ObjectPlacer(width=16, hidden=64, neighbours=8)
    t = model.nll({k: torch.from_numpy(v) for k, v in b.items()})
    assert all(torch.isfinite(v).all() for v in t.values())


# ------------------------------------------------------------------ Milestone 8 S: real-shape library
def test_shape_library_labels_rectangles_and_symmetries():
    from urbanmorph.layout.shape_library import ShapeLibrary
    from urbanmorph.layout.placer_data import _MIRROR_RAYS, shape_from_unbounded
    _, polys = _tile()
    l_shape = encode(polys[3])["ratios"]
    lib = ShapeLibrary(np.r_[np.ones((1, 20)), l_shape[None]], np.array([10, 5]))
    assert lib.n_classes == 8
    assert lib.nearest(np.ones((1, 20)))[0] == 0                               # a rectangle is class 0
    assert lib.nearest(l_shape[None])[0] == 4                                   # shape 1, identity
    assert lib.nearest(l_shape[_MIRROR_RAYS][None])[0] == 5                     # shape 1, mirrored
    assert np.allclose(shape_from_unbounded(lib.unbounded([4]))[0], l_shape, atol=1e-4)   # decodes exactly


def test_library_models_train_and_sample_library_shapes():
    from urbanmorph.layout.frontage_edge import crops_for, tile_edge_samples
    from urbanmorph.layout.placer_data import shape_from_unbounded, tile_samples_b2
    from urbanmorph.layout.shape_library import ShapeLibrary
    from urbanmorph.models.frontage_net import FrontageNetLib
    tile, polys = _tile()
    lib = ShapeLibrary(np.r_[np.ones((1, 20)), encode(polys[3])["ratios"][None]], np.array([10, 5]))
    smp = tile_edge_samples(tile)
    b = {k: torch.from_numpy(np.asarray(smp[k])) for k in ("scalars", "stop", "gapd", "phi", "size")}
    b["shape_cls"] = torch.from_numpy(lib.nearest(shape_from_unbounded(smp["shape"])))
    b["crops"] = torch.from_numpy(crops_for(smp))
    torch.manual_seed(0)
    m = FrontageNetLib(64, lib_logits=lib.unbounded(np.arange(lib.n_classes)))
    assert all(torch.isfinite(v).all() for v in m.nll(b).values())
    shape = m.sample(b["crops"], b["scalars"])[4]
    r = shape_from_unbounded(shape)
    assert all(np.abs(lib.variants - x).max(1).min() < 1e-4 for x in r)         # every sample is a library shape
    n_b = int((tile["objects"]["type"] == 0).sum())
    tile["front_mask"] = np.ones(n_b, bool)
    bb = tile_samples_b2(tile, 0, np.random.default_rng(0), front_drop=1.0)
    bb["shape_cls"] = lib.nearest(shape_from_unbounded(bb["shape"]))
    p = ObjectPlacer(width=16, hidden=64, neighbours=8, shape_lib=lib.n_classes)
    assert all(torch.isfinite(v).all() for v in p.nll({k: torch.from_numpy(v) for k, v in bb.items()}).values())


# ------------------------------------------------------------------ Milestone 8 C / D: generated parcels
class _FixedSampler:
    rng = np.random.default_rng(0)

    def lot(self, country, ctx):
        return 15.0, 30.0

    def jitter(self, country):
        return 0.0, 0.0

    def coverage(self, country, ctx, nb_known):
        return 0.4, 1

    def setbacks(self, country, ctx):
        return 3.0, 1.0, 1.0, 2.0


def test_generated_parcels_tile_blocks_and_hold_their_buildings():
    from urbanmorph.layout.blocks import extract_blocks
    from urbanmorph.layout.parcel_gen import generate_parcels
    from urbanmorph.layout.shape_library import ShapeLibrary
    tile, _ = _tile()
    lib = ShapeLibrary(np.ones((1, 20)), np.array([1]))
    st = LayoutState(tile["graph"], CW, True)
    P, stats = generate_parcels(st, "spain", _FixedSampler(), lib, np.random.default_rng(0), CW)
    land = shapely.union_all([b["polygon"] for b in extract_blocks(tile["graph"], CW)])
    polys = [p["polygon"] for p in P]
    assert all(p["status"] == "generated" for p in P)
    assert shapely.union_all(polys).difference(land.buffer(0.05)).area < 1.0           # parcels lie on blocks
    tree = shapely.STRtree(polys)
    for i, q in enumerate(polys):                                                     # parcels do not overlap
        for j in tree.query(q):
            assert j == i or q.intersection(polys[j]).area < 0.5
    owner = {b: p for p in P for b in p.get("building_idx", [])}
    assert len(owner) == len(st.buildings) > 0
    for b, p in owner.items():                                                        # buildings inside their parcel
        assert st.buildings[b].difference(p["polygon"].buffer(0.05)).area < 0.05
    inner = [p for p in P if not p["interior"] and abs(p["polygon"].centroid.x - 156) < 40 and abs(p["polygon"].centroid.y - 156) < 40]
    assert inner and np.median([p["polygon"].area for p in inner]) > 150               # ~15 m x 30 m lots
    assert stats["buildings_accepted"] <= stats["buildings_proposed"]


def test_parcel_net_likelihood_and_sampler_interface():
    from urbanmorph.models.parcel_net import NetSampler, ParcelNet, features
    torch.manual_seed(0)
    n = 16
    x = torch.from_numpy(features(["spain"] * 8 + ["united_states"] * 8, [2] * n, [9.0] * n, [False] * n, [0.5] * n, [True] * n))
    b = {"x": x, "lot": torch.randn(n, 2) + 2, "m_lot": torch.ones(n), "built": torch.ones(n), "coverage": torch.rand(n),
         "n_cls": torch.zeros(n, dtype=torch.long), "front": torch.rand(n) * 5, "m_sb": torch.ones(n),
         "side": torch.zeros(n), "m_side": torch.ones(n), "rear": torch.rand(n) * 5, "m_rear": torch.ones(n)}
    m = ParcelNet(32)
    assert all(torch.isfinite(v).all() for v in m.nll(b).values())
    s = NetSampler(m, {"spain": (0.1, 0.1), "united_states": (0.1, 0.1)})
    ctx = {"cls": 2, "log_block_area": 9.0, "interior": False}
    w, d = s.lot("spain", ctx)
    c, nb = s.coverage("spain", ctx, [0.3, 0.6])
    assert w > 0 and d > 0 and 0 <= c <= 1 and nb in (0, 1, 2, 3) and len(s.setbacks("spain", ctx)) == 4


def test_parcel_export_keeps_ids_and_building_parcel_relation(tmp_path):
    import geopandas as gpd
    from urbanmorph.layout.export import export_parcel_layout
    tile, _ = _tile()
    parcels = [{"parcel_id": 0, "status": "generated", "method": "C", "interior": False, "polygon": box(150, 150, 170, 180),
                "building_idx": [0]}]
    path = tmp_path / "x.gpkg"
    export_parcel_layout(path, tile["graph"], parcels, [box(152, 152, 165, 170), box(300, 300, 310, 310)], [], CW)
    b = gpd.read_file(path, layer="buildings")
    p = gpd.read_file(path, layer="parcels")
    assert list(b.parcel_id) == [0, -1] and list(p.status) == ["generated"]


def test_templates_are_right_angled_and_placed_in_the_street_frame():
    from urbanmorph.layout.frontage import interior_angles
    from urbanmorph.layout.templates import FILL, N_TEMPLATES, TEMPLATES, TYPE_OF, best_template, place, to_unit
    assert N_TEMPLATES > 100 and set(TYPE_OF) == set(range(7))
    assert all(interior_angles(affinity.scale(p, 20, 14, origin=(0, 0))).min() >= 89 for _, _, p in TEMPLATES)   # never acute (at building scale)
    assert TEMPLATES[0][0] == "rectangle" and abs(FILL[0] - 1) < 1e-9
    t, n = np.array([np.cos(0.3), np.sin(0.3)]), np.array([-np.sin(0.3), np.cos(0.3)])
    k = 5
    p = place(k, np.array([100.0, 50.0]), t, n, 2.0, 3.0, 12.0, 9.0)                 # stretched, not clipped
    assert abs(p.area - FILL[k] * 12 * 9) < 1e-6
    u, w, d, x0, y0 = to_unit(p, np.array([100.0, 50.0]), t, n)
    assert abs(w - 12) < 1e-6 and abs(d - 9) < 1e-6 and abs(x0 - 2) < 1e-6 and abs(y0 - 3) < 1e-6
    assert best_template(u)[0] == k and best_template(u)[1] > 0.99


def test_block_nets_likelihoods_and_sampling():
    from urbanmorph.models.block_net import N_MAX, BlockNets, block_x, shape_x, side_x, zone_x
    torch.manual_seed(0)
    m = BlockNets(32)
    bx = block_x(["spain", "united_states"], [8.0, 9.0], [0.8, 0.9], [0.6, 0.7], [4, 6], [2, 1])
    assert torch.isfinite(m.nll("type", {"x": torch.from_numpy(bx), "y": torch.tensor([3, 0])})).all()
    sx = side_x(bx, [3, 2], [3.5, 4.0], [2, 2])
    b = {"x": torch.from_numpy(sx), "N": torch.tensor([4, 0]), "ratio": torch.tensor([0.5, 0.5]), "m_ratio": torch.tensor([1.0, 0.0])}
    assert torch.isfinite(m.nll("side", b)).all()
    g = torch.Generator().manual_seed(0)
    n, r = m.sample_side(sx[:1], g)
    assert 0 <= n <= N_MAX and 0 < r < 1 and 0 <= m.sample_type(bx[:1], g) < 4
    assert 0 < m.p_green(zone_x(["spain"], [5.0], [False], [3], [0.0], [False])) < 1
    assert 0 <= m.sample_shape(shape_x(["spain"], [2.5], [3.0], [0.5], [False], [2]), g)


class _BlockNetsStub:
    """Deterministic stand-in: every block 'rows', 5 lots per side meeting back-to-back, rectangles."""

    def sample_type(self, x, g):
        return 3

    def sample_side(self, x, g):
        return 5, 0.5

    def p_green(self, x):
        return 0.0

    def sample_shape(self, x, g):
        return 0


def test_tidy_parcels_tile_blocks_hold_unclipped_template_buildings():
    from urbanmorph.layout.blocks import extract_blocks
    from urbanmorph.layout.templates import FILL
    from urbanmorph.layout.tidy_parcels import generate_tidy
    tile, _ = _tile()
    st = LayoutState(tile["graph"], CW, True)
    P, stats = generate_tidy(st, "spain", _BlockNetsStub(), _FixedSampler(), np.random.default_rng(0), CW, 0.0, zoning=True)
    land = shapely.union_all([b["polygon"] for b in extract_blocks(tile["graph"], CW)])
    polys = [p["polygon"] for p in P]
    assert all(p["status"] == "generated" and p["use"] in ("building", "green") for p in P)
    assert shapely.union_all(polys).difference(land.buffer(0.05)).area < 1.0
    tree = shapely.STRtree(polys)
    for i, q in enumerate(polys):
        for j in tree.query(q):
            assert j == i or q.intersection(polys[j]).area < 0.5
    owner = {b: p for p in P for b in p.get("building_idx", [])}
    assert len(owner) == len(st.buildings) > 0 and stats["clipped_to_parcel"] == 0
    for b, p in owner.items():                                    # inside the parcel and still a full template
        fp = st.buildings[b]
        assert fp.difference(p["polygon"].buffer(0.06)).area < 0.05
        rect = shapely.minimum_rotated_rectangle(fp)
        assert fp.area / rect.area > FILL[0] - 0.01                # rectangles (stub), not clipped shapes
    widths = [p["polygon"].area for p in P if not p["interior"] and p["use"] == "building"]
    assert np.std(widths) / np.mean(widths) < 0.6                  # equal division: similar lots


def test_street_access_split_leaves_no_building_parcel_without_frontage():
    from urbanmorph.layout.parcels import FRONT_TOL_M
    from urbanmorph.layout.tidy_parcels import MIN_FRONT_M, _TILE_EDGE, frontage, generate_tidy
    tile, _ = _tile()
    st = LayoutState(tile["graph"], CW, True)
    P, stats = generate_tidy(st, "spain", _BlockNetsStub(), _FixedSampler(), np.random.default_rng(0), CW, 0.0,
                             zoning=True, access=True, method="B")
    seg = tile["graph"].segments()
    near = shapely.MultiLineString([s.reshape(2, 2) for s in seg]).buffer(FRONT_TOL_M)
    for p in P:
        if p["use"] == "building":
            f = frontage(p["polygon"], near)
            assert (f is not None and f.length >= MIN_FRONT_M) or p["polygon"].exterior.intersection(_TILE_EDGE).length >= MIN_FRONT_M
    owner = {b: p for p in P for b in p.get("building_idx", [])}
    assert len(owner) == len(st.buildings) > 0 and stats["clipped_to_parcel"] == 0
    for b, p in owner.items():
        assert st.buildings[b].difference(p["polygon"].buffer(0.06)).area < 0.05
