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
