"""Morphology metrics on synthetic geometries with exact expected values."""

import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from urbanmorph.metrics import morphology as mm
from urbanmorph.preprocessing.projection import SquareBounds

UTM = "EPSG:32631"
B = SquareBounds(0.0, 0.0, 512.0, 512.0)


def grid_graph(spacing=100.0, n=3, origin=(100.0, 100.0), angle_deg=0.0):
    """n x n lattice of two-way streets, optionally rotated about the origin."""
    G = nx.MultiDiGraph(crs=UTM)
    a = math.radians(angle_deg)

    def xy(i, j):
        x, y = i * spacing, j * spacing
        return (origin[0] + x * math.cos(a) - y * math.sin(a), origin[1] + x * math.sin(a) + y * math.cos(a))

    for i in range(n):
        for j in range(n):
            x, y = xy(i, j)
            G.add_node((i, j), x=x, y=y)
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    u, v = (i, j), (i + di, j + dj)
                    for s, t in ((u, v), (v, u)):
                        G.add_edge(s, t, length=spacing, oneway=False, reversed=False, osmid=0)
    return nx.convert_node_labels_to_integers(G)


def test_raster_coverage():
    layout = np.zeros((3, 4, 4), np.uint8)
    layout[0, :2] = 1
    layout[2, 0, 0] = 1
    cov = mm.raster_coverage(layout)
    assert cov == {"roads_coverage": 0.5, "buildings_coverage": 0.0, "green_spaces_coverage": 1 / 16}


def test_vector_coverage_unions_overlaps():
    polys = gpd.GeoDataFrame(geometry=[box(0, 0, 256, 512), box(128, 0, 256, 512)], crs=UTM)
    assert mm.vector_coverage(polys, B) == pytest.approx(0.5)


def test_building_metrics_centroid_rule():
    blds = gpd.GeoDataFrame(geometry=[
        box(10, 10, 20, 20),          # 100 m2, inside
        box(100, 100, 130, 130),      # 900 m2, inside
        box(500, 10, 540, 20),        # centroid x=520 -> outside, not counted
    ], crs=UTM)
    m = mm.building_metrics(blds, B)
    assert m["building_count"] == 2
    assert m["building_area_mean_m2"] == pytest.approx(500.0)
    assert m["building_area_median_m2"] == pytest.approx(500.0)
    assert m["building_density_per_ha"] == pytest.approx(2 / (512 * 512 / 1e4))
    assert sum(m["building_area_hist_log10"]) == 2


def test_intersections_on_lattice():
    G = grid_graph()  # 3x3 nodes: centre has 4 streets, 4 edge-midpoints have 3, corners 2
    m = mm.intersection_metrics(G, B, tolerance_m=0)
    assert m["intersections"] == 5
    assert m["intersection_density_per_km2"] == pytest.approx(5 / (0.512 ** 2))


def test_consolidation_merges_split_junction():
    G = grid_graph()
    # split the centre junction into two nodes 6 m apart (like a dual carriageway)
    G2 = nx.MultiDiGraph(crs=UTM)
    G2.add_node("a", x=0, y=250.0); G2.add_node("b", x=100, y=250.0)
    G2.add_node("c", x=106, y=250.0); G2.add_node("d", x=206, y=250.0)
    G2.add_node("n", x=103, y=350.0); G2.add_node("s", x=103, y=150.0)
    for u, v in (("a", "b"), ("b", "c"), ("c", "d"), ("n", "b"), ("s", "c")):
        G2.add_edge(u, v, length=1.0, oneway=False, reversed=False, osmid=0)
        G2.add_edge(v, u, length=1.0, oneway=False, reversed=False, osmid=0)
    raw = mm.intersection_metrics(G2, B, tolerance_m=0)
    merged = mm.intersection_metrics(G2, B, tolerance_m=10)
    assert raw["intersections"] == 2          # b and c each have 3 streets
    assert merged["intersections"] == 1       # one real crossroads


def test_orientation_perfect_grid():
    G = grid_graph()
    m = mm.orientation_metrics(mm.street_segments(G, B))
    assert m["orientation_entropy"] == pytest.approx(math.log(2))
    assert m["orientation_order_phi"] == pytest.approx(1.0)
    assert m["grid_orientation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert m["grid_strength_r4"] == pytest.approx(1.0)
    assert m["dominant_axis_strength_r2"] == pytest.approx(0.0, abs=1e-9)  # two axes cancel
    assert m["street_length_m"] == pytest.approx(12 * 100.0)  # undirected: two-way counted once


def test_orientation_rotated_grid_and_single_direction():
    G = grid_graph(angle_deg=30.0, origin=(200.0, 50.0))
    m = mm.orientation_metrics(mm.street_segments(G, B))
    # rotating by +30 deg counter-clockwise = bearings 60 and 150 deg; grid orientation 60 mod 90
    assert m["grid_orientation_deg"] == pytest.approx(60.0, abs=1e-6)
    seg = np.array([[0, 0, 0, 100.0], [10, 0, 10, 50.0]])  # both due north
    s = mm.orientation_metrics(seg)
    assert s["dominant_bearing_deg"] == pytest.approx(0.0, abs=1e-9) or s["dominant_bearing_deg"] == pytest.approx(180.0)
    assert s["dominant_axis_strength_r2"] == pytest.approx(1.0)
    assert s["orientation_entropy"] == pytest.approx(0.0)


def test_bidirectional_bearings_fold():
    a = mm.orientation_metrics(np.array([[0, 0, 100, 100.0]]))
    b = mm.orientation_metrics(np.array([[100, 100, 0, 0.0]]))
    assert a["orientation_hist"] == b["orientation_hist"]
    assert a["dominant_bearing_deg"] == pytest.approx(45.0)


def test_empty_inputs_are_safe():
    assert mm.building_metrics(gpd.GeoDataFrame(geometry=[], crs=UTM), B)["building_count"] == 0
    assert math.isnan(mm.orientation_metrics(np.zeros((0, 4)))["orientation_entropy"])
    assert mm.intersection_metrics(nx.MultiDiGraph(crs=UTM), B, 10)["intersections"] == 0
