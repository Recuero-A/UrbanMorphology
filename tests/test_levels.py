"""3D-aware street graphs (Milestone 12): overpasses, river bridges, tunnels and ordinary intersections."""
import numpy as np
import shapely
from shapely.geometry import LineString

from urbanmorph.graphs.crossings import street_crossings, water_crossings
from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.levels import BRIDGE, GROUND, PASSAGE, TUNNEL, _overlay, level_of, parse_layer, structure_of
from urbanmorph.graphs.road_graph import RoadGraph, build_graph, components, graph_metrics, subgraph

H = [np.array([[106.0, 256.0], [406.0, 256.0]])]         # resampled points at x = 106 + 30k: one lies on (256, 256)
V = [np.array([[256.0, 106.0], [256.0, 406.0]])]


def max_degree(g):
    return int(g.degree().max())


def test_ordinary_intersection_is_a_confirmed_at_grade_junction():
    c = (256.0, 256.0)                                    # OSM: four ways meeting at one shared node
    lines = [np.array([[106, 256], c]), np.array([c, [406, 256]]), np.array([[256, 106], c]), np.array([c, [256, 406]])]
    g = build_graph(lines, levels=[0] * 4, structs=[GROUND] * 4)
    assert max_degree(g) == 4
    x = street_crossings(g)
    assert (x["at_grade"], x["grade_separated"], x["ambiguous"]) == (1, 0, 0)


def test_overpass_is_not_a_junction():
    old = build_graph(H + V)                              # Milestone 5 planar snap: a false junction
    assert max_degree(old) == 4
    g = build_graph(H + V, levels=[0, 1], structs=[GROUND, BRIDGE])
    assert max_degree(g) == 2 and len(np.unique(components(g))) == 2
    x = street_crossings(g)
    assert (x["at_grade"], x["grade_separated"], x["ambiguous"]) == (0, 1, 0)
    assert set(g.structs()) == {GROUND, BRIDGE} and set(g.levels()) == {0, 1}


def test_tunnel_under_a_street_is_not_a_junction():
    g = build_graph(H + V, levels=[0, -1], structs=[GROUND, TUNNEL])
    assert max_degree(g) == 2
    assert street_crossings(g)["grade_separated"] == 1
    assert graph_metrics(g)["intersections_per_km2"] == 0


def test_river_bridge_connects_through_portals_and_is_classified():
    lines = [np.array([[256, 40], [256, 200]]), np.array([[256, 200], [256, 300]]), np.array([[256, 300], [256, 460]])]
    g = build_graph(lines, levels=[0, 1, 0], structs=[GROUND, BRIDGE, GROUND])
    assert len(np.unique(components(g))) == 1           # approaches and deck are one street
    river = shapely.box(0, 230, 512, 270)
    w = water_crossings(g, river)
    assert w["bridge"] >= 1 and w["unknown"] == 0
    flat = build_graph([np.array([[100, 40], [100, 460]])], levels=[0], structs=[GROUND])
    assert water_crossings(flat, river) == {"bridge": 0, "tunnel": 0, "ford": 0, "unknown": 1}   # never assumed ground
    assert water_crossings(flat, river, ford_edges=np.ones(len(flat.edges), bool))["ford"] == 1


def test_untagged_crossing_without_shared_node_is_ambiguous():
    a = np.array([[100.0, 250.0], [400.0, 262.0]])        # resampled points never within MERGE_M of the other line
    b = np.array([[243.0, 100.0], [269.0, 420.0]])
    g = build_graph([a, b], levels=[0, 0], structs=[GROUND, GROUND])
    x = street_crossings(g)
    assert x["ambiguous"] == 1 and x["at_grade"] == 0 and x["grade_separated"] == 0


def test_all_ground_levels_reproduce_the_planar_build():
    rng = np.random.default_rng(0)
    lines = [rng.uniform(0, 512, (rng.integers(2, 5), 2)) for _ in range(12)]
    old = build_graph(lines, classes=[2] * 12)
    new = build_graph(lines, classes=[2] * 12, levels=[0] * 12, structs=[GROUND] * 12)
    assert np.array_equal(old.nodes, new.nodes) and np.array_equal(old.edges, new.edges)
    assert (new.levels() == 0).all() and (new.structs() == GROUND).all()


def test_metadata_survives_transform_and_subgraph():
    g = build_graph(H + V, classes=[2, 0], levels=[0, 1], structs=[GROUND, BRIDGE])
    for k in range(8):
        t = g.transform(k)
        assert np.array_equal(t.levels(), g.levels()) and np.array_equal(t.structs(), g.structs())
    s = subgraph(g, g.structs() == BRIDGE)
    assert (s.structs() == BRIDGE).all() and (s.levels() == 1).all() and (s.classes() == 0).all()


def test_tag_parsing():
    assert structure_of("yes", None) == BRIDGE and structure_of("viaduct", None) == BRIDGE
    assert structure_of(None, "yes") == TUNNEL and structure_of(None, "building_passage") == PASSAGE
    assert structure_of("no", "no") == GROUND and structure_of(float("nan"), None) == GROUND
    assert level_of(None, BRIDGE) == 1 and level_of(None, TUNNEL) == -1 and level_of("2", BRIDGE) == 2
    assert level_of("-1;0", GROUND) == -1 and parse_layer("x") is None


def test_multi_way_line_is_cut_along_the_tagged_member():
    line = LineString([(0, 256), (512, 256)])            # one OSMnx edge = ground way + bridge way + ground way
    bridge = LineString([(200, 256), (300, 256)])
    parts = _overlay(line, [(bridge, 1, BRIDGE)])
    kinds = [(round(p[0].length), p[2]) for p in parts]
    assert kinds == [(200, GROUND), (100, BRIDGE), (212, GROUND)]


# ------------------------------------------------------------------ generation respects given bridges
def bridge_given(meta: bool) -> RoadGraph:
    nodes = np.array([[10.0, 256.0], [256.0, 256.0], [500.0, 256.0]])
    edges = np.array([[0, 1], [1, 2]])
    kw = dict(edge_level=np.array([1, 1]), edge_struct=np.array([BRIDGE, BRIDGE])) if meta else {}
    return RoadGraph(nodes, edges, np.zeros(3, bool), np.array([0, 0]), **kw)


def grow_across(meta: bool) -> GenState:
    s = GenState()
    s.add_given(bridge_given(meta))
    u = s.add_node(np.array([200.0, 100.0]), push=False)
    s.apply_step(u, np.array([0.0, 300.0]))               # crosses the given street at (200, 256)
    return s


def test_generated_ground_street_passes_under_a_given_bridge():
    s = grow_across(meta=True)
    assert s.rules["grade_separated"] == 1 and s.rules["crossing_split"] == 0
    g = s.graph(with_class=True)
    assert (g.structs() == BRIDGE).sum() == 2              # the deck was not split
    assert max_degree(g) == 2


def test_without_metadata_the_crossing_still_splits():
    s = grow_across(meta=False)
    assert s.rules["crossing_split"] == 1 and s.rules["grade_separated"] == 0


def test_bridge_interior_is_not_a_merge_target():
    s = GenState()
    s.add_given(bridge_given(True))
    u = s.add_node(np.array([256.0, 200.0]), push=False)
    s.apply_step(u, np.array([0.0, 52.0]))                # ends 4 m below the deck's middle node
    assert s.rules["merge_node"] == 0 and s.rules["t_junction"] == 0
