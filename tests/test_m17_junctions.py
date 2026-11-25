"""Milestone 17 junction metrics on fixtures with known answers (graphs/junctions.py)."""
import math

import numpy as np
import pytest

from urbanmorph.graphs.junctions import clean, consolidate, junction_nodes, junction_row, nn_distances, summarise
from urbanmorph.graphs.road_graph import RoadGraph


def G(nodes, edges, struct=None):
    nodes = np.asarray(nodes, float)
    e = np.asarray(edges, int)
    st = np.zeros(len(e), int) if struct is None else np.asarray(struct, int)
    return RoadGraph(nodes, e, np.zeros(len(nodes), bool), edge_class=np.full(len(e), 2), edge_struct=st)


def close_t_pair(gap=8.0):
    """A street with two T-junctions ``gap`` m apart, side streets on opposite sides (a staggered junction)."""
    x = 256.0
    nodes = [(60, 256), (x - gap / 2, 256), (x + gap / 2, 256), (452, 256), (x - gap / 2, 420), (x + gap / 2, 90)]
    return G(nodes, [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])


def test_close_t_pair():
    g = close_t_pair(8.0)
    nn = nn_distances(g)
    assert np.allclose(nn, 8.0)
    r = junction_row(g)
    assert r["close15"] == 1.0 and r["nn_p10"] == pytest.approx(8.0)
    c = consolidate(g, 10.0)
    assert c["clusters"] == 1 and c["ratio"] == 2.0


def test_chamfered_corner_is_one_functional_junction():
    s = 5.0                                   # a 10 m chamfer square around the crossing of two streets
    c = np.array([256.0, 256.0])
    sq = [c + (-s, -s), c + (s, -s), c + (s, s), c + (-s, s)]
    arms = [c + (-s, -150), c + (150, -s), c + (s, 150), c + (-150, s)]
    nodes = sq + arms
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)]
    g = G(nodes, edges)
    assert len(junction_nodes(*clean(g)[:2])) == 4
    assert junction_row(g)["close15"] == 1.0
    k = consolidate(g, 10.0)
    assert k["clusters"] == 1 and k["ratio"] == 4.0 and k["max_diameter"] == pytest.approx(10 * math.sqrt(2))


def test_divided_carriageway_depends_on_tau():
    nodes = [(60, 250), (256, 250), (452, 250), (60, 262), (256, 262), (452, 262), (256, 90), (256, 420)]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 1), (1, 4), (4, 7)]
    g = G(nodes, edges)
    assert np.allclose(nn_distances(g), 12.0)
    assert consolidate(g, 10.0)["ratio"] == 1.0          # pairwise 12 m > 10 m: kept apart
    assert consolidate(g, 12.0)["ratio"] == 2.0          # pairwise <= 12 m and joined by a 12 m street


def test_short_block_not_close():
    g = close_t_pair(30.0)
    r = junction_row(g)
    assert r["close15"] == 0.0
    assert consolidate(g, 10.0)["ratio"] == 1.0 and consolidate(g, 15.0)["ratio"] == 1.0


def test_overpass_never_consolidated():
    """A ground crossing at A and, 4 m away in plan, a junction B on a bridge deck (structure 1) whose only at-grade
    link to A is 300 m long: close in plan (counted by the geometric share), never merged."""
    A = (256.0, 256.0)
    B = (260.0, 256.0)
    nodes = [A, (100, 256), (412, 256), (256, 100), (256, 412),           # ground crossing (0..4)
             B, (260, 120), (260, 400), (420, 256),                       # bridge deck junction B (5) and its arms
             ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8), (8, 2)]
    st = [0, 0, 0, 0, 1, 1, 1, 0]
    g = G(nodes, edges, st)
    assert junction_row(g)["close15"] == 1.0                # geometric proximity: yes
    assert consolidate(g, 10.0)["ratio"] == 1.0              # topology: no short at-grade link
    assert consolidate(g, 10.0)["clusters"] == 2


def test_chain_of_close_junctions():
    xs = [228.0, 236.0, 244.0, 252.0, 260.0]
    nodes = [(60, 256)] + [(x, 256) for x in xs] + [(452, 256)]
    edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    for k, x in enumerate(xs, start=1):                     # alternating side stubs make each node a junction
        nodes.append((x, 256 + (120 if k % 2 else -120)))
        edges.append((k, len(nodes) - 1))
    g = G(nodes, edges)
    k = consolidate(g, 10.0)
    assert k["ratio"] == 5.0 and k["clusters"] == 1 and k["max_diameter"] == pytest.approx(32.0)


def test_duplicated_nodes_are_one_junction():
    nodes = [(256, 256), (256.0 + 1e-3, 256), (100, 256), (412, 256), (256, 100), (256, 412)]
    edges = [(0, 2), (0, 3), (1, 4), (1, 5)]                # two coincident points, two edges each
    n, e, _ = clean(G(nodes, edges))
    assert len(junction_nodes(n, e)) == 1
    assert len(nn_distances(G(nodes, edges))) == 0          # a single junction: no neighbour distance


def test_boundary_junction_is_not_focal():
    g = close_t_pair(8.0)
    shifted = RoadGraph(g.nodes - [225.0, 0.0], g.edges, g.border, edge_class=g.classes(), edge_struct=g.structs())
    assert len(nn_distances(shifted)) == 0                  # both junctions now within 40 m of the west edge
    assert np.isnan(junction_row(shifted)["close15"])
    s = summarise([junction_row(shifted), junction_row(g)])
    assert s["windows"] == 2 and s["windows_scored"] == 1 and s["windows_lt2_focal"] == 1


def test_osmnx_tolerance_is_a_buffer_radius():
    """OSMnx 2.1.1 consolidate_intersections(tolerance=r) buffers every node by r: nodes less than 2r apart merge (and
    chains of overlapping buffers merge further). Our tau is a pairwise distance: tau = 12 groups the 12 m pair."""
    ox = pytest.importorskip("osmnx")
    import networkx as nx
    nodes = [(60, 250), (256, 250), (452, 250), (60, 262), (256, 262), (452, 262), (256, 90), (256, 420)]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 1), (1, 4), (4, 7)]
    M = nx.MultiDiGraph(crs="EPSG:32652")
    for i, (x, y) in enumerate(nodes):
        M.add_node(i, x=float(x), y=float(y))
    for a, b in edges:
        L = float(np.hypot(*(np.subtract(nodes[a], nodes[b]))))
        M.add_edge(a, b, length=L)
        M.add_edge(b, a, length=L)
    n65 = len(ox.consolidate_intersections(M, tolerance=6.5, rebuild_graph=False, dead_ends=False))
    n55 = len(ox.consolidate_intersections(M, tolerance=5.5, rebuild_graph=False, dead_ends=False))
    assert n65 < n55                       # 6.5 m buffers overlap across the 12 m gap (2r > 12); 5.5 m ones do not
    assert consolidate(G(nodes, edges), 12.0)["ratio"] == 2.0
