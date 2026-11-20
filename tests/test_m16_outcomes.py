"""Milestone 16 independent network outcomes on small graphs with known answers, and their sensitivity to rotation
and clipping."""
import math

import numpy as np
import pytest

from urbanmorph.graphs import fixtures as fx
from urbanmorph.graphs.network_outcomes import (blocks, junction_angles, outcome_row, phi_boeing, radial_structure,
                                                stroke_stats, terminal)
from urbanmorph.graphs.road_graph import RoadGraph


def G(nodes, edges):
    nodes = np.asarray(nodes, float)
    e = np.asarray(edges, int)
    return RoadGraph(nodes, e, np.zeros(len(nodes), bool), edge_class=np.full(len(e), 2), edge_struct=np.zeros(len(e), int))


def test_square_grid_known_values():
    r = outcome_row(fx.grid(64.0))
    assert r["block_rect_share"] == 1.0
    assert r["block_phi_med"] == pytest.approx(2 / math.pi, abs=1e-6)          # square in its circumcircle
    assert r["block_logA_med"] == pytest.approx(math.log(64 * 64), abs=1e-6)
    assert r["block_logA_sd"] == pytest.approx(0.0, abs=1e-9)
    assert r["jangle_right_share"] == 1.0
    assert r["through_share"] == 1.0
    assert r["tree_len_share"] == 0.0
    assert r["phi_boeing"] == pytest.approx(1.0)


def test_rectangle_shape_factor():
    g = G([(100, 100), (300, 100), (300, 200), (100, 200)], [(0, 1), (1, 2), (2, 3), (3, 0)])   # 2:1 block
    assert blocks(g)["block_phi_med"] == pytest.approx(200 * 100 / (math.pi * (math.hypot(200, 100) / 2) ** 2))


def test_crossing_strokes_and_t_junction():
    cross = G([(100, 256), (256, 256), (412, 256), (256, 100), (256, 412)], [(0, 1), (1, 2), (3, 1), (1, 4)])
    s = stroke_stats(cross)
    assert s["through_share"] == 1.0 and s["stroke_len_lw"] == pytest.approx(312.0)
    assert s["stroke_xjunc_lw"] == pytest.approx(1.0)
    tee = G([(100, 256), (256, 256), (412, 256), (256, 100)], [(0, 1), (1, 2), (3, 1)])
    j = junction_angles(tee)
    assert j["jangle_right_share"] == pytest.approx(2 / 3)                       # 90, 90 and 180 degrees
    assert stroke_stats(tee)["through_share"] == 1.0                              # the bar continues; the stem ends


def test_terminal_branch_depth_and_attachment():
    # a loop (backbone) with one two-edge branch carrying a side stub: 1 junction on the path from the far dead end
    loop = [(100, 100), (200, 100), (200, 200), (100, 200)]
    nodes = loop + [(250, 150), (300, 150), (300, 120)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4), (4, 5), (5, 6)]
    t = terminal(G(nodes, edges))
    assert t["term_attached_share"] == 1.0
    assert t["term_len_med"] == pytest.approx(50 + 30)
    assert t["tree_len_share"] == pytest.approx(80 / (400 + 2 * math.hypot(50, 50) + 80))


def test_spoke_ring_radial_structure():
    r = radial_structure(fx.spoke_ring(), (256.0, 256.0))
    assert r["ring_cover_max"] == pytest.approx(1.0)
    assert r["spoke_span_lw"] == pytest.approx(192 / 256)
    assert r["spokes_long"] == 4                     # opposite spokes join through the centre into one stroke each
    assert r["sector_unreach"] == 0.0


def test_rotation_invariance_and_clipping_sensitivity():
    """Rotating a grid changes nothing but boundary clipping: blocks stay rectangular and phi stays 1; junction angles
    and continuity stay within the clipping effect. Clipping a grid to a smaller window keeps the per-block outcomes."""
    a, b = outcome_row(fx.grid(64.0)), outcome_row(fx.grid(64.0, rot_deg=30.0))
    assert b["block_rect_share"] == 1.0 and b["phi_boeing"] == pytest.approx(1.0)
    assert b["block_phi_med"] == pytest.approx(a["block_phi_med"], abs=1e-6)
    assert b["jangle_right_share"] > 0.9
    g = fx.grid(64.0)
    keep = (g.nodes <= 300).all(1)
    idx = -np.ones(g.n, int)
    idx[keep] = np.arange(keep.sum())
    e = g.edges[keep[g.edges[:, 0]] & keep[g.edges[:, 1]]]
    c = outcome_row(G(g.nodes[keep], idx[e]))
    assert c["block_rect_share"] == 1.0 and c["block_logA_med"] == pytest.approx(a["block_logA_med"])


def test_phi_boeing_uniform_is_low():
    ang = np.radians(np.arange(0, 180, 5) + 2.5)
    nodes, edges = [], []
    for k, t in enumerate(ang):
        p = np.array([256.0, 256.0]) + 3.0 * k * np.array([1.0, 0.0])
        nodes += [p, p + 20 * np.array([math.cos(t), math.sin(t)])]
        edges.append((2 * k, 2 * k + 1))
    assert phi_boeing(G(nodes, edges)) < 0.05
