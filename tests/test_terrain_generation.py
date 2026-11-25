"""Terrain-aware generation (Milestone 12): baseline P and the terrain policy rules."""
import numpy as np

from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.procedural_terrain import chaikin, gabriel, sample_P
from urbanmorph.graphs.terrain_generate import TerrainPolicy, apply_bridge
from urbanmorph.graphs.terrain_metrics import tile_metrics
from urbanmorph.terrain.heightfield import Terrain


def tile(fz, water=None):
    c = (np.arange(256) - 64 + 0.5) * 4.0
    X, Y = np.meshgrid(c, 512 - c)
    d = {"elev": fz(X, Y).astype(np.float32), "valid": np.ones((256, 256), bool),
         "water": (np.zeros((256, 256)) if water is None else water(X, Y)).astype(np.int8)}
    return d


def test_gabriel_and_chaikin():
    pts = np.array([[0, 0], [10, 0], [5, 1], [5, 20.0]])
    e = gabriel(pts)
    assert (0, 1) not in e and (0, 2) in e                    # (5,1) lies in the circle on (0,0)-(10,0)
    p = chaikin(np.array([[0, 0], [10, 0], [10, 10.0]]))
    assert np.allclose(p[0], [0, 0]) and np.allclose(p[-1], [10, 10]) and len(p) > 3


def test_procedural_baseline_avoids_steep_ground_and_bridges_rivers():
    d = tile(lambda x, y: np.where(x > 300, (x - 300) * 0.5, 0.0) + 20,
             lambda x, y: ((y > 240) & (y < 256) & (x < 300)).astype(int))
    from urbanmorph.graphs.terrain_generate import terrain_of
    t = terrain_of(d)
    g = sample_P(t, np.random.default_rng(0), 150.0)
    assert len(g.edges) > 20
    assert (g.nodes[:, 0] > 330).sum() <= 2                    # the 50 % cliff zone stays (almost) empty
    m = tile_metrics(g, t)
    assert m["unbridged_water_m"] == 0 and (g.structs() == BRIDGE).any()


def test_policy_R_drops_a_step_into_a_lake_and_bridges_a_river():
    river = tile(lambda x, y: 0 * x + 10, lambda x, y: ((y > 240) & (y < 256)).astype(int))
    pol = TerrainPolicy([river], [], {"elev_scale": 5, "slope_scale": 0.1}, mode="R")
    s = GenState()
    u = s.add_node(np.array([200.0, 220.0]), push=False)
    pol.apply(0, s, u, np.array([0.0, 50.0]), 2, 0, None, 0.0, None)       # ground sample across a 16 m river
    g = s.graph(with_class=True)
    assert pol.stats[0]["bridge_rule"] == 1 and (g.structs() == BRIDGE).all()
    lake = tile(lambda x, y: 0 * x + 10, lambda x, y: (np.hypot(x - 256, y - 300) < 80).astype(int) * 2)
    pol = TerrainPolicy([lake], [], {"elev_scale": 5, "slope_scale": 0.1}, mode="R")
    s = GenState()
    u = s.add_node(np.array([256.0, 180.0]), push=False)
    pol.apply(0, s, u, np.array([0.0, 90.0]), 2, 0, None, 0.0, None)
    assert s.graph().edges.shape[0] == 0 and sum(v for k, v in pol.stats[0].items() if k.startswith("dropped")) == 1


def test_policy_HR_turns_to_the_contour_on_a_steep_slope():
    steep = tile(lambda x, y: 0.4 * y)
    pol = TerrainPolicy([steep], [], {"elev_scale": 5, "slope_scale": 0.1}, mode="HR")
    s = GenState()
    u = s.add_node(np.array([256.0, 200.0]), push=False)
    mix = (np.zeros(1, np.float32), np.array([[1.0, 0.0]], np.float32), np.full((1, 2), -5.0, np.float32),
           np.zeros(1, np.float32))                         # turtle frame: +x = ahead = uphill here
    import torch
    pol.apply(0, s, u, np.array([0.0, 30.0]), 2, 0, mix, np.pi / 2, torch.Generator().manual_seed(0))
    g = s.graph()
    assert len(g.edges) == 1 and pol.stats[0]["contour"] == 1
    d = g.nodes[1] - g.nodes[0] if np.allclose(g.nodes[0], [256, 200]) else g.nodes[0] - g.nodes[1]
    assert abs(d[1]) < 1.0                                     # along the contour (constant y)


def test_bridge_step_passes_over_streets():
    s = GenState()
    a, b = s.add_node(np.array([100.0, 256.0]), push=False), s.add_node(np.array([400.0, 256.0]), push=False)
    s.connect(a, b, 2)
    u = s.add_node(np.array([250.0, 200.0]), push=False)
    assert apply_bridge(s, u, np.array([0.0, 120.0]), 2)
    g = s.graph()
    assert g.degree().max() == 1 and (g.structs() == BRIDGE).sum() == 1
