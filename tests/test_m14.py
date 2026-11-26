"""Milestone 14: telescoping network potentials, M13 equivalence at lam = 0, trust region, closed-window metrics."""
import numpy as np
import pytest

from test_tc_sampler import flat_tile, neutral_pots, small_model
from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.m14_metrics import closed_components, tile_metrics_m14
from urbanmorph.graphs.m14_sampler import (NetSampler, comp_info, is_closed_island, kl_beta, net_preset,
                                           net_stop_lw, selection_probs)
from urbanmorph.graphs.road_graph import RoadGraph, is_border
from urbanmorph.graphs.tc_potentials import TerrainField
from urbanmorph.graphs.tc_sampler import TCSampler, preset
from urbanmorph.graphs.terrain_generate import terrain_of

NORM = {"elev_scale": 5.0, "slope_scale": 0.1}


def final_counts(g: RoadGraph):
    """Closed interior dead ends and closed islands (>= 40 m, no border node) of a FINAL graph."""
    deg = g.degree()
    D = int(((deg == 1) & ~is_border(g.nodes)).sum())
    n = len(g.nodes)
    lab, L, _ = closed_components(g)
    cl = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=lab.max() + 1)
    touches = lab[n]
    I = int(sum(1 for c in range(len(cl)) if c != touches and cl[c] >= 40))
    return D, I


@pytest.mark.parametrize("seed", [3, 5, 13, 21, 30])
def test_network_weights_telescope_to_final_counts(seed):
    """The product of the network weights of every committed decision equals exp(-lam_D D(x) - lam_I I(x))."""
    lam_d, lam_i = 1.3, 0.7
    cfg = net_preset("na_myopic", K=4, lam_dead=lam_d, lam_island=lam_i)
    sm = NetSampler(small_model(seed), [flat_tile()], [], NORM, neutral_pots(), cfg)
    acc = []
    orig = NetSampler.select

    def spy(self, cands, p_stop, gen):
        j, z = orig(self, cands, p_stop, gen)
        acc.append(cands.stop_lw if j < 0 else cands[j]["lw_net"])
        return j, z
    NetSampler.select = spy
    try:
        g = sm.run(seed=seed, max_rounds=400)[0]
    finally:
        NetSampler.select = orig
    D, I = final_counts(g)
    assert len(g.edges) > 5
    assert np.isclose(sum(acc), -lam_d * D - lam_i * I), (sum(acc), D, I)


def test_lambda_zero_reproduces_m13_energy():
    tiles = [flat_tile(water_cols=(100, 104))]
    g1 = TCSampler(small_model(6), tiles, [], NORM, neutral_pots(), preset("energy", K=4)).run(seed=9, max_rounds=200)[0]
    g2 = NetSampler(small_model(6), tiles, [], NORM, neutral_pots(), net_preset("na_myopic", K=4, lam_ratio=1.0)).run(
        seed=9, max_rounds=200)[0]
    assert np.array_equal(g1.nodes, g2.nodes) and np.array_equal(g1.edges, g2.edges)


def test_trust_region_bound():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lw = rng.normal(0, 3, 8)
        lw[rng.uniform(size=8) < 0.2] = -np.inf
        ps, sl, eps = float(rng.uniform(0.05, 0.6)), float(rng.normal(0, 2)), float(rng.uniform(0.01, 0.3))
        b = kl_beta(lw, ps, sl, eps)
        qs, qk, _ = selection_probs(lw, ps, sl, b)
        rs, rk, _ = selection_probs(np.where(np.isfinite(lw), 0.0, -np.inf), ps, 0.0)
        q, r = np.r_[qs, qk], np.r_[rs, rk]
        m = q > 0
        assert np.sum(q[m] * np.log(q[m] / r[m])) <= eps + 1e-6
        assert (q[~np.isfinite(np.r_[0.0, lw])] == 0).all()
    assert kl_beta(np.zeros(4), 0.3, 0.0, 0.01) == 1.0


def test_island_and_dead_end_bookkeeping():
    s = GenState()
    a = s.add_node(np.array([200.0, 200.0]), push=False)
    b = s.add_node(np.array([260.0, 200.0]), push=False)
    s.connect(a, b)
    cfg = net_preset("na_myopic", lam_dead=1.0, lam_island=2.0)
    lab, clen, cb = comp_info(s)
    assert is_closed_island(s, lab[a], lab, clen, cb, set())            # 60 m, interior, nothing open
    assert not is_closed_island(s, lab[a], lab, clen, cb, {b})          # b still open
    # STOP at b (deg 1, interior) closes a dead end and, a being closed, the island
    assert net_stop_lw(s, b, cfg, set()) == -3.0
    c = s.add_node(np.array([0.0, 200.0]), push=False)                  # on the border
    s.connect(a, c)
    lab, clen, cb = comp_info(s)
    assert cb[lab[a]] and net_stop_lw(s, b, cfg, set()) == -1.0          # touches the border: dead end only


def test_closed_window_metrics():
    t = flat_tile(water_cols=(127, 129))                                 # a river column through the tile centre
    tf = TerrainField(terrain_of(t))
    # a street crossing the whole tile west -> east (border to border) plus an interior island on the east bank
    nodes = np.array([[0.0, 100.0], [200.0, 100.0], [400.0, 100.0], [511.9, 100.0], [350.0, 400.0], [420.0, 400.0]])
    g = RoadGraph(nodes, np.array([[0, 1], [1, 2], [2, 3], [4, 5]]), is_border(nodes))
    m = tile_metrics_m14(g, tf, np.random.default_rng(0), n_pairs=400)
    assert m["islands"] == 1 and 0 < m["island_share"] < 0.2 and m["closed_giant"] > 0.8
    # island <-> east-bank street: same dry region (a failure); island <-> west bank: across the river (legitimate)
    assert m["od_unreach_land"] > 0 and m["od_unreach_water"] > 0
    assert np.isclose(m["od_unreach"], m["od_unreach_land"] + m["od_unreach_water"])
    empty = RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool))
    assert tile_metrics_m14(empty, tf, np.random.default_rng(0))["od_unreach_land"] == 1.0
