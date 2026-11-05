"""Milestone 13 sampler: state consistency, resolve / commit equivalence, hard constraints, particle weights."""
import numpy as np
import torch

from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.sequence import DISP_SCALE, rotate
from urbanmorph.graphs.tc_potentials import NEG_INF, Potentials, TerrainField, edge_features
from urbanmorph.graphs.tc_sampler import TCSampler, preset, resolve_structure
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle


def flat_tile(water_cols=None, z=None):
    elev = np.zeros((256, 256), np.float32) + 100.0 if z is None else z.astype(np.float32)
    water = np.zeros((256, 256), np.int8)
    if water_cols is not None:
        water[:, water_cols[0]:water_cols[1]] = 1
    return {"elev": elev, "valid": np.ones((256, 256), bool), "water": water}


def small_model(seed=0, n_terrain=0):
    """A random model biased to grow: starts rarely stop, arms stop half the time, steps ~ 1 unit (30 m) ahead."""
    torch.manual_seed(seed)
    m = TerrainGraphTurtle(hidden=32, n_mix=3, n_classes=4, n_terrain=n_terrain).eval()
    with torch.no_grad():
        m.start_stop[2].bias.fill_(-3.0)
        b = m.out[2].bias
        K = m.K
        b[0] = 0.0                                   # arm stop logit
        b[1 + K:1 + 2 * K] = 1.0                     # mean x (ahead) = 30 m
        b[1 + 3 * K:1 + 5 * K] = -1.5                # log sigma
        m.out[2].weight.mul_(0.1)
    return m


def neutral_pots():
    return Potentials(r_grade=np.zeros((4, 5, 15)), r_align=np.zeros((5, 10)), span_q={1: {-1: np.array([30.0, 60.0])}},
                      angle_q=np.array([60.0, 80.0]), junction_q=np.array([60.0, 90.0]), span_max={1: 500.0, 2: 100.0})


def test_resolve_then_commit_equals_apply_step():
    rng = np.random.default_rng(0)
    for trial in range(30):
        a, b = GenState(), GenState()
        for s in (a, b):
            s.add_node(np.array([100.0, 100.0]))
        for _ in range(25):
            u = int(rng.integers(a.n))
            d = rng.normal(0, 40, 2)
            a.apply_step(u, d, 2)
            b.commit(u, b.resolve_step(u, d), 2)
        assert np.array_equal(a.nodes[:a.n], b.nodes[:b.n]) and a.edges == b.edges and a.rules == b.rules


def test_resolve_does_not_mutate_and_copy_is_independent():
    s = GenState()
    s.add_node(np.array([100.0, 100.0]))
    s.apply_step(0, np.array([50.0, 0.0]))
    before = (s.n, [list(e) for e in s.edges], s.edge_map.copy())
    for d in (np.array([0.0, 60.0]), np.array([45.0, 3.0]), np.array([2.0, 0.0])):
        s.resolve_step(1, d)
        resolve_structure(s, 1, d)
    assert s.n == before[0] and s.edges == before[1] and np.array_equal(s.edge_map, before[2])
    c = s.copy()
    c.apply_step(1, np.array([0.0, 60.0]))
    assert c.n == s.n + 1 and s.n == before[0] and np.array_equal(s.edge_map, before[2])


def test_hard_constraints_on_committed_edges():
    tf = TerrainField(terrain_of(flat_tile(water_cols=(120, 130))))          # river at x ~ 224-264 (tile metres)
    pots = neutral_pots()
    across = edge_features(tf, np.array([200.0, 300.0]), np.array([290.0, 300.0]), struct=0)
    assert pots.log_potential(across, 2, "new_node")[1] == "ground_in_water"
    bridge = edge_features(tf, np.array([200.0, 300.0]), np.array([290.0, 300.0]), struct=BRIDGE)
    lw, reason, _ = pots.log_potential(bridge, 2, "struct_new")
    assert reason == "" and np.isfinite(lw)
    dry_bridge = edge_features(tf, np.array([20.0, 20.0]), np.array([80.0, 20.0]), struct=BRIDGE)
    assert pots.log_potential(dry_bridge, 2, "struct_new")[1] == "unsupported_structure"
    into = edge_features(tf, np.array([200.0, 300.0]), np.array([240.0, 300.0]), struct=0)
    assert pots.log_potential(into, 2, "new_node")[1] == "ends_in_water"
    steep = flat_tile(z=np.tile(np.linspace(0, 500, 256), (256, 1)))         # ~49 % grade: soft, never hard
    tfs = TerrainField(terrain_of(steep))
    f = edge_features(tfs, np.array([100.0, 100.0]), np.array([150.0, 100.0]))
    assert f["grade"] > 0.4 and pots.log_potential(f, 2, "new_node")[1] == ""
    assert pots.log_potential(f, 2, "dropped")[0] == NEG_INF


class Recorder:
    """Wraps a model and records every decoder input displacement."""

    def __init__(self, m):
        self.m, self.prev = m, []

    def __getattr__(self, k):
        return getattr(self.m, k)

    def decode_steps(self, ctx, prev, h=None, prev_class=None):
        self.prev.append(prev[:, 0].cpu().numpy().copy())
        return self.m.decode_steps(ctx, prev, h, prev_class)


def test_decoder_is_fed_the_committed_edge():
    """State consistency: after an arm is committed, the next decoder input is exactly the COMMITTED displacement
    (rotated into the turtle frame), even when the merge / T-junction rules moved the end point."""
    model = small_model(1)
    rec = Recorder(model)
    sm = TCSampler(rec, [flat_tile()], [], {"elev_scale": 5.0, "slope_scale": 0.1}, neutral_pots(), preset("energy", K=4))
    commits = []
    orig = TCSampler.commit_candidate

    def spy(s, u, c):
        v = orig(s, u, c)
        commits.append((s.nodes[u].copy(), c["o"]["q"].copy()))
        return v
    TCSampler.commit_candidate = staticmethod(spy)
    try:
        sm.run(seed=3, max_rounds=40)
    finally:
        TCSampler.commit_candidate = staticmethod(orig)
    fed = np.concatenate([p for p in rec.prev])
    fed = fed[np.abs(fed).sum(1) > 0]                                        # zero = first arm of a node
    lengths_fed = np.sort(np.round(np.hypot(*fed.T) * DISP_SCALE, 3))
    lengths_commit = np.sort(np.round([np.hypot(*(q - a)) for a, q in commits], 3))
    assert len(lengths_fed) <= len(lengths_commit)
    assert len(lengths_fed) > 0
    assert all(np.abs(lengths_commit - x).min() < 1e-3 for x in lengths_fed)


def test_scored_geometry_is_inserted_geometry():
    model = small_model(2)
    sm = TCSampler(model, [flat_tile(water_cols=(100, 104))], [], {"elev_scale": 5.0, "slope_scale": 0.1},
                   neutral_pots(), preset("filter", K=6))
    seen = []
    orig = TCSampler.commit_candidate

    def spy(s, u, c):
        n0 = s.n
        v = orig(s, u, c)
        assert np.allclose(s.nodes[v], c["o"]["q"])                         # inserted end == scored end
        assert c["reason"] == ""                                             # never a hard-infeasible candidate
        seen.append(n0)
        return v
    TCSampler.commit_candidate = staticmethod(spy)
    try:
        g = sm.run(seed=5, max_rounds=60)[0]
    finally:
        TCSampler.commit_candidate = staticmethod(orig)
    assert len(seen) > 0
    tf = sm.tfs[0]
    for (x0, y0, x1, y1), st in zip(g.segments(), g.structs()):
        f = edge_features(tf, np.array([x0, y0]), np.array([x1, y1]), struct=int(st))
        assert not (st == 0 and f["wet_length"] > 0)                         # no ground edge over water


def test_smc_weights_stay_equal_without_constraints_violations():
    """With neutral potentials and no hard constraints every candidate has w = 1, so Z = 1: no particle is preferred."""
    model = small_model(3)
    sm = TCSampler(model, [flat_tile()], [], {"elev_scale": 5.0, "slope_scale": 0.1}, neutral_pots(),
                   preset("smc", K=4, particles=3, hard=False))
    sm.run(seed=7, max_rounds=30)
    assert sm.stats[0]["resamples"] == 0


def test_sampler_weights_reach_the_potentials():
    """Regression (M13 bug found at selection): lam_ratio / lam_infra / lam_geom of the SamplerConfig must be used."""
    pots = neutral_pots()
    pots.r_grade = np.full((4, 5, 15), -1.0)
    sm1 = TCSampler(small_model(4), [flat_tile()], [], {"elev_scale": 5.0, "slope_scale": 0.1}, pots, preset("energy", lam_ratio=0.5))
    sm2 = TCSampler(small_model(4), [flat_tile()], [], {"elev_scale": 5.0, "slope_scale": 0.1}, pots, preset("energy", lam_ratio=2.0))
    assert sm1.pots.lam_ratio == 0.5 and sm2.pots.lam_ratio == 2.0 and pots.lam_ratio == 1.0
    f = edge_features(sm1.tfs[0], np.array([100.0, 100.0]), np.array([140.0, 100.0]))
    assert sm1.pots.log_potential(f, 2, "new_node")[0] == -0.5 and sm2.pots.log_potential(f, 2, "new_node")[0] == -2.0
