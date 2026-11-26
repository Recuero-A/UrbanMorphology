"""Milestone 5: street graphs, construction sequences, the GraphTurtle density and the sampler."""

import math

import numpy as np
import torch

from urbanmorph.graphs.generate import GenState, clip_to_tile, sample_graphs
from urbanmorph.graphs.road_graph import RoadGraph, build_graph, components, graph_metrics, rasterize_graph
from urbanmorph.graphs.sequence import (PAD, crops_from_maps, make_trace, pixel_rc, pop_samples,
                                        scalar_features, start_samples)
from urbanmorph.models.graph_gen import GraphTurtle, mixture_log_prob


def grid_lines(n=4, spacing=100.0, x0=106.0):
    """n x n street grid, every street as one polyline crossing the whole grid, plus reversed duplicates."""
    xs = x0 + spacing * np.arange(n)
    lines = []
    for v in xs:
        lines.append(np.array([[v, xs[0]], [v, xs[-1]]]))
        lines.append(np.array([[xs[0], v], [xs[-1], v]]))
    # junctions as polyline vertices, as in OSMnx edges (streets split at intersections)
    split = []
    for ln in lines:
        a, b = ln
        pts = [a + (b - a) * k / (n - 1) for k in range(n)]
        split += [np.array([p, q]) for p, q in zip(pts[:-1], pts[1:])]
    return split + [s[::-1] for s in split]


def test_build_graph_dedupes_two_way_streets_and_resamples():
    g = build_graph(grid_lines())
    # 4 x 4 junctions; every 100 m block side is resampled into 4 steps of 25 m (3 extra nodes)
    assert g.n == 16 + 24 * 3
    assert len(g.edges) == 24 * 4
    assert len(np.unique(components(g))) == 1
    m = graph_metrics(g)
    assert m["components"] == 1 and m["interior_components"] == 1    # the grid does not reach the tile edge
    assert m["loops_per_km2"] * (0.512 ** 2) == 9           # 3 x 3 blocks
    assert abs(m["intersections_per_km2"] * 0.512 ** 2 - 12) < 1e-9   # 4 corners have degree 2
    assert m["orientation_phi"] > 0.99                       # perfect grid
    assert rasterize_graph(g).sum() > 0


def test_border_nodes_and_symmetry():
    # a T-junction (shared vertex); lines that cross WITHOUT a shared vertex stay unjoined (bridges, as in OSM)
    g = build_graph([np.array([[0.0, 256.0], [256.0, 256.0], [512.0, 256.0]]), np.array([[256.0, 0.0], [256.0, 256.0]])])
    assert g.border.sum() == 3
    m = graph_metrics(g)
    assert m["components"] == 1 and m["interior_components"] == 0
    for k in range(8):
        h = g.transform(k)
        assert np.allclose(np.sort(np.hypot(*(h.segments()[:, 2:] - h.segments()[:, :2]).T)),
                           np.sort(np.hypot(*(g.segments()[:, 2:] - g.segments()[:, :2]).T)))
        assert ((h.nodes >= -1e-9) & (h.nodes <= 512 + 1e-9)).all()


def test_trace_emits_every_edge_once_and_rebuilds_the_graph():
    g = build_graph(grid_lines() + [np.array([[20.0, 20.0], [60.0, 20.0]])])     # + a separate piece
    tr = make_trace(g, np.random.default_rng(0))
    assert len(tr.start_t) == 2
    assert len(tr.pop_node) == g.n                          # every node popped once
    assert sum(len(t) for t in tr.targets) == len(g.edges)  # every edge emitted once
    deg = g.degree()
    for u, arms, tg in zip(tr.pop_node, tr.arms, tr.targets):
        assert len(arms) + len(tg) == deg[u]
    # replaying the displacements reproduces the edge set
    rebuilt = set()
    for u, tg in zip(tr.pop_node, tr.targets):
        for d in tg:
            w = int(np.argmin(np.hypot(*(g.nodes - (g.nodes[u] + d)).T)))
            rebuilt.add((min(u, w), max(u, w)))
    assert rebuilt == {tuple(e) for e in g.edges.tolist()}
    s = start_samples(tr)
    assert s["stop"].tolist() == [0.0, 0.0, 1.0]


def test_training_crops_equal_generation_crops_on_replay():
    """The sampler must see exactly what the model saw in training: replay the true steps through GenState."""
    g = build_graph(grid_lines())
    tr = make_trace(g, np.random.default_rng(1))
    ps = pop_samples(tr)
    st = GenState()
    st.add_node(g.nodes[tr.start_node[0]])
    for i, (u_true, tg) in enumerate(zip(tr.pop_node, tr.targets)):
        u = st.queue.popleft()
        assert np.allclose(st.nodes[u], g.nodes[u_true])
        r, c = pixel_rc(st.nodes[u])
        st.popped_map[r + PAD, c + PAD] = st.t
        crop = crops_from_maps(st.edge_map, st.created_map, st.popped_map, r, c, np.array([st.t]))[0]
        assert (crop == ps["crops"][i]).all(), f"crop mismatch at pop {i}"
        scal = scalar_features(st.nodes[u], np.array([st.border[u]]), np.array([len(st.adj[u])]),
                               np.array([st.n]), np.array([st.t]))[0]
        assert np.allclose(scal, ps["scalars"][i])
        for d in tg:
            st.apply_step(u, d)
        st.t += 1
    h = st.graph()
    assert h.n == g.n and len(h.edges) == len(g.edges)


def test_crossing_creates_a_junction_and_clipping_ends_on_border():
    st = GenState()
    a = st.add_node(np.array([100.0, 200.0]))
    b = st.add_node(np.array([200.0, 200.0]))
    st.connect(a, b)
    c = st.add_node(np.array([150.0, 180.0]))
    st.apply_step(c, np.array([0.0, 40.0]))                 # crosses the street a-b at (150, 200)
    g = st.graph()
    assert g.n == 4 and len(g.edges) == 3
    assert len(np.unique(components(g))) == 1
    j = int(np.argmin(np.hypot(*(g.nodes - [150.0, 200.0]).T)))
    assert np.allclose(g.nodes[j], [150.0, 200.0]) and g.degree()[j] == 3
    q, out = clip_to_tile(np.array([500.0, 100.0]), np.array([540.0, 100.0]))
    assert out and np.allclose(q, [512.0, 100.0])


def test_mixture_density_integrates_to_one():
    torch.manual_seed(0)
    K = 3
    lp, mu = torch.randn(K), 0.3 * torch.randn(K, 2)
    ls, rho = torch.full((K, 2), -1.0), torch.tensor([0.5, -0.3, 0.0])
    xs = torch.linspace(-4, 4, 401)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([X, Y], -1).view(-1, 2)
    dens = mixture_log_prob(pts, lp.expand(len(pts), K), mu.expand(len(pts), K, 2), ls.expand(len(pts), K, 2),
                            rho.expand(len(pts), K)).exp()
    assert abs(dens.sum().item() * (xs[1] - xs[0]).item() ** 2 - 1) < 1e-3


def test_training_lowers_nll_and_sampler_returns_valid_graphs():
    torch.manual_seed(0)
    g = build_graph(grid_lines())
    tr = make_trace(g, np.random.default_rng(0))
    b = {k: torch.from_numpy(v) for k, v in pop_samples(tr).items()}
    s = {k: torch.from_numpy(v) for k, v in start_samples(tr).items()}
    model = GraphTurtle(hidden=64, n_mix=3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = model.pop_nll(b).mean().item() + model.start_nll(s).mean().item()
    for _ in range(60):
        loss = model.pop_nll(b).mean() + model.start_nll(s).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item()) and loss.item() < first - 1
    graphs = sample_graphs(model, 3, seed=0, max_rounds=60)
    for h in graphs:
        assert isinstance(h, RoadGraph)
        assert ((h.nodes >= -1e-6) & (h.nodes <= 512 + 1e-6)).all()
        if len(h.edges):
            assert (h.edges[:, 0] < h.edges[:, 1]).all() and h.edges.max() < h.n


# ------------------------------------------------------------------ Milestone 6: turtle frame + classes
from urbanmorph.graphs.sequence import crops_turtle, heading_of, rotate  # noqa: E402


def test_turtle_replay_matches_training_features_and_classes():
    lines = grid_lines()
    g = build_graph(lines, classes=[k % 4 for k in range(len(lines))])
    assert g.edge_class is not None and set(g.edge_class.tolist()) <= {0, 1, 2, 3}
    tr = make_trace(g, np.random.default_rng(3), turtle=True)
    ps = pop_samples(tr, turtle=True)
    st = GenState()
    st.add_node(g.nodes[tr.start_node[0]])
    for i, u_true in enumerate(tr.pop_node):
        u = st.queue.popleft()
        assert np.allclose(st.nodes[u], g.nodes[u_true])
        r, c = pixel_rc(st.nodes[u])
        st.popped_map[r + PAD, c + PAD] = st.t
        hd, flag = heading_of(st.nodes, u, st.parent[u])
        crop = crops_turtle(st.edge_map, st.created_map, st.popped_map, st.nodes[u], hd, [st.t])[0]
        assert (crop == ps["crops"][i]).all(), f"turtle crop mismatch at pop {i}"
        assert ps["scalars"][i, -1] == flag
        for k in range(ps["n_targets"][i]):
            st.apply_step(u, rotate(ps["targets"][i, k] * 30.0, hd), int(ps["target_class"][i, k]))
        st.t += 1
    h = st.graph(with_class=True)
    assert h.n == g.n and len(h.edges) == len(g.edges)
    key = lambda gr: sorted((*sorted([tuple(np.round(gr.nodes[a], 3)), tuple(np.round(gr.nodes[b], 3))]), int(c))
                            for (a, b), c in zip(gr.edges, gr.classes()))
    assert key(h) == key(g)


def test_turtle_inputs_are_rotation_equivariant():
    """Rotating the world by 90 deg and the heading by 90 deg gives the same crop and the same local steps.

    (BFS order itself depends on the start node's absolute frame, so whole traces of a rotated graph
    are not compared pop by pop; equivariance is a property of each pop's inputs and outputs.)"""
    g = build_graph(grid_lines(x0=107.3))                     # avoid positions on pixel boundaries
    tr = make_trace(g, np.random.default_rng(5), turtle=True)
    ps = pop_samples(tr, turtle=True)
    rot_maps = [np.rot90(m) for m in (tr.edge_map, tr.created_map, tr.popped_map)]   # world rotated CCW 90
    checked = 0
    for i, u in enumerate(tr.pop_node):
        if tr.parent[u] < 0:
            continue
        hd = heading_of(g.nodes, u, tr.parent[u])[0]
        p = g.nodes[u] - 256.0
        p_rot = np.array([-p[1], p[0]]) + 256.0
        c1 = crops_turtle(tr.edge_map, tr.created_map, tr.popped_map, g.nodes[u], hd, [i])[0]
        c2 = crops_turtle(*rot_maps, p_rot, hd + np.pi / 2, [i])[0]
        assert (c1 == c2).all()
        for k in range(ps["n_targets"][i]):
            world = rotate(ps["targets"][i, k], hd)
            assert np.allclose(rotate(rotate(world, np.pi / 2), -(hd + np.pi / 2)), ps["targets"][i, k], atol=1e-6)
        checked += 1
    assert checked > 50


def test_turtle_model_with_classes_trains_and_samples():
    torch.manual_seed(0)
    lines = grid_lines()
    g = build_graph(lines, classes=[2] * len(lines))
    tr = make_trace(g, np.random.default_rng(0), turtle=True)
    b = {k: torch.from_numpy(v) for k, v in pop_samples(tr, turtle=True).items()}
    model = GraphTurtle(hidden=64, n_mix=3, turtle=True, n_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = model.pop_nll(b).mean().item()
    for _ in range(40):
        loss = model.pop_nll(b).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 1
    for h in sample_graphs(model, 2, seed=0, max_rounds=40):
        assert h.edge_class is not None and len(h.edge_class) == len(h.edges)
