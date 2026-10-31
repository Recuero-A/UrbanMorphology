"""Milestone 10: hierarchical / denoising traces, conditioned GraphTurtle, the procedural grid, the block canvas
and BlockGen."""

import numpy as np
import shapely
import torch

from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.procedural import perturbed_grid
from urbanmorph.graphs.road_graph import RoadGraph, build_graph, subgraph
from urbanmorph.graphs.sequence import make_trace, pop_samples, start_samples
from urbanmorph.layout.block_canvas import block_frame, from_canvas, to_canvas, windows
from urbanmorph.models.block_diffusion import BlockDiffusion
from urbanmorph.models.graph_gen import GraphTurtle


def classed_grid():
    """3 x 3 junction grid (100 m), middle row and column major (class 0), the rest local (class 2)."""
    xs = 156.0 + 100.0 * np.arange(3)
    lines, cls = [], []
    for i, v in enumerate(xs):
        for a, b in zip(xs[:-1], xs[1:]):
            lines += [np.array([[v, a], [v, b]]), np.array([[a, v], [b, v]])]
            cls += [0 if i == 1 else 2] * 2
    return build_graph(lines, classes=cls)


def test_subgraph_keeps_only_used_nodes():
    g = classed_grid()
    s = subgraph(g, g.edge_class == 0)
    assert len(s.edges) == (g.edge_class == 0).sum()
    assert set(np.unique(s.edges.ravel())) == set(range(s.n))
    assert (s.edge_class == 0).all()


def test_given_trace_emits_every_free_edge_once_and_starts_on_the_major_network():
    g = classed_grid()
    given = g.edge_class == 0
    tr = make_trace(g, np.random.default_rng(0), turtle=True, given=given)
    assert sum(len(t) for t in tr.targets) == (~given).sum()
    on_major = set(g.edges[given].ravel())
    assert all(int(s) in on_major for s in tr.start_node)          # every local component touches the major roads
    assert set(tr.head_override) == set(int(s) for s in tr.start_node)
    ps = pop_samples(tr, turtle=True)
    assert ps["crops"][0, 0].any()                                  # the major roads are on the canvas at pop 0
    assert (ps["scalars"][:, -1] == 0).all()                        # no "east" start flag: headings come from the arterial


def test_clean_nodes_targets_point_to_clean_positions():
    g = classed_grid()
    rng = np.random.default_rng(1)
    noisy = RoadGraph(g.nodes + rng.normal(0, 2.5, g.nodes.shape) * (~g.border)[:, None], g.edges, g.border, g.edge_class)
    tr = make_trace(noisy, np.random.default_rng(2), turtle=False, clean_nodes=g.nodes)
    for u, tg in zip(tr.pop_node, tr.targets):
        for d in tg:
            p = noisy.nodes[u] + d
            assert np.min(np.hypot(*(g.nodes - p).T)) < 1e-9


def test_conditioned_graph_turtle_and_sampler_with_given_network():
    torch.manual_seed(0)
    g = classed_grid()
    m = GraphTurtle(32, 3, turtle=True, n_classes=4, n_cond=2).eval()
    tr = make_trace(g, np.random.default_rng(0), turtle=True, given=g.edge_class == 0)
    pops, starts = pop_samples(tr, turtle=True), start_samples(tr)
    pops["cond"] = np.tile(np.array([[1, 0]], np.float32), (len(pops["n_targets"]), 1))
    starts["cond"] = np.tile(np.array([[1, 0]], np.float32), (len(starts["stop"]), 1))
    T = lambda d: {k: torch.from_numpy(v) for k, v in d.items()}
    assert torch.isfinite(m.pop_nll(T(pops))).all() and torch.isfinite(m.start_nll(T(starts))).all()
    major = subgraph(g, g.edge_class == 0)
    out = sample_graphs(m, 2, seed=0, max_rounds=30, cond=np.eye(2, dtype=np.float32), given=[major, major])
    for o in out:                                                    # the given network survives in the output
        assert ((o.edge_class == 0).sum()) >= len(major.edges)


def test_perturbed_grid_is_a_connected_grid():
    g = perturbed_grid(80.0, np.random.default_rng(0))
    assert len(g.edges) > 50 and g.edge_class is not None
    assert set(np.unique(g.edge_class)) <= {0, 1, 2}


def test_block_canvas_round_trip_preserves_area():
    block = shapely.affinity.rotate(shapely.box(100, 100, 220, 160), 27, origin=(160, 130))
    bld = [shapely.affinity.rotate(shapely.box(110, 110, 140, 130), 27, origin=(160, 130))]
    fr = block_frame(block)
    assert abs(fr["hl"] - 60) < 1e-6 and abs(fr["hs"] - 30) < 1e-6
    (win,) = windows(fr)
    m = to_canvas(bld, fr, win)
    back = from_canvas(m, fr, win, simplify_m=0.0)
    assert abs(sum(p.area for p in back) - bld[0].area) / bld[0].area < 0.05
    assert shapely.union_all(back).symmetric_difference(bld[0]).area / bld[0].area < 0.1


def test_large_block_gets_overlapping_windows():
    fr = block_frame(shapely.box(0, 0, 500, 100))
    w = windows(fr)
    assert len(w) == 3 and abs(w[1][0] - w[0][0]) < 2 * 128


def test_block_diffusion_loss_and_inpainting_constraint():
    torch.manual_seed(0)
    m = BlockDiffusion(3, ch=(8, 8, 16, 16)).eval()
    x0 = torch.zeros(2, 1, 128, 128) - 1
    cond = torch.zeros(2, 3, 128, 128)
    cond[:, 0, 20:100, 20:100] = 1                                    # block
    cond[:, 2, :, :64] = 1                                            # known: left half
    cond[:, 1, 30:40, 30:40] = 1                                      # one known building
    city = torch.tensor([0, 2])
    assert torch.isfinite(m.loss(x0, cond, city, torch.ones_like(x0)))
    out = m.sample(cond, city, steps=3)
    assert (out[:, :, 30:40, 30:40] == 1).all()                       # the known building is kept
    assert (out[:, :, 20:100, 20:64].sum((1, 2, 3)) == 100).all()     # nothing else in the known part
    assert (out[:, :, :20] == 0).all()                                # nothing outside the block
