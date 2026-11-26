"""Milestone 15: style factor, grade ceiling, fixtures and style / terrain metrics on analytic cases."""
import math

import numpy as np
import pytest

from urbanmorph.graphs import fixtures as fx
from urbanmorph.graphs.road_graph import RoadGraph
from urbanmorph.graphs.style import StyleField, closes_loop, grade_caps, parse_mix, stable_key
from urbanmorph.graphs.style_metrics import (block_areas, culdesac, estimate_centre, grade_exposure, junction_shares,
                                             morphology_row, od_circuity, orth_order, radial_tangential, winding)
from urbanmorph.graphs.tc_sampler import SamplerConfig
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.metrics.fidelity import prdc, real_vs_real


def field(style, **kw):
    return StyleField.build(SamplerConfig(style=style, style_kappa=2.0, **kw), "site-a", 1)


# ---------------------------------------------------------------- style factor
def test_off_is_inactive_and_defaults_unchanged():
    sf = StyleField.build(SamplerConfig(), "s", 1)
    assert not sf.active
    assert sf.heading_lw(np.zeros(2), np.array([30.0, 5.0]), 2) == 0.0


def test_grid_cross_field():
    sf = field("grid", style_theta0=10.0)
    a = np.array([100.0, 100.0])
    along = a + 30 * np.array([math.cos(math.radians(10)), math.sin(math.radians(10))])
    perp = a + 30 * np.array([math.cos(math.radians(100)), math.sin(math.radians(100))])
    diag = a + 30 * np.array([math.cos(math.radians(55)), math.sin(math.radians(55))])
    assert sf.heading_lw(a, along, 2) == pytest.approx(0.0, abs=1e-9)
    assert sf.heading_lw(a, perp, 2) == pytest.approx(0.0, abs=1e-9)
    assert sf.heading_lw(a, diag, 2) == pytest.approx(-2.0)
    assert sf.heading_lw(a, diag, 0) == pytest.approx(-0.5)             # backbone guided at gain 0.25


def test_radial_prefers_spokes_and_rings():
    sf = field("radial")
    p = np.array([356.0, 256.0])                                        # 100 m east of the centre
    assert sf.heading_lw(p, p + [30, 0], 2) == pytest.approx(0.0, abs=1e-9)       # radial
    off = math.atan2(15, 100)                                           # the midpoint sits 8.5 degrees round the ring
    assert sf.heading_lw(p, p + [0, 30], 2) == pytest.approx(2.0 * (math.cos(4 * off) - 1) / 2)   # tangential
    assert sf.heading_lw(p, p + [21, 21], 2) < -1.5                                # diagonal


def test_organic_field_smooth_and_keyed():
    cfg = SamplerConfig(style="organic", style_kappa=1.0)
    f1, f2 = StyleField.build(cfg, "site", 1), StyleField.build(cfg, "site", 1)
    f3 = StyleField.build(cfg, "site", 2)
    pts = [np.array([x, y]) for x in (50.0, 250.0, 450.0) for y in (50.0, 250.0, 450.0)]
    d1 = [f1.direction("organic", p) for p in pts]
    assert d1 == [f2.direction("organic", p) for p in pts]               # stable keyed randomness
    assert d1 != [f3.direction("organic", p) for p in pts]
    near = abs(f1.direction("organic", pts[4]) - f1.direction("organic", pts[4] + [10, 0]))
    assert near < math.radians(10)                                      # smooth at 10 m
    assert np.std(d1) > math.radians(5)                                 # but varies over the tile
    assert stable_key("a", 1) == stable_key("a", 1) != stable_key("a", 2)


def test_mixture_weights():
    assert parse_mix("mix:grid=1+organic=3") == {"grid": 0.25, "organic": 0.75}
    assert parse_mix("off") == {} and parse_mix("culdesac") == {}
    with pytest.raises(ValueError):
        parse_mix("hexagonal")


def test_loop_closure_and_culdesac_control():
    comp = np.array([0, 0, 0, 1])
    s = type("S", (), {})()
    assert closes_loop(s, 0, {"kind": "merge_node", "v": 2}, comp)
    assert not closes_loop(s, 0, {"kind": "merge_node", "v": 3}, comp)
    assert closes_loop(s, 0, {"kind": "t_junction", "edge": (1, 2)}, comp)
    assert not closes_loop(s, 0, {"kind": "new_node"}, comp)
    sf = StyleField.build(SamplerConfig(style="culdesac", style_loop=3.0), "s", 1)
    assert sf.active and sf.loop_lw(True, 2) == -3.0 and sf.loop_lw(False, 2) == 0.0


def test_grade_caps():
    assert grade_caps("off") is None
    assert grade_caps("0.12") == {c: 0.12 for c in range(4)}
    real = grade_caps("real_p99")
    assert set(real) == {0, 1, 2, 3} and all(0.05 < v < 0.3 for v in real.values())


# ---------------------------------------------------------------- fixtures and metrics
def test_terrain_fixtures():
    inc = terrain_of(fx.terrain("incline", grade=0.10, az=0.0))
    assert inc.sample(np.array([300.0]), np.array([256.0]))[0] - inc.sample(np.array([200.0]), np.array([256.0]))[0] == pytest.approx(10.0, abs=1e-3)
    w = fx.terrain("water", w=40.0)
    assert (w["water"] == 1).mean() == pytest.approx(40 / 1024, abs=0.004)   # a 40 m river across a 1024 m window
    r = fx.terrain("ridge", h=40.0)
    assert r["elev"].max() == pytest.approx(90.0, abs=0.5)


def test_grade_exposure_on_incline():
    t = terrain_of(fx.terrain("incline", grade=0.10, az=0.0))
    up = RoadGraph(np.array([[100.0, 256.0], [400.0, 256.0]]), np.array([[0, 1]]), np.zeros(2, bool))
    across = RoadGraph(np.array([[256.0, 100.0], [256.0, 400.0]]), np.array([[0, 1]]), np.zeros(2, bool))
    assert grade_exposure(up, t)["grade_p90"] == pytest.approx(0.10, abs=1e-3)
    assert grade_exposure(across, t)["grade_p90"] == pytest.approx(0.0, abs=1e-3)
    assert grade_exposure(up, t)["share_gt15"] == 0.0


def test_grid_metrics_and_rotation_invariance():
    g0, g30 = fx.grid(64, 0.0), fx.grid(64, 30.0)
    assert orth_order(g0) == pytest.approx(1.0, abs=1e-9)
    assert orth_order(g30) == pytest.approx(1.0, abs=1e-6)
    assert junction_shares(g0)["share_4way"] > 0.7
    ba = block_areas(g0)
    assert np.median(ba) == pytest.approx(64 * 64, rel=1e-6)
    assert culdesac(g0)["deadend_per_km"] == 0.0
    assert winding(g0)["turn_med_deg"] == pytest.approx(0.0, abs=1e-6)
    c = od_circuity(g0, np.random.default_rng(0))
    assert 1.0 <= c["circuity_200_350"] <= math.sqrt(2) + 0.05 and c["unreach_200_350"] == 0.0


def test_spoke_ring_radial_and_centre():
    g = fx.spoke_ring(centre=(256.0, 256.0))
    r = radial_tangential(g, (256.0, 256.0))
    assert r["radial_share"] > 0.3 and r["tangential_share"] > 0.3
    assert r["radial_share"] + r["tangential_share"] > 0.9
    cx, cy = estimate_centre(g)
    assert math.hypot(cx - 256, cy - 256) <= 32.0
    rg = radial_tangential(fx.grid(64, 0.0), (256.0, 256.0))           # a grid is only partly radial
    assert rg["radial_share"] + rg["tangential_share"] < r["radial_share"] + r["tangential_share"]


def test_dendritic_vs_grid_culdesac():
    d = culdesac(fx.dendritic())
    assert d["loops_per_km2"] == 0.0 and d["deadend_per_km"] > 1.0
    assert culdesac(fx.grid())["loops_per_km2"] > 50


def test_degraded_network_increases_dead_ends():
    g = fx.grid(64, 0.0)
    rng = np.random.default_rng(1)
    keep = rng.random(len(g.edges)) > 0.15
    deg = RoadGraph(g.nodes, g.edges[keep], g.border, edge_class=g.edge_class[keep], edge_struct=g.edge_struct[keep])
    assert culdesac(deg)["deadend_per_km"] > culdesac(g)["deadend_per_km"]
    assert culdesac(deg)["loops_per_km2"] < culdesac(g)["loops_per_km2"]


def test_morphology_row_complete():
    row = morphology_row(fx.grid(), terrain_of(fx.terrain("flat")))
    for k in ("street_km", "orth_order", "share_4way", "sinuosity", "radial_share", "deadend_per_km", "loops_per_km2",
              "block_area_med", "circuity_100_200", "grade_p90"):
        assert k in row


def test_prdc_sanity():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 4))
    same = prdc(real, real.copy())
    assert same["precision"] == 1.0 and same["coverage"] == 1.0
    far = prdc(real, real + 10.0)
    assert far["precision"] == 0.0 and far["coverage"] == 0.0
    rr = real_vs_real(real, np.arange(200) % 10)
    assert 0.5 < rr["coverage"] <= 1.0
