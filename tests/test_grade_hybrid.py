"""Milestone 20 D4 grade hybrid (scripts/m20_hybrid.py) on small synthetic terrains and graphs with known answers,
with the committee's D4a rulings (docs/M20_LOG.md) tested at the end.

The sampler is the injectable stub (GenState.resolve_step only); commits go through the real GenState.commit."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m20_hybrid as hy  # noqa: E402
from urbanmorph.graphs.generate import GenState  # noqa: E402
from urbanmorph.graphs.road_graph import RoadGraph  # noqa: E402
from urbanmorph.metrics import hilltown as H  # noqa: E402
from urbanmorph.terrain.heightfield import Terrain  # noqa: E402

SIZE = 512.0


# ---------------------------------------------------------------- fixtures
def terrain(zfun, size=SIZE, res=4.0, margin=256.0) -> Terrain:
    n = int((size + 2 * margin) / res)
    X, Y = np.meshgrid(-margin + (np.arange(n) + 0.5) * res, size + margin - (np.arange(n) + 0.5) * res)
    t = Terrain(zfun(X, Y).astype(np.float32), -margin, size + margin, res)
    t.water_kind = np.zeros(t.elev.shape, np.int8)
    return t


def state(lines, size=SIZE, step=25.0) -> GenState:
    """A GenState holding the polylines, cut into edges of at most ``step`` m (shared end points join)."""
    s = GenState(size=size)
    ids = {}

    def node(p):
        k = (round(float(p[0]), 3), round(float(p[1]), 3))
        if k not in ids:
            ids[k] = s.add_node(np.array(p, float), push=False)
        return ids[k]
    for L in lines:
        L = np.asarray(L, float)
        pts = [L[0]]
        for a, b in zip(L[:-1], L[1:]):
            m = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
            pts += [a + (b - a) * j / m for j in range(1, m + 1)]
        for a, b in zip(pts[:-1], pts[1:]):
            s.connect(node(a), node(b), 2)
    s.queue.clear()
    return s


def grad_of(T):
    gx, gy = T.gradient()

    def f(x, y):
        r, c = T.rc(np.atleast_1d(x), np.atleast_1d(y))
        r = np.clip(np.round(r).astype(int), 0, gx.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, gx.shape[1] - 1)
        return gx[r, c], gy[r, c]
    return f


def run(lines, T, **kw):
    s = state(lines)
    return s, hy.run_hybrid(s, T, grad_of(T), hy.stub_evaluator, **kw)


def mids(g):
    return (g.nodes[g.edges[:, 0]] + g.nodes[g.edges[:, 1]]) / 2


def on_line(P, a, b, tol=0.5):
    """Points of P within tol of segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return hy._seg_dist(P, np.repeat(a[None], len(P), 0), np.repeat(b[None], len(P), 0)) <= tol


def sub(g, mask):
    return RoadGraph(g.nodes, g.edges[mask], np.zeros(g.n, bool))


def field(T):
    return H.HillField(T, np.ones(T.elev.shape, bool))


PLANE20 = terrain(lambda X, Y: 0.20 * Y)
STREET = [(250, 100), (250, 250)]                       # straight up the slope: 150 m at 20 %
CONTOURS = [[(100, 100), (450, 100)], [(100, 250), (450, 250)]]
ALT = [(300, 100), (450, 175), (300, 250)]               # a gentle (8.9 %) two-leg alternative, detour 2.9
SHORT = [[(200, 100), (300, 100)], [(200, 250), (300, 250)]]   # contour streets too short to detour along


# ---------------------------------------------------------------- H2 rule (1): stairs
def test_steep_street_with_gentle_alternative_becomes_stairs():
    _, res = run([STREET] + CONTOURS + [ALT], PLANE20, h1=False)
    g, k = res["graph"], res["kind"]
    st = on_line(mids(g), *STREET)
    assert st.any() and (k[st] != 0).all()                # the whole run is pedestrian
    assert (k[st] == 1).all()                             # 15-25 %: typed stepped ramp (D4 stair placement)
    assert (k[~st] == 0).all()                            # nothing else changes
    log = res["log"]["H2"]
    assert log["runs_stairs"] == 1 and log.get("runs_switchback", 0) == 0 and log.get("runs_kept", 0) == 0
    assert res["gross_m"] == 0.0
    assert res["km"]["stepped_ramp"] == pytest.approx(0.15, abs=1e-6)


def test_steeper_than_25_percent_is_typed_stairs():
    T = terrain(lambda X, Y: 0.28 * Y)
    street = [(250, 100), (250, 200)]
    contours = [[(100, 100), (450, 100)], [(100, 200), (450, 200)]]
    alt = [(275, 100), (425, 150), (275, 200)]            # legs at 8.9 %, detour 3.7
    _, res = run([street] + contours + [alt], T, h1=False)
    g, k = res["graph"], res["kind"]
    st = on_line(mids(g), *street)
    assert (k[st] == 2).all() and res["log"]["H2"]["runs_stairs"] == 1


# ---------------------------------------------------------------- H2 rule (2): switchback
def test_switchback_without_alternative():
    s, res = run([STREET] + SHORT, PLANE20, h1=False)
    g, k = res["graph"], res["kind"]
    log = res["log"]["H2"]
    assert log["runs_switchback"] == 1 and log.get("runs_kept", 0) == 0 and log["drivable_claim"] == 1
    m = mids(g)
    old = on_line(m, *STREET) | on_line(m, *SHORT[0]) | on_line(m, *SHORT[1])
    assert (k[on_line(m, *STREET)] != 0).all()            # the run became stairs
    new = ~old
    assert new.sum() >= 4 and (k[new] == 0).all()         # the switchback is vehicular
    L = np.hypot(*(g.nodes[g.edges[new, 0]] - g.nodes[g.edges[new, 1]]).T).sum()
    assert res["gross_m"] == pytest.approx(L, rel=1e-6) == pytest.approx(log["switchback_added_m"], rel=1e-6)
    # pieces <= 15 % (30 m window, as H2 and G1 measure), except short ramps (<= 60 m, <= 25 %) which D4 allows:
    # a smooth hairpin on a uniform 20 % slope turns through the fall line, so its apex cannot be gentler
    sg = sub(g, new)
    F = field(PLANE20)
    pc = H.pieces(sg, F)
    wg = H.window_grades(sg, F, pc)
    assert np.nanmax(wg) <= hy.RAMP_PCT
    assert pc.length[wg > 15].sum() <= 0.15 * pc.length.sum()
    an = hy.Analysis(res["state"], F, hy.Params())
    assert all(an.short_ramp(r) for r in an.runs)          # no steep run left on the network
    hp = H.hairpins(sg)
    assert all(h[2] >= hy.RADIUS_M for h in hp)           # any G2 hairpin at >= 12 m centreline radius
    u = np.diff(np.vstack([P_ for P_ in H._M18.ground_polylines(sg)]), axis=0)
    assert (u[:, 0] > 1).any() and (u[:, 0] < -1).any()   # the legs reverse across the slope: a switchback


def test_switchback_commit_is_all_or_nothing_under_the_sampler_rules():
    calls = []

    def refuse_late(s, u, d):                             # an injected sampler that refuses the 4th step
        calls.append(1)
        o, ok, r = hy.stub_evaluator(s, u, d)
        return (o, False, "grade_cap") if len(calls) == 4 else (o, ok, r)
    s = state([STREET] + SHORT)
    res = hy.run_hybrid(s, PLANE20, grad_of(PLANE20), refuse_late, h1=False)
    log = res["log"]["H2"]
    assert log["runs_kept"] == 1 and log["kept_commit_failure"] == 1 and log["infeasible:grade_cap"] == 1
    assert (res["kind"] == 0).all() and res["gross_m"] == 0.0
    g0 = s.graph(with_class=True)                         # the partial switchback was discarded
    assert np.array_equal(res["graph"].nodes, g0.nodes) and np.array_equal(res["graph"].edges, g0.edges)


def test_kept_when_no_room():
    T = terrain(lambda X, Y: 0.20 * Y)
    water = np.zeros(T.elev.shape, np.int8)                # water on both sides of the street: one dry cell column,
    X = T.x0 + (np.arange(T.elev.shape[1]) + 0.5) * T.res  # so the only way up is the 20 % fall line itself
    water[:, (np.abs(X - 250) > 2.5) & (np.abs(X - 250) < 200)] = 3
    T.water_kind = water
    _, res = run([STREET, [(230, 100), (270, 100)], [(230, 250), (270, 250)]], T, h1=False)
    log = res["log"]["H2"]
    assert log["runs_kept"] == 1 and log["kept_failed_for_room"] == 1 and log["drivable_claim"] == 0
    assert log["kept_m"] == pytest.approx(150.0, abs=1e-6)
    assert (res["kind"] == 0).all()


# ---------------------------------------------------------------- forced stairs and short ramps
def test_forty_percent_piece_always_becomes_stairs():
    T = terrain(lambda X, Y: 0.40 * Y)
    street = [(250, 100), (250, 200)]
    _, res = run([street, [(150, 100), (350, 100)]], T, h1=False)
    g, k = res["graph"], res["kind"]
    st = on_line(mids(g), *street)
    assert st.any() and (k[st] == 2).all()
    assert res["log"]["H2"]["forced_stairs"] >= 1


def test_forced_stairs_log_the_vehicular_piece_cut_off():
    T = terrain(lambda X, Y: 0.40 * np.clip(Y - 150, 0, 50))     # flat, a 40 % step 150-200, flat
    street = [(250, 50), (250, 350)]
    _, res = run([street, [(150, 50), (350, 50)]], T, h1=False)
    g, k = res["graph"], res["kind"]
    F = field(T)
    pc = H.pieces(g, F)
    wg = H.window_grades(sub(g, np.ones(len(g.edges), bool)), F, pc)
    assert (k[pc.edge[wg > 30]] == 2).all()                # every piece above 30 % is stairs
    log = res["log"]["H2"]
    assert log["forced_stairs"] >= 1
    assert log["forced_cutoff_pieces"] >= 1 and log["forced_cutoff_m"] > 100


def ramp_terrain(length):
    return terrain(lambda X, Y: 0.20 * np.clip(Y - 150, 0, length))


def test_short_ramp_stays_vehicular():
    _, res = run([[(250, 50), (250, 300)], [(150, 50), (350, 50)]], ramp_terrain(50), h1=False)
    log = res["log"]["H2"]
    assert log["short_ramps_initial"] == 1 and log["runs_initial"] == 0
    assert (res["kind"] == 0).all() and res["gross_m"] == 0.0


def test_short_ramp_sensitivity_switch():
    lines = [[(250, 50), (250, 300)], [(150, 50), (350, 50)]]
    _, r60 = run(lines, ramp_terrain(70), h1=False)
    _, r40 = run(lines, ramp_terrain(70), h1=False, P=hy.Params(ramp_m=40.0))
    assert r60["log"]["H2"]["runs_initial"] == 0 and r60["log"]["H2"]["short_ramps_initial"] == 1
    assert r40["log"]["H2"]["runs_initial"] == 1 and r40["log"]["H2"]["short_ramps_initial"] == 0


# ---------------------------------------------------------------- the router
FLAT = terrain(lambda X, Y: 0.0 * X)


def street_net(lines, ped=False):
    s = state(lines)
    kinds = {}
    if ped:
        kinds = {hy._key(a, b): hy.PED for a, b in s.live_edges()}
    return s, hy.Net.of(s, kinds)


@pytest.mark.parametrize("offset", [6.0, 14.0])
def test_router_reuses_the_street_and_never_runs_parallel_within_10m(offset):
    s, net = street_net([[(50, 200), (450, 200)]])
    a, b = np.array([100, 200 + offset]), np.array([400, 200 + offset])
    R = hy.route(FLAT, a, b, SIZE, net)
    P = R["poly"]
    on = R["on"] >= 0
    seg = np.hypot(*np.diff(P, axis=0).T)
    if offset < hy.NEAR_M:                                 # inside the band a new parallel road costs x 10
        assert seg[on[:-1] & on[1:]].sum() >= 0.8 * 300    # so the route is the existing street (x 0.2)
    d = hy._seg_dist(P, np.repeat(np.array([[50.0, 200]]), len(P), 0), np.repeat(np.array([[450.0, 200]]), len(P), 0))
    band = (d > hy.ON_M) & (d <= hy.NEAR_M)               # near the street but not on it
    run_len, best = 0.0, 0.0
    for k in range(len(P) - 1):
        run_len = run_len + seg[k] if band[k] and band[k + 1] else 0.0
        best = max(best, run_len)
    assert best <= 12.0                                    # only the way in and out, never alongside
    if offset < hy.NEAR_M:                                 # only the new pieces are committed: in and out
        hb = hy.Hybrid(s, FLAT, grad_of(FLAT), hy.stub_evaluator)
        hb._net = net
        parts = hb.stretches(R)
        assert len(parts) == 2 and all(np.hypot(*np.diff(p, axis=0).T).sum() <= 20 for p, _ in parts)


def test_router_prefers_reuse_over_a_shorter_new_road():
    _, net = street_net([[(100, 300), (100, 250), (400, 250), (400, 300)]])
    R = hy.route(FLAT, np.array([100.0, 300]), np.array([400.0, 300]), SIZE, net)
    on = R["on"] >= 0
    assert on.mean() > 0.9                                 # 400 m on the street (cost 80) beats 300 m of new road


def crossings(P, a, b):
    """(angle deg, point) of proper crossings of polyline P with segment a-b."""
    out = []
    a, b = np.asarray(a, float), np.asarray(b, float)
    for p, q in zip(P[:-1], P[1:]):
        m, t, X, cos = hy._crossings(p[None], q[None], a[None], b[None])
        if m[0]:
            out.append((math.degrees(math.acos(min(cos[0], 1.0))), X[0]))
    return out


@pytest.mark.parametrize("ped", [False, True])
def test_crossings_only_at_nodes_or_at_60_degrees(ped):
    _, net = street_net([[(20, 200), (490, 200)]], ped)    # one long edge: no node near the crossing
    a, b = np.array([100.0, 100]), np.array([400.0, 300])  # the straight line would cross at 34 deg
    R = hy.route(FLAT, a, b, SIZE, net)
    assert R is not None
    P, on = R["poly"], R["on"]
    keep = [k for k in range(len(P) - 1) if not (on[k] >= 0 and on[k + 1] >= 0)]
    for k in keep:                                        # grid steps (steps along the street are reuse)
        for ang, _ in crossings(P[k:k + 2], (20, 200), (490, 200)):
            assert ang >= hy.CROSS_DEG - 1e-6
    hb = hy.Hybrid(state([[(20, 200), (490, 200)]]), FLAT, grad_of(FLAT), hy.stub_evaluator)
    hb._net = net
    n_cross = 0
    for pts, _ in hb.stretches(R):                        # the committed geometry of every new piece
        C, info, _ = hy.shape(hb.geo, pts, net)
        assert C is not None
        for ang, X in crossings(C, (20, 200), (490, 200)):
            if min(np.hypot(*(X - C[0])), np.hypot(*(X - C[-1]))) > 1.0:     # a join is not a crossing
                assert ang >= hy.CROSS_DEG - 1e-6
                n_cross += 1
    if ped:                                               # stairs are not reused: a real crossing
        assert n_cross >= 1


def test_shallow_crossing_allowed_at_a_junction_only():
    _, net = street_net([[(20, 200), (490, 200)], [(250, 200), (250, 120)]])    # a T-junction at (250, 200)
    geo = hy.Geo(FLAT)
    d = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))]) * 60
    at_j = np.array([[250.0, 200.0] - d, [250.0, 200.0] + d]) + [3.0, 0.0]     # 30 deg, 3 m from the junction
    away = at_j + [120.0, 0.0]                                                 # 30 deg, far from any node
    assert geo.crossings_ok(at_j, net) and not geo.crossings_ok(away, net)
    steep = np.array([[370.0, 150.0], [370.0 + 30, 150.0 + 100]])              # 73 deg, far from any node
    assert geo.crossings_ok(steep, net)


def test_reuse_set_excludes_stairs_and_the_replaced_run():
    s = state([STREET] + CONTOURS)
    st = hy.State(s)
    e = s.live_edges()
    run_e = [hy._key(a, b) for a, b in e if np.isclose(s.nodes[a][0], 250) and np.isclose(s.nodes[b][0], 250)]
    st.kinds[run_e[0]] = hy.PED                                                # an edge already typed stairs
    net = hy.Net.of(s, st.kinds, exclude=run_e[1:])                            # the rest: the run being replaced
    ids = {tuple(x) for x in net.ends[~net.reuse]}
    assert ids == set(run_e)
    assert net.reuse.sum() == len(net.ends) - len(run_e)


# ---------------------------------------------------------------- H1, snapshots, output
def test_connectors_and_monotone_gross_km_snapshots():
    lines = [[(100, 100), (400, 100)], [(100, 250), (400, 250)], [(400, 100), (400, 190)]]   # 2 components
    _, res = run(lines, PLANE20)
    l1, l2 = res["log"]["H1"], res["log"]["H2"]
    assert l1.get("connectors_complete", 0) >= 1
    gross = [x["gross_m"] for x in res["snapshots"]]
    assert gross[0] == 0.0 and gross[-1] == pytest.approx(res["gross_m"])
    assert all(a <= b + 1e-9 for a, b in zip(gross[:-1], gross[1:]))
    n_conn = l1.get("connectors_complete", 0) + l1.get("connectors_partial", 0)
    assert len(res["snapshots"]) == 2 + n_conn + l2.get("runs_switchback", 0)
    assert res["gross_m"] == pytest.approx(l1["added_m"] + l2.get("switchback_added_m", 0.0))
    stairs_m = l2.get("runs_stairs_m", 0.0) + l2.get("runs_switchback_m", 0.0)
    assert res["gross_m"] > 0 and (stairs_m > 0 or l2.get("runs_initial", 0) == 0)   # stairs are not subtracted


def test_class_tagged_output():
    _, res = run([STREET] + CONTOURS + [ALT], PLANE20, h1=False)
    g, k = res["graph"], res["kind"]
    assert len(k) == len(g.edges) and set(np.unique(k)) <= {0, 1, 2}
    j = res["snapshots"][-1]["graph"]
    assert len(j["kind"]) == len(j["edges"]) and set(j) >= {"nodes", "edges", "cls", "st", "kind"}
    assert hy.KIND_NAMES == ("vehicular", "stepped_ramp", "stairs")


# ---------------------------------------------------------------- geometry helpers
def test_douglas_peucker_keeps_a_zigzag_that_the_cap_needs():
    T = PLANE20
    geo = hy.Geo(T)
    zz = np.array([[250.0 + (8 if k % 2 else 0), 100 + 4 * k] for k in range(30)])   # 9 m legs, 8.9 % each
    D = hy.douglas_peucker(zz, lambda a, b: geo.chord_ok(a, b, hy.Net.empty()))
    assert len(D) > 2                                     # the 20 % chord straight up is rejected
    flat = hy.douglas_peucker(zz, lambda a, b: True)
    assert len(flat) == 2


def test_chaikin_keeps_end_points():
    P = np.array([[0.0, 0], [50, 0], [50, 50]])
    for k in (1, 2, 3):
        Q = hy.chaikin(P, k)
        assert np.allclose(Q[0], P[0]) and np.allclose(Q[-1], P[-1]) and len(Q) == 2 ** k + 2


# ---------------------------------------------------------------- D4a rulings
def test_d4a_turn_cost_is_a_declared_constant_and_gives_long_legs():
    assert hy.TURN_X == 1.0 and hy.Params().turn_x == hy.TURN_X
    net = hy.Net.empty()
    a, b = np.array([250.0, 150]), np.array([250.0, 300])

    def turns(P_):
        h = np.degrees(np.arctan2(*np.diff(P_, axis=0).T[::-1]))
        return int((np.abs(np.diff(h)) > 1).sum())
    with_cost = turns(hy.route(PLANE20, a, b, SIZE, net)["poly"])
    without = turns(hy.route(PLANE20, a, b, SIZE, net, hy.Params(turn_x=0.0))["poly"])
    assert with_cost <= 8 < 20 <= without                 # long legs, not a raster zig-zag


def two_ramps(gap, length=50):
    return terrain(lambda X, Y: 0.20 * (np.clip(Y - 150, 0, length) + np.clip(Y - (150 + length + gap), 0, length)))


def test_d4a_back_to_back_ramps_closer_than_30m_are_one_run():
    lines = [[(250, 50), (250, 400)], [(150, 50), (350, 50)]]
    close = hy.Analysis(hy.State.start(state(lines)), field(two_ramps(5)), hy.Params())
    apart = hy.Analysis(hy.State.start(state(lines)), field(two_ramps(40)), hy.Params())
    assert len(close.runs) == 1 and close.runs[0].length > hy.RAMP_M and not close.short_ramp(close.runs[0])
    assert len(apart.runs) == 2 and all(apart.short_ramp(r) for r in apart.runs)
    line = np.array([[250.0, 50], [250, 400]])               # the same rule binds new geometry
    assert not hy.Geo(two_ramps(5)).grade_ok(line) and hy.Geo(two_ramps(40)).grade_ok(line)
    _, res = run(lines, two_ramps(5), h1=False)
    assert res["log"]["H2"]["runs_initial"] == 1 and res["log"]["H2"].get("short_ramps_initial", 0) == 0


def hairpin_line(r=15.0, y=200.0, x0=100.0, x1=300.0):
    """Two contour legs joined by a semicircular hairpin climbing the slope (vertices every ~2 m)."""
    t = np.linspace(-math.pi / 2, math.pi / 2, int(math.pi * r / 2))
    return np.vstack([[x0, y], np.c_[x1 + r * np.cos(t), y + r + r * np.sin(t)], [x0, y + 2 * r]])


def test_d4a_hairpin_apex_is_a_short_ramp_only_at_20_percent_and_40m():
    C = hairpin_line()                                    # a G2 hairpin of 15 m radius
    assert H.hairpins(hy._poly_graph(C))
    steep = hy.Geo(terrain(lambda X, Y: 0.25 * Y))           # apex stretch ~26 m at ~21 %: a ramp anywhere else
    gentle = hy.Geo(terrain(lambda X, Y: 0.22 * Y))          # apex stretch ~17 m at ~18 %
    wg, _ = steep.window_grades(C)
    assert hy.APEX_PCT < wg.max() <= hy.RAMP_PCT
    assert not steep.grade_ok(C) and gentle.grade_ok(C)
    ramp = hy.Geo(terrain(lambda X, Y: 0.22 * np.clip(Y - 150, 0, 40)))   # 22 % without a hairpin: allowed
    assert ramp.grade_ok(np.array([[250.0, 50], [250, 300]]))
    for T, short in ((terrain(lambda X, Y: 0.25 * Y), False), (terrain(lambda X, Y: 0.22 * Y), True)):
        an = hy.Analysis(hy.State.start(state([C])), field(T), hy.Params())    # the network's own hairpins
        r = [x for x in an.runs if x.apex]
        assert r and all(an.short_ramp(x) == short for x in r)


def test_d4a_radius_is_binding_one_push_then_smoothing_fail(monkeypatch):
    geo = hy.Geo(FLAT)
    V = np.array([[200.0, 200], [230, 215], [200, 230]])     # a 30 m V hairpin: no Chaikin depth reaches 12 m
    C, info, D = hy.shape(geo, V, hy.Net.empty())
    assert C is None and info["fail"] == "radius" and len(info["apex"]) == 2
    wide = np.array([[200.0, 200], [300, 250], [200, 300]])  # a wide one passes at once
    C, info, _ = hy.shape(geo, wide, hy.Net.empty())
    assert C is not None and hy.radius_ok(C)
    routes, shapes = [], []
    real_route = hy.route

    def spy_route(*a, **k):
        routes.append(1)
        return real_route(*a, **k)

    def always_tight(geo_, pts, net):                     # every candidate geometry has a tight hairpin
        shapes.append(1)
        return None, {"fail": "radius", "apex": pts[len(pts) // 2]}, np.asarray(pts)[[0, len(pts) // 2, -1]]
    monkeypatch.setattr(hy, "route", spy_route)
    monkeypatch.setattr(hy, "shape", always_tight)
    hb = hy.Hybrid(state([[(100, 100), (150, 100)]]), FLAT, grad_of(FLAT), hy.stub_evaluator)
    from collections import Counter
    log = Counter()
    sh, why = hb.plan(hb.st, np.array([200.0, 300]), np.array([350.0, 320]), 1e9, log)
    assert sh is None and why == "smoothing"
    assert len(routes) == 3 and len(shapes) == 2 and log["pushed_reroute"] == 1     # re-routed once, then rejected
    assert log["smoothing_fail"] == 1 and log["smoothing_fail:radius"] == 1 and "tight_radius" not in log


def test_d4a_no_loop_on_a_stick():
    lines = [[(100, 100), (400, 100)], [(100, 250), (400, 250)], [(400, 100), (400, 190)]]   # spur: a dead end
    s = state(lines)
    u = int(np.argmin(np.hypot(*(s.nodes[:s.n] - [400, 190]).T)))
    chain = hy.own_chain(s, u)                            # through the degree-2 corner at (400, 100), to (100, 100)
    assert len(chain) == 16 and hy._key(*chain[-1]) == hy._key(*[int(np.argmin(np.hypot(*(s.nodes[:s.n] - q).T)))
                                                                 for q in ([100, 100], [125, 100])])
    s3 = state([[(100, 100), (400, 100)], [(250, 100), (250, 190)]])            # a spur off a junction
    u3 = int(np.argmin(np.hypot(*(s3.nodes[:s3.n] - [250, 190]).T)))
    assert len(hy.own_chain(s3, u3)) == 4
    net = hy.Net.of(s, {}, hidden=chain)
    hid = np.array([tuple(e) in set(chain) for e in net.ends])
    assert (~net.band[hid]).all() and (~net.reuse[hid]).all() and net.band[~hid].all()
    _, res = run(lines, PLANE20, h2=False)
    l1 = res["log"]["H1"]
    assert l1["connectors_complete"] >= 1 and l1["start_reused"] == 0     # every connector starts at its dead end
    g = res["graph"]
    end = int(np.argmin(np.hypot(*(g.nodes - [400, 190]).T)))
    assert np.bincount(g.edges.ravel(), minlength=g.n)[end] >= 2           # the spur's dead end is connected


def test_d4a_h2_router_pad_and_runtime_logged(monkeypatch):
    seen = []
    real = hy.route

    def spy(T, a, b, size, net, P=hy.Params(), pad=None):
        seen.append((float(np.hypot(*(np.asarray(a) - np.asarray(b)))), pad))
        return real(T, a, b, size, net, P, pad)
    monkeypatch.setattr(hy, "route", spy)
    _, res = run([STREET] + SHORT, PLANE20, h1=False)
    assert seen and all(pad == pytest.approx(max(hy.PAD_M, 2 * d)) for d, pad in seen[:1])
    log = res["log"]["H2"]
    assert log["pad_max_m"] == pytest.approx(300.0) and log["router_s"] > 0
    assert 0 < log["bend_r_min_m"] < float("inf") and "bends_under_12m" in log        # descriptive (D4a e)


def test_d4a_room_failure_reason_is_logged():
    # long contour streets: in the D4a box the cheapest route detours along them beyond 4 x straight, with reuse
    # x 0.2 and, since the x 10 band multiplies the grade cost (cheaper to leave a flat street past its end), x 1 too
    _, res = run([STREET] + CONTOURS, PLANE20, h1=False)
    log = res["log"]["H2"]
    assert log["runs_kept"] == 1 and log["kept_failed_for_room"] == 1 and log["room:detour"] == 1
    assert log["pass2_tried"] == 1 and log.get("pass2", 0) == 0                   # D4c: both passes failed: kept


# ---------------------------------------------------------------- D4c two-pass router
LOOP = [[(200, 300), (200, 100), (300, 100), (300, 300)]]   # a 500 m street between two points 100 m apart


def test_d4c_pass2_only_when_pass1_breaks_the_detour():
    from collections import Counter
    s = state(LOOP)
    hb = hy.Hybrid(s, FLAT, grad_of(FLAT), hy.stub_evaluator)
    a, b = np.array([200.0, 300]), np.array([300.0, 300])
    hb._net = hy.Net.of(hb.st.s, hb.st.kinds)
    log = Counter()
    R, pas, P_ = hb.two_pass(a, b, log, 4 * 100.0, pad=200.0)      # x 0.2: the 500 m street (detour 5) is cheapest
    assert pas == 2 and P_.reuse_x == 1.0 and R["length"] <= 400 and log["pass2_tried"] == 1
    assert (R["on"] < 0).mean() > 0.8                               # pass 2: a new road, not the street
    log = Counter()
    R, pas, P_ = hb.two_pass(a, b, log, 6 * 100.0, pad=200.0)      # within the detour: pass 1 stands (the street)
    assert pas == 1 and P_.reuse_x == hy.REUSE_X and log["pass2_tried"] == 0 and (R["on"] >= 0).mean() > 0.8
    p1 = hy.Params()                                                # pass 2 changes the reuse factor only
    assert {k: v for k, v in vars(P_).items() if k != "reuse_x"} == {k: v for k, v in vars(p1).items()
                                                                       if k != "reuse_x"}


def test_d4c_both_passes_fail_means_kept(monkeypatch):
    real = hy.route
    monkeypatch.setattr(hy, "route", lambda *a, **k: (lambda R: R and {**R, "length": R["length"] * 10})(real(*a, **k)))
    _, res = run([STREET] + SHORT, PLANE20, h1=False)              # every route "breaks" the detour
    log = res["log"]["H2"]
    assert log["pass2_tried"] == 1 and log["runs_kept"] == 1 and log["room:detour"] == 1 and res["gross_m"] == 0


def test_d4c_pass2_keeps_the_no_loop_on_a_stick_rules(monkeypatch):
    real, seen = hy.route, []

    def pass1_too_long(T, a, b, size, net, P=hy.Params(), pad=None):
        R = real(T, a, b, size, net, P, pad)
        seen.append((P.reuse_x, net))
        return R if R is None or P.reuse_x == 1.0 else {**R, "length": R["length"] * 10}
    monkeypatch.setattr(hy, "route", pass1_too_long)
    lines = [[(100, 100), (400, 100)], [(100, 250), (400, 250)], [(400, 100), (400, 190)]]
    _, res = run(lines, PLANE20, h2=False)
    l1 = res["log"]["H1"]
    assert l1["pass2"] >= 1 and l1.get("pass1", 0) == 0 and l1["pass2_share"] == 1.0
    assert l1["start_reused"] == 0                                  # pass 2 starts at the dead end too
    p2 = [net for rx, net in seen if rx == 1.0]
    assert p2 and all((~net.band).any() and not (net.reuse & ~net.band).any() for net in p2)   # its chain hidden
    assert 0.0 <= l1["pass2_lens_share"] <= 1.0 and l1["pass2_m"] > 0


def test_d4c_lens_share_of_pass2_pieces():
    lines = [[(50, 200), (100, 200), (300, 200), (350, 200)], [(100, 200), (200, 215), (300, 200)]]   # a lens
    hb = hy.Hybrid(state(lines), FLAT, grad_of(FLAT), hy.stub_evaluator)
    arc = [(np.array([100.0, 200]), np.array([200.0, 215])), (np.array([200.0, 215]), np.array([300.0, 200]))]
    hb.log["H2"]["pass2"] = 1
    hb.pass2_segs["H2"] = arc
    hb.pass2_segs["H1"] = [(np.array([300.0, 200]), np.array([350.0, 200]))]       # a street off the lens
    hb.log["H1"]["pass2"] = 1
    hb.pass2_lens()
    assert hb.log["H2"]["pass2_lens_share"] == pytest.approx(1.0) and hb.log["H2"]["pass2_share"] == 1.0
    assert hb.log["H1"]["pass2_lens_share"] == 0.0 and hb.log["H1"]["pass2_m"] == pytest.approx(50.0)
