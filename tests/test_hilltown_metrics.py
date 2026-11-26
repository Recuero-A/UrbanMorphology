"""Milestone 20 D3 hill-town metric suite (metrics/hilltown.py) on small synthetic terrains with known answers.

The adversarial tests at the end use a SMALL SYNTHETIC "real" set: they are unit tests of the gate helpers, not the
D3 validation gate (which runs on the dev-H real towns)."""
import math

import numpy as np
import pytest
import shapely

from urbanmorph.graphs.road_graph import RoadGraph
from urbanmorph.metrics import hilltown as H
from urbanmorph.terrain.heightfield import Terrain


# ---------------------------------------------------------------- fixtures
def terrain(zfun, size=(600.0, 600.0), res=4.0, margin=64.0) -> Terrain:
    W, Hh = int((size[0] + 2 * margin) / res), int((size[1] + 2 * margin) / res)
    X, Y = np.meshgrid(-margin + (np.arange(W) + 0.5) * res, size[1] + margin - (np.arange(Hh) + 0.5) * res)
    return Terrain(zfun(X, Y).astype(np.float32), -margin, size[1] + margin, res)


def field(T, size=(600.0, 600.0), uid="t") -> H.HillField:
    return H.HillField.from_polygon(T, shapely.box(0, 0, *size), uid=uid)


def plane(slope, size=(600.0, 600.0), res=4.0):
    T = terrain(lambda X, Y: slope * Y, size, res)
    return field(T, size)


def arc(c, r, a0, a1, n=24):
    t = np.linspace(a0, a1, n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def switchback(r, y=200.0, x0=100.0, x1=400.0):
    """Two contour legs joined by one semicircular hairpin of centreline radius r (vertices every ~2 m)."""
    n = max(8, int(math.pi * r / 2))
    return np.vstack([[x0, y], arc((x1, y + r), r, -math.pi / 2, math.pi / 2, n), [x0, y + 2 * r]])


# ---------------------------------------------------------------- M2
def test_fall_line_street_contour_index_one_and_all_steep():
    F = plane(0.22)              # 22 %: inside the 20-30 class (20 % itself is the class boundary)
    g = H.lines_graph([np.array([[300.0, 50.0], [300.0, 550.0]])])
    m = H.unit_metrics(g, F, access=False)
    assert m["contour_index@20-30"] == pytest.approx(1.0, abs=1e-3)
    assert m["share_gt15@20-30"] == 1.0
    assert m["grade_p90@20-30"] == pytest.approx(22.0, abs=0.1)
    assert m["street_km@20-30"] == pytest.approx(0.5, abs=1e-6)


def test_contour_street_contour_index_zero():
    F = plane(0.22)
    g = H.lines_graph([np.array([[50.0, 300.0], [550.0, 300.0]])])
    m = H.unit_metrics(g, F, access=False)
    assert m["contour_index@20-30"] < 1e-3
    assert m["share_gt15@20-30"] == 0.0
    assert m["grade_p90@20-30"] < 0.1


def test_density_and_coverage():
    F = plane(0.0)
    g = H.lines_graph([np.array([[0.0, 300.0], [600.0, 300.0]])])
    m = H.unit_metrics(g, F, access=False)
    assert m["density@0-5"] == pytest.approx(0.6 / 0.36, rel=1e-3)
    assert m["coverage@0-5"] == pytest.approx(120 / 600, abs=0.02)        # a 120 m band of the 600 m square


# ---------------------------------------------------------------- M3
def test_hairpin_counted_once_native_and_convention():
    F = plane(0.25)
    g = H.lines_graph([switchback(10.0)], node=False)
    hp = H.hairpins(g)
    assert len(hp) == 1
    assert 120.0 < hp[0][1] < 135.0                                        # the minimal window past 120 deg
    assert hp[0][2] == pytest.approx(10.0, rel=0.1)                        # centreline radius
    m = H.unit_metrics(g, F, access=False)
    assert m["hairpins_native@gt15"] == 1 and m["hairpins@gt15"] == 1
    assert m["hairpin_rate@gt15"] == pytest.approx(1 / m["street_km@gt15"])
    assert m["g2_tight_share@gt15"] == 1.0                                 # 10 m < 12 m
    wide = H.unit_metrics(H.lines_graph([switchback(20.0)], node=False), F, access=False)
    assert wide["hairpins_native@gt15"] == 1 and wide["g2_tight_share@gt15"] == 0.0


def test_s_bend_two_hairpins_and_zigzag_none():
    s = np.vstack([switchback(10.0), arc((100.0, 230.0), 10.0, -math.pi / 2, -3 * math.pi / 2, 16)[1:],
                   [[400.0, 240.0]]])
    assert len(H.hairpins(H.lines_graph([s], node=False))) == 2
    zig = np.array([[x, 5.0 * (k % 2)] for k, x in enumerate(np.arange(0, 200, 5.0))])   # alternating turns
    assert len(H.hairpins(H.lines_graph([zig], node=False))) == 0


def test_stroke_runs_up_the_slope():
    F = plane(0.25)
    # a straight climb of 300 m crossed by a contour street: one stroke through the junction, one run
    g = H.lines_graph([np.array([[300.0, 100.0], [300.0, 400.0]]), np.array([[100.0, 250.0], [500.0, 250.0]])])
    m = H.unit_metrics(g, F, access=False)
    assert m["run_p90@20-30"] == pytest.approx(300.0, rel=0.02)
    flat = H.unit_metrics(H.lines_graph([np.array([[100.0, 250.0], [500.0, 250.0]])]), F, access=False)
    assert flat["run_p90@20-30"] == 0.0                                    # streets, no run: defined as 0


# ---------------------------------------------------------------- M4, M5, M7
def lens_lines(offset=10.0):
    A, B = (200.0, 300.0), (320.0, 300.0)
    return [np.array([[100.0, 300.0], A, B, [420.0, 300.0]]),
            np.array([A, [A[0] + offset, 300 + offset], [B[0] - offset, 300 + offset], B])]


def test_lens_face_detected():
    F = plane(0.12)
    m = H.unit_metrics(H.lines_graph(lens_lines()), F, access=False)
    lens_len = 120 + 2 * math.hypot(10, 10) + 100
    assert m["lens_share@10-20"] == pytest.approx(lens_len / (320 + lens_len - 120), rel=0.05)
    assert m["lens_share_2ap@10-20"] > 0
    # a square block bounded by four junctions is not a lens
    grid = [np.array([[100.0, y], [500.0, y]]) for y in (200.0, 300.0)] + \
           [np.array([[x, 100.0], [x, 400.0]]) for x in (200.0, 300.0)]
    m2 = H.unit_metrics(H.lines_graph(grid), F, access=False)
    assert m2["lens_share@10-20"] == 0.0
    assert m2["phi_med@10-20"] == pytest.approx(2 / math.pi, abs=0.02)
    assert m2["face_area_med@10-20"] == pytest.approx(1e4, rel=0.02)
    assert m2["block_contour_cos@10-20"] in (0.0, 1.0) or 0 <= m2["block_contour_cos@10-20"] <= 1


def test_topology_parallel_streets_and_loops():
    F = plane(0.12)
    par = [np.array([[50.0, y], [550.0, y]]) for y in (100.0, 200.0, 300.0, 400.0)]
    m = H.unit_metrics(H.lines_graph(par), F, access=False)
    assert m["lcc_share@10-20"] == pytest.approx(0.25, abs=0.01)
    assert m["loops_per_km2@10-20"] == 0.0
    grid = par + [np.array([[x, 100.0], [x, 400.0]]) for x in (150.0, 300.0, 450.0)]
    m2 = H.unit_metrics(H.lines_graph(grid), F, access=False)
    assert m2["lcc_share@10-20"] == pytest.approx(1.0)
    assert m2["loops_per_km2@10-20"] == pytest.approx(6 / m2["land_km2@10-20"])   # 3 x 2 blocks
    assert m2["share_4way@10-20"] > 0 and m2["deadend_per_km@10-20"] >= 0


def test_tortuosity_of_raster_zigzag():
    F = plane(0.12)
    g = H.lines_graph([np.array([[50.0, 50.0], [550.0, 550.0]])])
    m = H.unit_metrics(g, F, access=False)
    assert m["tortuosity@10-20"] == pytest.approx(1.0, abs=1e-6)
    z = H.unit_metrics(H.adv_raster_zigzag(g), F, access=False)
    assert z["tortuosity@10-20"] == pytest.approx(math.sqrt(2), rel=0.06)   # a 45 deg staircase: |dx| + |dy|


# ---------------------------------------------------------------- M6, G1, pairs
def test_lexicographic_route_prefers_gentle_detour():
    nodes = np.array([[0.0, 0.0], [300.0, 0.0], [150.0, 200.0]])
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    steep = np.array([100.0, 0.0, 0.0])                     # the shortcut has 100 m above the threshold
    P, I, J = np.array([[-5.0, 0.0], [305.0, 0.0]]), np.array([0]), np.array([1])
    o, rl, rs = H.route_pairs(nodes, edges, P, I, J, steep=steep, lex=True)
    assert o[0] == 0 and rs[0] == 0.0 and rl[0] == pytest.approx(500.0)
    o, rl, _ = H.route_pairs(nodes, edges, P, I, J, steep=steep)          # plain: the shortcut
    assert o[0] == 0 and rl[0] == pytest.approx(300.0)
    o, _, rs = H.route_pairs(nodes[:2], edges[:1], P, I, J, steep=steep[:1], lex=True)
    assert o[0] == 4 and rs[0] == pytest.approx(100.0)                     # only the steep route: share 1/3 > 10 %
    far = np.array([[0.0, 0.0], [300.0, 0.0], [150.0, 700.0]])             # the gentle detour is longer than 4x
    o, _, rs = H.route_pairs(far, edges, P, I, J, steep=steep, lex=True)
    assert o[0] == 3 and rs[0] == 0.0


def test_route_pairs_matches_m19_served():
    ev = H._script("m19_eval")
    rng = np.random.default_rng(3)
    lines = [np.array([[0.0, y], [800.0, y + rng.normal(0, 20)]]) for y in np.arange(50, 800, 110)]
    lines += [np.array([[x, 0.0], [x + rng.normal(0, 20), 800.0]]) for x in np.arange(80, 800, 170)]
    g = H.lines_graph(lines)
    keep = rng.random(len(g.edges)) > 0.15                                   # holes: disconnected pieces
    g = RoadGraph(g.nodes, g.edges[keep], g.border, edge_struct=g.edge_struct[keep])
    P = rng.uniform(-50, 850, (300, 2))
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    I, J = np.nonzero(np.triu((D >= 150) & (D <= 500), 1))
    ref, _ = ev.served(g, P, I, J, closed=False)
    mine, _, _ = H.route_pairs(g.nodes, g.edges, P, I, J, chunk=7)
    assert len(I) > 1000 and np.array_equal(ref, mine)
    assert len(np.unique(ref)) >= 3


@pytest.mark.parametrize("n_px", [256, 200])
def test_od_pairs_match_m19_od_points(n_px):
    ev = H._script("m19_eval")
    size = n_px * 4.0 - 128.0
    T = terrain(lambda X, Y: 0.0004 * (X - 200) ** 2 + 0.12 * Y, (size, size), 4.0, 64.0)
    F = field(T, (size, size), uid=f"tile_{n_px}")
    assert F.land.shape[0] == n_px
    P0, I0, J0 = ev.od_points(F.uid, F)
    P1, I1, J1 = H.od_pairs(F)
    assert len(I0) > 50 and np.array_equal(P0, P1) and np.array_equal(I0, I1) and np.array_equal(J0, J1)


def test_window_grade_drivable_subgraph():
    F = plane(0.20)
    g = H.convention(H.lines_graph([np.array([[300.0, 100.0], [300.0, 500.0]]),
                                    np.array([[100.0, 300.0], [500.0, 300.0]])]))
    pc = H.pieces(g, F)
    wg = H.window_grades(g, F, pc)
    assert np.nanmax(wg) == pytest.approx(20.0, abs=0.2) and np.nanmin(wg) < 0.1
    nodes, edges = H.split_graph(g, pc, ~(wg > 15.0))
    L = np.hypot(*(nodes[edges[:, 0]] - nodes[edges[:, 1]]).T).sum()
    assert L == pytest.approx(400.0, abs=1.0)                                 # only the contour street is drivable


def test_pedestrian_graph_kinds():
    g = H.lines_graph([np.array([[0.0, 0.0], [200.0, 0.0]])])
    pg, kind = H.pedestrian_graph(g, ([np.array([[100.0, 0.0], [100.0, 150.0]])], [True]))
    assert set(kind.tolist()) == {0, 2}


# ---------------------------------------------------------------- scoring
def rows_of(values, metric="lens_share", cell="10-20", km=5.0):
    return {f"t{i}": {f"{metric}@{cell}": v, f"land_km2@{cell}": 1.0, f"street_km@{cell}": km}
            for i, v in enumerate(values)}


def test_robust_z_zero_inflated():
    vals = [0, 0, 0, 0, 0, 0.01, 0.02, 0, 0.03, 0, 0, 0.015]
    rows = rows_of(vals)
    k = "lens_share@10-20"
    ref = H.fit_reference(rows.values(), [k])
    c, med, scale, n = ref[k]
    assert c == pytest.approx(0.1 * np.median([0.01, 0.02, 0.03, 0.015]))
    t = np.log(np.array(vals) + c)
    q10, q90 = np.quantile(t, [0.1, 0.9])
    assert med == pytest.approx(np.median(t))
    assert scale == pytest.approx(max(1.4826 * np.median(np.abs(t - med)), (q90 - q10) / 2.563, 0.05))
    real = rows["t0"]
    z0 = H.score({**real, k: 0.0}, real, ref, [k])[k]
    assert np.isfinite(z0.z) and z0.absz < 5
    zhi = H.score({**real, k: 0.6}, real, ref, [k])[k]
    assert zhi.z == 5.0 and zhi.absz == 5.0                                # capped, sign kept
    # an identity metric with no spread falls back on the declared floor
    rows2 = rows_of([0.9] * 6, "lcc_share")
    r2 = H.fit_reference(rows2.values(), ["lcc_share@10-20"])
    assert r2["lcc_share@10-20"][2] == H.METRICS["lcc_share"].floor
    assert H.score({**rows2["t0"], "lcc_share@10-20": 0.2}, rows2["t0"], r2, ["lcc_share@10-20"])["lcc_share@10-20"].z == -5


def test_cell_rules():
    k = "phi_med@10-20"
    rows = rows_of([0.5, 0.55, 0.6, 0.52, 0.58], "phi_med")
    ref = H.fit_reference(rows.values(), [k])
    real = rows["t0"]
    s = H.score({**real, "street_km@10-20": 0.4}, real, ref, [k])[k]
    assert s.absz == 5 and s.flag == "gen_lt_1km"                          # deleted streets are never skipped
    s = H.score({**real, k: np.nan}, real, ref, [k])[k]
    assert s.absz == 5 and s.flag == "gen_undefined"
    assert H.score(real, {**real, "street_km@10-20": 0.9}, ref, [k]) == {}  # real < 1 km: not defined
    assert H.score(real, {**real, "land_km2@10-20": 0.04}, ref, [k]) == {}  # < 0.05 km2 of land: no cell
    d = "density@10-20"
    rd = {f"t{i}": {d: v, "land_km2@10-20": 1.0, "street_km@10-20": v} for i, v in enumerate([8, 9, 10, 11, 12])}
    refd = H.fit_reference(rd.values(), [d])
    sd = H.score({d: 0.0, "land_km2@10-20": 1.0, "street_km@10-20": 0.0}, rd["t0"], refd, [d])[d]
    assert sd.flag == "" and sd.z == -5                                    # M1 is scored, not ruled


def test_declared_cell_lists():
    assert not any(c.split("@")[0] in ("run_p50", "run_p90", "hairpin_rate") for c in H.P2_CELLS)  # D4: M3 targeted
    assert {c.split("@")[0] for c in H.P2_CELLS} == {"lens_share", "deadend_per_km", "lcc_share", "phi_med",
                                                     "block_contour_cos", "tortuosity"}
    assert "block_contour_cos@5-10" not in H.P2_CELLS and "block_contour_cos@30-inf" in H.P2_CELLS
    assert {"m6_15@unit", "m6_12@unit", "m6_10@unit"} <= set(H.GROUPS["M6"] + H.GROUPS["M6_sec"])
    assert all(g_ in H.GROUPS for t in H.ADVERSARIAL_TARGETS.values() for g_ in t)


def test_gate_real_on_exchangeable_towns():
    rng = np.random.default_rng(0)
    keys = ("lcc_share@10-20", "phi_med@10-20")
    rows = {f"t{i}": {"lcc_share@10-20": rng.normal(0.9, 0.03), "phi_med@10-20": rng.normal(0.5, 0.05),
                      "land_km2@10-20": 1.0, "street_km@10-20": 5.0} for i in range(25)}
    gate = H.gate_real(rows, keys)
    assert gate["coverage"] >= gate["threshold"] and gate["ks_p"] > 0.05 and gate["pass"]
    assert all(0 < p < 1 for p in gate["percentiles"].values())


def test_c2st_null_and_separable():
    rng = np.random.default_rng(1)
    town = np.repeat(np.arange(10), 20)
    y = np.tile(np.r_[np.zeros(10), np.ones(10)], 10).astype(int)
    X = rng.normal(size=(200, 3))
    null = H.c2st(X, y, town, n_boot=50, n_perm=20, seed=0)
    assert 0.3 < null["auc"] < 0.7 and null["p"] > 0.05
    sep = H.c2st(X + 2.0 * y[:, None], y, town, n_boot=50, n_perm=20, seed=0)
    assert sep["auc"] > 0.9 and sep["p"] <= 1 / 21 + 1e-12 and sep["ci"][0] > 0.8
    from scipy.optimize import check_grad
    from scipy.special import expit
    w, b = H.logistic_fit(X, y)
    grad = w + X.T @ (expit(X @ w + b) - y)                                 # optimality of the L2 objective
    assert np.abs(grad).max() < 1e-3 and check_grad is not None


def test_c2st_cells_pools_steep_classes():
    rng = np.random.default_rng(2)
    cells = []
    for t in range(6):
        for lab in (0, 1):
            for k in H.KEYS:
                cells.append({"town": f"t{t}", "label": lab, "cls": k, "land_km2": 0.1, "street_km": 1.0,
                              **{f: rng.normal() for f in H.C2ST_FEATURES}})
    out = H.c2st_cells(cells, n_boot=10, n_perm=5)
    assert out["pooled_steep"] and "20-inf|all" in out and "20-inf|no_M1" in out
    assert out["20-inf|all"]["cells"] == 24


# ---------------------------------------------------------------- adversarial networks (unit test, synthetic "real")
SX, SY = 1000.0, 1050.0
BANDS = ((0.0, 0.03), (350.0, 0.14), (700.0, 0.24))     # (start y, slope): classes 0-5, 10-20, 20-30


def z_hill(X, Y):
    z = np.zeros_like(Y, dtype=float)
    for (y0, s), (y1, _) in zip(BANDS, BANDS[1:] + ((1e9, 0),)):
        z += s * np.clip(Y - y0, 0, y1 - y0)
    return z


@pytest.fixture(scope="module")
def hill():
    return field(terrain(z_hill, (SX, SY), 8.0, 64.0), (SX, SY), uid="hill")


def slope_at(y):
    return [s for y0, s in BANDS if y >= y0][-1]


def real_town(seed):
    """Flat grid below, contour streets with gentle (10 %) diagonal connectors above, some hairpins and one lens."""
    rng = np.random.default_rng(seed)
    sp = rng.uniform(70, 90)
    ys = list(np.arange(370.0, SY - 15, rng.uniform(45, 60)))
    lines = [np.array([[10.0, y], [990.0, y]]) for y in np.arange(20.0, 350, sp)]
    xs = np.arange(25.0, 990, sp)
    lines += [np.array([[x, 10.0], [x, ys[0]]]) for x in xs]
    lines += [np.array([[10.0, y], [990.0, y]]) for y in ys]
    att = {y: [] for y in ys}
    for k, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        dx = (y1 - y0) * math.sqrt((slope_at(y0) / 0.10) ** 2 - 1)
        for i, x0 in enumerate(np.arange(rng.uniform(30, 120), 820 - dx, rng.uniform(220, 280))):
            a, b = (x0, y0), (x0 + dx, y1)
            if (i + k) % 2:
                a, b = (x0 + dx, y0), (x0, y1)
            lines.append(np.array([a, b]))
            att[y0].append(a[0])
            att[y1].append(b[0])
    # hairpin connectors on the 24 % land (0-2 per town)
    steep = [(y0, y1) for y0, y1 in zip(ys[:-1], ys[1:]) if y0 > 760]
    for y0, y1 in steep[:seed % 3]:
        x0 = 940.0
        r = (y1 - y0) / 4
        lines.append(np.vstack([[x0, y0], [x0 - 60, y0 + r], arc((x0 - 60, y0 + 2 * r), r, -math.pi / 2,
                                                                -3 * math.pi / 2, 14)[1:-1], [x0 - 60, y0 + 3 * r],
                                [x0, y1]]))
        att[y0].append(x0)
        att[y1].append(x0)
    # one lens (a 10 m divided stretch) on a street free of junctions, in band seed % 3
    yb = [y for y in ([*np.arange(20.0, 350, sp)] if seed % 3 == 0 else ys)
          if (seed % 3 == 0) or (y < 700) == (seed % 3 == 1)]
    y = yb[len(yb) // 2]
    taken = sorted(att.get(y, []) + (list(xs) if seed % 3 == 0 else []) + [10.0, 990.0])
    gaps = [(b - a, a) for a, b in zip(taken[:-1], taken[1:])]
    w, a = max(gaps)
    xa, xb = a + (w - 70) / 2, a + (w + 70) / 2
    lines.append(np.array([[xa, y], [xa + 10, y + 10], [xb - 10, y + 10], [xb, y]]))
    return H.lines_graph(lines)


def bundle(g):
    """leastcostpath stand-in for the unit test: every connector doubled 10 m to one side (a braid of lens faces)."""
    out = []
    for poly, _, _ in H.chains(g):
        out.append(poly)
        a, b = poly[0], poly[-1]
        d = b - a
        L = np.hypot(*d)
        if len(poly) == 2 and abs(d[0]) > 1 and abs(d[1]) > 1 and L > 60:      # a straight diagonal connector
            u = d / L
            n = np.array([-u[1], u[0]])
            n = n if n[1] > 0 else -n
            out.append(np.array([a, a + 15 * u + 10 * n, b - 15 * u + 10 * n, b]))
    return H.lines_graph(out)


@pytest.fixture(scope="module")
def real_set(hill):
    graphs = {f"t{s}": real_town(s) for s in range(9)}
    return graphs, {t: H.unit_metrics(g, hill, access=False) for t, g in graphs.items()}


def test_synthetic_real_towns_are_sane(real_set):
    _, rows = real_set
    for m in rows.values():
        assert m["lcc_share@20-30"] > 0.99 and m["tortuosity@20-30"] < 1.05
        assert m["share_gt15@10-20"] < 0.05 and m["loops_per_km2@20-30"] > 0
    assert sum(m["hairpins_native@gt15"] > 0 for m in rows.values()) == 6     # seeds with seed % 3 > 0
    assert sum(m["hairpins@gt15"] for m in rows.values()) < sum(m["hairpins_native@gt15"] for m in rows.values())
    assert sum(m["lens_share@20-30"] > 0 for m in rows.values()) >= 2


def test_adversarial_networks_fail_their_target(hill, real_set):
    """UNIT TEST on a synthetic real set (not the D3 validation gate)."""
    graphs, rows = real_set
    c = H.offsets(rows.values(), H.SUITE_CELLS)
    L = H.loo(rows, H.SUITE_CELLS, c)
    unit = "t0"
    adv = {"switchback": H.adv_switchback(hill), "contour_parallel": H.adv_contour_parallel(hill),
           "fall_line_grid": H.adv_fall_line_grid(hill), "raster_zigzag": H.adv_raster_zigzag(graphs[unit]),
           "leastcostpath": bundle(graphs[unit])}
    sw = adv["switchback"]
    assert len(H.lines_graph([]).edges) == 0 and (np.bincount(sw.edges.ravel()) <= 2).all()   # one road
    assert (np.bincount(adv["contour_parallel"].edges.ravel()) <= 2).all()                  # never joined
    for name, g in adv.items():
        m = H.unit_metrics(g, hill, access=False)
        res = H.gate_adversarial(m, rows[unit], rows, H.ADVERSARIAL_TARGETS[name], town=unit, c=c, loo_out=L)
        assert res["targeted_fails"], (name, res["groups"])
        assert res["composite_fails"], (name, res["composite"], res["null_p95"])
