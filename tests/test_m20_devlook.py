"""Milestone 20 D4 development-look runner (scripts/m20_devlook.py): the open / sealed split leaks no P2 key, the
stages cache and resume, and the arms are paired on one A0 state. Synthetic graphs and terrains only (the stub
sampler of tests/test_grade_hybrid.py); no town data is read."""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import m20_devlook as dl  # noqa: E402
import m20_hybrid as hy  # noqa: E402
from test_grade_hybrid import PLANE20, SHORT, STREET, grad_of, state  # noqa: E402
from urbanmorph.experiments.graph_io import graph_to_json  # noqa: E402
from urbanmorph.graphs.road_graph import RoadGraph  # noqa: E402
from urbanmorph.metrics import hilltown as H  # noqa: E402
from urbanmorph.terrain.heightfield import Terrain  # noqa: E402

P2_KEYS = set(H.P2_CELLS) | set(H.P2_CELLS_TOPO)
P2_NAMES = {k.split("@")[0] for k in P2_KEYS}
LINES = [[(100, 100), (400, 100)], [(100, 250), (400, 250)], [(400, 100), (400, 190)]]   # H1 and H2 both act


# ---------------------------------------------------------------- fixtures
def plane_field(slope=0.12, size=600.0, res=4.0, uid="t"):
    n = int(size / res)
    X, Y = np.meshgrid((np.arange(n) + 0.5) * res, size - (np.arange(n) + 0.5) * res)
    S = Terrain((slope * Y + 0.0002 * X * X).astype(np.float32), 0.0, size, res)
    return H.HillField(S, np.ones((n, n), bool), uid=uid)


def grid_graph(size=600.0, step=60.0):
    xs = np.arange(60.0, size - 59.0, step)
    nodes = np.array([(x, y) for y in xs for x in xs])
    k = len(xs)
    E = [(i * k + j, i * k + j + 1) for i in range(k) for j in range(k - 1)]
    E += [(i * k + j, (i + 1) * k + j) for i in range(k - 1) for j in range(k)]
    return RoadGraph(nodes, np.array(E), np.zeros(len(nodes), bool))


@pytest.fixture(scope="module")
def metrics():
    """A full unit_metrics dict (every key of the suite) on a synthetic town."""
    F = plane_field()
    ped = ([np.array([[90.0, 90.0], [150.0, 150.0]])], [True])
    return H.unit_metrics(grid_graph(), F, ped=ped, access=True)


def a0_stub(seed):
    return {"graph_sha256": f"sha{seed}", "seconds": 1.0, "run_s": 1.0, "forward_passes": 10, "km_domain": 2.0,
            "km_limits": 1.0, "round_cap_hit": False, "node_budget_hit": False, "nodes": 5, "edges": 4, "L": 2048.0,
            "chosen_particle": 0, "maxrss_mb": 10.0}


def hb_stub(seed):
    lg = {"H1": {"connectors_complete": 2, "pass1": 1, "pass2": 1, "pass2_share": 0.5, "pass2_m": 40.0,
                 "pass2_lens_m": 12.0, "pass2_lens_share": 0.3, "added_m": 80.0},
          "H2": {"runs_initial": 3, "runs_stairs": 1, "runs_kept": 1, "kept_m": 70.0, "pass2_lens_share": 0.1,
                 "drivable_claim": 0}, "seconds": {"H1": 1.0, "H2": 2.0}}
    left = {"steep_runs_domain": 1, "steep_km_domain": 0.07, "steep_runs_unit": 1, "steep_km_unit": 0.07,
            "forced_left_unit": 0, "drivable_unit": 0}
    arm = lambda: {"log": json.loads(json.dumps(lg)), "km": {k: 1.0 for k in hy.KIND_NAMES}, "gross_m": 80.0,
                   "seconds": 3.0, "steep_left": left}
    return {"a0_sha256": f"sha{seed}", "a0_steep_left": left, "arms": {"H1": arm(), "H12": arm()}}


def touch(f: Path, b: bytes = b"x"):
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b)


# ---------------------------------------------------------------- P2 sealed
def test_open_split_holds_no_p2_key(metrics):
    op, se = dl.open_split(metrics)
    assert not P2_KEYS & set(op)
    assert not any(k.split("@")[0] in P2_NAMES for k in op)
    assert P2_KEYS <= set(se)                                    # every P2 cell is computed and sealed
    opened = set(H.GROUPS["M1"] + H.GROUPS["M2"] + H.GROUPS["M6"] + H.GROUPS["M6_sec"] + H.GROUPS["G1"]
                 + H.GROUPS["pedestrian"] + H.MESH_CELLS)
    assert opened <= set(op)                                     # the declared open list is open
    assert not set(H.GROUPS["M3"] + H.GROUPS["M5"] + H.GROUPS["loops"]) & set(op)   # anything else is sealed
    logs = {"H1": {"pass2_share": 0.5, "pass2_lens_share": 0.2, "pass2_lens_m": 3.0}}
    assert dl.scrub(logs) == {"H1": {"pass2_share": 0.5}}
    assert dl.sealed_part(logs) == {"H1": {"pass2_lens_share": 0.2, "pass2_lens_m": 3.0}}


def test_report_output_contains_no_p2_key(metrics, tmp_path, capsys):
    lay = dl.Layout("scratch", "none", tmp_path)
    dl.store_real(lay, "T", metrics, 1.0)
    seeds = (1, 2, 3)
    for s in seeds:
        jit = {k: (v + 0.01 * s if isinstance(v, float) and np.isfinite(v) else v) for k, v in metrics.items()}
        meas = {arm: (jit, {k: 1.0 for k in hy.KIND_NAMES}, 1.0) for arm in dl.ARMS}
        op, se = dl.assemble("T", s, "none", a0_stub(s), hb_stub(s), meas, metrics)
        dl.put_json(lay.open_("T", s), op)
        dl.put_jgz(lay.sealed("T", s), se)
    rep = dl.stage_report(lay, ["T"], seeds, 32)
    printed = capsys.readouterr().out
    opened = [printed, json.dumps(rep), lay.report().read_text(), lay.manifest().read_text(),
              lay.real_open("T").read_text()] + [lay.open_("T", s).read_text() for s in seeds]
    for text in opened:
        for name in P2_NAMES | {"lens"}:
            assert name not in text, name
    assert "m6_15@unit" in printed and rep["towns"]["T"]["m6_15_pilot"]["n"] == 3
    bundle = dl.get_jgz(lay.bundle())                            # ... and the P2 cells are in the sealed bundle
    run = bundle["sealed/T/s01.json.gz"]
    assert P2_KEYS <= set(run["arms"]["H12"]) and run["logs"]["H12"]["H1"]["pass2_lens_share"] == 0.3
    man = json.loads(lay.manifest().read_text())
    assert man["bundle_sha256"] == dl.sha_bytes(lay.bundle().read_bytes()) and man["records"] == 4


def test_m6_pilot_variance_is_paired():
    rep = {"A0": [0.30, 0.20, 0.40], "H12": [0.35, 0.26, 0.44]}
    d = [b - a for a, b in zip(rep["A0"], rep["H12"])]
    assert dl.var1(d) == pytest.approx(np.var(d, ddof=1))
    assert dl.var1(rep["A0"]) == pytest.approx(0.01)
    assert math.isnan(dl.var1([0.3]))


# ---------------------------------------------------------------- caching and resume
def test_stages_skip_cached_runs_and_resume(tmp_path, monkeypatch):
    lay = dl.Layout("scratch", "none", tmp_path)
    calls = []

    def fake_a0(spec):
        calls.append(("a0", spec[2]))
        dl.put_jgz(lay.a0(spec[1], spec[2])[0], {"seed": spec[2]})
        return {"seconds": 1.0}

    def fake_hybrid(spec):
        calls.append(("hybrid", spec[2]))
        if spec[2] == 3 and not any(c == ("crash", 3) for c in calls):
            calls.append(("crash", 3))
            touch(dl._tmp(lay.hybrid(spec[1], spec[2])[0]))      # an interrupted write leaves only a .part
            raise KeyboardInterrupt
        dl.put_jgz(lay.hybrid(spec[1], spec[2])[0], {"seed": spec[2]})
        return {"seconds": 1.0}
    monkeypatch.setattr(dl, "a0_job", fake_a0)
    monkeypatch.setattr(dl, "hybrid_job", fake_hybrid)
    touch(lay.a0("T", 1)[0])                                     # seed 1 already generated
    dl.stage_generate(lay, ["T"], (1, 2, 3), 1)
    assert [c for c in calls if c[0] == "a0"] == [("a0", 2), ("a0", 3)]
    dl.stage_generate(lay, ["T"], (1, 2, 3), 1)                  # nothing left to do
    assert len([c for c in calls if c[0] == "a0"]) == 2
    touch(lay.hybrid("T", 2)[0])                                 # seed 2 already post-processed
    with pytest.raises(KeyboardInterrupt):
        dl.stage_simple(lay, ["T"], (1, 2, 3), 1, "hybrid")
    assert dl._tmp(lay.hybrid("T", 3)[0]).exists() and not lay.hybrid("T", 3)[0].exists()
    dl.stage_simple(lay, ["T"], (1, 2, 3), 1, "hybrid")          # resume: only the missing seed runs
    assert [c[1] for c in calls if c[0] == "hybrid"] == [1, 3, 3]
    dl.stage_simple(lay, ["T"], (1, 2, 3, 4), 1, "hybrid")       # seed 4 has no A0 yet: not attempted
    assert [c[1] for c in calls if c[0] == "hybrid"] == [1, 3, 3]
    assert len(lay.timing().read_text().splitlines()) == 4          # the interrupted call logged nothing


def test_atomic_writes_and_roundtrip(tmp_path):
    f = tmp_path / "a" / "x.json.gz"
    dl.put_jgz(f, {"v": [1, 2.5, float("nan")]})
    assert dl.get_jgz(f)["v"][:2] == [1, 2.5] and not dl._tmp(f).exists()
    g = tmp_path / "b.pkl.gz"
    dl.put_pkl(g, {"a": np.arange(3)})
    assert np.array_equal(dl.get_pkl(g)["a"], np.arange(3))


# ---------------------------------------------------------------- the hybrid runner and pairing
def _strip(log):
    drop = ("router_s", "seconds")
    return {st: {k: v for k, v in d.items() if k not in drop} if isinstance(d, dict) else d
            for st, d in log.items() if st != "seconds"}


def test_compact_runner_equals_run_hybrid():
    s = state(LINES)
    ref = hy.run_hybrid(s, PLANE20, grad_of(PLANE20), hy.stub_evaluator)
    got = dl.run_compact(s, PLANE20, grad_of(PLANE20), hy.stub_evaluator)
    assert ref["log"]["H1"].get("connectors_complete", 0) >= 1
    assert ref["log"]["H2"].get("runs_stairs", 0) + ref["log"]["H2"].get("runs_switchback", 0) >= 1
    assert np.array_equal(ref["graph"].nodes, got["graph"].nodes) and np.array_equal(ref["graph"].edges, got["graph"].edges)
    assert np.array_equal(ref["kind"], got["kind"]) and ref["gross_m"] == got["gross_m"] and ref["km"] == got["km"]
    assert _strip(ref["log"]) == _strip(got["log"])
    assert len(ref["snapshots"]) == len(got["snapshots"]) >= 3
    for a, (stage, gross, gj) in zip(ref["snapshots"], dl.snapshot_graphs(got["snapshots"])):
        assert a["stage"] == stage and a["gross_m"] == gross
        assert dl.edge_keys(a["graph"]) == dl.edge_keys(gj)        # every snapshot rebuilt exactly
    assert "graph" not in got["snapshots"][1]                     # stored as deltas


def test_arms_are_paired_on_one_a0_state():
    s = state(LINES)
    before = graph_to_json(s.graph(with_class=True))
    h1 = dl.run_compact(s, PLANE20, grad_of(PLANE20), hy.stub_evaluator, h2=False)
    assert graph_to_json(s.graph(with_class=True)) == before       # the A0 state is untouched by an arm
    h12 = dl.run_compact(s, PLANE20, grad_of(PLANE20), hy.stub_evaluator, h2=True)
    assert graph_to_json(s.graph(with_class=True)) == before
    keep = lambda d: {k: v for k, v in d.items() if k not in ("router_s",) and not k.startswith("pass2_")}
    assert keep(h1["log"]["H1"]) == keep(h12["log"]["H1"])         # H1 decisions identical in both arms
    snaps = list(dl.snapshot_graphs(h12["snapshots"]))
    after_h1 = [gj for stage, _, gj in snaps if stage in ("start", "H1")][-1]
    h1_final = {**graph_to_json(h1["graph"]), "kind": h1["kind"].tolist()}
    assert dl.edge_keys(after_h1) == dl.edge_keys(h1_final)       # H1 + H2 continues from H1 only's graph
    assert h12["gross_m"] >= h1["gross_m"] == pytest.approx(h1["log"]["H1"]["added_m"])


def test_switchback_logs_and_town_reading():
    s = state([STREET] + SHORT)
    res = dl.run_compact(s, PLANE20, grad_of(PLANE20), hy.stub_evaluator, h1=False)
    F_unit = H.HillField(PLANE20, np.ones(PLANE20.elev.shape, bool))
    left = dl.steep_left(res["state"], PLANE20, F_unit)
    assert res["log"]["H2"]["runs_switchback"] == 1 and left["steep_runs_unit"] == 0 and left["drivable_unit"] == 1
    a0 = dl.steep_left(hy.State(s), PLANE20, F_unit)
    assert a0["steep_runs_unit"] == 1 and a0["drivable_unit"] == 0 and a0["steep_km_unit"] == pytest.approx(0.15)


def test_fab_grad_matches_the_terrain_gradient():
    f = dl.fab_grad(PLANE20)
    gx, gy = f(250.0, 250.0)
    assert gx[0] == pytest.approx(0.0, abs=1e-6) and gy[0] == pytest.approx(0.20, abs=1e-6)


def test_split_kinds_carries_the_class_through_the_crop():
    import site_benchmark as sb
    site = sb.Site.__new__(sb.Site)
    site.L, site.n = 512.0, int(512 / sb.RES + 2 * sb.MARGIN / sb.RES)
    mask = np.ones((site.n, site.n), bool)
    gj = {"nodes": [[100, 100], [200, 100], [300, 100], [300, 200], [300, 300]], "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
          "cls": [2, 2, 3, 3], "st": [0, 0, 0, 0], "kind": [0, 0, 1, 2]}
    veh, ped, km = dl.split_kinds(gj, site, mask)
    assert len(veh.edges) == 2 and set(np.asarray(veh.classes()).tolist()) == {2}
    assert ped is not None and ped[1] == [False, True]               # steps flag = stairs only (choice 5)
    assert km == pytest.approx({"vehicular": 0.2, "stepped_ramp": 0.1, "stairs": 0.1})


# ---------------------------------------------------------------- guards
def test_never_touches_sealed_lists_or_the_town_guard():
    src = (ROOT / "scripts" / "m20_devlook.py").read_text()
    for bad in ("hold.csv", "study.csv", "m20_pool/lists", "\"lists\"", "guard("):
        assert bad not in src
    allowed = {"Paths", "TownSite", "load_town", "terrain_of", "hill_field"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module == "m20_towns":
            assert {a.name for a in node.names} <= allowed


def test_dev_towns_come_from_the_frozen_unit_list(tmp_path, monkeypatch):
    u = tmp_path / "dev_units.json"
    monkeypatch.setattr(dl, "UNITS", u)
    with pytest.raises(SystemExit):
        dl.town_ids("dev")
    u.write_text(json.dumps({"reference": ["A", "B"], "descriptive": ["V"]}))
    assert dl.town_ids("dev") == ["A", "B", "V"] and dl.town_ids("dev", ["V", "A"]) == ["A", "V"]
    with pytest.raises(SystemExit):
        dl.town_ids("dev", ["Z"])


def test_parse_seeds_and_worker_cap():
    assert dl.parse_seeds("1-3,7") == (1, 2, 3, 7) and dl.parse_seeds("1-24") == dl.SEEDS
    with pytest.raises(SystemExit):
        dl.main(["report", "--split", "scratch", "--workers", "7"])
