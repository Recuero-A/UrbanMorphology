"""Milestone 12: calibrate the feasibility thresholds from TRAINING tiles only.

    python scripts/m12_calibrate_feasibility.py   # -> configs/m12_feasibility.json, outputs/m12/audits/feasibility_calibration.json

* grade_max[class]: 99th percentile (length-weighted) of the maximum 30 m sub-span grade of real GROUND edges, on
  US train tiles only. AMENDED 2026-11-04 (before any model was trained): the first run used 10 m sub-spans and gave
  30 % for motorways - DEM artefacts at elevated structures and embankments, not street grades - so the window is
  30 m, the typical edge length, and road_feasibility.evaluate uses the same window (3DEP 1 m; the Spanish mosaic is terraced in places, docs/M12_DATA_PROTOCOL.md §5.3).
* bridge_span_max[class]: 99th percentile of real bridge-run lengths over water (all train tiles, OSM geometry;
  classes with < 10 runs take the pooled value of all classes at least as minor).
* bridge_angle_min_deg: 5th percentile of the real crossing angles (river width / wet length, as in evaluate()).
* lake_span_max: 99th percentile of real bridge runs over lakes / sea if >= 10, else the 30 m default.
* viaduct_clearance_min = 10 m, viaduct_span_max = 99th percentile of real DRY bridge runs: an engineering
  assumption plus data - the data contain no valley-spanning bridge (Phase 2 audit), so viaducts are rule-only.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import load_index_m12
from urbanmorph.graphs.levels import BRIDGE, TUNNEL
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.terrain.heightfield import grade_stats
from urbanmorph.terrain.road_feasibility import Thresholds, river_half_width, water_kind_at

GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
OUT = PROJECT_ROOT / "configs" / "m12_feasibility.json"
WINDOW_M = 30.0


def runs(g):
    bk = np.flatnonzero(g.structs() == BRIDGE)
    parent = {k: k for k in bk}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    by = {}
    for k in bk:
        for v in g.edges[k]:
            by.setdefault(int(v), []).append(k)
    for ks in by.values():
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    out = {}
    for k in bk:
        out.setdefault(find(k), []).append(k)
    return list(out.values())


def wq(v, w, q):
    v, w = np.asarray(v), np.asarray(w)
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(cw, q), len(o) - 1)])


def main() -> int:
    idx = load_index_m12()
    tr = idx[idx.split == "train"]
    country = dict(zip(tr.sample_id, tr.country))
    graphs = pickle.loads(GRAPHS3D.read_bytes())
    grades = {c: ([], []) for c in range(4)}
    spans = {"water": {c: [] for c in range(4)}, "lake": [], "dry": [], "angle": []}
    for sid, _, g in graphs["train"]:
        d = dict(np.load(TERRAIN / f"{sid}.npz"))
        t = terrain_of(d)
        seg, cls, st = g.segments(), g.classes(), g.structs()
        if country[sid] == "united_states" and d["valid"].all():
            for k in np.flatnonzero((st != BRIDGE) & (st != TUNNEL) & (g.levels() == 0)):
                s = grade_stats(*t.profile(seg[k, :2], seg[k, 2:]), window=WINDOW_M)
                if np.isfinite(s["max_grade"]):
                    grades[int(cls[k])][0].append(s["max_grade"])
                    grades[int(cls[k])][1].append(s["length"])
        hw = river_half_width(t)
        for ks in runs(g):
            L = float(np.hypot(seg[ks, 2] - seg[ks, 0], seg[ks, 3] - seg[ks, 1]).sum())
            c = int(cls[ks].min())
            p = np.concatenate([np.linspace(seg[k, :2], seg[k, 2:], 12) for k in ks])
            wk = water_kind_at(t, p[:, 0], p[:, 1])
            if (wk == 1).any():
                spans["water"][c].append(L)
                wet_len = (wk > 0).mean() * L
                r, cc = t.rc(p[wk > 0, 0], p[wk > 0, 1])
                r = np.clip(np.round(r).astype(int), 0, hw.shape[0] - 1)
                cc = np.clip(np.round(cc).astype(int), 0, hw.shape[1] - 1)
                if wet_len > 0:
                    spans["angle"].append(float(np.degrees(np.arcsin(np.clip(2 * hw[r, cc].max() / wet_len, 0, 1)))))
            elif (wk > 1).any():
                spans["lake"].append(L)
            else:
                spans["dry"].append(L)
    th = Thresholds()
    rep = {"grade_samples_m": {c: float(np.sum(grades[c][1])) for c in range(4)}}
    th.grade_max = {c: round(wq(*grades[c], 0.99), 3) if len(grades[c][0]) >= 50 else th.grade_max[c] for c in range(4)}
    span = {}
    for c in range(4):
        pool = sum((spans["water"][k] for k in range(c, 4)), [])
        v = spans["water"][c] if len(spans["water"][c]) >= 10 else pool
        span[c] = round(float(np.percentile(v, 99)), 1) if len(v) >= 10 else th.bridge_span_max[c]
    th.bridge_span_max = span
    if len(spans["angle"]) >= 10:
        th.bridge_angle_min_deg = round(float(np.percentile(spans["angle"], 5)), 1)
    if len(spans["lake"]) >= 10:
        th.lake_span_max = round(float(np.percentile(spans["lake"], 99)), 1)
    if len(spans["dry"]) >= 10:
        th.viaduct_span_max = round(float(np.percentile(spans["dry"], 99)), 1)
    th.source = {"grade_max": "p99 of max 30 m sub-span grade, real ground edges, US train tiles (3DEP)",
                 "bridge_span_max": "p99 of real bridge runs over rivers, train tiles (pooled to more minor classes if < 10)",
                 "bridge_angle_min_deg": "p5 of real crossing angles over rivers, train tiles",
                 "lake_span_max": "p99 of real bridges over lakes / sea if >= 10, else default 30 m",
                 "viaduct_clearance_min": "engineering assumption (10 m): no valley-spanning bridge in the data",
                 "viaduct_span_max": "p99 of real dry bridge runs (overpasses), train tiles"}
    th.save(OUT)
    rep.update({"n_water_runs": {c: len(spans["water"][c]) for c in range(4)}, "n_lake_runs": len(spans["lake"]),
                "n_dry_runs": len(spans["dry"]), "n_angles": len(spans["angle"]),
                "angle_quantiles": np.percentile(spans["angle"], [5, 25, 50]).tolist() if spans["angle"] else None,
                "thresholds": json.loads(OUT.read_text())})
    (PROJECT_ROOT / "outputs" / "m12" / "audits" / "feasibility_calibration.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
