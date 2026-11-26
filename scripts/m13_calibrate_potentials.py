"""Milestone 13: calibrate the sampler potentials from TRAINING tiles only (docs/M13_METHOD.md §3.3).

    python scripts/m13_calibrate_potentials.py --models N0 TN TNlp      # -> configs/m13_potentials/<model>.json

1. REAL features: every edge of the 3D-aware training graphs on training tiles with a DEM (US + Spain):
   * ground edges: maximum 30 m sub-span grade g, street class c, landform slope bin b (32 m) at the midpoint,
     contour alignment |cos| on land steeper than 5 %;
   * structures: bridge runs (M12 definition) over water -> span (run length) per class and crossing angle;
     tunnel runs -> length;
   * junctions: the smallest angle between edges at every node of degree >= 3.
2. PROPOSAL features: the same ground-edge features of the model's own unconstrained STATE-CONSISTENT generation
   (graphs/tc_sampler preset "standard", T = 0.5) on a seeded subset of the same training tiles (N_TILES tiles x 2
   samples, sampling seed 4242, distinct from every evaluation seed). (First version used the M12 legacy loop; replaced
   2026-11-06 before any selection result was examined, because the method defines the proposal as the
   state-consistent sampler.)
3. Log density ratios per (class, slope bin): log P_real(g-bin | c, b) - log P_prop(g-bin | c, b), add-one smoothed,
   clipped to [log 1/20, log 20]; alignment likewise per slope bin. Cells with fewer than MIN_N real or proposal
   edges get ratio 1 (log 0): no evidence, no correction.
No validation or test tile is read.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from m12_calibrate_feasibility import runs  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import load_index_m12
from urbanmorph.graphs.levels import BRIDGE, TUNNEL
from urbanmorph.graphs.tc_potentials import (ALIGN_BINS, GRADE_BINS, SLOPE_BINS, TerrainField, align_bin, edge_features,
                                             grade_bin, slope_bin)
from urbanmorph.graphs.terrain_generate import terrain_of

GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
OUT = PROJECT_ROOT / "configs" / "m13_potentials"
N_TILES, MIN_N, CLIP = 160, 30, np.log(20.0)
NG, NS, NA = len(GRADE_BINS) - 1, len(SLOPE_BINS) - 1, len(ALIGN_BINS) - 1


def ground_hist(graphs_and_tfs):
    """(4, NS, NG) grade counts and (NS, NA) alignment counts of the ground edges of (graph, TerrainField) pairs."""
    hg, ha = np.zeros((4, NS, NG)), np.zeros((NS, NA))
    for g, tf in graphs_and_tfs:
        if not tf.has_dem or not len(g.edges):
            continue
        seg, cls, st = g.segments(), g.classes(), g.structs()
        lv = g.levels() if getattr(g, "edge_level", None) is not None else np.zeros(len(seg), int)
        for k in np.flatnonzero((st == 0) & (lv == 0)):
            f = edge_features(tf, seg[k, :2], seg[k, 2:])
            if f["wet_length"] > 0 or not np.isfinite(f["grade"]) or f["length"] < 4:
                continue
            b = slope_bin(f["slope"])
            hg[int(cls[k]), b, grade_bin(f["grade"])] += 1
            if np.isfinite(f["align"]) and f["slope"] >= SLOPE_BINS[1]:
                ha[b, align_bin(f["align"])] += 1
    return hg, ha


def log_ratio(real, prop, axis=-1):
    nr, npr = real.sum(axis, keepdims=True), prop.sum(axis, keepdims=True)
    pr = (real + 1) / (nr + real.shape[axis])
    pp = (prop + 1) / (npr + prop.shape[axis])
    lr = np.clip(np.log(pr) - np.log(pp), -CLIP, CLIP)
    lr[np.broadcast_to((nr < MIN_N) | (npr < MIN_N), lr.shape)] = 0.0
    return lr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["N0", "TN", "TNlp"])
    ap.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint whose proposals are used")
    a = ap.parse_args()
    import torch
    import m12_evaluate as ev
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_m12()
    train_ids = set(idx[idx.split == "train"].sample_id)
    graphs = pickle.loads(GRAPHS3D.read_bytes())
    tr = [(sid, g) for sid, _, g in graphs["train"] if sid in train_ids]
    tfs, keep = {}, []
    for sid, g in tr:
        d = dict(np.load(TERRAIN / f"{sid}.npz"))
        if d["valid"].all():
            tfs[sid] = (TerrainField(terrain_of(d)), d)
            keep.append((sid, g))
    real_g, real_a = ground_hist([(g, tfs[s][0]) for s, g in keep])
    spans = {BRIDGE: {c: [] for c in range(4)}, TUNNEL: {c: [] for c in range(4)}}
    angles, junc = [], []
    for sid, g in keep:
        tf = tfs[sid][0]
        seg, cls, st = g.segments(), g.classes(), g.structs()
        for ks in runs(g):
            L = float(np.hypot(seg[ks, 2] - seg[ks, 0], seg[ks, 3] - seg[ks, 1]).sum())
            c = int(cls[ks].min())
            wet, ang = 0.0, []
            for k in ks:
                f = edge_features(tf, seg[k, :2], seg[k, 2:], struct=1)
                wet += f["wet_length"]
                if np.isfinite(f["angle"]):
                    ang.append(f["angle"])
            spans[BRIDGE][c].append(wet if wet > 0 else L)
            if ang:
                angles.append(float(np.median(ang)))
        tk = np.flatnonzero(st == TUNNEL)
        for k in tk:
            spans[TUNNEL][int(cls[k])].append(float(np.hypot(*(seg[k, 2:] - seg[k, :2]))))
        deg = g.degree()
        for v in np.flatnonzero(deg >= 3):
            inc = np.flatnonzero((g.edges[:, 0] == v) | (g.edges[:, 1] == v))
            dirs = [g.nodes[g.edges[k, 1] if g.edges[k, 0] == v else g.edges[k, 0]] - g.nodes[v] for k in inc]
            ang = np.sort(np.mod(np.degrees([np.arctan2(dd[1], dd[0]) for dd in dirs]), 360))
            gaps = np.diff(np.r_[ang, ang[0] + 360])
            junc.append(float(gaps.min()))

    def span_table(s):
        out = {}
        pooled = sorted(sum(s.values(), []))
        for c in range(4):
            v = sorted(sum((s[k] for k in range(c, 4)), [])) if len(s[c]) < MIN_N else sorted(s[c])
            out[c] = v if len(v) >= 10 else pooled
        out[-1] = pooled
        return out
    span_q = {BRIDGE: span_table(spans[BRIDGE]), TUNNEL: span_table(spans[TUNNEL])}
    common = {
        "span_q": {str(s): {str(c): v for c, v in t.items()} for s, t in span_q.items()},
        "angle_q": sorted(angles), "junction_q": sorted(junc),
        "span_max": {str(BRIDGE): float(max(sum(spans[BRIDGE].values(), []) or [np.inf])),
                     str(TUNNEL): float(max(sum(spans[TUNNEL].values(), []) or [np.inf]))},
        "angle_min": float(min(angles)) if angles else 0.0,
        "counts": {"train_tiles_with_dem": len(keep), "real_ground_edges": int(real_g.sum()),
                   "bridge_runs": {c: len(v) for c, v in spans[BRIDGE].items()},
                   "tunnel_edges": {c: len(v) for c, v in spans[TUNNEL].items()},
                   "crossing_angles": len(angles), "junctions": len(junc)},
    }
    rng = np.random.default_rng(20261106)
    sub = [keep[i] for i in sorted(rng.choice(len(keep), size=min(N_TILES, len(keep)), replace=False))]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    import m13_experiments as mx  # noqa: E402
    from urbanmorph.graphs.tc_potentials import Potentials
    from urbanmorph.graphs.tc_sampler import preset, sample_tc
    for spec in a.models:
        ds = [tfs[s][1] for s, _ in sub for _ in range(2)]
        sids = [s for s, _ in sub for _ in range(2)]
        model, names, norm, mcfg = mx.load_backbone(f"{spec}_s{a.seed}", device)
        crop = mx.model_inputs(mcfg, sids, ds)
        neutral = Potentials()                       # unused by the unconstrained state-consistent sampler
        gs, _, _ = sample_tc(model, ds, names, norm, neutral, preset("standard"), seed=4242, device=device, crop_tiles=crop)
        prop_g, prop_a = ground_hist([(g, tfs[s][0]) for g, (s, _) in zip(gs, [x for x in sub for _ in range(2)])])
        rep = {**common, "model": f"{spec}_s{a.seed}",
               "log_r_grade": log_ratio(real_g, prop_g).tolist(), "log_r_align": log_ratio(real_a, prop_a).tolist(),
               "real_grade_counts": real_g.tolist(), "prop_grade_counts": prop_g.tolist(),
               "real_align_counts": real_a.tolist(), "prop_align_counts": prop_a.tolist(),
               "grade_bins": GRADE_BINS.tolist(), "slope_bins": SLOPE_BINS.tolist(), "align_bins": ALIGN_BINS.tolist(),
               "source": {"real": "3D-aware graphs, train tiles with DEM (US + Spain)",
                          "proposal": f"{spec}_s{a.seed} state-consistent standard sampler, T=0.5, {len(sub)} train tiles x 2, seed 4242",
                          "version": "v2",
                          "smoothing": "add-one", "min_n": MIN_N, "clip": "|log r| <= log 20"}}
        (OUT / f"{spec}.json").write_text(json.dumps(rep, default=float))
        print(spec, "prop ground edges", int(prop_g.sum()), "real", int(real_g.sum()), flush=True)
    print(json.dumps(common["counts"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
