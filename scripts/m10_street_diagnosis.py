"""Milestone 10, Phase 1: why are generated streets irregular? Controlled diagnosis of GraphTurtle (M6 roads v2).

    python scripts/m10_street_diagnosis.py      # -> outputs/m10/diagnosis/street_diagnosis.json + figures

Train + validation tiles only (the test split is untouched). Checkpoints roads_v2_s0 / s1, unchanged.
  D1  reproduce the baseline: per-tile street metrics of 256 generated tiles per seed (sampling seed 123,
      as in M6) vs real validation (19) and real training (565) tiles;
  D2  raw model steps: the turn angle of every step the network samples, in its turtle frame (straight
      ahead = 0), BEFORE the geometric rules. "Snap deviation" = distance of the turn to the nearest of
      0 / +-90 / 180 deg. Compared for (a) real steps of validation traces, (b) teacher-forced samples
      (network given the REAL prefix), (c) free-running samples (network given its OWN prefix), by pop index;
  D3  junction origin: which mechanism made each generated junction (network branching, a merge into an
      existing node, or the crossing / T-junction rules), and its angular deviation;
  D4  compounding: junction deviation of generated graphs by node creation order (early vs late);
  D5  sampling temperature 0.25 / 0.5 / 1.0 (seed 0, 128 tiles): is the irregularity sampling variance?
A tile is the unit: per-tile medians are summarised, never the pooled samples of one tile as independent.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from m6_build_roads import load_graphs  # noqa: E402
from m6_evaluate import load_roads  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.road_graph import graph_metrics, junction_angle_deviations
from urbanmorph.graphs.sequence import DISP_SCALE, make_trace, pop_samples
from urbanmorph.models.graph_gen import mixture_sample
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "diagnosis"
KEYS = ["orientation_phi", "orientation_entropy", "junction_angle_dev_deg", "intersections_per_km2", "dead_end_ratio",
        "mean_link_length_m", "street_length_km_per_km2", "loops_per_km2", "share_deg4_of_junctions", "components"]
T_BINS = [0, 10, 40, 120, 10 ** 6]


def snap_dev(theta_deg):
    """Distance (deg) of a turn angle to the nearest of 0, +-90, 180."""
    a = np.mod(np.asarray(theta_deg), 90.0)
    return np.minimum(a, 90.0 - a)


def summarise(rows):
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        out[k] = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))} if len(v) else None
    return out


def w1_iqr(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    q = np.linspace(0, 1, 201)
    iqr = np.subtract(*np.percentile(b, [75, 25])) or 1e-9
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))) / iqr)


def teacher_forced(model, graphs, n_traces=2, seed=0):
    """Real and teacher-forced sampled turn angles (deg) with pop index; start pops excluded."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    real, samp, tidx = [], [], []
    for g in graphs:
        for _ in range(n_traces):
            ps = pop_samples(make_trace(g, rng, turtle=True), turtle=True)
            if not len(ps["crops"]):
                continue
            B = {k: torch.from_numpy(np.asarray(v)) for k, v in ps.items()}
            with torch.no_grad():
                ctx = model.context(B["crops"], B["scalars"], B["arms"], B["arm_mask"])
                tgt = B["targets"]
                prev = torch.cat([torch.zeros(len(tgt), 1, 2), tgt], 1)
                tc = B["target_class"]
                prev_c = torch.cat([torch.full((len(tgt), 1), -1, dtype=tc.dtype), tc], 1)
                _, (lp, mu, ls, rho), _, _ = model.decode_steps(ctx, prev, prev_class=prev_c)
            n = ps["n_targets"]
            start = ps["scalars"][:, -1] > 0.5
            for k in range(tgt.shape[1]):
                m = (n > k) & ~start
                if not m.any():
                    continue
                d = mixture_sample(lp[m, k], mu[m, k], ls[m, k], rho[m, k], gen).numpy()
                r = tgt[m, k].numpy()
                real += list(np.degrees(np.arctan2(r[:, 1], r[:, 0])))
                samp += list(np.degrees(np.arctan2(d[:, 1], d[:, 0])))
                tidx += list(np.flatnonzero(m))
    return np.array(real), np.array(samp), np.array(tidx)


def junctions(state):
    """(mechanism, median gap deviation, node index) per non-border junction of a GenState."""
    e = state.live_edges()
    adj = [set() for _ in range(state.n)]
    for a, b in e:
        adj[a].add(b)
        adj[b].add(a)
    out = []
    for u in range(state.n):
        if len(adj[u]) < 3 or state.border[u]:
            continue
        ang = np.sort([np.arctan2(*(state.nodes[w] - state.nodes[u])[::-1]) for w in adj[u]])
        gaps = np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]))
        dev = np.minimum(np.abs(gaps - 90), np.abs(gaps - 180))
        rule = state.node_rule.get(u, "new_node")
        mech = rule if rule in ("crossing_split", "t_junction") else ("merge" if state.merged_into.get(u, 0) else "branch")
        out.append((mech, float(np.median(dev)), u / max(state.n - 1, 1)))
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_graphs()
    val = [g for _, _, g in data["val"]]
    train = [g for _, _, g in data["train"]]
    res = {"real": {"val": summarise([graph_metrics(g) for g in val]), "train": summarise([graph_metrics(g) for g in train])}}
    real_val_rows = [graph_metrics(g) for g in val]
    res["real"]["junction_dev_pooled_median"] = {
        "val": float(np.median(np.concatenate([junction_angle_deviations(g) for g in val]))),
        "train": float(np.median(np.concatenate([junction_angle_deviations(g) for g in train if g.n])))}

    # D1 + D3 + D4 (+ step log for D2c)
    res["D1_baseline"], res["D3_junction_origin"], res["D4_by_creation_order"], logs = {}, {}, {}, {}
    for seed in (0, 1):
        model = load_roads(seed)
        states, log = [], []
        ts = time.time()
        gs = sample_graphs(model, 256, device="cpu", seed=123, states_out=states, step_log=log)
        rows = [graph_metrics(g) for g in gs]
        res["D1_baseline"][f"seed {seed}"] = {"summary": summarise(rows), "sampling_s": round(time.time() - ts, 1),
                                              "w1_iqr_vs_val": {k: w1_iqr([r[k] for r in rows], [r[k] for r in real_val_rows]) for k in KEYS},
                                              "rules_total": {k: int(sum(s.rules[k] for s in states)) for k in states[0].rules}}
        J = [j for s in states for j in junctions(s)]
        mechs = sorted({m for m, _, _ in J})
        res["D3_junction_origin"][f"seed {seed}"] = {
            m: {"share": float(np.mean([x == m for x, _, _ in J])), "median_dev_deg": float(np.median([d for x, d, _ in J if x == m])),
                "share_dev_gt_10deg": float(np.mean([d > 10 for x, d, _ in J if x == m]))} for m in mechs}
        pos = np.array([p for _, _, p in J])
        dev = np.array([d for _, d, _ in J])
        res["D4_by_creation_order"][f"seed {seed}"] = {f"{lo:.2f}-{hi:.2f}": float(np.median(dev[(pos >= lo) & (pos < hi + 1e-9)]))
                                                       for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0))}
        logs[seed] = np.array([(t, st, dx, dy) for _, t, st, dx, dy in log], float)
        print(f"seed {seed} sampled ({time.time() - t0:.0f}s)", flush=True)

    # D2: real vs teacher-forced vs free-running raw steps
    model = load_roads(0)
    real_th, tf_th, tf_t = teacher_forced(model, val)
    fr = logs[0]
    fr = fr[fr[:, 1] < 0.5]
    fr_th, fr_t = np.degrees(np.arctan2(fr[:, 3], fr[:, 2])), fr[:, 0]
    d2 = {}
    for lo, hi in zip(T_BINS[:-1], T_BINS[1:]):
        mr, mf = (tf_t >= lo) & (tf_t < hi), (fr_t >= lo) & (fr_t < hi)
        d2[f"pop {lo}-{hi if hi < 10 ** 6 else 'end'}"] = {
            "real_median_snap_dev": float(np.median(snap_dev(real_th[mr]))) if mr.any() else None,
            "teacher_forced_median_snap_dev": float(np.median(snap_dev(tf_th[mr]))) if mr.any() else None,
            "free_running_median_snap_dev": float(np.median(snap_dev(fr_th[mf]))) if mf.any() else None,
            "real_share_snap_lt_5": float(np.mean(snap_dev(real_th[mr]) < 5)) if mr.any() else None,
            "teacher_forced_share_snap_lt_5": float(np.mean(snap_dev(tf_th[mr]) < 5)) if mr.any() else None,
            "free_running_share_snap_lt_5": float(np.mean(snap_dev(fr_th[mf]) < 5)) if mf.any() else None,
            "n_real": int(mr.sum()), "n_free": int(mf.sum())}
    err = np.abs((tf_th - real_th + 180) % 360 - 180)
    res["D2_raw_steps"] = {"by_pop_index": d2, "teacher_forced_abs_turn_error_deg": {
        "median": float(np.median(err)), "share_gt_20": float(np.mean(err > 20))},
        "overall": {"real": float(np.median(snap_dev(real_th))), "teacher_forced": float(np.median(snap_dev(tf_th))),
                    "free_running": float(np.median(snap_dev(fr_th)))}}

    # D5: temperature
    res["D5_temperature"] = {}
    for temp in (0.25, 0.5, 1.0):
        states, log = [], []
        gs = sample_graphs(model, 128, device="cpu", seed=321, temperature=temp, states_out=states, step_log=log)
        rows = [graph_metrics(g) for g in gs]
        lg = np.array([(st, dx, dy) for _, _, st, dx, dy in log], float)
        lg = lg[lg[:, 0] < 0.5]
        J = [j for s in states for j in junctions(s)]
        res["D5_temperature"][str(temp)] = {
            "orientation_phi": summarise(rows)["orientation_phi"], "junction_angle_dev_deg": summarise(rows)["junction_angle_dev_deg"],
            "intersections_per_km2": summarise(rows)["intersections_per_km2"],
            "raw_step_median_snap_dev": float(np.median(snap_dev(np.degrees(np.arctan2(lg[:, 2], lg[:, 1]))))),
            "junction_share_by_mechanism": {m: float(np.mean([x == m for x, _, _ in J])) for m in sorted({x for x, _, _ in J})}}
        print(f"T={temp} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "street_diagnosis.json").write_text(json.dumps(res, indent=1))

    # figures: raw turn-angle histograms; junction deviation by mechanism
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), facecolor=SURFACE)
    bins = np.linspace(-180, 180, 145)
    for ax, (lab, th) in zip(axes, (("real steps (val traces)", real_th), ("teacher-forced samples", tf_th), ("free-running samples", fr_th))):
        ax.hist(th, bins, color="#3d6fb6")
        ax.set_title(f"{lab}: median snap dev {np.median(snap_dev(th)):.1f}°", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("turn angle in the turtle frame (deg, 0 = straight ahead)", fontsize=8)
        for x in (-180, -90, 0, 90, 180):
            ax.axvline(x, color=INK, lw=0.5, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "raw_turn_angles.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    d3 = res["D3_junction_origin"]["seed 0"]
    ax.bar(list(d3), [v["median_dev_deg"] for v in d3.values()], color="#3d6fb6")
    for i, (k, v) in enumerate(d3.items()):
        ax.text(i, v["median_dev_deg"], f"{v['share']:.0%} of junctions", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(res["real"]["junction_dev_pooled_median"]["val"], color=INK, ls="--", lw=0.8)
    ax.set_ylabel("median junction deviation (deg)")
    ax.set_title("Generated junctions by origin (seed 0); dashed = real validation", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "junction_origin.png", dpi=130, facecolor=SURFACE)
    print(json.dumps({k: v for k, v in res.items() if k != "real"}, indent=1))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
