"""Milestone 11 (exploratory): GraphTurtle N street samples at several temperatures, same seed and batch.

    python scripts/m11_temperature_streets.py --seed 2032 --temps 0.1 0.25 0.5 1.0 1.5

Writes outputs/m11/temperature/streets_by_temperature.{png,json}: per temperature, a batch of 10 tiles from the
same sampling seed (row = temperature, column = batch slot) and simple street statistics per slot.
Temperature multiplies the standard deviations of the step mixture only (graphs/generate.py).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs

N = 10
COLORS = {0: "#c0392b", 1: "#e67e22", 2: "#34495e", 3: "#95a5a6"}


def stats(g) -> dict:
    deg = g.degree()
    inner = ~np.asarray(g.border, bool)
    seg = g.segments() if len(g.edges) else np.zeros((0, 4))
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    junc = (deg >= 3) & inner
    adj = g.adjacency()
    bend = []
    for u in np.flatnonzero((deg == 2) & inner):  # mid-street nodes: deviation from straight on
        a, b = (g.nodes[w] - g.nodes[u] for w in adj[u])
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        bend.append(180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return {"bend_deg_median": round(float(np.median(bend)), 1) if bend else None,"edges": int(len(g.edges)), "length_km": round(float(lengths.sum()) / 1000, 2),
            "dead_end_share": round(float(((deg == 1) & inner).sum() / max(1, ((deg != 2) & inner).sum())), 3),
            "four_way_share": round(float(((deg >= 4) & inner).sum() / max(1, junc.sum())), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2032)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 1.5])
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    from m10_eval_streets import CK, load_model  # noqa: E402
    model = load_model(CK / "m10" / "N_noise_s0")
    out = PROJECT_ROOT / "outputs" / "m11" / "temperature"
    out.mkdir(parents=True, exist_ok=True)
    res = {"seed": a.seed, "batch": N, "model": "N_noise_s0", "temperatures": {}}
    fig, ax = plt.subplots(len(a.temps), a.cols, figsize=(2.2 * a.cols, 2.3 * len(a.temps)))
    for r, t in enumerate(a.temps):
        gs = sample_graphs(model, N, seed=a.seed, temperature=t)
        res["temperatures"][str(t)] = [stats(g) for g in gs]
        for c in range(a.cols):
            x = ax[r, c]
            g = gs[c]
            if len(g.edges):
                seg = g.segments()
                cls = g.classes()
                for i, s in enumerate(seg):
                    k = int(cls[i])
                    x.plot(s[[0, 2]], s[[1, 3]], color=COLORS.get(k, "#34495e"), lw=1.4 if k < 2 else 0.8)
            x.set_xlim(0, 512); x.set_ylim(0, 512); x.set_aspect("equal"); x.set_xticks([]); x.set_yticks([])
            if c == 0:
                x.set_ylabel(f"T = {t:g}", fontsize=12)
            if r == 0:
                x.set_title(f"slot {c}", fontsize=10)
            x.text(6, 6, f"{len(g.edges)} edges", fontsize=7, color="#555")
    fig.suptitle(f"GraphTurtle N streets, sampling seed {a.seed}: rows = temperature, columns = batch slot "
                 "(red major, orange mid, dark local, grey minor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "streets_by_temperature.png", dpi=110)
    (out / "streets_by_temperature.json").write_text(json.dumps(res, indent=1))
    for t, rows in res["temperatures"].items():
        print(t, [s["edges"] for s in rows], "median dead-end", np.median([s["dead_end_share"] for s in rows]),
              "4-way", np.median([s["four_way_share"] for s in rows]),
              "bend", np.median([s["bend_deg_median"] for s in rows]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
