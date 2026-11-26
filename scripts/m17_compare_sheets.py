"""Milestone 17: comparison sheets of every sampler and its improved versions on holdout windows (seed 101).

Windows (fixed rule): window 0 of the first metro of each country in configs/m17_holdout.json (KOR_GWJ_0, PER_PIU_0,
ESP_CAS_0). Rows: M*, NA-fixed, NA-SMC, M* + organic, M* + radial. Columns: real public streets | original |
F1 reuse (r = 15 m) | F4 degree prior (lambda 1, EXPLORATORY) | F1 + F4 | F0 post-hoc tau = 9 m. Markers (focal junctions): red circle = another
junction < 15 m; black square = 5 or more branches. Panel notes: close15 and the 5+-way count.

    python scripts/m17_compare_sheets.py   ->  outputs/m17/figures/fig_m17_compare_<window>.{png,pdf}
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.runner import load_run
from urbanmorph.experiments.sites import site_set
from urbanmorph.graphs.junction_reuse import consolidate_graph
from urbanmorph.graphs.junctions import clean, focal_mask, junction_row
from urbanmorph.visualization.m16_render import draw

OUT = ROOT / "outputs" / "m17" / "figures"
RUNS = ROOT / "outputs" / "m17" / "runs"
HOLD = ROOT / "data" / "processed" / "m17" / "holdout"
ARMS = [("M*", "off"), ("NA-fixed", "off"), ("NA-SMC", "off"), ("M*", "organic"), ("M*", "radial")]


def marks(ax, g):
    n, e, _ = clean(g)
    if len(e) == 0:
        return 0
    d = np.bincount(e.ravel(), minlength=len(n))
    J = np.flatnonzero(d >= 3)
    if len(J) >= 2:
        dist, _ = cKDTree(n[J]).query(n[J], 2)
        for k in np.flatnonzero(focal_mask(n[J]) & (dist[:, 1] < 15.0)):
            ax.add_patch(plt.Circle(n[J][k], 7.0, fill=False, ec="#e74c3c", lw=1.1, zorder=6))
    hi = np.flatnonzero((d >= 5) & focal_mask(n))
    ax.scatter(n[hi, 0], n[hi, 1], marker="s", s=28, facecolors="none", edgecolors="k", linewidths=1.3, zorder=7)
    return len(hi)


def main() -> int:
    cfg = json.loads((ROOT / "configs" / "m17_holdout.json").read_text())
    plan = json.loads((ROOT / "configs" / "m17_plans" / "holdout.json").read_text())["runs"]
    plan2 = json.loads((ROOT / "configs" / "m17_plans" / "degholdout.json").read_text())["runs"]
    terr = {d["site"]: d for d in site_set("holdout_m17")}
    first = {}
    for m in cfg["metros"]:
        first.setdefault(m["id"][:3], m["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    for metro in first.values():
        wid = f"{metro}_0"
        site = f"hold_{wid}"
        real = pickle.loads((HOLD / "streets" / f"{wid}.pkl").read_bytes())
        cols = ["real public streets", "original", "F1 reuse r=15", "F4 degree prior (λ 1)", "F1 + F4", "F0 post-hoc tau=9"]
        fig, ax = plt.subplots(len(ARMS), len(cols), figsize=(4.0 * len(cols), 4.1 * len(ARMS)))
        for i, (m, st) in enumerate(ARMS):
            sp = lambda pred: [s for s in plan if s["method"] == m and s["style"] == st and pred(s["overrides"])][0]
            r0 = load_run(sp(lambda o: set(o) == {"event_log"}), RUNS)["seeds"]["101"]
            r1 = load_run(sp(lambda o: "junction_reuse_m" in o), RUNS)["seeds"]["101"]
            k = [x["site"] for x in r0["rows"]].index(site)
            g0 = graph_from_json(r0["graphs"][k])
            g1 = graph_from_json(r1["graphs"][[x["site"] for x in r1["rows"]].index(site)])
            sp2 = lambda pred: [x for x in plan2 if x["method"] == m and x["style"] == st and pred(x["overrides"])][0]
            r4 = load_run(sp2(lambda o: "junction_reuse_m" not in o), RUNS)["seeds"]["101"]
            r14 = load_run(sp2(lambda o: "junction_reuse_m" in o), RUNS)["seeds"]["101"]
            g4 = graph_from_json(r4["graphs"][[x["site"] for x in r4["rows"]].index(site)])
            g14 = graph_from_json(r14["graphs"][[x["site"] for x in r14["rows"]].index(site)])
            gs = [real, g0, g1, g4, g14, consolidate_graph(g0, 9.0)]
            for j, (lab, g) in enumerate(zip(cols, gs)):
                a = ax[i, j]
                draw(a, terr[site], g, title=(lab if i == 0 else ""), scalebar=(i == 0 and j == 0))
                n5 = marks(a, g)
                c = junction_row(g)["close15"]
                a.text(8, 500, f"close15 {c:.2f} · 5+-way {n5}", fontsize=8, va="top",
                       bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1), zorder=8)
                if j == 0:
                    a.set_ylabel(m + (f" + {st}" if st != "off" else ""), fontsize=11)
        hs = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#e74c3c", label="junction with another < 15 m"),
              plt.Line2D([], [], marker="s", ls="", mfc="none", mec="k", label="5+ branch junction")]
        fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=11)
        fig.suptitle(f"M17 holdout window {wid} ({next(x['name'] for x in cfg['metros'] if x['id'] == metro)}), seed 101: "
                     "each sampler and its improved versions (GLO-30 hillshade)", fontsize=13)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig_m17_compare_{wid}.{ext}", dpi=90)
        plt.close(fig)
        print("wrote", wid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
