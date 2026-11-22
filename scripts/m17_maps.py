"""Milestone 17 maps (docs/M17_PROTOCOL.md §8): matched before / after maps with junction-event overlays.

For each arm (NA-SMC, NA-fixed, M* + organic) on the holdout, seed 101, two windows chosen by a fixed rule:
* TYPICAL: the window whose ORIGINAL excess close15 is the median over windows (ties: first by window id);
* FAILURE: the window with the largest remaining excess after F1.
Panels: real streets | original | F1 reuse | F0 post-hoc (same tau as F1). Junctions with another junction < 15 m are
circled, coloured by the edit that created the later junction of the pair (event log): T-snap, crossing split,
u-branch, merge, or unrecorded / post-hoc.

    python scripts/m17_maps.py   ->  outputs/m17/figures/fig_m17_<arm>_<case>.{png,pdf}
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

from m17_diagnose import rebuild
from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.runner import load_run
from urbanmorph.experiments.sites import site_set
from urbanmorph.graphs.junction_reuse import consolidate_graph
from urbanmorph.graphs.junctions import clean, focal_mask, junction_nodes, junction_row
from urbanmorph.visualization.m16_render import draw

OUT = ROOT / "outputs" / "m17" / "figures"
RUNS = ROOT / "outputs" / "m17" / "runs"
HOLD = ROOT / "data" / "processed" / "m17" / "holdout"
COL = {"t_junction": "#e67e22", "crossing_split": "#c0392b", "u": "#8e44ad", "merge": "#2980b9", "other": "#7f8c8d"}


def classify(ev):
    role = next((b[0] for b in ev["births"] if b[3] < 15.0), None)
    if ev["kind"] == "t_junction" and role == "split":
        return "t_junction"
    if ev["kind"] == "crossing_split" and role == "split":
        return "crossing_split"
    if role == "u":
        return "u"
    if ev["kind"] in ("merge_node", "crossing_endpoint"):
        return "merge"
    return "other"


def overlay(ax, g, events):
    nodes, e, _ = clean(g)
    J = nodes[junction_nodes(nodes, e)]
    if len(J) < 2:
        return
    d, nb = cKDTree(J).query(J, 2)
    close = np.flatnonzero(focal_mask(J) & (d[:, 1] < 15.0))
    born, kk = [], []
    for k, ev in enumerate(events or []):
        for x, y in ev["born"]:
            born.append((x, y))
            kk.append(k)
    bt = cKDTree(np.asarray(born)) if born else None
    for i in close:
        cls = "other"
        if bt is not None:
            dd, ii = bt.query(J[i])
            dj, jj = bt.query(J[nb[i, 1]])
            if dd < 0.05 and dj < 0.05:
                cls = classify(events[max(kk[ii], kk[jj])])
        ax.add_patch(plt.Circle(J[i], 7.5, fill=False, ec=COL[cls], lw=1.2, zorder=6))


def main() -> int:
    plan = json.loads((ROOT / "configs" / "m17_plans" / "holdout.json").read_text())["runs"]
    r = json.loads((ROOT / "outputs" / "m17" / "fixdev.json").read_text())["selected"]["r"]
    man = json.loads((ROOT / "configs" / "m17_holdout_windows.json").read_text())["windows"]
    real = {w["window"]: pickle.loads((HOLD / "streets" / f"{w['window']}.pkl").read_bytes()) for w in man}
    rc = {w: junction_row(g)["close15"] for w, g in real.items()}
    terr = {d["site"]: d for d in site_set("holdout_m17")}
    OUT.mkdir(parents=True, exist_ok=True)
    for m, st in (("NA-SMC", "off"), ("NA-fixed", "off"), ("M*", "organic")):
        sp = lambda pred: [s for s in plan if s["method"] == m and s["style"] == st and pred(s["overrides"])][0]
        R0, R1 = load_run(sp(lambda o: set(o) == {"event_log"}), RUNS), load_run(sp(lambda o: "junction_reuse_m" in o), RUNS)
        if R0 is None or R1 is None:
            continue
        v0, v1 = R0["seeds"]["101"], R1["seeds"]["101"]
        sites = [x["site"] for x in v0["rows"]]
        ex0 = {s: junction_row(rebuild(graph_from_json(v0["graphs"][k])))["close15"] - rc[s[5:]] for k, s in enumerate(sites)}
        ex1 = {s: junction_row(rebuild(graph_from_json(v1["graphs"][k])))["close15"] - rc[s[5:]] for k, s in enumerate(sites)}
        ok = sorted([s for s in sites if np.isfinite(ex0[s])], key=lambda s: (ex0[s], s))
        cases = {"typical": ok[len(ok) // 2], "failure": max([s for s in sites if np.isfinite(ex1[s])], key=lambda s: (ex1[s], s))}
        for case, site in cases.items():
            k = sites.index(site)
            g0, g1 = graph_from_json(v0["graphs"][k]), graph_from_json(v1["graphs"][k])
            e0 = (v0.get("events") or [None] * len(sites))[k]
            e1 = (v1.get("events") or [None] * len(sites))[k]
            gp = consolidate_graph(g0, r)
            fig, ax = plt.subplots(1, 4, figsize=(18, 5))
            panels = [("real public streets", real[site[5:]], None), (f"original (excess {ex0[site]:+.2f})", g0, e0),
                      (f"F1 reuse r={r:g} m (excess {ex1[site]:+.2f})", g1, e1), (f"F0 post-hoc tau={r:g} m", gp, [])]
            for a, (t, g, ev) in zip(ax, panels):
                draw(a, terr[site], g, title=t, note=f"close15 {junction_row(g)['close15']:.2f}")
                overlay(a, g, ev)
            hs = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=c, label=n) for n, c in
                  (("T-snap", COL["t_junction"]), ("crossing split", COL["crossing_split"]), ("u-branch", COL["u"]),
                   ("merge", COL["merge"]), ("unrecorded / post-hoc", COL["other"]))]
            fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=9)
            fig.suptitle(f"M17 holdout, {m}{' + ' + st if st != 'off' else ''}, window {site[5:]} ({case} case by rule), seed 101; "
                         "circles: junctions with another < 15 m, coloured by the creating edit", fontsize=10)
            fig.tight_layout(rect=(0, 0.06, 1, 0.94))
            name = f"fig_m17_{m.replace('*', 'star')}{'_' + st if st != 'off' else ''}_{case}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT / f"{name}.{ext}", dpi=120)
            plt.close(fig)
            print("wrote", name, site)
    return 0


if __name__ == "__main__":
    sys.exit(main())
