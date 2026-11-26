"""Milestone 17, step 2: what creates the excess close junctions? (docs/M17_PROTOCOL.md §4)

Reads the stored runs of configs/m17_plans/diagnose.json (outputs/m17/runs) and writes outputs/m17/diagnosis.{md,json}
and figures. For every final graph (RAW and REBUILT, focal junctions of the 512 m tile):
* close-junction statistics against the matched real window (the same pack window's public streets);
* attribution: every focal junction whose nearest other junction is < 15 m is traced, by its coordinates, to the edit
  at which it became a junction (event log); a close pair is assigned to the LATER of its two births; classes =
  edit kind x birth role (split: the node a crossing / T creates; v: an existing node reached by a merge; u: the
  expanding node gaining a branch), with the closure flag (network term > 0) and the source (model / style);
* H1: share of close births where reuse of the nearby junction was feasible, and where a feasible candidate without a
  close birth existed ("avoidable");
* H2: close births vs the network weight multiplier (NA-fixed, NA-SMC), and close-birth rates of chosen closures vs
  other edits;
* H3: close15 against focal junction density, with real windows (M15 pilot + training tiles) in the same density bins.

    python scripts/m17_diagnose.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.experiments.runner import load_run
from urbanmorph.graphs.junctions import clean, junction_nodes, junction_row, nn_distances
from urbanmorph.graphs.road_graph import build_graph

RUNS = ROOT / "outputs" / "m17" / "runs"
OUT = ROOT / "outputs" / "m17"
PACK = ROOT / "data" / "processed" / "m16" / "pack" / "streets"
BINS = [0, 75, 125, 175, 250, 350, 500, 10000]


def rebuild(g):
    lines = [g.nodes[[a, b]] for a, b in g.edges]
    return build_graph(lines) if lines else g


def label(spec) -> str:
    m = spec["method"]
    ov = spec.get("overrides", {})
    tag = m + (f" + {spec['style']}" if spec.get("proposals") else "")
    if "lam_dead" in ov:
        tag += f" lambda x{ov['lam_dead'] / 1.8143:.1f}"
    if "stop_bias" in ov:
        tag += f" stop {ov['stop_bias']:+.1f}"
    return tag


def attribute(g, events):
    """Counter of close-pair classes for the focal close junctions of g (pairs counted once)."""
    bxy, bk = [], []
    for k, ev in enumerate(events or []):
        for x, y in ev["born"]:
            bxy.append((x, y))
            bk.append(k)
    from scipy.spatial import cKDTree as _T
    btree = _T(np.asarray(bxy)) if bxy else None
    nodes, e, _ = clean(g)
    jid = junction_nodes(nodes, e)
    J = nodes[jid]
    out = Counter()
    if len(J) < 2:
        return out, 0
    from scipy.spatial import cKDTree
    from urbanmorph.graphs.junctions import focal_mask
    d, nb = cKDTree(J).query(J, 2)
    foc = focal_mask(J)
    seen = set()
    for i in np.flatnonzero(foc & (d[:, 1] < 15.0)):
        j = int(nb[i, 1])
        pair = tuple(sorted((int(i), j)))
        if pair in seen:
            continue
        seen.add(pair)
        ks = []
        for t in pair:
            if btree is None:
                ks.append(None)
                continue
            dd, ii = btree.query(J[t])
            ks.append(bk[ii] if dd < 0.05 else None)          # stored graphs round coordinates (graph_io)
        if any(k is None for k in ks):
            out["unrecorded (start / anchor / pre-log)"] += 1
            continue
        ev = events[max(ks)]
        role = next((b[0] for b in ev["births"] if b[3] < 15.0), ev["births"][0][0] if ev["births"] else "?")
        cls = f"{ev['kind']}/{role}" + (" +closure" if ev.get("lw_net", 0) > 0 else "") + (" [style]" if ev.get("src") == "style" else "")
        out[cls] += 1
    return out, len(seen)


def main() -> int:
    plan = json.loads((ROOT / "configs" / "m17_plans" / "diagnose.json").read_text())["runs"]
    real = {}
    for f in PACK.glob("*.pkl"):
        if "__" not in f.stem and "CAMPUS" not in f.stem:
            real[f.stem] = pickle.loads(f.read_bytes())
    rows, attr, evstats = [], {}, {}
    for spec in plan:
        r = load_run(spec, RUNS)
        if r is None:
            continue
        lab = label(spec)
        A, E = Counter(), Counter()
        for sd, v in r["seeds"].items():
            for k, row in enumerate(v["rows"]):
                site = row["site"]
                g = graph_from_json(v["graphs"][k])
                raw, reb = junction_row(g), junction_row(rebuild(g))
                w = site.split("_")[1] + ("_" + site.split("_")[2] if site.split("_")[2] not in ("glo30", "mdt05") else "")
                rr = junction_row(real[w]) if w in real else None
                rows.append({"run": lab, "sites": spec["sites"], "site": site, "seed": int(sd),
                             "raw_close15": raw["close15"], "reb_close15": reb["close15"], "raw_p10": raw["nn_p10"],
                             "reb_p10": reb["nn_p10"], "raw_consol10": raw["consol10_ratio"],
                             "reb_consol10": reb["consol10_ratio"], "density": reb["focal_density_km2"],
                             "real_close15": rr["close15"] if rr else np.nan, "street_km": row.get("street_km"),
                             "od_unreach": row.get("od_unreach_land"), "coverage": row.get("coverage_60"),
                             "empty": row.get("empty")})
                evs = (v.get("events") or [None] * len(v["rows"]))[k]
                if evs:
                    a, n = attribute(g, evs)
                    A.update(a)
                    for ev in evs:
                        E["events"] += 1
                        E["closure"] += int(ev.get("lw_net", 0) > 0)
                        if ev["chosen_close"]:
                            E["close_births"] += 1
                            E["close_closure"] += int(ev.get("lw_net", 0) > 0)
                            E["avoidable"] += int(ev["avoidable"])
                            if ev["reuse"] is not None:
                                E["reuse_applicable"] += 1
                                E["reuse_feasible"] += int(ev["reuse"])
        attr[lab] = attr.get(lab, Counter()) + A
        evstats[lab] = evstats.get(lab, Counter()) + E
    df = pd.DataFrame(rows)
    # real windows by density (H3 reference): M15 pilot + training tiles, REBUILT by construction
    ref = []
    for _, _, g, _ in pickle.loads((ROOT / "data" / "processed" / "m15" / "real" / "graphs.pkl").read_bytes()):
        j = junction_row(g)
        ref.append((j["focal_density_km2"], j["close15"]))
    for _, _, g in pickle.loads((ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())["train"]:
        j = junction_row(g)
        ref.append((j["focal_density_km2"], j["close15"]))
    ref = pd.DataFrame(ref, columns=["density", "close15"]).dropna()
    ref["bin"] = pd.cut(ref["density"], BINS)
    refbin = ref.groupby("bin", observed=False)["close15"].agg(["mean", "count"])
    df["bin"] = pd.cut(df["density"], BINS)
    df["real_at_density"] = df["bin"].map(refbin["mean"]).astype(float)
    df["excess_matched_density"] = df["reb_close15"] - df["real_at_density"]
    df["excess_window"] = df["reb_close15"] - df["real_close15"]
    g = df.groupby("run", sort=False)
    tab = g.agg(raw_close15=("raw_close15", "mean"), reb_close15=("reb_close15", "mean"), real_window=("real_close15", "mean"),
                excess_window=("excess_window", "mean"), density=("density", "mean"),
                excess_matched_density=("excess_matched_density", "mean"), reb_p10=("reb_p10", "mean"),
                raw_consol10=("raw_consol10", "mean"), street_km=("street_km", "mean"), od_unreach=("od_unreach", "mean"),
                windows=("site", "size"), scored=("reb_close15", lambda s: int(s.notna().sum())))
    L = ["# M17 step 2: diagnosis of excess close junctions (DEVELOPMENT: terrain pack windows, seeds 1-10)", "",
         "Generated by `scripts/m17_diagnose.py` from `configs/m17_plans/diagnose.json`. close15: share of focal junctions with "
         "another < 15 m (window mean over windows with >= 2 focal junctions). real_window: the same pack window's real public "
         "streets. excess_matched_density: REBUILT close15 minus the mean close15 of real windows (M15 pilot + training) in the "
         "same focal-density bin.", "",
         "| Run | close15 raw | close15 rebuilt | real (same window) | excess vs window | junctions / km2 | excess at matched density | nn p10 rebuilt | consol10 raw | street km | OD unreachable | windows (scored) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n=3: "–" if x is None or not np.isfinite(x) else f"{x:.{n}f}"
    for lab, r in tab.iterrows():
        L.append(f"| {lab} | {f(r.raw_close15)} | {f(r.reb_close15)} | {f(r.real_window)} | {f(r.excess_window)} | {f(r.density, 0)} | "
                 f"{f(r.excess_matched_density)} | {f(r.reb_p10, 1)} | {f(r.raw_consol10)} | {f(r.street_km, 2)} | {f(r.od_unreach)} | {int(r.windows)} ({int(r.scored)}) |")
    L += ["", "Real windows by focal-junction density (H3 reference: M15 pilot + training, rebuilt):", "",
          "| density bin (junctions / km2) | real close15 | windows |", "|---|---|---|"]
    for b, r in refbin.iterrows():
        L.append(f"| {b} | {f(r['mean'])} | {int(r['count'])} |")
    L += ["", "## H1. Which edit creates the close pair (later birth of each focal pair < 15 m; RAW final graphs)", ""]
    for lab, A in attr.items():
        n = sum(A.values())
        if not n:
            continue
        L.append(f"* **{lab}** ({n} pairs): " + ", ".join(f"{k} {v / n:.0%}" for k, v in A.most_common(6)))
    L += ["", "## H1 / H2. Edit-level statistics (all committed edits)", "",
          "| Run | edits | close births / 100 edits | of which closures | closures among all edits | close-birth rate: closures | close-birth rate: other edits | reuse feasible (when applicable) | avoidable |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, E in evstats.items():
        if not E["events"]:
            continue
        cb, ev, cl = E["close_births"], E["events"], E["closure"]
        rc = E["close_closure"] / max(cl, 1)
        ro = (cb - E["close_closure"]) / max(ev - cl, 1)
        L.append(f"| {lab} | {ev} | {100 * cb / ev:.2f} | {E['close_closure'] / max(cb, 1):.0%} | {cl / ev:.1%} | {rc:.3f} | {ro:.3f} | "
                 f"{E['reuse_feasible'] / max(E['reuse_applicable'], 1):.0%} ({E['reuse_applicable']}) | {E['avoidable'] / max(cb, 1):.0%} |")
    (OUT / "diagnosis.md").write_text("\n".join(L) + "\n")
    atomic_json(OUT / "diagnosis.json", {"table": tab.reset_index().to_dict("records"),
                                         "attribution": {k: dict(v) for k, v in attr.items()},
                                         "events": {k: dict(v) for k, v in evstats.items()},
                                         "real_by_density": refbin.reset_index().astype(str).to_dict("records")})
    df.drop(columns=["bin"]).to_csv(OUT / "diagnosis_windows.csv", index=False)
    # figure: close15 vs density
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ref["density"], ref["close15"], s=6, c="0.7", label="real windows (M15 pilot + training)")
    for lab, sub in df.groupby("run", sort=False):
        if "lambda" in lab or "stop" in lab:
            continue
        ax.scatter(sub["density"].mean(), sub["reb_close15"].mean(), s=40, label=lab)
    cen = refbin["mean"].values
    mids = [(a.left + min(a.right, 800)) / 2 for a in refbin.index]
    ax.plot(mids, cen, "k--", lw=1, label="real mean per density bin")
    ax.set_xlim(0, 800)
    ax.set_xlabel("focal junctions per km2 (rebuilt)")
    ax.set_ylabel("close15 (rebuilt)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Close junctions vs junction density (development windows)")
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "fig_close15_vs_density.png", dpi=130)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
