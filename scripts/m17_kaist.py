"""Milestone 17 on the KAIST campus (EXPLORATORY demo; GLO-30 DSM; 1536 m domain; same supplied arterials as M16).

Each method in two versions: ORIGINAL (the M16 campus cities, outputs/m16/kaist/all_cache) and IMPROVED = F1
topology-aware junction reuse (15 m) + F4 junction-degree prior (lambda 1; docs/M17_RESULTS.md §7). Seeds 1-3.
Scores per 512 m window (3 x 3 per city): close15 (REBUILT), 5+-way share (pooled focal junctions), junctions per km2,
street km, and realism typicality against the 156 real neighbourhoods of scripts/m16_realism.py.

    python scripts/m17_kaist.py --one KEY SEED     # one improved city (parallel-safe cache)
    python scripts/m17_kaist.py --assemble         # scores, table, 2D sheet; graphs for the 3D builder
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from m16_kaist_all import CACHE as ORIG_CACHE
from m16_kaist_all import AnchoredGlobalNet, key_of, window_rows
from m16_kaist_campus import L, OUT, PACK, W, AnchoredGlobal, anchors, domain, measures, merge, snap
from m16_kaist_styles import windows_of
from m16_plans import proposals
from m16_realism import K, real_pool, vec
from urbanmorph.experiments.backbones import load_backbone
from urbanmorph.experiments.methods import method
from urbanmorph.graphs.junctions import junction_row
from urbanmorph.graphs.m14_sampler import net_preset
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset
from urbanmorph.visualization.m16_render import draw

METHODS = ["A", "M*", "M13-SMC", "NA-fixed", "NA-SMC", "M* + organic", "M* + radial"]
FIX = {"junction_reuse_m": 15.0, "degree_prior": 1.0}
CACHE = OUT / "m17_cache"
FZ = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]


def generate_one(lab: str, seed: int):
    f = CACHE / f"{key_of(lab)}_s{seed}.pkl"
    if f.exists():
        return
    big, (x0, y0), crs = domain()
    model, names, norm, _ = load_backbone("TN_s0", "cpu")
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    base = lab.split(" + ")[0]
    m = method(base)
    ov = {**m.overrides, **FIX}
    if " + " in lab:
        st = lab.split(" + ")[1]
        ov.update(json.loads((ROOT / "configs" / "m15_style.json").read_text())["styles"][st])
        ov.update(proposals("style_topo", FZ["M*"][st]["style_topo"]["prop_alpha"]))
        ov.update(style_cx=L / 2, style_cy=L / 2)
    is_net = base.startswith("NA")
    cfg = net_preset(m.preset, **ov) if is_net else preset(m.preset, **ov)
    t0 = time.time()
    sm = (AnchoredGlobalNet if is_net else AnchoredGlobal)(model, {**big, "site": f"KAIST_CAMPUS_{lab}"}, L, names, norm,
                                                           pots, cfg)
    sm.anchor = anchors(x0, y0, crs)
    g = sm.run(seed)[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"label": lab, "seed": seed, "graph": g, "seconds": time.time() - t0,
                                  "reuse_applied": int(sm.stats[0].get("reuse_applied", 0))}))
    tmp.rename(f)
    print(f"done {lab} improved seed {seed} {time.time() - t0:.0f}s", flush=True)


def scores(g, big, X, Z, mu, sd, med, real_typ, rng):
    reb = [junction_row(wg) if wg is not None else None for _, wg in windows_of(g)]
    raw = [junction_row(g, window=(c * W, r * W, c * W + W, r * W + W)) for r in range(3) for c in range(3)]
    n = {k: sum(x[k] for x in reb if x) for k in ("n3", "n4", "n5p")}
    tot = max(sum(n.values()), 1)
    rows = [x for x in window_rows(g, big, rng) if x is not None]
    V = np.where(np.isfinite(np.array([vec(x) for x in rows], float)), np.array([vec(x) for x in rows], float), med)
    G = (V - mu) / sd
    typ = np.sort(np.sqrt(((G[:, None] - Z[None]) ** 2).sum(-1)), 1)[:, :K].mean(1) / real_typ
    return {"close15": float(np.nanmean([x["close15"] for x in reb if x])),
            "close15_raw": float(np.nanmean([x["close15"] for x in raw])),
            "share5p": n["n5p"] / tot, "share4": n["n4"] / tot, "junctions_km2": float(np.nanmean([x["focal_density_km2"] for x in reb if x])),
            "typicality": float(np.mean(typ)), **{k: measures(g, rng)[k] for k in ("street_km", "xwin_unreach")}}


def assemble():
    big, _, _ = domain()
    X, _ = real_pool()
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi.mean(0), np.where(Xi.std(0) > 1e-9, Xi.std(0), 1.0)
    Z = (Xi - mu) / sd
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    real_typ = float(np.mean(np.sort(D, 1)[:, :K].mean(1)))
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    real_g = snap(merge([(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}__vehicular.pkl").read_bytes()), off)
                         for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]))
    res, graphs = [], {"real": real_g}
    res.append({"method": "real KAIST (vehicular)", "version": "reference",
                **scores(real_g, big, X, Z, mu, sd, med, real_typ, np.random.default_rng(0))})
    for lab in METHODS:
        for ver, cache in (("original", ORIG_CACHE), ("improved (F1 + F4)", CACHE)):
            per = []
            for seed in (1, 2, 3):
                g = pickle.loads((cache / f"{key_of(lab)}_s{seed}.pkl").read_bytes())["graph"]
                graphs[(lab, ver, seed)] = g
                per.append(scores(g, big, X, Z, mu, sd, med, real_typ, np.random.default_rng([seed, 16])))
            res.append({"method": lab, "version": ver, **{k: float(np.mean([p[k] for p in per])) for k in per[0]}})
            print(lab, ver, {k: round(v, 3) for k, v in res[-1].items() if isinstance(v, float)}, flush=True)
    (OUT / "m17_graphs.pkl").write_bytes(pickle.dumps({"graphs": {(k[0] + (" [improved]" if k[1].startswith("impr") else ""), k[2]): v
                                                                  for k, v in graphs.items() if k != "real"},
                                                       "real": real_g}))
    json.dump(res, open(OUT / "m17_scores.json", "w"), indent=1, default=float)
    L_ = ["# KAIST campus: original vs improved junctions (F1 reuse 15 m + F4 degree prior λ 1) — EXPLORATORY", "",
          "1536 m campus, GLO-30 DSM, same real arterials; seeds 1-3. close15 = share of junctions with another < 15 m "
          "(REBUILT, window mean); 5+-way = pooled focal junctions; typicality vs 156 real neighbourhoods (1 = as typical as a real place).", "",
          "| Method | Version | close15 | 5+-way | 4-way | junctions / km² | typicality | street km | unreachable across windows |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L_.append(f"| {r['method']} | {r['version']} | {r['close15']:.3f} | {r['share5p']:.1%} | {r['share4']:.1%} | "
                  f"{r['junctions_km2']:.0f} | {r['typicality']:.2f} | {r['street_km']:.1f} | {r['xwin_unreach']:.2f} |")
    (OUT / "m17_kaist.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))
    fig, ax = plt.subplots(len(METHODS), 3, figsize=(15, 5.1 * len(METHODS)))
    for i, lab in enumerate(METHODS):
        for j, (ver, g) in enumerate((("real KAIST (vehicular)", real_g), ("original", graphs[(lab, "original", 1)]),
                                      ("improved (F1 + F4)", graphs[(lab, "improved (F1 + F4)", 1)]))):
            draw(ax[i, j], big, g, size=L, title=f"{lab} — {ver}" if j else ver, scalebar=False,
                 note=f"{measures(g, np.random.default_rng(0))['street_km']:.1f} km")
            n, e = g.nodes, g.edges
            deg = np.bincount(e.ravel(), minlength=len(n)) if len(e) else np.zeros(len(n), int)
            hi = np.flatnonzero(deg >= 5)
            ax[i, j].scatter(n[hi, 0], n[hi, 1], marker="s", s=22, facecolors="none", edgecolors="k", linewidths=1.2, zorder=7)
    fig.suptitle("KAIST campus (1536 m, seed 1): original vs improved junctions; black squares = 5+ branch junctions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "outputs" / "m17" / "figures" / f"fig_m17_kaist_maps.{ext}", dpi=80)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.one:
        generate_one({key_of(x): x for x in METHODS}[a.one[0]], int(a.one[1]))
    if a.assemble:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
