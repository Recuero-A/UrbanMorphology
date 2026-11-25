"""Milestone 16 (exploratory demo): every milestone's method on the whole KAIST campus (1536 m domain, GLO-30 DSM).

Methods (seeds 1-3): M12-HR (the M12 loop with hard grade rules, historical windowed growth, no anchors), P (rules
only), and the learned samplers with the SAME supplied anchors (real OSM arterials) on one global state: A, M*,
M* TNse, M13-SMC, NA-myopic, NA-fixed, NA-adaptive, NA-random, NA-SMC, and M* + grid / organic / radial / cul-de-sac
proposals + topology (the first eight already exist in outputs/m16/kaist/styles_cache and are reused). M*+repair is
left out: graphs/repair.py assumes a single 512 m tile.

Realism, two ways (neither is optimised by any method):
* KAIST distance: per window, |generated - real KAIST (all vehicular roads)| of the independent outcomes, each scaled
  by the pooled within-style SD of the 36 real style references, averaged over features and the 9 windows (lower =
  closer to this campus);
* general realism: precision / coverage of the generated windows against the 120 real M15 pilot windows on the M15
  Task A features (block areas, junction shares, circuity), with the real-vs-real benchmark for scale.

    python scripts/m16_kaist_all.py --one KEY SEED     # generate one cached graph (parallel-safe)
    python scripts/m16_kaist_all.py --assemble         # score, tables, 2D sheet -> outputs/m16/kaist/all.{md,json}
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
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
import pandas as pd
import torch

from m16_kaist_campus import L, OUT, PACK, W, AnchoredGlobal, anchors, domain, measures, merge, snap
from m16_kaist_styles import CACHE as STYLE_CACHE
from m16_kaist_styles import score, windows_of, window_terrain
from urbanmorph.experiments.backbones import load_backbone
from urbanmorph.experiments.methods import method
from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
from urbanmorph.graphs.network_outcomes import outcome_row
from urbanmorph.graphs.style_metrics import morphology_row
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.metrics.fidelity import prdc, real_vs_real
from urbanmorph.metrics.recognise import fit
from urbanmorph.visualization.m16_render import draw

CACHE = OUT / "all_cache"
REUSE = {"P": 0, "A": 1, "M*": 2, "M* TNse": 3, "M* + grid": 4, "M* + organic": 5, "M* + radial": 6, "M* + culdesac": 7}
ORDER = ["M12-HR", "P", "A", "M*", "M* TNse", "M13-SMC", "NA-myopic", "NA-fixed", "NA-adaptive", "NA-random", "NA-SMC",
         "M* + grid", "M* + organic", "M* + radial", "M* + culdesac"]
MILESTONE = {"M12-HR": "M12", "P": "M12", "A": "M13", "M*": "M13", "M* TNse": "M13 (TNse)", "M13-SMC": "M13",
             "NA-myopic": "M14", "NA-fixed": "M14", "NA-adaptive": "M14", "NA-random": "M14", "NA-SMC": "M14",
             "M* + grid": "M16", "M* + organic": "M16", "M* + radial": "M16", "M* + culdesac": "M16"}
OUTCOMES = ["share_4way", "share_3way", "deadend_per_km", "block_rect_share", "block_phi_med", "block_logA_med",
            "block_logA_sd", "jangle_right_share", "stroke_xjunc_lw", "tree_len_share", "phi_boeing", "circuity_200_350",
            "street_km"]
M15_FEATS = ("block_area_med", "block_area_p90", "share_3way", "share_4way", "circuity_100_200", "circuity_200_350")


def key_of(lab: str) -> str:
    return lab.replace("*", "star").replace(" + ", "_").replace(" ", "_")


class AnchoredGlobalNet(AnchoredGlobal, NetSampler):
    """The M14 network-aware sampler on the global anchored domain (same MRO hooks: global starts, anchors, NA weights)."""


def generate_one(lab: str, seed: int):
    f = CACHE / f"{key_of(lab)}_s{seed}.pkl"
    if f.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    if lab in REUSE:
        shutil.copy(STYLE_CACHE / f"v{REUSE[lab]}_s{seed}.pkl", f)
        return
    big, (x0, y0), crs = domain()
    t0 = time.time()
    if lab == "M12-HR":
        from m12_big_landscape import decode, grow
        from m11_big_city import to_graph
        from m12_evaluate import load_model
        from urbanmorph.experiments.references import TH
        from urbanmorph.terrain.heightfield import Terrain
        T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -256.0, L + 256.0, 4.0)
        T.water_kind = big["water"].astype(np.int8)
        T.water_surface = np.where(big["water"] > 0, big["elev"] - 1.0, np.nan).astype(np.float32)
        model, names, norm = load_model("TN_s0", "cpu")
        S, C, _ = grow(model, names, norm, "HR", TH, T, L, seed, 0.5)
        g = decode(to_graph(S, C, (0.0, 0.0)))
    else:
        m = method(lab)
        model, names, norm, _ = load_backbone("TN_s0", "cpu")
        pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
        is_net = lab.startswith("NA")
        cfg = net_preset(m.preset, **m.overrides) if is_net else preset(m.preset, **m.overrides)
        sm = (AnchoredGlobalNet if is_net else AnchoredGlobal)(model, {**big, "site": f"KAIST_CAMPUS_{lab}"}, L, names,
                                                               norm, pots, cfg)
        sm.anchor = anchors(x0, y0, crs)
        gs = sm.run(seed)
        g = gs[0]
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"label": lab, "seed": seed, "graph": g, "seconds": time.time() - t0}))
    tmp.rename(f)
    print(f"done {lab} seed {seed} {time.time() - t0:.0f}s", flush=True)


def window_rows(g, big, rng):
    rows = []
    for (r, c), wg in windows_of(g):
        if wg is None or len(wg.edges) == 0:
            rows.append(None)
            continue
        t = terrain_of(window_terrain(big, r, c))
        rows.append({**morphology_row(wg, t, centre=(256.0, 256.0), rng=rng, grade_span=30.0), **outcome_row(wg)})
    return rows


def assemble(seeds):
    big, _, _ = domain()
    refs = pd.read_csv(ROOT / "configs" / "m16_style_refs" / "metrics.csv")
    sd = {}
    for k in OUTCOMES:                                         # pooled within-style SD of the real references
        v = refs[[k, "style"]].dropna()
        res = np.concatenate([g[k].values - g[k].mean() for _, g in v.groupby("style")])
        sd[k] = float(np.sqrt((res ** 2).sum() / max(len(res) - v["style"].nunique(), 1))) or 1.0
    cal = refs[refs.subset == "calib"]
    clf = fit(cal.to_dict("records"), cal["style"].tolist())
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    real_g = snap(merge([(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}__vehicular.pkl").read_bytes()), off)
                         for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]))
    real_rows = window_rows(real_g, big, np.random.default_rng(0))
    m15 = json.loads((ROOT / "outputs" / "m15" / "real_rows.json").read_text())
    m15_rows = m15 if isinstance(m15, list) else m15.get("rows", [])
    feat = lambda r: [np.log(r[k]) if k.startswith("block_area") and r.get(k) and r[k] > 0 else
                      (r.get(k) if r.get(k) is not None else np.nan) for k in M15_FEATS]
    R = np.array([feat(r) for r in m15_rows], float)
    rvr = real_vs_real(R, np.array([r.get("metro", r.get("sid", "x")) for r in m15_rows]))
    out, graphs = [], {}
    for lab in ORDER:
        for seed in seeds:
            f = CACHE / f"{key_of(lab)}_s{seed}.pkl"
            if not f.exists():
                continue
            c = pickle.loads(f.read_bytes())
            g = c["graph"]
            graphs[(lab, seed)] = g
            rng = np.random.default_rng([seed, 16])
            s = score(g, big, clf, rng)
            rows = window_rows(g, big, rng)
            dist = [np.nanmean([abs(a[k] - b[k]) / sd[k] for k in OUTCOMES
                                if a.get(k) is not None and b.get(k) is not None and np.isfinite(a[k]) and np.isfinite(b[k])])
                    for a, b in zip(rows, real_rows) if a is not None and b is not None]
            Gx = np.array([feat(r) for r in rows if r is not None], float)
            pr = prdc(R, Gx) if len(Gx) else {}
            out.append({"method": lab, "milestone": MILESTONE[lab], "seed": seed, "seconds": c["seconds"],
                        "kaist_distance": float(np.nanmean(dist)) if dist else np.nan,
                        "precision": pr.get("precision", np.nan), "coverage_m15": pr.get("coverage", np.nan),
                        "windows_with_streets": int(sum(r is not None for r in rows)), **s})
            print(lab, seed, f"dist {out[-1]['kaist_distance']:.2f} prec {out[-1]['precision']:.2f}", flush=True)
    real_s = score(real_g, big, clf, np.random.default_rng(0))
    Gx = np.array([feat(r) for r in real_rows if r is not None], float)
    pr = prdc(R, Gx)
    out.append({"method": "real KAIST (vehicular)", "milestone": "reference", "seed": 0, "seconds": np.nan,
                "kaist_distance": 0.0, "precision": pr["precision"], "coverage_m15": pr["coverage"],
                "windows_with_streets": 9, **real_s})
    json.dump({"rows": out, "real_vs_real_m15": rvr}, open(OUT / "all.json", "w"), indent=1, default=float)
    (OUT / "all_graphs.pkl").write_bytes(pickle.dumps({"graphs": graphs, "real": real_g}))
    df = pd.DataFrame(out)
    cols = ["kaist_distance", "precision", "street_km", "coverage_60", "xwin_unreach", "share_4way", "deadend_per_km",
            "loops_per_km2", "block_rect_share", "jangle_right_share", "stroke_xjunc_lw", "tree_len_share", "share_gt15",
            "seconds"]
    tab = df.groupby(["method", "milestone"], sort=False)[cols].mean().reset_index()
    rec = df.groupby("method", sort=False)["recognised_as"].apply(
        lambda s: ", ".join(f"{k} {v:.0%}" for k, v in sorted(pd.DataFrame(list(s)).fillna(0).mean().items(),
                                                              key=lambda kv: -kv[1]) if v > 0))
    tab["classifier"] = tab["method"].map(rec)
    tab = pd.concat([tab[tab.milestone != "reference"].sort_values("kaist_distance"), tab[tab.milestone == "reference"]])
    md = ["# Every milestone's method on the whole KAIST campus (EXPLORATORY; GLO-30 DSM; 1536 m; seeds " + str(seeds) + ")", "",
          "Sorted by **KAIST distance** (mean scaled difference from the real campus roads, window by window, on the "
          "independent outcomes; lower = closer). **Precision** = share of generated windows that fall inside the "
          f"manifold of 120 real M15 windows (real-vs-real benchmark {rvr.get('precision', {}).get('mean', float('nan')) if isinstance(rvr.get('precision'), dict) else rvr.get('precision', float('nan'))}).", "",
          "| Method | Milestone | " + " | ".join(cols) + " | classifier says |", "|---|---|" + "---|" * (len(cols) + 1)]
    for _, r in tab.iterrows():
        md.append(f"| {r['method']} | {r['milestone']} | " + " | ".join(
            "–" if not np.isfinite(r[k]) else (f"{r[k]:.0f}" if k == "seconds" else f"{r[k]:.2f}") for k in cols)
                  + f" | {r['classifier'] if isinstance(r['classifier'], str) else ''} |")
    (OUT / "all.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    s0 = seeds[0]
    panels = [(f"{lab} ({MILESTONE[lab]})", graphs.get((lab, s0))) for lab in ORDER] + [("real KAIST (vehicular)", real_g)]
    fig, ax = plt.subplots(4, 4, figsize=(19, 19.6))
    for a, (lab, g) in zip(ax.ravel(), panels):
        draw(a, big, g, size=L, title=lab, note=(f"{measures(g, np.random.default_rng(0))['street_km']:.1f} km" if g is not None else "missing"))
    fig.suptitle(f"KAIST campus, every milestone's method (seed {s0}); learned methods start from the same real arterials", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "outputs" / "m16" / "figures" / f"fig_kaist_all_methods.{ext}", dpi=100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2, metavar=("KEY", "SEED"))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.one:
        lab = {key_of(x): x for x in ORDER}[a.one[0]]
        generate_one(lab, int(a.one[1]))
    if a.assemble:
        assemble(a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
