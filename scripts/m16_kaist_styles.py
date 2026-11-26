"""Milestone 16 (exploratory demo, recurring comparison site): different cities on the KAIST campus domain.

One change at a time, all with the SAME supplied anchors (the real OSM arterials of the domain), GLO-30 terrain,
seeds and evaluator:
  P (procedural, per 512 m window) -> A (learned turtle, unconstrained) -> M* (+ terrain rules and potentials)
  -> M* with the street-erased backbone TNse (terrain input) and M* + style proposals + topology rules
  (grid / organic / radial / cul-de-sac; alpha frozen in configs/m16_frozen.json; lattice and radial centre = domain
  centre). Real references: the KAIST public streets (training definition) and all vehicular roads (+ service).
Scoring: the domain graph is cut into its 3 x 3 windows of 512 m (lines clipped at the window and rebuilt with
build_graph, the same processing as the real references), each window is scored with the M16 outcomes, the M15
morphology / terrain metrics and the style classifier (real calib references), and window scores are averaged.
Domain-level: street km, 60 m coverage, cross-window OD reachability, seam dead-end ratio.

    python scripts/m16_kaist_styles.py [--seeds 1 2 3]  ->  outputs/m16/kaist/styles.{json,md}, figures/fig_kaist_styles*.png
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
import pandas as pd
import shapely
import torch

from m16_kaist_campus import L, OUT, PACK, W, AnchoredGlobal, anchors, domain, measures, merge, snap
from m16_plans import proposals
from urbanmorph.experiments.backbones import load_backbone
from urbanmorph.experiments.methods import method
from urbanmorph.experiments.references import TH, median_density
from urbanmorph.graphs.network_outcomes import outcome_row
from urbanmorph.graphs.procedural_terrain import sample_P
from urbanmorph.graphs.road_graph import build_graph
from urbanmorph.graphs.style_metrics import morphology_row
from urbanmorph.graphs.tc_global import window_tiles
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import preset
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.metrics.recognise import fit
from urbanmorph.visualization.m16_render import draw

FZ = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
STYLES = ("grid", "organic", "radial", "culdesac")
KEYS = ["street_km", "coverage_60", "xwin_unreach", "seam_dead_ratio", "share_4way", "share_3way", "deadend_per_km",
        "loops_per_km2", "block_rect_share", "block_phi_med", "block_logA_med", "jangle_right_share", "stroke_xjunc_lw",
        "tree_len_share", "phi_boeing", "circuity_200_350", "share_gt15", "share_gt10", "recognised_as"]


def variants():
    out = [("P (rules only)", None), ("A (learned)", ("TN_s0", "A", "off", None)),
           ("M* (+terrain rules)", ("TN_s0", "M*", "off", None)), ("M* TNse (terrain input)", ("TNse_s0", "M*", "off", None))]
    for st in STYLES:
        out.append((f"M* + {st}", ("TN_s0", "M*", st, proposals("style_topo", FZ["M*"][st]["style_topo"]["prop_alpha"]))))
    return out


def windows_of(g, crs_off=None):
    """The domain graph cut into its 9 windows (tile-local coordinates), rebuilt like the real references."""
    out = []
    for r in range(3):
        for c in range(3):
            x0, y0 = c * W, r * W
            box = shapely.box(x0, y0, x0 + W, y0 + W)
            lines = []
            for a, b in g.edges:
                seg = shapely.LineString(g.nodes[[a, b]]).intersection(box)
                for part in getattr(seg, "geoms", [seg]):
                    if part.geom_type == "LineString" and part.length > 0.5:
                        lines.append(np.asarray(part.coords) - [x0, y0])
            out.append(((r, c), build_graph(lines) if lines else None))
    return out


def window_terrain(big, r, c):
    n0 = int((L - (r + 1) * W) / 4)
    sl = (slice(n0, n0 + 256), slice(int(c * W / 4), int(c * W / 4) + 256))
    return {k: v[sl] for k, v in big.items()}


def score(g, big, clf, rng) -> dict:
    dom = measures(g, rng)
    rows = []
    for (r, c), wg in windows_of(g):
        if wg is None or len(wg.edges) == 0:
            continue
        t = terrain_of(window_terrain(big, r, c))
        row = {**morphology_row(wg, t, centre=(256.0, 256.0), rng=rng, grade_span=30.0), **outcome_row(wg)}
        row["recognised_as"] = clf.predict([row])[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = {k: dom[k] for k in ("street_km", "coverage_60", "xwin_unreach", "seam_dead_ratio")}
    for k in KEYS[4:-1]:
        out[k] = float(np.nanmean(df[k])) if k in df and df[k].notna().any() else np.nan
    out["recognised_as"] = df["recognised_as"].value_counts(normalize=True).to_dict() if len(df) else {}
    return out


CACHE = OUT / "styles_cache"


def generate_one(k: int, seed: int) -> None:
    """Generate variant k for one seed and cache the graph (resumable, parallel-safe)."""
    lab, spec = variants()[k]
    f = CACHE / f"v{k}_s{seed}.pkl"
    if f.exists():
        return
    big, (x0, y0), crs = domain()
    anc = anchors(x0, y0, crs)
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    t0 = time.time()
    if spec is None:
        tiles = window_tiles(big, L)
        gs = [sample_P(terrain_of(t), np.random.default_rng([seed, j]), median_density(), TH) for j, t in enumerate(tiles)]
        g = merge(list(zip(gs, offs)))
    else:
        bb, m, st, prop = spec
        model, names, norm, _ = load_backbone(bb, "cpu")
        pots = Potentials.load(ROOT / "configs" / "m13_potentials" / f"{bb.split('_')[0]}.json")
        ov = dict(method(m).overrides)
        if st != "off":
            ov.update(json.loads((ROOT / "configs" / "m15_style.json").read_text())["styles"][st])
            ov.update(prop)
            ov.update(style_cx=L / 2, style_cy=L / 2)
        sm = AnchoredGlobal(model, {**big, "site": f"KAIST_CAMPUS_{lab}"}, L, names, norm, pots, preset(method(m).preset, **ov))
        sm.anchor = anc
        g = sm.run(seed)[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"label": lab, "seed": seed, "graph": g, "seconds": time.time() - t0}))
    tmp.rename(f)
    print(f"done {lab} seed {seed} {time.time() - t0:.0f}s", flush=True)


def fig3d(panels, big, exag: float, name: str):
    """Terrain-draped 3D views (GLO-30 DSM), vertical exaggeration ``exag`` (1 = true elevation)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from urbanmorph.terrain.heightfield import Terrain
    from urbanmorph.visualization.m16_render import hillshade
    t = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -256.0, L + 256.0, 4.0)
    k, n0 = 4, 64
    Z = big["elev"][n0:n0 + int(L / 4):k, n0:n0 + int(L / 4):k].astype(float)
    xs = (np.arange(Z.shape[1]) * k + 0.5) * 4.0
    ys = L - (np.arange(Z.shape[0]) * k + 0.5) * 4.0
    X, Y = np.meshgrid(xs, ys)
    z0 = float(np.nanmin(Z))
    fc = plt.cm.gray(0.35 + 0.6 * hillshade(Z, res=16.0, vert=max(exag, 1.0)))
    fig = plt.figure(figsize=(22, 9.5))
    for j, (lab, g) in enumerate(panels):
        ax = fig.add_subplot(2, 5, j + 1, projection="3d", computed_zorder=False)
        ax.plot_surface(X, Y, (Z - z0) * exag, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, zorder=0)
        for a_, b_ in g.edges:
            p = g.nodes[[a_, b_]]
            f = np.linspace(0, 1, 5)
            q = p[0] + f[:, None] * (p[1] - p[0])
            ax.plot(q[:, 0], q[:, 1], (t.sample(q[:, 0], q[:, 1]) - z0) * exag + 1.5, color="#b03a2e", lw=0.5, zorder=5)
        ax.set_title(lab, fontsize=9)
        ax.set_axis_off()
        ax.view_init(elev=35, azim=-60)
        ax.set_box_aspect((1, 1, max(0.08, 0.06 * exag)))
    fig.suptitle(f"KAIST campus, 1536 m, terrain-draped (GLO-30 DSM, relief ~55 m), vertical exaggeration x{exag:g}",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "outputs" / "m16" / "figures" / f"{name}.{ext}", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--one", type=int, nargs=2, metavar=("VARIANT", "SEED"), help="generate one cached graph and exit")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.one:
        generate_one(*a.one)
        return 0
    big, (x0, y0), crs = domain()
    anc = anchors(x0, y0, crs)
    refs = pd.read_csv(ROOT / "configs" / "m16_style_refs" / "metrics.csv")
    cal = refs[refs.subset == "calib"]
    clf = fit(cal.to_dict("records"), cal["style"].tolist())
    res, graphs = [], {}
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    for k, (lab, spec) in enumerate(variants()):
        for seed in a.seeds:
            c = pickle.loads((CACHE / f"v{k}_s{seed}.pkl").read_bytes())
            g, sec = c["graph"], c["seconds"]
            rng = np.random.default_rng([seed, 16])
            graphs[(lab, seed)] = g
            res.append({"variant": lab, "seed": seed, "seconds": sec, **score(g, big, clf, rng)})
            print(lab, seed, "scored", flush=True)
    for lab, suffix in (("real KAIST: public streets", ""), ("real KAIST: vehicular (+service)", "__vehicular")):
        gs = [(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}{suffix}.pkl").read_bytes()), off)
              for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]
        g = snap(merge(gs))
        graphs[(lab, 0)] = g
        res.append({"variant": lab, "seed": 0, "seconds": np.nan, **score(g, big, clf, np.random.default_rng(0))})
    (OUT / "styles_graphs.pkl").write_bytes(pickle.dumps({"graphs": graphs, "anchors": anc}))
    json.dump(res, open(OUT / "styles.json", "w"), indent=1, default=float)
    # table
    df = pd.DataFrame(res)
    num = [k for k in KEYS if k != "recognised_as"]
    tab = df.groupby("variant", sort=False)[num].mean()
    rec = df.groupby("variant", sort=False)["recognised_as"].apply(
        lambda s: ", ".join(f"{k} {v:.0%}" for k, v in sorted(pd.DataFrame(list(s)).fillna(0).mean().items(), key=lambda kv: -kv[1]) if v > 0))
    md = ["# KAIST campus: different cities, one change at a time (EXPLORATORY demo; GLO-30 DSM; anchors = real arterials)", "",
          f"Seeds {a.seeds}; window scores = mean over the 3 x 3 windows of 512 m; > 15 % / > 10 % over 30 m sub-spans "
          "(the GLO-30 resolution). Style classifier fitted on the real calib references.", "",
          "| Variant | " + " | ".join(num) + " | classifier says |", "|---|" + "---|" * (len(num) + 1)]
    for v, r in tab.iterrows():
        md.append(f"| {v} | " + " | ".join("–" if not np.isfinite(r[k]) else f"{r[k]:.2f}" for k in num) + f" | {rec[v]} |")
    (OUT / "styles.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    # figure: one panel per variant (seed 1) + the two real references
    panels = [(lab, graphs[(lab, a.seeds[0])]) for lab, _ in variants()] + \
             [(lab, graphs[(lab, 0)]) for lab in ("real KAIST: public streets", "real KAIST: vehicular (+service)")]
    fig, ax = plt.subplots(2, 5, figsize=(20, 8.6))
    for axx, (lab, g) in zip(ax.ravel(), panels):
        draw(axx, big, g, size=L, title=lab, note=f"{measures(g, np.random.default_rng(0))['street_km']:.1f} km")
    fig.suptitle(f"KAIST main campus (1536 m, GLO-30 DSM hillshade), seed {a.seeds[0]}; every generated city starts from the "
                 "same real arterials (anchors)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (ROOT / "outputs" / "m16" / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "outputs" / "m16" / "figures" / f"fig_kaist_styles.{ext}", dpi=120)
    plt.close(fig)
    for ex in (1.0, 3.0):
        fig3d(panels, big, ex, f"fig_kaist_styles_3d_x{ex:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
