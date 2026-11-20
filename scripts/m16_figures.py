"""Milestone 16 figures from stored graphs and terrain only (docs/M16_PROTOCOL.md §8).

    python scripts/m16_figures.py [--only paired|named|kaist|replay]   ->  outputs/m16/figures/*.{png,pdf} + *_runs.json

* ``paired``  per style: rows = the six confirmatory conditions (syn_conf), columns = M13 (M* off), M14 (NA-fixed
              off), M15 (NA-fixed rerank), M16 (M* proposals + topology), M16 (NA-fixed proposals + topology); one sheet
              per predefined seed (301, 302) and one "weak runs" sheet: for each condition the seed whose NA-fixed
              proposals + topology network is shortest (empty networks first) - a fixed rule, not a choice.
* ``named``   the named real sites (GLO-30 and MDT05 rows), off / rerank / proposals + topology per style, seed 1.
* ``kaist``   the campus-wide pilot: plan views per approach (seed 1) and the real references.
* ``replay``  M12-HR vs M* vs NA-fixed on the demo landscapes (historical configurations, seed 101).
Every panel uses the same extent (the 512 m tile or the 1536 m domain), hillshade, water and road styling
(visualization/m16_render.py). The run ids behind every panel are written next to each figure.
"""
from __future__ import annotations

import argparse
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

from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.runner import load_run, run_id
from urbanmorph.experiments.sites import site_set
from urbanmorph.visualization.m16_render import draw

RUNS = ROOT / "outputs" / "m16" / "runs"
FIG = ROOT / "outputs" / "m16" / "figures"
STYLES = ("grid", "organic", "radial", "culdesac")
CMD = "python scripts/m16_figures.py"


def save(fig, name: str, ids: dict):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=130)
    fig.savefig(FIG / f"{name}.pdf")
    (FIG / f"{name}_runs.json").write_text(json.dumps({"render_command": CMD, "panels": ids}, indent=1))
    plt.close(fig)
    print("wrote", name)


def graph_of(spec, site: str, seed: int):
    r = load_run(spec, RUNS)
    if r is None or str(seed) not in r["seeds"]:
        return None, None, None
    v = r["seeds"][str(seed)]
    k = [x["site"] for x in v["rows"]].index(site)
    return graph_from_json(v["graphs"][k]), v["rows"][k], r["id"]


def paired():
    from m16_plans import CONF_SEEDS, proposals
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    sites = site_set("syn_conf")
    base = {"sites": "syn_conf", "backbone": "TN_s0", "start": "centre", "seeds": CONF_SEEDS}
    for st in STYLES:
        cols = [("M13: M* off", {**base, "method": "M*", "style": "off"}),
                ("M14: NA-fixed off", {**base, "method": "NA-fixed", "style": "off"}),
                ("M15: NA-fixed rerank", {**base, "method": "NA-fixed", "style": st}),
                ("M16: M* proposals+topology", {**base, "method": "M*", "style": st,
                                                "proposals": proposals("style_topo", fz["M*"][st]["style_topo"]["prop_alpha"])}),
                ("M16: NA-fixed proposals+topology", {**base, "method": "NA-fixed", "style": st,
                                                      "proposals": proposals("style_topo", fz["NA-fixed"][st]["style_topo"]["prop_alpha"])})]
        weak = {}
        r = load_run(cols[-1][1], RUNS)
        if r is not None:
            for d in sites:
                cand = [(x["empty"] == 0, x["street_km"], int(sd)) for sd, v in r["seeds"].items() for x in v["rows"]
                        if x["site"] == d["site"]]
                weak[d["site"]] = min(cand)[2]
        for sheet, seed_of in (("s301", lambda s: 301), ("s302", lambda s: 302), ("weak", lambda s: weak.get(s, 301))):
            fig, ax = plt.subplots(len(sites), len(cols), figsize=(2.3 * len(cols), 2.3 * len(sites)))
            ids = {}
            for i, d in enumerate(sites):
                sd = seed_of(d["site"])
                for j, (lab, spec) in enumerate(cols):
                    g, row, rid = graph_of(spec, d["site"], sd)
                    note = (f"{row['street_km']:.1f} km" if row else "missing")
                    draw(ax[i, j], d, g, title=(lab if i == 0 else "") , note=f"{d['site'][5:]} s{sd} · {note}",
                         scalebar=(i == 0 and j == 0))
                    ids[f"{d['site']}|{lab}|{sd}"] = rid
            fig.suptitle(f"M16 confirmatory sites, style = {st} ({sheet}); same seeds per row; DEVELOPMENT-FROZEN settings",
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            save(fig, f"fig_paired_{st}_{sheet}", ids)


def named():
    from m16_plans import proposals
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    for m in ("M*", "NA-fixed"):
        rows = []
        for sites in ("named_glo30", "named_mdt05"):
            for d in site_set(sites):
                rows.append((sites, d))
        cols = [("off", None, None)] + [(f"{st} rerank", st, None) for st in STYLES] + \
               [(f"{st} prop+topo", st, "topo") for st in STYLES]
        fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.1 * len(rows)))
        ids = {}
        for i, (sites, d) in enumerate(rows):
            base = {"sites": sites, "backbone": "TN_s0", "method": m, "start": "centre", "seeds": [1, 2, 3, 4, 5],
                    "grade_span": 30.0 if sites == "named_glo30" else 10.0}
            for j, (lab, st, mode) in enumerate(cols):
                spec = {**base, "style": st or "off"}
                if mode:
                    spec["proposals"] = proposals("style_topo", fz[m][st]["style_topo"]["prop_alpha"])
                g, row, rid = graph_of(spec, d["site"], 1)
                draw(ax[i, j], d, g, title=lab if i == 0 else "", note=f"{d['site'][6:]} · "
                     + (f"{row['street_km']:.1f} km" if row else "missing"), scalebar=(i == 0 and j == 0))
                ids[f"{d['site']}|{lab}"] = rid
        fig.suptitle(f"M16 named real sites, {m}, seed 1 (recurring comparison; GLO-30 = DSM, MDT05 = DTM)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, f"fig_named_{m.replace('*', 'star')}", ids)


def kaist():
    from m16_kaist_campus import L, OUT, PACK, W, merge, snap, domain
    gp = pickle.loads((OUT / "graphs.pkl").read_bytes())
    big, _, _ = domain()
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    real = {}
    for name, suffix in (("real streets", ""), ("real vehicular (+service)", "__vehicular"),
                         ("real pedestrian", "__pedestrian")):
        gs = [(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}{suffix}.pkl").read_bytes()), off)
              for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]
        real[name] = snap(merge(gs))
    panels = [(lab, gp["graphs"][(lab, 1)]) for lab in ("global M*", "anchored M*", "tiles M*", "tiles NA-fixed")] + \
             [("supplied anchors", gp["anchors"])] + list(real.items())
    fig, ax = plt.subplots(2, 4, figsize=(16, 9.0))
    for a, (lab, g) in zip(ax.ravel(), panels):
        draw(a, big, g, size=L, title=lab)
        for c in (W, 2 * W):
            a.axvline(c, color="#2e86c1", lw=0.4, ls=":")
            a.axhline(c, color="#2e86c1", lw=0.4, ls=":")
    fig.suptitle("KAIST main campus, 1536 m domain (GLO-30 DSM hillshade; dotted: 512 m window edges); seed 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_kaist_campus", {"source": "outputs/m16/kaist/graphs.pkl", "seed": 1})


def kaist3d():
    """Terrain-draped 3D views of the campus domain (GLO-30 DSM, vertical exaggeration 2): the anchored M* run and the
    real vehicular network, seed 1. Streets are draped on the bilinear surface and lifted 1 m."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from urbanmorph.terrain.heightfield import Terrain
    from m16_kaist_campus import L, OUT, PACK, W, merge, snap, domain
    gp = pickle.loads((OUT / "graphs.pkl").read_bytes())
    big, _, _ = domain()
    t = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -256.0, L + 256.0, 4.0)
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    gs = [(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}__vehicular.pkl").read_bytes()), off)
          for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]
    panels = [("anchored M* (generated, seed 1)", gp["graphs"][("anchored M*", 1)]), ("real vehicular (+ service)", snap(merge(gs)))]
    k = 4                                                          # 16 m surface mesh
    n0 = int(256 / 4)
    Z = big["elev"][n0:n0 + int(L / 4):k, n0:n0 + int(L / 4):k]
    xs = (np.arange(Z.shape[1]) * k + 0.5) * 4.0
    ys = L - (np.arange(Z.shape[0]) * k + 0.5) * 4.0
    X, Y = np.meshgrid(xs, ys)
    from urbanmorph.visualization.m16_render import hillshade
    hs = hillshade(Z, res=16.0, vert=2.0)
    fig = plt.figure(figsize=(16, 7.5))
    for j, (lab, g) in enumerate(panels):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d", computed_zorder=False)
        ax.plot_surface(X, Y, Z * 2.0, facecolors=plt.cm.gray(0.35 + 0.6 * hs), rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False, zorder=0)
        for a_, b_ in g.edges:
            p = g.nodes[[a_, b_]]
            f = np.linspace(0, 1, 6)
            q = p[0] + f[:, None] * (p[1] - p[0])
            z = t.sample(q[:, 0], q[:, 1]) * 2.0 + 2.0
            ax.plot(q[:, 0], q[:, 1], z, color="#b03a2e", lw=0.7, zorder=5)
        ax.set_title(lab, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=38, azim=-60)
        ax.set_box_aspect((1, 1, 0.25))
    fig.suptitle("KAIST main campus, 1536 m domain, terrain-draped (GLO-30 DSM, vertical exaggeration 2)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_kaist_campus_3d", {"source": "outputs/m16/kaist/graphs.pkl", "seed": 1})


def replay():
    sites = site_set("demo_scen")
    cols = [("M12-HR (M12 demos)", {"method": "M12-HR"}), ("M12 loop, no rules", {"method": "M12-loop"}),
            ("M* (M13)", {"method": "M*"}), ("NA-fixed (M14)", {"method": "NA-fixed"})]
    fig, ax = plt.subplots(len(sites), len(cols), figsize=(2.4 * len(cols), 2.4 * len(sites)))
    ids = {}
    for j, (lab, kw) in enumerate(cols):
        spec = {"sites": "demo_scen", "backbone": "TN_s0", "style": "off", "seeds": [101, 102, 103, 104, 105], **kw}
        for i, d in enumerate(sites):
            g, row, rid = graph_of(spec, d["site"], 101)
            draw(ax[i, j], d, g, title=lab if i == 0 else "", scalebar=(i == 0 and j == 0),
                 note=f"{d['site'][5:]} · >15 %: {row['share_gt15']:.2f}" if row else "missing")
            ids[f"{d['site']}|{lab}"] = rid
    fig.suptitle("Historical configurations on the demo landscapes (learned start, seed 101): steep share > 15 %", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_replay_demo", ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["paired", "named", "kaist", "kaist3d", "replay"])
    a = ap.parse_args()
    for name, f in (("replay", replay), ("kaist", kaist), ("kaist3d", kaist3d), ("paired", paired), ("named", named)):
        if a.only in (None, name):
            f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
