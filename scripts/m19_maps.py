"""Milestone 19, D5 deliverable: matched maps of a representative success and failure of online routing (A1) against
the same router after growth (A5), on the same unit, backbone and seed, with A0 for reference.

Tile scale, chosen by rule: among the unit-runs, d = A1's served pairs - A5's served pairs at A1's street length (per
pair); the success is the unit-run at the median of the positive d, the failure at the median of the negative d.
Whole towns: Petra at the seed with the largest A5 - A0 gain, and Machu Picchu at the seed with the largest A1 - A0 gain.
Steep endpoints are drawn as reached (a street within 60 m) or not; A1's router edges and A5's connectors are marked.

    python scripts/m19_maps.py  ->  outputs/m19/arms/fig_maps_tiles.png, fig_maps_towns.png
"""
from __future__ import annotations

import pickle
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m19_arms import OUT, run_file
from m19_eval import MEAS, a5_at, fields, od_points

INK, SURF = "#0b0b0b", "#fcfcfb"
STREET, ROUTER, CONN, OK_, NO_ = "#3a3a38", "#2a78d6", "#eb6834", "#1baf7a", "#c0392b"


def meas(unit, arm, bb, seed):
    return pickle.loads(zlib.decompress((MEAS / f"{unit}__{arm}__{bb}__s{seed}.pkl.gz").read_bytes()))


def runs(unit, arm, bb, seed):
    return pickle.loads(zlib.decompress(run_file(unit, arm, bb, seed).read_bytes()))


def pick_tiles():
    from m19_arms import BBS, TILE_SEEDS
    rows = []
    for unit in ("gen2", "natural"):
        for bb in BBS:
            for s in TILE_SEEDS:
                m0, m1 = meas(unit, "A0", bb, s), meas(unit, "A1", bb, s)
                for r0, r1 in zip(m0["rows"], m1["rows"]):
                    if r0["pairs"] < 50:
                        continue
                    d = (r1["out"][0] - a5_at(r0["A5"], r1["length"])[0]) / r0["pairs"]
                    rows.append((d, unit, bb, s, r0["uid"]))
    pos = sorted([r for r in rows if r[0] > 0])
    neg = sorted([r for r in rows if r[0] < 0])
    return pos[len(pos) // 2], neg[len(neg) // 2], len(pos), len(neg), len(rows)


def panel(ax, F, g, pts_steep, reached, extra=None, extra_col=None, title="", clip=None):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource
    z = np.asarray(F.S.elev, float)
    hs = LightSource(315, 40).hillshade(z, vert_exag=2.0, dx=F.S.res, dy=F.S.res)
    rgb = np.dstack([0.62 + 0.38 * hs] * 3)
    steep = (F.slope >= 15) & (F.slope < 35) & F.land
    rgb[steep] = rgb[steep] * 0.55 + np.array([0.93, 0.80, 0.62]) * 0.45
    rgb[~F.land] = rgb[~F.land] * 0.85 + 0.15
    H, W = z.shape
    ext = (F.S.x0, F.S.x0 + W * F.S.res, F.S.y1 - H * F.S.res, F.S.y1)
    ax.imshow(rgb, extent=ext)
    if len(g.edges):
        ax.add_collection(LineCollection(g.nodes[g.edges], colors=STREET, linewidths=0.8, zorder=3))
    if extra:
        ax.add_collection(LineCollection([np.array([a, b]) for a, b in extra], colors=extra_col, linewidths=2.2, zorder=4))
    ax.scatter(*pts_steep[reached].T, s=14, color=OK_, edgecolor=SURF, linewidth=0.5, zorder=5)
    ax.scatter(*pts_steep[~reached].T, s=18, marker="x", color=NO_, linewidth=1.1, zorder=5)
    if clip:
        ax.set_xlim(clip[0], clip[1])
        ax.set_ylim(clip[2], clip[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9.3)


def reached_mask(g, P):
    from scipy.spatial import cKDTree
    if len(g.edges) == 0:
        return np.zeros(len(P), bool)
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    L = np.hypot(*(b - a).T)
    m = np.maximum(1, np.ceil(L / 2.0)).astype(int)
    e = np.repeat(np.arange(len(L)), m + 1)
    t = np.concatenate([np.arange(k + 1) / k for k in m])
    d, _ = cKDTree(a[e] + t[:, None] * (b - a)[e]).query(P)
    return d <= 60.0


def new_edges(g_after, g_before):
    key = lambda g: {tuple(np.round(np.sort(g.nodes[e], axis=0).ravel(), 2)) for e in g.edges}
    kb = key(g_before)
    out = []
    for e in g_after.edges:
        k = tuple(np.round(np.sort(g_after.nodes[e], axis=0).ravel(), 2))
        if k not in kb:
            out.append(g_after.nodes[e])
    return out


def row(axs, unit, bb, seed, uid, label):
    from urbanmorph.experiments.graph_io import graph_from_json
    F = {u: (blk, F_, cl, cr) for u, blk, F_, cl, cr in fields(unit)}[uid][1]
    crop = {u: cr for u, blk, F_, cl, cr in fields(unit)}[uid] if unit in ("machu_picchu", "petra") else None
    P, I, J = od_points(uid, F)
    steep_flag = np.zeros(len(P), bool)
    r_, c_ = (((F.S.y1 - P[:, 1]) / F.S.res).astype(int), ((P[:, 0] - F.S.x0) / F.S.res).astype(int))
    r_, c_ = np.clip(r_, 0, F.slope.shape[0] - 1), np.clip(c_, 0, F.slope.shape[1] - 1)
    steep_flag = (F.slope[r_, c_] >= 15) & (F.slope[r_, c_] < 35)
    Ps = P[steep_flag]
    r0, r1 = runs(unit, "A0", bb, seed), runs(unit, "A1", bb, seed)
    k = r0["ids"].index(uid)
    g0, g1 = graph_from_json(r0["graphs"][k]), graph_from_json(r1["graphs"][k])
    g5 = graph_from_json(r0["A5"]["snapshots"][k][-1][1])
    if crop:
        g0, g1, g5 = crop(g0), crop(g1), crop(g5)
    m0, m1 = meas(unit, "A0", bb, seed)["rows"][k], meas(unit, "A1", bb, seed)["rows"][k]
    sh = lambda r: r["out"][0] / max(r["pairs"], 1)
    s5 = m0["A5"][-1][1] / max(m0["pairs"], 1)
    rt = [(a, b) for src, a, b, kind in r1.get("src_edges", [[]] * len(r1["ids"]))[k] if src == "router"]
    clip = None
    if crop:
        ys, xs = np.nonzero(F.land)
        clip = (F.S.x0 + xs.min() * F.S.res - 40, F.S.x0 + xs.max() * F.S.res + 40,
                F.S.y1 - ys.max() * F.S.res - 40, F.S.y1 - ys.min() * F.S.res + 40)
    else:
        clip = (0, 512, 0, 512)
    head = f"{label}: {uid} ({bb}, seed {seed})\n"
    panel(axs[0], F, g0, Ps, reached_mask(g0, Ps), title=f"{head}A0 frozen M18: served {sh(m0):.2f} · {m0['length'] / 1000:.1f} km", clip=clip)
    panel(axs[1], F, g1, Ps, reached_mask(g1, Ps), [tuple(map(np.asarray, e)) for e in rt], ROUTER,
          f"\nA1 online router proposals: served {sh(m1):.2f} · {m1['length'] / 1000:.1f} km · {len(rt)} router steps kept", clip)
    panel(axs[2], F, g5, Ps, reached_mask(g5, Ps), new_edges(g5, g0), CONN,
          f"\nA5 router after growth: served {s5:.2f} · {m0['A5'][-1][0] / 1000:.1f} km", clip)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    suc, fai, npos, nneg, n = pick_tiles()
    legend = [Line2D([], [], color=STREET, lw=1.5, label="street"), Line2D([], [], color=ROUTER, lw=3, label="A1: street from a router step"),
              Line2D([], [], color=CONN, lw=3, label="A5: connector added after growth"),
              Line2D([], [], color=OK_, lw=0, marker="o", ms=6, label="steep endpoint reached (street within 60 m)"),
              Line2D([], [], color=NO_, lw=0, marker="x", ms=7, mew=1.5, label="steep endpoint not reached"),
              Patch(facecolor=(0.93, 0.80, 0.62), label="steep eligible land (15-35 %)")]
    fig, axs = plt.subplots(2, 3, figsize=(16, 12.6), facecolor=SURF)
    row(axs[0], suc[1], suc[2], suc[3], suc[4], f"Representative success (median of {npos} runs where A1 > A5, d = {suc[0]:+.3f})")
    row(axs[1], fai[1], fai[2], fai[3], fai[4], f"Representative failure (median of {nneg} runs where A1 < A5, d = {fai[0]:+.3f})")
    fig.legend(handles=legend, loc="lower left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(f"Tile scale: online routing (A1) against the same router after growth (A5), matched units ({n} unit-runs with >= 50 pairs)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=3.0)
    fig.savefig(OUT / "fig_maps_tiles.png", dpi=95, facecolor=SURF)
    plt.close(fig)
    # whole towns
    import json
    rep = json.loads((OUT / "report.json").read_text())["whole_town"]["per_site_seed"]
    pe = max((v["A5"] - v["A0"], int(k.split(":s")[1])) for k, v in rep.items() if k.startswith("petra"))
    mp = max((v["A1"] - v["A0"], int(k.split(":s")[1])) for k, v in rep.items() if k.startswith("machu"))
    fig, axs = plt.subplots(2, 3, figsize=(16, 12.5), facecolor=SURF)
    row(axs[0], "petra", "TNg_s0", pe[1], "petra", f"Petra, the seed with the largest A5 gain ({pe[0]:+.3f})")
    row(axs[1], "machu_picchu", "TNg_s0", mp[1], "machu_picchu", f"Machu Picchu, the seed with the largest A1 gain ({mp[0]:+.3f})")
    fig.legend(handles=legend, loc="lower left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Whole towns: A0, online routing (A1) and the router after growth (A5)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "fig_maps_towns.png", dpi=90, facecolor=SURF)
    plt.close(fig)
    print("success", suc, "failure", fai, "petra", pe, "mp", mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
