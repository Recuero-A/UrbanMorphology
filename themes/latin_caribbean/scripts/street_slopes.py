"""Street grades of the KAIST jungle village (user question 2026-11-23: "some streets are very steep").

Every street is cut into 10 m pieces (the last piece keeps its remainder) and each piece's grade |dz| / length is
weighted by its length, so the distribution is "share of street length at each grade". Three street sets on the same
KAIST domain: the real KAIST vehicular streets, the generated village streets (M* + organic, F1 + F4, cropped to the
village) and the real main roads kept through the jungle. Three surfaces:
* DSM: GLO-30 as the generator saw it (a SURFACE model: roofs and tree crowns are in it);
* 3D road surface: what the renders and Godot show (city3d.terrain3d.Surfaces.road = max(DSM smoothed 8 m, DSM));
* bare-earth proxy: the DSM low-passed at sigma 16 m (the TN-lp convention), which removes most building / tree bumps.

    .venv/bin/python themes/latin_caribbean/scripts/street_slopes.py
    ->  outputs/latin_caribbean/kaist_village/street_slopes.{png,md,json}, street_slopes_map.png
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LightSource
from scipy import ndimage

from urbanmorph.terrain.heightfield import Terrain

OUT = ROOT / "outputs" / "latin_caribbean" / "kaist_village"
BINS = [0, 2, 5, 8, 12, 15, 20, 30, np.inf]                     # percent
BIN_LABELS = ["0–2", "2–5", "5–8", "8–12", "12–15", "15–20", "20–30", ">30"]
SERIES = [("real KAIST streets", "#2a78d6"), ("village streets (generated)", "#eb6834"), ("main roads (real)", "#1baf7a")]
MAP_BINS = [0, 5, 8, 12, 20, np.inf]
MAP_COLS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal blue ramp, validated
MAP_W = [0.5, 0.7, 0.9, 1.4, 2.2]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def pieces(g, sel, T, step=10.0):
    """(grade, length, midpoint, piece endpoints) of every <= step-metre piece of the selected edges."""
    G, W, A, B = [], [], [], []
    for e in np.flatnonzero(sel):
        a, b = g.nodes[g.edges[e]]
        Ln = float(np.hypot(*(b - a)))
        if Ln < 1.0:
            continue
        cuts = np.r_[np.arange(0.0, Ln, step), Ln]
        cuts = cuts[np.r_[True, np.diff(cuts) > 0.5]] if len(cuts) > 2 else cuts
        cuts[-1] = Ln
        P = a + np.outer(cuts / Ln, b - a)
        z = T.sample(P[:, 0], P[:, 1])
        d = np.diff(cuts)
        G.append(np.abs(np.diff(z)) / d)
        W.append(d)
        A.append(P[:-1])
        B.append(P[1:])
    return np.concatenate(G), np.concatenate(W), np.vstack(A), np.vstack(B)


def summary(G, W):
    o = np.argsort(G)
    cw = np.cumsum(W[o]) / W.sum()
    q = lambda p: float(G[o][min(np.searchsorted(cw, p), len(o) - 1)])
    return {"km": float(W.sum() / 1000), "median_pct": 100 * q(0.5), "p90_pct": 100 * q(0.9), "p99_pct": 100 * q(0.99),
            **{f"share_over_{t}pct": float(W[G > t / 100].sum() / W.sum()) for t in (8, 12, 15, 20)},
            "share_by_bin": [float(W[(G * 100 >= lo) & (G * 100 < hi)].sum() / W.sum()) for lo, hi in zip(BINS[:-1], BINS[1:])]}


def bars(ax, res, surface, title):
    x = np.arange(len(BIN_LABELS))
    w = 0.26
    for k, (name, col) in enumerate(SERIES):
        s = np.array(res[surface][name]["share_by_bin"]) * 100
        ax.bar(x + (k - 1) * (w + 0.02), s, w, color=col, label=name, zorder=3)
        for xi, v in zip(x[4:], s[4:]):                               # direct labels on the steep bins only
            if v >= 0.5:
                ax.text(xi + (k - 1) * (w + 0.02), v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, BIN_LABELS, color=INK2, fontsize=9)
    ax.set_xlabel("street grade (%), 10 m pieces", color=INK2)
    ax.set_ylabel("share of street length (%)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.axvspan(4.5, 7.6, color="#f3f2ee", zorder=0)


def main() -> int:
    from m16_kaist_campus import L, anchors, domain
    big, (x0, y0), crs = domain()
    z = big["elev"].astype(float)
    surfaces = {"DSM": z, "3D road surface": np.maximum(ndimage.gaussian_filter(z, 2.0), z),
                "bare-earth proxy": ndimage.gaussian_filter(z, 4.0)}
    an = anchors(x0, y0, crs)
    lines = shapely.union_all([shapely.LineString(an.nodes[e]) for e in an.edges])
    real = pickle.loads((ROOT / "outputs" / "m16" / "kaist" / "m17_graphs.pkl").read_bytes())["real"]
    vil = list(pickle.loads((OUT / "graphs.pkl").read_bytes())["graphs"].values())[0]
    mid = vil.nodes[vil.edges].mean(1)
    is_main = shapely.distance(lines, shapely.points(mid[:, 0], mid[:, 1])) < 3.0
    sets = {"real KAIST streets": (real, np.ones(len(real.edges), bool)), "village streets (generated)": (vil, ~is_main),
            "main roads (real)": (vil, is_main)}
    res, keep = {}, {}
    for sn, Z in surfaces.items():
        T = Terrain(Z.astype(np.float32), -256.0, L + 256.0, 4.0)
        res[sn] = {}
        for gn, (g, sel) in sets.items():
            G, W, A, B = pieces(g, sel, T)
            res[sn][gn] = summary(G, W)
            if sn == "3D road surface" and gn != "real KAIST streets":
                keep[gn] = (G, A, B)
    (OUT / "street_slopes.json").write_text(json.dumps(res, indent=1))
    # ---- figure: two distributions (rendered surface, bare-earth proxy) and the village map by grade
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6.4), facecolor=SURF)
    for a in (ax0, ax1):
        a.set_facecolor(SURF)
    bars(ax0, res, "3D road surface", "Grades as rendered (3D road surface on the GLO-30 DSM)")
    bars(ax1, res, "bare-earth proxy", "Same streets on a bare-earth proxy (DSM low-passed, σ 16 m)")
    top = max(ax0.get_ylim()[1], ax1.get_ylim()[1])
    for a in (ax0, ax1):
        a.set_ylim(0, top)
        a.text(4.55, top * 0.62, "steep (> 12 %)", color=MUTED, fontsize=8.5, va="top")
    ax0.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    r, v = res["3D road surface"]["real KAIST streets"], res["3D road surface"]["village streets (generated)"]
    fig.suptitle(f"KAIST jungle village: street grades. Generated streets are steeper than the real ones "
                 f"(> 15 %: {100 * v['share_over_15pct']:.1f} % vs {100 * r['share_over_15pct']:.1f} % of length), "
                 "and the DSM adds steepness to both", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(OUT / "street_slopes.png", dpi=110, bbox_inches="tight", facecolor=SURF)
    fig, ax2 = plt.subplots(figsize=(11, 11), facecolor=SURF)
    rgb = LightSource(315, 40).shade(z, cmap=plt.cm.Greys_r, vert_exag=2, blend_mode="soft", vmin=z.min() - 40, vmax=z.max() + 10)
    rgb[big["water"] > 0, :3] = [0.80, 0.88, 0.95]
    ax2.imshow(rgb, extent=(-256, L + 256, -256, L + 256))
    for gn in keep:
        G, A, B = keep[gn]
        k = np.digitize(G * 100, MAP_BINS[1:-1])
        for j in range(len(MAP_COLS)):
            m = k == j
            segs = np.stack([A[m], B[m]], 1)
            ax2.add_collection(matplotlib.collections.LineCollection(segs, colors=MAP_COLS[j], linewidths=MAP_W[j], zorder=3 + j))
    hs = [plt.Line2D([], [], color=c, lw=w + 0.6, label=f"{lo}–{hi} %" if np.isfinite(hi) else f"> {lo} %")
          for c, w, lo, hi in zip(MAP_COLS, MAP_W, MAP_BINS[:-1], MAP_BINS[1:])]
    ax2.legend(handles=hs, title="grade", frameon=True, fontsize=8.5, title_fontsize=9, loc="lower left", facecolor=SURF, edgecolor=GRID)
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, L)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("KAIST jungle village: streets and main roads by grade (10 m pieces, 3D road surface; grey = hillshade)",
                  color=INK, fontsize=11, loc="left")
    fig.savefig(OUT / "street_slopes_map.png", dpi=120, bbox_inches="tight", facecolor=SURF)
    # ---- table
    rows = ["# KAIST jungle village: street grades (share of street length, 10 m pieces)", "",
            "| Surface | Streets | km | median | p90 | p99 | > 8 % | > 12 % | > 15 % | > 20 % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for sn in surfaces:
        for gn in sets:
            s = res[sn][gn]
            rows.append(f"| {sn} | {gn} | {s['km']:.1f} | {s['median_pct']:.1f} % | {s['p90_pct']:.1f} % | {s['p99_pct']:.1f} % | "
                        f"{100 * s['share_over_8pct']:.1f} % | {100 * s['share_over_12pct']:.1f} % | {100 * s['share_over_15pct']:.1f} % | "
                        f"{100 * s['share_over_20pct']:.1f} % |")
    (OUT / "street_slopes.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
