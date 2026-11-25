"""Milestone 17 (EXPLORATORY, user request 2026-11-23): every street generator inside the same KAIST village limits.

Procedure, identical for every method and for the real streets:
* networks: the cached KAIST-campus generations of every method (M16 originals, outputs/m16/kaist/all_cache; M17 F1 + F4
  versions, outputs/m16/kaist/m17_cache), seeds 1-3; each was generated on the whole 1536 m campus with the same
  GLO-30 terrain and the same real main roads (anchors);
* village limits: the mask of themes/latin_caribbean/scripts/kaist_village.py (1.05 km2); every network, and the real
  KAIST vehicular streets, is cropped exactly as the village map was (kaist_village.crop: generated streets with both
  ends inside, main roads everywhere, stubs < 10 m and isolated pieces < 120 m removed);
* realism: typicality (scripts/m16_realism.py: 13 morphology features, k = 5 nearest of 156 real neighbourhoods,
  1 = as typical as a real neighbourhood, lower = more typical) on the five 512 m windows that lie >= 70 % inside the
  village (greedy, < 30 % overlap; the same windows for every network);
* inside the village: street density, junction mix (3 / 4 / 5+ way), close15 (junctions with another < 15 m), junction
  density, dead ends per km (away from the cut at the village edge), share of street length in the largest connected
  piece, and grades of 10 m pieces on the rendered 3D road surface and on a bare-earth proxy (DSM low-passed 16 m).

    .venv/bin/python scripts/m17_village_compare.py [--workers 6]   ->  outputs/m17/village/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "themes" / "latin_caribbean" / "scripts"))

import numpy as np

OUT = ROOT / "outputs" / "m17" / "village"
VMASK = ROOT / "outputs" / "latin_caribbean" / "kaist_village" / "village_mask.npz"
SEEDS = (1, 2, 3)
IMPROVED = ["A", "M*", "M13-SMC", "NA-fixed", "NA-SMC", "M* + organic", "M* + radial"]
W = 512.0
G = {}                                                        # per-worker state


def jobs():
    from m16_kaist_all import ORDER
    out = [(lab, "original", s) for lab in ORDER for s in SEEDS]
    out += [(lab, "F1 + F4", s) for lab in IMPROVED for s in SEEDS]
    return out


def windows(mask, L, cover=0.7, overlap=0.3):
    cov = lambda x0, y0: mask[int((L + 256 - (y0 + W)) / 4):int((L + 256 - y0) / 4), int((x0 + 256) / 4):int((x0 + W + 256) / 4)].mean()
    cands = sorted(((cov(x, y), x, y) for x in range(0, int(L - W) + 1, 32) for y in range(0, int(L - W) + 1, 32)), reverse=True)
    best = []
    for c, x, y in cands:
        if c < cover:
            break
        if all(max(0, W - abs(x - bx)) * max(0, W - abs(y - by)) / W ** 2 < overlap for _, bx, by in best):
            best.append((float(c), x, y))
    return best


def init():
    from scipy import ndimage
    from kaist_village import village_mask
    from m16_kaist_campus import L, anchors, domain
    from m16_realism import K, real_pool
    from urbanmorph.terrain.heightfield import Terrain
    big, (x0, y0), crs = domain()
    G["big"], G["L"] = big, L
    G["anchor"] = anchors(x0, y0, crs)
    G["mask"] = np.load(VMASK)["mask"] if VMASK.exists() else village_mask(big, G["anchor"], L)
    z = big["elev"].astype(float)
    G["T3"] = Terrain(np.maximum(ndimage.gaussian_filter(z, 2.0), z).astype(np.float32), -256.0, L + 256.0, 4.0)
    G["Tb"] = Terrain(ndimage.gaussian_filter(z, 4.0).astype(np.float32), -256.0, L + 256.0, 4.0)
    X, _ = real_pool()
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi.mean(0), np.where(Xi.std(0) > 1e-9, Xi.std(0), 1.0)
    Z = (Xi - mu) / sd
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    G["pool"] = (med, mu, sd, Z, float(np.mean(np.sort(D, 1)[:, :K].mean(1))), K)
    G["wins"] = windows(G["mask"], L)


def window_graph(g, x0, y0):
    import shapely
    from urbanmorph.graphs.road_graph import build_graph
    box = shapely.box(x0, y0, x0 + W, y0 + W)
    lines = []
    for a, b in g.edges:
        seg = shapely.LineString(g.nodes[[a, b]]).intersection(box)
        for part in getattr(seg, "geoms", [seg]):
            if part.geom_type == "LineString" and part.length > 0.5:
                lines.append(np.asarray(part.coords) - [x0, y0])
    return build_graph(lines) if lines else None


def typicality(g, rng):
    from m16_realism import vec
    from urbanmorph.graphs.network_outcomes import outcome_row
    from urbanmorph.graphs.style_metrics import morphology_row
    from urbanmorph.graphs.terrain_generate import terrain_of
    med, mu, sd, Z, real_typ, K = G["pool"]
    L, big = G["L"], G["big"]
    out = []
    for _, x0, y0 in G["wins"]:
        wg = window_graph(g, x0, y0)
        if wg is None or len(wg.edges) == 0:
            continue
        n0, c0 = int((L - y0 - W) / 4), int(x0 / 4)
        t = terrain_of({k: v[n0:n0 + 256, c0:c0 + 256] for k, v in big.items()})
        row = {**morphology_row(wg, t, centre=(256.0, 256.0), rng=rng, grade_span=30.0), **outcome_row(wg)}
        v = np.asarray(vec(row), float)
        v = np.where(np.isfinite(v), v, med)
        out.append(float(np.sort(np.sqrt(((((v - mu) / sd) - Z) ** 2).sum(1)))[:K].mean() / real_typ))
    return float(np.mean(out)) if out else np.nan


def grades(g, sel, T):
    from street_slopes import pieces
    if not sel.any():
        return np.zeros(0), np.zeros(0)
    Gr, Wt, _, _ = pieces(g, sel, T)
    return Gr, Wt


def village_stats(g):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from urbanmorph.graphs.junctions import clean
    mask, L = G["mask"], G["L"]
    n = mask.shape[0]
    rc = lambda P: (np.clip(((L + 256 - P[:, 1]) / 4).astype(int), 0, n - 1), np.clip(((P[:, 0] + 256) / 4).astype(int), 0, n - 1))
    area = float(mask[64:448, 64:448].sum() * 16 / 1e6)
    core = ndimage.binary_erosion(mask, iterations=8)                        # 32 m inside the village edge
    N, E, _ = clean(g)
    mid = N[E].mean(1)
    inside = mask[rc(mid)]
    Ln = np.linalg.norm(N[E[:, 0]] - N[E[:, 1]], axis=1)
    km = float(Ln[inside].sum() / 1000)
    deg = np.bincount(E.ravel(), minlength=len(N))
    vin = mask[rc(N)]
    J = np.flatnonzero((deg >= 3) & vin)
    n3, n4, n5 = (int(((deg[J] == 3)).sum()), int((deg[J] == 4).sum()), int((deg[J] >= 5).sum()))
    tot = max(len(J), 1)
    close = float((cKDTree(N[J]).query(N[J], 2)[0][:, 1] < 15.0).mean()) if len(J) >= 2 else np.nan
    dead = int(((deg == 1) & core[rc(N)]).sum())
    lab = np.arange(len(N))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for a, b in E[inside]:
        ra, rb = find(a), find(b)
        if ra != rb:
            lab[ra] = rb
    comp = np.array([find(a) for a in E[inside, 0]])
    lcc = float(max(np.bincount(np.unique(comp, return_inverse=True)[1], weights=Ln[inside])) / max(Ln[inside].sum(), 1e-9)) if len(comp) else np.nan
    sel = np.zeros(len(g.edges), bool)
    m2 = g.nodes[g.edges].mean(1)
    sel[mask[rc(m2)]] = True
    out = {"street_km_per_km2": km / area, "junctions_per_km2": len(J) / area, "share_3way": n3 / tot, "share_4way": n4 / tot,
           "share_5p": n5 / tot, "close15": close, "deadends_per_km": dead / max(km, 1e-9), "largest_piece_share": lcc}
    for name, T in (("", G["T3"]), ("bare_", G["Tb"])):
        Gr, Wt = grades(g, sel, T)
        if len(Gr):
            o = np.argsort(Gr)
            cw = np.cumsum(Wt[o]) / Wt.sum()
            out[f"{name}grade_median_pct"] = float(100 * Gr[o][min(np.searchsorted(cw, 0.5), len(o) - 1)])
            for th in (12, 15, 20):
                out[f"{name}over_{th}pct"] = float(Wt[Gr > th / 100].sum() / Wt.sum())
    return out


def run(job):
    import shapely
    from kaist_village import crop
    from m16_kaist_all import key_of
    lab, ver, seed = job
    if lab == "real":
        g = pickle.loads((ROOT / "outputs" / "m16" / "kaist" / "m17_graphs.pkl").read_bytes())["real"]
    else:
        cache = ROOT / "outputs" / "m16" / "kaist" / ("all_cache" if ver == "original" else "m17_cache")
        g = pickle.loads((cache / f"{key_of(lab)}_s{seed}.pkl").read_bytes())["graph"]
    gv, info = crop(g, G["anchor"], G["mask"], G["L"])
    row = {"method": lab, "version": ver, "seed": seed, **village_stats(gv),
           "typicality": typicality(gv, np.random.default_rng([seed, 17]))}
    return row, (gv if seed == 1 else None)


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = [("real", "reference", 1)] + jobs()
    with Pool(a.workers, initializer=init) as p:
        res = p.map(run, js, chunksize=1)
    init()
    rows = [r for r, _ in res]
    graphs = {(r["method"], r["version"]): gv for r, gv in res if gv is not None}
    (OUT / "graphs_seed1.pkl").write_bytes(pickle.dumps(graphs))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)
    num = [c for c in df.columns if c not in ("method", "version", "seed")]
    agg = df.groupby(["method", "version"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values("typicality_mean")
    agg.to_csv(OUT / "summary.csv", index=False)
    json.dump({"windows": G["wins"], "rows": agg.to_dict("records")}, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(agg[["method", "version", "typicality_mean", "street_km_per_km2_mean", "share_5p_mean", "close15_mean", "over_15pct_mean"]]
          .to_string(index=False))
    return 0


FAMILY = {"A": 0, "M*": 0, "M* TNse": 0, "M13-SMC": 0, "NA-myopic": 1, "NA-fixed": 1, "NA-adaptive": 1, "NA-random": 1,
          "NA-SMC": 1, "M* + grid": 2, "M* + organic": 2, "M* + radial": 2, "M* + culdesac": 2, "M12-HR": 3, "P": 3}
FAM = [("learned (M13 family)", "#2a78d6"), ("network-aware (M14 NA)", "#eb6834"), ("styles (M16)", "#1baf7a"),
       ("hard rules / procedural (M12)", "#eda100")]                    # validated categorical slots 1-4
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NUDGE = {"NA-SMC · F1+F4": (-92, -4), "NA-fixed · F1+F4": (-98, -12), "NA-fixed": (6, 7), "M* · F1+F4": (-62, 8),
         "M*": (6, -12), "A · F1+F4": (8, -12), "M* + organic": (-30, 9), "M* + radial": (6, -13)}


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LightSource
    init()
    agg = pd.read_csv(OUT / "summary.csv")
    graphs = pickle.loads((OUT / "graphs_seed1.pkl").read_bytes())
    # ---- realism vs steepness (one point per method version, mean of 3 seeds, bars = seed SD)
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=SURF)
    ax.set_facecolor(SURF)
    for lab in agg["method"].unique():
        if lab == "real":
            continue
        rows = agg[agg.method == lab].set_index("version")
        col = FAM[FAMILY[lab]][1]
        if {"original", "F1 + F4"} <= set(rows.index):
            o, f = rows.loc["original"], rows.loc["F1 + F4"]
            ax.annotate("", (f.typicality_mean, 100 * f.over_15pct_mean), (o.typicality_mean, 100 * o.over_15pct_mean),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.7), zorder=2)
        for ver, r in rows.iterrows():
            hollow = ver == "F1 + F4"
            ax.errorbar(r.typicality_mean, 100 * r.over_15pct_mean, xerr=r.typicality_std, yerr=100 * r.over_15pct_std,
                        fmt="o", ms=8, mfc=SURF if hollow else col, mec=col, mew=2, ecolor=col, elinewidth=0.8, alpha=0.95, zorder=3)
            txt = lab + (" · F1+F4" if hollow else "")
            ax.annotate(txt, (r.typicality_mean, 100 * r.over_15pct_mean), xytext=NUDGE.get(txt, (6, 5)),
                        textcoords="offset points", fontsize=8, color=INK2, zorder=4)
    rr = agg[agg.method == "real"].iloc[0]
    ax.plot(rr.typicality_mean, 100 * rr.over_15pct_mean, marker="*", ms=17, color=INK, zorder=5)
    ax.annotate("real KAIST streets (same village)", (rr.typicality_mean, 100 * rr.over_15pct_mean), xytext=(10, 4),
                textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(1.0, color=AXIS, lw=1, ls="--", zorder=1)
    ax.text(1.0, ax.get_ylim()[1], " 1 = as typical as a real neighbourhood", color=MUTED, fontsize=8.5, va="top")
    ax.set_xlabel("realism: typicality vs 156 real neighbourhoods (lower = more realistic)", color=INK2)
    ax.set_ylabel("street length steeper than 15 % (%), rendered 3D surface", color=INK2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec=c, label=n) for n, c in FAM]
    hs += [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=SURF, mec=INK2, mew=2, label="with F1 + F4 (arrow: original → fixed)"),
           plt.Line2D([], [], marker="*", ls="", ms=14, color=INK, label="real KAIST streets")]
    ax.legend(handles=hs, frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_title("KAIST village limits: every generator, realism vs steep streets (mean of seeds 1-3, bars = seed SD). "
                 "Best = bottom left", color=INK, fontsize=11.5, loc="left")
    fig.savefig(OUT / "fig_village_realism_vs_slope.png", dpi=120, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    # ---- maps, seed 1
    big, mask, L = G["big"], G["mask"], G["L"]
    z = big["elev"].astype(float)
    rgb = LightSource(315, 40).shade(z, cmap=plt.cm.Greys_r, vert_exag=2, blend_mode="soft", vmin=z.min() - 60, vmax=z.max() + 20)
    rgb[~mask, :3] = rgb[~mask, :3] * 0.55 + np.array([0.80, 0.86, 0.78]) * 0.45
    rgb[big["water"] > 0, :3] = [0.78, 0.86, 0.95]
    rows_ = agg.to_dict("records")
    rr_ = next(r for r in rows_ if r["method"] == "real")
    order = [rr_] + [r for r in rows_ if r["method"] != "real"]
    ys, xs = np.nonzero(mask)
    x0, x1 = -256 + xs.min() * 4 - 20, -256 + xs.max() * 4 + 20
    y1, y0 = L + 256 - ys.min() * 4 + 20, L + 256 - ys.max() * 4 - 20
    nc = 5
    nr = int(np.ceil(len(order) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.9 * nr), facecolor=SURF)
    lines_main = G["anchor"]
    for ax, r in zip(axs.ravel(), order):
        g = graphs[(r["method"], r["version"])]
        ax.imshow(rgb, extent=(-256, L + 256, -256, L + 256))
        segs = g.nodes[g.edges]
        ax.add_collection(matplotlib.collections.LineCollection(segs, colors=INK, linewidths=0.35))
        ax.add_collection(matplotlib.collections.LineCollection(lines_main.nodes[lines_main.edges], colors="#52514e", linewidths=1.1))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        name = "REAL KAIST streets" if r["method"] == "real" else r["method"] + ("  · F1 + F4" if r["version"] == "F1 + F4" else "")
        ax.set_title(f"{name}\nrealism {r['typicality_mean']:.2f} · {r['street_km_per_km2_mean']:.0f} km/km² · "
                     f"> 15 %: {100 * r['over_15pct_mean']:.1f} %", fontsize=9.5, color=INK, loc="left")
    for ax in axs.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("KAIST village limits (1.05 km²): every generator, seed 1, ordered by realism (lower = more realistic). "
                 "Black = generated streets, grey = real main roads, green tint = jungle outside the village", fontsize=13,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "fig_village_maps.png", dpi=85, facecolor=SURF)
    plt.close(fig)
    # ---- table
    f = lambda v, n=2: "–" if not np.isfinite(v) else f"{v:.{n}f}"
    pc = lambda v: "–" if not np.isfinite(v) else f"{100 * v:.1f} %"
    md = ["# KAIST village limits: every street generator (EXPLORATORY; seeds 1-3, means)", "",
          "Ordered by realism (typicality vs 156 real neighbourhoods; 1 = as typical as a real neighbourhood; lower = more "
          "realistic; mean of 4 windows >= 70 % inside the village). Grades: share of street length in 10 m pieces on the rendered "
          "3D road surface (bare-earth proxy in brackets). Real KAIST = the real vehicular streets cropped the same way.", "",
          "| # | Generator | Version | Realism | km / km² | Junctions / km² | 4-way | 5+-way | close15 | Dead ends / km | Largest piece | > 12 % | > 15 % | > 20 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    k = 0
    for r in rows_:
        k += r["method"] != "real"
        md.append(f"| {'' if r['method'] == 'real' else k} | {'**REAL KAIST**' if r['method'] == 'real' else r['method']} | {r['version']} | "
                  f"{f(r['typicality_mean'])} ± {f(r['typicality_std'])} | {f(r['street_km_per_km2_mean'], 1)} | {f(r['junctions_per_km2_mean'], 0)} | "
                  f"{pc(r['share_4way_mean'])} | {pc(r['share_5p_mean'])} | {f(r['close15_mean'])} | {f(r['deadends_per_km_mean'], 1)} | "
                  f"{pc(r['largest_piece_share_mean'])} | {pc(r['over_12pct_mean'])} | {pc(r['over_15pct_mean'])} ({pc(r['bare_over_15pct_mean'])}) | "
                  f"{pc(r['over_20pct_mean'])} |")
    (OUT / "village_compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(figures() if "--figures" in sys.argv else main())
