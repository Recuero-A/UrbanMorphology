"""Milestone 18: maps of the terrain-potential pilot (docs/M18_LOG.md F24, F25), for inspection. Nothing new is scored;
the numbers in the titles are recomputed from the stored runs by the code of the pilot.

sites   - Machu Picchu and Petra, seed 1 (the seed of the site benchmark's overview sheets): the real OSM vehicular
          network, leastcostpath, citygen P&M, NA-SMC on TN and TNg with the terrain potentials off and on, cropped to
          the city limits (site_benchmark.crop). fig_site_<site>.png: streets by grade (10 m pieces on the bare-earth
          proxy, sigma 16 m, as TAS). fig_site_connectivity.png: the largest connected piece against the rest, and the
          fixed land points of the geographic access test with no street within 60 m.
tiles   - fig_tiles.png: real, TN off / on, TNg off / on (backbone s0, seed 1) on three windows chosen by rule: the GEN2
          tile with the highest median land slope, the median one
          of the hilly GEN2 tiles (median land slope >= 10 %), and the natural window with the highest median slope.
          Grade on the DEM low-passed at sigma 16 m (the terrain-reference slope surface).
routes  - fig_routes.png: one leastcostpath anchor pair per site, the pair of median straight distance among those that
          leastcostpath, NA-SMC TN and NA-SMC TNg + potentials all serve (the first seed with one); map and profile.

    python scripts/m18_maps.py  ->  outputs/m18/terrain_potentials/maps/
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

OUT = ROOT / "outputs" / "m18" / "terrain_potentials" / "maps"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
GRADE_EDGES = (5, 10, 15)
RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")         # ordinal blue, validated (dataviz validate_palette --ordinal)
GRADE_LAB = ("0–5 %", "5–10 %", "10–15 %", "> 15 %")
MAIN, REST, UNREACHED = "#3a3a38", "#eb6834", "#6250d6"
ROUTE_COL = ("#2a78d6", "#eb6834", "#1baf7a")                # categorical slots 1-3 (all-pairs pass)
STEEP_TINT = np.array([0.93, 0.80, 0.62])
SITE_PANELS = [("Real streets (OSM, vehicular)", "real", "vehicular"),
               ("leastcostpath (external)", "leastcostpath (ext)", "original"),
               ("citygen, Parish & Müller (external)", "citygen P&M (ext)", "original"),
               None,
               ("NA-SMC · TN (trained on LiDAR)", "NA-SMC", "original"),
               ("NA-SMC · TN + terrain potentials", "NA-SMC", "TM"),
               ("NA-SMC · TNg (trained on GLO-30)", "NA-SMC", "TNg"),
               ("NA-SMC · TNg + terrain potentials", "NA-SMC", "TNg + TM")]
CONN_PANELS = [SITE_PANELS[1], SITE_PANELS[4], SITE_PANELS[6], SITE_PANELS[7]]
SITE_NAME = {"machu_picchu": "Machu Picchu", "petra": "Petra"}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": "#c3c2b7"})
    return plt


def graded(g, sample, step=10.0):
    """Ground edges cut into <= step m pieces: segments (n, 2, 2), grade (%), length (m), midpoints."""
    if len(g.edges) == 0:
        return np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2))
    st = np.asarray(g.structs())
    S, Gr, Ln = [], [], []
    for e in np.flatnonzero(st == 0):
        a, b = g.nodes[g.edges[e]]
        L = float(np.hypot(*(b - a)))
        if L < 1.0:
            continue
        m = max(1, int(np.ceil(L / step)))
        P = a + np.outer(np.linspace(0, 1, m + 1), b - a)
        z = sample(P[:, 0], P[:, 1])
        S.append(np.stack([P[:-1], P[1:]], 1))
        Gr.append(np.abs(np.diff(z)) / (L / m) * 100)
        Ln.append(np.full(m, L / m))
    if not S:
        return np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 2))
    S = np.concatenate(S)
    return S, np.concatenate(Gr), np.concatenate(Ln), S.mean(1)


def draw_grade(ax, g, sample, lw):
    from matplotlib.collections import LineCollection
    S, gr, ln, _ = graded(g, sample)
    k = np.digitize(gr, GRADE_EDGES)
    o = np.argsort(k, kind="stable")                         # steep pieces drawn last, on top
    ax.add_collection(LineCollection(S[o], colors=np.array(RAMP)[k[o]], linewidths=lw, capstyle="round", zorder=3))
    st = np.asarray(g.structs()) if len(g.edges) else np.zeros(0, int)
    if (st != 0).any():
        ax.add_collection(LineCollection(g.nodes[g.edges[st != 0]], colors=MUTED, linewidths=lw, linestyles=(0, (2, 1.5)), zorder=3))
    return gr, ln


def share15(gr, ln):
    return float(ln[gr > 15].sum() / ln.sum()) if ln.sum() > 0 else np.nan


def pale(rgb, a=0.5):
    out = rgb.copy()
    out[..., :3] = (1 - a) + a * rgb[..., :3]
    return out


def grade_legend(ax, x0=0.04, y0=0.93, dy=0.075, extra=()):
    ax.axis("off")
    ax.text(x0, y0, "Street grade (10 m pieces)", fontsize=10.5, weight="bold", transform=ax.transAxes, va="top")
    for i, (c, lab) in enumerate(zip(RAMP, GRADE_LAB)):
        y = y0 - (i + 1) * dy
        ax.plot([x0, x0 + 0.12], [y, y], color=c, lw=4, solid_capstyle="round", transform=ax.transAxes)
        ax.text(x0 + 0.16, y, lab, fontsize=10, va="center", transform=ax.transAxes)
    y = y0 - 5 * dy
    ax.plot([x0, x0 + 0.12], [y, y], color=MUTED, lw=2, ls=(0, (2, 1.5)), transform=ax.transAxes)
    ax.text(x0 + 0.16, y, "bridge / tunnel", fontsize=10, va="center", transform=ax.transAxes)
    ax.text(x0, y - dy, "\n".join(extra), fontsize=9.2, va="top", color=INK2, transform=ax.transAxes, linespacing=1.45)


# ---------------------------------------------------------------- sites
def site_graphs(name):
    import site_benchmark as sb
    site, mask = sb.G["site"], sb.G["mask"]
    out = {}
    for p in SITE_PANELS:
        if p is None:
            continue
        _, lab, ver = p
        if lab == "real":
            g = site.real(ver)
        else:
            g = pickle.loads((site.cache / f"{sb.key_of(lab, ver)}_s1.pkl").read_bytes())["graph"]
        out[(lab, ver)] = sb.crop(g, mask, site)
    return out


def components(g):
    """Edge mask of the largest connected piece (by length)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(g.nodes)
    A = coo_matrix((np.ones(len(g.edges)), (g.edges[:, 0], g.edges[:, 1])), shape=(n, n))
    lab = connected_components(A, directed=False)[1]
    L = np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T)
    tot = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=lab.max() + 1)
    big = int(np.argmax(tot))
    return lab[g.edges[:, 0]] == big, float(tot[big] / tot.sum())


def unreached(g, pts):
    from scipy.spatial import cKDTree
    from m18_terrain_potentials import GEO_ACCESS_M
    if len(g.edges) == 0:
        return np.ones(len(pts), bool)
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    L = np.hypot(*(b - a).T)
    m = np.maximum(1, np.ceil(L / 2.0)).astype(int)
    e = np.repeat(np.arange(len(L)), m + 1)
    t = np.concatenate([np.arange(k + 1) / k for k in m])
    d, _ = cKDTree(a[e] + t[:, None] * (b - a)[e]).query(pts)
    return d > GEO_ACCESS_M


def stage_sites():
    plt = plt_setup()
    from matplotlib.collections import LineCollection
    from scipy import ndimage
    import site_benchmark as sb
    from m18_terrain_potentials import GEO_N, SiteField, geo_access, geo_points, geo_stats
    from m18_terrain_reference import pool, score, tile_sums
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    conn = {}
    for name in ("machu_picchu", "petra"):
        sb.init(name)
        site, big, mask, Tb = sb.G["site"], sb.G["big"], sb.G["mask"], sb.G["Tb"]
        F = SiteField(site, big, mask)
        pts = geo_points(name, F, n=int(round(GEO_N * float(F.land.sum() * 16 / 1e6) / 0.262144)))
        G = site_graphs(name)
        bg = pale(sb.hillshade(site, big, mask), 0.55)
        ext = (-sb.MARGIN, site.L + sb.MARGIN, -sb.MARGIN, site.L + sb.MARGIN)
        ys, xs = np.nonzero(mask)
        view = (-sb.MARGIN + xs.min() * sb.RES - 40, -sb.MARGIN + xs.max() * sb.RES + 40,
                site.L + sb.MARGIN - ys.max() * sb.RES - 40, site.L + sb.MARGIN - ys.min() * sb.RES + 40)
        outline = ndimage.binary_fill_holes(mask).astype(float)

        def base(ax, site=site, big=big, bg=bg, ext=ext, view=view, outline=outline):
            ax.imshow(bg, extent=ext)
            sb.contours(ax, site, big, step=50, lw=0.22)
            ax.contour(*site.xy_grid(), outline, levels=[0.5], colors=site.cfg["outside"]["edge"], linewidths=1.0)
            ax.set_xlim(view[0], view[1])
            ax.set_ylim(view[2], view[3])
            ax.set_xticks([])
            ax.set_yticks([])

        def numbers(g):
            gs = geo_stats([geo_access(g, *pts, closed=False)])
            t = sb.tas(g)
            return {"realism": score(pool([tile_sums(g, F)]), ref, sd), "TAS": t["TAS"],
                    "km": sb.city_stats(g)["street_km_per_km2"], "piece": components(g)[1] if len(g.edges) else np.nan, **gs}
        N = {k: numbers(g) for k, g in G.items()}
        w, h = view[1] - view[0], view[3] - view[2]
        fig, axs = plt.subplots(2, 4, figsize=(19, 19 * 2 * h / (4 * w) + 1.3), facecolor=SURF)
        for ax, p in zip(axs.ravel(), SITE_PANELS):
            if p is None:
                grade_legend(ax, extra=("terrain realism: gap to the real-city",
                                        "terrain curves (0 = same; real cities",
                                        "0.24–1.55). TAS: terrain adaptiveness",
                                        "0–100, higher = better.",
                                        "50 m contours; city limits outlined.",
                                        "Seed 1 everywhere."))
                continue
            title, lab, ver = p
            base(ax)
            gr, ln = draw_grade(ax, G[(lab, ver)], Tb.sample, 0.75)
            n = N[(lab, ver)]
            ax.set_title(f"{title}\nterrain realism {n['realism']:.2f} · TAS {n['TAS']:.0f} · {n['km']:.0f} km/km² · "
                         f"> 15 %: {100 * share15(gr, ln):.0f} % of length", loc="left", fontsize=9.3)
        fig.suptitle(f"{SITE_NAME[name]}: the same city limits, every generator (bottom row: our model with the terrain "
                     "potentials off and on)", x=0.01, ha="left", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT / f"fig_site_{name}.png", dpi=105, facecolor=SURF)
        plt.close(fig)
        conn[name] = (G, N, pts, base, w, h)
        print(name, {f"{k[0]}|{k[1]}": {q: round(v, 3) for q, v in n.items()} for k, n in N.items()}, flush=True)
    # ---- connectivity sheet
    ratio = [c[5] / c[4] for c in conn.values()]
    fig, axs = plt.subplots(2, 4, figsize=(19, 19 / 4 * sum(ratio) + 2.6), facecolor=SURF,
                            gridspec_kw={"height_ratios": ratio})
    for row, (name, (G, N, pts, base, _, _)) in zip(axs, conn.items()):
        for ax, (title, lab, ver) in zip(row, CONN_PANELS):
            g, n = G[(lab, ver)], N[(lab, ver)]
            base(ax)
            if len(g.edges):
                keep, _ = components(g)
                ax.add_collection(LineCollection(g.nodes[g.edges[~keep]], colors=REST, linewidths=1.0, zorder=3))
                ax.add_collection(LineCollection(g.nodes[g.edges[keep]], colors=MAIN, linewidths=0.7, zorder=3))
            u = unreached(g, pts[0])
            ax.scatter(*pts[0][u].T, s=14, marker="x", color=UNREACHED, linewidths=1.0, zorder=4)
            ax.set_title(f"{SITE_NAME[name]} · {title}\nlargest piece: {100 * n['piece']:.0f} % of street length\n"
                         f"land points reached: {100 * n['access_60']:.0f} % · point pairs connected: "
                         f"{100 * n['geo_connected']:.0f} %", loc="left", fontsize=9.3)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=MAIN, lw=2, label="largest connected piece"),
                        Line2D([], [], color=REST, lw=2, label="pieces cut off from it"),
                        Line2D([], [], color=UNREACHED, lw=0, marker="x", ms=7, mew=1.5,
                               label="fixed land point with no street within 60 m")],
               loc="lower left", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Connectivity: the terrain potentials (right column) make TNg's streets gentler but split the network "
                 "and leave land unreached (seed 1)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / "fig_site_connectivity.png", dpi=105, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- tiles
def stage_tiles():
    plt = plt_setup()
    from matplotlib.colors import LightSource
    from m18_terrain_potentials import eval_tiles, load_graphs
    from m18_terrain_reference import TILE, Field
    rows = []
    T = eval_tiles("gen2")
    med = [float(np.median(Field(t["d"]).slope[Field(t["d"]).land])) for t in T]
    o = np.argsort(med)
    hilly = [k for k in o if med[k] >= 10]
    for i, why in ((int(o[-1]), "steepest GEN2 tile"), (int(hilly[(len(hilly) - 1) // 2]), "median hilly GEN2 tile")):
        rows.append(("gen2", i, T[i], why, med[i]))
    Tn = eval_tiles("natural")
    medn = [float(np.median(Field(t["d"]).slope[Field(t["d"]).land])) for t in Tn]
    i = int(np.argmax(medn))
    rows.append(("natural", i, Tn[i], "steepest natural window", medn[i]))
    arms = [("Real streets", None, None), ("NA-SMC · TN", "TN", "off"), ("TN + terrain potentials", "TN", "on"),
            ("NA-SMC · TNg", "TNg", "off"), ("TNg + terrain potentials", "TNg", "on")]
    cache = {}
    fig, axs = plt.subplots(len(rows), 5, figsize=(20, 4.9 * len(rows) + 1.2), facecolor=SURF)
    for r, (where, i, t, why, ms) in enumerate(rows):
        F = Field(t["d"])
        zs = np.asarray(F.S.elev, float)
        hs = LightSource(315, 40).hillshade(zs, vert_exag=2.0, dx=4.0, dy=4.0)
        rgb = np.dstack([0.55 + 0.45 * hs] * 3 + [np.ones_like(hs)])
        steep = F.slope >= 20
        rgb[steep, :3] = rgb[steep, :3] * 0.45 + STEEP_TINT * 0.55
        rgb = pale(rgb, 0.6)
        steep_land = F.land & steep
        a_steep = steep_land.sum() * 16 / 1e6
        for c, (title, fam, arm) in enumerate(arms):
            ax = axs[r, c]
            ax.imshow(rgb, extent=(-256, 768, -256, 768))
            X = -256 + (np.arange(zs.shape[1]) + 0.5) * 4
            Y = 768 - (np.arange(zs.shape[0]) + 0.5) * 4
            ax.contour(X, Y, zs, levels=np.arange(np.floor(zs.min() / 10) * 10, zs.max() + 10, 10), colors="#3d3a33",
                       linewidths=0.25, alpha=0.45)
            ax.set_xlim(0, TILE)
            ax.set_ylim(0, TILE)
            ax.set_xticks([])
            ax.set_yticks([])
            if fam is None:
                g = t.get("graph")
                if g is None:
                    ax.set_title(f"{why} (median slope {ms:.0f} %)\nno real streets: natural terrain", loc="left", fontsize=9.3)
                    continue
            else:
                key = (where, fam, arm)
                if key not in cache:
                    cache[key] = load_graphs(where, f"{fam}_s0", 1, arm, fam)
                g = cache[key][i]
            draw_grade(ax, g, F.S.sample, 1.1)
            _, gr2, ln2, M = graded(g, F.S.sample)
            inside = (M[:, 0] >= 0) & (M[:, 0] <= TILE) & (M[:, 1] >= 0) & (M[:, 1] <= TILE)
            land_km2 = F.land.sum() * 16 / 1e6
            on_steep = np.zeros(len(M), bool)
            if len(M):
                rr = np.clip(((768 - M[:, 1]) / 4).astype(int), 0, zs.shape[0] - 1)
                cc = np.clip(((M[:, 0] + 256) / 4).astype(int), 0, zs.shape[1] - 1)
                on_steep = steep_land[rr, cc] & inside
            head = f"{why} (median slope {ms:.0f} %) · {title}" if c == 0 else title
            ax.set_title(f"{head}\n{ln2[inside].sum() / 1000 / land_km2:.0f} km/km² · on land ≥ 20 %: "
                         f"{ln2[on_steep].sum() / 1000 / max(a_steep, 1e-9):.0f} km/km² · > 15 %: "
                         f"{100 * share15(gr2[inside], ln2[inside]):.0f} %", loc="left", fontsize=9.3)
        print(where, t["sid"], why, round(ms, 1), flush=True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=c, lw=3, label=l) for c, l in zip(RAMP, GRADE_LAB)]
    h.append(Patch(facecolor=0.4 + 0.6 * (STEEP_TINT * 0.55 + 0.45 * 0.75), edgecolor="none", label="terrain slope ≥ 20 %"))
    fig.legend(handles=h, loc="lower left", ncol=5, frameon=False, fontsize=10.5, title="Street grade (10 m pieces)",
               title_fontsize=10.5, bbox_to_anchor=(0.01, 0.0), alignment="left")
    fig.suptitle("512 m tiles: real streets and NA-SMC with the terrain potentials off and on (backbone s0, seed 1; "
                 "10 m contours)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))
    fig.savefig(OUT / "fig_tiles.png", dpi=100, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- routes
def stage_routes():
    plt = plt_setup()
    import pandas as pd
    import site_benchmark as sb
    from m18_routing import Router
    df = pd.read_csv(ROOT / "outputs" / "m18" / "routing" / "routes.csv")
    srcs = [("leastcostpath route", "leastcostpath route (ext)", None),
            ("NA-SMC · TN", "NA-SMC [TN]", "original"),
            ("NA-SMC · TNg + terrain potentials", "NA-SMC [TNg] + terrain potentials", "TNg + TM")]
    fig, axs = plt.subplots(2, 2, figsize=(17, 15.5), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.15]})
    for row, name in zip(axs, ("machu_picchu", "petra")):
        sb.init(name)
        site, big, mask, Tb = sb.G["site"], sb.G["big"], sb.G["mask"], sb.G["Tb"]
        d = df[(df.site == name) & df.served]
        pick = None
        for seed in sb.SEEDS:
            ok = None
            for _, lab, _ in srcs:
                s = set(d[(d.seed == seed) & (d.network == lab)].pair)
                ok = s if ok is None else ok & s
            if ok:
                c = d[(d.seed == seed) & (d.network == srcs[0][1]) & d.pair.isin(ok)].sort_values("straight_m")
                pick = (seed, c.iloc[(len(c) - 1) // 2].pair, len(c))
                break
        seed, pair, n_ok = pick
        i, j = map(int, pair.split("-"))
        wd = site.data / "external" / f"ext_leastcostpath_s{seed}"
        A = pd.read_csv(wd / "anchors.csv")
        anc = {int(r.id): np.array([r.x - site.x0, r.y - site.y0]) for r in A.itertuples()}
        P = pd.read_csv(wd / "paths.csv")
        polys = []
        best = None
        for _, g_ in P.groupby("path", sort=True):
            if tuple(sorted((int(g_.origin.iloc[0]), int(g_.destination.iloc[0])))) == (i, j):
                q = g_[["x", "y"]].to_numpy(float) - [site.x0, site.y0]
                Ln = np.hypot(*np.diff(q, axis=0).T).sum()
                if best is None or Ln < best[0]:
                    best = (Ln, q)
        polys.append(best[1])
        for _, _, ver in srcs[1:]:
            g = sb.crop(pickle.loads((site.cache / f"{sb.key_of('NA-SMC', ver)}_s{seed}.pkl").read_bytes())["graph"], mask, site)
            polys.append(Router(g).route(anc[i], anc[j]))
        allp = np.vstack(polys)
        pad = 120
        x0, x1, y0, y1 = allp[:, 0].min() - pad, allp[:, 0].max() + pad, allp[:, 1].min() - pad, allp[:, 1].max() + pad
        s_ = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax = row[0]
        ax.imshow(pale(sb.hillshade(site, big, mask), 0.55), extent=(-sb.MARGIN, site.L + sb.MARGIN, -sb.MARGIN, site.L + sb.MARGIN))
        sb.contours(ax, site, big, step=20, lw=0.3)
        for (lab, key, _), q, col in zip(srcs, polys, ROUTE_COL):
            ax.plot(*q.T, color=col, lw=2.6 if key.startswith("least") else 2.0, solid_capstyle="round", zorder=4)
        for k, pt in ((i, anc[i]), (j, anc[j])):
            ax.scatter(*pt, s=70, color=INK, zorder=5, edgecolor=SURF, linewidth=1.5)
            ax.annotate(f"anchor {k}", pt, xytext=(6, 6), textcoords="offset points", fontsize=9.5, zorder=6)
        ax.set_xlim(cx - s_ / 2, cx + s_ / 2)
        ax.set_ylim(cy - s_ / 2, cy + s_ / 2)
        ax.set_xticks([])
        ax.set_yticks([])
        straight = float(np.hypot(*(anc[i] - anc[j])))
        ax.set_title(f"{SITE_NAME[name]}: anchors {pair}, seed {seed}, {straight:.0f} m apart (median of the {n_ok} pairs all "
                     "three serve)\n20 m contours", loc="left", fontsize=10)
        ax = row[1]
        rec = d[(d.seed == seed) & (d.pair == pair)].set_index("network")
        for (lab, key, _), q, col in zip(srcs, polys, ROUTE_COL):
            seg = np.hypot(*np.diff(q, axis=0).T)
            m = np.maximum(1, np.ceil(seg / 10)).astype(int)
            Q = np.vstack([q[k] + np.outer(np.arange(m[k]) / m[k], q[k + 1] - q[k]) for k in range(len(seg))] + [q[-1:]])
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
            z = Tb.sample(Q[:, 0], Q[:, 1])
            r = rec.loc[key]
            ax.plot(s, z, color=col, lw=2, zorder=3, label=f"{lab}\n{s[-1]:.0f} m, detour {r.detour:.2f} · grade p90 "
                                                            f"{r.raw_grade_p90:.0f} % · > 15 %: {100 * r.raw_share_gt15:.0f} %")
        ax.set_xlabel("distance along the route (m)")
        ax.set_ylabel("elevation (m)")
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=True, facecolor=SURF, edgecolor="#e1e0d9", fontsize=9, labelspacing=0.9)
        ax.set_title("Elevation profile (native polylines, bare-earth proxy)", loc="left", fontsize=10)
        print(name, seed, pair, n_ok, flush=True)
    fig.suptitle("Routing between the same two anchors (one example pair per site; detour = route / straight distance, "
                 "grades on the native polylines)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "fig_routes.png", dpi=100, facecolor=SURF)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ["sites", "tiles", "routes"]
    for s in stages:
        {"sites": stage_sites, "tiles": stage_tiles, "routes": stage_routes}[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
