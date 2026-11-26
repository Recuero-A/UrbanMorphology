"""Milestone 18, phase 3 (first pass, GEN2): street distributions per model against the real networks of the same tiles.

Data: the 95 GEN2 tiles (Alcoi, Tarragona, Santa Barbara, Sacramento; LiDAR DEM on the 4 m grid with a 256 m margin)
with their real OSM networks (data/processed/m14/gen2/graphs3d_gen.pkl), and the stored graphs of the eight M14
confirmatory samplers (outputs/m14/runs: 3 TN backbones x 3 inference seeds; P has one backbone-free run x 3 seeds).
Development / already-inspected data only; nothing is generated here.

Distributions (pooled over tiles; measured inside the 512 m tile):
* grade: |dz| / length over pieces of at most 10 m along every GROUND edge (bridges and tunnels left out), DEM
  bilinear, weighted by piece length (%);
* link length: along-network length between junctions / dead ends through degree-2 nodes (m), one per link;
* nearest-junction distance: Euclidean distance from each focal junction (degree >= 3, >= 40 m inside the tile) to
  the nearest other junction (graphs/junctions.nn_distances), one per junction (m);
* block area and block compactness (Polsby-Popper 4 pi A / P^2): interior faces of the planar street graph, faces
  touching the tile edge left out, one per block;
* street density: street km per km2 inside the tile, one value per tile.

Same or different? Per metric, model and run: the Kolmogorov-Smirnov distance D between the pooled real and pooled
generated distributions (weighted ECDFs on fixed bins), with a paired permutation test that swaps real and generated
between matched 3 km blocks (999 swaps; the M14 inference unit). With tens of thousands of pieces almost any
difference is significant, so D is also set against two real-vs-real yardsticks: random halves of the real blocks
(median of 200 splits) and the Spanish against the US cities.

    python scripts/m18_distributions.py   ->  outputs/m18/distributions/{fig_distributions.png, fig_quantiles.png,
                                              tests.csv, summary.json, README.md}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import shapely

from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.graphs.junctions import nn_distances
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = ROOT / "outputs" / "m18" / "distributions"
CACHE = OUT / "cache"
TN = ("TN_s0", "TN_s1", "TN_s2")
MODELS = ("A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC", "P")
TILE = 512.0
KM2 = TILE * TILE / 1e6
METRICS = {   # key: (label, unit, bin edges, log x)
    "grade": ("Street grade", "%", np.r_[np.arange(0, 30.01, 1.0), 1e9], False),
    "link": ("Link length (junction to junction)", "m", np.r_[0, np.geomspace(5, 1500, 37), 1e9], True),
    "nn": ("Nearest-junction distance", "m", np.r_[0, np.geomspace(2, 500, 37), 1e9], True),
    "block_area": ("Block area", "m²", np.r_[0, np.geomspace(20, 3e5, 37), 1e12], True),
    "block_pp": ("Block compactness", "4πA/P²", np.linspace(0, 1, 41), False),
    "density": ("Street density per tile", "km/km²", np.r_[np.arange(0, 50.01, 2.5), 1e9], False),
}


def links(g) -> np.ndarray:
    from m18_audit_m14 import links as _links
    return _links(g)


def samples(g, T) -> dict:
    """Raw samples (values, weights) of every metric for one graph on one tile."""
    E = np.asarray(g.edges)
    P = np.asarray(g.nodes)[:, :2]
    out = {k: (np.zeros(0), np.zeros(0)) for k in METRICS}
    if len(E) == 0:
        out["density"] = (np.array([0.0]), np.array([1.0]))
        return out
    st = np.asarray(g.structs()) if hasattr(g, "structs") else np.zeros(len(E), int)
    A, B = P[E[:, 0]], P[E[:, 1]]
    L = np.hypot(*(B - A).T)
    # street length inside the tile (edges clipped by midpoint pieces of <= 10 m)
    n = np.maximum(1, np.ceil(L / 10.0)).astype(int)
    k = np.repeat(np.arange(len(E)), n)
    j = np.concatenate([np.arange(m) for m in n])
    t0, t1 = j / n[k], (j + 1) / n[k]
    p0 = A[k] + t0[:, None] * (B[k] - A[k])
    p1 = A[k] + t1[:, None] * (B[k] - A[k])
    mid = (p0 + p1) / 2
    inside = (mid[:, 0] >= 0) & (mid[:, 0] <= TILE) & (mid[:, 1] >= 0) & (mid[:, 1] <= TILE)
    pl = L[k] / n[k]
    out["density"] = (np.array([pl[inside].sum() / 1000.0 / KM2]), np.array([1.0]))
    ground = inside & (st[k] == 0) & (pl > 0.5)
    z0, z1 = T.sample(p0[ground, 0], p0[ground, 1]), T.sample(p1[ground, 0], p1[ground, 1])
    gr = np.abs(z1 - z0) / pl[ground] * 100.0
    ok = np.isfinite(gr)
    out["grade"] = (gr[ok], pl[ground][ok])
    lk = links(g)
    out["link"] = (lk, np.ones(len(lk)))
    nn = nn_distances(g)
    nn = nn[np.isfinite(nn)] if nn is not None else np.zeros(0)
    out["nn"] = (nn, np.ones(len(nn)))
    lines = shapely.node(shapely.multilinestrings(shapely.linestrings(P[E])))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    edge = shapely.box(0.5, 0.5, TILE - 0.5, TILE - 0.5).exterior
    box = shapely.box(0, 0, TILE, TILE)
    inner = [f for f in faces if f.area > 1.0 and not f.intersects(edge) and box.contains(f)]
    if inner:
        a = np.array([f.area for f in inner])
        pp = np.array([4 * np.pi * f.area / f.length ** 2 for f in inner])
        out["block_area"], out["block_pp"] = (a, np.ones(len(a))), (pp, np.ones(len(pp)))
    return out


def hist_of(s) -> dict:
    return {k: np.histogram(s[k][0], METRICS[k][2], weights=s[k][1])[0].astype(np.float64) for k in METRICS}


def ks(h1, h2) -> float:
    a, b = h1.sum(), h2.sum()
    if a <= 0 or b <= 0:
        return np.nan
    return float(np.max(np.abs(np.cumsum(h1) / a - np.cumsum(h2) / b)))


def quantiles(h, edges, qs=(0.1, 0.5, 0.9)) -> list:
    c = np.cumsum(h)
    if c[-1] <= 0:
        return [np.nan] * len(qs)
    c = c / c[-1]
    out = []
    for q in qs:
        i = int(np.searchsorted(c, q))
        lo, hi = edges[i], edges[min(i + 1, len(edges) - 1)]
        prev = c[i - 1] if i > 0 else 0.0
        f = (q - prev) / max(c[i] - prev, 1e-12)
        hi = hi if hi < 1e8 else lo * 1.5 + 1
        out.append(float(lo + f * (hi - lo)))
    return out


def build():
    """Per-tile histograms: real (95, nb) and model -> run -> (95, nb), cached."""
    from m14_build_gen2 import gen2_tiles
    from m14_report import get
    f = CACHE / "hist.npz"
    tiles = gen2_tiles()
    sids = [t["sid"] for t in tiles]
    blocks = [t["block"] for t in tiles]
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return sids, blocks, z["real"].item(), z["gen"].item()
    terr = {t["sid"]: terrain_of(t["d"]) for t in tiles}
    real = {k: np.stack([hist_of(samples(t["graph"], terr[t["sid"]]))[k] for t in tiles]) for k in METRICS}
    print("real done", flush=True)
    gen = {}
    for m in MODELS:
        gen[m] = []
        seen = set()
        for bb in TN:
            res = get("gen2", bb, m)
            if res is None or res["id"] in seen:
                continue
            seen.add(res["id"])
            for s in sorted(res["seeds"]):
                sd = res["seeds"][s]
                idx = {r["sid"]: i for i, r in enumerate(sd["rows"])}
                H = {k: np.zeros((len(tiles), len(METRICS[k][2]) - 1)) for k in METRICS}
                for ti, sid in enumerate(sids):
                    h = hist_of(samples(graph_from_json(sd["graphs"][idx[sid]]), terr[sid]))
                    for k in METRICS:
                        H[k][ti] = h[k]
                gen[m].append({"run": f"{res['spec']['backbone']}/seed{s}", **H})
        print("model done", m, len(gen[m]), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, real=np.array(real, dtype=object), gen=np.array(gen, dtype=object))
    return sids, blocks, real, gen


def perm_test(R, G, blocks, n=999, seed=7):
    """KS distance of pooled R vs pooled G (tile x bin) and a paired block-swap permutation p-value."""
    d0 = ks(R.sum(0), G.sum(0))
    ub = sorted(set(blocks))
    bi = np.array([ub.index(b) for b in blocks])
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        sw = rng.random(len(ub)) < 0.5
        m = sw[bi][:, None]
        R2, G2 = np.where(m, G, R), np.where(m, R, G)
        ge += ks(R2.sum(0), G2.sum(0)) >= d0
    return d0, (1 + ge) / (1 + n)


ROWLAB = {"grade": "Street grade\n(%, log scale)", "link": "Link length\n(m)", "nn": "Junction spacing\n(nearest, m)",
          "block_area": "Block area\n(m²)", "block_pp": "Block compactness\n(4πA/P²)", "density": "Street density\n(km/km² per tile)"}
TAILS = (10.0, 15.0, 20.0)


def tail_ci(real, gens, blocks, thr, n_boot=4000, seed=5):
    """Share of ground street length steeper than thr (%): real, model (mean of runs), and the model - real difference
    with a block-bootstrap 95 % interval (pooled ratio per resample)."""
    edges = METRICS["grade"][2]
    i = int(np.searchsorted(edges, thr))
    G = np.mean(gens, axis=0)
    ub = sorted(set(blocks))
    bi = np.array([ub.index(b) for b in blocks])
    ra, rt = np.bincount(bi, real[:, i:].sum(1)), np.bincount(bi, real.sum(1))
    ga, gt = np.bincount(bi, G[:, i:].sum(1)), np.bincount(bi, G.sum(1))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ub), len(ub))
        d.append(ga[k].sum() / gt[k].sum() - ra[k].sum() / rt[k].sum())
    return {"real": float(ra.sum() / rt.sum()), "model": float(ga.sum() / gt.sum()),
            "diff": float(ga.sum() / gt.sum() - ra.sum() / rt.sum()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "ratio": float((ga.sum() / gt.sum()) / max(ra.sum() / rt.sum(), 1e-12))}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    sids, blocks, real, gen = build()
    ub = sorted(set(blocks))
    bi = np.array([ub.index(b) for b in blocks])
    spain = np.array([s.startswith("ESP") for s in sids])
    rng = np.random.default_rng(11)
    floor = {}
    for k in METRICS:
        ds = []
        for _ in range(200):
            half = rng.random(len(ub)) < 0.5
            ds.append(ks(real[k][half[bi]].sum(0), real[k][~half[bi]].sum(0)))
        floor[k] = {"halves_median_D": float(np.nanmedian(ds)), "spain_vs_usa_D": ks(real[k][spain].sum(0), real[k][~spain].sum(0))}
    rows, summ = [], {"real": {k: quantiles(real[k].sum(0), METRICS[k][2]) for k in METRICS}, "floor": floor, "models": {}}
    for m in MODELS:
        summ["models"][m] = {}
        for k in METRICS:
            Ds, ps = [], []
            for r in gen[m]:
                d, p = perm_test(real[k], r[k], blocks)
                Ds.append(d)
                ps.append(p)
                rows.append({"metric": k, "model": m, "run": r["run"], "D": round(d, 4), "p": round(p, 4)})
            pooled = np.mean([r[k] for r in gen[m]], axis=0)
            summ["models"][m][k] = {"D_median": float(np.median(Ds)), "D_min": float(np.min(Ds)), "D_max": float(np.max(Ds)),
                                    "runs": len(Ds), "runs_p_lt_05": int(sum(p < 0.05 for p in ps)),
                                    "quantiles_10_50_90": quantiles(pooled.sum(0), METRICS[k][2])}
        summ["models"][m]["grade_tails"] = {f"gt{int(t)}": tail_ci(real["grade"], [r["grade"] for r in gen[m]], blocks, t)
                                             for t in TAILS}
        print("tested", m, flush=True)
    with open(OUT / "tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "run", "D", "p"])
        w.writeheader()
        w.writerows(rows)
    atomic_json(OUT / "summary.json", summ)

    # figure 1: small multiples, rows = metrics, columns = models; real grey fill, model line (mean of runs)
    ink, ink2, grid_c, accent, realc = "#1f2328", "#59636e", "#e6e8eb", "#2a78d6", "#b9bfc6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8c959f", "axes.linewidth": 0.6})
    keys = list(METRICS)
    fig, axs = plt.subplots(len(keys), len(MODELS), figsize=(2.35 * len(MODELS), 1.9 * len(keys)), sharey="row")
    for i, k in enumerate(keys):
        lab, unit, edges, logx = METRICS[k]
        e = edges.copy()
        e[-1] = e[-2] * (1.2 if logx else 1.0) + (0 if logx else (e[-2] - e[-3]))
        if e[0] == 0 and logx:
            e[0] = e[1] / 1.5
        wid = np.log10(e[1:] / e[:-1]) if logx else np.diff(e)
        rd = real[k].sum(0)
        rd = rd / max(rd.sum(), 1e-12) / wid
        for j, m in enumerate(MODELS):
            ax = axs[i, j]
            gd = np.mean([r[k] for r in gen[m]], axis=0).sum(0)
            gd = gd / max(gd.sum(), 1e-12) / wid
            ax.stairs(rd, e, fill=True, color=realc, alpha=0.9, lw=0)
            ax.stairs(gd, e, color=accent, lw=1.4)
            if logx:
                ax.set_xscale("log")
            if k == "grade":                                  # the steep tail, not the flat spike, is what matters
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 1.0)
            else:
                ax.set_ylim(0, 1.3 * max(rd.max(), gd.max()))
            ax.set_xlim(e[0], e[-2])
            ax.grid(axis="y", color=grid_c, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=2, colors=ink2, labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            s = summ["models"][m][k]
            left = k == "block_pp"
            ax.text(0.03 if left else 0.97, 0.95, f"D {s['D_median']:.2f}\n{'different' if s['runs_p_lt_05'] == s['runs'] else str(s['runs_p_lt_05']) + '/' + str(s['runs']) + ' p<.05'}",
                    transform=ax.transAxes, ha="left" if left else "right", va="top", fontsize=7.5, color=ink)
            if i == 0:
                ax.set_title(m, fontsize=10, color=ink, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ROWLAB[k], fontsize=8.5, color=ink)
            if i == len(keys) - 1:
                ax.set_xlabel(unit, fontsize=8, color=ink2)
    fig.suptitle("Generated vs real street distributions on the 95 GEN2 tiles (Alcoi, Tarragona, Santa Barbara, Sacramento)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=ink, y=0.993)
    fig.text(0.01, 0.958, "Grey: real OSM networks of the same tiles. Blue: the model, pooled over its runs (3 backbones × 3 seeds). "
             "D = Kolmogorov–Smirnov distance (median over runs); 'different' = p < 0.05 in every run (paired block-swap test).\n"
             "Grade on a log scale (the steep tail). Real-vs-real yardstick D: random halves of the real blocks / Spain vs USA, in the README.",
             fontsize=8.5, color=ink2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(OUT / "fig_distributions.png", dpi=130)
    plt.close(fig)

    # figure 2: p10 - p50 - p90 per model against the real band, one panel per metric
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6))
    names = ["real", *MODELS]
    for ax, k in zip(axs.ravel(), keys):
        lab, unit, edges, logx = METRICS[k]
        rq = summ["real"][k]
        ax.axvspan(rq[0], rq[2], color=realc, alpha=0.45, lw=0)
        ax.axvline(rq[1], color=ink2, lw=1.0, ls="--")
        for y, nm in enumerate(names):
            q = rq if nm == "real" else summ["models"][nm][k]["quantiles_10_50_90"]
            c = ink if nm == "real" else accent
            ax.plot([q[0], q[2]], [y, y], color=c, lw=2.0, solid_capstyle="round")
            ax.plot(q[1], y, "o", ms=6.5, color=c, mec="white", mew=1.2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9, color=ink)
        ax.invert_yaxis()
        if logx:
            from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
            ax.set_xscale("log")
            lo, hi = ax.get_xlim()
            nice = [v for v in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if lo <= v <= hi]
            ax.xaxis.set_major_locator(FixedLocator(nice))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{lab} ({unit})", fontsize=10.5, color=ink, loc="left", fontweight="bold")
        ax.grid(axis="x", color=grid_c, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2, colors=ink2, labelsize=8)
    fig.suptitle("Where each model sits: 10th, 50th and 90th percentile against the real range (grey band = real p10–p90, dashed = real median)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_quantiles.png", dpi=130)
    plt.close(fig)

    md = ["# M18: street distributions per model against real networks (GEN2, 95 tiles)", "",
          "Script `scripts/m18_distributions.py`; figures `fig_distributions.png`, `fig_quantiles.png`; per-run tests `tests.csv`.", "",
          "| Metric | real p10 / p50 / p90 | real-vs-real D (halves / Spain vs USA) | " + " | ".join(MODELS) + " |",
          "|---|---|---|" + "---|" * len(MODELS)]
    for k in keys:
        rq = summ["real"][k]
        cells = []
        for m in MODELS:
            s = summ["models"][m][k]
            q = s["quantiles_10_50_90"]
            cells.append(f"D {s['D_median']:.2f} ({s['runs_p_lt_05']}/{s['runs']}); {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g}")
        md.append(f"| {METRICS[k][0]} ({METRICS[k][1]}) | {rq[0]:.3g} / {rq[1]:.3g} / {rq[2]:.3g} | "
                  f"{floor[k]['halves_median_D']:.2f} / {floor[k]['spain_vs_usa_D']:.2f} | " + " | ".join(cells) + " |")
    md += ["", "## Steep tail of the grade distribution (share of ground street length steeper than the threshold)", "",
           "| Model | " + " | ".join(f"> {int(t)} %: model vs real (diff [95 % CI], ratio)" for t in TAILS) + " |",
           "|---|" + "---|" * len(TAILS)]
    for m in MODELS:
        cells = []
        for t in TAILS:
            c = summ["models"][m]["grade_tails"][f"gt{int(t)}"]
            cells.append(f"{c['model']:.4f} vs {c['real']:.4f} ({c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], ×{c['ratio']:.1f})")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md += ["", "Cells: KS distance D to the real distribution (median over runs), runs with p < 0.05 out of all runs, "
           "then the model's p10 / p50 / p90 (pooled over runs).",
           "Grades are measured on the LiDAR DEM, which carries the imprint of the real streets: real grades are those of the "
           "road surface, generated streets off the real street beds read the natural slope."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
