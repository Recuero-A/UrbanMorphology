"""Milestone 18, D11: leastcostpath on its own task, routing between shared anchors.

leastcostpath routes between anchors; it does not build a town. So it is compared where it is defined. For every pair
of anchors of its site runs (12 anchors per seed, farthest-point sampled on the gentler land inside the city limits,
seeds 1-3; m18_external.lcp_anchors), its least-cost route (wheeled-transport cost, critical slope 12 %, maximum 20 %,
16 neighbours; m18_external.LCP) is compared with the shortest street route between the SAME anchors on each network:
the real OSM vehicular network, the whole-town generators of the same seed, and the merged leastcostpath network.

Network route: each anchor is joined to the nearest street point (<= 60 m, else the pair is not served), then the
shortest path by length. Every route gets the common smoothing of m18_terrain_reference (30 m resampling, <= 10 m
pieces) and is measured on the site benchmark's bare-earth proxy (sigma 16 m, as TAS): length over the straight anchor
distance (detour), climb (sum of positive rises, mean of the two directions) per km of straight distance, route grade
p90 (length-weighted), share of route length above 15 %, and the longest sustained climb (m18_terrain_potentials.
profile_runs: > 5 % in one direction). Unordered pairs; the leastcostpath route of a pair = the shorter of its two
directions; a pair it cannot route (every path above the 20 % maximum) is not served by it.

    python scripts/m18_routing.py  ->  outputs/m18/routing/{routing.json, README.md, fig_routing.png}
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

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "routing"
SITES = ("machu_picchu", "petra")
JOIN_M = 60.0
NETWORKS = [("real (OSM vehicular)", None, None), ("NA-SMC [TN]", "NA-SMC", "original"),
            ("NA-SMC [TN] + terrain potentials", "NA-SMC", "TM"), ("NA-SMC [TNg]", "NA-SMC", "TNg"),
            ("NA-SMC [TNg] + terrain potentials", "NA-SMC", "TNg + TM"), ("M*", "M*", "original"), ("P", "P", "original"),
            ("citygen P&M (ext)", "citygen P&M (ext)", "original"), ("leastcostpath merged network (ext)",
                                                                       "leastcostpath (ext)", "original")]


def profile(poly, sample, smooth=True):
    """Route profile (piece lengths and rises, m): smoothed (30 m resampling) or on the native polyline."""
    from m18_terrain_reference import resample
    q = resample(np.asarray(poly, float)) if smooth else np.asarray(poly, float)
    if q is None or len(q) < 2:
        return np.zeros(0), np.zeros(0)
    P = []
    for a, b in zip(q[:-1], q[1:]):
        m = max(1, int(np.ceil(np.hypot(*(b - a)) / 10.0)))
        P += [a + (b - a) * t for t in np.arange(m) / m]
    P = np.array(P + [q[-1]])
    return np.hypot(*np.diff(P, axis=0).T), np.diff(sample(P[:, 0], P[:, 1]))


def route_metrics(poly, straight, sample):
    m = _metrics(poly, straight, sample, True)
    if m is not None:
        raw = _metrics(poly, straight, sample, False)
        m.update({f"raw_{k}": raw[k] for k in ("grade_p90", "share_gt15")} if raw else {})
    return m


def _metrics(poly, straight, sample, smooth):
    from m18_terrain_potentials import profile_runs
    ln, dz = profile(poly, sample, smooth)
    ok = np.isfinite(dz) & (ln > 0.5)
    ln, dz = ln[ok], dz[ok]
    if ln.sum() <= 0:
        return None
    gr = np.abs(dz) / ln * 100
    o = np.argsort(gr)
    c = np.cumsum(ln[o]) / ln.sum()
    runs, _ = profile_runs(ln, dz, np.ones(len(ln), bool))
    return {"detour": float(ln.sum() / straight), "climb_per_km": float((dz[dz > 0].sum() - dz[dz < 0].sum()) / 2 / (straight / 1000)),
            "grade_p90": float(gr[o][min(int(np.searchsorted(c, 0.9)), len(o) - 1)]),
            "share_gt15": float(ln[gr > 15].sum() / ln.sum()), "longest_climb_m": float(max(runs) if runs else 0.0),
            "length_m": float(ln.sum())}


class Router:
    """Shortest street routes between points joined to the nearest street point."""

    def __init__(self, g):
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree
        self.g = g
        a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
        self.L = np.hypot(*(b - a).T)
        m = np.maximum(1, np.ceil(self.L / 2.0)).astype(int)
        self.e = np.repeat(np.arange(len(self.L)), m + 1)
        self.t = np.concatenate([np.arange(k + 1) / k for k in m])
        self.Q = a[self.e] + self.t[:, None] * (b - a)[self.e]
        self.tree = cKDTree(self.Q)
        n = len(g.nodes)
        self.A = coo_matrix((np.r_[self.L, self.L], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])),
                            shape=(n, n)).tocsr()
        self.cache = {}

    def join(self, p):
        d, k = self.tree.query(p)
        return float(d), int(self.e[k]), float(self.t[k]), self.Q[k]

    def sp(self, u):
        if u not in self.cache:
            from scipy.sparse.csgraph import dijkstra
            self.cache[u] = dijkstra(self.A, indices=u, return_predecessors=True)
        return self.cache[u]

    def route(self, p, q):
        """Polyline from p to q through the network, or None (not joined within JOIN_M, or not connected)."""
        dp, ep, tp_, P = self.join(p)
        dq, eq, tq, Qp = self.join(q)
        if dp > JOIN_M or dq > JOIN_M:
            return None
        E, N = self.g.edges, self.g.nodes
        if ep == eq:
            return np.array([p, P, Qp, q])
        best = None
        for u, du in ((E[ep, 0], tp_ * self.L[ep]), (E[ep, 1], (1 - tp_) * self.L[ep])):
            D, pred = self.sp(int(u))
            for v, dv in ((E[eq, 0], tq * self.L[eq]), (E[eq, 1], (1 - tq) * self.L[eq])):
                tot = D[v] + du + dv
                if np.isfinite(tot) and (best is None or tot < best[0]):
                    best = (tot, int(u), int(v), pred)
        if best is None:
            return None
        _, u, v, pred = best
        path = [v]
        while path[-1] != u:
            path.append(int(pred[path[-1]]))
        return np.vstack([p, P, N[path[::-1]], Qp, q])


def main() -> int:
    import pandas as pd
    import site_benchmark as sb
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in SITES:
        sb.init(name)
        site, mask, Tb = sb.G["site"], sb.G["mask"], sb.G["Tb"]
        for seed in sb.SEEDS:
            wd = site.data / "external" / f"ext_leastcostpath_s{seed}"
            A = pd.read_csv(wd / "anchors.csv")
            anchors = {int(r.id): np.array([r.x - site.x0, r.y - site.y0]) for r in A.itertuples()}
            df = pd.read_csv(wd / "paths.csv")
            lcp = {}
            for _, g_ in df.groupby("path", sort=True):
                o, d = int(g_.origin.iloc[0]), int(g_.destination.iloc[0])
                poly = g_[["x", "y"]].to_numpy(float) - [site.x0, site.y0]
                key = tuple(sorted((o, d)))
                Ln = np.hypot(*np.diff(poly, axis=0).T).sum()
                if key not in lcp or Ln < lcp[key][0]:
                    lcp[key] = (Ln, poly)
            ids = sorted(anchors)
            pairs = [(i, j) for k, i in enumerate(ids) for j in ids[k + 1:]]
            routers = {}
            for lab, meth, ver in NETWORKS:
                if meth is None:
                    g = site.real("vehicular")
                else:
                    f = site.cache / f"{sb.key_of(meth, ver)}_s{seed}.pkl"
                    if not f.exists():
                        continue
                    g = pickle.loads(f.read_bytes())["graph"]
                g = sb.crop(g, mask, site)
                if len(g.edges):
                    routers[lab] = Router(g)
            for i, j in pairs:
                a, b = anchors[i], anchors[j]
                straight = float(np.hypot(*(b - a)))
                cand = {"leastcostpath route (ext)": lcp[(i, j)][1] if (i, j) in lcp else None}
                cand.update({lab: R.route(a, b) for lab, R in routers.items()})
                for lab, poly in cand.items():
                    m = route_metrics(poly, straight, Tb.sample) if poly is not None else None
                    rows.append({"site": name, "seed": seed, "pair": f"{i}-{j}", "straight_m": straight, "network": lab,
                                 "served": m is not None, **(m or {})})
            print(name, seed, len(pairs), "pairs", sorted(routers), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "routes.csv", index=False)
    summ = []
    for (name, lab), d in df.groupby(["site", "network"], sort=False):
        s = d[d.served]
        summ.append({"site": name, "network": lab, "pairs": len(d), "served": float(d.served.mean()),
                     **{f"{k}_p50": float(s[k].median()) if len(s) else np.nan for k in ("detour", "climb_per_km", "grade_p90", "longest_climb_m")},
                     "share_gt15_mean": float(s.share_gt15.mean()) if len(s) else np.nan,
                     "raw_grade_p90_p50": float(s.raw_grade_p90.median()) if len(s) else np.nan,
                     "raw_share_gt15_mean": float(s.raw_share_gt15.mean()) if len(s) else np.nan,
                     "grade_p90_p90": float(s.grade_p90.quantile(0.9)) if len(s) else np.nan})
    # pairs served by BOTH leastcostpath and a network: paired medians of the network / leastcostpath ratios
    lc = df[df.network == "leastcostpath route (ext)"].set_index(["site", "seed", "pair"])
    for r in summ:
        if r["network"] == "leastcostpath route (ext)":
            continue
        d = df[(df.site == r["site"]) & (df.network == r["network"]) & df.served].set_index(["site", "seed", "pair"])
        j = d.join(lc.add_suffix("_lcp"), how="inner")
        j = j[j.served_lcp]
        r["both_served"] = int(len(j))
        r["detour_ratio_vs_lcp_p50"] = float((j.detour / j.detour_lcp).median()) if len(j) else np.nan
        r["climb_ratio_vs_lcp_p50"] = float((j.climb_per_km / j.climb_per_km_lcp).median()) if len(j) else np.nan
    atomic_json(OUT / "routing.json", {"summary": summ})
    md = ["# M18 D11: leastcostpath on a routing task with shared anchors", "",
          "Anchors: the 12 farthest-point anchors of each leastcostpath site run (seeds 1-3), 66 unordered pairs per seed. "
          "Network routes: shortest path by length after joining each anchor to the nearest street point (<= 60 m). All "
          "routes smoothed alike (30 m resampling, 10 m pieces) and measured on the bare-earth proxy. Script "
          "`scripts/m18_routing.py`.", "",
          "Unsmoothed = the same measures on the native polylines (sensitivity: 30 m resampling turns switchbacks "
          "into chords up the slope).", "",
          "| Site | Route source | served | detour p50 | climb m per km p50 | route grade p90 (median) | share > 15 % | unsmoothed grade p90 | unsmoothed share > 15 % | longest climb p50 (m) | detour / LCP | climb / LCP |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        md.append(f"| {r['site']} | {r['network']} | {r['served']:.2f} | {r['detour_p50']:.2f} | {r['climb_per_km_p50']:.0f} | "
                  f"{r['grade_p90_p50']:.1f} | {r['share_gt15_mean']:.3f} | {r['raw_grade_p90_p50']:.1f} | "
                  f"{r['raw_share_gt15_mean']:.3f} | {r['longest_climb_m_p50']:.0f} | "
                  f"{r.get('detour_ratio_vs_lcp_p50', np.nan):.2f} | {r.get('climb_ratio_vs_lcp_p50', np.nan):.2f} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
