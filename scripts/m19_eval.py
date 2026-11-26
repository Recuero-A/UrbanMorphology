"""Milestone 19, D5: evaluation of the development arms (docs/M19_LOG.md D5).

Primary endpoint: steep-land geographic access. Fixed points per unit (seed crc32(unit id), never shown to a generator):
40 on steep eligible land (15-35 %) and 40 on any eligible land (< 35 %); sites scaled by eligible area / 0.262 km2.
Pairs: at least one steep endpoint, 150-500 m apart, on the same dry-land region. Served: both endpoints within 60 m of
a street, joined by the network (closed window on tiles, open graph for whole towns), detour <= 4 where the path lies
inside the unit (a tile pair joined only through the outside counts as served, flagged). Failure modes: endpoint not
reached, disconnected, detour > 4. Preservation: flat-land density, steep-land coverage, grade p90 (F22 smoothing);
independent: sustained climbs. Matched length: A5's access interpolated at each online arm's street length.

    measure   every run file and A5 snapshot                                    ->  measures/<run>.pkl.gz
    report    contrasts (block bootstrap over the tile units; whole towns per site x seed), margins, selection,
              compute, figure                                                   ->  report.json, README.md, fig_access_length.png

    python scripts/m19_eval.py measure|report [--workers 7]  ->  outputs/m19/arms/
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m19_arms import ARMS, BBS, OUT, SITE_SEEDS, TILE_SEEDS, run_file, unit_tiles
from urbanmorph.experiments.jsonio import atomic_json

MEAS = OUT / "measures"
N_STEEP, N_ANY, D_LO, D_HI, JOIN_M, DETOUR_MAX = 40, 40, 150.0, 500.0, 60.0, 4.0
STEEP_LO, STEEP_HI = 15.0, 35.0
TILE_UNITS, SITE_UNITS = ("gen2", "natural"), ("machu_picchu", "petra")
ONLINE = ("A1", "A2", "A3", "A4")
SIMPLICITY = ("A5", "A4", "A2", "A1", "A3")


# ---------------------------------------------------------------- fields and points
def fields(unit):
    """[(unit id, block, field, closed window, crop)] for a unit set."""
    if unit in TILE_UNITS:
        from m18_terrain_reference import Field
        return [(t["sid"], str(t["block"]), Field(t["d"]), True, None) for t in unit_tiles(unit)]
    import site_benchmark as sb
    from m18_terrain_potentials import SiteField
    sb.init(unit)
    site, big, mask = sb.G["site"], sb.G["big"], sb.G["mask"]
    return [(unit, unit, SiteField(site, big, mask), False, lambda g: sb.crop(g, mask, site))]


def od_points(uid, F):
    import zlib as z_
    from scipy import ndimage
    elig = F.land & (F.slope < STEEP_HI)
    steep = elig & (F.slope >= STEEP_LO)
    reg, _ = ndimage.label(F.land, structure=np.ones((3, 3)))
    rng = np.random.default_rng(z_.crc32(uid.encode()))
    scale = 1.0 if F.land.shape[0] == 256 else float(elig.sum() * 16 / 1e6) / 0.262144
    res = F.S.res

    def pick(m, n):
        r, c = np.nonzero(m)
        if not len(r):
            return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, bool)
        k = rng.choice(len(r), size=min(int(round(n)), len(r)), replace=False)
        x = F.S.x0 + (c[k] + 0.5) * res + rng.uniform(-res / 2, res / 2, len(k))
        y = F.S.y1 - (r[k] + 0.5) * res + rng.uniform(-res / 2, res / 2, len(k))
        return np.c_[x, y], reg[r[k], c[k]], steep[r[k], c[k]]
    P1, R1, S1 = pick(steep, N_STEEP * scale)
    P2, R2, S2 = pick(elig, N_ANY * scale)
    P, R, S = np.vstack([P1, P2]), np.r_[R1, R2], np.r_[S1, S2]
    if len(P) < 2:
        return P, np.zeros(0, int), np.zeros(0, int)
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    ok = np.triu((D >= D_LO) & (D <= D_HI) & (R[:, None] == R[None]) & (R[:, None] > 0) & (S[:, None] | S[None]), 1)
    I, J = np.nonzero(ok)
    return P, I, J


def served(g, P, I, J, closed):
    """Per-pair outcome: 0 served, 1 an endpoint not reached, 2 disconnected, 3 detour > 4; and the via-outside flag."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree
    out = np.ones(len(I), int)
    via = np.zeros(len(I), bool)
    if not len(I) or len(g.edges) == 0:
        return out, via
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    L = np.hypot(*(b - a).T)
    m = np.maximum(1, np.ceil(L / 2.0)).astype(int)
    e = np.repeat(np.arange(len(L)), m + 1)
    t = np.concatenate([np.arange(k + 1) / k for k in m])
    dist, idx = cKDTree(a[e] + t[:, None] * (b - a)[e]).query(P)
    joined = dist <= JOIN_M
    both = joined[I] & joined[J]
    n = len(g.nodes)
    if closed:
        from urbanmorph.graphs.m14_metrics import closed_components
        lab, _, A = closed_components(g)
        A = A[:n, :n]
    else:
        A = coo_matrix((np.r_[L, L], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])), shape=(n, n)).tocsr()
        lab = connected_components(A, directed=False)[1]
    ea, te = e[idx], t[idx]
    na, nb = g.edges[ea, 0], g.edges[ea, 1]
    conn = both & (lab[na[I]] == lab[na[J]])
    out[both & ~conn] = 2
    src = np.unique(np.r_[na[I[conn]], nb[I[conn]]])
    row = {s: k for k, s in enumerate(src)}
    Dn = dijkstra(A, indices=src) if len(src) else None
    for q in np.flatnonzero(conn):
        i, j = I[q], J[q]
        if ea[i] == ea[j]:
            net = abs(te[i] - te[j]) * L[ea[i]]
        else:
            net = min(Dn[row[u], v] + du + dv for u, du in ((na[i], te[i] * L[ea[i]]), (nb[i], (1 - te[i]) * L[ea[i]]))
                      for v, dv in ((na[j], te[j] * L[ea[j]]), (nb[j], (1 - te[j]) * L[ea[j]])))
        straight = float(np.hypot(*(P[i] - P[j])))
        if not np.isfinite(net):
            out[q], via[q] = 0, True                      # joined only through the outside of the tile
        else:
            out[q] = 0 if (net + dist[i] + dist[j]) / straight <= DETOUR_MAX else 3
    return out, via


def length_m(g):
    return float(np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T).sum()) if len(g.edges) else 0.0


# ---------------------------------------------------------------- measure
def measure_one(spec):
    unit, arm, bb, seed = spec
    from m18_terrain_potentials import climb_runs
    from m18_terrain_reference import tile_sums
    from urbanmorph.experiments.graph_io import graph_from_json
    f = MEAS / f"{unit}__{arm}__{bb}__s{seed}.pkl.gz"
    if f.exists():
        return spec, "cached"
    res = pickle.loads(zlib.decompress(run_file(unit, arm, bb, seed).read_bytes()))
    FL = {uid: (blk, F, closed, crop) for uid, blk, F, closed, crop in fields(unit)}
    rows = []
    for k, uid in enumerate(res["ids"]):
        blk, F, closed, crop = FL[uid]
        P, I, J = od_points(uid, F)
        g = graph_from_json(res["graphs"][k])
        gc = crop(g) if crop else g
        if crop:
            mask_in = lambda M, F=F: F.land[np.clip(((F.S.y1 - M[:, 1]) / F.S.res).astype(int), 0, F.land.shape[0] - 1),
                                            np.clip(((M[:, 0] - F.S.x0) / F.S.res).astype(int), 0, F.land.shape[1] - 1)]
        else:
            mask_in = lambda M: (M[:, 0] >= 0) & (M[:, 0] <= 512) & (M[:, 1] >= 0) & (M[:, 1] <= 512)
        o, via = served(gc, P, I, J, closed)
        runs, _, tot = climb_runs(gc, F.T.sample, mask_in)
        row = {"uid": uid, "block": blk, "pairs": len(I), "out": np.bincount(o, minlength=4).tolist(), "via": int(via.sum()),
               "length": length_m(gc), "sums": tile_sums(gc, F), "runs": runs, "climb_tot": tot,
               "src": res.get("src_edges", [[]] * len(res["ids"]))[k]}
        if arm == "A0":
            curve = []
            for added, gj in res["A5"]["snapshots"][k]:
                g5 = graph_from_json(gj)
                g5 = crop(g5) if crop else g5
                o5, _ = served(g5, P, I, J, closed)
                curve.append((length_m(g5), int((o5 == 0).sum()), np.bincount(o5, minlength=4).tolist()))
            row["A5"] = curve
            last = graph_from_json(res["A5"]["snapshots"][k][-1][1])
            last = crop(last) if crop else last
            row["A5_sums"] = tile_sums(last, F)
            r5, _, t5 = climb_runs(last, F.T.sample, mask_in)
            row["A5_runs"], row["A5_climb_tot"] = r5, t5
        rows.append(row)
    comp = {"seconds": res["seconds"], "forward_passes": res["forward_passes"], "mech": res["mech"],
            "cand_evals": sum(st.get("cand_evals", 0) for st in res["stats"]),
            "la_steps": sum(st.get("lookahead_steps", 0) for st in res["stats"]),
            "rounds": sum(st.get("ess_n", 0) for st in res["stats"]),
            "chosen": {k_: sum(st.get(k_, 0) for st in res["stats"]) for k_ in ("chosen_router", "chosen_big")}}
    if arm == "A0":
        comp["A5"] = {"seconds": res["A5"]["seconds"], "log": res["A5"]["log"]}
    MEAS.mkdir(parents=True, exist_ok=True)
    f.write_bytes(zlib.compress(pickle.dumps({"spec": spec, "rows": rows, "compute": comp}), 6))
    return spec, "done"


def specs():
    out = []
    for unit in TILE_UNITS + SITE_UNITS:
        bbs, seeds = (BBS, TILE_SEEDS) if unit in TILE_UNITS else (("TNg_s0",), SITE_SEEDS)
        out += [(unit, a, b, s) for a in ARMS for b in bbs for s in seeds if run_file(unit, a, b, s).exists()]
    return out


def stage_measure(workers):
    from multiprocessing import Pool
    with Pool(workers, maxtasksperchild=4) as p:
        for spec, how in p.imap_unordered(measure_one, specs(), chunksize=1):
            print(spec, how, flush=True)
    return 0


# ---------------------------------------------------------------- report
def load():
    D = {}
    for f in sorted(MEAS.glob("*.pkl.gz")):
        x = pickle.loads(zlib.decompress(f.read_bytes()))
        D[tuple(x["spec"])] = x
    return D


def a5_at(curve, L):
    """A5's served count at street length L (linear along its snapshots; A0 below A0's length; capped at the last)."""
    Ls = np.array([c[0] for c in curve])
    S = np.array([c[1] for c in curve], float)
    if L <= Ls[0]:
        return S[0], False
    if L >= Ls[-1]:
        return S[-1], L > Ls[-1] + 1.0
    return float(np.interp(L, Ls, S)), False


def pooled_stats(rows, w, sums_key="sums", runs_key="runs", tot_key="climb_tot"):
    """Steep access and preservation statistics of weighted rows (the bootstrap weights)."""
    from m18_terrain_reference import GBINS, KEYS, pool
    pairs = sum(wi * r["pairs"] for r, wi in zip(rows, w))
    outs = np.sum([wi * np.array(r["out"]) for r, wi in zip(rows, w)], axis=0)
    t = {k: sum(wi * np.asarray(r[sums_key][k], float) for r, wi in zip(rows, w)) for k in rows[0][sums_key]}
    cur = pool([t])
    h = t["hist"].sum(0)
    c = np.cumsum(h) / max(h.sum(), 1e-9)
    gp90 = float(GBINS[min(int(np.searchsorted(c, 0.9)), len(GBINS) - 2)] + 0.25) if h.sum() > 0 else float("nan")
    tot = sum(wi * r[tot_key] for r, wi in zip(rows, w))
    long_ = sum(wi * float(np.asarray(r[runs_key])[np.asarray(r[runs_key]) >= 100].sum()) for r, wi in zip(rows, w))
    nruns = sum(wi * len(r[runs_key]) for r, wi in zip(rows, w))
    return {"access": float(outs[0] / max(pairs, 1e-9)), "unreached": float(outs[1] / max(pairs, 1e-9)),
            "disconnected": float(outs[2] / max(pairs, 1e-9)), "detour": float(outs[3] / max(pairs, 1e-9)),
            "density_0-5": cur["0-5"]["density"], "density_5-10": cur["5-10"]["density"],
            "coverage_20-30": cur["20-30"]["coverage"], "coverage_30-inf": cur["30-inf"]["coverage"],
            "grade_p90": gp90, "share_in_climbs_ge100": float(long_ / max(tot, 1e-9)), "climbs_per_km": float(nruns / max(tot / 1000, 1e-9)),
            "km": float(sum(wi * r["length"] for r, wi in zip(rows, w)) / 1000)}


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = load()
    ref = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    sd = ref["between_city_sd"]
    rep = {"rule": "docs/M19_LOG.md D5", "tile": {}, "whole_town": {}, "compute": {}}
    # ---- tile scale: rows keyed by (unit, uid, bb, seed); bootstrap over blocks
    T = {}
    for (unit, arm, bb, seed), x in D.items():
        if unit not in TILE_UNITS:
            continue
        for r in x["rows"]:
            T.setdefault(arm, {})[(unit, r["uid"], bb, seed)] = {**r, "block": f"{unit}:{r['block']}"}
    arms = [a for a in ARMS if a in T]
    keys = sorted(set.intersection(*[set(T[a]) for a in arms])) if arms else []
    blocks = sorted({T["A0"][k]["block"] for k in keys})
    bidx = {b: i for i, b in enumerate(blocks)}
    kb = np.array([bidx[T["A0"][k]["block"]] for k in keys])
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks))[kb].astype(float) for _ in range(1000)]
    one = np.ones(len(keys))
    rows = {a: [T[a][k] for k in keys] for a in arms}
    rows5 = [{**T["A0"][k], "out": T["A0"][k]["A5"][-1][2], "length": T["A0"][k]["A5"][-1][0], "sums": T["A0"][k]["A5_sums"],
              "runs": T["A0"][k]["A5_runs"], "climb_tot": T["A0"][k]["A5_climb_tot"]} for k in keys]
    stat = lambda a, w: pooled_stats(rows5 if a == "A5" else rows[a], w)
    level = {a: stat(a, one) for a in arms + ["A5"]}
    rep["tile"]["units"] = {"rows": len(keys), "blocks": len(blocks)}
    rep["tile"]["level"] = level

    SD = {a: [stat(a, w) for w in draws] for a in arms + ["A5"]}     # every arm's statistics, once per draw

    def contrast2(a, b, k_):
        d0 = level[a][k_] - level[b][k_]
        bs = np.array([x[k_] - y[k_] for x, y in zip(SD[a], SD[b])])
        return {"delta": float(d0), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}

    def matched(a, w):
        """Online arm a minus A5 at a's street length, per row; pooled served share difference."""
        sv_a = sum(wi * r["out"][0] for r, wi in zip(rows[a], w))
        sv_5 = sum(wi * a5_at(T["A0"][k]["A5"], r["length"])[0] for k, r, wi in zip(keys, rows[a], w))
        pairs = sum(wi * r["pairs"] for r, wi in zip(rows[a], w))
        return (sv_a - sv_5) / max(pairs, 1e-9)
    C = {}
    for a in arms[1:] + ["A5"]:
        C[f"{a}-A0"] = {k_: contrast2(a, "A0", k_) for k_ in ("access", "density_0-5", "density_5-10", "coverage_20-30",
                                                             "coverage_30-inf", "grade_p90", "share_in_climbs_ge100", "km")}
    for a in [x for x in ONLINE if x in arms]:
        d0 = matched(a, one)
        bs = np.array([matched(a, w) for w in draws])
        C[f"{a}-A5@matched"] = {"access": {"delta": float(d0), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                           "rows_beyond_A5_range": int(sum(a5_at(T["A0"][k]["A5"], r["length"])[1] for k, r in zip(keys, rows[a])))}}
        if "A4" in arms and a != "A4":
            C[f"{a}-A4"] = {"access": contrast2(a, "A4", "access")}
    rep["tile"]["contrasts"] = C
    # margins: grade from A0's per-row spread across its 6 runs
    g_rows = {}
    for k in keys:
        g_rows.setdefault((k[0], k[1]), []).append(pooled_stats([T["A0"][k]], [1.0])["grade_p90"])
    g_sd = float(np.nanmedian([np.nanstd(v, ddof=1) for v in g_rows.values() if len(v) > 1]))
    gm = max(1.0, g_sd)
    rep["tile"]["grade_margin_pp"] = {"margin": gm, "A0_per_unit_sd_median": g_sd}

    def qualifies(a):
        c = C[f"{a}-A0"]
        m = {"access_gain": c["access"]["lo"] > 0,
             "flat_0-5": abs(c["density_0-5"]["delta"]) <= sd["0-5"]["density"],
             "flat_5-10": abs(c["density_5-10"]["delta"]) <= sd["5-10"]["density"],
             "steep_cov_20-30": c["coverage_20-30"]["lo"] >= -0.02, "steep_cov_30+": c["coverage_30-inf"]["lo"] >= -0.02,
             "grade_p90": c["grade_p90"]["delta"] <= gm}
        return all(m.values()), m
    Q = {a: qualifies(a) for a in arms[1:] + ["A5"]}
    rep["tile"]["qualifies"] = {a: {"all": q[0], **q[1]} for a, q in Q.items()}
    online_win = [a for a in ("A1", "A3") if a in arms and Q[a][0] and C[f"{a}-A5@matched"]["access"]["lo"] > 0
                  and ("A4" not in arms or C[f"{a}-A4"]["access"]["lo"] > 0)]
    if online_win:
        sel = min(online_win, key=SIMPLICITY.index)
        verdict = f"online routing selected ({sel}): qualifies and beats A5 at matched length and A4"
    else:
        qual = [a for a in SIMPLICITY if a in Q and Q[a][0]]
        sel = qual[0] if qual else None
        verdict = f"simplest qualifier selected ({sel})" if sel else "no arm qualifies: the round closes; report the trade-off"
    rep["tile"]["selection"] = {"selected": sel, "verdict": verdict}
    # ---- whole towns: per site x seed
    W = {}
    for (unit, arm, bb, seed), x in D.items():
        if unit in SITE_UNITS:
            W[(arm, unit, seed)] = x["rows"][0]
    wt = {}
    for unit in SITE_UNITS:
        for seed in SITE_SEEDS:
            r0 = W.get(("A0", unit, seed))
            if r0 is None:
                continue
            row = {"A0": r0["out"][0] / max(r0["pairs"], 1), "A0_km": r0["length"] / 1000,
                   "A5": r0["A5"][-1][1] / max(r0["pairs"], 1), "A5_km": r0["A5"][-1][0] / 1000, "pairs": r0["pairs"]}
            for a in ONLINE:
                r = W.get((a, unit, seed))
                if r is not None:
                    row[a] = r["out"][0] / max(r["pairs"], 1)
                    row[a + "_km"] = r["length"] / 1000
                    row[a + "_vs_A5_matched"] = (r["out"][0] - a5_at(r0["A5"], r["length"])[0]) / max(r["pairs"], 1)
            wt[f"{unit}:s{seed}"] = row
    rep["whole_town"]["per_site_seed"] = wt
    rep["whole_town"]["all_positive_vs_A0"] = {a: bool(wt) and all(v.get(a, -1) > v["A0"] for v in wt.values()) for a in ONLINE + ("A5",)}
    rep["whole_town"]["all_positive_vs_A5_matched"] = {a: bool(wt) and all(v.get(a + "_vs_A5_matched", -1) > 0 for v in wt.values()) for a in ONLINE}
    # ---- compute
    for (unit, arm, bb, seed), x in D.items():
        c = rep["compute"].setdefault(f"{'tile' if unit in TILE_UNITS else 'whole_town'}:{arm}", {"seconds": 0.0, "forward_passes": 0, "cand_evals": 0,
                                                                                                 "router_calls": 0, "chosen_router": 0, "chosen_big": 0})
        c["seconds"] += x["compute"]["seconds"]
        c["forward_passes"] += x["compute"]["forward_passes"]
        c["cand_evals"] += x["compute"]["cand_evals"]
        c["router_calls"] += x["compute"]["mech"].get("router_calls", 0)
        c["chosen_router"] += x["compute"]["chosen"]["chosen_router"]
        c["chosen_big"] += x["compute"]["chosen"]["chosen_big"]
        if arm == "A0":
            c5 = rep["compute"].setdefault(f"{'tile' if unit in TILE_UNITS else 'whole_town'}:A5", {"seconds": 0.0, "router_calls": 0})
            c5["seconds"] += x["compute"]["A5"]["seconds"]
            c5["router_calls"] += sum(l_.get("router_calls", 0) for l_ in x["compute"]["A5"]["log"])
    atomic_json(OUT / "report.json", rep)
    # ---- figure: access vs street length
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    col = {"A0": "#52514e", "A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#6250d6", "A5": "#898781"}
    ax = axs[0]
    if keys:
        phis = np.linspace(0, 1, 11)
        xs, ys = [], []
        for ph in phis:
            sv, pr, ln = 0.0, 0, 0.0
            for k in keys:
                cv = T["A0"][k]["A5"]
                Lt = cv[0][0] + ph * (cv[-1][0] - cv[0][0])
                sv += a5_at(cv, Lt)[0]
                pr += T["A0"][k]["pairs"]
                ln += Lt
            xs.append(ln / 1000)
            ys.append(sv / max(pr, 1))
        ax.plot(xs, ys, color=col["A5"], lw=2, label="A5: A0 + the router after growth")
        for a in arms:
            ax.scatter(level[a]["km"], level[a]["access"], s=70, color=col[a], zorder=3, label=a)
        ax.set_xlabel("total street length, all tile units and runs (km)")
        ax.set_ylabel("steep-land pairs served")
        ax.set_title("Tile scale (39 GEN2 tiles + 16 natural windows, TNg s0-s2 x seeds 1-2)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
    ax = axs[1]
    for k_, v in wt.items():
        ax.plot([v["A0_km"], v["A5_km"]], [v["A0"], v["A5"]], color=col["A5"], lw=1)
        for a in ("A0",) + ONLINE:
            if a in v:
                ax.scatter(v[a + "_km"] if a != "A0" else v["A0_km"], v[a], s=40, color=col[a], zorder=3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=col[a], lw=0, marker="o", ms=7, label=a) for a in ("A0",) + ONLINE]
              + [Line2D([], [], color=col["A5"], lw=1.5, label="A0 to A5 (router after growth)")], frameon=False, fontsize=8.5)
    ax.set_xlabel("street length inside the city limits (km)")
    ax.set_title("Whole towns (Machu Picchu, Petra; seeds 1-3; lines: A0 to A5)", loc="left", fontsize=10)
    for a_ in axs:
        a_.grid(color="#e1e0d9", lw=0.6)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_access_length.png", dpi=110)
    # ---- README
    f3 = lambda c: f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
    md = ["# M19 D5: development comparison of the arms", "", f"Tile scale: {len(keys)} unit-runs in {len(blocks)} blocks. "
          f"Grade margin {gm:.2f} pp.", "", "| Arm | steep access | − A0 | − A5 at matched length | − A4 | km | flat density 0–5 / 5–10 (Δ) | "
          "steep coverage 20–30 / 30+ (Δ) | grade p90 (Δ pp) | climbs ≥ 100 m share | qualifies |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms + ["A5"]:
        lv = level[a]
        c0 = C.get(f"{a}-A0")
        md.append(f"| {a} | {lv['access']:.3f} | {f3(c0['access']) if c0 else '–'} | "
                  f"{f3(C[f'{a}-A5@matched']['access']) if f'{a}-A5@matched' in C else '–'} | "
                  f"{f3(C[f'{a}-A4']['access']) if f'{a}-A4' in C else '–'} | {lv['km']:.0f} | "
                  f"{(f'{c0['density_0-5']['delta']:+.2f} / {c0['density_5-10']['delta']:+.2f}') if c0 else '–'} | "
                  f"{(f3(c0['coverage_20-30']) + ' / ' + f3(c0['coverage_30-inf'])) if c0 else '–'} | "
                  f"{(f'{c0['grade_p90']['delta']:+.2f}') if c0 else '–'} | {lv['share_in_climbs_ge100']:.3f} | "
                  f"{Q[a][0] if a in Q else '–'} |")
    md += ["", f"**Tile selection:** {verdict}", "", "Whole towns (served share; A5 = all connectors):", ""]
    for k_, v in wt.items():
        md.append(f"* {k_}: " + ", ".join(f"{a} {v[a]:.3f}" for a in ("A0",) + ONLINE + ("A5",) if a in v))
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def stage_secondary():
    """D5 secondary (descriptive): A5 - A0 steep access on TN_s0 and TNse_s0 (tiles, seeds 1-2), next to TNg."""
    out = {}
    for bb in ("TNg_s0", "TN_s0", "TNse_s0"):
        sv0 = sv5 = pr = 0
        for unit in TILE_UNITS:
            for s_ in TILE_SEEDS:
                if not run_file(unit, "A0", bb, s_).exists():
                    continue
                measure_one((unit, "A0", bb, s_))
                x = pickle.loads(zlib.decompress((MEAS / f"{unit}__A0__{bb}__s{s_}.pkl.gz").read_bytes()))
                for r in x["rows"]:
                    sv0 += r["out"][0]
                    sv5 += r["A5"][-1][1]
                    pr += r["pairs"]
        out[bb] = {"A0": sv0 / max(pr, 1), "A5": sv5 / max(pr, 1), "A5_minus_A0": (sv5 - sv0) / max(pr, 1), "pairs": pr}
    atomic_json(OUT / "secondary.json", out)
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "report", "secondary"])
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    return {"measure": lambda: stage_measure(a.workers), "report": stage_report, "secondary": stage_secondary}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
