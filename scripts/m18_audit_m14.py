"""Milestone 18, audit step 1: does the M14 connectivity result reproduce from the stored runs, and what does it cost?

1. Reproduction. The stored GEN2 runs (outputs/m14/runs, the M14 confirmatory specs found by m14_report.get) are read
   back and same-land OD unreachability is RECOMPUTED from every stored graph of M* and NA-fixed:
   * exact replay: one random stream per inference seed, default_rng(1000 + seed), shared across the tiles in order
     and consumed first by tile_metrics_m13, then tile_metrics_m14 (as m14_experiments.run scored generated graphs;
     real tiles were scored with a fresh default_rng(0) each);
   * precise re-estimate: the estimator samples about 200 pairs of the network's own nodes (30 sources x 7 targets),
     so it is re-run with 10 independent streams per tile (default_rng(r), r = 0..9) and averaged, with the spread
     across streams as the Monte Carlo noise.
   The H1 contrast NA-fixed - M* is recomputed with m14_analyze.block_contrast (3 TN backbones x 3 seeds).
2. Trade-offs of the connectivity gain, per method, on the same stored graphs (development / confirmation data that
   M14 already used; no new generation, no held-out or sealed data):
   * street km per km2 and functional junctions per km2 (degree >= 3 inside the tile);
   * close15 and nearest-junction p10 (graphs/junctions.junction_row; RAW generated graphs against real graphs built
     with the 8 m snap: M17 found RAW reads 0.01-0.04 higher, so the comparison is labelled RAW);
   * link lengths (junction or dead end to junction or dead end, through degree-2 nodes): p10, median, and the shares
     under 15 m and 25 m (short connectors);
   * blocks: faces of the planar street graph (shapely.polygonize of the noded lines) that do not touch the tile edge
     (boundary-truncated faces excluded); per tile the count per km2, the median area (m2), the share under 2,000 m2,
     and the median Polsby-Popper compactness 4 pi A / P^2.
   Real GEN2 graphs (data/processed/m14/gen2/graphs3d_gen.pkl) are scored the same way.

    python scripts/m18_audit_m14.py [--quick]   ->  outputs/m18/audit_m14/{tradeoffs.md,tradeoffs.json,per_tile.csv}
"""
from __future__ import annotations

import argparse
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
from urbanmorph.graphs.junctions import junction_row
from urbanmorph.graphs.m13_metrics import tile_metrics_m13
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.tc_potentials import TerrainField
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = ROOT / "outputs" / "m18" / "audit_m14"
TN = ("TN_s0", "TN_s1", "TN_s2")
LABELS = ("A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC", "P")
TILE_M = 512.0
SMALL_BLOCK_M2 = 2000.0


def links(g) -> np.ndarray:
    """Lengths (m) of the links between junctions / dead ends, through degree-2 nodes (loops of degree-2 nodes skipped)."""
    E = np.asarray(g.edges)
    if len(E) == 0:
        return np.zeros(0)
    P = np.asarray(g.nodes)[:, :2]
    n = len(P)
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(E):
        adj[a].append((b, k))
        adj[b].append((a, k))
    deg = np.array([len(x) for x in adj])
    L = np.hypot(*(P[E[:, 1]] - P[E[:, 0]]).T)
    seen = np.zeros(len(E), bool)
    out = []
    for s in np.flatnonzero(deg != 2):
        for v, k in adj[s]:
            if seen[k]:
                continue
            tot, prev, cur, kk = 0.0, s, v, k
            while True:
                seen[kk] = True
                tot += L[kk]
                if deg[cur] != 2:
                    break
                nxt = [(w, j) for w, j in adj[cur] if j != kk]
                if not nxt:
                    break
                prev, (cur, kk) = cur, nxt[0]
            out.append(tot)
    return np.asarray(out)


def blocks(g) -> dict:
    """Interior faces of the planar street graph (faces touching the tile edge are boundary-truncated: excluded)."""
    E = np.asarray(g.edges)
    nan = {"blocks_per_km2": 0.0, "block_area_med_m2": np.nan, "block_small_share": np.nan, "block_pp_med": np.nan}
    if len(E) < 3:
        return nan
    P = np.asarray(g.nodes)[:, :2]
    lines = shapely.node(shapely.multilinestrings(shapely.linestrings(P[E])))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    edge = shapely.box(0.5, 0.5, TILE_M - 0.5, TILE_M - 0.5).exterior
    inner = [f for f in faces if f.area > 1.0 and not f.intersects(edge) and shapely.box(0, 0, TILE_M, TILE_M).contains(f)]
    if not inner:
        return nan
    A = np.array([f.area for f in inner])
    pp = np.array([4 * np.pi * f.area / f.length ** 2 for f in inner])
    return {"blocks_per_km2": len(inner) / (TILE_M * TILE_M / 1e6), "block_area_med_m2": float(np.median(A)),
            "block_small_share": float(np.mean(A < SMALL_BLOCK_M2)), "block_pp_med": float(np.median(pp))}


def tile_row(g) -> dict:
    P = np.asarray(g.nodes)[:, :2]
    E = np.asarray(g.edges)
    deg = np.bincount(E.ravel(), minlength=len(P)) if len(E) else np.zeros(len(P), int)
    inside = (P[:, 0] >= 0) & (P[:, 0] <= TILE_M) & (P[:, 1] >= 0) & (P[:, 1] <= TILE_M)
    jr = junction_row(g) or {}
    lk = links(g)
    return {"junctions_per_km2": float(((deg >= 3) & inside).sum() / (TILE_M * TILE_M / 1e6)),
            "close15": jr.get("close15", np.nan), "nn_p10": jr.get("nn_p10", np.nan),
            "link_p10_m": float(np.percentile(lk, 10)) if len(lk) else np.nan,
            "link_med_m": float(np.median(lk)) if len(lk) else np.nan,
            "link_lt15_share": float(np.mean(lk < 15)) if len(lk) else np.nan,
            "link_lt25_share": float(np.mean(lk < 25)) if len(lk) else np.nan, **blocks(g)}


CI_KEYS = ("od_unreach_land", "street_km_per_km2", "junctions_per_km2", "close15", "link_lt15_share", "blocks_per_km2",
           "block_area_med_m2", "block_small_share", "block_pp_med")


def paired_ci(rows, a, b, key, n_boot=4000, seed=2026):
    """Mean tile difference b - a over matched (backbone, seed, tile), averaged per 3 km block, with a block bootstrap
    95 % interval (the M14 inference unit). Tiles where either value is NaN are skipped."""
    va = {(r["backbone"], r["seed"], r["sid"]): float(r[key]) for r in rows if r["method"] == a}
    per_block = {}
    for r in rows:
        if r["method"] != b:
            continue
        k = (r["backbone"], r["seed"], r["sid"])
        x, y = va.get(k, np.nan), float(r[key])
        if np.isfinite(x) and np.isfinite(y):
            per_block.setdefault(r["block"], []).append(y - x)
    if not per_block:
        return None
    m = np.array([np.mean(v) for v in per_block.values()])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(m, len(m)).mean() for _ in range(n_boot)]
    return {"delta": float(m.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "blocks": len(m)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="TN_s0, seed 1 only")
    a = ap.parse_args()
    import m14_analyze as ma
    from m14_build_gen2 import gen2_tiles
    from m14_report import get
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = gen2_tiles()
    tfs = {t["sid"]: TerrainField(terrain_of(t["d"])) for t in tiles}
    bbs = TN[:1] if a.quick else TN
    rows, repro, runs = [], [], {}
    for t in tiles:
        rows.append({"method": "real", "backbone": "-", "seed": 0, "sid": t["sid"], "block": t["block"],
                     "street_km_per_km2": float(np.nan), **tile_row(t["graph"])})
    real_rows = {r["sid"]: r for r in ma.real("gen2")}
    for r in rows:
        r["street_km_per_km2"] = real_rows[r["sid"]]["street_km_per_km2"]
        r["od_unreach_land"] = real_rows[r["sid"]]["od_unreach_land"]
    for lab in LABELS:
        for bb in bbs:
            res = get("gen2", bb, lab)
            if res is None:
                print("missing run", lab, bb, flush=True)
                continue
            runs[(lab, bb)] = res
            seeds = ["1"] if a.quick else sorted(res["seeds"])
            for s in seeds:
                sd = res["seeds"][s]
                stream = np.random.default_rng(1000 + int(s))
                for i, (st, gj) in enumerate(zip(sd["rows"], sd["graphs"])):
                    g = graph_from_json(gj)
                    row = {"method": lab, "backbone": res["spec"]["backbone"], "seed": int(s), "sid": st["sid"],
                           "block": st["block"], "street_km_per_km2": st["street_km_per_km2"],
                           "od_unreach_land": st["od_unreach_land"], **tile_row(g)}
                    if lab in ("M*", "NA-fixed"):                 # recompute the primary outcome from the stored graph
                        tf = tfs[st["sid"]]
                        tile_metrics_m13(g, tf, stream)
                        od = tile_metrics_m14(g, tf, stream)["od_unreach_land"]
                        reps = [tile_metrics_m14(g, tf, np.random.default_rng(r))["od_unreach_land"] for r in range(10)]
                        repro.append({"method": lab, "backbone": bb, "seed": int(s), "sid": st["sid"],
                                      "stored": st["od_unreach_land"], "recomputed": od,
                                      "precise": float(np.mean(reps)), "mc_sd": float(np.std(reps))})
                    rows.append(row)
            print("scored", lab, bb, flush=True)
    keys = [k for k in rows[0] if k not in ("method", "backbone", "seed", "sid", "block")]
    with open(OUT / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "backbone", "seed", "sid", "block", *keys])
        w.writeheader()
        w.writerows(rows)
    # 1. reproduction
    st_ = np.array([r["stored"] for r in repro], float)
    rc_ = np.array([r["recomputed"] for r in repro], float)
    both = np.isfinite(st_) & np.isfinite(rc_)
    nan_agree = int((np.isnan(st_) == np.isnan(rc_)).sum())
    rep = {"graphs_rescored": len(repro), "nan_pattern_agrees": nan_agree,
           "max_abs_diff": float(np.max(np.abs(st_[both] - rc_[both]))) if both.any() else None,
           "tiles_differing": int((np.abs(st_[both] - rc_[both]) > 1e-12).sum()),
           "mean_stored": {m: float(np.nanmean([r["stored"] for r in repro if r["method"] == m])) for m in ("M*", "NA-fixed")},
           "mean_recomputed": {m: float(np.nanmean([r["recomputed"] for r in repro if r["method"] == m])) for m in ("M*", "NA-fixed")},
           "mean_precise": {m: float(np.nanmean([r["precise"] for r in repro if r["method"] == m])) for m in ("M*", "NA-fixed")},
           "tile_mc_sd_median": float(np.nanmedian([r["mc_sd"] for r in repro if r["mc_sd"] > 0] or [np.nan])),
           "tile_mc_sd_p90": float(np.nanpercentile([r["mc_sd"] for r in repro if r["mc_sd"] > 0] or [np.nan], 90))}
    pm = {}
    for r in repro:
        pm.setdefault((r["backbone"], r["seed"], r["sid"]), {})[r["method"]] = r["precise"]
    d = [v["NA-fixed"] - v["M*"] for v in pm.values() if "M*" in v and "NA-fixed" in v and np.isfinite(v["M*"]) and np.isfinite(v["NA-fixed"])]
    rep["precise_paired_delta_mean"] = float(np.mean(d)) if d else None
    if not a.quick and all((m, b) in runs for m in ("M*", "NA-fixed") for b in TN):
        rep["H1_contrast"] = ma.block_contrast([(runs[("M*", b)], runs[("NA-fixed", b)]) for b in TN], "od_unreach_land")
    # 2. trade-offs: mean over tiles of each run, then over runs (backbone x seed); real = mean over tiles
    summ = {}
    for lab in ("real", *LABELS):
        R = [r for r in rows if r["method"] == lab]
        if not R:
            continue
        groups = {}
        for r in R:
            groups.setdefault((r["backbone"], r["seed"]), []).append(r)
        summ[lab] = {k: float(np.nanmean([np.nanmean([float(x[k]) for x in G]) for G in groups.values()])) for k in keys}
        summ[lab]["runs"] = len(groups)
        summ[lab]["tiles"] = len(R) // max(len(groups), 1)
    atomic_json(OUT / "tradeoffs.json", {"reproduction": rep, "summary": summ, "quick": a.quick})
    cols = [("od_unreach_land", "OD unreach.", 3), ("street_km_per_km2", "km/km²", 1), ("junctions_per_km2", "junctions/km²", 0),
            ("close15", "close15 (RAW)", 3), ("nn_p10", "nn p10 m", 1), ("link_p10_m", "link p10 m", 1),
            ("link_med_m", "link median m", 0), ("link_lt15_share", "links < 15 m", 3), ("link_lt25_share", "links < 25 m", 3),
            ("blocks_per_km2", "blocks/km²", 0), ("block_area_med_m2", "block median m²", 0),
            ("block_small_share", "blocks < 2,000 m²", 2), ("block_pp_med", "compactness", 2)]
    md = ["# M18 audit 1: the M14 connectivity result, reproduced from stored runs, and its trade-offs (GEN2)", "",
          f"Source: `outputs/m14/runs` (M14 confirmatory specs), real `data/processed/m14/gen2/graphs3d_gen.pkl`. "
          f"{'QUICK: TN_s0, seed 1 only. ' if a.quick else ''}Script: `scripts/m18_audit_m14.py`.", "",
          "## 1. Reproduction of the primary outcome", "",
          f"* Graphs re-scored (M\\* and NA-fixed): {rep['graphs_rescored']}; empty-tile pattern agrees on {nan_agree}.",
          f"* Largest |stored − recomputed| OD unreachability: {rep['max_abs_diff']:.4f}, on {rep['tiles_differing']} of "
          f"{rep['graphs_rescored']} graphs.",
          f"* Means stored → recomputed: M\\* {rep['mean_stored']['M*']:.4f} → {rep['mean_recomputed']['M*']:.4f}; "
          f"NA-fixed {rep['mean_stored']['NA-fixed']:.4f} → {rep['mean_recomputed']['NA-fixed']:.4f}.",
          f"* Precise re-estimate (10 streams per tile): M\\* {rep['mean_precise']['M*']:.4f}, NA-fixed "
          f"{rep['mean_precise']['NA-fixed']:.4f}; paired tile mean difference {rep['precise_paired_delta_mean']:+.4f}.",
          f"* Monte Carlo noise of one 200-pair estimate (SD across streams, tiles with any unreachable pair): median "
          f"{rep['tile_mc_sd_median']:.4f}, p90 {rep['tile_mc_sd_p90']:.4f}.",
          "* Definition as implemented: pairs of the network's own nodes (30 sources x up to 7 targets, 150-500 m apart), "
          "unreachable on the closed graph and on the same dry-land region, over all pairs drawn; an empty network "
          "scores 1."]
    if "H1_contrast" in rep:
        c = rep["H1_contrast"]
        md.append(f"* H1 contrast NA-fixed − M\\* (block sign-flip, as M14): {json.dumps(c)[:300]}")
    md += ["", "## 2. Trade-offs (mean over tiles per run, then over runs)", "",
           "| Method | runs | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for lab, s in summ.items():
        md.append(f"| {lab} | {s['runs']} | " + " | ".join(f"{s[k]:.{d}f}" if np.isfinite(s[k]) else "—" for k, _, d in cols) + " |")
    cis = {f"{b} - M*": {k: paired_ci(rows, "M*", b, k) for k in CI_KEYS} for b in ("M*+repair", "NA-myopic", "NA-fixed", "NA-SMC")}
    data = json.loads((OUT / "tradeoffs.json").read_text())
    data["paired_vs_Mstar"] = cis
    atomic_json(OUT / "tradeoffs.json", data)
    md += ["", "## 3. Paired differences from M\\* (matched backbone, seed and tile; block bootstrap 95 % interval)", "",
           "| Contrast | " + " | ".join(CI_KEYS) + " |", "|---|" + "---|" * len(CI_KEYS)]
    for name, d in cis.items():
        md.append(f"| {name} | " + " | ".join(f"{c['delta']:+.3g} [{c['lo']:+.3g}, {c['hi']:+.3g}]" if c else "—"
                                             for c in (d[k] for k in CI_KEYS)) + " |")
    md += ["", "close15 is RAW for generated graphs and built (8 m snap) for real ones; blocks exclude faces touching the "
           "tile edge. Means are of tile values; empty tiles give NaN block and link statistics and are skipped."]
    (OUT / "tradeoffs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
