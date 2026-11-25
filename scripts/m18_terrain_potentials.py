"""Milestone 18, D11 step 2: pilot of terrain moment potentials (graphs/m14_sampler.NetConfig.tm_*).

fit   - Iterative scaling on the 48 TRAINING calibration tiles of M14 (m14_experiments.train_cal_tiles, seed 20261117),
        NA-SMC on <family>_s0 (TN: raw LiDAR; TNg: GLO-30 input), inference seed 1, torch threads 2. Targets = the
        training-city reference curves of scripts/m18_terrain_reference.py (same smoothing and measurement). Updates per
        iteration (k = slope class):
          tm_len[k]  += 1.0 * log(generated density_k / reference density_k)
          tm_ci[k]   += 20  * (generated contour index_k - reference contour index_k), the feature centred on the
                        reference contour index of the class (tm_ci_centre), so that it moves orientation, not length
          tm_steep   += 2.0 * log(generated share > 15 % / reference share > 15 %)
        clipped to [-3, 6], [-10, 10] and [0, 20]; 8 iterations; every iteration's parameters and moments are kept
        (outputs/m18/terrain_potentials/fit_<family>.json). A first, uncentred attempt on TN (stopped after 3
        iterations: the negative contour weights acted as a length reward and density moved the wrong way) is kept in
        fit_TN_uncentred.json. The final parameters are those of the iteration with the
        lowest terrain-realism score on the training tiles, frozen in configs/m18_terrain_potentials_<family>.json.
eval  - Paired on / off comparison (same backbone, seed and tiles; threads 2): NA-SMC on <family>_s0-s2, inference
        seeds 1-2, on GEN2 (95 tiles) and on the 16 natural windows. "off" on GEN2 = the stored runs (TN: M14, which
        replay bit for bit at 2 threads, F23; TNg: configs/m18_plans/gen2_tng.json, same runner and threads); every
        other arm is generated here.
report- Targeted: per-class curves and the terrain-realism score against the training reference. Untargeted: sustained
        climbs (climb_runs), against the REAL networks of the same GEN2 tiles (run-statistics gap in between-city SD,
        and the KS distance of run lengths); descriptive on the natural windows (no real streets there). Preservation:
        coverage, and a shared geographic access test (geo_access). Paired block bootstrap CIs (3 km blocks on GEN2, one
        block per natural window) and block-swap permutation p-values.

sites / site_report - Machu Picchu and Petra (development): NA-SMC TN off / on and TNg off / on, seeds 1-3; TAS,
        typicality, the targeted score (SiteField), climbs against the site's real OSM vehicular network, geographic access.

    python scripts/m18_terrain_potentials.py fit|eval|report|sites|site_report [--family TN|TNg] [--worker k/n]
      ->  outputs/m18/terrain_potentials/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "terrain_potentials"
ITERS = 8
SEEDS = (1, 2)
CLIMB_PCT, CLIMB_MIN_M = 5.0, 20.0            # a sustained climb: consecutive 10 m pieces > 5 % in one direction, >= 20 m
RUN_STATS = ["runs_per_km", "run_p50_m", "run_p90_m", "share_in_runs_ge100", "gain_p90_m"]
GEO_N, GEO_DMIN, GEO_DMAX, GEO_ACCESS_M = 60, 150.0, 500.0, 60.0


def frozen_path(fam):
    return ROOT / "configs" / f"m18_terrain_potentials_{fam}.json"


def frozen_params(fam="TN"):
    """The frozen NetConfig overrides (tuples, as NetConfig expects)."""
    p = json.loads(frozen_path(fam).read_text())["params"]
    return {k: tuple(v) if isinstance(v, list) else float(v) for k, v in p.items()}


def setup():
    import torch
    torch.set_num_threads(2)


def generate(bb, tiles, seed, tm=None):
    from urbanmorph.config import PROJECT_ROOT
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import net_preset, sample_net
    from urbanmorph.graphs.tc_potentials import Potentials
    model, names, norm, mcfg = load_backbone(bb, "cpu")
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")    # TNg: as m18_natural_terrain
    crop = model_inputs(mcfg, [t["sid"] for t in tiles], [t["d"] for t in tiles]) if bb.startswith("TNg") else None
    m = method("NA-SMC")
    cfg = net_preset(m.preset, **m.overrides, **(tm or {}))
    gs, _, _ = sample_net(model, [t["d"] for t in tiles], names, norm, pots, cfg, seed=seed, device="cpu",
                          crop_tiles=crop, voc_log=None)
    return gs


def moments(gs, tiles):
    from m18_terrain_reference import Field, pool, tile_sums
    return pool([tile_sums(g, Field(t["d"])) for g, t in zip(gs, tiles)])


def stage_fit(fam):
    setup()
    from m14_experiments import train_cal_tiles
    from m18_terrain_reference import KEYS, score
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    tiles = train_cal_tiles(48)
    tm = {"tm_len": [0.0] * 5, "tm_ci": [0.0] * 5, "tm_steep": 0.0}
    centre = tuple(float(ref[k]["contour_index"]) for k in KEYS)
    log = []
    for it in range(ITERS + 1):
        t0 = time.time()
        par = {"tm_len": tuple(tm["tm_len"]), "tm_ci": tuple(tm["tm_ci"]), "tm_ci_centre": centre,
               "tm_steep": float(tm["tm_steep"])}
        cur = moments(generate(f"{fam}_s0", tiles, 1, par if it else None), tiles)
        sc = score(cur, ref, sd)
        share15 = sum(cur[k]["share_gt15"] * cur[k]["street_km"] for k in KEYS if np.isfinite(cur[k]["share_gt15"])) / \
            max(sum(cur[k]["street_km"] for k in KEYS), 1e-9)
        rshare = sum(ref[k]["share_gt15"] * ref[k]["street_km"] for k in KEYS) / sum(ref[k]["street_km"] for k in KEYS)
        log.append({"iter": it, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in par.items()},
                    "score": sc, "curves": cur, "share_gt15": share15, "seconds": round(time.time() - t0, 1)})
        atomic_json(OUT / f"fit_{fam}.json", {"family": fam, "reference_share_gt15": rshare, "log": log})
        print(f"iter {it} score {sc:.3f} share15 {share15:.3f} (ref {rshare:.3f}) len {np.round(tm['tm_len'], 2)} "
              f"ci {np.round(tm['tm_ci'], 2)} steep {tm['tm_steep']:.2f}", flush=True)
        if it == ITERS:
            break
        for q, k in enumerate(KEYS):
            g, r = cur[k]["density"], ref[k]["density"]
            if np.isfinite(g) and np.isfinite(r) and g > 0 and r > 0:
                tm["tm_len"][q] = float(np.clip(tm["tm_len"][q] + np.log(g / r), -3, 6))
            g, r = cur[k]["contour_index"], ref[k]["contour_index"]
            if np.isfinite(g) and np.isfinite(r):
                tm["tm_ci"][q] = float(np.clip(tm["tm_ci"][q] + 20 * (g - r), -10, 10))
        if share15 > 0 and rshare > 0:
            tm["tm_steep"] = float(np.clip(tm["tm_steep"] + 2.0 * np.log(share15 / rshare), 0, 20))
    best = min(log[1:], key=lambda r: r["score"])
    atomic_json(frozen_path(fam), {"frozen": time.strftime("%Y-%m-%d %H:%M"), "family": fam, "rule": "lowest training-tile "
                                   "terrain-realism score over the fit iterations (scripts/m18_terrain_potentials.py fit)",
                                   "iteration": best["iter"], "score_training": best["score"],
                                   "score_off_training": log[0]["score"], "params": best["params"]})
    print("frozen iteration", best["iter"], round(best["score"], 3), "off", round(log[0]["score"], 3), flush=True)
    return 0


def stage_noise(fam):
    """Noise floor of the fit's score: potentials off, inference seeds 2 and 3, same 48 training tiles."""
    setup()
    from m14_experiments import train_cal_tiles
    from m18_terrain_reference import score
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    tiles = train_cal_tiles(48)
    out = {}
    for s in (2, 3):
        out[s] = score(moments(generate(f"{fam}_s0", tiles, s), tiles), ref_all["reference"], ref_all["between_city_sd"])
        print(fam, "off seed", s, round(out[s], 3), flush=True)
        atomic_json(OUT / f"fit_noise_{fam}.json", out)
    return 0


# ---------------------------------------------------------------- eval
def eval_tiles(where):
    if where == "gen2":
        from m14_build_gen2 import gen2_tiles
        return gen2_tiles()
    from m18_natural_terrain import windows
    return [{"sid": w["sid"], "d": w["d"], "block": w["sid"]} for w in windows()]


def eval_jobs(fam):
    out = []
    for where in ("natural", "gen2"):
        for b in range(3):
            for s in SEEDS:
                for arm in ("on", "off"):
                    if not (where == "gen2" and arm == "off"):         # stored M14 / M18 runs
                        out.append((where, f"{fam}_s{b}", s, arm))
    return out


def run_file(where, bb, s, arm):
    return OUT / "runs" / f"{where}__{bb}__s{s}__{arm}.json"


def stage_eval(fam, worker):
    setup()
    from urbanmorph.experiments.graph_io import graph_to_json
    tm = frozen_params(fam)
    k, n = worker
    tiles = {}
    for j, (where, bb, s, arm) in enumerate(eval_jobs(fam)):
        f = run_file(where, bb, s, arm)
        if j % n != k or f.exists():
            continue
        t0 = time.time()
        T = tiles.setdefault(where, eval_tiles(where))
        gs = generate(bb, T, s, tm if arm == "on" else None)
        atomic_json(f, {"where": where, "backbone": bb, "seed": s, "arm": arm, "tm": tm if arm == "on" else None,
                        "sids": [t["sid"] for t in T], "graphs": [graph_to_json(g) for g in gs],
                        "seconds": round(time.time() - t0, 1)})
        print(where, bb, s, arm, f"{time.time() - t0:.0f}s", flush=True)
    return 0


def load_graphs(where, bb, s, arm, fam):
    from urbanmorph.experiments.graph_io import graph_from_json
    if where == "gen2" and arm == "off":
        from m14_report import get
        return [graph_from_json(gj) for gj in get("gen2", bb, "NA-SMC")["seeds"][str(s)]["graphs"]]
    return [graph_from_json(gj) for gj in json.loads(run_file(where, bb, s, arm).read_text())["graphs"]]


# ---------------------------------------------------------------- untargeted and preservation outcomes
def climb_runs(g, sample, inside):
    """Sustained climbs on the commonly smoothed network (m18_terrain_reference: ground junction-to-junction polylines,
    30 m resampling, <= 10 m pieces): maximal sequences of consecutive pieces along a polyline whose signed grade is
    > CLIMB_PCT in the same direction, kept if >= CLIMB_MIN_M. A piece outside the scored area ends a run.
    Returns (run lengths m, run elevation gains m, scored street length m)."""
    from m18_terrain_reference import ground_polylines, resample
    runs, gains, tot = [], [], 0.0
    for poly in ground_polylines(g):
        q = resample(poly)
        if q is None or len(q) < 2:
            continue
        P = []
        for a, b in zip(q[:-1], q[1:]):
            m = max(1, int(np.ceil(np.hypot(*(b - a)) / 10.0)))
            P += [a + (b - a) * t for t in np.arange(m) / m]
        P = np.array(P + [q[-1]])
        mid = (P[1:] + P[:-1]) / 2
        ln = np.hypot(*np.diff(P, axis=0).T)
        dz = np.diff(sample(P[:, 0], P[:, 1]))
        ok = inside(mid) & (ln > 0.5) & np.isfinite(dz)
        tot += ln[ok].sum()
        r_, g_ = profile_runs(ln, dz, ok)
        runs += r_
        gains += g_
    return np.array(runs), np.array(gains), tot


def profile_runs(ln, dz, ok):
    """Sustained climbs of one profile (piece lengths, rises, pieces scored): lengths and gains of the runs."""
    sg = np.where(ok, np.sign(dz) * (np.abs(dz) / np.maximum(ln, 1e-9) * 100 > CLIMB_PCT), 0)
    runs, gains = [], []
    cur, L, Z = 0, 0.0, 0.0
    for s_, l_, z_ in zip(np.r_[sg, 0], np.r_[ln, 0], np.r_[dz, 0]):
        if s_ != 0 and s_ == cur:
            L, Z = L + l_, Z + z_
            continue
        if cur != 0 and L >= CLIMB_MIN_M:
            runs.append(L)
            gains.append(abs(Z))
        cur, L, Z = s_, (l_ if s_ else 0.0), (z_ if s_ else 0.0)
    return runs, gains


def wquant(x, w, p):
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(int(np.searchsorted(c, p)), len(o) - 1)])


def run_stats(R, w=None):
    """R: list of per-tile (runs, gains, street m); w: tile weights (bootstrap counts)."""
    w = np.ones(len(R)) if w is None else w
    runs = np.concatenate([r[0] for r in R]) if R else np.zeros(0)
    gains = np.concatenate([r[1] for r in R]) if R else np.zeros(0)
    rw = np.concatenate([np.full(len(r[0]), wi) for r, wi in zip(R, w)]) if R else np.zeros(0)
    tot = float(sum(r[2] * wi for r, wi in zip(R, w)))
    return {"runs_per_km": float(rw.sum() / max(tot / 1000, 1e-9)), "run_p50_m": wquant(runs, rw, 0.5),
            "run_p90_m": wquant(runs, rw, 0.9),
            "share_in_runs_ge100": float((runs * rw)[runs >= 100].sum() / max(tot, 1e-9)),
            "gain_p90_m": wquant(gains, rw, 0.9)}


def wks(x, w, y):
    """Kolmogorov-Smirnov distance between a weighted sample x and an unweighted sample y."""
    if len(x) == 0 or len(y) == 0 or w.sum() <= 0:
        return np.nan
    grid = np.unique(np.r_[x, y])
    o = np.argsort(x)
    Fx = np.cumsum(w[o])[np.searchsorted(x[o], grid, side="right") - 1] / w.sum()
    Fx = np.where(np.searchsorted(x[o], grid, side="right") == 0, 0.0, Fx)
    Fy = np.searchsorted(np.sort(y), grid, side="right") / len(y)
    return float(np.max(np.abs(Fx - Fy)))


def run_score(st, ref, sd):
    v = [abs(st[k] - ref[k]) / sd[k] for k in RUN_STATS if np.isfinite(st[k]) and np.isfinite(ref[k]) and sd[k] > 0]
    return float(np.mean(v)) if v else np.nan


def geo_points(sid, F, n=GEO_N):
    """GEO_N fixed land points of a tile, the same for every network scored on it: buildable land (dry, in the tile,
    smoothed slope < 30 %), drawn with a seed from the tile id. Returns (points, dry-land region of each)."""
    from scipy import ndimage
    from m18_terrain_reference import RES
    ok = F.land & (F.slope < 30)
    reg, _ = ndimage.label(F.land, structure=np.ones((3, 3)))
    r, c = np.nonzero(ok)
    if len(r) == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    k = rng.choice(len(r), size=min(n, len(r)), replace=False)
    x = F.S.x0 + (c[k] + 0.5) * RES + rng.uniform(-RES / 2, RES / 2, len(k))
    y = F.S.y1 - (r[k] + 0.5) * RES + rng.uniform(-RES / 2, RES / 2, len(k))
    return np.c_[x, y], reg[r[k], c[k]]


def geo_access(g, pts, reg, closed=True):
    """Shared geographic access test: each fixed point is joined to the nearest street point if within 60 m. Pairs of
    points 150-500 m apart on the same dry-land region: both joined, and joined by the network (closed window, as M14:
    a street that leaves the tile continues outside). Sums per tile: points, joined points, same-land pairs, pairs with
    both joined, connected pairs; detours (network + 2 joins over straight distance, open graph) of connected pairs."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from urbanmorph.graphs.m14_metrics import closed_components
    z = {"points": len(pts), "joined": 0, "pairs": 0, "pairs_joined": 0, "connected": 0, "detours": []}
    if len(pts) == 0:
        return z
    D = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    I, J = np.nonzero(np.triu((D >= GEO_DMIN) & (D <= GEO_DMAX) & (reg[:, None] == reg[None]) & (reg[:, None] > 0), 1))
    z["pairs"] = len(I)
    if len(g.edges) == 0:
        return z
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    L = np.hypot(*(b - a).T)
    m = np.maximum(1, np.ceil(L / 2.0)).astype(int)
    e = np.repeat(np.arange(len(L)), m + 1)
    t = np.concatenate([np.arange(k + 1) / k for k in m])
    Q = a[e] + t[:, None] * (b - a)[e]
    dist, idx = cKDTree(Q).query(pts)
    joined = dist <= GEO_ACCESS_M
    z["joined"] = int(joined.sum())
    both = joined[I] & joined[J]
    z["pairs_joined"] = int(both.sum())
    if not both.any():
        return z
    if closed:
        lab, _, A = closed_components(g)
    else:                                      # a whole town (site): no window to close
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n0 = len(g.nodes)
        A = coo_matrix((np.r_[L, L], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])),
                       shape=(n0, n0)).tocsr()
        lab = connected_components(A, directed=False)[1]
    ea, te = e[idx], t[idx]
    na, nb = g.edges[ea, 0], g.edges[ea, 1]
    conn = both & (lab[na[I]] == lab[na[J]])
    z["connected"] = int(conn.sum())
    n = len(g.nodes)
    src = np.unique(np.r_[na[I[conn]], nb[I[conn]]])
    if len(src):
        Dn = dijkstra(A[:n, :n], indices=src)
        row = {s: k for k, s in enumerate(src)}
        for i, j in zip(I[conn], J[conn]):
            if ea[i] == ea[j]:
                net = abs(te[i] - te[j]) * L[ea[i]]
            else:
                net = min(Dn[row[u], v] + du + dv
                          for u, du in ((na[i], te[i] * L[ea[i]]), (nb[i], (1 - te[i]) * L[ea[i]]))
                          for v, dv in ((na[j], te[j] * L[ea[j]]), (nb[j], (1 - te[j]) * L[ea[j]])))
            if np.isfinite(net):
                z["detours"].append(float((net + dist[i] + dist[j]) / D[i, j]))
    return z


def geo_stats(Z, w=None):
    w = np.ones(len(Z)) if w is None else w
    s = lambda k: float(sum(z[k] * wi for z, wi in zip(Z, w)))
    det = np.concatenate([np.array(z["detours"]) for z in Z]) if Z else np.zeros(0)
    dw = np.concatenate([np.full(len(z["detours"]), wi) for z, wi in zip(Z, w)]) if Z else np.zeros(0)
    return {"access_60": s("joined") / max(s("points"), 1e-9), "geo_connected": s("connected") / max(s("pairs"), 1e-9),
            "geo_unreach_joined": 1 - s("connected") / max(s("pairs_joined"), 1e-9), "geo_detour_p50": wquant(det, dw, 0.5)}


# ---------------------------------------------------------------- report
def tile_outcomes(g, F, sid, pts_cache):
    from m18_terrain_reference import TILE, tile_sums
    inside = lambda M: (M[:, 0] >= 0) & (M[:, 0] <= TILE) & (M[:, 1] >= 0) & (M[:, 1] <= TILE)
    if sid not in pts_cache:
        pts_cache[sid] = geo_points(sid, F)
    return tile_sums(g, F), climb_runs(g, F.T.sample, inside), geo_access(g, *pts_cache[sid])


def reference_runs():
    """Sustained climbs of the real training networks (the same 473 tiles as the reference curves), per city."""
    f = OUT / "reference_runs.json"
    if f.exists():
        z = json.loads(f.read_text())
        return {c: [(np.array(r), np.array(gn), t) for r, gn, t in v] for c, v in z.items()}
    from m18_terrain_reference import TILE, Field, train_tiles
    inside = lambda M: (M[:, 0] >= 0) & (M[:, 0] <= TILE) & (M[:, 1] >= 0) & (M[:, 1] <= TILE)
    by = {}
    for sid, g, d in train_tiles():
        by.setdefault(sid[:7], []).append(climb_runs(g, Field(d).T.sample, inside))
    atomic_json(f, {c: [(r.tolist(), gn.tolist(), t) for r, gn, t in v] for c, v in by.items()})
    return by


def paired(on, off, blocks, stat, B=2000, seed=0):
    """Paired block bootstrap (95 % CI) and block-swap permutation p of stat(on) - stat(off). on / off: per-unit lists in
    the same order; blocks: block label per unit. stat(units, weights) -> float."""
    ub, inv = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    d0 = stat(on, None) - stat(off, None)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[inv].astype(float)
        boot.append(stat(on, w) - stat(off, w))
    perm = []
    for _ in range(B):
        flip = rng.random(nb) < 0.5
        f = flip[inv]
        a = [y if s else x for x, y, s in zip(on, off, f)]
        b = [x if s else y for x, y, s in zip(on, off, f)]
        perm.append(stat(a, None) - stat(b, None))
    boot, perm = np.array(boot), np.array(perm)
    return {"delta": float(d0), "lo": float(np.nanpercentile(boot, 2.5)), "hi": float(np.nanpercentile(boot, 97.5)),
            "p_perm": float((1 + np.sum(np.abs(perm) >= abs(d0) - 1e-12)) / (1 + len(perm)))}


def stage_report(fam):
    from m18_terrain_reference import KEYS, Field, pool, score
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    rr = reference_runs()
    ref_run = run_stats([x for v in rr.values() for x in v])
    city_run = {c: run_stats(v) for c, v in rr.items()}
    run_sd = {k: float(np.nanstd([city_run[c][k] for c in city_run], ddof=1)) for k in RUN_STATS}
    res = {"family": fam, "params": frozen_params(fam), "reference_runs": ref_run, "run_between_city_sd": run_sd,
           "real_city_run_scores": {c: run_score(v, ref_run, run_sd) for c, v in city_run.items()}, "where": {}}
    pool_w = lambda S, w: pool([{k: np.tensordot(w, np.array([s[k] for s in S]), axes=1) for k in S[0]}]) if w is not None else pool(S)
    stats = {"terrain_realism": lambda U, w: score(pool_w([u[0] for u in U], w), ref, sd),
             **{k: (lambda kk: lambda U, w: run_stats([u[1] for u in U], w)[kk])(k) for k in RUN_STATS},
             "coverage_all": lambda U, w: (lambda t: float(t["covered"].sum() / t["land"].sum()))(
                 {k: np.tensordot(np.ones(len(U)) if w is None else w, np.array([u[0][k] for u in U]), axes=1) for k in ("covered", "land")}),
             **{f"coverage_{key}": (lambda q: lambda U, w: pool_w([u[0] for u in U], w)[KEYS[q]]["coverage"])(q) for q, key in enumerate(KEYS)},
             **{k: (lambda kk: lambda U, w: geo_stats([u[2] for u in U], w)[kk])(k)
                for k in ("access_60", "geo_connected", "geo_unreach_joined", "geo_detour_p50")}}
    for where in ("gen2", "natural"):
        T = eval_tiles(where)
        F = {t["sid"]: Field(t["d"]) for t in T}
        pc = {}
        # sustained climbs depend on the terrain mix, so their realism target is the real network on the SAME tiles
        # (GEN2); the natural windows have no real streets, and their climb statistics are descriptive only
        real = [tile_outcomes(t["graph"], F[t["sid"]], t["sid"], pc) for t in T] if where == "gen2" else None
        if real is not None:
            tgt = run_stats([u[1] for u in real])
            tgt_runs = np.concatenate([u[1][0] for u in real])
            stats["climb_run_score"] = lambda U, w: run_score(run_stats([u[1] for u in U], w), tgt, run_sd)
            stats["climb_run_ks"] = lambda U, w: wks(np.concatenate([u[1][0] for u in U]), np.concatenate(
                [np.full(len(u[1][0]), 1.0 if w is None else w[i]) for i, u in enumerate(U)]), tgt_runs)
        else:
            stats.pop("climb_run_score", None)
            stats.pop("climb_run_ks", None)
        units = {"on": [], "off": []}
        blocks = []
        for b in range(3):
            for s in SEEDS:
                for arm in ("on", "off"):
                    gs = load_graphs(where, f"{fam}_s{b}", s, arm, fam)
                    units[arm] += [tile_outcomes(g, F[t["sid"]], t["sid"], pc) for g, t in zip(gs, T)]
                blocks += [str(t["block"]) for t in T]
        out = {"tiles": len(T), "runs": 6, "contrasts": {}}
        for arm in ("on", "off"):
            out[arm] = {"curves": pool([u[0] for u in units[arm]]), **{k: f(units[arm], None) for k, f in stats.items()},
                        **{f"runs.{k}": v for k, v in run_stats([u[1] for u in units[arm]]).items()}}
        if real is not None:
            out["real"] = {**{k: f(real, None) for k, f in stats.items()},
                           **{f"runs.{k}": v for k, v in run_stats([u[1] for u in real]).items()}}
        for k, f in stats.items():
            out["contrasts"][k] = paired(units["on"], units["off"], np.array(blocks), f, B=1000)
            print(where, k, {a: round(out[a][k], 4) for a in ("on", "off")}, out["contrasts"][k], flush=True)
        res["where"][where] = out
        atomic_json(OUT / f"report_{fam}.json", res)
    return 0


# ---------------------------------------------------------------- the two inspected sites (development)
SITES = ("machu_picchu", "petra")
SITE_VERS = ("original", "TM", "TNg", "TNg + TM")    # NA-SMC: TN off / on, TNg off / on (site_benchmark.generate_one)


class SiteField:
    """m18_terrain_reference.Field for a whole site: grade and slope classes on the site benchmark's bare-earth proxy of
    the GLO-30 DSM (sigma 16 m, as TAS there), land = the city mask, dry and valid."""
    def __init__(self, site, big, mask):
        self.T = self.S = site.bare(big)
        self.gx, self.gy = self.S.gradient()
        self.slope = np.hypot(self.gx, self.gy) * 100
        self.land = mask & (big["water"] == 0) & big["valid"]
        self.cls = np.digitize(self.slope, [5, 10, 20, 30])
        self.extent = site.L


def stage_sites(workers):
    """NA-SMC TN with the potentials (TM), TNg off and TNg with the potentials, seeds 1-3 (cached by site_benchmark;
    one torch thread per job, as the stored originals)."""
    from multiprocessing import Pool
    import site_benchmark as sb
    js = [(name, "NA-SMC", ver, s) for name in SITES for ver in SITE_VERS[1:] for s in sb.SEEDS]
    js = [j for j in js if not (sb.Site(j[0]).cache / f"{sb.key_of(j[1], j[2])}_s{j[3]}.pkl").exists()]
    print(len(js), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for job, how, sec in p.imap_unordered(sb.generate_one, js, chunksize=1):
            print(job, how, round(sec), flush=True)
    return 0


def stage_site_report():
    import site_benchmark as sb
    from m18_terrain_reference import pool, score, tile_sums
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    rr = reference_runs()
    city_run = {c: run_stats(v) for c, v in rr.items()}
    run_sd = {k: float(np.nanstd([city_run[c][k] for c in city_run], ddof=1)) for k in RUN_STATS}
    res = {}
    for name in SITES:
        sb.init(name)
        site, big, mask = sb.G["site"], sb.G["big"], sb.G["mask"]
        F = SiteField(site, big, mask)
        inside = lambda M: mask[site.rc(M)] & (M[:, 0] >= 0) & (M[:, 0] <= site.L) & (M[:, 1] >= 0) & (M[:, 1] <= site.L)
        land_km2 = float(F.land.sum() * 16 / 1e6)
        pts = geo_points(name, F, n=int(round(GEO_N * land_km2 / 0.262144)))
        real = sb.crop(site.real("vehicular"), mask, site)
        real_runs = climb_runs(real, F.T.sample, inside)
        tgt, tgt_runs = run_stats([real_runs]), real_runs[0]

        def outcomes(gc):
            ru = climb_runs(gc, F.T.sample, inside)
            st = run_stats([ru])
            return {"terrain_realism": score(pool([tile_sums(gc, F)]), ref, sd), "climb_run_score": run_score(st, tgt, run_sd),
                    "climb_run_ks": wks(ru[0], np.ones(len(ru[0])), tgt_runs), **{f"runs.{k}": v for k, v in st.items()},
                    **geo_stats([geo_access(gc, *pts, closed=False)])}
        rows = [{"version": "real (OSM vehicular)", "seed": 0, **outcomes(real)}]
        for ver in SITE_VERS:
            for s_ in sb.SEEDS:
                if not (site.cache / f"{sb.key_of('NA-SMC', ver)}_s{s_}.pkl").exists():
                    continue
                row, gc = sb.score_one(("NA-SMC", ver, s_))
                rows.append({"version": ver, "seed": s_, **{k: row[k] for k in ("typicality", "TAS", "G_gentle", "C_contour",
                             "S_siting", "street_km_per_km2", "coverage_60", "largest_piece_share")}, **outcomes(gc)})
                print(name, ver, s_, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        res[name] = {"land_km2": land_km2, "geo_points": len(pts[0]), "rows": rows}
        atomic_json(OUT / "sites.json", res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "noise", "eval", "report", "sites", "site_report"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--family", default="TN", choices=["TN", "TNg"])
    ap.add_argument("--worker", default="0/1")
    a = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        return stage_fit(a.family)
    if a.stage == "eval":
        return stage_eval(a.family, tuple(map(int, a.worker.split("/"))))
    if a.stage == "noise":
        return stage_noise(a.family)
    if a.stage == "sites":
        return stage_sites(a.workers)
    if a.stage == "site_report":
        return stage_site_report()
    return stage_report(a.family)


if __name__ == "__main__":
    sys.exit(main())
