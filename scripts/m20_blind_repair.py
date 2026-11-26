"""Milestone 20, D2: terrain-blind repair control (A5b) on the M19 development units (docs/M20_LOG.md D2).

A5b = M19 A5 with a router whose cost is step length only and that has no grade cap (water and the domain bounds still
block); destinations, eligibility, route-length limit, order and commit rules are A5's. A0 is replayed at its stored
thread counts and must reproduce the stored A0 graphs bit for bit.

    python scripts/m20_blind_repair.py run [--workers 6]   ->  outputs/m20/blind_repair/runs/*.pkl.gz
    python scripts/m20_blind_repair.py report              ->  outputs/m20/blind_repair/report.json, README.md
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m19_arms import BBS, SITE_SEEDS, TILE_SEEDS, Arm, destination, run_file, unit_tiles
from m19_proposal_diag import NB, PAD_M, ROUTE_PAD, ROUTE_X, SLOPE_MIN, along

OUT = ROOT / "outputs" / "m20" / "blind_repair"
GRADE = 0.15


def route_blind(tf, a, b, size):
    """M19 route() with cost = step length and no grade cap (water and bounds still block)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    t = tf.t
    lo = np.maximum(np.minimum(a, b) - PAD_M, 0.0)
    hi = np.minimum(np.maximum(a, b) + PAD_M, size)
    r0, c0 = t.rc(np.array([lo[0]]), np.array([hi[1]]))
    r1, c1 = t.rc(np.array([hi[0]]), np.array([lo[1]]))
    r0, c0, r1, c1 = int(np.floor(r0[0])), int(np.floor(c0[0])), int(np.ceil(r1[0])), int(np.ceil(c1[0]))
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, t.elev.shape[0] - 1), min(c1, t.elev.shape[1] - 1)
    Z = t.elev[r0:r1 + 1, c0:c1 + 1].astype(float)
    H, W = Z.shape
    ok = np.isfinite(Z)
    wk = getattr(t, "water_kind", None)
    if wk is not None:
        ok &= np.asarray(wk)[r0:r1 + 1, c0:c1 + 1] == 0
    xs = t.x0 + (np.arange(c0, c1 + 1) + 0.5) * t.res
    ys = t.y1 - (np.arange(r0, r1 + 1) + 0.5) * t.res
    ok &= ((xs >= 0) & (xs <= size))[None, :] & ((ys >= 0) & (ys <= size))[:, None]
    idx = np.arange(H * W).reshape(H, W)
    I, J, C = [], [], []
    for dr, dc in NB:
        ra, rb = max(0, -dr), H - max(0, dr)
        ca, cb = max(0, -dc), W - max(0, dc)
        A = (slice(ra, rb), slice(ca, cb))
        B = (slice(ra + dr, rb + dr), slice(ca + dc, cb + dc))
        L = t.res * math.hypot(dr, dc)
        m = ok[A] & ok[B]
        I.append(idx[A][m])
        J.append(idx[B][m])
        C.append(np.full(int(m.sum()), L))
    I, J, C = np.concatenate(I), np.concatenate(J), np.concatenate(C)
    if not len(I):
        return None
    G = coo_matrix((C, (I, J)), shape=(H * W, H * W)).tocsr()
    rc = lambda p: (int(np.clip(round((t.y1 - p[1]) / t.res - 0.5) - r0, 0, H - 1)), int(np.clip(round((p[0] - t.x0) / t.res - 0.5) - c0, 0, W - 1)))
    sa, sb = rc(a), rc(b)
    if not ok[sa] or not ok[sb]:
        return None
    dist, pred = dijkstra(G, indices=int(idx[sa]), return_predecessors=True)
    k = int(idx[sb])
    if not np.isfinite(dist[k]):
        return None
    path = [k]
    while path[-1] != int(idx[sa]):
        path.append(int(pred[path[-1]]))
    rr, cc = np.divmod(np.array(path[::-1]), W)
    P = np.c_[xs[cc], ys[rr]]
    P[0], P[-1] = a, b
    return {"poly": P, "length": float(np.hypot(*np.diff(P, axis=0).T).sum())}


def posthoc_blind(sm, i, s, snap_every=1, nocap=False):
    """m19_arms.posthoc with route_blind (same eligibility, destinations, limit, order and commit rules); nocap: the
    sampler's grade cap (reason "grade_cap") does not stop a connector step (D2 secondary arm A5b-nocap)."""
    from urbanmorph.experiments.graph_io import graph_to_json
    s = s.copy()
    s.__dict__["_m14_extra_open"] = ()
    snaps = [(0.0, graph_to_json(s.graph(with_class=True)))]
    log = Counter()
    cand = []
    for u in range(s.n):
        if len(s.adj[u]) != 1 or s.border[u]:
            continue
        gx, gy = sm.tfs[i].grad_at(*s.nodes[u])
        if float(np.hypot(gx[0], gy[0])) < SLOPE_MIN:
            continue
        t = destination(s, u)
        if t is None:
            continue
        R = route_blind(sm.tfs[i], s.nodes[u], t["xy"], s.size)
        log["router_calls"] += 1
        if R is not None and R["length"] <= ROUTE_X * t["straight"] + ROUTE_PAD:
            cand.append((R["length"], u))
    added, n_conn = 0.0, 0
    for _, u in sorted(cand):
        if len(s.adj[u]) != 1:
            continue
        t = destination(s, u)
        if t is None:
            continue
        R = route_blind(sm.tfs[i], s.nodes[u], t["xy"], s.size)
        log["router_calls"] += 1
        if R is None or R["length"] > ROUTE_X * t["straight"] + ROUTE_PAD:
            continue
        tip, d, ok = u, 0.0, False
        while d < R["length"] - 1e-6:
            d = min(d + 25.0, R["length"])
            q = along(R["poly"], d)
            o, st, lw, reason = sm.evaluate_candidate(i, s, tip, q - s.nodes[tip], 3, 0)
            if (not np.isfinite(lw) and not (nocap and reason == "grade_cap")) or o["kind"] == "dropped":
                log["step_infeasible"] += 1
                log[f"infeasible:{reason}"] += 1
                break
            added += float(np.hypot(*(o["q"] - s.nodes[tip])))
            v = s.commit(tip, o, 3)
            if v < 0:
                break
            if o["kind"] != "new_node":
                ok = True
                break
            tip = v
        log["connectors_complete" if ok else "connectors_partial"] += 1
        n_conn += 1
        if n_conn % snap_every == 0:
            snaps.append((added, graph_to_json(s.graph(with_class=True))))
    if n_conn % snap_every:
        snaps.append((added, graph_to_json(s.graph(with_class=True))))
    return snaps, dict(log)


def out_file(unit, bb, seed):
    return OUT / "runs" / f"{unit}__A5b__{bb}__s{seed}.pkl.gz"


def job(spec):
    """Replay A0 exactly as m19_arms.job builds it, check it against the stored A0 run, then run A5b."""
    import torch
    unit, bb, seed = spec
    from m18_joint_round import cross_over, joint_params
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_to_json
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    tile_mode = unit in ("gen2", "natural")
    torch.set_num_threads(2 if tile_mode else 1)
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    model, names, norm, mcfg = load_backbone(bb, "cpu")
    over = dict(joint_params()) if tile_mode else dict(cross_over(unit)[0])
    cfg = net_preset(m.preset, **{**m.overrides, **over})
    if tile_mode:
        T = unit_tiles(unit)
        ds = [t["d"] for t in T]

        class S_(Arm, NetSampler):
            pass
        try:
            crop = model_inputs(mcfg, [t["sid"] for t in T], ds)
        except FileNotFoundError:
            crop = None
        sm = S_(model, ds, names, norm, pots, cfg, "cpu", crop)
    else:
        import site_benchmark as sb
        from m16_kaist_all import AnchoredGlobalNet
        site = sb.Site(unit)
        big = site.domain()

        class S_(Arm, AnchoredGlobalNet):
            pass
        sm = S_(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
        sm.anchor = None
    sm._arm_init("A0", seed)
    sm.voc_log = None
    gs = sm.run(seed)
    stored = pickle.loads(zlib.decompress(run_file(unit, "A0", bb, seed).read_bytes()))
    replay = [graph_to_json(g) for g in gs]
    if replay != stored["graphs"]:
        raise RuntimeError(f"A0 replay differs from the stored run: {spec}")
    res = {"unit": unit, "backbone": bb, "seed": seed, "ids": stored["ids"]}
    for arm, nocap in (("A5b", False), ("A5bn", True)):
        t1 = time.time()
        snaps, logs = [], []
        for i, g in enumerate(gs):
            S = sm.states[i]
            p = next((k for k, x in enumerate(S) if np.array_equal(x.graph(with_class=True).nodes, g.nodes)
                      and np.array_equal(x.graph(with_class=True).edges, g.edges)), 0)
            sn, lg = posthoc_blind(sm, i, S[p], 1 if tile_mode else 10, nocap)
            snaps.append(sn)
            logs.append(lg)
        res[arm] = {"snapshots": snaps, "log": logs, "seconds": time.time() - t1}
    f = out_file(unit, bb, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.with_suffix(".tmp").write_bytes(zlib.compress(pickle.dumps(res), 6))
    f.with_suffix(".tmp").rename(f)
    return spec, round(res["A5b"]["seconds"] + res["A5bn"]["seconds"], 1)


def stage_run(workers):
    from multiprocessing import Pool
    specs = [(u, b, s) for u in ("gen2", "natural") for b in BBS for s in TILE_SEEDS]
    specs += [(u, "TNg_s0", s) for u in ("machu_picchu", "petra") for s in SITE_SEEDS]
    specs = [x for x in specs if not out_file(*x).exists()]
    print(len(specs), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for spec, sec in p.imap_unordered(job, specs, chunksize=1):
            print(spec, sec, "s", flush=True)
    return 0


def new_edge_grades(g_after, g_before, sample):
    """(length, length steeper than 15 %) of the connector edges: edges of g_after not in g_before, sampled every
    <= 4 m. Halves of an old street split by a connector's T-junction or crossing get new keys, so an edge whose
    midpoint lies within 0.5 m of an edge of g_before is old street, not connector."""
    import shapely
    key = lambda g, e: tuple(np.round(np.sort(g.nodes[e], axis=0).ravel(), 2))
    kb = {key(g_before, e) for e in g_before.edges}
    old = shapely.STRtree(shapely.linestrings(g_before.nodes[g_before.edges])) if len(g_before.edges) else None
    L, G = [], []
    for e in g_after.edges:
        if key(g_after, e) in kb:
            continue
        a, b = g_after.nodes[e[0]], g_after.nodes[e[1]]
        if old is not None and len(old.query(shapely.Point((a + b) / 2), predicate="dwithin", distance=0.5)):
            continue
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / 4.0)))
        P = a + np.linspace(0, 1, n + 1)[:, None] * (b - a)
        z = np.asarray(sample(P[:, 0], P[:, 1]), float).ravel()
        seg = np.hypot(*np.diff(P, axis=0).T)
        ok = np.isfinite(np.diff(z))
        L.append(seg[ok])
        G.append(np.abs(np.diff(z))[ok] / np.maximum(seg[ok], 1e-9))
    if not L:
        return 0.0, 0.0
    L, G = np.concatenate(L), np.concatenate(G)
    return float(L.sum()), float(L[G > GRADE].sum())


def measure(spec):
    from m19_eval import MEAS, a5_at, fields, length_m, od_points, served
    from urbanmorph.experiments.graph_io import graph_from_json
    unit, bb, seed, arm = spec
    rb = pickle.loads(zlib.decompress(out_file(unit, bb, seed).read_bytes()))[arm]
    r0 = pickle.loads(zlib.decompress(run_file(unit, "A0", bb, seed).read_bytes()))
    m0 = pickle.loads(zlib.decompress((MEAS / f"{unit}__A0__{bb}__s{seed}.pkl.gz").read_bytes()))
    FL = {uid: (blk, F, closed, crop) for uid, blk, F, closed, crop in fields(unit)}
    rows = []
    for k, uid in enumerate(r0["ids"]):
        blk, F, closed, crop = FL[uid]
        P, I, J = od_points(uid, F)
        cut = (lambda g: crop(g)) if crop else (lambda g: g)
        curve = []
        for added, gj in rb["snapshots"][k]:
            g = cut(graph_from_json(gj))
            o, _ = served(g, P, I, J, closed)
            curve.append((length_m(g), int((o == 0).sum()), np.bincount(o, minlength=4).tolist()))
        c5 = m0["rows"][k]["A5"]
        Ls = min(c5[-1][0], curve[-1][0])
        g0 = cut(graph_from_json(r0["graphs"][k]))
        g5 = cut(graph_from_json(r0["A5"]["snapshots"][k][-1][1]))
        g5b = cut(graph_from_json(rb["snapshots"][k][-1][1]))
        rows.append({"uid": uid, "block": blk, "pairs": len(I), "L0": length_m(g0), "L5": c5[-1][0], "L5b": curve[-1][0],
                     "Lstar": Ls, "s5": a5_at(c5, Ls)[0], "s5b": a5_at(curve, Ls)[0], "s5_end": c5[-1][1],
                     "s5b_end": curve[-1][1], "s0": m0["rows"][k]["out"][0],
                     "grade5": new_edge_grades(g5, g0, F.T.sample), "grade5b": new_edge_grades(g5b, g0, F.T.sample),
                     "log5b": rb["log"][k], "log5": r0["A5"]["log"][k]})
    return spec, rows


def contrast(rows, rng, B=2000):
    """A5 vs a blind arm on tile rows: access at matched length (pair-weighted primary, unweighted block means as the
    declared sensitivity) and the connector share > 15 % grade, with a block bootstrap."""
    blocks = sorted({r["block"] for r in rows})
    by = {b: [r for r in rows if r["block"] == b] for b in blocks}

    def stat(rs):
        d = sum(r["s5"] - r["s5b"] for r in rs) / max(sum(r["pairs"] for r in rs), 1)
        grp = {}
        for r in rs:
            grp.setdefault(r["block"], []).append(r)
        bm = [sum(r["s5"] - r["s5b"] for r in g) / max(sum(r["pairs"] for r in g), 1) for g in grp.values()]
        g5 = sum(r["grade5"][1] for r in rs) / max(sum(r["grade5"][0] for r in rs), 1e-9)
        g5b = sum(r["grade5b"][1] for r in rs) / max(sum(r["grade5b"][0] for r in rs), 1e-9)
        return d, float(np.mean(bm)), g5, g5b, g5 - g5b
    est = stat(rows)
    draws = []
    for _ in range(B):
        pick = rng.choice(blocks, len(blocks))
        rs = [dict(r, block=f"{r['block']}#{k}") for k, b in enumerate(pick) for r in by[b]]   # resampled copies
        draws.append(stat(rs))
    ci = lambda j: [float(np.percentile([x[j] for x in draws], 2.5)), float(np.percentile([x[j] for x in draws], 97.5))]
    names = ["access_A5_minus_blind_matched", "access_unweighted_block_mean", "connector_gt15_A5", "connector_gt15_blind",
             "connector_gt15_gap"]
    out = {n: {"est": float(est[j]), "ci95": ci(j)} for j, n in enumerate(names)}
    n = sum(r["pairs"] for r in rows)
    out.update(unit_runs=len(rows), blocks=len(blocks),
               added_km={"A5": sum(r["L5"] - r["L0"] for r in rows) / 1e3, "blind": sum(r["L5b"] - r["L0"] for r in rows) / 1e3},
               end_access={"A0": sum(r["s0"] for r in rows) / n, "A5": sum(r["s5_end"] for r in rows) / n,
                           "blind": sum(r["s5b_end"] for r in rows) / n},
               connectors={a: dict(sum((Counter(r[f"log{a}"]) for r in rows), Counter())) for a in ("5", "5b")})
    out["verdict"] = ("supported" if out["access_A5_minus_blind_matched"]["ci95"][0] >= -0.005
                      and out["connector_gt15_gap"]["ci95"][1] < 0 else "not supported")
    return out


def stage_report(workers):
    from multiprocessing import Pool
    tiles = [(u, b, s) for u in ("gen2", "natural") for b in BBS for s in TILE_SEEDS]
    sites = [(u, "TNg_s0", s) for u in ("machu_picchu", "petra") for s in SITE_SEEDS]
    rep = {"reading_rule": "terrain-aware supported if A5 - A5b >= -0.005 at matched length (95 % block-bootstrap lower "
                           "bound) and A5's connector share > 15 % grade is lower (upper bound of the gap < 0); A5b "
                           "primary, A5b-nocap secondary (D2 and its amendment)"}
    for arm, label in (("A5b", "primary"), ("A5bn", "secondary_nocap")):
        with Pool(workers) as p:
            got = dict(p.map(measure, [(*x, arm) for x in tiles + sites]))
        rows = [dict(r, spec=list(sp)) for sp in tiles for r in got[(*sp, arm)]]
        rep[label] = {"arm": arm, "tile": contrast(rows, np.random.default_rng(20)),
                      "towns": {f"{sp[0]}:s{sp[2]}": {k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in got[(*sp, arm)][0].items() if k not in ("log5b", "log5")}
                                for sp in sites}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k]["tile"] for k in ("primary", "secondary_nocap")}, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    return stage_run(a.workers) if a.stage == "run" else stage_report(a.workers)


if __name__ == "__main__":
    sys.exit(main())
