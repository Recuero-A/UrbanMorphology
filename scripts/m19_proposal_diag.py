"""Milestone 19, D2: the proposal diagnostic (docs/M19_LOG.md D2). Does the frozen M18 sampler stop on hillsides because
its learned proposals hold no gentle continuation (a proposal gap), or because its weights discard the ones they hold
(a scoring problem)?

The frozen M18 runs are replayed with an observer (Diag) that never changes them: at every STOP that leaves a hillside
node a closed interior dead end, it records, on the identical state,
  (a) the 8 learned candidates of that decision with their full weights;
  (b) 64 learned candidates drawn from the same decoder output (own random generator), scored by the same rules;
  (c) a terrain-aware route (Dijkstra on the terrain grid, 4 m, 16 neighbours, cost length x (1 + (grade / 0.08)^2),
      no water, no piece steeper than the sampler's local-street cap) toward a destination taken from the partial
      network only: the nearest node 30-250 m away in another component or more than 3x as far by the network;
and the route's first step scored by the sampler's own rules. Each run must reproduce its stored M18 run bit for bit.

    run      the four replays (GEN2 and natural windows in tile mode; Machu Picchu and Petra whole-town)  ->  events_<run>.jsonl
    report   p_orig, p_big, p_route and D2's decision rule                         ->  summary.json, README.md

    python scripts/m19_proposal_diag.py run|report  ->  outputs/m19/proposal_diag/
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m19" / "proposal_diag"
RUNS = ("gen2", "natural", "machu_picchu", "petra")
SLOPE_MIN, D_MIN, D_MAX, LOOP_X = 0.15, 30.0, 250.0, 3.0
K_BIG, ALIGN_DEG, LEG_M = 64, 30.0, 20.0
G_MAX, G0, ROUTE_X, ROUTE_PAD, PAD_M = 0.218, 0.08, 3.0, 60.0, 60.0
NB = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
      (1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]


# ---------------------------------------------------------------- the router
def route(tf, a, b, size):
    """Least-cost path a -> b on the terrain grid inside [0, size]^2 (D2 (c)); None if not found."""
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
    I, J, C, Ls = [], [], [], []
    for dr, dc in NB:
        ra, rb = max(0, -dr), H - max(0, dr)
        ca, cb = max(0, -dc), W - max(0, dc)
        A = (slice(ra, rb), slice(ca, cb))
        B = (slice(ra + dr, rb + dr), slice(ca + dc, cb + dc))
        L = t.res * math.hypot(dr, dc)
        g = np.abs(Z[B] - Z[A]) / L
        m = ok[A] & ok[B] & (g <= G_MAX)
        I.append(idx[A][m])
        J.append(idx[B][m])
        C.append(L * (1 + (g[m] / G0) ** 2))
        Ls.append(np.full(int(m.sum()), L))
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
    path = path[::-1]
    rr, cc = np.divmod(np.array(path), W)
    P = np.c_[xs[cc], ys[rr]]
    P[0], P[-1] = a, b
    z = Z[rr, cc]
    seg = np.hypot(*np.diff(P, axis=0).T)
    gr = np.abs(np.diff(z)) / np.maximum(seg, 1e-9)
    o = np.argsort(gr)
    cum = np.cumsum(seg[o]) / max(seg.sum(), 1e-9)
    return {"poly": P, "length": float(seg.sum()), "grade_p90": float(gr[o][min(int(np.searchsorted(cum, 0.9)), len(o) - 1)]) if len(o) else 0.0}


def along(P, d):
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    d = min(d, cum[-1])
    k = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
    f = (d - cum[k]) / max(seg[k], 1e-9)
    return P[k] + f * (P[k + 1] - P[k])


# ---------------------------------------------------------------- the observer
class Diag:
    """Mixin over the M14 samplers: records D2 events at STOP decisions; the run itself is untouched."""

    def _diag_init(self, run):
        self.events, self._ctx, self.run_name = [], {}, run
        import torch
        self._g = torch.Generator().manual_seed(19)
        self._r = np.random.default_rng(19)

    def candidates(self, i, s, u, hd, prop, b):
        out = super().candidates(i, s, u, hd, prop, b)
        if not s.__dict__.get("_m14_extra_open") and not prop.get("_rollout") and not prop.get("_diag"):
            self._ctx[id(out)] = (i, s, u, hd, prop, b)
        return out

    def select(self, cands, p_stop, gen):
        j, logZ = super().select(cands, p_stop, gen)
        ctx = self._ctx.pop(id(cands), None)
        if j < 0 and ctx is not None:
            i, s, u, hd, prop, b = ctx
            gx, gy = self.tfs[i].grad_at(*s.nodes[u])
            if len(s.adj[u]) == 1 and not s.border[u] and u not in s.queue and float(np.hypot(gx[0], gy[0])) >= SLOPE_MIN:
                self.events.append(self.diagnose(i, s, u, hd, prop, b, cands, p_stop, float(np.hypot(gx[0], gy[0]))))
        if len(self._ctx) > 4096:
            self._ctx.clear()
        return j, logZ

    def score_moves(self, i, s, u, hd, prop, b, D, cls, st):
        """Candidates at world displacements D (M, 2), scored by the sampler's own candidates() on a copy of the state."""
        from urbanmorph.graphs.sequence import DISP_SCALE, rotate
        s2 = s.copy()
        s2.__dict__["_m14_extra_open"] = (u,)
        M = len(D)
        pr = {"d": (rotate(np.asarray(D, float), -hd) / DISP_SCALE)[None], "cls": np.asarray(cls, int)[None],
              "struct": np.asarray(st, int)[None], "logp": np.zeros((1, M)), "p_stop": np.asarray(prop["p_stop"])[b:b + 1],
              "_diag": True}
        return self.candidates(i, s2, u, hd, pr, 0)

    def diagnose(self, i, s, u, hd, prop, b, cands, p_stop, slope):
        import torch
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from urbanmorph.graphs.m14_sampler import comp_info, selection_probs
        from urbanmorph.graphs.sequence import DISP_SCALE, rotate
        a = s.nodes[u].copy()
        ev = {"run": self.run_name, "tile": int(i), "u": a.round(2).tolist(), "slope": slope}
        # (a) the decision's own candidates
        lw = np.array([c["lw"] for c in cands])
        p_s, p_k, _ = selection_probs(lw, p_stop, getattr(cands, "stop_lw", 0.0))
        live = [k for k, c in enumerate(cands) if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"]
        Lf = [float(np.hypot(*(cands[k]["o"]["q"] - a))) for k in live]
        ev["orig"] = {"n": len(cands), "feasible": len(live), "p_stop_model": float(p_stop), "P_stop": float(p_s)}
        # (b) 64 learned candidates from the same decoder output
        lp, mu, ls, rho = (x[b:b + 1].float() for x in prop["mix"])
        comp = torch.multinomial(torch.softmax(lp, -1), K_BIG, replacement=True, generator=self._g)[0]
        e = torch.randn(K_BIG, 2, generator=self._g)
        m_, s_, r_ = mu[0, comp], ls[0, comp].exp(), rho[0, comp]
        d = m_ + s_ * torch.stack([e[:, 0], r_ * e[:, 0] + torch.sqrt(1 - r_ ** 2) * e[:, 1]], -1)
        pc = np.exp(prop["lc"][b] - prop["lc"][b].max())
        cls = self._r.choice(len(pc), K_BIG, p=pc / pc.sum())
        if prop.get("ls_struct") is not None:
            ps_ = np.exp(prop["ls_struct"][b] - prop["ls_struct"][b].max())
            st = self._r.choice(len(ps_), K_BIG, p=ps_ / ps_.sum())
        else:
            st = np.zeros(K_BIG, int)
        big = self.score_moves(i, s, u, hd, prop, b, rotate(d.numpy() * DISP_SCALE, hd), cls, st)
        live_b = [c for c in big if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"]
        ev["big"] = {"n": K_BIG, "feasible": len(live_b)}
        # destination from the partial network only
        lab, _, _ = comp_info(s)
        n = s.n
        dd = np.hypot(*(s.nodes[:n] - a).T)
        near = np.flatnonzero((dd >= D_MIN) & (dd <= D_MAX))
        E = np.array([x[:2] for x in s.edges if x[0] >= 0], int).reshape(-1, 2)
        if len(near) and len(E):
            Le = np.hypot(*(s.nodes[E[:, 0]] - s.nodes[E[:, 1]]).T)
            A = coo_matrix((np.r_[Le, Le], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
            dn = dijkstra(A, indices=u, limit=LOOP_X * D_MAX + 1.0)
            elig = [v for v in near if lab[v] != lab[u] or dn[v] > LOOP_X * dd[v]]
        else:
            elig = []
        if not elig:
            ev["target"] = None
            return ev
        v = int(min(elig, key=lambda x: dd[x]))
        tgt = s.nodes[v].copy()
        ev["target"] = {"kind": "other_component" if lab[v] != lab[u] else "loop", "xy": tgt.round(2).tolist(),
                        "straight_m": float(dd[v])}
        # (c) the terrain-aware route and its first step
        R = route(self.tfs[i], a, tgt, s.size)
        viable = R is not None and R["length"] <= ROUTE_X * dd[v] + ROUTE_PAD
        ev["route"] = {"found": R is not None, "viable": bool(viable),
                       **({"length_m": R["length"], "detour": R["length"] / max(dd[v], 1e-9), "grade_p90": R["grade_p90"]} if R else {})}
        if not viable:
            return ev
        leg = along(R["poly"], LEG_M) - a
        leg /= max(np.hypot(*leg), 1e-9)
        aligned = lambda c: (np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"
                             and float(np.dot((c["o"]["q"] - a) / max(np.hypot(*(c["o"]["q"] - a)), 1e-9), leg))
                             >= math.cos(math.radians(ALIGN_DEG)))
        al = [k for k in live if aligned(cands[k])]
        ev["orig"].update({"aligned": len(al), "mass_aligned": float(p_k[al].sum()) if al else 0.0})
        ev["big"]["aligned"] = int(sum(aligned(c) for c in big))
        L0 = float(np.clip(np.median(Lf), 12.0, 60.0)) if Lf else 20.0
        step = along(R["poly"], L0) - a
        rc = self.score_moves(i, s, u, hd, prop, b, step[None], [int(np.argmax(prop["lc"][b]))], [0])[0]
        ev["route_step"] = {"feasible": bool(np.isfinite(rc["lw"]) and rc["o"]["kind"] != "dropped"),
                            "reason": rc["reason"], "lw": float(rc["lw"]) if np.isfinite(rc["lw"]) else None, "len_m": L0}
        return ev


# ---------------------------------------------------------------- the replays
def replay(run):
    import torch
    from m18_joint_round import cross_over, eval_file, joint_params, whole_file
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_from_json, graph_to_json
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    model, names, norm, mcfg = load_backbone("TNg_s0", "cpu")
    t0 = time.time()
    if run in ("gen2", "natural"):
        torch.set_num_threads(2)                         # as the stored M18 runs
        from m18_terrain_potentials import eval_tiles
        T = eval_tiles(run)
        ds = [t["d"] for t in T]
        cfg = net_preset(m.preset, **{**m.overrides, **joint_params()})

        class DiagNet(Diag, NetSampler):
            pass
        sm = DiagNet(model, ds, names, norm, pots, cfg, "cpu", model_inputs(mcfg, [t["sid"] for t in T], ds))
        sm._diag_init(run)
        sm.voc_log = None
        gs = sm.run(1)
        ref = [graph_from_json(x) for x in json.loads(eval_file(run, "TNg_s0", 1).read_text())["graphs"]]
        same = all(graph_to_json(g) == graph_to_json(r) for g, r in zip(gs, ref)) and len(gs) == len(ref)
        tiles = [t["sid"] for t in T]
    else:
        torch.set_num_threads(1)                         # as the stored site runs
        import site_benchmark as sb
        from m16_kaist_all import AnchoredGlobalNet
        site = sb.Site(run)
        big = site.domain()
        over, _ = cross_over(run)
        cfg = net_preset(m.preset, **{**m.overrides, **over})

        class DiagGlobal(Diag, AnchoredGlobalNet):
            pass
        sm = DiagGlobal(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
        sm.anchor = None
        sm._diag_init(run)
        g = sm.run(1)[0]
        ref = pickle.loads(whole_file(run, 1).read_bytes())["graph"]
        same = bool(np.array_equal(g.nodes, ref.nodes) and np.array_equal(g.edges, ref.edges))
        tiles = [run]
    with open(OUT / f"events_{run}.jsonl", "w") as f:
        for ev in sm.events:
            f.write(json.dumps({**ev, "tile": tiles[ev["tile"]]}) + "\n")
    meta = {"run": run, "identical_to_stored": bool(same), "events": len(sm.events), "seconds": round(time.time() - t0)}
    atomic_json(OUT / f"meta_{run}.json", meta)
    return meta


def stage_run(workers=4):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [r for r in RUNS if not (OUT / f"meta_{r}.json").exists()]
    with Pool(min(workers, len(todo)) or 1, maxtasksperchild=1) as p:
        for meta in p.imap_unordered(replay, todo):
            print(meta, flush=True)
    return 0


# ---------------------------------------------------------------- the report
def shares(E):
    el = [e for e in E if e.get("route", {}).get("viable")]
    n = len(el)
    f = lambda key: float(np.mean([key(e) for e in el])) if n else float("nan")
    return {"events": len(E), "with_target": sum(e.get("target") is not None for e in E),
            "route_found": sum(e.get("route", {}).get("found", False) for e in E), "eligible": n,
            "p_orig": f(lambda e: e["orig"]["aligned"] > 0), "p_big": f(lambda e: e["big"]["aligned"] > 0),
            "p_route": f(lambda e: e["route_step"]["feasible"]),
            "orig_feasible_any": float(np.mean([e["orig"]["feasible"] > 0 for e in E])) if E else float("nan"),
            "P_stop_median": float(np.median([e["orig"]["P_stop"] for e in E])) if E else float("nan"),
            "mass_aligned_median_when_present": float(np.median([e["orig"]["mass_aligned"] for e in el if e["orig"]["aligned"] > 0]))
            if any(e["orig"]["aligned"] > 0 for e in el) else float("nan"),
            "route_detour_median": float(np.median([e["route"]["detour"] for e in el])) if n else float("nan"),
            "route_grade_p90_median": float(np.median([e["route"]["grade_p90"] for e in el])) if n else float("nan")}


def decide(sh):
    if not sh["eligible"] or sh["eligible"] < 50:
        return "inconclusive (fewer than 50 eligible events)"
    if sh["p_orig"] < 0.25 and sh["p_route"] >= 0.5:
        return "proposal gap" + ("; more learned proposals may already suffice (p_big >= 0.5)" if sh["p_big"] >= 0.5 else "")
    if sh["p_orig"] >= 0.5:
        return "scoring problem"
    return "mixed"


def stage_report():
    E, per, meta = [], {}, {}
    for r in RUNS:
        f = OUT / f"events_{r}.jsonl"
        if not f.exists():
            continue
        ev = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        E += ev
        per[r] = shares(ev)
        meta[r] = json.loads((OUT / f"meta_{r}.json").read_text())
    pooled = shares(E)
    res = {"rule": "docs/M19_LOG.md D2", "pooled": pooled, "decision": decide(pooled), "per_run": per,
           "per_run_decision": {r: decide(v) for r, v in per.items()}, "meta": meta}
    atomic_json(OUT / "summary.json", res)
    fmt = lambda v: "–" if v != v else (f"{v:.2f}" if isinstance(v, float) else str(v))
    md = ["# M19 D2: proposal diagnostic (frozen M18 baseline replayed; events = STOPs leaving hillside dead ends)", "",
          "| Run | identical to stored | events | with target | eligible (viable route) | p_orig (8 learned) | p_big (64 learned) | "
          "p_route (router step feasible) | median P(STOP) | decision |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r, v in list(per.items()) + [("pooled", pooled)]:
        md.append(f"| {r} | {meta[r]['identical_to_stored'] if r in meta else ''} | {v['events']} | {v['with_target']} | {v['eligible']} | "
                  f"{fmt(v['p_orig'])} | {fmt(v['p_big'])} | {fmt(v['p_route'])} | {fmt(v['P_stop_median'])} | "
                  f"{res['decision'] if r == 'pooled' else res['per_run_decision'][r]} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    return {"run": stage_run, "report": stage_report}[stage]()


if __name__ == "__main__":
    sys.exit(main())
