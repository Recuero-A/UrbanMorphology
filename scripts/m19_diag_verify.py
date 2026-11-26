"""Milestone 19, D4: verification of the proposal diagnostic (docs/M19_LOG.md D4).

The frozen M18 runs are replayed again (bit-identical, as D2) with a fuller observer:
* every eligible frontier (the pop of a node with at most one street - a street end or a new street's start - off the
  border, on land >= 15 % slope) is counted once; the events are D2's: a STOP during that pop that leaves the node a
  closed dead end;
* per event: the decision's 8 candidates re-scored on a copy of the state (they must equal the live weights), with
  their weight components (M13 terrain potentials, degree prior, network term, terrain-moment term), selection
  probabilities against STOP, and 1,000 paired redraws of the decision;
* the D2 destination; the router's first step and its complete continuation (the route cut into steps of the median
  learned step, committed in turn under the sampler's own rules); 32 learned roll-outs (8 proposals per step, the
  feasible ones picked by the frozen weights, STOP excluded) with the same length budget (3 x straight + 60 m);
* the state, the decoder output and the candidates are stored, so that the classification can be recomputed.

    run          the four replays                                       ->  events_<run>.jsonl, states/<run>/*.pkl.gz
    reclassify   recompute every event's classification from its stored state; must match
    report       counts, shares with block-bootstrap intervals, land affected, selection  ->  summary.json, README.md

    python scripts/m19_diag_verify.py run|reclassify|report  ->  outputs/m19/diag_verify/
"""
from __future__ import annotations

import gzip
import json
import math
import pickle
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m19_proposal_diag import ALIGN_DEG, D_MAX, D_MIN, K_BIG, LEG_M, LOOP_X, ROUTE_PAD, ROUTE_X, RUNS, SLOPE_MIN, along, route
from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m19" / "diag_verify"
N_ROLL, ROLL_STEPS, REACH_M, N_DRAWS = 32, 40, 25.0, 1000
LAND_R, COVER_M, LAND_SLOPE = 120.0, 60.0, 30.0


def conv(o):
    """JSON fallback for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def unit(v):
    return v / max(float(np.hypot(*v)), 1e-9)


class Verify:
    """Mixin over the M14 samplers (as m19_proposal_diag.Diag); the run itself is untouched."""

    def _init(self, run):
        self.run_name, self.events, self._ctx, self.frontiers, self._pops = run, [], {}, [], {}
        self.state_dir = OUT / "states" / run
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def candidates(self, i, s, u, hd, prop, b):
        out = super().candidates(i, s, u, hd, prop, b)
        if not s.__dict__.get("_m14_extra_open") and not prop.get("_rollout") and not prop.get("_diag"):
            key = (u, s.t)                                   # one pop = one node in one round of one state
            if self._pops.get(id(s), (None,))[0] != key:      # first decision of the pop: is it an eligible frontier?
                fr = None
                if len(s.adj[u]) <= 1 and not s.border[u]:     # a street end, or a new street's start node
                    gx, gy = self.tfs[i].grad_at(*s.nodes[u])
                    if float(np.hypot(gx[0], gy[0])) >= SLOPE_MIN:
                        self.frontiers.append({"tile": int(i), "u": s.nodes[u].round(2).tolist(), "stop": False,
                                               "start": len(s.adj[u]) == 0})
                        fr = len(self.frontiers) - 1
                self._pops[id(s)] = (key, fr)
            self._ctx[id(out)] = (i, s, u, hd, prop, b, self._pops[id(s)][1])
        return out

    def select(self, cands, p_stop, gen):
        j, logZ = super().select(cands, p_stop, gen)
        ctx = self._ctx.pop(id(cands), None)
        if ctx is not None and ctx[6] is not None:          # D2 events: a STOP that leaves the frontier a dead end
            i, s, u, hd, prop, b, fr = ctx
            if j < 0 and len(s.adj[u]) == 1 and not s.border[u] and u not in s.queue:
                self.frontiers[fr]["stop"] = True
                k = len(self.events)
                row = {"mix": tuple(x[b:b + 1].clone() for x in prop["mix"]), "lc": prop["lc"][b].copy(),
                       "ls_struct": None if prop.get("ls_struct") is None else prop["ls_struct"][b].copy(),
                       "p_stop": float(prop["p_stop"][b])}
                moves = [(np.asarray(c["dd"], float), int(c["cls"]), int(c["st"])) for c in cands]
                live = [float(c["lw"]) for c in cands]
                blob = {"tile": int(i), "u": int(u), "hd": float(hd), "row": row, "moves": moves, "live_lw": live,
                        "stop_lw": float(getattr(cands, "stop_lw", 0.0)), "state": s}
                (self.state_dir / f"{k:05d}.pkl.gz").write_bytes(zlib.compress(pickle.dumps(blob), 6))
                self.events.append(self.analyse(k, i, s, u, hd, row, moves, live, float(getattr(cands, "stop_lw", 0.0))))
        if len(self._ctx) > 4096:
            self._ctx.clear()
        return j, logZ

    # -- helpers ------------------------------------------------------------------------------------
    def score_moves(self, i, s, u, hd, p_stop, D, cls, st):
        from urbanmorph.graphs.sequence import DISP_SCALE, rotate
        s2 = s.copy()
        s2.__dict__["_m14_extra_open"] = (u,)
        M = len(D)
        pr = {"d": (rotate(np.asarray(D, float), -hd) / DISP_SCALE)[None], "cls": np.asarray(cls, int)[None],
              "struct": np.asarray(st, int)[None], "logp": np.zeros((1, M)), "p_stop": np.array([p_stop]), "_diag": True}
        return self.candidates(i, s2, u, hd, pr, 0)

    def components(self, s, u, c):
        from urbanmorph.graphs import junction_reuse as jr
        from urbanmorph.graphs.m14_sampler import terrain_moment_lw
        if not np.isfinite(c["lw"]):
            return {"total": None}
        tm = terrain_moment_lw(c, self.cfg)
        net = float(c.get("lw_net", 0.0)) - tm
        deg = self.cfg.degree_prior * jr.degree_lw(s, u, c["o"]) if self.cfg.degree_prior else 0.0
        return {"total": float(c["lw"]), "m13": float(c["lw"] - c.get("lw_net", 0.0) - deg), "degree": float(deg),
                "net": float(net), "terrain_moment": float(tm)}

    def destination(self, s, u):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from urbanmorph.graphs.m14_sampler import comp_info
        a = s.nodes[u]
        lab, _, _ = comp_info(s)
        n = s.n
        dd = np.hypot(*(s.nodes[:n] - a).T)
        near = np.flatnonzero((dd >= D_MIN) & (dd <= D_MAX))
        E = np.array([x[:2] for x in s.edges if x[0] >= 0], int).reshape(-1, 2)
        if not len(near) or not len(E):
            return None
        Le = np.hypot(*(s.nodes[E[:, 0]] - s.nodes[E[:, 1]]).T)
        A = coo_matrix((np.r_[Le, Le], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        dn = dijkstra(A, indices=u, limit=LOOP_X * D_MAX + 1.0)
        elig = [v for v in near if lab[v] != lab[u] or dn[v] > LOOP_X * dd[v]]
        if not elig:
            return None
        v = int(min(elig, key=lambda x: dd[x]))
        return {"v": v, "xy": s.nodes[v].copy(), "comp": int(lab[v]), "kind": "other_component" if lab[v] != lab[u] else "loop",
                "straight": float(dd[v])}

    def reached(self, s, tip, tgt):
        from urbanmorph.graphs.m14_sampler import comp_info
        if float(np.hypot(*(s.nodes[tip] - tgt["xy"]))) <= REACH_M:
            return True
        if tgt["kind"] == "other_component":
            lab, _, _ = comp_info(s)
            return bool(lab[tip] == lab[tgt["v"]])
        return False

    def router_complete(self, i, s, u, hd, p_stop, R, L0, cls, tgt):
        """Commit the route in steps of L0 under the sampler's rules; (complete, steps, reason)."""
        from urbanmorph.graphs.sequence import heading_of
        s3 = s.copy()
        s3.__dict__["_m14_extra_open"] = (u,)
        P = R["poly"]
        total = R["length"]
        tip, h, d = u, hd, 0.0
        n = 0
        while d < total - 1e-6 and n < 200:
            d = min(d + L0, total)
            q = along(P, d)
            c = self.score_moves(i, s3, tip, h, p_stop, [q - s3.nodes[tip]], [cls], [0])[0]
            if not np.isfinite(c["lw"]) or c["o"]["kind"] == "dropped":
                return False, n, c["reason"] or "dropped"
            v = self.commit_candidate(s3, tip, c)
            n += 1
            if v < 0:
                return False, n, "dropped"
            if self.reached(s3, v, tgt):
                return True, n, "ok"
            if c["o"]["kind"] != "new_node":
                return False, n, "joined_" + c["o"]["kind"]
            tip = v
            h, _ = heading_of(s3.nodes, tip, s3.parent[tip])
        return self.reached(s3, tip, tgt), n, "end"

    def rollouts(self, i, s, u, hd, p_stop, tgt, budget, gen):
        """N_ROLL learned roll-outs from u (batched): 8 proposals per step, a feasible one picked by the frozen weights,
        STOP excluded; returns the share that reach the destination within the length budget, and their reasons."""
        import torch
        from urbanmorph.graphs.m14_sampler import selection_probs
        from urbanmorph.graphs.sequence import heading_of
        from urbanmorph.graphs.tc_sampler import propose, tempered
        S = []
        for _ in range(N_ROLL):
            s2 = s.copy()
            s2.__dict__["_m14_extra_open"] = (u,)
            S.append({"s": s2, "tip": u, "hd": hd, "len": 0.0, "done": None})
        for step in range(ROLL_STEPS):
            alive = [r for r in S if r["done"] is None]
            if not alive:
                break
            items = []
            for r in alive:
                items.append((i, r["s"], r["tip"], r["hd"], heading_of(r["s"].nodes, r["tip"], r["s"].parent[r["tip"]])[1]))
            ctx = self.contexts(items)
            B = len(items)
            stop_logit, mix, _, cl, sl = self.decode(ctx, torch.zeros(B, 1, 2, device=self.device), None,
                                                     torch.full((B, 1), -1, device=self.device, dtype=torch.long))
            stop_logit, mix, cl, sl = tempered(stop_logit, mix, cl, sl, self.cfg)
            prop = propose(stop_logit, mix, cl, sl, self.cfg.K, gen, self.cfg.structures)
            prop["_diag"] = True
            for b, r in enumerate(alive):
                cands = self.candidates(i, r["s"], r["tip"], r["hd"], prop, b)
                lw = np.array([c["lw"] for c in cands])
                _, pk, _ = selection_probs(lw, 0.0, 0.0)              # STOP excluded: p_stop = 0
                if pk.sum() <= 0:
                    r["done"] = "infeasible"
                    continue
                c = cands[int(torch.multinomial(torch.as_tensor(pk / pk.sum()), 1, generator=gen))]
                L = float(np.hypot(*(c["o"]["q"] - r["s"].nodes[r["tip"]])))
                v = self.commit_candidate(r["s"], r["tip"], c)
                r["len"] += L
                if v < 0:
                    r["done"] = "dropped"
                elif self.reached(r["s"], v, tgt):
                    r["done"] = "reached"
                elif r["len"] > budget:
                    r["done"] = "budget"
                elif c["o"]["kind"] != "new_node":
                    r["done"] = "joined_other"
                else:
                    r["tip"] = v
                    r["hd"] = heading_of(r["s"].nodes, v, r["s"].parent[v])[0]
        reasons = {}
        for r in S:
            k = r["done"] or "steps"
            reasons[k] = reasons.get(k, 0) + 1
        return reasons.get("reached", 0) / N_ROLL, reasons

    # -- one event -----------------------------------------------------------------------------------
    def analyse(self, k, i, s, u, hd, row, moves, live_lw, stop_lw):
        import torch
        from urbanmorph.graphs.m14_sampler import selection_probs
        from urbanmorph.graphs.sequence import DISP_SCALE, rotate
        gen = torch.Generator().manual_seed(1900000 + k)
        rng = np.random.default_rng(1900000 + k)
        a = s.nodes[u].copy()
        gx, gy = self.tfs[i].grad_at(*a)
        ev = {"k": k, "tile": int(i), "u": a.round(2).tolist(), "slope": float(np.hypot(gx[0], gy[0]))}
        p_stop = row["p_stop"]
        orig = self.score_moves(i, s, u, hd, p_stop, [m[0] for m in moves], [m[1] for m in moves], [m[2] for m in moves])
        lw = np.array([c["lw"] for c in orig])
        same = bool(np.allclose(np.nan_to_num(lw, neginf=-1e9), np.nan_to_num(np.array(live_lw), neginf=-1e9), atol=1e-9))
        p_s, p_k, _ = selection_probs(lw, p_stop, stop_lw)
        draws = rng.choice(len(lw) + 1, N_DRAWS, p=np.r_[p_s, p_k] / (p_s + p_k.sum()))
        ev["selection"] = {"rescored_equals_live": same, "p_stop_model": p_stop, "stop_lw": stop_lw, "P_stop": float(p_s),
                           "stop_freq_draws": float(np.mean(draws == 0)),
                           "cands": [{"kind": c["o"]["kind"], "len": float(np.hypot(*(c["o"]["q"] - a))),
                                      "grade": float(c["o"].get("feat", {}).get("grade", np.nan)) if c["o"].get("feat") else None,
                                      "reason": c["reason"], "p": float(pk), **self.components(s, u, c)}
                                     for c, pk in zip(orig, p_k)]}
        live = [c for c in orig if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"]
        Lf = [float(np.hypot(*(c["o"]["q"] - a))) for c in live]
        # 64 learned proposals from the same decoder output (as D2)
        lp, mu, ls, rho = (x.float() for x in row["mix"])
        comp = torch.multinomial(torch.softmax(lp, -1), K_BIG, replacement=True, generator=gen)[0]
        e = torch.randn(K_BIG, 2, generator=gen)
        m_, s_, r_ = mu[0, comp], ls[0, comp].exp(), rho[0, comp]
        d = m_ + s_ * torch.stack([e[:, 0], r_ * e[:, 0] + torch.sqrt(1 - r_ ** 2) * e[:, 1]], -1)
        pc = np.exp(row["lc"] - row["lc"].max())
        cls = rng.choice(len(pc), K_BIG, p=pc / pc.sum())
        if row["ls_struct"] is not None:
            ps_ = np.exp(row["ls_struct"] - row["ls_struct"].max())
            st = rng.choice(len(ps_), K_BIG, p=ps_ / ps_.sum())
        else:
            st = np.zeros(K_BIG, int)
        big = self.score_moves(i, s, u, hd, p_stop, rotate(d.numpy() * DISP_SCALE, hd), cls, st)
        ev["orig"] = {"n": len(orig), "feasible": len(live)}
        ev["big"] = {"n": K_BIG, "feasible": sum(np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped" for c in big)}
        tgt = self.destination(s, u)
        if tgt is None:
            ev["target"] = None
            return ev
        ev["target"] = {"kind": tgt["kind"], "xy": tgt["xy"].round(2).tolist(), "straight_m": tgt["straight"]}
        R = route(self.tfs[i], a, tgt["xy"], s.size)
        budget = ROUTE_X * tgt["straight"] + ROUTE_PAD
        viable = R is not None and R["length"] <= budget
        ev["route"] = {"found": R is not None, "viable": bool(viable), "budget_m": budget,
                       **({"length_m": R["length"], "detour": R["length"] / max(tgt["straight"], 1e-9), "grade_p90": R["grade_p90"]} if R else {})}
        if not viable:
            return ev
        leg = unit(along(R["poly"], LEG_M) - a)
        cosmin = math.cos(math.radians(ALIGN_DEG))
        aligned = lambda c: (np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"
                             and float(np.dot(unit(c["o"]["q"] - a), leg)) >= cosmin)
        al = [j for j, c in enumerate(orig) if aligned(c)]
        ev["orig"].update({"aligned": len(al), "mass_aligned": float(p_k[al].sum()) if al else 0.0})
        ev["big"]["aligned"] = int(sum(aligned(c) for c in big))
        L0 = float(np.clip(np.median(Lf), 12.0, 60.0)) if Lf else 20.0
        cls0 = int(np.argmax(row["lc"]))
        step = along(R["poly"], L0) - a
        rc = self.score_moves(i, s, u, hd, p_stop, [step], [cls0], [0])[0]
        ok, n_steps, why = self.router_complete(i, s, u, hd, p_stop, R, L0, cls0, tgt)
        ev["route_step"] = {"feasible": bool(np.isfinite(rc["lw"]) and rc["o"]["kind"] != "dropped"), "reason": rc["reason"], "len_m": L0}
        ev["route_complete"] = {"complete": bool(ok), "steps": n_steps, "reason": why}
        share, reasons = self.rollouts(i, s, u, hd, p_stop, tgt, budget, gen)
        ev["learned_rollouts"] = {"n": N_ROLL, "reached_share": share, "reasons": reasons}
        return ev


# ---------------------------------------------------------------- replays and the samplers
def sampler(run):
    from m18_joint_round import cross_over, joint_params
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    model, names, norm, mcfg = load_backbone("TNg_s0", "cpu")
    if run in ("gen2", "natural"):
        from m18_terrain_potentials import eval_tiles
        T = eval_tiles(run)
        ds = [t["d"] for t in T]
        cfg = net_preset(m.preset, **{**m.overrides, **joint_params()})

        class V(Verify, NetSampler):
            pass
        sm = V(model, ds, names, norm, pots, cfg, "cpu", model_inputs(mcfg, [t["sid"] for t in T], ds))
        return sm, T
    import site_benchmark as sb
    from m16_kaist_all import AnchoredGlobalNet
    site = sb.Site(run)
    big = site.domain()
    over, _ = cross_over(run)
    cfg = net_preset(m.preset, **{**m.overrides, **over})

    class VG(Verify, AnchoredGlobalNet):
        pass
    sm = VG(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
    sm.anchor = None
    return sm, [{"sid": run, "block": run}]


def replay(run):
    import torch
    from m18_joint_round import eval_file, whole_file
    from urbanmorph.experiments.graph_io import graph_from_json, graph_to_json
    torch.set_num_threads(2 if run in ("gen2", "natural") else 1)   # as the stored M18 runs
    sm, T = sampler(run)
    sm._init(run)
    sm.voc_log = None
    t0 = time.time()
    gs = sm.run(1)
    if run in ("gen2", "natural"):
        ref = [graph_from_json(x) for x in json.loads(eval_file(run, "TNg_s0", 1).read_text())["graphs"]]
        same = len(gs) == len(ref) and all(graph_to_json(g) == graph_to_json(r) for g, r in zip(gs, ref))
    else:
        ref = [pickle.loads(whole_file(run, 1).read_bytes())["graph"]]
        same = bool(np.array_equal(gs[0].nodes, ref[0].nodes) and np.array_equal(gs[0].edges, ref[0].edges))
    sid = [t["sid"] for t in T]
    with open(OUT / f"events_{run}.jsonl", "w") as f:
        for ev in sm.events:
            f.write(json.dumps({**ev, "run": run, "tile": sid[ev["tile"]], "block": str(T[ev["tile"]]["block"])}, default=conv) + "\n")
    atomic_json(OUT / f"frontiers_{run}.json", [{**x, "tile": sid[x["tile"]], "block": str(T[x["tile"]]["block"])} for x in sm.frontiers])
    meta = {"run": run, "identical_to_stored": bool(same), "events": len(sm.events), "frontiers": len(sm.frontiers),
            "rescored_equals_live": all(e["selection"]["rescored_equals_live"] for e in sm.events), "seconds": round(time.time() - t0)}
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


# ---------------------------------------------------------------- recompute from stored states
CLASS_KEYS = ("orig", "big", "target", "route", "route_step", "route_complete", "learned_rollouts")


def reclassify_one(run):
    import torch
    torch.set_num_threads(2 if run in ("gen2", "natural") else 1)
    sm, _ = sampler(run)
    sm._init(run + "_re")
    sm.styles, sm.fields = None, [None] * len(sm.tfs)                # set by run(); no style fields in M18
    logged = [json.loads(x) for x in (OUT / f"events_{run}.jsonl").read_text().splitlines() if x.strip()]
    bad = 0
    torch.set_grad_enabled(False)                                    # as inside the sampler's run()
    for ev in logged:
        blob = pickle.loads(zlib.decompress((OUT / "states" / run / f"{ev['k']:05d}.pkl.gz").read_bytes()))
        s = blob["state"]
        re = sm.analyse(ev["k"], blob["tile"], s, blob["u"], blob["hd"], blob["row"], blob["moves"], blob["live_lw"], blob["stop_lw"])
        a = json.dumps({k: ev.get(k) for k in CLASS_KEYS}, sort_keys=True)
        b = json.dumps({k: json.loads(json.dumps(re.get(k), default=conv)) for k in CLASS_KEYS}, sort_keys=True)
        bad += int(a != b)
    return {"run": run, "events": len(logged), "mismatches": bad}


def stage_reclassify():
    from multiprocessing import Pool
    with Pool(len(RUNS), maxtasksperchild=1) as p:
        res = p.map(reclassify_one, RUNS)
    atomic_json(OUT / "reclassify.json", res)
    print(res, flush=True)
    return 0


# ---------------------------------------------------------------- report
def boot(E, fn, units, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    U = sorted(set(units))
    idx = {u: [j for j, x in enumerate(units) if x == u] for u in U}
    vals = []
    for _ in range(B):
        pick = rng.choice(len(U), len(U))
        sel = [j for q in pick for j in idx[U[q]]]
        if sel:
            vals.append(fn([E[j] for j in sel]))
    v = np.array([x for x in vals if x == x])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def share(E, pred):
    return float(np.mean([pred(e) for e in E])) if E else float("nan")


def land_affected(run, E):
    """Eligible land within LAND_R of an event node and farther than COVER_M from every street of the final network,
    as a share of all eligible land (D4)."""
    from scipy import ndimage
    from m18_joint_round import eval_file, whole_file
    from urbanmorph.experiments.graph_io import graph_from_json
    tot_land, aff = 0.0, 0.0
    if run in ("gen2", "natural"):
        from m18_terrain_potentials import eval_tiles
        from m18_terrain_reference import Field
        T = {t["sid"]: t for t in eval_tiles(run)}
        G = dict(zip([t for t in T], [graph_from_json(x) for x in json.loads(eval_file(run, "TNg_s0", 1).read_text())["graphs"]]))
        items = [(sid, Field(T[sid]["d"]), G[sid], -256.0, 768.0) for sid in T]
    else:
        import site_benchmark as sb
        from m18_terrain_potentials import SiteField
        sb.init(run)
        F = SiteField(sb.G["site"], sb.G["big"], sb.G["mask"])
        items = [(run, F, pickle.loads(whole_file(run, 1).read_bytes())["graph"], F.S.x0, F.S.y1)]
    for sid, F, g, x0, y1 in items:
        land = F.land & (F.slope < LAND_SLOPE)
        H, W = land.shape
        res = 4.0
        X = x0 + (np.arange(W) + 0.5) * res
        Y = y1 - (np.arange(H) + 0.5) * res
        ras = np.zeros((H, W), bool)
        for a_, b_ in g.edges:
            p, q = g.nodes[a_], g.nodes[b_]
            n = max(2, int(np.hypot(*(q - p)) / 2.0))
            P = p + np.outer(np.linspace(0, 1, n), q - p)
            r = np.clip(((y1 - P[:, 1]) / res).astype(int), 0, H - 1)
            c = np.clip(((P[:, 0] - x0) / res).astype(int), 0, W - 1)
            ras[r, c] = True
        far = ndimage.distance_transform_edt(~ras) * res > COVER_M
        near = np.zeros((H, W), bool)
        for e in E:
            if e["tile"] != sid:
                continue
            ux, uy = e["u"]
            near |= ((X[None, :] - ux) ** 2 + (Y[:, None] - uy) ** 2) <= LAND_R ** 2
        tot_land += float(land.sum())
        aff += float((land & far & near).sum())
    return aff / max(tot_land, 1e-9), aff * 16 / 1e6


def stage_report():
    E, Fr, per, meta = [], [], {}, {}
    for r in RUNS:
        if not (OUT / f"events_{r}.jsonl").exists():
            continue
        ev = [json.loads(x) for x in (OUT / f"events_{r}.jsonl").read_text().splitlines() if x.strip()]
        fr = json.loads((OUT / f"frontiers_{r}.json").read_text())
        meta[r] = json.loads((OUT / f"meta_{r}.json").read_text())
        E += ev
        Fr += [{**x, "run": r} for x in fr]
        per[r] = (ev, fr)
    unit_of = lambda e: f"{e['run']}:{e['block']}"
    out = {"rule": "docs/M19_LOG.md D4", "meta": meta}
    if (OUT / "reclassify.json").exists():
        out["reclassify"] = json.loads((OUT / "reclassify.json").read_text())

    def summarise(ev, fr):
        el = [e for e in ev if e.get("route", {}).get("viable")]
        u_ev, u_el = [unit_of(e) for e in ev], [unit_of(e) for e in el]
        p = lambda E_, f: share(E_, f)
        S = {"frontiers": len(fr), "events": len(ev), "events_per_frontier": len(ev) / max(len(fr), 1),
             "with_target": sum(e["target"] is not None for e in ev), "eligible": len(el),
             "units_events": len(set(u_ev)), "units_eligible": len(set(u_el))}
        for name, f in (("p_orig", lambda e: e["orig"]["aligned"] > 0), ("p_big", lambda e: e["big"]["aligned"] > 0),
                        ("p_route_first", lambda e: e["route_step"]["feasible"]),
                        ("p_route_complete", lambda e: e["route_complete"]["complete"]),
                        ("p_learned_complete_any", lambda e: e["learned_rollouts"]["reached_share"] > 0)):
            S[name] = {"value": p(el, f), "num": int(sum(f(e) for e in el)), "den": len(el),
                       "ci": boot(el, lambda X: share(X, f), u_el)}
        S["learned_complete_mean_share"] = {"value": float(np.mean([e["learned_rollouts"]["reached_share"] for e in el])) if el else float("nan"),
                                            "ci": boot(el, lambda X: float(np.mean([e["learned_rollouts"]["reached_share"] for e in X])), u_el)}
        Ps = np.array([e["selection"]["P_stop"] for e in ev])
        S["P_stop"] = {"median": float(np.median(Ps)) if len(Ps) else float("nan"),
                       "systematic_gt_0.5": float(np.mean(Ps > 0.5)) if len(Ps) else float("nan"),
                       "occasional_lt_0.2": float(np.mean(Ps < 0.2)) if len(Ps) else float("nan"),
                       "draws_vs_P_max_abs": float(np.max(np.abs(Ps - np.array([e["selection"]["stop_freq_draws"] for e in ev])))) if len(Ps) else float("nan")}
        comp = {}
        for e in ev:
            for c in e["selection"]["cands"]:
                if c.get("total") is None:
                    continue
                for k_ in ("m13", "degree", "net", "terrain_moment"):
                    comp.setdefault(k_, []).append(c[k_])
        S["weight_components_median"] = {k_: float(np.median(v)) for k_, v in comp.items()}
        S["stop_lw_median"] = float(np.median([e["selection"]["stop_lw"] for e in ev])) if ev else float("nan")
        return S
    out["per_run"] = {r: summarise(*v) for r, v in per.items()}
    out["pooled"] = summarise(E, Fr)
    for r, (ev, _) in per.items():
        sh, km2 = land_affected(r, ev)
        out["per_run"][r]["land_affected_share"], out["per_run"][r]["land_affected_km2"] = sh, km2
    atomic_json(OUT / "summary.json", out)
    f = lambda d: f"{d['value']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] ({d['num']}/{d['den']})"
    md = ["# M19 D4: verification of the proposal diagnostic", "",
          "| Run | frontiers | events (share) | eligible / units | p_orig | p_big | router first step | router complete | learned roll-out complete (any of 32) | median P(STOP) | land affected |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, S in list(out["per_run"].items()) + [("pooled", out["pooled"])]:
        md.append(f"| {r} | {S['frontiers']} | {S['events']} ({S['events_per_frontier']:.2f}) | {S['eligible']} / {S['units_eligible']} | "
                  f"{f(S['p_orig'])} | {f(S['p_big'])} | {f(S['p_route_first'])} | {f(S['p_route_complete'])} | "
                  f"{f(S['p_learned_complete_any'])} | {S['P_stop']['median']:.2f} | "
                  f"{(100 * S['land_affected_share']):.1f} % ({S['land_affected_km2']:.3f} km²) |" if "land_affected_share" in S else
                  f"{f(S['p_learned_complete_any'])} | {S['P_stop']['median']:.2f} | – |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({k: out["pooled"][k] for k in ("P_stop", "weight_components_median", "stop_lw_median", "learned_complete_mean_share")}, indent=1))
    return 0


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    return {"run": stage_run, "reclassify": stage_reclassify, "report": stage_report}[stage]()


if __name__ == "__main__":
    sys.exit(main())
