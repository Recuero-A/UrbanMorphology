"""Milestone 19, D5: the development comparison of the arms (docs/M19_LOG.md D5).

Every arm inherits the frozen M18 configuration (D14b) and differs only at eligible frontier decisions (a node with at
most one street, off the border, landform slope >= 15 %):
  A0  frozen M18;
  A1  router proposals mixed with the learned ones, importance-weighted (Q = (K p + M q) / (K + M)), frozen scoring;
  A2  learned proposals, horizon-matched look-ahead twist (M13/M14 approximate twisted SMC, H = 8, 4 per step);
  A3  A1 + A2;
  A4  K = 64 learned proposals (56 extra from the same decoder output);
  A5  A0's final state + the same router after growth (connectors in order of route length, snapshots).

    run        tile sets (39 hilly GEN2 tiles, 16 natural windows) x TNg_s0-s2 x seeds 1-2, and the two sites x TNg_s0 x
               seeds 1-3, every arm (A5 inside the A0 job)          ->  runs/<unit set>__<arm>__<backbone>__s<seed>.pkl.gz

    python scripts/m19_arms.py run [--workers 7] [--only gen2,natural,machu_picchu,petra]  ->  outputs/m19/arms/
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

from m19_proposal_diag import D_MAX, D_MIN, LOOP_X, ROUTE_PAD, ROUTE_X, SLOPE_MIN, along, route

OUT = ROOT / "outputs" / "m19" / "arms"
ARMS = ("A0", "A1", "A2", "A3", "A4")
SIGMA_M, M_ROUTER, K_EXTRA, H_TWIST, M_TWIST = 3.0, 2, 56, 8, 4
BBS, TILE_SEEDS, SITE_SEEDS = ("TNg_s0", "TNg_s1", "TNg_s2"), (1, 2), (1, 2, 3)
STEEP_LO, STEEP_HI, MIN_STEEP_KM2 = 15.0, 35.0, 0.01


def gen2_units():
    """D5: the GEN2 tiles with >= 0.01 km2 of steep eligible land (15-35 %), in GEN2 order (cached by tile id)."""
    from m18_terrain_potentials import eval_tiles
    from m18_terrain_reference import Field
    T = eval_tiles("gen2")
    f = OUT / "gen2_units.json"
    if f.exists():
        keep = set(json.loads(f.read_text()))
    else:
        keep = {t["sid"] for t in T if (lambda F: (F.land & (F.slope >= STEEP_LO) & (F.slope < STEEP_HI)).sum() * 16 / 1e6
                                       >= MIN_STEEP_KM2)(Field(t["d"]))}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(keep)))
    return [t for t in T if t["sid"] in keep]


def unit_tiles(which):
    from m18_terrain_potentials import eval_tiles
    return gen2_units() if which == "gen2" else eval_tiles("natural")


def destination(s, u):
    """D2's destination from the partial network only (or None)."""
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
    return {"v": v, "xy": s.nodes[v].copy(), "straight": float(dd[v]), "other": bool(lab[v] != lab[u])}


class Arm:
    """Mixin over the M14 samplers: the D5 arm's changes at frontier decisions."""

    def _arm_init(self, arm, seed):
        import torch
        self.arm = arm
        self.mech = Counter()
        self._g = torch.Generator().manual_seed(1910000 + seed)
        self._r = np.random.default_rng(1910000 + seed)
        self.router_window_nodes = 0

    def is_frontier(self, i, s, u):
        if len(s.adj[u]) > 1 or s.border[u]:
            return False
        gx, gy = self.tfs[i].grad_at(*s.nodes[u])
        return float(np.hypot(gx[0], gy[0])) >= SLOPE_MIN

    def features(self, i, s, u, cands, p_stop):
        f = super().features(i, s, u, cands, p_stop)
        f["frontier"] = self.is_frontier(i, s, u)
        return f

    def fires(self, f):
        if self.arm in ("A2", "A3"):
            return bool(f.get("frontier"))
        return super().fires(f)

    def finish(self, i, s, u, cands, ps):
        """The network and terrain-moment terms, as NetSampler.candidates adds them."""
        from urbanmorph.graphs.m14_sampler import net_candidate_lw, net_stop_lw, open_nodes, terrain_moment_lw
        from urbanmorph.graphs.sequence import MAX_T
        opened = open_nodes(s) | {u}
        last = s.__dict__.get("_m14_arm")
        closing = bool(last and last[0] == u and last[1] == MAX_T and (self.cfg.lam_dead or self.cfg.lam_island))
        for c in cands:
            c["lw_net"] = net_candidate_lw(s, u, c, self.cfg, opened) + terrain_moment_lw(c, self.cfg)
            if np.isfinite(c["lw"]):
                c["lw"] = c["lw"] + c["lw_net"]
                if closing and c["o"]["kind"] != "dropped":
                    s2 = s.copy()
                    self.commit_candidate(s2, u, c)
                    x = net_stop_lw(s2, u, self.cfg, open_nodes(s2) - {u})
                    c["lw_net"] += x
                    c["lw"] += x
            c["_p_stop"] = ps

    def model_lp(self, prop, b, hd, cands):
        """log p of each candidate under the model: displacement density (normalised units) x class x structure."""
        from urbanmorph.graphs import style_proposals as sp
        lpd = sp.logpdf_d(prop, b, np.array([c["dd"] for c in cands]), hd)
        lc = prop["lc"][b]
        lsf = prop.get("ls_struct")
        return np.array([x + lc[c["cls"]] + (lsf[b][c["st"]] if lsf is not None else 0.0) for c, x in zip(cands, lpd)])

    def new_cands(self, i, s, u, D, cls, st, src):
        out = []
        for dd, c_, s_ in zip(D, cls, st):
            o, stt, lw, reason = self.evaluate_candidate(i, s, u, np.asarray(dd, float), int(c_), int(s_))
            out.append({"o": o, "st": stt, "cls": int(c_), "lw": lw, "reason": reason, "logp": 0.0,
                        "dd": np.asarray(dd, float), "src": src})
            self.stats[i]["cand_evals"] += 1
        return out

    def commit_candidate(self, s, u, c):
        """As the sampler's; a committed router or extra learned candidate is recorded in the state itself, so the record
        follows the particle through resampling (mechanism, D5)."""
        from urbanmorph.graphs.tc_sampler import TCSampler
        v = TCSampler.commit_candidate(s, u, c)
        if c.get("src") and v >= 0:
            s.__dict__.setdefault("_src_edges", []).append((c["src"], s.nodes[u].tolist(), s.nodes[v].tolist(), c["o"]["kind"]))
        return v

    def candidates(self, i, s, u, hd, prop, b):
        from urbanmorph.graphs.m14_sampler import CandList
        out = super().candidates(i, s, u, hd, prop, b)
        if self.arm not in ("A1", "A3", "A4") or prop.get("_rollout") or s.__dict__.get("_m14_extra_open"):
            return out
        if not self.is_frontier(i, s, u):
            return out
        self.mech["frontier_decisions"] += 1
        ps = float(prop["p_stop"][b])
        if self.arm == "A4":
            extra = self.extra_learned(i, s, u, hd, prop, b)
            self.finish(i, s, u, extra, ps)
            pool = CandList(list(out) + extra)
            pool.stop_lw = out.stop_lw
            self.mech["big_offered"] += len(extra)
            return pool
        extra, centre = self.router_props(i, s, u, hd, prop, b, out)
        if not extra:
            return out
        self.finish(i, s, u, extra, ps)
        pool = CandList(list(out) + extra)
        pool.stop_lw = out.stop_lw
        self.is_weights(pool, prop, b, hd, centre, len(out), len(extra))
        self.mech["router_decisions"] += 1
        self.mech["router_offered"] += len(extra)
        self.mech["router_feasible"] += sum(np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped" for c in extra)
        return pool

    def extra_learned(self, i, s, u, hd, prop, b):
        import torch
        from urbanmorph.graphs.sequence import DISP_SCALE, rotate
        lp, mu, ls, rho = (x[b:b + 1].float() for x in prop["mix"])
        comp = torch.multinomial(torch.softmax(lp, -1), K_EXTRA, replacement=True, generator=self._g)[0]
        e = torch.randn(K_EXTRA, 2, generator=self._g)
        m_, s_, r_ = mu[0, comp], ls[0, comp].exp(), rho[0, comp]
        d = (m_ + s_ * torch.stack([e[:, 0], r_ * e[:, 0] + torch.sqrt(1 - r_ ** 2) * e[:, 1]], -1)).numpy()
        pc = np.exp(prop["lc"][b] - prop["lc"][b].max())
        cls = self._r.choice(len(pc), K_EXTRA, p=pc / pc.sum())
        lsf = prop.get("ls_struct")
        if lsf is not None:
            q = np.exp(lsf[b] - lsf[b].max())
            st = self._r.choice(len(q), K_EXTRA, p=q / q.sum())
        else:
            st = np.zeros(K_EXTRA, int)
        return self.new_cands(i, s, u, rotate(d * DISP_SCALE, hd), cls, st, "big")

    def router_props(self, i, s, u, hd, prop, b, learned):
        tgt = destination(s, u)
        if tgt is None:
            return [], None
        a = s.nodes[u]
        R = route(self.tfs[i], a, tgt["xy"], s.size)
        self.mech["router_calls"] += 1
        if R is None or R["length"] > ROUTE_X * tgt["straight"] + ROUTE_PAD:
            return [], None
        live = [float(np.hypot(*(c["o"]["q"] - a))) for c in learned if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"]
        L0 = float(np.clip(np.median(live), 12.0, 60.0)) if live else 20.0
        centre = along(R["poly"], L0) - a
        D = centre + self._r.normal(0.0, SIGMA_M, (M_ROUTER, 2))
        pc = np.exp(prop["lc"][b] - prop["lc"][b].max())
        cls = self._r.choice(len(pc), M_ROUTER, p=pc / pc.sum())
        return self.new_cands(i, s, u, D, cls, np.zeros(M_ROUTER, int), "router"), centre

    def is_weights(self, pool, prop, b, hd, centre, K, M):
        """Mixture importance weights: every candidate's weight x p / Q, Q = (K p + M q) / (K + M)."""
        from urbanmorph.graphs import style_proposals as sp
        lp = self.model_lp(prop, b, hd, pool)
        lc = prop["lc"][b]
        lq = np.array([sp.logq_d(np.array([c["dd"]]), centre[None], SIGMA_M, hd)[0] + lc[c["cls"]] if c["st"] == 0 else -np.inf
                       for c in pool])
        logQ = np.logaddexp(math.log(K) + lp, math.log(M) + lq) - math.log(K + M)
        ps = float(prop["p_stop"][b])
        for c, x, q in zip(pool, lp, logQ):
            if c.get("src") == "router":
                c["logp"] = float(x + math.log1p(-min(ps, 1 - 1e-6)))
            if np.isfinite(c["lw"]):
                c["lw"] = float(c["lw"] + x - q)


# ---------------------------------------------------------------- post-hoc routing (A5)
def posthoc(sm, i, s, snap_every=1):
    """The same router after growth: connectors from every eligible dead end, shortest route first, committed in steps
    of L0 under the sampler's own rules. Returns (snapshots [(added length m, graph json)], log)."""
    from urbanmorph.experiments.graph_io import graph_to_json
    from urbanmorph.graphs.sequence import heading_of
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
        R = route(sm.tfs[i], s.nodes[u], t["xy"], s.size)
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
        R = route(sm.tfs[i], s.nodes[u], t["xy"], s.size)
        log["router_calls"] += 1
        if R is None or R["length"] > ROUTE_X * t["straight"] + ROUTE_PAD:
            continue
        tip, d, ok = u, 0.0, False
        h, _ = heading_of(s.nodes, tip, s.parent[tip])
        L0 = 25.0
        while d < R["length"] - 1e-6:
            d = min(d + L0, R["length"])
            q = along(R["poly"], d)
            o, st, lw, reason = sm.evaluate_candidate(i, s, tip, q - s.nodes[tip], 3, 0)
            if not np.isfinite(lw) or o["kind"] == "dropped":
                log["step_infeasible"] += 1
                break
            L = float(np.hypot(*(o["q"] - s.nodes[tip])))
            v = s.commit(tip, o, 3)
            added += L
            if v < 0:
                break
            if o["kind"] != "new_node":
                ok = True
                break
            tip = v
            h, _ = heading_of(s.nodes, tip, s.parent[tip])
        log["connectors_complete" if ok else "connectors_partial"] += 1
        n_conn += 1
        if n_conn % snap_every == 0:
            snaps.append((added, graph_to_json(s.graph(with_class=True))))
    if n_conn % snap_every:
        snaps.append((added, graph_to_json(s.graph(with_class=True))))
    return snaps, dict(log)


# ---------------------------------------------------------------- one job
def run_file(unit, arm, bb, seed):
    return OUT / "runs" / f"{unit}__{arm}__{bb}__s{seed}.pkl.gz"


def job(spec):
    import torch
    unit, arm, bb, seed = spec
    from m18_joint_round import cross_over, joint_params
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_to_json
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    tile_mode = unit in ("gen2", "natural")
    torch.set_num_threads(2 if tile_mode else 1)                 # as the stored M18 runs
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    model, names, norm, mcfg = load_backbone(bb, "cpu")
    over = dict(joint_params()) if tile_mode else dict(cross_over(unit)[0])
    if arm in ("A2", "A3"):
        over.update(horizon=H_TWIST, rollout_m=M_TWIST, trigger="frontier")
    cfg = net_preset(m.preset, **{**m.overrides, **over})
    t0 = time.time()
    if tile_mode:
        T = unit_tiles(unit)
        ds = [t["d"] for t in T]

        class S_(Arm, NetSampler):
            pass
        try:
            crop = model_inputs(mcfg, [t["sid"] for t in T], ds)
        except FileNotFoundError:                                # natural windows: no street beds to erase (TNse)
            crop = None
        sm = S_(model, ds, names, norm, pots, cfg, "cpu", crop)
        ids = [t["sid"] for t in T]
    else:
        import site_benchmark as sb
        from m16_kaist_all import AnchoredGlobalNet
        site = sb.Site(unit)
        big = site.domain()

        class S_(Arm, AnchoredGlobalNet):
            pass
        sm = S_(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
        sm.anchor = None
        ids = [unit]
    sm._arm_init(arm, seed)
    sm.voc_log = None
    gs = sm.run(seed)
    secs = time.time() - t0
    res = {"unit": unit, "arm": arm, "backbone": bb, "seed": seed, "ids": ids, "graphs": [graph_to_json(g) for g in gs],
           "seconds": secs, "forward_passes": sm.n_fwd, "mech": dict(sm.mech),
           "stats": [{k: float(v) for k, v in st.items() if isinstance(v, (int, float, np.integer, np.floating))} for st in sm.stats]}
    chosen = []
    for i, g in enumerate(gs):
        S = sm.states[i]
        chosen.append(next((k for k, x in enumerate(S) if np.array_equal(x.graph(with_class=True).nodes, g.nodes)
                            and np.array_equal(x.graph(with_class=True).edges, g.edges)), 0))
    res["src_edges"] = [list(sm.states[i][p].__dict__.get("_src_edges", [])) for i, p in enumerate(chosen)]
    if arm == "A0":                                              # A5: the same router after growth
        t1 = time.time()
        snaps, logs = [], []
        for i, g in enumerate(gs):
            sn, lg = posthoc(sm, i, sm.states[i][chosen[i]], 1 if tile_mode else 10)
            snaps.append(sn)
            logs.append(lg)
        res["A5"] = {"snapshots": snaps, "log": logs, "seconds": time.time() - t1}
    f = run_file(unit, arm, bb, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.with_suffix(".tmp").write_bytes(zlib.compress(pickle.dumps(res), 6))
    f.with_suffix(".tmp").rename(f)
    return spec, round(secs), dict(sm.mech)


def stage_run(workers, only):
    from multiprocessing import Pool
    specs = []
    for unit in only:
        bbs, seeds = (BBS, TILE_SEEDS) if unit in ("gen2", "natural") else (("TNg_s0",), SITE_SEEDS)
        specs += [(unit, a, b, s) for a in ARMS for b in bbs for s in seeds]
    specs = [x for x in specs if not run_file(*x).exists()]
    order = {"A2": 0, "A3": 0, "A0": 1, "A4": 2, "A1": 3}            # longest first
    specs.sort(key=lambda x: (order[x[1]], x[0] != "gen2"))
    print(len(specs), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for spec, sec, mech in p.imap_unordered(job, specs, chunksize=1):
            print(spec, sec, "s", mech, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--only", default="gen2,natural,machu_picchu,petra")
    ap.add_argument("--secondary", action="store_true", help="D5 secondary: A0 (+ A5) on TN_s0 and TNse_s0, tiles, seeds 1-2")
    a = ap.parse_args()
    if a.secondary:
        from multiprocessing import Pool
        js = [(u, "A0", b, s_) for u in ("gen2", "natural") for b in ("TN_s0", "TNse_s0") for s_ in TILE_SEEDS
              if not run_file(u, "A0", b, s_).exists()]
        with Pool(a.workers, maxtasksperchild=1) as p:
            for spec, sec, mech in p.imap_unordered(job, js, chunksize=1):
                print(spec, sec, "s", flush=True)
        return 0
    return stage_run(a.workers, a.only.split(","))


if __name__ == "__main__":
    sys.exit(main())
