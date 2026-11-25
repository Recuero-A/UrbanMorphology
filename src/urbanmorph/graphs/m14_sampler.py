"""Milestone 14: network-aware and adaptive inference over the trained TerrainGraphTurtle (docs/M14_MATHEMATICAL_FORMULATION.md).

The M13 sampler (graphs/tc_sampler.py) is unchanged; this module subclasses it. Additions:

1. Network potentials that TELESCOPE to graph-level counts (formulation §3):
   D(s) = closed interior dead ends (degree-1 nodes off the border that are neither queued nor the node being expanded);
   I(s) = closed islands (components with >= ISLAND_M of street that touch no border node and contain no open node).
   Every transition s -> s' multiplies the weight by exp(-lam_dead [D(s') - D(s)] - lam_island [I(s') - I(s)]), so the
   product over a run equals exp(-lam_dead D(x) - lam_island I(x)) of the FINAL graph x exactly.
   Events: STOP at node u closes u (D += 1 if deg(u) = 1 off the border) and may close its component (I += 1);
   a candidate that joins a closed dead end gives D -= 1; one that joins a closed island gives I -= 1.
2. The STOP decision carries its network weight: P(stop) = p_stop w_stop / Z, Z = p_stop w_stop + (1 - p_stop) mean_k w_k
   (the M13 Rao-Blackwellised stop with w_stop = 1 is the special case lam = 0).
3. Network-aware look-ahead: Monte Carlo roll-outs of a candidate's continuation (the M13 roll-out) with the network
   potentials inside, the stop weight included, and the node being expanded kept open.
4. Adaptive look-ahead: the roll-outs run only at decisions where ``trigger`` fires (formulation §5); with
   ``voc_log`` every decision with roll-outs records its context features and the value of computation
   VoC = TV(selection distribution without the twist, with the twist), used to validate triggers on DEV-A.
5. Optional per-decision KL trust region (formulation §6): the selection distribution is q_beta ∝ w^beta over the K
   prior candidates (plus stop), beta in [0, 1] the largest value with KL(q_beta || q_0) <= kl_eps, q_0 = the
   candidate-uniform (prior) selection. Hard zeros stay zeros.

The single-chain samplers (method ``energy``) sample the locally normalised distribution, not the tilted target;
method ``smc`` with these potentials is properly weighted for it (formulation §4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.junction_reuse import closure_gain
from urbanmorph.graphs.sequence import MAX_T, heading_of
from urbanmorph.graphs.tc_potentials import NEG_INF
from urbanmorph.graphs.tc_sampler import (PRESETS, SamplerConfig, TCSampler, open_end, propose, tempered)
from urbanmorph.terrain.road_feasibility import water_kind_at

ISLAND_M = 40.0
FEATURES = ("island", "deg1", "water", "steep", "feas_low", "w_spread", "p_stop")


@dataclass
class NetConfig(SamplerConfig):
    lam_dead: float = 0.0            # weight of the dead-end count potential
    lam_island: float = 0.0          # weight of the island count potential
    trigger: str = "always"          # always | never | <feature> | any:<f1>+<f2> | random:<p>
    kl_eps: float = 0.0              # > 0: per-decision KL trust region on the selection distribution
    water_m: float = 48.0            # "water" trigger radius
    steep_slope: float = 0.15        # "steep" trigger threshold (landform slope, M13 slope bin boundary)
    feas_low: float = 0.75           # "feas_low" trigger: share of feasible candidates below this
    closure_gain: float = 0.0        # M17 F2: closure reward only if it shortens the u-v network distance by this fraction
    # M18 D11: terrain moment potentials, exp(-sum_k tm_len[k] L_k - sum_k tm_ci[k] (L x contour index)_k - tm_steep
    # L_steep) with L in units of 100 m of ground street, k = terrain-slope class of the edge (tm_edges, landform
    # slope); edge-additive, so the factor telescopes exactly like the network counts. Empty / 0 = off (bit-identical).
    tm_len: tuple = ()
    tm_ci: tuple = ()
    tm_steep: float = 0.0
    tm_edges: tuple = (0.05, 0.10, 0.20, 0.30)
    tm_steep_grade: float = 0.15


class CandList(list):
    """Candidates of one decision plus the STOP option's network log-weight."""
    stop_lw: float = 0.0


# ------------------------------------------------------------------------------------------------ network state
def comp_info(s: GenState):
    """(labels over nodes, component street length, component touches the border). Cached per state version."""
    key = (s.n, len(s.edges))
    c = s.__dict__.get("_m14c")
    if c is not None and c[0] == key:
        return c[1]
    e = np.array([x[:2] for x in s.edges if x[0] >= 0], int).reshape(-1, 2)
    n = s.n
    if len(e):
        L = np.hypot(*(s.nodes[e[:, 0]] - s.nodes[e[:, 1]]).T)
        A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(A, directed=False)
        clen = np.bincount(lab[e[:, 0]], weights=L, minlength=lab.max() + 1)
    else:
        lab, clen = np.arange(n), np.zeros(n)
    cb = np.bincount(lab, weights=s.border[:n].astype(float), minlength=len(clen)) > 0
    out = (lab, clen, cb)
    s.__dict__["_m14c"] = (key, out)
    return out


def open_nodes(s: GenState, extra=()) -> set:
    return set(s.queue) | set(extra)


def is_closed_dead_end(s: GenState, v: int, opened: set) -> bool:
    return len(s.adj[v]) == 1 and not s.border[v] and v not in opened


def is_closed_island(s: GenState, comp: int, lab, clen, cb, opened: set) -> bool:
    if cb[comp] or clen[comp] < ISLAND_M:
        return False
    return not any(lab[v] == comp for v in opened if v < len(lab))


def target_node(o: dict) -> int:
    """Existing node a committed edge joins (merge / crossing endpoint / structure merge), else -1."""
    return int(o.get("v", -1)) if o["kind"] in ("merge_node", "crossing_endpoint", "struct_merge") else -1


def net_candidate_lw(s: GenState, u: int, c: dict, cfg: NetConfig, opened: set) -> float:
    """Network log-weight of a continuation candidate from u: + lam_dead if it joins a closed dead end, + lam_island
    if it joins a closed island (both events reduce the final count by one)."""
    if not (cfg.lam_dead or cfg.lam_island) or c["lw"] == NEG_INF:
        return 0.0
    o = c["o"]
    v = target_node(o)
    joined = v if v >= 0 else (o["edge"][0] if o["kind"] in ("crossing_split", "t_junction") else -1)
    if joined < 0:
        return 0.0
    lw = 0.0
    if v >= 0 and cfg.lam_dead and is_closed_dead_end(s, v, opened):
        if not (cfg.closure_gain and closure_gain(s, u, v, float(np.hypot(*(o["q"] - s.nodes[u])))) < cfg.closure_gain):
            lw += cfg.lam_dead
    if cfg.lam_island:
        lab, clen, cb = comp_info(s)
        if lab[joined] != lab[u] and is_closed_island(s, lab[joined], lab, clen, cb, opened):
            lw += cfg.lam_island
    return lw


def terrain_moment_lw(c: dict, cfg: NetConfig) -> float:
    """M18 D11 terrain moment potential of a candidate edge (ground edges only; 0 when off)."""
    if not (cfg.tm_len or cfg.tm_ci or cfg.tm_steep) or c["lw"] == NEG_INF:
        return 0.0
    f = c["o"].get("feat")
    if f is None or f.get("struct", 0) != 0 or not np.isfinite(f.get("slope", np.nan)):
        return 0.0
    L = float(f["length"]) / 100.0
    k = int(np.searchsorted(cfg.tm_edges, f["slope"], side="right"))
    lw = 0.0
    if cfg.tm_len:
        lw -= cfg.tm_len[k] * L
    if cfg.tm_ci and np.isfinite(f.get("align", np.nan)):
        lw -= cfg.tm_ci[k] * L * float(f["align"])
    if cfg.tm_steep and np.isfinite(f.get("grade", np.nan)) and f["grade"] > cfg.tm_steep_grade:
        lw -= cfg.tm_steep * L
    return lw


def net_stop_lw(s: GenState, u: int, cfg: NetConfig, opened_others: set) -> float:
    """Network log-weight of STOP at u (u closes; its component closes if no other node of it is open)."""
    if not (cfg.lam_dead or cfg.lam_island):
        return 0.0
    lw = 0.0
    if cfg.lam_dead and len(s.adj[u]) == 1 and not s.border[u] and not (cfg.dead_exempt_local and _rooted_local(s, u)):
        lw -= cfg.lam_dead
    if cfg.lam_island and len(s.adj[u]):
        lab, clen, cb = comp_info(s)
        if is_closed_island(s, lab[u], lab, clen, cb, opened_others):
            lw -= cfg.lam_island
    return lw


def _rooted_local(s: GenState, u: int) -> bool:
    """M16 ablation: is u the end of a local branch rooted on a collector (an intended cul-de-sac)?"""
    from urbanmorph.graphs.style_proposals import local_depth
    return local_depth(s, u)[1] and local_depth(s, u)[0] > 0


def selection_probs(lw_cands: np.ndarray, p_stop: float, stop_lw: float, beta: float = 1.0):
    """(P(stop), P(candidate k) for every k) of the Rao-Blackwellised selection with tempered weights w^beta."""
    w = np.where(np.isfinite(lw_cands), np.exp(beta * np.minimum(lw_cands, 50)), 0.0)
    ws = p_stop * math.exp(beta * stop_lw)
    Z = ws + (1 - p_stop) * w.mean() if len(w) else ws
    if Z <= 0:
        return 1.0, np.zeros(len(w)), Z
    return ws / Z, (1 - p_stop) * w / len(w) / Z, Z


def kl_beta(lw, p_stop, stop_lw, eps):
    """Largest beta in [0, 1] with KL(q_beta || q_0) <= eps (bisection; q_0 = beta 0 with hard zeros kept)."""
    p0s, p0, _ = selection_probs(np.where(np.isfinite(lw), 0.0, -np.inf), p_stop, 0.0)

    def kl(b):
        ps, pk, _ = selection_probs(lw, p_stop, stop_lw, b)
        q, r = np.r_[ps, pk], np.r_[p0s, p0]
        m = q > 0
        return float(np.sum(q[m] * (np.log(q[m]) - np.log(np.maximum(r[m], 1e-300)))))
    if kl(1.0) <= eps:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if kl(mid) <= eps else (lo, mid)
    return lo


# ------------------------------------------------------------------------------------------------ the sampler
class NetSampler(TCSampler):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cfg: NetConfig
        self.voc_log = None                   # list -> record (features, VoC) at decisions with roll-outs
        self.trig_rng = np.random.default_rng(7)
        self._head = {}                       # id(state) -> node being expanded (for roll-outs)

    # -- context features of a decision (formulation §5) ---------------------------------------------
    def features(self, i, s, u, cands, p_stop) -> dict:
        cfg = self.cfg
        lab, clen, cb = comp_info(s)
        a = s.nodes[u]
        t = self.tfs[i].t
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = a + cfg.water_m * np.c_[np.cos(ang), np.sin(ang)]
        wk = water_kind_at(t, np.r_[a[0], ring[:, 0]], np.r_[a[1], ring[:, 1]]) if getattr(t, "water_kind", None) is not None else np.zeros(1)
        gx, gy = self.tfs[i].grad_at(a[0], a[1])
        lw = np.array([c["lw"] for c in cands])
        fin = np.isfinite(lw)
        return {"island": bool(not cb[lab[u]]) if len(s.adj[u]) else True,
                "deg1": bool(len(s.adj[u]) == 1 and not s.border[u]),
                "water": bool((np.asarray(wk) > 0).any()),
                "steep": bool(float(np.hypot(gx[0], gy[0])) >= cfg.steep_slope),
                "feas_low": bool(fin.mean() < cfg.feas_low),
                "w_spread": float(np.std(lw[fin])) if fin.sum() > 1 else 0.0,
                "p_stop": float(p_stop)}

    def fires(self, f: dict) -> bool:
        tr = self.cfg.trigger
        if tr == "always":
            return True
        if tr == "never":
            return False
        if tr.startswith("random:"):
            return bool(self.trig_rng.uniform() < float(tr.split(":")[1]))
        if tr.startswith("any:"):
            return any(self._one(f, x) for x in tr[4:].split("+"))
        return self._one(f, tr)

    @staticmethod
    def _one(f, name):
        if name.startswith("w_spread>"):
            return f["w_spread"] > float(name.split(">")[1])
        return bool(f[name])

    # -- candidates with network weights ---------------------------------------------------------------
    def candidates(self, i, s, u, hd, prop, b):
        out = CandList(super().candidates(i, s, u, hd, prop, b))
        extra = s.__dict__.get("_m14_extra_open", ())
        if not extra:                                  # main state: count arms of u (the loop closes u after MAX_T)
            last = s.__dict__.get("_m14_arm")
            n_arm = last[1] + 1 if last and last[0] == u else 1
            s.__dict__["_m14_arm"] = (u, n_arm)
            if n_arm == MAX_T:
                self.stats[i]["maxT_last_arm"] += 1    # the loop closes u after this arm: add the closure weight
        opened = open_nodes(s, extra) | {u}
        for c in out:
            c["lw_net"] = net_candidate_lw(s, u, c, self.cfg, opened) + terrain_moment_lw(c, self.cfg)
            c["lw"] = c["lw"] + c["lw_net"] if np.isfinite(c["lw"]) else c["lw"]
        if not extra and s.__dict__["_m14_arm"][1] == MAX_T and (self.cfg.lam_dead or self.cfg.lam_island):
            for c in out:                              # telescoping: commit c, then u closes without a STOP
                if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped":
                    s2 = s.copy()
                    self.commit_candidate(s2, u, c)
                    extra_lw = net_stop_lw(s2, u, self.cfg, open_nodes(s2) - {u})
                    c["lw_net"] += extra_lw
                    c["lw"] += extra_lw
        out.stop_lw = net_stop_lw(s, u, self.cfg, open_nodes(s, extra) - {u})
        ps = float(prop["p_stop"][b])
        for c in out:
            c["_p_stop"] = ps
        return out

    # -- look-ahead: M13 roll-outs + network potentials + stop weight + adaptive trigger ---------------
    def lookahead(self, jobs, gen):
        cfg = self.cfg
        groups = {}
        for i, s, u, c in jobs:
            groups.setdefault((i, id(s), u), []).append((i, s, u, c))
        live = []
        for (i, _, u), js in groups.items():
            s = js[0][1]
            cands = [c for _, _, _, c in js]
            for c in cands:
                c["log_psi"] = 0.0
            f = self.features(i, s, u, cands, cands[0]["_p_stop"] if cands else 0.0)
            if not self.fires(f):
                self.stats[i]["la_skipped"] += 1
                continue
            self.stats[i]["la_fired"] += 1
            if self.voc_log is not None:
                for c in cands:
                    c["_feats"] = f                       # read back by select (same decision)
            for c in cands:
                if c["lw"] == NEG_INF or not open_end(c["o"]):
                    continue
                s2 = s.copy()
                s2.__dict__["_m14_extra_open"] = (u,)
                v = self.commit_candidate(s2, u, c)
                live.append([i, s2, v, c])
        for h in range(cfg.horizon):
            if not live:
                break
            items = []
            for i, s2, v, c in live:
                hd, flag = heading_of(s2.nodes, v, s2.parent[v])
                items.append((i, s2, v, hd, flag))
            ctx = self.contexts(items)
            B = len(items)
            stop_logit, mix, _, cl, sl = self.decode(ctx, torch.zeros(B, 1, 2, device=self.device), None,
                                                     torch.full((B, 1), -1, device=self.device, dtype=torch.long))
            stop_logit, mix, cl, sl = tempered(stop_logit, mix, cl, sl, cfg)
            prop = propose(stop_logit, mix, cl, sl, cfg.rollout_m, gen, cfg.structures)
            prop["_rollout"] = True
            nxt = []
            for b, (i, s2, v, c) in enumerate(live):
                cands = self.candidates(i, s2, v, items[b][3], prop, b)
                lw = np.array([x["lw"] for x in cands])
                ps = float(prop["p_stop"][b])
                p_s, p_k, Z = selection_probs(lw, ps, cands.stop_lw)
                c["log_psi"] += math.log(max(Z, 1e-300))
                self.stats[i]["lookahead_steps"] += 1
                if Z <= 0 or torch.rand(1, generator=gen).item() < p_s or p_k.sum() == 0:
                    continue
                k = int(torch.multinomial(torch.as_tensor(p_k / p_k.sum()), 1, generator=gen))
                vv = self.commit_candidate(s2, v, cands[k])
                if open_end(cands[k]["o"]):
                    nxt.append([i, s2, vv, c])
            live = nxt

    # -- selection: network stop weight, twist, trust region, VoC record -----------------------------
    def select(self, cands, p_stop, gen):
        cfg = self.cfg
        m = cfg.method
        if m not in ("energy", "smc", "filter"):
            return super().select(cands, p_stop, gen)
        lw0 = np.array([c["lw"] for c in cands])
        lw = lw0 + cfg.twist_beta * np.array([c.get("log_psi", 0.0) for c in cands])
        stop_lw = getattr(cands, "stop_lw", 0.0)
        beta = kl_beta(lw, p_stop, stop_lw, cfg.kl_eps) if cfg.kl_eps > 0 else 1.0
        if beta < 1.0:
            self._beta_sum += beta
            self._beta_n += 1
        p_s, p_k, Z = selection_probs(lw, p_stop, stop_lw, beta)
        if self.voc_log is not None and cands and "_feats" in cands[0]:
            q0s, q0k, _ = selection_probs(lw0, p_stop, stop_lw, beta)
            voc = 0.5 * (abs(p_s - q0s) + float(np.abs(p_k - q0k).sum()))
            self.voc_log.append({**cands[0]["_feats"], "voc": voc})
        # weight increment for SMC: the local normaliser of the UNTWISTED-at-beta-1 potentials (proper weighting);
        # the twist is corrected by the M13 twist bookkeeping when the node is popped
        if Z <= 0 or torch.rand(1, generator=gen).item() < p_s:
            return -1, math.log(max(Z, 1e-300))
        k = int(torch.multinomial(torch.as_tensor(p_k / p_k.sum()), 1, generator=gen))
        return k, math.log(max(Z, 1e-300))

    @torch.no_grad()
    def run(self, seed: int = 0, max_rounds: int = 1500):
        self._beta_sum, self._beta_n = 0.0, 0
        return super().run(seed, max_rounds)


NET_PRESETS = {
    "na_myopic": dict(method="energy", K=8, lam_ratio=0.5, hard=True, soft=True),
    "na_fixed": dict(method="energy", K=8, lam_ratio=0.5, hard=True, soft=True, horizon=2, rollout_m=4),
    "na_adaptive": dict(method="energy", K=8, lam_ratio=0.5, hard=True, soft=True, horizon=2, rollout_m=4),
    "na_smc": dict(method="smc", K=8, lam_ratio=0.5, hard=True, soft=True, particles=4),
}


def net_preset(name: str, **kw) -> NetConfig:
    base = NET_PRESETS.get(name) or PRESETS[name]
    allowed = {f.name for f in fields(NetConfig)}
    return NetConfig(**{k: v for k, v in {**base, **kw}.items() if k in allowed})


def sample_net(model, tiles, names, norm, pots, cfg: NetConfig, seed=0, device="cpu", crop_tiles=None, voc_log=None):
    import time
    sm = NetSampler(model, tiles, names, norm, pots, cfg, device, crop_tiles)
    sm.voc_log = voc_log
    gs = sm.run(seed)
    info = {"seconds": sm.seconds, "forward_passes": sm.n_fwd,
            "mean_beta": sm._beta_sum / sm._beta_n if sm._beta_n else 1.0}
    return gs, [dict(c) for c in sm.stats], info
