"""Milestone 13: TerrainConstrainedGraphTurtle - probabilistic inference over a trained TerrainGraphTurtle.

docs/M13_METHOD.md gives the formulation. The trained network is NOT changed; this module replaces the generation loop
of graphs/generate.sample_graphs with a family of samplers that share six separated parts:

1. proposal    ``propose``: K candidate actions from ONE decoder call (displacement, class, structure; plus the arm's
               stop probability) with their exact model log-probabilities;
2. feasibility ``evaluate_candidate``: every candidate is resolved by the SAME geometric rules that insert it
               (GenState.resolve_step / resolve_structure), then scored by graphs/tc_potentials on the committed edge;
3. ranking     ``select``: method-specific choice among candidates (likelihood, filter, potential-weighted, argmax);
4. search      ``lookahead``: Monte Carlo roll-outs of the new road's continuation (horizon H, M candidates per step)
               giving an estimated twist psi(s') of each candidate;
5. commitment  ``commit_candidate``: inserts the committed edge; the decoder is fed the COMMITTED displacement and class
               (state consistency), never a rejected or replaced sample;
6. management  particles (independent GenStates per tile), weights, ESS, systematic resampling (SMC) or top-B
               selection (beam), frontier scheduling.

Methods (SamplerConfig.method):
* ``legacy``    - the M12 loop exactly (graphs/generate.sample_graphs with TerrainPolicy "none"); control A0;
* ``standard``  - state-consistent ancestral sampling (K = 1, no constraints); control A;
* ``topk``      - K candidates, keep the k most likely, sample among them by likelihood (no terrain); B;
* ``fulltemp``  - K = 1 with a full-distribution temperature on every head; C;
* ``filter``    - K candidates, hard constraints only, Rao-Blackwellised stop; D;
* ``energy``    - K candidates, hard constraints and soft potentials; E (with H > 0: + look-ahead);
* ``beam``      - B particles, argmax candidate, top-B selection on length-normalised score; F;
* ``smc``       - N particles, potential-weighted K-candidate proposal, weights = local normaliser estimates, ESS
                  resampling; G (with H > 0: approximate twisted SMC).
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from urbanmorph.graphs.generate import (MAX_COMPONENTS, MAX_NODES, SNAP_START_M, GenState, clip_to_tile)
from urbanmorph.graphs.levels import BRIDGE, TUNNEL
from urbanmorph.graphs.road_graph import MERGE_M, TILE_M
from urbanmorph.graphs.sequence import (CELL, DISP_SCALE, GRID, MAX_T, PAD, arm_features, crops_turtle, heading_of,
                                        pixel_rc, rotate, scalar_features)
from urbanmorph.graphs.tc_potentials import NEG_INF, Potentials, TerrainField, edge_features
from urbanmorph.graphs.terrain_generate import TerrainPolicy, crosses_street, terrain_of
from urbanmorph.terrain.road_feasibility import water_kind_at


@dataclass
class SamplerConfig:
    method: str = "energy"
    K: int = 8                       # candidates per decision
    topk: int = 2                    # method topk: candidates kept
    temperature: float = 0.5         # displacement temperature of the base model (the M12 reference T = 0.5)
    full_temperature: float = 1.0    # method fulltemp: temperature of EVERY head (stop, class, structure, mixture)
    lam_ratio: float = 1.0           # weight of the density-ratio terrain potentials
    lam_infra: float = 1.0           # weight of the structure (span / angle) potentials
    lam_geom: float = 0.0            # weight of the junction-angle potential
    hard: bool = True                # apply hard constraints
    soft: bool = True                # apply soft potentials
    horizon: int = 0                 # look-ahead depth H (0 = none)
    rollout_m: int = 4               # candidates per roll-out step
    twist_beta: float = 1.0          # exponent of the estimated twist in selection
    particles: int = 1               # SMC particles / beam width per tile
    ess_frac: float = 0.5            # resample when ESS < ess_frac * particles
    scheduler: str = "fifo"          # fifo | class | confidence | connect | stochastic
    sched_window: int = 4            # stochastic scheduler: choose among the first L queue entries
    viaducts: bool = False           # exploratory: allow dry-valley bridges
    structures: bool = True          # use the learned structure head (else every step is ground)
    stop_bias: float = 0.0           # added to the arm stop logit (density-matched comparisons only, calibrated on DEV-A)

    def tag(self) -> str:
        return "_".join(f"{k}{v}" for k, v in asdict(self).items())


# ------------------------------------------------------------------------------------------------ geometry
def resolve_structure(s: GenState, u: int, d: np.ndarray) -> dict:
    """The edge a BRIDGE / TUNNEL step u -> u + d would create (graphs/terrain_generate.apply_bridge semantics: it passes
    over or under streets and water and lands on an existing ground node within MERGE_M or on a new node)."""
    a = s.nodes[u]
    q, clipped = clip_to_tile(a, a + d, s.size)
    out = {"kind": "dropped", "q": q, "clipped": bool(clipped), "grade_separated": False}
    if np.hypot(*(q - a)) < MERGE_M or s.n >= s.max_nodes:
        return out
    dist = np.hypot(*(s.nodes[:s.n] - q).T)
    dist[u] = np.inf
    if s.raised_nodes:
        dist[list(s.raised_nodes)] = np.inf
    v = int(dist.argmin()) if s.n > 1 else -1
    if v >= 0 and dist[v] < MERGE_M:
        return {**out, "kind": "struct_merge", "v": v, "q": s.nodes[v].copy()}
    return {**out, "kind": "struct_new"}


def commit_structure(s: GenState, u: int, o: dict, cls: int, st: int) -> int:
    if o["kind"] == "struct_merge":
        v = o["v"]
        s.reach(v, u)
    else:
        v = s.add_node(o["q"], parent=u)
        s.node_rule[v] = "bridge_end" if st == BRIDGE else "tunnel_end"
    s.connect(u, v, cls)
    s.raised[(min(u, v), max(u, v))] = (1, BRIDGE) if st == BRIDGE else (-1, TUNNEL)
    s.rules["bridge" if st == BRIDGE else "tunnel"] = s.rules.get("bridge" if st == BRIDGE else "tunnel", 0) + 1
    return v


def open_end(o: dict) -> bool:
    """Does the committed edge end at a NEW node (an open road end the turtle will continue)?"""
    return o["kind"] in ("new_node", "struct_new") and not o["clipped"]


def junction_angle(s: GenState, v: int, a: np.ndarray, q: np.ndarray) -> float:
    """Smallest angle (deg) between the new edge (arriving at q from a) and the existing edges at node v."""
    if v < 0 or not s.adj[v]:
        return np.nan
    back = a - q
    hb = math.degrees(math.atan2(back[1], back[0]))
    angs = [abs((math.degrees(math.atan2(*(s.nodes[w] - s.nodes[v])[::-1])) - hb + 180) % 360 - 180) for w in s.adj[v]]
    return float(min(angs)) if angs else np.nan


# ------------------------------------------------------------------------------------------------ proposal
def tempered(stop_logit, mix, cls_logit, struct_logit, cfg: SamplerConfig):
    lp, mu, ls, rho = mix
    ls = ls + math.log(cfg.temperature)
    stop_logit = stop_logit + cfg.stop_bias
    if cfg.method == "fulltemp":
        T = cfg.full_temperature
        lp, ls = lp / T, ls - math.log(cfg.temperature) + math.log(T)
        stop_logit, cls_logit = stop_logit / T, cls_logit / T
        struct_logit = struct_logit / T if struct_logit is not None else None
    return stop_logit, (lp, mu, ls, rho), cls_logit, struct_logit


def mixture_logpdf(x, lp, mu, ls, rho):
    """log density of x (B, K, 2) under per-row mixtures lp (B, C), mu/ls (B, C, 2), rho (B, C)."""
    z = (x.unsqueeze(-2) - mu.unsqueeze(1)) / ls.exp().unsqueeze(1)
    r = rho.unsqueeze(1)
    one_m = 1 - r ** 2
    quad = (z[..., 0] ** 2 + z[..., 1] ** 2 - 2 * r * z[..., 0] * z[..., 1]) / one_m
    comp = -math.log(2 * math.pi) - ls.sum(-1).unsqueeze(1) - 0.5 * torch.log(one_m) - 0.5 * quad
    return torch.logsumexp(F.log_softmax(lp, -1).unsqueeze(1) + comp, -1)


def propose(stop_logit, mix, cls_logit, struct_logit, K: int, gen, structures=True):
    """K iid continuation candidates per row (B rows). Returns dicts of numpy arrays (B, K, ...)."""
    lp, mu, ls, rho = (x.float().cpu() for x in mix)
    B, C = lp.shape
    comp = torch.multinomial(F.softmax(lp, -1), K, replacement=True, generator=gen)          # (B, K)
    idx = torch.arange(B)[:, None].expand(B, K)
    m, s, r = mu[idx, comp], ls[idx, comp].exp(), rho[idx, comp]
    e = torch.randn(B, K, 2, generator=gen)
    d = m + s * torch.stack([e[..., 0], r * e[..., 0] + torch.sqrt(1 - r ** 2) * e[..., 1]], -1)
    logp_d = mixture_logpdf(d, lp, mu, ls, rho)
    lc = F.log_softmax(cls_logit.float().cpu(), -1)
    c = torch.multinomial(lc.exp(), K, replacement=True, generator=gen)
    logp_c = lc.gather(1, c)
    if structures and struct_logit is not None:
        lsf = F.log_softmax(struct_logit.float().cpu(), -1)
        st = torch.multinomial(lsf.exp(), K, replacement=True, generator=gen)
        logp_s = lsf.gather(1, st)
    else:
        st, logp_s = torch.zeros(B, K, dtype=torch.long), torch.zeros(B, K)
    p_stop = torch.sigmoid(stop_logit.float().cpu())
    logp = logp_d + logp_c + logp_s + torch.log1p(-p_stop.clamp(max=1 - 1e-6))[:, None]
    return {"d": d.numpy(), "cls": c.numpy(), "struct": st.numpy(), "logp": logp.numpy(), "p_stop": p_stop.numpy()}


# ------------------------------------------------------------------------------------------------ the sampler
class TCSampler:
    def __init__(self, model, tiles: list[dict], names, norm, pots: Potentials, cfg: SamplerConfig, device="cpu",
                 crop_tiles: list[dict] | None = None):
        self.model, self.cfg, self.device = model, cfg, device
        self.pots = pots
        self.pots.viaducts = cfg.viaducts
        self.pol = TerrainPolicy(tiles, names, norm, mode="none", crop_tiles=crop_tiles)
        self.tfs = [TerrainField(terrain_of(d)) for d in tiles]
        self.stats = [Counter() for _ in tiles]
        self.n_fwd = 0

    # -- model calls -------------------------------------------------------------------------------
    def contexts(self, items):
        """items: list of (tile i, GenState s, node u, heading hd). One batched context call."""
        crops, arms, masks, scal, tcrops = [], [], [], [], []
        for i, s, u, hd, flag in items:
            crops.append(crops_turtle(s.edge_map, s.created_map, s.popped_map, s.nodes[u], hd, [s.t], s.size)[0])
            sc = scalar_features(self.local_pos(s.nodes[u]), np.array([s.border[u]]), np.array([len(s.adj[u])]),
                                 np.array([s.n]), np.array([s.t]))[0]
            scal.append(np.r_[sc, np.float32(flag)].astype(np.float32))
            a, m = arm_features([rotate(s.nodes[w] - s.nodes[u], -hd) for w in s.adj[u]])
            arms.append(a)
            masks.append(m)
            tcrops.append(self.pol.crop_extra(i, s.nodes[u], hd))
        T = lambda x: torch.from_numpy(np.stack(x)).to(self.device)
        self.n_fwd += 1
        return self.model.context(T(crops), T(scal), T(arms), T(masks), tcrops=T(tcrops))

    def decode(self, ctx, prev, h, prev_c):
        self.n_fwd += 1
        stop_logit, mix, h, cl = self.model.decode_steps(ctx, prev, h, prev_c)
        sl = self.model.last_struct_logit
        mix = tuple(x[:, 0] for x in mix)
        return stop_logit[:, 0], mix, h, cl[:, 0], (sl[:, 0] if sl is not None else None)

    # -- candidates --------------------------------------------------------------------------------
    def evaluate_candidate(self, i, s: GenState, u, dd, cls, st):
        """Resolve a candidate by the insertion rules and score the COMMITTED edge."""
        cfg = self.cfg
        a = s.nodes[u]
        if st in (BRIDGE, TUNNEL):
            o = resolve_structure(s, u, dd)
        else:
            o = s.resolve_step(u, dd)
            st = 0
        q = o["q"]
        if o["kind"] == "dropped":
            return o, st, (NEG_INF if cfg.hard else 0.0), "unrepresentable"
        over = bool(st == BRIDGE and crosses_street(s, a, q))
        f = edge_features(self.tfs[i], a, q, struct=st, over_street=over)
        v = o.get("v", -1)
        ja = junction_angle(s, v, a, q) if (self.pots.lam_geom and v >= 0) else np.nan
        lw, reason, _ = self.pots.log_potential(f, int(cls), o["kind"], ja)
        if reason and not cfg.hard:
            lw = 0.0
        elif not reason and not cfg.soft:
            lw = 0.0
        o["feat"] = f
        return o, st, lw, reason

    def candidates(self, i, s, u, hd, prop, b):
        out = []
        for k in range(prop["d"].shape[1]):
            dd = rotate(prop["d"][b, k] * DISP_SCALE, hd)
            st = int(prop["struct"][b, k])
            o, st, lw, reason = self.evaluate_candidate(i, s, u, dd, int(prop["cls"][b, k]), st)
            out.append({"o": o, "st": st, "cls": int(prop["cls"][b, k]), "lw": lw, "reason": reason,
                        "logp": float(prop["logp"][b, k])})
            self.stats[i]["cand_evals"] += 1
            if reason:
                self.stats[i][f"hard_{reason}"] += 1
        keys = {(c["o"]["kind"], c["st"], round(float(c["o"]["q"][0])), round(float(c["o"]["q"][1]))) for c in out}
        self.stats[i]["cand_unique"] += len(keys)
        return out

    # -- look-ahead --------------------------------------------------------------------------------
    def lookahead(self, jobs, gen):
        """jobs: list of (tile i, GenState s, node u, candidate c). Sets c["log_psi"] by Monte Carlo roll-outs of
        the new road's continuation: psi = prod_h Z_h, Z_h = p_stop + (1 - p_stop) mean_m w_m (local normaliser)."""
        cfg = self.cfg
        live = []
        for i, s, u, c in jobs:
            c["log_psi"] = 0.0
            if c["lw"] == NEG_INF or not open_end(c["o"]):
                continue
            s2 = s.copy()
            v = self.commit_candidate(s2, u, c)
            if getattr(self, "trace", None) is not None:
                c["rollout"] = [s2.nodes[u].tolist(), s2.nodes[v].tolist()]
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
            nxt = []
            for b, (i, s2, v, c) in enumerate(live):
                cands = self.candidates(i, s2, v, items[b][3], prop, b)
                w = np.array([math.exp(x["lw"]) if x["lw"] > NEG_INF else 0.0 for x in cands])
                ps = float(prop["p_stop"][b])
                Z = ps + (1 - ps) * w.mean()
                c["log_psi"] += math.log(max(Z, 1e-300))
                self.stats[i]["lookahead_steps"] += 1
                if torch.rand(1, generator=gen).item() < ps / max(Z, 1e-300) or w.sum() == 0:
                    continue                                   # the road stops (or cannot continue): roll-out ends
                k = int(torch.multinomial(torch.as_tensor(w / w.sum()), 1, generator=gen))
                vv = self.commit_candidate(s2, v, cands[k])
                if "rollout" in c:
                    c["rollout"].append(s2.nodes[vv].tolist() if vv >= 0 else cands[k]["o"]["q"].tolist())
                if open_end(cands[k]["o"]):
                    nxt.append([i, s2, vv, c])
            live = nxt

    # -- commit ------------------------------------------------------------------------------------
    @staticmethod
    def commit_candidate(s: GenState, u, c) -> int:
        if c["st"] in (BRIDGE, TUNNEL):
            return commit_structure(s, u, c["o"], c["cls"], c["st"])
        return s.commit(u, c["o"], c["cls"])

    # -- selection ---------------------------------------------------------------------------------
    def select(self, cands, p_stop, gen):
        """(index or -1 for STOP, log Z estimate). Rao-Blackwellised over the stop decision where weights apply."""
        m, cfg = cfg_method(self.cfg), self.cfg
        if m in ("standard", "fulltemp"):
            if torch.rand(1, generator=gen).item() < p_stop:
                return -1, 0.0
            return 0, 0.0
        if m == "topk":
            if torch.rand(1, generator=gen).item() < p_stop:
                return -1, 0.0
            lp = np.array([c["logp"] for c in cands])
            keep = np.argsort(-lp)[:cfg.topk]
            w = np.exp(lp[keep] - lp[keep].max())
            return int(keep[int(torch.multinomial(torch.as_tensor(w / w.sum()), 1, generator=gen))]), 0.0
        lw = np.array([c["lw"] + cfg.twist_beta * c.get("log_psi", 0.0) for c in cands])
        w = np.where(np.isfinite(lw), np.exp(np.minimum(lw, 50)), 0.0)
        Z = p_stop + (1 - p_stop) * w.mean()
        if m == "beam":
            # the stop decision is SAMPLED: comparing log p(stop) with continuation scores that contain log DENSITIES
            # of the continuous displacement would almost never stop (mode-seeking degeneracy); the continuation is
            # the argmax of log p + log w among feasible candidates, and particles are pruned on the mean score.
            score = np.where(np.isfinite(lw), np.array([c["logp"] for c in cands]) + lw, -np.inf)
            k = int(np.argmax(score))
            if not np.isfinite(score[k]) or torch.rand(1, generator=gen).item() < p_stop:
                return -1, math.log(max(Z, 1e-300))
            return k, math.log(max(Z, 1e-300))
        if Z <= 0 or torch.rand(1, generator=gen).item() < p_stop / Z:
            return -1, math.log(max(Z, 1e-300))
        k = int(torch.multinomial(torch.as_tensor(w / w.sum()), 1, generator=gen))
        return k, math.log(max(Z, 1e-300))

    # -- scheduler ---------------------------------------------------------------------------------
    def pop(self, s: GenState, gen, conf: dict):
        cfg = self.cfg
        q = s.queue
        if cfg.scheduler == "fifo" or len(q) == 1:
            return q.popleft()
        L = list(q)
        if cfg.scheduler == "class":
            pr = [min((c for a_, b_, c in s.edges if a_ >= 0 and v in (a_, b_)), default=3) for v in L]
            k = int(np.argmin(pr))
        elif cfg.scheduler == "confidence":
            k = int(np.argmax([conf.get(v, -1e9) for v in L]))
        elif cfg.scheduler == "connect":
            comp = components(s)
            best, k = np.inf, 0
            for j, v in enumerate(L):
                other = np.flatnonzero(comp[:s.n] != comp[v])
                if len(other):
                    dmin = float(np.hypot(*(s.nodes[other] - s.nodes[v]).T).min())
                    if dmin < best:
                        best, k = dmin, j
        else:                                                                    # bounded stochastic
            W = L[:cfg.sched_window]
            pr = np.array([min((c for a_, b_, c in s.edges if a_ >= 0 and v in (a_, b_)), default=3) for v in W], float)
            w = np.exp(-pr)
            k = int(torch.multinomial(torch.as_tensor(w / w.sum()), 1, generator=gen))
        v = L[k]
        del q[k]
        return v

    # -- hooks (overridden by the global-domain sampler) ----------------------------------------------
    def local_pos(self, p):
        """Position for the model's scalar features (tile coordinates)."""
        return p

    def new_state(self):
        return GenState()

    def do_starts(self, need, gen, rng):
        """Start-head decisions for states whose queue is empty: end the tile or place a start node."""
        cfg, model = self.cfg, self.model
        canvas = torch.from_numpy(np.stack([(s.edge_map[PAD:PAD + GRID, PAD:PAD + GRID] < s.t + 0.5)[None]
                                            for _, _, s in need])).to(self.device)
        ncomp = torch.tensor([float(s.ncomp) for _, _, s in need], device=self.device)
        tcan = torch.from_numpy(np.stack([self.pol.start_extra(i) for i, _, _ in need])).to(self.device)
        stop_logit, loc = model.start_outputs(canvas, ncomp, tcanvas=tcan)
        self.n_fwd += 1
        p_stop = torch.sigmoid(stop_logit).cpu()
        cells = torch.multinomial(torch.softmax(loc, -1).cpu(), 1, generator=gen).squeeze(1).numpy()
        stop = (torch.rand(len(need), generator=gen) < p_stop).numpy()
        for (i, p, s), stp, cell in zip(need, stop, cells):
            if stp or s.ncomp >= MAX_COMPONENTS or s.n >= s.max_nodes:
                s.done = True
                continue
            row, col = divmod(int(cell), GRID // CELL)
            px = (col * CELL + rng.uniform(0, CELL)) * 4.0
            py = TILE_M - (row * CELL + rng.uniform(0, CELL)) * 4.0
            s.ncomp += 1
            self.place_start(i, s, px, py)

    def place_start(self, i, s, px, py):
        cfg = self.cfg
        t = self.tfs[i].t
        wk = water_kind_at(t, np.array([px]), np.array([py]))[0] if getattr(t, "water_kind", None) is not None else 0
        if cfg.hard and cfg_method(cfg) not in ("standard", "fulltemp", "topk") and wk > 0:
            self.stats[i]["start_in_water"] += 1
            return
        s.node_rule[s.add_node(np.array([px, py]))] = "start"

    # -- main loop ---------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, seed: int = 0, max_rounds: int = 1500):
        cfg, model = self.cfg, self.model
        model.eval()
        gen = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        nT, P = len(self.tfs), max(1, cfg.particles)
        S = [[self.new_state() for _ in range(P)] for _ in range(nT)]
        logW = np.zeros((nT, P))
        score = np.zeros((nT, P))
        ndec = np.zeros((nT, P))
        conf = [[{} for _ in range(P)] for _ in range(nT)]
        twist = [[{} for _ in range(P)] for _ in range(nT)]
        t0 = time.time()
        for rnd in range(max_rounds):
            flat = [(i, p, S[i][p]) for i in range(nT) for p in range(P)]
            need = [(i, p, s) for i, p, s in flat if not s.done and not s.queue]
            if need:
                self.do_starts(need, gen, rng)
            active = [(i, p, s) for i, p, s in flat if not s.done and s.queue]
            if not active:
                if all(s.done for _, _, s in flat):
                    break
                continue
            items, us = [], []
            for i, p, s in active:
                u = self.pop(s, gen, conf[i][p])
                r, c = pixel_rc(s.nodes[u:u + 1], s.size)
                s.popped_map[r[0] + PAD, c[0] + PAD] = s.t
                if u in twist[i][p]:                                       # approximate twisted SMC: consume psi
                    logW[i, p] -= cfg.twist_beta * twist[i][p].pop(u)
                hd, flag = heading_of(s.nodes, u, s.parent[u])
                items.append((i, s, u, hd, flag))
                us.append(u)
            ctx = self.contexts(items)
            B = len(items)
            prev = torch.zeros(B, 1, 2, device=self.device)
            prev_c = torch.full((B, 1), -1, device=self.device, dtype=torch.long)
            h = None
            alive = np.ones(B, bool)
            for k in range(MAX_T):
                stop_logit, mix, h, cl, sl = self.decode(ctx, prev, h, prev_c)
                stop_logit, mix, cl, sl = tempered(stop_logit, mix, cl, sl, cfg)
                Kc = 1 if cfg_method(cfg) in ("standard", "fulltemp") else cfg.K
                prop = propose(stop_logit, mix, cl, sl, Kc, gen, cfg.structures)
                new_prev = prev.clone()
                new_pc = prev_c.clone()
                per = []
                for b in np.flatnonzero(alive):
                    i, s, u, hd, _ = items[b]
                    per.append((b, self.candidates(i, s, u, hd, prop, b)))
                if cfg.horizon > 0:
                    self.lookahead([(items[b][0], items[b][1], items[b][2], c) for b, cs in per for c in cs], gen)
                for b, cs in per:
                    i, s, u, hd, _ = items[b]
                    p = active[b][1]
                    j, logZ = self.select(cs, float(prop["p_stop"][b]), gen)
                    if getattr(self, "trace", None) is not None and i == 0 and active[b][1] == 0:
                        self.trace.append({"u": s.nodes[u].tolist(), "heading": hd, "chosen": j, "p_stop": float(prop["p_stop"][b]),
                                           "round": rnd, "n_nodes": s.n,
                                           "cands": [{"q": c["o"]["q"].tolist(), "kind": c["o"]["kind"], "st": c["st"],
                                                      "lw": c["lw"] if np.isfinite(c["lw"]) else None, "reason": c["reason"],
                                                      "logp": c["logp"], "log_psi": c.get("log_psi"), "rollout": c.get("rollout")}
                                                     for c in cs]})
                    if cfg_method(cfg) in ("smc", "beam", "filter", "energy"):
                        logW[i, p] += logZ
                    if j < 0:
                        alive[b] = False
                        self.stats[i]["stop"] += 1
                        if logZ < math.log(1e-3) and cfg_method(cfg) not in ("standard", "fulltemp", "topk"):
                            self.stats[i]["forced_stop"] += 1
                        continue
                    c = cs[j]
                    if c["reason"] and cfg.hard and cfg_method(cfg) not in ("standard", "fulltemp", "topk"):
                        alive[b] = False                                # cannot happen: zero weight is never chosen
                        continue
                    if c["o"]["kind"] == "dropped":                     # unconstrained methods: nothing inserted
                        self.stats[i]["dropped"] += 1
                        alive[b] = False
                        continue
                    v = self.commit_candidate(s, u, c)
                    self.stats[i]["committed"] += 1
                    self.stats[i][f"committed_struct{c['st']}"] += 1
                    score[i, p] += c["logp"] + (c["lw"] if np.isfinite(c["lw"]) else 0.0)
                    ndec[i, p] += 1
                    if v >= 0:
                        conf[i][p][v] = c["logp"]
                        if cfg.horizon > 0 and open_end(c["o"]):
                            twist[i][p][v] = c.get("log_psi", 0.0)
                    dq = rotate(c["o"]["q"] - s.nodes[u], -hd) / DISP_SCALE     # the COMMITTED displacement
                    new_prev[b, 0] = torch.as_tensor(dq, dtype=new_prev.dtype)
                    new_pc[b, 0] = c["cls"]
                prev, prev_c = new_prev, new_pc
                if not alive.any():
                    break
            for _, _, s in active:
                s.t += 1
            if P > 1:
                self.resample(S, logW, score, ndec, conf, twist, rng)
        self.seconds = time.time() - t0
        self.states = S                                     # kept for figures and diagnostics
        out = []
        for i in range(nT):
            if cfg_method(cfg) == "beam":
                p = int(np.argmax(score[i] / np.maximum(ndec[i], 1)))
            elif P > 1:
                w = np.exp(logW[i] - logW[i].max())
                p = int(rng.choice(P, p=w / w.sum()))
            else:
                p = 0
            self.stats[i]["chosen_particle_logw"] = float(logW[i, p])
            out.append(S[i][p].graph(with_class=True))
        return out

    def resample(self, S, logW, score, ndec, conf, twist, rng):
        cfg = self.cfg
        P = cfg.particles
        for i in range(len(S)):
            if cfg_method(cfg) == "beam":
                sc = score[i] / np.maximum(ndec[i], 1)
                order = np.argsort(-sc)
                keep = order[: max(1, P // 2)]
                anc = np.r_[keep, keep][:P]
                self.stats[i]["beam_unique_ancestors"] += len(set(anc.tolist()))
            else:
                w = np.exp(logW[i] - logW[i].max())
                w /= w.sum()
                ess = 1.0 / np.sum(w ** 2)
                self.stats[i]["ess_sum"] += ess
                self.stats[i]["ess_n"] += 1
                if ess >= cfg.ess_frac * P:
                    continue
                pos = (rng.uniform() + np.arange(P)) / P                     # systematic resampling
                anc = np.minimum(np.searchsorted(np.cumsum(w), pos), P - 1)
                self.stats[i]["resamples"] += 1
                logW[i] = np.log(np.mean(np.exp(logW[i] - logW[i].max()))) + logW[i].max()
            S[i] = [S[i][a].copy() for a in anc]
            score[i], ndec[i] = score[i][anc], ndec[i][anc]
            conf[i] = [dict(conf[i][a]) for a in anc]
            twist[i] = [dict(twist[i][a]) for a in anc]
            if cfg_method(cfg) == "beam":
                continue


def cfg_method(cfg: SamplerConfig) -> str:
    return cfg.method


PRESETS = {
    "legacy": dict(method="legacy"),
    "standard": dict(method="standard", K=1, hard=False, soft=False),
    "topk": dict(method="topk", K=8, topk=2, hard=False, soft=False),
    "fulltemp": dict(method="fulltemp", K=1, hard=False, soft=False, full_temperature=0.7),
    "filter": dict(method="filter", K=8, hard=True, soft=False),
    "softonly": dict(method="energy", K=8, hard=False, soft=True),
    "energy": dict(method="energy", K=8, hard=True, soft=True),
    "energy_la": dict(method="energy", K=8, hard=True, soft=True, horizon=4, rollout_m=4),
    "beam": dict(method="beam", K=8, hard=True, soft=True, particles=4),
    "smc": dict(method="smc", K=8, hard=True, soft=True, particles=8),
    "smc_la": dict(method="smc", K=8, hard=True, soft=True, particles=8, horizon=4, rollout_m=4),
}


def preset(name: str, **kw) -> SamplerConfig:
    return SamplerConfig(**{**PRESETS[name], **kw})


def components(s: GenState) -> np.ndarray:
    parent = np.arange(s.n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in s.edges:
        if a >= 0:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(x) for x in range(s.n)])


def sample_tc(model, tiles, names, norm, pots, cfg: SamplerConfig, seed=0, device="cpu", crop_tiles=None):
    """Convenience wrapper: graphs and per-tile statistics."""
    if cfg.method == "legacy":
        from urbanmorph.graphs.generate import sample_graphs
        pol = TerrainPolicy(tiles, names, norm, mode="none", crop_tiles=crop_tiles)
        t0 = time.time()
        gs = sample_graphs(model, len(tiles), device=device, seed=seed, temperature=cfg.temperature, terrain=pol)
        return gs, [dict(c) for c in pol.stats], {"seconds": time.time() - t0, "forward_passes": np.nan}
    sm = TCSampler(model, tiles, names, norm, pots, cfg, device, crop_tiles)
    gs = sm.run(seed)
    return gs, [dict(c) for c in sm.stats], {"seconds": sm.seconds, "forward_passes": sm.n_fwd}
