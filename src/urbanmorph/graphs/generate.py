"""Sampling street graphs from the trained GraphTurtle (Milestone 5).

Many tiles grow in lock-step (one pop per tile per round) so the network runs on batches.
Each sampled step from node u to p = u + d is turned into geometry by fixed rules, in order:

1. clip u->p to the tile (a street leaving the window ends on the border, like the clipped data);
2. if u->p CROSSES an existing street, stop at the crossing and create a junction there
   (in the OSM graph streets that cross at grade share a node);
3. else if p is within MERGE_M of an existing node, join it (loop closing, NTG's merge);
4. else if p is within MERGE_M of an existing street, join it with a T-junction;
5. else if the step is shorter than MERGE_M, drop it; otherwise create a new node and queue it.

Every rule firing is recorded (``GenState.rules`` counts, ``GenState.node_rule`` = the rule that created
a node; Milestone 10 diagnosis). Recording changes nothing in the sampling.

Milestone 10 options of ``sample_graphs`` (defaults reproduce Milestone 6 exactly): ``cond`` (one conditioning
vector per tile, for models with ``n_cond``) and ``given`` (one RoadGraph per tile, pre-drawn before sampling:
the hierarchical stage 2). With ``given``, a sampled start point within SNAP_START_M of a not-yet-reached given
node starts there, heading along a random given edge into it (as in training, sequence.py); a given node is
queued the first time a generated street joins it.

These rules make every component connected by construction. They do not impose grids, angles,
lengths or a number of components: those come only from the learned distributions.
Rules 2 and 4 are not in the training data's step sequences (a train/generate mismatch,
documented in docs/MILESTONE_5.md).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from urbanmorph.graphs.road_graph import MERGE_M, TILE_M, RoadGraph, is_border
from urbanmorph.graphs.sequence import (CANVAS, CELL, DISP_SCALE, GRID, MAX_T, PAD, arm_features,
                                        crops_from_maps, crops_turtle, heading_of, pixel_rc, rotate,
                                        scalar_features, segment_pixels)

MAX_NODES = 1000
MAX_COMPONENTS = 40
SNAP_START_M = 16.0     # one start cell (training start cells contain the given node exactly)


def clip_to_tile(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, bool]:
    """Liang-Barsky: point where a->b leaves [0, TILE_M]^2 (a is inside or on the border)."""
    d = b - a
    t1 = 1.0
    for k in range(2):
        if d[k] > 0:
            t1 = min(t1, (TILE_M - a[k]) / d[k])
        elif d[k] < 0:
            t1 = min(t1, (0 - a[k]) / d[k])
    t1 = max(t1, 0.0)
    return a + t1 * d, t1 < 1.0


class GenState:
    def __init__(self):
        self.nodes = np.zeros((MAX_NODES + 8, 2))
        self.border = np.zeros(MAX_NODES + 8, bool)
        self.n = 0
        self.adj: list[set] = []
        self.edges: list[list[int]] = []   # [a, b, street class]; removed edges get a = -1
        self.parent: list[int] = []
        self.queue: deque = deque()
        self.edge_map = np.full((CANVAS, CANVAS), 1e9)   # drawn = 0 (same "< t" convention as training)
        self.created_map = np.full((CANVAS, CANVAS), 1e9)
        self.popped_map = np.full((CANVAS, CANVAS), 1e9)
        self.t = 0
        self.ncomp = 0
        self.done = False
        self.rules = {"clipped": 0, "crossing_split": 0, "crossing_endpoint": 0, "merge_node": 0, "t_junction": 0,
                      "new_node": 0, "dropped_short": 0}
        self.node_rule: dict = {}          # node -> "start" / "new_node" / "crossing_split" / "t_junction"
        self.merged_into: dict = {}        # node -> times a step was joined to it (rules 2 endpoint, 3)
        self.given_adj: dict = {}          # given node -> its given neighbours (Milestone 10)
        self.reached: set = set()          # given nodes already queued
        self.head_override: dict = {}      # start node on the given network -> heading

    def add_given(self, g: RoadGraph) -> None:
        """Pre-draw a graph (time -1; its nodes are never "open")."""
        ids = [self.add_node(p, push=False) for p in g.nodes]
        for i in ids:
            r, c = pixel_rc(self.nodes[i])
            self.created_map[r[0] + PAD, c[0] + PAD] = -1.0
            self.popped_map[r[0] + PAD, c[0] + PAD] = -1.0
            self.node_rule[i] = "given"
        for (a, b), cls in zip(g.edges, g.classes()):
            self.connect(ids[a], ids[b], int(cls))
            self.given_adj.setdefault(ids[a], []).append(ids[b])
            self.given_adj.setdefault(ids[b], []).append(ids[a])

    def reach(self, v: int, u: int) -> None:
        if v in self.given_adj and v not in self.reached:
            self.reached.add(v)
            self.parent[v] = u
            self.queue.append(v)

    # -- geometry helpers
    def add_node(self, p, push=True, parent=-1) -> int:
        i = self.n
        self.parent.append(parent)
        self.nodes[i] = p
        self.border[i] = bool(is_border(p)[0])
        self.adj.append(set())
        self.n += 1
        r, c = pixel_rc(p)
        self.created_map[r[0] + PAD, c[0] + PAD] = self.t - 0.5
        if push:
            self.queue.append(i)
        return i

    def connect(self, a, b, cls=2):
        if a == b or b in self.adj[a]:
            return
        self.adj[a].add(b)
        self.adj[b].add(a)
        self.edges.append([a, b, cls])
        r, c = segment_pixels(self.nodes[a], self.nodes[b])
        self.edge_map[r, c] = -1.0

    def live_edges(self, with_class=False):
        e = np.array([x for x in self.edges if x[0] >= 0], int).reshape(-1, 3)
        return e if with_class else e[:, :2]

    def split_edge(self, a, b, p, parent=-1) -> int:
        cls = 2
        for x in self.edges:
            if x[0] >= 0 and {x[0], x[1]} == {a, b}:
                x[0] = -1
                cls = x[2]
                break
        self.adj[a].discard(b)
        self.adj[b].discard(a)
        s = self.add_node(p, parent=parent)
        for v in (a, b):
            self.adj[s].add(v)
            self.adj[v].add(s)
            self.edges.append([s, v, cls])        # both halves keep the street's class
        return s

    def apply_step(self, u: int, d: np.ndarray, cls: int = 2) -> None:
        a = self.nodes[u]
        q, clipped = clip_to_tile(a, a + d)
        self.rules["clipped"] += int(clipped)
        if np.hypot(*(q - a)) < 1.0:
            self.rules["dropped_short"] += 1
            return
        e = self.live_edges()
        if len(e):
            inc = (e[:, 0] == u) | (e[:, 1] == u)
            A, B = self.nodes[e[:, 0]], self.nodes[e[:, 1]]
            # 2. first crossing of a->q with an existing street
            r, s = q - a, B - A
            den = r[0] * s[:, 1] - r[1] * s[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = ((A[:, 0] - a[0]) * s[:, 1] - (A[:, 1] - a[1]) * s[:, 0]) / den
                uu = ((A[:, 0] - a[0]) * r[1] - (A[:, 1] - a[1]) * r[0]) / den
            hit = (~inc) & (np.abs(den) > 1e-9) & (tt > 1e-6) & (tt <= 1) & (uu >= 0) & (uu <= 1)
            hit &= tt * np.hypot(*r) > 1.0
            if hit.any():
                k = int(np.flatnonzero(hit)[np.argmin(tt[hit])])
                x = a + tt[k] * r
                ea, eb = e[k]
                ends = [v for v in (ea, eb) if np.hypot(*(self.nodes[v] - x)) < MERGE_M]
                if ends:
                    v = min(ends, key=lambda v: np.hypot(*(self.nodes[v] - x)))
                    self.rules["crossing_endpoint"] += 1
                    self.merged_into[v] = self.merged_into.get(v, 0) + 1
                    self.reach(v, u)
                else:
                    v = self.split_edge(ea, eb, x, u)
                    self.rules["crossing_split"] += 1
                    self.node_rule[v] = "crossing_split"
                self.connect(u, v, cls)
                return
        # 3. nearest existing node
        if self.n > 1:
            dist = np.hypot(*(self.nodes[:self.n] - q).T)
            dist[u] = np.inf
            v = int(dist.argmin())
            if dist[v] < MERGE_M:
                self.rules["merge_node"] += 1
                self.merged_into[v] = self.merged_into.get(v, 0) + 1
                self.reach(v, u)
                self.connect(u, v, cls)
                return
        # 4. nearest existing street (T-junction)
        if len(e):
            keep = ~inc
            if keep.any():
                A2, B2 = A[keep], B[keep]
                s2 = B2 - A2
                lam = np.clip(((q - A2) * s2).sum(1) / np.maximum((s2 ** 2).sum(1), 1e-9), 0, 1)
                proj = A2 + lam[:, None] * s2
                dd = np.hypot(*(proj - q).T)
                k = int(dd.argmin())
                if dd[k] < MERGE_M and 0 < lam[k] < 1:
                    ea, eb = e[keep][k]
                    v = self.split_edge(ea, eb, proj[k], u)
                    self.rules["t_junction"] += 1
                    self.node_rule[v] = "t_junction"
                    self.connect(u, v, cls)
                    return
        # 5. new node
        if np.hypot(*(q - a)) < MERGE_M or self.n >= MAX_NODES:
            self.rules["dropped_short"] += 1
            return
        v = self.add_node(q, parent=u)
        self.rules["new_node"] += 1
        self.node_rule[v] = "new_node"
        self.connect(u, v, cls)

    def graph(self, with_class=False) -> RoadGraph:
        ec = self.live_edges(with_class=True)
        if len(ec):
            ec[:, :2] = np.sort(ec[:, :2], axis=1)
            _, first = np.unique(ec[:, :2], axis=0, return_index=True)
            ec = ec[np.sort(first)]
        e, cls = ec[:, :2], ec[:, 2]
        used = np.unique(e.ravel()) if len(e) else np.zeros(0, int)
        remap = -np.ones(self.n, int)
        remap[used] = np.arange(len(used))
        nodes = self.nodes[used]
        return RoadGraph(nodes, remap[e] if len(e) else e.reshape(0, 2), is_border(nodes) if len(nodes) else np.zeros(0, bool),
                         cls.copy() if with_class else None)


@torch.no_grad()
def sample_graphs(model, n: int, device="cpu", seed: int = 0, temperature: float = 1.0,
                  max_rounds: int = 1500, states_out: list | None = None, step_log: list | None = None,
                  cond: np.ndarray | None = None, given: list | None = None) -> list[RoadGraph]:
    """Grow n street graphs. temperature scales the step-mixture spread (1.0 = the learned model).
    ``states_out`` receives the GenStates (rule counts); ``step_log`` receives one tuple per sampled step
    (tile, pop round t, start flag, dx, dy in the turtle frame in metres) - diagnostics only."""
    from urbanmorph.models.graph_gen import mixture_sample
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = [GenState() for _ in range(n)]
    sid = {id(s): i for i, s in enumerate(states)}
    C = None if cond is None else torch.as_tensor(np.asarray(cond, np.float32), device=device)
    if given is not None:
        for s, g in zip(states, given):
            s.add_given(g)
    for _ in range(max_rounds):
        # ---- start a new component where the queue is empty
        need = [s for s in states if not s.done and not s.queue]
        if need:
            canvas = torch.from_numpy(np.stack([(s.edge_map[PAD:PAD + GRID, PAD:PAD + GRID] < s.t + 0.5)[None]
                                                for s in need])).to(device)
            ncomp = torch.tensor([float(s.ncomp) for s in need], device=device)
            stop_logit, loc = model.start_outputs(canvas, ncomp, None if C is None else C[[sid[id(s)] for s in need]])
            p_stop = torch.sigmoid(stop_logit).cpu()
            cells = torch.multinomial(torch.softmax(loc, -1).cpu(), 1, generator=gen).squeeze(1).numpy()
            stop = (torch.rand(len(need), generator=gen) < p_stop).numpy()
            for s, st, cell in zip(need, stop, cells):
                if st or s.ncomp >= MAX_COMPONENTS or s.n >= MAX_NODES:
                    s.done = True
                    continue
                n_side = GRID // CELL
                row, col = divmod(int(cell), n_side)
                px = (col * CELL + rng.uniform(0, CELL)) * 4.0
                py = TILE_M - (row * CELL + rng.uniform(0, CELL)) * 4.0
                s.ncomp += 1
                free = [v for v in s.given_adj if v not in s.reached]
                if free:
                    dist = np.hypot(*(s.nodes[free] - np.array([px, py])).T)
                    if dist.min() < SNAP_START_M:
                        v = free[int(dist.argmin())]
                        w = s.given_adj[v][int(rng.integers(len(s.given_adj[v])))]
                        d = s.nodes[v] - s.nodes[w]
                        s.head_override[v] = float(np.arctan2(d[1], d[0]))
                        s.reached.add(v)
                        s.queue.append(v)
                        continue
                s.node_rule[s.add_node(np.array([px, py]))] = "start"
        active = [s for s in states if not s.done and s.queue]
        if not active:
            if all(s.done for s in states):
                break
            continue
        # ---- one pop per active tile
        us = [s.queue.popleft() for s in active]
        pos = np.stack([s.nodes[u] for s, u in zip(active, us)])
        rows, cols = pixel_rc(pos)
        turtle = getattr(model, "turtle", False)
        n_cls = getattr(model, "n_classes", 0)
        crops, arms, masks, scal, heads = [], [], [], [], []
        for s, u, r, c in zip(active, us, rows, cols):
            s.popped_map[r + PAD, c + PAD] = s.t
            sc = scalar_features(s.nodes[u], np.array([s.border[u]]), np.array([len(s.adj[u])]),
                                 np.array([s.n]), np.array([s.t]))[0]
            disp = [s.nodes[w] - s.nodes[u] for w in s.adj[u]]
            if turtle:
                hd, flag = (s.head_override[u], False) if u in s.head_override else heading_of(s.nodes, u, s.parent[u])
                crops.append(crops_turtle(s.edge_map, s.created_map, s.popped_map, s.nodes[u], hd, [s.t])[0])
                disp = [rotate(x, -hd) for x in disp]
                sc = np.r_[sc, np.float32(flag)].astype(np.float32)
            else:
                hd = 0.0
                crops.append(crops_from_maps(s.edge_map, s.created_map, s.popped_map, np.array([r]), np.array([c]),
                                             np.array([s.t]))[0])
            heads.append(hd)
            a, m = arm_features(disp)
            arms.append(a)
            masks.append(m)
            scal.append(sc)
        T = lambda x: torch.from_numpy(np.stack(x)).to(device)
        ctx = model.context(T(crops), T(scal), T(arms), T(masks), None if C is None else C[[sid[id(s)] for s in active]])
        prev = torch.zeros(len(active), 1, 2, device=device)
        prev_c = torch.full((len(active), 1), -1, device=device, dtype=torch.long)
        h = None
        alive = np.ones(len(active), bool)
        steps = [[] for _ in active]
        for k in range(MAX_T):
            if n_cls:
                stop_logit, (lp, mu, ls, rho), h, cl = model.decode_steps(ctx, prev, h, prev_c)
                c = torch.multinomial(torch.softmax(cl[:, 0], -1).cpu(), 1, generator=gen).squeeze(1)
            else:
                stop_logit, (lp, mu, ls, rho), h = model.decode_steps(ctx, prev, h)
                c = torch.full((len(active),), 2)
            stop = (torch.rand(len(active), generator=gen) < torch.sigmoid(stop_logit[:, 0]).cpu()).numpy()
            d = mixture_sample(lp[:, 0], mu[:, 0], ls[:, 0] + np.log(temperature), rho[:, 0], gen)
            alive &= ~stop
            if not alive.any():
                break
            dn = d.cpu().numpy()
            for i in np.flatnonzero(alive):
                steps[i].append((rotate(dn[i] * DISP_SCALE, heads[i]), int(c[i])))
                if step_log is not None:
                    s_ = active[i]
                    step_log.append((sid[id(s_)], s_.t, bool(s_.parent[us[i]] < 0), *(dn[i] * DISP_SCALE)))
            prev = d.unsqueeze(1)
            prev_c = c.view(-1, 1).to(device)
        for s, u, st in zip(active, us, steps):
            for d, c in st:
                s.apply_step(u, d, c)
            s.t += 1
    if states_out is not None:
        states_out.extend(states)
    return [s.graph(with_class=bool(getattr(model, "n_classes", 0))) for s in states]
