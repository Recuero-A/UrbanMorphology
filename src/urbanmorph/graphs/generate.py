"""Sampling street graphs from the trained GraphTurtle (Milestone 5).

Many tiles grow in lock-step (one pop per tile per round) so the network runs on batches.
Each sampled step from node u to p = u + d is turned into geometry by fixed rules, in order:

1. clip u->p to the tile (a street leaving the window ends on the border, like the clipped data);
2. if u->p CROSSES an existing street, stop at the crossing and create a junction there
   (in the OSM graph streets that cross at grade share a node);
3. else if p is within MERGE_M of an existing node, join it (loop closing, NTG's merge);
4. else if p is within MERGE_M of an existing street, join it with a T-junction;
5. else if the step is shorter than MERGE_M, drop it; otherwise create a new node and queue it.

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
                                        crops_from_maps, pixel_rc, scalar_features, segment_pixels)

MAX_NODES = 1000
MAX_COMPONENTS = 40


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
        self.edges: list[list[int]] = []   # [a, b]; removed edges get a = -1
        self.queue: deque = deque()
        self.edge_map = np.full((CANVAS, CANVAS), 1e9)   # drawn = 0 (same "< t" convention as training)
        self.created_map = np.full((CANVAS, CANVAS), 1e9)
        self.popped_map = np.full((CANVAS, CANVAS), 1e9)
        self.t = 0
        self.ncomp = 0
        self.done = False

    # -- geometry helpers
    def add_node(self, p, push=True) -> int:
        i = self.n
        self.nodes[i] = p
        self.border[i] = bool(is_border(p)[0])
        self.adj.append(set())
        self.n += 1
        r, c = pixel_rc(p)
        self.created_map[r[0] + PAD, c[0] + PAD] = self.t - 0.5
        if push:
            self.queue.append(i)
        return i

    def connect(self, a, b):
        if a == b or b in self.adj[a]:
            return
        self.adj[a].add(b)
        self.adj[b].add(a)
        self.edges.append([a, b])
        r, c = segment_pixels(self.nodes[a], self.nodes[b])
        self.edge_map[r, c] = -1.0

    def live_edges(self):
        e = np.array([x for x in self.edges if x[0] >= 0], int).reshape(-1, 2)
        return e

    def split_edge(self, a, b, p) -> int:
        for x in self.edges:
            if x[0] >= 0 and {x[0], x[1]} == {a, b}:
                x[0] = -1
                break
        self.adj[a].discard(b)
        self.adj[b].discard(a)
        s = self.add_node(p)
        for v in (a, b):
            self.adj[s].add(v)
            self.adj[v].add(s)
            self.edges.append([s, v])
        return s

    def apply_step(self, u: int, d: np.ndarray) -> None:
        a = self.nodes[u]
        q, _ = clip_to_tile(a, a + d)
        if np.hypot(*(q - a)) < 1.0:
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
                v = min(ends, key=lambda v: np.hypot(*(self.nodes[v] - x))) if ends else self.split_edge(ea, eb, x)
                self.connect(u, v)
                return
        # 3. nearest existing node
        if self.n > 1:
            dist = np.hypot(*(self.nodes[:self.n] - q).T)
            dist[u] = np.inf
            v = int(dist.argmin())
            if dist[v] < MERGE_M:
                self.connect(u, v)
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
                    self.connect(u, self.split_edge(ea, eb, proj[k]))
                    return
        # 5. new node
        if np.hypot(*(q - a)) < MERGE_M or self.n >= MAX_NODES:
            return
        self.connect(u, self.add_node(q))

    def graph(self) -> RoadGraph:
        e = self.live_edges()
        e = np.sort(e, axis=1) if len(e) else e
        e = np.unique(e, axis=0) if len(e) else np.zeros((0, 2), int)
        used = np.unique(e.ravel()) if len(e) else np.zeros(0, int)
        remap = -np.ones(self.n, int)
        remap[used] = np.arange(len(used))
        nodes = self.nodes[used]
        return RoadGraph(nodes, remap[e] if len(e) else e, is_border(nodes) if len(nodes) else np.zeros(0, bool))


@torch.no_grad()
def sample_graphs(model, n: int, device="cpu", seed: int = 0, temperature: float = 1.0,
                  max_rounds: int = 1500) -> list[RoadGraph]:
    """Grow n street graphs. temperature scales the step-mixture spread (1.0 = the learned model)."""
    from urbanmorph.models.graph_gen import mixture_sample
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = [GenState() for _ in range(n)]
    for _ in range(max_rounds):
        # ---- start a new component where the queue is empty
        need = [s for s in states if not s.done and not s.queue]
        if need:
            canvas = torch.from_numpy(np.stack([(s.edge_map[PAD:PAD + GRID, PAD:PAD + GRID] < s.t + 0.5)[None]
                                                for s in need])).to(device)
            ncomp = torch.tensor([float(s.ncomp) for s in need], device=device)
            stop_logit, loc = model.start_outputs(canvas, ncomp)
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
                s.add_node(np.array([px, py]))
                s.ncomp += 1
        active = [s for s in states if not s.done and s.queue]
        if not active:
            if all(s.done for s in states):
                break
            continue
        # ---- one pop per active tile
        us = [s.queue.popleft() for s in active]
        pos = np.stack([s.nodes[u] for s, u in zip(active, us)])
        rows, cols = pixel_rc(pos)
        crops, arms, masks, scal = [], [], [], []
        for s, u, r, c in zip(active, us, rows, cols):
            s.popped_map[r + PAD, c + PAD] = s.t
            crops.append(crops_from_maps(s.edge_map, s.created_map, s.popped_map, np.array([r]), np.array([c]),
                                         np.array([s.t]))[0])
            a, m = arm_features([s.nodes[w] - s.nodes[u] for w in s.adj[u]])
            arms.append(a)
            masks.append(m)
            scal.append(scalar_features(s.nodes[u], np.array([s.border[u]]), np.array([len(s.adj[u])]),
                                        np.array([s.n]), np.array([s.t]))[0])
        T = lambda x: torch.from_numpy(np.stack(x)).to(device)
        ctx = model.context(T(crops), T(scal), T(arms), T(masks))
        prev = torch.zeros(len(active), 1, 2, device=device)
        h = None
        alive = np.ones(len(active), bool)
        steps = [[] for _ in active]
        for k in range(MAX_T):
            stop_logit, (lp, mu, ls, rho), h = model.decode_steps(ctx, prev, h)
            stop = (torch.rand(len(active), generator=gen) < torch.sigmoid(stop_logit[:, 0]).cpu()).numpy()
            d = mixture_sample(lp[:, 0], mu[:, 0], ls[:, 0] + np.log(temperature), rho[:, 0], gen)
            alive &= ~stop
            if not alive.any():
                break
            dn = d.cpu().numpy()
            for i in np.flatnonzero(alive):
                steps[i].append(dn[i] * DISP_SCALE)
            prev = d.unsqueeze(1)
        for s, u, st in zip(active, us, steps):
            for d in st:
                s.apply_step(u, d)
            s.t += 1
    return [s.graph() for s in states]
