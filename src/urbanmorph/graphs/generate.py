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

Milestone 12: a ``given`` graph may carry vertical metadata (``edge_level`` / ``edge_struct``, graphs/levels.py).
Generated streets are at ground level, so a step that crosses a given bridge or tunnel passes under or over it
(``rules["grade_separated"]``) instead of creating a junction, and bridge / tunnel interiors are never merge or
T-junction targets. Graphs without metadata take exactly the Milestone 6 path.

These rules make every component connected by construction. They do not impose grids, angles,
lengths or a number of components: those come only from the learned distributions.
Rules 2 and 4 are not in the training data's step sequences (a train/generate mismatch,
documented in docs/MILESTONE_5.md).
"""

from __future__ import annotations

import gc
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


def clip_to_tile(a: np.ndarray, b: np.ndarray, size: float = TILE_M) -> tuple[np.ndarray, bool]:
    """Liang-Barsky: point where a->b leaves [0, size]^2 (a is inside or on the border)."""
    d = b - a
    t1 = 1.0
    for k in range(2):
        if d[k] > 0:
            t1 = min(t1, (size - a[k]) / d[k])
        elif d[k] < 0:
            t1 = min(t1, (0 - a[k]) / d[k])
    t1 = max(t1, 0.0)
    return a + t1 * d, t1 < 1.0


def raised_keys(raised) -> np.ndarray:
    return np.array([(int(a) << 32) | int(b) for a, b in raised], np.int64)


def raised_mask(e: np.ndarray, raised, keys: np.ndarray | None = None) -> np.ndarray:
    """Per edge (a, b): (min, max) in ``raised``; vectorised, the same result as testing each edge in Python (M20:
    that per-edge loop was 87 % of a whole-town run's time). ``keys``: raised_keys(raised), if already at hand."""
    lo, hi = e.min(1).astype(np.int64), e.max(1).astype(np.int64)
    return np.isin((lo << 32) | hi, raised_keys(raised) if keys is None else keys)


# M20 D4e: a spatial index over the edges and nodes of a GenState, kept beside (never instead of) ``edges`` / ``nodes``.
# Each entry (edge or node) has a box (xmin, ymin, xmax, ymax). Entries [0, m) form a static part, sorted along a Morton
# curve of their box centres and grouped in blocks of _IX_B with the union box of each block; entries [m, n) (the newest,
# at most _IX_TAIL) are tested one by one; the static part is rebuilt when the tail is full. A box query returns the
# indices of every entry whose own box meets the query box, in ascending (global insertion) order. It is a
# conservative pre-filter: every edge / node that the full scans of resolve_step could select is returned, so the
# selection (argmin ties included) and the float arithmetic on the survivors are those of the full scan. The index lives
# in GenState.__dict__ under _IX_KEYS, is copied by ``copy``, left out of pickles (rebuilt on first use) and rebuilt
# whenever its counts disagree with ``edges`` / ``n``.
_IX_B = 32
_IX_TAIL = 4096
_IX_CELL = 64.0
_IX_KEYS = ("_ix_ne", "_ix_ei", "_ix_eg", "_ix_lmax", "_ix_es", "_ix_nn", "_ix_ng", "_ix_ns", "_ix_rk")
# Crossing query margin (m) about the step's box. A computed "hit" needs |den| > 1e-9, so its rounding error moves the
# crossing point by at most ~eps |A - a| |r| |s| / 1e-9 along each segment: under 10 m while the step plus the longest
# edge is at most _IX_LMAX; beyond that the crossing test scans every live edge, as before.
_IX_PAD = 64.0
_IX_LMAX = 400.0
_IX_AR = np.arange(_IX_B)


def _ix_empty(cap: int) -> np.ndarray:
    g = np.empty((cap, 4))
    g[:, :2], g[:, 2:] = np.inf, -np.inf
    return g


def _ix_static(g: np.ndarray, n: int) -> tuple:
    """(m, order, block boxes): entries [0, n) sorted along a Morton curve of their box centres (empty boxes last)."""
    b = g[:n]
    ok = np.isfinite(b).all(1)
    c = np.zeros((n, 2), np.int64)
    c[ok] = np.clip(((b[ok, :2] + b[ok, 2:]) * (0.5 / _IX_CELL)).astype(np.int64) + 1024, 0, 65535)
    key = np.zeros(n, np.int64)
    for bit in range(16):
        key |= ((c[:, 0] >> bit) & 1) << (2 * bit) | ((c[:, 1] >> bit) & 1) << (2 * bit + 1)
    key[~ok] = np.iinfo(np.int64).max
    order = np.argsort(key, kind="stable")
    nb = (n + _IX_B - 1) // _IX_B
    pad = _ix_empty(nb * _IX_B)
    pad[:n] = b[order]
    pad = pad.reshape(nb, _IX_B, 4)
    blk = np.c_[pad[:, :, :2].min(1), pad[:, :, 2:].max(1)]
    return n, order, blk


def _ix_query(g: np.ndarray, st: tuple, n: int, x0, y0, x1, y1) -> np.ndarray:
    """Ascending indices k < n whose box g[k] meets [x0, x1] x [y0, y1] (closed); st = _ix_static of [0, m)."""
    m, order, blk = st
    out = []
    if m:
        hit = np.flatnonzero((blk[:, 2] >= x0) & (blk[:, 0] <= x1) & (blk[:, 3] >= y0) & (blk[:, 1] <= y1))
        if len(hit):
            pos = (hit[:, None] * _IX_B + _IX_AR[None, :]).ravel()
            ids = order[pos[pos < m]]
            b = g[ids]
            ids = ids[(b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1)]
            ids.sort()
            out.append(ids)
    b = g[m:n]
    out.append(m + np.flatnonzero((b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1)))
    return out[0] if len(out) == 1 else np.concatenate(out)


class GenState:
    def __init__(self, size: float = TILE_M, max_nodes: int = MAX_NODES):
        """``size``: side of the square domain (the 512 m tile; Milestone 13 global generation uses larger)."""
        self.size, self.max_nodes = size, max_nodes
        canvas = CANVAS if size == TILE_M else int(round(size / 4.0)) + 2 * PAD
        self.nodes = np.zeros((max_nodes + 8, 2))
        self.border = np.zeros(max_nodes + 8, bool)
        self.n = 0
        self.adj: list[set] = []
        self.edges: list[list[int]] = []   # [a, b, street class]; removed edges get a = -1
        self.parent: list[int] = []
        self.queue: deque = deque()
        self.edge_map = np.full((canvas, canvas), 1e9)   # drawn = 0 (same "< t" convention as training)
        self.created_map = np.full((canvas, canvas), 1e9)
        self.popped_map = np.full((canvas, canvas), 1e9)
        self.t = 0
        self.ncomp = 0
        self.done = False
        self.rules = {"clipped": 0, "crossing_split": 0, "crossing_endpoint": 0, "merge_node": 0, "t_junction": 0,
                      "new_node": 0, "dropped_short": 0, "grade_separated": 0}
        self.node_rule: dict = {}          # node -> "start" / "new_node" / "crossing_split" / "t_junction"
        self.merged_into: dict = {}        # node -> times a step was joined to it (rules 2 endpoint, 3)
        self.given_adj: dict = {}          # given node -> its given neighbours (Milestone 10)
        self.reached: set = set()          # given nodes already queued
        self.head_override: dict = {}      # start node on the given network -> heading
        self.raised: dict = {}             # (a, b) sorted -> (level, struct) of given non-ground edges (Milestone 12)
        self.raised_nodes: set = set()     # given nodes whose given edges are all non-ground (bridge / tunnel interior)

    def add_given(self, g: RoadGraph) -> None:
        """Pre-draw a graph (time -1; its nodes are never "open")."""
        ids = [self.add_node(p, push=False) for p in g.nodes]
        for i in ids:
            r, c = pixel_rc(self.nodes[i], self.size)
            self.created_map[r[0] + PAD, c[0] + PAD] = -1.0
            self.popped_map[r[0] + PAD, c[0] + PAD] = -1.0
            self.node_rule[i] = "given"
        meta = getattr(g, "edge_level", None) is not None
        ground: dict = {}
        for (a, b), cls, lv, st in zip(g.edges, g.classes(), g.levels(), g.structs()):
            self.connect(ids[a], ids[b], int(cls))
            self.given_adj.setdefault(ids[a], []).append(ids[b])
            self.given_adj.setdefault(ids[b], []).append(ids[a])
            raised = meta and (lv != 0 or st in (1, 2))               # levels.BRIDGE, levels.TUNNEL
            if raised:
                self.raised[(min(ids[a], ids[b]), max(ids[a], ids[b]))] = (int(lv), int(st))
            for v in (ids[a], ids[b]):
                ground[v] = ground.get(v, False) or not raised
        self.raised_nodes |= {v for v, gr in ground.items() if not gr}

    def __getstate__(self):
        d = self.__dict__
        return {k: v for k, v in d.items() if k not in _IX_KEYS} if any(k in d for k in _IX_KEYS) else d

    # -- the spatial index (M20 D4e; see _IX_B)
    def _ix_edges(self):
        """(edge rows [a, b, class] with a = -1 for removed edges, as ``edges``; their boxes; their count), in sync
        with ``edges``."""
        d = self.__dict__
        if d.get("_ix_ne") != len(self.edges):
            E = np.array(self.edges, int).reshape(-1, 3)
            n = len(E)
            cap = max(1024, 2 * n)
            ei = np.zeros((cap, 3), int)
            ei[:n] = E
            eg = _ix_empty(cap)
            live = E[:, 0] >= 0
            P, Q = self.nodes[E[live, 0]], self.nodes[E[live, 1]]
            eg[:n][live] = np.c_[np.minimum(P, Q), np.maximum(P, Q)]
            d["_ix_ei"], d["_ix_eg"] = ei, eg
            d["_ix_lmax"] = float(np.hypot(*(Q - P).T).max()) if len(P) else 0.0
            d["_ix_es"] = _ix_static(eg, n)
            d["_ix_ne"] = n
        return d["_ix_ei"], d["_ix_eg"], d["_ix_ne"]

    def _ix_add_edge(self, a, b, cls) -> None:
        """Index the edge just appended to ``edges`` (only when the index was in sync before it)."""
        d = self.__dict__
        k = d.get("_ix_ne")
        if k is None or k != len(self.edges) - 1:
            return
        ei, eg = d["_ix_ei"], d["_ix_eg"]
        if k >= len(ei):
            d.pop("_ix_ne")                               # grow by a rebuild
            return
        P, Q = self.nodes[a], self.nodes[b]
        ei[k] = (a, b, cls)
        eg[k] = (min(P[0], Q[0]), min(P[1], Q[1]), max(P[0], Q[0]), max(P[1], Q[1]))
        d["_ix_lmax"] = max(d["_ix_lmax"], float(np.hypot(*(Q - P))))
        d["_ix_ne"] = k + 1

    def _ix_live_in(self, x0, y0, x1, y1) -> np.ndarray:
        """Ascending indices (into ``edges``) of the live edges whose box meets the closed box."""
        ei, eg, n = self._ix_edges()
        d = self.__dict__
        if n - d["_ix_es"][0] > _IX_TAIL:
            d["_ix_es"] = _ix_static(eg, n)
        idx = _ix_query(eg, d["_ix_es"], n, x0, y0, x1, y1)
        return idx[ei[idx, 0] >= 0]

    def _ix_nodes_in(self, x0, y0, x1, y1) -> np.ndarray:
        """Ascending ids of the nodes inside the closed box."""
        d = self.__dict__
        n = self.n
        if d.get("_ix_nn") != n:
            cap = max(1024, 2 * n)
            ng = _ix_empty(cap)
            ng[:n, :2] = ng[:n, 2:] = self.nodes[:n]
            d["_ix_ng"], d["_ix_ns"], d["_ix_nn"] = ng, _ix_static(ng, n), n
        elif n - d["_ix_ns"][0] > _IX_TAIL:
            d["_ix_ns"] = _ix_static(d["_ix_ng"], n)
        return _ix_query(d["_ix_ng"], d["_ix_ns"], n, x0, y0, x1, y1)

    def _raised_keys(self) -> np.ndarray:
        """raised_keys(self.raised), cached while ``raised`` is the same dict with the same size (entries are only ever
        added, never removed, so the size fixes the key set)."""
        c = self.__dict__.get("_ix_rk")
        if c is None or c[0] is not self.raised or c[1] != len(self.raised):
            c = (self.raised, len(self.raised), raised_keys(self.raised))
            self.__dict__["_ix_rk"] = c
        return c[2]

    def _ix_add_node(self, i) -> None:
        d = self.__dict__
        if d.get("_ix_nn") != i:
            return
        ng = d["_ix_ng"]
        if i >= len(ng):
            d.pop("_ix_nn")
            return
        ng[i, :2] = ng[i, 2:] = self.nodes[i]
        d["_ix_nn"] = i + 1

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
        self.border[i] = bool(is_border(p, size=self.size)[0])
        self.adj.append(set())
        self.n += 1
        self._ix_add_node(i)
        r, c = pixel_rc(p, self.size)
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
        self._ix_add_edge(a, b, cls)
        r, c = segment_pixels(self.nodes[a], self.nodes[b], self.size)
        self.edge_map[r, c] = -1.0

    def live_edges(self, with_class=False):
        ei, _, n = self._ix_edges()                   # the same rows as a scan of ``edges`` (M20 D4e)
        e = ei[:n][ei[:n, 0] >= 0]
        return e if with_class else e[:, :2]

    def split_edge(self, a, b, p, parent=-1) -> int:
        cls = 2
        P, Q = self.nodes[a], self.nodes[b]           # the first live edge {a, b}, found through the index
        ei = self._ix_edges()[0]
        k = self._ix_live_in(min(P[0], Q[0]), min(P[1], Q[1]), max(P[0], Q[0]), max(P[1], Q[1]))
        k = k[((ei[k, 0] == a) & (ei[k, 1] == b)) | ((ei[k, 0] == b) & (ei[k, 1] == a))]
        if len(k):
            x = self.edges[int(k[0])]
            x[0] = -1
            ei[k[0], 0] = -1
            cls = x[2]
        self.adj[a].discard(b)
        self.adj[b].discard(a)
        s = self.add_node(p, parent=parent)
        for v in (a, b):
            self.adj[s].add(v)
            self.adj[v].add(s)
            self.edges.append([s, v, cls])        # both halves keep the street's class
            self._ix_add_edge(s, v, cls)
        return s

    def apply_step(self, u: int, d: np.ndarray, cls: int = 2) -> None:
        self.commit(u, self.resolve_step(u, d), cls)

    def resolve_step(self, u: int, d: np.ndarray) -> dict:
        """The edge that a ground step u -> u + d WOULD create, by the five rules of the module docstring, without
        changing the state (Milestone 13: candidates are scored on exactly the geometry that ``commit`` inserts).
        Returns {"kind", "q" (committed end point), "clipped", "grade_separated", and "edge" / "v" as needed}."""
        a = self.nodes[u]
        q, clipped = clip_to_tile(a, a + d, self.size)
        out = {"kind": "dropped", "q": q, "clipped": bool(clipped), "grade_separated": False}
        if np.hypot(*(q - a)) < 1.0:
            return out
        # M20 D4e: each test runs on the live edges / nodes that the spatial index returns for its box, in edge order;
        # every edge / node the full scan could select is among them, so the outcome and its floats are unchanged.
        ei = self._ix_edges()[0]
        lo, hi = np.minimum(a, q), np.maximum(a, q)
        if np.hypot(*(q - a)) + self.__dict__["_ix_lmax"] > _IX_LMAX:
            C = self._ix_live_in(-np.inf, -np.inf, np.inf, np.inf)
        else:
            C = self._ix_live_in(lo[0] - _IX_PAD, lo[1] - _IX_PAD, hi[0] + _IX_PAD, hi[1] + _IX_PAD)
        if len(C):
            e = ei[C, :2]
            up = raised_mask(e, self.raised, self._raised_keys()) if self.raised else np.zeros(len(e), bool)   # bridges / tunnels: never joined
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
            if up.any():
                out["grade_separated"] = bool((hit & up).any())
                hit &= ~up                                            # pass under / over: no junction
            if hit.any() and self.n >= len(self.nodes):          # node array full (M16 guard; earlier runs never
                return out                                    # reached it: a split here raised IndexError)
            if hit.any():
                k = int(np.flatnonzero(hit)[np.argmin(tt[hit])])
                x = a + tt[k] * r
                ea, eb = e[k]
                ends = [v for v in (ea, eb) if np.hypot(*(self.nodes[v] - x)) < MERGE_M]
                if ends:
                    v = min(ends, key=lambda v: np.hypot(*(self.nodes[v] - x)))
                    return {**out, "kind": "crossing_endpoint", "v": int(v), "q": self.nodes[v].copy()}
                return {**out, "kind": "crossing_split", "edge": (int(ea), int(eb)), "q": x}
        # 3. nearest existing node
        if self.n > 1:
            m = MERGE_M + 1.0
            idx = self._ix_nodes_in(q[0] - m, q[1] - m, q[0] + m, q[1] + m)
            if len(idx):
                dist = np.hypot(*(self.nodes[idx] - q).T)
                dist[idx == u] = np.inf
                if self.raised_nodes:
                    dist[np.isin(idx, list(self.raised_nodes))] = np.inf
                j = int(dist.argmin())
                if dist[j] < MERGE_M:
                    v = int(idx[j])
                    return {**out, "kind": "merge_node", "v": v, "q": self.nodes[v].copy()}
        # 4. nearest existing street (T-junction)
        m = MERGE_M + 1.0
        T = self._ix_live_in(q[0] - m, q[1] - m, q[0] + m, q[1] + m)
        if len(T):
            e = ei[T, :2]
            up = raised_mask(e, self.raised, self._raised_keys()) if self.raised else np.zeros(len(e), bool)
            keep = ~((e[:, 0] == u) | (e[:, 1] == u)) & ~up
            if keep.any():
                A2, B2 = self.nodes[e[keep, 0]], self.nodes[e[keep, 1]]
                s2 = B2 - A2
                lam = np.clip(((q - A2) * s2).sum(1) / np.maximum((s2 ** 2).sum(1), 1e-9), 0, 1)
                proj = A2 + lam[:, None] * s2
                dd = np.hypot(*(proj - q).T)
                k = int(dd.argmin())
                if dd[k] < MERGE_M and 0 < lam[k] < 1:
                    if self.n >= len(self.nodes):                 # node array full (M16 guard, as above)
                        return out
                    ea, eb = e[keep][k]
                    return {**out, "kind": "t_junction", "edge": (int(ea), int(eb)), "q": proj[k]}
        # 5. new node
        if np.hypot(*(q - a)) < MERGE_M or self.n >= self.max_nodes:
            return out
        return {**out, "kind": "new_node"}

    def commit(self, u: int, o: dict, cls: int = 2) -> int:
        """Insert the outcome of ``resolve_step``; returns the node the new edge ends at (-1 if dropped)."""
        self.rules["clipped"] += int(o["clipped"])
        self.rules["grade_separated"] += int(o["grade_separated"])
        k = o["kind"]
        if k == "dropped":
            self.rules["dropped_short"] += 1
            return -1
        if k in ("crossing_endpoint", "merge_node"):
            v = o["v"]
            self.rules[k] += 1
            self.merged_into[v] = self.merged_into.get(v, 0) + 1
            self.reach(v, u)
        elif k in ("crossing_split", "t_junction"):
            v = self.split_edge(*o["edge"], o["q"], u)
            self.rules[k] += 1
            self.node_rule[v] = k
        else:
            v = self.add_node(o["q"], parent=u)
            self.rules["new_node"] += 1
            self.node_rule[v] = "new_node"
        self.connect(u, v, cls)
        return v

    def copy(self) -> "GenState":
        """An independent copy (particles, look-ahead roll-outs). The cyclic garbage collector is paused while the
        ~10^5 per-edge lists and per-node sets are built (M20 D4e: its passes, triggered by these allocations, were most
        of the time of a copy late in a whole-town run); the copy itself is unchanged."""
        c = GenState.__new__(GenState)
        paused = gc.isenabled()
        if paused:
            gc.disable()
        try:
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    c.__dict__[k] = v.copy()
                elif isinstance(v, deque):
                    c.__dict__[k] = deque(v)
                elif k == "adj":
                    c.__dict__[k] = list(map(set, v))
                elif k == "edges":
                    c.__dict__[k] = list(map(list, v))
                elif k in ("given_adj",):
                    c.__dict__[k] = {a: list(b) for a, b in v.items()}
                elif isinstance(v, (dict, set, list)):
                    c.__dict__[k] = type(v)(v) if not isinstance(v, dict) else dict(v)
                else:
                    c.__dict__[k] = v
        finally:
            if paused:
                gc.enable()
        return c

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
        lv = st = None
        if self.raised:                                               # keep the given vertical metadata
            m = [self.raised.get((int(a), int(b)), (0, 0)) for a, b in e]
            lv, st = np.array([x[0] for x in m], int), np.array([x[1] for x in m], int)
        return RoadGraph(nodes, remap[e] if len(e) else e.reshape(0, 2),
                         is_border(nodes, size=self.size) if len(nodes) else np.zeros(0, bool),
                         cls.copy() if with_class else None, lv, st)


@torch.no_grad()
def sample_graphs(model, n: int, device="cpu", seed: int = 0, temperature: float = 1.0,
                  max_rounds: int = 1500, states_out: list | None = None, step_log: list | None = None,
                  cond: np.ndarray | None = None, given: list | None = None,
                  frontier: list | None = None, terrain=None) -> list[RoadGraph]:
    """Grow n street graphs. temperature scales the step-mixture spread (1.0 = the learned model).
    ``states_out`` receives the GenStates (rule counts); ``step_log`` receives one tuple per sampled step
    (tile, pop round t, start flag, dx, dy in the turtle frame in metres) - diagnostics only.
    ``frontier`` (Milestone 11, needs ``given``): per tile, indices of given nodes of degree 1 that are queued from the
    start, heading along their given edge into them (the turtle arrives along a street cut by an earlier window and
    continues it). Default None changes nothing.
    ``terrain`` (Milestone 12, graphs/terrain_generate.TerrainPolicy): supplies terrain crops and start canvases to a
    TerrainGraphTurtle, may reject start points, and applies every step (feasibility, replanning, structure). With
    None the loop is exactly the Milestone 6/10 one."""
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
    if frontier is not None:
        for s, fr in zip(states, frontier):
            for v in fr:                       # add_given numbers the given nodes 0..n-1 in order
                w = s.given_adj[v][0]
                d = s.nodes[v] - s.nodes[w]
                s.head_override[v] = float(np.arctan2(d[1], d[0]))
                s.reached.add(v)
                s.queue.append(v)
    for _ in range(max_rounds):
        # ---- start a new component where the queue is empty
        need = [s for s in states if not s.done and not s.queue]
        if need:
            canvas = torch.from_numpy(np.stack([(s.edge_map[PAD:PAD + GRID, PAD:PAD + GRID] < s.t + 0.5)[None]
                                                for s in need])).to(device)
            ncomp = torch.tensor([float(s.ncomp) for s in need], device=device)
            if terrain is None:
                stop_logit, loc = model.start_outputs(canvas, ncomp, None if C is None else C[[sid[id(s)] for s in need]])
            else:
                tcan = torch.from_numpy(np.stack([terrain.start_extra(sid[id(s)]) for s in need])).to(device)
                stop_logit, loc = model.start_outputs(canvas, ncomp, tcanvas=tcan)
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
                if terrain is not None and not terrain.start_ok(sid[id(s)], np.array([px, py])):
                    s.ncomp += 1                           # a start in water / no data is spent, not placed
                    continue
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
        crops, arms, masks, scal, heads, tcrops = [], [], [], [], [], []
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
            if terrain is not None:
                tcrops.append(terrain.crop_extra(sid[id(s)], s.nodes[u], hd))
            a, m = arm_features(disp)
            arms.append(a)
            masks.append(m)
            scal.append(sc)
        T = lambda x: torch.from_numpy(np.stack(x)).to(device)
        if terrain is None:
            ctx = model.context(T(crops), T(scal), T(arms), T(masks), None if C is None else C[[sid[id(s)] for s in active]])
        else:
            ctx = model.context(T(crops), T(scal), T(arms), T(masks), tcrops=T(tcrops))
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
            if terrain is not None:                        # learned structure and the step distribution, for replanning
                sl = getattr(model, "last_struct_logit", None)
                spred = (torch.multinomial(torch.softmax(sl[:, 0], -1).cpu(), 1, generator=gen).squeeze(1).numpy()
                         if sl is not None else np.zeros(len(active), int))
                mix_k = (lp[:, 0].cpu(), mu[:, 0].cpu(), (ls[:, 0] + np.log(temperature)).cpu(), rho[:, 0].cpu())
            alive &= ~stop
            if not alive.any():
                break
            dn = d.cpu().numpy()
            for i in np.flatnonzero(alive):
                if terrain is not None:
                    steps[i].append((rotate(dn[i] * DISP_SCALE, heads[i]), int(c[i]), int(spred[i]),
                                     tuple(x[i] for x in mix_k), heads[i]))
                    continue
                steps[i].append((rotate(dn[i] * DISP_SCALE, heads[i]), int(c[i])))
                if step_log is not None:
                    s_ = active[i]
                    step_log.append((sid[id(s_)], s_.t, bool(s_.parent[us[i]] < 0), *(dn[i] * DISP_SCALE)))
            prev = d.unsqueeze(1)
            prev_c = c.view(-1, 1).to(device)
        for s, u, st in zip(active, us, steps):
            for x in st:
                if terrain is None:
                    s.apply_step(u, x[0], x[1])
                else:
                    terrain.apply(sid[id(s)], s, u, *x, gen)
            s.t += 1
    if states_out is not None:
        states_out.extend(states)
    return [s.graph(with_class=bool(getattr(model, "n_classes", 0))) for s in states]
