"""Turn a street graph into the step-by-step construction sequence the generator learns (Milestone 5).

Construction order (breadth-first, as in NTG's node queue):

    for each connected component (largest first):
        a START node is placed (a random node of the component: augmentation)
        queue = [start]
        while queue: pop u; emit, in counter-clockwise order from east, the displacement
                     to every neighbour w whose edge (u, w) is not drawn yet; then STOP.
                     Neighbours not created yet are created and appended to the queue.
    finally: STOP (no more components)

A neighbour that already exists (a loop closing) is emitted exactly like a new one: at
generation time a step that lands within MERGE_M of an existing node joins it.

What the model sees when popping u at step t (identical in training and generation):
  * a 48 x 48 crop (4 m pixels = 192 m) centred on u with 3 channels:
    roads drawn before t, "open" nodes (created, not yet expanded), outside-the-tile;
  * the "arms" of u: displacements of its edges drawn before t (NTG encodes incoming paths);
  * scalars: position in the tile, sub-pixel offset, border flag, counts.
Training renders the whole construction ONCE per trace with time stamps
(pixel = time it was first drawn), so the canvas at any step is a threshold: map < t.

Turtle frame (Milestone 6, ``turtle=True``): everything is expressed relative to the HEADING of
the street that reached u (the direction parent -> u; east for a component's start node, flagged).
The crop is sampled on a grid rotated so the heading points up; arms and steps are rotated by
-heading; steps are ordered by their turn angle. A step "straight on" is then the same number
whatever the street's compass direction, so junction angles are learned once, not per heading
(rotation equivariance by construction; no angle is preferred).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph, components

PX = 4.0
GRID = int(TILE_M / PX)       # 128
CROP = 48
HALF = CROP // 2
PAD = HALF
CANVAS = GRID + 2 * PAD       # 176
CELL = 4                      # start-location cells of 4 x 4 pixels = 16 m -> 32 x 32 cells
DISP_SCALE = 30.0             # displacements are modelled in units of 30 m
MAX_T = 7                     # at most 7 steps per pop (+ STOP)
MAX_ARMS = 8
N_SCALARS = 8
N_SCALARS_TURTLE = 9
N_CLASSES = 4
INF = 1e9


def pixel_rc(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of the 4 m pixel containing each point (row 0 = north), clipped to the tile."""
    p = np.atleast_2d(p)
    col = np.clip(np.floor(p[:, 0] / PX), 0, GRID - 1).astype(int)
    row = np.clip(np.floor((TILE_M - p[:, 1]) / PX), 0, GRID - 1).astype(int)
    return row, col


def segment_pixels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Padded-canvas pixels touched by the segment a-b (sampled every metre)."""
    n = max(2, int(np.ceil(np.hypot(*(b - a)))) + 1)
    pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
    r, c = pixel_rc(pts)
    return r + PAD, c + PAD


def outside_mask() -> np.ndarray:
    m = np.ones((CANVAS, CANVAS), bool)
    m[PAD:PAD + GRID, PAD:PAD + GRID] = False
    return m


OUTSIDE = outside_mask()


def scalar_features(p: np.ndarray, border: np.ndarray, n_arms: np.ndarray, n_created: np.ndarray,
                    t: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(p)
    row, col = pixel_rc(p)
    sub_x = p[:, 0] / PX - col - 0.5
    sub_y = (TILE_M - p[:, 1]) / PX - row - 0.5
    return np.c_[p / TILE_M, sub_x, sub_y, border, n_arms / 4, n_created / 400, t / 400].astype(np.float32)


def arm_features(disps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(MAX_ARMS, 3) = dx, dy (/30 m), length; mask (MAX_ARMS,)."""
    a = np.zeros((MAX_ARMS, 3), np.float32)
    m = np.zeros(MAX_ARMS, bool)
    for k, d in enumerate(disps[:MAX_ARMS]):
        d = np.asarray(d) / DISP_SCALE
        a[k] = (d[0], d[1], np.hypot(*d))
        m[k] = True
    return a, m


@dataclass
class Trace:
    g: RoadGraph
    pop_node: np.ndarray        # (P,)
    targets: list               # P lists of displacements (metres)
    arms: list                  # P lists of displacements of already-drawn edges
    n_created: np.ndarray       # (P,) nodes existing when popping
    target_class: list          # P lists of street classes of the emitted edges
    parent: np.ndarray          # (N,) node whose pop created this node (-1 for component starts)
    start_t: np.ndarray         # (C,) pop index at which component c starts
    start_node: np.ndarray      # (C,)
    end_t: int
    edge_map: np.ndarray        # (CANVAS, CANVAS) time each pixel is first drawn
    created_map: np.ndarray     # node pixel -> creation time
    popped_map: np.ndarray      # node pixel -> pop time


def make_trace(g: RoadGraph, rng: np.random.Generator, turtle: bool = False) -> Trace:
    """BFS construction sequence. With ``turtle`` the steps of a pop are ordered by turn angle relative
    to the incoming street (the order the turtle-frame model emits them), else counter-clockwise from east."""
    adj = g.adjacency()
    lab = components(g)
    sizes = np.bincount(lab) if g.n else np.zeros(0, int)
    order = np.argsort(-sizes, kind="stable")
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(g.edges)}
    edge_t = np.full(len(g.edges), INF)
    created = np.full(g.n, INF)
    popped = np.full(g.n, INF)
    parent = -np.ones(g.n, int)
    ecls = g.classes()
    pop_node, targets, arms, n_created, start_t, start_node, tcls = [], [], [], [], [], [], []
    t = 0
    for c in order:
        s = int(rng.choice(np.flatnonzero(lab == c)))
        start_t.append(t)
        start_node.append(s)
        created[s] = t - 0.5
        q = deque([s])
        while q:
            u = q.popleft()
            popped[u] = t
            here = g.nodes[u]
            drawn, todo = [], []
            for w in adj[u]:
                e = eid[(min(u, w), max(u, w))]
                (drawn if edge_t[e] < t else todo).append((w, e))
            if turtle:
                hd = heading_of(g.nodes, u, parent[u])[0]
                todo.sort(key=lambda we: np.arctan2(*rotate(g.nodes[we[0]] - here, -hd)[::-1]))
            else:
                todo.sort(key=lambda we: np.arctan2(*(g.nodes[we[0]] - here)[::-1]) % (2 * np.pi))
            pop_node.append(u)
            arms.append([g.nodes[w] - here for w, _ in drawn])
            targets.append([g.nodes[w] - here for w, _ in todo])
            tcls.append([int(ecls[e]) for _, e in todo])
            n_created.append(int((created < t).sum()))
            for w, e in todo:
                edge_t[e] = t
                if created[w] == INF:
                    created[w] = t
                    parent[w] = u
                    q.append(w)
            t += 1
    edge_map = np.full((CANVAS, CANVAS), INF)
    for (a, b), te in zip(g.edges, edge_t):
        r, cc = segment_pixels(g.nodes[a], g.nodes[b])
        np.minimum.at(edge_map, (r, cc), te)
    created_map = np.full((CANVAS, CANVAS), INF)
    popped_map = np.full((CANVAS, CANVAS), INF)
    if g.n:
        r, cc = pixel_rc(g.nodes)
        created_map[r + PAD, cc + PAD] = created
        popped_map[r + PAD, cc + PAD] = popped
    return Trace(g, np.array(pop_node, int), targets, arms, np.array(n_created), tcls, parent, np.array(start_t),
                 np.array(start_node, int), t, edge_map, created_map, popped_map)


def crops_from_maps(edge_map, created_map, popped_map, rows, cols, t) -> np.ndarray:
    """(P, 3, CROP, CROP) bool crops centred on (rows, cols) of the tile, at times t (P,)."""
    rr = rows[:, None] + np.arange(-HALF, HALF)[None] + PAD
    cc = cols[:, None] + np.arange(-HALF, HALF)[None] + PAD
    R, C = rr[:, :, None], cc[:, None, :]
    tt = np.asarray(t, float)[:, None, None]
    roads = edge_map[R, C] < tt
    open_ = (created_map[R, C] < tt) & (tt < popped_map[R, C])
    out = np.broadcast_to(OUTSIDE[R, C], roads.shape)
    return np.stack([roads, open_, out], axis=1)


def rotate(v, h):
    """Rotate vectors (..., 2) by angle h (radians, counter-clockwise)."""
    v = np.asarray(v, float)
    c, s = np.cos(h), np.sin(h)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def heading_of(nodes, u, parent) -> tuple[float, bool]:
    """Heading (radians, CCW from east) of the street that reached u; (0, True) for a start node."""
    if parent < 0:
        return 0.0, True
    d = nodes[u] - nodes[parent]
    return float(np.arctan2(d[1], d[0])), False


_LOCAL = (np.arange(CROP) - HALF + 0.5)


def crops_turtle(edge_map, created_map, popped_map, pos, heading, t) -> np.ndarray:
    """(P, 3, CROP, CROP) crops on a grid rotated so that ``heading`` points up (row 0 = ahead)."""
    pos, heading = np.atleast_2d(pos), np.atleast_1d(heading)
    fwd = -_LOCAL[None, :, None]                      # rows: ahead is up
    right = _LOCAL[None, None, :]
    c, s = np.cos(heading)[:, None, None], np.sin(heading)[:, None, None]
    dx = fwd * c + right * s                          # pixels, east
    dy = fwd * s - right * c                          # pixels, north
    col = np.floor(pos[:, 0, None, None] / PX + dx).astype(int) + PAD
    row = np.floor((TILE_M - pos[:, 1, None, None]) / PX - dy).astype(int) + PAD
    inside = (col >= 0) & (col < CANVAS) & (row >= 0) & (row < CANVAS)
    col, row = np.clip(col, 0, CANVAS - 1), np.clip(row, 0, CANVAS - 1)
    tt = np.asarray(t, float)[:, None, None]
    roads = (edge_map[row, col] < tt) & inside
    open_ = (created_map[row, col] < tt) & (tt < popped_map[row, col]) & inside
    out = OUTSIDE[row, col] | ~inside
    return np.stack([roads, open_, out], axis=1)


def pop_samples(tr: Trace, turtle: bool = False) -> dict:
    """All teacher-forcing samples of one trace (one per pop)."""
    P = len(tr.pop_node)
    t = np.arange(P)
    pos = tr.g.nodes[tr.pop_node] if P else np.zeros((0, 2))
    n_arms = np.array([len(a) for a in tr.arms])
    scal = scalar_features(pos, tr.g.border[tr.pop_node], n_arms, tr.n_created, t)
    head = np.zeros(P)
    if turtle:
        hs = [heading_of(tr.g.nodes, u, tr.parent[u]) for u in tr.pop_node]
        head = np.array([h for h, _ in hs])
        scal = np.c_[scal, np.array([f for _, f in hs], np.float32)].astype(np.float32)
        crops = crops_turtle(tr.edge_map, tr.created_map, tr.popped_map, pos, head, t)
    else:
        rows, cols = pixel_rc(pos) if P else (np.zeros(0, int), np.zeros(0, int))
        crops = crops_from_maps(tr.edge_map, tr.created_map, tr.popped_map, rows, cols, t)
    arms = np.zeros((P, MAX_ARMS, 3), np.float32)
    arm_mask = np.zeros((P, MAX_ARMS), bool)
    tgt = np.zeros((P, MAX_T, 2), np.float32)
    tcls = np.zeros((P, MAX_T), np.int64)
    n_tgt = np.zeros(P, np.int64)
    for i in range(P):
        a, d, c = tr.arms[i], tr.targets[i], tr.target_class[i]
        if turtle:
            a = [rotate(x, -head[i]) for x in a]
            d = [rotate(x, -head[i]) for x in d]
            order = np.argsort([np.arctan2(x[1], x[0]) for x in d], kind="stable")   # turn angle, right to left
            d, c = [d[j] for j in order], [c[j] for j in order]
        arms[i], arm_mask[i] = arm_features(a)
        k = min(len(d), MAX_T)
        n_tgt[i] = k
        if k:
            tgt[i, :k] = np.asarray(d[:k]) / DISP_SCALE
            tcls[i, :k] = c[:k]
    out = {"crops": crops, "scalars": scal, "arms": arms, "arm_mask": arm_mask, "targets": tgt, "n_targets": n_tgt}
    if turtle:
        out["target_class"] = tcls
    return out


def global_canvas(edge_map: np.ndarray, t: float) -> np.ndarray:
    return edge_map[PAD:PAD + GRID, PAD:PAD + GRID] < t


def start_cell(p: np.ndarray) -> int:
    r, c = pixel_rc(p)
    return int((r[0] // CELL) * (GRID // CELL) + c[0] // CELL)


def start_samples(tr: Trace) -> dict:
    """One sample per component start (continue + location) and a final STOP."""
    canv, ncomp, stop, cell = [], [], [], []
    for k, (t, s) in enumerate(zip(tr.start_t, tr.start_node)):
        canv.append(global_canvas(tr.edge_map, t))
        ncomp.append(k)
        stop.append(0.0)
        cell.append(start_cell(tr.g.nodes[s]))
    canv.append(global_canvas(tr.edge_map, tr.end_t))
    ncomp.append(len(tr.start_t))
    stop.append(1.0)
    cell.append(0)
    return {"canvas": np.stack(canv)[:, None], "ncomp": np.array(ncomp, np.float32),
            "stop": np.array(stop, np.float32), "cell": np.array(cell, np.int64)}
