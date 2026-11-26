"""Milestone 15 analytic fixtures: synthetic terrains for the intervention experiments and hand-built street graphs
for metric checks.

Terrains use the sampler's terrain-dict format: a 1024 m x 1024 m window on a 4 m grid (256 x 256) whose central
512 m is the tile; x east, y north, tile-local metres; ``water`` 0 dry, 1 river; ``valid`` all True (no no-data).
* ``flat``           constant 50 m;
* ``incline``        a plane rising ``grade`` per metre toward azimuth ``az`` (degrees, 0 = east);
* ``ridge`` / ``valley``  a Gaussian ridge / valley of height ``h`` and width ``w`` through the tile centre,
                     its axis at azimuth ``az``;
* ``water``          flat land cut by a straight river of width ``w`` through the tile centre, axis at azimuth ``az``.
Graphs: an orthogonal ``grid``, a ``spoke_ring`` layout around a centre, and a ``dendritic`` (tree) layout.
"""
from __future__ import annotations

import math

import numpy as np

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph

RES, N, X0, Y1 = 4.0, 256, -256.0, 768.0
CENTRE = np.array([TILE_M / 2, TILE_M / 2])


def _xy():
    c = (np.arange(N) + 0.5) * RES
    return np.meshgrid(X0 + c, Y1 - c)                     # x east, y north (row 0 = north)


def _axis_dist(x, y, az):
    """Signed distance from the line through the tile centre at azimuth az (degrees)."""
    a = math.radians(az)
    return -(x - CENTRE[0]) * math.sin(a) + (y - CENTRE[1]) * math.cos(a)


def terrain(kind: str, grade: float = 0.10, az: float = 0.0, h: float = 40.0, w: float = 150.0, base: float = 50.0) -> dict:
    x, y = _xy()
    water = np.zeros((N, N), np.int8)
    if kind == "flat":
        z = np.full((N, N), base)
    elif kind == "incline":
        a = math.radians(az)
        z = base + grade * ((x - CENTRE[0]) * math.cos(a) + (y - CENTRE[1]) * math.sin(a))
    elif kind in ("ridge", "valley"):
        d = _axis_dist(x, y, az)
        z = base + (1 if kind == "ridge" else -1) * h * np.exp(-0.5 * (d / (w / 2)) ** 2)
    elif kind == "water":
        z = np.full((N, N), base)
        water[np.abs(_axis_dist(x, y, az)) <= w / 2] = 1
    else:
        raise ValueError(kind)
    z = z.astype(np.float32)
    return {"elev": z, "valid": np.ones((N, N), bool), "water": water,
            "water_surface": np.where(water > 0, z - 1.0, np.nan).astype(np.float32)}


def _graph(nodes, edges) -> RoadGraph:
    nodes = np.asarray(nodes, float)
    e = np.unique(np.sort(np.asarray(edges, int), axis=1), axis=0)
    return RoadGraph(nodes, e, np.zeros(len(nodes), bool), edge_class=np.full(len(e), 2), edge_struct=np.zeros(len(e), int))


def grid(spacing: float = 64.0, rot_deg: float = 0.0, step: float = 32.0, size: float = TILE_M) -> RoadGraph:
    """An orthogonal grid of ``spacing`` m blocks, edges every ``step`` m, rotated about the tile centre and clipped."""
    pts, idx, edges = [], {}, []
    n = int(size // step)
    for i in range(n + 1):
        for j in range(n + 1):
            idx[i, j] = len(pts)
            pts.append((i * step, j * step))
    k = int(round(spacing / step))
    for i in range(n + 1):
        for j in range(n + 1):
            if i < n and j % k == 0:
                edges.append((idx[i, j], idx[i + 1, j]))
            if j < n and i % k == 0:
                edges.append((idx[i, j], idx[i, j + 1]))
    P = np.array(pts) - CENTRE
    a = math.radians(rot_deg)
    P = P @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + CENTRE
    g = _graph(P, edges)
    return clip(g)


def clip(g: RoadGraph, size: float = TILE_M) -> RoadGraph:
    """Keep the edges with both ends inside the tile (a crude clip for rotated fixtures)."""
    ok = ((g.nodes >= -1e-6) & (g.nodes <= size + 1e-6)).all(1)
    keep = ok[g.edges[:, 0]] & ok[g.edges[:, 1]]
    used = np.unique(g.edges[keep].ravel())
    remap = -np.ones(g.n, int)
    remap[used] = np.arange(len(used))
    return _graph(g.nodes[used], remap[g.edges[keep]])


def spoke_ring(centre=(256.0, 256.0), n_spokes: int = 8, radii=(64, 128, 192), step: float = 16.0) -> RoadGraph:
    c = np.asarray(centre, float)
    pts, edges = [c], []
    rmax = max(radii)
    spoke_nodes = {}
    for s in range(n_spokes):
        a = 2 * math.pi * s / n_spokes
        prev = 0
        for r in np.arange(step, rmax + 1e-9, step):
            pts.append(c + r * np.array([math.cos(a), math.sin(a)]))
            edges.append((prev, len(pts) - 1))
            prev = len(pts) - 1
            spoke_nodes[s, round(r)] = prev
    for r in radii:
        m = max(int(2 * math.pi * r / step / n_spokes), 1)
        for s in range(n_spokes):
            a0 = 2 * math.pi * s / n_spokes
            prev = spoke_nodes[s, round(r)]
            for t in range(1, m + 1):
                if t == m:
                    nxt = spoke_nodes[(s + 1) % n_spokes, round(r)]
                else:
                    a = a0 + (2 * math.pi / n_spokes) * t / m
                    pts.append(c + r * np.array([math.cos(a), math.sin(a)]))
                    nxt = len(pts) - 1
                edges.append((prev, nxt))
                prev = nxt
    return _graph(pts, edges)


def dendritic(depth: int = 4, length: float = 90.0, seed: int = 0) -> RoadGraph:
    """A tree: a trunk from the south edge, each branch splitting in two at +-40 degrees, segments of 30 m."""
    rng = np.random.default_rng(seed)
    pts, edges = [np.array([256.0, 0.0])], []
    tips = [(0, math.pi / 2, length)]
    for _ in range(depth):
        nxt = []
        for node, ang, L in tips:
            for turn in (-0.7, 0.7):
                a = ang + turn + rng.normal(0, 0.05)
                prev = node
                for _k in range(3):
                    p = pts[prev] + (L / 3) * np.array([math.cos(a), math.sin(a)])
                    if not ((p > 2) & (p < TILE_M - 2)).all():
                        break
                    pts.append(p)
                    edges.append((prev, len(pts) - 1))
                    prev = len(pts) - 1
                if prev != node:
                    nxt.append((prev, a, L * 0.8))
        tips = nxt
    return _graph(pts, edges)
