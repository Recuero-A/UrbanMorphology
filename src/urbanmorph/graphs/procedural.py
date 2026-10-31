"""Milestone 10, street baseline E: a procedural perturbed grid with parameters taken from TRAINING tiles only.

Per generated tile:
  * a city is drawn with the training-tile proportions, then one training tile of that city; its mean
    link length L (from ``graph_metrics``) sets the block size;
  * two spacings sx, sy = L * (u, 2 - u), u ~ U(0.6, 1.4) (rectangular blocks, mean link length about L);
  * a random grid orientation in [0, 90) deg;
  * every grid intersection jittered by N(0, (0.04 L)^2);
  * each grid segment dropped with probability P_DROP (dead ends, irregular blocks);
  * street classes: every 10th grid line major, every 5th mid, others local;
  * segments clipped to the tile and turned into a graph by ``build_graph`` (30 m resampling, 8 m snapping),
    exactly like the real data.
It is a reference for what a plain grid scores on the M10 metrics, not a candidate model.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, box

from urbanmorph.graphs.road_graph import TILE_M, RoadGraph, build_graph

P_DROP = 0.08
_TILE = box(0, 0, TILE_M, TILE_M)


def perturbed_grid(L: float, rng: np.random.Generator) -> RoadGraph:
    u = rng.uniform(0.6, 1.4)
    sx, sy = max(L * u, 25.0), max(L * (2 - u), 25.0)
    th = rng.uniform(0, np.pi / 2)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ext = TILE_M * 0.75                                   # half-extent covering the rotated tile
    xs = np.arange(-ext, ext + sx, sx) + rng.uniform(0, sx)
    ys = np.arange(-ext, ext + sy, sy) + rng.uniform(0, sy)
    X, Y = np.meshgrid(xs, ys)                            # (ny, nx)
    P = np.stack([X, Y], -1) + rng.normal(0, 0.04 * L, X.shape + (2,))
    P = P @ R.T + TILE_M / 2
    lines, classes = [], []
    cls = lambda k: 0 if k % 10 == 0 else (1 if k % 5 == 0 else 2)
    ox, oy = rng.integers(10), rng.integers(10)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for di, dj, k in ((0, 1, i + oy), (1, 0, j + ox)):
                if i + di >= P.shape[0] or j + dj >= P.shape[1] or rng.random() < P_DROP:
                    continue
                seg = shapely.intersection(LineString([P[i, j], P[i + di, j + dj]]), _TILE)
                if seg.is_empty or seg.length < 1.0 or seg.geom_type != "LineString":
                    continue
                lines.append(np.asarray(seg.coords))
                classes.append(cls(k))
    return build_graph(lines, classes=classes)


def sample_procedural(n: int, city_links: dict[str, list[float]], seed: int = 0) -> tuple[list[RoadGraph], list[str]]:
    """n tiles; ``city_links`` = training tiles' mean link lengths per city (sets the city proportions too)."""
    rng = np.random.default_rng(seed)
    cities = sorted(city_links)
    w = np.array([len(city_links[c]) for c in cities], float)
    out, names = [], []
    for _ in range(n):
        c = cities[rng.choice(len(cities), p=w / w.sum())]
        out.append(perturbed_grid(float(rng.choice(city_links[c])), rng))
        names.append(c)
    return out, names
