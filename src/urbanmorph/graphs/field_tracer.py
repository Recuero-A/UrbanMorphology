"""Milestone 15 external baseline: a cross-field street tracer in the spirit of Chen et al. (2008), "Interactive
procedural street modeling" (streets traced along a direction field), documented here as "Chen-inspired"; it is NOT
the authors' implementation (none is available).

The tracer follows the SAME style field as the samplers' style guidance (graphs/style.StyleField: grid, organic,
radial, mixtures), and uses the SAME graph operations (GenState.resolve_step / commit: snapping, edge splitting,
merging) and the SAME hard feasibility rules (tc_potentials: no ground edge over or ending in water) plus the optional
grade ceiling, so post-processing is identical across variants. It has no learned prior and no terrain preference
beyond those hard rules.

Algorithm: streamlines of fixed step (``step`` m) start at a start point near the tile centre in the four cross-field
directions; at each step the field direction closest to the current heading is followed. A streamline ends when it
leaves the tile, would violate a hard rule, or joins existing street (merge / T-junction / crossing). Every
``d_sep`` metres a streamline seeds two perpendicular streamlines (FIFO queue). Tracing stops at the length budget
``max_km`` or when no seed is left; the achieved length is reported.
"""
from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np

from urbanmorph.graphs.generate import GenState
from urbanmorph.graphs.style import StyleField, grade_caps
from urbanmorph.graphs.tc_potentials import TerrainField, edge_features
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.terrain.road_feasibility import water_kind_at


def trace(d: dict, sf: StyleField, pots, rng, step: float = 26.0, d_sep: float = 78.0, max_km: float = 6.0,
          max_steps: int = 60, cls: int = 2, grade_cap: str = "off"):
    """(RoadGraph, stats). ``sf`` must carry a heading field (grid / organic / radial / mixture)."""
    if not sf.weights:
        raise ValueError("the tracer needs a heading field")
    tf = TerrainField(terrain_of(d))
    caps = grade_caps(grade_cap)
    stats = Counter()
    s = GenState()
    start = np.array([256.0, 256.0]) + rng.uniform(-30, 30, 2)
    for _ in range(200):                                     # a dry start point near the centre
        if water_kind_at(tf.t, start[:1], start[1:])[0] == 0:
            break
        start = np.array([256.0, 256.0]) + rng.uniform(-200, 200, 2)
    u0 = s.add_node(start, push=False)
    th0 = field_dir(sf, start)
    seeds = deque((u0, th0 + k * math.pi / 2) for k in range(4))
    total = 0.0
    while seeds and total < max_km * 1000.0:
        u, heading = seeds.popleft()
        cur, run = u, 0.0
        for _ in range(max_steps):
            th = closest_cross(field_dir(sf, s.nodes[cur]), heading)
            o = s.resolve_step(cur, step * np.array([math.cos(th), math.sin(th)]))
            if o["kind"] == "dropped":
                stats["end_dropped"] += 1
                break
            f = edge_features(tf, s.nodes[cur], o["q"], struct=0)
            _, reason, _ = pots.log_potential(f, cls, o["kind"])
            if not reason and caps and f["finite"] and np.isfinite(f["grade"]) and f["grade"] > caps[cls]:
                reason = "grade_cap"
            a = s.nodes[cur].copy()
            if reason in ("ground_in_water", "ends_in_water"):
                v = bridge(s, tf, pots, cur, th, cls)           # the samplers' bridge eligibility rule, shared
                if v < 0:
                    stats[f"end_{reason}"] += 1
                    break
                stats["bridges"] += 1
                total += float(np.hypot(*(s.nodes[v] - a)))
                heading, run, cur = th, 0.0, v
                continue
            if reason:
                stats[f"end_{reason}"] += 1
                break
            v = s.commit(cur, o, cls)
            seg = float(np.hypot(*(s.nodes[v] - a)))
            total += seg
            stats["steps"] += 1
            if o["kind"] != "new_node" or o["clipped"]:
                stats["end_joined" if o["kind"] != "new_node" else "end_tile"] += 1
                break
            heading = math.atan2(*(s.nodes[v] - a)[::-1])
            run += seg
            if run >= d_sep:
                seeds.extend([(v, heading + math.pi / 2), (v, heading - math.pi / 2)])
                run = 0.0
            cur = v
    stats["km"] = total / 1000.0
    stats["budget_hit"] = int(total >= max_km * 1000.0)
    return s.graph(with_class=True), dict(stats)


def bridge(s: GenState, tf: TerrainField, pots, u: int, th: float, cls: int, inc: float = 4.0) -> int:
    """Cross water straight ahead with the SAME structure rules the samplers use (tc_potentials: river only, span <=
    the real p99 of the class, crossing angle >= the real p5): the shortest feasible bridge landing on dry land, or -1."""
    from urbanmorph.graphs.tc_sampler import commit_structure, resolve_structure
    d = np.array([math.cos(th), math.sin(th)])
    a = s.nodes[u]
    span_max = pots.span_max.get(1, 300.0)
    for L in np.arange(2 * inc, span_max + inc, inc):
        q = a + L * d
        if water_kind_at(tf.t, q[:1], q[1:])[0] != 0:
            continue
        o = resolve_structure(s, u, L * d)
        f = edge_features(tf, a, o["q"], struct=1)
        _, reason, _ = pots.log_potential(f, cls, o["kind"])
        return -1 if reason else commit_structure(s, u, o, cls, 1)
    return -1


def field_dir(sf: StyleField, p) -> float:
    """The mixture's field direction at p: the weighted circular mean of the fields' directions on the 4-fold circle."""
    z = sum(w * np.exp(4j * sf.direction(k, np.asarray(p, float))) for k, w in sf.weights.items())
    return float(np.angle(z) / 4)


def closest_cross(theta: float, heading: float) -> float:
    """Of theta + k * 90 degrees, the direction closest to ``heading``."""
    k = round((heading - theta) / (math.pi / 2))
    return theta + k * math.pi / 2
