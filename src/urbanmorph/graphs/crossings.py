"""Classify where streets meet streets or water (Milestone 12).

Street-street, for a RoadGraph (built 3D-aware, graphs/levels.py):
* ``at_grade``: a junction node (degree >= 3, not on the border) whose edges are all at one vertical key - a
  shared OSM node, i.e. a CONFIRMED ground-level (or same-level) crossing;
* ``portal``: a junction joining different keys (the end of a bridge, a ramp onto a viaduct);
* ``grade_separated``: two edges that cross in plan view without a shared node, at different keys - confirmed by
  bridge / tunnel / layer metadata;
* ``ambiguous``: two edges that cross in plan view without a shared node at the SAME key - OSM says they do not
  meet, but no tag explains why (missing bridge / layer tag, or a mapping error). Never counted as a junction.

Street-water, for a RoadGraph and water geometry in the same tile coordinates:
* ``bridge`` (edge struct BRIDGE), ``tunnel`` (TUNNEL, or a negative level), ``ford`` (ground edge whose OSM way
  has ford=*: a confirmed ground-level crossing), and ``unknown`` (a ground edge with no tag: the crossing is
  NOT assumed to be at ground level).
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.graphs.levels import BRIDGE, TUNNEL, merge_key


def edge_keys(g) -> np.ndarray:
    return np.array([merge_key(lv, st) for lv, st in zip(g.levels(), g.structs())], int)


def street_crossings(g) -> dict:
    deg = g.degree()
    keys = edge_keys(g)
    inc: list[list[int]] = [[] for _ in range(g.n)]
    for k, (a, b) in enumerate(g.edges):
        inc[a].append(k)
        inc[b].append(k)
    at_grade = portal = 0
    for u in range(g.n):
        if deg[u] >= 3 and not g.border[u]:
            if len({keys[k] for k in inc[u]}) == 1:
                at_grade += 1
            else:
                portal += 1
    pairs = {"grade_separated": [], "ambiguous": []}
    if len(g.edges) > 1:
        segs = shapely.linestrings(g.segments().reshape(-1, 2, 2))
        tree = shapely.STRtree(segs)
        i, j = tree.query(segs, predicate="intersects")
        seen = {"grade_separated": set(), "ambiguous": set()}
        for a, b in zip(i, j):
            if a >= b or set(g.edges[a]) & set(g.edges[b]):
                continue                                              # same edge, or edges sharing a node
            kind = "grade_separated" if keys[a] != keys[b] else "ambiguous"
            x = shapely.intersection(segs[a], segs[b])
            c = shapely.get_coordinates(x).mean(0) if not x.is_empty else g.nodes[g.edges[a][0]]
            spot = (tuple(np.round(c).astype(int)), frozenset((keys[a], keys[b])))
            if spot not in seen[kind]:                                # one crossing touches several resampled edges
                seen[kind].add(spot)
                pairs[kind].append((int(a), int(b)))
    return {"at_grade": at_grade, "portal": portal, "grade_separated": len(pairs["grade_separated"]),
            "ambiguous": len(pairs["ambiguous"]), "pairs": pairs}


def water_crossings(g, water, ford_edges=None) -> dict:
    """``water``: a shapely geometry (polygons and / or waterway lines) in tile coordinates; ``ford_edges``: optional
    boolean per edge (from the OSM ford tag)."""
    out = {"bridge": 0, "tunnel": 0, "ford": 0, "unknown": 0}
    if water is None or water.is_empty or not len(g.edges):
        return out
    segs = shapely.linestrings(g.segments().reshape(-1, 2, 2))
    hit = np.flatnonzero(shapely.intersects(segs, water))
    lv, st = g.levels(), g.structs()
    ford = np.zeros(len(g.edges), bool) if ford_edges is None else np.asarray(ford_edges, bool)
    # one crossing = a connected run of water-touching edges (a deck is resampled into several edges)
    parent = {k: k for k in hit}
    by_node: dict = {}
    for k in hit:
        for v in g.edges[k]:
            by_node.setdefault(int(v), []).append(k)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for ks in by_node.values():
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    runs: dict = {}
    for k in hit:
        runs.setdefault(find(k), []).append(k)
    for ks in runs.values():
        if any(st[k] == BRIDGE for k in ks):
            out["bridge"] += 1
        elif any(st[k] == TUNNEL or lv[k] < 0 for k in ks):
            out["tunnel"] += 1
        elif any(ford[k] for k in ks):
            out["ford"] += 1
        else:
            out["unknown"] += 1
    return out
