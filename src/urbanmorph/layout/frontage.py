"""Street frontages: buildings as sequences along each side of each street (Milestone 7).

Real buildings are not independent points: 67-82 % lie within 20 m of a street, and 78-92 % are
aligned within 5 deg of their nearest neighbour (training tiles). Following the row / ring
representations of GlobalMapper (He & Aliaga 2023) and BlockPlanner (Xu et al. 2021), frontage
buildings are generated IN ORDER along each street side, each one predicted relative to the
previous one: gap along the street, setback, orientation relative to the street, size, shape.
Rows, rhythm (regular spacing) and consistent setbacks can then be learned directly.

Definitions (all programmed, applied identically to real and generated layouts):
* street link: a chain of edges between nodes of degree != 2 (junctions, dead ends, border nodes);
* frontage: one side of one link, as a directed polyline with its buildings on the LEFT
  (the right side is the reversed link);
* a building belongs to the frontage of its nearest link if its polygon is within FRONT_M of the
  street centreline; the side is given by the anchor; its position s is the anchor's projection;
* order: links by street class (major first) then length (longest first); left side, then right.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString

from urbanmorph.layout.objects import mrr_params

FRONT_M = 20.0


def street_links(graph) -> list[tuple[np.ndarray, int]]:
    """(polyline (n, 2), street class) for every chain of edges between nodes of degree != 2."""
    adj = graph.adjacency()
    deg = graph.degree()
    cls = graph.classes()
    eid = {(int(min(a, b)), int(max(a, b))): i for i, (a, b) in enumerate(graph.edges)}
    used = np.zeros(len(graph.edges), bool)
    links = []

    def walk(u, v):
        path, classes = [u], []
        prev, cur = u, v
        while True:
            e = eid[(min(prev, cur), max(prev, cur))]
            used[e] = True
            classes.append(cls[e])
            path.append(cur)
            if deg[cur] != 2 or graph.border[cur]:
                break
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt or used[eid[(min(cur, nxt[0]), max(cur, nxt[0]))]]:
                break
            prev, cur = cur, nxt[0]
        return graph.nodes[path], int(np.bincount(classes).argmax())

    starts = [u for u in range(graph.n) if deg[u] != 2 or graph.border[u]]
    for u in starts:
        for v in adj[u]:
            if not used[eid[(min(u, v), max(u, v))]]:
                links.append(walk(u, v))
    for i, (a, b) in enumerate(graph.edges):          # pure cycles of degree-2 nodes
        if not used[i]:
            links.append(walk(int(a), int(b)))
    links = [(p, c) for p, c in links if len(p) >= 2 and LineString(p).length > 1.0]
    links.sort(key=lambda pc: (pc[1], -LineString(pc[0]).length))
    return links


def frontages(graph) -> list[dict]:
    """Directed frontages (left side of each link, then its right side = reversed link)."""
    out = []
    for i, (p, c) in enumerate(street_links(graph)):
        for side, q in ((0, p), (1, p[::-1])):
            line = LineString(q)
            out.append({"link": i, "side": side, "line": line, "length": line.length, "cls": c})
    return out


def tangent(line: LineString, s: float) -> float:
    """Direction (radians) of the polyline at arc length s."""
    L = line.length
    a = np.asarray(line.interpolate(max(0.0, min(s, L) - 0.5)).coords[0])
    b = np.asarray(line.interpolate(min(L, max(s, 0.0) + 0.5)).coords[0])
    d = b - a
    return float(np.arctan2(d[1], d[0]))


def point_at(line: LineString, s: float) -> np.ndarray:
    return np.asarray(line.interpolate(min(max(s, 0.0), line.length)).coords[0])


def assign(polys: list, graph, anchors: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    """Assign buildings to frontages. Returns the frontage list (each with 'members': building
    indices ordered by s, and 's', 'd' arrays) and a boolean mask of frontage buildings."""
    fr = frontages(graph)
    n = len(polys)
    is_front = np.zeros(n, bool)
    for f in fr:
        f["members"], f["s"], f["d"] = [], [], []
    if not n or not fr:
        return fr, is_front
    if anchors is None:
        anchors = np.array([mrr_params(p)[0] for p in polys])
    links = {}
    for f in fr:
        links.setdefault(f["link"], f["line"] if f["side"] == 0 else LineString(np.asarray(f["line"].coords)[::-1]))
    link_ids = sorted(links)
    link_lines = np.array([links[i] for i in link_ids], dtype=object)
    parr = np.array(polys, dtype=object)
    for b in range(n):
        dist = shapely.distance(link_lines, parr[b])
        k = int(np.argmin(dist))
        if dist[k] > FRONT_M:
            continue
        line = link_lines[k]
        s = line.project(shapely.Point(anchors[b]))
        a = tangent(line, s)
        rel = anchors[b] - point_at(line, s)
        side = 0 if np.cos(a) * rel[1] - np.sin(a) * rel[0] >= 0 else 1
        f = next(f for f in fr if f["link"] == link_ids[k] and f["side"] == side)
        s_dir = s if side == 0 else line.length - s
        f["members"].append(b)
        f["s"].append(s_dir)
        f["d"].append(float(np.hypot(*rel)))
        is_front[b] = True
    for f in fr:
        o = np.argsort(f["s"], kind="stable")
        f["members"] = [f["members"][i] for i in o]
        f["s"] = np.asarray(f["s"])[o]
        f["d"] = np.asarray(f["d"])[o]
    return fr, is_front


# ------------------------------------------------------------------ pattern metrics (real and generated)
def interior_angles(poly) -> np.ndarray:
    """Interior angles (degrees) of the exterior ring after removing sub-0.5 m detail."""
    c = np.asarray(shapely.simplify(poly, 0.5).exterior.coords)[:-1]
    if len(c) < 3:
        return np.array([180.0])
    if not shapely.LinearRing(c).is_ccw:
        c = c[::-1]
    a = np.roll(c, 1, 0) - c
    b = np.roll(c, -1, 0) - c
    x = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))) % 360
    return 360 - x


def pattern_metrics(buildings: list, graph) -> dict:
    """How buildings relate to each other and to the streets (per tile).

    * nn_aligned_5deg: share of buildings whose orientation is within 5 deg (mod 90) of their nearest
      neighbour's (rows, terraces, grids give high values; scattered layouts low);
    * frontage_share: share within FRONT_M of a street;
    * setback_std_m: per frontage with >= 3 buildings, std of the setback; median over frontages
      (low = a regular building line);
    * spacing_cv: per frontage with >= 3 buildings, coefficient of variation of the gaps between
      consecutive buildings; median (low = regular rhythm);
    * acute_share: share of footprints with an interior angle < 45 deg.
    """
    nan = float("nan")
    b = [p for p in buildings if not p.is_empty]
    if len(b) < 3:
        return {"nn_aligned_5deg": nan, "frontage_share": nan, "setback_std_m": nan, "spacing_cv": nan,
                "acute_share": nan, "frontage_size_similarity": nan}
    enc = [mrr_params(p) for p in b]
    anchors = np.array([e[0] for e in enc])
    theta = np.array([e[1] for e in enc])
    D = np.hypot(*(anchors[:, None] - anchors[None]).transpose(2, 0, 1))
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    dth = np.degrees(np.abs((theta - theta[nn]) % (np.pi / 2)))
    dth = np.minimum(dth, 90 - dth)
    fr, is_front = assign(b, graph, anchors)
    sb, cv, sim = [], [], []
    area = np.array([p.area for p in b])
    for f in fr:
        m = f["members"]
        sim += list(np.abs(np.log(area[m[1:]]) - np.log(area[m[:-1]])) < np.log(1.5)) if len(m) >= 2 else []
        if len(m) >= 3:
            sb.append(np.std(f["d"]))
            gaps = np.diff(f["s"])
            if gaps.mean() > 0:
                cv.append(gaps.std() / gaps.mean())
    acute = np.array([interior_angles(p).min() < 45 for p in b])
    return {"nn_aligned_5deg": float((dth < 5).mean()), "frontage_share": float(is_front.mean()),
            "setback_std_m": float(np.median(sb)) if sb else nan, "spacing_cv": float(np.median(cv)) if cv else nan,
            "acute_share": float(acute.mean()),
            "frontage_size_similarity": float(np.mean(sim)) if sim else nan}
