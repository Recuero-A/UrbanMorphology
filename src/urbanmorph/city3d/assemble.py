"""Shared pieces of the real and generated city converters: streets, road surface, blocks, parcels, land use."""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString

from urbanmorph.city3d.format import ROAD_CLASSES, polygon, polygons, rings

TILE = shapely.box(0, 0, 512.0, 512.0)
ACCESS_M = 2.0          # a parcel has street access if it lies within this distance of the road surface
MIN_BLOCK_M2 = 50.0


def streets_from_graph(g, class_widths) -> tuple[list, list]:
    """Streets = maximal chains of graph edges between nodes of degree != 2 (junctions, dead ends, border).
    Each keeps its per-segment class widths (identical to ``road_polygons``) and its majority class."""
    deg = g.degree()
    adj = g.adjacency()
    cls = g.classes()
    eidx = {}
    for k, (a, b) in enumerate(g.edges):
        eidx[(int(a), int(b))] = eidx[(int(b), int(a))] = k
    stop = {v for v in range(g.n) if deg[v] != 2 or g.border[v]}
    used = np.zeros(len(g.edges), bool)
    streets = []

    def walk(u, v):
        path, edges = [u, v], [eidx[(u, v)]]
        while v not in stop:
            nxt = [w for w in adj[v] if w != path[-2] and not used[eidx[(v, w)]] and eidx[(v, w)] not in edges]
            if not nxt:
                break
            path.append(nxt[0])
            edges.append(eidx[(v, nxt[0])])
            v = nxt[0]
        return path, edges

    starts = sorted(stop) + list(range(g.n))                  # loops without junctions are picked up last
    for u in starts:
        for v in adj[u]:
            k = eidx[(u, v)]
            if used[k]:
                continue
            path, edges = walk(u, v)
            used[edges] = True
            c = cls[edges]
            streets.append({"id": f"s{len(streets)}", "centerline": [[round(float(x), 3), round(float(y), 3)] for x, y in g.nodes[path]],
                            "segment_widths_m": [float(class_widths[int(x)]) for x in c], "width_source": "class_default",
                            "class": ROAD_CLASSES[int(np.bincount(c).argmax())], "segment_classes": [ROAD_CLASSES[int(x)] for x in c],
                            "nodes": [f"n{path[0]}", f"n{path[-1]}"]})
    nodes = [{"id": f"n{v}", "xy": [round(float(x), 3), round(float(y), 3)], "degree": int(deg[v]), "border": bool(g.border[v])}
             for v, (x, y) in enumerate(g.nodes) if deg[v] > 0 and (v in stop)]
    keep = {n["id"] for n in nodes}
    for s in streets:                                          # a chain closing on itself ends on a degree-2 node
        for nid in s["nodes"]:
            if nid not in keep:
                v = int(nid[1:])
                nodes.append({"id": nid, "xy": [round(float(x), 3) for x in g.nodes[v]], "degree": int(deg[v]),
                              "border": bool(g.border[v])})
                keep.add(nid)
    return streets, nodes


def street_surface(streets) -> shapely.Geometry:
    """Union of every street segment buffered by half its width (flat caps would leave gaps at bends)."""
    parts = []
    for s in streets:
        xy = np.asarray(s["centerline"])
        for (a, b), w in zip(zip(xy[:-1], xy[1:]), s["segment_widths_m"]):
            parts.append(LineString([a, b]).buffer(w / 2))
    return shapely.union_all(parts).intersection(TILE) if parts else shapely.Polygon()


def blocks_from_surface(surface, streets) -> list:
    land = TILE.difference(surface)
    out = []
    lines = [(s["id"], LineString(s["centerline"]), max(s["segment_widths_m"])) for s in streets]
    for p in polygons(land):
        if p.area < MIN_BLOCK_M2:
            continue
        ring = p.boundary
        adj = [sid for sid, ln, w in lines if ln.distance(ring) <= w / 2 + 1.0]
        out.append({"id": f"b{len(out)}", "polygon": rings(p), "streets": adj, "_poly": p})
    return out


def block_of(geom, blocks) -> str | None:
    c = geom.representative_point()
    for b in blocks:
        if b["_poly"].contains(c):
            return b["id"]
    best = max(blocks, key=lambda b: b["_poly"].intersection(geom).area, default=None)
    return best["id"] if best is not None and best["_poly"].intersects(geom) else None


def street_access(geom, surface, streets) -> tuple[bool, str | None]:
    if surface.is_empty or geom.distance(surface) > ACCESS_M:
        return False, None
    lines = [(s["id"], LineString(s["centerline"])) for s in streets]
    return True, min(lines, key=lambda t: t[1].distance(geom))[0] if lines else None


def strip_private(c: dict) -> dict:
    """Drop helper shapely objects (keys starting with '_') before saving."""
    for k in ("blocks", "parcels", "buildings", "landuse"):
        for x in c[k]:
            for key in [key for key in x if key.startswith("_")]:
                del x[key]
    return c


def shapes(c: dict, key: str, field: str = "polygon") -> list:
    return [polygon(x[field]) for x in c[key]]
