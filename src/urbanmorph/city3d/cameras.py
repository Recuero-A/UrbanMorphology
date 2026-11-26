"""Deterministic camera placement from the city format (written to cameras.json; Blender and Godot read it).

* aerial: orthographic, the whole tile.
* oblique: perspective from the south-west, looking at the tile centre.
* street: eye height 1.7 m on the longest street near the centre, looking 80 m along it.
* closeup: a building of >= 3 floors near the centre, seen from a street-centerline point 18-45 m away with a
  clear line of sight (no other footprint crosses it), at 5 m height.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from urbanmorph.city3d.format import polygon

EYE_M = 1.7


def street_view(city):
    best, score = None, -np.inf
    for s in city["streets"]:
        xy = np.asarray(s["centerline"], float)
        L = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        sc = L - 0.5 * float(np.linalg.norm(xy[len(xy) // 2] - 256))
        if L > 60 and sc > score:
            best, score = xy, sc
    if best is None:
        return [256.0, 40.0, EYE_M], [256.0, 256.0, EYE_M]
    line = LineString(best)
    a = line.interpolate(0.15, normalized=True)
    b = line.interpolate(min(line.length, line.project(a) + 80.0))
    return [a.x, a.y, EYE_M], [b.x, b.y, 2.5]


def closeup_view(city):
    foot = [polygon(b["footprint"]) for b in city["buildings"]]
    if not foot:
        return None
    tree = shapely.STRtree(foot)
    pts = []
    for s in city["streets"]:
        line = LineString(s["centerline"])
        pts += [line.interpolate(d) for d in np.arange(0, line.length, 5.0)]
    order = sorted(range(len(foot)), key=lambda i: foot[i].centroid.distance(Point(256, 256)))
    for i in order:
        b = city["buildings"][i]
        if b["floors"] < 3:
            continue
        P = foot[i]
        for p in sorted(pts, key=lambda q: abs(q.distance(P) - 28.0)):
            d = p.distance(P)
            if not 18.0 <= d <= 45.0:
                continue
            tgt = shapely.ops.nearest_points(P.exterior, p)[0]
            sight = LineString([p, tgt]).interpolate(d - 1.0)
            seg = LineString([p, sight])
            if any(foot[j].intersects(seg) for j in tree.query(seg)):
                continue
            return {"location": [p.x, p.y, 5.0], "look_at": [tgt.x, tgt.y, b["eave_m"] * 0.5], "building": b["id"],
                    "distance_m": round(d, 1)}
    return None


def cameras(city) -> dict:
    s0, s1 = street_view(city)
    out = {"aerial": {"type": "ORTHO", "location": [256, 256, 800], "ortho_scale": 512},
           "oblique": {"type": "PERSP", "lens": 30, "location": [-120, -160, 330], "look_at": [256, 256, 0]},
           "street": {"type": "PERSP", "lens": 22, "location": s0, "look_at": s1}}
    c = closeup_view(city)
    if c is not None:
        out["closeup"] = {"type": "PERSP", "lens": 35, **c}
    out["godot_spawn"] = {"spawn": [s0[0], s0[2], -s0[1]], "look_at": [s1[0], s1[2], -s1[1]],
                          "frame": "godot (x, y_up, -north), metres"}
    return out
