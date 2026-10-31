"""Milestone 9 subdivision metrics S1-S4 (docs/MILESTONE_9_PLAN.md §4), same code for observed and generated parcels.

Per tile, from parcel polygons + their frontage / interior / cut flags (layout/parcels.parcel_table rules),
the street graph and the green polygons:
  * S1: per street side with >= 3 lots, robust width spread = IQR / median of frontage widths;
  * S2: per street lot, rear ratio = (distance from its street edge to the lot's rear) / block depth along
        the edge's inward normal (lots assigned to the edge their boundary runs along most, within 8 m);
  * S2b (M9b): as S2 but the block depth is the median over normals at 25, 37.5, 50, 62.5, 75 % of the
        edge, and ratios are clipped to [0, 1.5];
  * S3: block type of each CLOSED block from its parcel count (none / single / few 2-3 / rows >= 4);
  * S4: per closed block, share of the block area that is neither parcel nor green and lies more than
        8 m inside the block edge (interior leftover).
Street edges are layout/parcel_gen.street_edges, the same edges the tidy generator uses.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.layout.blocks import block_of, extract_blocks
from urbanmorph.layout.parcel_gen import street_edges
from urbanmorph.layout.parcels import FRONT_TOL_M
from urbanmorph.layout.tidy_parcels import block_type, full_depth

SIDE_TOL_M = 8.0
INNER_M = 8.0


def _depth_at(block, a, t, n, L, frac):
    p = a + t * L * frac
    r = np.arange(0.5, 600, 1.0)
    pts = p[None] + r[:, None] * n[None]
    out = np.flatnonzero(~shapely.contains_xy(block, pts[:, 0], pts[:, 1]))
    return float(r[out[0]]) if len(out) else 600.0


def tile_subdivision(polys, frontage, interior, cut, graph, class_widths, green=()) -> dict:
    seg = graph.segments()
    near = shapely.MultiLineString([s.reshape(2, 2) for s in seg]).buffer(FRONT_TOL_M) if len(seg) else shapely.Polygon()
    blocks = extract_blocks(graph, class_widths)
    out = {"S1": [], "S2": [], "S2b": [], "S3": [], "S4": []}
    if not len(polys):
        rp = np.zeros((0, 2))
    else:
        rp = np.array([p.representative_point().coords[0] for p in polys])
    bid = block_of(rp, blocks) if len(rp) else np.zeros(0, int)
    cover = shapely.union_all(list(polys) + list(green)) if (len(polys) or len(green)) else shapely.Polygon()
    for b in blocks:
        if b["open"]:
            continue
        poly = b["polygon"]
        idx = np.flatnonzero(bid == b["block_id"])
        out["S3"].append(block_type(len(idx)))
        inner = poly.buffer(-INNER_M)
        left = inner.difference(cover).area if not inner.is_empty else 0.0
        out["S4"].append(left / poly.area)
        edges = street_edges(poly, near)
        if not edges or not len(idx):
            continue
        segs = [shapely.LineString([a, a + t * L]) for L, a, t, n in edges]
        side = {}
        for i in idx:
            if interior[i] or cut[i] or not frontage[i] > 1:
                continue
            ov = [polys[i].buffer(SIDE_TOL_M).intersection(s).length for s in segs]
            k = int(np.argmax(ov))
            if ov[k] > 1.0:
                side.setdefault(k, []).append(i)
        for k, lots in side.items():
            L, a, t, n = edges[k]
            D = full_depth(poly, a, t, n, L)
            rear = [float(((np.asarray(polys[i].exterior.coords) - a) @ n).max()) for i in lots]
            out["S2"] += [r / D for r in rear]
            Db = float(np.median([_depth_at(poly, a, t, n, L, f) for f in (0.25, 0.375, 0.5, 0.625, 0.75)]))
            out["S2b"] += [float(np.clip(r / Db, 0, 1.5)) for r in rear]
            w = np.asarray(frontage)[lots]
            if len(w) >= 3:
                out["S1"].append(float(np.subtract(*np.percentile(w, [75, 25])) / np.median(w)))
    return out
