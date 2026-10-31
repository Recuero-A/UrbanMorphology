"""Observed parcels: measurement rules (Milestone 8). Applied identically to observed and generated parcels.

All definitions are PROGRAMMED approximations, documented here:
* parcels are clipped to the tile; slivers < 20 m2 dropped; a parcel touching the tile edge is CUT
  (its true shape is unknown; excluded from shape statistics);
* street frontage = the part of the parcel outline NOT shared with another parcel, not on the tile
  edge, and within FRONT_TOL_M of a street centreline (half of a wide street plus the sidewalk), i.e.
  the edge facing a street; a parcel without it is "interior" (rear edges backing onto unparcelled
  land such as rail or parks are not frontage);
* depth = largest distance from the frontage to any vertex of the parcel;
* a building belongs to the parcel holding the largest share of its area; it SPANS parcels if that
  share is < 90 %;
* coverage = building area inside the parcel / parcel area;
* setbacks (parcels with frontage and exactly one building): front = building-to-frontage distance;
  rear = building-to-outline distance over the outline part farther than 80 % of the depth from the
  frontage; side = building-to-outline distance over the remaining non-frontage outline.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.graphs.road_graph import TILE_M

TILE = shapely.box(0, 0, TILE_M, TILE_M)
MIN_PARCEL_M2 = 20.0
FRONT_TOL_M = 12.0


def clean_parcels(polys) -> list:
    out = []
    for p in polys:
        q = shapely.make_valid(p).intersection(TILE)
        for g in getattr(q, "geoms", [q]):
            if g.geom_type == "Polygon" and g.area >= MIN_PARCEL_M2:
                out.append(g)
    return out


def parcel_table(parcels: list, buildings: list, street_segments=None) -> tuple[list[dict], dict]:
    """``street_segments``: (E, 4) street centreline segments (tile-local metres)."""
    n = len(parcels)
    if not n:
        return [], {}
    arr = np.array(parcels, dtype=object)
    tree = shapely.STRtree(arr)
    edge = TILE.exterior.buffer(0.5)
    near_street = (shapely.MultiLineString([sg.reshape(2, 2) for sg in street_segments]).buffer(FRONT_TOL_M)
                   if street_segments is not None and len(street_segments) else None)
    rows = []
    for i, p in enumerate(parcels):
        nb = [j for j in tree.query(p.buffer(0.5), predicate="intersects") if j != i]
        shared = shapely.union_all([parcels[j].buffer(0.5) for j in nb]) if nb else shapely.Polygon()
        front = p.exterior.difference(shared).difference(edge)
        front = front.intersection(near_street) if near_street is not None else shapely.LineString()
        rect = shapely.minimum_rotated_rectangle(p)
        r = {"area_m2": p.area, "cut": bool(p.intersects(edge)), "frontage_m": front.length,
             "rectangularity": p.area / rect.area if rect.area else np.nan, "n_vertices": len(shapely.simplify(p, 0.5).exterior.coords) - 1,
             "front": front}
        if front.length > 1.0:
            v = shapely.points(np.asarray(p.exterior.coords))
            r["depth_m"] = float(shapely.distance(v, front).max())
        else:
            r["depth_m"] = np.nan
        rows.append(r)
    # buildings -> parcels
    owner, spans = [], 0
    for b in buildings:
        cand = tree.query(b, predicate="intersects")
        if not len(cand):
            owner.append(-1)
            continue
        inter = shapely.area(shapely.intersection(arr[cand], b))
        k = int(cand[np.argmax(inter)])
        owner.append(k)
        spans += inter.max() < 0.9 * b.area
    owner = np.array(owner)
    for i, r in enumerate(rows):
        mine = [buildings[j] for j in np.flatnonzero(owner == i)]
        r["n_buildings"] = len(mine)
        r["coverage"] = float(shapely.union_all(mine).intersection(parcels[i]).area / parcels[i].area) if mine else 0.0
        if len(mine) == 1 and r["frontage_m"] > 1.0 and np.isfinite(r["depth_m"]):
            b, p, front = mine[0], parcels[i], r["front"]
            c = np.asarray(p.exterior.coords)
            edges = [shapely.LineString([c[k], c[k + 1]]) for k in range(len(c) - 1)]
            mid = shapely.points([(c[k] + c[k + 1]) / 2 for k in range(len(c) - 1)])
            dmid = shapely.distance(mid, front)
            r["front_setback_m"] = float(b.distance(front))
            far = [e for e, d in zip(edges, dmid) if d > 0.8 * r["depth_m"]]
            side = [e for e, d in zip(edges, dmid) if 0.5 < d <= 0.8 * r["depth_m"]]
            r["rear_setback_m"] = float(min(b.distance(e) for e in far)) if far else np.nan
            r["side_setback_m"] = float(min(b.distance(e) for e in side)) if side else np.nan
        del r["front"]
    summary = {"buildings": len(buildings), "buildings_without_parcel": int((owner < 0).sum()),
               "buildings_spanning_parcels": int(spans)}
    return rows, summary
