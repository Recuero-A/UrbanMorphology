"""Vertical structure of streets (Milestone 12): bridges, tunnels and OSM layers.

OSM draws a street that passes over or under another WITHOUT a shared node; a real at-grade junction is a shared
node. The Milestone 5 graph build snapped every point within MERGE_M of another point, so an overpass and the
street below it could be fused into a false junction. This module supplies, per street piece, the information the
3D-aware build needs (``road_graph.build_graph(..., levels=, structs=)``):

* ``struct``: GROUND, BRIDGE (bridge=* except "no"), TUNNEL (tunnel=* except "no" / building_passage), PASSAGE
  (tunnel=building_passage: through a building, at street level);
* ``level``: the OSM ``layer`` tag when present, else +1 for a bridge, -1 for a tunnel, 0 otherwise (the OSM default
  convention). Points of different (level, structure) never merge; endpoints that coincide within PORTAL_M (a
  shared OSM node: the end of a bridge or a ramp) still join.

Tag source. Tile centerlines (OSMnx edges) carry only `bridge` and `tunnel`, and 8.6 % of them concatenate several
OSM ways, whose tags then describe only part of the line. The tags are therefore taken from the OSM ways
themselves (city clip, same snapshot; ``scripts/m12_build_structures.py``): a line made of one way gets that way's
tags; a line made of several ways is cut into the parts that lie along each tagged member way.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

GROUND, BRIDGE, TUNNEL, PASSAGE = 0, 1, 2, 3
STRUCT_NAMES = ("ground", "bridge", "tunnel", "passage")
PORTAL_M = 1.0          # endpoints closer than this are one OSM node even across levels
OVERLAY_M = 0.5         # a part of a multi-way line belongs to a member way if it lies within this distance


def _is_set(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip().lower() not in ("", "no", "nan", "none")


def parse_layer(v) -> int | None:
    if not _is_set(v):
        return None
    try:
        return int(round(float(str(v).split(";")[0].strip())))
    except ValueError:
        return None


def structure_of(bridge, tunnel) -> int:
    if _is_set(tunnel):
        return PASSAGE if str(tunnel).strip() == "building_passage" else TUNNEL
    if _is_set(bridge):
        return BRIDGE
    return GROUND


def level_of(layer, struct: int) -> int:
    lay = parse_layer(layer)
    if lay is not None:
        return lay
    return {BRIDGE: 1, TUNNEL: -1}.get(struct, 0)


def merge_key(level: int, struct: int) -> int:
    """Points merge only within one key: the level, and whether the piece is a bridge / tunnel structure."""
    return 2 * int(level) + int(struct in (BRIDGE, TUNNEL))


def osm_ids(v) -> list[int]:
    if isinstance(v, str) and v.startswith("["):
        v = ast.literal_eval(v)
    vals = v if isinstance(v, (list, tuple)) else [v]
    out = []
    for x in vals:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


# ------------------------------------------------------------------ OSM way table (one per city)
def load_structure_ways(path) -> dict:
    """{way id: {"struct", "level", "layer", "ford", "coords" (lon, lat)}} for street ways tagged bridge / tunnel /
    layer / ford, from the osmium geojsonseq export (only ways are kept)."""
    from urbanmorph.data.osm_extract import STREET_VALUES
    ways = {}
    for line in Path(path).read_text().split("\n"):            # not splitlines(): it also splits at \x1e
        line = line.strip("\x1e \r")
        if not line:
            continue
        f = json.loads(line)
        p, geom = f["properties"], f["geometry"]
        if p.get("@type") != "way" or p.get("highway") not in STREET_VALUES or geom["type"] != "LineString":
            continue
        st = structure_of(p.get("bridge"), p.get("tunnel"))
        ways[int(p["@id"])] = {"struct": st, "level": level_of(p.get("layer"), st), "layer": parse_layer(p.get("layer")),
                               "ford": _is_set(p.get("ford")), "coords": geom["coordinates"]}
    return ways


def tile_pieces(rc, ways: dict, to_local, xmin: float, ymin: float, class_fn):
    """Cut the tile centerlines ``rc`` (GeoDataFrame in the tile CRS) into pieces with vertical metadata.

    ``to_local``: pyproj transform lon/lat -> tile CRS. Returns lists (coords in tile-local metres, class, level,
    struct, source) where source is "way" (one OSM way), "overlay" (cut along member ways) or "untagged"."""
    lines, cls, lev, st, src = [], [], [], [], []
    local = {}

    def way_line(i):
        if i not in local:
            x, y = to_local(*np.asarray(ways[i]["coords"]).T)
            local[i] = LineString(np.c_[x, y])
        return local[i]

    tags = rc["highway"] if "highway" in rc else ["unclassified"] * len(rc)
    for geom, tag, oid in zip(rc.geometry, tags, rc["osmid"] if "osmid" in rc else [None] * len(rc)):
        c = class_fn(tag)
        ids = osm_ids(oid)
        members = [i for i in ids if i in ways and (ways[i]["struct"] != GROUND or ways[i]["level"] != 0)]
        for part in (geom.geoms if isinstance(geom, MultiLineString) else [geom]):
            if part.length <= 0:
                continue
            if not members:
                pieces = [(part, 0, GROUND, "untagged" if not any(i in ways for i in ids) else "way")]
            elif len(ids) == 1:
                w = ways[ids[0]]
                pieces = [(part, w["level"], w["struct"], "way")]
            else:
                pieces = _overlay(part, [(way_line(i), ways[i]["level"], ways[i]["struct"]) for i in members])
            for g, lv, s, how in pieces:
                lines.append(np.asarray(g.coords)[:, :2] - [xmin, ymin])
                cls.append(c), lev.append(int(lv)), st.append(int(s)), src.append(how)
    return lines, cls, lev, st, src


def _overlay(line: LineString, members) -> list:
    """Split ``line`` into intervals that run along a tagged member way (its tags) and the rest (ground)."""
    iv = []
    for wl, lv, s in members:
        hit = line.intersection(wl.buffer(OVERLAY_M, cap_style="flat"))
        for h in getattr(hit, "geoms", [hit]):
            if h.geom_type == "LineString" and h.length > 2 * OVERLAY_M:
                a, b = sorted((line.project(shapely.Point(h.coords[0])), line.project(shapely.Point(h.coords[-1]))))
                iv.append((a, b, lv, s))
    iv.sort()
    out, pos = [], 0.0
    for a, b, lv, s in iv:
        a = max(a, pos)
        if b - a < 1e-6:
            continue
        if a - pos > 1e-6:
            out.append((substring(line, pos, a), 0, GROUND, "overlay"))
        out.append((substring(line, a, b), lv, s, "overlay"))
        pos = b
    if line.length - pos > 1e-6:
        out.append((substring(line, pos, line.length), 0, GROUND, "overlay"))
    return [p for p in out if p[0].length > 0 and len(p[0].coords) >= 2]
