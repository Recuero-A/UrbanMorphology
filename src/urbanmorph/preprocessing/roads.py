"""Turn road centerlines into approximate road surface polygons.

OSM stores most roads as *centerlines* (zero-width LineStrings). To rasterize
them as areas we buffer each line by half of an estimated width. The estimate
uses, in order of preference:

1. the ``width`` tag (metres), if present and plausible, unless the
   ``lanes`` tag implies a wider road (lanes are a lower bound);
2. the ``lanes`` tag * a nominal lane width;
3. a documented default for the ``highway`` class (see ``RoadWidthRules``).

This is an approximation of the carriageway, not a survey of the real street.

Some roads are mapped as AREAS (closed ways with ``area=yes``, e.g. pedestrian
squares). Those are downloaded separately as polygons and used as-is; their
outline edges in the graph would otherwise produce hollow rings.
"""

from __future__ import annotations

import ast
import json
import re

import geopandas as gpd
import pandas as pd

from urbanmorph.config import ROAD_WIDTH_RULES, RoadWidthRules

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _values(value) -> list:
    """All values of a possibly list-valued attribute.

    OSMnx simplification merges consecutive OSM ways into one edge; when their
    tags differ the attribute becomes a list (e.g. lanes = ['2', '3']). The
    list ORDER is not guaranteed (it differed between Overpass and extract
    sources), so every rule below must be order-independent.
    """
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:  # Python-style list string, e.g. "['2', '3']"
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    return [v for v in items if v is not None and not (isinstance(v, float) and v != v)]  # drop None/NaN


def _numbers(value) -> list[float]:
    out = []
    for v in _values(value):  # findall also handles OSM multi-values such as "2;3"
        out.extend(float(m.replace(",", ".")) for m in _NUMBER.findall(str(v)))
    return out


def _class_width(highway: str, rules: RoadWidthRules) -> float:
    if highway.endswith("_link"):
        return rules.default_by_class["link"]
    return rules.default_by_class.get(highway, rules.fallback_m)


def estimate_width(row, rules: RoadWidthRules = ROAD_WIDTH_RULES) -> tuple[float, str]:
    """Return (width_m, source) for one edge row.

    For merged edges with several values the MAXIMUM plausible value is used
    (widest part of the street), which is independent of list order.
    """
    lo, hi = rules.plausible_range_m

    widths = [w for w in _numbers(row.get("width")) if lo <= w <= hi]
    width = max(widths) if widths else None
    lanes = [n for n in _numbers(row.get("lanes")) if 1 <= n <= 12]
    lanes_width = max(lanes) * rules.lane_width_m if lanes else None

    # Tags can contradict each other (e.g. width=5 with lanes=4). The lane
    # count is treated as a lower bound, so the larger estimate wins.
    if width is not None and (lanes_width is None or width >= lanes_width):
        return width, "width_tag"
    if lanes_width is not None:
        return lanes_width, "lanes_tag"

    classes = [str(h) for h in _values(row.get("highway"))] or [""]
    return max(_class_width(h, rules) for h in classes), "class_default"


def is_tunnel(row) -> bool:
    """True only if EVERY merged part is a tunnel (a partly-surface edge is kept)."""
    values = [str(v) for v in _values(row.get("tunnel"))]
    return bool(values) and all(v not in ("no", "") for v in values)


def buffer_roads(edges: gpd.GeoDataFrame, rules: RoadWidthRules = ROAD_WIDTH_RULES) -> gpd.GeoDataFrame:
    """Buffer centerlines (in a *metric* CRS) into road polygons.

    Tunnels are dropped: they are underground and do not occupy surface space.
    Round caps/joins are used so that segments meeting at an intersection merge
    into a continuous surface.
    """
    if edges.empty:
        return gpd.GeoDataFrame(columns=["width_m", "width_source"], geometry=[], crs=edges.crs)
    if not edges.crs or edges.crs.is_geographic:
        raise ValueError("buffer_roads needs a projected (metric) CRS; buffering degrees is meaningless")

    surface = edges[~edges.apply(is_tunnel, axis=1)].copy()
    est = surface.apply(lambda r: estimate_width(r, rules), axis=1, result_type="expand")
    surface["width_m"] = est[0].astype(float)
    surface["width_source"] = est[1]
    surface["geometry"] = surface.geometry.buffer(surface["width_m"] / 2, cap_style="round")
    return surface


def combine_road_surfaces(buffered: gpd.GeoDataFrame, road_areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Append road-area polygons (already metric) to the buffered centerlines."""
    if road_areas.empty:
        return buffered
    areas = road_areas[["geometry"]].copy()
    areas["highway"] = road_areas.get("highway")
    areas["width_m"] = float("nan")
    areas["width_source"] = "area_polygon"
    return gpd.GeoDataFrame(
        pd.concat([buffered, areas.to_crs(buffered.crs)], ignore_index=True), crs=buffered.crs
    )
