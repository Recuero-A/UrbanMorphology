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

import json
import re

import geopandas as gpd
import pandas as pd

from urbanmorph.config import ROAD_WIDTH_RULES, RoadWidthRules

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _first(value):
    """OSMnx may merge edges into list-valued attributes; take the first item."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _parse_number(value) -> float | None:
    value = _first(value)
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return None
    match = _NUMBER.search(str(value))
    return float(match.group().replace(",", ".")) if match else None


def estimate_width(row, rules: RoadWidthRules = ROAD_WIDTH_RULES) -> tuple[float, str]:
    """Return (width_m, source) for one edge row."""
    lo, hi = rules.plausible_range_m

    width = _parse_number(row.get("width"))
    if width is not None and not lo <= width <= hi:
        width = None
    lanes = _parse_number(row.get("lanes"))
    lanes_width = lanes * rules.lane_width_m if lanes is not None and 1 <= lanes <= 12 else None

    # Tags can contradict each other (e.g. width=5 with lanes=4). The lane
    # count is treated as a lower bound, so the larger estimate wins.
    if width is not None and (lanes_width is None or width >= lanes_width):
        return width, "width_tag"
    if lanes_width is not None:
        return lanes_width, "lanes_tag"

    highway = str(_first(row.get("highway")) or "")
    if highway.endswith("_link"):
        return rules.default_by_class["link"], "class_default"
    return rules.default_by_class.get(highway, rules.fallback_m), "class_default"


def is_tunnel(row) -> bool:
    value = _first(row.get("tunnel"))
    return value not in (None, "no", "") and value == value  # value==value rejects NaN


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
