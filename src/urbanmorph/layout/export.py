"""2D map rendering and vector export of generated layouts (Milestone 6).

Export: one GeoPackage per layout with layers
    roads         LineString per edge, attributes street_class (name) and width_m
    road_surface  the buffered road surface (Polygon / MultiPolygon)
    buildings     Polygon per building, attribute area_m2
    green         Polygon per green space, attribute area_m2
Coordinates are tile-local metres (x east, y north, origin = south-west corner), i.e. a local
engineering CRS with no geographic location (generated tiles are not anywhere on Earth). These
layers are what a later 3D stage extrudes.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString

from urbanmorph.graphs.road_graph import CLASS_NAMES, TILE_M, road_polygons

ROAD_FILL = "#c4c4c4"
BUILDING_FILL = "#3f3f46"
GREEN_FILL = "#6fae6b"
GROUND = "#f7f6f2"


def _fill(ax, geom, color, lw=0.0, edge=None, z=1):
    if geom.is_empty:
        return
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon":
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, color=color, lw=lw, ec=edge or color, zorder=z)
        for hole in p.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color=GROUND, lw=0, zorder=z)


def draw_layout(ax, graph, buildings, green, class_widths, title=None):
    ax.set_facecolor(GROUND)
    for g in green:
        _fill(ax, g, GREEN_FILL, z=1)
    _fill(ax, road_polygons(graph, class_widths=class_widths), ROAD_FILL, z=2)
    for b in buildings:
        _fill(ax, b, BUILDING_FILL, z=3)
    ax.set_xlim(0, TILE_M)
    ax.set_ylim(0, TILE_M)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d4d4d4")
    if title:
        ax.set_title(title, fontsize=7.5, loc="left", color="#333333")


def export_layout(path, graph, buildings, green, class_widths) -> None:
    import geopandas as gpd
    cls = graph.classes()
    seg = graph.segments()
    roads = gpd.GeoDataFrame({"street_class": [CLASS_NAMES[c] for c in cls],
                              "width_m": [class_widths[c] for c in cls]},
                             geometry=[LineString(s.reshape(2, 2)) for s in seg], crs=None)
    roads.to_file(path, layer="roads", driver="GPKG")
    gpd.GeoDataFrame(geometry=[road_polygons(graph, class_widths=class_widths)], crs=None).to_file(
        path, layer="road_surface", driver="GPKG")
    for name, polys in (("buildings", buildings), ("green", green)):
        gpd.GeoDataFrame({"area_m2": [p.area for p in polys]}, geometry=list(polys), crs=None).to_file(
            path, layer=name, driver="GPKG")


def layout_raster(graph_road_raster: np.ndarray, buildings, green) -> np.ndarray:
    """(3, 128, 128) uint8 raster with the pixel-centre rule (same channels as the dataset)."""
    from urbanmorph.layout.objects import polygon_pixels
    out = np.zeros((3, 128 * 128), np.uint8)
    out[0] = graph_road_raster.ravel()
    for b in buildings:
        out[1, polygon_pixels(b)] = 1
    for g in green:
        out[2, polygon_pixels(g)] = 1
    return out.reshape(3, 128, 128)
