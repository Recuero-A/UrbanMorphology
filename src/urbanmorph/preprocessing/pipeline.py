"""End-to-end: raw OSM layers -> (3, H, W) uint8 layout + metadata.

Overlap policy: the three channels are INDEPENDENT binary masks (multi-label).
No priority is applied, so a pixel may be 1 in several channels (e.g. a road
buffer that reaches into a building footprint, or a garden polygon that is
mapped on top of a building roof). Overlap counts are recorded in the
metadata; any exclusive "one class per pixel" view can be derived later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import geopandas as gpd
import numpy as np

from urbanmorph.config import CHANNELS, ROAD_WIDTH_RULES, AreaSpec
from urbanmorph.preprocessing.projection import SquareBounds, square_bounds, to_metric_and_clip
from urbanmorph.preprocessing.rasterize import rasterize_geometries
from urbanmorph.preprocessing.roads import buffer_roads, combine_road_surfaces


@dataclass
class ProcessedSample:
    layout: np.ndarray  # (3, H, W) uint8
    bounds: SquareBounds
    road_centerlines: gpd.GeoDataFrame
    road_polygons: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    green: gpd.GeoDataFrame


def build_layout(
    area: AreaSpec,
    edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    green: gpd.GeoDataFrame,
    road_areas: gpd.GeoDataFrame | None = None,
    bounds: SquareBounds | None = None,
) -> ProcessedSample:
    """Project, clip, buffer and rasterize. Pure function of its inputs."""
    bounds = bounds or square_bounds(area)
    crs = area.projected_crs

    # Buffer roads BEFORE clipping, so a road just outside the square whose
    # surface reaches inside is still represented; then clip the polygons.
    edges_m = edges.to_crs(crs) if not edges.empty else gpd.GeoDataFrame(geometry=[], crs=crs)
    road_polys = buffer_roads(edges_m)
    if road_areas is not None and not road_areas.empty:
        road_polys = combine_road_surfaces(road_polys, road_areas.to_crs(crs))
    road_polys = to_metric_and_clip(road_polys, crs, bounds)
    centerlines = to_metric_and_clip(edges_m, crs, bounds)

    buildings_m = to_metric_and_clip(buildings, crs, bounds)
    green_m = to_metric_and_clip(green, crs, bounds)

    res = area.resolution
    layout = np.stack(
        [
            rasterize_geometries(road_polys.geometry, bounds, res),
            rasterize_geometries(buildings_m.geometry, bounds, res),
            rasterize_geometries(green_m.geometry, bounds, res),
        ]
    ).astype(np.uint8)

    return ProcessedSample(layout, bounds, centerlines, road_polys, buildings_m, green_m)


def layout_statistics(layout: np.ndarray) -> dict:
    total = layout.shape[1] * layout.shape[2]
    r, b, g = (layout[i].astype(bool) for i in range(3))
    return {
        "coverage_fraction": {CHANNELS[i]: round(float(layout[i].sum()) / total, 4) for i in range(3)},
        "overlap_pixels": {
            "roads&buildings": int((r & b).sum()),
            "roads&green_spaces": int((r & g).sum()),
            "buildings&green_spaces": int((b & g).sum()),
        },
        "empty_pixels": int((~(r | b | g)).sum()),
    }


def road_buffer_metadata(sample: ProcessedSample) -> dict:
    sources = (
        sample.road_polygons["width_source"].value_counts().to_dict()
        if "width_source" in sample.road_polygons
        else {}
    )
    return {
        "method": "Centerlines buffered by width/2 with round caps; tunnels excluded. "
        "Roads mapped as areas (area=yes) are added as their own polygons.",
        "width_priority": [
            "max(OSM width tag, OSM lanes tag x lane_width_m)",
            "OSM lanes tag x lane_width_m",
            "highway-class default",
        ],
        "rules": asdict(ROAD_WIDTH_RULES),
        "segments_by_width_source": {k: int(v) for k, v in sources.items()},
        "caveat": "Buffered widths approximate the carriageway; they are not measured road widths "
        "and exclude sidewalks.",
    }


def make_metadata(area: AreaSpec, sample: ProcessedSample, download_info: dict, processing_date: str) -> dict:
    from urbanmorph.preprocessing.projection import wgs84_corners

    b = sample.bounds
    return {
        "sample_id": area.sample_id,
        "description": area.description,
        "source": download_info.get("source", "OpenStreetMap"),
        "attribution": download_info.get("attribution"),
        "osm_retrieved_at_utc": download_info.get("retrieved_at_utc"),
        "osm_query": download_info.get("query"),
        "center_wgs84": {"lat": area.center_lat, "lon": area.center_lon},
        "geographic_bounds": wgs84_corners(area, b),
        "projected_crs": area.projected_crs,
        "projected_bounds_m": {"xmin": b.xmin, "ymin": b.ymin, "xmax": b.xmax, "ymax": b.ymax},
        "physical_size_m": area.size_m,
        "resolution_px": area.resolution,
        "pixel_size_m": area.pixel_size_m,
        "array": {"shape": list(sample.layout.shape), "dtype": "uint8", "values": [0, 1],
                  "axis_order": "(channel, row, col); row 0 = north edge, col 0 = west edge"},
        "channels": {str(k): v for k, v in CHANNELS.items()},
        "rasterization_rule": "pixel = 1 iff its centre lies inside or on the boundary of a layer polygon",
        "overlap_policy": "channels are independent binary masks; overlaps allowed and counted",
        "road_buffer": road_buffer_metadata(sample),
        "feature_counts_in_square": {
            "road_segments": int(len(sample.road_centerlines)),
            "buildings": int(len(sample.buildings)),
            "green_polygons": int(len(sample.green)),
        },
        "statistics": layout_statistics(sample.layout),
        "processing_date_utc": processing_date,
    }


_KEEP_COLUMNS = {
    "road_centerlines": ["osmid", "highway", "name", "width", "lanes", "tunnel", "bridge", "oneway"],
    "road_polygons": ["osmid", "highway", "name", "width_m", "width_source"],
    "buildings": ["id", "building", "name", "height", "building:levels"],
    "green": ["id", "leisure", "landuse", "natural", "name", "access"],
}


def export_vectors(sample: ProcessedSample, path) -> None:
    """Save the clipped, projected vector layers (one GeoPackage, four layers)."""
    path.unlink(missing_ok=True)
    for layer, cols in _KEEP_COLUMNS.items():
        gdf: gpd.GeoDataFrame = getattr(sample, layer)
        keep = [c for c in cols if c in gdf.columns]
        out = gdf[keep + ["geometry"]].copy()
        for c in keep:  # lists / mixed types -> plain strings for GeoPackage
            if out[c].dtype == object:
                out[c] = out[c].map(lambda v: None if v is None or v != v else str(v))
        out.to_file(path, layer=layer, driver="GPKG")
