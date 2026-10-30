"""Coordinate reference systems: from lat/lon degrees to a metric grid.

Why project? Latitude/longitude are *angles* on an ellipsoid. One degree of
latitude is ~111 km everywhere, but one degree of longitude is ~111 km * cos(lat):
at Barcelona (41.4°N) that is only ~83.5 km. Treating (lon, lat) as Euclidean
(x, y) would therefore stretch everything east-west by ~33% and make "distance"
and "area" meaningless. A projected CRS such as UTM maps the curved surface to a
plane where 1 unit = 1 metre with very low distortion inside its zone.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

from urbanmorph.config import AreaSpec


@dataclass(frozen=True)
class SquareBounds:
    """Axis-aligned square in projected metres: (xmin, ymin, xmax, ymax)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def polygon(self):
        return box(*self.as_tuple())


def project_center(area: AreaSpec) -> tuple[float, float]:
    """Centre (lat, lon) -> projected (x, y) metres (or the exact projected centre if given)."""
    if area.projected_center is not None:
        return area.projected_center
    t = Transformer.from_crs("EPSG:4326", area.projected_crs, always_xy=True)
    return t.transform(area.center_lon, area.center_lat)


def square_bounds(area: AreaSpec) -> SquareBounds:
    cx, cy = project_center(area)
    h = area.size_m / 2
    return SquareBounds(cx - h, cy - h, cx + h, cy + h)


def wgs84_corners(area: AreaSpec, bounds: SquareBounds) -> dict:
    """Lat/lon of the square's corners (it is a slightly rotated quad in lat/lon)."""
    t = Transformer.from_crs(area.projected_crs, "EPSG:4326", always_xy=True)
    corners = {
        "south_west": (bounds.xmin, bounds.ymin),
        "south_east": (bounds.xmax, bounds.ymin),
        "north_east": (bounds.xmax, bounds.ymax),
        "north_west": (bounds.xmin, bounds.ymax),
    }
    out = {}
    for name, (x, y) in corners.items():
        lon, lat = t.transform(x, y)
        out[name] = {"lat": round(lat, 7), "lon": round(lon, 7)}
    lats = [c["lat"] for c in out.values()]
    lons = [c["lon"] for c in out.values()]
    out["bbox"] = {"south": min(lats), "west": min(lons), "north": max(lats), "east": max(lons)}
    return out


def to_metric_and_clip(
    gdf: gpd.GeoDataFrame, crs: str, bounds: SquareBounds
) -> gpd.GeoDataFrame:
    """Reproject a layer to ``crs`` and clip it to the sample square."""
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    projected = gdf.to_crs(crs)
    # make_valid fixes self-intersecting OSM polygons before clipping.
    projected = projected.set_geometry(projected.geometry.make_valid())
    clipped = gpd.clip(projected, bounds.polygon())
    return clipped[~clipped.geometry.is_empty].copy()
