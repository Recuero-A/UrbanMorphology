"""Rasterization: vector polygons -> binary pixel grid.

Rule (deterministic, same as GDAL/rasterio's default ``all_touched=False``):

    A pixel is 1 if and only if its CENTRE lies inside or on the boundary
    of at least one polygon of the layer.

Grid convention (standard image layout):
    * row 0 is the NORTH edge, rows increase southwards;
    * column 0 is the WEST edge, columns increase eastwards;
    * pixel (r, c) has centre  x = xmin + (c + 0.5) * px,
                               y = ymax - (r + 0.5) * px.

Consequence: features narrower than ~one pixel (4 m) may vanish if they happen
to miss every pixel centre. That is a real, documented information loss.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from urbanmorph.preprocessing.projection import SquareBounds


def pixel_centers(bounds: SquareBounds, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) arrays of shape (resolution, resolution) with pixel-centre coordinates."""
    px_w = bounds.width / resolution
    px_h = bounds.height / resolution
    xs = bounds.xmin + (np.arange(resolution) + 0.5) * px_w
    ys = bounds.ymax - (np.arange(resolution) + 0.5) * px_h
    return np.meshgrid(xs, ys)  # X varies along columns, Y along rows


def rasterize_geometries(
    geometries, bounds: SquareBounds, resolution: int
) -> np.ndarray:
    """Rasterize an iterable of (multi)polygons into a uint8 {0,1} mask."""
    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((resolution, resolution), dtype=np.uint8)
    merged: BaseGeometry = shapely.union_all(geoms)
    shapely.prepare(merged)  # builds a spatial index -> fast repeated point tests
    X, Y = pixel_centers(bounds, resolution)
    inside = shapely.intersects_xy(merged, X, Y)  # True inside OR on boundary
    return inside.astype(np.uint8)
