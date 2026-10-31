"""Urban blocks from a street graph (Milestone 8, Stage A, analysis version).

Rules (programmed, documented):
* road surface = every edge buffered by its class's median OSM width (as in Milestone 6);
* blocks = the tile minus the road surface, split into connected polygons; this uses road
  BOUNDARIES, not centrelines, and needs no closed cycles: dead ends simply cut into a block;
* a block touching the tile edge is OPEN (truncated by the 512 m window, so its true size is unknown);
* slivers below MIN_BLOCK_M2 (left between parallel carriageways, at junctions) are dropped.
"""

from __future__ import annotations

import numpy as np
import shapely

from urbanmorph.graphs.road_graph import TILE_M, road_polygons

MIN_BLOCK_M2 = 200.0
TILE = shapely.box(0, 0, TILE_M, TILE_M)


def extract_blocks(graph, class_widths) -> list[dict]:
    land = TILE.difference(road_polygons(graph, class_widths=class_widths)) if len(graph.edges) else TILE
    parts = [g for g in getattr(land, "geoms", [land]) if g.geom_type == "Polygon" and g.area >= MIN_BLOCK_M2]
    edge = TILE.exterior.buffer(0.5)
    out = []
    for i, p in enumerate(parts):
        rect = shapely.minimum_rotated_rectangle(p)
        out.append({"block_id": i, "polygon": p, "area_m2": p.area, "open": bool(p.intersects(edge)),
                    "rectangularity": p.area / rect.area if rect.area else np.nan,
                    "compactness": 4 * np.pi * p.area / p.length ** 2})
    return out


def block_of(points: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Index of the block containing each point (-1 if none)."""
    out = -np.ones(len(points), int)
    for b in blocks:
        inside = shapely.contains_xy(b["polygon"], points[:, 0], points[:, 1])
        out[inside & (out < 0)] = b["block_id"]
    return out
