"""Urban-morphology metrics for one 512 m tile.

Coverage ratios come from the raster (what a model will see) and, for
buildings and green, are cross-checked against exact vector areas. Network
metrics use the OSM street GRAPH, never raster pixels. Building statistics use
the original (unclipped) footprints.

Conventions
-----------
* Bearings are measured clockwise from UTM grid north (differs from true
  north by the grid convergence, < 1.5 deg for our cities; negligible for 10 deg bins).
* Streets are undirected, so bearings are folded to [0, 180): a street
  running 30 deg is the same as one running 210 deg.
"""

from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import shapely

from urbanmorph.config import CHANNELS
from urbanmorph.preprocessing.projection import SquareBounds


# ---------------------------------------------------------------- coverage
def raster_coverage(layout: np.ndarray) -> dict:
    return {f"{CHANNELS[i]}_coverage": float(layout[i].mean()) for i in range(layout.shape[0])}


def vector_coverage(polygons: gpd.GeoDataFrame, bounds: SquareBounds) -> float:
    """Exact area fraction of the tile covered by the union of (projected) polygons."""
    if polygons.empty:
        return 0.0
    merged = shapely.union_all(polygons.geometry.values)
    return float(merged.intersection(bounds.polygon()).area / (bounds.width * bounds.height))


# ---------------------------------------------------------------- buildings
def building_metrics(buildings_projected: gpd.GeoDataFrame, bounds: SquareBounds) -> dict:
    """Count and footprint-area statistics of buildings whose CENTROID lies in the tile.

    The centroid rule assigns each building to exactly one tile, so buildings
    crossing the edge are neither lost nor double counted across tiles. Areas
    are of the full footprint (not clipped). Heights are deliberately not
    inferred.
    """
    tile_ha = bounds.width * bounds.height / 1e4
    empty = {"building_count": 0, "building_density_per_ha": 0.0, "building_area_mean_m2": np.nan,
             "building_area_median_m2": np.nan, "building_area_p10_m2": np.nan,
             "building_area_p90_m2": np.nan, "building_area_hist_log10": []}
    if buildings_projected.empty:
        return empty
    polys = buildings_projected[buildings_projected.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    c = polys.geometry.centroid
    inside = polys[(c.x >= bounds.xmin) & (c.x < bounds.xmax) & (c.y >= bounds.ymin) & (c.y < bounds.ymax)]
    if inside.empty:
        return empty
    a = inside.geometry.area.to_numpy()
    hist, _ = np.histogram(np.log10(np.clip(a, 1, None)), bins=LOG10_AREA_BINS)
    return {
        "building_count": int(len(a)),
        "building_density_per_ha": float(len(a) / tile_ha),
        "building_area_mean_m2": float(a.mean()),
        "building_area_median_m2": float(np.median(a)),
        "building_area_p10_m2": float(np.percentile(a, 10)),
        "building_area_p90_m2": float(np.percentile(a, 90)),
        "building_area_hist_log10": hist.tolist(),
    }


# 0.25-decade bins from 1 m2 to 100,000 m2.
LOG10_AREA_BINS = np.arange(0, 5.01, 0.25)


# ---------------------------------------------------------------- network
def project_graph(graph: nx.MultiDiGraph, crs: str) -> nx.MultiDiGraph:
    return ox.project_graph(graph, to_crs=crs)


def intersection_metrics(graph_projected: nx.MultiDiGraph, bounds: SquareBounds, tolerance_m: float) -> dict:
    """Intersections (nodes where >= 3 street segments meet) per km2 inside the tile.

    * ``street_count`` counts physical streets at a node, ignoring direction,
      so a two-way street is counted once.
    * OSMnx simplification has already removed degree-2 "pass-through" nodes.
    * Complex junctions (dual carriageways, roundabouts) create several nodes
      for one real-world intersection. ``consolidate_intersections`` buffers
      each node by ``tolerance_m`` and merges overlapping buffers (nodes closer
      than 2 x tolerance), after removing dead ends.
    * Boundary: only nodes inside the tile are counted; the graph was downloaded
      with a margin so these nodes keep their true degree.
    """
    area_km2 = bounds.width * bounds.height / 1e6

    def count(G) -> int:
        if G.number_of_nodes() == 0:
            return 0
        sc = ox.stats.count_streets_per_node(G)
        n = 0
        for node, data in G.nodes(data=True):
            if sc.get(node, 0) >= 3 and bounds.xmin <= data["x"] < bounds.xmax and bounds.ymin <= data["y"] < bounds.ymax:
                n += 1
        return n

    raw = count(graph_projected)
    if tolerance_m > 0 and graph_projected.number_of_nodes() > 1:
        consolidated = ox.consolidate_intersections(
            graph_projected, tolerance=tolerance_m, rebuild_graph=True, dead_ends=False, reconnect_edges=True
        )
        merged = count(consolidated)
    else:
        merged = raw
    return {
        "intersections_raw": raw,
        "intersections": merged,
        "intersection_density_per_km2": merged / area_km2,
    }


def street_segments(graph_projected: nx.MultiDiGraph, bounds: SquareBounds) -> np.ndarray:
    """(n, 4) array of straight segments x0, y0, x1, y1 of undirected streets clipped to the tile."""
    if graph_projected.number_of_edges() == 0:
        return np.zeros((0, 4))
    undirected = ox.convert.to_undirected(graph_projected)  # removes u->v / v->u duplicates
    edges = ox.graph_to_gdfs(undirected, nodes=False, fill_edge_geometry=True)
    clipped = shapely.intersection(edges.geometry.values, bounds.polygon())
    segs = []
    for g in clipped:
        for line in getattr(g, "geoms", [g]):
            if line.is_empty or line.geom_type != "LineString":
                continue
            c = np.asarray(line.coords)[:, :2]
            segs.append(np.hstack([c[:-1], c[1:]]))
    return np.vstack(segs) if segs else np.zeros((0, 4))


def orientation_metrics(segments: np.ndarray, n_bins: int = 18) -> dict:
    """Length-weighted street-orientation statistics.

    * Histogram: ``n_bins`` bins over [0, 180), centred on 0, 10, ..., 170 deg
      (so N-S streets fall in the bin centred on 0).
    * Entropy H = -sum p ln p. H_max = ln(n_bins) (all directions equally
      common); a perfect orthogonal grid fills 2 axial bins, H_grid = ln 2.
    * Order phi = 1 - ((H - H_grid) / (H_max - H_grid))^2  (Boeing 2019):
      1 = perfect grid, 0 = uniform. Equivalent to Boeing's 36-bin version.
    * Dominant axis: circular mean of DOUBLED angles (axial data), in [0, 180).
      R2 in [0, 1] measures how strongly one axis dominates. For an orthogonal
      grid, the two axes cancel (R2 ~ 0), so we also report the grid orientation
      from QUADRUPLED angles, in [0, 90), with its strength R4.
    """
    if len(segments) == 0:
        nan = float("nan")
        return {"street_length_m": 0.0, "orientation_entropy": nan, "orientation_order_phi": nan,
                "dominant_bearing_deg": nan, "dominant_axis_strength_r2": nan,
                "grid_orientation_deg": nan, "grid_strength_r4": nan, "orientation_hist": [0.0] * n_bins}
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    w = np.hypot(dx, dy)
    keep = w > 0
    dx, dy, w = dx[keep], dy[keep], w[keep]
    theta = np.degrees(np.arctan2(dx, dy)) % 180.0  # clockwise from north, folded

    width = 180.0 / n_bins
    idx = (((theta + width / 2) % 180.0) // width).astype(int)
    hist = np.bincount(idx, weights=w, minlength=n_bins)
    p = hist / hist.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    h_max, h_grid = math.log(n_bins), math.log(2)
    phi = 1 - ((H - h_grid) / (h_max - h_grid)) ** 2

    def circ(k: int):
        r = np.radians(k * theta)
        s, c = (w * np.sin(r)).sum(), (w * np.cos(r)).sum()
        period = 360.0 / k
        angle = (math.degrees(math.atan2(s, c)) / k) % period
        if period - angle < 1e-6:  # floating-point noise just below the wrap-around is 0
            angle = 0.0
        return angle, float(math.hypot(s, c) / w.sum())

    dom, r2 = circ(2)
    grid, r4 = circ(4)
    return {
        "street_length_m": float(w.sum()),
        "orientation_entropy": H,
        "orientation_order_phi": float(phi),
        "dominant_bearing_deg": float(dom),
        "dominant_axis_strength_r2": r2,
        "grid_orientation_deg": float(grid),
        "grid_strength_r4": r4,
        "orientation_hist": (p).round(5).tolist(),
    }
