"""Planar street graphs of the 512 m tiles, in tile-local metres (Milestone 5).

Coordinates: x east, y north, origin = south-west corner of the tile, so a tile is [0, 512]^2.

Extraction from the ``road_centerlines`` layer (OSMnx edges clipped to the tile, the same
vectors the road raster was made from):

1. Resample every line at equal spacing of at most ``STEP_M`` metres. Line endpoints
   (junctions, dead ends, points where the tile edge cuts a street) are always kept.
2. Snap all points greedily to node centres at least ``MERGE_M`` apart. Endpoints are snapped
   first, so junction and border positions are preserved. This (a) merges the two directions
   of a two-way street, which OSMnx stores twice, and (b) consolidates complex junctions,
   like the 10 m consolidation of Milestone 2. Greedy assignment to a fixed centre is NOT
   transitive, so long chains of nearby points cannot collapse into one node.
3. Edges join consecutive points of each line after snapping (self-loops and duplicates dropped).
4. ``border`` marks nodes within ``BORDER_M`` of the tile edge: streets cut by the window.

The generator uses the same ``MERGE_M`` to decide when a new step reaches an existing node,
so training data and generated graphs share one notion of "the same place".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString

TILE_M = 512.0
STEP_M = 30.0
MERGE_M = 8.0
BORDER_M = 1.0
ROAD_WIDTH_M = 7.0   # median OSM width in the training data (road_polygons.width_m), used to rasterise
TILE_KM2 = (TILE_M / 1000) ** 2


@dataclass
class RoadGraph:
    nodes: np.ndarray    # (N, 2) float, metres
    edges: np.ndarray    # (E, 2) int, i < j, unique
    border: np.ndarray   # (N,) bool

    @property
    def n(self) -> int:
        return len(self.nodes)

    def adjacency(self) -> list[list[int]]:
        adj = [[] for _ in range(self.n)]
        for a, b in self.edges:
            adj[a].append(int(b))
            adj[b].append(int(a))
        return adj

    def degree(self) -> np.ndarray:
        return np.bincount(self.edges.ravel(), minlength=self.n) if len(self.edges) else np.zeros(self.n, int)

    def segments(self) -> np.ndarray:
        """(E, 4) array x1, y1, x2, y2."""
        if not len(self.edges):
            return np.zeros((0, 4))
        return np.c_[self.nodes[self.edges[:, 0]], self.nodes[self.edges[:, 1]]]

    def transform(self, k: int) -> "RoadGraph":
        """One of the 8 symmetries of the square tile (rotation k%4 x 90 deg about the centre, mirrored if k>=4)."""
        p = self.nodes - TILE_M / 2
        if k >= 4:
            p = p * np.array([-1.0, 1.0])
        for _ in range(k % 4):
            p = np.c_[-p[:, 1], p[:, 0]]
        return RoadGraph(p + TILE_M / 2, self.edges.copy(), self.border.copy())


def is_border(p: np.ndarray, tol: float = BORDER_M) -> np.ndarray:
    p = np.atleast_2d(p)
    return np.minimum(p, TILE_M - p).min(axis=1) < tol


def _resample(coords: np.ndarray, step: float) -> np.ndarray:
    line = LineString(coords)
    n = max(1, int(np.ceil(line.length / step)))
    d = np.linspace(0, line.length, n + 1)
    return shapely.get_coordinates(shapely.line_interpolate_point(line, d))


def build_graph(lines: list[np.ndarray], step: float = STEP_M, merge: float = MERGE_M) -> RoadGraph:
    """Planar graph from polylines in tile-local metres (see module docstring)."""
    lines = [np.asarray(c, float)[:, :2] for c in lines if len(c) >= 2 and LineString(c).length > 0]
    pts = [_resample(c, step) for c in lines]
    if not pts:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool))
    allp = np.concatenate(pts)
    is_end = np.concatenate([np.r_[True, np.zeros(len(p) - 2, bool), True] if len(p) > 1 else np.array([True]) for p in pts])
    # priority: border endpoints, other endpoints, interior resampled points
    prio = np.where(is_end & is_border(allp), 0, np.where(is_end, 1, 2))
    order = np.argsort(prio, kind="stable")
    centres = np.zeros((len(allp), 2))
    n = 0
    assign = np.empty(len(allp), int)
    for i in order:
        if n:
            d = np.hypot(*(centres[:n] - allp[i]).T)
            j = int(d.argmin())
            if d[j] < merge:
                assign[i] = j
                continue
        centres[n] = allp[i]
        assign[i] = n
        n += 1
    edges, start = set(), 0
    for p in pts:
        ids = assign[start:start + len(p)]
        start += len(p)
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                edges.add((min(a, b), max(a, b)))
    nodes = centres[:n]
    e = np.array(sorted(edges), int).reshape(-1, 2)
    used = np.unique(e.ravel()) if len(e) else np.zeros(0, int)
    remap = -np.ones(n, int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used]
    return RoadGraph(nodes, remap[e], is_border(nodes) if len(nodes) else np.zeros(0, bool))


def load_tile_graph(gpkg_path, xmin: float, ymin: float) -> RoadGraph:
    import geopandas as gpd
    rc = gpd.read_file(gpkg_path, layer="road_centerlines")
    lines = []
    for geom in rc.geometry:
        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for g in parts:
            lines.append(np.asarray(g.coords)[:, :2] - [xmin, ymin])
    return build_graph(lines)


# ------------------------------------------------------------------ connectivity and metrics
def components(g: RoadGraph) -> np.ndarray:
    """Component label per node (union-find)."""
    parent = np.arange(g.n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in g.edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(g.n)], int)
    return np.unique(roots, return_inverse=True)[1] if g.n else roots


def graph_metrics(g: RoadGraph, min_component_m: float = 40.0) -> dict:
    """Per-tile statistics computed directly on the graph (no rasterisation).

    * components: connected pieces with >= ``min_component_m`` of street (10 px, like the raster
      metrics); interior_components: those with no border node, i.e. "floating" networks.
    * dead_end_ratio: degree-1 non-border nodes / (those + junctions of degree >= 3).
    * intersections_per_km2: non-border nodes of degree >= 3.
    * loops_per_km2: cycle rank E - V + C (independent cycles; for a planar graph = enclosed blocks).
    * mean_link_length_m: street length / number of links between junctions or ends.
    * orientation: the same length-weighted axial statistics as Milestone 2 (Boeing phi).
    """
    from urbanmorph.metrics.morphology import orientation_metrics
    nan = float("nan")
    if not len(g.edges):
        return {"street_length_km_per_km2": 0.0, "components": 0, "interior_components": 0, "dead_end_ratio": nan,
                "intersections_per_km2": 0.0, "loops_per_km2": 0.0, "mean_link_length_m": nan,
                "share_deg4_of_junctions": nan, "orientation_phi": nan, "orientation_entropy": nan,
                "grid_strength_r4": nan, "n_nodes": int(g.n)}
    seg = g.segments()
    seg_len = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    deg = g.degree()
    lab = components(g)
    comp_len = np.bincount(lab[g.edges[:, 0]], weights=seg_len, minlength=lab.max() + 1)
    comp_border = np.bincount(lab, weights=g.border.astype(float), minlength=lab.max() + 1) > 0
    big = comp_len >= min_component_m
    inner = ~g.border
    ends = int(((deg == 1) & inner).sum())
    junc = int(((deg >= 3) & inner).sum())
    links = max(1, len(g.edges) - int((deg == 2).sum()))
    om = orientation_metrics(seg)
    return {
        "street_length_km_per_km2": float(seg_len.sum() / 1000 / TILE_KM2),
        "components": int(big.sum()),
        "interior_components": int((big & ~comp_border).sum()),
        "dead_end_ratio": ends / (ends + junc) if ends + junc else nan,
        "intersections_per_km2": junc / TILE_KM2,
        "loops_per_km2": (len(g.edges) - g.n + int(lab.max() + 1)) / TILE_KM2,
        "mean_link_length_m": float(seg_len.sum() / links),
        "share_deg4_of_junctions": float(((deg >= 4) & inner).sum() / junc) if junc else nan,
        "orientation_phi": om["orientation_order_phi"],
        "orientation_entropy": om["orientation_entropy"],
        "grid_strength_r4": om["grid_strength_r4"],
        "n_nodes": int(g.n),
    }


def rasterize_graph(g: RoadGraph, width_m: float = ROAD_WIDTH_M, resolution: int = 128) -> np.ndarray:
    """Road mask with the Milestone 1 pixel-centre rule; every street gets the same width."""
    from urbanmorph.preprocessing.projection import SquareBounds
    from urbanmorph.preprocessing.rasterize import rasterize_geometries
    if not len(g.edges):
        return np.zeros((resolution, resolution), np.uint8)
    lines = MultiLineString([s.reshape(2, 2) for s in g.segments()])
    return rasterize_geometries([lines.buffer(width_m / 2)], SquareBounds(0, 0, TILE_M, TILE_M), resolution)
