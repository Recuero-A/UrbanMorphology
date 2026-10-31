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


# Street classes (Milestone 6), from the OSM highway tag; a list-valued tag takes its most major class.
CLASS_NAMES = ("major", "mid", "local", "minor")
_CLASS_OF = {"motorway": 0, "trunk": 0, "primary": 0, "motorway_link": 0, "trunk_link": 0, "primary_link": 0,
             "secondary": 1, "tertiary": 1, "secondary_link": 1, "tertiary_link": 1,
             "residential": 2, "unclassified": 2, "living_street": 2, "road": 2}


def highway_class(tag) -> int:
    """0 major, 1 mid, 2 local, 3 minor (pedestrian and everything else)."""
    import ast
    vals = tag
    if isinstance(tag, str) and tag.startswith("["):
        vals = ast.literal_eval(tag)
    if isinstance(vals, str):
        vals = [vals]
    return min((_CLASS_OF.get(str(v), 3) for v in vals), default=3)


@dataclass
class RoadGraph:
    nodes: np.ndarray    # (N, 2) float, metres
    edges: np.ndarray    # (E, 2) int, i < j, unique
    border: np.ndarray   # (N,) bool
    edge_class: np.ndarray | None = None   # (E,) int in 0..3 (Milestone 6); None for Milestone 5 graphs

    def classes(self) -> np.ndarray:
        ec = getattr(self, "edge_class", None)
        return ec if ec is not None else np.full(len(self.edges), 2)

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
        ec = getattr(self, "edge_class", None)
        return RoadGraph(p + TILE_M / 2, self.edges.copy(), self.border.copy(), None if ec is None else ec.copy())


def subgraph(g: RoadGraph, edge_mask: np.ndarray) -> RoadGraph:
    """The edges in ``edge_mask`` and only the nodes they use (Milestone 10: the major + mid network)."""
    e = g.edges[np.asarray(edge_mask, bool)]
    used = np.unique(e.ravel()) if len(e) else np.zeros(0, int)
    remap = -np.ones(g.n, int)
    remap[used] = np.arange(len(used))
    ec = getattr(g, "edge_class", None)
    return RoadGraph(g.nodes[used], remap[e] if len(e) else e.reshape(0, 2), g.border[used],
                     None if ec is None else ec[np.asarray(edge_mask, bool)])


def is_border(p: np.ndarray, tol: float = BORDER_M) -> np.ndarray:
    p = np.atleast_2d(p)
    return np.minimum(p, TILE_M - p).min(axis=1) < tol


def _resample(coords: np.ndarray, step: float) -> np.ndarray:
    line = LineString(coords)
    n = max(1, int(np.ceil(line.length / step)))
    d = np.linspace(0, line.length, n + 1)
    return shapely.get_coordinates(shapely.line_interpolate_point(line, d))


def build_graph(lines: list[np.ndarray], step: float = STEP_M, merge: float = MERGE_M,
                classes: list[int] | None = None) -> RoadGraph:
    """Planar graph from polylines in tile-local metres (see module docstring).

    ``classes`` (one street class per line) gives every edge the most major class of the lines
    that produced it; without it the graph has no classes (Milestone 5)."""
    keep = [i for i, c in enumerate(lines) if len(c) >= 2 and LineString(c).length > 0]
    lines = [np.asarray(lines[i], float)[:, :2] for i in keep]
    cls_in = [classes[i] for i in keep] if classes is not None else [2] * len(lines)
    pts = [_resample(c, step) for c in lines]
    if not pts:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool),
                         np.zeros(0, int) if classes is not None else None)
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
    edges, start = {}, 0
    for p, c in zip(pts, cls_in):
        ids = assign[start:start + len(p)]
        start += len(p)
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                k = (min(a, b), max(a, b))
                edges[k] = min(edges.get(k, 3), c)
    nodes = centres[:n]
    keys = sorted(edges)
    e = np.array(keys, int).reshape(-1, 2)
    ec = np.array([edges[k] for k in keys], int)
    used = np.unique(e.ravel()) if len(e) else np.zeros(0, int)
    remap = -np.ones(n, int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used]
    return RoadGraph(nodes, remap[e], is_border(nodes) if len(nodes) else np.zeros(0, bool),
                     ec if classes is not None else None)


def load_tile_graph(gpkg_path, xmin: float, ymin: float, with_classes: bool = False) -> RoadGraph:
    import geopandas as gpd
    rc = gpd.read_file(gpkg_path, layer="road_centerlines")
    lines, classes = [], []
    tags = rc["highway"] if "highway" in rc else ["unclassified"] * len(rc)
    for geom, tag in zip(rc.geometry, tags):
        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for g in parts:
            lines.append(np.asarray(g.coords)[:, :2] - [xmin, ymin])
            classes.append(highway_class(tag))
    return build_graph(lines, classes=classes if with_classes else None)


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


def junction_angle_deviations(g: RoadGraph) -> np.ndarray:
    """For every angular gap between consecutive streets at a non-border junction (degree >= 3):
    its distance in degrees from the nearest of 90 and 180 deg (0 = a right-angle / straight-through
    junction). Descriptive only; nothing in the model prefers these angles."""
    out = []
    adj = g.adjacency()
    for u, nb in enumerate(adj):
        if len(nb) >= 3 and not g.border[u]:
            ang = np.sort([np.arctan2(*(g.nodes[w] - g.nodes[u])[::-1]) for w in nb])
            gaps = np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]))
            out += list(np.minimum(np.abs(gaps - 90), np.abs(gaps - 180)))
    return np.array(out)


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
                "grid_strength_r4": nan, "n_nodes": int(g.n), "junction_angle_dev_deg": nan}
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
        "junction_angle_dev_deg": float(np.median(dev)) if len(dev := junction_angle_deviations(g)) else nan,
    }


def road_polygons(g: RoadGraph, width_m: float = ROAD_WIDTH_M, class_widths=None):
    """Road surface: every edge buffered by half its width (one width, or a width per street class)."""
    if not len(g.edges):
        return shapely.Polygon()
    seg = g.segments()
    if class_widths is None:
        return MultiLineString([s.reshape(2, 2) for s in seg]).buffer(width_m / 2)
    cls = g.classes()
    parts = [MultiLineString([s.reshape(2, 2) for s in seg[cls == c]]).buffer(class_widths[c] / 2)
             for c in np.unique(cls)]
    return shapely.union_all(parts)


def rasterize_graph(g: RoadGraph, width_m: float = ROAD_WIDTH_M, resolution: int = 128, class_widths=None) -> np.ndarray:
    """Road mask with the Milestone 1 pixel-centre rule (one width, or a width per street class)."""
    from urbanmorph.preprocessing.projection import SquareBounds
    from urbanmorph.preprocessing.rasterize import rasterize_geometries
    if not len(g.edges):
        return np.zeros((resolution, resolution), np.uint8)
    return rasterize_geometries([road_polygons(g, width_m, class_widths)], SquareBounds(0, 0, TILE_M, TILE_M), resolution)
