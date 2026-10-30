"""Parameterised footprints of buildings and green spaces (Milestone 6).

Every polygon (tile-local metres) is described by
    anchor (x, y)        centre of its minimum rotated rectangle (MRR)
    theta                direction of the MRR's LONGER side (axial: theta and theta + pi are the same)
    length >= width      MRR side lengths
    radial r_1..r_20     shape: along 20 rays from the anchor (object frame: 16 rays 22.5 deg apart plus
                         4 rays aimed at the MRR corners), the polygon's extent as a fraction of the
                         MRR's extent in that direction (all 1 = exactly the rectangle; an L-shape has
                         small values towards its missing corner).
This is not a fixed template library: any rectangle, L, U, T, wedge or rounded shape that is
star-shaped from its MRR centre is covered approximately; ``representation_iou`` measures the loss.

For learning, the orientation is expressed RELATIVE to the nearest street (``phi`` = theta minus the
street's axis, folded to [-pi/2, pi/2)), so "parallel to the street" is the same value everywhere.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon

from urbanmorph.graphs.road_graph import TILE_M

N_UNIFORM = 16
N_RAYS = N_UNIFORM + 4
UNIFORM_ANGLES = 2 * np.pi * np.arange(N_UNIFORM) / N_UNIFORM


def ray_angles(length, width) -> np.ndarray:
    """16 uniform directions + the 4 corner directions of a length x width rectangle (object frame)."""
    c = np.arctan2(width, length)
    return np.r_[UNIFORM_ANGLES, c, np.pi - c, np.pi + c, 2 * np.pi - c]
MIN_BUILDING_M2 = 10.0
MIN_GREEN_M2 = 20.0
RATIO_MIN, RATIO_MAX = 0.05, 1.0


def fold_axial(a):
    """Angle folded to [-pi/2, pi/2) (axial data: a and a + pi are the same direction)."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def rect_extent(alpha, length, width):
    """Distance from the centre to the edge of a length x width rectangle along direction alpha."""
    c, s = np.abs(np.cos(alpha)), np.abs(np.sin(alpha))
    with np.errstate(divide="ignore"):
        return np.minimum(np.where(c > 1e-9, length / 2 / np.maximum(c, 1e-9), np.inf),
                          np.where(s > 1e-9, width / 2 / np.maximum(s, 1e-9), np.inf))


def mrr_params(poly) -> tuple[np.ndarray, float, float, float]:
    """(anchor, theta, length, width) of the minimum rotated rectangle."""
    rect = shapely.minimum_rotated_rectangle(poly)
    xy = np.asarray(rect.exterior.coords)[:4] if rect.geom_type == "Polygon" else np.asarray(poly.centroid.coords)
    if len(xy) < 4:
        b = poly.bounds
        return np.array(poly.centroid.coords[0]), 0.0, max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    e1, e2 = xy[1] - xy[0], xy[2] - xy[1]
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    main = e1 if l1 >= l2 else e2
    return xy.mean(0), float(np.arctan2(main[1], main[0])), float(max(l1, l2)), float(max(min(l1, l2), 0.5))


def encode(poly) -> dict:
    anchor, theta, L, W = mrr_params(poly)
    rays = ray_angles(L, W)
    reach = rect_extent(rays, L, W)
    ratios = np.empty(N_RAYS)
    for k, (a, R) in enumerate(zip(theta + rays, reach)):
        ray = LineString([anchor, anchor + 1.05 * R * np.array([np.cos(a), np.sin(a)])])
        hit = ray.intersection(poly)
        if hit.is_empty:
            ratios[k] = RATIO_MIN
            continue
        pts = shapely.get_coordinates(hit)
        ratios[k] = np.hypot(*(pts - anchor).T).max() / R
    return {"anchor": anchor, "theta": theta, "length": L, "width": W,
            "ratios": np.clip(ratios, RATIO_MIN, RATIO_MAX)}


def decode(anchor, theta, length, width, ratios) -> Polygon:
    """Polygon from parameters: the 20 ray end points joined in angular order (made valid)."""
    rays = ray_angles(length, width)
    r = np.clip(np.asarray(ratios, float), RATIO_MIN, RATIO_MAX) * rect_extent(rays, length, width)
    order = np.argsort(rays % (2 * np.pi), kind="stable")
    a = theta + rays[order]
    pts = np.asarray(anchor) + np.c_[r[order] * np.cos(a), r[order] * np.sin(a)]
    poly = shapely.make_valid(Polygon(pts))
    if poly.geom_type != "Polygon":
        parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
        poly = max(parts, key=lambda g: g.area) if parts else Polygon()
    return poly


def representation_iou(poly) -> float:
    e = encode(poly)
    q = decode(e["anchor"], e["theta"], e["length"], e["width"], e["ratios"])
    u = poly.union(q).area
    return float(poly.intersection(q).area / u) if u else 1.0


# ------------------------------------------------------------------ geometry relative to the streets
def street_axis_and_distance(points: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point: axis angle of the nearest street segment and the distance to it (metres)."""
    points = np.atleast_2d(points)
    if not len(segments):
        return np.zeros(len(points)), np.full(len(points), 999.0)
    A, B = segments[:, :2], segments[:, 2:]
    s = B - A
    ss = np.maximum((s ** 2).sum(1), 1e-9)
    lam = np.clip(((points[:, None, :] - A[None]) * s[None]).sum(-1) / ss[None], 0, 1)
    proj = A[None] + lam[..., None] * s[None]
    d = np.hypot(*(points[:, None, :] - proj).transpose(2, 0, 1))
    k = d.argmin(1)
    return np.arctan2(s[k, 1], s[k, 0]), d[np.arange(len(points)), k]


# ------------------------------------------------------------------ rasterisation (pixel-centre rule)
_C = (np.arange(128) + 0.5) * 4.0


def polygon_pixels(poly) -> np.ndarray:
    """Flat indices (row * 128 + col) of the 4 m pixels whose centre lies in the polygon (row 0 = north)."""
    if poly.is_empty:
        return np.zeros(0, int)
    x0, y0, x1, y1 = poly.bounds
    c0, c1 = max(0, int(np.floor(x0 / 4 - 0.5))), min(127, int(np.ceil(x1 / 4)))
    r0, r1 = max(0, int(np.floor((TILE_M - y1) / 4 - 0.5))), min(127, int(np.ceil((TILE_M - y0) / 4)))
    if c1 < c0 or r1 < r0:
        return np.zeros(0, int)
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    inside = shapely.intersects_xy(poly, _C[cc], TILE_M - _C[rr])
    return (rr[inside] * 128 + cc[inside]).astype(int)


def transform_polygon(poly, k: int):
    """The 8 symmetries of the tile, matching RoadGraph.transform (mirror x, then rotate k%4 x 90 deg)."""
    if k >= 4:
        poly = affinity.scale(poly, -1, 1, origin=(TILE_M / 2, TILE_M / 2))
    return affinity.rotate(poly, 90 * (k % 4), origin=(TILE_M / 2, TILE_M / 2))


# ------------------------------------------------------------------ per-tile metrics
def building_metrics(buildings: list, segments: np.ndarray) -> dict:
    """Per-tile statistics of building footprints (vectors)."""
    nan = float("nan")
    b = [p for p in buildings if not p.is_empty and p.area >= MIN_BUILDING_M2]
    if not b:
        return {"building_count": 0, "building_coverage": 0.0, "median_log_area": nan, "median_aspect": nan,
                "median_rectangularity": nan, "share_aligned_10deg": nan, "median_dist_road_m": nan,
                "clark_evans": nan}
    arr = np.array(b, dtype=object)
    area = shapely.area(arr)
    enc = [mrr_params(p) for p in b]
    anchors = np.array([e[0] for e in enc])
    theta = np.array([e[1] for e in enc])
    aspect = np.array([e[2] / e[3] for e in enc])
    rect = area / np.maximum(np.array([e[2] * e[3] for e in enc]), 1e-9)
    axis, dist = street_axis_and_distance(anchors, segments)
    phi = fold_axial(theta - axis)
    aligned = np.minimum(np.abs(phi), np.pi / 2 - np.abs(phi)) < np.radians(10)   # parallel OR perpendicular
    cov = shapely.area(shapely.union_all(arr).intersection(shapely.box(0, 0, TILE_M, TILE_M))) / TILE_M ** 2
    if len(anchors) >= 2:
        dd = np.hypot(*(anchors[:, None] - anchors[None]).transpose(2, 0, 1))
        np.fill_diagonal(dd, np.inf)
        ce = dd.min(1).mean() / (0.5 / np.sqrt(len(anchors) / TILE_M ** 2))
    else:
        ce = nan
    return {"building_count": len(b), "building_coverage": float(cov), "median_log_area": float(np.median(np.log(area))),
            "median_aspect": float(np.median(aspect)), "median_rectangularity": float(np.median(rect)),
            "share_aligned_10deg": float(aligned.mean()), "median_dist_road_m": float(np.median(dist)),
            "clark_evans": float(ce)}


def green_metrics(green: list) -> dict:
    g = [p for p in green if not p.is_empty and p.area >= MIN_GREEN_M2]
    if not g:
        return {"green_count": 0, "green_coverage": 0.0, "green_median_log_area": float("nan")}
    arr = np.array(g, dtype=object)
    cov = shapely.area(shapely.union_all(arr).intersection(shapely.box(0, 0, TILE_M, TILE_M))) / TILE_M ** 2
    return {"green_count": len(g), "green_coverage": float(cov),
            "green_median_log_area": float(np.median(np.log(shapely.area(arr))))}


def validity(buildings: list, green: list, road_surface) -> dict:
    """Geometric validity of a generated (or real) layout."""
    b = [p for p in buildings if not p.is_empty]
    arr = np.array(b, dtype=object)
    out = {"invalid_polygons": int(sum(not p.is_valid for p in b) + sum(not p.is_valid for p in green if not p.is_empty)),
           "outside_tile": int(sum(not shapely.box(-0.01, -0.01, TILE_M + 0.01, TILE_M + 0.01).contains(p) for p in b))}
    if len(b):
        on_road = shapely.area(shapely.intersection(arr, road_surface)) / shapely.area(arr) if not road_surface.is_empty else np.zeros(len(b))
        out["share_buildings_over_25pct_on_road"] = float((on_road > 0.25).mean())
        tree = shapely.STRtree(arr)
        pairs = tree.query(arr, predicate="intersects")
        pairs = pairs[:, pairs[0] < pairs[1]]
        if pairs.shape[1]:
            ov = shapely.area(shapely.intersection(arr[pairs[0]], arr[pairs[1]])) / np.minimum(
                shapely.area(arr[pairs[0]]), shapely.area(arr[pairs[1]]))
            out["overlapping_building_pairs_per_building"] = float((ov > 0.05).sum() / len(b))
        else:
            out["overlapping_building_pairs_per_building"] = 0.0
    else:
        out["share_buildings_over_25pct_on_road"] = float("nan")
        out["overlapping_building_pairs_per_building"] = float("nan")
    return out
