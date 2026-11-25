"""Milestone 10: the canonical raster canvas of one city block (BlockGen, docs/M10_BLOCK_GENERATION.md §4).

Frame: the block's minimum rotated rectangle; x along its long side, origin at its centre. A window is
WIN_PX x WIN_PX pixels of PX_M metres (256 m) centred at (wx, wy) in that frame. Blocks larger than one
window get several windows on a grid with stride STRIDE_M (overlap 64 m), generated in order by outpainting.
Pixel (row, col) covers window coordinates x = -128 + 2 col .. +2, y = 128 - 2 row .. -2.
"""

from __future__ import annotations

import numpy as np
import shapely
from rasterio import features
from rasterio.transform import Affine
from shapely import affinity

PX_M = 2.0
WIN_PX = 128
HALF_M = WIN_PX * PX_M / 2          # 128 m
STRIDE_M = 192.0
MIN_AREA_M2 = 10.0
PIX = Affine(PX_M, 0, -HALF_M, 0, -PX_M, HALF_M)


def block_frame(poly) -> dict:
    """Centre, angle (radians, long side) and half-sizes of the block's minimum rotated rectangle."""
    rect = np.asarray(shapely.minimum_rotated_rectangle(poly).exterior.coords)[:4]
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    long_, short = (e1, e2) if np.hypot(*e1) >= np.hypot(*e2) else (e2, e1)
    return {"cx": float(rect.mean(0)[0]), "cy": float(rect.mean(0)[1]), "theta": float(np.arctan2(long_[1], long_[0])),
            "hl": float(np.hypot(*long_) / 2), "hs": float(np.hypot(*short) / 2)}


def windows(fr: dict) -> list[tuple[float, float]]:
    """Window centres (frame coordinates) covering the block rectangle."""
    def axis(h):
        if h <= HALF_M - 4:
            return [0.0]
        n = int(np.ceil((2 * h - 2 * HALF_M) / STRIDE_M)) + 1
        return list(np.linspace(-h + HALF_M - 4, h - HALF_M + 4, n))
    return [(x, y) for x in axis(fr["hl"]) for y in axis(fr["hs"])]


def _world_to_win(fr, win):
    c, s = np.cos(-fr["theta"]), np.sin(-fr["theta"])
    # p_win = R(-theta) (p - centre) - win
    tx = -(c * fr["cx"] - s * fr["cy"]) - win[0]
    ty = -(s * fr["cx"] + c * fr["cy"]) - win[1]
    return [c, -s, s, c, tx, ty]


def _win_to_world(fr, win):
    c, s = np.cos(fr["theta"]), np.sin(fr["theta"])
    # p = R(theta) (p_win + win) + centre
    return [c, -s, s, c, c * win[0] - s * win[1] + fr["cx"], s * win[0] + c * win[1] + fr["cy"]]


def to_canvas(geoms, fr, win) -> np.ndarray:
    """(WIN_PX, WIN_PX) uint8 mask of the polygons (world coordinates) in the window."""
    m = _world_to_win(fr, win)
    shp = [(affinity.affine_transform(g, m), 1) for g in geoms if g is not None and not g.is_empty]
    if not shp:
        return np.zeros((WIN_PX, WIN_PX), np.uint8)
    return features.rasterize(shp, out_shape=(WIN_PX, WIN_PX), transform=PIX, fill=0, dtype="uint8", all_touched=False)


def from_canvas(mask: np.ndarray, fr, win, clip=None, simplify_m: float = 1.0) -> list:
    """Polygons (world coordinates) of the connected components of ``mask``, clipped to ``clip``."""
    m = _win_to_world(fr, win)
    out = []
    for geom, v in features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=PIX, connectivity=4):
        p = affinity.affine_transform(shapely.geometry.shape(geom), m).simplify(simplify_m)
        if clip is not None:
            p = p.intersection(clip)
        for q in getattr(p, "geoms", [p]):
            if q.geom_type == "Polygon" and q.area >= MIN_AREA_M2:
                out.append(q)
    return out


def window_region(fr, win):
    """The window's square in world coordinates."""
    sq = shapely.box(-HALF_M, -HALF_M, HALF_M, HALF_M)
    return affinity.affine_transform(sq, _win_to_world(fr, win))
