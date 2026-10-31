"""Milestone 9: standard building footprint templates (rectangle, L, T / wing, Z, U, H, cross).

A template is a polygon in the unit square (x = along the street / the lot's long axis, y = away from
it). A building is the template STRETCHED to its envelope (width x depth); it is never clipped. Cut sizes
a, b in CUTS (fraction of the side), every orientation: 1 + 16 x (4 L + 4 T + 4 U + 2 Z + 2 H + 1 cross)
classes before dropping degenerate ones. All corners are right angles, so no template is acute.
Fit on real footprints (scripts/m9_checks.py): 82.7 % of validation footprints at IoU >= 0.9.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import box

CUTS = (0.15, 0.25, 0.35, 0.45)
TYPES = ("rectangle", "L", "T / wing", "Z", "U", "H", "cross")
R = 48
_xy = (np.arange(R) + 0.5) / R
_GX, _GY = np.meshgrid(_xy, _xy)


def _build():
    U = box(0, 0, 1, 1)
    out = [("rectangle", (), U)]
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def corner(c, a, b):
        x0, y0 = c
        return box(1 - a if x0 else 0, 1 - b if y0 else 0, 1 if x0 else a, 1 if y0 else b)

    def notch(side, a, b):   # side 0 bottom (street), 1 right, 2 top (rear), 3 left; a = width, b = depth
        lo, hi = 0.5 - a / 2, 0.5 + a / 2
        return [box(lo, 0, hi, b), box(1 - b, lo, 1, hi), box(lo, 1 - b, hi, 1), box(0, lo, b, hi)][side]

    for a, b in product(CUTS, CUTS):
        for i, c in enumerate(corners):
            out.append(("L", (a, b, i), U.difference(corner(c, a, b))))
        for i in range(4):
            c1, c2 = corners[i], corners[(i + 1) % 4]
            out.append(("T / wing", (a, b, i), U.difference(corner(c1, a, b)).difference(corner(c2, a, b))))
            out.append(("U", (a, b, i), U.difference(notch(i, a, b))))
        for i in range(2):
            out.append(("Z", (a, b, i), U.difference(corner(corners[i], a, b)).difference(corner(corners[i + 2], a, b))))
            out.append(("H", (a, b, i), U.difference(notch(i, a, b)).difference(notch(i + 2, a, b))))
        p = U
        for c in corners:
            p = p.difference(corner(c, a, b))
        out.append(("cross", (a, b), p))
    return [(t, prm, shapely.normalize(p)) for t, prm, p in out if p.geom_type == "Polygon" and p.area > 0.3]


TEMPLATES = _build()
N_TEMPLATES = len(TEMPLATES)
TYPE_OF = np.array([TYPES.index(t) for t, _, _ in TEMPLATES])
FILL = np.array([p.area for _, _, p in TEMPLATES])          # share of the envelope rectangle covered


def mask(poly) -> np.ndarray:
    """Boolean R x R mask (flattened) of a polygon in the unit square."""
    return shapely.contains_xy(poly, _GX.ravel(), _GY.ravel())


_MASKS = None


def best_template(unit_poly) -> tuple[int, float]:
    """Index and IoU of the template that best matches a footprint normalised to its unit square."""
    global _MASKS
    if _MASKS is None:
        _MASKS = np.stack([mask(p) for _, _, p in TEMPLATES])
    m = mask(unit_poly)
    iou = (_MASKS & m).sum(1) / np.maximum((_MASKS | m).sum(1), 1)
    k = int(iou.argmax())
    return k, float(iou[k])


def to_unit(poly, origin, t, n):
    """Polygon in the frame (origin, x = t, y = n) normalised by its axis-aligned extent -> unit square.
    Returns (unit polygon, width, depth, x0, y0) with (x0, y0) the extent's lower corner in the frame."""
    c = np.asarray(poly.exterior.coords) - origin
    xy = np.c_[c @ t, c @ n]
    x0, y0 = xy.min(0)
    w, d = np.ptp(xy, 0)
    u = shapely.Polygon((xy - [x0, y0]) / [max(w, 1e-6), max(d, 1e-6)])
    return u, float(w), float(d), float(x0), float(y0)


def place(k, origin, t, n, x0, y0, w, d):
    """Template k stretched to w x d with its lower corner at (x0, y0) of the frame (origin, t, n)."""
    p = affinity.scale(TEMPLATES[k][2], w, d, origin=(0, 0))
    p = affinity.translate(p, x0, y0)
    return affinity.affine_transform(p, [t[0], n[0], t[1], n[1], origin[0], origin[1]])
