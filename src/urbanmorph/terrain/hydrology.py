"""Hydrology on a heightfield (Milestone 12): depression filling, D8 flow, drainage area, rivers and lakes.

* ``priority_flood``: Barnes, Lehman & Mulla (2014) priority-flood with an epsilon gradient, so that every cell
  drains to the grid edge (or the sea). The filled height minus the original is the depression depth.
* ``d8``: steepest-descent direction to one of 8 neighbours on the filled surface; ``drainage_area`` accumulates
  cell areas downstream in order of decreasing height.
* Rivers are cells whose drainage area exceeds a threshold. Width follows hydraulic geometry, w = a * (A / 1 km^2)^b
  (Leopold & Maddock 1953 relate width to discharge; discharge is taken proportional to drainage area). ``a`` and
  ``b`` are parameters (calibrated to OSM river widths in docs/M12_PROGRESS_LOG.md); no physical discharge is
  claimed.
* Lakes are depressions that exceed BOTH a minimum area and a minimum depth; smaller depressions are treated as
  noise and filled. A lake surface is flat, at the depression's spill height.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(z: np.ndarray, sea: np.ndarray | None = None, eps: float = 1e-4) -> np.ndarray:
    """Filled surface (every cell has a downhill path to the edge or the sea)."""
    H, W = z.shape
    filled = z.astype(float).copy()
    done = np.zeros((H, W), bool)
    pq = []
    edge = np.zeros((H, W), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    if sea is not None:
        edge |= sea
    for r, c in zip(*np.nonzero(edge)):
        heapq.heappush(pq, (filled[r, c], r, c))
        done[r, c] = True
    while pq:
        h, r, c = heapq.heappop(pq)
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not done[rr, cc]:
                done[rr, cc] = True
                filled[rr, cc] = max(filled[rr, cc], h + eps)
                heapq.heappush(pq, (filled[rr, cc], rr, cc))
    return filled


def d8(filled: np.ndarray) -> np.ndarray:
    """Index 0..7 into NB of the steepest downhill neighbour, -1 at outlets."""
    H, W = filled.shape
    best = np.zeros((H, W))
    out = np.full((H, W), -1, int)
    pad = np.pad(filled, 1, constant_values=np.inf)
    for k, (dr, dc) in enumerate(NB):
        nb = pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        drop = (filled - nb) / np.hypot(dr, dc)
        m = drop > best
        best[m], out[m] = drop[m], k
    return out


def drainage_area(filled: np.ndarray, flow: np.ndarray, cell_area: float) -> np.ndarray:
    H, W = filled.shape
    acc = np.full(H * W, cell_area)
    order = np.argsort(-filled, axis=None)
    dr = np.array([d[0] for d in NB])
    dc = np.array([d[1] for d in NB])
    f = flow.ravel()
    for i in order:
        k = f[i]
        if k >= 0:
            r, c = divmod(i, W)
            j = (r + dr[k]) * W + (c + dc[k])
            acc[j] += acc[i]
    return acc.reshape(H, W)


def river_mask(area: np.ndarray, res: float, threshold_km2: float, a: float, b: float, max_width: float = 400.0):
    """(mask, width per river cell) from drainage area; the channel is painted with its local width."""
    centre = area >= threshold_km2 * 1e6
    width = np.where(centre, np.clip(a * (area / 1e6) ** b, res, max_width), 0.0)
    if not centre.any():
        return np.zeros_like(centre), width
    dist, (ir, ic) = ndimage.distance_transform_edt(~centre, return_indices=True)
    return dist * res <= width[ir, ic] / 2, width


def lakes(z: np.ndarray, filled: np.ndarray, res: float, min_area_m2: float, min_depth_m: float):
    """(lake mask, lake surface height) for depressions large and deep enough; others are left as land."""
    depth = filled - z
    lab, n = ndimage.label(depth > 0.05)
    mask = np.zeros(z.shape, bool)
    surface = np.full(z.shape, np.nan)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() * res * res >= min_area_m2 and depth[m].max() >= min_depth_m:
            mask |= m
            surface[m] = filled[m].max()                          # flat water at the spill level
    return mask, surface
