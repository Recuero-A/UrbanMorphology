"""Terrain inputs for the turtle (Milestone 12): rasters in the tile frame, their symmetries, and rotated crops.

A tile's terrain is a set of rasters on the 4 m grid of the tile plus a 256 m margin (256 x 256, row 0 = north;
scripts/m12_prepare_data.py). ``transform_raster`` applies the k-th of the 8 square symmetries exactly as
``RoadGraph.transform`` does to the street graph (mirror x when k >= 4, then k % 4 counter-clockwise quarter turns
about the tile centre), so streets and terrain stay aligned under augmentation (tests/test_terrain_model.py).

``terrain_crops`` samples the SAME rotated 48 x 48 grid as ``sequence.crops_turtle`` (heading up, 4 m pixels).
Channels, by name:
* ``elev``  (z - z at the node) / elev_scale, clipped to [-3, 3]: uphill / downhill in the turtle frame;
* ``slope`` |grad z| / slope_scale, clipped to [0, 4];
* ``water`` 1 on rivers, lakes and sea;
* ``valid`` 1 where a DEM exists (always appended when ``elev`` or ``slope`` is used: Japanese tiles have water but
  no DEM, and the model must be able to tell "flat" from "unknown").
The terrain gradient vector is not a separate channel: it is implied by ``elev`` on a rotated grid, which is how
it is rotated into the local frame (avoids duplicate information; spec §8.1).
"""

from __future__ import annotations

import numpy as np

from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.graphs.sequence import HALF, PX, _LOCAL

MARGIN_PX = 64                                       # 256 m margin at 4 m
CHANNELS = ("elev", "slope", "water")


def transform_raster(a: np.ndarray, k: int) -> np.ndarray:
    if k >= 4:
        a = a[:, ::-1]
    return np.rot90(a, k % 4).copy()


def channel_list(names) -> list[str]:
    names = [n for n in CHANNELS if n in names]
    return names + (["valid"] if ("elev" in names or "slope" in names) else [])


def tile_layers(d: dict, k: int, names, norm: dict) -> dict:
    """Transformed rasters needed for ``names`` (elevation kept raw: the crop subtracts the node height)."""
    out = {}
    elev, valid = transform_raster(d["elev"], k), transform_raster(d["valid"], k)
    if "elev" in names or "slope" in names:
        out["z"], out["valid"] = elev.astype(np.float32), valid.astype(np.float32)
    if "slope" in names:
        gy, gx = np.gradient(elev.astype(float), PX)
        out["slope"] = np.clip(np.hypot(gx, gy) / norm["slope_scale"], 0, 4).astype(np.float32) * out["valid"]
    if "water" in names:
        out["water"] = (transform_raster(d["water"], k) > 0).astype(np.float32)
    return out


def grid_rc(pos, heading):
    """Rows / cols in the margin grid of the rotated crop pixels, (P, 48, 48) each."""
    pos, heading = np.atleast_2d(pos), np.atleast_1d(heading)
    fwd = -_LOCAL[None, :, None]
    right = _LOCAL[None, None, :]
    c, s = np.cos(heading)[:, None, None], np.sin(heading)[:, None, None]
    dx, dy = fwd * c + right * s, fwd * s - right * c
    col = np.floor(pos[:, 0, None, None] / PX + dx).astype(int) + MARGIN_PX
    row = np.floor((TILE_M - pos[:, 1, None, None]) / PX - dy).astype(int) + MARGIN_PX
    return np.clip(row, 0, 255), np.clip(col, 0, 255)


def node_height(z, pos):
    col = np.clip((pos[:, 0] / PX).astype(int) + MARGIN_PX, 0, 255)
    row = np.clip(((TILE_M - pos[:, 1]) / PX).astype(int) + MARGIN_PX, 0, 255)
    return z[row, col]


def terrain_crops(layers: dict, names, pos, heading, norm: dict) -> np.ndarray:
    """(P, C, 48, 48) float16 terrain channels for pops at ``pos`` with ``heading``."""
    chans = channel_list(names)
    pos = np.atleast_2d(np.asarray(pos, float))
    if not len(pos):
        return np.zeros((0, len(chans), 2 * HALF, 2 * HALF), np.float16)
    row, col = grid_rc(pos, heading)
    out = []
    for n in chans:
        if n == "elev":
            z0 = node_height(layers["z"], pos)[:, None, None]
            out.append(np.clip((layers["z"][row, col] - z0) / norm["elev_scale"], -3, 3) * layers["valid"][row, col])
        else:
            out.append(layers[n][row, col])
    return np.stack(out, 1).astype(np.float16)


def tile_canvas(layers: dict, names, norm: dict) -> np.ndarray:
    """(C, 128, 128) terrain channels of the tile itself (no margin) for the start head."""
    chans = channel_list(names)
    sl = slice(MARGIN_PX, MARGIN_PX + 128)
    out = []
    for n in chans:
        if n == "elev":
            z = layers["z"][sl, sl]
            v = layers["valid"][sl, sl]
            med = np.median(z[v > 0]) if (v > 0).any() else 0.0
            out.append(np.clip((z - med) / norm["elev_scale"], -3, 3) * v)
        else:
            out.append(layers[n][sl, sl])
    return np.stack(out, 0).astype(np.float16)
