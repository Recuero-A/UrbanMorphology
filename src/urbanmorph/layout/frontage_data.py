"""Teacher-forcing samples of the frontage model (Milestone 7).

Order of one tile: frontage after frontage (frontage.py order); along each frontage, buildings by
position s, then STOP. Step k of a frontage is predicted from
  * a 32 x 32 crop (4 m pixels) centred on the street point at s_{k-1} and rotated to the street's
    direction there (the street runs along the crop's x axis, the building side is "up"), showing
    roads and every frontage building placed earlier (this and other frontages);
  * scalars: street class, link length, position, remaining length, step index, and the parameters
    of the previous TWO buildings on this frontage (gap, setback, orientation, size) -> rhythm.
Targets: STOP or (log(gap + 1 m), log setback), orientation relative to the street (axial, with the
canonical shape frame), log size, shape.
Inputs live in the street's frame, so no symmetry augmentation is needed.

``relative=True`` (Milestone 7b): for every building after the first on a frontage, the targets
(log gap, log setback, orientation) are RESIDUALS w.r.t. the previous building: "same as my neighbour"
is then the value 0, which a mixture can represent sharply. Real neighbours on a frontage differ by
a median of 0.7 deg in orientation; predicting absolute angles gave 2.9 deg.
"""

from __future__ import annotations

import numpy as np

from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.layout.frontage import assign, point_at, tangent
from urbanmorph.layout.objects import decode
from urbanmorph.layout.placer_data import CROP, N, canonical, shape_to_unbounded

N_FSCAL = 20
MAX_PER_FRONTAGE = 60


def prev_features(p1, p2) -> np.ndarray:
    """Parameters of the previous two buildings: (log gap+1)/3, log d/3, phi, log L/3, log W/3, present."""
    out = []
    for p in (p1, p2):
        out += [p[0] / 3, p[1] / 3, p[2], p[3] / 3, p[4] / 3, 1.0] if p is not None else [0.0] * 6
    return np.array(out, np.float32)


def step_scalars(cls, length, s_prev, k, p1, p2) -> np.ndarray:
    return np.r_[np.eye(4)[cls], length / 200, s_prev / max(length, 1.0), (length - s_prev) / 200, k / 20,
                 prev_features(p1, p2)].astype(np.float32)


def time_crops(roads, bmap, centres, angles, t) -> np.ndarray:
    """(P, 5, CROP, CROP) rotated crops of [roads, buildings with time < t, green (empty), phase 0, outside]."""
    P = len(centres)
    loc = np.arange(CROP) - CROP / 2 + 0.5
    along, across = loc[None, None, :], -loc[None, :, None]
    c, s = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    col = np.floor(centres[:, 0, None, None] / 4.0 + along * c - across * s).astype(int)
    row = np.floor((TILE_M - centres[:, 1, None, None]) / 4.0 - (along * s + across * c)).astype(int)
    inside = (col >= 0) & (col < N) & (row >= 0) & (row < N)
    col, row = np.clip(col, 0, N - 1), np.clip(row, 0, N - 1)
    r = roads.reshape(N, N)[row, col] & inside
    b = (bmap.reshape(N, N)[row, col] < np.asarray(t, float)[:, None, None]) & inside
    z = np.zeros_like(r)
    return np.stack([r, b, z, z, ~inside], 1)


def tile_frontage_samples(tile: dict, relative: bool = False) -> dict:
    """All steps of one tile (k = 0 geometry)."""
    o, g = tile["objects"], tile["graph"]
    bidx = np.flatnonzero(o["type"] == 0)
    polys = [decode(o["anchor"][i], o["theta"][i], o["length"][i], o["width"][i], o["ratios"][i]) for i in bidx]
    fr, is_front = assign(polys, g, o["anchor"][bidx])
    order = [bidx[m] for f in fr for m in f["members"][:MAX_PER_FRONTAGE]]
    time_of = {int(b): t for t, b in enumerate(order)}
    bmap = np.full(N * N, np.inf)
    for b, t in time_of.items():
        np.minimum.at(bmap, o["pix"][o["pix_ptr"][b]:o["pix_ptr"][b + 1]], t)
    rows = {k: [] for k in ("centre", "angle", "t", "scalars", "stop", "gapd", "phi", "size", "shape")}
    t = 0
    for f in fr:
        line, L = f["line"], f["length"]
        s_prev, p1, p2 = 0.0, None, None
        members = f["members"][:MAX_PER_FRONTAGE]
        for k in range(len(members) + 1):
            a = tangent(line, s_prev)
            rows["centre"].append(point_at(line, s_prev))
            rows["angle"].append(a)
            rows["t"].append(t)
            rows["scalars"].append(step_scalars(f["cls"], L, s_prev, k, p1, p2))
            if k == len(members):
                rows["stop"].append(1.0)
                rows["gapd"].append(np.zeros(2, np.float32))
                rows["phi"].append(0.0)
                rows["size"].append(np.zeros(2, np.float32))
                rows["shape"].append(np.zeros(20, np.float32))
                break
            b = bidx[members[k]]
            s, d = f["s"][k], f["d"][k]
            axis = tangent(line, s)
            phi, ratios = canonical(o["theta"][b:b + 1], o["ratios"][b:b + 1], np.array([axis]))
            gap = max(s - s_prev, 0.0)
            par = (np.log(gap + 1), np.log(max(d, 0.5)), float(phi[0]), np.log(o["length"][b]), np.log(o["width"][b]))
            rows["stop"].append(0.0)
            if relative and p1 is not None:
                rows["gapd"].append(np.array([par[0] - p1[0], par[1] - p1[1]], np.float32))
                rows["phi"].append(float((par[2] - p1[2] + np.pi / 2) % np.pi - np.pi / 2))
            else:
                rows["gapd"].append(np.array(par[:2], np.float32))
                rows["phi"].append(par[2])
            rows["size"].append(np.array(par[3:], np.float32))
            rows["shape"].append(shape_to_unbounded(ratios)[0])
            p1, p2, s_prev = par, p1, s
            t += 1
    shapes = {"centre": (2,), "scalars": (N_FSCAL,), "gapd": (2,), "size": (2,), "shape": (20,)}
    out = {k: np.asarray(v, np.float32).reshape(-1, *shapes.get(k, ())) for k, v in rows.items()}
    out["bmap"], out["roads"] = bmap, tile["road_raster"].ravel()
    return out


def crops_for(samples: dict, idx=None) -> np.ndarray:
    idx = np.arange(len(samples["t"])) if idx is None else idx
    return time_crops(samples["roads"], samples["bmap"], samples["centre"][idx], samples["angle"][idx], samples["t"][idx])
