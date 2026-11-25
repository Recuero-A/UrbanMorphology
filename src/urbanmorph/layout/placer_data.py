"""Features shared by training and sampling of the object placer (Milestone 6).

Construction order of one tile: all buildings (random order, re-drawn every epoch), STOP, all green
spaces (random order), STOP. Step t sees a 128 x 128 canvas with 4 channels:
    roads (drawn from the street graph with per-class widths), buildings placed so far,
    green placed so far, phase (0 = buildings, 1 = green).
Attributes are predicted from a 32 x 32 crop (4 m pixels = 128 m) ROTATED so that the nearest
street runs along the crop's x axis, plus scalars. Orientation is therefore learned relative to
the street, in the same spirit as the turtle frame for roads.
"""

from __future__ import annotations

import numpy as np

from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.graphs.sequence import pixel_rc
from urbanmorph.layout.objects import encode, fold_axial, polygon_pixels, street_axis_and_distance

N = 128
CROP = 32
N_SCAL = 10
_MIRROR_RAYS = np.r_[(16 - np.arange(16)) % 16, 19, 18, 17, 16]     # local angle a -> -a
_HALF_TURN_RAYS = np.r_[(np.arange(16) + 8) % 16, 18, 19, 16, 17]    # frame rotated by pi


# ------------------------------------------------------------------ encoding a tile
def encode_tile(polys_b, polys_g) -> dict:
    rows = [(0, p) for p in polys_b] + [(1, p) for p in polys_g]
    n = len(rows)
    out = {"type": np.array([t for t, _ in rows], np.int8), "anchor": np.zeros((n, 2), np.float32),
           "theta": np.zeros(n, np.float32), "length": np.zeros(n, np.float32), "width": np.zeros(n, np.float32),
           "ratios": np.zeros((n, 20), np.float32), "pix_ptr": np.zeros(n + 1, np.int64)}
    pix = []
    for i, (_, p) in enumerate(rows):
        e = encode(p)
        out["anchor"][i], out["theta"][i], out["length"][i], out["width"][i] = e["anchor"], e["theta"], e["length"], e["width"]
        out["ratios"][i] = e["ratios"]
        px = polygon_pixels(p)
        pix.append(px)
        out["pix_ptr"][i + 1] = out["pix_ptr"][i] + len(px)
    out["pix"] = np.concatenate(pix).astype(np.int32) if pix else np.zeros(0, np.int32)
    return out


# ------------------------------------------------------------------ symmetries
def transform_points(p, k):
    p = np.asarray(p, float) - TILE_M / 2
    if k >= 4:
        p = p * np.array([-1.0, 1.0])
    for _ in range(k % 4):
        p = np.c_[-p[:, 1], p[:, 0]]
    return p + TILE_M / 2


def transform_pixels(flat, k):
    r, c = np.divmod(np.asarray(flat), N)
    if k >= 4:
        c = N - 1 - c
    for _ in range(k % 4):
        r, c = N - 1 - c, r
    return r * N + c


def transform_raster(a, k):
    if k >= 4:
        a = a[..., ::-1]
    return np.rot90(a, k % 4, axes=(-2, -1)).copy()


def transform_objects(o: dict, k: int) -> dict:
    out = dict(o)
    out["anchor"] = transform_points(o["anchor"], k).astype(np.float32)
    th = o["theta"].astype(float)          # DIRECTED: the shape rays are measured from this direction
    ratios = o["ratios"]
    if k >= 4:
        th = np.pi - th
        ratios = ratios[:, _MIRROR_RAYS]
    out["theta"] = (th + (k % 4) * np.pi / 2).astype(np.float32)
    out["ratios"] = ratios
    out["pix"] = transform_pixels(o["pix"], k).astype(np.int32)
    return out


# ------------------------------------------------------------------ street context
def street_context(points, graph):
    """Direction, distance and class of the nearest street for each point.

    The direction is DIRECTED so that the point lies on the street's left: "the street, seen from
    the building" is then well defined, and a footprint's facing (e.g. an L opening to the street)
    is preserved."""
    pts = np.atleast_2d(points)
    seg = graph.segments()
    if not len(seg):
        return np.zeros(len(pts)), np.full(len(pts), 999.0), np.full(len(pts), 2)
    A, B = seg[:, :2], seg[:, 2:]
    s = B - A
    lam = np.clip(((pts[:, None] - A[None]) * s[None]).sum(-1) / np.maximum((s ** 2).sum(1), 1e-9)[None], 0, 1)
    proj = A[None] + lam[..., None] * s[None]
    d = np.hypot(*(pts[:, None] - proj).transpose(2, 0, 1))
    k = d.argmin(1)
    i = np.arange(len(pts))
    sk, rel = s[k], pts - proj[i, k]
    axis = np.arctan2(sk[:, 1], sk[:, 0])
    axis = np.where(sk[:, 0] * rel[:, 1] - sk[:, 1] * rel[:, 0] < 0, axis + np.pi, axis)
    return axis, d[i, k], graph.classes()[k]


def canonical(theta, ratios, axis):
    """Orientation relative to the (directed) street, folded to [-pi/2, pi/2); the shape rays are
    rotated by pi whenever the fold flips the object's frame, so the polygon is unchanged."""
    rel = np.asarray(theta, float) - axis
    phi = fold_axial(rel)
    flipped = np.abs(((rel - phi) / np.pi) % 2 - 1) < 1e-6        # rel - phi is an odd multiple of pi
    ratios = np.where(flipped[:, None], np.asarray(ratios)[:, _HALF_TURN_RAYS], ratios)
    return phi.astype(np.float32), ratios


def rotated_crops(canvas: np.ndarray, centres: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(P, C + 1, CROP, CROP) crops of (P, C, N, N) canvases, x axis along ``angles``; last channel = outside tile."""
    P = len(centres)
    loc = np.arange(CROP) - CROP / 2 + 0.5
    along, across = loc[None, None, :], -loc[None, :, None]           # crop col = along street, row = across
    c, s = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    dx = along * c - across * s
    dy = along * s + across * c
    col = np.floor(centres[:, 0, None, None] / 4.0 + dx).astype(int)
    row = np.floor((TILE_M - centres[:, 1, None, None]) / 4.0 - dy).astype(int)
    inside = (col >= 0) & (col < N) & (row >= 0) & (row < N)
    col, row = np.clip(col, 0, N - 1), np.clip(row, 0, N - 1)
    idx = np.arange(P)[:, None, None]
    crops = canvas[idx, :, row, col].transpose(0, 3, 1, 2) & inside[:, None]
    return np.concatenate([crops, ~inside[:, None]], 1)


def scalars(anchors, dist, cls, phase, n_b, n_g) -> np.ndarray:
    anchors = np.atleast_2d(anchors)
    row, col = pixel_rc(anchors)
    sub_x = anchors[:, 0] / 4 - col - 0.5
    sub_y = (TILE_M - anchors[:, 1]) / 4 - row - 0.5
    onehot = np.eye(4)[np.asarray(cls, int)]
    return np.c_[sub_x, sub_y, np.minimum(dist, 120) / 30, phase, onehot, np.asarray(n_b) / 1000,
                 np.asarray(n_g) / 50].astype(np.float32)


def shape_to_unbounded(ratios):
    r = np.clip((np.asarray(ratios) - 0.05) / 0.95, 0.005, 0.995)
    return np.log(r / (1 - r)).astype(np.float32)


def shape_from_unbounded(s):
    return 0.05 + 0.95 / (1 + np.exp(-np.asarray(s)))


# ------------------------------------------------------------------ teacher-forcing samples
def tile_samples(tile: dict, k: int, rng: np.random.Generator, frac: float = 1.0) -> dict:
    """Samples of one tile under symmetry k with a random object order; a fraction ``frac`` of steps is kept."""
    o = transform_objects(tile["objects"], k)
    g = tile["graph"].transform(k)
    roads = transform_raster(tile["road_raster"], k)
    typ = o["type"]
    order = np.r_[rng.permutation(np.flatnonzero(typ == 0)), rng.permutation(np.flatnonzero(typ == 1))]
    nb, ng = int((typ == 0).sum()), int((typ == 1).sum())
    # step sequence: buildings (t = 0..nb-1), STOP (t = nb), green (t = nb+1 .. nb+ng), STOP (t = nb+ng+1)
    steps = np.r_[order[:nb], -1, order[nb:], -1]
    phase = np.r_[np.zeros(nb + 1), np.ones(ng + 1)]
    T = len(steps)
    keep = np.flatnonzero(rng.random(T) < frac)
    if not len(keep):
        return {}
    # time stamp of each object = its position in the sequence; canvas at step t shows objects with time < t
    time_of = np.empty(len(typ))
    time_of[order[:nb]] = np.arange(nb)
    time_of[order[nb:]] = nb + 1 + np.arange(ng)
    bmap = np.full(N * N, np.inf)
    gmap = np.full(N * N, np.inf)
    for i in range(len(typ)):
        px = o["pix"][o["pix_ptr"][i]:o["pix_ptr"][i + 1]]
        np.minimum.at(bmap if typ[i] == 0 else gmap, px, time_of[i])
    t = keep.astype(float)
    canv = np.zeros((len(keep), 4, N, N), bool)
    canv[:, 0] = roads[None]
    canv[:, 1] = (bmap[None] < t[:, None]).reshape(-1, N, N)
    canv[:, 2] = (gmap[None] < t[:, None]).reshape(-1, N, N)
    canv[:, 3] = phase[keep, None, None] > 0
    n_b = np.minimum(keep, nb)
    n_g = np.maximum(keep - nb - 1, 0)
    obj = steps[keep]
    is_obj = obj >= 0
    oi = np.where(is_obj, obj, 0)
    anchors = o["anchor"][oi]
    axis, dist, cls = street_context(anchors, g)
    row, col = pixel_rc(anchors)
    L, W = o["length"][oi], o["width"][oi]
    phi, ratios = canonical(o["theta"][oi], o["ratios"][oi], axis)
    return {"canvas": canv, "stop": (~is_obj).astype(np.float32), "phase": phase[keep].astype(np.float32),
            "counts": np.c_[n_b / 1000, n_g / 50].astype(np.float32),
            "cell": (row * N + col).astype(np.int64),
            "crops": rotated_crops(canv, anchors, axis),
            "scalars": scalars(anchors, dist, cls, phase[keep], n_b, n_g),
            "phi": phi,
            "size": np.c_[np.log(L), np.log(W)].astype(np.float32),
            "shape": shape_to_unbounded(ratios)}


# ------------------------------------------------------------------ Milestone 8 B2: neighbour-aware fallback
K_NB, NB_RADIUS, NB_FEAT = 8, 60.0, 7


def neighbour_features(anchor, axis, nb_anchor, nb_theta, nb_logL, nb_logW):
    """(K_NB, 7) features of the nearest placed buildings, in the frame of the (directed) street axis:
    offset (along, across) / 30 m, orientation relative to the street as (cos 2phi, sin 2phi), log sizes / 3,
    distance / 30 m; plus a mask."""
    f = np.zeros((K_NB, NB_FEAT), np.float32)
    m = np.zeros(K_NB, bool)
    if len(nb_anchor):
        d = np.hypot(*(nb_anchor - anchor).T)
        sel = np.argsort(d)[:K_NB]
        sel = sel[d[sel] < NB_RADIUS]
        c, s = np.cos(-axis), np.sin(-axis)
        rel = nb_anchor[sel] - anchor
        rel = np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]
        ph = 2 * (nb_theta[sel] - axis)
        f[:len(sel)] = np.c_[rel / 30, np.cos(ph), np.sin(ph), nb_logL[sel] / 3, nb_logW[sel] / 3, d[sel] / 30]
        m[:len(sel)] = True
    return f, m


def tile_samples_b2(tile: dict, k: int, rng: np.random.Generator, frac: float = 1.0, front_drop: float = 0.0) -> dict:
    """Like tile_samples, but the tile's FRONTAGE buildings (tile['front_mask']) are already placed (time -1),
    as after the frontage phase; the sequence is interior buildings (random order), STOP, green, STOP.
    Adds neighbour features of the placed buildings at each step.
    ``front_drop`` (Milestone 8 S): each frontage building is removed from the pre-placed set with
    probability q ~ U(0, front_drop) (q drawn per tile) and must be placed by the fallback instead, as
    when the frontage phase rejects proposals."""
    o = transform_objects(tile["objects"], k)
    g = tile["graph"].transform(k)
    roads = transform_raster(tile["road_raster"], k)
    typ = o["type"]
    front = np.zeros(len(typ), bool)
    front[np.flatnonzero(typ == 0)[tile["front_mask"]]] = True
    if front_drop > 0:
        front &= rng.random(len(typ)) >= rng.uniform(0, front_drop)
    interior = np.flatnonzero((typ == 0) & ~front)
    greens = np.flatnonzero(typ == 1)
    order = np.r_[rng.permutation(interior), rng.permutation(greens)]
    ni, ng, nf = len(interior), len(greens), int(front.sum())
    steps = np.r_[order[:ni], -1, order[ni:], -1]
    phase = np.r_[np.zeros(ni + 1), np.ones(ng + 1)]
    keep = np.flatnonzero(rng.random(len(steps)) < frac)
    if not len(keep):
        return {}
    time_of = np.full(len(typ), -1.0)
    time_of[order[:ni]] = np.arange(ni)
    time_of[order[ni:]] = ni + 1 + np.arange(ng)
    bmap = np.full(N * N, np.inf)
    gmap = np.full(N * N, np.inf)
    for i in range(len(typ)):
        np.minimum.at(bmap if typ[i] == 0 else gmap, o["pix"][o["pix_ptr"][i]:o["pix_ptr"][i + 1]], time_of[i])
    t = keep.astype(float)
    canv = np.zeros((len(keep), 4, N, N), bool)
    canv[:, 0] = roads[None]
    canv[:, 1] = (bmap[None] < t[:, None]).reshape(-1, N, N)
    canv[:, 2] = (gmap[None] < t[:, None]).reshape(-1, N, N)
    canv[:, 3] = phase[keep, None, None] > 0
    n_b = nf + np.minimum(keep, ni)
    n_g = np.maximum(keep - ni - 1, 0)
    obj = steps[keep]
    is_obj = obj >= 0
    oi = np.where(is_obj, obj, 0)
    anchors = o["anchor"][oi]
    axis, dist, cls = street_context(anchors, g)
    row, col = pixel_rc(anchors)
    L, W = o["length"][oi], o["width"][oi]
    phi, ratios = canonical(o["theta"][oi], o["ratios"][oi], axis)
    bidx = np.flatnonzero(typ == 0)
    nb, nbm = [], []
    for q in range(len(keep)):
        placed = bidx[time_of[bidx] < t[q]]
        f, m = neighbour_features(anchors[q], axis[q], o["anchor"][placed], o["theta"][placed],
                                  np.log(o["length"][placed]), np.log(o["width"][placed]))
        nb.append(f)
        nbm.append(m)
    return {"canvas": canv, "stop": (~is_obj).astype(np.float32), "phase": phase[keep].astype(np.float32),
            "counts": np.c_[n_b / 1000, n_g / 50].astype(np.float32), "cell": (row * N + col).astype(np.int64),
            "crops": rotated_crops(canv, anchors, axis), "scalars": scalars(anchors, dist, cls, phase[keep], n_b, n_g),
            "phi": phi, "size": np.c_[np.log(L), np.log(W)].astype(np.float32), "shape": shape_to_unbounded(ratios),
            "nb": np.stack(nb), "nb_mask": np.stack(nbm)}
