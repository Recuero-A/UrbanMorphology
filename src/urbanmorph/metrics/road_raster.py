"""Road-network metrics computed from a RASTER road mask (Milestone 3b).

Pipeline: mask -> 1-pixel skeleton (Zhang-Suen thinning) -> short spurs pruned ->
pixels classified by their number of skeleton neighbours:
    1 neighbour  = end point (dead end, or the tile edge)
    2 neighbours = part of a segment
    >= 3         = junction pixel; touching junction pixels form ONE intersection

Metrics and their limitations
-----------------------------
* skeleton_components: pieces of the network (fragmentation). Tile edges cut real
  streets, so even a perfect network can have several pieces.
* intersections_per_km2: junction clusters. Skeletons of thick or blobby shapes
  create spurious branches and junctions; pruning removes only short spurs.
  Compare against the vector-graph intersection density only as a sanity check
  (different definitions: the graph merges junctions within 20 m, the raster cannot
  see under bridges, and buffered parallel carriageways can merge).
* dead_end_ratio: end points / (end points + intersections). End points at the tile
  edge are counted too.
* mean_segment_length_m: skeleton length between junctions/end points (continuity).
* mean_road_width_m = road area / skeleton length. Real buffered streets are ~3-20 m;
  a blob has a large "width". This is the main guard against treating connected
  blobs as street networks.
* largest_component_share: share of skeleton length in the largest piece.
"""

from __future__ import annotations

import numpy as np

from urbanmorph.models.evaluate import label_components

PIXEL_M = 4.0
TILE_KM2 = (128 * PIXEL_M / 1000) ** 2


def _neighbours(img: np.ndarray) -> list[np.ndarray]:
    """P2..P9 in Zhang-Suen order (clockwise from north) for every pixel (zero padded)."""
    p = np.pad(img, 1)
    h, w = img.shape
    s = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return [s(-1, 0), s(-1, 1), s(0, 1), s(1, 1), s(1, 0), s(1, -1), s(0, -1), s(-1, -1)]


def zhang_suen_thin(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Thin a binary mask to a 1-pixel-wide skeleton (Zhang & Suen 1984), vectorised."""
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            n = _neighbours(img)
            b = sum(n)
            seq = n + [n[0]]
            a = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            rm = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    return sum(_neighbours(skel.astype(np.uint8))).astype(np.int32) * skel


def prune_spurs(skel: np.ndarray, length: int = 3) -> np.ndarray:
    """Remove end-point pixels ``length`` times (shortens spurs; long dead ends survive, shortened)."""
    s = skel.copy()
    for _ in range(length):
        ends = (neighbour_count(s) == 1)
        if not ends.any():
            break
        s[ends] = False
    return s


def road_network_metrics(mask: np.ndarray, spur_px: int = 3) -> dict:
    mask = mask.astype(bool)
    area_px = int(mask.sum())
    if area_px == 0:
        return {"road_px": 0, "skeleton_length_m": 0.0, "skeleton_components": 0, "intersections_per_km2": 0.0,
                "dead_end_ratio": np.nan, "mean_segment_length_m": np.nan, "mean_road_width_m": np.nan,
                "largest_component_share": np.nan}
    skel = prune_spurs(zhang_suen_thin(mask), spur_px)
    length_px = int(skel.sum())
    if length_px == 0:  # tiny specks vanish under pruning
        return {"road_px": area_px, "skeleton_length_m": 0.0, "skeleton_components": 0, "intersections_per_km2": 0.0,
                "dead_end_ratio": np.nan, "mean_segment_length_m": np.nan, "mean_road_width_m": np.nan,
                "largest_component_share": np.nan}
    nc = neighbour_count(skel)
    ends = int((nc == 1).sum())
    junction_px = nc >= 3
    n_inter = int(label_components(junction_px).max()) if junction_px.any() else 0
    comp = label_components(skel)
    sizes = np.bincount(comp.ravel())[1:]
    segs = label_components(skel & ~junction_px)
    seg_sizes = np.bincount(segs.ravel())[1:]
    return {
        "road_px": area_px,
        "skeleton_length_m": length_px * PIXEL_M,
        "skeleton_components": int(len(sizes)),
        "intersections_per_km2": n_inter / TILE_KM2,
        "dead_end_ratio": ends / (ends + n_inter) if (ends + n_inter) else np.nan,
        "mean_segment_length_m": float(seg_sizes.mean() * PIXEL_M) if seg_sizes.size else np.nan,
        "mean_road_width_m": area_px * PIXEL_M ** 2 / (length_px * PIXEL_M),
        "largest_component_share": float(sizes.max() / sizes.sum()),
    }


def summarise(masks: np.ndarray) -> dict:
    """Median of each metric over a stack of road masks (N, H, W)."""
    rows = [road_network_metrics(m) for m in masks]
    keys = rows[0].keys()
    return {k: float(np.nanmedian([r[k] for r in rows])) if any(np.isfinite(r[k]) for r in rows) else float("nan")
            for k in keys}


# ------------------------------------------------------------------ topology (road-topology experiment)
def _label(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Connected-component labels with 4- or 8-connectivity (0 = background)."""
    from collections import deque
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    n = 0
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        n += 1
        lab[i, j] = n
        q = deque([(i, j)])
        while q:
            a, b = q.popleft()
            for da, db in offs:
                y, x = a + da, b + db
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not lab[y, x]:
                    lab[y, x] = n
                    q.append((y, x))
    return lab


def topology_metrics(mask: np.ndarray, min_hole_px: int = 2, min_piece_px: int = 10) -> dict:
    """Betti numbers and boundary-aware fragmentation of a road mask.

    * betti0: road pieces (8-connected), and betti0_10 counting only pieces >= 10 px.
    * interior_fragments: pieces (>= 10 px) that touch NO tile edge. A street cut by the
      512 m window is legitimate; a road piece floating inside the tile is not, except
      for real cases (e.g. a closed campus loop, a mapping gap).
    * betti1 / loops_per_km2: holes = 4-connected background regions that do not touch
      the tile edge and have >= ``min_hole_px`` pixels. City blocks cut by the tile edge
      are not counted as loops (boundary truncation). 1-pixel holes from rasterisation
      noise are ignored.
    Limitations: at 4 m/px, parallel carriageways can merge (fewer loops) or leave thin
    gaps (extra loops); a loop's existence depends on the 8/4-connectivity convention.
    """
    m = mask.astype(bool)
    fg = _label(m, 8)
    sizes = np.bincount(fg.ravel())[1:]
    border = np.unique(np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]]))
    big = np.flatnonzero(sizes >= min_piece_px) + 1
    interior = [k for k in big if k not in set(border.tolist())]
    bg = _label(~m, 4)
    bsizes = np.bincount(bg.ravel())[1:]
    bborder = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])).tolist())
    holes = [k for k in range(1, len(bsizes) + 1) if k not in bborder and bsizes[k - 1] >= min_hole_px]
    return {"betti0": int(len(sizes)), "betti0_10": int(len(big)), "interior_fragments": int(len(interior)),
            "betti1": int(len(holes)), "loops_per_km2": len(holes) / TILE_KM2}


def hard_cldice(pred: np.ndarray, true: np.ndarray) -> float:
    """clDice on binary masks with exact (Zhang-Suen) skeletons: 2 Tprec Tsens / (Tprec + Tsens)."""
    p, t = pred.astype(bool), true.astype(bool)
    if not p.any() and not t.any():
        return 1.0
    sp, st = zhang_suen_thin(p), zhang_suen_thin(t)
    tprec = (sp & t).sum() / sp.sum() if sp.any() else 0.0
    tsens = (st & p).sum() / st.sum() if st.any() else 0.0
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0


def orientation_from_skeleton(mask: np.ndarray, min_seg_px: int = 4) -> dict:
    """Street-orientation statistics of a raster road network.

    The skeleton is split at junctions into segments; each segment of >= 4 px gets its
    principal-axis direction (PCA of its pixel coordinates), weighted by its length,
    and the result is passed to the SAME orientation statistics used for the vector
    street graph (18 axial bins, entropy, Boeing's phi, grid strength R4). Curved
    segments are summarised by one mean direction (limitation).
    """
    from urbanmorph.metrics.morphology import orientation_metrics
    skel = prune_spurs(zhang_suen_thin(mask.astype(bool)), 3)
    nc = neighbour_count(skel)
    segs = _label(skel & ~(nc >= 3), 8)
    lines = []
    for k in range(1, segs.max() + 1):
        r, c = np.nonzero(segs == k)
        if len(r) < min_seg_px:
            continue
        pts = np.c_[c.astype(float), -r.astype(float)]  # x east, y north (row 0 = north)
        ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        d = vt[0] * (len(r) * PIXEL_M) / 2
        lines.append([*(ctr * PIXEL_M - d), *(ctr * PIXEL_M + d)])
    om = orientation_metrics(np.array(lines) if lines else np.zeros((0, 4)))
    return {"orientation_entropy": om["orientation_entropy"], "orientation_phi": om["orientation_order_phi"],
            "grid_strength_r4": om["grid_strength_r4"]}


def full_road_profile(mask: np.ndarray) -> dict:
    """All per-tile road metrics used in the road-topology experiment."""
    m = mask.astype(bool)
    out = {"road_coverage": float(m.mean())}
    out.update(road_network_metrics(m))
    out.update(topology_metrics(m))
    out.update(orientation_from_skeleton(m))
    return out
