"""Milestone 9: tidy subdivision (BlockNet), green zoning and template buildings (docs/MILESTONE_9_PLAN.md §3).

Every parcel is GENERATED (status "generated"), never cadastral truth. Per block of the street graph:

LEARNED (models/block_net.py, fitted on observed train parcels)
  * block type {none, single, few, rows};
  * per street edge (parcel_gen.street_edges, longest first): number of lots N and rear ratio r;
  * (TZ only) land use of each building parcel {building, green}, sequential in random order, given
    the mean label of already-assigned neighbours;
  * template of each building (layout/templates.py) given lot width, depth, coverage target, country;
  * coverage, number of buildings and setbacks: ParcelNet (M8 D, NetSampler), coverage with Gibbs
    sweeps as in M8.
PROGRAMMED
  * none: green parcel if >= GREEN_MIN_M2, else left as road verge; single: the whole block is one parcel;
  * few / rows: the edge is cut into N EQUAL-width lots, each width x exp(N(0, s)) with s the measured
    robust spread of widths along real street sides (outputs/m9/analysis/jitter.json), rescaled to the
    edge length; lot depth = r x block depth, snapped to half the block depth ("back-to-back") when
    r > SNAP_RATIO and the far side is also a street; lots are clipped to the block and to lots cut
    earlier; slivers merged (parcel_gen.merge_slivers);
  * block remainder: pieces < GREEN_MIN_M2 are merged into the adjacent lot with the longest shared
    boundary; pieces up to GREEN_MAX_M2 become green parcels; larger pieces are split recursively
    (across the long side of their minimum rotated rectangle, at the middle) into interior lots of about
    the block's median street-lot area. CHANGE vs the plan (made on train tiles, before evaluation):
    the plan turned every piece >= 400 m2 into green, which covered half of some tiles (large blocks and
    tile-edge blocks); GREEN_MAX_M2 = 4000 m2 is about the 75th percentile of real green parcels;
  * zoning is sampled WITHOUT the neighbour feature (has_nb = 0). CHANGE vs the plan: with a green
    neighbour the learned probability is 0.6-0.7, so sequential sampling spread green lot to lot
    (25-40 % of parcels green on train tiles vs 1-3 % real): a branching process above 1;
  * green parcel -> green polygon = parcel inset by 0.5 m;
  * building = template STRETCHED to its envelope (lot minus setbacks), never clipped; if it does not
    fit inside its parcel it is shrunk (x 0.85, up to 4 times), else dropped (counted).
"""

from __future__ import annotations

import numpy as np
import shapely
import torch
from shapely.geometry import LineString, Polygon

from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.parcel_gen import (GIBBS_SWEEPS, MAIN_SHARE, MIN_DIM_M, _largest, context, depth_cap,
                                          merge_slivers, street_class_at, street_edges)
from urbanmorph.layout.parcels import FRONT_TOL_M, MIN_PARCEL_M2, parcel_neighbours
from urbanmorph.layout.templates import FILL, place
from urbanmorph.models.block_net import block_x, shape_x, side_x, zone_x

GREEN_MIN_M2 = 400.0
GREEN_MAX_M2 = 4000.0
SNAP_RATIO = 0.45
SHRINK, N_SHRINK = 0.85, 4


def block_type(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3


def full_depth(block, a, t, n, L):
    mid = a + t * L / 2
    r = np.arange(0.5, 600, 1.0)
    pts = mid[None] + r[:, None] * n[None]
    out = np.flatnonzero(~shapely.contains_xy(block, pts[:, 0], pts[:, 1]))
    return float(r[out[0]]) if len(out) else 600.0


def block_features(b, edges, graph):
    cls = street_class_at(np.array([a + t * L / 2 for L, a, t, n in edges]), graph) if edges else np.array([2])
    return {"log_area": float(np.log(b["polygon"].area)), "rect": float(b["rectangularity"]),
            "compact": float(b["compactness"]), "n_edges": len(edges), "cls": int(np.bincount(cls, minlength=4).argmax())}


def split_piece(piece, target):
    """Recursive halving across the long side of the minimum rotated rectangle until <= 2 x target."""
    if piece.area <= 2 * target:
        return [piece]
    rect = np.asarray(shapely.minimum_rotated_rectangle(piece).exterior.coords)[:4]
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    if np.hypot(*e1) < np.hypot(*e2):
        rect = np.roll(rect, -1, 0)                  # long side first
        e1 = rect[1] - rect[0]
    half =Polygon([rect[0], rect[0] + e1 / 2, rect[3] + e1 / 2, rect[3]]).buffer(0.01)
    out = []
    for part in (piece.intersection(half), piece.difference(half)):
        for q in getattr(shapely.make_valid(part), "geoms", [shapely.make_valid(part)]):
            if q.geom_type == "Polygon" and q.area >= MIN_PARCEL_M2:
                out += split_piece(q, target)
    return out or [piece]


def _merge_into(lots, piece):
    """Merge a small remainder piece into the lot with the longest shared boundary; False if none."""
    best, best_len = None, 0.0
    for j, q in enumerate(lots):
        if q["polygon"].distance(piece) < 0.1:
            shared = q["polygon"].buffer(0.1).intersection(piece.exterior).length
            if shared > best_len:
                best, best_len = j, shared
    if best is None or best_len < 1.0:
        return False
    merged = _largest(shapely.make_valid(lots[best]["polygon"].union(piece.buffer(0.01)).buffer(-0.01)))
    if merged.is_empty or merged.area < lots[best]["polygon"].area:
        return False
    lots[best]["polygon"] = merged
    return True


def subdivide_block(b, near, country, nets, graph, jitter, rng, g, stats):
    poly = b["polygon"]
    edges = sorted(street_edges(poly, near), key=lambda e: -e[0])
    f = block_features(b, edges, graph)
    bx = block_x([country], [f["log_area"]], [f["rect"]], [f["compact"]], [f["n_edges"]], [f["cls"]])
    bt = nets.sample_type(bx, g)
    stats["block_type"][bt] += 1
    green = lambda p: {"polygon": p, "front": None, "interior": True, "use": "green", "btype": bt}
    if bt == 0:
        if poly.area >= GREEN_MIN_M2:
            return [green(poly)]
        stats["verge_m2"] += poly.area
        return []
    if bt == 1:
        return [{"polygon": poly, "front": None, "interior": True, "use": "building", "btype": bt}]
    lots, taken = [], Polygon()
    for L, a, t, n in edges:
        cls = int(street_class_at((a + t * L / 2)[None], graph)[0])
        N, r = nets.sample_side(side_x(bx, [bt], [np.log(L)], [cls]), g)
        if N == 0:
            continue
        w = L / N * np.exp(rng.normal(0, jitter, N))
        w *= L / w.sum()
        D = full_depth(poly, a, t, n, L)
        cap = depth_cap(poly, near, a, t, n, L)
        d = D / 2 if (r > SNAP_RATIO and cap < D) else min(r * D, cap)
        d = max(d, 5.0)
        s = 0.0
        for wi in w:
            p0, p1 = a + t * s, a + t * (s + wi)
            quad = Polygon([p0, p1, p1 + n * d, p0 + n * d])
            lot = _largest(shapely.make_valid(quad.intersection(poly).difference(taken)))
            if not lot.is_empty and lot.area >= MIN_PARCEL_M2:
                lots.append({"polygon": lot, "front": LineString([p0, p1]), "t": t, "n": n, "w": float(wi), "d": float(d),
                             "interior": False, "use": "building", "btype": bt})
                taken = taken.union(lot)
            s += wi
    rest = shapely.make_valid(poly.difference(taken.buffer(0.01)))
    greens = []
    target = float(np.median([q["polygon"].area for q in lots])) if lots else 300.0
    for piece in sorted([q for q in getattr(rest, "geoms", [rest]) if q.geom_type == "Polygon"], key=lambda q: q.area):
        if piece.area > GREEN_MAX_M2:
            for q in split_piece(piece, target):
                lots.append({"polygon": q, "front": None, "interior": True, "use": "building", "btype": bt})
                stats["interior_lots"] += 1
        elif piece.area >= GREEN_MIN_M2:
            greens.append(green(piece))
            stats["interior_green"] += 1
        elif not _merge_into(lots, piece):
            stats["verge_m2"] += piece.area
        else:
            stats["remainder_merged"] += 1
    if not lots and not greens:                      # no street lot at all: the block is one parcel
        return [{"polygon": poly, "front": None, "interior": True, "use": "building", "btype": bt}]
    return merge_slivers(lots, stats) + greens


def _fits(parcel, poly):
    return not poly.is_empty and parcel.buffer(0.05).contains(poly)


def template_buildings(p, cov, n_b, country, ctx, sampler, nets, g, stats):
    """Template footprints (stretched, never clipped) for one building parcel."""
    if cov <= 0 or n_b < 1:
        return []
    poly = p["polygon"]
    target = cov * poly.area
    n_b = int(min(n_b, 3))
    main = target * (MAIN_SHARE if n_b > 1 else 1.0)
    if not p["interior"]:
        fr, sl, sr, rear = (max(float(x), 0.0) for x in sampler.setbacks(country, ctx))
        t, n = p["t"], p["n"]
        p0 = np.asarray(p["front"].coords[0])
        xy = (np.asarray(poly.exterior.coords) - p0) @ np.c_[t, n]   # the clipped lot in its street frame
        x_lo, x_hi = float(xy[:, 0].min()), float(xy[:, 0].max())
        w, d = x_hi - x_lo, float(xy[:, 1].max())
        bw = w - sl - sr
        if bw < MIN_DIM_M:
            sl = sr = 0.0
            bw = w
        room = d - fr - rear
        if room < MIN_DIM_M:
            rear, room = 0.0, d - fr
        if room < MIN_DIM_M:
            fr, room = 0.0, d
        k = nets.sample_shape(shape_x([country], [np.log(max(w, 1))], [np.log(max(d, 1))], [cov], [False], [ctx["cls"]]), g)
        bd = min(main / (FILL[k] * bw), room)
        out = []
        for _ in range(N_SHRINK + 1):
            if bd < MIN_DIM_M or bw < MIN_DIM_M:
                break
            free = w - sl - sr - bw                  # slack along the street: try centre, left, right
            for x0 in (x_lo + sl + free / 2, x_lo + sl, x_lo + sl + free):
                fp = place(k, p0, t, n, x0, fr, bw, bd)
                if _fits(poly, fp):
                    out.append(fp)
                    break
            if out:
                break
            bw, bd = bw * SHRINK, bd * SHRINK
            stats["shrunk"] += 1
        if not out:
            stats["not_fitted"] += 1
            return out
        if n_b > 1:
            side = np.sqrt((target - main) / (n_b - 1))
            back = d - max(rear, 1.0) - side
            if back > fr + bd + 1.0 and side >= MIN_DIM_M:
                for j in range(n_b - 1):
                    sw = min(side, (w - sl - sr) / (n_b - 1))
                    fp = place(0, p0, t, n, x_lo + sl + j * (w - sl - sr) / (n_b - 1), back, sw, side)
                    if _fits(poly, fp):
                        out.append(fp)
        return out
    rect = np.asarray(shapely.minimum_rotated_rectangle(poly).exterior.coords)[:4]
    if np.hypot(*(rect[1] - rect[0])) < np.hypot(*(rect[2] - rect[1])):
        rect = np.roll(rect, -1, 0)                  # long side first
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    W, D = float(np.hypot(*e1)), float(np.hypot(*e2))
    t, n = e1 / W, e2 / D
    s = max(float(sampler.setbacks(country, ctx)[1]), 0.0)
    if min(W, D) - 2 * s < MIN_DIM_M:
        s = 0.0
    k = nets.sample_shape(shape_x([country], [np.log(max(W, 1))], [np.log(max(D, 1))], [cov], [True], [ctx["cls"]]), g)
    ew, ed = W - 2 * s, D - 2 * s
    f = min(np.sqrt(main / (FILL[k] * ew * ed)), 1.0)
    bw, bd = f * ew, f * ed
    for _ in range(N_SHRINK + 1):
        if min(bw, bd) < MIN_DIM_M:
            break
        fp = place(k, rect[0], t, n, (W - bw) / 2, (D - bd) / 2, bw, bd)
        if _fits(poly, fp):
            return [fp]
        bw, bd = bw * SHRINK, bd * SHRINK
        stats["shrunk"] += 1
    stats["not_fitted"] += 1
    return []


def generate_tidy(state, country, nets, sampler, rng, class_widths, jitter, zoning=False, seed=0, method="T"):
    """Subdivide every block, zone (TZ), assign coverage, add template buildings and green parcels to
    ``state`` (strict rules). Returns (parcels, stats)."""
    gr = state.graph
    g = torch.Generator().manual_seed(seed)
    blocks = extract_blocks(gr, class_widths)
    seg = gr.segments()
    near = shapely.MultiLineString([s.reshape(2, 2) for s in seg]).buffer(FRONT_TOL_M) if len(seg) else Polygon()
    stats = {"block_type": [0, 0, 0, 0], "verge_m2": 0.0, "interior_green": 0, "interior_lots": 0, "remainder_merged": 0, "slivers_merged": 0,
             "zoned_green": 0, "shrunk": 0, "not_fitted": 0, "clipped_to_parcel": 0,
             "buildings_proposed": 0, "buildings_accepted": 0, "green_parcels": 0}
    parcels = []
    for b in blocks:
        parcels += subdivide_block(b, near, country, nets, gr, jitter, rng, g, stats)
    polys = [p["polygon"] for p in parcels]
    nbs = parcel_neighbours(polys)
    if zoning:
        lab = np.array([1.0 if p["use"] == "green" else np.nan for p in parcels])
        for i in rng.permutation(len(parcels)):
            if not np.isnan(lab[i]):
                continue
            p = parcels[i]
            x = zone_x([country], [np.log(p["polygon"].area)], [p["interior"]], [p["btype"]], [0.0], [False])
            lab[i] = float(rng.random() < nets.p_green(x))
            if lab[i]:
                p["use"] = "green"
                stats["zoned_green"] += 1
    ctx = context(polys, gr, blocks, [p["interior"] for p in parcels])
    built = np.array([p["use"] == "building" for p in parcels])
    cov = np.where(built, np.nan, 0.0)
    nbld = np.zeros(len(parcels), int)
    for _ in range(1 + GIBBS_SWEEPS):
        for i in rng.permutation(len(parcels)):
            if not built[i]:
                continue
            known = cov[nbs[i]]
            known = known[~np.isnan(known)]
            cov[i], nbld[i] = sampler.coverage(country, {k: v[i] for k, v in ctx.items()}, known)
    for i, p in enumerate(parcels):
        ci = {k: v[i] for k, v in ctx.items()}
        p.update(parcel_id=i, status="generated", method=method, coverage_target=float(cov[i]),
                 n_buildings_target=int(nbld[i]), cls=int(ci["cls"]))
        if p["use"] == "green":
            q = _largest(shapely.make_valid(p["polygon"].buffer(-0.5)))
            if not q.is_empty and state.try_add(q, 1):
                stats["green_parcels"] += 1
            continue
        for fp in template_buildings(p, cov[i], nbld[i], country, ci, sampler, nets, g, stats):
            stats["buildings_proposed"] += 1
            if state.try_add(fp, 0):
                stats["buildings_accepted"] += 1
                p.setdefault("building_idx", []).append(len(state.buildings) - 1)
    return parcels, stats
