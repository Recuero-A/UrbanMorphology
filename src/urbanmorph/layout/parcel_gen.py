"""Milestone 8, Experiments C and D: GENERATED parcels and parcel-conditioned buildings.

Every parcel produced here has status "generated". It is never cadastral truth; observed parcels
(layout/parcels.py, scripts/m8_parcels.py) are used only to fit the distributions below (TRAIN tiles of
Spain and the US) and to evaluate (validation tiles).

Pipeline per tile: street network -> blocks (layout/blocks.py) -> parcels -> buildings, then green by
the S fallback. Decision types:

PROGRAMMED rules
  * subdivision: each block edge (block outline simplified at 1 m) that is >= 6 m long, not on the tile
    edge and within 12 m of a street centreline is a street edge. Street edges are processed longest
    first. Per edge one TYPICAL lot (w0, d0) is drawn (length-bias correction: of 8 draws, one is
    chosen with probability proportional to w0, because an edge holds ~ L / w0 lots; without it
    narrow lots are over-represented: generated median width 8 m vs 13 m observed on train tiles);
    each lot along the edge then has
    w = w0 exp(N(0, s_w)), d = d0 exp(N(0, s_d)), where s_w, s_d are the measured spread between
    neighbouring observed street parcels (fitted; Spain 0.66 / 0.37, US 0.53 / 0.30 in log units).
    The depth is capped at half the block's depth when the opposite side is also a street (else the
    full depth), measured along the inward normal from the edge midpoint. Lots are rectangles
    extruded inwards, clipped to the block and to the lots already cut; a last piece shorter than
    w / 2 is merged into the previous lot. The block remainder becomes interior parcels (>= 20 m2).
    Slivers are merged into the neighbouring parcel with the longest shared boundary: a street lot
    keeping < 40 % of its nominal w x d rectangle after clipping, or an interior piece smaller than half
    the block's median street lot (corner clipping otherwise left 30 % of parcels < 60 m2 vs 2-12 %
    observed, measured on train tiles);
  * coverage is assigned sequentially in random order: each parcel's (coverage, number of buildings)
    is drawn GIVEN the mean coverage of its already-assigned neighbours (the user's density idea;
    observed Moran's I 0.45-0.65), followed by GIBBS_SWEEPS = 2 Gibbs sweeps in which every parcel is
    re-drawn given all its neighbours (one pass alone gave Moran's I ~ 0.2 on train tiles);
  * main building of a street lot: facade parallel to the lot's street edge, at the front setback,
    between the two side setbacks; its depth follows from coverage x parcel area / width, capped by
    the lot depth minus front and rear setbacks. Extra buildings (2nd, 3rd) take the remaining area,
    at the rear of the lot. Interior parcels: one building centred in the parcel, aligned with its
    minimum rotated rectangle, scaled to the target area. 75 % of a multi-building parcel's built
    area goes to the main building (programmed; not measurable from our parcel table);
  * shape: a library shape (layout/shape_library.py) drawn with its training frequency, random symmetry;
POST-GENERATION corrections (counted): building grown (<= 1.5x per side) towards its target area and
  clipped to its parcel; if rejected, retried unscaled, at 0.8x, then as its plain rectangle; the strict
  rules of
  LayoutState (trim to the road edge, overlap, acute angles, minimum area).
LEARNED from observed parcels
  * C (``EmpiricalSampler``): per country, real (w, d) pairs; (coverage, buildings) records binned by
    the neighbours' mean coverage (5 quantile bins; separate for street and interior parcels);
    front / side / rear setbacks of one-building parcels. Resampled, no network.
  * D (``NetSampler``, models/parcel_net.py): the same quantities from a neural mixture-density model
    conditioned on context (country, street class, block size, neighbours' coverage).
Not modelled: buildings spanning several parcels (16-33 % in Madrid, Valencia, New York).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon

from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.objects import decode
from urbanmorph.layout.parcels import FRONT_TOL_M, MIN_PARCEL_M2, parcel_neighbours

COUNTRIES = ("spain", "united_states")
TILE = shapely.box(0, 0, TILE_M, TILE_M)
MIN_EDGE_M = 6.0
N_COV_BINS = 5
MAIN_SHARE = 0.75
MIN_DIM_M = 3.0
GIBBS_SWEEPS = 2


def _largest(g):
    if g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def street_class_at(points, graph) -> np.ndarray:
    """Class of the nearest street centreline segment for each point (2 = local when there is none)."""
    seg = graph.segments()
    pts = np.atleast_2d(points)
    if not len(seg):
        return np.full(len(pts), 2)
    A, B = seg[:, :2], seg[:, 2:]
    s = B - A
    lam = np.clip(((pts[:, None] - A[None]) * s[None]).sum(-1) / np.maximum((s ** 2).sum(1), 1e-9)[None], 0, 1)
    d = np.hypot(*(pts[:, None] - (A[None] + lam[..., None] * s[None])).transpose(2, 0, 1))
    return graph.classes()[d.argmin(1)]


def block_area_at(points, blocks) -> np.ndarray:
    out = np.full(len(points), np.nan)
    for b in blocks:
        inside = shapely.contains_xy(b["polygon"], points[:, 0], points[:, 1])
        out[inside & np.isnan(out)] = b["area_m2"]
    return out


def context(parcels, graph, blocks, interior) -> dict:
    """Context features shared by fitting (observed parcels) and generation (generated parcels)."""
    rp = np.array([p.representative_point().coords[0] for p in parcels]).reshape(-1, 2)
    ba = block_area_at(rp, blocks) if len(rp) else np.zeros(0)
    area = np.array([p.area for p in parcels])
    return {"cls": street_class_at(rp, graph) if len(rp) else np.zeros(0, int),
            "log_block_area": np.log(np.where(np.isnan(ba), area, ba)), "interior": np.asarray(interior, bool)}


def lot_jitter(rec, country) -> tuple[float, float]:
    """Spread (log units) of frontage width and depth between neighbouring uncut street parcels:
    sd(log ratio) / sqrt(2), i.e. the spread of each lot around a shared typical value."""
    ok = ((~rec.interior) & (~rec.cut) & (rec.frontage_m > 1) & np.isfinite(rec.depth_m)).to_numpy()
    pos = {g: k for k, g in enumerate(rec.index)}
    W, D, C = rec.frontage_m.to_numpy(), rec.depth_m.to_numpy(), (rec.country == country).to_numpy()
    lw, ld = [], []
    for i in np.flatnonzero(ok & C):
        for g in rec.nb.iat[i]:
            j = pos.get(g)
            if j is not None and j > i and ok[j]:
                lw.append(np.log(W[i] / W[j]))
                ld.append(np.log(D[i] / D[j]))
    return float(np.std(lw) / np.sqrt(2)), float(np.std(ld) / np.sqrt(2))


# ---------------------------------------------------------------- C: empirical sampler
class EmpiricalSampler:
    """Resamples real records (TRAIN observed parcels) per country; see module docstring."""

    def __init__(self, rec, rng=None):
        self.rng = rng or np.random.default_rng(0)
        self.by = {}
        for c in COUNTRIES:
            r = rec[rec.country == c]
            st = r[~r.interior & ~r.cut & np.isfinite(r.depth_m)]
            one = st[st.n_buildings == 1]
            cov = {}
            for inter in (False, True):
                q = r[(r.interior == inter) & ~r.cut]
                edges = np.nanquantile(q.nb_cov, np.linspace(0, 1, N_COV_BINS + 1)[1:-1])
                bins = np.digitize(q.nb_cov.fillna(-1), edges)
                cov[inter] = {"edges": edges, "all": q[["coverage", "n_buildings"]].to_numpy(),
                              "bins": [q[bins == k][["coverage", "n_buildings"]].to_numpy() for k in range(N_COV_BINS)]}
            self.by[c] = {"lots": st[["frontage_m", "depth_m"]].to_numpy(), "cov": cov, "jitter": lot_jitter(rec, c),
                          "front": one.front_setback_m.dropna().to_numpy(), "side": one.side_setback_m.dropna().to_numpy(),
                          "rear": one.rear_setback_m.dropna().to_numpy()}

    def _pick(self, a):
        return a[self.rng.integers(len(a))]

    def lot(self, country, ctx):
        """The typical (w0, d0) of one street edge."""
        return tuple(self._pick(self.by[country]["lots"]))

    def jitter(self, country):
        return self.by[country]["jitter"]

    def coverage(self, country, ctx, nb_known):
        c = self.by[country]["cov"][bool(ctx["interior"])]
        if not len(nb_known):
            return tuple(self._pick(c["all"]))
        b = c["bins"][int(np.digitize(np.mean(nb_known), c["edges"]))]
        return tuple(self._pick(b if len(b) else c["all"]))

    def setbacks(self, country, ctx):
        s = self.by[country]
        return self._pick(s["front"]), self._pick(s["side"]), self._pick(s["side"]), self._pick(s["rear"])


# ---------------------------------------------------------------- subdivision
def street_edges(block, near_street):
    c = np.asarray(shapely.simplify(block.exterior, 1.0).coords)
    out = []
    for a, b in zip(c[:-1], c[1:]):
        L = float(np.hypot(*(b - a)))
        if L < MIN_EDGE_M:
            continue
        mid = shapely.Point((a + b) / 2)
        if mid.distance(TILE.exterior) < 1.0 or not near_street.contains(mid):
            continue
        t = (b - a) / L
        n = np.array([-t[1], t[0]])
        if not block.contains(shapely.Point((a + b) / 2 + n * 1.0)):
            n = -n
        out.append((L, a, t, n))
    return sorted(out, key=lambda e: -e[0])


def depth_cap(block, near_street, a, t, n, L) -> float:
    """Half the block depth along the inward normal from the edge midpoint if the far side is a street,
    else the full depth."""
    mid = a + t * L / 2
    r = np.arange(0.5, 600, 1.0)
    pts = mid[None] + r[:, None] * n[None]
    inside = shapely.contains_xy(block, pts[:, 0], pts[:, 1])
    out = np.flatnonzero(~inside)
    depth = float(r[out[0]]) if len(out) else 600.0
    far = shapely.Point(mid + n * depth)
    return depth / 2 if near_street.contains(far) else depth


def subdivide(block, near_street, country, sampler, graph, blocks, stats):
    lots, taken = [], Polygon()
    for L, a, t, n in street_edges(block, near_street):
        s = 0.0
        ctx = {"cls": int(street_class_at((a + t * L / 2)[None], graph)[0]), "log_block_area": float(np.log(block.area)),
               "interior": False}
        cand = np.array([sampler.lot(country, ctx) for _ in range(8)], float)
        w0, d0 = cand[sampler.rng.choice(8, p=cand[:, 0] / cand[:, 0].sum())]
        sw, sd = sampler.jitter(country)
        cap = depth_cap(block, near_street, a, t, n, L)
        while s < L - 2.0:
            w = max(float(w0 * np.exp(sampler.rng.normal(0, sw))), 3.0)
            d = min(max(float(d0 * np.exp(sampler.rng.normal(0, sd))), 5.0), max(cap, 5.0))
            if L - (s + w) < 0.5 * w:
                w = L - s
            p0, p1 = a + t * s, a + t * (s + w)
            quad = Polygon([p0, p1, p1 + n * d, p0 + n * d])
            lot = _largest(shapely.make_valid(quad.intersection(block).difference(taken)))
            if not lot.is_empty and lot.area >= MIN_PARCEL_M2:
                lots.append({"polygon": lot, "front": LineString([p0, p1]), "t": t, "n": n, "w": w, "d": d, "interior": False})
                taken = taken.union(lot)
            s += w
    rest = shapely.make_valid(block.difference(taken.buffer(0.01)))
    for g in getattr(rest, "geoms", [rest]):
        if g.geom_type == "Polygon" and g.area >= MIN_PARCEL_M2:
            lots.append({"polygon": g, "front": None, "interior": True})
        elif g.geom_type == "Polygon":
            stats["unparcelled_m2"] += g.area
    return merge_slivers(lots, stats)


def merge_slivers(lots, stats):
    street = [p["polygon"].area for p in lots if not p["interior"]]
    med = np.median(street) if street else 0.0
    small = lambda p: (p["polygon"].area < 0.4 * p["w"] * p["d"]) if not p["interior"] else p["polygon"].area < 0.5 * med
    order = sorted(range(len(lots)), key=lambda i: lots[i]["polygon"].area)
    alive = [True] * len(lots)
    for i in order:
        if not small(lots[i]) or sum(alive) < 2:
            continue
        pi = lots[i]["polygon"]
        best, best_len = None, 0.0
        for j, q in enumerate(lots):
            if j != i and alive[j] and q["polygon"].distance(pi) < 0.1:
                shared = q["polygon"].buffer(0.1).intersection(pi.exterior).length
                if shared > best_len:
                    best, best_len = j, shared
        if best is None or best_len < 1.0:
            continue
        merged = _largest(shapely.make_valid(lots[best]["polygon"].union(pi.buffer(0.01)).buffer(-0.01)))
        if merged.is_empty or merged.area < lots[best]["polygon"].area:
            continue
        lots[best]["polygon"] = merged
        alive[i] = False
        stats["slivers_merged"] = stats.get("slivers_merged", 0) + 1
    return [p for p, a in zip(lots, alive) if a]


# ---------------------------------------------------------------- buildings
def _shape(lib, rng):
    k = rng.choice(len(lib.shapes), p=lib.counts / lib.counts.sum())
    return lib.ratios(4 * k + rng.integers(4))


def _footprint(centre, axis, along, across, ratios):
    if along >= across:
        return decode(centre, axis, along, across, ratios)
    return decode(centre, axis + np.pi / 2, across, along, ratios)


def parcel_buildings(p, cov, n_b, country, ctx, sampler, lib, rng):
    """Footprints (before validity rules) for one parcel."""
    if cov <= 0 or n_b < 1:
        return []
    poly = p["polygon"]
    target = cov * poly.area
    n_b = int(min(n_b, 3))
    if p["interior"]:
        rect = shapely.minimum_rotated_rectangle(poly)
        c = np.asarray(rect.exterior.coords)[:4]
        e1, e2 = c[1] - c[0], c[2] - c[1]
        Lm, Wm = np.hypot(*e1), np.hypot(*e2)
        k = np.sqrt(min(target / max(Lm * Wm, 1e-6), 1.0))
        centre = np.asarray(poly.representative_point().coords[0]) if not poly.contains(poly.centroid) else np.asarray(poly.centroid.coords[0])
        return [_footprint(centre, np.arctan2(e1[1], e1[0]), k * Lm, k * Wm, _shape(lib, rng))]
    f, sl, sr, rear = (max(float(x), 0.0) for x in sampler.setbacks(country, ctx))
    w, d, t, n = p["w"], p["d"], p["t"], p["n"]
    axis = np.arctan2(t[1], t[0])
    p0 = np.asarray(p["front"].coords[0])
    main = target * (MAIN_SHARE if n_b > 1 else 1.0)
    bw = w - sl - sr
    if bw < MIN_DIM_M:
        sl = sr = 0.0
        bw = w
    room = d - f - rear
    if room < MIN_DIM_M:
        rear, room = 0.0, d - f
    if room < MIN_DIM_M:
        f, room = 0.0, d
    bd = min(main / bw, room)
    if bd < MIN_DIM_M:
        return []
    out = [_footprint(p0 + t * (sl + bw / 2) + n * (f + bd / 2), axis, bw, bd, _shape(lib, rng))]
    if n_b > 1:
        a2 = (target - main) / (n_b - 1)
        side = np.sqrt(a2)
        back = d - max(rear, 1.0) - side / 2
        if back - side / 2 > f + bd + 1.0 and side >= MIN_DIM_M:
            for j in range(n_b - 1):
                along = sl + (j + 0.5) * bw / (n_b - 1)
                out.append(_footprint(p0 + t * along + n * back, axis, min(side, bw / (n_b - 1)), side, lib.ratios(0)))
    return out


def generate_parcels(state, country, sampler, lib, rng, class_widths, method="C"):
    """Subdivide every block of ``state.graph``, assign coverage, add buildings to ``state`` (strict
    rules). Returns (parcels, stats); every parcel dict has status 'generated'."""
    g = state.graph
    blocks = extract_blocks(g, class_widths)
    seg = g.segments()
    near = (shapely.MultiLineString([sg.reshape(2, 2) for sg in seg]).buffer(FRONT_TOL_M) if len(seg) else Polygon())
    stats = {"unparcelled_m2": 0.0, "clipped_to_parcel": 0, "buildings_proposed": 0, "buildings_accepted": 0}
    parcels = []
    for b in blocks:
        parcels += subdivide(b["polygon"], near, country, sampler, g, blocks, stats)
    polys = [p["polygon"] for p in parcels]
    ctx = context(polys, g, blocks, [p["interior"] for p in parcels])
    nbs = parcel_neighbours(polys)
    cov = np.full(len(parcels), np.nan)
    nbld = np.zeros(len(parcels), int)
    for sweep in range(1 + GIBBS_SWEEPS):
        for i in rng.permutation(len(parcels)):
            known = cov[nbs[i]]
            known = known[~np.isnan(known)]
            cov[i], nbld[i] = sampler.coverage(country, {k: v[i] for k, v in ctx.items()}, known)
    for i, p in enumerate(parcels):
        ci = {k: v[i] for k, v in ctx.items()}
        p.update(parcel_id=i, status="generated", method=method, coverage_target=float(cov[i]),
                 n_buildings_target=int(nbld[i]), cls=int(ci["cls"]))
        fps = parcel_buildings(p, cov[i], nbld[i], country, ci, sampler, lib, rng)
        for k, fp in enumerate(fps):
            stats["buildings_proposed"] += 1
            q = _largest(shapely.make_valid(fp.intersection(p["polygon"])))
            # correction: grow the footprint (<= 1.5x per side in total) until its clipped area reaches the
            # target share of this building (library shapes do not fill their rectangle; merged slivers)
            want = cov[i] * p["polygon"].area * ((MAIN_SHARE if len(fps) > 1 else 1.0) if k == 0 else
                                                  (1 - MAIN_SHARE) / max(len(fps) - 1, 1))
            f = 1.0
            for _ in range(3):
                if q.is_empty or q.area >= 0.97 * want or f >= 1.5:
                    break
                f = min(f * np.sqrt(want / max(q.area, 1.0)), 1.5)
                q = _largest(shapely.make_valid(affinity.scale(fp, f, f, origin=fp.centroid).intersection(p["polygon"])))
                stats["grown_to_target"] = stats.get("grown_to_target", 0) + 1
            if q.area < 0.99 * fp.area * f * f:
                stats["clipped_to_parcel"] += 1
            # if the rules reject it (mostly acute angles created by clipping at oblique parcel corners), try
            # the unscaled footprint, 0.8x, then its plain rectangle, each clipped to the parcel (counted)
            clip = lambda g_: _largest(shapely.make_valid(g_.intersection(p["polygon"])))
            tries = [q, clip(fp), clip(affinity.scale(fp, 0.8, 0.8, origin=fp.centroid)), clip(shapely.minimum_rotated_rectangle(fp))]
            for n_try, cand in enumerate(tries):
                if cand.is_empty:
                    continue
                if state.try_add(cand, 0):
                    stats["buildings_accepted"] += 1
                    stats["accepted_after_retry"] = stats.get("accepted_after_retry", 0) + int(n_try > 0)
                    p.setdefault("building_idx", []).append(len(state.buildings) - 1)
                    break
    return parcels, stats
