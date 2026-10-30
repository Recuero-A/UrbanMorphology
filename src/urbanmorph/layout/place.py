"""Placing buildings and green spaces around a street network (Milestone 6, Stages B-D).

Two placers share the SAME geometry rules and bookkeeping:
  * P2 ``ObjectPlacer`` (neural, models/placer.py): where / what learned from real tiles.
  * P1 ``StatisticalPlacer`` (no neural network): object counts drawn from real tiles, positions
    drawn from the real density of objects as a function of distance to the nearest street,
    attributes resampled from real objects. It answers "what does the network add?".

Programmed validity rules (thresholds measured on real training tiles, see docs/MILESTONE_6.md):
  1. every polygon is clipped to the tile and made valid (largest part kept);
  2. a building is rejected if more than 25 % of it lies on the road surface
     (real data: 14.6 % of buildings touch the class-width road surface, 6.7 % by more than 10 %);
  3. a building is rejected if it overlaps an earlier building by more than 5 % of the smaller one
     (real buildings touch but essentially never overlap);
  4. a green space is trimmed to land not covered by roads or buildings, and rejected if under
     20 m2 remain or it overlaps earlier green by more than 5 %.
Up to 4 candidate attribute sets are tried per location; every rule firing is counted.

STRICT rules (Milestone 7, ``strict=True``): nothing may lie on top of anything else.
  2'. a building is TRIMMED to the land outside the road surface (flush with the road edge) and
      rejected if it loses more than 20 % of its area;
  3'. buildings may touch but not overlap by more than 1 % of the smaller one;
  4'. green spaces likewise may overlap other green by at most 1 %;
  5.  a building is rejected if any interior angle is below 45 deg (after removing < 0.5 m detail).
      Real OSM footprints have such angles in about 0.1 % of buildings, so the rule removes artefacts
      of the shape representation, not real shapes.
Blocks (Stage B) are not modelled as parcels: the road surface simply removes land.
"""

from __future__ import annotations

import numpy as np
import shapely
import torch

from urbanmorph.graphs.road_graph import TILE_M, road_polygons, rasterize_graph
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2, decode, polygon_pixels, street_axis_and_distance
from urbanmorph.layout.placer_data import N, canonical, rotated_crops, scalars, shape_from_unbounded, street_context

TILE = shapely.box(0, 0, TILE_M, TILE_M)
MAX_ON_ROAD = 0.25
MAX_OVERLAP = 0.05
N_CANDIDATES = 4


def _largest(poly):
    if poly.is_empty:
        return poly
    if poly.geom_type == "Polygon":
        return poly
    parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else shapely.Polygon()


MIN_ANGLE_DEG = 45.0
MAX_TRIM = 0.20
STRICT_OVERLAP = 0.01


class LayoutState:
    def __init__(self, graph, class_widths, strict: bool = False):
        self.strict = strict
        self.graph = graph
        self.road = rasterize_graph(graph, class_widths=class_widths).astype(bool)
        self.surface = road_polygons(graph, class_widths=class_widths)
        shapely.prepare(self.surface)
        self.buildings: list = []
        self.green: list = []
        self.bcanv = np.zeros(N * N, bool)
        self.gcanv = np.zeros(N * N, bool)
        self.phase = 0
        self.done = False
        self.fails = 0
        self.rules = {"accepted_building": 0, "accepted_green": 0, "rejected_on_road": 0,
                      "rejected_overlap": 0, "rejected_too_small": 0, "green_trimmed": 0, "location_given_up": 0,
                      "building_trimmed": 0, "rejected_acute": 0}

    def canvas(self) -> np.ndarray:
        return np.stack([self.road, self.bcanv.reshape(N, N), self.gcanv.reshape(N, N),
                         np.full((N, N), self.phase > 0)])

    def try_add(self, poly, typ) -> bool:
        poly = _largest(shapely.make_valid(poly).intersection(TILE))
        if typ == 0:
            if poly.is_empty or poly.area < MIN_BUILDING_M2:
                self.rules["rejected_too_small"] += 1
                return False
            if self.strict:
                land = _largest(shapely.make_valid(poly.difference(self.surface)))
                if land.is_empty or land.area < (1 - MAX_TRIM) * poly.area:
                    self.rules["rejected_on_road"] += 1
                    return False
                if land.area < poly.area - 1e-6:          # any overlap with the road is cut off
                    self.rules["building_trimmed"] += 1
                    poly = land
                from urbanmorph.layout.frontage import interior_angles
                if interior_angles(poly).min() < MIN_ANGLE_DEG:
                    self.rules["rejected_acute"] += 1
                    return False
            elif poly.intersection(self.surface).area > MAX_ON_ROAD * poly.area:
                self.rules["rejected_on_road"] += 1
                return False
            max_ov = STRICT_OVERLAP if self.strict else MAX_OVERLAP
            if self.buildings:
                arr = np.array(self.buildings, dtype=object)
                hit = shapely.intersects(arr, poly)
                if hit.any():
                    ov = shapely.area(shapely.intersection(arr[hit], poly)) / np.minimum(shapely.area(arr[hit]), poly.area)
                    if (ov > max_ov).any():
                        self.rules["rejected_overlap"] += 1
                        return False
            self.buildings.append(poly)
            self.bcanv[polygon_pixels(poly)] = True
            self.rules["accepted_building"] += 1
            return True
        land = poly.difference(self.surface)
        if self.buildings:
            arr = np.array(self.buildings, dtype=object)
            hit = shapely.intersects(arr, land)
            if hit.any():
                land = land.difference(shapely.union_all(arr[hit]))
        land = _largest(shapely.make_valid(land))
        if land.is_empty or land.area < MIN_GREEN_M2:
            self.rules["rejected_too_small"] += 1
            return False
        if land.area < 0.99 * poly.area:
            self.rules["green_trimmed"] += 1
        if self.green:
            arr = np.array(self.green, dtype=object)
            hit = shapely.intersects(arr, land)
            if hit.any():
                ov = shapely.area(shapely.intersection(arr[hit], land)) / np.minimum(shapely.area(arr[hit]), land.area)
                if (ov > (STRICT_OVERLAP if self.strict else MAX_OVERLAP)).any():
                    self.rules["rejected_overlap"] += 1
                    return False
        self.green.append(land)
        self.gcanv[polygon_pixels(land)] = True
        self.rules["accepted_green"] += 1
        return True

    def next_phase(self):
        self.fails = 0
        if self.phase == 0:
            self.phase = 1
        else:
            self.done = True


def _polygon_from(anchor, axis, phi, size, shape):
    L, W = np.exp(size[0]), np.exp(size[1])
    W = min(W, L)
    return decode(anchor, axis + phi, L, W, shape_from_unbounded(shape))


@torch.no_grad()
def place_neural(model, graphs, class_widths, device="cpu", seed=0, max_buildings=2500, max_green=300,
                 max_fails=40, strict=False, states=None) -> list[LayoutState]:
    """Grow buildings then green spaces on every graph with the neural placer (lock-step batches).
    ``states`` continues existing layouts (e.g. after the frontage phase) instead of starting empty."""
    model.eval()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = states if states is not None else [LayoutState(g, class_widths, strict) for g in graphs]
    while True:
        act = [s for s in states if not s.done]
        if not act:
            break
        canv = np.stack([s.canvas() for s in act])
        phase = torch.tensor([float(s.phase) for s in act], device=device)
        counts = torch.tensor([[len(s.buildings) / 1000, len(s.green) / 50] for s in act], device=device)
        stop_logit, loc, feat = model.where(torch.from_numpy(canv).to(device), phase, counts)
        stop = (torch.rand(len(act), generator=gen) < torch.sigmoid(stop_logit).cpu()).numpy()
        cells = torch.multinomial(torch.softmax(loc, -1).cpu(), 1, generator=gen).squeeze(1).numpy()
        go = []
        for i, s in enumerate(act):
            cap = max_buildings if s.phase == 0 else max_green
            if stop[i] or (len(s.buildings) if s.phase == 0 else len(s.green)) >= cap or s.fails >= max_fails:
                s.next_phase()
            else:
                go.append(i)
        if not go:
            continue
        r, c = np.divmod(cells[go], N)
        anchors = np.c_[(c + rng.random(len(go))) * 4.0, TILE_M - (r + rng.random(len(go))) * 4.0]
        sts = [act[i] for i in go]
        ax_d_c = [street_context(anchors[j:j + 1], s.graph) for j, s in enumerate(sts)]
        axis = np.array([a[0][0] for a in ax_d_c])
        dist = np.array([a[1][0] for a in ax_d_c])
        cls = np.array([a[2][0] for a in ax_d_c])
        crops = rotated_crops(canv[go], anchors, axis)
        ph = np.array([float(s.phase) for s in sts])
        scal = scalars(anchors, dist, cls, ph, [len(s.buildings) for s in sts], [len(s.green) for s in sts])
        cell_t = torch.from_numpy(cells[go]).to(device)
        feat_at = feat[torch.as_tensor(go, device=device)].flatten(2)
        feat_at = feat_at.gather(2, cell_t[:, None, None].expand(-1, feat_at.shape[1], 1)).squeeze(2)
        rep = lambda x: x.repeat_interleave(N_CANDIDATES, 0)
        phi, size, shape = model.sample_what(rep(torch.from_numpy(crops).to(device)), rep(feat_at),
                                             rep(torch.from_numpy(scal).to(device)),
                                             rep(torch.from_numpy(ph.astype(np.float32)).to(device)), gen)
        for j, s in enumerate(sts):
            ok = False
            for m in range(N_CANDIDATES):
                q = j * N_CANDIDATES + m
                if s.try_add(_polygon_from(anchors[j], axis[j], phi[q], size[q], shape[q]), s.phase):
                    ok = True
                    break
            if ok:
                s.fails = 0
            else:
                s.fails += 1
                s.rules["location_given_up"] += 1
    return states


class StatisticalPlacer:
    """P1: no neural network. Fitted from training tiles only."""

    BINS = np.arange(0, 124, 4.0)

    def __init__(self, tiles: list):
        from urbanmorph.layout.placer_data import shape_to_unbounded
        self.counts = np.array([[(t["objects"]["type"] == 0).sum(), (t["objects"]["type"] == 1).sum()] for t in tiles])
        px = (np.arange(N) + 0.5) * 4.0
        X, Y = np.meshgrid(px, TILE_M - px)
        self.pix_xy = np.c_[X.ravel(), Y.ravel()]
        hist_pix = np.zeros(len(self.BINS) - 1)
        hist_obj = np.zeros((2, len(self.BINS) - 1))
        self.pool = {0: [], 1: []}
        for t in tiles:
            seg = t["graph"].segments()
            _, dpix = street_axis_and_distance(self.pix_xy, seg)
            hist_pix += np.histogram(np.minimum(dpix, 119.9), self.BINS)[0]
            o = t["objects"]
            axis, d, _ = street_context(o["anchor"], t["graph"])
            phi, ratios = canonical(o["theta"], o["ratios"], axis)
            sh = shape_to_unbounded(ratios)
            for ty in (0, 1):
                m = o["type"] == ty
                hist_obj[ty] += np.histogram(np.minimum(d[m], 119.9), self.BINS)[0]
                self.pool[ty] += [(p, np.log(L), np.log(W), s) for p, L, W, s in
                                  zip(phi[m], o["length"][m], o["width"][m], sh[m])]
        self.intensity = hist_obj / np.maximum(hist_pix, 1)[None]

    def place(self, graphs, class_widths, seed=0, max_fails=40, strict=False) -> list[LayoutState]:
        rng = np.random.default_rng(seed)
        states = []
        for g in graphs:
            s = LayoutState(g, class_widths, strict)
            nb, ng = self.counts[rng.integers(len(self.counts))]
            seg = g.segments()
            _, dpix = street_axis_and_distance(self.pix_xy, seg)
            binix = np.clip(np.digitize(np.minimum(dpix, 119.9), self.BINS) - 1, 0, len(self.BINS) - 2)
            for ty, target in ((0, nb), (1, ng)):
                s.phase = ty
                w = self.intensity[ty][binix]
                w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                placed, attempts = 0, 0
                while placed < target and attempts < 3 * target + max_fails:
                    attempts += 1
                    cell = rng.choice(len(w), p=w)
                    anchor = self.pix_xy[cell] + rng.uniform(-2, 2, 2)
                    axis = street_context(anchor[None], g)[0][0]
                    ok = False
                    for _ in range(N_CANDIDATES):
                        phi, lL, lW, sh = self.pool[ty][rng.integers(len(self.pool[ty]))]
                        if s.try_add(_polygon_from(anchor, axis, phi, (lL, lW), sh), ty):
                            ok = True
                            break
                    placed += ok
                    if not ok:
                        s.rules["location_given_up"] += 1
            s.done = True
            states.append(s)
        return states


@torch.no_grad()
def place_frontage(model, graphs, class_widths, device="cpu", seed=0, strict=True, states=None,
                   relative=False) -> list[LayoutState]:
    """Milestone 7, phase 1: buildings in sequence along every street side (layout/frontage.py order).

    Each step samples STOP or (gap, setback, orientation, size, shape) relative to the street and the
    previous building; the anchor is placed at s = s_prev + gap along the street and ``setback`` metres
    to its left. Rejected buildings (rules) still advance the position, so the row continues.
    Afterwards call ``place_neural(..., states=...)`` for interior buildings and green spaces."""
    from urbanmorph.layout.frontage import frontages, point_at, tangent
    from urbanmorph.layout.frontage_data import MAX_PER_FRONTAGE, step_scalars, time_crops
    model.eval()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    states = states if states is not None else [LayoutState(g, class_widths, strict) for g in graphs]
    cur = [{"fr": frontages(s.graph), "i": 0, "s": 0.0, "p1": None, "p2": None, "k": 0} for s in states]

    def advance(c):
        c.update(i=c["i"] + 1, s=0.0, p1=None, p2=None, k=0)

    while True:
        act = [j for j, c in enumerate(cur) if c["i"] < len(c["fr"])]
        if not act:
            break
        centres, angles, scal, crops = [], [], [], []
        for j in act:
            c, st = cur[j], states[j]
            f = c["fr"][c["i"]]
            a = tangent(f["line"], c["s"])
            p = point_at(f["line"], c["s"])
            centres.append(p)
            angles.append(a)
            scal.append(step_scalars(f["cls"], f["length"], c["s"], c["k"], c["p1"], c["p2"]))
            crops.append(time_crops(st.road.ravel(), np.where(st.bcanv, 0.0, np.inf), p[None], np.array([a]), [1.0])[0])
        T = lambda x: torch.from_numpy(np.stack(x)).to(device)
        stop, gapd, phi, size, shape = model.sample(T(crops), T(scal), gen)
        for q, j in enumerate(act):
            c, st = cur[j], states[j]
            f = c["fr"][c["i"]]
            if stop[q] or c["k"] >= MAX_PER_FRONTAGE:
                advance(c)
                continue
            g_log, d_log, ph = gapd[q, 0], gapd[q, 1], float(phi[q])
            if relative and c["p1"] is not None:      # residuals w.r.t. the previous building on this frontage
                g_log, d_log = g_log + c["p1"][0], d_log + c["p1"][1]
                ph = float((ph + c["p1"][2] + np.pi / 2) % np.pi - np.pi / 2)
            s_new = c["s"] + max(np.exp(g_log) - 1, 0.0)
            if s_new > f["length"]:
                advance(c)
                continue
            a = tangent(f["line"], s_new)
            d = np.exp(d_log)
            anchor = point_at(f["line"], s_new) + d * np.array([-np.sin(a), np.cos(a)])   # left of the street
            st.try_add(_polygon_from(anchor, a, ph, size[q], shape[q]), 0)
            par = (g_log, d_log, ph, size[q, 0], size[q, 1])
            c.update(s=s_new, p2=c["p1"], p1=par, k=c["k"] + 1)
    return states
