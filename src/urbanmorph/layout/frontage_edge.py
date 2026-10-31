"""Milestone 8, Experiment B1: frontage generation parameterised by building EDGES.

Phase 1 diagnosis (MILESTONE_8_PLAN.md §1.1): in Milestone 7, 77 % of overlap rejections hit the
previous building on the same frontage, because the gap was predicted between building CENTRES
independently of each building's length along the street. B1 changes only the parameterisation and
the correction step; the network, data and training budget stay as in M7.

Per frontage building k (real data -> targets; generation -> placement):
  * [a_k, b_k]: the building's extent along the street (projection of its outline on the frontage line);
  * gap g_k = a_k - b_{k-1}, SIGNED: 0 = attached (shared wall), > 0 = space, < 0 = staggered
    (overlapping extents at different depths; 31-78 % of real consecutive pairs). Modelled as
    asinh(g / 2 m), so attached and staggered cases need no special rule;
  * front distance d_k = distance from the street centreline to the outline, modelled as log(d + 1 m);
  * orientation relative to the street at the building, size (log L, log W);
  * shape: a learned Bernoulli "exactly the minimum rotated rectangle" (22-53 % of real buildings)
    or the 20-ray shape otherwise.
Buildings in a frontage are ordered by a_k. Placement puts the outline so that its extent starts at
b_{k-1} + g_k and its front lies at d_k from the centreline (two correction iterations, so curved
streets are handled). Programmed correction: if the rules reject it, it is shrunk about its anchor
(0.9, 0.8, 0.7) before being rejected; every shrink is counted.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point

from urbanmorph.layout.frontage import assign, point_at, tangent
from urbanmorph.layout.frontage_data import MAX_PER_FRONTAGE, N_FSCAL, step_scalars, time_crops
from urbanmorph.layout.objects import decode
from urbanmorph.layout.placer_data import N, canonical, shape_to_unbounded

RECT_MIN_RATIO = 0.95
SHRINK = (0.9, 0.8, 0.7)


def gap_to_z(g):
    return np.arcsinh(np.asarray(g) / 2.0)


def z_to_gap(z):
    return 2.0 * np.sinh(np.asarray(z))


def extent_along(poly, line) -> tuple[float, float]:
    c = np.asarray(poly.exterior.coords)
    s = np.array([line.project(Point(x)) for x in c])
    return float(s.min()), float(s.max())


def tile_edge_samples(tile: dict) -> dict:
    o, g = tile["objects"], tile["graph"]
    bidx = np.flatnonzero(o["type"] == 0)
    polys = [decode(o["anchor"][i], o["theta"][i], o["length"][i], o["width"][i], o["ratios"][i]) for i in bidx]
    fr, _ = assign(polys, g, o["anchor"][bidx])
    rows = {k: [] for k in ("centre", "angle", "t", "scalars", "stop", "gapd", "phi", "size", "shape", "rect")}
    seqs = []
    for f in fr:
        ext = [extent_along(polys[m], f["line"]) for m in f["members"]]
        order = np.argsort([e[0] for e in ext], kind="stable")[:MAX_PER_FRONTAGE]
        seqs.append((f, [f["members"][i] for i in order], [ext[i] for i in order]))
    time_of, t = {}, 0
    for _, mem, _ in seqs:
        for m in mem:
            time_of[int(bidx[m])] = t
            t += 1
    bmap = np.full(N * N, np.inf)
    for b, tt in time_of.items():
        np.minimum.at(bmap, o["pix"][o["pix_ptr"][b]:o["pix_ptr"][b + 1]], tt)
    t = 0
    for f, mem, ext in seqs:
        line, L = f["line"], f["length"]
        end_prev, p1, p2 = 0.0, None, None
        for k in range(len(mem) + 1):
            a_here = tangent(line, end_prev)
            rows["centre"].append(point_at(line, end_prev))
            rows["angle"].append(a_here)
            rows["t"].append(t)
            rows["scalars"].append(step_scalars(f["cls"], L, end_prev, k, p1, p2))
            if k == len(mem):
                rows["stop"].append(1.0)
                rows["gapd"].append(np.zeros(2, np.float32))
                rows["phi"].append(0.0)
                rows["size"].append(np.zeros(2, np.float32))
                rows["shape"].append(np.zeros(20, np.float32))
                rows["rect"].append(0.0)
                break
            b = bidx[mem[k]]
            a, e = ext[k]
            d = polys[mem[k]].distance(line)
            axis = tangent(line, 0.5 * (a + e))
            phi, ratios = canonical(o["theta"][b:b + 1], o["ratios"][b:b + 1], np.array([axis]))
            par = (float(gap_to_z(a - end_prev)), float(np.log(d + 1.0)), float(phi[0]),
                   float(np.log(o["length"][b])), float(np.log(o["width"][b])))
            rows["stop"].append(0.0)
            rows["gapd"].append(np.array(par[:2], np.float32))
            rows["phi"].append(par[2])
            rows["size"].append(np.array(par[3:], np.float32))
            rows["shape"].append(shape_to_unbounded(ratios)[0])
            rows["rect"].append(float(o["ratios"][b].min() >= RECT_MIN_RATIO))
            p1, p2, end_prev = par, p1, max(end_prev, e) if k else e
            t += 1
    shapes = {"centre": (2,), "scalars": (N_FSCAL,), "gapd": (2,), "size": (2,), "shape": (20,)}
    out = {k: np.asarray(v, np.float32).reshape(-1, *shapes.get(k, ())) for k, v in rows.items()}
    out["bmap"], out["roads"] = bmap, tile["road_raster"].ravel()
    return out


def crops_for(samples, idx=None):
    idx = np.arange(len(samples["t"])) if idx is None else idx
    return time_crops(samples["roads"], samples["bmap"], samples["centre"][idx], samples["angle"][idx], samples["t"][idx])


def place_by_edges(poly, line, start, front_d):
    """Translate ``poly`` so its extent along ``line`` starts at ``start`` and its outline is ``front_d``
    from the centreline on the left side (two fixed-point iterations for curved streets)."""
    for _ in range(2):
        a, b = extent_along(poly, line)
        s_mid = 0.5 * (a + b)
        ang = tangent(line, s_mid)
        t_vec = np.array([np.cos(ang), np.sin(ang)])
        n_vec = np.array([-np.sin(ang), np.cos(ang)])
        d_now = poly.distance(line)
        c = np.asarray(poly.centroid.coords[0]) - point_at(line, s_mid)
        side = 1.0 if c @ n_vec >= 0 else -1.0
        shift = (start - a) * t_vec + (front_d - side * d_now) * n_vec if side > 0 else (start - a) * t_vec + (front_d + d_now) * n_vec
        poly = affinity.translate(poly, *shift)
    return poly


def try_add_fit(state, poly) -> tuple[bool, object]:
    """Programmed correction: try the building, then shrink it about its centroid before rejecting."""
    if state.try_add(poly, 0):
        return True, state.buildings[-1]
    c = poly.centroid
    for f in SHRINK:
        q = affinity.scale(poly, f, f, origin=c)
        if state.try_add(q, 0):
            state.rules["building_shrunk"] = state.rules.get("building_shrunk", 0) + 1
            return True, state.buildings[-1]
    return False, None


def place_frontage_edge(model, graphs, class_widths, device="cpu", seed=0, states=None):
    """B1 frontage phase (strict rules). Afterwards: place_neural(..., states=...) for interior + green."""
    import torch
    from urbanmorph.layout.frontage import frontages
    from urbanmorph.layout.place import LayoutState, _polygon_from
    model.eval()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    states = states if states is not None else [LayoutState(g, class_widths, True) for g in graphs]
    cur = [{"fr": frontages(s.graph), "i": 0, "end": 0.0, "p1": None, "p2": None, "k": 0} for s in states]
    for s in states:
        s.rules.setdefault("frontage_proposed", 0)
        s.rules.setdefault("frontage_accepted", 0)
        s.rules.setdefault("building_shrunk", 0)
        s.rules.setdefault("frontage_rejected_acute", 0)

    def advance(c):
        c.update(i=c["i"] + 1, end=0.0, p1=None, p2=None, k=0)

    with torch.no_grad():
        while True:
            act = [j for j, c in enumerate(cur) if c["i"] < len(c["fr"])]
            if not act:
                break
            crops, scal = [], []
            for j in act:
                c, st = cur[j], states[j]
                f = c["fr"][c["i"]]
                p = point_at(f["line"], c["end"])
                a = tangent(f["line"], c["end"])
                scal.append(step_scalars(f["cls"], f["length"], c["end"], c["k"], c["p1"], c["p2"]))
                crops.append(time_crops(st.road.ravel(), np.where(st.bcanv, 0.0, np.inf), p[None], np.array([a]), [1.0])[0])
            T = lambda x: torch.from_numpy(np.stack(x)).to(device)
            stop, gapd, phi, size, shape = model.sample(T(crops), T(scal), gen)
            for q, j in enumerate(act):
                c, st = cur[j], states[j]
                f = c["fr"][c["i"]]
                line = f["line"]
                if stop[q] or c["k"] >= MAX_PER_FRONTAGE:
                    advance(c)
                    continue
                start = max(c["end"] + float(z_to_gap(gapd[q, 0])), 0.0)
                if start > f["length"]:
                    advance(c)
                    continue
                d = max(float(np.exp(gapd[q, 1]) - 1.0), 0.0)
                ang = tangent(line, start)
                W = float(np.exp(size[q, 1]))
                anchor = point_at(line, start) + (d + W / 2) * np.array([-np.sin(ang), np.cos(ang)])
                poly = _polygon_from(anchor, ang, phi[q], size[q], shape[q])
                if poly.is_empty:
                    advance(c)
                    continue
                poly = place_by_edges(poly, line, start, d)
                st.rules["frontage_proposed"] += 1
                acute0 = st.rules["rejected_acute"]
                ok, _ = try_add_fit(st, poly)
                st.rules["frontage_accepted"] += int(ok)
                st.rules["frontage_rejected_acute"] += int(not ok and st.rules["rejected_acute"] > acute0)
                _, b = extent_along(poly, line)
                par = (float(gapd[q, 0]), float(gapd[q, 1]), float(phi[q]), float(size[q, 0]), float(size[q, 1]))
                c.update(end=max(c["end"], b) if c["k"] else b, p2=c["p1"], p1=par, k=c["k"] + 1)
    return states
