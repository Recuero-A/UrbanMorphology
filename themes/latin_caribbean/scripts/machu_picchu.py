"""Latin Caribbean theme on Machu Picchu (user request 2026-11-24): a dense colonial hill town in the jungle on the
NA-SMC · F1+F4 streets (seed 2) of the Machu Picchu site benchmark (docs/SITE_BENCHMARKS.md), with a small landmark
downtown, plaza churches with giant towers, the photo churches and the KAIST landmark city's palette.

* Streets: the benchmark's city network (cropped to the city limits), then cropped again to the TOWN MASK.
* Town mask (4 m grid): city limits, no water, within 40 m of a street, where the street density within ~120 m is
  >= 10 km/km2 (the steep west, with only a few streets, stays jungle); opened / closed, parts < 2 ha dropped, holes
  filled. Streets are kept when both ends are inside the mask dilated by 20 m; pieces < 120 m are dropped.
* Cut and fill (the terrain of a steep hill town, inside the town only): every street is level ACROSS its width (the
  road level of the nearest centreline point: the DSM low-passed over 12 m), over the carriageway and the 1.6 m
  sidewalks, with 6 m earth banks back to the slope; plazas are level terraces (their mean street level) with 6 m
  banks. The road level is written as ``road_z`` (terrain3d.Surfaces uses it). Buildings: terrain3d seat "street" (each
  house's front at its street's level, a terrace cut under it, a stone plinth where it stands above the slope).
* Plazas (3) on the gentlest blocks (slope < 35 %); the downtown (a small skyscraper district, ~8 streets) about the
  main plaza; six cohesive landmark towers in it, red-family colours; photo churches on lottery sites.

    .venv/bin/python themes/latin_caribbean/scripts/machu_picchu.py plan      # mask, graph, terrain, plazas, district, map
    .venv/bin/python themes/latin_caribbean/scripts/machu_picchu.py build     # the 3D city (m16_city3d_big.py)
    .venv/bin/python themes/latin_caribbean/scripts/machu_picchu.py render    # views + Blender renders at sunset
    ->  outputs/latin_caribbean/machu_picchu/ (plan) and outputs/latin_caribbean/cities/machu_picchu/<city>/
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import shapely
from scipy import ndimage

OUT = ROOT / "outputs" / "latin_caribbean" / "machu_picchu"
CITIES = ROOT / "outputs" / "latin_caribbean" / "cities" / "machu_picchu"
METHOD, VERSION, SEED = "NA-SMC", "F1 + F4", 2
LABEL = "NA-SMC [improved]"
TAG = "machu_picchu"
CLASS_W = (14.0, 10.0, 7.0, 5.0)                 # street widths by class (terrain3d.apply_terrain)
WALK_M = 1.6                                     # colonial sidewalk (latin.infill)
BANK_M = 6.0
GIANT = "150,115,90"                             # plaza church towers (m), tallest first
GIANT_LOTTERY = "60,85"
CHURCH_SITES = [("gothic", None), ("bahia", None), (None, 75.0), ("gothic", None), (None, None), ("bahia", None), (None, 65.0),
                (None, None)]                            # small church squares: photo style, giant tower (m)
DISTRICT_N, DISTRICT_H = 14, "60,130"
DISTRICT2_N, DISTRICT2_H = 10, "60,130"                # the second downtown, on the southern crest
LANDMARKS2 = ["golden:rojo:thin", "coral:bermellon:cohesive", "eagle:coral:thin", "golden:terracota:cohesive"]
CLUSTER = ["golden:thin:rojo", "coral:cohesive:bermellon", "eagle:thin:coral", "golden:cohesive:terracota",
           "coral:thin:rojo_ladrillo", "eagle:cohesive:rojo", "golden:thin:coral", "coral:cohesive:rojo",
           "eagle:thin:bermellon", "golden:cohesive:rojo_ladrillo"]
LANDMARKS = ["golden:rojo", "coral:bermellon", "eagle:coral", "coral:rojo_ladrillo", "golden:coral", "eagle:rojo"]


def site():
    from site_benchmark import Site
    return Site("machu_picchu")


CITADEL = (949.0, 874.0)                          # the Inca citadel, domain metres (configs/sites/machu_picchu.json)
PLAZA_LONG, PLAZA_DEEP = 70.0, 40.0


def chain(E):
    """The node sequence of a stroke (its edges form a path or a loop; every node has degree <= 2 in it)."""
    adj = {}
    for a, b in E:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    ends = [v for v, nb in adj.items() if len(nb) == 1]
    seq = [ends[0] if ends else int(E[0, 0])]
    seen = {seq[0]}
    while True:
        nxt = [v for v in adj[seq[-1]] if v not in seen]
        if not nxt:
            return seq
        seq.append(nxt[0])
        seen.add(nxt[0])


def street_plazas(st, g, mask, slope, cw, n=3, spacing=240.0, max_w=34.0, W=512.0, sizes=None, avoid=(), max_slope=0.4,
                  centre=CITADEL):
    """Plaza + church sites for long contour blocks: a PLAZA_LONG x PLAZA_DEEP rectangle fronting a street, inside one
    block, the town and one tile; the church at one end along the street (as latin.infill.plaza_sites). The main plaza
    is the gentlest near ``centre`` (the citadel saddle), the others the gentlest >= ``spacing`` away. "sub_block": the rest of the
    block keeps its houses (latin.infill.colonial_infill)."""
    from kaist_village import street_strokes
    from urbanmorph.city3d.format import polygons, rings
    road = shapely.union_all([shapely.LineString(g.nodes[e]).buffer(cw[int(c)] / 2) for e, c in zip(g.edges, g.classes())])
    blocks = [b for b in polygons(shapely.box(0, 0, st.L, st.L).difference(road)) if b.area > PLAZA_LONG * PLAZA_DEEP]
    tree = shapely.STRtree(blocks)
    X, Y = st.xy_grid()
    cls = g.classes()
    cands = []
    for ch in street_strokes(g):
        P = g.nodes[chain(g.edges[ch])]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        if d[-1] < (sizes[0][0] if sizes else PLAZA_LONG) + 20:
            continue
        hw = max(cw[int(cls[e])] for e in ch) / 2 + 2.0
        at = lambda s_: np.array([np.interp(s_, d, P[:, 0]), np.interp(s_, d, P[:, 1])])
        LONG, DEEP = sizes[0] if sizes else (PLAZA_LONG, PLAZA_DEEP)
        for s_ in np.arange(LONG / 2 + 5, d[-1] - LONG / 2 - 5, 12.0):
            u = at(s_ + 35) - at(s_ - 35)
            u /= np.linalg.norm(u)
            nrm = np.array([-u[1], u[0]])
            for sg in (-1.0, 1.0):
                f = at(s_) + sg * nrm * hw
                R = shapely.Polygon([f - u * LONG / 2, f + u * LONG / 2, f + u * LONG / 2 + sg * nrm * DEEP,
                                     f - u * LONG / 2 + sg * nrm * DEEP])
                x0, y0, x1, y1 = R.bounds
                if min(x0, y0) < 0 or int(x0 // W) != int(x1 // W) or int(y0 // W) != int(y1 // W):
                    continue
                if not len(tree.query(R, predicate="within")):
                    continue
                cells = shapely.contains_xy(R, X, Y)
                if mask[cells].mean() < 0.98:
                    continue
                cands.append((float(np.median(slope[cells])), R, u, sg * nrm))
    # also rectangles anywhere on gentle land inside a block, along the nearest street, preferably touching it
    lines = shapely.linestrings(g.nodes[g.edges])
    ltree = shapely.STRtree(lines)
    gx, gy = np.meshgrid(np.arange(0, st.L, 8.0), np.arange(0, st.L, 8.0))
    G = np.c_[gx.ravel(), gy.ravel()]
    r_, c_ = st.rc(G)
    G = G[mask[r_, c_] & (slope[r_, c_] < 0.3)]
    for x, y in G:
        pt = shapely.Point(x, y)
        ln = lines[ltree.nearest(pt)]
        a_, b_ = np.asarray(ln.coords)
        u = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
        nrm = np.array([-u[1], u[0]])
        for lo, de in (sizes or ((PLAZA_LONG, PLAZA_DEEP), (56.0, 36.0))):
            c0 = np.array([x, y])
            R = shapely.Polygon([c0 + sx * u * lo / 2 + sy * nrm * de / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            x0, y0, x1, y1 = R.bounds
            if min(x0, y0) < 0 or int(x0 // W) != int(x1 // W) or int(y0 // W) != int(y1 // W):
                continue
            if not len(tree.query(R, predicate="within")):
                continue
            cells = shapely.contains_xy(R, X, Y)
            if mask[cells].mean() < 0.98:
                continue
            gap = float(shapely.distance(lines[ltree.nearest(R)], R))
            cands.append((float(np.median(slope[cells])) + (0.05 if gap > 8.0 else 0.0), R, u, nrm))
            break
    out = []
    for k in range(n):
        dist = lambda R: np.hypot(R.centroid.x - centre[0], R.centroid.y - centre[1])
        taken = [shapely.Point(o["church"]["center"]) for o in out] + [shapely.Point(xy) for xy in avoid]
        ok = [c for c in cands if c[0] < max_slope and all(c[1].centroid.distance(p_) >= spacing for p_ in taken)]
        if not ok:
            break
        sl, R, u, v = min(ok, key=lambda c: c[0] + (0.1 * dist(c[1]) / 300.0 if k == 0 else 0.0))
        cR = np.asarray(R.centroid.coords)[0]
        xy = np.asarray(R.exterior.coords)[:4]
        LR, WR = np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])
        Lc, Wc = float(min(44.0, 0.55 * LR)), float(min(WR - 6.0, max_w))
        ctr = cR + u * (LR / 2 - Lc / 2)
        out.append({"plaza": rings(R), "church": {"center": ctr.tolist(), "axis": u.tolist(), "L": Lc, "W": Wc},
                    "sub_block": True, "slope": sl})
    return out


def cluster_sites(st, g, mask, cw, dist, n=14, W=20.0, D=20.0, gap=4.0, seed=3):
    """Explicit tower sites for the landmark cluster (the second downtown): W x D rectangles along both sides of the
    district's streets (clear of the carriageway and sidewalks, inside the town), from the centre out, ``gap`` m apart;
    heights from the district's (lo, hi), taller to the centre; designs CLUSTER in turn around the centre, so that
    neighbours differ."""
    from kaist_village import street_strokes
    poly = shapely.Polygon(dist["polygon"][0])
    c0, R = np.asarray(dist["center"], float), float(dist["radius"])
    road = shapely.union_all([shapely.LineString(g.nodes[e]).buffer(cw[int(k)] / 2 + WALK_M) for e, k in zip(g.edges, g.classes())])
    X, Y = st.xy_grid()
    cand = []
    for ch in street_strokes(g):
        P = g.nodes[chain(g.edges[ch])]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        hw = max(cw[int(k)] for k in g.classes()[ch]) / 2 + WALK_M
        at = lambda s_: np.array([np.interp(s_, d, P[:, 0]), np.interp(s_, d, P[:, 1])])
        for s_ in np.arange(0.0, d[-1], 6.0):
            pt = at(s_)
            if not poly.contains(shapely.Point(pt)):
                continue
            u = at(min(s_ + 10, d[-1])) - at(max(s_ - 10, 0))
            if np.hypot(*u) < 1e-6:
                continue
            u /= np.hypot(*u)
            nrm = np.array([-u[1], u[0]])
            for sg in (-1.0, 1.0):
                ctr = pt + sg * nrm * (hw + 0.8 + D / 2)
                Rr = shapely.Polygon([ctr + sx * u * W / 2 + sy * nrm * D / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                if Rr.intersects(road) or not poly.contains(Rr.centroid):
                    continue
                cells = shapely.contains_xy(Rr, X, Y)
                if not cells.any() or mask[cells].mean() < 0.95:
                    continue
                cand.append((float(np.hypot(*(ctr - c0))), ctr, u, Rr))
    cand.sort(key=lambda c: c[0])
    picked = []
    for dd, ctr, u, Rr in cand:
        if all(not Rr.buffer(gap / 2).intersects(q[3].buffer(gap / 2)) for q in picked):
            picked.append((dd, ctr, u, Rr))
        if len(picked) >= n:
            break
    lo, hi = dist["h"]
    rng = np.random.default_rng(seed)
    picked.sort(key=lambda q: np.arctan2(*(q[1] - c0)[::-1]))
    out = []
    for k, (dd, ctr, u, Rr) in enumerate(picked):
        H = float(np.clip((lo + (hi - lo) * max(0.0, 1 - dd / R) ** 0.8) * rng.uniform(0.92, 1.08), lo, hi))
        out.append({"center": ctr.tolist(), "axis": u.tolist(), "W": W, "D": D, "H": H, "wall": "rojo", "dome": "cupula_naranja",
                    "landmark": CLUSTER[k % len(CLUSTER)]})
    return out


def town_mask(st, big, limits, g):
    """City limits, no water, near streets, where the streets are dense; cleaned."""
    n = st.n
    on = np.zeros((n, n), np.float32)
    for a, b in g.edges:
        p, q = g.nodes[a], g.nodes[b]
        k = max(2, int(np.hypot(*(q - p)) / 2.0))
        P = p + np.outer(np.linspace(0, 1, k), q - p)
        np.add.at(on, st.rc(P), 2.0 / 1000)                      # km of street per cell
    dens = ndimage.gaussian_filter(on, 120.0 / 4 / 2) / ((4.0 / 1000) ** 2)       # km / km2 over ~120 m
    near = ndimage.distance_transform_edt(on == 0) * 4.0 <= 40.0
    m = limits & (big["water"] == 0) & near & (dens >= 10.0)
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=2), iterations=4) & limits & (big["water"] == 0)
    lab, k = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    m = np.isin(lab, 1 + np.flatnonzero(sizes * 16 >= 20000))
    return ndimage.binary_fill_holes(m) & (big["water"] == 0), dens


def crop_to(g, st, mask):
    """Streets with both ends in the mask; isolated pieces < 120 m dropped (site_benchmark.crop)."""
    import site_benchmark as sb
    return sb.crop(g, mask, st)


def engineer(st, big, g, mask, plazas, cw):
    """Cut and fill inside the town: (elev, road_z) on the domain grid (see the module docstring). The road level is
    continuous: every cell takes the level of the nearest point of its street's centreline (sampled every 0.5 m on
    the DSM low-passed over 12 m), so a street is level across its width and smooth along it; under the carriageway
    and the sidewalks the ground sits 0.35 m below the paving, so no grass shows through."""
    from scipy.spatial import cKDTree
    z = big["elev"].astype(float)
    zs = ndimage.gaussian_filter(z, 3.0)
    L = st.L
    sample = lambda P: ndimage.map_coordinates(zs, [(L + 256 - P[:, 1]) / 4.0 - 0.5, (P[:, 0] + 256) / 4.0 - 0.5], order=1,
                                               mode="nearest")
    pts, hws = [], []
    cls, structs = g.classes(), np.asarray(g.structs())
    for e, (a, b) in enumerate(g.edges):
        if structs[e] != 0:
            continue
        p, q = g.nodes[a], g.nodes[b]
        P = p + np.outer(np.linspace(0, 1, max(2, int(np.hypot(*(q - p)) / 0.5) + 1)), q - p)
        pts.append(P)
        hws.append(np.full(len(P), cw[int(cls[e])] / 2 + WALK_M))
    P, hw = np.vstack(pts), np.concatenate(hws)
    zp = sample(P)
    town = ndimage.binary_dilation(mask, iterations=8)              # 32 m around the town
    X, Y = st.xy_grid()
    d = np.full(X.shape, np.inf)
    rz, hwn = np.full(X.shape, np.nan), np.zeros(X.shape)
    dd, ii = cKDTree(P).query(np.c_[X[town], Y[town]], distance_upper_bound=60.0)
    ok = np.isfinite(dd)
    tmp_d, tmp_z, tmp_h = np.full(len(dd), np.inf), np.full(len(dd), np.nan), np.zeros(len(dd))
    tmp_d[ok], tmp_z[ok], tmp_h[ok] = dd[ok], zp[ii[ok]], hw[ii[ok]]
    d[town], rz[town], hwn[town] = tmp_d, tmp_z, tmp_h
    near = np.isfinite(d)
    rz = np.where(near, rz, z)
    road = town & near & (d <= hwn + 3.0)                           # junctions: where two streets meet at different levels,
    w = ndimage.gaussian_filter(road.astype(float), 1.5)            # a normalised blur over the corridors turns the crease
    rzb = ndimage.gaussian_filter(np.where(road, rz, 0.0), 1.5) / np.maximum(w, 1e-6)   # into a short ramp
    rz = np.where(road, rzb, rz)
    sunk = rz - 0.35                                                # under the paving and 2.5 m past it: no grass shows
    s = np.clip((d - hwn - 2.5) / BANK_M, 0, 1)
    s = s * s * (3 - 2 * s)
    elev = np.where(town & near, sunk + (z - sunk) * s, z)
    road_z = np.where(town & near & (d <= hwn + 3.0), rz, np.nan)
    corridor = town & near & (d <= hwn + 2.5)
    rep = []
    for pz in plazas:                                               # level plaza terraces and tower pads
        poly = shapely.Polygon(pz["plaza"][0])
        inside = shapely.contains_xy(poly, X, Y) & ~corridor
        edge = shapely.contains_xy(poly.buffer(8.0), X, Y) & corridor
        lvl = float(np.mean(rz[edge])) if edge.any() else float(np.mean(z[inside]))
        dp = ndimage.distance_transform_edt(~inside) * 4.0
        sp = np.clip(dp / BANK_M, 0, 1)
        sp = sp * sp * (3 - 2 * sp)
        zone = (dp <= BANK_M) & ~corridor & town
        elev = np.where(zone, lvl + (elev - lvl) * sp, elev)
        rep.append({"level_m": lvl, "cut_fill_max_m": float(np.abs(z[inside] - lvl).max()) if inside.any() else 0.0})
    return elev.astype(np.float32), road_z.astype(np.float32), {"plazas": rep, "corridor_km2": float(corridor.sum() * 16 / 1e6),
                                                                 "max_cut_m": float((z - elev)[town].max()),
                                                                 "max_fill_m": float((elev - z)[town].max())}


def stage_plan():
    import site_benchmark as sb
    from m6_build_objects import load_objects
    from kaist_village import district_of
    st = site()
    big = st.domain()
    limits = np.load(st.data / "city_mask.npz")["mask"]
    L = st.L
    g0 = pickle.loads((st.out / "city_graphs.pkl").read_bytes())[(METHOD, VERSION, SEED)]
    mask, dens = town_mask(st, big, limits, g0)
    g = crop_to(g0, st, ndimage.binary_dilation(mask, iterations=5))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "town_mask.npz", mask=mask)
    slope, _, _ = st.slope(big)
    cw = load_objects()["class_widths"]
    sites = street_plazas(st, g, mask, slope, cw)
    for s_, h in zip(sites, [float(x) for x in GIANT.split(",")]):
        s_["church"]["tower_h"] = h
    # small church squares along the streets (a church and its atrium), some of them the churches from the photos
    small = street_plazas(st, g, mask, slope, cw, n=len(CHURCH_SITES), spacing=150.0, max_w=26.0,
                          sizes=((46.0, 30.0), (40.0, 28.0)), avoid=[s_["church"]["center"] for s_ in sites], max_slope=0.45)
    for s_, (style, h) in zip(small, CHURCH_SITES):
        if style:
            s_["church"]["style"] = style
        if h:
            s_["church"]["tower_h"] = h
    sites += small
    (OUT / "plaza_sites.json").write_text(json.dumps(sites, indent=1))
    centre = np.asarray(shapely.Polygon(sites[0]["plaza"][0]).centroid.coords)[0]
    district = district_of(g, centre, DISTRICT_N, [float(x) for x in DISTRICT_H.split(",")])
    district.update(lot_front=[21.0, 28.0], lot_depth=[20.0, 26.0], fill="houses")   # wider lots; no gardens: a dense downtown
    # the second downtown (user request 2026-11-24): landmark towers side by side on the southern crest, the town's
    # high ground under Machu Picchu mountain (the highest gentle cells of the southern town)
    X, Y = st.xy_grid()
    sel = mask & (Y < 600) & (slope < 0.3)
    top = np.argsort(big["elev"][sel])[-200:]
    c2 = np.array([X[sel][top].mean(), Y[sel][top].mean()])
    district2 = district_of(g, c2, DISTRICT2_N, [float(x) for x in DISTRICT2_H.split(",")])
    district2.update(lot_front=[21.0, 28.0], lot_depth=[20.0, 26.0], fill="houses")
    (OUT / "district.json").write_text(json.dumps([district, district2], indent=1))
    elev, road_z, erep = engineer(st, big, g, mask, sites, cw)
    np.savez_compressed(OUT / "terrain_town.npz", elev=elev, water=big["water"], road_z=road_z,
                        masonry=ndimage.binary_dilation(mask, iterations=8),
                        water_surface=np.where(big["water"] > 0, big["elev"] + 0.3, np.nan).astype(np.float32))
    (OUT / "graphs.pkl").write_bytes(pickle.dumps({"graphs": {(LABEL, SEED): g}, "real": None}))
    Ln = lambda gg: float(np.linalg.norm(gg.nodes[gg.edges[:, 0]] - gg.nodes[gg.edges[:, 1]], axis=1).sum() / 1000)
    info = {"method": METHOD, "version": VERSION, "seed": SEED, "town_km2": float(mask.sum() * 16 / 1e6),
            "city_limits_km2": float(limits.sum() * 16 / 1e6), "street_km_city": Ln(g0), "street_km_town": Ln(g),
            "plazas": len(sites), "plaza_slope": [float(np.median(slope[shapely.contains_xy(shapely.Polygon(s_["plaza"][0]), *st.xy_grid())]))
                                                  for s_ in sites],
            "district": {k: v for k, v in district.items() if k != "polygon"}, "tiles": district_tiles(district),
            "district2": {k: v for k, v in district2.items() if k not in ("polygon", "sites")}, "tiles2": district_tiles(district2),
            "engineering": erep}
    (OUT / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))
    plan_map(st, big, limits, mask, g0, g, sites, [district, district2], elev)


def district_tiles(district, W=512.0):
    """Tiles the district overlaps, with the share of its area in each (landmarks go where the district is)."""
    poly = shapely.Polygon(district["polygon"][0])
    out = {}
    for r in range(4):
        for c in range(4):
            a = poly.intersection(shapely.box(c * W, r * W, c * W + W, r * W + W)).area / poly.area
            if a > 0.05:
                out[f"r{r}c{c}"] = round(float(a), 3)
    return out


def plan_map(st, big, limits, mask, g0, g, sites, district, elev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from site_benchmark import MARGIN, contours, hillshade
    L = st.L
    fig, axs = plt.subplots(1, 2, figsize=(20, 11))
    for ax, (title, gg, show) in zip(axs, (("benchmark city network (NA-SMC · F1+F4, seed 2) and the town mask", g0, mask),
                                           ("the town: streets kept, plazas (gold), the two downtowns (red), cut-and-fill terrain", g, mask))):
        ax.imshow(hillshade(st, {**big, "elev": elev} if gg is g else big, show), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN))
        contours(ax, st, {**big, "elev": elev} if gg is g else big, step=25, lw=0.3)
        ax.contour(*st.xy_grid(), limits.astype(float), levels=[0.5], colors="#f5c242", linewidths=1.2, linestyles="--")
        ax.add_collection(LineCollection(gg.nodes[gg.edges], colors="#111", linewidths=0.8))
        if gg is g:
            for s_ in sites:
                p = np.asarray(s_["plaza"][0])
                ax.fill(p[:, 0], p[:, 1], color="#e3a92b", alpha=0.8)
            for dist_ in district:
                p = np.asarray(dist_["polygon"][0])
                ax.plot(p[:, 0], p[:, 1], color="#c0392b", lw=2)
        ys, xs = np.nonzero(limits)
        ax.set_xlim(-MARGIN + xs.min() * 4 - 60, -MARGIN + xs.max() * 4 + 60)
        ax.set_ylim(L + MARGIN - ys.max() * 4 - 60, L + MARGIN - ys.min() * 4 + 60)
        ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "town_plan.png", dpi=85)
    plt.close(fig)


PHOTO_CHURCHES = {"r0c1": "gothic", "r1c2": "bahia", "r2c2": "gothic", "r2c1": "bahia", "r0c2": "bahia", "r1c3": "gothic"}


def city_dir():
    return CITIES / f"{TAG}_{LABEL.replace(' [', '_').replace(']', '')}_s{SEED}_latin"


def stage_build():
    info = json.loads((OUT / "info.json").read_text())
    per = {}
    for share, lms in ((info["tiles"], LANDMARKS), (info["tiles2"], LANDMARKS2)):   # landmarks by each district's share per tile
        cnt = {t: int(round(len(lms) * a)) for t, a in share.items()}
        cnt[max(share, key=share.get)] += len(lms) - sum(cnt.values())
        k = 0
        for t, m in cnt.items():
            if m > 0:
                per[t], k = per.get(t, []) + lms[k:k + m], k + m
    topts = {"church_colour": "rojo", "church_dome": "cupula_naranja", "landmarks": per, "photo_churches": PHOTO_CHURCHES,
             "landmark_min": [13.0, 60.0], "landmark_spacing_m": 30.0,
             "churches_per_tile": 2, "church_spacing_m": 150.0, "giant_lottery_p": 0.35}
    py = str(ROOT / ".venv" / "bin" / "python")
    cmd = [py, str(ROOT / "scripts" / "m16_city3d_big.py"), "--method", LABEL, "--seed", str(SEED), "--graphs", str(OUT / "graphs.pkl"),
           "--tag", TAG, "--theme", "latin_caribbean", "--cities-dir", str(CITIES), "--terrain-npz", str(OUT / "terrain_town.npz"),
           "--village-npz", str(OUT / "town_mask.npz"), "--jungle", "dense", "--infill", "colonial",
           "--plazas", str(OUT / "plaza_sites.json"), "--giant-lottery", GIANT_LOTTERY, "--lanes", "60", "--below-max", "2",
           "--cliff-m", "9",
           "--district", str(OUT / "district.json"), "--theme-opts", json.dumps(topts), "--seat", "street",
           "--plinth", "latin:masonry", "--build-slope", "9", "--village-clip", "--site", "Machu Picchu site benchmark"]
    CITIES.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "build.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode:
        print(r.stderr[-3000:])
        return r.returncode
    print(r.stdout.strip().splitlines()[-1])
    return 0


def town_terrain():
    """The town's terrain dict (cut and fill, road level) on the domain grid, for cameras."""
    z = np.load(OUT / "terrain_town.npz")
    return {"elev": z["elev"], "valid": np.ones(z["elev"].shape, bool), "water": z["water"], "road_z": z["road_z"]}


def stage_render(views_only=False):
    from kaist_village import BLENDER, tower_views
    st = site()
    d = city_dir()
    sites = json.loads((OUT / "plaza_sites.json").read_text())
    district = json.loads((OUT / "district.json").read_text())[0]
    mask = np.load(OUT / "town_mask.npz")["mask"]
    tv = tower_views(d, town_terrain(), sites, mask, st.L, district=district)
    (d / "giant_towers.json").write_text(json.dumps(tv, indent=1, default=float))
    print("views", tv.get("views"), flush=True)
    if views_only:
        return 0
    views = "aerial,oblique,street" + "".join("," + v for v in tv.get("views", []))
    rc = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
                         "--", "--tile", str(d), "--res", "2000", "--light", "sunset", "--views", views],
                        stdout=open(d / "blender.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    return rc


RINGS = ((-4096.0, 10240.0, 32.0), (-14336.0, 30720.0, 160.0))     # backdrop squares: west / south edge, side, cell (m)


def dem_corners(st, x0, side, res):
    """GLO-30 heights at the (n + 1)^2 cell corners of a square (domain metres, west / south edge x0, row 0 = north)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    n = int(round(side / res))
    dst = np.full((n + 1, n + 1), np.nan, np.float32)
    tr = from_origin(st.x0 + x0 - res / 2, st.y0 + x0 + side + res / 2, res, res)
    with rasterio.open(ROOT / st.cfg["dem"]) as r:
        reproject(rasterio.band(r, 1), dst, src_nodata=r.nodata, dst_transform=tr, dst_crs=st.crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return np.nan_to_num(dst, nan=float(np.nanmin(dst))), n


def backdrop_water(st, x0, side):
    """Rivers and lakes of the backdrop (OSM, Peru extract) as one polygon in domain metres."""
    import geopandas as gpd
    from pyproj import Transformer
    seq = OUT / "backdrop_water.geojsonseq"
    if not seq.exists():
        back = Transformer.from_crs(st.crs, "EPSG:4326", always_xy=True).transform
        lo, hi = back(st.x0 + x0, st.y0 + x0), back(st.x0 + x0 + side, st.y0 + x0 + side)
        clip, wat = OUT / "_backdrop.osm.pbf", OUT / "_backdrop_water.osm.pbf"
        subprocess.run(["osmium", "extract", "-O", "-b", f"{lo[0]},{lo[1]},{hi[0]},{hi[1]}", "-o", str(clip),
                        str(ROOT / st.cfg["osm_extract"])], check=True, capture_output=True)
        subprocess.run(["osmium", "tags-filter", "-O", "-o", str(wat), str(clip), "nwr/natural=water", "w/waterway=river,riverbank"],
                       check=True, capture_output=True)
        subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "-o", str(seq), str(wat)], check=True, capture_output=True)
        clip.unlink()
        wat.unlink()
    g = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(st.crs).translate(-st.x0, -st.y0)
    parts = [geom.buffer(18.0) if geom.geom_type in ("LineString", "MultiLineString") else geom for geom in g.geometry]
    return shapely.make_valid(shapely.union_all(parts)) if parts else shapely.Polygon()


def stage_backdrop():
    """The mountains around the city: GLO-30 terrain in two rings (32 m to 10 km, 160 m to 30 km) with a hole for the
    2048 m domain (its edges on the grid), distant rainforest on every slope, the Urubamba and other OSM rivers."""
    from urbanmorph.city3d.materials import MaterialTable
    from urbanmorph.city3d.scene import corner_uvs, save_scene
    st = site()
    d = city_dir()
    sc = np.load(d / "meshes.npz", allow_pickle=True)
    groups = {}
    for k in sc.files:
        if k != "materials":
            g_, f_ = k.split("__")
            groups.setdefault(g_, {})[f_] = sc[k]
    mats = json.loads(str(sc["materials"]))
    groups.pop("backdrop", None)

    def mat(name):
        tbl = MaterialTable()
        tbl.get(name)
        prm = tbl.params[0]
        for k, p in enumerate(mats):
            if p["name"] == prm["name"]:
                return k
        mats.append(prm)
        return len(mats) - 1
    m_far, m_water = mat("latin:canopy_far"), mat("latin:water")
    water = backdrop_water(st, RINGS[1][0], RINGS[1][1])
    VV, FF, MM, n0 = [], [], [], 0
    holes = [(0.0, st.L), (RINGS[0][0], RINGS[0][0] + RINGS[0][1])]
    for (x0, side, res), (h0, h1) in zip(RINGS, holes):
        z, n = dem_corners(st, x0, side, res)
        xs = x0 + np.arange(n + 1) * res
        ys = x0 + side - np.arange(n + 1) * res
        X, Y = np.meshgrid(xs, ys)
        V = np.c_[X.ravel(), Y.ravel(), z.ravel()]
        r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cx, cy = x0 + (c + 0.5) * res, x0 + side - (r + 0.5) * res
        keep = ~((cx > h0) & (cx < h1) & (cy > h0) & (cy < h1))
        r, c, cx, cy = r[keep], c[keep], cx[keep], cy[keep]
        a, b, dd, e = r * (n + 1) + c, r * (n + 1) + c + 1, (r + 1) * (n + 1) + c + 1, (r + 1) * (n + 1) + c
        F = np.vstack([np.c_[a, e, dd], np.c_[a, dd, b]])
        zc = z[:-1, :-1][keep], z[:-1, 1:][keep], z[1:, :-1][keep], z[1:, 1:][keep]
        wet = shapely.contains_xy(water, cx, cy)                 # forest to the cliffs (rock squares looked blocky)
        m = np.where(wet, m_water, m_far)
        MM.append(np.tile(m, 2))
        if res == RINGS[0][2]:                                   # a skirt under the inner ring's outer edge (ring cracks)
            ring = [(0, j) for j in range(n)] + [(i, n) for i in range(n)] + [(n, n - j) for j in range(n)] + [(n - i, 0) for i in range(n)]
            ids = np.array([i * (n + 1) + j for i, j in ring] + [0])
            low = V[ids] - [0, 0, 40.0]
            k0 = len(V)
            V = np.vstack([V, low])
            q = np.arange(len(ids) - 1)
            sk = np.vstack([np.c_[ids[q], ids[q + 1], k0 + q + 1], np.c_[ids[q], k0 + q + 1, k0 + q]])
            F = np.vstack([F, sk, sk[:, ::-1]])
            MM.append(np.full(2 * len(sk), m_far))
        VV.append(V)
        FF.append(F + n0)
        n0 += len(V)
    V, F = np.vstack(VV).astype(np.float32), np.vstack(FF).astype(np.int32)
    groups["backdrop"] = {"V": V, "F": F, "mat": np.concatenate(MM).astype(np.int16), "obj": np.full(len(F), -1, np.int32),
                          "UV": corner_uvs(V.astype(float), F)}
    save_scene(d / "meshes.npz", groups, mats)
    print("backdrop triangles", len(F), "water faces", int((np.concatenate(MM) == m_water).sum()), flush=True)
    return 0


HUAYNA = (898.0, 1738.0)                          # Huayna Picchu's summit on the GLO-30 DSM (domain metres)


STREET_VIEWS = {"main_tower": "church_tower", "landmark_street": "landmark", "photo_church": "church_"}   # best view of each kind
STREET_SKIP = {"a07_church_tower"}                     # checked by eye: a paved slab fills the foreground


def plan_photos(d):
    """Ten hero photos (photo_tour.py). The town faces east, so its slope takes a low eastern sun and the skyline the
    western sunset. Street views: the best clear views of the towers (alleys.search over the town's streets ->
    street_candidates.json) and the alley of tower_views, as level shift-lens cameras (alleys.level)."""
    import alleys
    from alleys import level
    from kaist_village import terrain_z
    st = site()
    z_at = terrain_z(town_terrain(), st.L)
    zg = lambda x, y: float(st.domain()["elev"][tuple(i[0] for i in st.rc([[x, y]]))])
    at = lambda xy, h, z=None: [float(xy[0]), float(xy[1]), (z_at(*xy) if z is None else z) + h]
    cams = json.loads((d / "cameras.json").read_text())
    dt = json.loads((OUT / "district.json").read_text())[0]["center"]
    zd = z_at(*dt)
    cf = OUT / "street_candidates.json"
    if not cf.exists():
        cf.write_text(json.dumps(alleys.search(d, z_at, n_best=16), indent=1))
    cand = {p_["name"]: p_ for p_ in json.loads(cf.read_text())}
    air = dict(haze_m=20000, clouds=0.55, glow=0.3, exposure=-0.35)
    P = []
    # the classic view from the south: the town over the saddle, the downtown, Huayna Picchu behind
    P.append(dict(name="postcard", location=at((1010, 330), 55), look_at=[930.0, 1300.0, zd + 20], lens=32, sun_el=8,
                  sun_az=15, **air))
    # from the Urubamba (a drone 45 m over the water, above the bank canopy): the town climbing 450 m of slope
    P.append(dict(name="from_the_river", location=[1806.0, 1018.0, zg(1806, 1018) + 45.0], look_at=[1150.0, 900.0, zd - 60],
                  lens=26, sun_el=9, sun_az=-10, **{**air, "haze_m": 15000}))
    # the ridge against the sunset: the towers backlit over the gorge
    P.append(dict(name="sunset_ridge", location=[2050.0, 780.0, zd + 60], look_at=[944.0, 930.0, zd + 60], lens=40,
                  sun_el=5.0, sun_az=178, sun_disc=True, haze_m=30000, clouds=0.6, glow=0.15, exposure=-1.1))
    # from Huayna Picchu's summit down over the town (the hikers' view)
    P.append(dict(name="from_huayna_picchu", location=at(HUAYNA, 25.0, zg(*HUAYNA)), look_at=[1050.0, 700.0, zd - 80],
                  lens=28, sun_el=7, sun_az=40, **air))
    # street views of the towers
    for nm, kind in STREET_VIEWS.items():
        best = [p_ for p_ in cand.values() if p_["target"].startswith(kind) and (kind != "church_" or p_["target"] != "church_tower")
                and p_["name"] not in STREET_SKIP]
        if best:
            v = max(best, key=lambda p_: p_["score"])
            P.append(level_godot(dict(name=nm, location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=6,
                                sun_az=v["sun_az"], haze_m=9000, clouds=0.55, glow=0.6, top_pad_deg=2.5)))
    if "alley" in cams:
        v = cams["alley"]
        P.append(level_godot(dict(name="alley", location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=6, sun_az=20,
                            haze_m=9000, clouds=0.55, glow=0.7, top_pad_deg=2.0)))
    # the terraced east slope: a drone over the lower town, up the stacked houses to the towers on the saddle
    P.append(dict(name="east_slope", location=at((1460, 700), 60.0, zg(1460, 700)), look_at=[1050.0, 880.0, zd - 30],
                  lens=30, sun_el=9, sun_az=-5, **{**air, "haze_m": 15000}))
    # the second downtown on the southern crest, the Andes behind, the sun behind the camera
    c2 = np.asarray(json.loads((OUT / "district.json").read_text())[1]["center"], float)
    z2 = z_at(*c2)
    P.append(dict(name="second_downtown", location=[c2[0] - 260, c2[1] - 200, z2 + 90], look_at=[c2[0], c2[1], z2 + 45], lens=30,
                  sun_el=7, sun_az=215, **air))
    # a drone over the east slope: rows of houses along the contour streets
    P.append(dict(name="slope_drone", location=[1420.0, 980.0, zd], look_at=[1150.0, 820.0, zd - 200], lens=28, sun_el=9,
                  sun_az=-15, **{**air, "haze_m": 15000}))
    # the downtown at dusk: lit windows, the sun just down
    if "district" in cams:
        v = cams["district"]
        P.append(dict(name="downtown_dusk", location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=0.4, sun_az=185,
                      haze_m=12000, glow=1.0, clouds=0.65, exposure=0.7))
    for k, p_ in enumerate(P):
        p_["name"] = f"{k + 1:02d}_{p_['name']}"
        p_.setdefault("look", "AgX - Medium High Contrast")
    return P


def level_godot(p):
    """alleys.level for Blender, keeping the tilted camera for Godot (no lens shift there)."""
    from alleys import level
    q = level(p)
    q["godot"] = {"look_at": p["look_at"], "lens": p["lens"]}
    return q


def stage_photos(engine="CYCLES", res=2400, samples=192, only=None, godot_only=False):
    from kaist_village import BLENDER
    from photo_tour import GODOT, sheet
    st = site()
    d = city_dir()
    photos = plan_photos(d)
    pj = d / "photos.json"
    pj.write_text(json.dumps(photos, indent=1))
    rc = 0 if godot_only else subprocess.run([BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"),
                         "--", "--tile", str(d), "--photos", str(pj), "--res", str(res), "--samples", str(samples), "--out", "photos",
                         "--engine", engine, *(["--only", only] if only else [])],
                        stdout=open(d / "photos.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    if rc:
        return rc
    if not godot_only:
        sheet(d / "photos", [p["name"] for p in photos])
    if not GODOT.exists():
        return rc
    used = {p["name"]: p["location"] for p in photos}                # the cameras as Blender placed them (if rendered)
    used.update(json.loads((d / "photos" / "photo_timing.json").read_text())["cameras"])
    gd = lambda v: [v[0], v[2], -v[1]]
    vj = json.loads((d / "viewer.json").read_text())
    vj["shots"] = {p["name"]: {"pos": gd(used[p["name"]]), "look": gd(p.get("godot", p)["look_at"]),
                               "fov": float(np.degrees(2 * np.arctan(18.0 / p.get("godot", p)["lens"] * 9 / 16)))}
                   for p in photos if p["name"] in used}
    vp = d / "photos" / "godot_viewer.json"
    vp.write_text(json.dumps(vj))
    rc = subprocess.run([str(GODOT), "--path", str(ROOT / "godot" / "viewer"), "--resolution", "1920x1080", "--",
                         "--glb", str(d / "scene.glb"), "--viewer", str(vp), "--light", "sunset",
                         "--shot", str(d / "photos" / "godot"), "--extent", str(st.L), "--frames", "20"],
                        stdout=open(d / "photos_godot.log", "w"), stderr=subprocess.STDOUT, timeout=1200).returncode
    print("godot", rc, flush=True)
    return rc


def stage_qa(n_street=12, n_drone=6, glb=True):
    """Review in Godot, as the user explores it: the GLB (m11_scene with one quick render), then fixed random cameras
    (seeded, comparable between builds) at street level along the town's streets and as drones over it, plus the two
    downtowns -> <city>/qa/godot_*.png and qa/sheet.jpg."""
    from kaist_village import BLENDER, terrain_z
    from photo_tour import GODOT
    from PIL import Image
    st = site()
    d = city_dir()
    if glb:
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
                        "--", "--tile", str(d), "--res", "800", "--light", "sunset", "--views", "aerial"],
                       stdout=open(d / "blender_qa.log", "w"), stderr=subprocess.STDOUT, check=True)
    mask = np.load(OUT / "town_mask.npz")["mask"]
    z_at = terrain_z(town_terrain(), st.L)
    S_ = np.load(d / "streets.npy")
    mid = (S_[:, :2] + S_[:, 2:4]) / 2
    r, c = st.rc(mid)
    ok = mask[r, c] & (np.hypot(*(S_[:, 2:4] - S_[:, :2]).T) > 8) & (S_[:, 5] == 0)
    rng = np.random.default_rng(7)
    idx = rng.choice(np.flatnonzero(ok), n_street + n_drone, replace=False)
    gd = lambda v: [float(v[0]), float(v[2]), float(-v[1])]
    shots = {}
    for k, i in enumerate(idx):
        p0 = mid[i]
        u = (S_[i, 2:4] - S_[i, :2]) / np.hypot(*(S_[i, 2:4] - S_[i, :2]))
        u = u if k % 2 else -u
        z0 = z_at(*p0)
        if k < n_street:
            pos, look = [*p0, z0 + 1.7], [*(p0 + u * 60), z_at(*(p0 + u * 60)) + 1.7]
            shots[f"street_{k:02d}"] = {"pos": gd(pos), "look": gd(look), "fov": 62.0}
        else:
            pos = [*(p0 - u * 30), z0 + 35.0]
            shots[f"drone_{k - n_street:02d}"] = {"pos": gd(pos), "look": gd([*(p0 + u * 40), z0]), "fov": 58.0}
    for j, dist_ in enumerate(json.loads((OUT / "district.json").read_text())):
        cx, cy = dist_["center"]
        zc = z_at(cx, cy)
        shots[f"downtown_{j + 1}"] = {"pos": gd([cx - 260, cy - 200, zc + 90]), "look": gd([cx, cy, zc + 40]), "fov": 50.0}
    vj = json.loads((d / "viewer.json").read_text())
    vj["shots"] = shots
    q = d / "qa"
    q.mkdir(exist_ok=True)
    (q / "qa_viewer.json").write_text(json.dumps(vj))
    rc = subprocess.run([str(GODOT), "--path", str(ROOT / "godot" / "viewer"), "--resolution", "1600x900", "--",
                         "--glb", str(d / "scene.glb"), "--viewer", str(q / "qa_viewer.json"), "--light", "sunset",
                         "--shot", str(q / "godot"), "--extent", str(st.L), "--frames", "20"],
                        stdout=open(d / "qa_godot.log", "w"), stderr=subprocess.STDOUT, timeout=1500).returncode
    names = list(shots)
    cols, w = 4, 640
    h = int(w * 9 / 16)
    sheet_ = Image.new("RGB", (cols * w, -(-len(names) // cols) * h), "white")
    for k, nm in enumerate(names):
        f = q / f"godot_{nm}.png"
        if f.exists():
            sheet_.paste(Image.open(f).convert("RGB").resize((w, h)), ((k % cols) * w, (k // cols) * h))
    sheet_.save(q / "sheet.jpg", quality=85)
    print("qa", rc, len(names), flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "build", "backdrop", "render", "photos", "qa"])
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--only", default=None)
    ap.add_argument("--godot-only", action="store_true")
    a = ap.parse_args()
    if a.stage == "plan":
        stage_plan()
        return 0
    if a.stage == "photos":
        return stage_photos(a.engine, a.res, a.samples, a.only, a.godot_only)
    if a.stage == "qa":
        return stage_qa()
    return {"build": stage_build, "backdrop": stage_backdrop, "render": stage_render}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
