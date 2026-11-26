"""Latin theme on Petra (user request 2026-11-25): a dense colonial desert town on the NA-fixed · F1+F4 streets (seed 1)
of the Petra site benchmark (docs/SITE_BENCHMARKS.md), with one small downtown (landmark towers, red skyscrapers,
colonial blocks), plaza churches with giant towers and the photo churches; the desert of Petra around it.

* Town edge = the outer streets. The benchmark's city network (cropped to the city limits) is cut into the faces its
  streets enclose (shapely.polygonize); the town is their union (pieces < 1 ha dropped), so every block is bounded by
  streets and the outermost streets are the edge of the town: an organic outline, desert beyond. Streets are kept when
  both ends lie on or inside the town; spurs leading out into the desert are dropped.
* Streets, plazas, cut and fill, houses: as the Machu Picchu town v3 (machu_picchu.py, DESIGN.md §5.6): smooth
  continuous road levels with blended junctions, the ground sunk under the paving, houses seated at street level with at
  most 2 storeys below it, lanes through big blocks, the dense colonial infill.
* Desert (``--jungle desert``, latin.plants / latin.build): desert floor and Petra's rose sandstone on the terrain,
  sparse dry shrubs, agaves and columnar cacti outside the town; Washingtonia and date palms on the streets and plazas,
  bougainvillea in the yards; the GLO-30 mountains around in sandstone and sand.

    .venv/bin/python themes/latin_caribbean/scripts/petra.py plan       # town, graph, terrain, plazas, downtown, map
    .venv/bin/python themes/latin_caribbean/scripts/petra.py build      # the 3D city (m16_city3d_big.py)
    .venv/bin/python themes/latin_caribbean/scripts/petra.py backdrop   # the mountains around
    .venv/bin/python themes/latin_caribbean/scripts/petra.py render     # Blender views at sunset
    .venv/bin/python themes/latin_caribbean/scripts/petra.py qa         # Godot street-level review sheet
    .venv/bin/python themes/latin_caribbean/scripts/petra.py photos     # Cycles photos + Godot shots
    .venv/bin/python themes/latin_caribbean/scripts/petra.py art        # art photos of one evening (+ Godot shots)
    .venv/bin/python themes/latin_caribbean/scripts/petra.py skyline    # user request 2026-11-26: 8 towers + a park (skyline.json)
    ... build | backdrop | render | viewer | qa --variant skyline       # that town, in its own folder (petra_skyline_...)
    .venv/bin/python themes/latin_caribbean/scripts/petra.py viewer     # crowns for the dusk / night floodlights (viewer.json)
    ->  outputs/latin_caribbean/petra/ (plan) and outputs/latin_caribbean/cities/petra/<city>/
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

from machu_picchu import WALK_M, dem_corners, district_tiles, engineer, level_godot, street_plazas

OUT = ROOT / "outputs" / "latin_caribbean" / "petra"
CITIES = ROOT / "outputs" / "latin_caribbean" / "cities" / "petra"
METHOD, VERSION, SEED = "NA-fixed", "F1 + F4", 1
LABEL = "NA-fixed [improved]"
TAG = "petra"
CENTRE = (1037.0, 841.0)                          # Petra's ancient city centre, domain metres (configs/sites/petra.json)
GIANT = "120,95"                                  # plaza church towers (m), tallest first
GIANT_LOTTERY = "55,80"
CHURCH_SITES = [("gothic", None), ("bahia", None), (None, 70.0), ("gothic", None), (None, None), ("bahia", None)]
DISTRICT_N, DISTRICT_H = 10, "55,115"             # one small downtown about the main plaza
LANDMARKS = ["golden:rojo", "coral:bermellon", "eagle:coral", "golden:terracota"]
PHOTO_CHURCHES = {"r1c1": "gothic", "r1c2": "bahia", "r2c1": "bahia", "r2c2": "gothic", "r0c2": "gothic", "r2c3": "bahia"}
MIN_FACE_HA = 1.0
SKY_N, SKY_H = 8, (68.0, 92.0)                # the skyline cluster: 8 towers side by side, taller toward its centre; the
                                               # scale of the downtown's own landmarks (60-88 m, 5.5-6.4 x the shaft), so
                                               # that the crowns keep their proportion (user, 2026-11-26: 85-135 m too tall)
PARK_SIZE = (56.0, 40.0)                       # the park: a lawn fronting a street, facing the cluster
VARIANT = None                                 # --variant skyline: build that town into its own folder
SLIVER_M = 70.0


def site():
    from site_benchmark import Site
    return Site("petra")


def street_town(st, g, limits, cw):
    """The town = the union of the faces the streets enclose (pieces < MIN_FACE_HA dropped); the streets on or inside it;
    the 4 m mask = the town plus the outer half of its edge streets and their sidewalks."""
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    seg = shapely.linestrings(g.nodes[g.edges])
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(shapely.node(shapely.multilinestrings(seg)))))
    parts = [p for p in shapely.get_parts(shapely.union_all(faces)) if p.area >= MIN_FACE_HA * 1e4]
    town = shapely.union_all([shapely.Polygon(p.exterior) for p in parts])          # no holes: a face is a block
    core = town.buffer(-SLIVER_M / 2).buffer(SLIVER_M / 2)          # edge faces mostly narrower than SLIVER_M: spikes
    faces = [f for f in faces if f.intersects(town) and f.intersection(core).area >= 0.5 * f.area]
    parts = [p for p in shapely.get_parts(shapely.union_all(faces)) if p.area >= MIN_FACE_HA * 1e4]
    town = shapely.union_all([shapely.Polygon(p.exterior) for p in parts])
    inside = shapely.contains_xy(town.buffer(0.5), g.nodes[:, 0], g.nodes[:, 1])
    keep = inside[g.edges].all(1)
    L_e = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1)
    for _ in range(3):                                            # stubs < 10 m (site_benchmark.crop)
        deg = np.bincount(g.edges[keep].ravel(), minlength=len(g.nodes))
        keep &= ~((L_e < 10) & ((deg[g.edges[:, 0]] == 1) | (deg[g.edges[:, 1]] == 1)))
    E = g.edges[keep]
    used, inv = np.unique(E, return_inverse=True)
    N = g.nodes[used]
    gt = RoadGraph(N, inv.reshape(-1, 2), is_border(N, size=st.L), edge_class=g.classes()[keep], edge_struct=g.structs()[keep])
    edge = shapely.union_all([shapely.LineString(gt.nodes[e]).buffer(cw[int(k)] / 2 + WALK_M + 0.5)
                              for e, k in zip(gt.edges, gt.classes())])
    X, Y = st.xy_grid()
    mask = shapely.contains_xy(shapely.union_all([town, edge]), X, Y)
    return mask, town, gt


def stage_plan():
    from m6_build_objects import load_objects
    from kaist_village import district_of
    st = site()
    big = st.domain()
    limits = np.load(st.data / "city_mask.npz")["mask"]
    g0 = pickle.loads((st.out / "city_graphs.pkl").read_bytes())[(METHOD, VERSION, SEED)]
    cw = load_objects()["class_widths"]
    mask, town, g = street_town(st, g0, limits, cw)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "town_mask.npz", mask=mask)
    slope, _, _ = st.slope(big)
    sites = street_plazas(st, g, mask, slope, cw, n=2, spacing=260.0, centre=CENTRE)
    for s_, h in zip(sites, [float(x) for x in GIANT.split(",")]):
        s_["church"]["tower_h"] = h
    small = street_plazas(st, g, mask, slope, cw, n=len(CHURCH_SITES), spacing=150.0, max_w=26.0,
                          sizes=((46.0, 30.0), (40.0, 28.0)), avoid=[s_["church"]["center"] for s_ in sites], max_slope=0.45,
                          centre=CENTRE)
    for s_, (style, h) in zip(small, CHURCH_SITES):
        if style:
            s_["church"]["style"] = style
        if h:
            s_["church"]["tower_h"] = h
    sites += small
    (OUT / "plaza_sites.json").write_text(json.dumps(sites, indent=1))
    centre = np.asarray(shapely.Polygon(sites[0]["plaza"][0]).centroid.coords)[0]
    district = district_of(g, centre, DISTRICT_N, [float(x) for x in DISTRICT_H.split(",")])
    district.update(lot_front=[21.0, 28.0], lot_depth=[20.0, 26.0], fill="houses")   # wider lots, no gardens: dense
    (OUT / "district.json").write_text(json.dumps([district], indent=1))
    elev, road_z, erep = engineer(st, big, g, mask, sites, cw)
    np.savez_compressed(OUT / "terrain_town.npz", elev=elev, water=big["water"], road_z=road_z,
                        masonry=ndimage.binary_dilation(mask, iterations=3),
                        water_surface=np.full(elev.shape, np.nan, np.float32))
    (OUT / "graphs.pkl").write_bytes(pickle.dumps({"graphs": {(LABEL, SEED): g}, "real": None}))
    Ln = lambda gg: float(np.linalg.norm(gg.nodes[gg.edges[:, 0]] - gg.nodes[gg.edges[:, 1]], axis=1).sum() / 1000)
    X, Y = st.xy_grid()
    info = {"method": METHOD, "version": VERSION, "seed": SEED, "town_km2": float(town.area / 1e6),
            "town_mask_km2": float(mask.sum() * 16 / 1e6), "town_pieces": len(shapely.get_parts(town)),
            "city_limits_km2": float(limits.sum() * 16 / 1e6), "street_km_city": Ln(g0), "street_km_town": Ln(g),
            "plazas": len(sites), "plaza_slope": [float(np.median(slope[shapely.contains_xy(shapely.Polygon(s_["plaza"][0]), X, Y)]))
                                                  for s_ in sites],
            "town_slope_median": float(np.median(slope[mask])),
            "district": {k: v for k, v in district.items() if k != "polygon"}, "tiles": district_tiles(district),
            "engineering": erep}
    (OUT / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))
    plan_map(st, big, limits, mask, town, g0, g, sites, [district], elev)


def stage_skyline():
    """User request 2026-11-26: eight more tall buildings close together in the downtown, and a small park with a fine
    view of them. Planned on top of the existing town (the town itself is unchanged: latin.infill.overlay removes only
    the houses the towers and the park touch).
    * Cluster: the gentlest spot of the downtown (median slope within 45 m), 70-150 m from the main plaza's church and
      40 m clear of every plaza; a district of 3 streets about it (more if needed); machu_picchu.cluster_sites (20 x 20 m
      sites, 4 m apart, taller toward the centre, SKY_H), the SKY_N nearest the centre clear of the plazas; the CLUSTER
      landmark designs (theme option landmark_districts, district 1).
    * Park: PARK_SIZE lawns fronting a street (machu_picchu.street_plazas candidates, 50 m apart), 180-380 m from the
      cluster; the one with the most clearance between the line of sight (eye 1.7 m above the park to 60 % of the
      cluster's mean height) and the rooftops (terrain + 13 m) along the way.
    -> skyline.json, skyline_plan.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kaist_village import district_of
    from m6_build_objects import load_objects
    from machu_picchu import CLUSTER, cluster_sites
    st = site()
    big = st.domain()
    g = pickle.loads((OUT / "graphs.pkl").read_bytes())["graphs"][(LABEL, SEED)]
    mask = np.load(OUT / "town_mask.npz")["mask"]
    cw = load_objects()["class_widths"]
    slope, _, _ = st.slope(big)
    elev = np.load(OUT / "terrain_town.npz")["elev"]
    sites = json.loads((OUT / "plaza_sites.json").read_text())
    down = json.loads((OUT / "district.json").read_text())[0]
    plazas = shapely.union_all([shapely.Polygon(s_["plaza"][0]) for s_ in sites])
    main = np.asarray(sites[0]["church"]["center"], float)
    X, Y = st.xy_grid()
    z_at = lambda P: elev[st.rc(P)]
    c0, R0 = np.asarray(down["center"], float), float(down["radius"])
    best = None
    for x in np.arange(c0[0] - R0, c0[0] + R0, 10.0):
        for y in np.arange(c0[1] - R0, c0[1] + R0, 10.0):
            q = np.array([x, y])
            if np.hypot(*(q - c0)) > R0 or not 70.0 <= np.hypot(*(q - main)) <= 150.0 or plazas.distance(shapely.Point(q)) < 40.0:
                continue
            if not mask[st.rc(q)][0]:
                continue
            near = ((X - x) ** 2 + (Y - y) ** 2 < 45.0 ** 2) & mask
            sc = float(np.median(slope[near]))
            if best is None or sc < best[0]:
                best = (sc, q)
    tc = best[1]
    for n_st in (3, 4, 5, 6):
        dist = district_of(g, tc, n_st, SKY_H)
        cand = cluster_sites(st, g, mask, cw, dist, n=3 * SKY_N, W=20.0, D=20.0, gap=4.0)
        cand = [t for t in cand if not shapely.Point(t["center"]).buffer(16.0).intersects(plazas.buffer(5.0))]
        if len(cand) >= SKY_N:
            break
    cand.sort(key=lambda t: np.hypot(*(np.asarray(t["center"]) - tc)))
    towers = [{**t, "district": 1, "landmark": CLUSTER[k % len(CLUSTER)]} for k, t in enumerate(cand[:SKY_N])]
    lo, hi = SKY_H                                       # heights: taller toward the cluster's centre, as cluster_sites
    from urbanmorph.city3d.latin.landmarks import city_colours, landmark_on_lot
    used = []
    for k, t in enumerate(towers):
        t["H"] = float(hi - (hi - lo) * k / max(len(towers) - 1, 1))
        for j in range(len(CLUSTER)):                    # a design that can be built at that height (cohesive towers
            spec = CLUSTER[(k + j) % len(CLUSTER)]       # come in whole tiers under a big crown: some cannot be short)
            if spec in used:
                continue
            d_, f_, w_ = spec.split(":")
            _, info = landmark_on_lot(d_, t["center"], t["axis"], t["W"], t["D"], t["H"], np.random.default_rng(0),
                                      city_colours(d_, w_), form=f_)
            if abs(info["height_m"] - t["H"]) <= 6.0:
                t["landmark"] = spec
                used.append(spec)
                break
    tc = np.mean([t["center"] for t in towers], axis=0)
    z_c, h_mean = float(z_at(tc[None])[0]), float(np.mean([t["H"] for t in towers]))
    pk = street_plazas(st, g, mask, slope, cw, n=60, spacing=50.0, sizes=(PARK_SIZE,), avoid=[s_["church"]["center"] for s_ in sites] + [tc.tolist()],
                       max_slope=0.35, centre=tc)
    prev = json.loads((OUT / "skyline.json").read_text()) if (OUT / "skyline.json").exists() else None
    rows = []
    for s_ in pk:
        R = shapely.Polygon(s_["plaza"][0])
        q = np.asarray(R.centroid.coords[0])
        d = float(np.hypot(*(q - tc)))
        if not 180.0 <= d <= 380.0 or R.intersects(plazas.buffer(10.0)):
            continue
        z0 = float(z_at(q[None])[0]) + 1.7
        z1 = z_c + 0.6 * h_mean
        t_ = np.arange(30.0, d - 40.0, 5.0) / d
        P = q + np.outer(t_, tc - q)
        clear = float(np.min(z0 + t_ * (z1 - z0) - (z_at(P) + 13.0))) if len(P) else 0.0
        rows.append({"polygon": s_["plaza"], "center": q.tolist(), "distance_m": d, "ground_z": z0 - 1.7, "clearance_m": clear})
    park = max(rows, key=lambda r: r["clearance_m"])
    if prev is not None:                                 # keep the park the user has seen; its view to the new towers
        old = prev["parks"][0]
        same = [r for r in rows if np.hypot(*(np.asarray(r["center"]) - old["center"])) < 1.0]
        park = same[0] if same else {k: v for k, v in old.items() if k != "view_to"}
    out = {"request": "2026-11-26: 8 more tall buildings close together in the downtown, a small park with a view of the skyline",
           "rule": stage_skyline.__doc__.split("*", 1)[1].strip(), "cluster": {"center": tc.tolist(), "district": {k: v for k, v in dist.items() if k != "polygon"},
           "base_z": z_c}, "towers": towers, "parks": [{**park, "view_to": tc.tolist()}], "park_candidates": len(rows)}
    (OUT / "skyline.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"cluster_centre": tc.round(1).tolist(), "towers": [(round(t["H"]), t["landmark"]) for t in towers],
                      "park": {k: round(v, 1) for k, v in park.items() if isinstance(v, float)}, "park_candidates": len(rows)}, indent=1))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.where(mask, elev, np.nan), extent=(-256, st.L + 256, -256, st.L + 256), cmap="Greys_r")
    for e in g.edges:
        ax.plot(*g.nodes[e].T, color="#333", lw=0.6)
    for s_ in sites:
        ax.fill(*np.asarray(s_["plaza"][0]).T, color="#e0a020", alpha=0.7)
    for t in towers:
        c_, u = np.asarray(t["center"]), np.asarray(t["axis"])
        nrm = np.array([-u[1], u[0]])
        ax.fill(*np.array([c_ + sx * u * 10 + sy * nrm * 10 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]).T, color="#c0392b")
    ax.fill(*np.asarray(park["polygon"][0]).T, color="#2e8b57")
    ax.plot(*np.array([park["center"], tc]).T, color="#2e8b57", ls="--", lw=1)
    ax.set_xlim(tc[0] - 450, tc[0] + 450)
    ax.set_ylim(tc[1] - 450, tc[1] + 450)
    ax.set_title("skyline: 8 towers (red), the park (green) and its view; plazas (gold)", loc="left", fontsize=10)
    fig.savefig(OUT / "skyline_plan.png", dpi=90)
    return 0


def plan_map(st, big, limits, mask, town, g0, g, sites, district, elev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from site_benchmark import MARGIN, contours, hillshade
    L = st.L
    fig, axs = plt.subplots(1, 2, figsize=(20, 11))
    for ax, (title, gg) in zip(axs, (("benchmark city network (NA-fixed · F1+F4, seed 1); town = the faces its streets enclose "
                                      "(yellow)", g0),
                                     ("the town: streets kept, plazas (gold), the downtown (red), cut-and-fill terrain", g))):
        ax.imshow(hillshade(st, {**big, "elev": elev} if gg is g else big, mask), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN))
        contours(ax, st, {**big, "elev": elev} if gg is g else big, step=25, lw=0.3)
        ax.contour(*st.xy_grid(), limits.astype(float), levels=[0.5], colors="#7a4a22", linewidths=1.0, linestyles="--")
        if gg is g0:
            for p in shapely.get_parts(town):
                xy = np.asarray(p.exterior.coords)
                ax.fill(xy[:, 0], xy[:, 1], color="#f5c242", alpha=0.25)
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


def city_dir():
    return CITIES / f"{TAG}_{LABEL.replace(' [', '_').replace(']', '')}_s{SEED}_latin"


def stage_build():
    info = json.loads((OUT / "info.json").read_text())
    share = info["tiles"]                                         # landmarks by the downtown's share per tile
    cnt = {t: int(round(len(LANDMARKS) * a)) for t, a in share.items()}
    cnt[max(share, key=share.get)] += len(LANDMARKS) - sum(cnt.values())
    per, k = {}, 0
    for t, m in cnt.items():
        if m > 0:
            per[t], k = LANDMARKS[k:k + m], k + m
    topts = {"church_colour": "rojo", "church_dome": "cupula_naranja", "landmarks": per, "photo_churches": PHOTO_CHURCHES,
             "landmark_min": [13.0, 60.0], "landmark_spacing_m": 30.0,
             "churches_per_tile": 2, "church_spacing_m": 150.0, "giant_lottery_p": 0.35}
    py = str(ROOT / ".venv" / "bin" / "python")
    cmd = [py, str(ROOT / "scripts" / "m16_city3d_big.py"), "--method", LABEL, "--seed", str(SEED), "--graphs", str(OUT / "graphs.pkl"),
           "--tag", TAG, "--theme", "latin_caribbean", "--cities-dir", str(CITIES), "--terrain-npz", str(OUT / "terrain_town.npz"),
           "--village-npz", str(OUT / "town_mask.npz"), "--jungle", "desert", "--infill", "colonial",
           "--plazas", str(OUT / "plaza_sites.json"), "--giant-lottery", GIANT_LOTTERY, "--lanes", "60", "--below-max", "2",
           "--cliff-m", "9", "--district", str(OUT / "district.json"), "--theme-opts", json.dumps(topts), "--seat", "street",
           "--plinth", "latin:masonry", "--build-slope", "9", "--village-clip", "--site", "Petra site benchmark"]
    if VARIANT == "skyline":
        from machu_picchu import CLUSTER
        topts["landmark_districts"] = {"1": CLUSTER}
        cmd[cmd.index("--theme-opts") + 1] = json.dumps(topts)
        cmd += ["--overlay", str(OUT / "skyline.json")]
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


def stage_viewer():
    """Godot viewer extras in the city's viewer.json (run after build / render):
    * "crowns": the crowns of the tall towers (giant church towers, landmarks, skyscrapers), which the viewer floodlights
      in its --light dusk / night modes (user request 2026-11-26: warm, romantic crowns seen from a distance);
    * the skyline variant: the walker starts in the park facing the skyline, plus the skyline views."""
    st = site()
    d = city_dir()
    vj = json.loads((d / "viewer.json").read_text())
    elev = town_terrain()["elev"]
    z = lambda p_: float(elev[st.rc(np.asarray(p_, float)[None])][0])
    gd = lambda p_: [float(p_[0]), float(p_[2]), -float(p_[1])]
    gt = json.loads((d / "giant_towers.json").read_text())
    crowns, seen = [], []
    for kind, lst, share, r in (("church", gt["towers"], 0.28, None), ("landmark", gt["landmarks"], 0.40, 8.5),
                                ("skyscraper", gt.get("skyscrapers", []), 0.18, 8.0)):
        for t in lst:
            xy = np.asarray(t["xy"], float)
            if any(np.hypot(*(xy - q)) < 3.0 for q in seen):
                continue
            seen.append(xy)
            H = float(t["height_m"])
            rr = float(t["side_m"]) / 2 if r is None else r
            crowns.append({"kind": kind, "c": gd(np.r_[xy, z(xy) + H * (1 - share / 2)]), "r": rr, "h": share * H})
    vj["crowns"] = crowns
    if VARIANT == "skyline":
        sky = json.loads((OUT / "skyline.json").read_text())
        tc, zc = np.asarray(sky["cluster"]["center"]), sky["cluster"]["base_z"]
        pc = np.asarray(sky["parks"][0]["center"])
        v = (tc - pc) / np.hypot(*(tc - pc))
        n_ = np.array([-v[1], v[0]])
        eye = pc - v * 14.0
        base = json.loads((CITIES / f"petra_{LABEL.replace(' [', '_').replace(']', '')}_s{SEED}_latin" / "viewer.json").read_text())
        vj["shots"] = {**base.get("shots", {}), **vj.get("shots", {})}
        vj["spawn"], vj["look_at"] = gd(np.r_[eye, z(eye) + 2.0]), gd(np.r_[tc, zc + 40.0])
        side = pc - v * 4.0 + n_ * 18.0
        dr = pc + (tc - pc) * 0.25 - n_ * 120.0
        t0, u0 = np.asarray(sky["towers"][0]["center"]), np.asarray(sky["towers"][0]["axis"])
        cs = t0 + np.array([-u0[1], u0[0]]) * 24.0
        vj["shots"].update({
            "park_view": {"pos": gd(np.r_[eye, z(eye) + 1.7]), "look": gd(np.r_[tc, zc + 38.0]), "fov": 50.0},
            "park_view_tele": {"pos": gd(np.r_[eye, z(eye) + 1.7]), "look": gd(np.r_[tc, zc + 45.0]), "fov": 24.0},
            "park_lawn": {"pos": gd(np.r_[side, z(side) + 1.7]), "look": gd(np.r_[tc, zc + 35.0]), "fov": 62.0},
            "skyline_drone": {"pos": gd(np.r_[dr, z(dr) + 80.0]), "look": gd(np.r_[tc, zc + 40.0]), "fov": 45.0},
            "cluster_street": {"pos": gd(np.r_[cs, z(cs) + 1.7]), "look": gd(np.r_[tc, z(cs) + 55.0]), "fov": 70.0}})
    (d / "viewer.json").write_text(json.dumps(vj))
    print("crowns", len(crowns), {k: sum(c["kind"] == k for c in crowns) for k in ("church", "landmark", "skyscraper")})
    return 0


RINGS = ((-4096.0, 10240.0, 32.0), (-14336.0, 30720.0, 160.0))     # backdrop squares: west / south edge, side, cell (m)
CLIFF_DEG = (22.0, 12.0)                     # backdrop faces steeper than this (inner, outer ring): sandstone
SMOOTH_DEG = 60.0                            # Blender / GLB: smooth shading of the terrain and the backdrop


def stage_backdrop():
    """The mountains around the town: GLO-30 terrain in two rings (32 m to 10 km, 160 m to 30 km) with a hole for the
    2048 m domain, desert sand on the gentle land and Petra's sandstone on the slopes (no water: the wadis are dry)."""
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
    m_sand, m_rock = mat("latin:desert_far"), mat("latin:sandstone_far")
    VV, FF, MM, n0 = [], [], [], 0
    holes = [(0.0, st.L), (RINGS[0][0], RINGS[0][0] + RINGS[0][1])]
    for (x0, side, res), (h0, h1), cliff in zip(RINGS, holes, CLIFF_DEG):
        z, n = dem_corners(st, x0, side, res)
        xs = x0 + np.arange(n + 1) * res
        ys = x0 + side - np.arange(n + 1) * res
        X, Y = np.meshgrid(xs, ys)
        V = np.c_[X.ravel(), Y.ravel(), z.ravel()]
        r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cx, cy = x0 + (c + 0.5) * res, x0 + side - (r + 0.5) * res
        keep = ~((cx > h0) & (cx < h1) & (cy > h0) & (cy < h1))
        r, c = r[keep], c[keep]
        a, b, dd, e = r * (n + 1) + c, r * (n + 1) + c + 1, (r + 1) * (n + 1) + c + 1, (r + 1) * (n + 1) + c
        F = np.vstack([np.c_[a, e, dd], np.c_[a, dd, b]])
        P = V[F]
        nz = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        steep = np.degrees(np.arccos(np.clip(np.abs(nz[:, 2]) / np.maximum(np.linalg.norm(nz, axis=1), 1e-9), 0, 1))) > cliff
        MM.append(np.where(steep, m_rock, m_sand))
        if res == RINGS[0][2]:                                   # a skirt under the inner ring's outer edge (ring cracks)
            ring = [(0, j) for j in range(n)] + [(i, n) for i in range(n)] + [(n, n - j) for j in range(n)] + [(n - i, 0) for i in range(n)]
            ids = np.array([i * (n + 1) + j for i, j in ring] + [0])
            low = V[ids] - [0, 0, 40.0]
            k0 = len(V)
            V = np.vstack([V, low])
            q = np.arange(len(ids) - 1)
            sk = np.vstack([np.c_[ids[q], ids[q + 1], k0 + q + 1], np.c_[ids[q], k0 + q + 1, k0 + q]])
            F = np.vstack([F, sk, sk[:, ::-1]])
            MM.append(np.full(2 * len(sk), m_rock))
        VV.append(V)
        FF.append(F + n0)
        n0 += len(V)
    V, F = np.vstack(VV).astype(np.float32), np.vstack(FF).astype(np.int32)
    M = np.concatenate(MM).astype(np.int16)
    groups["backdrop"] = {"V": V, "F": F, "mat": M, "obj": np.full(len(F), -1, np.int32), "UV": corner_uvs(V.astype(float), F)}
    save_scene(d / "meshes.npz", groups, mats)
    cams = json.loads((d / "cameras.json").read_text())             # the render-only surround plane under the backdrop
    cams["surround_z"] = float(V[:, 2].min()) - 100.0               # (at the town's lowest level it showed as lakes in
    cams["smooth_terrain"] = SMOOTH_DEG                             # the western valleys, which fall to Wadi Araba)
    (d / "cameras.json").write_text(json.dumps(cams, indent=1))
    print("backdrop triangles", len(F), "sandstone faces", int((M == m_rock).sum()), flush=True)
    return 0


STREET_VIEWS = {"main_tower": "church_tower", "landmark_street": "landmark", "photo_church": "church_"}   # best view of each kind
STREET_SKIP = {"a04_church_bahia"}                    # checked by eye: a street crest hides the church
UMM_AL_BIYARA = (430.0, 640.0)                    # the massif west of the town (domain metres, GLO-30)
AD_DEIR = (110.0, 1800.0)                         # the Monastery plateau, north-west


def high_point(st, xy, r=120.0):
    """The highest GLO-30 cell within r of xy (a viewpoint on a summit)."""
    big = st.domain()
    X, Y = st.xy_grid()
    sel = np.hypot(X - xy[0], Y - xy[1]) < r
    k = np.argmax(np.where(sel, big["elev"], -np.inf))
    return float(X.ravel()[k]), float(Y.ravel()[k]), float(big["elev"].ravel()[k])


def plan_photos(d):
    """Hero photos (photo_tour.py; sun_az counter-clockwise from east: 180 = the west). The sunset sun comes from the
    west, over Umm al-Biyara: views from the west and south have it behind or beside the camera."""
    import alleys
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
    air = dict(haze_m=30000, clouds=0.2, glow=0.3, exposure=-0.35)              # dry desert air, a few clouds
    P = []
    # the town from Umm al-Biyara (the western massif), the sun behind the camera, the eastern cliffs behind the town
    ux, uy, uz = high_point(st, UMM_AL_BIYARA)
    P.append(dict(name="postcard", location=[ux, uy, uz + 12.0], look_at=[dt[0] + 250, dt[1] + 60, zd - 10], lens=30, sun_el=8,
                  sun_az=190, **air))
    # the whole town from the south-west: its edge along the outer streets, the desert all around
    P.append(dict(name="town_and_desert", location=[350.0, 80.0, zd + 420], look_at=[1080.0, 820.0, zd - 40], lens=28, sun_el=10,
                  sun_az=200, **air))
    # from the Monastery plateau down over the town
    ax_, ay_, az_ = high_point(st, AD_DEIR)
    P.append(dict(name="from_ad_deir", location=[ax_, ay_, az_ + 15.0], look_at=[1000.0, 850.0, zd - 40], lens=34, sun_el=8,
                  sun_az=195, **air))
    # into the sunset from the eastern cliffs: the towers against the sky over Umm al-Biyara
    P.append(dict(name="sunset_basin", location=[1820.0, 820.0, zd + 140], look_at=[700.0, 780.0, zd + 70], lens=40,
                  sun_el=4.0, sun_az=182, sun_disc=True, haze_m=40000, clouds=0.3, glow=0.15, exposure=-1.1))
    # street views of the towers
    for nm, kind in STREET_VIEWS.items():
        best = [p_ for p_ in cand.values() if p_["target"].startswith(kind) and (kind != "church_" or p_["target"] != "church_tower")
                and p_["name"] not in STREET_SKIP]
        if best:
            v = max(best, key=lambda p_: p_["score"])
            P.append(level_godot(dict(name=nm, location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=6,
                                sun_az=v["sun_az"], haze_m=12000, clouds=0.2, glow=0.6, top_pad_deg=2.5)))
    if "alley" in cams:
        v = cams["alley"]
        P.append(level_godot(dict(name="alley", location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=6, sun_az=190,
                            haze_m=12000, clouds=0.2, glow=0.7, top_pad_deg=2.0)))
    # the edge of the town from the desert to the south: the outer street, the houses behind it. The camera stands
    # 120 m out where the desert is about level with the edge street (else a rise hides the town or it is seen from above)
    mask = np.load(OUT / "town_mask.npz")["mask"]
    X, Y = st.xy_grid()
    best = None
    for x in np.arange(700.0, 1500.0, 25.0):
        col = int(np.argmin(np.abs(X[0] - x)))
        if not mask[:, col].any():
            continue
        ye = float(Y[:, col][mask[:, col]].min())
        dz = zg(x, ye - 120) - z_at(x, ye + 6)
        if -5 <= dz <= 25 and (best is None or abs(dz - 8) < best[0]):
            best = (abs(dz - 8), x, ye)
    if best:
        _, x, ye = best
        P.append(dict(name="desert_edge", location=at((x, ye - 120), 9.0, zg(x, ye - 120)), look_at=[x, ye + 160, z_at(x, ye + 6) + 12],
                      lens=26, sun_el=7, sun_az=195, **{**air, "haze_m": 20000}))
    # a drone over the town: rows of houses, plazas, palms
    P.append(dict(name="town_drone", location=[dt[0] + 380, dt[1] - 300, zd + 120], look_at=[dt[0], dt[1], zd - 10], lens=28,
                  sun_el=9, sun_az=195, **{**air, "haze_m": 20000}))
    # the downtown at dusk: lit windows, the sun just down
    if "district" in cams:
        v = cams["district"]
        P.append(dict(name="downtown_dusk", location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=0.4, sun_az=185,
                      haze_m=15000, glow=1.0, clouds=0.3, exposure=0.7))
    for k, p_ in enumerate(P):
        p_["name"] = f"{k + 1:02d}_{p_['name']}"
        p_.setdefault("look", "AgX - Medium High Contrast")
    return P


# ---- art photos (user request 2026-11-25: "warm sunset lights, realistic like a Mexican village, very artistic photos,
# aerial, from the street, with a long lens"). One evening, the sun sinking photo by photo. Petra's town lies in a basin:
# the western massif shades it once the sun is below ~10 deg, so the golden hour is a summer evening sun at 11 deg from
# the west-north-west (sun_az 160; sunset azimuths at 30.3 N run 152-208), reddened by desert dust (sun_tint_el) and
# strong against the dusty sky (sun_k: test renders 2026-11-25, the sun at its m11 strength left the town flat).
GOLDEN = dict(sun_el=11.0, sun_az=160.0, sun_k=4.0, sun_tint_el=5.0, aerosol=4.0, exposure=-1.1, clouds=0.3,
              haze_m=9000, glow=0.2)
DUSK = dict(sun_el=-2.0, sun_az=165.0, sun_k=0.0, aerosol=3.0, exposure=1.4, clouds=0.4, haze_m=9000, glow=1.2,
            house_lights={"share": 0.35, "strength": 5.0})           # blue hour: the sun 2 deg down, a third of the windows lit
SUN_WINDOW = (152.0, 208.0)


class Sight:
    """Terrain (cut and fill on the domain grid; the backdrop beyond) and roofs, for sight lines and the horizon."""

    def __init__(self, d, st):
        from alleys import buildings
        from scipy.ndimage import map_coordinates
        self.st, self.mc = st, map_coordinates
        self.elev = town_terrain()["elev"].astype(float)
        self.foot, h = buildings(d)
        self.tree = shapely.STRtree(self.foot)
        self.top = self.z(np.array([np.asarray(p.representative_point().coords)[0] for p in self.foot])) + h
        with np.load(d / "meshes.npz") as mz:
            self.V = np.vstack([mz["ground__V"], mz["backdrop__V"]])
            self.inst = {k[5:-3]: mz[k] for k in mz.files if k.startswith("inst_") and k.endswith("__T")}

    def z(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        r = (self.st.L + 256.0 - 2.0 - P[:, 1]) / 4.0
        c = (P[:, 0] + 256.0 - 2.0) / 4.0
        return self.mc(self.elev, [r, c], order=1, mode="nearest")

    def clear(self, a, b, skip_end=0.0):
        """The segment a -> b (3D) passes over the terrain and every roof (its last skip_end m not tested)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.hypot(*(b - a)[:2]) / 3.0), 2)
        t = np.linspace(0, 1, n)[1:-1]
        t = t[t * np.hypot(*(b - a)[:2]) < np.hypot(*(b - a)[:2]) - skip_end]
        P = a + np.outer(t, b - a)
        if (P[:, 2] < self.z(P[:, :2]) + 0.5).any():
            return False
        i, j = self.tree.query(shapely.points(P[:, :2]), predicate="within")
        return not (self.top[j] > P[i, 2]).any()

    def horizon(self, p, az, near=30.0):
        """The terrain's elevation angle (deg) seen from p towards az (counter-clockwise from east)."""
        d = self.V[:, :2] - np.asarray(p[:2])
        r = np.hypot(*d.T)
        a = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        sel = (np.abs((a - az + 180) % 360 - 180) < 0.8) & (r > near)
        return float(np.degrees(np.arctan2(self.V[sel, 2] - p[2], r[sel])).max())


def bearing(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def unit3(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def sun_for(view_az):
    """Light for a street view: into the sun when it looks west, the sun behind the camera when it looks east, from
    the side otherwise (sun azimuths kept within the year's sunset window)."""
    w = (view_az + 180) % 360 - 180
    if abs(w) >= 125:
        return "backlit", dict(GOLDEN, sun_az=float(np.clip(w % 360 + 4, *SUN_WINDOW)), sun_el=9.0, sun_disc=True,
                               disc_k=0.25, exposure=-0.7, haze_m=4000, dust={"r": 220.0, "density": 0.0012, "g": 0.75})
    if abs(w) <= 45:
        return "frontlit", dict(GOLDEN, sun_az=float(np.clip(w + 180, *SUN_WINDOW)), sun_el=12.0)
    return "sidelit", dict(GOLDEN, sun_el=12.0)


def tele_sun(S, top, els=np.arange(3.0, 9.6, 0.5), azs=np.arange(152.0, 177.0, 2.0)):
    """Where to stand for the sun behind a tower's crown (top: 3D, a few m below the cross): along the line from the
    crown away from the sun, where it meets the ground (a person on the far slope), the crown in sight and the disc
    above the hills behind. -> (camera, az, el): the camera lowest above its ground, then the lowest sun, then the
    longest line."""
    best = None
    for el in els:
        for az in azs:
            s = unit3(az, el)
            for r in np.arange(450.0, 1500.0, 15.0):
                c = top - r * s
                if not (-200 < c[0] < S.st.L + 200 and -200 < c[1] < S.st.L + 200):
                    break
                lift = c[2] - S.z(c[:2])[0]
                if lift < 1.7:
                    break
                if lift > 25.0 or not S.clear(c, top, skip_end=12.0) or S.horizon(c, az) > el - 0.4:
                    continue
                key = (round(lift / 5), el, -r)
                if best is None or key < best[0]:
                    best = (key, c, float(az), float(el))
    if best is None:
        raise RuntimeError("no place for the sun behind the tower")
    return best[1:]


def vantage(S, target, centre, r_max, want_down, lift=1.7):
    """The ground point within r_max of centre (8 m grid) with the target in sight whose view down to it is nearest
    want_down (deg)."""
    g = np.arange(-r_max, r_max + 0.1, 8.0)
    Q = np.array([(x, y) for x in g for y in g if x * x + y * y <= r_max * r_max]) + centre
    Z = S.z(Q) + lift
    down = np.degrees(np.arctan2(Z - target[2], np.hypot(*(Q - target[:2]).T)))
    for k in np.argsort(np.abs(down - want_down)):
        c = np.array([*Q[k], Z[k]])
        if S.clear(c, target, skip_end=30.0) and S.z(Q[k] + 6.0 * (target[:2] - Q[k]) / np.hypot(*(target[:2] - Q[k])))[0] < Z[k] - 1.0:
            return c
    raise RuntimeError(f"no vantage near {centre}")


def grade(f, warm=0.1, vignette=0.2):
    """A photographer's finish on the rendered jpg: warmer white balance (warm: blue down, red up) and a soft
    vignette, in linear light. The render is kept as raw/<name>.jpg."""
    from PIL import Image
    raw = f.parent / "raw" / f.name
    raw.parent.mkdir(exist_ok=True)
    if not raw.exists() or raw.stat().st_mtime < f.stat().st_mtime:
        raw.write_bytes(f.read_bytes())
    x = (np.asarray(Image.open(raw).convert("RGB"), float) / 255.0) ** 2.2
    x *= [1.0 + 0.35 * warm, 1.0 - 0.1 * warm, 1.0 - 0.9 * warm]
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    x *= (1.0 - vignette * np.clip(r, 0, 1) ** 2.4)[..., None]
    Image.fromarray((np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)).save(f, quality=95)


# the shots kept from the previews (2026-11-25, checked by eye) -> the photo's name. Dropped: a drone over long
# shadows (busy), the basin into the sun (murky), a long lens from the southern hills (blocked), the plaza at dusk
# (the plaza is too small to step back from the 120 m tower), a lantern at dusk (cramped), the other roof terraces,
# cacti and streets (weaker repeats)
ART_PICKS = {"golden_drone": "golden_drone", "tele_from_massif": "long_lens_from_the_massif", "rooftop_0": "rooftop",
             "desert_cactus_1": "cactus_and_town", "s_a03": "street_side_light", "s_a12": "pink_house_street",
             "s_a13": "town_from_the_plaza", "s_a09": "street_to_the_desert", "s_a02": "dusty_street_backlit",
             "s_a08": "dust_and_tower", "s_a11": "landmarks_in_the_glow", "tele_sun_tower": "sun_behind_the_cross",
             "dusk_a03": "street_at_dusk", "tele_dusk": "long_lens_at_dusk", "blue_hour_drone": "blue_hour_drone"}


def plan_art(d):
    """The art photos (photo_tour.py), one evening: golden hour on the town, the sun behind the tallest tower, dusk
    with the windows lit."""
    import alleys
    from kaist_village import terrain_z
    st = site()
    S = Sight(d, st)
    z_at = terrain_z(town_terrain(), st.L)
    gt = json.loads((d / "giant_towers.json").read_text())
    T1 = np.asarray(gt["towers"][0]["xy"])                           # the main plaza's 120 m tower
    H1 = gt["towers"][0]["height_m"]
    z1 = float(S.z(T1)[0])
    dt = json.loads((OUT / "district.json").read_text())[0]["center"]
    zd = float(S.z(dt)[0])
    P = []

    def add(name, loc, look, lens, light, **kw):
        P.append({"name": name, "location": [float(v) for v in loc], "look_at": [float(v) for v in look],
                  "lens": float(lens), **light, **kw})

    # aerial: the downtown from the south-south-west in the golden hour, the tower on the left third
    add("golden_drone", [760.0, 470.0, zd + 170], [990.0, 820.0, zd + 5], 35, GOLDEN)
    # long lens from Umm al-Biyara, front-lit: the houses of the far side of the town stacked one over the other, a
    # domed tower and the 120 m tower cutting through the frame (the same frame at dusk). Framed on the houses: higher,
    # the long lens fills the frame with the sandstone hills behind the town, whose strata read as stripes up close
    tg = np.array([dt[0] + 20, dt[1] + 10, zd + 40])
    cam = vantage(S, tg, np.array(UMM_AL_BIYARA), 300.0, 9.0)
    add("tele_from_massif", cam, tg, 200, dict(GOLDEN, haze_m=6000))
    add("tele_dusk", cam, tg, 200, DUSK, grade={"warm": 0.0})
    # long lens: the sun going down behind the crown of the 120 m tower (exposed for the sky: a silhouette)
    top = np.array([*T1, z1 + H1 - 4.0])
    cam, az, el = tele_sun(S, top)
    add("tele_sun_tower", cam, top - [0, 0, 14.0], 400,
        dict(GOLDEN, sun_el=el, sun_az=az, sun_disc=True, disc_k=2.0, aerosol=6.0, haze_m=15000, exposure=-6.0),
        grade={"warm": 0.35, "vignette": 0.25})
    # the streets (alleys.search: a street looking at a tower, houses close on both sides), lit by where they look;
    # the colourful side-lit street again at dusk
    cf = OUT / "art_street_candidates.json"
    if not cf.exists():
        cf.write_text(json.dumps(alleys.search(d, z_at, n_best=40), indent=1))
    for c in json.loads(cf.read_text()):
        kind, light = sun_for(bearing(np.asarray(c["look_at"][:2]) - c["location"][:2]))
        frame = dict(location=c["location"], look_at=c["look_at"], lens=c["lens"], top_pad_deg=2.5)
        P.append(level_godot(dict(name=f"s_{c['name'][:3]}", **frame, **light)))
        P.append(level_godot(dict(name=f"dusk_{c['name'][:3]}", **frame, **DUSK, grade={"warm": 0.0})))
    # rooftops: from a house roof west-south-west of the main tower over the roofs to the tower, front-lit
    n_roof = 0
    for r in (150.0, 190.0, 120.0, 230.0):
        for b in (235.0, 215.0, 255.0, 200.0):
            p = T1 + r * np.array([np.cos(np.radians(b)), np.sin(np.radians(b))])
            j = S.tree.query(shapely.Point(p), predicate="within")
            if not len(j) or S.top[j[0]] - S.z(p)[0] < 5.0:
                continue
            cam = np.array([*p, S.top[j[0]] + 1.7])
            if S.clear(cam, [*T1, z1 + 0.5 * H1], skip_end=12.0) and n_roof < 3:
                add(f"rooftop_{n_roof}", [*p, 1.7], [T1[0], T1[1], z1 + 0.45 * H1], 35, GOLDEN, on_surface=True)
                n_roof += 1
    # the desert edge: a columnar cactus in the sun close in front, the town behind it (three sides of the town)
    C = S.inst["cactus"]
    mask = np.load(OUT / "town_mask.npz")["mask"]
    rr, cc = st.rc(C[:, :2])
    dc = np.hypot(*(C[:, :2] - T1).T)
    ok = (~mask[rr, cc]) & (dc > 350) & (dc < 900) & (C[:, 3] > 0.9)
    used = []
    for k in np.flatnonzero(ok)[np.argsort(dc[ok])]:
        u = (T1 - C[k, :2]) / dc[k]
        if any(abs((bearing(-u) - b + 180) % 360 - 180) < 40 for b in used):
            continue
        if S.horizon(C[k, :3] + [0, 0, 2.0], GOLDEN["sun_az"]) > GOLDEN["sun_el"] - 1.0:
            continue
        xy = C[k, :2] - 5.5 * u + 1.6 * np.array([-u[1], u[0]])
        cam = np.array([*xy, S.z(xy)[0] + 1.5])
        if (S.clear(cam, [*C[k, :2], C[k, 2] + 2.0]) and S.clear(cam, [*T1, z1 + 0.6 * H1], skip_end=12.0)
                and len(used) < 3):
            add(f"desert_cactus_{len(used)}", cam, [T1[0], T1[1], z1 + 30.0], 40, GOLDEN,
                dof={"focus": [float(C[k, 0]), float(C[k, 1]), float(C[k, 2] + 2.2)], "fstop": 4.0})
            used.append(bearing(-u))
    # dusk from above, looking east (away from the afterglow): the lights come on under the blue sky
    add("blue_hour_drone", [640.0, 1060.0, zd + 150], [930.0, 760.0, zd + 20], 30, DUSK, grade={"warm": 0.0})
    P = [dict(p, name=ART_PICKS[p["name"]]) for p in P if p["name"] in ART_PICKS]
    P.sort(key=lambda p: -p["sun_el"])                               # one evening: the sun sinks photo by photo
    for k, p in enumerate(P):
        p["name"] = f"{k + 1:02d}_{p['name']}"
    return P


def stage_photos(engine="CYCLES", res=2400, samples=160, only=None, godot_only=False, kind="photos", out=None):
    from kaist_village import BLENDER
    from photo_tour import GODOT, sheet
    st = site()
    d = city_dir()
    out = out or kind
    photos = plan_photos(d) if kind == "photos" else plan_art(d)
    pj = d / f"{out}.json"
    pj.write_text(json.dumps(photos, indent=1))
    rc = 0 if godot_only else subprocess.run([BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"),
                         "--", "--tile", str(d), "--photos", str(pj), "--res", str(res), "--samples", str(samples), "--out", out,
                         "--engine", engine, *(["--only", only] if only else [])],
                        stdout=open(d / f"{out}.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    if rc:
        return rc
    if kind == "art" and not godot_only:
        for p in photos:
            f = d / out / f"{p['name']}.jpg"
            if f.exists() and (not only or p["name"] in only.split(",")):
                grade(f, **p.get("grade", {}))
    if not godot_only:
        sheet(d / out, [p["name"] for p in photos])
    if out != kind or not GODOT.exists():                          # previews: no Godot pass
        return rc
    used = {p["name"]: p["location"] for p in photos}                # the cameras as Blender placed them (if rendered)
    used.update(json.loads((d / out / "photo_timing.json").read_text())["cameras"])
    gd = lambda v: [v[0], v[2], -v[1]]
    vj = json.loads((d / "viewer.json").read_text())
    vj["shots"] = {p["name"]: {"pos": gd(used[p["name"]]), "look": gd(p.get("godot", p)["look_at"]),
                               "fov": float(np.degrees(2 * np.arctan(18.0 / p.get("godot", p)["lens"] * 9 / 16)))}
                   for p in photos if p["name"] in used}
    vp = d / out / "godot_viewer.json"
    vp.write_text(json.dumps(vj))
    rc = subprocess.run([str(GODOT), "--path", str(ROOT / "godot" / "viewer"), "--resolution", "1920x1080", "--",
                         "--glb", str(d / "scene.glb"), "--viewer", str(vp), "--light", "sunset",
                         "--shot", str(d / out / "godot"), "--extent", str(st.L), "--frames", "20"],
                        stdout=open(d / f"{out}_godot.log", "w"), stderr=subprocess.STDOUT, timeout=1200).returncode
    print("godot", rc, flush=True)
    return rc


def stage_qa(n_street=12, n_drone=6, glb=True):
    """Review in Godot, as the user explores it (machu_picchu.stage_qa): the GLB, then seeded random cameras at street
    level and as drones over the town, the downtown and the town edge -> <city>/qa/godot_*.png and qa/sheet.jpg."""
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
    cx, cy = json.loads((OUT / "district.json").read_text())[0]["center"]
    zc = z_at(cx, cy)
    shots["downtown"] = {"pos": gd([cx - 260, cy - 200, zc + 90]), "look": gd([cx, cy, zc + 40]), "fov": 50.0}
    ex, ey = cx - 380, cy - 40                                     # over the western slope (it rises: ground + 30 m)
    ze = float(st.domain()["elev"][tuple(i[0] for i in st.rc([[ex, ey]]))])
    shots["edge_west"] = {"pos": gd([ex, ey, max(ze, zc) + 30]), "look": gd([cx, cy, zc]), "fov": 55.0}
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
    ap.add_argument("stage", choices=["plan", "skyline", "build", "backdrop", "render", "viewer", "photos", "art", "qa"])
    ap.add_argument("--variant", default=None, choices=["skyline"], help="the town with the skyline cluster and park")
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--only", default=None)
    ap.add_argument("--godot-only", action="store_true")
    ap.add_argument("--out", default=None, help="photos / art: output folder (another than the stage's: a preview, no Godot)")
    a = ap.parse_args()
    global TAG, VARIANT
    if a.variant:
        TAG, VARIANT = f"{TAG}_{a.variant}", a.variant
    if a.stage == "skyline":
        return stage_skyline()
    if a.stage == "plan":
        stage_plan()
        return 0
    if a.stage in ("photos", "art"):
        return stage_photos(a.engine, a.res, a.samples, a.only, a.godot_only, a.stage, a.out)
    if a.stage == "qa":
        return stage_qa()
    return {"build": stage_build, "backdrop": stage_backdrop, "render": stage_render, "viewer": stage_viewer}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
