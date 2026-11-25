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


def stage_photos(engine="CYCLES", res=2400, samples=160, only=None, godot_only=False):
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
    ap.add_argument("stage", choices=["plan", "build", "backdrop", "render", "photos", "qa"])
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=160)
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
