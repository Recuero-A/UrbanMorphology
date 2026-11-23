"""Latin Caribbean theme on KAIST: a colonial village in the jungle (user request 2026-11-22).

Streets: the Milestone 17 generation "M* + organic [improved]" (F1 junction reuse 15 m + F4 junction-degree prior,
seed 1; outputs/m16/kaist/m17_cache) on the KAIST campus domain, kept only inside a VILLAGE MASK; the real KAIST main
roads (OSM trunk / primary / secondary / tertiary, the anchors of every campus generation, bridges included) are kept
everywhere. Outside the village: no buildings or yards, a closed rainforest canopy (--jungle dense). The Gapcheon and
its tributary stay as water (shown at the DSM water surface).

Village mask (4 m domain grid): a noisy ellipse over the flat campus basin (centre 700, 720; semi-axes 640 x 470 m,
35 deg), slope < 11 deg, no water; opened / closed, parts < 2.5 ha dropped, holes filled; plus the flat land within 70 m
of a main road and 220 m of the village, so the village reaches its roads. Cropping: a generated street is kept when
both ends are in the mask; then isolated pieces shorter than 120 m and dead-end stubs shorter than 10 m are removed.

    .venv/bin/python themes/latin_caribbean/scripts/kaist_village.py [--dense] [--method NA-adaptive --version original]
                                                                     [--giant 165,125,95 [--giant-lottery 60,85]] [--lanes 60]
    --dense: colonial row houses on every village block (latin.infill), three plazas with a church at their head
    --method / --version: another street generator of the KAIST campus runs (M17 village comparison), seed 1
    --giant: giant Cartagena-type towers (m, tallest first) on the plaza churches (church.giant_tower), wider church
             sites, and three more views (tower: from the main plaza; alley: down a street at the tallest tower;
             skyline); --giant-lottery 'lo,hi': giant towers on the other churches too
    ->  outputs/latin_caribbean/kaist_village/ (mask, graph, map; another method in a sub-folder) and
        outputs/m16/cities/kaist_village[_dense|_giant]_<method>_s1_latin/
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

import numpy as np
import shapely
from scipy import ndimage

OUT = ROOT / "outputs" / "latin_caribbean" / "kaist_village"
LABEL = "M* + organic [improved]"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def village_mask(big, anchor, L):
    from urbanmorph.terrain.synthesis import gradient_noise
    z, w = big["elev"].astype(float), big["water"]
    n = z.shape[0]
    xs = -256 + (np.arange(n) + 0.5) * 4
    ys = L + 256 - (np.arange(n) + 0.5) * 4
    X, Y = np.meshgrid(xs, ys)
    a, cx, cy = np.radians(35), 700.0, 720.0
    u = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    v = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
    r = np.hypot(u / 640, v / 470) + 0.18 * gradient_noise(X / 260, Y / 260, 5) + 0.08 * gradient_noise(X / 90, Y / 90, 6)
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 2), 4.0)
    flat = np.degrees(np.arctan(np.hypot(gx, gy))) < 11
    m = (r < 1) & (w == 0) & flat
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=3), iterations=4)
    lab, k = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    m = ndimage.binary_fill_holes(np.isin(lab, 1 + np.flatnonzero(sizes * 16 > 25000))) & (w == 0)
    lines = shapely.union_all([shapely.LineString(anchor.nodes[e]) for e in anchor.edges])
    d_road = shapely.distance(lines, shapely.points(X.ravel(), Y.ravel())).reshape(X.shape)
    d_village = ndimage.distance_transform_edt(~m) * 4.0
    return m | ((d_road < 70) & (d_village < 220) & flat & (w == 0))


def crop(g, anchor, mask, L):
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    n = mask.shape[0]
    rc = lambda P: (np.clip(((L + 256 - P[:, 1]) / 4).astype(int), 0, n - 1), np.clip(((P[:, 0] + 256) / 4).astype(int), 0, n - 1))
    inside = mask[rc(g.nodes)]
    lines = shapely.union_all([shapely.LineString(anchor.nodes[e]) for e in anchor.edges])
    mid = g.nodes[g.edges].mean(1)
    is_anchor = shapely.distance(lines, shapely.points(mid[:, 0], mid[:, 1])) < 3.0
    keep = is_anchor | inside[g.edges].all(1)
    L_e = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1)
    for _ in range(3):                                      # dead-end stubs < 10 m (not on main roads)
        deg = np.bincount(g.edges[keep].ravel(), minlength=len(g.nodes))
        stub = keep & ~is_anchor & (L_e < 10) & ((deg[g.edges[:, 0]] == 1) | (deg[g.edges[:, 1]] == 1))
        keep &= ~stub
    idx = np.flatnonzero(keep)                              # isolated pieces < 120 m without a main road
    lab = np.arange(len(g.nodes))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for e in idx:
        a, b = find(g.edges[e, 0]), find(g.edges[e, 1])
        if a != b:
            lab[a] = b
    comp = np.array([find(g.edges[e, 0]) for e in idx])
    tot, has_anchor = {}, {}
    for e, c in zip(idx, comp):
        tot[c] = tot.get(c, 0.0) + L_e[e]
        has_anchor[c] = has_anchor.get(c, False) or bool(is_anchor[e])
    keep[idx] = [tot[c] >= 120 or has_anchor[c] for c in comp]
    E = g.edges[keep]
    used, inv = np.unique(E, return_inverse=True)
    N = g.nodes[used]
    return RoadGraph(N, inv.reshape(-1, 2), is_border(N, size=L), edge_class=g.classes()[keep], edge_struct=g.structs()[keep]), \
        {"edges_in": int(len(g.edges)), "edges_kept": int(keep.sum()), "anchor_edges": int(is_anchor.sum()),
         "street_km_in": float(L_e.sum() / 1000), "street_km_kept": float(L_e[keep].sum() / 1000)}


def figure(big, mask, g, anchor, L, path, title="M* + organic (F1 + F4)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    z, w = big["elev"].astype(float), big["water"]
    rgb = LightSource(315, 40).shade(z, cmap=plt.cm.gist_earth, vert_exag=2, blend_mode="soft")
    rgb[~mask, :3] = rgb[~mask, :3] * 0.4 + np.array([0.12, 0.38, 0.12]) * 0.6
    rgb[w > 0, :3] = [0.2, 0.45, 0.9]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, extent=(-256, L + 256, -256, L + 256))
    for e in g.edges:
        ax.plot(*g.nodes[e].T, color="k", lw=0.5)
    for e in anchor.edges:
        ax.plot(*anchor.nodes[e].T, color="#d62728", lw=1.6)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title(f"KAIST village: {title} inside the village (black), real main roads (red), jungle (green)")
    fig.savefig(path, dpi=90, bbox_inches="tight")


def tower_views(d, big, sites, mask, L, W=512.0):
    """Cameras of the giant towers (Blender cameras.json; Godot viewer.json "shots" and the walk spawn): ``tower`` from
    the far side of the main plaza looking up at the tallest tower, ``alley`` down a straight street at it (clear line
    of sight past every house), ``skyline`` over the jungle with the low sun behind the camera."""
    from m12_big_landscape import MARGIN, window_dict
    from urbanmorph.city3d.terrain3d import Surfaces
    from urbanmorph.graphs.terrain_generate import terrain_of
    from urbanmorph.terrain.heightfield import Terrain
    val = json.loads((d / "validation.json").read_text())
    towers = []
    for key, t in val["tiles"].items():
        r, c = int(key[1]), int(key[3])
        towers += [(np.asarray(x["xy"]) + [c * W, r * W], x["height_m"], x["side_m"]) for x in t["summary"].get("giant_towers", [])]
    if not towers:
        return {}
    towers.sort(key=lambda x: -x[1])
    T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -MARGIN, L + MARGIN, 4.0)
    T.water_kind = big["water"].astype(np.int8)
    T.water_surface = np.where(big["water"] > 0, big["elev"] + 0.3, np.nan).astype(np.float32)
    surf = {}

    def z_at(x, y):
        tc, tr = int(np.clip(x // W, 0, L // W - 1)), int(np.clip(y // W, 0, L // W - 1))
        if (tr, tc) not in surf:
            surf[(tr, tc)] = Surfaces(terrain_of(window_dict(T, tc * W, tr * W)))
        return float(surf[(tr, tc)].road(np.array([x - tc * W]), np.array([y - tr * W]))[0])

    def lens_for(vfov_deg):
        h = 2 * np.arctan(np.tan(np.radians(vfov_deg) / 2) * 16 / 9)
        return float(18.0 / np.tan(h / 2))

    def view(cam, look, lens):
        return {"type": "PERSP", "lens": lens, "location": [float(v) for v in cam], "look_at": [float(v) for v in look]}

    def aim(cam, xy, H, zb, lo=-3.0, margin=3.0, min_lens=12.0):
        """Lens and look-at that frame the tower from its foot to its top (pitched camera)."""
        D = float(np.hypot(*(xy - cam[:2])))
        top = np.degrees(np.arctan2(zb + H - cam[2], D)) + margin
        bot = max(np.degrees(np.arctan2(zb - cam[2], D)) - 1.0, lo)
        lens = max(min_lens, lens_for(top - bot))
        pitch = np.radians((top + bot) / 2)
        u = (xy - cam[:2]) / D
        return lens, [cam[0] + u[0] * 100 * np.cos(pitch), cam[1] + u[1] * 100 * np.cos(pitch), cam[2] + 100 * np.sin(pitch)]

    tx, H, side = towers[0]
    zb = z_at(*tx)
    views = {}
    # plaza: the free point of the main plaza farthest from the tower
    site = min(sites, key=lambda s_: np.hypot(*(np.asarray(s_["church"]["center"]) - tx)))
    ch = site["church"]
    ax_, ay_ = np.asarray(ch["axis"]), np.array([-ch["axis"][1], ch["axis"][0]])
    ctr = np.asarray(ch["center"])
    rect = shapely.Polygon([ctr + sx * ax_ * ch["L"] / 2 + sy * ay_ * ch["W"] / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    free = shapely.Polygon(site["plaza"][0]).buffer(-4.0).difference(rect.buffer(4.0)).difference(
        shapely.box(*(tx - side / 2 - 4), *(tx + side / 2 + 4)))
    if not free.is_empty:
        x0, y0, x1, y1 = free.bounds
        P = np.array([(x, y) for x in np.arange(x0, x1, 2.0) for y in np.arange(y0, y1, 2.0)])
        P = P[shapely.contains_xy(free, P[:, 0], P[:, 1])]
        if len(P):
            p = P[np.argmax(np.hypot(*(P - tx).T))]
            cam = np.r_[p, z_at(*p) + 1.7]
            lens, look = aim(cam, tx, H, zb)
            views["tower"] = view(cam, look, lens)
    # alley: a point of a street within 15 deg of the tower direction, 120-500 m away, no house in the line of sight
    houses = []
    for f in sorted(d.glob("city_r*c*.json")):
        r, c = int(f.stem[6]), int(f.stem[8])
        for b in json.loads(f.read_text())["buildings"]:
            if "church_site" not in b:
                houses.append(shapely.Polygon(np.asarray(b["footprint"][0])[:, :2] + [c * W, r * W]))
    tree = shapely.STRtree(houses)
    st = np.load(d / "streets.npy")
    n = mask.shape[0]
    inside = lambda x, y: bool(mask[min(max(int((L + 256 - y) / 4), 0), n - 1), min(max(int((x + 256) / 4), 0), n - 1)])
    best = None
    for x0, y0, x1, y1, cls, stc in st:
        a_, b_ = np.array([x0, y0]), np.array([x1, y1])
        Ls = float(np.hypot(*(b_ - a_)))
        if Ls < 15:
            continue
        e = (b_ - a_) / Ls
        for t_ in np.arange(0.0, Ls, 5.0):
            p = a_ + e * t_
            D = float(np.hypot(*(tx - p)))
            if not (120 <= D <= 500 and inside(*p)):
                continue
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(e, (tx - p) / D)), 0, 1)))
            if ang > 15:
                continue
            sight = shapely.LineString([p + (tx - p) / D * 3, tx - (tx - p) / D * (side / 2 + 2)]).buffer(1.0)
            zc = z_at(*p) + 1.7
            need = np.arctan2(zb + 0.35 * H - zc, D)
            hit = tree.query(sight, predicate="intersects")
            if any(np.arctan2(z_at(*np.asarray(houses[k].centroid.coords)[0]) + 14.0 - zc,
                              max(houses[k].distance(shapely.Point(p)), 0.5)) > need for k in hit):
                continue
            sun = np.dot((p - tx) / D, [np.cos(np.radians(140)), np.sin(np.radians(140))])   # > 0: sun behind camera
            score = ang - 4 * sun + abs(D - 260) / 80
            if best is None or score < best[0]:
                best = (score, p)
    if best is not None:
        cam = np.r_[best[1], z_at(*best[1]) + 1.7]
        lens, look = aim(cam, tx, H, zb, lo=-6.0, margin=4.0, min_lens=18.0)
        views["alley"] = view(cam, look, lens)
    # skyline: from the sun side, over the jungle
    cxy = np.mean([t[0] for t in towers], 0)
    zc = z_at(*cxy)
    u = np.array([np.cos(np.radians(125)), np.sin(np.radians(125))])
    camxy = np.clip(cxy + u * 560, 40, L - 40)
    cam = np.r_[camxy, max(z_at(*camxy), zc) + 250]                # pitched down ~22 deg: the map edge stays out
    views["skyline"] = view(cam, [cxy[0], cxy[1], zc + 0.15 * H], 32.0)
    cams = json.loads((d / "cameras.json").read_text())
    cams.update(views)
    (d / "cameras.json").write_text(json.dumps(cams, indent=1, default=float))
    gd = lambda v: [v[0], v[2], -v[1]]
    vj = json.loads((d / "viewer.json").read_text())
    vj["shots"] = {k: {"pos": gd(v["location"]), "look": gd(v["look_at"]),
                       "fov": float(np.degrees(2 * np.arctan(np.tan(np.arctan(18.0 / v["lens"])) * 9 / 16)))}
                   for k, v in views.items()}
    if "tower" in views:                                          # walk from the main plaza, facing the tower
        loc = views["tower"]["location"]
        vj["spawn"] = gd([loc[0], loc[1], loc[2] + 0.3])
        vj["look_at"] = gd([tx[0], tx[1], loc[2] + 0.3])
    (d / "viewer.json").write_text(json.dumps(vj, default=float))
    return {"towers": [{"xy": t[0].tolist(), "height_m": t[1], "side_m": t[2]} for t in towers], "views": sorted(views)}


def main() -> int:
    from m16_kaist_all import key_of
    from m16_kaist_campus import L, anchors, domain
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--method", default="M* + organic")
    ap.add_argument("--version", default="F1 + F4", choices=["F1 + F4", "original"])
    ap.add_argument("--giant", default=None, help="'150,115,90': giant towers (m) on the plaza churches, tallest first")
    ap.add_argument("--giant-lottery", default=None, help="'lo,hi': giant towers on the other churches too")
    ap.add_argument("--lanes", type=float, default=None, help="cut blocks bigger than ~1.6 x M^2 by lanes every M m")
    ap.add_argument("--skip-build", action="store_true", help="the 3D city exists: only views and Blender renders")
    a = ap.parse_args()
    dense = a.dense or bool(a.giant)
    label = a.method + (" [improved]" if a.version == "F1 + F4" else "")
    run = OUT if label == LABEL else OUT / key_of(label).replace("[", "").replace("]", "")
    run.mkdir(parents=True, exist_ok=True)
    big, (x0, y0), crs = domain()
    anchor = anchors(x0, y0, crs)
    mp = OUT / "village_mask.npz"
    if mp.exists():
        mask = np.load(mp)["mask"]
    else:
        mask = village_mask(big, anchor, L)
        np.savez_compressed(mp, mask=mask)
    cache = "m17_cache" if a.version == "F1 + F4" else "all_cache"
    org = pickle.loads((ROOT / "outputs" / "m16" / "kaist" / cache / f"{key_of(a.method)}_s1.pkl").read_bytes())["graph"]
    g, info = crop(org, anchor, mask, L)
    info["village_km2"] = float(mask[64:448, 64:448].sum() * 16 / 1e6)
    info["method"], info["version"] = a.method, a.version
    (run / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    gp = run / "graphs.pkl"
    gp.write_bytes(pickle.dumps({"graphs": {(label, 1): g}, "real": None}))
    figure(big, mask, g, anchor, L, run / "village_map.png", f"{a.method} ({a.version})")
    extra, sites = [], []
    if dense:                                                # plazas with their churches, then colonial row houses
        from m6_build_objects import load_objects
        from urbanmorph.city3d.latin.infill import plaza_sites
        sites = plaza_sites(g, mask, L, load_objects()["class_widths"], max_w=34.0 if a.giant else 22.0)
        if a.giant:
            for s_, h in zip(sites, [float(x) for x in a.giant.split(",")]):
                s_["church"]["tower_h"] = h
        pj = run / ("plaza_sites_giant.json" if a.giant else "plaza_sites.json")
        pj.write_text(json.dumps(sites, indent=1))
        print("plaza sites", len(sites), flush=True)
        extra = ["--infill", "colonial", "--plazas", str(pj)]
        if a.giant_lottery:
            extra += ["--giant-lottery", a.giant_lottery]
        if a.lanes:
            extra += ["--lanes", str(a.lanes)]
    tag = "kaist_village" + ("_giant" if a.giant else "_dense" if dense else "")
    py = str(ROOT / ".venv" / "bin" / "python")
    d = ROOT / "outputs" / "m16" / "cities" / f"{tag}_{key_of(label).replace('[', '').replace(']', '')}_s1_latin"
    if not (a.skip_build and (d / "meshes.npz").exists()):
        r = subprocess.run([py, str(ROOT / "scripts" / "m16_city3d_big.py"), "--method", label, "--seed", "1", "--graphs", str(gp),
                            "--tag", tag, "--theme", "latin_caribbean",
                            "--village-npz", str(mp), "--jungle", "dense", *extra], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            return r.returncode
        d = Path(json.loads(r.stdout.strip().splitlines()[-1])["dir"])
    print("3D", d, flush=True)
    views = "aerial,oblique,street"
    if a.giant:
        tv = tower_views(d, big, sites, mask, L)
        (d / "giant_towers.json").write_text(json.dumps(tv, indent=1, default=float))
        print("towers", json.dumps(tv, default=float), flush=True)
        views += "".join("," + v for v in tv.get("views", []))
    rc = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
                         "--", "--tile", str(d), "--res", "2000", "--light", "sunset", "--views", views],
                        stdout=open(d / "blender.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
