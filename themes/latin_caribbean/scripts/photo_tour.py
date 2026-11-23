"""Sunset photo tour of the KAIST Latin village (user request 2026-11-23: "take beautiful photos around the city at sunset").

A photographer's evening walk: hand-placed viewpoints (streets, plazas, palm gardens, a roof terrace, the hills, a drone
and long lenses), each with its own sun height (the sun sinks from 9 to 1 deg; it stays at azimuth 140 deg, the
north-west, as in every sunset render of the theme), lens, haze and lit windows. Cameras stand on the terrain (the 3D
city's road surface, kaist_village.terrain_z) or on a skyscraper's crown terrace (ray cast in Blender). Writes
<city>/photos.json, renders blender/photo_tour.py into <city>/photos/<name>.jpg and the same views in Godot (sunset
light, <city>/photos/godot_<name>.png).

    .venv/bin/python themes/latin_caribbean/scripts/photo_tour.py [--city <dir>] [--only a,b] [--res 2400] [--samples 128]
                                                                  [--no-godot]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from kaist_village import BLENDER, terrain_z

CITY = ROOT / "outputs" / "m16" / "cities" / "kaist_village_downtown_NA-adaptive_s1_latin"
GODOT = Path.home() / "Downloads" / "Godot.app" / "Contents" / "MacOS" / "Godot"
W = 512.0
SUN = np.array([np.cos(np.radians(140)), np.sin(np.radians(140))])     # towards the sun (Blender m11 sunset preset)


def skyscrapers(d):
    """Every Latin skyscraper of the city: centre, axis, W, D, H (world metres)."""
    out = []
    for f in sorted(d.glob("city_r*c*.json")):
        r, c = int(f.stem[6]), int(f.stem[8])
        for b in json.loads(f.read_text())["buildings"]:
            if b.get("skyscraper"):
                sk = dict(b["skyscraper"])
                sk["center"] = np.asarray(sk["center"]) + [c * W, r * W]
                out.append(sk)
    return out


def terrace_point(sk, target):
    """A point on the crown terrace of skyscraper ``sk`` (between the drum and the balustrade) on the side facing target."""
    ax = np.asarray(sk["axis"], float)
    ay = np.array([-ax[1], ax[0]])
    si = 1.4 if min(sk["W"], sk["D"]) >= 14 else 0.8
    hw, hd = sk["W"] / 2 - si, sk["D"] / 2 - si
    rd = 0.43 * 2 * min(hw, hd)
    v = np.asarray(target[:2]) - sk["center"]
    lx, ly = v @ ax, v @ ay
    if abs(lx) / hw >= abs(ly) / hd:                      # the face the target is in front of
        p = np.sign(lx) * ax * (rd + 0.25 + hw + 0.3) / 2
    else:
        p = np.sign(ly) * ay * (rd + 0.25 + hd + 0.3) / 2
    return sk["center"] + p


class Ground:
    """Free places to stand: off every building and away from the low crowns (flamboyants, jungle trees, bananas)."""

    def __init__(self, d):
        import shapely
        from scipy.spatial import cKDTree
        with np.load(d / "meshes.npz") as mz:
            obst = np.vstack([np.zeros((0, 2))] + [mz[k][:, :2] for k in mz.files if k.endswith("__T") and any(
                t in k for t in ("flamboyant", "jungle_tree", "canopy", "banana", "fountain"))])
        self.kd = cKDTree(obst)
        foot = []
        for f in sorted(d.glob("city_r*c*.json")):
            r, c = int(f.stem[6]), int(f.stem[8])
            foot += [shapely.Polygon(np.asarray(b["footprint"][0])[:, :2] + [c * W, r * W])
                     for b in json.loads(f.read_text())["buildings"]]
        self.houses = shapely.STRtree(foot)

    def spot(self, xy, target, reach=30.0, clear=7.0, sight=40.0, within=None):
        """The free point nearest ``xy`` (within ``reach`` m, inside the polygon ``within``) whose first ``sight`` m
        towards ``target`` pass no crown and no building."""
        import shapely
        g = np.arange(-reach, reach + 0.1, 2.0)
        P = np.asarray(xy, float) + np.array([(x, y) for x in g for y in g])
        P = P[np.argsort(np.hypot(*(P - xy).T))]
        for p in P:
            if within is not None and not within.contains(shapely.Point(p)):
                continue
            if self.kd.query(p)[0] < clear or len(self.houses.query(shapely.Point(p).buffer(1.5), predicate="intersects")):
                continue
            v = np.asarray(target[:2]) - p
            u = v / np.hypot(*v)
            sight_ = min(sight, 0.6 * np.hypot(*v))                  # never into the target itself
            ray = p + np.outer(np.arange(2.0, sight_, 2.5), u)
            if (self.kd.query(ray)[0] < 5.0).any():
                continue
            if len(self.houses.query(shapely.LineString([p + 2 * u, p + sight_ * u]), predicate="intersects")):
                continue
            return p
        raise RuntimeError(f"no free spot near {xy}")


def balcony_point(xy, side, axis, view):
    """A point on the balcony of the first belfry of a giant tower (church.giant_tower: belfry 0.86 x side on the
    side-wide clock stage) on the face looking along ``view``."""
    a = np.asarray(axis, float)
    b = np.array([-a[1], a[0]])
    n = max((a, -a, b, -b), key=lambda e: float(e @ view))
    return np.asarray(xy) + n * (0.43 * side + 0.5 * side + 0.6) / 2


def shore_spot(big, L, target, kind=2):
    """The pond's edge with the longest stretch of water towards ``target`` (camera on the water's edge, 4 m grid)."""
    wm = big["water"] == kind
    n = wm.shape[0]
    cell = lambda x, y: (int((L + 256 - y) / 4), int((x + 256) / 4))
    ys, xs = np.nonzero(wm)
    pts = np.c_[xs * 4 - 256 + 2, L + 256 - ys * 4 - 2]
    best, out = -1.0, None
    for p in pts:
        u = (np.asarray(target[:2]) - p) / np.hypot(*(np.asarray(target[:2]) - p))
        r, c = cell(*(p - 6 * u))
        if 0 <= r < n and 0 <= c < n and wm[r, c]:          # not at the edge: water behind the camera
            continue
        k = 0
        while True:
            r, c = cell(*(p + (k + 1) * 4.0 * u))
            if not (0 <= r < n and 0 <= c < n and wm[r, c]):
                break
            k += 1
        if k * 4.0 > best:
            best, out = k * 4.0, p
    r, c = cell(*out)
    return out, float(big["elev"][r, c]) + 0.3, best


def plan(d, z_at, big=None, L=1536.0):
    """The tour: one dict per photo for blender/photo_tour.py (location / look_at in world metres)."""
    import shapely
    cams = json.loads((d / "cameras.json").read_text())
    gt = json.loads((d / "giant_towers.json").read_text())
    sites = json.loads((ROOT / "outputs" / "latin_caribbean" / "kaist_village" / "NA-adaptive" / "plaza_sites_giant.json").read_text())
    cat, cat_side = np.asarray(gt["towers"][0]["xy"]), gt["towers"][0]["side_m"]     # the 165 m cathedral tower
    zc = z_at(*cat)
    sks = skyscrapers(d)
    near = lambda xy: min(sks, key=lambda s: np.hypot(*(s["center"] - xy)))
    ground = Ground(d)

    def at(xy, h):
        return [float(xy[0]), float(xy[1]), z_at(*xy) + h]

    P = []

    def add(name, loc, look, lens, **kw):
        P.append({"name": name, "location": [float(v) for v in loc], "look_at": [float(v) for v in look], "lens": float(lens), **kw})

    # from the western hill (83 m), 400 m out, the low sun behind the left shoulder: the district glows
    hill = np.array([300.0, 790.0])
    add("hill_golden_hour", at(hill, 45), [660, 930, zc + 55], 30, sun_el=7, haze_m=9000, clouds=0.55)
    add("hill_long_lens", at(hill, 45), [655, 945, zc + 110], 70, sun_el=6, haze_m=9000, clouds=0.55)
    # into the sun from the south-east, the towers against the glow
    add("backlit_skyline", at([1250, 660], 60), [686, 936, zc + 60], 45, sun_el=3, haze_m=16000, clouds=0.6, sun_disc=True)
    # the street views of kaist_village.tower_views, re-lit
    for k, el in (("tower", 7), ("alley", 6)):
        v = cams[k]
        add(f"street_{k}", v["location"], v["look_at"], v["lens"], sun_el=el, haze_m=12000, glow=0.8, clouds=0.6)
    # the pond south of the district: the sunset in the water, the towers against the glow
    if big is not None:
        tgt = [cat[0], cat[1], zc + 30]
        sp, zw, _ = shore_spot(big, L, tgt)
        add("pond_sunset", [sp[0], sp[1], zw + 1.4], tgt, 22, sun_el=3, haze_m=12000, glow=0.8, clouds=0.6,
            sun_disc=True)
    # the cathedral's belfry balcony (about 100 m up), over the skyscraper crowns to the south
    ch = sites[0]["church"]
    u = np.array([np.cos(np.radians(-75.0)), np.sin(np.radians(-75.0))])
    bp = balcony_point(cat, cat_side, ch["axis"], u)
    add("bell_tower_view", [bp[0], bp[1], 1.75], [bp[0] + 100 * u[0], bp[1] + 100 * u[1], zc + 100 - 14], 24,
        on_surface=True, sun_el=5, haze_m=12000, glow=1.0, clouds=0.6)
    # a crown terrace west of the cathedral: a long lens on the tallest skyscraper's crown and the cathedral's bells
    sk = near(np.array([575.0, 982.0]))
    top = near(np.array([641.0, 967.0]))
    tp = terrace_point(sk, top["center"])
    add("crown", [tp[0], tp[1], 1.6], [*top["center"], z_at(*top["center"]) + top["H"] - 12], 70, on_surface=True,
        sun_el=5, haze_m=12000, glow=1.0, clouds=0.6)
    # the sun going down beside the cathedral tower: 850 m south-east, a 70 mm lens, 2 deg between them
    ub = np.array([np.cos(np.radians(142.0)), np.sin(np.radians(142.0))])
    cb = cat - 850 * ub
    add("sun_and_tower", at(cb, 40), [cat[0], cat[1], zc + 70], 70, sun_el=2.5, haze_m=9000, clouds=0.6, sun_disc=True)
    # a drone north-west of the district: the whole town in the last sun, the hills and the clouds
    dc = np.array([686.0, 936.0])
    add("drone", at(dc + SUN * 520, 150), [dc[0], dc[1], zc + 40], 28, sun_el=4, haze_m=8000, clouds=0.6)
    # over the roofs of the old town to the second giant tower (125 m)
    t2 = np.asarray(gt["towers"][1]["xy"])
    u2 = np.array([np.cos(np.radians(200.0)), np.sin(np.radians(200.0))])
    add("old_town_roofs", at(t2 + 240 * u2, 25), at(t2, 45), 28, sun_el=6, haze_m=10000, glow=0.8, clouds=0.6)
    # last light: the sun on the horizon, the tops of the towers still red, windows and lanterns lit
    add("last_light", at(hill, 45), [655, 945, zc + 70], 50, sun_el=0.3, haze_m=9000, glow=0.7, clouds=0.65,
        exposure=0.4)
    P.sort(key=lambda p: -p["sun_el"])                               # one evening: the sun sinks photo by photo
    for i, p in enumerate(P):
        p["name"] = f"{i + 1:02d}_{p['name']}"
    return P


def sheet(out, names, cols=3, w=800):
    """contact_sheet.jpg: every photo, in order."""
    from PIL import Image
    ims = [Image.open(out / f"{n}.jpg").convert("RGB") for n in names if (out / f"{n}.jpg").exists()]
    h = int(w * 9 / 16)
    S = Image.new("RGB", (cols * w, -(-len(ims) // cols) * h), "white")
    for i, im in enumerate(ims):
        S.paste(im.resize((w, h), Image.LANCZOS), ((i % cols) * w, (i // cols) * h))
    S.save(out / "contact_sheet.jpg", quality=88)


def main() -> int:
    from m16_kaist_campus import L, domain
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=str(CITY))
    ap.add_argument("--only", default=None)
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--out", default="photos")
    ap.add_argument("--no-godot", action="store_true")
    a = ap.parse_args()
    d = Path(a.city).resolve()
    big, _, _ = domain()
    photos = plan(d, terrain_z(big, L, W), big, L)
    (d / "photos.json").write_text(json.dumps(photos, indent=1))
    rc = subprocess.run([BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python",
                         str(ROOT / "blender" / "photo_tour.py"), "--", "--tile", str(d), "--res", str(a.res),
                         "--samples", str(a.samples), "--out", a.out, *(["--only", a.only] if a.only else [])],
                        stdout=open(d / f"{a.out}.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    if rc:
        return rc
    sheet(d / a.out, [p["name"] for p in photos])
    if a.no_godot or not GODOT.exists():
        return rc
    # Godot: the same cameras (as Blender placed them) under the viewer's sunset light
    used = json.loads((d / a.out / "photo_timing.json").read_text())["cameras"]
    gd = lambda v: [v[0], v[2], -v[1]]
    vj = json.loads((d / "viewer.json").read_text())
    vj["shots"] = {p["name"]: {"pos": gd(used[p["name"]]), "look": gd(p["look_at"]),
                               "fov": float(np.degrees(2 * np.arctan(18.0 / p["lens"] * 9 / 16)))}
                   for p in photos if p["name"] in used}
    vp = d / a.out / "godot_viewer.json"
    vp.write_text(json.dumps(vj))
    rc = subprocess.run([str(GODOT), "--path", str(ROOT / "godot" / "viewer"), "--resolution", "1920x1080", "--",
                         "--glb", str(d / "scene.glb"), "--viewer", str(vp), "--light", "sunset",
                         "--shot", str(d / a.out / "godot"), "--extent", str(L), "--frames", "20"],
                        stdout=open(d / f"{a.out}_godot.log", "w"), stderr=subprocess.STDOUT, timeout=900).returncode
    for k in ("street", "oblique", "aerial"):                     # the viewer's own three views are not part of the tour
        (d / a.out / f"godot_{k}.png").unlink(missing_ok=True)
    print("godot", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
