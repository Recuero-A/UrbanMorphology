"""Latin Caribbean theme: the six landmark towers (city3d.latin.landmarks; user references in architectural features/thin
and architectural features/cohesive) on a showcase plaza.

Sheets laid out like the references (front elevation, three-quarter view from the south-west, crown and tier
close-ups): a studio scene (each tower on its own plaza pad, daylight, transparent sky composited on paper). Then the
six on one plaza: sunset photos (blender/photo_tour.py: sun and sky aligned, haze, clouds), its GLB + viewer.json to
walk among them in Godot, and Godot shots. ``--colours`` repaints every tower with house-palette roles (e.g. wall=wall:rosa).

    .venv/bin/python themes/latin_caribbean/scripts/preview_landmarks.py [--height 150] [--res 1000] [--out landmarks]
    ->  themes/latin_caribbean/previews/<out>/{sheets/*.jpg, photos/*.jpg, studio/, godot/, scene.glb, viewer.json,
        report.json}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont

from urbanmorph.city3d.latin.build import group_of
from urbanmorph.city3d.latin.landmarks import landmark
from urbanmorph.city3d.latin.plants import prototypes
from urbanmorph.city3d.materials import MaterialTable
from urbanmorph.city3d.mesh import surface
from urbanmorph.city3d.scene import Group, Z_BASE, Z_ROAD, save_scene

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
GODOT = str(Path.home() / "Downloads/Godot.app/Contents/MacOS/Godot")
OUT = ROOT / "themes" / "latin_caribbean" / "previews"
ORDER = [("coral", "thin"), ("coral", "cohesive"), ("eagle", "thin"), ("eagle", "cohesive"), ("golden", "thin"), ("golden", "cohesive")]
TITLES = {"coral": "Coral Lantern", "eagle": "Eagle Belfry", "golden": "Golden Clock Tower"}
SP, X0, Y, EXT, DEPTH = 100.0, 100.0, 350.0, 700.0, 700.0
FONT = "/System/Library/Fonts/Supplemental/Georgia.ttf"


def scene(H: float, seed: int, colours: dict, studio: bool = False):
    mt = MaterialTable()
    M = lambda k: mt.get("latin:" + k)
    names = ("walls", "roofs", "trim", "glass", "doors", "carpentry", "metal", "roads", "green", "ground", "sidewalks", "markings")
    groups = {k: Group() for k in names}
    towers = []
    for i, (d, f) in enumerate(ORDER):
        c = (X0 + SP * i, Y)
        S, info = landmark(d, f, c, (1.0, 0.0), H, np.random.default_rng([seed, i]), colours=colours)
        for key, lst in S.parts.items():
            for V, F, UV in lst:
                groups[group_of(key)].add(V, F, M(key), i, UV)
        towers.append({"name": f"{d}_{f}", "title": f"{TITLES[d]} ({f})", "xy": list(c), **info})
    rng = np.random.default_rng(seed)
    if studio:                                  # a plaza pad per tower, nothing else (the sky is not filmed)
        pads = [shapely.box(t["xy"][0] - t["half_extent_m"] - 9, Y - t["half_extent_m"] - 9, t["xy"][0] + t["half_extent_m"] + 9,
                            Y + t["half_extent_m"] + 9) for t in towers]
        V, F = surface(shapely.union_all(pads), Z_ROAD)
        groups["green"].add(V, F, M("plaza"))
        inst = {"palm_royal": [[t["xy"][0] + sx * (t["half_extent_m"] + 5), Y + sy * (t["half_extent_m"] + 5), 0.0,
                                float(rng.uniform(0.85, 1.0)), float(rng.uniform(0, 6.28))]
                               for t in towers for sx in (-1, 1) for sy in (-1, 1)]}
    else:
        pads = [shapely.box(t["xy"][0] - 32, Y - 32, t["xy"][0] + 32, Y + 32) for t in towers]
        walk = shapely.box(X0 - 45, Y - 52, X0 + SP * 5 + 45, Y - 40)
        plaza = shapely.union_all(pads + [walk] + [shapely.box(t["xy"][0] - 4, Y - 40, t["xy"][0] + 4, Y - 32) for t in towers])
        V, F = surface(plaza, Z_ROAD)
        groups["green"].add(V, F, M("plaza"))
        V, F = surface(shapely.box(0, 0, EXT, DEPTH).difference(plaza), Z_ROAD)
        groups["green"].add(V, F, M("grass_public"))
        groups["ground"].add(np.array([[0, 0, Z_BASE], [EXT, 0, Z_BASE], [EXT, DEPTH, Z_BASE], [0, DEPTH, Z_BASE]], float),
                             [[0, 1, 2], [0, 2, 3]], M("ground"))
        inst = {"palm_royal": [[x, Y - 56, 0.0, float(rng.uniform(0.9, 1.15)), float(rng.uniform(0, 6.28))]
                               for x in np.arange(X0 - 40, X0 + SP * 5 + 41, 20.0)],
                "palm_coconut": [[t["xy"][0] + sx * 27, Y + 26, 0.0, float(rng.uniform(0.85, 1.1)), float(rng.uniform(0, 6.28))]
                                 for t in towers for sx in (-1, 1)],
                "lamp": [[x + 10, Y - 38, 0.0, 0.0] for x in np.arange(X0 - 40, X0 + SP * 5 + 41, 20.0)]}
    out = {k: g.arrays() for k, g in groups.items()}
    protos = prototypes()
    for name, T in inst.items():
        PV, PF, part, PUV, mats = protos[name]
        mi = np.array([M(m) for m in mats], np.int16)
        out[f"proto_{name}"] = {"V": PV.astype(np.float32), "F": PF.astype(np.int32), "mat": mi[part],
                                "obj": np.full(len(PF), -1, np.int32), "UV": PUV.astype(np.float32)}
        out[f"inst_{name}"] = {"T": np.asarray(T, np.float32)}
    return out, mt.params, towers


def cameras(towers, res):
    """Studio views per tower: front elevation, three-quarter view, crown, tier (cohesive)."""
    RW, RH = res, int(res * 2.3)
    cams = {"extent_m": EXT, "surround_z": -0.05, "transparent": True, "surround": False}
    views = []
    for t in towers:
        cx, cy = t["xy"]
        top = t["height_m"]
        D = 650.0
        cams[f"{t['name']}__front"] = {"type": "PERSP", "lens": 36 * D / (1.08 * top), "location": [cx, cy - D, top / 2],
                                       "look_at": [cx, cy, top / 2], "res": [RW, RH]}
        D = 520.0
        a = np.radians(30.0)
        cams[f"{t['name']}__quarter"] = {"type": "PERSP", "lens": 36 * D / (1.16 * top),
                                         "location": [cx - D * np.sin(a), cy - D * np.cos(a), 0.3 * top],
                                         "look_at": [cx, cy, 0.49 * top], "res": [RW, RH]}
        zc = t["crown_z"]
        hc = top - zc + 3.0
        D = 1.12 * hc * 70 / 36
        a = np.radians(22.0)
        cams[f"{t['name']}__crown"] = {"type": "PERSP", "lens": 70, "location": [cx - D * np.sin(a), cy - D * np.cos(a), zc + 0.42 * hc],
                                       "look_at": [cx, cy, zc + 0.5 * hc - 1.0], "res": [int(res * 1.3), int(res * 1.3)]}
        views += [f"{t['name']}__front", f"{t['name']}__quarter", f"{t['name']}__crown"]
        if "tier" in t:
            z0, z1 = t["tier"]
            ht = z1 - z0 + 3.0
            D = 1.1 * ht * 100 / 36
            cams[f"{t['name']}__tier"] = {"type": "PERSP", "lens": 100, "location": [cx - 0.12 * D, cy - D, (z0 + z1) / 2],
                                          "look_at": [cx, cy, (z0 + z1) / 2], "res": [int(res * 0.95), int(res * 1.3)]}
            views.append(f"{t['name']}__tier")
    return cams, views


MID = X0 + SP * 2.5
PHOTOS = [                                     # sunset photos of the plaza (blender/photo_tour.py; sun_az: light from)
    {"name": "01_lineup", "location": [MID - 70, Y - 330, 22.0], "look_at": [MID + 10, Y, 74.0], "lens": 19, "sun_el": 6,
     "sun_az": 200, "haze_m": 6000, "clouds": 0.5, "glow": 0.3},
    {"name": "02_avenue", "location": [X0 - 48, Y - 46, 1.7], "look_at": [X0 + 230, Y - 8, 78.0], "lens": 18, "sun_el": 7,
     "sun_az": 195, "haze_m": 5000, "clouds": 0.55, "glow": 0.3},
    {"name": "03_backlit", "location": [MID + 330, Y + 280, 35.0], "look_at": [MID - 80, Y, 85.0], "lens": 28, "sun_el": 3,
     "sun_az": 190, "sun_disc": True, "haze_m": 4000, "clouds": 0.45, "glow": 0.5},
    {"name": "04_drone", "location": [MID + 120, Y - 460, 200.0], "look_at": [MID - 10, Y, 70.0], "lens": 24, "sun_el": 5,
     "sun_az": 200, "haze_m": 7000, "clouds": 0.5, "glow": 0.4},
]


def plaza_cameras():
    return {"extent_m": EXT, "surround_z": -0.05,
            "aerial": {"type": "PERSP", "lens": 30, "location": [MID + 330, Y - 420, 260.0], "look_at": [MID, Y, 70.0]}}


def blender(d, views, light, res, sun_az, log, extra=()):
    cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
           "--", "--tile", str(d), "--res", str(res), "--views", ",".join(views), "--light", light, "--sun-az", str(sun_az),
           "--samples", "64", *extra]
    return subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode


def font(size):
    return ImageFont.truetype(FONT, size)


PAPER = (246, 240, 228)


def on_paper(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, PAPER + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def sheet(d, t):
    """Reference-style sheet: front elevation | three-quarter view | crown (and tier) close-ups, on paper."""
    R = d / "studio" / "renders"
    f, q, c = (on_paper(R / f"{t['name']}__{v}.png") for v in ("front", "quarter", "crown"))
    H = 1500
    f, q = (im.resize((int(im.width * H / im.height), H), Image.LANCZOS) for im in (f, q))
    colw = int(H * 0.5)
    right = [c.resize((colw, int(c.height * colw / c.width)), Image.LANCZOS)]
    if (R / f"{t['name']}__tier.png").exists():
        tr = on_paper(R / f"{t['name']}__tier.png")
        right.append(tr.resize((colw, int(tr.height * colw / tr.width)), Image.LANCZOS))
    rh = sum(im.height for im in right) + 40 * (len(right) - 1)
    if rh > H:
        k = H / rh
        right = [im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS) for im in right]
        colw = right[0].width
    pad, top = 40, 120
    W = pad * 4 + f.width + q.width + colw
    out = Image.new("RGB", (W, top + H + 100), PAPER)
    dr = ImageDraw.Draw(out)
    title = t["title"].upper()
    dr.text((W / 2, 60), title, fill=(40, 30, 25), font=font(54), anchor="mm")
    x = pad
    for im, lab in ((f, "FRONT ELEVATION"), (q, "THREE-QUARTER VIEW (FROM SOUTHWEST)")):
        out.paste(im, (x, top))
        dr.text((x + im.width / 2, top + H + 45), lab, fill=(60, 50, 40), font=font(26), anchor="mm")
        x += im.width + pad
    y = top
    labs = ["CROWN", "TYPICAL MID-SHAFT TIER"]
    for im, lab in zip(right, labs):
        out.paste(im, (x, y))
        dr.rectangle([x, y, x + im.width - 1, y + im.height - 1], outline=(200, 188, 168), width=2)
        dr.text((x + im.width / 2, y + im.height + 22), lab, fill=(60, 50, 40), font=font(24), anchor="mm")
        y += im.height + 60
    dr.text((W / 2, 102), f"{t['height_m']:.0f} m  ·  {t['floors']} storeys  ·  shaft {t['shaft_m']:.0f} m wide",
            fill=(120, 104, 88), font=font(22), anchor="mm")
    out.save(d / "sheets" / f"{t['name']}.jpg", quality=92)


def godot(d, towers):
    """viewer.json (spawn on the plaza walk, named shots) and Godot screenshots."""
    gd = lambda p: [float(p[0]), float(p[2]), float(-p[1])]
    vf = lambda lens: float(np.degrees(2 * np.arctan(36 * 9 / 16 / 2 / lens)))
    shots = {p["name"]: {"pos": gd(p["location"]), "look": gd(p["look_at"]), "fov": vf(p["lens"])} for p in PHOTOS}
    for t in towers:
        cx, cy = t["xy"]
        shots[t["name"]] = {"pos": gd([cx - 110, cy - 230, 1.7]), "look": gd([cx, cy, 0.5 * t["height_m"]]), "fov": vf(20)}
    vj = {"spawn": gd([X0 + 42, Y - 47, 1.7]), "look_at": gd([X0 + 140, Y - 5, 1.7]), "extent": EXT, "shots": shots}
    (d / "viewer.json").write_text(json.dumps(vj, indent=1))
    cmd = [GODOT, "--path", str(ROOT / "godot" / "viewer"), "--resolution", "1920x1080", "--", "--glb", str(d / "scene.glb"),
           "--viewer", str(d / "viewer.json"), "--light", "sunset", "--shot", str(d / "godot" / "godot"), "--extent", str(EXT),
           "--frames", "20"]
    (d / "godot").mkdir(exist_ok=True)
    return subprocess.run(cmd, stdout=open(d / "godot.log", "w"), stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--res", type=int, default=1000)
    ap.add_argument("--out", default="landmarks")
    ap.add_argument("--colours", default="", help="role=key,... applied to every tower (e.g. wall=wall:rosa)")
    ap.add_argument("--skip", default="", help="comma list of stages to skip: day,sunset,godot")
    a = ap.parse_args()
    d = OUT / a.out
    st = d / "studio"
    for x in (d / "sheets", st):
        x.mkdir(parents=True, exist_ok=True)
    colours = dict(kv.split("=", 1) for kv in a.colours.split(",") if kv)
    out, mats, towers = scene(a.height, a.seed, colours, studio=True)
    save_scene(st / "meshes.npz", out, mats)
    cams, views = cameras(towers, a.res)
    (st / "cameras.json").write_text(json.dumps(cams, indent=1))
    out, mats, towers = scene(a.height, a.seed, colours)
    save_scene(d / "meshes.npz", out, mats)
    (d / "cameras.json").write_text(json.dumps(plaza_cameras(), indent=1))
    rep = {"height_target_m": a.height, "towers": towers}
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    skip = set(a.skip.split(","))
    rc = {}
    if "day" not in skip:                      # studio daylight, sun from the south-west (lights the front and west faces)
        rc["day"] = blender(st, views, "studio", a.res, 235.0, d / "blender_studio.log", ["--no-export"])
        for t in towers:
            sheet(d, t)
    if "sunset" not in skip:                   # the plaza: scene.blend + scene.glb (m11), then the sunset photos
        rc["scene"] = blender(d, ["aerial"], "sunset", 1200, 200.0, d / "blender_scene.log")
        (d / "photos.json").write_text(json.dumps(PHOTOS, indent=1))
        cmd = [BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"),
               "--", "--tile", str(d), "--photos", str(d / "photos.json"), "--res", "2400", "--samples", "128"]
        rc["photos"] = subprocess.run(cmd, stdout=open(d / "photos.log", "w"), stderr=subprocess.STDOUT).returncode
    if "godot" not in skip and (d / "scene.glb").exists():
        rc["godot"] = godot(d, towers)
    print(json.dumps({"dir": str(d), "rc": rc, "towers": [{k: (round(t[k], 1) if isinstance(t[k], float) else t[k])
                                                           for k in ("name", "height_m", "floors", "triangles")} for t in towers]}))
    return max(rc.values(), default=0)


if __name__ == "__main__":
    sys.exit(main())
