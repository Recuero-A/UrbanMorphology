"""Latin Caribbean theme: the six landmark towers (city3d.latin.landmarks; user references in architectural features/thin
and architectural features/cohesive) and the two church styles from the user's photos (city3d.latin.church_styles;
themes/latin_caribbean/references/church_*.png) on a showcase plaza.

* Studio sheets laid out like the references: each building on its own pad, daylight preset 'studio' (warm ambient,
  Standard view), transparent sky composited on paper. Towers: front elevation, three-quarter view from the
  south-west, crown and (cohesive) tier close-ups; churches: front, three-quarter view, tower top.
* compare/*.jpg: every model beside its reference, to review fidelity.
* The eight on one plaza: sunset photos (blender/photo_tour.py: sun and sky aligned, haze, clouds), a GLB +
  viewer.json to walk among them in Godot, Godot shots.

``--colours`` repaints every tower with house-palette roles (e.g. wall=wall:rosa).

    .venv/bin/python themes/latin_caribbean/scripts/preview_landmarks.py [--height 150] [--res 1000] [--out landmarks]
    ->  themes/latin_caribbean/previews/<out>/{sheets/, compare/, photos/, studio/, godot/, scene.glb, viewer.json,
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
from urbanmorph.city3d.latin.church_styles import church_style
from urbanmorph.city3d.latin.landmarks import landmark
from urbanmorph.city3d.latin.plants import prototypes
from urbanmorph.city3d.materials import MaterialTable
from urbanmorph.city3d.mesh import surface
from urbanmorph.city3d.scene import Group, Z_BASE, Z_ROAD, save_scene

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
GODOT = str(Path.home() / "Downloads/Godot.app/Contents/MacOS/Godot")
OUT = ROOT / "themes" / "latin_caribbean" / "previews"
A = ROOT / "architectural features"
TOWERS = [("coral", "thin"), ("coral", "cohesive"), ("eagle", "thin"), ("eagle", "cohesive"), ("golden", "thin"), ("golden", "cohesive")]
CHURCHES = [("gothic", 26.0, 42.0), ("bahia", 27.0, 42.0)]           # style, facade width, nave length (m)
TITLES = {"coral": "Coral Lantern", "eagle": "Eagle Belfry", "golden": "Golden Clock Tower", "gothic": "Blue Gothic Church",
          "bahia": "Bahia Baroque Church"}
REFS = {("coral", "thin"): (f"{A}/thin/Coral Lantern_ Slim Shaft, Grand Crown-1.png", (835, 60, 1190, 860)),
        ("eagle", "thin"): (f"{A}/thin/Slender Eagle Belfry with Enlarged Crown-3.png", (750, 20, 1185, 900)),
        ("golden", "thin"): (f"{A}/thin/Slender Golden Clock Tower Elevations-2.png", (820, 60, 1190, 790)),
        ("coral", "cohesive"): (f"{A}/cohesive/Coral Lantern_ Slender Ornate Tropical Tower-2.png", (830, 80, 1170, 1180)),
        ("eagle", "cohesive"): (f"{A}/cohesive/Eagle Belfry_ Slender Tropical Tower-1.png", (755, 145, 1150, 1100)),
        ("golden", "cohesive"): (f"{A}/cohesive/Golden tropical clocktower skyscraper-3.png", (820, 60, 1190, 1250)),
        "gothic": (str(ROOT / "themes/latin_caribbean/references/church_gothic_blue.png"), None),
        "bahia": (str(ROOT / "themes/latin_caribbean/references/church_bahia_rosario.png"), None)}
SP, X0, Y, EXT, DEPTH = 100.0, 100.0, 350.0, 920.0, 700.0
CX = (720.0, 820.0)                                                  # church facade centres (x), facades at Y - 30
FONT = "/System/Library/Fonts/Supplemental/Georgia.ttf"
PAPER = (246, 240, 228)


def scene(H: float, seed: int, colours: dict, studio: bool = False):
    mt = MaterialTable()
    M = lambda k: mt.get("latin:" + k)
    names = ("walls", "roofs", "trim", "glass", "doors", "carpentry", "metal", "roads", "green", "ground", "sidewalks", "markings")
    groups = {k: Group() for k in names}
    items = []
    for i, (d, f) in enumerate(TOWERS):
        c = (X0 + SP * i, Y)
        S, info = landmark(d, f, c, (1.0, 0.0), H, np.random.default_rng([seed, i]), colours=colours)
        for key, lst in S.parts.items():
            for V, F, UV in lst:
                groups[group_of(key)].add(V, F, M(key), i, UV)
        he = info["half_extent_m"]
        items.append({"name": f"{d}_{f}", "kind": "tower", "ref": (d, f), "title": f"{TITLES[d]} ({f})", "xy": list(c),
                      "box": [c[0] - he, c[1] - he, c[0] + he, c[1] + he], **info})
    for j, (st, W, L) in enumerate(CHURCHES):
        c = (CX[j], Y - 30.0)
        S, info = church_style(st, c, (0.0, 1.0), L, W, np.random.default_rng([seed, 50 + j]))
        for key, lst in S.parts.items():
            for V, F, UV in lst:
                groups[group_of(key)].add(V, F, M(key), 100 + j, UV)
        items.append({"name": f"church_{st}", "kind": "church", "ref": st, "title": TITLES[st], "xy": list(c),
                      "box": [c[0] - W / 2 - 1.5, c[1] - 2.0, c[0] + W / 2 + 1.5, c[1] + L + 1.0], **info})
    rng = np.random.default_rng(seed)
    if studio:                                  # a plaza pad per building, nothing else (the sky is not filmed)
        pads = [shapely.box(t["box"][0] - 9, t["box"][1] - 9, t["box"][2] + 9, t["box"][3] + 9) for t in items]
        V, F = surface(shapely.union_all(pads), Z_ROAD)
        groups["green"].add(V, F, M("plaza"))
        inst = {"palm_royal": [[x, y, 0.0, float(rng.uniform(0.85, 1.0)), float(rng.uniform(0, 6.28))] for t in items
                               for x in (t["box"][0] - 5, t["box"][2] + 5) for y in (t["box"][1] - 5, t["box"][3] + 5)]}
    else:
        pads = [shapely.box(t["box"][0] - 12, Y - 32, t["box"][2] + 12, max(Y + 32, t["box"][3] + 8)) for t in items]
        walk = shapely.box(X0 - 45, Y - 52, CX[-1] + 45, Y - 40)
        plaza = shapely.union_all(pads + [walk] + [shapely.box(t["xy"][0] - 4, Y - 40, t["xy"][0] + 4, Y - 30) for t in items])
        V, F = surface(plaza, Z_ROAD)
        groups["green"].add(V, F, M("plaza"))
        V, F = surface(shapely.box(0, 0, EXT, DEPTH).difference(plaza), Z_ROAD)
        groups["green"].add(V, F, M("grass_public"))
        groups["ground"].add(np.array([[0, 0, Z_BASE], [EXT, 0, Z_BASE], [EXT, DEPTH, Z_BASE], [0, DEPTH, Z_BASE]], float),
                             [[0, 1, 2], [0, 2, 3]], M("ground"))
        xs = np.arange(X0 - 40, CX[-1] + 41, 20.0)
        inst = {"palm_royal": [[x, Y - 56, 0.0, float(rng.uniform(0.9, 1.15)), float(rng.uniform(0, 6.28))] for x in xs],
                "palm_coconut": [[t["xy"][0] + sx * (t["box"][2] - t["box"][0] + 10) / 2, Y + 26, 0.0, float(rng.uniform(0.85, 1.1)),
                                  float(rng.uniform(0, 6.28))] for t in items if t["kind"] == "tower" for sx in (-1, 1)],
                "lamp": [[x + 10, Y - 38, 0.0, 0.0] for x in xs]}
    out = {k: g.arrays() for k, g in groups.items()}
    protos = prototypes()
    for name, T in inst.items():
        PV, PF, part, PUV, mats = protos[name]
        mi = np.array([M(m) for m in mats], np.int16)
        out[f"proto_{name}"] = {"V": PV.astype(np.float32), "F": PF.astype(np.int32), "mat": mi[part],
                                "obj": np.full(len(PF), -1, np.int32), "UV": PUV.astype(np.float32)}
        out[f"inst_{name}"] = {"T": np.asarray(T, np.float32)}
    return out, mt.params, items


def cam(loc, look, lens, res):
    return {"type": "PERSP", "lens": float(lens), "location": [float(v) for v in loc], "look_at": [float(v) for v in look],
            "res": [int(res[0]), int(res[1])]}


def cameras(items, res):
    """Studio views: towers - front elevation, three-quarter view, crown, tier (cohesive); churches - front,
    three-quarter view, tower top."""
    cams = {"extent_m": EXT, "surround_z": -0.05, "transparent": True, "surround": False}
    views = []
    for t in items:
        cx, cy = t["xy"]
        top = t["height_m"]
        nm = t["name"]
        if t["kind"] == "tower":
            D = 650.0
            cams[f"{nm}__front"] = cam((cx, cy - D, top / 2), (cx, cy, top / 2), 36 * D / (1.08 * top), (res, int(res * 2.3)))
            D, a = 520.0, np.radians(30.0)
            cams[f"{nm}__quarter"] = cam((cx - D * np.sin(a), cy - D * np.cos(a), 0.3 * top), (cx, cy, 0.49 * top),
                                         36 * D / (1.16 * top), (res, int(res * 2.3)))
            zc = t["crown_z"]
            hc = top - zc + 3.0
            D, a = 1.1 * hc * 80 / 36, np.radians(20.0)
            cams[f"{nm}__crown"] = cam((cx - D * np.sin(a), cy - D * np.cos(a), zc + 0.45 * hc), (cx, cy, zc + 0.5 * hc - 1.0), 80,
                                       (int(res * 1.1), int(res * 1.4)))
            views += [f"{nm}__front", f"{nm}__quarter", f"{nm}__crown"]
            if "tier" in t:
                z0, z1 = t["tier"][0] - 2, t["tier"][1] + 4
                ht = z1 - z0
                D = 1.05 * ht * 90 / 36
                cams[f"{nm}__tier"] = cam((cx - 0.15 * D, cy - D, (z0 + z1) / 2), (cx, cy, (z0 + z1) / 2), 90,
                                          (int(res * 1.0), int(res * 1.3)))
                views.append(f"{nm}__tier")
        else:
            W = t["W"]
            span = 1.18 * max(top, W + 8)
            D = 260.0
            cams[f"{nm}__front"] = cam((cx, cy - D, 0.47 * top), (cx, cy, 0.47 * top), 36 * D / span, (res, res))
            D, a = 220.0, np.radians(32.0)
            cams[f"{nm}__quarter"] = cam((cx - D * np.sin(a), cy - D * np.cos(a), 0.35 * top), (cx + 2, cy + 10, 0.42 * top),
                                         36 * D / (1.15 * span), (res, res))
            D, a = 60.0, np.radians(25.0)
            tx = cx - W / 2 + 0.11 * W
            cams[f"{nm}__crown"] = cam((tx - D * np.sin(a), cy - D * np.cos(a), 0.66 * top), (tx, cy, 0.72 * top), 110,
                                       (int(res * 1.0), int(res * 1.3)))
            views += [f"{nm}__front", f"{nm}__quarter", f"{nm}__crown"]
    return cams, views


MID = X0 + SP * 2.5
PHOTOS = [                                     # sunset photos of the plaza (blender/photo_tour.py; sun_az: light from)
    {"name": "01_lineup", "location": [MID - 10, Y - 360, 24.0], "look_at": [MID - 10, Y, 78.0], "lens": 19, "sun_el": 6,
     "sun_az": 200, "haze_m": 6000, "clouds": 0.5, "glow": 0.3},
    {"name": "02_avenue", "location": [X0 - 48, Y - 46, 1.7], "look_at": [X0 + 230, Y - 8, 78.0], "lens": 18, "sun_el": 7,
     "sun_az": 195, "haze_m": 5000, "clouds": 0.55, "glow": 0.3},
    {"name": "03_backlit", "location": [MID + 330, Y + 280, 35.0], "look_at": [MID - 80, Y, 85.0], "lens": 28, "sun_el": 3,
     "sun_az": 190, "sun_disc": True, "haze_m": 4000, "clouds": 0.45, "glow": 0.5},
    {"name": "04_drone", "location": [MID + 160, Y - 500, 210.0], "look_at": [MID + 60, Y, 60.0], "lens": 22, "sun_el": 5,
     "sun_az": 200, "haze_m": 7000, "clouds": 0.5, "glow": 0.4},
    {"name": "05_churches", "location": [CX[0] - 25, Y - 120, 6.0], "look_at": [(CX[0] + CX[1]) / 2, Y, 18.0], "lens": 24,
     "sun_el": 6, "sun_az": 205, "haze_m": 5000, "clouds": 0.5, "glow": 0.4},
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


def on_paper(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, PAPER + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def sheet(d, t):
    """Reference-style sheet: front | three-quarter view | close-ups (crown, tier / tower top), on paper."""
    R = d / "studio" / "renders"
    f, q, c = (on_paper(R / f"{t['name']}__{v}.png") for v in ("front", "quarter", "crown"))
    H = 1500 if t["kind"] == "tower" else 1000
    f, q = (im.resize((int(im.width * H / im.height), H), Image.LANCZOS) for im in (f, q))
    colw = int(H * 0.5) if t["kind"] == "tower" else int(H * 0.62)
    right = [c.resize((colw, int(c.height * colw / c.width)), Image.LANCZOS)]
    if (R / f"{t['name']}__tier.png").exists():
        tr = on_paper(R / f"{t['name']}__tier.png")
        right.append(tr.resize((colw, int(tr.height * colw / tr.width)), Image.LANCZOS))
    rh = sum(im.height for im in right) + 60 * (len(right) - 1)
    if rh > H:
        k = H / rh
        right = [im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS) for im in right]
        colw = right[0].width
    pad, top = 40, 130
    W = pad * 4 + f.width + q.width + colw
    out = Image.new("RGB", (W, top + H + 100), PAPER)
    dr = ImageDraw.Draw(out)
    dr.text((W / 2, 60), t["title"].upper(), fill=(40, 30, 25), font=font(54), anchor="mm")
    sub = (f"{t['height_m']:.0f} m  ·  {t['floors']} storeys  ·  shaft {t['shaft_m']:.0f} m wide" if t["kind"] == "tower" else
           f"{t['height_m']:.0f} m to the cross  ·  facade {t['W']:.0f} m  ·  nave {t['L']:.0f} m")
    dr.text((W / 2, 108), sub, fill=(120, 104, 88), font=font(22), anchor="mm")
    x = pad
    for im, lab in ((f, "FRONT ELEVATION"), (q, "THREE-QUARTER VIEW (FROM SOUTHWEST)")):
        out.paste(im, (x, top))
        dr.text((x + im.width / 2, top + H + 45), lab, fill=(60, 50, 40), font=font(26), anchor="mm")
        x += im.width + pad
    y = top
    labs = ["CROWN", "TYPICAL MID-SHAFT TIER"] if t["kind"] == "tower" else ["TOWER TOP"]
    for im, lab in zip(right, labs):
        out.paste(im, (x, y))
        dr.rectangle([x, y, x + im.width - 1, y + im.height - 1], outline=(200, 188, 168), width=2)
        dr.text((x + im.width / 2, y + im.height + 22), lab, fill=(60, 50, 40), font=font(24), anchor="mm")
        y += im.height + 60
    out.save(d / "sheets" / f"{t['name']}.jpg", quality=92)


def compare(d, t):
    """The model beside its reference: towers - reference front | model front | reference detail | model detail;
    churches - photo | model front | model three-quarter."""
    R = d / "studio" / "renders"
    ref, box = REFS[t["ref"]]
    r = Image.open(ref).convert("RGB")
    if t["kind"] == "tower":
        detail = "tier" if (R / f"{t['name']}__tier.png").exists() else "crown"
        ims = [r.crop((20, 40, 485, 1295)), on_paper(R / f"{t['name']}__front.png"), r.crop(box), on_paper(R / f"{t['name']}__{detail}.png")]
        labs = ("REFERENCE", "MODEL", "REFERENCE", "MODEL")
    else:
        ims = [r, on_paper(R / f"{t['name']}__front.png"), on_paper(R / f"{t['name']}__quarter.png")]
        labs = ("PHOTO", "MODEL", "MODEL")
    h = 1100
    ims = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    out = Image.new("RGB", (sum(i.width for i in ims) + 10 * len(ims), h + 50), "white")
    dr = ImageDraw.Draw(out)
    x = 0
    for im, lab in zip(ims, labs):
        out.paste(im, (x, 50))
        dr.text((x + 10, 12), lab, fill=(40, 30, 25), font=font(24))
        x += im.width + 10
    out.save(d / "compare" / f"{t['name']}.jpg", quality=90)


def godot(d, items):
    """viewer.json (spawn on the plaza walk, named shots) and Godot screenshots."""
    gd = lambda p: [float(p[0]), float(p[2]), float(-p[1])]
    vf = lambda lens: float(np.degrees(2 * np.arctan(36 * 9 / 16 / 2 / lens)))
    shots = {p["name"]: {"pos": gd(p["location"]), "look": gd(p["look_at"]), "fov": vf(p["lens"])} for p in PHOTOS}
    for t in items:
        cx, cy = t["xy"]
        back = 230 if t["kind"] == "tower" else 75
        shots[t["name"]] = {"pos": gd([cx - 0.48 * back, cy - back, 1.7]), "look": gd([cx, cy, 0.5 * t["height_m"]]), "fov": vf(20)}
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
    for x in (d / "sheets", d / "compare", st):
        x.mkdir(parents=True, exist_ok=True)
    colours = dict(kv.split("=", 1) for kv in a.colours.split(",") if kv)
    out, mats, items = scene(a.height, a.seed, colours, studio=True)
    save_scene(st / "meshes.npz", out, mats)
    cams, views = cameras(items, a.res)
    (st / "cameras.json").write_text(json.dumps(cams, indent=1))
    out, mats, items = scene(a.height, a.seed, colours)
    save_scene(d / "meshes.npz", out, mats)
    (d / "cameras.json").write_text(json.dumps(plaza_cameras(), indent=1))
    rep = {"height_target_m": a.height, "items": items}
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))
    skip = set(a.skip.split(","))
    rc = {}
    if "day" not in skip:                      # studio daylight, sun from the south-west (lights the front and west faces)
        rc["day"] = blender(st, views, "studio", a.res, 225.0, d / "blender_studio.log", ["--no-export"])
        for t in items:
            sheet(d, t)
            compare(d, t)
    if "sunset" not in skip:                   # the plaza: scene.blend + scene.glb (m11), then the sunset photos
        rc["scene"] = blender(d, ["aerial"], "sunset", 1200, 200.0, d / "blender_scene.log")
        (d / "photos.json").write_text(json.dumps(PHOTOS, indent=1))
        cmd = [BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"),
               "--", "--tile", str(d), "--photos", str(d / "photos.json"), "--res", "2400", "--samples", "128"]
        rc["photos"] = subprocess.run(cmd, stdout=open(d / "photos.log", "w"), stderr=subprocess.STDOUT).returncode
    if "godot" not in skip and (d / "scene.glb").exists():
        rc["godot"] = godot(d, items)
    print(json.dumps({"dir": str(d), "rc": rc, "items": [{"name": t["name"], "height_m": round(t["height_m"], 1),
                                                          "triangles": t["triangles"]} for t in items]}))
    return max(rc.values(), default=0)


if __name__ == "__main__":
    sys.exit(main())
