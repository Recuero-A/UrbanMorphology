"""Latin Caribbean theme: preliminary design preview on a hand-laid showcase street (no learned streets).

A colonial main street (row houses on both sides, 1-3 storeys) ending at a plaza with the church, a cross street, a
few detached villas with gardens, and jungle on the free land. Built with the same code path the generated maps use
(city3d.scene.build_scene(theme="latin_caribbean")), rendered in Blender at golden hour (and one midday view).

    .venv/bin/python themes/latin_caribbean/scripts/preview_kit.py [--seed 3] [--light sunset] [--res 1600]
    ->  themes/latin_caribbean/previews/kit_s<seed>/renders/*.png
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

from urbanmorph.city3d.format import rings
from urbanmorph.city3d.scene import build_scene, save_scene

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
OUT = ROOT / "themes" / "latin_caribbean" / "previews"


def street(sid, pts, cls, w):
    return {"id": sid, "centerline": [list(map(float, p)) for p in pts], "segment_widths_m": [w] * (len(pts) - 1),
            "width_source": "preview", "class": cls, "segment_classes": [cls] * (len(pts) - 1), "nodes": []}


def showcase(seed: int) -> dict:
    r = np.random.default_rng(seed)
    streets = [street("main", [(120, 256), (330, 256)], "mid", 10.0), street("cross", [(236, 120), (236, 400)], "local", 7.0),
               street("back", [(120, 300), (236, 300)], "local", 7.0), street("plaza_n", [(330, 300), (470, 300)], "local", 7.0),
               street("plaza_s", [(330, 212), (470, 212)], "local", 7.0), street("plaza_w", [(330, 212), (330, 300)], "local", 7.0)]
    road = shapely.union_all([shapely.LineString(s["centerline"]).buffer(s["segment_widths_m"][0] / 2, cap_style="square")
                              for s in streets])
    fps, kinds = [], []
    # row houses on both sides of the main street and the cross street (frontage 7-12 m, depth 12-16 m)
    for side in (1, -1):
        for x0, x1 in ((126, 230), (242, 326)):
            x = x0
            while x < x1 - 6:
                w = min(float(r.uniform(7, 12)), x1 - x)
                d = float(r.uniform(12, 16))
                y0 = 256 + side * 7.6
                fps.append(shapely.box(x, min(y0, y0 + side * d), x + w, max(y0, y0 + side * d)))
                kinds.append(int(r.choice([1, 2, 2, 3])))
                x += w
    for side in (1, -1):
        for y0, y1 in ((126, 240), (272, 296), (306, 396)):
            y = y0
            while y < y1 - 6:
                w = min(float(r.uniform(8, 12)), y1 - y)
                x0 = 236 + side * 6.2
                d = float(r.uniform(11, 15))
                fps.append(shapely.box(min(x0, x0 + side * d), y, max(x0, x0 + side * d), y + w))
                kinds.append(int(r.choice([1, 2, 2, 3])))
                y += w
    # the church parcel east of the plaza, and two blocks of the plaza edge
    fps.append(shapely.box(392, 244, 444, 268))
    kinds.append(2)
    # detached villas with gardens south-west
    villas = [(150, 170), (180, 165), (152, 200), (190, 196), (150, 330), (186, 340)]
    for vx, vy in villas:
        w, d = float(r.uniform(10, 14)), float(r.uniform(8, 11))
        fps.append(shapely.box(vx, vy, vx + w, vy + d))
        kinds.append(int(r.choice([1, 2])))
    fps = [shapely.orient_polygons(p.difference(road)) if p.intersects(road) else shapely.orient_polygons(p) for p in fps]
    tree = shapely.STRtree(fps)
    B = []
    for i, (p, f) in enumerate(zip(fps, kinds)):
        att = [f"B{j}" for j in tree.query(p.buffer(0.3)) if j != i and p.distance(fps[j]) < 0.3]
        B.append({"id": f"B{i}", "block": "b0", "parcels": [], "footprint": rings(p), "floors": f, "floors_source": "preview",
                  "height_m": 3.5 * f, "height_source": "synthetic", "typology": "attached_low", "eave_m": 3.5 * f,
                  "roof": {"type": "flat", "pitch_deg": 0.0, "rise_m": 0.0, "source": "synthetic"},
                  "style": {"attached_to": att, "source": "synthetic", "facade_family": "plaster", "palette": i % 6,
                            "bay_width_m": 3.4, "window_width_ratio": 0.5, "window_height_ratio": 0.55, "balconies": False,
                            "balcony_pattern": "all", "cornice": True, "ground_floor": "residential"}})
    landuse = [{"class": "public_green", "source": "preview", "id": "plaza", "polygon": rings(shapely.box(336, 218, 388, 294))},
               {"class": "public_green", "source": "preview", "id": "park", "polygon": rings(shapely.box(244, 140, 320, 230))}]
    for k, (vx, vy) in enumerate(villas):
        g = shapely.box(vx - 4, vy - 5, vx + 18, vy + 16).difference(shapely.union_all(fps)).difference(road.buffer(3))
        if not g.is_empty and g.geom_type == "Polygon":
            landuse.append({"class": "private_green", "source": "preview", "id": f"garden{k}", "polygon": rings(g)})
    for k, (x0, y0, x1, y1) in enumerate(((126, 278, 230, 294), (126, 218, 230, 234), (242, 278, 326, 294))):
        landuse.append({"class": "private_green", "source": "preview", "id": f"yard{k}",
                        "polygon": rings(shapely.box(x0, y0, x1, y1).difference(shapely.union_all(fps)))})
    from urbanmorph.city3d.format import polygons
    surf = [rings(p) for p in polygons(road)]
    return {"format": "urbanmorph-city", "version": "0.1", "metadata": {"tile_id": f"latin_kit_s{seed}", "seed": seed,
            "storey_height_m": {"value": 3.5, "source": "preview"}}, "nodes": [], "streets": streets, "road_surface": surf,
            "blocks": [], "parcels": [], "buildings": B, "landuse": landuse}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--light", default="sunset")
    ap.add_argument("--res", type=int, default=1600)
    ap.add_argument("--views", default="street,oblique,church,plaza,aerial")
    ap.add_argument("--sun-az", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    d = OUT / f"kit_s{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    city = showcase(a.seed)
    groups, mats, rep = build_scene(city, theme="latin_caribbean")
    save_scene(d / "meshes.npz", groups, mats)
    (d / "report.json").write_text(json.dumps({k: rep[k] for k in ("summary", "churches", "facade")}, indent=1, default=float))
    cams = {"extent_m": 512.0, "surround_z": -0.05,
            "aerial": {"type": "ORTHO", "location": [300, 256, 400], "ortho_scale": 300},
            "oblique": {"type": "PERSP", "lens": 32, "location": [130, 120, 95], "look_at": [300, 262, 8]},
            "street": {"type": "PERSP", "lens": 24, "location": [150, 253.5, 1.7], "look_at": [330, 258, 9]},
            "church": {"type": "PERSP", "lens": 28, "location": [345, 238, 8], "look_at": [400, 257, 16]},
            "plaza": {"type": "PERSP", "lens": 22, "location": [338, 228, 1.8], "look_at": [390, 262, 8]}}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1))
    log = d / f"blender_{a.light}.log"
    cmd = [BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"), "--",
           "--tile", str(d), "--res", str(a.res), "--views", a.views, "--light", a.light]
    cmd += ["--sun-az", a.sun_az] if a.sun_az else []
    if a.light != "sunset" or a.suffix:
        cmd += ["--suffix", a.suffix or f"_{a.light}"]
    rc = subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
    print(json.dumps({"dir": str(d), "blender_rc": rc, **{k: rep["summary"][k] for k in ("n_buildings", "churches", "roofs")},
                      "triangles": sum(rep["summary"]["triangles"].values())}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
