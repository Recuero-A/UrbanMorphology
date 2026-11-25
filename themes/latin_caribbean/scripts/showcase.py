"""Sunset showcase of the theme's cities (user request 2026-11-23: "good photos to showcase the urbanistic landscapes and
the architecture of the cities ... at sunset ... the tropical and epic feeling").

Six hand-placed photos per city, rendered by blender/photo_tour.py (sun and sky aligned, aerial haze, clouds, lit
windows), each with its own sun: in the KAIST landmark city the low sun of the theme (towards the north-west) lights
the towers gold or stands behind them; in the coastal town it sets over the sea (south-west).

    .venv/bin/python themes/latin_caribbean/scripts/showcase.py [--city landmarks|coast|all] [--only a,b] [--res 2400]
    ->  <city>/showcase/<nn>_<name>.jpg, contact_sheet.jpg
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
from photo_tour import SUN, Ground, sheet, shore_spot

CITIES = ROOT / "outputs" / "latin_caribbean" / "cities"
LANDMARKS = CITIES / "kaist" / "kaist_village_landmarks_NA-adaptive_s1_latin"
COAST = CITIES / "coast" / "coast_NA-fixed_improved_s1_latin"
L = 1536.0


def open_spot(g, xy, target, reach=40.0, clear=8.0):
    """A free spot (Ground.spot) at least ``clear`` m from any building: no wall or balcony fills the foreground."""
    import shapely
    g_ = np.arange(-reach, reach + 0.1, 3.0)
    P = np.asarray(xy, float) + np.array([(x, y) for x in g_ for y in g_])
    for p in P[np.argsort(np.hypot(*(P - xy).T))]:
        if len(g.houses.query(shapely.Point(p).buffer(clear), predicate="intersects")):
            continue
        try:
            return g.spot(p, target, reach=0.1)
        except RuntimeError:
            continue
    return g.spot(xy, target, reach=reach)


def unit(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def plan_landmarks(d, z_at, big):
    """The KAIST landmark city: golden towers, backlit skyline, raking drone, the red tower among palms, the river,
    the Bahia church at dusk."""
    gt = json.loads((d / "giant_towers.json").read_text())
    dist = json.loads((ROOT / "outputs" / "latin_caribbean" / "kaist_village" / "NA-adaptive" / "district.json").read_text())
    dc = np.asarray(dist["center"])
    zc = z_at(*dc)
    red = max(gt["landmarks"], key=lambda t: t["height_m"])          # the red Golden Clock Tower, 122 m
    rx, rH = np.asarray(red["xy"]), red["height_m"]
    at = lambda xy, h: [float(xy[0]), float(xy[1]), z_at(*xy) + h]
    P = []
    # the towers front-lit across the jungle, a long lens from the north-west hills
    c = dc + SUN * 620
    P.append(dict(name="golden_towers", location=at(c, 60), look_at=[*dc, zc + 85], lens=78, sun_el=5, haze_m=9000, clouds=0.55))
    # into the sun: the skyline against the glow, the sun beside the towers
    c = dc - unit(128) * 720
    P.append(dict(name="backlit_skyline", location=at(c, 45), look_at=[*(dc + unit(150) * 40), zc + 70], lens=40, sun_el=2.5,
                  sun_disc=True, haze_m=7000, clouds=0.6, glow=0.6))
    # a drone from the south-west: the low sun rakes across the downtown, long shadows, jungle hills beyond
    c = dc + unit(232) * 520
    P.append(dict(name="raking_drone", location=at(c, 210), look_at=[*dc, zc + 30], lens=30, sun_el=4, haze_m=9000, clouds=0.6))
    # the red tower from a street 150 m away, front-lit, palms in front, foot to crown
    p = open_spot(Ground(d), rx + SUN * 150, [*rx, 0], reach=40)
    P.append(dict(name="red_tower_palms", location=at(p, 1.7), look_at=[*rx, z_at(*rx) + 0.45 * rH], lens=24, sun_el=6,
                  haze_m=12000, glow=0.8, clouds=0.6))
    # over the canopy from the north: the towers rising out of the rainforest, the low sun on the right
    c = dc + unit(95) * 480
    P.append(dict(name="above_the_canopy", location=at(c, 45), look_at=[*dc, zc + 70], lens=35, sun_el=4, haze_m=9000,
                  clouds=0.6))
    # dusk at the Bahia church: the sun just down, windows and lanterns lit
    cams = json.loads((d / "cameras.json").read_text())
    if "church_bahia" in cams:
        v = cams["church_bahia"]
        P.append(dict(name="church_dusk", location=v["location"], look_at=v["look_at"], lens=v["lens"], sun_el=0.4,
                      haze_m=9000, glow=1.2, clouds=0.65, exposure=0.45))
    return P


def plan_coast(d, z_at):
    """The coastal town: the town from the beach, the sun over the sea from the hills, a drone over the bay, palms on
    the beach at sunset, the vista from the mountain road, a street down to the sea."""
    at = lambda xy, h: [float(xy[0]), float(xy[1]), z_at(*xy) + h]
    P = []
    # from the beach: the town climbing the hills, the jungle mountains behind, the sun behind the camera
    P.append(dict(name="town_from_beach", location=at([800.0, 335.0], 1.7), look_at=at([830.0, 820.0], 70.0), lens=24,
                  sun_el=6, sun_az=250, haze_m=6000, clouds=0.5, glow=0.4))
    # from the hillside over the roofs: the sun sets into the sea
    P.append(dict(name="sea_sunset", location=at([700.0, 1060.0], 14.0), look_at=[560.0, 80.0, 0.0], lens=28, sun_el=2.2,
                  sun_az=252, sun_disc=True, haze_m=8000, clouds=0.5, glow=0.7))
    # a drone over the bay: the whole town and the mountains in the last sun
    P.append(dict(name="bay_drone", location=[850.0, 0.0, 260.0], look_at=at([840.0, 520.0], 10.0), lens=34, sun_el=5,
                  sun_az=250, haze_m=9000, clouds=0.55))
    # palms on the beach against the sunset over the sea, looking west along the shore
    g = Ground(d)
    p = open_spot(g, np.array([1290.0, 195.0]), [600.0, 330.0], reach=30)
    P.append(dict(name="palm_beach", location=at(p, 1.5), look_at=at([600.0, 330.0], 8.0), lens=24, sun_el=2.0,
                  sun_az=184, sun_disc=True, haze_m=6000, clouds=0.5, glow=0.6))
    # the vista from the mountain road: the town below, the sea to the horizon
    P.append(dict(name="mountain_vista", location=at([800.0, 1330.0], 4.0), look_at=[800.0, 250.0, 0.0], lens=22, sun_el=4,
                  sun_az=232, haze_m=9000, clouds=0.55, glow=0.5))
    # the beach promenade: palms, the town on the left, the low sun from the sea on the right
    p = g.spot(np.array([560.0, 345.0]), [1250.0, 330.0], reach=25)
    P.append(dict(name="promenade", location=at(p, 1.7), look_at=at([1250.0, 345.0], 14.0), lens=24, sun_el=4, sun_az=215,
                  haze_m=7000, clouds=0.5, glow=0.7))
    return P


def render(d, photos, a):
    photos.sort(key=lambda p: -p["sun_el"])                          # one evening: the sun sinks photo by photo
    for i, p in enumerate(photos):
        p["name"] = f"{i + 1:02d}_{p['name']}"
    pj = d / "showcase.json"
    pj.write_text(json.dumps(photos, indent=1))
    cmd = [BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"), "--",
           "--tile", str(d), "--photos", str(pj), "--res", str(a.res), "--samples", str(a.samples), "--out", "showcase"]
    if a.only:
        cmd += ["--only", ",".join(p["name"] for p in photos if any(o in p["name"] for o in a.only.split(",")))]
    rc = subprocess.run(cmd, stdout=open(d / "showcase.log", "w"), stderr=subprocess.STDOUT).returncode
    if rc == 0:
        sheet(d / "showcase", [p["name"] for p in photos])
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="all", choices=["landmarks", "coast", "all"])
    ap.add_argument("--only", default=None, help="comma list of name fragments")
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=128)
    a = ap.parse_args()
    rc = {}
    if a.city in ("landmarks", "all"):
        from m16_kaist_campus import domain
        big, _, _ = domain()
        rc["landmarks"] = render(LANDMARKS, plan_landmarks(LANDMARKS, terrain_z(big, L), big), a)
    if a.city in ("coast", "all"):
        z = np.load(ROOT / "outputs" / "latin_caribbean" / "coast" / "terrain_t1.npz")
        big = {"elev": z["elev"], "water": z["water"], "valid": np.ones(z["elev"].shape, bool)}
        rc["coast"] = render(COAST, plan_coast(COAST, terrain_z(big, L)), a)
    print(json.dumps(rc))
    return max(rc.values())


if __name__ == "__main__":
    sys.exit(main())
