"""Alleys of the landmark city in the last light (user request 2026-11-24: "explore the different alleys to create the
best picture ... the ambient light and the picture overall hyper-realistic").

search:  every street point (every 6 m) that looks down its own street (within 24 deg) at a target 60-420 m away
         (the six landmark towers, the giant church towers, the photo churches) with at least the upper 40 % of the
         target in sight over the houses (building heights, terrain) and no low crown in the first 30 m. Scored by
         the share of the target in view, houses close on both sides (an alley, not an avenue), alignment and
         distance; then a diverse best set (cameras 45 m apart, at most three per target). Each candidate is framed
         from the street to the target's top (16:9) and lit by the last sun behind the camera's shoulder.
         ->  <city>/alleys/candidates.json, EEVEE previews <city>/alleys/preview/*.jpg and labelled contact sheets
final:   the chosen candidates (--pick, optional --edits json {name: {key: value}}) path traced in Cycles
         ->  <city>/alleys/final/*.jpg

    .venv/bin/python themes/latin_caribbean/scripts/alleys.py search [--n 32]
    .venv/bin/python themes/latin_caribbean/scripts/alleys.py final --pick a,b,c [--edits edits.json] [--samples 256]
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
import shapely

from kaist_village import BLENDER, terrain_z
from photo_tour import Ground

CITY = ROOT / "outputs" / "latin_caribbean" / "cities" / "kaist" / "kaist_village_landmarks_NA-adaptive_s1_latin"
L, W = 1536.0, 512.0


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def targets(d):
    gt = json.loads((d / "giant_towers.json").read_text())
    T = [(np.asarray(t["xy"]), t["height_m"], 16.0, f"landmark_{t['design']}") for t in gt.get("landmarks", [])]
    T += [(np.asarray(t["xy"]), t["height_m"], t["side_m"], "church_tower") for t in gt.get("towers", [])]
    T += [(np.asarray(t["xy"]), t["height_m"], 26.0, f"church_{t['style']}") for t in gt.get("photo_churches", [])]
    return T


def buildings(d):
    foot, top = [], []
    for f in sorted(d.glob("city_r*c*.json")):
        r, c = int(f.stem[6]), int(f.stem[8])
        for b in json.loads(f.read_text())["buildings"]:
            p = shapely.Polygon(np.asarray(b["footprint"][0])[:, :2] + [c * W, r * W])
            sk = b.get("skyscraper")
            h = sk["H"] if sk else (22.0 if b.get("church_site") else float(b.get("height_m") or 9.0) + 2.5)
            foot.append(p)
            top.append(h)
    return foot, np.asarray(top)


def lens_for(vfov):
    return float(18.0 / (np.tan(np.radians(vfov) / 2) * 16 / 9))


def search(d, z_at, n_best=32):
    T = targets(d)
    foot, top = buildings(d)
    tree = shapely.STRtree(foot)
    zb = np.array([z_at(*np.asarray(p.representative_point().coords)[0]) for p in foot])
    g = Ground(d)
    st = np.load(d / "streets.npy")
    own = [set(tree.query(shapely.Point(t[0]).buffer(t[2] / 2 + 6.0))) for t in T]
    cands = []
    for x0, y0, x1, y1, cls, stc in st:
        a, b = np.array([x0, y0]), np.array([x1, y1])
        Ls = float(np.hypot(*(b - a)))
        if Ls < 20 or stc != 0:                                   # bridges / tunnels off
            continue
        e = (b - a) / Ls
        for s_ in np.arange(3.0, Ls - 3.0, 6.0):
            p = a + e * s_
            if g.kd.query(p)[0] < 5.0 or len(tree.query(shapely.Point(p).buffer(1.2), predicate="intersects")):
                continue
            zc = None
            for k, (tx, H, side, kind) in enumerate(T):
                v = tx - p
                D = float(np.hypot(*v))
                if not 60.0 <= D <= 420.0:
                    continue
                u = v / D
                ang = np.degrees(np.arccos(np.clip(abs(np.dot(u, e)), 0, 1)))
                if ang > 24.0:
                    continue
                zc = z_at(*p) + 1.65 if zc is None else zc
                zt = z_at(*tx)
                sight = shapely.LineString([p + u * 2.0, tx - u * (side / 2 + 2.0)]).buffer(0.8)
                occ = -np.inf
                for j in tree.query(sight, predicate="intersects"):
                    if j in own[k]:
                        continue
                    dj = max(foot[j].distance(shapely.Point(p)), 0.8)
                    occ = max(occ, np.arctan2(zb[j] + top[j] - zc, dj))
                fmin = (np.tan(occ) * D + zc - zt) / H if np.isfinite(occ) else 0.0
                vis = 1.0 - float(np.clip(fmin, 0.0, 1.0))
                if vis < 0.4:
                    continue
                if (g.kd.query(p + np.outer(np.arange(3.0, min(30.0, D / 2), 3.0), u))[0] < 4.5).any():
                    continue
                nrm = np.array([-u[1], u[0]])                     # houses close on both sides: an alley
                sides = sum(bool(len(tree.query(shapely.LineString([p, p + sgn * nrm * 13.0]), predicate="intersects")))
                            for sgn in (-1, 1))
                ahead = len(tree.query(shapely.LineString([p, p + u * 40.0]).buffer(14.0), predicate="intersects"))
                score = (2.0 * vis + 0.7 * sides + 0.04 * min(ahead, 12) - ang / 40.0 - abs(D - 190.0) / 400.0
                         + (0.35 if kind.startswith("landmark") else 0.0))
                cands.append((score, p, zc, k, D, u, vis, sides))
    cands.sort(key=lambda c: -c[0])
    picked, per = [], {}
    for c in cands:
        score, p, zc, k, D, u, vis, sides = c
        if per.get(k, 0) >= 3 or any(np.hypot(*(p - q[1])) < 45.0 for q in picked):
            continue
        if any(q[3] == k and np.degrees(np.arccos(np.clip(np.dot(u, q[5]), -1, 1))) < 20 for q in picked):
            continue
        picked.append(c)
        per[k] = per.get(k, 0) + 1
        if len(picked) >= n_best:
            break
    photos = []
    for i, (score, p, zc, k, D, u, vis, sides) in enumerate(picked):
        tx, H, side, kind = T[k]
        zt = z_at(*tx)
        topa = np.degrees(np.arctan2(zt + H - zc, D)) + 3.5
        bot = -9.0
        vf = float(np.clip(topa - bot, 30.0, 72.0))
        pitch = np.radians(topa - vf / 2)
        look = [p[0] + u[0] * 100 * np.cos(pitch), p[1] + u[1] * 100 * np.cos(pitch), zc + 100 * np.sin(pitch)]
        sun_az = float(np.degrees(np.arctan2(-u[1], -u[0])) + (35.0 if i % 2 else -35.0))   # behind the shoulder
        photos.append({"name": f"a{i:02d}_{kind}", "location": [float(p[0]), float(p[1]), float(zc)], "look_at": look,
                       "lens": lens_for(vf), "sun_el": 2.5, "sun_az": sun_az, "haze_m": 6000, "clouds": 0.55, "glow": 0.9,
                       "score": round(float(score), 3), "visible": round(float(vis), 2), "sides": int(sides),
                       "distance_m": round(D, 1), "target": kind})
    return photos


def level(p):
    """The same frame from a level camera with a shift lens (architectural photography: verticals stay vertical)."""
    loc, look = np.asarray(p["location"], float), np.asarray(p["look_at"], float)
    v = look - loc
    hz = float(np.hypot(v[0], v[1]))
    pitch = np.arctan2(v[2], hz)
    hfov = 2 * np.arctan(18.0 / p["lens"])
    half = np.arctan(np.tan(hfov / 2) * 9 / 16)                    # vertical half angle of the 16:9 frame
    pad = np.radians(float(p.get("top_pad_deg", 0.0)))              # more sky over the tallest tower
    t_top, t_bot = np.tan(pitch + half + pad), np.tan(pitch - half)
    f = 2 * 10.125 / (t_top - t_bot)                               # 36 x 20.25 mm frame
    q = dict(p)
    q["lens"] = float(f)
    q["shift_y"] = float(f * (t_top + t_bot) / 2 / 36.0)
    q["look_at"] = [float(loc[0] + v[0] / hz * 100), float(loc[1] + v[1] / hz * 100), float(loc[2])]
    return q


def render(d, photos, out, res, samples, engine="EEVEE"):
    pj = d / "alleys" / f"{out}.json"
    pj.write_text(json.dumps(photos, indent=1))
    cmd = [BLENDER, "-b", str(d / "scene.blend"), "--python-exit-code", "1", "--python", str(ROOT / "blender" / "photo_tour.py"), "--",
           "--tile", str(d), "--photos", str(pj), "--res", str(res), "--samples", str(samples), "--out", f"alleys/{out}",
           "--engine", engine]
    return subprocess.run(cmd, stdout=open(d / "alleys" / f"{out}.log", "w"), stderr=subprocess.STDOUT).returncode


def sheets(folder, names, per=12, cols=4, w=640):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    h = int(w * 9 / 16)
    for s in range(0, len(names), per):
        chunk = names[s:s + per]
        S = Image.new("RGB", (cols * w, -(-len(chunk) // cols) * (h + 30)), "white")
        dr = ImageDraw.Draw(S)
        for i, n in enumerate(chunk):
            f = folder / f"{n}.jpg"
            if not f.exists():
                continue
            x, y = (i % cols) * w, (i // cols) * (h + 30)
            S.paste(Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS), (x, y + 30))
            dr.text((x + 6, y + 4), n, fill=(0, 0, 0), font=font)
        S.save(folder / f"sheet_{s // per + 1}.jpg", quality=86)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "final"])
    ap.add_argument("--city", default=str(CITY))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--pick", default=None)
    ap.add_argument("--edits", default=None, help="json {name: {key: value}} applied to the picked candidates")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args()
    d = Path(a.city).resolve()
    (d / "alleys").mkdir(exist_ok=True)
    if a.stage == "search":
        from m16_kaist_campus import domain
        big, _, _ = domain()
        photos = search(d, terrain_z(big, L), a.n)
        (d / "alleys" / "candidates.json").write_text(json.dumps(photos, indent=1))
        print(json.dumps([(p["name"], p["score"], p["visible"], p["sides"], p["distance_m"]) for p in photos]), flush=True)
        rc = render(d, photos, "preview", a.res or 960, a.samples or 24)
        sheets(d / "alleys" / "preview", [p["name"] for p in photos])
        return rc
    cand = {p["name"]: p for p in json.loads((d / "alleys" / "candidates.json").read_text())}
    edits = json.loads(Path(a.edits).read_text()) if a.edits else {}
    picks = []
    for k, nm in enumerate(a.pick.split(",")):
        base = dict(cand.get(nm.split("+")[0], {}))
        p = {**base, **edits.get(nm, {})}
        p["name"] = f"{k + 1:02d}_{nm.replace('+', '_')}"
        picks.append(level(p) if p.get("level", True) else p)
    rc = render(d, picks, "final", a.res or 2400, a.samples or 256, "CYCLES")
    sheets(d / "alleys" / "final", [p["name"] for p in picks], per=12, cols=3, w=800)
    return rc


if __name__ == "__main__":
    sys.exit(main())
