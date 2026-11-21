"""Milestone 16: a 3 x 3 (1536 m) explorable 3D city from a SAVED street graph on the KAIST campus terrain.

The 3D part of scripts/m12_big_landscape.py unchanged (BF parcels and buildings per 512 m tile, building filter for
water and slope, M11 scene, terrain draping, bridges, merge into one scene, cameras and the Godot spawn), but the
streets come from outputs/m16/kaist/all_graphs.pkl (or the real campus network) and the terrain is the GLO-30 domain
of the terrain pack. Then: Blender (blender/m11_scene.py) and Godot screenshots (godot/viewer --shot).

    python scripts/m16_city3d_big.py --method "NA-fixed" [--seed 1]   ->  outputs/m16/cities/kaist_all_<key>_s<seed>/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m11_big_city import clip, merge, to_graph
from m12_big_landscape import MARGIN, decode, window_dict
from m16_kaist_all import key_of
from m16_kaist_campus import L, OUT, W, domain
from urbanmorph.city3d.convert import from_generated
from urbanmorph.city3d.format import save_city, validate_city
from urbanmorph.city3d.scene import build_scene, save_scene
from urbanmorph.city3d.terrain3d import Surfaces, apply_terrain
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.terrain.heightfield import Terrain

CITIES = ROOT / "outputs" / "m16" / "cities"


def segments(g):
    """(segments x0 y0 x1 y1, code = class + 4 x structure) of a RoadGraph, as the M12 big city stores streets."""
    S = np.c_[g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]]
    st = np.clip(np.asarray(g.structs()), 0, 2)
    return S, np.asarray(g.classes()) % 4 + 4 * st


def main() -> int:
    from m10_integrate import bf_layouts  # noqa: E402
    from m12_build_city3d import buildable  # noqa: E402
    from m6_build_objects import load_objects  # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help='a label of scripts/m16_kaist_all.py ORDER, or "real"')
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    big, _, _ = domain()
    T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -MARGIN, L + MARGIN, 4.0)
    T.water_kind = big["water"].astype(np.int8)
    T.water_surface = np.where(big["water"] > 0, big["elev"] - 1.0, np.nan).astype(np.float32)
    gp = pickle.loads((OUT / "all_graphs.pkl").read_bytes())
    g = gp["real"] if a.method == "real" else gp["graphs"][(a.method, a.seed)]
    S, C = segments(g)
    cid = f"kaist_all_{'real' if a.method == 'real' else key_of(a.method)}_s{a.seed}"
    d = CITIES / cid
    d.mkdir(parents=True, exist_ok=True)
    cw = load_objects()["class_widths"]
    priors = json.loads((ROOT / "outputs" / "m11" / "audit" / "height_priors.json").read_text())
    n = int(L // W)
    tiles = [(r, c) for r in range(n) for c in range(n)]
    graphs = [decode(to_graph(*clip(S, C, (c * W, r * W, c * W + W, r * W + W)), (c * W, r * W))) for r, c in tiles]
    lay = bf_layouts(graphs, ["spain"] * len(tiles), cw, seed=0)
    parts, reps, zmin, zmax = [], {}, np.inf, -np.inf
    for k, ((r, c), (st, P), gg) in enumerate(zip(tiles, lay, graphs)):
        city = from_generated(f"{cid}_r{r}c{c}", st.graph, st, P, cw, "spain", priors, {"value": 3.0, "source": "default"},
                              a.seed + k, {"streets": f"{a.method} (M16 KAIST campus)", "terrain": "GLO-30 DSM, terrain pack",
                                           "parcels_buildings": "BF (M10) filtered for water and slope",
                                           "heights": "synthetic", "roofs_styles": "synthetic"})
        wd = window_dict(T, c * W, r * W)
        tt = terrain_of(wd)
        tt.water_surface = wd["water_surface"]
        city["buildings"], rm_w, rm_s = buildable(city["buildings"], tt)
        save_city(city, d / f"city_r{r}c{c}.json")
        groups, mats, rep = build_scene(city)
        groups, mats, trep = apply_terrain(groups, mats, city, tt, gg)
        parts.append((c * W, r * W, groups, mats))
        reps[f"r{r}c{c}"] = {"summary": rep["summary"], "terrain3d": trep, "format_errors": len(validate_city(city)),
                             "buildability_filter": {"removed_water": rm_w, "removed_slope": rm_s}}
        zmin, zmax = min(zmin, trep["surround_z"]), max(zmax, float(np.nanmax(tt.elev)))
        print(f"r{r}c{c}", rep["summary"]["n_buildings"], "buildings", flush=True)
    groups, mats = merge(parts, L)
    save_scene(d / "meshes.npz", groups, mats)
    np.save(d / "streets.npy", np.c_[S, C % 4, C // 4])
    mid = L / 2
    seg = S[(C // 4 == 0) & (np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]) > 15)]
    m = 0.5 * (seg[:, :2] + seg[:, 2:]) if len(seg) else np.array([[mid, mid]])
    k0 = int(np.argmin(np.hypot(m[:, 0] - mid, m[:, 1] - mid)))
    sx, sy = m[k0]
    dv = (seg[k0, 2:] - seg[k0, :2]) / np.hypot(*(seg[k0, 2:] - seg[k0, :2])) if len(seg) else np.array([0.0, 1.0])
    tc, tr = int(min(sx // W, n - 1)), int(min(sy // W, n - 1))
    Sx = Surfaces(terrain_of(window_dict(T, tc * W, tr * W)))
    zs = float(Sx.road(np.array([sx - tc * W]), np.array([sy - tr * W]))[0])
    lx, ly = sx + 100 * dv[0], sy + 100 * dv[1]
    cams = {"extent_m": L, "surround_z": zmin,
            "aerial": {"type": "ORTHO", "location": [L / 2, L / 2, 1200.0 + zmax], "ortho_scale": L},
            "oblique": {"type": "PERSP", "lens": 30, "location": [L / 2 - 376 * n, L / 2 - 416 * n, 330 * n + zmax],
                        "look_at": [L / 2, L / 2, (zmin + zmax) / 2]},
            "street": {"type": "PERSP", "lens": 22, "location": [sx, sy, zs + 1.7], "look_at": [lx, ly, zs + 1.7]},
            "godot_spawn": {"spawn": [sx, zs + 2.0, -sy], "look_at": [lx, zs + 2.0, -ly], "extent": L}}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1, default=float))
    (d / "viewer.json").write_text(json.dumps(cams["godot_spawn"], default=float))
    total = {k: int(sum(r["summary"][k] for r in reps.values())) for k in ("n_buildings", "watertight", "outward", "floating")}
    (d / "validation.json").write_text(json.dumps({"city": cid, "size_m": L, "method": a.method, "seed": a.seed,
                                                    "total": total, "tiles": reps, "seconds": round(time.time() - t0, 1)},
                                                   indent=1, default=float))
    print(json.dumps({"dir": str(d), **total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
