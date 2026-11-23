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
    ap.add_argument("--graphs", default=str(OUT / "all_graphs.pkl"), help="M17: another graphs pickle (same format)")
    ap.add_argument("--tag", default="kaist_all", help="M17: city directory prefix")
    ap.add_argument("--theme", default="standard", choices=["standard", "latin_caribbean"], help="building / plant theme")
    ap.add_argument("--terrain-npz", default=None, help="a synthetic domain (elev, water, water_surface) instead of KAIST")
    ap.add_argument("--village-npz", default=None, help="themes: a mask on the domain grid; no buildings or yards outside it")
    ap.add_argument("--jungle", default="normal", choices=["normal", "dense"], help="themes: jungle density on free land")
    ap.add_argument("--infill", default="bf", choices=["bf", "colonial"], help="themes: dense colonial row houses in the village")
    ap.add_argument("--plazas", default=None, help="themes: plaza + church sites (json, world coordinates; latin.infill)")
    ap.add_argument("--giant-lottery", default=None, help="themes: 'lo,hi' m, giant towers on the other churches too")
    ap.add_argument("--lanes", type=float, default=None, help="themes: colonial infill cuts big blocks by lanes every M m")
    a = ap.parse_args()
    t0 = time.time()
    if a.terrain_npz:
        z = np.load(a.terrain_npz)
        big = {"elev": z["elev"], "valid": np.ones(z["elev"].shape, bool), "water": z["water"]}
    else:
        big, _, _ = domain()
    T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -MARGIN, L + MARGIN, 4.0)
    T.water_kind = big["water"].astype(np.int8)
    T.water_surface = (z["water_surface"].astype(np.float32) if a.terrain_npz else
                       np.where(big["water"] > 0, big["elev"] - 1.0, np.nan).astype(np.float32))
    if a.theme == "latin_caribbean" and not a.terrain_npz:
        # the DSM already sits at the river's surface: water just above it is visible (elev - 1 hid it under the terrain)
        T.water_surface = np.where(big["water"] > 0, big["elev"] + 0.3, np.nan).astype(np.float32)
    village = np.load(a.village_npz)["mask"] if a.village_npz else None
    gp = pickle.loads(Path(a.graphs).read_bytes())
    g = gp["real"] if a.method == "real" else gp["graphs"][(a.method, a.seed)]
    S, C = segments(g)
    cid = f"{a.tag}_{'real' if a.method == 'real' else key_of(a.method).replace('[', '').replace(']', '')}_s{a.seed}"
    cid += "_latin" if a.theme == "latin_caribbean" else ""
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
        infill = None
        if a.infill == "colonial":
            from urbanmorph.city3d.latin.infill import colonial_infill
            vin = (lambda x, y: bool(village[min(max(int((L + MARGIN - (y + r * W)) / 4.0), 0), village.shape[0] - 1),
                                             min(max(int((x + c * W + MARGIN) / 4.0), 0), village.shape[1] - 1)])) if village is not None else None
            sites = json.loads(Path(a.plazas).read_text()) if a.plazas else []
            infill = colonial_infill(city, np.random.default_rng([a.seed, k, 5]), vin, sites, (c * W, r * W), a.lanes)
        city["buildings"], rm_w, rm_s = buildable(city["buildings"], tt)
        if village is not None:                                  # themes: the village only; the rest is jungle
            inside = lambda x, y: bool(village[min(max(int((L + MARGIN - (y + r * W)) / 4.0), 0), village.shape[0] - 1),
                                               min(max(int((x + c * W + MARGIN) / 4.0), 0), village.shape[1] - 1)])
            rp = lambda rings_: np.asarray(rings_[0])[:, :2].mean(0)
            city["buildings"] = [b for b in city["buildings"] if inside(*rp(b["footprint"]))]
            city["landuse"] = [g for g in city["landuse"] if inside(*rp(g["polygon"]))]
        if a.theme == "latin_caribbean":
            city["metadata"]["theme_options"] = {"jungle": a.jungle, "infill": a.infill}
            if a.giant_lottery:
                city["metadata"]["theme_options"]["giant_lottery_m"] = [float(x) for x in a.giant_lottery.split(",")]
        save_city(city, d / f"city_r{r}c{c}.json")
        groups, mats, rep = build_scene(city, theme=a.theme)
        groups, mats, trep = apply_terrain(groups, mats, city, tt, gg)
        if a.theme == "latin_caribbean":
            from urbanmorph.city3d.latin.build import dress_terrain
            trep["latin_dressing"] = dress_terrain(groups, mats, city, tt, a.seed + k)
        parts.append((c * W, r * W, groups, mats))
        reps[f"r{r}c{c}"] = {"summary": rep["summary"], "terrain3d": trep, "format_errors": len(validate_city(city)),
                             "buildability_filter": {"removed_water": rm_w, "removed_slope": rm_s}, "infill": infill}
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
    if (T.water_kind == 3).any():                                  # themes: an ocean to the horizon at sea level
        cams["sea_z"] = cams["godot_spawn"]["sea_z"] = float(np.nanmedian(T.water_surface[T.water_kind == 3]))
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
