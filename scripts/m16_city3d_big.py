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


def village_polygon(village, Ld, x0, y0):
    """The village mask as a polygon in the tile's local metres (tile south-west corner at world (x0, y0))."""
    import shapely
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    tr = from_origin(-MARGIN - x0, Ld + MARGIN - y0, 4.0, 4.0)
    polys = [shapely.geometry.shape(gm) for gm, v in shapes(village.astype(np.uint8), mask=village, transform=tr) if v == 1]
    return shapely.make_valid(shapely.union_all(polys)).intersection(shapely.box(-20, -20, W + 20, W + 20))


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
    ap.add_argument("--jungle", default="normal", choices=["normal", "dense", "desert"],
                    help="themes: jungle density on free land, or the desert (petra.py)")
    ap.add_argument("--infill", default="bf", choices=["bf", "colonial"], help="themes: dense colonial row houses in the village")
    ap.add_argument("--plazas", default=None, help="themes: plaza + church sites (json, world coordinates; latin.infill)")
    ap.add_argument("--giant-lottery", default=None, help="themes: 'lo,hi' m, giant towers on the other churches too")
    ap.add_argument("--lanes", type=float, default=None, help="themes: colonial infill cuts big blocks by lanes every M m")
    ap.add_argument("--district", default=None, help="themes: a Latin skyscraper district (json, world coordinates)")
    ap.add_argument("--theme-opts", default=None, help="themes: extra theme options (json), e.g. church_colour, church_dome")
    ap.add_argument("--cities-dir", default=str(CITIES), help="where the city directory goes (themes keep their own folder)")
    ap.add_argument("--build-slope", type=float, default=None, help="buildability: max landform slope (default BUILD_SLOPE)")
    ap.add_argument("--seat", default="min", choices=["min", "street"], help="building bases (terrain3d.apply_terrain)")
    ap.add_argument("--plinth", default="bridge_concrete", help="seat street: the plinth material")
    ap.add_argument("--site", default="M16 KAIST campus", help="provenance label of the streets and terrain")
    ap.add_argument("--village-clip", action="store_true", help="colonial infill: blocks cut to the village outline")
    ap.add_argument("--lot-depth", default=None, help="colonial infill: 'lo,hi' m row-house band (default 11,15)")
    ap.add_argument("--below-max", type=int, default=3, help="seat street: most storeys a downhill house adds below")
    ap.add_argument("--terraces", type=float, default=None, help="seat street: open land as terraces of this step (m)")
    ap.add_argument("--stairs", type=float, default=None, help="seat street: streets steeper than this grade as stairs")
    ap.add_argument("--cliff-m", type=float, default=9.0, help="seat street: drop a house over a wall higher than this (m)")
    a = ap.parse_args()
    t0 = time.time()
    Ld = L
    if a.terrain_npz:
        z = np.load(a.terrain_npz)
        big = {"elev": z["elev"], "valid": np.ones(z["elev"].shape, bool), "water": z["water"]}
        Ld = float((z["elev"].shape[0] - 2 * MARGIN / 4.0) * 4.0)       # the domain of the npz (KAIST: 1536 m)
    else:
        big, _, _ = domain()
    T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -MARGIN, Ld + MARGIN, 4.0)
    T.water_kind = big["water"].astype(np.int8)
    T.water_surface = (z["water_surface"].astype(np.float32) if a.terrain_npz and "water_surface" in z else
                       np.where(big["water"] > 0, big["elev"] - 1.0, np.nan).astype(np.float32))
    road_z = z["road_z"].astype(np.float32) if a.terrain_npz and "road_z" in z else None
    masonry = z["masonry"].astype(bool) if a.terrain_npz and "masonry" in z else None
    if a.theme == "latin_caribbean" and (not a.terrain_npz or "water_surface" not in z):
        # the DSM already sits at the river's surface: water just above it is visible (elev - 1 hid it under the terrain)
        T.water_surface = np.where(big["water"] > 0, big["elev"] + 0.3, np.nan).astype(np.float32)
    village = np.load(a.village_npz)["mask"] if a.village_npz else None
    gp = pickle.loads(Path(a.graphs).read_bytes())
    g = gp["real"] if a.method == "real" else gp["graphs"][(a.method, a.seed)]
    S, C = segments(g)
    cid = f"{a.tag}_{'real' if a.method == 'real' else key_of(a.method).replace('[', '').replace(']', '')}_s{a.seed}"
    cid += "_latin" if a.theme == "latin_caribbean" else ""
    d = Path(a.cities_dir) / cid
    d.mkdir(parents=True, exist_ok=True)
    cw = load_objects()["class_widths"]
    priors = json.loads((ROOT / "outputs" / "m11" / "audit" / "height_priors.json").read_text())
    n = int(Ld // W)
    tiles = [(r, c) for r in range(n) for c in range(n)]
    graphs = [decode(to_graph(*clip(S, C, (c * W, r * W, c * W + W, r * W + W)), (c * W, r * W))) for r, c in tiles]
    lay = bf_layouts(graphs, ["spain"] * len(tiles), cw, seed=0)
    parts, reps, zmin, zmax = [], {}, np.inf, -np.inf
    for k, ((r, c), (st, P), gg) in enumerate(zip(tiles, lay, graphs)):
        city = from_generated(f"{cid}_r{r}c{c}", st.graph, st, P, cw, "spain", priors, {"value": 3.0, "source": "default"},
                              a.seed + k, {"streets": f"{a.method} ({a.site})", "terrain": "GLO-30 DSM, " + ("terrain pack" if a.site.startswith("M16") else a.site),
                                           "parcels_buildings": "BF (M10) filtered for water and slope",
                                           "heights": "synthetic", "roofs_styles": "synthetic"})
        wd = window_dict(T, c * W, r * W)
        tt = terrain_of(wd)
        tt.water_surface = wd["water_surface"]
        if road_z is not None:
            r0, c0 = int(round((T.y1 - (r * W + W + MARGIN)) / T.res)), int(round((c * W - MARGIN - T.x0) / T.res))
            tt.road_z = road_z[r0:r0 + 256, c0:c0 + 256]
        if masonry is not None:
            r0, c0 = int(round((T.y1 - (r * W + W + MARGIN)) / T.res)), int(round((c * W - MARGIN - T.x0) / T.res))
            tt.masonry = masonry[r0:r0 + 256, c0:c0 + 256]
        infill = None
        if a.infill == "colonial":
            from urbanmorph.city3d.latin.infill import colonial_infill
            vin = (lambda x, y: bool(village[min(max(int((Ld + MARGIN - (y + r * W)) / 4.0), 0), village.shape[0] - 1),
                                             min(max(int((x + c * W + MARGIN) / 4.0), 0), village.shape[1] - 1)])) if village is not None else None
            sites = json.loads(Path(a.plazas).read_text()) if a.plazas else []
            dist = json.loads(Path(a.district).read_text()) if a.district else None
            vclip = village_polygon(village, Ld, c * W, r * W) if a.village_clip and village is not None else None
            infill = colonial_infill(city, np.random.default_rng([a.seed, k, 5]), vin, sites, (c * W, r * W), a.lanes, dist, vclip,
                                     [float(x) for x in a.lot_depth.split(",")] if a.lot_depth else None)
        city["buildings"], rm_w, rm_s = buildable(city["buildings"], tt, *([a.build_slope] if a.build_slope else []))
        if village is not None:                                  # themes: the village only; the rest is jungle
            inside = lambda x, y: bool(village[min(max(int((Ld + MARGIN - (y + r * W)) / 4.0), 0), village.shape[0] - 1),
                                               min(max(int((x + c * W + MARGIN) / 4.0), 0), village.shape[1] - 1)])
            rp = lambda rings_: np.asarray(rings_[0])[:, :2].mean(0)
            city["buildings"] = [b for b in city["buildings"] if inside(*rp(b["footprint"]))]
            city["landuse"] = [g for g in city["landuse"] if inside(*rp(g["polygon"]))]
            if a.village_clip:                                   # and no park or yard reaching out into the jungle
                import shapely
                from urbanmorph.city3d.format import polygon, rings
                vc = village_polygon(village, Ld, c * W, r * W)
                lu = []
                for lu_ in city["landuse"]:
                    q = shapely.make_valid(polygon(lu_["polygon"])).intersection(vc)
                    for k_, part in enumerate(getattr(q, "geoms", [q])):
                        if part.geom_type == "Polygon" and part.area >= 2.0:
                            lu.append({**lu_, "id": f"{lu_['id']}" + (f"_{k_}" if k_ else ""), "polygon": rings(part)})
                city["landuse"] = lu
        if a.theme == "latin_caribbean":
            city["metadata"]["theme_options"] = {"jungle": a.jungle, "infill": a.infill, "tile": f"r{r}c{c}"}
            if a.giant_lottery:
                city["metadata"]["theme_options"]["giant_lottery_m"] = [float(x) for x in a.giant_lottery.split(",")]
            if a.theme_opts:
                city["metadata"]["theme_options"].update(json.loads(a.theme_opts))
        if a.seat == "street":                                   # steep towns: downhill houses get storeys below the street
            from urbanmorph.city3d.terrain3d import street_levels
            gseg = gg.segments()[np.asarray(gg.structs()) == 0] if len(gg.edges) else np.zeros((0, 4))
            keep_b, cliff, cliff_lots = [], 0, []
            for b, lv in zip(city["buildings"], street_levels(Surfaces(tt), city, gseg)):
                special = b.get("church_site") or b.get("skyscraper")
                if lv and not special and lv[0] - lv[1] > 3.5 and a.below_max > 0:
                    b["below_floors"] = int(min((lv[0] - lv[1]) // 3.5, a.below_max))   # the rest: a stone terrace wall
                if lv and not special and lv[0] - lv[1] - 3.5 * b.get("below_floors", 0) > a.cliff_m:
                    cliff += 1                                   # a cliff lot: too high a wall under the house
                    cliff_lots.append(b["footprint"])
                    continue
                if lv:
                    b["street_z"] = float(lv[0])                 # the house builder: walls a stepped neighbour leaves free
                keep_b.append(b)
            city["buildings"] = keep_b
        save_city(city, d / f"city_r{r}c{c}.json")
        terr = stairs = None
        if a.seat == "street" and a.terraces:                    # steep towns: yards, patios and leftovers as terraces
            import shapely
            from urbanmorph.city3d.format import polygon, rings
            road_b = shapely.union_all([shapely.LineString(sg.reshape(2, 2)).buffer(cw[int(cl)] / 2 + 2.2)
                                        for sg, cl in zip(gg.segments(), gg.classes())]) if len(gg.edges) else shapely.Polygon()
            is_open = lambda lu_: lu_["class"] in ("private_green", "paved") or (lu_["class"] == "public_green"
                                                                                  and lu_.get("source") != "theme:plaza")
            open_ = [lu_ for lu_ in city["landuse"] if is_open(lu_)]
            city["landuse"] = [lu_ for lu_ in city["landuse"] if not is_open(lu_)]
            open_ += [{"polygon": fp_} for fp_ in (cliff_lots if a.seat == "street" else [])]     # empty cliff lots too
            city["terraces"] = [rings(q) for lu_ in open_ for q in
                                getattr(shapely.make_valid(polygon(lu_["polygon"])).difference(road_b), "geoms",
                                        [shapely.make_valid(polygon(lu_["polygon"])).difference(road_b)])
                                if q.geom_type == "Polygon" and q.area >= 25.0]
            mat_ = "latin:" if a.theme == "latin_caribbean" else ""
            terr = {"step": a.terraces, "top": mat_ + ("grass_public" if mat_ else "ground"), "wall": a.plinth}
        if a.seat == "street" and a.stairs:
            stairs = {"grade": a.stairs, "material": "latin:cobble" if a.theme == "latin_caribbean" else "bridge_concrete",
                      "walk": 1.6}
        groups, mats, rep = build_scene(city, theme=a.theme)
        groups, mats, trep = apply_terrain(groups, mats, city, tt, gg, seat=a.seat, plinth=a.plinth, terraces=terr, stairs=stairs,
                                           **({"class_widths": tuple(cw)} if a.seat == "street" else {}))
        if a.theme == "latin_caribbean":
            from urbanmorph.city3d.latin.build import dress_terrain
            trep["latin_dressing"] = dress_terrain(groups, mats, city, tt, a.seed + k)
        parts.append((c * W, r * W, groups, mats))
        reps[f"r{r}c{c}"] = {"summary": rep["summary"], "terrain3d": trep, "format_errors": len(validate_city(city)),
                             "buildability_filter": {"removed_water": rm_w, "removed_slope": rm_s,
                                                     "removed_cliff": cliff if a.seat == "street" else 0}, "infill": infill}
        zmin, zmax = min(zmin, trep["surround_z"]), max(zmax, float(np.nanmax(tt.elev)))
        print(f"r{r}c{c}", rep["summary"]["n_buildings"], "buildings", flush=True)
    groups, mats = merge(parts, Ld)
    save_scene(d / "meshes.npz", groups, mats)
    np.save(d / "streets.npy", np.c_[S, C % 4, C // 4])
    mid = Ld / 2
    seg = S[(C // 4 == 0) & (np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]) > 15)]
    m = 0.5 * (seg[:, :2] + seg[:, 2:]) if len(seg) else np.array([[mid, mid]])
    k0 = int(np.argmin(np.hypot(m[:, 0] - mid, m[:, 1] - mid)))
    sx, sy = m[k0]
    dv = (seg[k0, 2:] - seg[k0, :2]) / np.hypot(*(seg[k0, 2:] - seg[k0, :2])) if len(seg) else np.array([0.0, 1.0])
    tc, tr = int(min(sx // W, n - 1)), int(min(sy // W, n - 1))
    Sx = Surfaces(terrain_of(window_dict(T, tc * W, tr * W)))
    zs = float(Sx.road(np.array([sx - tc * W]), np.array([sy - tr * W]))[0])
    lx, ly = sx + 100 * dv[0], sy + 100 * dv[1]
    cams = {"extent_m": Ld, "surround_z": zmin,
            "aerial": {"type": "ORTHO", "location": [Ld / 2, Ld / 2, 1200.0 + zmax], "ortho_scale": Ld},
            "oblique": {"type": "PERSP", "lens": 30, "location": [Ld / 2 - 376 * n, Ld / 2 - 416 * n, 330 * n + zmax],
                        "look_at": [Ld / 2, Ld / 2, (zmin + zmax) / 2]},
            "street": {"type": "PERSP", "lens": 22, "location": [sx, sy, zs + 1.7], "look_at": [lx, ly, zs + 1.7]},
            "godot_spawn": {"spawn": [sx, zs + 2.0, -sy], "look_at": [lx, zs + 2.0, -ly], "extent": Ld}}
    if (T.water_kind == 3).any():                                  # themes: an ocean to the horizon at sea level
        cams["sea_z"] = cams["godot_spawn"]["sea_z"] = float(np.nanmedian(T.water_surface[T.water_kind == 3]))
    (d / "cameras.json").write_text(json.dumps(cams, indent=1, default=float))
    (d / "viewer.json").write_text(json.dumps(cams["godot_spawn"], default=float))
    total = {k: int(sum(r["summary"][k] for r in reps.values())) for k in ("n_buildings", "watertight", "outward", "floating")}
    (d / "validation.json").write_text(json.dumps({"city": cid, "size_m": Ld, "method": a.method, "seed": a.seed,
                                                    "total": total, "tiles": reps, "seconds": round(time.time() - t0, 1)},
                                                   indent=1, default=float))
    print(json.dumps({"dir": str(d), **total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
