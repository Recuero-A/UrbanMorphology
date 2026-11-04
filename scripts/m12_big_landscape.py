"""Milestone 12: a multi-tile terrain-aware city on ONE continuous procedural landscape (spec §21).

    python scripts/m12_big_landscape.py --scenario wide_river --scenario-seed 5 --n 3 --model TN_s0 --mode HR

The landscape (terrain + rivers + lakes + sea) is generated once for the whole region (n x 512 m plus a 256 m
margin), so terrain, rivers and lakes are continuous across tiles. Streets grow window by window exactly as in
scripts/m11_big_city.py (512 m windows, 128 m overlap, existing streets given, dangling ends continued, stubs on
internal window edges pruned); each window's turtle crops and feasibility checks read the shared field. Bridge and
tunnel flags travel with the street segments (class + 4 * structure), so a bridge cut by a window seam stays a
bridge. Parcels and buildings per 512 m tile (BF), 3D per tile on its slice of the same field, meshes merged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m11_big_city import W, clip, edge_lines, merge, on_lines, to_graph, windows  # noqa: E402
from urbanmorph.city3d.convert import from_generated
from urbanmorph.city3d.format import save_city, validate_city
from urbanmorph.city3d.scene import build_scene, save_scene
from urbanmorph.city3d.terrain3d import Surfaces, apply_terrain
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.road_graph import RoadGraph, is_border
from urbanmorph.graphs.terrain_generate import TerrainPolicy, terrain_of
from urbanmorph.terrain.road_feasibility import Thresholds
from urbanmorph.terrain.synthesis import generate, scenario

OUT = PROJECT_ROOT / "outputs" / "m12" / "cities"
MARGIN = 256.0


def window_dict(T, x, y):
    """The 256 x 256 terrain dict of the window whose south-west corner is (x, y) (world metres)."""
    r0 = int(round((T.y1 - (y + W + MARGIN)) / T.res))
    c0 = int(round((x - MARGIN - T.x0) / T.res))
    sl = (slice(r0, r0 + 256), slice(c0, c0 + 256))
    return {"elev": T.elev[sl], "valid": np.ones((256, 256), bool), "water": T.water_kind[sl], "water_surface": T.water_surface[sl]}


def decode(g: RoadGraph) -> RoadGraph:
    code = g.classes()
    st = code // 4
    return RoadGraph(g.nodes, g.edges, g.border, code % 4, (st == 1).astype(int) - (st == 2).astype(int), st)


def grow(model, names, norm, mode, th, T, L, seed, temp):
    segs, code = np.zeros((0, 4)), np.zeros(0, int)
    done, log, stats = [], [], {}
    for k, (x, y) in enumerate(windows(L)):
        box = (x, y, x + W, y + W)
        gs, gc = clip(segs, code, box, True)
        given = decode(to_graph(gs, gc, (x, y)))
        deg = given.degree()
        fr = [i for i in np.flatnonzero(deg == 1) if not is_border(given.nodes[i:i + 1])[0] and on_lines(given.nodes[i] + (x, y), done)]
        pol = TerrainPolicy([window_dict(T, x, y)], names, norm, mode=mode, th=th)
        g = sample_graphs(model, 1, device=model.crop_cnn[0].weight.device, seed=seed + k, temperature=temp,
                          given=[given], frontier=[fr], terrain=pol)[0]
        for kk, v in pol.stats[0].items():
            stats[kk] = stats.get(kk, 0) + v
        out_s, out_c = clip(segs, code, box, False)
        segs = np.r_[out_s, g.segments() + np.array([x, y, x, y])]
        code = np.r_[out_c, g.classes() + 4 * g.structs()]
        log.append({"window": [x, y], "given_edges": int(len(given.edges)), "edges_after": int(len(g.edges))})
        done += edge_lines(x, y, L)
    return segs, code, {"windows": log, "policy": stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="wide_river")
    ap.add_argument("--scenario-seed", type=int, default=5)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="TN_s0")
    ap.add_argument("--mode", default="HR", choices=["none", "R", "HR"])
    ap.add_argument("--seed", type=int, default=2050)
    ap.add_argument("--country", default="united_states")
    a = ap.parse_args()
    from m10_integrate import bf_layouts  # noqa: E402
    from m12_evaluate import load_model  # noqa: E402
    from m6_build_objects import load_objects  # noqa: E402
    t0 = time.time()
    L = a.n * W
    T = generate(scenario(a.scenario, a.scenario_seed), -MARGIN, -MARGIN, L + MARGIN, L + MARGIN)
    th = Thresholds.load(PROJECT_ROOT / "configs" / "m12_feasibility.json")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, names, norm = load_model(a.model, device)
    S, C, slog = grow(model, names, norm, a.mode, th, T, L, a.seed, 0.5)
    cid = f"big_{a.model}_{a.mode}_{a.scenario}{a.scenario_seed}_{a.n}x{a.n}_s{a.seed}"
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    cw = load_objects()["class_widths"]
    priors = json.loads((PROJECT_ROOT / "outputs" / "m11" / "audit" / "height_priors.json").read_text())
    tiles = [(r, c) for r in range(a.n) for c in range(a.n)]
    graphs = [decode(to_graph(*clip(S, C, (c * W, r * W, c * W + W, r * W + W)), (c * W, r * W))) for r, c in tiles]
    lay = bf_layouts(graphs, [a.country] * len(tiles), cw, seed=0)
    parts, reps, zmin, zmax = [], {}, np.inf, -np.inf
    for k, ((r, c), (st, P), g) in enumerate(zip(tiles, lay, graphs)):
        city = from_generated(f"{cid}_r{r}c{c}", st.graph, st, P, cw, a.country, priors,
                              {"value": 3.0, "source": "default"}, a.seed + k,
                              {"streets": f"{a.model} mode {a.mode}, windowed growth", "terrain": f"procedural {a.scenario}",
                               "parcels_buildings": "BF (M10) filtered for water", "heights": "synthetic", "roofs_styles": "synthetic"})
        tt = terrain_of(window_dict(T, c * W, r * W))
        tt.water_surface = window_dict(T, c * W, r * W)["water_surface"]
        from urbanmorph.terrain.road_feasibility import water_kind_at
        import shapely
        city["buildings"] = [b for b in city["buildings"]
                             if not (water_kind_at(tt, *np.asarray(shapely.Polygon(np.asarray(b["footprint"][0])[:, :2]).exterior.coords).T) > 0).any()]
        save_city(city, d / f"city_r{r}c{c}.json")
        groups, mats, rep = build_scene(city)
        groups, mats, trep = apply_terrain(groups, mats, city, tt, g)
        parts.append((c * W, r * W, groups, mats))
        reps[f"r{r}c{c}"] = {"summary": rep["summary"], "terrain3d": trep, "format_errors": len(validate_city(city))}
        zmin, zmax = min(zmin, trep["surround_z"]), max(zmax, float(np.nanmax(tt.elev)))
        print(f"r{r}c{c}", rep["summary"]["n_buildings"], "buildings", trep["bridges"], "bridges", flush=True)
    groups, mats = merge(parts, L)
    save_scene(d / "meshes.npz", groups, mats)
    Sx = Surfaces(terrain_of(window_dict(T, (a.n // 2) * W, (a.n // 2) * W)))
    mid = (a.n // 2) * W + 256.0
    zs = float(Sx.road(np.array([256.0]), np.array([256.0]))[0])
    cams = {"extent_m": L, "surround_z": zmin,
            "aerial": {"type": "ORTHO", "location": [L / 2, L / 2, 1200.0 + zmax], "ortho_scale": L},
            "oblique": {"type": "PERSP", "lens": 30, "location": [L / 2 - 376 * a.n, L / 2 - 416 * a.n, 330 * a.n + zmax],
                        "look_at": [L / 2, L / 2, (zmin + zmax) / 2]},
            "street": {"type": "PERSP", "lens": 22, "location": [mid, mid - 60, zs + 1.7], "look_at": [mid, mid + 100, zs + 1.7]},
            "godot_spawn": {"spawn": [mid, zs + 2.0, -mid], "look_at": [mid, zs + 2.0, -(mid + 50)]}}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1, default=float))
    (d / "viewer.json").write_text(json.dumps(cams["godot_spawn"], default=float))
    keys = ("n_buildings", "watertight", "outward", "floating")
    total = {k: int(sum(r["summary"][k] for r in reps.values())) for k in keys}
    total["bridges"] = int(sum(r["terrain3d"]["bridges"] for r in reps.values()))
    total["piers"] = int(sum(r["terrain3d"]["piers"] for r in reps.values()))
    total["street_segments"] = int(len(S))
    total["street_bridge_segments"] = int((C // 4 == 1).sum())
    (d / "validation.json").write_text(json.dumps({"city": cid, "size_m": L, "scenario": a.scenario, "scenario_seed": a.scenario_seed,
                                                    "total": total, "streets": slog, "tiles": reps,
                                                    "seconds": round(time.time() - t0, 1)}, indent=1, default=float))
    print(json.dumps(total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
