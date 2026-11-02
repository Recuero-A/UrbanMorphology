"""Milestone 11 (exploratory): an n x n tile city (default 3 x 3 = 1,536 m) generated from scratch, then built in 3D.

    python scripts/m11_big_city.py --seed 2033 --temp 0.5 --country spain        # -> outputs/m11/big_N_T05_s2033_3x3_spain/

1. Streets: GraphTurtle N grows the city window by window (512 m windows, 128 m overlap, row by row from the
   south-west). Each window starts with the streets already generated inside it pre-drawn (``given``), and the
   street ends cut by an earlier window's edge are queued (``frontier``), so streets continue across seams.
   Ends left on a window edge inside the city that no later window continued are pruned back to their junction.
   The model was trained on single 512 m tiles: windows are its native size, the stitching is ours.
2. Parcels and buildings: BF (M10) on each of the n x n 512 m tiles of the final street graph (BF works on tile
   rasters), so blocks are cut at tile seams.
3. 3D: city3d on every tile, meshes merged with offsets (one material table), one camera set for the whole city.
Writes meshes.npz, cameras.json, viewer.json, validation.json, streets.json and plan.png in the city folder.
Nothing reads validation or test tiles.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from urbanmorph.city3d.cameras import cameras
from urbanmorph.city3d.convert import from_generated
from urbanmorph.city3d.format import polygon, save_city, validate_city
from urbanmorph.city3d.scene import build_scene, corner_uvs, save_scene
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.road_graph import RoadGraph, is_border

W = 512.0
OUT = PROJECT_ROOT / "outputs" / "m11"
SNAP = 2           # node identity: coordinates rounded to 1 cm


def clip_t(a, b, x0, y0, x1, y1):
    """Liang-Barsky parameters [t0, t1] of segment a->b inside the box, or None."""
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return (t0, t1) if t1 - t0 > 1e-9 else None


def clip(segs, cls, box, inside=True):
    """Pieces of the segments inside (or outside) box = (x0, y0, x1, y1)."""
    out, oc = [], []
    for s, c in zip(segs, cls):
        a, b = s[:2], s[2:]
        t = clip_t(a, b, *box)
        if inside:
            spans = [t] if t else []
        else:
            spans = [(0.0, 1.0)] if t is None else [(0.0, t[0]), (t[1], 1.0)]
        for t0, t1 in spans:
            if t1 - t0 < 1e-9:
                continue
            p, q = a + t0 * (b - a), a + t1 * (b - a)
            if np.hypot(*(q - p)) > 0.05:
                out.append(np.r_[p, q])
                oc.append(c)
    return np.array(out).reshape(-1, 4), np.array(oc, int)


def to_graph(segs, cls, origin=(0.0, 0.0)) -> RoadGraph:
    """Planar graph of segments (nodes = endpoints rounded to 1 cm), in coordinates relative to origin."""
    P = np.round(np.r_[segs[:, :2], segs[:, 2:]] - np.asarray(origin), SNAP)
    nodes, inv = np.unique(P, axis=0, return_inverse=True)
    m = len(segs)
    e = np.c_[inv[:m], inv[m:]]
    keep = e[:, 0] != e[:, 1]
    e, c = np.sort(e[keep], 1), cls[keep]
    e, first = np.unique(e, axis=0, return_index=True)
    return RoadGraph(nodes.astype(float), e, is_border(nodes) if len(nodes) else np.zeros(0, bool), c[first])


def windows(L):
    xs = sorted({min(x, L - W) for x in np.arange(0, L - W + 1e-6 + 384, 384.0)})
    return [(x, y) for y in xs for x in xs]


def on_lines(p, lines, tol=0.05):
    """p lies on one of the axis-aligned window edge segments (x0, y0, x1, y1)."""
    for x0, y0, x1, y1 in lines:
        if abs(x0 - x1) < 1e-9 and abs(p[0] - x0) < tol and y0 - tol <= p[1] <= y1 + tol:
            return True
        if abs(y0 - y1) < 1e-9 and abs(p[1] - y0) < tol and x0 - tol <= p[0] <= x1 + tol:
            return True
    return False


def edge_lines(x, y, L):
    """The 4 edges of window (x, y) minus the parts on the city's outer border."""
    out = []
    for e in ((x, y, x, y + W), (x + W, y, x + W, y + W), (x, y, x + W, y), (x, y + W, x + W, y + W)):
        vertical = e[0] == e[2]
        c = e[0] if vertical else e[1]
        if 1e-6 < c < L - 1e-6:
            out.append(e)
    return out


def grow_streets(model, L, seed, temp):
    from urbanmorph.graphs.generate import sample_graphs
    segs, cls = np.zeros((0, 4)), np.zeros(0, int)
    done_lines, log = [], []
    for k, (x, y) in enumerate(windows(L)):
        box = (x, y, x + W, y + W)
        gs, gc = clip(segs, cls, box, True)
        given = to_graph(gs, gc, (x, y))
        deg = given.degree()
        fr = [i for i in np.flatnonzero(deg == 1)
              if not is_border(given.nodes[i:i + 1])[0] and on_lines(given.nodes[i] + (x, y), done_lines)]
        g = sample_graphs(model, 1, seed=seed + k, temperature=temp, given=[given], frontier=[fr])[0]
        out_s, out_c = clip(segs, cls, box, False)
        new = g.segments() + np.array([x, y, x, y])
        segs, cls = np.r_[out_s, new], np.r_[out_c, g.classes()]
        log.append({"window": [x, y], "given_edges": int(len(given.edges)), "frontier": len(fr),
                    "edges_after": int(len(g.edges)), "added": int(len(g.edges) - len(given.edges))})
        done_lines += edge_lines(x, y, L)
    # prune street ends left on an internal window edge (cut by a window, never continued)
    G = to_graph(segs, cls)
    alive = np.ones(len(G.edges), bool)
    adj = [[] for _ in range(G.n)]
    for j, (a, b) in enumerate(G.edges):
        adj[a].append(j)
        adj[b].append(j)
    deg = G.degree().copy()
    pruned = 0
    for v in np.flatnonzero(deg == 1):
        if not on_lines(G.nodes[v], done_lines):
            continue
        pruned += 1
        u = v
        while deg[u] == 1:
            j = next(j for j in adj[u] if alive[j])
            alive[j] = False
            a, b = G.edges[j]
            w = b if a == u else a
            deg[u] -= 1
            deg[w] -= 1
            u = w
            if deg[u] != 1:
                break
    S = G.segments()[alive]
    return S, G.classes()[alive], {"windows": log, "stubs_pruned": pruned, "edges_pruned": int((~alive).sum())}


def merge(parts, L):
    """parts: [(dx, dy, groups, materials)] -> one groups dict + one material list."""
    names, params, merged = [], [], {}
    for dx, dy, groups, mats in parts:
        remap = []
        for p in mats:
            if p["name"] not in names:
                names.append(p["name"])
                params.append(p)
            remap.append(names.index(p["name"]))
        remap = np.array(remap, np.int16)
        for g, a in groups.items():
            if g.startswith("inst_"):
                T = a["T"].copy()
                T[:, 0] += dx
                T[:, 1] += dy
                merged.setdefault(g, []).append({"T": T})
                continue
            if g.startswith("proto_"):
                if g not in merged:
                    merged[g] = [{**a, "mat": remap[a["mat"]]}]
                continue
            V = a["V"] + np.array([dx, dy, 0], np.float32)
            merged.setdefault(g, []).append({"V": V, "F": a["F"], "mat": remap[a["mat"]] if len(a["mat"]) else a["mat"],
                                             "obj": a["obj"]})
    out = {}
    for g, lst in merged.items():
        if g.startswith("inst_"):
            out[g] = {"T": np.vstack([x["T"] for x in lst])}
        elif g.startswith("proto_"):
            out[g] = lst[0]
        else:
            n, F = 0, []
            for x in lst:
                F.append(x["F"] + n)
                n += len(x["V"])
            V = np.vstack([x["V"] for x in lst])
            Fa = np.vstack(F).astype(np.int32)
            out[g] = {"V": V, "F": Fa, "mat": np.concatenate([x["mat"] for x in lst]).astype(np.int16),
                      "obj": np.concatenate([x["obj"] for x in lst]), "UV": corner_uvs(V.astype(float), Fa)}
    return out, params


def plan(cities, n, L, S, C, path, title):
    fig, ax = plt.subplots(figsize=(13, 13))
    col = {0: "#c0392b", 1: "#e67e22", 2: "#2c3e50", 3: "#95a5a6"}
    for (r, c), city in cities.items():
        dx, dy = c * W, r * W
        for lu in city["landuse"]:
            p = polygon(lu["polygon"])
            ax.fill(*(np.asarray(p.exterior.coords) + (dx, dy)).T, lw=0,
                    color={"public_green": "#8fbf6a", "private_green": "#c5e0a5"}.get(lu["class"], "#f1ece1"))
        for rs in city["road_surface"]:
            p = polygon(rs)
            ax.fill(*(np.asarray(p.exterior.coords) + (dx, dy)).T, color="#d5d5d5", lw=0)
        for p in city["parcels"]:
            q = polygon(p["polygon"])
            ax.plot(*(np.asarray(q.exterior.coords) + (dx, dy)).T, color="#7f8c8d", lw=0.25)
        for b in city["buildings"]:
            q = polygon(b["footprint"])
            ax.fill(*(np.asarray(q.exterior.coords) + (dx, dy)).T, color="#5d6d7e", lw=0)
    for s, k in zip(S, C):
        ax.plot(s[[0, 2]], s[[1, 3]], color=col.get(int(k), "#2c3e50"), lw=0.7)
    for i in range(1, n):
        ax.axvline(i * W, color="#2980b9", ls="--", lw=0.8)
        ax.axhline(i * W, color="#2980b9", ls="--", lw=0.8)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("metres. Green: public green; light green: private green; beige: other open land; grey: roads; blue dashes: 512 m tile seams")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2033)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--country", default="spain", choices=["spain", "united_states"])
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    from m6_build_objects import load_objects  # noqa: E402
    from m10_eval_streets import CK, load_model  # noqa: E402
    from m10_integrate import bf_layouts  # noqa: E402
    t0 = time.time()
    L = a.n * W
    tag = "T" + ("%.2f" % a.temp).rstrip("0").rstrip(".").replace(".", "")
    cid = f"big_N_{tag}_s{a.seed}_{a.n}x{a.n}_{a.country}"
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    model = load_model(CK / "m10" / "N_noise_s0")
    S, C, slog = grow_streets(model, L, a.seed, a.temp)
    t_streets = time.time() - t0
    print("streets", len(S), "segments;", slog["stubs_pruned"], "stubs pruned;", round(t_streets, 1), "s", flush=True)
    cw = load_objects()["class_widths"]
    priors = json.loads((OUT / "audit" / "height_priors.json").read_text())
    tiles = [(r, c) for r in range(a.n) for c in range(a.n)]
    graphs = [to_graph(*clip(S, C, (c * W, r * W, c * W + W, r * W + W)), (c * W, r * W)) for r, c in tiles]
    lay = bf_layouts(graphs, [a.country] * len(tiles), cw, seed=0)
    t_bf = time.time() - t0 - t_streets
    sh = {"value": 3.0, "source": "default 3.0 m (generated tile: no city; country-level floor priors)"}
    cities, parts, reps = {}, [], {}
    for k, ((r, c), (st, P)) in enumerate(zip(tiles, lay)):
        tid = f"{cid}_r{r}c{c}"
        city = from_generated(tid, st.graph, st, P, cw, a.country, priors, sh, a.seed + k,
                              {"streets": f"GraphTurtle N_noise_s0, T={a.temp:g}, windowed growth seed {a.seed} "
                                          f"(scripts/m11_big_city.py), tile r{r} c{c} of {a.n}x{a.n}",
                               "parcels_buildings": f"BF (M10), parcel net m8 s0, block nets m9 s0, country {a.country}",
                               "heights": f"synthetic (city3d.heights, {a.country} floor priors from train tiles)",
                               "roofs_styles": "synthetic (city3d.style)"})
        city["metadata"]["offset_m"] = [c * W, r * W]
        err = validate_city(city)
        save_city(city, d / f"city_r{r}c{c}.json")
        groups, mats, rep = build_scene(city)
        rep.pop("buildings")
        rep["format_errors"] = len(err)
        cities[(r, c)] = city
        parts.append((c * W, r * W, groups, mats))
        reps[f"r{r}c{c}"] = rep
        print(tid, rep["summary"]["n_buildings"], "buildings", flush=True)
    t_3d = time.time() - t0 - t_streets - t_bf
    groups, mats = merge(parts, L)
    save_scene(d / "meshes.npz", groups, mats)
    mid = a.n // 2
    cc = cameras(cities[(mid, mid)])
    off = np.array([mid * W, mid * W, 0.0])
    cams = {"extent_m": L,
            "aerial": {"type": "ORTHO", "location": [L / 2, L / 2, 1200.0], "ortho_scale": L},
            "oblique": {"type": "PERSP", "lens": 30, "location": [L / 2 - 376 * a.n, L / 2 - 416 * a.n, 330 * a.n],
                        "look_at": [L / 2, L / 2, 0]}}
    for v in ("street", "closeup"):
        if v in cc:
            cams[v] = {**cc[v], "location": list(np.asarray(cc[v]["location"]) + off), "look_at": list(np.asarray(cc[v]["look_at"]) + off)}
    sp = cc["godot_spawn"]
    cams["godot_spawn"] = {**sp, "spawn": [sp["spawn"][0] + off[0], sp["spawn"][1], sp["spawn"][2] - off[1]],
                           "look_at": [sp["look_at"][0] + off[0], sp["look_at"][1], sp["look_at"][2] - off[1]]}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1, default=float))
    (d / "viewer.json").write_text(json.dumps(cams["godot_spawn"], default=float))
    keys = ("n_buildings", "watertight", "outward", "floating", "missing_roof", "footprint_overlap_pairs_gt_0.5m2",
            "buildings_on_road_gt_1m2")
    total = {k: int(sum(r["summary"][k] for r in reps.values())) for k in keys}
    total["doors"] = int(sum(r["facade"]["doors"] for r in reps.values()))
    total["openings_outside_wall"] = int(sum(r["facade"]["openings_outside_wall"] for r in reps.values()))
    total["format_errors"] = int(sum(r["format_errors"] for r in reps.values()))
    total["triangles"] = int(sum(len(g["F"]) for g in groups.values() if "F" in g))
    per_tile = {t: {"buildings": r["summary"]["n_buildings"], "street_edges": int(len(g.edges)),
                    "parcels": len(cities[rc]["parcels"])}
                for (t, r), g, rc in zip(reps.items(), graphs, tiles)}
    timing = {"streets_s": round(t_streets, 1), "bf_s": round(t_bf, 1), "city3d_s": round(t_3d, 1),
              "total_s": round(time.time() - t0, 1)}
    (d / "streets.json").write_text(json.dumps({**slog, "segments": int(len(S))}, indent=1))
    (d / "validation.json").write_text(json.dumps({"city": cid, "size_m": L, "temperature": a.temp, "seed": a.seed,
                                                    "total": total, "per_tile": per_tile, "timing": timing,
                                                    "tiles": {t: {k: r[k] for k in ("summary", "facade")} for t, r in reps.items()}},
                                                   indent=1, default=float))
    plan(cities, a.n, L, S, C, d / "plan.png",
         f"{cid}: {total['n_buildings']} buildings, {len(S)} street segments (GraphTurtle N, T={a.temp:g}; BF parcels per tile)")
    print(json.dumps({"total": total, "per_tile": per_tile, "timing": timing}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
