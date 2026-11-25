"""Latin Caribbean theme, first full map: a coastal town between the sea and jungle mountains (DESIGN.md §8).

1. terrain: the showcase landscape ``caribbean_coast`` (terrain.synthesis; sea + bay south, beach, coastal plain,
   foothills, mountains north) over the 1536 m domain plus its 256 m margin, 4 m grid;
2. backbone: a waterfront road (malecon) along the 5 m contour and two roads climbing inland (anchors, as the real
   arterials of the KAIST campus maps);
3. streets: the Milestone 17 IMPROVED sampler (F1 junction reuse 15 m + F4 junction-degree prior) on the whole domain;
   default NA-fixed (the most realistic sampler on the campus, outputs/m16/kaist/m17_kaist.md);
4. 3D: scripts/m16_city3d_big.py --theme latin_caribbean (BF parcels and buildings per 512 m tile, terrain, beach
   dressing), Blender at golden hour, Godot screenshots.

    .venv/bin/python themes/latin_caribbean/scripts/coastal_town.py [--method NA-fixed] [--seed 1] [--terrain-seed 1]
    ->  outputs/latin_caribbean/coast/ (terrain, anchors, graphs, map) and outputs/latin_caribbean/cities/coast/coast_<method>_s<seed>_latin/
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

OUT = ROOT / "outputs" / "latin_caribbean" / "coast"
L, MARGIN, RES = 1536.0, 256.0, 4.0
FIX = {"junction_reuse_m": 15.0, "degree_prior": 1.0}
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def terrain(seed: int):
    from urbanmorph.terrain.synthesis import generate, scenario
    T = generate(scenario("caribbean_coast", seed), -MARGIN, -MARGIN, L + MARGIN, L + MARGIN, RES)
    return T


def backbone(T):
    """Malecon along the 5 m contour (smoothed), and two inland roads following the gentlest climb."""
    from scipy import ndimage
    from urbanmorph.graphs.road_graph import RoadGraph, build_graph, is_border
    z = ndimage.gaussian_filter(np.where(T.water_kind == 3, -5.0, T.elev.astype(float)), 3)
    col = lambda x: int(round((x + MARGIN) / RES))
    row = lambda y: int(round((L + MARGIN - y) / RES))
    xs = np.arange(0.0, L + 1, 16.0)
    ys = []
    for x in xs:
        c = col(x)
        prof = z[::-1, c]                                        # south -> north
        k = int(np.argmax(prof >= 5.0))
        ys.append(-MARGIN + k * RES)
    ys = ndimage.uniform_filter1d(np.asarray(ys), 7, mode="nearest") + 6.0
    lines, cls = [np.c_[xs, ys]], [0]
    for x_start in (0.2 * L, 0.68 * L):
        y = float(np.interp(x_start, xs, ys))
        pts, x = [(x_start, y)], x_start
        while y < 0.78 * L:
            best = None
            for dx in (-24.0, -12.0, 0.0, 12.0, 24.0):           # the gentlest of five headings, 40 m steps
                nx, ny = x + dx, y + 40.0
                g = abs(z[row(ny), col(nx)] - z[row(y), col(x)]) / np.hypot(dx, 40.0)
                if best is None or g < best[0]:
                    best = (g, nx, ny)
            _, x, y = best
            pts.append((x, y))
        lines.append(np.asarray(pts))
        cls.append(1)
    g = build_graph(lines, classes=cls)
    return RoadGraph(g.nodes, g.edges, is_border(g.nodes, size=L), edge_class=g.classes(), edge_struct=g.structs())


def generate(method_label: str, seed: int, T, anchor):
    import torch
    from m16_kaist_all import AnchoredGlobalNet
    from m16_kaist_campus import AnchoredGlobal
    from urbanmorph.experiments.backbones import load_backbone
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    from urbanmorph.graphs.tc_sampler import preset
    torch.set_num_threads(4)
    model, names, norm, _ = load_backbone("TN_s0", "cpu")
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method(method_label)
    ov = {**m.overrides, **FIX}
    is_net = method_label.startswith("NA")
    cfg = net_preset(m.preset, **ov) if is_net else preset(m.preset, **ov)
    big = {"elev": T.elev.astype(np.float32), "valid": np.ones(T.elev.shape, bool), "water": T.water_kind.astype(np.int8),
           "site": f"demo_coast_{method_label}"}
    sm = (AnchoredGlobalNet if is_net else AnchoredGlobal)(model, big, L, names, norm, pots, cfg)
    sm.anchor = anchor
    t0 = time.time()
    g = sm.run(seed)[0]
    return g, time.time() - t0, int(sm.stats[0].get("reuse_applied", 0))


def figure(T, anchor, g, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    z = T.elev.astype(float)
    rgb = LightSource(315, 40).shade(np.maximum(z, 0), cmap=plt.cm.gist_earth, vert_exag=1.5, blend_mode="soft", vmin=-40, vmax=330)
    rgb[T.water_kind == 3, :3] = [0.25, 0.6, 0.7]
    rgb[T.water_kind == 1, :3] = [0.2, 0.4, 0.9]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN))
    for gg, col, lw in ((g, "k", 0.9), (anchor, "#d62728", 2.0)):
        for a, b in gg.edges:
            ax.plot(*gg.nodes[[a, b]].T, color=col, lw=lw)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title("Caribbean coast: generated streets (black) on the backbone (red)")
    fig.savefig(path, dpi=90, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="NA-fixed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--terrain-seed", type=int, default=1)
    ap.add_argument("--res", type=int, default=2000)
    ap.add_argument("--skip-3d", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T = terrain(a.terrain_seed)
    tz = OUT / f"terrain_t{a.terrain_seed}.npz"
    np.savez_compressed(tz, elev=T.elev, water=T.water_kind, water_surface=T.water_surface.astype(np.float32))
    anchor = backbone(T)
    key = a.method.replace("*", "star").replace(" ", "")
    gp = OUT / f"graphs_{key}_s{a.seed}_t{a.terrain_seed}.pkl"
    if gp.exists():
        d = pickle.loads(gp.read_bytes())
        g = d["graphs"][(a.method + " [improved]", a.seed)]
    else:
        g, sec, reuse = generate(a.method, a.seed, T, anchor)
        gp.write_bytes(pickle.dumps({"graphs": {(a.method + " [improved]", a.seed): g}, "real": None,
                                     "meta": {"seconds": sec, "reuse_applied": reuse, "terrain_seed": a.terrain_seed,
                                              "fix": FIX, "anchor_edges": int(len(anchor.edges))}}))
        print(f"generated {a.method} seed {a.seed}: {len(g.edges)} edges, {sec:.0f} s", flush=True)
    figure(T, anchor, g, OUT / f"map_{key}_s{a.seed}_t{a.terrain_seed}.png")
    if a.skip_3d:
        return 0
    py = str(ROOT / ".venv" / "bin" / "python")
    r = subprocess.run([py, str(ROOT / "scripts" / "m16_city3d_big.py"), "--method", a.method + " [improved]", "--seed", str(a.seed),
                        "--graphs", str(gp), "--tag", "coast", "--theme", "latin_caribbean", "--terrain-npz", str(tz),
                        "--cities-dir", str(ROOT / "outputs" / "latin_caribbean" / "cities" / "coast")],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return r.returncode
    d = Path(json.loads(r.stdout.strip().splitlines()[-1])["dir"])
    print("3D", d, flush=True)
    rc = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
                         "--", "--tile", str(d), "--res", str(a.res), "--light", "sunset"],
                        stdout=open(d / "blender.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
