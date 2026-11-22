"""Latin Caribbean theme on KAIST: a colonial village in the jungle (user request 2026-11-22).

Streets: the Milestone 17 generation "M* + organic [improved]" (F1 junction reuse 15 m + F4 junction-degree prior,
seed 1; outputs/m16/kaist/m17_cache) on the KAIST campus domain, kept only inside a VILLAGE MASK; the real KAIST main
roads (OSM trunk / primary / secondary / tertiary, the anchors of every campus generation, bridges included) are kept
everywhere. Outside the village: no buildings or yards, a closed rainforest canopy (--jungle dense). The Gapcheon and
its tributary stay as water (shown at the DSM water surface).

Village mask (4 m domain grid): a noisy ellipse over the flat campus basin (centre 700, 720; semi-axes 640 x 470 m,
35 deg), slope < 11 deg, no water; opened / closed, parts < 2.5 ha dropped, holes filled; plus the flat land within 70 m
of a main road and 220 m of the village, so the village reaches its roads. Cropping: a generated street is kept when
both ends are in the mask; then isolated pieces shorter than 120 m and dead-end stubs shorter than 10 m are removed.

    .venv/bin/python themes/latin_caribbean/scripts/kaist_village.py
    ->  outputs/latin_caribbean/kaist_village/ (mask, graph, map) and outputs/m16/cities/kaist_village_Mstar_organic_improved_s1_latin/
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import shapely
from scipy import ndimage

OUT = ROOT / "outputs" / "latin_caribbean" / "kaist_village"
LABEL = "M* + organic [improved]"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def village_mask(big, anchor, L):
    from urbanmorph.terrain.synthesis import gradient_noise
    z, w = big["elev"].astype(float), big["water"]
    n = z.shape[0]
    xs = -256 + (np.arange(n) + 0.5) * 4
    ys = L + 256 - (np.arange(n) + 0.5) * 4
    X, Y = np.meshgrid(xs, ys)
    a, cx, cy = np.radians(35), 700.0, 720.0
    u = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    v = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
    r = np.hypot(u / 640, v / 470) + 0.18 * gradient_noise(X / 260, Y / 260, 5) + 0.08 * gradient_noise(X / 90, Y / 90, 6)
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 2), 4.0)
    flat = np.degrees(np.arctan(np.hypot(gx, gy))) < 11
    m = (r < 1) & (w == 0) & flat
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=3), iterations=4)
    lab, k = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    m = ndimage.binary_fill_holes(np.isin(lab, 1 + np.flatnonzero(sizes * 16 > 25000))) & (w == 0)
    lines = shapely.union_all([shapely.LineString(anchor.nodes[e]) for e in anchor.edges])
    d_road = shapely.distance(lines, shapely.points(X.ravel(), Y.ravel())).reshape(X.shape)
    d_village = ndimage.distance_transform_edt(~m) * 4.0
    return m | ((d_road < 70) & (d_village < 220) & flat & (w == 0))


def crop(g, anchor, mask, L):
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    n = mask.shape[0]
    rc = lambda P: (np.clip(((L + 256 - P[:, 1]) / 4).astype(int), 0, n - 1), np.clip(((P[:, 0] + 256) / 4).astype(int), 0, n - 1))
    inside = mask[rc(g.nodes)]
    lines = shapely.union_all([shapely.LineString(anchor.nodes[e]) for e in anchor.edges])
    mid = g.nodes[g.edges].mean(1)
    is_anchor = shapely.distance(lines, shapely.points(mid[:, 0], mid[:, 1])) < 3.0
    keep = is_anchor | inside[g.edges].all(1)
    L_e = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1)
    for _ in range(3):                                      # dead-end stubs < 10 m (not on main roads)
        deg = np.bincount(g.edges[keep].ravel(), minlength=len(g.nodes))
        stub = keep & ~is_anchor & (L_e < 10) & ((deg[g.edges[:, 0]] == 1) | (deg[g.edges[:, 1]] == 1))
        keep &= ~stub
    idx = np.flatnonzero(keep)                              # isolated pieces < 120 m without a main road
    lab = np.arange(len(g.nodes))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for e in idx:
        a, b = find(g.edges[e, 0]), find(g.edges[e, 1])
        if a != b:
            lab[a] = b
    comp = np.array([find(g.edges[e, 0]) for e in idx])
    tot, has_anchor = {}, {}
    for e, c in zip(idx, comp):
        tot[c] = tot.get(c, 0.0) + L_e[e]
        has_anchor[c] = has_anchor.get(c, False) or bool(is_anchor[e])
    keep[idx] = [tot[c] >= 120 or has_anchor[c] for c in comp]
    E = g.edges[keep]
    used, inv = np.unique(E, return_inverse=True)
    N = g.nodes[used]
    return RoadGraph(N, inv.reshape(-1, 2), is_border(N, size=L), edge_class=g.classes()[keep], edge_struct=g.structs()[keep]), \
        {"edges_in": int(len(g.edges)), "edges_kept": int(keep.sum()), "anchor_edges": int(is_anchor.sum()),
         "street_km_in": float(L_e.sum() / 1000), "street_km_kept": float(L_e[keep].sum() / 1000)}


def figure(big, mask, g, anchor, L, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    z, w = big["elev"].astype(float), big["water"]
    rgb = LightSource(315, 40).shade(z, cmap=plt.cm.gist_earth, vert_exag=2, blend_mode="soft")
    rgb[~mask, :3] = rgb[~mask, :3] * 0.4 + np.array([0.12, 0.38, 0.12]) * 0.6
    rgb[w > 0, :3] = [0.2, 0.45, 0.9]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, extent=(-256, L + 256, -256, L + 256))
    for e in g.edges:
        ax.plot(*g.nodes[e].T, color="k", lw=0.5)
    for e in anchor.edges:
        ax.plot(*anchor.nodes[e].T, color="#d62728", lw=1.6)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title("KAIST village: M* + organic (F1 + F4) inside the village (black), real main roads (red), jungle (green)")
    fig.savefig(path, dpi=90, bbox_inches="tight")


def main() -> int:
    from m16_kaist_all import key_of
    from m16_kaist_campus import L, anchors, domain
    OUT.mkdir(parents=True, exist_ok=True)
    big, (x0, y0), crs = domain()
    anchor = anchors(x0, y0, crs)
    mask = village_mask(big, anchor, L)
    np.savez_compressed(OUT / "village_mask.npz", mask=mask)
    org = pickle.loads((ROOT / "outputs" / "m16" / "kaist" / "m17_cache" / f"{key_of('M* + organic')}_s1.pkl").read_bytes())["graph"]
    g, info = crop(org, anchor, mask, L)
    info["village_km2"] = float(mask[64:448, 64:448].sum() * 16 / 1e6)
    (OUT / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    gp = OUT / "graphs.pkl"
    gp.write_bytes(pickle.dumps({"graphs": {(LABEL, 1): g}, "real": None}))
    figure(big, mask, g, anchor, L, OUT / "village_map.png")
    py = str(ROOT / ".venv" / "bin" / "python")
    r = subprocess.run([py, str(ROOT / "scripts" / "m16_city3d_big.py"), "--method", LABEL, "--seed", "1", "--graphs", str(gp),
                        "--tag", "kaist_village", "--theme", "latin_caribbean", "--village-npz", str(OUT / "village_mask.npz"),
                        "--jungle", "dense"], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        return r.returncode
    d = Path(json.loads(r.stdout.strip().splitlines()[-1])["dir"])
    print("3D", d, flush=True)
    rc = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT / "blender" / "m11_scene.py"),
                         "--", "--tile", str(d), "--res", "2000", "--light", "sunset"],
                        stdout=open(d / "blender.log", "w"), stderr=subprocess.STDOUT).returncode
    print("blender", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
