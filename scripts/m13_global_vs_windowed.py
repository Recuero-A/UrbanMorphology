"""Milestone 13, experiment X5 (secondary; synthetic terrain = robustness only): global vs windowed generation.

    python scripts/m13_global_vs_windowed.py --done outputs/m13/x5/DONE

Terrain: 1536 m procedural landscapes, scenarios rolling_hills, high_hills, steep_hillside, narrow_river,
mountain_pass, seeds 0-2. Backbone TN_s0. Methods:
* windowed  - the M11 / M12 windowed growth (scripts/m12_big_landscape.grow, legacy loop, policy none, T = 0.5);
* global_A  - the tile-free sampler (graphs/tc_global), standard (state-consistent, unconstrained);
* global_M  - the tile-free sampler with the selected M* configuration (outputs/m13/selection.json).
Metrics per landscape: street density, components (>= 40 m), giant-component share over the whole 1536 m domain,
dead-end ratio, and SEAM metrics - street density and dead-end share as a function of distance to the windowed
generator's window edges (x, y in {384, 512, 768, 896, 1024, 1280}); a tile-free generator has no seams, so a
near / far ratio far from 1 for "windowed" only would show window artefacts.
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import m13_experiments as ex  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.road_graph import RoadGraph, is_border
from urbanmorph.graphs.tc_global import GlobalTCSampler, big_terrain
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset
from urbanmorph.terrain.synthesis import generate, scenario

OUT = PROJECT_ROOT / "outputs" / "m13" / "x5"
L = 1536.0
SCEN = ("rolling_hills", "high_hills", "steep_hillside", "narrow_river", "mountain_pass")
SEAMS = np.array([384.0, 512.0, 768.0, 896.0, 1024.0, 1280.0])
BANDS = np.array([0.0, 16.0, 48.0, 96.0, 1e9])


def metrics(nodes, edges):
    out = {}
    if not len(edges):
        return {"density": 0.0, "components": 0, "giant_share": 0.0, "dead_end_ratio": np.nan}
    seg = np.c_[nodes[edges[:, 0]], nodes[edges[:, 1]]]
    Ls = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    out["density"] = float(Ls.sum() / 1000 / (L / 1000) ** 2)
    n = len(nodes)
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    cl = np.bincount(lab[edges[:, 0]], weights=Ls, minlength=k)
    out["components"] = int((cl >= 40).sum())
    out["giant_share"] = float(cl.max() / Ls.sum())
    deg = np.bincount(edges.ravel(), minlength=n)
    border = is_border(nodes, size=L)
    out["dead_end_ratio"] = float(((deg == 1) & ~border).sum() / max((deg > 0).sum(), 1))
    mid = 0.5 * (seg[:, :2] + seg[:, 2:])
    dseam = np.minimum(np.abs(mid[:, 0, None] - SEAMS[None]).min(1), np.abs(mid[:, 1, None] - SEAMS[None]).min(1))
    dn = np.minimum(np.abs(nodes[:, 0, None] - SEAMS[None]).min(1), np.abs(nodes[:, 1, None] - SEAMS[None]).min(1))
    # band areas (approximate, by sampling the domain)
    g = np.linspace(2, L - 2, 384)
    X, Y = np.meshgrid(g, g)
    dpix = np.minimum(np.abs(X.ravel()[:, None] - SEAMS[None]).min(1), np.abs(Y.ravel()[:, None] - SEAMS[None]).min(1))
    for b in range(len(BANDS) - 1):
        area = ((dpix >= BANDS[b]) & (dpix < BANDS[b + 1])).mean() * (L / 1000) ** 2
        m = (dseam >= BANDS[b]) & (dseam < BANDS[b + 1])
        out[f"density_band{b}"] = float(Ls[m].sum() / 1000 / max(area, 1e-9))
        mn = (dn >= BANDS[b]) & (dn < BANDS[b + 1]) & (deg > 0)
        out[f"deadend_band{b}"] = float(((deg == 1) & ~border & mn).sum() / max(mn.sum(), 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from m12_big_landscape import grow  # noqa: E402
    from m12_evaluate import TH  # noqa: E402
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, names, norm, _ = ex.load_backbone("TN_s0", device)
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    sel = json.loads((PROJECT_ROOT / "outputs" / "m13" / "selection.json").read_text())
    ms = sel["M_star"]
    f = OUT / "results.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for sc in SCEN:
        for sd in range(3):
            big = big_terrain(scenario(sc, sd), L)
            T = generate(scenario(sc, sd), -256.0, -256.0, L + 256.0, L + 256.0)   # same field as ``big``, with water surface
            assert np.array_equal(T.elev.astype(np.float32), big["elev"])
            for meth in ("windowed", "global_A", "global_M"):
                key = f"{sc}{sd}_{meth}"
                if key in res:
                    continue
                t0 = time.time()
                if meth == "windowed":
                    S, C, _ = grow(model, names, norm, "none", TH, T, L, 2050 + sd, 0.5)
                    pts = np.r_[S[:, :2], S[:, 2:]]
                    uniq, inv = np.unique(np.round(pts, 2), axis=0, return_inverse=True)
                    nodes, edges = uniq, np.c_[inv[:len(S)], inv[len(S):]]
                    edges = edges[edges[:, 0] != edges[:, 1]]
                else:
                    cfg = preset("standard") if meth == "global_A" else preset(ms["preset"], **ms["overrides"])
                    g = GlobalTCSampler(model, big, L, names, norm, pots, cfg, device).run(seed=2050 + sd)[0]
                    nodes, edges = g.nodes, g.edges
                res[key] = {"scenario": sc, "seed": sd, "method": meth, "seconds": time.time() - t0, **metrics(nodes, edges)}
                f.write_text(json.dumps(res, indent=1, default=float))
                print(key, {k: round(v, 3) for k, v in res[key].items() if isinstance(v, float)}, flush=True)
    if a.done:
        Path(a.done).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
