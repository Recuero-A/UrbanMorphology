"""Milestone 16, part 3: campus-wide KAIST pilot (recurring comparison / demo; GLO-30 DSM; docs/M16_PROTOCOL.md).

Domain: the 1536 m square of configs/m16_pack (3 x 3 windows of 512 m, holding the OSM campus polygon with ~100 m to
spare), terrain GLO-30 resampled to 4 m with a 256 m margin, water from OSM. Approaches (style off, M* unless noted):
* ``global``    the existing model over the whole domain (graphs/tc_global.GlobalTCSampler: one state, start head on
                512 m windows in a seeded round robin, streets cross windows);
* ``tiles``     independent 512 m windows (M* and NA-fixed, fixed start at each window centre), merged without joining;
* ``anchored``  a connected backbone of SUPPLIED anchors (the real OSM trunk / primary / secondary / tertiary roads of
                the domain, recorded below) pre-drawn, then the global turtle infill joins and extends it.
Measures: street km, 60 m coverage of the domain, cross-window OD reachability (node pairs in different windows,
>= 300 m apart), seam artefacts (dead ends per km within 24 m of an internal window edge vs elsewhere), runtime;
real references: the training street definition and all vehicular roads (+ service).

    python scripts/m16_kaist_campus.py [--seeds 1 2 3]   ->  outputs/m16/kaist/{runs.json, report.md, graphs.pkl}
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
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from urbanmorph.experiments.backbones import load_backbone
from urbanmorph.experiments.methods import method
from urbanmorph.graphs.m14_sampler import net_preset, sample_net
from urbanmorph.graphs.road_graph import RoadGraph, is_border
from urbanmorph.graphs.tc_global import GlobalTCSampler, window_tiles
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset, sample_tc

L = 1536.0
W = 512.0
OUT = ROOT / "outputs" / "m16" / "kaist"
PACK = ROOT / "data" / "processed" / "m16" / "pack"
ANCHOR_HW = ("trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link")


def domain():
    """Big terrain dict over [-256, L + 256]^2 and the domain origin (UTM SW corner)."""
    from rasterio.transform import from_origin
    from m16_build_terrain_pack import RAW, GLO, SITES, origin, window_terrain
    from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers
    s = [x for x in SITES if x["id"] == "KAIST_CAMPUS"][0]
    x0, y0 = origin(s)
    z, tr, _ = window_terrain(RAW / "glo30" / GLO["N36E127"], s["crs"], x0, y0, size=L)
    feats = city_features("M16_KAIST_CAMPUS", s["crs"])
    layers = water_layers(feats, tile_window(x0, y0, 256.0, size=L))
    n = z.shape[0]
    water = rasterize_kinds(layers, from_origin(x0 - 256, y0 + L + 256, 4.0, 4.0), (n, n))
    return {"elev": np.nan_to_num(z).astype(np.float32), "valid": np.isfinite(z), "water": water}, (x0, y0), s["crs"]


def anchors(x0, y0, crs) -> RoadGraph:
    import geopandas as gpd
    import shapely
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    seq = PACK / "_M16_KAIST_CAMPUS_streets.geojsonseq"
    g = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(crs)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"]) & g.highway.isin(ANCHOR_HW)]
    win = shapely.box(x0, y0, x0 + L, y0 + L)
    lines, cls = [], []
    for geom, hw in zip(g.geometry.intersection(win), g.highway):
        for part in getattr(geom, "geoms", [geom]):
            if part.geom_type == "LineString" and part.length > 1:
                lines.append(np.asarray(part.coords)[:, :2] - [x0, y0])
                cls.append(highway_class(hw))
    if not lines:
        return None
    g = build_graph(lines, classes=cls)
    return RoadGraph(g.nodes, g.edges, is_border(g.nodes, size=L), edge_class=g.classes(), edge_struct=g.structs())


class AnchoredGlobal(GlobalTCSampler):
    """Global generation with a pre-drawn anchor graph (GenState.add_given: nodes never open; a street that joins
    an anchor node queues it, so the infill continues from the backbone)."""
    anchor: RoadGraph | None = None

    def new_state(self):
        s = super().new_state()
        if self.anchor is not None:
            s.add_given(self.anchor)
        return s


def merge(graphs_offsets) -> RoadGraph:
    nodes, edges, cls, st = [], [], [], []
    k = 0
    for g, (dx, dy) in graphs_offsets:
        if g.n == 0:
            continue
        nodes.append(g.nodes + [dx, dy])
        edges.append(g.edges + k)
        cls.append(g.classes())
        st.append(g.structs())
        k += g.n
    if not nodes:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool))
    N = np.vstack(nodes)
    return RoadGraph(N, np.vstack(edges), is_border(N, size=L), edge_class=np.concatenate(cls),
                     edge_struct=np.concatenate(st))


def measures(g: RoadGraph, rng) -> dict:
    out = {"street_km": 0.0, "coverage_60": 0.0, "xwin_unreach": np.nan, "seam_dead_ratio": np.nan, "edges": int(len(g.edges))}
    if len(g.edges) == 0:
        return out
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    le = np.hypot(*(b - a).T)
    out["street_km"] = float(le.sum() / 1000)
    gx = np.arange(10, L, 20.0)
    X, Y = np.meshgrid(gx, gx)
    P = np.c_[X.ravel(), Y.ravel()]
    from scipy.spatial import cKDTree
    pts = np.vstack([a + t * (b - a) for t in np.linspace(0, 1, 6)])
    d, _ = cKDTree(pts).query(P)
    out["coverage_60"] = float(np.mean(d <= 60.0))
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    A = coo_matrix((np.r_[le, le], (np.r_[g.edges[:, 0], g.edges[:, 1]], np.r_[g.edges[:, 1], g.edges[:, 0]])),
                   shape=(g.n, g.n)).tocsr()
    used = np.flatnonzero(deg > 0)
    win = (np.floor(g.nodes[:, 0] / W).clip(0, 2) + 3 * np.floor(g.nodes[:, 1] / W).clip(0, 2)).astype(int)
    i, j = rng.choice(used, 3000), rng.choice(used, 3000)
    ok = (win[i] != win[j]) & (np.hypot(*(g.nodes[i] - g.nodes[j]).T) >= 300)
    i, j = i[ok][:300], j[ok][:300]
    if len(i) >= 20:
        src, inv = np.unique(i, return_inverse=True)
        D = dijkstra(A, indices=src)[inv, j]
        out["xwin_unreach"] = float(np.mean(~np.isfinite(D)))
    dead = (deg == 1) & ~is_border(g.nodes, size=L)
    seam = np.zeros(g.n, bool)
    for c in (W, 2 * W):
        seam |= (np.abs(g.nodes[:, 0] - c) < 24) | (np.abs(g.nodes[:, 1] - c) < 24)
    mid = (a + b) / 2
    seam_e = np.zeros(len(le), bool)
    for c in (W, 2 * W):
        seam_e |= (np.abs(mid[:, 0] - c) < 24) | (np.abs(mid[:, 1] - c) < 24)
    km_s, km_o = le[seam_e].sum() / 1000, le[~seam_e].sum() / 1000
    if km_s > 0 and km_o > 0 and (dead & ~seam).sum() > 0:
        out["seam_dead_ratio"] = float(((dead & seam).sum() / km_s) / ((dead & ~seam).sum() / km_o))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    big, (x0, y0), crs = domain()
    anc = anchors(x0, y0, crs)
    model, names, norm, _ = load_backbone("TN_s0", "cpu")
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    tiles = window_tiles(big, L)
    for k, t in enumerate(tiles):
        t["site"] = f"KAIST_CAMPUS_w{k}"
    offs = [(c * W, r * W) for r in range(3) for c in range(3)]
    mstar = preset("energy", **method("M*").overrides)
    nafix = net_preset("na_fixed", **method("NA-fixed").overrides)
    res, graphs = [], {}
    for seed in a.seeds:
        rng = np.random.default_rng([seed, 16])
        runs = {}
        t0 = time.time()
        runs["global M*"] = (GlobalTCSampler(model, big, L, names, norm, pots, mstar).run(seed)[0], time.time() - t0)
        t0 = time.time()
        sm = AnchoredGlobal(model, big, L, names, norm, pots, mstar)
        sm.anchor = anc
        runs["anchored M*"] = (sm.run(seed)[0], time.time() - t0)
        for lab, cfg, f in (("tiles M*", preset("energy", fixed_start="centre", **method("M*").overrides), sample_tc),
                            ("tiles NA-fixed", net_preset("na_fixed", fixed_start="centre", **method("NA-fixed").overrides),
                             sample_net)):
            t0 = time.time()
            gs, _, _ = f(model, tiles, names, norm, pots, cfg, seed=seed)
            runs[lab] = (merge(list(zip(gs, offs))), time.time() - t0)
        for lab, (g, sec) in runs.items():
            res.append({"approach": lab, "seed": seed, "seconds": sec, **measures(g, rng)})
            graphs[(lab, seed)] = g
            print(res[-1], flush=True)
    refs = {}
    for name, suffix in (("real streets (training definition)", ""), ("real vehicular (+ service)", "__vehicular")):
        gs = [(pickle.loads((PACK / "streets" / f"KAIST_CAMPUS_r{2 - r}c{c}{suffix}.pkl").read_bytes()), off)
              for (c, r), off in zip([(c, r) for r in range(3) for c in range(3)], offs)]
        refs[name] = measures(merge(gs), np.random.default_rng(0))
    (OUT / "graphs.pkl").write_bytes(pickle.dumps({"graphs": graphs, "anchors": anc, "origin": (x0, y0), "crs": crs}))
    json.dump({"runs": res, "refs": refs, "anchor_km": float(sum(np.hypot(*(anc.nodes[anc.edges[:, 0]] - anc.nodes[anc.edges[:, 1]]).T)) / 1000) if anc else 0.0,
               "origin_utm": [x0, y0], "crs": crs}, open(OUT / "runs.json", "w"), indent=1, default=float)
    import pandas as pd
    df = pd.DataFrame(res)
    tab = df.groupby("approach")[["street_km", "coverage_60", "xwin_unreach", "seam_dead_ratio", "seconds"]].mean()
    md = ["# KAIST campus-wide pilot (recurring comparison; GLO-30 DSM; style off)", "",
          f"Domain 1536 m x 1536 m, origin UTM 52N {x0}, {y0}; seeds {a.seeds}. Anchors: {len(anc.edges) if anc else 0} OSM "
          "trunk / primary / secondary / tertiary edges (supplied, not generated).", "",
          "| Approach | street km | 60 m coverage | cross-window OD unreachable | seam dead-end ratio | seconds |",
          "|---|---|---|---|---|---|"]
    for k, r in tab.iterrows():
        md.append(f"| {k} | {r.street_km:.2f} | {r.coverage_60:.3f} | {r.xwin_unreach:.3f} | {r.seam_dead_ratio:.2f} | {r.seconds:.0f} |")
    for k, r in refs.items():
        md.append(f"| {k} | {r['street_km']:.2f} | {r['coverage_60']:.3f} | {r['xwin_unreach']:.3f} | {r['seam_dead_ratio']:.2f} | – |")
    md += ["", "Seam dead-end ratio: dead ends per km within 24 m of an internal window edge / elsewhere (1 = no seam "
           "artefact). Anchored runs include the supplied anchors in their street km."]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
