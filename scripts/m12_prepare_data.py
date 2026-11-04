"""Milestone 12, Phase 5 data: terrain + water arrays and structure labels for every development tile.

    python scripts/m12_prepare_data.py   # -> data/processed/m12/terrain/<sample_id>.npz, .../norm.json

Per tile (train, val, val_t; test splits are never touched), on the 4 m grid of the tile plus a 256 m margin
(256 x 256 pixels, row 0 = north; the same grid as terrain.data.tile_terrain):
* ``elev`` float32 metres (0 where unavailable) and ``valid`` (bool): DEM available - Spain (IGN), US (3DEP). Japan
  has no DEM (GSI licence unresolved), so its tiles train with valid = 0 and only the water channel;
* ``water`` int8 kinds 0 land, 1 river, 2 lake, 3 sea, from OSM (terrain.water);
* ``struct_label`` per edge of the 3D-aware graph (graphs3d.pkl): 0 ground, 1 bridge, 2 tunnel, 3 UNKNOWN = an
  untagged ground edge crossing water (a missing tag is not evidence of a ground-level crossing; such steps are
  excluded from the structure loss).
``norm.json``: scales for the terrain channels from TRAINING tiles only - ``elev_scale`` = std of (z - z at a point
<= 96 m away), ``slope_scale`` = p90 of slope.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import shapely
from rasterio.transform import from_origin

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import active_splits, load_index_m12, test_authorised
from urbanmorph.graphs.levels import BRIDGE, TUNNEL
from urbanmorph.terrain.data import tile_terrain
from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers

OUT = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
MARGIN, RES, N = 256.0, 4.0, 256


def struct_labels(g, water_union):
    st = g.structs()
    lab = np.where(st == BRIDGE, 1, np.where(st == TUNNEL, 2, 0))
    if not water_union.is_empty and len(g.edges):
        segs = shapely.linestrings(g.segments().reshape(-1, 2, 2))
        wet = shapely.intersects(segs, water_union)
        lab[(lab == 0) & wet & (g.levels() == 0)] = 3
    return lab.astype(np.int8)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_m12()
    assert test_authorised() or not idx.split.isin(["test", "test_t"]).any()
    if test_authorised():                       # final test run: test tiles only; dev arrays and norm.json stay frozen
        idx = idx[idx.split.isin(["test", "test_t"])]
        GRAPHS = GRAPHS3D.with_name("graphs3d_test.pkl")
    else:
        GRAPHS = GRAPHS3D
    graphs = pickle.loads(GRAPHS.read_bytes())
    gmap = {sid: g for sp in graphs for sid, _, g in graphs[sp]}
    rel, slopes = [], []
    rng = np.random.default_rng(0)
    for code, sub in idx.groupby("city_code"):
        feats = city_features(code, sub.crs.iloc[0])
        for r in sub.itertuples():
            t = tile_terrain(r.country, r.sample_id, r.crs, r.xmin, r.ymin)
            if t is not None:
                elev, valid = np.nan_to_num(t.elev, nan=0.0), np.isfinite(t.elev)
            else:
                elev, valid = np.zeros((N, N), np.float32), np.zeros((N, N), bool)
            win = tile_window(r.xmin, r.ymin, MARGIN)
            layers = water_layers(feats, win)
            tr = from_origin(r.xmin - MARGIN, r.ymax + MARGIN, RES, RES)
            water = rasterize_kinds(layers, tr, (N, N))
            wu = shapely.affinity.translate(shapely.union_all([layers[1], layers[2]]), -r.xmin, -r.ymin)
            lab = struct_labels(gmap[r.sample_id], wu)
            np.savez_compressed(OUT / f"{r.sample_id}.npz", elev=elev.astype(np.float32), valid=valid, water=water,
                                struct_label=lab)
            if r.split == "train" and valid.all():
                gy, gx = np.gradient(elev, RES)
                slopes.append(np.percentile(np.hypot(gx, gy)[64:192, 64:192], 90))
                a = rng.integers(64, 192, (200, 2))
                b = np.clip(a + rng.integers(-24, 25, (200, 2)), 0, N - 1)
                rel.append(elev[b[:, 0], b[:, 1]] - elev[a[:, 0], a[:, 1]])
        print(code, len(sub), flush=True)
    if test_authorised():
        print("test tiles prepared; norm.json unchanged")
        return 0
    norm = {"elev_scale": float(np.std(np.concatenate(rel))), "slope_scale": float(np.median(slopes)),
            "note": "training tiles with a DEM only; elev_scale = std of elevation differences within 96 m, "
                    "slope_scale = median over tiles of the tile p90 slope"}
    (OUT / "norm.json").write_text(json.dumps(norm, indent=1))
    print(norm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
