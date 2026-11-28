"""Milestone 20, D1: the training data of the three retrained backbones TNg-32, TNg-H and TNg-R (docs/M20_LOG.md D1,
"Training data and recipes"; the implementation note "D1-impl" records what this builds).

    python scripts/m20_tng_data.py m12               FABDEM inputs of the existing development tiles + norm.json
    python scripts/m20_tng_data.py towns --set H     training windows of the train-H settlements (data/processed/m20/towns)
    python scripts/m20_tng_data.py towns --set R     training windows of the flat TNg-R settlements (scripts/m20_flat.py)
    python scripts/m20_tng_data.py report            street-km per slope class, windows dropped, region shares, weights

One input recipe for every backbone (D1: "FABDEM, 32 m slope channel"):
* FABDEM V1-2 (the tiles of m20_walk.fab_tile) is averaged (rasterio Resampling.average, as m20_walk.terrain_one and
  m20_towns' terrain32) onto a 32 m grid of the tile's own CRS covering the tile plus its 256 m margin (32 x 32 cells;
  cell edges on the tile's 4 m grid), and that 32 m surface is interpolated bilinearly (cell centres, edge cells held)
  onto the unchanged 4 m model grid (256 x 256, row 0 = north). The model's own slope channel
  (terrain_sequence.tile_layers, unchanged) is then the slope of the 32 m surface. ``valid`` = the nearest 32 m cell
  has FABDEM data. Invalid cells are filled from the nearest valid cell before interpolation (and zeroed after).
* Existing tiles: the 525 development tiles (train, val, val_t) that have a DEM in M12 (Spain, US) — exactly the
  tiles of the M18 GLO-30 build (m18_build_glo30_terrain.select_tiles; the test splits never enter). ``water`` and
  ``struct_label`` are copied from the M12 arrays. Japanese tiles (no DEM in M12) keep their M12 arrays (valid = 0),
  as in TNg.
* Town windows (train-H, TNg-R): the kept 512 m windows of the town build (m20_towns ``real`` stage: >= 70 % inside the
  polygon or city core, FABDEM and GLO-30 valid), their ``graph`` (road_graph.build_graph, 30 m / 8 m, 3D-aware: the
  training build; OSM STREET_VALUES = vehicular training classes + links), their 4 m OSM water layer, and the 32 m
  FABDEM of the town (real.pkl terrain32, the D3 suite's grid; the window + margin is exactly 32 x 32 of its cells).
  Windows without a vehicular edge are dropped (D1) and counted. ``struct_label`` follows m12_prepare_data.struct_labels
  (bridge 1, tunnel 2, else 0; 3 = an untagged ground-level edge crossing a river or lake), with the crossing tested on
  the 4 m water raster (kinds 1-2) at 9 points per edge instead of the water polygons (declared approximation).
* ``cls32``: the D3 slope class (0-5 / 5-10 / 10-20 / 20-30 / > 30 %, FABDEM 32 m, |np.gradient| as
  heightfield.Terrain.slope) of every 32 m cell of tile + margin; -1 where there is no DEM. It is only used for the
  loss weights and the reports, never as a model input.
* norm.json: the M12 normalisation recipe (m18_build_glo30_terrain.norm_recipe, verbatim) on the FABDEM training tiles;
  one normalisation for all three backbones (the input recipe is identical across them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

OUT = ROOT / "data" / "processed" / "m20" / "tng"
M12 = ROOT / "data" / "processed" / "m12" / "terrain"
GRAPHS3D = ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TOWNS = ROOT / "data" / "processed" / "m20" / "towns"
WALK = ROOT / "data" / "processed" / "m20" / "walk"
REP = ROOT / "outputs" / "m20" / "tng"
MARGIN, RES, N, TILE, FRES = 256.0, 4.0, 256, 512.0, 32.0
K32 = int((TILE + 2 * MARGIN) / FRES)                      # 32 cells of 32 m: tile + margin
EDGES = (5.0, 10.0, 20.0, 30.0)                            # D3 slope classes (%): 0-5, 5-10, 10-20, 20-30, > 30
CLASS_NAMES = ("0-5", "5-10", "10-20", "20-30", ">30")
STEEP_MIN_CLASS = 3                                        # "steep" street-km = land of 20 % and more (D1 framing)
FAB_NOTE = ("FABDEM V1-2 (Hawker et al. 2022), bare earth, CC BY-NC-SA 4.0 (research use only), averaged to 32 m "
            "and interpolated bilinearly onto the 4 m model grid")


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slope_class(slope_pct: np.ndarray) -> np.ndarray:
    c = np.digitize(slope_pct, EDGES).astype(np.int8)
    c[~np.isfinite(slope_pct)] = -1
    return c


def classes32(z32: np.ndarray) -> np.ndarray:
    """D3 class of every 32 m cell (NaN = no DEM -> -1); |np.gradient| on the 32 m grid, as Terrain.slope."""
    gy, gx = np.gradient(z32.astype(float), FRES)
    return slope_class(100.0 * np.hypot(gx, gy))


def upsample(z32: np.ndarray):
    """(elev float32 0 where invalid, valid bool) on the 256 x 256 4 m grid from the 32 x 32 32 m grid of the same
    extent: bilinear between cell centres (edge cells held), valid = nearest cell has data."""
    from scipy import ndimage
    assert z32.shape == (K32, K32), z32.shape
    ok = np.isfinite(z32)
    if not ok.any():
        return np.zeros((N, N), np.float32), np.zeros((N, N), bool)
    if not ok.all():
        _, (ri, ci) = ndimage.distance_transform_edt(~ok, return_indices=True)
        z32 = z32[ri, ci]
    k = N // K32
    u = (np.arange(N) + 0.5) / k - 0.5                       # 4 m pixel centre in 32 m cell-centre units
    R, C = np.meshgrid(np.clip(u, 0, K32 - 1), np.clip(u, 0, K32 - 1), indexing="ij")
    z = ndimage.map_coordinates(z32.astype(np.float64), [R, C], order=1, mode="nearest")
    valid = np.repeat(np.repeat(ok, k, 0), k, 1)
    return np.where(valid, z, 0.0).astype(np.float32), valid


def node_class(cls32: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Class of the 32 m cell under each tile-frame position (x east, y north, 0..512; margin grid row 0 = north)."""
    if not len(pos):
        return np.zeros(0, np.int8)
    col = np.clip(np.floor((pos[:, 0] + MARGIN) / FRES).astype(int), 0, K32 - 1)
    row = np.clip(np.floor((TILE + MARGIN - pos[:, 1]) / FRES).astype(int), 0, K32 - 1)
    return cls32[row, col]


def edge_km_by_class(g, cls32) -> np.ndarray:
    """Street km per class (index 0-4; 5 = no DEM) by the class under each edge's midpoint."""
    out = np.zeros(6)
    if not len(g.edges):
        return out
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    L = np.hypot(*(a - b).T) / 1000.0
    c = node_class(cls32, (a + b) / 2).astype(int)
    np.add.at(out, np.where(c < 0, 5, c), L)
    return out


# ---------------------------------------------------------------- existing tiles
def stage_m12() -> int:
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    import m18_build_glo30_terrain as G
    from m20_walk import fab_tile
    tiles = G.select_tiles()                                     # 525 dev tiles with an M12 DEM; test refused
    assert len(tiles) == 525, len(tiles)
    d = OUT / "terrain_m12"
    d.mkdir(parents=True, exist_ok=True)
    per_tile, used_fab = [], set()
    for code, sub in tiles.groupby("city_code"):
        boxes = [G.lonlat_box(r, MARGIN + 600.0) for r in sub.itertuples()]
        b = (min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes))
        fts = [fab_tile(la, lo) for la in range(math.floor(b[1]), math.floor(b[3]) + 1)
               for lo in range(math.floor(b[0]), math.floor(b[2]) + 1)]
        fts = [t for t in fts if t is not None]
        used_fab |= set(fts)
        srcs = [rasterio.open(t) for t in fts]
        dem, dtr = merge(srcs, bounds=b, nodata=np.nan, dtype="float32")
        nod = srcs[0].nodata
        for s in srcs:
            s.close()
        dem = dem[0]
        if nod is not None:
            dem[dem == nod] = np.nan
        for r in sub.itertuples():
            f = d / f"{r.sample_id}.npz"
            if f.exists():
                z = np.load(f)
                per_tile.append({"sample_id": r.sample_id, "city_code": code, "split": r.split,
                                 "valid_share": float(z["valid"].mean())})
                continue
            z32 = np.full((K32, K32), np.nan, np.float32)
            reproject(dem, z32, src_transform=dtr, src_crs="EPSG:4326", src_nodata=np.nan,
                      dst_transform=from_origin(r.xmin - MARGIN, r.ymin + TILE + MARGIN, FRES, FRES), dst_crs=r.crs,
                      dst_nodata=np.nan, resampling=Resampling.average)
            elev, valid = upsample(z32)
            lid = dict(np.load(M12 / f"{r.sample_id}.npz"))
            np.savez_compressed(f, elev=elev, valid=valid, water=lid["water"], struct_label=lid["struct_label"],
                                cls32=classes32(z32), z32=z32)
            per_tile.append({"sample_id": r.sample_id, "city_code": code, "split": r.split,
                             "valid_share": float(valid.mean())})
        print(code, len(sub), "tiles;", [t.name for t in fts], flush=True)

    def load(s):
        z = np.load(d / f"{s}.npz")
        return z["elev"], z["valid"]
    gn = G.norm_recipe(tiles, load)
    norm = {"elev_scale": gn["elev_scale"], "slope_scale": gn["slope_scale"],
            "note": f"FABDEM-32 m training tiles with a full DEM only ({gn['n_tiles']}); recipe of "
                    "scripts/m12_prepare_data.py via m18_build_glo30_terrain.norm_recipe (same seed and tile order); "
                    "one normalisation for TNg-32, TNg-H and TNg-R"}
    (OUT / "norm.json").write_text(json.dumps(norm, indent=1))
    REP.mkdir(parents=True, exist_ok=True)
    man = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "script": "scripts/m20_tng_data.py m12",
           "source": FAB_NOTE, "fabdem_tiles": {t.name: sha_file(t) for t in sorted(used_fab)}, "norm": norm,
           "n_tiles": len(per_tile), "tiles": per_tile,
           "split_guard": "m18_build_glo30_terrain.select_tiles (load_index_m12(include_test=False); dev splits only)"}
    (REP / "manifest_m12.json").write_text(json.dumps(man, indent=1))
    vs = np.array([t["valid_share"] for t in per_tile])
    print("norm", norm, "| valid share min", vs.min(), "| tiles not fully valid", int((vs < 1).sum()))
    return 0


# ---------------------------------------------------------------- town windows
def struct_label_from_raster(g, water) -> np.ndarray:
    from urbanmorph.graphs.levels import BRIDGE, TUNNEL
    st = g.structs()
    lab = np.where(st == BRIDGE, 1, np.where(st == TUNNEL, 2, 0))
    if len(g.edges):
        a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
        t = np.linspace(0, 1, 9)[None, :, None]
        p = a[:, None] + t * (b - a)[:, None]
        col = np.clip(np.floor(p[..., 0] / RES).astype(int) + 64, 0, N - 1)
        row = np.clip(np.floor((TILE - p[..., 1]) / RES).astype(int) + 64, 0, N - 1)
        wet = np.isin(water[row, col], (1, 2)).any(1)
        lab[(lab == 0) & wet & (g.levels() == 0)] = 3
    return lab.astype(np.int8)


def town_ids(which: str) -> list[str]:
    import pandas as pd
    f = WALK / ("train_accepted.csv" if which == "H" else "train_flat_selected.csv")
    return list(pd.read_csv(f).id)


def stage_towns(which: str) -> int:
    import pandas as pd
    from m20_towns import guard
    ids = town_ids(which)
    guard(ids, "train")                                          # train list only; never hold-H / study-H
    d = OUT / "towns"
    d.mkdir(parents=True, exist_ok=True)
    items, rows = [], []
    for sid in ids:
        real = pickle.loads((TOWNS / sid / "real.pkl").read_bytes())
        assert real["split"] == "train" and not real["scratch"], (sid, real["split"])
        t32 = real["terrain32"]
        Z = np.asarray(t32["elev"], float)
        for w in real["windows"]:
            if not w["keep"]:
                continue
            g = w["graph"]
            R0, C0 = w["fab_rc"]
            r0, c0 = R0 - 8, C0 - 8                               # window + 256 m margin = 32 x 32 cells
            if r0 < 0 or c0 < 0 or r0 + K32 > Z.shape[0] or c0 + K32 > Z.shape[1]:
                z32 = np.full((K32, K32), np.nan)
                rr, cc = np.arange(r0, r0 + K32), np.arange(c0, c0 + K32)
                okr, okc = (rr >= 0) & (rr < Z.shape[0]), (cc >= 0) & (cc < Z.shape[1])
                z32[np.ix_(okr, okc)] = Z[np.ix_(rr[okr], cc[okc])]
            else:
                z32 = Z[r0:r0 + K32, c0:c0 + K32]
            z32 = z32.astype(np.float32)
            cls = classes32(z32)
            km_c = edge_km_by_class(g, cls)
            rec = {"set": which, "town": sid, "window": w["id"], "region": None, "n_edges": int(len(g.edges)),
                   "street_km": float(km_c.sum()), **{f"km_{n}": float(v) for n, v in zip(CLASS_NAMES + ("nodem",), km_c)},
                   "cover": w["cover"]}
            if not len(g.edges):
                rec["dropped"] = "no_vehicular_edge"
                rows.append(rec)
                continue
            rec["dropped"] = ""
            gt = np.load(ROOT / w["tile_npz"])
            elev, valid = upsample(z32)
            lab = struct_label_from_raster(g, gt["water"])
            np.savez_compressed(d / f"{w['id']}.npz", elev=elev, valid=valid, water=gt["water"], struct_label=lab,
                                cls32=cls, z32=z32)
            items.append((w["id"], sid, g))
            rows.append(rec)
        print(sid, sum(1 for x in rows if x["town"] == sid and not x["dropped"]), "windows kept,",
              sum(1 for x in rows if x["town"] == sid and x["dropped"]), "dropped", flush=True)
    A = pd.read_csv(WALK / ("train_accepted.csv" if which == "H" else "train_flat_selected.csv"))
    reg = dict(zip(A.id, A.region))
    for r in rows:
        r["region"] = reg[r["town"]]
    (OUT / f"towns_{which}.pkl").write_bytes(pickle.dumps(items))
    REP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REP / f"windows_{which}.csv", index=False)
    print(which, len(items), "windows;", sum(1 for r in rows if r["dropped"]), "dropped (no vehicular edge)")
    return 0


# ---------------------------------------------------------------- report
def m12_rows():
    graphs = pickle.loads(GRAPHS3D.read_bytes())
    out = []
    for sid, _, g in graphs["train"]:
        f = OUT / "terrain_m12" / f"{sid}.npz"
        cls = np.load(f)["cls32"] if f.exists() else np.full((K32, K32), -1, np.int8)
        km_c = edge_km_by_class(g, cls)
        cc = sid.split("_")
        out.append({"set": "M12", "town": "_".join(cc[:2]) if len(cc) > 2 else sid, "window": sid,
                    "region": {"ESP": "R1", "USA": "R5", "JPN": "R3"}.get(sid[:3], "?"), "n_edges": int(len(g.edges)),
                    "street_km": float(km_c.sum()), **{f"km_{n}": float(v) for n, v in zip(CLASS_NAMES + ("nodem",), km_c)},
                    "dropped": ""})
    return out


def stage_report() -> int:
    import pandas as pd
    D = pd.DataFrame(m12_rows())
    for w in ("H", "R"):
        f = REP / f"windows_{w}.csv"
        if f.exists():
            D = pd.concat([D, pd.read_csv(f).fillna({"dropped": ""})])
    cols = [f"km_{n}" for n in CLASS_NAMES + ("nodem",)]
    K = D[D.dropped == ""]
    print("windows dropped (no vehicular edge):", D[D.dropped != ""].groupby("set").size().to_dict())
    print("street-km per slope class (kept):")
    print(K.groupby("set")[cols + ["street_km"]].sum().round(1).to_string())
    print(K.groupby("set").size().rename("windows").to_string())
    steep = [f"km_{n}" for n in CLASS_NAMES[STEEP_MIN_CLASS:]]
    for name, sets in (("TNg-32", ["M12"]), ("TNg-H", ["M12", "H"]), ("TNg-R", ["M12", "R"])):
        S = K[K.set.isin(sets)]
        sh = S.groupby("region")[steep].sum().sum(1)
        print(name, "steep (>= 20 %) street-km by region:", (sh / sh.sum()).round(3).to_dict(), f"total {sh.sum():.1f}")
    print("added street-km per stratum:")
    for w in ("H", "R"):
        S = K[K.set == w]
        if len(S):
            print(w, S.groupby("region").street_km.sum().round(1).to_dict(), f"total {S.street_km.sum():.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["m12", "towns", "report"])
    ap.add_argument("--set", choices=["H", "R"], default="H")
    a = ap.parse_args()
    if a.stage == "m12":
        return stage_m12()
    if a.stage == "towns":
        return stage_towns(a.set)
    return stage_report()


if __name__ == "__main__":
    sys.exit(main())
