"""Milestone 18, decision D7 (c): GLO-30 versions of the EVALUATION windows, so that TNg is tested on exactly the
tiles TN was tested on: the 16 natural windows of leakage control (b) and the 95 GEN2 tiles.

Same recipe as scripts/m18_build_glo30_terrain.py (whose functions are reused): the 1-degree Copernicus GLO-30 cells
are mosaicked on their native grid, reprojected to the window's own CRS on the same 4 m grid (tile + 256 m margin,
256 x 256), bilinear. ``water`` (and ``struct_label`` where present) are copied from the LiDAR arrays, so only the
elevation differs. Output data/processed/m18/glo30_eval/<sid>.npz (git-ignored); provenance in
outputs/m18/glo30/eval_manifest.json.

    python scripts/m18_glo30_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "data" / "processed" / "m18" / "glo30_eval"


def records():
    """(sid, record with crs / xmin / ymin / xmax / ymax, path of the LiDAR npz) for every evaluation window."""
    from pyproj import Transformer
    w = json.loads((ROOT / "outputs" / "m18" / "natural" / "windows.json").read_text())["windows"]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    for x in w:                                   # as m18_erasure_validation.natural_window: a 512 m tile at the centre
        cx, cy = tr.transform(x["lon"], x["lat"])
        r = SimpleNamespace(crs="EPSG:25830", xmin=round(cx) - 256, ymin=round(cy) - 256, xmax=round(cx) + 256, ymax=round(cy) + 256)
        yield x["name"], r, ROOT / "data" / "processed" / "m18" / "natural" / f"{x['name']}.npz"
    plan = pd.read_csv(ROOT / "data" / "processed" / "m14" / "gen2" / "plan.csv").set_index("sample_id")
    for f in sorted((ROOT / "data" / "processed" / "m14" / "gen2" / "terrain").glob("*.npz")):
        p = plan.loc[f.stem]
        yield f.stem, SimpleNamespace(crs=p.crs, xmin=p.xmin, ymin=p.ymin, xmax=p.xmax, ymax=p.ymax), f


def main() -> int:
    import m18_build_glo30_terrain as b
    OUT.mkdir(parents=True, exist_ok=True)
    log, made = [], []
    for sid, r, lidar in records():
        f = OUT / f"{sid}.npz"
        if f.exists():
            made.append(sid)
            continue
        box = b.lonlat_box(r, b.MARGIN + 600.0)
        paths = [b.fetch(n, log) for n in b.cells_of_box(box)]
        arr, tr, nodata, _ = b.mosaic(paths, box)
        z = b.tile_grid(r, arr, tr, nodata)
        valid = np.isfinite(z) & (z != nodata if nodata is not None else True)
        ref = dict(np.load(lidar))
        out = {k: v for k, v in ref.items() if k not in ("elev", "valid")}
        out.update(elev=np.where(valid, z, 0.0).astype(np.float32), valid=valid)
        np.savez_compressed(f, **out)
        made.append(sid)
        lz = ref["elev"].astype(np.float64)
        print(sid, "valid", round(float(valid.mean()), 3), "rmse vs LiDAR %.1f m" % float(np.sqrt(np.mean((z[valid] - lz[valid]) ** 2))), flush=True)
    atomic_json(ROOT / "outputs" / "m18" / "glo30" / "eval_manifest.json", {"windows": made, "cells": log})
    print("built", len(made), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
