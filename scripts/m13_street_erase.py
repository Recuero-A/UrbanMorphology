"""Milestone 13, leakage control C: street-erased DEMs for every development tile (train, val, val_t; never test).

    python scripts/m13_street_erase.py --factor 1.5 2.5 4.0     # -> data/processed/m13/terrain_se/f<f>/<sid>.npz
    python scripts/m13_street_erase.py --audit                  # -> outputs/m13/audits/street_erasure.json
    python scripts/m13_street_erase.py --gen --factor 4.0       # GEN cities (M13 generalisation set), frozen factor

Why: an urban bare-earth DEM still contains the street beds (M12: |Laplacian| on street pixels is 0.53x its
off-street value), so a terrain-conditioned model can partly read the answer. Erasure removes street-shaped relief from
the MODEL INPUT; rules and metrics keep the original DEM.

Method, per tile, on the tile's own 4 m grid (256 x 256 px, 256 m margin, row 0 = north):
1. corridor mask: every OSM street way of the city clip (terrain.water.city_features, so the margin is erased too),
   buffered by ``factor * w_c / 2`` with w_c the class carriageway width (9.75 / 10 / 8 / 6 m, Milestone 6), i.e. the
   carriageway plus verges and cut / fill shoulders for factor > 1;
2. smooth fill: biharmonic inpainting of the elevation inside the mask (skimage.restoration.inpaint_biharmonic);
3. micro-relief: the inpainted surface is smoother than natural ground, which would itself mark the corridors. The
   residual R = z - G_8m(z) of the ORIGINAL DEM is transplanted into the mask from off-corridor pixels at a random
   offset (seeded per tile), so corridor pixels carry natural-ground roughness.
The corridor width is chosen on TRAINING tiles only (--audit): the factor whose on-street / off-street median
|Laplacian| ratio is closest to 1 after erasure, with the slope distribution change reported. This is a sensitivity
analysis, not a reconstruction of pre-development terrain.
"""
import argparse
import json
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import load_index_m12
from urbanmorph.graphs.road_graph import highway_class
from urbanmorph.terrain.water import city_features

SRC = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
OUT = PROJECT_ROOT / "data" / "processed" / "m13" / "terrain_se"
AUD = PROJECT_ROOT / "outputs" / "m13" / "audits"
WIDTH = np.array([9.75, 10.0, 8.0, 6.0])
MARGIN, RES, N = 256.0, 4.0, 256


def corridor_masks(streets, r, factors, halo=0.0):
    """{factor: bool (N, N)} of buffered streets on the tile grid; also the on-street (carriageway) mask."""
    tr = from_origin(r.xmin - MARGIN, r.ymin + 512 + MARGIN, RES, RES)
    box = shapely.box(r.xmin - MARGIN - 50, r.ymin - MARGIN - 50, r.xmin + 512 + MARGIN + 50, r.ymin + 512 + MARGIN + 50)
    s = streets[streets.intersects(box)]
    cls = np.array([highway_class(h) for h in s.highway]) if len(s) else np.zeros(0, int)
    out = {}
    for f in list(factors) + ["carriageway"]:
        k = 1.0 if f == "carriageway" else float(f)
        geoms = [(g.buffer(k * WIDTH[c] / 2 + halo), 1) for g, c in zip(s.geometry, cls)]
        out[f] = (rasterize(geoms, out_shape=(N, N), transform=tr, fill=0, dtype="uint8").astype(bool)
                  if geoms else np.zeros((N, N), bool))
    return out


def erase(elev, valid, mask, rng):
    from skimage.restoration import inpaint_biharmonic
    m = mask & valid
    if not m.any():
        return elev.copy()
    z = np.where(valid, elev, np.nan)
    fill = np.where(np.isfinite(z), z, np.nanmean(z))
    sm = inpaint_biharmonic(fill.astype(np.float64), m)
    resid = fill - ndimage.gaussian_filter(fill, 2.0)
    out = np.where(m, sm, fill)
    todo = m.copy()
    rr, cc = np.nonzero(m)
    for _ in range(30):                                   # transplant natural roughness from off-corridor pixels
        if not todo.any():
            break
        dr, dc = rng.integers(-96, 97, 2)
        sr, sc = np.clip(rr + dr, 0, N - 1), np.clip(cc + dc, 0, N - 1)
        ok = todo[rr, cc] & ~mask[sr, sc] & valid[sr, sc]
        out[rr[ok], cc[ok]] += resid[sr[ok], sc[ok]]
        todo[rr[ok], cc[ok]] = False
    return np.where(valid, out, 0.0).astype(np.float32)


def lap_ratio(elev, valid, on, off):
    L = np.abs(ndimage.laplace(elev.astype(np.float64)))
    inner = np.zeros_like(valid)
    inner[2:-2, 2:-2] = True
    a, b = L[on & valid & inner], L[off & valid & inner]
    return float(np.median(a) / max(np.median(b), 1e-9)) if len(a) > 20 and len(b) > 20 else np.nan


def slope_of(elev):
    z = ndimage.uniform_filter(elev.astype(np.float64), 8)       # 32 m landform slope (M12 definition)
    gy, gx = np.gradient(z, RES)
    return np.hypot(gx, gy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, nargs="+", default=[1.5, 2.5, 4.0])
    ap.add_argument("--audit", action="store_true", help="only compute the selection statistics on training tiles")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen", action="store_true", help="erase the GEN tiles (data/processed/m13/gen) with the frozen factor")
    a = ap.parse_args()
    global SRC
    if a.gen:
        import pandas as pd
        assert not a.audit and a.factor == [4.0], "GEN uses the factor frozen on training tiles (4.0)"
        gen = PROJECT_ROOT / "data" / "processed" / "m13" / "gen"
        idx, SRC = pd.read_csv(gen / "samples.csv"), gen / "terrain"
    else:
        idx = load_index_m12()
    assert not idx.split.isin(["test", "test_t"]).any()
    rows, done = [], 0
    for code, sub in idx.groupby("city_code"):
        feats = city_features(code, sub.crs.iloc[0])["streets"]
        for r in sub.itertuples():
            src = SRC / f"{r.sample_id}.npz"
            if not src.exists():
                continue
            d = dict(np.load(src))
            if a.audit and r.split != "train":
                continue
            if not a.audit and all((OUT / f"f{f:g}" / f"{r.sample_id}.npz").exists() for f in a.factor):
                continue
            rng = np.random.default_rng(zlib.crc32(r.sample_id.encode()))
            masks = corridor_masks(feats, r, a.factor)
            on = masks["carriageway"]
            far = ~ndimage.binary_dilation(masks[max(a.factor)], iterations=10)
            rec = {"sample_id": r.sample_id, "split": r.split, "city": code, "has_dem": bool(d["valid"].all())}
            if rec["has_dem"]:
                rec["ratio_orig"] = lap_ratio(d["elev"], d["valid"], on, far)
                s0 = slope_of(d["elev"])
            for f in a.factor:
                e = erase(d["elev"], d["valid"], masks[f], rng) if rec["has_dem"] else d["elev"]
                if rec["has_dem"]:
                    rec[f"ratio_f{f:g}"] = lap_ratio(e, d["valid"], on, far)
                    rec[f"mask_share_f{f:g}"] = float(masks[f].mean())
                    s1 = slope_of(e)
                    rec[f"slope_p50_shift_f{f:g}"] = float(np.median(s1) - np.median(s0))
                    rec[f"slope_p90_shift_f{f:g}"] = float(np.percentile(s1, 90) - np.percentile(s0, 90))
                    rec[f"abs_dz_corridor_f{f:g}"] = float(np.mean(np.abs(e - d["elev"])[masks[f] & d["valid"]])) if masks[f].any() else 0.0
                if not a.audit:
                    (OUT / f"f{f:g}").mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(OUT / f"f{f:g}" / f"{r.sample_id}.npz", **{**d, "elev": e})
            rows.append(rec)
            done += 1
            if a.limit and done >= a.limit:
                break
        print(code, len(sub), flush=True)
        if a.limit and done >= a.limit:
            break
    if a.audit:
        AUD.mkdir(parents=True, exist_ok=True)
        dem = [r for r in rows if r["has_dem"]]
        summ = {"n_train_tiles_with_dem": len(dem), "ratio_orig_median": float(np.nanmedian([r["ratio_orig"] for r in dem]))}
        for f in a.factor:
            summ[f"f{f:g}"] = {k: float(np.nanmedian([r[f"{k}_f{f:g}"] for r in dem]))
                               for k in ("ratio", "mask_share", "slope_p50_shift", "slope_p90_shift", "abs_dz_corridor")}
        best = min(a.factor, key=lambda f: abs(np.log(summ[f"f{f:g}"]["ratio"])))
        summ["selected_factor"] = best
        summ["rule"] = "factor whose median on/off-street |Laplacian| ratio after erasure is closest to 1 (training tiles)"
        (AUD / "street_erasure.json").write_text(json.dumps({"summary": summ, "tiles": rows}, indent=1, default=float))
        print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
