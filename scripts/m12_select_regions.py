"""Milestone 12, Phase 2b: choose terrain-diverse expansion tiles in Los Angeles and San Diego objectively.

    python scripts/m12_select_regions.py      # -> data/processed/m12/sampling/{candidates,plan_m12}.csv,
                                              #    outputs/m12/audits/region_selection.json

Rules (fixed before any terrain value was looked at):
1. Candidates: 512 m cells of the M2 grid inside the city boundary (LA: the M2 boundary; San Diego: the M2 backup
   city, boundary from the same Nominatim query) with GHSL 2020 built-up >= 0.15, the M2 urban eligibility rule.
2. Terrain screening: USGS 3DEP bare earth at 32 m over the candidate area (one coarse request per city), landform
   slope p90 per cell at 32 m, the audit's landform definition (docs/MILESTONE_12.md §10); class flat < 5 %,
   moderate 5-15 %, steep >= 15 %.
3. Los Angeles -> TRAIN only. Cells in a 3 km spatial block holding an LA val/test tile, or within 1,536 m (edge to
   edge) of one, are excluded; new tiles keep >= 1,024 m centre spacing to every LA train tile and to each other.
   Greedy order: steep first, then moderate, each in a seeded random order; at most N_LA tiles.
4. San Diego -> a held-out city (never used for training): 3 km blocks are assigned at random (seeded) to
   ``val_t`` (terrain validation, development) or ``test_t`` (terrain test, untouched until authorisation);
   tiles closer than 1,536 m to a tile of the other split are dropped; within each split, steep and moderate tiles
   are taken first (seeded order) up to N_SD per split, then flat ones up to N_SD_FLAT, so that every terrain class
   is represented for evaluation.
"""
import json
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from m12_fetch_dem import USGS, get  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.sampling.grid import candidate_tiles, sample_id
from urbanmorph.sampling.strata import add_ghsl_columns
from urbanmorph.sampling.study_area import City, cities, fetch_boundary, load_config
from urbanmorph.splits.geographic import block_ids, edge_distance

OUT = PROJECT_ROOT / "data" / "processed" / "m12" / "sampling"
AUD = PROJECT_ROOT / "outputs" / "m12" / "audits"
COARSE = PROJECT_ROOT / "data" / "raw" / "dem" / "coarse"
SEED, N_LA, N_SD, N_SD_FLAT = 20261104, 120, 30, 10
BUFFER_M, SPACING_M, BLOCK_M, RES = 1536.0, 1024.0, 3000.0, 32.0
CITY = {"USA_LAX": ("los_angeles", "EPSG:32611"), "USA_SAN": ("san_diego", "EPSG:32611")}


def coarse_dem(code, crs, bounds):
    """3DEP bare earth at RES metres over ``bounds`` (tile CRS); cached GeoTIFF."""
    COARSE.mkdir(parents=True, exist_ok=True)
    tif = COARSE / f"{code}_{int(RES)}m.tif"
    x0, y0, x1, y1 = (np.floor(bounds[0] / RES) * RES - RES, np.floor(bounds[1] / RES) * RES - RES,
                      np.ceil(bounds[2] / RES) * RES + RES, np.ceil(bounds[3] / RES) * RES + RES)
    if not tif.exists():
        epsg = int(crs.split(":")[1])
        get(USGS + "/exportImage", {"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": epsg, "imageSR": epsg,
                                    "size": f"{int((x1 - x0) / RES)},{int((y1 - y0) / RES)}", "format": "tiff",
                                    "pixelType": "F32", "noData": -9999, "interpolation": "RSP_BilinearInterpolation",
                                    "f": "image"}, tif)
    with rasterio.open(tif) as d:
        z = d.read(1).astype(float)
        z[z < -1000] = np.nan
        return z, d.transform


def landform_p90(cands, z, tr):
    gy, gx = np.gradient(z, RES)
    s = np.hypot(gx, gy)
    out = []
    for r in cands.itertuples():
        c0, r0 = ~tr * (r.xmin, r.ymax)
        c0, r0 = int(round(c0)), int(round(r0))
        w = s[r0:r0 + 16, c0:c0 + 16]
        out.append(float(np.nanpercentile(w, 90)) if np.isfinite(w).any() else np.nan)
    return np.array(out)


def classify(p):
    return np.where(p >= 0.15, "steep", np.where(p >= 0.05, "moderate", "flat"))


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    AUD.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    ds = idx[idx.in_dataset]
    by_code = {c.code: c for c in cities(cfg)}
    by_code["USA_SAN"] = City(country="united_states", city="san_diego", code="USA_SAN",
                              query="San Diego, San Diego County, California, United States")
    rng = np.random.default_rng(SEED)
    plan, report = [], {"seed": SEED, "rules": __doc__}
    for code, (city, crs) in CITY.items():
        b = fetch_boundary(by_code[code])
        cands = candidate_tiles(b, crs, cfg["tile"]["size_m"])
        cands = add_ghsl_columns(pd.DataFrame(cands.drop(columns="geometry")), code, b.total_bounds, crs, cfg)
        cands = cands[cands.ghsl_built_2020 >= cfg["study_area"]["min_builtup_fraction"]].reset_index(drop=True)
        z, tr = coarse_dem(code, crs, (cands.xmin.min(), cands.ymin.min(), cands.xmax.max(), cands.ymax.max()))
        cands["landform_p90"] = landform_p90(cands, z, tr)
        cands = cands[np.isfinite(cands.landform_p90)].reset_index(drop=True)
        cands["terrain_class"] = classify(cands.landform_p90)
        cands["sample_id"] = [sample_id(code, e, n) for e, n in zip(cands.grid_e, cands.grid_n)]
        cands["city_code"], cands["city"], cands["country"], cands["crs"] = code, city, "united_states", crs
        cands["block_id"] = block_ids(cands, cfg["splits"]["block_size_m"])
        cands.to_csv(OUT / f"candidates_{code}.csv", index=False)
        report[code] = {"eligible_cells": len(cands), "classes": cands.terrain_class.value_counts().to_dict()}
        existing = ds[ds.city_code == code]
        cands = cands[~cands.sample_id.isin(existing.sample_id)]
        if code == "USA_LAX":
            held = existing[existing.split.isin(["val", "test"])]
            hb = set(block_ids(held, cfg["splits"]["block_size_m"]))
            ok = [not (bid in hb) and all(edge_distance(r, h) >= BUFFER_M for _, h in held.iterrows())
                  for (_, r), bid in zip(cands.iterrows(), cands.block_id)]
            pool = cands[np.array(ok)]
            taken = [np.c_[(existing.xmin + existing.xmax) / 2, (existing.ymin + existing.ymax) / 2][i]
                     for i in range(len(existing)) if existing.split.iloc[i] == "train"]
            chosen = []
            for cls in ("steep", "moderate"):
                sub = pool[pool.terrain_class == cls]
                for i in rng.permutation(len(sub)):
                    r = sub.iloc[i]
                    c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
                    if len(chosen) < N_LA and all(np.hypot(*(c - t)) >= SPACING_M for t in taken):
                        taken.append(c)
                        chosen.append({**r.to_dict(), "split": "train"})
            plan += chosen
            report[code]["selected_train"] = pd.Series([c["terrain_class"] for c in chosen]).value_counts().to_dict()
        else:
            blocks = np.array(sorted(cands.block_id.unique()))
            assign = dict(zip(blocks, np.where(rng.permutation(len(blocks)) % 2 == 0, "val_t", "test_t")))
            cands = cands.assign(split=cands.block_id.map(assign))
            v, t = cands[cands.split == "val_t"], cands[cands.split == "test_t"]
            keep = []
            for _, r in cands.iterrows():
                other = t if r.split == "val_t" else v
                keep.append(all(edge_distance(r, o) >= BUFFER_M for _, o in other.iterrows()))
            cands = cands[np.array(keep)]
            for sp in ("val_t", "test_t"):
                taken, chosen = [], []
                sub_all = cands[cands.split == sp]
                for cls, cap in (("steep", N_SD), ("moderate", N_SD), ("flat", N_SD + N_SD_FLAT)):
                    sub = sub_all[sub_all.terrain_class == cls]
                    for i in rng.permutation(len(sub)):
                        r = sub.iloc[i]
                        c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
                        if len(chosen) < cap and all(np.hypot(*(c - x)) >= SPACING_M for x in taken):
                            taken.append(c)
                            chosen.append(r.to_dict())
                plan += chosen
                report[code][f"selected_{sp}"] = pd.Series([c["terrain_class"] for c in chosen]).value_counts().to_dict()
        print(code, json.dumps(report[code]), flush=True)
    p = pd.DataFrame(plan)
    to_wgs = {c: Transformer.from_crs(c, 4326, always_xy=True) for c in p.crs.unique()}
    lon, lat = zip(*[to_wgs[r.crs].transform((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2) for r in p.itertuples()])
    p["center_lon"], p["center_lat"] = lon, lat
    p["role"], p["rank"] = "expansion_m12", np.arange(len(p))
    p.to_csv(OUT / "plan_m12.csv", index=False)
    report["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (AUD / "region_selection.json").write_text(json.dumps(report, indent=1, default=str))
    print(p.groupby(["city", "split", "terrain_class"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
