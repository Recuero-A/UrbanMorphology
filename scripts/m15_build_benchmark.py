"""Milestone 15 real-city pilot benchmark: versioned site manifest and preprocessing (docs/M15_PROTOCOL.md §5).

    python scripts/m15_build_benchmark.py --stage manifest   # development windows + declared held-out metros
    python scripts/m15_build_benchmark.py --stage glo30      # Copernicus GLO-30 DSM per window (public AWS bucket)
    python scripts/m15_build_benchmark.py --stage prepare    # street graphs, water, terrain arrays (GLO-30; MDT05 for Spain)
    python scripts/m15_build_benchmark.py --stage strata     # flat / mixed / hilly from GLO-30 landform slope
    python scripts/m15_build_benchmark.py --stage all

Development metros (pilot only): Girona, Manresa (Spain; the M13 GEN windows), Busan, Daejeon (South Korea) and
Arequipa, Cusco (Peru) (the M2 v1 pilot windows). Windows were chosen by the earlier GHSL built-up sampling rules,
never by street metrics. Held-out metros are DECLARED in the manifest only: nothing is fetched, built or evaluated
for them in M15 (building them needs the project owner's authorisation).

Sources (pinned in the manifest): OSM = the cached dated Geofabrik extracts (region file, replication timestamp, MD5);
elevation = Copernicus GLO-30 public DSM (30 m, EGM2008 heights) from s3://copernicus-dem-30m (HTTPS), cropped per
window (tile + 600 m) in EPSG:4326 and resampled by the M12 loader to the tile's UTM grid at 4 m (bilinear); Spain also
keeps the IGN MDT05 bare-earth terrain of M13 as a sensitivity check. Failed acquisitions, empty street graphs and
missing elevation are recorded in the manifest with their reason, never dropped silently.
Outputs: data/processed/m15/real/{manifest.csv, manifest.json, graphs.pkl, terrain_glo30/, terrain_mdt05/}; the manifest
(csv + json) and the GLO-30 log are also published to configs/m15_benchmark/ (tracked: the versioned site manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from urbanmorph.config import PROJECT_ROOT

REAL = PROJECT_ROOT / "data" / "processed" / "m15" / "real"
GLO_SRC = PROJECT_ROOT / "data" / "raw" / "dem" / "glo30_src"
GLO_DIR = PROJECT_ROOT / "data" / "raw" / "dem" / "glo30"
GLO_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
DEV = {
    "ESP_GRO": dict(country="spain", city="girona", source="m13_gen", region="europe/spain/cataluna"),
    "ESP_MRS": dict(country="spain", city="manresa", source="m13_gen", region="europe/spain/cataluna"),
    "KOR_BSN": dict(country="south_korea", city="busan", source="m2_v1", region="asia/south-korea"),
    "KOR_DJN": dict(country="south_korea", city="daejeon", source="m2_v1", region="asia/south-korea"),
    "PER_AQP": dict(country="peru", city="arequipa", source="m2_v1", region="south-america/peru"),
    "PER_CUZ": dict(country="peru", city="cusco", source="m2_v1", region="south-america/peru"),
}
HELD_OUT = {"spain": ["Lleida", "Alicante"], "south_korea": ["Seoul", "Daegu"], "peru": ["Lima", "Trujillo"],
            "external_country_usa": ["Sacramento", "Santa Barbara"]}
SOURCES = {"m13_gen": (PROJECT_ROOT / "data" / "processed" / "m13" / "gen" / "samples.csv", PROJECT_ROOT / "data" / "processed" / "m2"),
           "m2_v1": (PROJECT_ROOT / "data" / "processed" / "m2_v1_peru_korea" / "samples.csv",
                     PROJECT_ROOT / "data" / "processed" / "m2_v1_peru_korea")}


def stage_manifest():
    from urbanmorph.data.osm_extract import snapshot_info
    rows, extracts = [], {}
    for code, c in DEV.items():
        f, root = SOURCES[c["source"]]
        df = pd.read_csv(f)
        df = df[df.city_code == code]
        if c["region"] not in extracts:
            extracts[c["region"]] = snapshot_info(c["region"])
        for r in df.itertuples():
            rows.append({"sample_id": r.sample_id, "metro": code, "country": c["country"], "city": c["city"],
                         "role": "development", "window_source": c["source"], "crs": r.crs, "xmin": r.xmin, "ymin": r.ymin,
                         "center_lat": r.center_lat, "center_lon": r.center_lon,
                         "block": f"{code}_b{int(r.xmin // 3000)}_{int(r.ymin // 3000)}",
                         "vectors": str((root / c["country"] / c["city"] / f"{r.sample_id}_vectors.gpkg").relative_to(PROJECT_ROOT)),
                         "osm_region": c["region"], "status": "planned", "reason": ""})
    REAL.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(REAL / "manifest.csv", index=False)
    meta = {"version": "m15-pilot-1", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "development_metros": list(DEV), "held_out_declared": HELD_OUT,
            "held_out_rule": "declared before any M15 real-city result; not fetched, built or evaluated without authorisation",
            "window_rule": "earlier GHSL built-up sampling rules (M13 GEN: GHSL 2020 >= 0.15, random cap; M2 v1: stratified "
                           "GHSL design); never street metrics. Strata assigned afterwards from GLO-30 slope",
            "osm_extracts": extracts, "elevation": {"primary": "Copernicus GLO-30 public DSM (30 m)",
                                                    "sensitivity": "IGN MDT05 bare-earth DTM (Spain only, M13 terrain)"},
            "known_limitations": ["GLO-30 is a DSM: buildings and vegetation raise the surface; at 30 m, building blocks "
                                  "and street canyons can imprint the street pattern (leakage risk)",
                                  "Korea and Peru were dropped in M2 for incomplete OSM building mapping; street completeness "
                                  "is not independently verified",
                                  "Spanish development windows were used in M13/M14 for other questions"]}
    (REAL / "manifest.json").write_text(json.dumps(meta, indent=1, default=str))
    print(m.groupby("metro").size().to_dict())


def glo_names(lat0, lat1, lon0, lon1):
    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            yield f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"


def stage_glo30():
    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    m = pd.read_csv(REAL / "manifest.csv")
    GLO_SRC.mkdir(parents=True, exist_ok=True)
    GLO_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for r in m.itertuples():
        tif = GLO_DIR / f"{r.sample_id}.tif"
        if tif.exists():
            continue
        pad = 0.012                                                     # ~1.3 km of latitude: tile + context + margin
        lat0, lat1 = r.center_lat - pad, r.center_lat + pad
        dlon = pad / max(math.cos(math.radians(r.center_lat)), 0.2)
        lon0, lon1 = r.center_lon - dlon, r.center_lon + dlon
        srcs, names = [], []
        try:
            for n in glo_names(lat0, lat1, lon0, lon1):
                f = GLO_SRC / f"{n}.tif"
                if not f.exists():
                    urllib.request.urlretrieve(GLO_URL.format(n=n), f)
                names.append(n)
                srcs.append(rasterio.open(f))
            arr, tr = merge(srcs, bounds=(lon0, lat0, lon1, lat1))
            prof = srcs[0].profile | {"height": arr.shape[1], "width": arr.shape[2], "transform": tr, "count": 1,
                                      "driver": "GTiff", "compress": "deflate", "tiled": False}
            prof.pop("blockxsize", None), prof.pop("blockysize", None)
            with rasterio.open(tif, "w", **prof) as dst:
                dst.write(arr[:1])
            info = {"provider": "Copernicus DEM GLO-30 (public release), DSM", "native_resolution_m": 30.0,
                    "source_tiles": names, "url_template": GLO_URL, "crs": "EPSG:4326", "vertical_datum": "EGM2008",
                    "bounds_wgs84": [lon0, lat0, lon1, lat1], "sha1_crop": hashlib.sha1(tif.read_bytes()).hexdigest(),
                    "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            tif.with_suffix(".json").write_text(json.dumps(info, indent=1))
            log.append({"sample_id": r.sample_id, "ok": True})
        except Exception as e:                                          # recorded, never silently dropped
            log.append({"sample_id": r.sample_id, "ok": False, "error": repr(e)[:300]})
        finally:
            for s in srcs:
                s.close()
        print(r.sample_id, log[-1]["ok"], flush=True)
    prev = json.loads((REAL / "glo30_log.json").read_text()) if (REAL / "glo30_log.json").exists() else []
    (REAL / "glo30_log.json").write_text(json.dumps(prev + log, indent=1))


def stage_prepare():
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from m12_prepare_data import struct_labels  # noqa: E402
    from urbanmorph.graphs.levels import load_structure_ways, tile_pieces
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    from urbanmorph.terrain.data import tile_terrain
    from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers
    STRUCT = PROJECT_ROOT / "data" / "processed" / "m12" / "structures"
    CLIPS = PROJECT_ROOT / "data" / "raw" / "osm_extracts" / "cities"
    MDT05 = PROJECT_ROOT / "data" / "processed" / "m13" / "gen" / "terrain"
    m = pd.read_csv(REAL / "manifest.csv")
    for d in ("terrain_glo30", "terrain_mdt05"):
        (REAL / d).mkdir(parents=True, exist_ok=True)
    graphs, status = [], []
    for code, sub in m.groupby("metro", sort=False):
        sf = STRUCT / f"{code}.geojsonseq"
        if not sf.exists():
            tmp = STRUCT / f"_{code}.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{code}.osm.pbf"), "w/bridge", "w/tunnel",
                            "w/layer", "w/ford"], check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(sf), str(tmp)],
                           check=True, capture_output=True)
            tmp.unlink()
        ways = load_structure_ways(sf)
        crs = sub.crs.iloc[0]
        to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        feats = city_features(code, crs)
        for r in sub.itertuples():
            reason = ""
            rc = gpd.read_file(PROJECT_ROOT / r.vectors, layer="road_centerlines")
            if isinstance(rc, gpd.GeoDataFrame) and not rc.empty:
                lines, cls, lev, st, _ = tile_pieces(rc, ways, to_local, r.xmin, r.ymin, highway_class)
            else:
                lines, cls, lev, st = [], [], [], []
            g = build_graph(lines, classes=cls, levels=lev, structs=st)
            layers = water_layers(feats, tile_window(r.xmin, r.ymin, 256.0))
            water = rasterize_kinds(layers, from_origin(r.xmin - 256.0, r.ymin + 512.0 + 256.0, 4.0, 4.0), (256, 256))
            wu = shapely.affinity.translate(shapely.union_all([layers[1], layers[2]]), -r.xmin, -r.ymin)
            t = tile_terrain("glo30", r.sample_id, crs, r.xmin, r.ymin)
            if t is None:
                reason = "no GLO-30 elevation"
            else:
                valid = np.isfinite(t.elev)
                if not valid.all():
                    reason = f"GLO-30 no-data share {1 - valid.mean():.3f}"
                np.savez_compressed(REAL / "terrain_glo30" / f"{r.sample_id}.npz", elev=np.nan_to_num(t.elev).astype(np.float32),
                                    valid=valid, water=water, struct_label=struct_labels(g, wu))
            if r.country == "spain" and (MDT05 / f"{r.sample_id}.npz").exists():
                d5 = dict(np.load(MDT05 / f"{r.sample_id}.npz"))
                np.savez_compressed(REAL / "terrain_mdt05" / f"{r.sample_id}.npz", **d5)
            if len(g.edges) == 0:
                reason = (reason + "; " if reason else "") + "empty OSM street graph"
            graphs.append((r.sample_id, r.metro, g, r.block))
            status.append(("excluded" if reason else "ok", reason))
            print(r.sample_id, status[-1], flush=True)
    m["status"], m["reason"] = [s for s, _ in status], [x for _, x in status]
    m.to_csv(REAL / "manifest.csv", index=False)
    (REAL / "graphs.pkl").write_bytes(pickle.dumps(graphs))


def stage_strata():
    from urbanmorph.graphs.terrain_metrics import landform_gradient
    from urbanmorph.graphs.terrain_generate import terrain_of
    m = pd.read_csv(REAL / "manifest.csv")
    slope, stratum = [], []
    for r in m.itertuples():
        f = REAL / "terrain_glo30" / f"{r.sample_id}.npz"
        if not f.exists():
            slope.append(np.nan), stratum.append("")
            continue
        t = terrain_of(dict(np.load(f)))
        gx, gy = landform_gradient(t)
        s = float(np.nanmedian(np.hypot(gx, gy)[64:192, 64:192]))
        slope.append(s)
        stratum.append("flat" if s < 0.05 else "hilly" if s >= 0.10 else "mixed")
    m["glo30_slope_med"], m["stratum"] = slope, stratum
    m.to_csv(REAL / "manifest.csv", index=False)
    print(m.groupby(["metro", "stratum"]).size().unstack(fill_value=0))


from urbanmorph.experiments.sites import real_tiles  # noqa: E402,F401 (the loader lives in the library)


def publish():
    """Copy the versioned manifest to the tracked configs/m15_benchmark/."""
    import shutil
    dst = PROJECT_ROOT / "configs" / "m15_benchmark"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.csv", "manifest.json", "glo30_log.json"):
        if (REAL / f).exists():
            shutil.copy2(REAL / f, dst / f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["manifest", "glo30", "prepare", "strata", "all"])
    a = ap.parse_args()
    for st, f in (("manifest", stage_manifest), ("glo30", stage_glo30), ("prepare", stage_prepare), ("strata", stage_strata)):
        if a.stage in (st, "all"):
            f()
    publish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
