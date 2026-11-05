"""Milestone 13 generalisation cities (GEN): Girona, Manresa (Spain) and San Francisco (USA).

    python scripts/m13_build_gen_cities.py --stage plan      # boundaries, M2 grid, GHSL, seeded terrain-blind sample
    python scripts/m13_build_gen_cities.py --stage build     # tiles from the cached OSM extracts (unchanged M2 code)
    python scripts/m13_build_gen_cities.py --stage dem       # IGN MDT05 / USGS 3DEP per tile (M12 fetchers)
    python scripts/m13_build_gen_cities.py --stage prepare   # 3D-aware graphs, terrain + water arrays

Selection rule (pre-registered, docs/M13_EVALUATION_PROTOCOL.md §2):
* every 512 m cell of the M2 grid inside the Nominatim city boundary with GHSL 2020 built-up >= 0.15;
* at most CAP[city] tiles drawn at random (seed 20261106); NO terrain screening, so the terrain mix is the city's own;
* blocks of 3 km for the block bootstrap.

Licences:
* OSM: ODbL, from the cached Geofabrik extracts (Catalonia; NorCal for San Francisco);
* IGN MDT05: CC BY 4.0, as in M12;
* USGS 3DEP: public domain.
Outputs: data/processed/m13/gen/{plan,samples}.csv, graphs3d_gen.pkl, terrain/<sid>.npz. GEN is never used for
training, calibration or selection.
"""
import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from urbanmorph.config import PROJECT_ROOT

GEN = PROJECT_ROOT / "data" / "processed" / "m13" / "gen"
SEED = 20261106
CITIES = {
    "ESP_GRO": dict(country="spain", city="girona", query="Girona, Gironès, Catalonia, Spain", crs="EPSG:32631",
                    region="europe/spain/cataluna", cap=30),
    "ESP_MRS": dict(country="spain", city="manresa", query="Manresa, Bages, Catalonia, Spain", crs="EPSG:32631",
                    region="europe/spain/cataluna", cap=30),
    "USA_SFO": dict(country="united_states", city="san_francisco", query="San Francisco, California, United States",
                    crs="EPSG:32610", region="north-america/us/california/norcal", cap=40),
}


def city_obj(code):
    from urbanmorph.sampling.study_area import City
    c = CITIES[code]
    return City(country=c["country"], city=c["city"], code=code, query=c["query"])


def stage_plan(codes):
    from pyproj import Transformer
    from urbanmorph.sampling.grid import candidate_tiles
    from urbanmorph.sampling.strata import add_ghsl_columns
    from urbanmorph.sampling.study_area import fetch_boundary, load_config
    cfg = load_config()
    rng = np.random.default_rng(SEED)
    rows, report = [], {}
    for code in codes:
        c = CITIES[code]
        b = fetch_boundary(city_obj(code))
        cands = candidate_tiles(b, c["crs"], cfg["tile"]["size_m"])
        cands = add_ghsl_columns(pd.DataFrame(cands.drop(columns="geometry")), code, b.total_bounds, c["crs"], cfg)
        elig = cands[cands.ghsl_built_2020 >= cfg["study_area"]["min_builtup_fraction"]].reset_index(drop=True)
        pick = elig.iloc[sorted(rng.permutation(len(elig))[:c["cap"]])].copy() if len(elig) else elig
        from urbanmorph.sampling.grid import sample_id
        pick["sample_id"] = [sample_id(code, r.grid_e, r.grid_n) for r in pick.itertuples()] if "grid_e" in pick else \
            [f"{code}_e{int(r.xmin // 512)}_n{int(r.ymin // 512)}" for r in pick.itertuples()]
        pick["city_code"], pick["city"], pick["country"], pick["crs"] = code, c["city"], c["country"], c["crs"]
        pick["block_id"] = [f"{code}_b{int(r.xmin // 3000)}_{int(r.ymin // 3000)}" for r in pick.itertuples()]
        pick["split"] = "gen"
        to_wgs = Transformer.from_crs(c["crs"], 4326, always_xy=True)
        ll = [to_wgs.transform((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2) for r in pick.itertuples()]
        pick["center_lon"], pick["center_lat"] = [x for x, _ in ll], [y for _, y in ll]
        pick["role"], pick["rank"] = "generalisation_m13", np.arange(len(pick))
        rows.append(pick)
        report[code] = {"candidates": int(len(cands)), "eligible": int(len(elig)), "selected": int(len(pick))}
        print(code, report[code], flush=True)
    GEN.mkdir(parents=True, exist_ok=True)
    plan = pd.concat(rows, ignore_index=True)
    if (GEN / "plan.csv").exists():                                   # keep the cities planned earlier
        old = pd.read_csv(GEN / "plan.csv")
        plan = pd.concat([old[~old.city_code.isin(codes)], plan], ignore_index=True)
    plan.to_csv(GEN / "plan.csv", index=False)
    rep_f = GEN / "plan_report.json"
    prev = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    rep_f.write_text(json.dumps({**prev, **report, "seed": SEED, "date_utc": datetime.now(timezone.utc).isoformat()}, indent=1))


def stage_build():
    from urbanmorph.data.m2_dataset import index_row, process_sample, save_m2_sample
    from urbanmorph.data.osm import is_cached, load_raw
    from urbanmorph.data.osm_extract import build_tile_from_extract, city_pbf, snapshot_info
    from urbanmorph.sampling.plan_io import area_from_row
    from urbanmorph.sampling.study_area import fetch_boundary, load_config
    cfg = load_config()
    plan = pd.read_csv(GEN / "plan.csv")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, skipped = [], []
    for code, sub in plan.groupby("city_code"):
        c = CITIES[code]
        snap = snapshot_info(c["region"])
        clip = city_pbf(code, c["region"], fetch_boundary(city_obj(code)).total_bounds)
        for _, r in sub.iterrows():
            area = area_from_row(r, cfg)
            if not is_cached(area):
                build_tile_from_extract(area, clip, snap)
            layout, meta, sample = process_sample(area, load_raw(area), r.to_dict(), cfg, now)
            meta["sampling"]["role"] = "generalisation_m13"
            meta["split"], meta["in_dataset"], meta["block_id"] = "gen", True, r.block_id
            if not meta["usable"]:
                skipped.append((r.sample_id, ";".join(f["code"] for f in meta["quality_flags"] if f["severity"] == "error")))
                continue
            save_m2_sample(layout, meta, sample)
            row = index_row(meta)
            row.update(split="gen", in_dataset=True, role="generalisation_m13", block_id=r.block_id)
            rows.append(row)
            print(r.sample_id, flush=True)
    pd.DataFrame(rows).to_csv(GEN / "samples.csv", index=False)
    (GEN / "build_report.json").write_text(json.dumps({"built": len(rows), "skipped": skipped}, indent=1))
    print("built", len(rows), "skipped", len(skipped))


def stage_dem():
    from m12_fetch_dem import fetch_spain, fetch_us  # noqa: E402
    idx = pd.read_csv(GEN / "samples.csv")
    log = []
    for r in idx.itertuples():
        tif = PROJECT_ROOT / "data" / "raw" / "dem" / r.country / f"{r.sample_id}.tif"
        tif.parent.mkdir(parents=True, exist_ok=True)
        if tif.exists():
            continue
        try:
            info = fetch_spain(r, tif) if r.country == "spain" else fetch_us(r, tif)
            log.append({"sample_id": r.sample_id, "ok": True, **{k: v for k, v in info.items() if isinstance(v, (str, int, float))}})
        except Exception as e:                                   # recorded, never silently dropped
            log.append({"sample_id": r.sample_id, "ok": False, "error": repr(e)[:300]})
        print(r.sample_id, log[-1]["ok"] if log else "cached", flush=True)
    (GEN / "dem_log.json").write_text(json.dumps(log, indent=1))


def stage_prepare():
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from m12_prepare_data import struct_labels  # noqa: E402
    from urbanmorph.data.m2_dataset import sample_dir
    from urbanmorph.graphs.levels import load_structure_ways, tile_pieces
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    from urbanmorph.terrain.data import tile_terrain
    from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers
    import subprocess
    STRUCT = PROJECT_ROOT / "data" / "processed" / "m12" / "structures"
    CLIPS = PROJECT_ROOT / "data" / "raw" / "osm_extracts" / "cities"
    idx = pd.read_csv(GEN / "samples.csv")
    out = GEN / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    graphs = []
    for code, sub in idx.groupby("city_code"):
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
            f = sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg"
            rc = gpd.read_file(f, layer="road_centerlines")
            if isinstance(rc, gpd.GeoDataFrame) and not rc.empty:
                lines, cls, lev, st, _ = tile_pieces(rc, ways, to_local, r.xmin, r.ymin, highway_class)
            else:
                lines, cls, lev, st = [], [], [], []
            g = build_graph(lines, classes=cls, levels=lev, structs=st)
            graphs.append((r.sample_id, r.country, g, r.block_id))
            t = tile_terrain(r.country, r.sample_id, r.crs, r.xmin, r.ymin)
            if t is not None:
                elev, valid = np.nan_to_num(t.elev, nan=0.0), np.isfinite(t.elev)
            else:
                elev, valid = np.zeros((256, 256), np.float32), np.zeros((256, 256), bool)
            layers = water_layers(feats, tile_window(r.xmin, r.ymin, 256.0))
            tr = from_origin(r.xmin - 256.0, r.ymax + 256.0, 4.0, 4.0)
            water = rasterize_kinds(layers, tr, (256, 256))
            wu = shapely.affinity.translate(shapely.union_all([layers[1], layers[2]]), -r.xmin, -r.ymin)
            np.savez_compressed(out / f"{r.sample_id}.npz", elev=elev.astype(np.float32), valid=valid, water=water,
                                struct_label=struct_labels(g, wu))
            print(r.sample_id, "dem" if valid.all() else "NO DEM", flush=True)
    (GEN / "graphs3d_gen.pkl").write_bytes(pickle.dumps(graphs))


def gen_tiles():
    """GEN tiles with a complete DEM, in the experiment-driver format."""
    graphs = pickle.loads((GEN / "graphs3d_gen.pkl").read_bytes())
    out = []
    for sid, _, g, block in graphs:
        d = dict(np.load(GEN / "terrain" / f"{sid}.npz"))
        if d["valid"].all():
            out.append({"sid": sid, "split": "gen", "block": block, "graph": g, "d": d})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "build", "dem", "prepare", "all"])
    ap.add_argument("--cities", nargs="+", default=list(CITIES))
    a = ap.parse_args()
    if a.stage in ("plan", "all"):
        stage_plan(a.cities)
    if a.stage in ("build", "all"):
        stage_build()
    if a.stage in ("dem", "all"):
        stage_dem()
    if a.stage in ("prepare", "all"):
        stage_prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
