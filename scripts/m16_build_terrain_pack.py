"""Milestone 16, part 3: the versioned real-terrain benchmark pack (docs/M16_TERRAIN_PACK.md).

Named sites (recurring comparisons and demos; sources and choices in docs/M16_TERRAIN_SOURCES.md):
* KAIST Daejeon main campus: one 512 m window at the campus centre, plus the campus-wide domain (3 x 3 windows,
  1536 m, holding the OSM campus polygon with ~100 m to spare) - GLO-30 DSM (no open Korean DTM without a login);
* Seoul, Haebangchon (Namsan south slope) - GLO-30 DSM;
* Madrid, Lavapies / Embajadores - IGN MDT05 DTM, with GLO-30 as the matched-source control;
* Guadarrama foothills, San Lorenzo de El Escorial upper town (a deliberately steep stress-test window, selected on
  street grades) - IGN MDT05 DTM, with GLO-30 control.
Plus the pinned synthetic cases (experiments/sites.py: syn_dev, syn_eval, syn_conf) with their parameters.

    python scripts/m16_build_terrain_pack.py --stage clips      # OSM clips of each site from the regional extracts
    python scripts/m16_build_terrain_pack.py --stage build      # terrain npz + canonical GeoTIFFs, water, streets
    python scripts/m16_build_terrain_pack.py --stage refs       # extra references: + service roads; pedestrian paths
    python scripts/m16_build_terrain_pack.py --stage audit      # street-imprint audit (on/off-street |Laplacian|)
    python scripts/m16_build_terrain_pack.py --stage all

Model inputs (data/processed/m16/pack/terrain/<id>__<dem>.npz: elev, valid, water on the 4 m tile grid with the 256 m
margin, the format of every earlier milestone) are stored apart from the street reference graphs
(data/processed/m16/pack/streets/<id>.pkl), which no model input reads. Canonical GeoTIFFs (UTM, float32, NaN no-data,
metres) sit next to the npz. The manifest (configs/m16_pack/manifest.json) records boundaries, sources, checksums,
CRS, vertical datum, resolution, preprocessing and attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.style_metrics import street_km

RAW = PROJECT_ROOT / "data" / "raw" / "m16"
EXTRACTS = PROJECT_ROOT / "data" / "raw" / "osm_extracts"
CLIPS = EXTRACTS / "cities"
PACK = PROJECT_ROOT / "data" / "processed" / "m16" / "pack"
CFG = PROJECT_ROOT / "configs" / "m16_pack"
TILE, MARGIN, RES = 512.0, 256.0, 4.0
GLO = {"N36E127": "Copernicus_DSM_COG_10_N36_00_E127_00_DEM.tif", "N37E126": "Copernicus_DSM_COG_10_N37_00_E126_00_DEM.tif",
       "N40W004": "Copernicus_DSM_COG_10_N40_00_W004_00_DEM.tif", "N40W005": "Copernicus_DSM_COG_10_N40_00_W005_00_DEM.tif"}
ATTR = {"glo30": "Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided "
                 "under COPERNICUS by the European Union and ESA; all rights reserved (free licence, attribution)",
        "mdt05": "MDT05 (c) Instituto Geografico Nacional de Espana, CC BY 4.0 scne.es",
        "osm": "(c) OpenStreetMap contributors, ODbL 1.0"}

# id, name, lat, lon, crs, regional extract, dem sources ({dem: raster}), grid (rows, cols) of 512 m windows, role
SITES = [
    dict(id="KAIST", name="KAIST Daejeon main campus (centre window)", lat=36.370164, lon=127.362599, crs="EPSG:32652",
         extract="asia_south-korea-latest.osm.pbf", dems={"glo30": RAW / "glo30" / GLO["N36E127"]}, grid=(1, 1)),
    dict(id="KAIST_CAMPUS", name="KAIST Daejeon main campus, campus-wide (3 x 3 windows)", lat=36.370164, lon=127.362599,
         crs="EPSG:32652", extract="asia_south-korea-latest.osm.pbf", dems={"glo30": RAW / "glo30" / GLO["N36E127"]},
         grid=(3, 3), boundary=RAW / "kaist" / "kaist_main_campus_osm.geojson"),
    dict(id="SEL_HBC", name="Seoul, Haebangchon (Namsan south slope)", lat=37.5425, lon=126.9875, crs="EPSG:32652",
         extract="asia_south-korea-latest.osm.pbf", dems={"glo30": RAW / "glo30" / GLO["N37E126"]}, grid=(1, 1)),
    dict(id="MAD_LAV", name="Madrid, Lavapies / Embajadores", lat=40.4087, lon=-3.7020, crs="EPSG:25830",
         extract="europe_spain_madrid-latest.osm.pbf",
         dems={"mdt05": RAW / "madrid" / "MAD_LAVAPIES_mdt05.tif", "glo30": RAW / "glo30" / GLO["N40W004"]}, grid=(1, 1)),
    dict(id="GUA_ESC", name="San Lorenzo de El Escorial, upper town (steep stress test)", lat=40.5928, lon=-4.1490,
         crs="EPSG:25830", extract="europe_spain_madrid-latest.osm.pbf",
         dems={"mdt05": RAW / "guadarrama" / "GUA_ESCORIAL_NW_mdt05.tif", "glo30": RAW / "glo30" / GLO["N40W005"]},
         grid=(1, 1)),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def origin(site) -> tuple[float, float]:
    """SW corner (UTM) of the site's window grid, the grid centred on the declared centre."""
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", site["crs"], always_xy=True).transform(site["lon"], site["lat"])
    r, c = site["grid"]
    return round(x - c * TILE / 2), round(y - r * TILE / 2)


def windows(site) -> list[tuple[str, float, float]]:
    x0, y0 = origin(site)
    r, c = site["grid"]
    if (r, c) == (1, 1):
        return [(site["id"], x0, y0)]
    return [(f"{site['id']}_r{i}c{j}", x0 + j * TILE, y0 + (r - 1 - i) * TILE) for i in range(r) for j in range(c)]


def stage_clips():
    from pyproj import Transformer
    for s in SITES:
        out = CLIPS / f"M16_{s['id']}.osm.pbf"
        if out.exists():
            continue
        x0, y0 = origin(s)
        r, c = s["grid"]
        back = Transformer.from_crs(s["crs"], "EPSG:4326", always_xy=True).transform
        pad = MARGIN + 300.0
        lo = back(x0 - pad, y0 - pad)
        hi = back(x0 + c * TILE + pad, y0 + r * TILE + pad)
        bbox = f"{min(lo[0], hi[0])},{min(lo[1], hi[1])},{max(lo[0], hi[0])},{max(lo[1], hi[1])}"
        subprocess.run(["osmium", "extract", "-O", "-b", bbox, "-o", str(out), str(EXTRACTS / s["extract"])], check=True,
                       capture_output=True)
        print("clip", s["id"], bbox, flush=True)


def window_terrain(src: Path, crs: str, xmin: float, ymin: float, size: float = TILE):
    """Bilinear reprojection of ``src`` onto the tile grid (``size`` + 2 x margin, 4 m), elevation in metres, NaN outside."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    n = int(round((size + 2 * MARGIN) / RES))
    dst = np.full((n, n), np.nan, np.float32)
    tr = from_origin(xmin - MARGIN, ymin + size + MARGIN, RES, RES)
    with rasterio.open(src) as r:
        reproject(rasterio.band(r, 1), dst, src_nodata=r.nodata, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        native = float(abs(r.res[0])) if r.crs.is_projected else 30.0
    return dst, tr, native


def write_tif(path: Path, a: np.ndarray, tr, crs: str):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32", crs=crs,
                       transform=tr, nodata=np.nan, compress="deflate") as dst:
        dst.write(a.astype(np.float32), 1)


def stage_build():
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from urbanmorph.graphs.levels import load_structure_ways, tile_pieces
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers
    for d in ("terrain", "streets", "geotiff"):
        (PACK / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SITES:
        code = f"M16_{s['id']}"
        feats = city_features(code, s["crs"])                     # cached geojsonseq of the clip (water, streets)
        sf = PACK / f"_{code}_struct.geojsonseq"
        if not sf.exists():
            tmp = PACK / f"_{code}.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{code}.osm.pbf"), "w/bridge",
                            "w/tunnel", "w/layer", "w/ford"], check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(sf), str(tmp)],
                           check=True, capture_output=True)
            tmp.unlink()
        ways = load_structure_ways(sf)
        st_seq = PACK / f"_{code}_streets.geojsonseq"
        if not st_seq.exists():
            from urbanmorph.terrain.water import STREET_VALUES
            tmp = PACK / f"_{code}_s.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{code}.osm.pbf"),
                            "w/highway=" + ",".join(STREET_VALUES)], check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(st_seq),
                            str(tmp)], check=True, capture_output=True)
            tmp.unlink()
        streets = gpd.read_file(st_seq).set_crs(4326, allow_override=True).to_crs(s["crs"])
        streets = streets[streets.geom_type.isin(["LineString", "MultiLineString"])]
        streets["osmid"] = streets["@id"]
        to_local = Transformer.from_crs("EPSG:4326", s["crs"], always_xy=True).transform
        for wid, x0, y0 in windows(s):
            win = shapely.box(x0, y0, x0 + TILE, y0 + TILE)
            rc = streets[streets.intersects(win)].copy()
            rc["geometry"] = rc.geometry.intersection(win)
            lines, cls, lev, stt, _ = tile_pieces(rc, ways, to_local, x0, y0, highway_class) if len(rc) else ([],) * 5
            g = build_graph(lines, classes=cls, levels=lev, structs=stt)
            (PACK / "streets" / f"{wid}.pkl").write_bytes(pickle.dumps(g))
            layers = water_layers(feats, tile_window(x0, y0, MARGIN))
            from rasterio.transform import from_origin
            water = rasterize_kinds(layers, from_origin(x0 - MARGIN, y0 + TILE + MARGIN, RES, RES), (256, 256))
            for dem, src in s["dems"].items():
                z, tr, native = window_terrain(src, s["crs"], x0, y0)
                valid = np.isfinite(z)
                f = PACK / "terrain" / f"{wid}__{dem}.npz"
                np.savez_compressed(f, elev=np.nan_to_num(z).astype(np.float32), valid=valid, water=water)
                tif = PACK / "geotiff" / f"{wid}__{dem}.tif"
                write_tif(tif, z, tr, s["crs"])
                rows.append({"window": wid, "site": s["id"], "name": s["name"], "dem": dem,
                             "surface": "DSM" if dem == "glo30" else "DTM", "crs": s["crs"], "xmin": x0, "ymin": y0,
                             "size_m": TILE, "margin_m": MARGIN, "grid_m": RES, "native_res_m": native,
                             "vertical": "EGM2008 orthometric, m" if dem == "glo30" else "REDNAP orthometric (Alicante), m",
                             "resampling": "bilinear reprojection to the 4 m tile grid (adds no detail)",
                             "nodata_share": float(1 - valid.mean()), "source": str(src.relative_to(PROJECT_ROOT)),
                             "source_sha256": sha256(src), "npz_sha256": sha256(f), "tif_sha256": sha256(tif),
                             "water_share": float((water > 0).mean()), "street_km": street_km(g),
                             "street_edges": int(len(g.edges)),
                             "osm_clip": f"data/raw/osm_extracts/cities/{code}.osm.pbf",
                             "osm_extract": s["extract"], "osm_extract_md5": (EXTRACTS / f"{s['extract']}.md5").read_text().split()[0],
                             "attribution": [ATTR[dem], ATTR["osm"]]})
                print(wid, dem, f"nodata {rows[-1]['nodata_share']:.3f} streets {rows[-1]['street_km']:.1f} km", flush=True)
    syn = synthetic_pins()
    CFG.mkdir(parents=True, exist_ok=True)
    man = {"created": time.strftime("%Y-%m-%d"), "version": "m16-pack-1",
           "sites": [{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items() if k != "dems"} | {"dems": list(s["dems"])}
                     for s in SITES],
           "windows": rows, "synthetic": syn,
           "notes": ["GLO-30 is a surface model (buildings, canopy); MDT05 a LiDAR bare-earth DTM. Results are labelled by source.",
                     "Street graphs are reference data only; model inputs never read them.",
                     "KAIST_CAMPUS windows r1c1 equals the KAIST centre window."]}
    (CFG / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    print("manifest", len(rows), "window x source rows")


def synthetic_pins() -> list[dict]:
    from urbanmorph.experiments.sites import _CONF, _DEV, _EVAL
    out = []
    (PACK / "synthetic").mkdir(parents=True, exist_ok=True)
    from urbanmorph.experiments.sites import site_set
    for name, params in (("syn_dev", _DEV), ("syn_eval", _EVAL), ("syn_conf", _CONF)):
        for d in site_set(name):
            f = PACK / "synthetic" / f"{d['site']}.npz"
            np.savez_compressed(f, elev=d["elev"], valid=d["valid"], water=d["water"])
            out.append({"set": name, "site": d["site"], "params": params[d["site"].split("_", 1)[1]],
                        "npz_sha256": sha256(f), "role": {"syn_dev": "development", "syn_eval": "M15 evaluation (inspected)",
                                                         "syn_conf": "M16 confirmation (fresh)"}[name]})
    return out


VEHICULAR_EXTRA = ("service",)
PEDESTRIAN = ("footway", "path", "steps", "cycleway", "pedestrian", "corridor", "bridleway")


def stage_refs():
    """Extra reference graphs per window (never model inputs): ``vehicular`` = the training street definition plus
    service roads (campus and estate roads are mostly ``service``), ``pedestrian`` = footways, paths, steps, cycleways."""
    import geopandas as gpd
    import shapely
    from urbanmorph.graphs.road_graph import build_graph
    man = json.loads((CFG / "manifest.json").read_text())
    extra = {}
    for s in SITES:
        code = f"M16_{s['id']}"
        seq = PACK / f"_{code}_all_highways.geojsonseq"
        if not seq.exists():
            tmp = PACK / f"_{code}_h.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{code}.osm.pbf"), "w/highway"],
                           check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "-o", str(seq), str(tmp)], check=True,
                           capture_output=True)
            tmp.unlink()
        h = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(s["crs"])
        h = h[h.geom_type.isin(["LineString", "MultiLineString"])]
        from urbanmorph.terrain.water import STREET_VALUES
        for wid, x0, y0 in windows(s):
            win = shapely.box(x0, y0, x0 + TILE, y0 + TILE)
            sub = h[h.intersects(win)]
            rec = {}
            for name, keep in (("vehicular", list(STREET_VALUES) + list(VEHICULAR_EXTRA)), ("pedestrian", list(PEDESTRIAN))):
                lines = []
                for geom in sub[sub.highway.isin(keep)].geometry.intersection(win):
                    for part in getattr(geom, "geoms", [geom]):
                        if part.geom_type == "LineString" and part.length > 0:
                            lines.append(np.asarray(part.coords)[:, :2] - [x0, y0])
                g = build_graph(lines)
                (PACK / "streets" / f"{wid}__{name}.pkl").write_bytes(pickle.dumps(g))
                rec[f"{name}_km"] = street_km(g)
            extra[wid] = rec
            print(wid, rec, flush=True)
    for r in man["windows"]:
        r.update(extra.get(r["window"], {}))
    (CFG / "manifest.json").write_text(json.dumps(man, indent=1, default=float))


def stage_audit():
    """Street imprint: median |Laplacian| on carriageway pixels / off-street pixels (M13 rule; 1 = no imprint), per
    window and source; and the share of on-street pixels lying below the 16 m low-pass surface (street beds as troughs)."""
    import geopandas as gpd
    from scipy import ndimage
    from m13_street_erase import corridor_masks, lap_ratio
    man = json.loads((CFG / "manifest.json").read_text())
    out = []
    for r in man["windows"]:
        if r["window"].startswith("KAIST_CAMPUS"):
            continue
        d = dict(np.load(PACK / "terrain" / f"{r['window']}__{r['dem']}.npz"))
        code = f"M16_{r['site']}"
        st = gpd.read_file(PACK / f"_{code}_streets.geojsonseq").set_crs(4326, allow_override=True).to_crs(r["crs"])
        st = st[st.geom_type.isin(["LineString", "MultiLineString"])]

        class W:
            xmin, ymin = r["xmin"], r["ymin"]
        m = corridor_masks(st, W, [3.0])
        on, off = m["carriageway"], ~m[3.0]
        lp = ndimage.gaussian_filter(d["elev"].astype(float), 16.0 / RES)
        below = float(np.mean((d["elev"] - lp)[on & d["valid"]] < 0)) if (on & d["valid"]).any() else np.nan
        below_off = float(np.mean((d["elev"] - lp)[off & d["valid"]] < 0)) if (off & d["valid"]).any() else np.nan
        out.append({"window": r["window"], "dem": r["dem"], "lap_ratio_on_off": lap_ratio(d["elev"], d["valid"], on, off),
                    "below_lowpass_on": below, "below_lowpass_off": below_off})
        print(out[-1], flush=True)
    (CFG / "imprint_audit.json").write_text(json.dumps(out, indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["clips", "build", "refs", "audit", "all"])
    a = ap.parse_args()
    for st, f in (("clips", stage_clips), ("build", stage_build), ("refs", stage_refs), ("audit", stage_audit)):
        if a.stage in (st, "all"):
            f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
