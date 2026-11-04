"""Milestone 12, Phase 2: fetch bare-earth elevation for the TRAIN + VALIDATION tiles of Spain and the US.

    python scripts/m12_fetch_dem.py --country spain            # IGN MDT 5 m (WCS), CC BY 4.0
    python scripts/m12_fetch_dem.py --country united_states    # USGS 3DEP bare earth (ImageServer), public domain

Japan is NOT fetched: the GSI terms for derived use of survey results were unclear (docs/M12_DATA_PROTOCOL.md).

Per tile, the request box is the tile plus MARGIN_M on every side, in the tile's UTM CRS (test tiles are filtered
out before any request). Outputs (git-ignored):
* data/raw/dem/<country>/<sample_id>.tif - the provider's raster at its native grid (Spain: 5 m in EPSG:25830;
  US: 1 m requested in the tile CRS from the 3DEP dynamic mosaic);
* data/raw/dem/<country>/<sample_id>.json - provenance: service, coverage / mosaic, request, CRS, vertical datum,
  licence, and for the US the catalogue records of every source raster intersecting the box.
Nothing is resampled here; data/processed/m12/terrain is built by scripts/m12_build_terrain.py.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from pyproj import Transformer

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion

OUT = PROJECT_ROOT / "data" / "raw" / "dem"
MARGIN_M = 256.0
UA = {"User-Agent": "UrbanMorph research (KAIST student project; one request per tile)"}

IGN_WCS = "https://servicios.idee.es/wcs-inspire/mdt"
IGN_COVERAGE = "Elevacion25830_5"
IGN_EDGE = 2.5          # native pixel edges at 2.5 m modulo 5 (DescribeCoverage: origin = pixel centre -19450)
IGN_META = {"provider": "Instituto Geografico Nacional (IGN), Spain", "product": "MDT (5 m grid), INSPIRE WCS coverage "
            + IGN_COVERAGE, "type": "DTM (bare earth, PNOA LiDAR)", "crs": "EPSG:25830 (ETRS89 / UTM 30N, extended)",
            "vertical_datum": "orthometric heights, REDNAP (geoid EGM08-REDNAP)", "native_resolution_m": 5.0,
            "licence": "CC BY 4.0 (scne.es)", "attribution": "MDT05 CC BY 4.0 scne.es",
            "acquisition_date": "per MTN50 sheet, not exposed by the WCS (unknown here)"}
USGS = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
USGS_META = {"provider": "U.S. Geological Survey, 3D Elevation Program", "product": "3DEP Bare Earth DEM dynamic "
             "service (mosaic ByAttribute on 'Best': best available source per location)", "type": "DTM (bare earth)",
             "vertical_datum": "per source record (NAVD88 expected); see 'sources'", "licence": "public domain (credit USGS)",
             "attribution": "USGS National Map 3D Elevation Program (3DEP)"}


def get(url: str, params: dict, path: Path | None = None, tries: int = 4) -> bytes:
    q = url + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=120) as r:
                data = r.read()
            if path is not None:
                path.write_bytes(data)
            return data
        except Exception as e:                                       # transient network errors: back off
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {type(e).__name__}", flush=True)
            time.sleep(5 * (k + 1))


def box(r):
    return r.xmin - MARGIN_M, r.ymin - MARGIN_M, r.xmax + MARGIN_M, r.ymax + MARGIN_M


def fetch_spain(r, tif: Path) -> dict:
    t = Transformer.from_crs(r.crs, "EPSG:25830", always_xy=True)
    x0, y0, x1, y1 = box(r)
    xs, ys = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    snap = lambda v, up: (np.ceil if up else np.floor)((v - IGN_EDGE) / 5.0) * 5.0 + IGN_EDGE
    sub = (snap(min(xs) - 10, False), snap(min(ys) - 10, False), snap(max(xs) + 10, True), snap(max(ys) + 10, True))
    # ^ covers the rotated box, snapped to native pixel edges: an unaligned subset is silently resampled by the WCS
    params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"), ("CoverageId", IGN_COVERAGE),
              ("format", "ArcGrid"), ("subset", f"x({sub[0]:.1f},{sub[2]:.1f})"), ("subset", f"y({sub[1]:.1f},{sub[3]:.1f})")]
    q = IGN_WCS + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=180) as resp:
        body = resp.read().decode("latin-1")
    write_ascii_grid(body, tif, "EPSG:25830")
    return {**IGN_META, "request": q, "subset_epsg25830": sub,
            "format_note": "ArcGrid (mm precision) converted to float32 GeoTIFF; the WCS image/tiff output is int16 (1 m steps)"}


def write_ascii_grid(body: str, tif: Path, crs: str) -> None:
    """Parse the ESRI ASCII grid inside the WCS multipart response and write it as a float32 GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    lines = body[body.index("ncols"):].splitlines()
    hdr, k = {}, 0
    while lines[k].split()[0].lower() in ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "dx", "dy", "nodata_value"):
        key, val = lines[k].split()[:2]
        hdr[key.lower()] = float(val)
        k += 1
    nc, nr = int(hdr["ncols"]), int(hdr["nrows"])
    cs = hdr.get("cellsize", hdr.get("dx"))
    if abs(cs - 5.0) > 1e-6 or abs(hdr.get("dy", cs) - 5.0) > 1e-6 or abs((hdr["xllcorner"] - IGN_EDGE) % 5.0) > 1e-3:
        raise ValueError(f"WCS returned a resampled grid {hdr}")       # never accept silently resampled data
    vals = np.array(" ".join(l for l in lines[k:] if not l.startswith("--")).split()[:nr * nc], np.float32).reshape(nr, nc)
    nd = hdr.get("nodata_value", -9999.0)
    with rasterio.open(tif, "w", driver="GTiff", height=nr, width=nc, count=1, dtype="float32", crs=crs, nodata=nd,
                       transform=from_origin(hdr["xllcorner"], hdr["yllcorner"] + nr * cs, cs, cs), compress="deflate") as d:
        d.write(vals, 1)


def fetch_us(r, tif: Path) -> dict:
    epsg = int(r.crs.split(":")[1])
    x0, y0, x1, y1 = box(r)
    env = f"{x0},{y0},{x1},{y1}"
    params = {"bbox": env, "bboxSR": epsg, "imageSR": epsg, "size": f"{int(x1 - x0)},{int(y1 - y0)}", "format": "tiff",
              "pixelType": "F32", "noData": -9999, "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    get(USGS + "/exportImage", params, tif)
    src = json.loads(get(USGS + "/query", {"geometry": env, "geometryType": "esriGeometryEnvelope", "inSR": epsg,
                                           "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "false", "f": "json",
                                           "outFields": "Name,ProductName,DEM_Type,Source,VerticalDatum,AcquisitionDate,"
                                                        "Resolution_X,Best,LowPS,HighPS,pubdate"}))
    recs = [f["attributes"] for f in src.get("features", [])]
    return {**USGS_META, "crs": r.crs, "requested_resolution_m": 1.0, "request": {"url": USGS + "/exportImage", **params},
            "sources": recs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, choices=["spain", "united_states"])
    ap.add_argument("--limit", type=int, default=None, help="fetch at most this many new tiles (pilot)")
    ap.add_argument("--shard", default="0/1", help="k/n: this process fetches every n-th tile starting at k")
    a = ap.parse_args()
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & idx.split.isin(["train", "val"]) & (idx.country == a.country)]   # never the test split
    assert not (idx.split == "test").any()
    out = OUT / a.country
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    k, n = (int(v) for v in a.shard.split("/"))
    for i, r in enumerate(idx.itertuples()):
        if i % n != k:
            continue
        tif, meta = out / f"{r.sample_id}.tif", out / f"{r.sample_id}.json"
        if tif.exists() and meta.exists():                            # resumable: skip finished tiles
            continue
        info = (fetch_spain if a.country == "spain" else fetch_us)(r, tif)
        info.update(sample_id=r.sample_id, split=r.split, city=r.city, tile_crs=r.crs, margin_m=MARGIN_M,
                    tile_bounds=[r.xmin, r.ymin, r.xmax, r.ymax], fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    bytes=tif.stat().st_size)
        meta.write_text(json.dumps(info, indent=1, default=str))
        done += 1
        if a.limit and done >= a.limit:
            break
        print(f"{r.sample_id} {r.split} {tif.stat().st_size / 1e6:.2f} MB", flush=True)
        time.sleep(0.5)                                               # be polite to public services
    print(f"fetched {done} new tiles; {len(idx)} in total for {a.country}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
