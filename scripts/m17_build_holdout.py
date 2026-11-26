"""Milestone 17: build the fresh geographic holdout (configs/m17_holdout.json; rule declared before sampling).

Per metro: an OSM clip of the pinned regional extract around the declared centre; candidate 512 m windows on a 512 m
grid aligned to the centre within 2560 m; the OSM public streets of each candidate (training road definition, the
same tile_pieces / build_graph path as the terrain pack); GLO-30 tiles fetched from the public AWS bucket; windows
kept when streets >= 3 km and no GLO-30 no-data; 6 chosen per metro with numpy default_rng(17).

    python scripts/m17_build_holdout.py   ->  data/processed/m17/holdout/{terrain/<wid>.npz, streets/<wid>.pkl},
                                               configs/m17_holdout_windows.json (manifest with checksums)
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from m16_build_terrain_pack import EXTRACTS, MARGIN, RES, TILE, sha256, window_terrain
from urbanmorph.graphs.style_metrics import street_km

OUTD = ROOT / "data" / "processed" / "m17" / "holdout"
GLO_DIR = ROOT / "data" / "raw" / "dem" / "glo30_src"
GLO_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
CLIPS = EXTRACTS / "cities"


def glo_name(lat: int, lon: int) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def glo_mosaic(lat0, lat1, lon0, lon1) -> Path:
    """A GeoTIFF mosaic of the GLO-30 tiles covering the bbox (tiles downloaded once)."""
    GLO_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            n = glo_name(la, lo)
            f = GLO_DIR / f"{n}.tif"
            if not f.exists():
                urllib.request.urlretrieve(GLO_URL.format(n=n), f)
            files.append(str(f))
    out = GLO_DIR / f"m17_mosaic_{lat0:.3f}_{lon0:.3f}.tif"
    if not out.exists():
        import rasterio
        from rasterio.merge import merge
        srcs = [rasterio.open(f) for f in files]
        arr, tr = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(height=arr.shape[1], width=arr.shape[2], transform=tr, driver="GTiff", compress="deflate")
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(arr)
        for x in srcs:
            x.close()
    return out


def main() -> int:
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from urbanmorph.graphs.levels import load_structure_ways, tile_pieces
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    from urbanmorph.terrain.water import STREET_VALUES, city_features, rasterize_kinds, tile_window, water_layers
    from rasterio.transform import from_origin
    cfg = json.loads((ROOT / "configs" / "m17_holdout.json").read_text())
    (OUTD / "terrain").mkdir(parents=True, exist_ok=True)
    (OUTD / "streets").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    man = []
    for m in cfg["metros"]:
        code = f"M17_{m['id']}"
        to_utm = Transformer.from_crs("EPSG:4326", m["crs"], always_xy=True).transform
        back = Transformer.from_crs(m["crs"], "EPSG:4326", always_xy=True).transform
        cx, cy = to_utm(m["lon"], m["lat"])
        R = 2560.0
        pad = R + MARGIN + 300.0
        lo, hi = back(cx - pad, cy - pad), back(cx + pad, cy + pad)
        clip = CLIPS / f"{code}.osm.pbf"
        if not clip.exists():
            subprocess.run(["osmium", "extract", "-O", "-b", f"{lo[0]},{lo[1]},{hi[0]},{hi[1]}", "-o", str(clip),
                            str(EXTRACTS / m["extract"])], check=True, capture_output=True)
        seq = OUTD / f"_{code}_streets.geojsonseq"
        if not seq.exists():
            tmp = OUTD / f"_{code}_s.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(clip), "w/highway=" + ",".join(STREET_VALUES)],
                           check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(seq), str(tmp)],
                           check=True, capture_output=True)
            tmp.unlink()
        sf = OUTD / f"_{code}_struct.geojsonseq"
        if not sf.exists():
            tmp = OUTD / f"_{code}.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(clip), "w/bridge", "w/tunnel", "w/layer", "w/ford"],
                           check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(sf), str(tmp)],
                           check=True, capture_output=True)
            tmp.unlink()
        ways = load_structure_ways(sf)
        streets = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(m["crs"])
        streets = streets[streets.geom_type.isin(["LineString", "MultiLineString"])]
        streets["osmid"] = streets["@id"]
        feats = city_features(code, m["crs"])
        vrt = glo_mosaic(min(lo[1], hi[1]), max(lo[1], hi[1]), min(lo[0], hi[0]), max(lo[0], hi[0]))
        cands = []
        k = int(R // TILE)
        for i in range(-k, k):
            for j in range(-k, k):
                x0, y0 = cx + j * TILE, cy + i * TILE
                if math.hypot(x0 + TILE / 2 - cx, y0 + TILE / 2 - cy) > R:
                    continue
                win = shapely.box(x0, y0, x0 + TILE, y0 + TILE)
                rc = streets[streets.intersects(win)].copy()
                if not len(rc):
                    continue
                rc["geometry"] = rc.geometry.intersection(win)
                lines, cls, lev, stt, _ = tile_pieces(rc, ways, to_utm, x0, y0, highway_class)
                g = build_graph(lines, classes=cls, levels=lev, structs=stt)
                if street_km(g) >= 3.0:
                    cands.append((round(x0, 2), round(y0, 2), g))
        cands.sort(key=lambda c: (c[1], c[0]))
        ok = []
        for x0, y0, g in cands:
            z, _, _ = window_terrain(vrt, m["crs"], x0, y0)
            if np.isfinite(z).all():
                ok.append((x0, y0, g, z))
        pick = sorted(rng.choice(len(ok), min(cfg["windows_per_metro"], len(ok)), replace=False).tolist())
        for n, idx in enumerate(pick):
            x0, y0, g, z = ok[idx]
            wid = f"{m['id']}_{n}"
            layers = water_layers(feats, tile_window(x0, y0, MARGIN))
            water = rasterize_kinds(layers, from_origin(x0 - MARGIN, y0 + TILE + MARGIN, RES, RES), (256, 256))
            f = OUTD / "terrain" / f"{wid}.npz"
            np.savez_compressed(f, elev=z.astype(np.float32), valid=np.isfinite(z), water=water)
            (OUTD / "streets" / f"{wid}.pkl").write_bytes(pickle.dumps(g))
            man.append({"window": wid, "metro": m["id"], "name": m["name"], "crs": m["crs"], "xmin": x0, "ymin": y0,
                        "street_km": street_km(g), "candidates": len(cands), "eligible": len(ok), "npz_sha256": sha256(f),
                        "dem": "Copernicus GLO-30 DSM (EGM2008), bilinear to 4 m", "osm_extract": m["extract"],
                        "osm_extract_md5": (EXTRACTS / f"{m['extract']}.md5").read_text().split()[0]})
            print(wid, f"{street_km(g):.1f} km", flush=True)
        print(m["id"], "candidates", len(cands), "eligible", len(ok), "picked", len(pick), flush=True)
    (ROOT / "configs" / "m17_holdout_windows.json").write_text(json.dumps({"windows": man}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
