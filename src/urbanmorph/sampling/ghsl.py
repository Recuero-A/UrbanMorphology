"""GHSL built-up surface (GHS-BUILT-S R2023A) for OSM-independent stratification.

GHS-BUILT-S gives, for each 100 m x 100 m cell (World Mollweide, ESRI:54009),
the square metres covered by buildings, estimated from Landsat/Sentinel-2
imagery, for epochs 1975-2030. It also has a non-residential (NRES) layer.

Why use it: stratifying by OSM building density would confuse "dense" with
"well mapped in OSM". GHSL is derived from satellite imagery, independently
of OSM, so it can define density and era-of-urbanisation strata without that
circularity. Citation: Pesaresi & Politis (2023), doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA.
License: CC BY 4.0.
"""

from __future__ import annotations

import logging
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.windows import from_bounds

from urbanmorph.config import RAW_DIR

log = logging.getLogger(__name__)

GHSL_DIR = RAW_DIR / "ghsl"
MOLLWEIDE = "ESRI:54009"
BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A"
CELL_AREA_M2 = 100.0 * 100.0
# GHSL 100 m tiling: 1000 km x 1000 km tiles, origin at the top-left of the
# Mollweide extent. Verified against downloaded tile bounds (see tile_index tests).
TILE_ORIGIN = (-18_041_000.0, 9_000_000.0)
TILE_SIZE = 1_000_000.0
GHSL_ATTRIBUTION = (
    "GHS-BUILT-S R2023A, European Commission Joint Research Centre (Pesaresi & Politis 2023), CC BY 4.0"
)


def product_name(epoch: int, nres: bool = False) -> str:
    kind = "GHS_BUILT_S_NRES" if nres else "GHS_BUILT_S"
    return f"{kind}_E{epoch}_GLOBE_R2023A_54009_100"


def tile_index(x: float, y: float) -> tuple[int, int]:
    """(row, col), 1-based, of the GHSL tile containing Mollweide point (x, y)."""
    col = int((x - TILE_ORIGIN[0]) // TILE_SIZE) + 1
    row = int((TILE_ORIGIN[1] - y) // TILE_SIZE) + 1
    return row, col


def tiles_for_bounds(bounds_moll: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    xmin, ymin, xmax, ymax = bounds_moll
    r0, c0 = tile_index(xmin, ymax)
    r1, c1 = tile_index(xmax, ymin)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _download(url: str, dest: Path, attempts: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for i in range(attempts):
        try:
            log.info("GET %s", url)
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return
        except Exception as exc:
            log.warning("download failed (%s), retry %d", exc, i + 1)
            time.sleep(5 * 2**i)
    raise RuntimeError(f"could not download {url}")


def tile_tif(epoch: int, row: int, col: int, nres: bool = False) -> Path:
    """Path to a GHSL tile GeoTIFF, downloading and unzipping it if needed."""
    name = product_name(epoch, nres)
    stem = f"{name}_V1_0_R{row}_C{col}"
    tif = GHSL_DIR / "tiles" / f"{stem}.tif"
    if tif.exists():
        return tif
    zpath = tif.with_suffix(".zip")
    if not zpath.exists():
        _download(f"{BASE_URL}/{name}/V1-0/tiles/{stem}.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        member = next(m for m in z.namelist() if m.endswith(".tif"))
        with z.open(member) as src, open(tif, "wb") as dst:
            dst.write(src.read())
    zpath.unlink()  # the GeoTIFF is the cache; the zip would only duplicate it
    return tif


def city_extract(code: str, bounds_wgs84, epoch: int, nres: bool = False, pad_m: float = 2000.0) -> Path:
    """Crop (and mosaic, if needed) GHSL tiles to a city's bounding box."""
    out = GHSL_DIR / code / f"{product_name(epoch, nres)}.tif"
    if out.exists():
        return out
    t = Transformer.from_crs("EPSG:4326", MOLLWEIDE, always_xy=True)
    w, s, e, n = bounds_wgs84
    xs, ys = t.transform([w, e, w, e], [s, s, n, n])
    bb = (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)
    tifs = [tile_tif(epoch, r, c, nres) for r, c in tiles_for_bounds(bb)]
    srcs = [rasterio.open(p) for p in tifs]
    try:
        covered = _union_bounds([s_.bounds for s_ in srcs])
        if not (covered[0] <= bb[0] and covered[1] <= bb[1] and covered[2] >= bb[2] and covered[3] >= bb[3]):
            raise RuntimeError(f"GHSL tiles {tifs} do not cover {bb} (tile index assumption wrong?)")
        mosaic, transform = merge(srcs, bounds=bb)
        profile = {k: v for k, v in srcs[0].profile.items() if k not in ("blockxsize", "blockysize", "tiled")}
        profile |= {"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform, "compress": "deflate"}
    finally:
        for s_ in srcs:
            s_.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic)
    return out


def _union_bounds(bs):
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def builtup_fraction(tif: Path, polygons_bounds: np.ndarray, crs: str, samples_per_side: int = 16) -> np.ndarray:
    """Mean built-up fraction of each square (xmin, ymin, xmax, ymax) in ``crs``.

    Each square is sampled on a regular ``samples_per_side``^2 point grid
    (32 m spacing for 512 m tiles); each point reads the 100 m GHSL cell it falls
    in. The mean of (built-up m2 / 10,000 m2) is the tile's built-up fraction.
    """
    b = np.asarray(polygons_bounds, dtype=float)
    k = samples_per_side
    offs = (np.arange(k) + 0.5) / k
    gx = b[:, [0]] + offs[None, :] * (b[:, [2]] - b[:, [0]])  # (n, k)
    gy = b[:, [1]] + offs[None, :] * (b[:, [3]] - b[:, [1]])
    X = np.repeat(gx[:, None, :], k, axis=1).reshape(len(b), -1)
    Y = np.repeat(gy[:, :, None], k, axis=2).reshape(len(b), -1)
    t = Transformer.from_crs(crs, MOLLWEIDE, always_xy=True)
    mx, my = t.transform(X.ravel(), Y.ravel())
    with rasterio.open(tif) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        rows, cols = rasterio.transform.rowcol(src.transform, mx, my)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    vals = np.full(rows.shape, np.nan)
    vals[inside] = data[rows[inside], cols[inside]]
    if nodata is not None:
        vals[vals == nodata] = np.nan
    frac = np.clip(vals / CELL_AREA_M2, 0, 1).reshape(len(b), -1)
    return np.nanmean(frac, axis=1)
