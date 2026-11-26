"""Load provider DEMs onto the tile grid (Milestone 12).

``tile_terrain`` reprojects the raw provider raster (data/raw/dem/<country>/<sample_id>.tif, written by
scripts/m12_fetch_dem.py) into the tile's UTM CRS on a ``res``-metre grid covering the tile plus ``margin`` metres,
bilinear. The provenance JSON travels with the result, and ``native_res`` records the source resolution: the 4 m
grid is a resampling, not 4 m data. Vertical datums are not converted (one datum per country; cross-country
analyses use slopes and relative elevations only, which a constant offset does not change).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.terrain.heightfield import Terrain

DEM_DIR = PROJECT_ROOT / "data" / "raw" / "dem"
TILE_M = 512.0


def tile_terrain(country: str, sample_id: str, crs: str, xmin: float, ymin: float, res: float = 4.0,
                 margin: float = 256.0, native: bool = False) -> Terrain | None:
    """``native=True`` keeps the source resolution (for the resolution-sensitivity analysis)."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    tif = DEM_DIR / country / f"{sample_id}.tif"
    if not tif.exists():
        return None
    prov = json.loads(tif.with_suffix(".json").read_text())
    with rasterio.open(tif) as src:
        src_res = float(abs(src.res[0]))
        r = src_res if native else res
        n = int(round((TILE_M + 2 * margin) / r))
        dst = np.full((n, n), np.nan, np.float32)
        transform = from_origin(xmin - margin, ymin + TILE_M + margin, r, r)
        reproject(rasterio.band(src, 1), dst, src_nodata=src.nodata, dst_transform=transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    if prov.get("provider", "").startswith("U.S."):
        dst[dst < -1000] = np.nan                                  # -9999 no-data that survived resampling
    return Terrain(dst, -margin, TILE_M + margin, r, native_res=prov.get("native_resolution_m", src_res),
                   provenance=prov)
