"""Water of a real tile (Milestone 12): OSM polygons, waterway lines and the sea, rasterised on the terrain grid.

* Polygons: natural=water, waterway=riverbank|dock, landuse=reservoir|basin (the Phase 1 audit filter).
* Lines: waterway=river|canal|stream|drain|ditch that are not underground (culverts excluded), buffered by a nominal
  half-width per class (river 6 m, canal 4 m, others 1 m) - an assumption used only where no polygon exists.
* Sea: OSM has no sea polygons; coastline ways have the LAND ON THEIR LEFT. The window is split by the coastlines and
  a piece is sea when the nearest coastline passes it on its right.
Kinds: 1 river / canal / stream, 2 lake / reservoir, 3 sea (a polygon's ``water`` tag decides river vs lake).
"""

from __future__ import annotations

import subprocess

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.osm_extract import STREET_VALUES

CACHE = PROJECT_ROOT / "data" / "processed" / "m12" / "osm"
CLIPS = PROJECT_ROOT / "data" / "raw" / "osm_extracts" / "cities"
FILTERS = ["nwr/natural=water,coastline", "nwr/waterway", "nwr/landuse=reservoir,basin", "w/highway=" + ",".join(STREET_VALUES)]

HALF_WIDTH = {"river": 6.0, "canal": 4.0, "stream": 1.0, "drain": 1.0, "ditch": 1.0}
RIVER, LAKE, SEA = 1, 2, 3


def city_features(code: str, crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Water polygons, waterway lines, coastlines and street ways of one city clip, in the tile CRS."""
    seq = CACHE / f"{code}.geojsonseq"
    if not seq.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"_{code}.osm.pbf"
        subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{code}.osm.pbf"), *FILTERS],
                       check=True, capture_output=True)
        subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "-o", str(seq), str(tmp)], check=True, capture_output=True)
        tmp.unlink()
    g = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(crs)
    for c in ("natural", "waterway", "landuse", "water", "highway", "bridge", "tunnel", "layer", "incline"):
        if c not in g:
            g[c] = None
    poly = g.geom_type.isin(["Polygon", "MultiPolygon"])
    line = g.geom_type.isin(["LineString", "MultiLineString"])
    water = poly & ((g.natural == "water") | g.waterway.isin(["riverbank", "dock"]) | g.landuse.isin(["reservoir", "basin"]))
    underground = g.tunnel.map(tagged) | g.layer.astype(str).str.startswith("-")       # culverts, buried streams
    return {"water": g[water], "waterway": g[line & g.waterway.isin(["river", "canal", "stream", "drain", "ditch"]) & ~underground],
            "coast": g[line & (g.natural == "coastline")], "streets": g[line & g.highway.isin(STREET_VALUES)]}


def water_kind(r) -> str:
    w = str(r.water or "")
    if r.waterway == "riverbank" or w in ("river", "canal", "stream", "oxbow", "ditch", "drain"):
        return "river"
    if w in ("lake", "pond", "reservoir", "lagoon", "basin") or r.landuse in ("reservoir", "basin"):
        return "lake"
    return "other"


def tagged(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("no", "")


def sea_polygons(coast_lines, window) -> list:
    lines = [l.intersection(window) for l in coast_lines if l.intersects(window)]
    lines = [g for l in lines for g in getattr(l, "geoms", [l]) if g.geom_type == "LineString" and g.length > 0]
    if not lines:
        return []
    pieces = shapely.polygonize(shapely.get_parts(shapely.union_all(lines + [window.exterior])))
    out = []
    for poly in shapely.get_parts(pieces):
        p = poly.representative_point()
        best, side = np.inf, 0.0
        for l in lines:
            d = l.distance(p)
            if d < best:
                s = l.project(p)
                a = np.array(l.interpolate(max(s - 1.0, 0)).coords[0])
                b = np.array(l.interpolate(min(s + 1.0, l.length)).coords[0])
                q = np.array(p.coords[0])
                u, v = b - a, q - a
                best, side = d, u[0] * v[1] - u[1] * v[0]
        if side < 0:                                           # right of the coastline: sea
            out.append(poly)
    return out


def water_layers(feats: dict, window) -> dict:
    """Shapely geometries per kind inside ``window`` (tile CRS), from the city feature tables."""
    w = feats["water"][feats["water"].intersects(window)]
    kinds = w.apply(water_kind, axis=1) if len(w) else []
    river = [g for g, k in zip(w.geometry, kinds) if k == "river"]
    lake = [g for g, k in zip(w.geometry, kinds) if k != "river"]
    ww = feats["waterway"][feats["waterway"].intersects(window)]
    river += [g.buffer(HALF_WIDTH.get(k, 1.0)) for g, k in zip(ww.geometry, ww.waterway)]
    sea = sea_polygons(list(feats["coast"].geometry), window)
    clip = lambda gs: shapely.union_all(gs).intersection(window) if gs else shapely.Polygon()
    return {RIVER: clip(river), LAKE: clip(lake), SEA: clip(sea)}


def rasterize_kinds(layers: dict, transform, shape) -> np.ndarray:
    from rasterio.features import rasterize
    out = np.zeros(shape, np.int8)
    for k in (SEA, LAKE, RIVER):                               # rivers drawn last (bridges cross rivers)
        g = layers[k]
        if not g.is_empty:
            out[rasterize([(g, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)] = k
    return out


def tile_window(xmin, ymin, margin=256.0, size=512.0):
    return box(xmin - margin, ymin - margin, xmin + size + margin, ymin + size + margin)
