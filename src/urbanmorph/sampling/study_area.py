"""Pilot configuration, city boundaries and per-city metric CRS."""

from __future__ import annotations

import hashlib
import re
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from urbanmorph.config import PROJECT_ROOT, RAW_DIR

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "m2_pilot.toml"
BOUNDARY_DIR = RAW_DIR / "boundaries"


@dataclass(frozen=True)
class City:
    country: str
    city: str
    code: str
    query: str


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(path)
    cfg["_hash"] = config_hash(path)
    return cfg


def config_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def cities(cfg: dict) -> list[City]:
    return [City(**c) for c in cfg["cities"]]


def fetch_boundary(city: City, force: bool = False) -> gpd.GeoDataFrame:
    """City boundary polygon (EPSG:4326) from OSM via Nominatim, cached as GeoJSON."""
    path = BOUNDARY_DIR / f"{city.code}.geojson"
    if path.exists() and not force:
        return gpd.read_file(path)
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(BOUNDARY_DIR / "http_cache")  # keep raw responses under data/raw
    # A query like "R19631009" is an exact OSM relation ID (used when names are ambiguous).
    by_id = re.fullmatch(r"[NWR]\d+", city.query) is not None
    gdf = ox.geocode_to_gdf(city.query, by_osmid=by_id)  # one Nominatim request; OSMnx rate-limits
    gdf = gdf[["geometry", "osm_type", "osm_id", "display_name", "class", "type"]].copy()
    gdf["query"] = city.query
    gdf.to_file(path, driver="GeoJSON")
    return gdf


def utm_crs(boundary: gpd.GeoDataFrame) -> str:
    """WGS84 / UTM zone containing the boundary centroid, e.g. 'EPSG:32631'."""
    return boundary.estimate_utm_crs(datum_name="WGS 84").to_string()


def boundary_summary(boundary: gpd.GeoDataFrame) -> dict:
    row = boundary.iloc[0]
    area_km2 = float(boundary.to_crs(utm_crs(boundary)).area.sum() / 1e6)
    return {
        "display_name": row["display_name"],
        "osm": f"{row['osm_type']}/{row['osm_id']}",
        "type": f"{row['class']}={row['type']}",
        "area_km2": round(area_km2, 1),
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
