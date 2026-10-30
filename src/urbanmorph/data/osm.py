"""Download roads, buildings and green spaces from OpenStreetMap, with caching.

Raw data is stored exactly as received (WGS84 / EPSG:4326) under
``data/raw/<sample_id>/`` so later processing never needs the network:

* ``streets.graphml``   - the street network graph (nodes = intersections,
                          edges = street segments), as returned by OSMnx.
* ``road_areas.geojson``- roads mapped as polygons (pedestrian squares etc.).
* ``buildings.geojson`` - building footprint polygons.
* ``green.geojson``     - park / garden / vegetation polygons.
* ``download_info.json``- query parameters, timestamp, versions, attribution.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError

from urbanmorph.config import (
    BUILDING_TAGS,
    GREEN_TAGS,
    OSM_ATTRIBUTION,
    ROAD_AREA_TAGS,
    ROAD_HIGHWAY_CLASSES,
    AreaSpec,
)

log = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
)

STREETS_FILE = "streets.graphml"
ROAD_AREAS_FILE = "road_areas.geojson"
BUILDINGS_FILE = "buildings.geojson"
GREEN_FILE = "green.geojson"
INFO_FILE = "download_info.json"


def raw_paths(area: AreaSpec) -> dict[str, Path]:
    d = area.raw_dir
    return {
        "streets": d / STREETS_FILE,
        "road_areas": d / ROAD_AREAS_FILE,
        "buildings": d / BUILDINGS_FILE,
        "green": d / GREEN_FILE,
        "info": d / INFO_FILE,
    }


def is_cached(area: AreaSpec) -> bool:
    return all(p.exists() for p in raw_paths(area).values())


def _configure_osmnx(area: AreaSpec) -> None:
    # OSMnx also keeps its own HTTP-response cache; keep it inside data/raw.
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(area.raw_dir / "http_cache")
    ox.settings.requests_timeout = 180
    ox.settings.log_console = False


def _with_retries(fn, what: str, attempts_per_endpoint: int = 3):
    """Call ``fn`` with exponential backoff, rotating Overpass endpoints."""
    last_exc: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint
        for attempt in range(attempts_per_endpoint):
            try:
                return fn()
            except InsufficientResponseError:
                raise  # a valid but empty answer: retrying will not help
            except Exception as exc:  # network errors, timeouts, HTTP 429/504
                last_exc = exc
                wait = 5 * 2**attempt
                log.warning("%s failed via %s (%s); retry in %ss", what, endpoint, exc, wait)
                time.sleep(wait)
    raise RuntimeError(f"Could not download {what} from any Overpass endpoint") from last_exc


def _polygons_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep Polygon/MultiPolygon features (OSM tags can also match points/lines)."""
    if gdf.empty:
        return gdf
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def _features(area: AreaSpec, tags: dict, what: str) -> gpd.GeoDataFrame:
    center = (area.center_lat, area.center_lon)
    try:
        gdf = _with_retries(
            lambda: ox.features_from_point(center, tags=tags, dist=area.download_dist_m),
            what,
        )
    except InsufficientResponseError:
        # Genuinely no matching features: return an empty layer, never invent data.
        log.warning("No %s found in the query area", what)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return _polygons_only(gdf)


def _to_serialisable(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """GeoJSON cannot store list/dict cells; encode them as JSON strings."""
    gdf = gdf.reset_index()  # (element, id) index -> columns, keeps OSM ids
    for col in gdf.columns:
        if col == "geometry" or gdf[col].dtype != object:
            continue
        gdf[col] = gdf[col].map(
            lambda v: json.dumps(v) if isinstance(v, (list, dict, set, tuple)) else v
        )
    return gdf


def download_area(area: AreaSpec, force: bool = False) -> dict[str, Path]:
    """Download (or reuse cached) raw OSM layers for ``area``."""
    paths = raw_paths(area)
    if is_cached(area) and not force:
        log.info("Using cached OSM data in %s", area.raw_dir)
        return paths

    area.raw_dir.mkdir(parents=True, exist_ok=True)
    _configure_osmnx(area)
    center = (area.center_lat, area.center_lon)

    log.info("Downloading street network...")
    graph: nx.MultiDiGraph = _with_retries(
        lambda: ox.graph_from_point(
            center,
            dist=area.download_dist_m,
            dist_type="bbox",
            custom_filter=f'["highway"~"{ROAD_HIGHWAY_CLASSES}"]',
            simplify=True,
            retain_all=True,  # keep disconnected pieces; we want geometry, not routing
            truncate_by_edge=True,
        ),
        "street network",
    )
    ox.save_graphml(graph, paths["streets"])

    log.info("Downloading road areas...")
    road_areas = _features(area, ROAD_AREA_TAGS, "road areas")
    _to_serialisable(road_areas).to_file(paths["road_areas"], driver="GeoJSON")

    log.info("Downloading building footprints...")
    buildings = _features(area, BUILDING_TAGS, "buildings")
    if not buildings.empty:
        buildings = buildings[buildings["building"] != "no"]
    _to_serialisable(buildings).to_file(paths["buildings"], driver="GeoJSON")

    log.info("Downloading green spaces...")
    green = _features(area, GREEN_TAGS, "green spaces")
    _to_serialisable(green).to_file(paths["green"], driver="GeoJSON")

    info = {
        "sample_id": area.sample_id,
        "source": "OpenStreetMap via Overpass API (OSMnx)",
        "attribution": OSM_ATTRIBUTION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overpass_url": ox.settings.overpass_url,
        "osmnx_version": ox.__version__,
        "query": {
            "center_lat": area.center_lat,
            "center_lon": area.center_lon,
            "dist_m": area.download_dist_m,
            "street_filter": ROAD_HIGHWAY_CLASSES,
            "road_area_tags": ROAD_AREA_TAGS,
            "building_tags": {k: str(v) for k, v in BUILDING_TAGS.items()},
            "green_tags": GREEN_TAGS,
        },
        "counts": {
            "street_nodes": graph.number_of_nodes(),
            "street_edges": graph.number_of_edges(),
            "road_area_polygons": int(len(road_areas)),
            "buildings": int(len(buildings)),
            "green_polygons": int(len(green)),
        },
    }
    paths["info"].write_text(json.dumps(info, indent=2))
    return paths


def load_raw(area: AreaSpec) -> dict:
    """Load cached raw layers (no network).

    Returns a dict with keys ``edges``, ``road_areas``, ``buildings``, ``green``
    (GeoDataFrames in EPSG:4326) and ``info`` (download metadata).
    """
    if not is_cached(area):
        raise FileNotFoundError(
            f"No cached OSM data for {area.sample_id!r} in {area.raw_dir}. "
            "Run scripts/download_osm.py first."
        )
    paths = raw_paths(area)
    graph = ox.load_graphml(paths["streets"])
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).reset_index()
    return {
        "edges": edges,
        "road_areas": gpd.read_file(paths["road_areas"]),
        "buildings": gpd.read_file(paths["buildings"]),
        "green": gpd.read_file(paths["green"]),
        "info": json.loads(paths["info"].read_text()),
    }
