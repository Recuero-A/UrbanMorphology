"""Build a tile's raw OSM cache from a local Geofabrik extract (no API calls).

Why: the public Overpass API is shared infrastructure and was overloaded during
Milestone 2. A Geofabrik ``.osm.pbf`` is a dated snapshot of a region, so
(1) no load is put on public APIs, (2) every tile in the dataset comes from
the same, recorded snapshot, which makes the dataset reproducible.

The steps mirror ``osmnx.graph_from_point(..., dist_type="bbox",
truncate_by_edge=True)`` exactly, so results match the Overpass path:

1. bbox = the same lat/lon box OSMnx would query around the tile centre;
2. ``osmium extract`` a 500 m-buffered box (``smart`` strategy keeps ways and
   multipolygons complete), ``osmium tags-filter`` to the street classes;
3. build the unsimplified graph, truncate to the buffered polygon, simplify,
   truncate to the bbox (keeping edges that cross it) - OSMnx's own order;
4. polygon features: tags-filter to the same combined tags, parse with
   ``features_from_xml`` restricted to the bbox polygon.

Requires the ``osmium`` command (``brew install osmium-tool``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import osmnx as ox

from urbanmorph.config import RAW_DIR, ROAD_HIGHWAY_CLASSES, AreaSpec
from urbanmorph.data.osm import _polygons_only, combined_tags, raw_paths, save_raw_layers

EXTRACT_DIR = RAW_DIR / "osm_extracts"
GEOFABRIK_URL = "https://download.geofabrik.de"

# Overpass matched the street classes as a REGEX ("trunk" also matched
# "trunk_link"); osmium needs explicit values, so list the *_link ramps too.
STREET_VALUES = sorted(
    set(ROAD_HIGHWAY_CLASSES.split("|"))
    | {f"{c}_link" for c in ("motorway", "trunk", "primary", "secondary", "tertiary")}
)


def region_pbf(region: str) -> Path:
    return EXTRACT_DIR / f"{region.replace('/', '_')}-latest.osm.pbf"


def _run(args: list[str]) -> str:
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {res.stderr.strip()[:500]}")
    return res.stdout


def snapshot_info(region: str) -> dict:
    """Provenance of a regional extract: replication timestamp, size, MD5 check."""
    pbf = region_pbf(region)
    header = json.loads(_run(["osmium", "fileinfo", "-j", str(pbf)]))
    opts = header.get("header", {}).get("option", {})
    md5 = hashlib.md5(pbf.read_bytes()).hexdigest()
    expected = None
    md5_file = pbf.with_name(pbf.name + ".md5")
    if md5_file.exists():
        expected = md5_file.read_text().split()[0]
    return {
        "file": pbf.name,
        "url": f"{GEOFABRIK_URL}/{region}-latest.osm.pbf",
        "osm_replication_timestamp": opts.get("osmosis_replication_timestamp") or opts.get("timestamp"),
        "bytes": pbf.stat().st_size,
        "md5": md5,
        "md5_verified": expected == md5 if expected else None,
    }


def city_pbf(code: str, region: str, bounds_wgs84, pad_deg: float = 0.03) -> Path:
    """Cut the city's bounding box (+ ~3 km) out of the regional extract once."""
    out = EXTRACT_DIR / "cities" / f"{code}.osm.pbf"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bounds_wgs84
    bbox = f"{w - pad_deg},{s - pad_deg},{e + pad_deg},{n + pad_deg}"
    _run(["osmium", "extract", "-b", bbox, "--strategy", "smart", "-o", str(out), "--overwrite", str(region_pbf(region))])
    return out


def _tags_filter_expressions(tags: dict) -> list[str]:
    exprs = []
    for key, value in tags.items():
        exprs.append(f"nwr/{key}" if value is True else f"nwr/{key}={','.join(value)}")
    return exprs


def build_tile_from_extract(area: AreaSpec, city_extract: Path, snapshot: dict, force: bool = False) -> dict[str, Path]:
    paths = raw_paths(area)
    if paths["info"].exists() and not force:
        return paths
    work = area.raw_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        # 1. same bbox as osmnx.graph_from_point(dist_type="bbox")
        bbox = ox.utils_geo.bbox_from_point((area.center_lat, area.center_lon), dist=area.download_dist_m)
        poly = ox.utils_geo.bbox_to_poly(bbox)
        poly_proj, crs_utm = ox.projection.project_geometry(poly)
        poly_buff, _ = ox.projection.project_geometry(poly_proj.buffer(500), crs=crs_utm, to_latlong=True)

        # 2. clip + filter with osmium
        tile = work / "tile.osm.pbf"
        w, s, e, n = poly_buff.bounds
        _run(["osmium", "extract", "-b", f"{w},{s},{e},{n}", "--strategy", "smart", "-o", str(tile),
              "--overwrite", str(city_extract)])
        streets_xml = work / "streets.osm"
        _run(["osmium", "tags-filter", str(tile), f"w/highway={','.join(STREET_VALUES)}", "-o", str(streets_xml), "--overwrite"])
        feats_xml = work / "features.osm"
        _run(["osmium", "tags-filter", str(tile), *_tags_filter_expressions(combined_tags()), "-o", str(feats_xml), "--overwrite"])

        # 3. graph, in OSMnx's order of operations. A tile can legitimately have
        # no street of the selected classes (e.g. inside a large campus); that
        # is recorded as an empty graph and flagged later, not treated as a crash.
        try:
            G = ox.graph_from_xml(streets_xml, bidirectional=False, simplify=False, retain_all=True)
            G = ox.truncate.truncate_graph_polygon(G, poly_buff, truncate_by_edge=True)
            G = ox.simplify_graph(G)
            G = ox.truncate.truncate_graph_polygon(G, poly, truncate_by_edge=True)
        except ValueError as exc:
            if "no graph nodes" not in str(exc).lower() and "empty" not in str(exc).lower():
                raise
            G = nx.MultiDiGraph(crs="epsg:4326")

        # 4. polygon features intersecting the bbox
        features = ox.features_from_xml(feats_xml, polygon=poly, tags=combined_tags())
        features = _polygons_only(features)

        return save_raw_layers(area, G, features, {
            "source": "OpenStreetMap via Geofabrik extract (clipped locally with osmium)",
            "retrieved_at_utc": snapshot.get("osm_replication_timestamp"),
            "snapshot": snapshot,
            "city_extract": city_extract.name,
            "feature_request": "osmium tags-filter with the combined tags, split locally by tag",
        })
    finally:
        shutil.rmtree(work, ignore_errors=True)
