"""The ``urbanmorph-city`` structured city format, v0.1 (docs/M11_3D_ARCHITECTURE.md §3).

A city is a plain JSON-serialisable dict. Frame: x east, y north, z up, metres, origin at the tile's SW corner.
Polygons are ``[exterior, hole, ...]`` rings of ``[x, y]`` (exterior counter-clockwise, holes clockwise).
Every inferable attribute carries a source: observed | derived | generated | synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely

FORMAT, VERSION = "urbanmorph-city", "0.1"
SOURCES = {"observed", "derived", "generated", "synthetic"}
LANDUSE = {"public_green", "private_green", "built", "other_open"}
ROOFS = {"flat", "gable", "hip"}
ROAD_CLASSES = ("major", "mid", "local", "minor")


def rings(poly: shapely.Polygon, nd: int = 3) -> list:
    """Polygon -> [exterior, holes...] with the format's orientation; coordinates rounded to ``nd`` decimals."""
    p = shapely.orient_polygons(poly) if hasattr(shapely, "orient_polygons") else shapely.geometry.polygon.orient(poly, 1.0)
    r = lambda ring: [[round(float(x), nd), round(float(y), nd)] for x, y in np.asarray(ring.coords)[:-1]]
    return [r(p.exterior)] + [r(h) for h in p.interiors]


def polygon(rs: list) -> shapely.Polygon:
    return shapely.Polygon(rs[0], rs[1:])


def polygons(geom) -> list[shapely.Polygon]:
    """Polygon parts of any geometry (empty and non-areal parts dropped)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) for g in polygons(g)]


def new_city(tile_id: str, kind: str, *, crs=None, origin=None, seed=None, provenance=None, storey=None) -> dict:
    assert kind in ("real", "generated")
    if kind == "generated":
        assert crs is None and origin is None, "synthetic cities get no geographic coordinates"
    return {"format": FORMAT, "version": VERSION,
            "metadata": {"tile_id": tile_id, "kind": kind, "frame": "local_tile", "tile_size_m": [512.0, 512.0],
                         "crs": crs, "origin_crs_xy": origin, "seed": seed, "provenance": provenance or {},
                         "units": {"length": "m", "angle": "deg"}, "storey_height_m": storey},
            "nodes": [], "streets": [], "road_surface": None, "blocks": [], "parcels": [], "buildings": [], "landuse": []}


def validate_city(c: dict) -> list[str]:
    """Schema and consistency problems (empty list = valid)."""
    err = []
    if c.get("format") != FORMAT or c.get("version") != VERSION:
        err.append("format/version")
    m = c.get("metadata", {})
    if m.get("kind") == "generated" and (m.get("crs") is not None or m.get("origin_crs_xy") is not None):
        err.append("generated city has geographic coordinates")
    ids = {k: {x["id"] for x in c.get(k, [])} for k in ("streets", "blocks", "parcels", "buildings", "nodes")}
    for k, s in ids.items():
        if len(s) != len(c.get(k, [])):
            err.append(f"duplicate ids in {k}")
    for s in c["streets"]:
        if len(s["centerline"]) < 2 or len(s["segment_widths_m"]) != len(s["centerline"]) - 1:
            err.append(f"street {s['id']}: geometry")
        if s["class"] not in ROAD_CLASSES or not set(s["nodes"]) <= ids["nodes"]:
            err.append(f"street {s['id']}: class/nodes")
    for p in c["parcels"]:
        if p["block"] is not None and p["block"] not in ids["blocks"]:
            err.append(f"parcel {p['id']}: block")
        if p["use"] not in LANDUSE or p["source"] not in SOURCES:
            err.append(f"parcel {p['id']}: use/source")
    for b in c["buildings"]:
        if not set(b["parcels"]) <= ids["parcels"]:
            err.append(f"building {b['id']}: parcels")
        if b["height_source"] not in SOURCES or b["floors_source"] not in SOURCES or b["roof"]["source"] not in SOURCES:
            err.append(f"building {b['id']}: sources")
        if not (b["height_m"] > 0 and b["floors"] >= 1) or b["roof"]["type"] not in ROOFS:
            err.append(f"building {b['id']}: height/floors/roof")
        if not polygon(b["footprint"]).is_valid:
            err.append(f"building {b['id']}: invalid footprint")
    for g in c["landuse"]:
        if g["class"] not in LANDUSE or g["source"] not in SOURCES:
            err.append(f"landuse {g['id']}: class/source")
    return err


def save_city(c: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(c, separators=(",", ":")))


def load_city(path: Path) -> dict:
    c = json.loads(Path(path).read_text())
    assert c.get("format") == FORMAT, path
    return c
