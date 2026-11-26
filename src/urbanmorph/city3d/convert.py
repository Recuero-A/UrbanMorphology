"""UrbanMorph 2D layouts -> ``urbanmorph-city`` v0.1 (docs/M11_3D_ARCHITECTURE.md §3).

* ``from_real``: a real OSM tile (validation split; the test split is never used) with observed buildings,
  green, road polygons and, for Spanish tiles, observed cadastral parcels (licensed for local use only: city
  files of real tiles are never committed).
* ``from_generated``: an M10 layout (GraphTurtle streets + BF parcels and buildings). No geographic
  coordinates are attached.
Heights: city3d.heights; roofs and styles: city3d.style. Helper shapely objects live under '_' keys and are
stripped before saving.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import shapely
from shapely import affinity

from urbanmorph.city3d.assemble import (TILE, block_of, blocks_from_surface, street_access, street_surface,
                                        streets_from_graph, strip_private)
from urbanmorph.city3d.format import new_city, polygons, rings
from urbanmorph.city3d.heights import assign_heights
from urbanmorph.city3d.mesh import clean_footprint
from urbanmorph.city3d.style import assign_style

PRIVATE_ACCESS = {"private", "no", "customers"}        # as in scripts/m10_landuse_audit.py
MIN_GREEN_M2 = 20.0


def parse_height(v) -> float:
    """OSM height string -> metres (plain number, 'm' or feet); NaN if absent or unparseable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(m|ft|')?$", str(v).strip().lower().replace(",", "."))
    if not m:
        return np.nan
    return float(m.group(1)) * (0.3048 if m.group(2) in ("ft", "'") else 1.0)


def parse_levels(v) -> float:
    try:
        x = float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return x if 0 < x < 200 else np.nan


def _local(gdf, xmin, ymin):
    for _, r in gdf.explode(index_parts=False).iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        for p in polygons(affinity.translate(shapely.make_valid(g), -xmin, -ymin).intersection(TILE)):
            yield r, p


def _finish(c, buildings, parcels, landuse, surface, priors_key, priors, storey, rng):
    blocks = blocks_from_surface(surface, c["streets"])
    for i, p in enumerate(parcels):
        p["id"] = f"p{i}"
        p["block"] = block_of(p["_poly"], blocks)
        acc, front = street_access(p["_poly"], surface, c["streets"])
        p.update(street_access=acc, front_street=front, polygon=rings(p["_poly"]))
        if p.get("_green"):
            p["use"] = "public_green" if acc else "private_green"
    ptree = shapely.STRtree([p["_poly"] for p in parcels]) if parcels else None
    for i, b in enumerate(buildings):
        b["id"] = f"B{i}"
        b["block"] = block_of(b["_poly"], blocks)
        if "parcels" not in b:
            b["parcels"] = [parcels[j]["id"] for j in ptree.query(b["_poly"])
                            if parcels[j]["_poly"].intersection(b["_poly"]).area > 0.2 * b["_poly"].area] if ptree else []
        b["footprint"] = rings(b["_poly"])
    assign_heights(buildings, priors, priors_key, storey["value"], rng)
    for b in buildings:
        b["_storey"] = storey["value"]
    assign_style(buildings, rng)
    for i, g in enumerate(landuse):
        g["id"] = f"g{i}"
        g["polygon"] = rings(g["_poly"])
    c.update(blocks=blocks, parcels=parcels, buildings=buildings, landuse=landuse,
             road_surface=[rings(p) for p in polygons(surface)])
    return c


def _parcel_use(poly, bpolys_tree, bpolys, pub, pri) -> str:
    near = [bpolys[j] for j in bpolys_tree.query(poly)] if bpolys else []
    if any(poly.intersection(q).area >= 0.05 * poly.area or q.intersection(poly).area >= 0.5 * q.area for q in near):
        return "built"
    if not pub.is_empty and poly.intersection(pub).area >= 0.5 * poly.area:
        return "public_green"
    if not pri.is_empty and poly.intersection(pri).area >= 0.5 * poly.area:
        return "private_green"
    return "other_open"


def from_real(tile: dict, row, gpkg, class_widths, priors, storey, observed_parcels=None, seed=0) -> dict:
    """``tile``: an entry of load_objects()['val'] (sample_id, graph); ``row``: its dataset-index row."""
    rng = np.random.default_rng(seed)
    c = new_city(tile["sample_id"], "real", crs=str(row.crs), origin=[float(row.xmin), float(row.ymin)], seed=seed,
                 provenance={"streets": "OSM (Geofabrik extracts, replication 2026-10-28), OSMnx graph, class widths",
                             "road_surface": "OSM road polygons (observed or estimated widths, road_polygons layer)",
                             "buildings": "OSM building footprints", "green": "OSM leisure/landuse/natural",
                             "parcels": "Catastro INSPIRE (observed; local use only)" if observed_parcels else "none",
                             "heights": "OSM height / building:levels, else synthetic (city3d.heights)",
                             "roofs_styles": "synthetic (city3d.style)"},
                 storey=storey)
    c["streets"], c["nodes"] = streets_from_graph(tile["graph"], class_widths)
    roads = gpd.read_file(gpkg, layer="road_polygons")
    surface = shapely.union_all([p for _, p in _local(roads, row.xmin, row.ymin)]) if len(roads) else street_surface(c["streets"])
    buildings = []
    bld = gpd.read_file(gpkg, layer="buildings")
    for r, p in _local(bld, row.xmin, row.ymin):
        q = clean_footprint(p)
        if q is None:
            continue
        buildings.append({"_poly": q, "_obs_height": parse_height(r.get("height")), "_obs_levels": parse_levels(r.get("building:levels"))})
    green = gpd.read_file(gpkg, layer="green")
    pub, pri = [], []
    for r, p in _local(green, row.xmin, row.ymin):
        acc, leis = str(r.get("access") or "").lower(), str(r.get("leisure") or "")
        (pri if acc in PRIVATE_ACCESS or leis == "garden" else pub).append(p)
    U = lambda L: shapely.union_all(L) if L else shapely.Polygon()
    pri_u = U(pri)
    pub_u = U(pub).difference(pri_u)
    landuse = [{"_poly": p, "class": cl, "source": "observed"} for geom, cl in ((pub_u, "public_green"), (pri_u, "private_green"))
               for p in polygons(geom) if p.area >= MIN_GREEN_M2]
    parcels = []
    if observed_parcels:
        bp = [b["_poly"] for b in buildings]
        tree = shapely.STRtree(bp) if bp else None
        for poly in observed_parcels:
            for p in polygons(shapely.make_valid(poly).intersection(TILE)):
                if p.area >= 5:
                    parcels.append({"_poly": p, "use": _parcel_use(p, tree, bp, pub_u, pri_u), "source": "observed",
                                    "use_source": "derived"})
        landuse += [{"_poly": p["_poly"], "class": "other_open", "source": "derived"} for p in parcels if p["use"] == "other_open"]
    return strip_private(_finish(c, buildings, parcels, landuse, surface, row.city, priors, storey, rng))


def from_generated(tile_id: str, graph, state, parcels_bf, class_widths, country, priors, storey, seed, provenance) -> dict:
    """``state``/``parcels_bf``: a LayoutState and BF parcel list (scripts/m10_integrate.bf_layouts)."""
    rng = np.random.default_rng(seed)
    c = new_city(tile_id, "generated", seed=seed, provenance=provenance, storey=storey)
    c["metadata"]["parcel_model_country"] = country
    c["streets"], c["nodes"] = streets_from_graph(graph, class_widths)
    surface = street_surface(c["streets"])
    buildings = []
    bidx = {}
    for k, poly in enumerate(state.buildings):
        q = clean_footprint(poly)
        if q is not None:
            bidx[k] = len(buildings)
            buildings.append({"_poly": q, "_obs_height": np.nan, "_obs_levels": np.nan, "parcels": []})
    parcels = []
    for p in parcels_bf:
        if p["polygon"].is_empty or p["polygon"].area < 1:
            continue
        own = [bidx[k] for k in p.get("building_idx", []) if k in bidx]
        green = p["use"] == "green"
        parcels.append({"_poly": shapely.make_valid(p["polygon"]), "_green": green, "_own": own, "source": "generated",
                        "use": "built" if own else ("public_green" if green else "other_open"), "use_source": "generated"})
    landuse = []
    for g in state.green:
        for p in polygons(shapely.make_valid(g)):
            if p.area >= MIN_GREEN_M2:
                landuse.append({"_poly": p, "class": "public_green", "source": "generated"})
    c = _finish(c, buildings, parcels, landuse, surface, country, priors, storey, rng)
    for p in c["parcels"]:
        for j in p.pop("_own", []):
            c["buildings"][j]["parcels"].append(p["id"])
    pt = shapely.STRtree([p["_poly"] for p in c["parcels"]]) if c["parcels"] else None
    for g in c["landuse"]:                       # green inherits public / private from the parcel it lies on
        if pt is not None:
            best = max(pt.query(g["_poly"]), key=lambda j: c["parcels"][j]["_poly"].intersection(g["_poly"]).area, default=None)
            if best is not None and c["parcels"][best]["use"] == "private_green":
                g["class"] = "private_green"
    landuse_open = [{"_poly": p["_poly"], "class": "other_open", "source": "generated", "id": f"g{len(c['landuse']) + i}",
                     "polygon": p["polygon"]} for i, p in enumerate(q for q in c["parcels"] if q["use"] == "other_open")]
    c["landuse"] += landuse_open
    return strip_private(c)
