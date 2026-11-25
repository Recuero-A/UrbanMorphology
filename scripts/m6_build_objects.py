"""Milestone 6: encode the buildings and green spaces of every TRAIN and VAL tile (test tiles are not read).

    python scripts/m6_build_objects.py

For each tile: its street graph (with classes, from m6_build_roads.py), the road raster drawn from
that graph with per-class widths (the same way generated roads are drawn), and every building
(>= 10 m2) and green polygon (>= 20 m2) as parameters (layout/objects.py) plus its pixels.
Writes data/processed/m6/objects.pkl (git-ignored) and outputs/m6/object_stats.json.
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import shapely
from shapely import affinity

from m6_build_roads import class_widths, load_graphs  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import TILE_M, rasterize_graph
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.layout.placer_data import encode_tile

OBJECTS = PROJECT_ROOT / "data" / "processed" / "m6" / "objects.pkl"


def load_objects() -> dict:
    return pickle.loads(OBJECTS.read_bytes())


def tile_polygons(gpkg, xmin, ymin, layer, min_area):
    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.empty:
        return []
    tile = shapely.box(0, 0, TILE_M, TILE_M)
    out = []
    for geom in gdf.explode(index_parts=False).geometry:
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        p = affinity.translate(shapely.make_valid(geom), -xmin, -ymin).intersection(tile)
        parts = [p] if p.geom_type == "Polygon" else [g for g in getattr(p, "geoms", []) if g.geom_type == "Polygon"]
        out += [q for q in parts if q.area >= min_area]
    return out


def main() -> int:
    t0 = time.time()
    idx = load_index_with_expansion().set_index("sample_id")
    graphs = load_graphs()
    cw = class_widths()
    data = {"class_widths": cw, "train": [], "val": []}
    stats = {}
    for split in ("train", "val"):
        for sid, country, g in graphs[split]:
            r = idx.loc[sid]
            f = sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg"
            b = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
            gr = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
            data[split].append({"sample_id": sid, "country": country, "graph": g,
                                "road_raster": rasterize_graph(g, class_widths=cw).astype(bool),
                                "objects": encode_tile(b, gr)})
            st = stats.setdefault(f"{split}/{country}", {"tiles": 0, "buildings": 0, "green": 0})
            st["tiles"] += 1
            st["buildings"] += len(b)
            st["green"] += len(gr)
        print(split, f"{time.time() - t0:.0f}s", flush=True)
    OBJECTS.write_bytes(pickle.dumps(data))
    (PROJECT_ROOT / "outputs" / "m6" / "object_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
