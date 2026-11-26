"""Milestone 12, Phase 1: how much water, coastline and crossing infrastructure do our tiles contain?

    python scripts/m12_audit_terrain.py      # -> outputs/m12/audits/terrain_osm_audit.{json,csv}

Everything here comes from the cached OSM city clips (same snapshot as the tiles; osmium CLI). Nothing is
downloaded and the test split is never read (tiles are filtered to train + val before any geometry is touched).

Per tile (512 m, tile CRS). Waterways mapped underground (tunnel=* such as culverts, or a negative layer) are
excluded, so a street over a culverted stream is not counted as a water crossing:
* water share: union of OSM water polygons (natural=water, waterway=riverbank|dock, landuse=reservoir|basin),
  split into river / lake / other by the `water` tag. The SEA is not a polygon in OSM (only natural=coastline
  lines), so coastal tiles are flagged by coastline length, and their sea area is NOT in the water share;
* waterway length by class (river, canal, stream, drain + ditch);
* street-water crossings: tile street centerlines (the model's own data) that intersect a waterway line or
  enter a water polygon, classified by the street's OSM tag: bridge, tunnel, or untagged;
* bridges and tunnels on streets (count of ways, length), and the share of bridge length over water
  (bridge within 5 m of water) - the rest crosses roads, rails or dry valleys;
* `incline` and `layer` tag coverage on streets (city clip ways, clipped to the tile).

Elevation, slope and street grades need a terrain model (DEM) and are NOT computed here (approval gate).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box
from shapely.ops import unary_union

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.data.osm_extract import STREET_VALUES

from urbanmorph.terrain.water import city_features, water_kind  # noqa: E402  (moved to the terrain package)

OUT = PROJECT_ROOT / "outputs" / "m12" / "audits"
NEAR_WATER_M = 5.0


def tagged(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("no", "")


def tile_stats(row, f: dict) -> dict:
    tile = box(row.xmin, row.ymin, row.xmax, row.ymax)
    area = tile.area
    out = {"sample_id": row.sample_id, "city": row.city, "split": row.split}
    w = f["water"][f["water"].intersects(tile)]
    wu = unary_union([g.intersection(tile) for g in w.geometry]) if len(w) else None
    out["water_share"] = wu.area / area if wu is not None else 0.0
    for k in ("river", "lake", "other"):
        sub = w[w.apply(water_kind, axis=1) == k] if len(w) else w
        out[f"water_share_{k}"] = unary_union([g.intersection(tile) for g in sub.geometry]).area / area if len(sub) else 0.0
    ww = f["waterway"][f["waterway"].intersects(tile)]
    ww_cls = ww.waterway.replace({"ditch": "drain"})
    for k in ("river", "canal", "stream", "drain"):
        out[f"waterway_{k}_m"] = float(ww[ww_cls == k].geometry.intersection(tile).length.sum())
    out["coastline_m"] = float(f["coast"][f["coast"].intersects(tile)].geometry.intersection(tile).length.sum())

    rc = gpd.read_file(sample_dir(row.country, row.city) / f"{row.sample_id}_vectors.gpkg", layer="road_centerlines")
    if not isinstance(rc, gpd.GeoDataFrame) or rc.empty:                  # a tile without streets
        rc = gpd.GeoDataFrame({"bridge": [], "tunnel": []}, geometry=gpd.GeoSeries([], crs=row.crs))
    br = (rc.bridge.map(tagged) if "bridge" in rc else pd.Series(False, index=rc.index)).astype(bool)
    tu = (rc.tunnel.map(tagged) if "tunnel" in rc else pd.Series(False, index=rc.index)).astype(bool)
    L = rc.geometry.length
    out["street_m"] = float(L.sum())
    out["bridge_m"], out["tunnel_m"] = float(L[br].sum()), float(L[tu].sum())
    out["bridge_ways"], out["tunnel_ways"] = int(br.sum()), int(tu.sum())
    wet = unary_union(list(w.geometry.intersection(tile.buffer(50))) + list(ww.geometry.intersection(tile.buffer(50))))
    if wet.is_empty:
        out.update(cross_bridge=0, cross_tunnel=0, cross_untagged=0, cross_untagged_river=0, bridge_m_over_water=0.0)
    else:
        hit = rc.intersects(wet)
        out["cross_bridge"] = int((hit & br).sum())
        out["cross_tunnel"] = int((hit & tu & ~br).sum())
        out["cross_untagged"] = int((hit & ~br & ~tu).sum())
        big = unary_union(list(w[w.apply(water_kind, axis=1) == "river"].geometry) + list(ww[ww.waterway.isin(["river", "canal"])].geometry))
        out["cross_untagged_river"] = int((rc.intersects(big) & ~br & ~tu).sum()) if not big.is_empty else 0
        out["bridge_m_over_water"] = float(L[br & rc.intersects(wet.buffer(NEAR_WATER_M))].sum())
    s = f["streets"][f["streets"].intersects(tile)]
    ls = s.geometry.intersection(tile).length
    out["incline_share"] = float(ls[s.incline.apply(tagged)].sum() / max(ls.sum(), 1e-9))
    out["layer_share"] = float(ls[s.layer.apply(tagged)].sum() / max(ls.sum(), 1e-9))
    return out


def q(a, ps=(50, 90, 100)):
    a = np.asarray(a, float)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & idx.split.isin(["train", "val"])]          # the test split is never touched
    assert not (idx.split == "test").any()
    rows = []
    for code, sub in idx.groupby("city_code"):
        f = city_features(code, sub.crs.iloc[0])
        rows += [tile_stats(r, f) for r in sub.itertuples()]
        print(code, len(sub), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "terrain_osm_audit.csv", index=False)
    summ = {"note": "train + val tiles; OSM only (no elevation); sea area excluded from water_share", "cities": {}}
    for (city, split), d in [((c, s), d) for (c, s), d in df.groupby(["city", "split"])] + [(("ALL", "train"), df[df.split == "train"]), (("ALL", "val"), df[df.split == "val"])]:
        summ["cities"].setdefault(city, {})[split] = {
            "tiles": len(d),
            "water_share": q(d.water_share),
            "tiles_water_ge_1pct": int((d.water_share >= 0.01).sum()),
            "tiles_water_ge_10pct": int((d.water_share >= 0.10).sum()),
            "tiles_with_river_or_canal": int(((d.waterway_river_m + d.waterway_canal_m) > 0).sum()),
            "tiles_with_river_polygon": int((d.water_share_river > 0).sum()),
            "tiles_with_lake": int((d.water_share_lake > 0).sum()),
            "tiles_coastal": int((d.coastline_m > 0).sum()),
            "tiles_with_bridge": int((d.bridge_ways > 0).sum()),
            "tiles_with_water_bridge": int((d.bridge_m_over_water > 0).sum()),
            "street_water_crossings": {k: int(d[f"cross_{k}"].sum()) for k in ("bridge", "tunnel", "untagged", "untagged_river")},
            "bridge_share_of_street_length": float(d.bridge_m.sum() / max(d.street_m.sum(), 1e-9)),
            "bridge_length_over_water_share": float(d.bridge_m_over_water.sum() / max(d.bridge_m.sum(), 1e-9)),
            "tunnel_share_of_street_length": float(d.tunnel_m.sum() / max(d.street_m.sum(), 1e-9)),
            "incline_tag_share": q(d.incline_share), "layer_tag_share": q(d.layer_share)}
    (OUT / "terrain_osm_audit.json").write_text(json.dumps(summ, indent=1))
    a = summ["cities"]["ALL"]["train"]
    print(json.dumps(a, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
