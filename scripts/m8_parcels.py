"""Milestone 8: clip OBSERVED parcels to our train / val tiles and measure them (layout/parcels.py rules).

    python scripts/m8_parcels.py

Inputs: data/raw/parcels/ (scripts/m8_fetch_parcels.py). Output:
* data/processed/m8/parcels_observed.pkl: tile-local parcel polygons (git-ignored; licence: no
  redistribution of the original data);
* outputs/m8/parcels/observed_parcel_stats.json + .csv: DERIVED statistics only (shareable);
* outputs/m8/parcels/observed_examples.png: figure (observed parcels drawn with a solid outline).
Spain: Barcelona, Madrid, Valencia. US: New York, Los Angeles. Test tiles are never read.
"""
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import affinity

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.export import BUILDING_FILL, GROUND, ROAD_FILL, _fill
from urbanmorph.layout.objects import MIN_BUILDING_M2
from urbanmorph.layout.parcels import clean_parcels, parcel_table
from urbanmorph.visualization.layout_plot import INK, SURFACE

RAW = PROJECT_ROOT / "data" / "raw" / "parcels"
OUTD = PROJECT_ROOT / "data" / "processed" / "m8" / "parcels_observed.pkl"
OUT = PROJECT_ROOT / "outputs" / "m8" / "parcels"
SPAIN = {"barcelona": "08900", "madrid": "28900", "valencia": "46900"}


def spain_layer(code):
    g = glob.glob(str(RAW / "spain" / f"A.ES.SDGC.CP.{code}" / "*cadastralparcel.gml"))[0]
    p = gpd.read_file(g)
    p["geometry"] = shapely.make_valid(p.geometry.values)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx["split"].isin(["train", "val"]) & idx["city"].isin(list(SPAIN) + ["new_york", "los_angeles"])]
    layers = {c: spain_layer(code) for c, code in SPAIN.items()}
    objs = load_objects()
    data_graphs = {t["sample_id"]: t["graph"] for t in objs["train"] + objs["val"]}
    tiles, rows, summ = {}, [], {}
    for _, r in idx.iterrows():
        box = shapely.box(r.xmin, r.ymin, r.xmax, r.ymax)
        if r.city in SPAIN:
            lay = layers[r.city]
            box_l = gpd.GeoSeries([box], crs=r.crs).to_crs(lay.crs).iloc[0]
            sub = lay.iloc[lay.sindex.query(box_l, predicate="intersects")].to_crs(r.crs)
            polys = list(sub.geometry)
            source = "Catastro INSPIRE"
        else:
            f = RAW / "us" / r.city / f"{r.sample_id}.geojson"
            if not f.exists():
                continue
            fc = json.loads(f.read_text())
            polys = [shapely.geometry.shape(ft["geometry"]) for ft in fc["features"] if ft.get("geometry")]
            source = "NYC MapPLUTO" if r.city == "new_york" else "LA County Assessor"
        polys = clean_parcels([affinity.translate(shapely.make_valid(p), -r.xmin, -r.ymin) for p in polys])
        b = tile_polygons(sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg", r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        seg = data_graphs[r.sample_id].segments()
        table, s = parcel_table(polys, b, seg)
        tiles[r.sample_id] = {"country": r.country, "city": r.city, "split": r.split, "source": source, "status": "observed",
                              "parcels": polys}
        for t in table:
            rows.append({"sample_id": r.sample_id, "country": r.country, "city": r.city, "split": r.split, **t})
        summ[r.sample_id] = {"city": r.city, "parcels": len(polys), **s}
        print(r.city, r.sample_id, len(polys), flush=True)
    OUTD.parent.mkdir(parents=True, exist_ok=True)
    OUTD.write_bytes(pickle.dumps(tiles))
    df = pd.DataFrame(rows)
    df.drop(columns=[]).to_csv(OUT / "observed_parcel_table.csv", index=False)
    stats = {}
    for c, d in df.groupby("city"):
        full = d[~d.cut]
        fr = full[full.frontage_m > 1]
        one = fr[fr.n_buildings == 1]
        sm = pd.DataFrame(summ).T
        sm = sm[sm.city == c]
        q = lambda x: [round(float(v), 2) for v in np.nanpercentile(x, [10, 50, 90])] if len(x) else None
        stats[c] = {"tiles": int(d.sample_id.nunique()), "parcels": int(len(d)), "uncut_parcels": int(len(full)),
                    "area_m2_p10_p50_p90": q(full.area_m2), "frontage_m_p10_p50_p90": q(fr.frontage_m),
                    "depth_m_p10_p50_p90": q(fr.depth_m), "rectangularity_median": float(full.rectangularity.median()),
                    "share_interior_no_frontage": float((full.frontage_m <= 1).mean()),
                    "buildings_per_parcel_shares": {k: float(v) for k, v in
                                                    full.n_buildings.clip(upper=3).value_counts(normalize=True).sort_index().items()},
                    "coverage_p10_p50_p90_built_parcels": q(full[full.n_buildings > 0].coverage),
                    "front_setback_m_p10_p50_p90": q(one.front_setback_m), "side_setback_m_p10_p50_p90": q(one.side_setback_m),
                    "rear_setback_m_p10_p50_p90": q(one.rear_setback_m),
                    "share_front_setback_lt_1m": float((one.front_setback_m < 1).mean()) if len(one) else None,
                    "share_buildings_spanning_parcels": float(sm.buildings_spanning_parcels.sum() / max(sm.buildings.sum(), 1)),
                    "share_buildings_without_parcel": float(sm.buildings_without_parcel.sum() / max(sm.buildings.sum(), 1))}
    (OUT / "observed_parcel_stats.json").write_text(json.dumps(stats, indent=1))
    # figure: two tiles per city, observed parcels (solid outline) + buildings
    data = {t["sample_id"]: t for t in load_objects()["train"] + load_objects()["val"]}
    cw = load_objects()["class_widths"]
    picks = []
    for c in list(SPAIN) + ["new_york", "los_angeles"]:
        ids = [s for s, t in tiles.items() if t["city"] == c and len(t["parcels"]) > 50][:1]
        picks += ids
    fig, axes = plt.subplots(1, len(picks), figsize=(3.3 * len(picks), 3.6), facecolor=SURFACE)
    for ax, sid in zip(np.atleast_1d(axes), picks):
        r = idx.set_index("sample_id").loc[sid]
        ax.set_facecolor(GROUND)
        _fill(ax, road_polygons(data[sid]["graph"], class_widths=cw), ROAD_FILL, z=1)
        for b in tile_polygons(sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg", r.xmin, r.ymin, "buildings", MIN_BUILDING_M2):
            _fill(ax, b, BUILDING_FILL, z=2)
        for p in tiles[sid]["parcels"]:
            x, y = p.exterior.xy
            ax.plot(x, y, color="#c0392b", lw=0.6, zorder=3)
        ax.set_xlim(156, 356), ax.set_ylim(156, 356), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{tiles[sid]['city'].replace('_', ' ')} · OBSERVED parcels ({tiles[sid]['source']})", fontsize=7, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "observed_examples.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
