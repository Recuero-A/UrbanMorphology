"""Milestone 12, Phase 2: terrain diversity of the existing train + validation tiles (test never read).

    python scripts/m12_terrain_audit.py     # -> outputs/m12/audits/terrain_audit.{json,csv}, terrain_audit.png

Per tile with a DEM (data/raw/dem; Spain and the US - Japan awaits a licence decision):
* elevation: min, max, range, relief p95 - p5 (tile only, no margin);
* terrain slope at 4 m (the model grid) and at the native DEM resolution (sensitivity): median, p90, p99, area
  share above 5 / 10 / 20 / 30 %;
* valley proxy: topographic position index TPI = z - mean(z) in a 200 m window; share of tile area with
  TPI < -5 m (ground at least 5 m below its surroundings);
* street grades on the 3D-aware graph (data/processed/m12/graphs3d.pkl), ground and passage edges only (bridges
  and tunnels do not follow the terrain): every edge profiled every 2 m on the 4 m grid; length-weighted median and
  p90 of the mean edge grade, share of street length above 5 / 10 / 15 % (max 10 m sub-span grade);
* bridges: runs of connected bridge edges; span; over water (OSM water within 5 m) or not; clearance = straight
  deck between the terrain at the two portals minus the lowest terrain under it (a proxy: deck heights are not in
  OSM). A dry bridge with clearance >= 10 m is counted as valley-spanning (viaduct-like);
* street-water crossings on the 3D-aware graph: bridge / tunnel / unknown (no tag; never assumed ground level).

Terrain class of a tile (fixed before the results were seen): p90 of 4 m slope < 5 % flat, 5-15 % moderate,
>= 15 % steep. "Valley tile": relief >= 40 m and TPI < -5 m on >= 10 % of the area.
AMENDMENT 2026-11-04, after a dry run on 227 tiles and before the final run: the 4 m slope p90 measures urban
micro-relief (embankments, underpasses, retaining walls; Chicago tiles with 2-3 m relief reach 5-8 %) and, in Spain,
1 m terracing of the source mosaic (65 % of Valencia pixels are whole metres). A LANDFORM slope is added: the 4 m
grid block-averaged to 32 m, then central differences; ``terrain_class_landform`` uses the same thresholds on its
p90. Both classes are reported; adequacy is given for both. Quantisation diagnostics per tile: share of native
pixels with whole-metre values, share with zero gradient.
Adequacy rule (fixed before the results; an engineering assumption, not a statistical guarantee): a feature is
adequately represented for TRAINING if >= 30 training tiles from >= 2 cities and >= 10 spatial blocks show it, and
for EVALUATION if >= 5 validation tiles show it.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import uniform_filter
from shapely import box

from m12_audit_terrain import city_features  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import load_index_m12
from urbanmorph.graphs.crossings import water_crossings
from urbanmorph.graphs.levels import BRIDGE, GROUND, PASSAGE
from urbanmorph.terrain.data import tile_terrain
from urbanmorph.terrain.heightfield import Terrain, grade_stats

OUT = PROJECT_ROOT / "outputs" / "m12" / "audits"
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TPI_WINDOW_M, VALLEY_TPI_M = 200.0, -5.0
LANDFORM_M = 32.0
MIN_TRAIN_TILES, MIN_CITIES, MIN_BLOCKS, MIN_VAL_TILES = 30, 2, 10, 5


def slope_stats(t, mask, prefix):
    s = t.slope()[mask]
    s = s[np.isfinite(s)]
    out = {f"{prefix}_median": float(np.median(s)), f"{prefix}_p90": float(np.percentile(s, 90)),
           f"{prefix}_p99": float(np.percentile(s, 99))}
    for th in (0.05, 0.10, 0.20, 0.30):
        out[f"{prefix}_share_gt{int(th * 100)}"] = float((s > th).mean())
    return out


def bridge_runs(g):
    st = g.structs()
    bk = np.flatnonzero(st == BRIDGE)
    parent = {k: k for k in bk}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    by_node = {}
    for k in bk:
        for v in g.edges[k]:
            by_node.setdefault(int(v), []).append(k)
    for ks in by_node.values():
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    runs = {}
    for k in bk:
        runs.setdefault(find(k), []).append(k)
    return list(runs.values())


def tile_row(r, g, feats):
    t = tile_terrain(r.country, r.sample_id, r.crs, r.xmin, r.ymin)
    if t is None:
        return None
    mask = t.window(0, 0, 512, 512)
    z = t.elev[mask]
    row = {"sample_id": r.sample_id, "city": r.city, "split": r.split, "block_id": r.block_id,
           "nodata_share": float(np.isnan(z).mean()), "elev_min": float(np.nanmin(z)), "elev_max": float(np.nanmax(z)),
           "relief_p95_p5": float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5))}
    row["elev_range"] = row["elev_max"] - row["elev_min"]
    row.update(slope_stats(t, mask, "slope4m"))
    tn = tile_terrain(r.country, r.sample_id, r.crs, r.xmin, r.ymin, native=True)
    row.update(slope_stats(tn, tn.window(0, 0, 512, 512), "slope_native"))
    row["native_res_m"] = float(tn.res)
    zn = tn.elev[tn.window(0, 0, 512, 512)]
    zn = zn[np.isfinite(zn)]
    row["quant_integer_share"] = float(np.mean(zn == np.round(zn)))
    gn = tn.slope()[tn.window(0, 0, 512, 512)]
    row["quant_zero_gradient_share"] = float(np.mean(gn[np.isfinite(gn)] == 0))
    k = int(round(LANDFORM_M / t.res))
    H = t.elev.shape[0] // k * k
    coarse = np.nanmean(t.elev[:H, :H].reshape(H // k, k, H // k, k), axis=(1, 3))
    tl = Terrain(coarse.astype(np.float32), t.x0, t.y1, t.res * k)
    row.update(slope_stats(tl, tl.window(0, 0, 512, 512), "slope_landform"))
    w = int(round(TPI_WINDOW_M / t.res)) | 1
    e = np.where(np.isfinite(t.elev), t.elev, np.nanmean(t.elev))
    tpi = (e - uniform_filter(e.astype(float), w, mode="nearest"))[mask]
    row["valley_share"] = float((tpi < VALLEY_TPI_M).mean())
    # street grades (ground and passage edges)
    st = g.structs()
    lens, mean_g, max_g = [], [], []
    for k, (a, b) in enumerate(g.edges):
        if st[k] in (GROUND, PASSAGE):
            s = grade_stats(*t.profile(g.nodes[a], g.nodes[b]))
            if np.isfinite(s["mean_grade"]):
                lens.append(s["length"]), mean_g.append(s["mean_grade"]), max_g.append(s["max_grade"])
    lens, mean_g, max_g = map(np.asarray, (lens, mean_g, max_g))
    if lens.sum() > 0:
        o = np.argsort(mean_g)
        cw = np.cumsum(lens[o]) / lens.sum()
        row["grade_median"] = float(mean_g[o][np.searchsorted(cw, 0.5)])
        row["grade_p90"] = float(mean_g[o][min(np.searchsorted(cw, 0.9), len(o) - 1)])
        for th in (0.05, 0.10, 0.15):
            row[f"grade_share_gt{int(th * 100)}"] = float(lens[max_g > th].sum() / lens.sum())
    # water geometry near the tile, bridges and crossings
    tb = box(r.xmin, r.ymin, r.xmax, r.ymax)
    wp = [gg for gg in feats["water"].geometry if gg.intersects(tb)]
    ww = [gg for gg in feats["waterway"].geometry if gg.intersects(tb)]
    water = shapely.union_all(wp + [gg.buffer(2.0) for gg in ww]) if wp or ww else shapely.Polygon()
    water = shapely.affinity.translate(water, -r.xmin, -r.ymin)
    row.update({f"wx_{k}": v for k, v in water_crossings(g, water).items()})
    runs = bridge_runs(g)
    row["bridge_runs"], row["bridge_runs_water"], row["bridge_runs_valley"] = len(runs), 0, 0
    clear = []
    for ks in runs:
        nodes = np.unique(g.edges[ks].ravel())
        deg_in = np.bincount(g.edges[ks].ravel(), minlength=g.n)[nodes]
        ends = nodes[deg_in == 1] if (deg_in == 1).sum() >= 2 else nodes[[0, -1]]
        pa, pb = g.nodes[ends[0]], g.nodes[ends[-1]]
        za, zb = t.sample(*pa[:, None]), t.sample(*pb[:, None])
        segs = shapely.union_all([shapely.LineString(g.nodes[e]) for e in g.edges[ks]])
        over_water = (not water.is_empty) and segs.distance(water) < 5.0
        lowest = []
        for e in g.edges[ks]:
            d, zz = t.profile(g.nodes[e[0]], g.nodes[e[1]])
            frac = np.linspace(0, 1, len(zz))
            p = g.nodes[e[0]] + frac[:, None] * (g.nodes[e[1]] - g.nodes[e[0]])
            L = np.hypot(*(pb - pa)) or 1.0
            lam = np.clip(((p - pa) @ (pb - pa)) / L ** 2, 0, 1)
            deck = za[0] + lam * (zb[0] - za[0])
            lowest.append(np.nanmax(deck - zz))
        c = float(np.nanmax(lowest)) if lowest else np.nan
        clear.append(c)
        row["bridge_runs_water"] += int(over_water)
        row["bridge_runs_valley"] += int((not over_water) and np.isfinite(c) and c >= 10.0)
    row["bridge_clearance_max"] = float(np.nanmax(clear)) if clear and np.isfinite(clear).any() else np.nan
    return row


def main() -> int:
    idx = load_index_m12()
    idx = idx[idx.in_dataset & idx.split.isin(["train", "val", "val_t"])]
    assert not idx.split.isin(["test", "test_t"]).any()
    graphs = pickle.loads(GRAPHS3D.read_bytes())
    gmap = {sid: g for sp in graphs for sid, _, g in graphs[sp]}
    rows = []
    for code, sub in idx.groupby("city_code"):
        if not any((tile_terrain_path := PROJECT_ROOT / "data" / "raw" / "dem" / c / f"{s}.tif").exists()
                   for c, s in zip(sub.country, sub.sample_id)):
            continue
        feats = city_features(code, sub.crs.iloc[0])
        for r in sub.itertuples():
            x = tile_row(r, gmap[r.sample_id], feats)
            if x is not None:
                rows.append(x)
        print(code, flush=True)
    df = pd.DataFrame(rows)
    df["terrain_class"] = pd.cut(df.slope4m_p90, [-1, 0.05, 0.15, 10], labels=["flat", "moderate", "steep"])
    df["terrain_class_landform"] = pd.cut(df.slope_landform_p90, [-1, 0.05, 0.15, 10], labels=["flat", "moderate", "steep"])
    df["valley_tile"] = (df.relief_p95_p5 >= 40) & (df.valley_share >= 0.10)
    df.to_csv(OUT / "terrain_audit.csv", index=False)

    def desc(d):
        q = lambda c: {p: float(np.nanpercentile(d[c], p)) for p in (10, 50, 90)} if d[c].notna().any() else None
        return {"tiles": len(d), "relief_m": q("relief_p95_p5"), "slope4m_p90": q("slope4m_p90"),
                "slope_native_p90": q("slope_native_p90"), "grade_median": q("grade_median"), "grade_p90": q("grade_p90"),
                "street_length_share_grade_gt10": q("grade_share_gt10"),
                "slope_landform_p90": q("slope_landform_p90"), "quant_integer_share": q("quant_integer_share"),
                "terrain_class": d.terrain_class.value_counts().to_dict(),
                "terrain_class_landform": d.terrain_class_landform.value_counts().to_dict(), "valley_tiles": int(d.valley_tile.sum()),
                "bridge_runs": int(d.bridge_runs.sum()), "bridge_runs_water": int(d.bridge_runs_water.sum()),
                "bridge_runs_valley": int(d.bridge_runs_valley.sum()),
                "water_crossings": {k: int(d[f"wx_{k}"].sum()) for k in ("bridge", "tunnel", "ford", "unknown")}}

    res = {"note": "train + val tiles with a DEM (Spain IGN MDT 5 m, US 3DEP 1 m); Japan pending", "by_city": {}, "all": {}}
    for city, d in df.groupby("city"):
        res["by_city"][city] = {sp: desc(d[d.split == sp]) for sp in ("train", "val", "val_t") if (d.split == sp).any()}
    for sp in ("train", "val", "val_t"):
        res["all"][sp] = desc(df[df.split == sp])
    tr = df[df.split == "train"]
    feats = {"steep_4m": df.terrain_class == "steep", "moderate_or_steep_4m": df.terrain_class.isin(["moderate", "steep"]),
             "steep_landform": df.terrain_class_landform == "steep",
             "moderate_or_steep_landform": df.terrain_class_landform.isin(["moderate", "steep"]),
             "valley": df.valley_tile, "dry_valley_bridge": df.bridge_runs_valley > 0,
             "street_grade_p90_gt10": df.grade_p90 > 0.10}
    adequacy = {}
    for name, m in feats.items():
        mt, mv = m[df.split == "train"], m[df.split.isin(["val", "val_t"])]
        adequacy[name] = {"train_tiles": int(mt.sum()), "train_cities": int(tr[mt].city.nunique()),
                          "train_blocks": int(tr[mt].block_id.nunique()), "val_tiles": int(mv.sum()),
                          "val_t_tiles": int(m[df.split == "val_t"].sum())}
        a = adequacy[name]
        a["adequate_for_training"] = a["train_tiles"] >= MIN_TRAIN_TILES and a["train_cities"] >= MIN_CITIES and a["train_blocks"] >= MIN_BLOCKS
        a["adequate_for_evaluation"] = a["val_tiles"] >= MIN_VAL_TILES
    res["adequacy"] = adequacy
    (OUT / "terrain_audit.json").write_text(json.dumps(res, indent=1, default=str))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cities = sorted(df.city.unique())
    for i, (col, lab) in enumerate([("slope_landform_p90", "tile p90 landform slope (32 m)"), ("relief_p95_p5", "tile relief p95-p5 (m)"),
                                    ("grade_p90", "tile p90 street grade (ground edges)")]):
        ax[i].boxplot([df[df.city == c][col].dropna() for c in cities], tick_labels=cities, orientation="horizontal")
        ax[i].set_xlabel(lab)
        if col != "relief_p95_p5":
            for th in (0.05, 0.15):
                ax[i].axvline(th, color="0.6", lw=0.8, ls="--")
    fig.suptitle("Terrain of the train + val tiles (DEM: IGN MDT 5 m, USGS 3DEP 1 m; dashed: 5 % / 15 %). Japan pending.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "terrain_audit.png", dpi=130)
    print(json.dumps({"all": res["all"], "adequacy": adequacy}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
