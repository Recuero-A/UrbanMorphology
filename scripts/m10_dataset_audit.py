"""Milestone 10, Phase 1: does the street dataset support hierarchical / conditioned generators?

    python scripts/m10_dataset_audit.py     # -> outputs/m10/audit/dataset_audit.json, audit.png

Uses the tile index (all splits for COUNTS and positions only) and the train / val street graphs; no test
geometry is loaded. Per city:
  * tiles per split, spatial blocks (index ``block_id``), clusters of edge-adjacent tiles (independent units);
  * distance from each val / test tile to the nearest train tile (leakage check);
  * street pattern: orientation order phi (M2 metric) quantiles, share grid-like (phi > 0.5) / irregular (< 0.2);
  * street classes: length share per class; continuity of the major + mid network (share of its length in
    components that reach the tile border at >= 2 points, i.e. through-corridors);
  * closed blocks per tile; OSM completeness proxies (building coverage, tiles with streets but almost no buildings).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m6_build_roads import class_widths, load_graphs  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.graphs.road_graph import CLASS_NAMES, components
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "audit"


def clusters(xy, size=512.0):
    """Connected groups of tiles whose squares touch (edge or corner)."""
    n = len(xy)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(xy[i] - xy[j]).max() <= size + 1:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def corridor_share(g):
    if g.edge_class is None or not len(g.edges):
        return np.nan
    m = g.edge_class <= 1
    if not m.any():
        return np.nan
    from urbanmorph.graphs.road_graph import RoadGraph
    sub = RoadGraph(g.nodes, g.edges[m], g.border, g.edge_class[m])
    lab = components(sub)
    seg = sub.segments()
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    used = np.unique(sub.edges.ravel())
    nb = np.bincount(lab[used], weights=sub.border[used].astype(float), minlength=lab.max() + 1)
    comp_of_edge = lab[sub.edges[:, 0]]
    return float(L[nb[comp_of_edge] >= 2].sum() / L.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset]
    graphs = load_graphs()
    cw = class_widths()
    G = {sid: (c, g) for split in ("train", "val") for sid, c, g in graphs[split]}
    res = {"cities": {}}
    for city, d in idx.groupby("city"):
        tr = d[d.split == "train"]
        xy = lambda q: q[["xmin", "ymin"]].to_numpy(float)
        dist = {}
        for sp in ("val", "test"):
            q = d[d.split == sp]
            dist[sp] = [float(np.min(np.abs(xy(tr) - p).max(1))) for p in xy(q)] if len(q) else []
        dv = d[d.split.isin(["train", "val"])]
        phi = dv.orientation_order_phi.to_numpy(float)
        cls_len, corr, nblocks, bc = np.zeros(4), [], [], []
        for sid in dv.sample_id:
            if sid not in G:
                continue
            g = G[sid][1]
            if g.edge_class is not None and len(g.edges):
                seg = g.segments()
                L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
                cls_len += np.bincount(g.edge_class, weights=L, minlength=4)[:4]
            corr.append(corridor_share(g))
            nblocks.append(sum(not b["open"] for b in extract_blocks(g, cw)))
        bcov = dv.building_coverage_vector.to_numpy(float)
        res["cities"][city] = {
            "country": d.country.iat[0],
            "tiles": {s: int((d.split == s).sum()) for s in ("train", "val", "test")},
            "spatial_blocks_train": int(tr.block_id.nunique()),
            "adjacent_tile_clusters_train": clusters(xy(tr)),
            "nearest_train_tile_m": {s: [round(x) for x in v] for s, v in dist.items()},
            "phi_quantiles_10_50_90": [float(q) for q in np.nanpercentile(phi, [10, 50, 90])],
            "share_grid_like_phi_gt_0.5": float(np.nanmean(phi > 0.5)), "share_irregular_phi_lt_0.2": float(np.nanmean(phi < 0.2)),
            "class_length_share": {CLASS_NAMES[k]: float(cls_len[k] / max(cls_len.sum(), 1)) for k in range(4)},
            "major_mid_corridor_share_median": float(np.nanmedian(corr)),
            "closed_blocks_per_tile_median": float(np.median(nblocks)), "closed_blocks_total": int(np.sum(nblocks)),
            "building_coverage_median": float(np.nanmedian(bcov)),
            "tiles_streets_but_building_cov_lt_0.02": int(np.sum((bcov < 0.02) & (dv.street_length_m.to_numpy(float) > 2000))),
        }
        print(city, res["cities"][city]["tiles"], flush=True)
    c = res["cities"]
    res["totals"] = {"train_tiles": int(sum(v["tiles"]["train"] for v in c.values())),
                     "adjacent_tile_clusters_train": int(sum(v["adjacent_tile_clusters_train"] for v in c.values())),
                     "spatial_blocks_train": int(sum(v["spatial_blocks_train"] for v in c.values())),
                     "closed_blocks_train_val": int(sum(v["closed_blocks_total"] for v in c.values())),
                     "min_val_test_to_train_distance_m": float(min(min(v["nearest_train_tile_m"][s] or [1e9]) for v in c.values() for s in ("val", "test")))}
    size = sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "processed").rglob("*") if f.is_file() and "_vectors" in f.name)
    nvec = sum(1 for f in (PROJECT_ROOT / "data" / "processed").rglob("*_vectors.gpkg"))
    res["storage"] = {"vector_tiles": nvec, "vector_bytes_total": size, "mb_per_tile": round(size / max(nvec, 1) / 1e6, 2)}
    (OUT / "dataset_audit.json").write_text(json.dumps(res, indent=1))

    cities = list(c)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=SURFACE)
    idv = idx[idx.split.isin(["train", "val"])]
    axes[0].boxplot([idv[idv.city == k].orientation_order_phi.dropna() for k in cities], tick_labels=cities, orientation="horizontal")
    axes[0].set_title("orientation order φ per tile (train + val)", fontsize=9, loc="left", color=INK)
    left = np.zeros(len(cities))
    for k, col in zip(range(4), ("#2f4f7f", "#5a86c2", "#a9c3e6", "#d8e3f2")):
        v = np.array([c[x]["class_length_share"][CLASS_NAMES[k]] for x in cities])
        axes[1].barh(cities, v, left=left, color=col, label=CLASS_NAMES[k])
        left += v
    axes[1].legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axes[1].set_title("street length by OSM class", fontsize=9, loc="left", color=INK, pad=22)
    axes[2].barh(cities, [c[x]["closed_blocks_per_tile_median"] for x in cities], color="#3d6fb6")
    axes[2].set_title("closed blocks per tile (median)", fontsize=9, loc="left", color=INK)
    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "audit.png", dpi=130, facecolor=SURFACE)
    print(json.dumps(res["totals"], indent=1), res["storage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
