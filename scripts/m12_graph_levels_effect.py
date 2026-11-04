"""Milestone 12: how does the 3D-aware street graph change the GraphTurtle training and validation data?

    python scripts/m12_graph_levels_effect.py   # -> outputs/m12/graph_levels/effect.json,
                                                #    data/processed/m12/graphs3d.pkl (same layout as m6/graphs.pkl)

For every train + val tile (test never read) the street graph is built twice from the same centerlines:
* old: the Milestone 5/6 planar snap (``load_tile_graph(..., with_classes=True)``, what N_noise was trained on);
* new: 3D-aware (bridges, tunnels, layers from the OSM ways; graphs/levels.py).
Reported per city and split: tiles whose graph changed, junctions removed, crossing classes of the new graph
(at-grade, portal, grade-separated, ambiguous), street length by structure, and how the tag was found. The S10
reference shift is W1/IQR between the old and new per-tile metric distributions (IQR of the old ones), i.e. how
far the ground truth that S10 compares against has moved; compare it with the model gaps (N at T 0.5: 0.30).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
from pyproj import Transformer

from m10_street_diagnosis import KEYS  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.crossings import street_crossings
from urbanmorph.graphs.levels import STRUCT_NAMES, load_structure_ways, tile_pieces
from urbanmorph.graphs.road_graph import build_graph, graph_metrics, highway_class, load_tile_graph

OUT = PROJECT_ROOT / "outputs" / "m12" / "graph_levels"
STRUCT = PROJECT_ROOT / "data" / "processed" / "m12" / "structures"
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"


def w1(a, b):
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & idx.split.isin(["train", "val"])].sort_values("sample_id")
    assert not (idx.split == "test").any()
    rows, graphs3d = [], {"train": [], "val": []}
    for code, sub in idx.groupby("city_code"):
        ways = load_structure_ways(STRUCT / f"{code}.geojsonseq")
        to_local = Transformer.from_crs("EPSG:4326", sub.crs.iloc[0], always_xy=True).transform
        for r in sub.itertuples():
            f = sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg"
            old = load_tile_graph(f, r.xmin, r.ymin, with_classes=True)
            rc = gpd.read_file(f, layer="road_centerlines")
            if isinstance(rc, gpd.GeoDataFrame) and not rc.empty:
                lines, cls, lev, st, src = tile_pieces(rc, ways, to_local, r.xmin, r.ymin, highway_class)
            else:
                lines, cls, lev, st, src = [], [], [], [], []
            new = build_graph(lines, classes=cls, levels=lev, structs=st)
            graphs3d[r.split].append((r.sample_id, r.country, new))
            plen = np.array([np.hypot(*np.diff(c, axis=0).T).sum() for c in lines]) if lines else np.zeros(0)
            mo, mn = graph_metrics(old), graph_metrics(new)
            x = street_crossings(new)
            changed = not (old.nodes.shape == new.nodes.shape and np.allclose(old.nodes, new.nodes)
                           and np.array_equal(old.edges, new.edges))
            rows.append({"sample_id": r.sample_id, "city": r.city, "split": r.split, "changed": changed,
                         "junctions_old": int(((old.degree() >= 3) & ~old.border).sum()),
                         "junctions_new": int(((new.degree() >= 3) & ~new.border).sum()),
                         **{f"x_{k}": x[k] for k in ("at_grade", "portal", "grade_separated", "ambiguous")},
                         **{f"len_{n}": float(plen[np.array(st) == i].sum()) if len(plen) else 0.0 for i, n in enumerate(STRUCT_NAMES)},
                         **{f"src_{s}": float(plen[np.array(src) == s].sum()) if len(plen) else 0.0 for s in ("way", "overlay", "untagged")},
                         "old": {k: mo[k] for k in KEYS}, "new": {k: mn[k] for k in KEYS}})
        print(code, len(sub), flush=True)
    GRAPHS3D.write_bytes(pickle.dumps(graphs3d))

    def summ(d):
        tot = sum(r["len_ground"] + r["len_bridge"] + r["len_tunnel"] + r["len_passage"] for r in d) or 1.0
        out = {"tiles": len(d), "tiles_changed": sum(r["changed"] for r in d),
               "junctions_old": sum(r["junctions_old"] for r in d), "junctions_new": sum(r["junctions_new"] for r in d),
               **{f"crossings_{k}": sum(r[f"x_{k}"] for r in d) for k in ("at_grade", "portal", "grade_separated", "ambiguous")},
               **{f"length_share_{n}": sum(r[f"len_{n}"] for r in d) / tot for n in STRUCT_NAMES},
               **{f"tag_source_share_{s}": sum(r[f"src_{s}"] for r in d) / tot for s in ("way", "overlay", "untagged")}}
        out["junctions_removed_share"] = 1 - out["junctions_new"] / max(out["junctions_old"], 1)
        return out

    res = {"note": "train + val only; old = planar snap (M5/M6 training data), new = 3D-aware", "by_city": {}, "total": {}}
    for city in sorted({r["city"] for r in rows}):
        res["by_city"][city] = {sp: summ([r for r in rows if r["city"] == city and r["split"] == sp]) for sp in ("train", "val")}
    for sp in ("train", "val"):
        d = [r for r in rows if r["split"] == sp]
        res["total"][sp] = summ(d)
        shift = {}
        for k in KEYS:
            a = np.array([r["old"][k] for r in d], float)
            b = np.array([r["new"][k] for r in d], float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            iqr = np.subtract(*np.percentile(a, [75, 25]))
            shift[k] = w1(a, b) / iqr if iqr > 0 else float("nan")
        res["total"][sp]["reference_shift_w1_iqr"] = shift
        res["total"][sp]["reference_shift_S10"] = float(np.nanmean(list(shift.values())))
    (OUT / "effect.json").write_text(json.dumps(res, indent=1))
    (OUT / "per_tile.json").write_text(json.dumps(rows, indent=0))
    print(json.dumps(res["total"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
