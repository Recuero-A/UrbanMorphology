"""Milestone 5: extract the planar street graph of every TRAIN and VAL tile (test tiles are not read).

    python scripts/m5_build_graphs.py

Writes data/processed/m5/graphs.pkl (git-ignored) and outputs/m5/real_graph_profiles.csv.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import graph_metrics, load_tile_graph

OUT_DATA = PROJECT_ROOT / "data" / "processed" / "m5"
OUT = PROJECT_ROOT / "outputs" / "m5"
GRAPHS = OUT_DATA / "graphs.pkl"


def load_graphs() -> dict:
    """{'train': [(sample_id, country, RoadGraph), ...], 'val': [...]}"""
    return pickle.loads(GRAPHS.read_bytes())


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx["in_dataset"] & idx["split"].isin(["train", "val"])].sort_values("sample_id")
    graphs, rows = {"train": [], "val": []}, []
    for _, r in idx.iterrows():
        g = load_tile_graph(sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg", r.xmin, r.ymin)
        graphs[r.split].append((r.sample_id, r.country, g))
        rows.append({"sample_id": r.sample_id, "country": r.country, "split": r.split, **graph_metrics(g)})
    GRAPHS.write_bytes(pickle.dumps(graphs))
    prof = pd.DataFrame(rows)
    prof.to_csv(OUT / "real_graph_profiles.csv", index=False)
    print({k: len(v) for k, v in graphs.items()})
    print(prof.groupby("split")[["n_nodes", "components", "interior_components", "intersections_per_km2",
                                 "orientation_phi"]].median())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
