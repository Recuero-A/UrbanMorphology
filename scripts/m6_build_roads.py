"""Milestone 6: street graphs WITH street classes for train and val tiles (test tiles are not read).

    python scripts/m6_build_roads.py

Writes data/processed/m6/graphs.pkl (git-ignored; the Milestone 5 file is left untouched) and
outputs/m6/class_widths.json: median OSM road width per street class, from TRAINING tiles only.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import CLASS_NAMES, highway_class, load_tile_graph

OUT_DATA = PROJECT_ROOT / "data" / "processed" / "m6"
OUT = PROJECT_ROOT / "outputs" / "m6"
GRAPHS = OUT_DATA / "graphs.pkl"


def load_graphs() -> dict:
    return pickle.loads(GRAPHS.read_bytes())


def class_widths() -> list[float]:
    return json.loads((OUT / "class_widths.json").read_text())["median_width_m"]


def main() -> int:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx["in_dataset"] & idx["split"].isin(["train", "val"])].sort_values("sample_id")
    graphs = {"train": [], "val": []}
    widths = {c: [] for c in range(4)}
    for _, r in idx.iterrows():
        f = sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg"
        graphs[r.split].append((r.sample_id, r.country, load_tile_graph(f, r.xmin, r.ymin, with_classes=True)))
        if r.split == "train":
            rp = gpd.read_file(f, layer="road_polygons")
            if rp.empty:
                continue
            for tag, w in zip(rp["highway"], rp["width_m"]):
                if np.isfinite(w):
                    widths[highway_class(tag)].append(float(w))
    GRAPHS.write_bytes(pickle.dumps(graphs))
    med = [float(np.median(widths[c])) for c in range(4)]
    (OUT / "class_widths.json").write_text(json.dumps({"classes": CLASS_NAMES, "median_width_m": med,
                                                       "n_polygons": [len(widths[c]) for c in range(4)]}, indent=1))
    share = np.bincount(np.concatenate([g.classes() for _, _, g in graphs["train"]]), minlength=4)
    print({k: len(v) for k, v in graphs.items()}, "class widths", med, "edge share", (share / share.sum()).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
