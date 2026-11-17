"""Milestone 14 fresh generalisation cities (GEN2): never used for training, selection or evaluation before M14.

    python scripts/m14_build_gen2.py --stage all      # plan, build, dem, prepare (the unchanged M13 GEN stages)

Same pipeline and pre-registered selection rule as scripts/m13_build_gen_cities.py (M2 grid cells inside the city
boundary with GHSL 2020 built-up >= 0.15, at most CAP tiles drawn at random with the M13 seed, no terrain screening),
with a different city list and output directory (data/processed/m14/gen2). Cities were fixed in
docs/M14_EXPERIMENTAL_PROTOCOL.md §2 before any M14 result: four cities outside every earlier split (train, val,
val_t, the sealed test cities, GEN), with a cached OpenStreetMap extract and an open DEM (IGN MDT05, USGS 3DEP),
chosen for contrasting terrain by known geography (not by measured data):
* Alcoi (Spain): mountain town in river gorges;            * Tarragona (Spain): coastal, moderate relief;
* Santa Barbara (USA): coast and steep foothills;          * Sacramento (USA): flat river city.
GEN2 is confirmatory only (never used to tune M14).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m13_build_gen_cities as bg  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

GEN2 = PROJECT_ROOT / "data" / "processed" / "m14" / "gen2"
CITIES = {
    "ESP_ALY": dict(country="spain", city="alcoi", query="Alcoi, Alacant, Comunitat Valenciana, Spain",
                    crs="EPSG:32630", region="europe/spain/valencia", cap=25),
    "ESP_TGN": dict(country="spain", city="tarragona", query="Tarragona, Tarragonès, Catalonia, Spain",
                    crs="EPSG:32631", region="europe/spain/cataluna", cap=25),
    "USA_SBA": dict(country="united_states", city="santa_barbara", query="Santa Barbara, California, United States",
                    crs="EPSG:32611", region="north-america/us/california/socal", cap=25),
    "USA_SAC": dict(country="united_states", city="sacramento", query="Sacramento, California, United States",
                    crs="EPSG:32610", region="north-america/us/california/norcal", cap=25),
}
bg.GEN, bg.CITIES = GEN2, CITIES


def gen2_tiles():
    graphs_f = GEN2 / "graphs3d_gen.pkl"
    import pickle

    import numpy as np
    out = []
    for sid, _, g, block in pickle.loads(graphs_f.read_bytes()):
        d = dict(np.load(GEN2 / "terrain" / f"{sid}.npz"))
        if d["valid"].all():
            out.append({"sid": sid, "split": "gen2", "block": block, "graph": g, "d": d})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "build", "dem", "prepare", "all"])
    ap.add_argument("--cities", nargs="+", default=list(CITIES))
    a = ap.parse_args()
    if a.stage in ("plan", "all"):
        bg.stage_plan(a.cities)
    if a.stage in ("build", "all"):
        bg.stage_build()
    if a.stage in ("dem", "all"):
        bg.stage_dem()
    if a.stage in ("prepare", "all"):
        bg.stage_prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
