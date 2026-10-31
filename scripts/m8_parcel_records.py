"""Milestone 8: per-parcel records of the OBSERVED parcels, with the context used by C and D.

    python scripts/m8_parcel_records.py         # builds data/processed/m8/parcel_records.pkl (git-ignored)

One row per observed parcel (train + val tiles of Spain and the US): the measurements of
outputs/m8/parcels/observed_parcel_table.csv (same rules, layout/parcels.py) plus
* context (layout/parcel_gen.context): nearest street class, log area of the containing block, interior flag;
* nb_cov: mean coverage of the parcels sharing a boundary (NaN without neighbours);
* nb: global row indices of those neighbours (D masks a random subset during training).
Used to FIT C / train D on split == "train" only.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from m6_build_objects import load_objects  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.parcel_gen import context
from urbanmorph.layout.parcels import parcel_neighbours

OBS = PROJECT_ROOT / "data" / "processed" / "m8" / "parcels_observed.pkl"
TABLE = PROJECT_ROOT / "outputs" / "m8" / "parcels" / "observed_parcel_table.csv"
OUT = PROJECT_ROOT / "data" / "processed" / "m8" / "parcel_records.pkl"


def load_records() -> pd.DataFrame:
    if not OUT.exists():
        build()
    return pickle.loads(OUT.read_bytes())


def build():
    tiles = pickle.loads(OBS.read_bytes())
    tab = pd.read_csv(TABLE)
    objs = load_objects()
    graphs = {t["sample_id"]: t["graph"] for t in objs["train"] + objs["val"]}
    cw = objs["class_widths"]
    parts, offset = [], 0
    for sid, d in tab.groupby("sample_id", sort=False):
        polys = tiles[sid]["parcels"]
        assert len(polys) == len(d)
        d = d.reset_index(drop=True).copy()
        d["interior"] = d.frontage_m <= 1.0
        ctx = context(polys, graphs[sid], extract_blocks(graphs[sid], cw), d.interior.to_numpy())
        for k, v in ctx.items():
            if k != "interior":
                d[k] = v
        nbs = parcel_neighbours(polys)
        cov = d.coverage.to_numpy()
        d["nb_cov"] = [cov[nb].mean() if len(nb) else np.nan for nb in nbs]
        d["nb"] = [nb + offset for nb in nbs]
        offset += len(d)
        parts.append(d)
        print(sid, len(d), flush=True)
    rec = pd.concat(parts, ignore_index=True)
    OUT.write_bytes(pickle.dumps(rec))
    return rec


if __name__ == "__main__":
    build()
