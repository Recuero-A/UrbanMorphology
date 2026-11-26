"""Milestone 14: street-erased DEMs (leakage control, TNse backbone input) for the GEN2 tiles, with the factor frozen
on training tiles in M13 (4.0). Same erasure as scripts/m13_street_erase.py (corridor mask, biharmonic fill,
transplanted micro-relief, per-tile seeded); output data/processed/m13/terrain_se/f4/<sid>.npz (the path the TNse
runner reads). Model input only: rules and metrics keep the original DEM.

    python scripts/m14_street_erase_gen2.py
"""
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import m13_street_erase as se  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.terrain.water import city_features

F = 4.0
GEN2 = PROJECT_ROOT / "data" / "processed" / "m14" / "gen2"


def main() -> int:
    idx = pd.read_csv(GEN2 / "samples.csv")
    assert not idx.split.isin(["test", "test_t"]).any()
    out = se.OUT / f"f{F:g}"
    out.mkdir(parents=True, exist_ok=True)
    for code, sub in idx.groupby("city_code"):
        feats = city_features(code, sub.crs.iloc[0])["streets"]
        for r in sub.itertuples():
            src = GEN2 / "terrain" / f"{r.sample_id}.npz"
            dst = out / f"{r.sample_id}.npz"
            if not src.exists() or dst.exists():
                continue
            d = dict(np.load(src))
            rng = np.random.default_rng(zlib.crc32(r.sample_id.encode()))
            m = se.corridor_masks(feats, r, [F])[F]
            e = se.erase(d["elev"], d["valid"], m, rng) if d["valid"].all() else d["elev"]
            np.savez_compressed(dst, **{**d, "elev": e})
        print(code, len(sub), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
