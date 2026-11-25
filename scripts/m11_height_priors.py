"""Milestone 11: empirical floor-count priors for synthetic building heights (TRAIN tiles only).

    python scripts/m11_height_priors.py      # -> outputs/m11/audit/height_priors.json

Per city and footprint-area bin: quantiles of the floor count of buildings whose floors are observed
(`building:levels`) or derived (`height` / city storey height, see heights.storey_table). Pooled per country too.
Only training tiles are read, so validation tiles rendered later never inform their own priors.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyogrio
import shapely

from m11_height_audit import to_levels, to_m  # noqa: E402
from urbanmorph.city3d.heights import AREA_BINS, QS, storey_table
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir

OUT = PROJECT_ROOT / "outputs" / "m11" / "audit"


def main() -> int:
    storey = storey_table(json.loads((OUT / "height_audit.json").read_text()))
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & (idx.split == "train")]
    samples = {}                                           # (key, bin) -> floors
    for _, r in idx.iterrows():
        f = sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg"
        if not f.exists():
            continue
        df = pyogrio.read_dataframe(f, layer="buildings")
        col = lambda c: df[c] if c in df.columns else [None] * len(df)
        sh = storey[r.city]["value"]
        for g, h, lv in zip(df.geometry, col("height"), col("building:levels")):
            if g is None or g.is_empty:
                continue
            fl = to_levels(lv)
            if not np.isfinite(fl):
                hm = to_m(h)
                fl = round(hm / sh) if np.isfinite(hm) and 2.0 <= hm <= 400 and r.city != "chicago" else np.nan
            if not np.isfinite(fl) or fl < 1:
                continue
            b = int(np.searchsorted(AREA_BINS, shapely.area(g), side="right"))
            for key in (r.city, r.country):
                samples.setdefault(key, {}).setdefault(b, []).append(float(fl))
    res = {"note": "train tiles only; floors per footprint-area bin (edges m2: %s)" % AREA_BINS, "quantile_levels": QS,
           "storey_height_m": storey, "priors": {}}
    for key, bins in samples.items():
        res["priors"][key] = {str(b): {"n": len(v), "q": [float(x) for x in np.quantile(v, QS)]} for b, v in sorted(bins.items())}
        print(key, {b: (d["n"], d["q"][len(QS) // 2]) for b, d in res["priors"][key].items()})
    (OUT / "height_priors.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
