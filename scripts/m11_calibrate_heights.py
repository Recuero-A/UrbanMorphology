"""Milestone 11: calibrate the within-block spread of synthetic floors on TRAINING tiles (validation untouched).

    python scripts/m11_calibrate_heights.py      # -> outputs/m11/audit/height_calibration.json, sets "u_sd" in height_priors.json

For 80 training tiles (fixed seed) with >= 50 buildings of known floors: the median within-block SD of observed
floors (blocks with >= 3 known buildings) is compared with that of all-synthetic floors (city3d.heights) for each
U_SD on a grid; the U_SD whose synthetic SD is closest to the observed one (median over tiles) is kept.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from m11_evaluate import _rows, _within_block_sd  # noqa: E402
from urbanmorph.city3d import heights
from urbanmorph.city3d.convert import from_real
from urbanmorph.config import PROJECT_ROOT

AUDIT = PROJECT_ROOT / "outputs" / "m11" / "audit"
GRID = [0.04, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4]


def main() -> int:
    from m6_build_objects import load_objects
    from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
    priors = json.loads((AUDIT / "height_priors.json").read_text())
    storey = priors["storey_height_m"]
    data = load_objects()
    idx = load_index_with_expansion().set_index("sample_id")
    train = data["train"]
    order = np.random.default_rng(0).permutation(len(train))
    obs, syn = [], {u: [] for u in GRID}
    used = 0
    for k in order:
        t = train[k]
        row = idx.loc[t["sample_id"]]
        assert row.split == "train"
        city = from_real(t, row, sample_dir(row.country, row.city) / f"{t['sample_id']}_vectors.gpkg", data["class_widths"],
                         priors, storey[row.city], None, seed=0)
        rows = _rows(city)
        kr = [r for r in rows if r["known"]]
        if len(kr) < 50:
            continue
        o = _within_block_sd(kr, [r["floors_true"] for r in kr])
        if o is None:
            continue
        obs.append(o)
        for u in GRID:
            heights.U_SD = u
            bs = [{"_poly": r["_poly"], "block": r["block"], "_obs_height": np.nan, "_obs_levels": np.nan} for r in rows]
            heights.assign_heights(bs, priors, row.city, storey[row.city]["value"], np.random.default_rng(1))
            syn[u].append(_within_block_sd(rows, [b["floors"] for b in bs]))
        used += 1
        if used >= 80:
            break
    med_obs = float(np.median(obs))
    med_syn = {u: float(np.median(v)) for u, v in syn.items()}
    best = min(GRID, key=lambda u: abs(med_syn[u] - med_obs))
    res = {"n_train_tiles": used, "median_within_block_sd_observed": med_obs, "median_within_block_sd_synthetic": med_syn,
           "chosen_u_sd": best}
    (AUDIT / "height_calibration.json").write_text(json.dumps(res, indent=1))
    priors["u_sd"] = best
    (AUDIT / "height_priors.json").write_text(json.dumps(priors, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
