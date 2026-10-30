"""Milestone 3b: process the expansion tiles into TRAIN-only samples (offline).

Same processing and quality rules as Milestone 2 (``process_sample``). Writes the
arrays/metadata next to the M2 tiles and a SEPARATE index,
data/processed/m2/samples_expansion_m3b.csv. The M2 index (samples.csv) and all
existing split assignments are not modified. Before writing, the script asserts:
  * every expansion tile is >= buffer (1000 m) edge-to-edge from every val/test tile;
  * no expansion tile lies in a spatial block that holds a val/test tile;
  * no expansion tile reuses a grid cell of any existing tile, and none is adjacent
    (centre distance >= spacing) to another train tile.
"""

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from urbanmorph.data.m2_dataset import EXPANSION_INDEX_PATH, index_row, load_index, process_sample, save_m2_sample, write_meta
from urbanmorph.data.osm import is_cached, load_raw
from urbanmorph.quality.checks import dataset_flags
from urbanmorph.sampling.plan_io import area_from_row, load_plan
from urbanmorph.sampling.study_area import load_config
from urbanmorph.splits.geographic import block_ids, edge_distance

PLAN = Path(__file__).resolve().parents[1] / "data" / "processed" / "m2" / "sampling" / "expansion_m3b.csv"
SPACING_M = 1024.0


def main() -> int:
    cfg = load_config()
    plan = load_plan(PLAN)
    base = load_index()
    ds = base[base["in_dataset"]]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows, skipped = [], []
    for _, row in plan.iterrows():
        area = area_from_row(row, cfg)
        if not is_cached(area):
            skipped.append((row.sample_id, "not extracted"))
            continue
        layout, meta, sample = process_sample(area, load_raw(area), row.to_dict(), cfg, now)
        meta["sampling"]["role"] = "expansion_m3b"
        meta["split"], meta["in_dataset"] = "train", True
        meta["block_id"] = row["block_id"]
        if not meta["usable"]:
            skipped.append((row.sample_id, ";".join(f["code"] for f in meta["quality_flags"] if f["severity"] == "error")))
            continue
        save_m2_sample(layout, meta, sample)
        r = index_row(meta)
        r.update(split="train", in_dataset=True, role="expansion_m3b", block_id=row["block_id"])
        rows.append(r)
    exp = pd.DataFrame(rows)

    # ---- invariants: splits untouched, separation preserved
    sp = cfg["splits"]
    held = ds[ds.split.isin(["val", "test"])]
    held_blocks = set(block_ids(held, sp["block_size_m"]))
    assert not set(block_ids(exp, sp["block_size_m"])) & held_blocks, "expansion tile inside a val/test block"
    for code, g in exp.groupby("city_code"):
        h = held[held.city_code == code]
        tr = ds[(ds.city_code == code) & (ds.split == "train")]
        for _, a in g.iterrows():
            assert all(edge_distance(a, b) >= sp["buffer_m"] for _, b in h.iterrows()), f"{a.sample_id} too close to val/test"
        allt = pd.concat([tr, g])
        c = np.c_[(allt.xmin + allt.xmax) / 2, (allt.ymin + allt.ymax) / 2]
        d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        assert d.min() >= SPACING_M - 1e-6, f"{code}: train tiles closer than {SPACING_M} m"
    combined = pd.concat([ds, exp], ignore_index=True)
    problems = {k: v for k, v in dataset_flags(combined, cfg["tile"]["size_m"]).items() if v}
    assert not problems, f"overlap/duplicate problems: {list(problems.items())[:3]}"

    exp.to_csv(EXPANSION_INDEX_PATH, index=False)
    print(exp.groupby(["country", "city"]).size().to_string())
    print(f"\n{len(exp)} expansion train tiles written to {EXPANSION_INDEX_PATH}; skipped {len(skipped)}: {skipped[:5]}")
    print(f"Train tiles: {int((ds.split == 'train').sum())} original + {len(exp)} expansion = {int((ds.split == 'train').sum()) + len(exp)}; "
          f"val/test unchanged ({int((ds.split == 'val').sum())}/{int((ds.split == 'test').sum())})")
    print("Flags:", exp["flags_warning"].fillna("").str.split(";").explode().replace("", None).dropna().value_counts().to_dict(),
          exp["flags_info"].fillna("").str.split(";").explode().replace("", None).dropna().value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
