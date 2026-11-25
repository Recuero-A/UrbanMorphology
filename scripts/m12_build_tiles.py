"""Milestone 12, Phase 2b: build the expansion tiles of plan_m12.csv with the unchanged M2 processing.

    python scripts/m12_build_tiles.py                 # train (LA) + val_t (San Diego) tiles
    python scripts/m12_build_tiles.py --authorised-test   # ALSO test_t: only after explicit user authorisation

OSM comes from the cached SoCal extract (same replication state as the M2 LA tiles); the San Diego city clip is cut
locally with osmium (data/raw/osm_extracts/cities/USA_SAN.osm.pbf). Tiles are written next to the M2 tiles and
indexed in a SEPARATE file, data/processed/m12/samples_m12.csv; the M2 and M3b indexes are not modified.
Invariants checked before writing: no new tile within 1,536 m of an M2 val/test tile; new train tiles >= 1,024 m
(centre) from every other train tile; val_t and test_t >= 1,536 m apart.
"""
import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from urbanmorph.data.m12_index import M12_INDEX
from urbanmorph.data.m2_dataset import index_row, load_index_with_expansion, process_sample, save_m2_sample
from urbanmorph.data.osm import is_cached, load_raw
from urbanmorph.data.osm_extract import build_tile_from_extract, city_pbf, snapshot_info
from urbanmorph.sampling.plan_io import area_from_row
from urbanmorph.sampling.study_area import City, fetch_boundary, load_config
from urbanmorph.splits.geographic import edge_distance

PLAN = Path(__file__).resolve().parents[1] / "data" / "processed" / "m12" / "sampling" / "plan_m12.csv"
REGION = "north-america/us/california/socal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorised-test", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    plan = pd.read_csv(PLAN)
    if not a.authorised_test:
        plan = plan[plan.split != "test_t"]
    snap = snapshot_info(REGION)
    clips = {"USA_LAX": city_pbf("USA_LAX", REGION, None),
             "USA_SAN": city_pbf("USA_SAN", REGION, fetch_boundary(City("united_states", "san_diego", "USA_SAN",
                                 "San Diego, San Diego County, California, United States")).total_bounds)}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, skipped = [], []
    for _, r in plan.iterrows():
        area = area_from_row(r, cfg)
        if not is_cached(area):
            build_tile_from_extract(area, clips[r.city_code], snap)
        layout, meta, sample = process_sample(area, load_raw(area), r.to_dict(), cfg, now)
        meta["sampling"]["role"] = "expansion_m12"
        meta["split"], meta["in_dataset"], meta["block_id"] = r.split, True, r.block_id
        if not meta["usable"]:
            skipped.append((r.sample_id, ";".join(f["code"] for f in meta["quality_flags"] if f["severity"] == "error")))
            continue
        save_m2_sample(layout, meta, sample)
        row = index_row(meta)
        row.update(split=r.split, in_dataset=True, role="expansion_m12", block_id=r.block_id,
                   terrain_class_screen=r.terrain_class, landform_p90_screen=r.landform_p90)
        rows.append(row)
        print(r.sample_id, r.split, flush=True)
    new = pd.DataFrame(rows)
    base = load_index_with_expansion()
    held = base[base.in_dataset & base.split.isin(["val", "test"])]
    for _, x in new.iterrows():
        h = held[held.city_code == x.city_code]
        assert all(edge_distance(x, y) >= 1536.0 for _, y in h.iterrows()), f"{x.sample_id} too close to M2 val/test"
    tr = pd.concat([base[base.in_dataset & (base.split == "train")], new[new.split == "train"]])
    for code, g in tr.groupby("city_code"):
        c = np.c_[(g.xmin + g.xmax) / 2, (g.ymin + g.ymax) / 2]
        d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        assert d.min() >= 1024.0 - 1e-6, f"{code}: train tiles too close"
    if (new.split == "test_t").any():
        v, t = new[new.split == "val_t"], new[new.split == "test_t"]
        assert all(edge_distance(x, y) >= 1536.0 for _, x in v.iterrows() for _, y in t.iterrows())
    M12_INDEX.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(M12_INDEX, index=False)
    print(new.groupby(["city", "split"]).size().to_string(), f"\nskipped {len(skipped)}: {skipped[:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
