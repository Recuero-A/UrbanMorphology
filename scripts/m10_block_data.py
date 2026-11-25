"""Milestone 10: BlockGen training / evaluation windows (docs/M10_BLOCK_GENERATION.md §4).

    python scripts/m10_block_data.py     # -> data/processed/m10/block_windows.pkl (git-ignored)

Train and validation tiles (test never read). Every block of the REAL street network (layout/blocks.py,
open blocks included) -> one or more canonical windows (layout/block_canvas.py) with the block mask and the
mask of the real OSM buildings inside the block. Bits are packed (np.packbits).
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import shapely

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.layout.block_canvas import block_frame, to_canvas, windows
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.objects import MIN_BUILDING_M2

OUT = PROJECT_ROOT / "data" / "processed" / "m10" / "block_windows.pkl"


def main() -> int:
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion().set_index("sample_id")
    data = load_objects()
    cw = data["class_widths"]
    cities = sorted(idx[idx.in_dataset & (idx.split == "train")].city.unique())
    rows, blk, bld = [], [], []
    for split in ("train", "val"):
        for t in data[split]:
            sid = t["sample_id"]
            r = idx.loc[sid]
            f = sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg"
            B = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
            tree = shapely.STRtree(B) if B else None
            for b in extract_blocks(t["graph"], cw):
                poly = b["polygon"]
                fr = block_frame(poly)
                inside = [B[i].intersection(poly) for i in tree.query(poly)] if tree is not None else []
                for k, win in enumerate(windows(fr)):
                    rows.append({"split": split, "sample_id": sid, "city": cities.index(r.city), "country": r.country,
                                 "block_id": b["block_id"], "open": b["open"], "frame": fr, "win": win, "win_k": k,
                                 "block_area": poly.area})
                    blk.append(np.packbits(to_canvas([poly], fr, win)))
                    bld.append(np.packbits(to_canvas(inside, fr, win)))
        print(split, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    OUT.write_bytes(pickle.dumps({"rows": rows, "block": np.stack(blk), "building": np.stack(bld), "cities": cities}))
    tr = [r for r in rows if r["split"] == "train"]
    print({"windows": len(rows), "train": len(tr), "train_blocks": len({(r['sample_id'], r['block_id']) for r in tr})})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
