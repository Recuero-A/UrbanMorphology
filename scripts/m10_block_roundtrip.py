"""Milestone 10: representation ceiling of BlockGen's binary building canvas (diagnostic, no criteria).

    python scripts/m10_block_roundtrip.py      # -> outputs/m10/blocks/roundtrip.json

The REAL buildings of each block of the 19 validation tiles are drawn on the block's first canonical window
(2 m / px) and vectorised back exactly as BlockGen's samples are (connected components). Attached buildings
(shared walls) become one component, so the round trip bounds what a perfect binary-mask generator could score.
Test tiles are never read.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m10_eval_blocks import OUT, SKEYS, tile_metrics  # noqa: E402
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.layout.block_canvas import block_frame, from_canvas, to_canvas, windows
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.objects import MIN_BUILDING_M2


def main() -> int:
    data = load_objects()
    cw = data["class_widths"]
    ix = load_index_with_expansion().set_index("sample_id")
    real, rt = [], []
    for t in data["val"]:
        r = ix.loc[t["sample_id"]]
        B = tile_polygons(sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg", r.xmin, r.ymin,
                          "buildings", MIN_BUILDING_M2)
        out = []
        for b in extract_blocks(t["graph"], cw):
            inb = [q for q in (p.intersection(b["polygon"]) for p in B if p.intersects(b["polygon"])) if q.area > 1]
            if not inb:
                continue
            fr = block_frame(b["polygon"])
            win = windows(fr)[0]
            out += [p for p in from_canvas(to_canvas(inb, fr, win) > 0.5, fr, win, clip=b["polygon"])
                    if p.area >= MIN_BUILDING_M2]
        real.append(B)
        rt.append(out)
    co = [t["country"] for t in data["val"]]
    R, T = tile_metrics(data["val"], cw, real, co), tile_metrics(data["val"], cw, rt, co)
    res = {"n_tiles": len(real), "real_median": {k: float(R[k].median()) for k in SKEYS},
           "roundtrip_median": {k: float(T[k].median()) for k in SKEYS}}
    (OUT / "roundtrip.json").write_text(json.dumps(res, indent=1))
    for k in SKEYS:
        print(f"{k:24s} real {res['real_median'][k]:8.3f}  round trip {res['roundtrip_median'][k]:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
