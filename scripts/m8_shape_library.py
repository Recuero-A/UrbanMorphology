"""Milestone 8, Experiment S: fit the real-footprint shape library on TRAIN buildings.

    python scripts/m8_shape_library.py

Writes outputs/m8/shape_library/library.npz (100 x 20 ray fractions + cluster sizes; derived from
OpenStreetMap, ODbL), library.png (all shapes, drawn at their cluster's median aspect) and
library_eval.json (validation footprints: IoU with the nearest class, stretched to their own size).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m6_build_objects import load_objects  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.frontage import interior_angles
from urbanmorph.layout.objects import decode
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m8" / "shape_library"


def footprints(tiles):
    r, lw = [], []
    for t in tiles:
        o = t["objects"]
        b = o["type"] == 0
        r.append(o["ratios"][b])
        lw.append(np.c_[o["length"][b], o["width"][b]])
    return np.concatenate(r), np.concatenate(lw)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_objects()
    tr, tr_lw = footprints(data["train"])
    lib = ShapeLibrary.fit(tr, tr_lw[:, 0] / tr_lw[:, 1])
    lib.save(OUT / "library.npz")
    va, va_lw = footprints(data["val"])
    sel = np.random.default_rng(0).choice(len(va), min(4000, len(va)), replace=False)
    cls = lib.nearest(va[sel])
    ious, acute = [], 0
    for i, c in zip(sel, cls):
        L, W = va_lw[i]
        real = decode(np.zeros(2), 0.0, L, W, va[i])
        p = decode(np.zeros(2), 0.0, L, W, lib.ratios(c))
        u = real.union(p).area
        ious.append(real.intersection(p).area / u if u else 1.0)
        acute += interior_angles(p).min() < 45
    ious = np.array(ious)
    ev = {"shapes": len(lib.shapes), "classes": lib.n_classes, "train_footprints": int(len(tr)),
          "train_share_rectangle": float(lib.counts[0] / len(tr)),
          "val_evaluated": int(len(sel)), "iou_median": float(np.median(ious)), "iou_p10": float(np.percentile(ious, 10)),
          "share_iou_ge_0.9": float((ious >= 0.9).mean()), "share_acute_after_stretch": float(acute / len(sel)),
          "val_class_usage_top10": np.bincount(cls // 4, minlength=len(lib.shapes)).argsort()[::-1][:10].tolist()}
    (OUT / "library_eval.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps(ev, indent=1))
    n = len(lib.shapes)
    cols = 10
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 1.2, np.ceil(n / cols) * 1.3), facecolor=SURFACE)
    for k, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if k >= n:
            continue
        p = decode(np.zeros(2), 0.0, 20.0, 13.0, lib.shapes[k])
        x, y = p.exterior.xy
        ax.fill(x, y, color="#3c3c43")
        ax.set_aspect("equal")
        ax.set_xlim(-12, 12), ax.set_ylim(-12, 12)
        ax.set_title(f"{k} · {lib.counts[k]:,}", fontsize=6, color=INK)
    fig.suptitle("Real-footprint shape library (train buildings; number = cluster size); drawn at 20 x 13 m", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "library.png", dpi=130, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
