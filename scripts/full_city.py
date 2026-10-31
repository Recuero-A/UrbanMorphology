"""One full 512 m tile generated from scratch, drawn large: roads v2 (M6) -> M8 S, and the same streets -> M8 C
(generated parcels drawn) for comparison.

    python scripts/full_city.py --seed 0 --country spain    # -> outputs/compare/full_city_seed0_spain.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_milestones import LIB, OUT, parcels_layout  # noqa: E402
from m6_build_objects import load_objects  # noqa: E402
from m6_evaluate import load_roads  # noqa: E402
from m8_evaluate import run as run_m8  # noqa: E402
from m8_parcel_records import load_records  # noqa: E402
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.layout.export import draw_layout, draw_parcels
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--country", default="spain", choices=["spain", "united_states"])
    a = ap.parse_args()
    t0 = time.time()
    device = pick_device("auto")
    cw = load_objects()["class_widths"]
    graph = sample_graphs(load_roads(0), 1, device="cpu", seed=a.seed)[0]
    s = run_m8("S", 0, [graph], cw, device)[0][0]
    rec = load_records()
    c, parcels = parcels_layout("C", graph, a.country, cw, ShapeLibrary.load(LIB), rec[rec.split == "train"], device, a.seed)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.4), facecolor=SURFACE)
    draw_layout(axes[0], s.graph, s.buildings, s.green, cw, f"M8 S · {len(s.buildings)} buildings, no parcels")
    draw_layout(axes[1], c.graph, c.buildings, c.green, cw,
                f"M8 C · {len(parcels)} generated parcels ({a.country}), {len(c.buildings)} buildings")
    draw_parcels(axes[1], parcels)
    for ax in axes:
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=12, color=INK, pad=8)
    fig.suptitle(f"Fully generated 512 m × 512 m tile (seed {a.seed}): streets from roads v2, same streets in both panels",
                 fontsize=13, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"full_city_seed{a.seed}_{a.country}.png"
    fig.savefig(f, dpi=150, facecolor=SURFACE)
    print(f"{time.time() - t0:.0f}s ->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
