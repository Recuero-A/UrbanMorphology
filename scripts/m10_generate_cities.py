"""Milestone 10: generate new complete tiles with the recommended pipeline (no evaluation, no criteria).

    python scripts/m10_generate_cities.py --n 10 --seed 2026     # -> outputs/m10/cities/

Streets: GraphTurtle N (denoising context noise, checkpoint N_noise_s0) sampled at temperature 0.5, from a fresh
sampling seed (not the cached evaluation samples). Tiles whose street graph is too sparse to form blocks
(< MIN_EDGES edges) are skipped and counted. Blocks: BF (procedural parcels + buildings, data-based green fix)
with the M8 parcel net and M9 block nets of seed 0. Countries alternate Spain / US (BF has parcel models for
those two only); the street model is unconditioned, so only parcels and buildings follow the country.
One figure per tile (streets, parcels, buildings, green), an overview, and a GeoPackage per tile
(outputs/m10/cities/export/, git-ignored). Test tiles are never read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m6_build_objects import load_objects  # noqa: E402
from m10_eval_streets import CK, load_model  # noqa: E402
from m10_integrate import bf_layouts  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.layout.export import draw_layout, draw_parcels, export_parcel_layout
from urbanmorph.visualization.layout_plot import SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "cities"
MIN_EDGES = 100
TEMPERATURE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    (OUT / "export").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cw = load_objects()["class_widths"]
    model = load_model(CK / "m10" / "N_noise_s0")
    graphs = sample_graphs(model, 2 * a.n, seed=a.seed, temperature=TEMPERATURE)
    keep = [g for g in graphs if len(g.edges) >= MIN_EDGES][:a.n]
    countries = [("spain", "united_states")[i % 2] for i in range(len(keep))]
    layouts = bf_layouts(keep, countries, cw, seed=0)
    info = []
    for i, ((st, P), c) in enumerate(zip(layouts, countries)):
        name = f"city_{i + 1:02d}_{c}"
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=SURFACE)
        draw_layout(ax, st.graph, st.buildings, st.green, cw, f"{name} · generated streets (N, T = 0.5) + BF parcels")
        draw_parcels(ax, P)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=130, facecolor=SURFACE)
        plt.close(fig)
        export_parcel_layout(OUT / "export" / f"{name}.gpkg", st.graph, P, st.buildings, st.green, cw)
        info.append({"name": name, "country": c, "street_edges": int(len(st.graph.edges)), "parcels": len(P),
                     "buildings": len(st.buildings), "green_areas": len(st.green)})
    cols = 5
    rows = -(-len(layouts) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), facecolor=SURFACE)
    for ax, (st, P), d in zip(np.ravel(axes), layouts, info):
        draw_layout(ax, st.graph, st.buildings, st.green, cw, d["name"])
        draw_parcels(ax, P)
    for ax in np.ravel(axes)[len(layouts):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=110, facecolor=SURFACE)
    res = {"seed": a.seed, "temperature": TEMPERATURE, "street_model": "N_noise_s0", "sampled": len(graphs),
           "skipped_sparse": int(sum(len(g.edges) < MIN_EDGES for g in graphs)), "tiles": info,
           "seconds": round(time.time() - t0, 1)}
    (OUT / "cities.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
