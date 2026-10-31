"""New cities fully from scratch: street graph (M6 roads v2) -> TZ (M9 tidy parcels + zoning + templates).

    python scripts/generate_tz.py --n 5 --seed 0              # -> outputs/compare/tz_cities_seed0.png + .gpkg per city
    python scripts/generate_tz.py --n 5 --seed 0 --method B   # M9b -> outputs/compare/b_cities_seed0.png

Seed-0 checkpoints (roads_v2_s0, block_nets_s0, parcel_net_s0). Countries alternate Spain / US (TZ was
fitted on those two). Every panel draws the generated parcels; green parcels are filled.
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
import torch

from m6_build_objects import load_objects  # noqa: E402
from m6_evaluate import load_roads  # noqa: E402
from m8_parcel_records import load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.layout.export import draw_layout, draw_parcels, export_parcel_layout
from urbanmorph.layout.parcel_gen import lot_jitter
from urbanmorph.layout.place import LayoutState
from urbanmorph.layout.tidy_parcels import generate_tidy
from urbanmorph.models.block_net import BlockNets
from urbanmorph.models.parcel_net import NetSampler, ParcelNet
from urbanmorph.visualization.layout_plot import INK, SURFACE

CK = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs" / "compare"
JIT = PROJECT_ROOT / "outputs" / "m9" / "analysis" / "jitter.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="TZ", choices=["TZ", "B"])
    a = ap.parse_args()
    t0 = time.time()
    cw = load_objects()["class_widths"]
    rec = load_records()
    rt = rec[rec.split == "train"]
    nets = BlockNets()
    nets.load_state_dict(torch.load(CK / "m9" / "block_nets_s0" / "model.pt", map_location="cpu")["model"])
    nets.eval()
    pn = ParcelNet()
    pn.load_state_dict(torch.load(CK / "m8" / "parcel_net_s0" / "model.pt", map_location="cpu")["model"])
    sampler = NetSampler(pn, {c: lot_jitter(rt, c) for c in ("spain", "united_states")}, seed=100 + a.seed)
    jit = json.loads(JIT.read_text())["width_jitter_log_sd"]
    graphs = sample_graphs(load_roads(0), a.n, device="cpu", seed=a.seed)
    fig, axes = plt.subplots(1, a.n, figsize=(5.2 * a.n, 5.6), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (g, ax) in enumerate(zip(graphs, axes)):
        c = ("spain", "united_states")[k % 2]
        s = LayoutState(g, cw, True)
        P, S = generate_tidy(s, c, nets, sampler, np.random.default_rng(1000 * a.seed + k), cw, jit[c], zoning=True,
                             seed=1000 * a.seed + k, method=a.method, access=(a.method == "B"))
        draw_layout(ax, g, s.buildings, s.green, cw)
        draw_parcels(ax, P)
        n_green = sum(p["use"] == "green" for p in P)
        ax.set_title(f"city {k + 1} · {c} style · {len(P)} parcels ({n_green} green), {len(s.buildings)} buildings",
                     loc="left", fontsize=9, color=INK)
        export_parcel_layout(OUT / f"{a.method.lower()}_city{k + 1}_seed{a.seed}.gpkg", g, P, s.buildings, s.green, cw)
    fig.suptitle(f"{a.n} new 512 m tiles generated from scratch: streets (M6 roads v2) → {a.method} parcels, zoning, "
                 f"template buildings (seed {a.seed})", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f = OUT / f"{a.method.lower()}_cities_seed{a.seed}.png"
    fig.savefig(f, dpi=140, facecolor=SURFACE)
    print(f"{time.time() - t0:.0f}s ->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
