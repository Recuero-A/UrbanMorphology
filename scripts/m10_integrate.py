"""Milestone 10, Phase 5 (exploratory, no pass/fail criteria): complete generated tiles.

    python scripts/m10_integrate.py --streets C      # -> outputs/m10/integration/

Streets: the first N_TILES cached unconditional samples of a street model, seed 0
(outputs/m10/streets/samples/<model>_s0.pkl, scripts/m10_eval_streets.py). Each tile gets a city: the sample's own
city for city-conditioned models, else drawn with the training-tile proportions (seed 0).
Blocks on those GENERATED streets:
  * BlockGen seed 0 (buildings) - every city;
  * BF (procedural parcels + buildings, data-based green fix) - Spanish and US cities only (parcel models exist
    for those two countries).
Building metrics (tile_row) are compared with the real validation tiles (all 19). Figures draw streets,
parcels (BF) and buildings. Test tiles are never read.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m6_evaluate import w1_iqr  # noqa: E402
from m8_parcel_records import load_records  # noqa: E402
from m10_eval_blocks import SKEYS, load_model, tile_metrics  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.layout.block_gen import generate_buildings
from urbanmorph.layout.export import draw_layout, draw_parcels
from urbanmorph.layout.objects import MIN_BUILDING_M2
from urbanmorph.layout.parcel_gen import lot_jitter
from urbanmorph.layout.place import LayoutState
from urbanmorph.layout.tidy_parcels import generate_tidy
from urbanmorph.models.block_net import BlockNets
from urbanmorph.models.parcel_net import NetSampler, ParcelNet
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "integration"
SAMPLES = PROJECT_ROOT / "outputs" / "m10" / "streets" / "samples"
CK = PROJECT_ROOT / "checkpoints"
JIT = PROJECT_ROOT / "outputs" / "m9" / "analysis" / "jitter.json"
N_TILES = 32
COUNTRY = {"barcelona": "spain", "madrid": "spain", "valencia": "spain", "chicago": "united_states",
           "los_angeles": "united_states", "new_york": "united_states", "tokyo": "japan", "osaka": "japan", "nagoya": "japan"}


def bf_layouts(graphs, countries, cw, seed=0):
    rec = load_records()
    rec_train = rec[rec.split == "train"]
    pn = ParcelNet()
    pn.load_state_dict(torch.load(CK / "m8" / f"parcel_net_s{seed}" / "model.pt", map_location="cpu")["model"])
    sampler = NetSampler(pn, {c: lot_jitter(rec_train, c) for c in ("spain", "united_states")}, seed=100 + seed)
    nets = BlockNets()
    nets.load_state_dict(torch.load(CK / "m9" / f"block_nets_s{seed}" / "model.pt", map_location="cpu")["model"])
    nets.eval()
    jit = json.loads(JIT.read_text())["width_jitter_log_sd"]
    rng = np.random.default_rng(200 + seed)
    out = []
    for k, (g, c) in enumerate(zip(graphs, countries)):
        if c not in ("spain", "united_states"):
            out.append(None)
            continue
        s = LayoutState(g, cw, True)
        P, _ = generate_tidy(s, c, nets, sampler, rng, cw, jit[c], zoning=True, seed=1000 * seed + k, method="BF",
                             access=True, green_fix=True)
        out.append((s, P))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streets", default="C")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion()
    s = pickle.loads((SAMPLES / f"{a.streets}_s0.pkl").read_bytes())
    graphs = s["graphs"][:N_TILES]
    if s["cities"] is not None:
        cities = s["cities"][:N_TILES]
    else:
        tr = idx[idx.in_dataset & (idx.split == "train")].city.tolist()
        cities = [tr[i] for i in np.random.default_rng(0).integers(len(tr), size=N_TILES)]
    countries = [COUNTRY[c] for c in cities]
    model, mc = load_model(0, device)
    bg = generate_buildings(model, graphs, [mc.index(c) for c in cities], cw, device=device, seed=0)
    bf = bf_layouts(graphs, countries, cw)
    # real validation reference (all 19 tiles)
    ix = idx.set_index("sample_id")
    real_b = []
    for t in data["val"]:
        r = ix.loc[t["sample_id"]]
        real_b.append(tile_polygons(sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg", r.xmin, r.ymin,
                                    "buildings", MIN_BUILDING_M2))
    real = tile_metrics(data["val"], cw, real_b, [t["country"] for t in data["val"]])
    tiles = [{"graph": g} for g in graphs]
    gen_bg = tile_metrics(tiles, cw, bg, countries)
    keep = [i for i, x in enumerate(bf) if x is not None]
    gen_bf = tile_metrics([tiles[i] for i in keep], cw, [bf[i][0].buildings for i in keep], [countries[i] for i in keep])
    res = {"streets": a.streets, "n_tiles": N_TILES, "cities": cities, "real_val_median": {k: float(real[k].median()) for k in SKEYS}}
    for name, df in (("BlockGen", gen_bg), ("BF", gen_bf)):
        w = {k: w1_iqr(df[k], real[k]) for k in SKEYS}
        res[name] = {"n_tiles": len(df), "median": {k: float(df[k].median()) for k in SKEYS}, "w1_iqr": w,
                     "S_B": float(np.mean(list(w.values())))}
    (OUT / f"integration_{a.streets}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k]["S_B"] for k in ("BlockGen", "BF")}))

    show = keep[:4] if len(keep) >= 4 else list(range(4))
    fig, axes = plt.subplots(len(show), 2, figsize=(7, 3.5 * len(show)), facecolor=SURFACE)
    for r_, i in enumerate(show):
        draw_layout(axes[r_, 0], graphs[i], bg[i], [], cw, f"{a.streets} streets + BlockGen · {cities[i]}")
        if bf[i] is not None:
            st, P = bf[i]
            draw_layout(axes[r_, 1], st.graph, st.buildings, st.green, cw, f"{a.streets} streets + BF parcels")
            draw_parcels(axes[r_, 1], P)
        else:
            axes[r_, 1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"integration_{a.streets}.png", dpi=100, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
