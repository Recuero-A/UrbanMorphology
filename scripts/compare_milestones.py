"""Generate one new tile FROM SCRATCH with the model of every milestone (M3 to M8), side by side.

    python scripts/compare_milestones.py --seed 0 --country spain

M1 / M2 built the dataset only (no generative model). Seed-0 checkpoints throughout.
  * M3: spatial VAE (C2), latent drawn from N(0, I), decoded to a raster;
  * M4: same decoder, latent drawn from the PixelCNN prior;
  * M5: street graph only (absolute-step turtle);
  * M6 to M8: ONE street graph from roads v2 (M6), shared by every building method below, so the
    differences are the building / parcel models only:
      M6 placer; M7b relative frontage + placer; M8 B1 edge frontage + placer; B2 edge frontage +
      neighbour fallback; S library frontage + library fallback; C / D generated parcels (country given,
      Spain or US only) + library fallback for green.
Output: outputs/compare/city_seed{seed}_{country}.png (C / D panels draw the generated parcels).
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
import numpy as np
import torch

from m4_evaluate import decode as vae_decode, load_prior  # noqa: E402
from m5_evaluate_graph import draw_graph  # noqa: E402
from m6_build_objects import load_objects  # noqa: E402
from m6_evaluate import load_placer, load_roads  # noqa: E402
from m7_evaluate import load_frontage  # noqa: E402
from m8_evaluate import run as run_m8  # noqa: E402
from m8_parcel_records import load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.layout.export import draw_layout, draw_parcels
from urbanmorph.layout.parcel_gen import EmpiricalSampler, generate_parcels, lot_jitter
from urbanmorph.layout.place import LayoutState, place_frontage, place_neural
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.parcel_net import NetSampler, ParcelNet
from urbanmorph.models.train import load_checkpoint, pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE, composite_rgb

CK = PROJECT_ROOT / "checkpoints"
OUT = PROJECT_ROOT / "outputs" / "compare"
LIB = PROJECT_ROOT / "outputs" / "m8" / "shape_library" / "library.npz"


def parcels_layout(method, graph, country, cw, lib, rec_train, device, seed):
    if method == "C":
        sampler = EmpiricalSampler(rec_train, np.random.default_rng(100 + seed))
    else:
        m = ParcelNet()
        m.load_state_dict(torch.load(CK / "m8" / "parcel_net_s0" / "model.pt", map_location="cpu")["model"])
        sampler = NetSampler(m, {c: lot_jitter(rec_train, c) for c in ("spain", "united_states")}, seed=100 + seed)
    s = LayoutState(graph, cw, True)
    parcels, _ = generate_parcels(s, country, sampler, lib, np.random.default_rng(200 + seed), cw, method=method)
    s.phase = 1                                       # green only from the fallback
    from m8_evaluate import load_fallback_b2
    return place_neural(load_fallback_b2(0, device, "fallback_lib"), None, cw, device=device, seed=21 + seed,
                        states=[s])[0], parcels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--country", default="spain", choices=["spain", "united_states"])
    a = ap.parse_args()
    device = pick_device("auto")
    cw = load_objects()["class_widths"]
    times, panels = {}, []

    t = time.time()
    vae, _ = load_checkpoint(CK / "m3" / "C2_spatial16_roadx3" / "best.pt", device)
    vae.eval()
    g = torch.Generator().manual_seed(a.seed)
    panels.append(("M3 · VAE, N(0, I)", "raster", vae_decode(vae, torch.randn(1, 16, 8, 8, generator=g), device)[0]))
    times["M3"] = time.time() - t
    t = time.time()
    z = load_prior(0, 16, device).sample(1, 8, 8, generator=torch.Generator().manual_seed(a.seed), device=device).cpu()
    panels.append(("M4 · VAE + PixelCNN prior", "raster", vae_decode(vae, z, device)[0]))
    times["M4"] = time.time() - t

    t = time.time()
    ck = torch.load(CK / "m5" / "graph_turtle_s0" / "model.pt", map_location="cpu")
    m5 = GraphTurtle(ck["config"]["model"]["hidden"], ck["config"]["model"]["n_mix"])
    m5.load_state_dict(ck["model"])
    panels.append(("M5 · street graph (turtle)", "graph", sample_graphs(m5, 1, seed=a.seed)[0]))
    times["M5"] = time.time() - t

    t = time.time()
    graph = sample_graphs(load_roads(0), 1, device="cpu", seed=a.seed)[0]
    times["M6 roads"] = time.time() - t
    t = time.time()
    s = place_neural(load_placer(0, device), [graph], cw, device=device, seed=21 + a.seed)[0]
    panels.append(("M6 · roads v2 + placer", "layout", (s, None)))
    times["M6 buildings"] = time.time() - t
    t = time.time()
    st = place_frontage(load_frontage(0, device, True), [graph], cw, device=device, seed=21 + a.seed, strict=True,
                        relative=True)
    s = place_neural(load_placer(0, device), None, cw, device=device, seed=21 + a.seed, states=st)[0]
    panels.append(("M7b · relative frontage + placer", "layout", (s, None)))
    times["M7"] = time.time() - t
    for m, lab in (("B1", "M8 B1 · edge frontage"), ("B2", "M8 B2 · + neighbour fallback"),
                   ("S", "M8 S · shape library")):
        t = time.time()
        st, _ = run_m8(m, 0, [graph], cw, device)
        panels.append((lab, "layout", (st[0], None)))
        times[f"M8 {m}"] = time.time() - t
    lib = ShapeLibrary.load(LIB)
    rec = load_records()
    rec_train = rec[rec.split == "train"]
    for m, lab in (("C", "M8 C · procedural parcels"), ("D", "M8 D · ParcelNet parcels")):
        t = time.time()
        panels.append((f"{lab} ({a.country})", "layout", parcels_layout(m, graph, a.country, cw, lib, rec_train,
                                                                          device, a.seed)))
        times[f"M8 {m}"] = time.time() - t

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4), facecolor=SURFACE)
    for ax, (title, kind, obj) in zip(axes.ravel(), panels):
        if kind == "raster":
            ax.imshow(composite_rgb(obj.astype(np.uint8)), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        elif kind == "graph":
            draw_graph(ax, obj)
            ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        else:
            st, parcels = obj
            draw_layout(ax, st.graph, st.buildings, st.green, cw, title)
            if parcels is not None:
                draw_parcels(ax, parcels)
    fig.suptitle(f"One new 512 m tile per milestone, generated from scratch (seed {a.seed}); M6–M8 share the same "
                 "generated street graph", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"city_seed{a.seed}_{a.country}.png"
    fig.savefig(f, dpi=130, facecolor=SURFACE)
    print({k: round(v, 1) for k, v in times.items()}, "->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
