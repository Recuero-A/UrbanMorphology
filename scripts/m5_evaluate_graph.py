"""Milestone 5: evaluate the NTG-inspired street-graph generator against real validation graphs.

    python scripts/m5_evaluate_graph.py

Sets compared (256 generated tiles per training seed, fixed sampling seed):
  real val    19 validation graphs (primary reference); real train: 565 training graphs
  G seed 0/1  generated graphs
Two views of every tile:
  * GRAPH metrics (road_graph.graph_metrics), computed directly on the vectors;
  * RASTER topology (road_raster.full_road_profile) after rasterising the graph with ONE road width
    (7 m, the median OSM width). Real graphs are rasterised the same way ("real val, same raster"),
    so raster differences come from the network, not from widths. The Milestone 4 numbers
    (VAE + PixelCNN prior, real widths) are listed for context only.
Diversity and memorisation use the road raster (1 - pairwise IoU; nearest training tile under 8 symmetries).

Decision criteria, fixed BEFORE looking at generated samples (docs/MILESTONE_5.md, section 5):
  C1 fragmentation: median floating fragments <= 1 AND |median road pieces - real| <= 1 (same raster)
  C2 structure: W1/IQR to real val <= 1 for at least 5 of 6 graph metrics
  C3 diversity >= real val diversity - 0.05
  C4 memorisation: median nearest-train IoU <= real val's + 0.10
  success = C1-C4 hold for BOTH seeds
"""
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from m5_build_graphs import load_graphs  # noqa: E402
from topo_evaluate import wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.road_graph import components, graph_metrics, rasterize_graph
from urbanmorph.metrics.road_raster import full_road_profile
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m5"
CKPT = PROJECT_ROOT / "checkpoints" / "m5"
GRAPH_KEYS = ["intersections_per_km2", "dead_end_ratio", "loops_per_km2", "orientation_phi",
              "mean_link_length_m", "street_length_km_per_km2"]
RASTER_KEYS = ["betti0_10", "interior_fragments", "intersections_per_km2", "loops_per_km2", "dead_end_ratio",
               "orientation_phi", "road_coverage"]


def road_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2, b2 = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    inter = a2 @ b2.T
    union = a2.sum(1)[:, None] + b2.sum(1)[None] - inter
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def diversity(m: np.ndarray) -> float:
    iou = road_iou_matrix(m, m)
    iu = np.triu_indices(len(m), 1)
    return float((1 - iou[iu]).mean())


def nearest_train(m: np.ndarray, train: np.ndarray) -> np.ndarray:
    variants = []
    for k in range(8):
        t = train[:, :, ::-1] if k >= 4 else train
        variants.append(np.rot90(t, k % 4, axes=(1, 2)))
    return road_iou_matrix(m, np.concatenate(variants)).max(1)


def summarise(df: pd.DataFrame, ref: pd.DataFrame, keys) -> dict:
    out = {}
    for k in keys:
        v, r = df[k].to_numpy(float), ref[k].to_numpy(float)
        iqr = np.nanpercentile(r, 75) - np.nanpercentile(r, 25)
        out[k] = {"median": float(np.nanmedian(v)), "p05": float(np.nanpercentile(v, 5)),
                  "p95": float(np.nanpercentile(v, 95)), "w1_val_iqr": wasserstein1(v, r) / (iqr if iqr else 1.0)}
    return out


def draw_graph(ax, g, color="#1f1f1f"):
    if len(g.edges):
        seg = g.segments()
        lab = components(g)
        big = np.bincount(lab).argmax()
        for (x1, y1, x2, y2), c in zip(seg, lab[g.edges[:, 0]]):
            ax.plot([x1, x2], [y1, y2], color=color if c == big else "#d6402b", lw=0.8)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cfcfcf")


def main() -> int:
    cfg = tomllib.loads((PROJECT_ROOT / "configs" / "m5_graph.toml").read_text())
    N, temp = cfg["sample"]["n"], cfg["sample"]["temperature"]
    data = load_graphs()
    real = {"real val": [g for _, _, g in data["val"]], "real train": [g for _, _, g in data["train"]]}
    graphs = dict(real)
    timing = {}
    for s in cfg["train"]["seeds"]:
        ck = torch.load(CKPT / f"graph_turtle_s{s}" / "model.pt", map_location="cpu")
        model = GraphTurtle(ck["config"]["model"]["hidden"], ck["config"]["model"]["n_mix"])
        model.load_state_dict(ck["model"])
        t0 = time.time()
        graphs[f"G seed {s}"] = sample_graphs(model, N, device="cpu", seed=123, temperature=temp)
        timing[s] = round(time.time() - t0, 1)
        print(f"seed {s}: {N} graphs in {timing[s]} s", flush=True)

    gprof, rprof, rasters = {}, {}, {}
    for name, gs in graphs.items():
        gprof[name] = pd.DataFrame([graph_metrics(g) for g in gs])
        rasters[name] = np.stack([rasterize_graph(g) for g in gs]).astype(bool)
        rprof[name] = pd.DataFrame([full_road_profile(m) for m in rasters[name]])
        gprof[name].to_csv(OUT / f"graph_profiles_{name.replace(' ', '_')}.csv", index=False)
        rprof[name].to_csv(OUT / f"raster_profiles_{name.replace(' ', '_')}.csv", index=False)

    ref_g, ref_r = gprof["real val"], rprof["real val"]
    real_div = diversity(rasters["real val"])
    real_near = float(np.median(nearest_train(rasters["real val"], rasters["real train"])))
    m4 = pd.read_csv(PROJECT_ROOT / "outputs" / "m4" / "profiles_C_seed0.csv")
    m4_real = pd.read_csv(PROJECT_ROOT / "outputs" / "topology" / "eval" / "real_val_profiles.csv")
    results = {"n_generated": N, "temperature": temp, "sampling_seconds": timing, "real_val_diversity": real_div,
               "real_val_nearest_train_iou_median": real_near, "sets": {},
               "context_m4_pixelcnn_seed0_real_widths": {k: float(m4[k].median()) for k in RASTER_KEYS},
               "context_real_val_real_widths": {k: float(m4_real[k].median()) for k in RASTER_KEYS}}
    for name in graphs:
        e = {"graph": summarise(gprof[name], ref_g, GRAPH_KEYS + ["components", "interior_components", "n_nodes"]),
             "raster": summarise(rprof[name], ref_r, RASTER_KEYS)}
        if name != "real train":
            m = rasters[name][:96] if name.startswith("G") else rasters[name]
            e["diversity"] = diversity(m)
            if name.startswith("G"):
                e["nearest_train_iou_median"] = float(np.median(nearest_train(rasters[name][:128], rasters["real train"])))
        results["sets"][name] = e

    dec = {}
    for s in cfg["train"]["seeds"]:
        e = results["sets"][f"G seed {s}"]
        r = e["raster"]
        c1 = r["interior_fragments"]["median"] <= 1 and abs(r["betti0_10"]["median"] - float(ref_r["betti0_10"].median())) <= 1
        close = {k: e["graph"][k]["w1_val_iqr"] <= 1.0 for k in GRAPH_KEYS}
        c2 = sum(close.values()) >= 5
        c3 = e["diversity"] >= real_div - 0.05
        c4 = e["nearest_train_iou_median"] <= real_near + 0.10
        c1, c2, c3, c4 = bool(c1), bool(c2), bool(c3), bool(c4)
        close = {k: bool(v) for k, v in close.items()}
        dec[str(s)] = {"C1_fragmentation": c1, "C2_structure": c2, "C2_detail": close, "C3_diversity": c3,
                       "C4_memorisation": c4, "success": bool(c1 and c2 and c3 and c4)}
    results["decision"] = dec
    results["success_both_seeds"] = all(d["success"] for d in dec.values())
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- figure 1: real validation graphs vs generated graphs
    rows = [("real val", graphs["real val"])] + [(n, graphs[n]) for n in graphs if n.startswith("G")]
    fig, axes = plt.subplots(len(rows), 7, figsize=(14, 2.1 * len(rows) + 0.4), facecolor=SURFACE)
    for i, (n, gs) in enumerate(rows):
        for j in range(7):
            draw_graph(axes[i, j], gs[(j * 2) % len(gs)])
        axes[i, 0].set_ylabel(n, fontsize=8, color=INK)
    fig.suptitle("Milestone 5: street graphs, real validation tiles vs generated (largest component dark, "
                 "other components red)", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "samples_graphs.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 2: metric distributions (box plots, same layout as Milestone 4)
    names = ["real val", "real train"] + [n for n in graphs if n.startswith("G")]
    colors = ["#6b6b6b", "#a8a8a8", "#2a78d6", "#1baf7a"]
    panels = [("raster", "betti0_10", "road pieces (≥10 px)"), ("raster", "interior_fragments", "floating fragments"),
              ("graph", "intersections_per_km2", "junctions / km² (graph)"), ("graph", "loops_per_km2", "loops / km² (graph)"),
              ("graph", "dead_end_ratio", "dead-end ratio (graph)"), ("graph", "orientation_phi", "orientation order φ (graph)"),
              ("graph", "mean_link_length_m", "mean link length, m (graph)"),
              ("graph", "street_length_km_per_km2", "street length, km / km² (graph)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), facecolor=SURFACE)
    for ax, (kind, k, lab) in zip(axes.ravel(), panels):
        src = rprof if kind == "raster" else gprof
        bp = ax.boxplot([src[n][k].dropna().to_numpy(float) for n in names], widths=0.55, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7, color=INK)
        ax.set_title(lab, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#ececec")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Milestone 5: generated street-graph metrics vs real tiles (raster panels use one 7 m road width "
                 "for all sets)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "distributions_graphs.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    short = {n: {"pieces": e["raster"]["betti0_10"]["median"], "floating": e["raster"]["interior_fragments"]["median"],
                 **{k: round(e["graph"][k]["median"], 3) for k in GRAPH_KEYS},
                 "w1": {k: round(e["graph"][k]["w1_val_iqr"], 2) for k in GRAPH_KEYS},
                 "diversity": e.get("diversity"), "near_train": e.get("nearest_train_iou_median")}
             for n, e in results["sets"].items()}
    print(json.dumps({"sets": short, "real_val_near_train": real_near, "decision": dec,
                      "m4_context": results["context_m4_pixelcnn_seed0_real_widths"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
