"""Milestone 6 evaluation: roads v2, buildings + green on real roads (P1 vs P2), full pipeline vs raster baselines.

    python scripts/m6_evaluate.py

Protocol and pre-registered criteria: docs/MILESTONE_6.md, section 4. Test tiles are never read.
Writes outputs/m6/results.json, figures and example exports (outputs/m6/export/).
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

from m4_evaluate import decode as vae_decode, load_prior  # noqa: E402
from m5_evaluate_graph import GRAPH_KEYS, RASTER_KEYS, diversity as road_diversity, nearest_train, summarise  # noqa: E402
from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from topo_evaluate import wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.road_graph import (CLASS_NAMES, components, graph_metrics, junction_angle_deviations,
                                          rasterize_graph, road_polygons)
from urbanmorph.layout.export import draw_layout, export_layout, layout_raster
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2, building_metrics, green_metrics, validity
from urbanmorph.layout.place import StatisticalPlacer, place_neural
from urbanmorph.metrics.road_raster import _label, full_road_profile
from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.placer import ObjectPlacer
from urbanmorph.models.train import load_checkpoint, pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE, composite_rgb

OUT = PROJECT_ROOT / "outputs" / "m6"
CK = PROJECT_ROOT / "checkpoints"
SEEDS = (0, 1)
N_ROADS = 256
SAMPLES_PER_VAL_TILE = 4
N_FULL = 64
BKEYS = ["building_count", "building_coverage", "median_log_area", "median_aspect", "median_rectangularity",
         "share_aligned_10deg", "median_dist_road_m", "clark_evans", "green_coverage"]
COUNTRIES = ("spain", "united_states", "japan")


def load_roads(seed):
    ck = torch.load(CK / "m6" / f"roads_v2_s{seed}" / "model.pt", map_location="cpu")
    mc = ck["config"]["model"]
    m = GraphTurtle(mc["hidden"], mc["n_mix"], turtle=mc["turtle"], n_classes=mc["n_classes"])
    m.load_state_dict(ck["model"])
    return m.eval()


def load_placer(seed, device):
    ck = torch.load(CK / "m6" / f"placer_s{seed}" / "model.pt", map_location=device)
    m = ObjectPlacer(ck["config"]["model"]["width"], ck["config"]["model"]["hidden"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def layout_metrics(graph, buildings, green):
    return {**building_metrics(buildings, graph.segments()), **green_metrics(green)}


def w1_iqr(v, r):
    v, r = np.asarray(v, float), np.asarray(r, float)
    iqr = np.nanpercentile(r, 75) - np.nanpercentile(r, 25)
    return wasserstein1(v, r) / (iqr if iqr else 1.0)


def raster_profile(layouts: np.ndarray) -> pd.DataFrame:
    rows = []
    for m in layouts:
        p = full_road_profile(m[0].astype(bool))
        lab = _label(m[1].astype(bool), 8)
        sizes = np.bincount(lab.ravel())[1:]
        sizes = sizes[sizes >= 2]
        rows.append({**{k: p[k] for k in ("betti0_10", "interior_fragments", "intersections_per_km2", "orientation_phi")},
                     "building_coverage": float(m[1].mean()), "green_coverage": float(m[2].mean()),
                     "building_pieces": int(len(sizes)), "median_piece_m2": float(np.median(sizes) * 16) if len(sizes) else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    global OUT, SEEDS, N_ROADS, SAMPLES_PER_VAL_TILE, N_FULL
    if "--quick" in sys.argv:            # smoke test of the script itself: tiny sizes, scratch output
        OUT = Path(sys.argv[sys.argv.index("--quick") + 1])
        SEEDS, N_ROADS, SAMPLES_PER_VAL_TILE, N_FULL = (0,), 14, 1, 12
    t0 = time.time()
    device = pick_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export").mkdir(exist_ok=True)
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")
    results = {"class_widths_m": dict(zip(CLASS_NAMES, cw))}

    # ================================================================ Part 1: roads v2 vs M5
    real_val_g = [t["graph"] for t in data["val"]]
    real_train_g = [t["graph"] for t in data["train"]]
    sets = {"real val": real_val_g, "real train": real_train_g}
    for s in SEEDS:
        sets[f"roads v2 seed {s}"] = sample_graphs(load_roads(s), N_ROADS, device="cpu", seed=123)
        print(f"roads v2 seed {s} sampled ({time.time() - t0:.0f}s)", flush=True)
    gprof = {n: pd.DataFrame([graph_metrics(g) for g in gs]) for n, gs in sets.items()}
    rast7 = {n: np.stack([rasterize_graph(g) for g in gs]).astype(bool) for n, gs in sets.items()}
    rprof = {n: pd.DataFrame([full_road_profile(m) for m in r]) for n, r in rast7.items()}
    real_div = road_diversity(rast7["real val"])
    real_near = float(np.median(nearest_train(rast7["real val"], rast7["real train"])))
    m5 = json.loads((PROJECT_ROOT / "outputs" / "m5" / "results.json").read_text())
    part1 = {"real_val_diversity": real_div, "real_val_nearest_train_iou": real_near, "sets": {}, "decision": {},
             "m5_for_comparison": {k: {"graph": {m: v["median"] for m, v in m5["sets"][k]["graph"].items()},
                                       "raster": {m: v["median"] for m, v in m5["sets"][k]["raster"].items()},
                                       "w1": {m: v["w1_val_iqr"] for m, v in m5["sets"][k]["graph"].items()}}
                                   for k in ("G seed 0", "G seed 1")}}
    for n, gs in sets.items():
        e = {"graph": summarise(gprof[n], gprof["real val"], GRAPH_KEYS + ["components", "interior_components", "n_nodes",
                                                                         "junction_angle_dev_deg"]),
             "raster": summarise(rprof[n], rprof["real val"], RASTER_KEYS),
             "share_single_component": float((gprof[n]["components"] == 1).mean()),
             "junction_angle_dev_pooled_median": float(np.median(np.concatenate([junction_angle_deviations(g) for g in gs]))),
             "class_share": (np.bincount(np.concatenate([g.classes() for g in gs]), minlength=4) /
                             max(1, sum(len(g.edges) for g in gs))).round(4).tolist()}
        if n.startswith("roads"):
            e["diversity"] = road_diversity(rast7[n][:96])
            e["nearest_train_iou_median"] = float(np.median(nearest_train(rast7[n][:128], rast7["real train"])))
            r = e["raster"]
            c1 = r["interior_fragments"]["median"] <= 1 and abs(r["betti0_10"]["median"] - float(rprof["real val"]["betti0_10"].median())) <= 1
            close = {k: bool(e["graph"][k]["w1_val_iqr"] <= 1.0) for k in GRAPH_KEYS}
            part1["decision"][n] = {"C1_fragmentation": bool(c1), "C2_structure": sum(close.values()) >= 5, "C2_detail": close,
                                    "C3_diversity": bool(e["diversity"] >= real_div - 0.05),
                                    "C4_memorisation": bool(e["nearest_train_iou_median"] <= real_near + 0.10)}
            part1["decision"][n]["success"] = all(part1["decision"][n][k] for k in ("C1_fragmentation", "C2_structure",
                                                                                     "C3_diversity", "C4_memorisation"))
        part1["sets"][n] = e
    results["part1_roads"] = part1
    print("part 1 done", f"({time.time() - t0:.0f}s)", flush=True)

    # ================================================================ Part 2: buildings + green on REAL val roads
    real_rows, val_meta = [], []
    real_val_layouts = []
    for t in data["val"]:
        r = idx.loc[t["sample_id"]]
        f = sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg"
        b = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        g = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
        surface = road_polygons(t["graph"], class_widths=cw)
        real_rows.append({"country": t["country"], **layout_metrics(t["graph"], b, g), **validity(b, g, surface)})
        val_meta.append((t, b, g))
        real_val_layouts.append(layout_raster(t["road_raster"], b, g))
    real_df = pd.DataFrame(real_rows)
    train_layouts = []
    for t in data["train"]:
        o = t["objects"]
        lay = np.zeros((3, 128 * 128), np.uint8)
        lay[0] = t["road_raster"].ravel()
        for i, ty in enumerate(o["type"]):
            lay[1 + ty, o["pix"][o["pix_ptr"][i]:o["pix_ptr"][i + 1]]] = 1
        train_layouts.append(lay.reshape(3, 128, 128))
    train_layouts = np.stack(train_layouts)
    real_val_layouts = np.stack(real_val_layouts)
    real_near3 = float(np.median(ev.nearest_train_similarity(real_val_layouts, train_layouts)))
    graphs_rep = [t["graph"] for t, _, _ in val_meta for _ in range(SAMPLES_PER_VAL_TILE)]
    country_rep = [t["country"] for t, _, _ in val_meta for _ in range(SAMPLES_PER_VAL_TILE)]
    methods = {}
    p1 = StatisticalPlacer(data["train"])
    methods["P1 statistical"] = p1.place(graphs_rep, cw, seed=7)
    print("P1 placed", f"({time.time() - t0:.0f}s)", flush=True)
    for s in SEEDS:
        methods[f"P2 neural seed {s}"] = place_neural(load_placer(s, device), graphs_rep, cw, device=device, seed=7)
        print(f"P2 seed {s} placed ({time.time() - t0:.0f}s)", flush=True)
    part2 = {"real_val": {k: float(real_df[k].median()) for k in real_df.columns if k != "country"},
             "real_val_nearest_train_iou_3ch": real_near3, "methods": {}, "decision": {}}
    real_rank = {k: list(real_df.groupby("country")[k].median().reindex(COUNTRIES).rank().values)
                 for k in ("building_count", "median_log_area", "median_dist_road_m")}
    part2["real_val_per_country"] = {c: {k: float(real_df[real_df.country == c][k].median()) for k in BKEYS}
                                     for c in COUNTRIES}
    method_dfs = {}
    for name, states in methods.items():
        df = pd.DataFrame([{"country": c, **layout_metrics(s.graph, s.buildings, s.green),
                            **validity(s.buildings, s.green, s.surface)} for s, c in zip(states, country_rep)])
        method_dfs[name] = df
        rules = pd.DataFrame([s.rules for s in states]).sum().to_dict()
        lays = np.stack([layout_raster(s.road, s.buildings, s.green) for s in states])
        w1 = {k: w1_iqr(df[k], real_df[k]) for k in BKEYS}
        e = {"median": {k: float(df[k].median()) for k in df.columns if k != "country"}, "w1_val_iqr": w1,
             "mean_w1_iqr": float(np.mean(list(w1.values()))), "rules_fired": {k: int(v) for k, v in rules.items()},
             "diversity_3ch": ev.diversity(lays[:76]),
             "nearest_train_iou_3ch_median": float(np.median(ev.nearest_train_similarity(lays[:76], train_layouts))),
             "per_country": {c: {k: float(df[df.country == c][k].median()) for k in BKEYS} for c in COUNTRIES}}
        part2["methods"][name] = e
    part2["real_val_diversity_3ch"] = ev.diversity(real_val_layouts)
    for s in SEEDS:
        n = f"P2 neural seed {s}"
        e, df = part2["methods"][n], method_dfs[n]
        rank = {k: list(df.groupby("country")[k].median().reindex(COUNTRIES).rank().values) for k in real_rank}
        b3 = (e["median"]["invalid_polygons"] == 0 and df["invalid_polygons"].sum() == 0
              and df["overlapping_building_pairs_per_building"].median() <= real_df["overlapping_building_pairs_per_building"].median()
              and df["share_buildings_over_25pct_on_road"].median() <= real_df["share_buildings_over_25pct_on_road"].median())
        part2["decision"][n] = {
            "B1_realism": sum(v <= 1.0 for v in e["w1_val_iqr"].values()) >= 7,
            "B1_detail": {k: bool(v <= 1.0) for k, v in e["w1_val_iqr"].items()},
            "B2_beats_P1": e["mean_w1_iqr"] < part2["methods"]["P1 statistical"]["mean_w1_iqr"],
            "B3_validity": bool(b3),
            "B4_country_ranking": all(rank[k] == real_rank[k] for k in real_rank),
            "B4_detail": {k: {"generated": rank[k], "real": real_rank[k]} for k in real_rank},
            "B5_memorisation": e["nearest_train_iou_3ch_median"] <= real_near3 + 0.10}
        part2["decision"][n]["success"] = all(part2["decision"][n][k] for k in
                                              ("B1_realism", "B2_beats_P1", "B3_validity", "B4_country_ranking", "B5_memorisation"))
    results["part2_buildings_on_real_roads"] = part2
    print("part 2 done", f"({time.time() - t0:.0f}s)", flush=True)

    # ================================================================ Part 3: full pipeline vs raster baselines
    full_graphs = sets["roads v2 seed 0"][:N_FULL]
    full = place_neural(load_placer(0, device), full_graphs, cw, device=device, seed=11)
    full_lays = np.stack([layout_raster(s.road, s.buildings, s.green) for s in full])
    vae, _ = load_checkpoint(CK / "m3" / "C2_spatial16_roadx3" / "best.pt", device)
    vae.eval()
    gA = torch.Generator().manual_seed(123)
    A = vae_decode(vae, torch.randn(N_FULL, 16, 8, 8, generator=gA), device).astype(np.uint8)
    prior = load_prior(0, 16, device)
    B = vae_decode(vae, prior.sample(N_FULL, 8, 8, generator=torch.Generator().manual_seed(123), device=device).cpu(),
                   device).astype(np.uint8)
    val_ds = LayoutDataset("val")
    real_orig = val_ds.x.numpy().astype(np.uint8)
    train_orig = LayoutDataset("train", index=load_index_with_expansion()).x.numpy().astype(np.uint8)
    p3 = {}
    for name, lays in (("real val (OSM widths)", real_orig), ("real val (class widths)", real_val_layouts),
                       ("A: Spatial VAE, N(0,I)", A), ("B: Spatial VAE + PixelCNN prior", B), ("D: hybrid (roads v2 + P2)", full_lays)):
        prof = raster_profile(lays)
        p3[name] = {"median": {k: float(prof[k].median()) for k in prof.columns},
                    "diversity_3ch": ev.diversity(lays[:64]),
                    "nearest_train_iou_3ch_median": float(np.median(ev.nearest_train_similarity(lays[:64], train_orig)))}
    p3["D rules fired"] = {k: int(v) for k, v in pd.DataFrame([s.rules for s in full]).sum().to_dict().items()}
    p3["D vector validity"] = pd.DataFrame([validity(s.buildings, s.green, s.surface) for s in full]).median().to_dict()
    results["part3_full_pipeline"] = p3
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("part 3 done", f"({time.time() - t0:.0f}s)", flush=True)

    # ================================================================ figures + exports
    for i, s in enumerate(full[:4]):
        export_layout(OUT / "export" / f"generated_{i}.gpkg", s.graph, s.buildings, s.green, cw)
    import geopandas as gpd
    gpd.GeoDataFrame({"layer": ["building"] * len(full[0].buildings) + ["green"] * len(full[0].green)},
                     geometry=full[0].buildings + full[0].green, crs=None).to_file(OUT / "export" / "generated_0.geojson",
                                                                                  driver="GeoJSON")
    # fig 1: conditional layouts (2 tiles per country): real | P1 | P2 s0
    pick = []
    for c in COUNTRIES:
        pick += [i for i, (t, _, _) in enumerate(val_meta) if t["country"] == c][:2]
    fig, axes = plt.subplots(3, len(pick), figsize=(2.3 * len(pick), 7.2), facecolor=SURFACE)
    for j, i in enumerate(pick):
        t, b, g = val_meta[i]
        draw_layout(axes[0, j], t["graph"], b, g, cw, f"{t['country'].replace('_', ' ')} · real")
        s1 = methods["P1 statistical"][i * SAMPLES_PER_VAL_TILE]
        draw_layout(axes[1, j], s1.graph, s1.buildings, s1.green, cw, "P1 statistical")
        s2 = methods["P2 neural seed 0"][i * SAMPLES_PER_VAL_TILE]
        draw_layout(axes[2, j], s2.graph, s2.buildings, s2.green, cw, "P2 neural seed 0")
    fig.suptitle("Milestone 6: buildings and green on REAL validation street networks (roads grey, buildings dark, green)",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "layouts_on_real_roads.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # fig 2: full pipeline
    fig, axes = plt.subplots(2, 6, figsize=(14, 5), facecolor=SURFACE)
    for j, ax in enumerate(axes.ravel()):
        s = full[j]
        draw_layout(ax, s.graph, s.buildings, s.green, cw, f"generated {j}")
    fig.suptitle("Milestone 6: fully generated layouts (roads v2 → buildings → green)", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "full_pipeline.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # fig 3: roads v2 vs M5 vs real (graphs coloured by class)
    from m5_evaluate_graph import draw_graph
    from m5_build_graphs import load_graphs as load_m5
    from urbanmorph.graphs.road_graph import TILE_M
    m5_model = GraphTurtle(256, 10)
    m5_model.load_state_dict(torch.load(CK / "m5" / "graph_turtle_s0" / "model.pt", map_location="cpu")["model"])
    m5_samples = sample_graphs(m5_model, 7, seed=5)
    rows = [("real val", real_val_g), ("M5 (absolute steps)", m5_samples)] + [
        (f"v2 seed {sd}" + (" (turtle frame)" if sd == 0 else ""), sets[f"roads v2 seed {sd}"]) for sd in SEEDS]
    fig, axes = plt.subplots(len(rows), 7, figsize=(14, 2.1 * len(rows) + 0.4), facecolor=SURFACE)
    for i, (n, gs) in enumerate(rows):
        for j in range(7):
            draw_graph(axes[i, j], gs[(j * 2) % len(gs)])
        axes[i, 0].set_ylabel(n, fontsize=8, color=INK)
    fig.suptitle("Street graphs: real vs Milestone 5 vs roads v2 (largest component dark, others red)",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "roads_v2_samples.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    # fig 4: junction angle deviation histograms
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    bins = np.arange(0, 46, 2.5)
    m5_dev = np.concatenate([junction_angle_deviations(g) for g in sample_graphs(m5_model, 48, seed=6)])
    for lab, dev, col in (("real val", np.concatenate([junction_angle_deviations(g) for g in real_val_g]), "#6b6b6b"),
                          ("M5", m5_dev, "#eb6834"),
                          ("roads v2 seed 0", np.concatenate([junction_angle_deviations(g) for g in sets["roads v2 seed 0"]]), "#2a78d6")):
        ax.hist(dev, bins=bins, density=True, histtype="step", lw=2, color=col, label=f"{lab} (median {np.median(dev):.1f}°)")
    ax.set_xlabel("angle between neighbouring streets at a junction: distance from 90° / 180° (degrees)", fontsize=8, color=INK)
    ax.set_ylabel("density", fontsize=8, color=INK)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "junction_angles.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # fig 5: building metric distributions
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), facecolor=SURFACE)
    names = ["real val"] + list(method_dfs)
    dfs = [real_df] + [method_dfs[n] for n in method_dfs]
    colors = ["#6b6b6b", "#a8a8a8", "#2a78d6", "#1baf7a"]
    for ax, k in zip(axes.ravel(), BKEYS + ["green_count"]):
        bp = ax.boxplot([d[k].dropna().to_numpy(float) for d in dfs], widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7, color=INK)
        ax.set_title(k.replace("_", " "), fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#ececec")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Milestone 6: building and green metrics on real validation street networks (per tile)", fontsize=10,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "building_metrics.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # fig 6: raster comparison with baselines
    fig, axes = plt.subplots(4, 6, figsize=(13, 9), facecolor=SURFACE)
    for i, (n, lays) in enumerate((("real val", real_orig), ("A: VAE N(0,I)", A), ("B: VAE + PixelCNN", B),
                                   ("D: hybrid", full_lays))):
        for j in range(6):
            axes[i, j].imshow(composite_rgb(lays[j * 3 % len(lays)]), interpolation="nearest")
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(n, fontsize=8, color=INK)
    fig.suptitle("Raster view (same colours as Milestones 3-4): real vs VAE baselines vs hybrid", fontsize=10,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "raster_comparison.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    print(json.dumps({"roads": {n: {"decision": part1["decision"].get(n), "junction_dev": e["junction_angle_dev_pooled_median"],
                                    "single_comp": e["share_single_component"],
                                    **{k: round(e["graph"][k]["median"], 3) for k in GRAPH_KEYS + ["components"]},
                                    "pieces": e["raster"]["betti0_10"]["median"], "floating": e["raster"]["interior_fragments"]["median"]}
                               for n, e in part1["sets"].items()},
                      "buildings": {n: {"median": {k: round(v, 3) for k, v in e["median"].items()}, "mean_w1": e["mean_w1_iqr"],
                                        "w1": {k: round(v, 2) for k, v in e["w1_val_iqr"].items()}, "rules": e["rules_fired"],
                                        "div": e["diversity_3ch"], "near": e["nearest_train_iou_3ch_median"]}
                                    for n, e in part2["methods"].items()},
                      "real_val_buildings": part2["real_val"], "decision_buildings": part2["decision"],
                      "per_country_real": part2["real_val_per_country"],
                      "per_country_P2s0": part2["methods"]["P2 neural seed 0"]["per_country"],
                      "per_country_P1": part2["methods"]["P1 statistical"]["per_country"],
                      "full": p3}, indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
