"""Milestone 7 evaluation: strict constraints + frontage model vs Milestone 6, on real validation street networks.

    python scripts/m7_evaluate.py

Protocol and pre-registered criteria: docs/MILESTONE_7.md, section 4. Test tiles are never read.
"""
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
import pandas as pd
import torch

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m6_evaluate import BKEYS, COUNTRIES, load_placer, load_roads, w1_iqr  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.export import draw_layout, export_layout, layout_raster
from urbanmorph.layout.frontage import pattern_metrics
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2, building_metrics, green_metrics
from urbanmorph.layout.place import place_frontage, place_neural
from urbanmorph.models import evaluate as ev
from urbanmorph.models.frontage_net import FrontageNet
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m7"
CK = PROJECT_ROOT / "checkpoints"
SEEDS = (0, 1)
PER_TILE = 4
N_FULL = 12
PKEYS = ["nn_aligned_5deg", "frontage_share", "setback_std_m", "spacing_cv"]


def load_frontage(seed, device, relative=False):
    ck = torch.load(CK / "m7" / (f"frontage_rel_s{seed}" if relative else f"frontage_s{seed}") / "model.pt", map_location=device)
    m = FrontageNet(ck["config"]["model"]["hidden"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def strict_validity(buildings, green, surface) -> dict:
    import shapely
    b = [p for p in buildings if not p.is_empty]
    out = {"invalid_polygons": int(sum(not p.is_valid for p in b + [g for g in green if not g.is_empty]))}
    if not b:
        return {**out, "share_on_road_gt1m2": np.nan, "overlap_pairs_gt1pct": 0}
    arr = np.array(b, dtype=object)
    on = shapely.area(shapely.intersection(arr, surface)) if not surface.is_empty else np.zeros(len(b))
    out["share_on_road_gt1m2"] = float((on > 1.0).mean())
    pairs = shapely.STRtree(arr).query(arr, predicate="intersects")
    pairs = pairs[:, pairs[0] < pairs[1]]
    if pairs.shape[1]:
        ov = shapely.area(shapely.intersection(arr[pairs[0]], arr[pairs[1]])) / np.minimum(shapely.area(arr[pairs[0]]), shapely.area(arr[pairs[1]]))
        out["overlap_pairs_gt1pct"] = int((ov > 0.01).sum())
    else:
        out["overlap_pairs_gt1pct"] = 0
    return out


def tile_row(graph, buildings, green, surface, country):
    return {"country": country, **building_metrics(buildings, graph.segments()), **green_metrics(green),
            **pattern_metrics(buildings, graph), **strict_validity(buildings, green, surface)}


def main() -> int:
    global OUT, SEEDS, PER_TILE, N_FULL
    if "--quick" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--quick") + 1])
        SEEDS, PER_TILE, N_FULL = (0,), 1, 6
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export").mkdir(exist_ok=True)
    device = pick_device("auto")
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")

    real_rows, val_meta, real_lays = [], [], []
    for t in data["val"]:
        r = idx.loc[t["sample_id"]]
        f = sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg"
        b = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        g = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
        real_rows.append(tile_row(t["graph"], b, g, road_polygons(t["graph"], class_widths=cw), t["country"]))
        val_meta.append((t, b, g))
        real_lays.append(layout_raster(t["road_raster"], b, g))
    real = pd.DataFrame(real_rows)
    real_lays = np.stack(real_lays)
    train_lays = []
    for t in data["train"]:
        o = t["objects"]
        lay = np.zeros((3, 128 * 128), np.uint8)
        lay[0] = t["road_raster"].ravel()
        for i, ty in enumerate(o["type"]):
            lay[1 + ty, o["pix"][o["pix_ptr"][i]:o["pix_ptr"][i + 1]]] = 1
        train_lays.append(lay.reshape(3, 128, 128))
    train_lays = np.stack(train_lays)
    real_near = float(np.median(ev.nearest_train_similarity(real_lays, train_lays)))
    graphs = [t["graph"] for t, _, _ in val_meta for _ in range(PER_TILE)]
    countries = [t["country"] for t, _, _ in val_meta for _ in range(PER_TILE)]

    methods = {}
    for s in SEEDS:
        p2 = load_placer(s, device)
        methods[f"M6 P2 seed {s}"] = place_neural(p2, graphs, cw, device=device, seed=21)
        methods[f"M6 P2 + strict seed {s}"] = place_neural(p2, graphs, cw, device=device, seed=21, strict=True)
        st = place_frontage(load_frontage(s, device), graphs, cw, device=device, seed=21, strict=True)
        methods[f"M7 seed {s}"] = place_neural(p2, None, cw, device=device, seed=21, states=st)
        st = place_frontage(load_frontage(s, device, True), graphs, cw, device=device, seed=21, strict=True, relative=True)
        methods[f"M7b relative seed {s}"] = place_neural(p2, None, cw, device=device, seed=21, states=st)
        print(f"seed {s} placed ({time.time() - t0:.0f}s)", flush=True)

    res = {"real_val": {k: float(real[k].median()) for k in real.columns if k != "country"},
           "real_val_per_country": {c: {k: float(real[real.country == c][k].median()) for k in BKEYS + PKEYS}
                                    for c in COUNTRIES},
           "real_val_nearest_train_iou_3ch": real_near, "methods": {}, "decision": {}}
    dfs = {}
    for name, states in methods.items():
        df = pd.DataFrame([tile_row(x.graph, x.buildings, x.green, x.surface, c) for x, c in zip(states, countries)])
        dfs[name] = df
        lays = np.stack([layout_raster(x.road, x.buildings, x.green) for x in states])
        res["methods"][name] = {
            "median": {k: float(df[k].median()) for k in df.columns if k != "country"},
            "w1_val_iqr": {k: w1_iqr(df[k], real[k]) for k in BKEYS + PKEYS},
            "totals": {"buildings": int(sum(len(x.buildings) for x in states)),
                       "buildings_on_road_gt1m2": int(round((df["share_on_road_gt1m2"].fillna(0) * df["building_count"]).sum())),
                       "overlap_pairs_gt1pct": int(df["overlap_pairs_gt1pct"].sum()),
                       "invalid_polygons": int(df["invalid_polygons"].sum())},
            "rules_fired": {k: int(v) for k, v in pd.DataFrame([x.rules for x in states]).sum().to_dict().items()},
            "diversity_3ch": ev.diversity(lays[:76]),
            "nearest_train_iou_3ch_median": float(np.median(ev.nearest_train_similarity(lays[:76], train_lays))),
            "per_country": {c: {k: float(df[df.country == c][k].median()) for k in BKEYS + PKEYS} for c in COUNTRIES}}
    real_rank = {k: list(real.groupby("country")[k].median().reindex(COUNTRIES).rank().values)
                 for k in ("building_count", "median_log_area", "median_dist_road_m")}
    for s, variant in [(s, v) for s in SEEDS for v in ("M7", "M7b relative")]:
        m7, m6s = res["methods"][f"{variant} seed {s}"], res["methods"][f"M6 P2 + strict seed {s}"]
        df = dfs[f"{variant} seed {s}"]
        tot = m7["totals"]
        s1 = (tot["buildings_on_road_gt1m2"] == 0 and tot["overlap_pairs_gt1pct"] == 0 and tot["invalid_polygons"] == 0
              and df["acute_share"].mean() <= 0.01)
        within = {k: bool(m7["w1_val_iqr"][k] <= 1.0) for k in PKEYS}
        closer = {k: bool(m7["w1_val_iqr"][k] < m6s["w1_val_iqr"][k]) for k in PKEYS}
        s2 = all(within.values()) and sum(closer.values()) >= 3
        s3 = sum(m7["w1_val_iqr"][k] <= 1.0 for k in BKEYS) >= 7 and m7["nearest_train_iou_3ch_median"] <= real_near + 0.10
        rank = {k: list(df.groupby("country")[k].median().reindex(COUNTRIES).rank().values) for k in real_rank}
        res["decision"][f"{variant} seed {s}"] = {
            "S1_constraints": bool(s1), "S1_acute_share_mean": float(df["acute_share"].mean()),
            "S2_patterns": bool(s2), "S2_within_1iqr": within, "S2_closer_than_M6_strict": closer,
            "S3_no_regression": bool(s3), "S3_detail": {k: bool(m7["w1_val_iqr"][k] <= 1.0) for k in BKEYS},
            "S4_country_ranking": all(rank[k] == real_rank[k] for k in real_rank),
            "success": bool(s1 and s2 and s3)}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print("evaluation done", f"({time.time() - t0:.0f}s)", flush=True)

    # ---- full pipeline examples: roads v2 -> frontage -> interior + green
    full_g = sample_graphs(load_roads(0), N_FULL, device="cpu", seed=321)
    st = place_frontage(load_frontage(0, device, True), full_g, cw, device=device, seed=5, strict=True, relative=True)
    full = place_neural(load_placer(0, device), None, cw, device=device, seed=5, states=st)
    for i, x in enumerate(full[:4]):
        export_layout(OUT / "export" / f"generated_{i}.gpkg", x.graph, x.buildings, x.green, cw)
    # ---- figures
    pick = []
    for c in COUNTRIES:
        pick += [i for i, (t, _, _) in enumerate(val_meta) if t["country"] == c][:2]
    rows = [("real", None), ("M6 P2", "M6 P2 seed 0"), ("M6 P2 + strict rules", "M6 P2 + strict seed 0"),
            ("M7: frontage + strict", "M7 seed 0"), ("M7b: relative frontage", "M7b relative seed 0")]
    fig, axes = plt.subplots(len(rows), len(pick), figsize=(2.3 * len(pick), 2.4 * len(rows)), facecolor=SURFACE)
    for j, i in enumerate(pick):
        t, b, g = val_meta[i]
        for r, (lab, key) in enumerate(rows):
            if key is None:
                draw_layout(axes[r, j], t["graph"], b, g, cw, f"{t['country'].replace('_', ' ')} · real")
            else:
                x = methods[key][i * PER_TILE]
                draw_layout(axes[r, j], x.graph, x.buildings, x.green, cw, lab)
    fig.suptitle("Milestone 7: buildings on REAL validation streets: real vs M6 vs M6 + strict rules vs frontage model",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "layouts_on_real_roads.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # zoom: one tile per country, 200 m window
    fig, axes = plt.subplots(3, len(rows), figsize=(3.2 * len(rows), 9.6), facecolor=SURFACE)
    for r, c in enumerate(COUNTRIES):
        i = [k for k, (t, _, _) in enumerate(val_meta) if t["country"] == c][0]
        t, b, g = val_meta[i]
        for j, (lab, key) in enumerate(rows):
            ax = axes[r, j]
            if key is None:
                draw_layout(ax, t["graph"], b, g, cw, f"{c.replace('_', ' ')} · real (zoom)")
            else:
                x = methods[key][i * PER_TILE]
                draw_layout(ax, x.graph, x.buildings, x.green, cw, f"{lab} (zoom)")
            ax.set_xlim(156, 356)
            ax.set_ylim(156, 356)
    fig.tight_layout()
    fig.savefig(OUT / "zoom_patterns.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5), facecolor=SURFACE)
    for j, ax in enumerate(axes.ravel()[:len(full)]):
        draw_layout(ax, full[j].graph, full[j].buildings, full[j].green, cw, f"generated {j}")
    fig.suptitle("Milestone 7b: fully generated layouts (roads v2 → relative frontage buildings → interior buildings → green)",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "full_pipeline.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    # pattern metric distributions
    names = ["real val"] + [k for k in methods if k.endswith("seed 0")]
    frames = [real] + [dfs[k] for k in names[1:]]
    colors = ["#6b6b6b", "#a8a8a8", "#eb6834", "#2a78d6", "#1baf7a"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), facecolor=SURFACE)
    for ax, k in zip(axes, PKEYS + ["acute_share"]):
        bp = ax.boxplot([d[k].dropna().to_numpy(float) for d in frames], widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([n.replace(" seed 0", "").replace(" + ", "\n+ ").replace("real val", "real") for n in names], fontsize=7, color=INK)
        ax.set_title(k.replace("_", " "), fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#ececec")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Milestone 7: building pattern metrics per tile (real validation streets)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "pattern_metrics.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    show = BKEYS + PKEYS + ["acute_share", "share_on_road_gt1m2"]
    print(json.dumps({"real": {k: round(res["real_val"][k], 3) for k in show},
                      "methods": {n: {"median": {k: round(e["median"][k], 3) for k in show},
                                      "w1": {k: round(v, 2) for k, v in e["w1_val_iqr"].items()}, "totals": e["totals"],
                                      "rules": e["rules_fired"], "div": round(e["diversity_3ch"], 3),
                                      "near": round(e["nearest_train_iou_3ch_median"], 3)} for n, e in res["methods"].items()},
                      "decision": res["decision"], "real_near": real_near,
                      "per_country_real": res["real_val_per_country"],
                      "per_country_M7s0": res["methods"][f"M7 seed {SEEDS[0]}"]["per_country"],
                      "per_country_M7bs0": res["methods"][f"M7b relative seed {SEEDS[0]}"]["per_country"]}, indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
