"""Milestone 8 evaluation of the no-parcel experiments (A, B1, B2, S) on the real validation street networks.

    python scripts/m8_evaluate.py A B1 B2 S

Pre-registered criteria: docs/MILESTONE_8.md §2. Every method: 19 val tiles x 4 samples, sampling
seed 21, strict rules, seeds 0 and 1. Writes outputs/m8/eval/results_<methods>.json and figures.
Test tiles are never read.
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
from m6_evaluate import BKEYS, COUNTRIES, load_placer, w1_iqr  # noqa: E402
from m7_evaluate import load_frontage, tile_row  # noqa: E402
from topo_evaluate import wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.export import draw_layout, layout_raster
from urbanmorph.layout.frontage_edge import place_frontage_edge
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.layout.place import place_frontage, place_neural
from urbanmorph.models import evaluate as ev
from urbanmorph.models.frontage_net import FrontageNetEdge, FrontageNetLib
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m8" / "eval"
CK = PROJECT_ROOT / "checkpoints"
SEEDS = (0, 1)
PER_TILE = 4
PRIMARY = ["nn_aligned_5deg", "spacing_cv", "setback_std_m", "frontage_share", "frontage_size_similarity"]   # + pooled log area
PREDECESSOR = {"B1": "A", "B2": "B1", "S": "B2"}


def load_edge(seed, device):
    m = FrontageNetEdge(256).to(device)
    m.load_state_dict(torch.load(CK / "m8" / f"frontage_edge_s{seed}" / "model.pt", map_location=device)["model"])
    return m.eval()


def load_frontage_lib(seed, device):
    m = FrontageNetLib(256).to(device)
    m.load_state_dict(torch.load(CK / "m8" / f"frontage_lib_s{seed}" / "model.pt", map_location=device)["model"])
    return m.eval()


def load_fallback_b2(seed, device, name="fallback_nb"):
    from urbanmorph.models.placer import ObjectPlacer
    ck = torch.load(CK / "m8" / f"{name}_s{seed}" / "model.pt", map_location=device)
    m = ObjectPlacer(ck["config"]["model"]["width"], ck["config"]["model"]["hidden"], **ck["config"].get("extra", {})).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def run(method, seed, graphs, cw, device):
    """Returns layout states and the number of buildings each had after the frontage phase."""
    p2 = load_placer(seed, device)
    if method == "A":
        st = place_frontage(load_frontage(seed, device), graphs, cw, device=device, seed=21, strict=True)
    else:
        front = load_frontage_lib(seed, device) if method == "S" else load_edge(seed, device)
        st = place_frontage_edge(front, graphs, cw, device=device, seed=21)
    n_front = [len(s.buildings) for s in st]
    fallback = {"B2": lambda: load_fallback_b2(seed, device), "S": lambda: load_fallback_b2(seed, device, "fallback_lib")}.get(
        method, lambda: p2)()
    st = place_neural(fallback, None, cw, device=device, seed=21, states=st)
    return st, n_front


def main() -> int:
    methods = sys.argv[1:] or ["A", "B1"]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")
    real_rows, real_areas, val_meta, real_lays = [], [], [], []
    for t in data["val"]:
        r = idx.loc[t["sample_id"]]
        f = sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg"
        b = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        g = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
        real_rows.append(tile_row(t["graph"], b, g, road_polygons(t["graph"], class_widths=cw), t["country"]))
        real_areas.append((t["country"], np.log([p.area for p in b])))
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
    real_div = ev.diversity(real_lays)
    real_la = np.concatenate([a for _, a in real_areas])
    la_iqr = np.percentile(real_la, 75) - np.percentile(real_la, 25)
    graphs = [t["graph"] for t, _, _ in val_meta for _ in range(PER_TILE)]
    countries = [t["country"] for t, _, _ in val_meta for _ in range(PER_TILE)]

    res = {"real": {k: float(real[k].median()) for k in real.columns if k != "country"},
           "real_diversity": real_div, "real_nearest_train_iou": real_near, "methods": {}, "decision": {}}
    layouts = {}
    for m in methods:
        for s in SEEDS:
            name = f"{m} seed {s}"
            st, n_front = run(m, s, graphs, cw, device)
            layouts[name] = st
            df = pd.DataFrame([tile_row(x.graph, x.buildings, x.green, x.surface, c) for x, c in zip(st, countries)])
            gen_la = np.concatenate([np.log([p.area for p in x.buildings]) for x in st if x.buildings])
            lays = np.stack([layout_raster(x.road, x.buildings, x.green) for x in st])
            rules = pd.DataFrame([x.rules for x in st]).sum().to_dict()
            tot_b = int(sum(len(x.buildings) for x in st))
            e = {"median": {k: float(df[k].median()) for k in df.columns if k != "country"},
                 "w1_iqr": {**{k: w1_iqr(df[k], real[k]) for k in BKEYS + PRIMARY},
                            "pooled_log_area": wasserstein1(gen_la, real_la) / la_iqr},
                 "frontage_acceptance": rules.get("frontage_accepted", 0) / max(rules.get("frontage_proposed", 0), 1),
                 "fallback_share": 1 - sum(n_front) / max(tot_b, 1),
                 "frontage_acute_share": rules.get("frontage_rejected_acute", 0) / max(rules.get("frontage_proposed", 0), 1),
                 "buildings": tot_b, "rules": {k: int(v) for k, v in rules.items()},
                 "validity": {"on_road_gt1m2": int(round((df["share_on_road_gt1m2"].fillna(0) * df["building_count"]).sum())),
                              "overlap_pairs": int(df["overlap_pairs_gt1pct"].sum()), "invalid": int(df["invalid_polygons"].sum()),
                              "acute_share": float(df["acute_share"].mean())},
                 "diversity": ev.diversity(lays[:76]),
                 "nearest_train_iou": float(np.median(ev.nearest_train_similarity(lays[:76], train_lays))),
                 "per_country_w1": {c: {k: w1_iqr(df[df.country == c][k], real[real.country == c][k]) for k in PRIMARY}
                                    for c in COUNTRIES},
                 "per_country_median": {c: {k: float(df[df.country == c][k].median()) for k in PRIMARY + ["building_count"]}
                                        for c in COUNTRIES}}
            res["methods"][name] = e
            print(f"{name}: acceptance {e['frontage_acceptance']:.2f} fallback {e['fallback_share']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    keys6 = PRIMARY + ["pooled_log_area"]
    for m in methods:
        if m not in PREDECESSOR or PREDECESSOR[m] not in methods:
            continue
        for s in SEEDS:
            e, p = res["methods"][f"{m} seed {s}"], res["methods"][f"{PREDECESSOR[m]} seed {s}"]
            better = {k: bool(e["w1_iqr"][k] < p["w1_iqr"][k]) for k in keys6}
            v = e["validity"]
            dec = {"improves_on_predecessor": sum(better.values()) >= 4, "detail": better,
                   "G1_validity": v["on_road_gt1m2"] == 0 and v["overlap_pairs"] == 0 and v["invalid"] == 0 and v["acute_share"] == 0,
                   "G2_no_regression": bool(sum(e["w1_iqr"][k] <= 1.0 for k in BKEYS) >= 7),
                   "G3_diversity": e["diversity"] >= real_div - 0.05,
                   "G4_memorisation": e["nearest_train_iou"] <= real_near + 0.10}
            if m in ("S",):                     # G5 pre-registered from S on (docs §5)
                dec["G5_density"] = bool(abs(e["median"]["building_count"] / res["real"]["building_count"] - 1) <= 0.25
                                     and e["w1_iqr"]["building_coverage"] <= 1.0)
            if m == "B1":
                dec["mechanism"] = e["frontage_acceptance"] >= 0.75 and e["fallback_share"] <= 0.50
            if m == "S":
                dec["mechanism"] = e["frontage_acceptance"] >= 0.75 and e["frontage_acute_share"] <= 0.02
            dec["success"] = all(v for k, v in dec.items() if k not in ("detail",))
            res["decision"][f"{m} seed {s}"] = dec
    tag = "_".join(methods)
    (OUT / f"results_{tag}.json").write_text(json.dumps(res, indent=2, default=float))

    # figure: real vs each method (seed 0), one tile per country, zoomed
    rows = [("real", None)] + [(m, f"{m} seed 0") for m in methods]
    fig, axes = plt.subplots(3, len(rows), figsize=(3.2 * len(rows), 9.6), facecolor=SURFACE)
    for r_, c in enumerate(COUNTRIES):
        i = [k for k, (t, _, _) in enumerate(val_meta) if t["country"] == c][0]
        t, b, g = val_meta[i]
        for j, (lab, key) in enumerate(rows):
            ax = axes[r_, j]
            if key is None:
                draw_layout(ax, t["graph"], b, g, cw, f"{c.replace('_', ' ')} · real")
            else:
                x = layouts[key][i * PER_TILE]
                draw_layout(ax, x.graph, x.buildings, x.green, cw, lab)
            ax.set_xlim(156, 356)
            ax.set_ylim(156, 356)
    fig.tight_layout()
    fig.savefig(OUT / f"zoom_{tag}.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    show = {n: {"acc": round(e["frontage_acceptance"], 3), "fallback": round(e["fallback_share"], 3),
                "w1": {k: round(e["w1_iqr"][k], 2) for k in keys6}, "med": {k: round(e["median"][k], 3) for k in PRIMARY},
                "validity": e["validity"], "div": round(e["diversity"], 3), "near": round(e["nearest_train_iou"], 3),
                "g2": sum(e["w1_iqr"][k] <= 1.0 for k in BKEYS), "shrunk": e["rules"].get("building_shrunk", 0),
                "front_acute": round(e.get("frontage_acute_share", 0), 3), "buildings_median": e["median"]["building_count"]}
            for n, e in res["methods"].items()}
    print(json.dumps({"real": {k: round(res["real"][k], 3) for k in PRIMARY}, "methods": show, "decision": res["decision"]},
                     indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
