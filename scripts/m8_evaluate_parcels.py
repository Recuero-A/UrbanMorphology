"""Milestone 8 evaluation of the parcel experiments (C, D) against S, on the validation tiles that have
OBSERVED parcels (Spain 6, US 4; docs/MILESTONE_8.md §5).

    python scripts/m8_evaluate_parcels.py

Per method and seed: 10 tiles x 4 samples on the real street networks, strict rules.
* S: frontage_lib + fallback_lib (no parcels);
* C: generated parcels, EmpiricalSampler (sampling seed s), buildings per parcel, green by fallback_lib s;
* D: generated parcels, ParcelNet seed s, otherwise as C.
Parcel metrics use the SAME measurement rules for generated and observed parcels (layout/parcels.py).
Writes outputs/m8/eval_parcels/results.json, figures (parcels drawn: observed solid red, generated
dashed blue / orange = interior) and GeoPackage exports with parcel IDs and the building -> parcel relation.
Test tiles are never read.
"""
import json
import pickle
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
from m6_evaluate import BKEYS, w1_iqr  # noqa: E402
from m7_evaluate import tile_row  # noqa: E402
from m8_evaluate import PRIMARY, load_fallback_b2, load_frontage_lib  # noqa: E402
from m8_parcel_records import load_records  # noqa: E402
from topo_evaluate import wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.export import draw_layout, draw_parcels, export_parcel_layout, layout_raster
from urbanmorph.layout.frontage_edge import place_frontage_edge
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.layout.parcel_gen import EmpiricalSampler, generate_parcels, lot_jitter
from urbanmorph.layout.parcels import coverage_autocorrelation, morans_i, parcel_neighbours, parcel_table
from urbanmorph.layout.place import LayoutState, place_neural
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.models import evaluate as ev
from urbanmorph.models.parcel_net import NetSampler, ParcelNet
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import SURFACE

OUT = PROJECT_ROOT / "outputs" / "m8" / "eval_parcels"
CK = PROJECT_ROOT / "checkpoints" / "m8"
LIB = PROJECT_ROOT / "outputs" / "m8" / "shape_library" / "library.npz"
OBS = PROJECT_ROOT / "data" / "processed" / "m8" / "parcels_observed.pkl"
SEEDS = (0, 1)
PER_TILE = 4
PARCEL_KEYS = ["log_area", "frontage_m", "depth_m", "coverage", "front_setback_m", "n_buildings"]
PREDECESSOR = {"C": "S", "D": "C"}


def parcel_samples(rows: pd.DataFrame) -> dict:
    """The six parcel distributions (same selection for observed and generated parcels)."""
    full = rows[~rows.cut]
    fr = full[full.frontage_m > 1]
    one = fr[fr.n_buildings == 1]
    return {"log_area": np.log(full.area_m2), "frontage_m": fr.frontage_m, "depth_m": fr.depth_m.dropna(),
            "coverage": full.coverage, "front_setback_m": one.front_setback_m.dropna(), "n_buildings": full.n_buildings}


def run(method, seed, metas, cw, lib, rec_train, device):
    graphs = [t["graph"] for t, *_ in metas for _ in range(PER_TILE)]
    countries = [t["country"] for t, *_ in metas for _ in range(PER_TILE)]
    fallback = load_fallback_b2(seed, device, "fallback_lib")
    if method == "S":
        st = place_frontage_edge(load_frontage_lib(seed, device), graphs, cw, device=device, seed=21)
        return place_neural(fallback, None, cw, device=device, seed=21, states=st), [None] * len(st), {}
    if method == "C":
        sampler = EmpiricalSampler(rec_train, np.random.default_rng(100 + seed))
    else:
        m = ParcelNet()
        m.load_state_dict(torch.load(CK / f"parcel_net_s{seed}" / "model.pt", map_location="cpu")["model"])
        sampler = NetSampler(m, {c: lot_jitter(rec_train, c) for c in ("spain", "united_states")}, seed=100 + seed)
    rng = np.random.default_rng(200 + seed)
    states, parcels, stats = [], [], {}
    for g, c in zip(graphs, countries):
        s = LayoutState(g, cw, True)
        P, S = generate_parcels(s, c, sampler, lib, rng, cw, method=method)
        s.phase = 1                                   # buildings come from the parcels; the fallback adds green only
        states.append(s)
        parcels.append(P)
        for k, v in S.items():
            stats[k] = stats.get(k, 0) + v
    return place_neural(fallback, None, cw, device=device, seed=21, states=states), parcels, stats


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export").mkdir(exist_ok=True)
    device = pick_device("auto")
    data = load_objects()
    cw = data["class_widths"]
    lib = ShapeLibrary.load(LIB)
    rec = load_records()
    rec_train = rec[rec.split == "train"]
    obs = pickle.loads(OBS.read_bytes())
    idx = load_index_with_expansion().set_index("sample_id")
    metas, real_rows, real_lays = [], [], []
    for t in data["val"]:
        if t["sample_id"] not in obs:
            continue
        r = idx.loc[t["sample_id"]]
        f = sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg"
        b = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        g = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
        metas.append((t, b, g, obs[t["sample_id"]]["parcels"]))
        real_rows.append(tile_row(t["graph"], b, g, road_polygons(t["graph"], class_widths=cw), t["country"]))
        real_lays.append(layout_raster(t["road_raster"], b, g))
    real = pd.DataFrame(real_rows)
    real_lays = np.stack(real_lays)
    real_la = np.concatenate([np.log([p.area for p in b]) for _, b, _, _ in metas])
    la_iqr = np.percentile(real_la, 75) - np.percentile(real_la, 25)
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
    # observed parcels of these tiles: distributions + Moran's I per country
    rv = rec[rec.sample_id.isin([t["sample_id"] for t, *_ in metas])]
    obs_samp = {c: parcel_samples(rv[rv.country == c]) for c in ("spain", "united_states")}
    obs_moran = {}
    for c in obs_samp:
        xs, ys = [], []
        for t, _, _, P in metas:
            if t["country"] != c:
                continue
            rr = rv[rv.sample_id == t["sample_id"]]
            x, y = coverage_autocorrelation(rr.coverage.to_numpy(), rr.cut.to_numpy(), parcel_neighbours(P))
            xs.append(x)
            ys.append(y)
        obs_moran[c] = morans_i(np.concatenate(xs), np.concatenate(ys))
    countries = [t["country"] for t, *_ in metas for _ in range(PER_TILE)]
    res = {"tiles": [t["sample_id"] for t, *_ in metas], "real": {k: float(real[k].median()) for k in real.columns if k != "country"},
           "observed_morans_I": obs_moran, "observed_parcels": {c: {k: float(np.median(v)) for k, v in s.items()} for c, s in obs_samp.items()},
           "methods": {}, "decision": {}}
    layouts = {}
    for m in ("S", "C", "D"):
        for s in SEEDS:
            name = f"{m} seed {s}"
            st, P, pstats = run(m, s, metas, cw, lib, rec_train, device)
            layouts[name] = (st, P)
            df = pd.DataFrame([tile_row(x.graph, x.buildings, x.green, x.surface, c) for x, c in zip(st, countries)])
            gen_la = np.concatenate([np.log([p.area for p in x.buildings]) for x in st if x.buildings])
            lays = np.stack([layout_raster(x.road, x.buildings, x.green) for x in st])
            v = {"on_road_gt1m2": int(round((df["share_on_road_gt1m2"].fillna(0) * df["building_count"]).sum())),
                 "overlap_pairs": int(df["overlap_pairs_gt1pct"].sum()), "invalid": int(df["invalid_polygons"].sum()),
                 "acute_share": float(df["acute_share"].mean())}
            e = {"median": {k: float(df[k].median()) for k in df.columns if k != "country"},
                 "w1_iqr": {**{k: w1_iqr(df[k], real[k]) for k in BKEYS + PRIMARY}, "pooled_log_area": wasserstein1(gen_la, real_la) / la_iqr},
                 "validity": v, "diversity": ev.diversity(lays), "nearest_train_iou": float(np.median(ev.nearest_train_similarity(lays, train_lays))),
                 "rules": {k: int(v_) for k, v_ in pd.DataFrame([x.rules for x in st]).sum().items()}, "parcel_stats": pstats}
            if m != "S":
                prow = {c: [] for c in obs_samp}
                mor = {c: ([], []) for c in obs_samp}
                for x, parc, c in zip(st, P, countries):
                    polys = [p["polygon"] for p in parc]
                    rows, _ = parcel_table(polys, x.buildings, x.graph.segments())
                    rows = pd.DataFrame(rows)
                    if not len(rows):
                        continue
                    prow[c].append(rows)
                    xx, yy = coverage_autocorrelation(rows.coverage.to_numpy(), rows.cut.to_numpy(), parcel_neighbours(polys))
                    mor[c][0].append(xx)
                    mor[c][1].append(yy)
                e["parcels"] = {}
                for c in obs_samp:
                    gs = parcel_samples(pd.concat(prow[c], ignore_index=True))
                    e["parcels"][c] = {"w1_iqr": {k: w1_iqr(gs[k], obs_samp[c][k]) for k in PARCEL_KEYS},
                                       "median": {k: float(np.median(gs[k])) for k in PARCEL_KEYS},
                                       "morans_I": morans_i(np.concatenate(mor[c][0]), np.concatenate(mor[c][1]))}
            res["methods"][name] = e
            print(f"{name}: buildings {int(df.building_count.sum())} ({time.time() - t0:.0f}s)", flush=True)
    keys6 = PRIMARY + ["pooled_log_area"]
    for m, p_ in PREDECESSOR.items():
        for s in SEEDS:
            e, p = res["methods"][f"{m} seed {s}"], res["methods"][f"{p_} seed {s}"]
            better = {k: bool(e["w1_iqr"][k] < p["w1_iqr"][k]) for k in keys6}
            v = e["validity"]
            dec = {"improves_on_predecessor": sum(better.values()) >= 4, "detail": better,
                   "G1_validity": v["on_road_gt1m2"] == 0 and v["overlap_pairs"] == 0 and v["invalid"] == 0 and v["acute_share"] == 0,
                   "G2_no_regression": bool(sum(e["w1_iqr"][k] <= 1.0 for k in BKEYS) >= 7),
                   "G3_diversity": e["diversity"] >= real_div - 0.05,
                   "G4_memorisation": e["nearest_train_iou"] <= real_near + 0.10,
                   "G5_density": bool(abs(e["median"]["building_count"] / res["real"]["building_count"] - 1) <= 0.25
                   and e["w1_iqr"]["building_coverage"] <= 1.0),
                   "density_autocorrelation": all(abs(e["parcels"][c]["morans_I"] - obs_moran[c]) <= 0.15 for c in obs_moran)}
            if m == "C":
                dec["parcel_realism"] = all(sum(e["parcels"][c]["w1_iqr"][k] <= 0.5 for k in PARCEL_KEYS) >= 4 for c in obs_moran)
            else:
                mean_w = lambda x: np.mean([x["parcels"][c]["w1_iqr"][k] for c in obs_moran for k in PARCEL_KEYS])
                dec["parcel_realism_better_than_C"] = bool(mean_w(e) < mean_w(p))
            dec["success"] = all(v_ for k, v_ in dec.items() if k != "detail")
            res["decision"][f"{m} seed {s}"] = dec
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))

    # figures: real + observed parcels | S | C | D (seed 0), one tile per city, zoomed; and full tiles
    cities = []
    for k, (t, *_) in enumerate(metas):
        city = t["sample_id"][:7]
        if city not in [c for c, _ in cities]:
            cities.append((city, k))
    for zoom in (True, False):
        fig, axes = plt.subplots(len(cities), 4, figsize=(13, 3.3 * len(cities)), facecolor=SURFACE)
        for r_, (city, i) in enumerate(cities):
            t, b, g, P = metas[i]
            draw_layout(axes[r_, 0], t["graph"], b, g, cw, f"{t['sample_id']} · real, OBSERVED parcels")
            draw_parcels(axes[r_, 0], P, generated=False)
            for j, m in enumerate(("S", "C", "D"), 1):
                st, parcels = layouts[f"{m} seed 0"]
                x = st[i * PER_TILE]
                draw_layout(axes[r_, j], x.graph, x.buildings, x.green, cw, f"{m}" + ("" if m == "S" else " · GENERATED parcels"))
                if parcels[i * PER_TILE] is not None:
                    draw_parcels(axes[r_, j], parcels[i * PER_TILE])
            if zoom:
                for a in axes[r_]:
                    a.set_xlim(156, 356)
                    a.set_ylim(156, 356)
        fig.tight_layout()
        fig.savefig(OUT / ("zoom_S_C_D.png" if zoom else "full_S_C_D.png"), dpi=110, facecolor=SURFACE)
        plt.close(fig)
    for m in ("C", "D"):
        st, parcels = layouts[f"{m} seed 0"]
        for city, i in cities[:2]:
            x = st[i * PER_TILE]
            export_parcel_layout(OUT / "export" / f"{m}_s0_{metas[i][0]['sample_id']}.gpkg", x.graph, parcels[i * PER_TILE],
                                 x.buildings, x.green, cw)
    show = {n: {"w1": {k: round(e["w1_iqr"][k], 2) for k in keys6}, "buildings_median": e["median"]["building_count"],
                "coverage_w1": round(e["w1_iqr"]["building_coverage"], 2), "validity": e["validity"],
                "g2": int(sum(e["w1_iqr"][k] <= 1.0 for k in BKEYS)), "div": round(e["diversity"], 3), "near": round(e["nearest_train_iou"], 3),
                "parcels": {c: {"w1": {k: round(x, 2) for k, x in d["w1_iqr"].items()}, "moran": round(d["morans_I"], 2)}
                            for c, d in e.get("parcels", {}).items()}}
            for n, e in res["methods"].items()}
    print(json.dumps({"real_buildings_median": res["real"]["building_count"], "observed_moran": obs_moran,
                      "methods": show, "decision": res["decision"]}, indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
