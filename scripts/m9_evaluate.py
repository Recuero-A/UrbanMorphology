"""Milestone 9 evaluation: C (M8) vs T (tidy subdivision + templates) vs TZ (T + green zoning).

    python scripts/m9_evaluate.py      # -> outputs/m9/eval/results.json, zoom_C_T_TZ.png, full_C_T_TZ.png, export/
    python scripts/m9_evaluate.py --b  # M9b: TZ vs B -> outputs/m9/eval_b/ (criteria: plan §6)

Same 10 validation tiles with observed parcels as M8 (Spain 6, US 4), 2 seeds, 4 samples per tile, real
validation street networks. Criteria: docs/MILESTONE_9_PLAN.md §4, applied unchanged.
  C  : M8 C (EmpiricalSampler rng 100 + seed; 100-shape library, clipped) + fallback_lib green.
  T  : BlockNets seed s + ParcelNet seed s (coverage / setbacks); green = programmed green parcels + fallback_lib.
  TZ : as T + zoning; no fallback green.
  B  : TZ + M9b (street-access splitting, no-access parcels -> green / merged, inscribed-rectangle fit).
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
import shapely
import torch

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m6_evaluate import BKEYS, w1_iqr  # noqa: E402
from m7_evaluate import tile_row  # noqa: E402
from m8_evaluate import PRIMARY, load_fallback_b2  # noqa: E402
from m8_evaluate_parcels import PARCEL_KEYS, parcel_samples  # noqa: E402
from m8_parcel_records import OBS, load_records  # noqa: E402
from topo_evaluate import wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.export import draw_layout, draw_parcels, export_parcel_layout, layout_raster
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.layout.parcel_gen import EmpiricalSampler, generate_parcels, lot_jitter
from urbanmorph.layout.parcels import coverage_autocorrelation, morans_i, parcel_neighbours, parcel_table
from urbanmorph.layout.place import LayoutState, place_neural
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.layout.subdivision_metrics import tile_subdivision
from urbanmorph.layout.tidy_parcels import generate_tidy
from urbanmorph.models import evaluate as ev
from urbanmorph.models.block_net import BlockNets
from urbanmorph.models.parcel_net import NetSampler, ParcelNet
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import SURFACE

OUT = PROJECT_ROOT / "outputs" / "m9" / "eval"
CK = PROJECT_ROOT / "checkpoints"
LIB = PROJECT_ROOT / "outputs" / "m8" / "shape_library" / "library.npz"
JIT = PROJECT_ROOT / "outputs" / "m9" / "analysis" / "jitter.json"
SEEDS = (0, 1)
PER_TILE = 4
METHODS = ("C", "T", "TZ")
if "--b" in sys.argv:
    METHODS = ("TZ", "B")
    OUT = PROJECT_ROOT / "outputs" / "m9" / "eval_b"
if "--m10" in sys.argv:          # Milestone 10: B vs BF (data-based green fix), docs/M10_BLOCK_GENERATION.md §5
    METHODS = ("B", "BF")
    OUT = PROJECT_ROOT / "outputs" / "m10" / "procedural_fix"
COUNTRIES = ("spain", "united_states")
PREDECESSOR = {"T": "C"}


def green_rect(green):
    v = [g.area / shapely.minimum_rotated_rectangle(g).area for g in green if g.area > 0]
    return float(np.median(v)) if v else np.nan


def run(method, seed, metas, cw, lib, rec_train, device):
    graphs = [t["graph"] for t, *_ in metas for _ in range(PER_TILE)]
    countries = [t["country"] for t, *_ in metas for _ in range(PER_TILE)]
    fallback = load_fallback_b2(seed, device, "fallback_lib")
    rng = np.random.default_rng(200 + seed)
    states, parcels, stats = [], [], {}
    if method == "C":
        sampler = EmpiricalSampler(rec_train, np.random.default_rng(100 + seed))
    else:
        pn = ParcelNet()
        pn.load_state_dict(torch.load(CK / "m8" / f"parcel_net_s{seed}" / "model.pt", map_location="cpu")["model"])
        sampler = NetSampler(pn, {c: lot_jitter(rec_train, c) for c in COUNTRIES}, seed=100 + seed)
        nets = BlockNets()
        nets.load_state_dict(torch.load(CK / "m9" / f"block_nets_s{seed}" / "model.pt", map_location="cpu")["model"])
        nets.eval()
        jit = json.loads(JIT.read_text())["width_jitter_log_sd"]
    for k, (g, c) in enumerate(zip(graphs, countries)):
        s = LayoutState(g, cw, True)
        if method == "C":
            P, S = generate_parcels(s, c, sampler, lib, rng, cw, method="C")
        else:
            P, S = generate_tidy(s, c, nets, sampler, rng, cw, jit[c], zoning=(method in ("TZ", "B", "BF")), seed=1000 * seed + k,
                                 method=method, access=(method in ("B", "BF")), green_fix=(method == "BF"))
        s.phase = 1
        states.append(s)
        parcels.append(P)
        for key, v in S.items():
            if isinstance(v, list):
                stats[key] = np.add(stats.get(key, [0] * len(v)), v).tolist()
            else:
                stats[key] = stats.get(key, 0) + v
    if method not in ("TZ", "B", "BF"):               # TZ / B / BF: green comes from zoning only
        states = place_neural(fallback, None, cw, device=device, seed=21, states=states)
    return states, parcels, stats


def subdiv_summary(per_country_lists):
    return {c: {k: list(map(float, v)) for k, v in d.items()} for c, d in per_country_lists.items()}


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
        real_rows.append({**tile_row(t["graph"], b, g, road_polygons(t["graph"], class_widths=cw), t["country"]),
                          "green_rect": green_rect(g)})
        real_lays.append(layout_raster(t["road_raster"], b, g))
    real = pd.DataFrame(real_rows)
    real_lays = np.stack(real_lays)
    real_la = np.concatenate([np.log([p.area for p in b]) for _, b, _, _ in metas])
    la_iqr = np.percentile(real_la, 75) - np.percentile(real_la, 25)
    real_p99 = float(np.percentile(np.concatenate([[p.area for p in b] for _, b, _, _ in metas]), 99))
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
    # observed parcels: M8 parcel distributions, Moran's I, and S1-S4
    rv = rec[rec.sample_id.isin([t["sample_id"] for t, *_ in metas])]
    obs_samp = {c: parcel_samples(rv[rv.country == c]) for c in COUNTRIES}
    obs_moran, obs_sub = {}, {c: {"S1": [], "S2": [], "S2b": [], "S3": [], "S4": []} for c in COUNTRIES}
    obs_built = {c: float(((rv.coverage > 0) & (rv.n_buildings >= 1))[rv.country == c].mean()) for c in COUNTRIES}
    for c in COUNTRIES:
        xs, ys = [], []
        for t, b, g, P in metas:
            if t["country"] != c:
                continue
            rr = rv[rv.sample_id == t["sample_id"]]
            x, y = coverage_autocorrelation(rr.coverage.to_numpy(), rr.cut.to_numpy(), parcel_neighbours(P))
            xs.append(x)
            ys.append(y)
            sub = tile_subdivision(P, rr.frontage_m.to_numpy(), rr.interior.to_numpy(), rr.cut.to_numpy(), t["graph"], cw, g)
            for k in sub:
                obs_sub[c][k] += sub[k]
        obs_moran[c] = morans_i(np.concatenate(xs), np.concatenate(ys))
    obs_types = {c: np.bincount(obs_sub[c]["S3"], minlength=4) / max(len(obs_sub[c]["S3"]), 1) for c in COUNTRIES}
    countries = [t["country"] for t, *_ in metas for _ in range(PER_TILE)]
    res = {"tiles": [t["sample_id"] for t, *_ in metas], "real": {k: float(real[k].median()) for k in real.columns if k != "country"},
           "real_building_area_p99": real_p99, "observed_morans_I": obs_moran, "observed_built_share": obs_built,
           "observed_subdivision": {c: {"S1_median": float(np.median(d["S1"])), "S2_median": float(np.median(d["S2"])),
                                        "S3_type_shares": obs_types[c].tolist(), "S4_median": float(np.median(d["S4"])),
                                        "n_closed_blocks": len(d["S3"]), "n_sides": len(d["S1"])} for c, d in obs_sub.items()},
           "methods": {}, "decision": {}}
    layouts = {}
    for m in METHODS:
        for s in SEEDS:
            name = f"{m} seed {s}"
            st, P, pstats = run(m, s, metas, cw, lib, rec_train, device)
            layouts[name] = (st, P)
            df = pd.DataFrame([{**tile_row(x.graph, x.buildings, x.green, x.surface, c), "green_rect": green_rect(x.green)}
                               for x, c in zip(st, countries)])
            gen_area = np.concatenate([[p.area for p in x.buildings] for x in st if x.buildings])
            lays = np.stack([layout_raster(x.road, x.buildings, x.green) for x in st])
            v = {"on_road_gt1m2": int(round((df["share_on_road_gt1m2"].fillna(0) * df["building_count"]).sum())),
                 "overlap_pairs": int(df["overlap_pairs_gt1pct"].sum()), "invalid": int(df["invalid_polygons"].sum()),
                 "acute_share": float(df["acute_share"].mean())}
            e = {"median": {k: float(df[k].median()) for k in df.columns if k != "country"},
                 "w1_iqr": {**{k: w1_iqr(df[k], real[k]) for k in BKEYS + PRIMARY + ["green_rect"]},
                            "pooled_log_area": wasserstein1(np.log(gen_area), real_la) / la_iqr},
                 "building_area_p99": float(np.percentile(gen_area, 99)),
                 "validity": v, "diversity": ev.diversity(lays), "nearest_train_iou": float(np.median(ev.nearest_train_similarity(lays, train_lays))),
                 "rules": {k: int(v_) for k, v_ in pd.DataFrame([x.rules for x in st]).sum().items()}, "parcel_stats": pstats}
            prow = {c: [] for c in COUNTRIES}
            mor = {c: ([], []) for c in COUNTRIES}
            sub = {c: {"S1": [], "S2": [], "S2b": [], "S3": [], "S4": []} for c in COUNTRIES}
            built = {c: [] for c in COUNTRIES}
            for x, parc, c in zip(st, P, countries):
                polys = [p["polygon"] for p in parc]
                built[c] += [bool(p.get("building_idx")) for p in parc if p.get("use", "building") != "green"]
                rows, _ = parcel_table(polys, x.buildings, x.graph.segments())
                rows = pd.DataFrame(rows)
                if not len(rows):
                    continue
                rows["interior"] = rows.frontage_m <= 1.0
                prow[c].append(rows)
                xx, yy = coverage_autocorrelation(rows.coverage.to_numpy(), rows.cut.to_numpy(), parcel_neighbours(polys))
                mor[c][0].append(xx)
                mor[c][1].append(yy)
                sd = tile_subdivision(polys, rows.frontage_m.to_numpy(), rows.interior.to_numpy(), rows.cut.to_numpy(), x.graph, cw, x.green)
                for k in sd:
                    sub[c][k] += sd[k]
            e["parcels"], e["subdivision"] = {}, {}
            for c in COUNTRIES:
                gs = parcel_samples(pd.concat(prow[c], ignore_index=True))
                e["parcels"][c] = {"w1_iqr": {k: w1_iqr(gs[k], obs_samp[c][k]) for k in PARCEL_KEYS},
                                   "median": {k: float(np.median(gs[k])) for k in PARCEL_KEYS},
                                   "morans_I": morans_i(np.concatenate(mor[c][0]), np.concatenate(mor[c][1]))}
                types = np.bincount(sub[c]["S3"], minlength=4) / max(len(sub[c]["S3"]), 1)
                iqr_b = max(float(np.subtract(*np.percentile(obs_sub[c]["S2b"], [75, 25]))), 0.05)
                e["subdivision"][c] = {"S1_w1_iqr": w1_iqr(sub[c]["S1"], obs_sub[c]["S1"]), "S2_w1_iqr": w1_iqr(sub[c]["S2"], obs_sub[c]["S2"]),
                                       "S2b_w1": float(wasserstein1(np.array(sub[c]["S2b"]), np.array(obs_sub[c]["S2b"])) / iqr_b) if sub[c]["S2b"] else np.nan,
                                       "built_share": float(np.mean(built[c])) if built[c] else np.nan,
                                       "S3_tv": float(0.5 * np.abs(types - obs_types[c]).sum()), "S3_type_shares": types.tolist(),
                                       "S4_median": float(np.median(sub[c]["S4"])) if sub[c]["S4"] else np.nan,
                                       "S1_median": float(np.median(sub[c]["S1"])) if sub[c]["S1"] else np.nan,
                                       "S2_median": float(np.median(sub[c]["S2"])) if sub[c]["S2"] else np.nan}
            prop, acc = pstats.get("buildings_proposed", 0), pstats.get("buildings_accepted", 0)
            e["mechanism"] = {"acceptance": acc / max(prop, 1), "clipped_to_parcel": int(pstats.get("clipped_to_parcel", 0)),
                              "not_fitted": int(pstats.get("not_fitted", 0)), "not_fitted_share": pstats.get("not_fitted", 0) / max(prop + pstats.get("not_fitted", 0), 1)}
            res["methods"][name] = e
            print(f"{name}: buildings {int(df.building_count.sum())} ({time.time() - t0:.0f}s)", flush=True)

    keys6 = PRIMARY + ["pooled_log_area"]
    mean_w = lambda e: float(np.mean([e["parcels"][c]["w1_iqr"][k] for c in COUNTRIES for k in PARCEL_KEYS]))
    for m in [x for x in ("T", "TZ", "B") if x in METHODS and not (x == "TZ" and "T" not in METHODS)
              and not (x == "B" and "TZ" not in METHODS)]:
        for s in SEEDS:
            e = res["methods"][f"{m} seed {s}"]
            v = e["validity"]
            guards = {"G1_validity": bool(v["on_road_gt1m2"] == 0 and v["overlap_pairs"] == 0 and v["invalid"] == 0 and v["acute_share"] == 0),
                      "G2_no_regression": bool(sum(e["w1_iqr"][k] <= 1.0 for k in BKEYS) >= 7),
                      "G3_diversity": bool(e["diversity"] >= real_div - 0.05),
                      "G4_memorisation": bool(e["nearest_train_iou"] <= real_near + 0.10),
                      "G5_density": bool(abs(e["median"]["building_count"] / res["real"]["building_count"] - 1) <= 0.25
                                         and e["w1_iqr"]["building_coverage"] <= 1.0)}
            if m == "T":
                p = res["methods"][f"C seed {s}"]
                better = {k: bool(e["w1_iqr"][k] < p["w1_iqr"][k]) for k in keys6}
                dec = {"parcel_realism": all(sum(e["parcels"][c]["w1_iqr"][k] <= 0.5 for k in PARCEL_KEYS) >= 4 for c in COUNTRIES),
                       "tidiness": all(e["subdivision"][c]["S1_w1_iqr"] <= 0.5 and e["subdivision"][c]["S2_w1_iqr"] <= 0.5
                                       and e["subdivision"][c]["S3_tv"] <= 0.15 for c in COUNTRIES),
                       "no_leftovers": all(e["subdivision"][c]["S4_median"] <= res["observed_subdivision"][c]["S4_median"] for c in COUNTRIES),
                       "no_giant_buildings": bool(e["building_area_p99"] <= 1.25 * real_p99),
                       "mechanism": bool(e["mechanism"]["clipped_to_parcel"] == 0 and v["acute_share"] == 0 and e["mechanism"]["acceptance"] >= 0.95),
                       "improves_on_C": sum(better.values()) >= 4, "detail": better, **guards}
            elif m == "B":
                p = res["methods"][f"TZ seed {s}"]
                better = {k: bool(e["w1_iqr"][k] < p["w1_iqr"][k]) for k in keys6}
                sd = e["subdivision"]
                dec = {"built_parcels": all(abs(sd[c]["built_share"] - obs_built[c]) <= 0.10 for c in COUNTRIES),
                       "density_G5": guards.pop("G5_density"),
                       "parcel_realism": all(sum(e["parcels"][c]["w1_iqr"][k] <= 0.5 for k in PARCEL_KEYS) >= 4 for c in COUNTRIES),
                       "tidiness": all(sd[c]["S1_w1_iqr"] <= 0.5 and sd[c]["S2b_w1"] <= 0.5 and sd[c]["S3_tv"] <= 0.15 for c in COUNTRIES),
                       "no_leftovers": all(sd[c]["S4_median"] <= res["observed_subdivision"][c]["S4_median"] + 0.001 for c in COUNTRIES),
                       "no_giant_buildings": bool(e["building_area_p99"] <= 1.25 * real_p99),
                       "mechanism": bool(e["mechanism"]["clipped_to_parcel"] == 0 and v["acute_share"] == 0
                                         and e["mechanism"]["acceptance"] >= 0.95 and e["mechanism"]["not_fitted_share"] <= 0.05),
                       "improves_on_TZ": sum(better.values()) >= 4, "detail": better, **guards}
            else:
                dec = {"green_coverage": bool(e["w1_iqr"]["green_coverage"] <= 0.5),
                       "green_shape": bool(e["w1_iqr"]["green_rect"] <= 0.5),
                       "parcel_realism_vs_T": bool(mean_w(e) <= mean_w(res["methods"][f"T seed {s}"]) + 0.05), **guards}
            dec["success"] = all(x for k, x in dec.items() if k != "detail")
            res["decision"][f"{m} seed {s}"] = dec
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))

    cities = []
    for k, (t, *_) in enumerate(metas):
        city = t["sample_id"][:7]
        if city not in [c for c, _ in cities]:
            cities.append((city, k))
    for zoom in (True, False):
        fig, axes = plt.subplots(len(cities), 1 + len(METHODS), figsize=(3.3 * (1 + len(METHODS)), 3.3 * len(cities)), facecolor=SURFACE)
        for r_, (city, i) in enumerate(cities):
            t, b, g, P = metas[i]
            draw_layout(axes[r_, 0], t["graph"], b, g, cw, f"{t['sample_id']} · real, OBSERVED parcels")
            draw_parcels(axes[r_, 0], P, generated=False)
            for j, m in enumerate(METHODS, 1):
                st, parcels = layouts[f"{m} seed 0"]
                x = st[i * PER_TILE]
                draw_layout(axes[r_, j], x.graph, x.buildings, x.green, cw, f"{m} · GENERATED parcels")
                draw_parcels(axes[r_, j], parcels[i * PER_TILE])
            if zoom:
                for a in axes[r_]:
                    a.set_xlim(156, 356)
                    a.set_ylim(156, 356)
        fig.tight_layout()
        tag = "_".join(METHODS)
        fig.savefig(OUT / (f"zoom_{tag}.png" if zoom else f"full_{tag}.png"), dpi=110, facecolor=SURFACE)
        plt.close(fig)
    for m in [x for x in ("T", "TZ", "B") if x in METHODS]:
        st, parcels = layouts[f"{m} seed 0"]
        for city, i in cities[:2]:
            x = st[i * PER_TILE]
            export_parcel_layout(OUT / "export" / f"{m}_s0_{metas[i][0]['sample_id']}.gpkg", x.graph, parcels[i * PER_TILE],
                                 x.buildings, x.green, cw)
    print(json.dumps(res["decision"], indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
