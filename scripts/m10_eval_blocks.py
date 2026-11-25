"""Milestone 10: evaluate BlockGen as pre-registered (docs/M10_BLOCK_GENERATION.md §5.2-5.3).

    python scripts/m10_eval_blocks.py        # -> outputs/m10/blocks/eval.json + figures

G  : buildings on the REAL validation streets of the 10 M9 evaluation tiles (observed parcels: Spain 6, US 4),
     2 seeds x 4 samples per tile, M9 building metrics (tile_row) vs real; compared with the procedural B / BF
     numbers of outputs/m10/procedural_fix/results.json (same tiles, same metric code). Also all 19 val tiles.
I  : inpainting on validation blocks (single-window blocks, >= 3 real buildings, >= 1500 m2): the real
     buildings on one side of a random line through the block's centroid are given; the hidden side is
     generated (a) by inpainting, (b) unconditionally; per-block absolute errors of building coverage and
     count on the hidden side; tile medians; paired tile bootstrap.
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
from rasterio import features

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m6_evaluate import BKEYS, w1_iqr  # noqa: E402
from m7_evaluate import tile_row  # noqa: E402
from m8_parcel_records import OBS  # noqa: E402
from m10_train_blockgen import load_windows  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.block_gen import generate_buildings
from urbanmorph.layout.export import draw_layout
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.models.block_diffusion import BlockDiffusion
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "blocks"
CK = PROJECT_ROOT / "checkpoints" / "m10"
PROC = PROJECT_ROOT / "outputs" / "m10" / "procedural_fix" / "results.json"
SKEYS = [k for k in BKEYS if k != "green_coverage"]
SEEDS, PER_TILE, N_BOOT = (0, 1), 4, 1000


def load_model(seed, device):
    ck = torch.load(CK / f"blockgen_s{seed}" / "model.pt", map_location=device)
    m = BlockDiffusion(len(ck["cities"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval(), ck["cities"]


def n_components(mask):
    return sum(1 for g, v in features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), connectivity=4)
               if len(g["coordinates"][0]) and np.abs(np.asarray(g["coordinates"][0])).size and _area(g) >= 3)


def _area(g):
    c = np.asarray(g["coordinates"][0])
    return 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) - np.dot(c[:, 1], np.roll(c[:, 0], 1)))


def tile_metrics(tiles, cw, buildings, countries):
    return pd.DataFrame([tile_row(t["graph"], b, [], road_polygons(t["graph"], class_widths=cw), c)
                         for t, b, c in zip(tiles, buildings, countries)])


def validity(tiles, cw, buildings):
    on_road, total, overl = 0.0, 0.0, 0
    for t, bs in zip(tiles, buildings):
        rp = road_polygons(t["graph"], class_widths=cw)
        for b in bs:
            total += b.area
            on_road += b.intersection(rp).area
        import shapely
        tree = shapely.STRtree(bs) if bs else None
        if tree is not None:
            for i, b in enumerate(bs):
                overl += sum(1 for j in tree.query(b) if j > i and bs[j].intersection(b).area > 1.0)
    return {"share_area_on_road": on_road / max(total, 1.0), "overlap_pairs_gt1m2": overl}


def inpainting(model, cities, device, seed):
    rows, blk, bld, _ = load_windows()
    nwin = {}
    for r in rows:
        key = (r["sample_id"], r["block_id"])
        nwin[key] = nwin.get(key, 0) + 1
    rng = np.random.default_rng(500 + seed)
    sel, conds_i, conds_u, hidden, city, tiles = [], [], [], [], [], []
    for i, r in enumerate(rows):
        if r["split"] != "val" or nwin[(r["sample_id"], r["block_id"])] != 1 or r["block_area"] < 1500:
            continue
        b, x = blk[i].astype(np.float32), bld[i].astype(np.float32)
        if n_components(x) < 3:
            continue
        ys, xs = np.nonzero(b)
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[:128, :128]
        km = (((xx - xs.mean()) * np.cos(a) + (yy - ys.mean()) * np.sin(a)) > 0).astype(np.float32)
        conds_i.append(np.stack([b, x * km, km]))
        conds_u.append(np.stack([b, np.zeros_like(b), np.zeros_like(b)]))
        hidden.append((b * (1 - km), x))
        city.append(r["city"])
        tiles.append(r["sample_id"])
        sel.append(i)
    gen = torch.Generator().manual_seed(700 + seed)
    outs = {}
    for name, conds in (("inpaint", conds_i), ("uncond", conds_u)):
        o = []
        for s in range(0, len(conds), 96):
            c = torch.from_numpy(np.stack(conds[s:s + 96])).to(device)
            ct = torch.as_tensor(city[s:s + 96], device=device)
            o.append(model.sample(c, ct, generator=gen).cpu().numpy()[:, 0])
        outs[name] = np.concatenate(o)
    per = []
    for k, (H, x) in enumerate(hidden):
        hs = H.sum()
        real_cov, real_n = (x * H).sum() / hs, n_components(x * H)
        row = {"tile": tiles[k]}
        for name in outs:
            g = outs[name][k] * H
            row[f"{name}_cov_err"] = abs(g.sum() / hs - real_cov)
            row[f"{name}_n_err"] = abs(n_components(g) - real_n)
        per.append(row)
    df = pd.DataFrame(per)
    tm = df.groupby("tile").median()
    res = {"n_blocks": len(df), "n_tiles": len(tm)}
    brng = np.random.default_rng(0)
    for q in ("cov", "n"):
        d = (tm[f"inpaint_{q}_err"] - tm[f"uncond_{q}_err"]).to_numpy()
        boot = [np.mean(d[brng.integers(len(d), size=len(d))]) for _ in range(N_BOOT)]
        res[q] = {"inpaint_median_err": float(df[f"inpaint_{q}_err"].median()), "uncond_median_err": float(df[f"uncond_{q}_err"].median()),
                  "tile_mean_diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    res["I1_pass"] = bool(res["cov"]["inpaint_median_err"] < res["cov"]["uncond_median_err"] and res["cov"]["ci95"][1] < 0)
    res["I2_pass"] = bool(res["n"]["inpaint_median_err"] < res["n"]["uncond_median_err"])
    return res, (sel[:6], conds_i[:6], outs["inpaint"][:6], outs["uncond"][:6], hidden[:6])


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")
    obs = pickle.loads(OBS.read_bytes())
    val = data["val"]
    m9 = [t for t in val if t["sample_id"] in obs]
    real_b = {}
    for t in val:
        r = idx.loc[t["sample_id"]]
        real_b[t["sample_id"]] = tile_polygons(sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg",
                                               r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
    proc = json.loads(PROC.read_text())
    res = {"tiles_m9": [t["sample_id"] for t in m9], "methods": {}, "procedural": {}}
    for tag, tiles in (("m9_tiles", m9), ("all_val", val)):
        real = tile_metrics(tiles, cw, [real_b[t["sample_id"]] for t in tiles], [t["country"] for t in tiles])
        res[f"real_{tag}"] = {k: float(real[k].median()) for k in SKEYS}
        for seed in SEEDS:
            model, cities = load_model(seed, device)
            city_of = [cities.index(idx.loc[t["sample_id"]].city) for t in tiles for _ in range(PER_TILE)]
            graphs = [t["graph"] for t in tiles for _ in range(PER_TILE)]
            ts = time.time()
            bs = generate_buildings(model, graphs, city_of, cw, device=device, seed=100 + seed)
            secs = time.time() - ts
            rep = [t for t in tiles for _ in range(PER_TILE)]
            df = tile_metrics(rep, cw, bs, [t["country"] for t in rep])
            e = {"median": {k: float(df[k].median()) for k in SKEYS}, "w1_iqr": {k: w1_iqr(df[k], real[k]) for k in SKEYS},
                 "validity": validity(rep, cw, bs), "seconds_per_tile": secs / len(rep)}
            e["S_B"] = float(np.mean(list(e["w1_iqr"].values())))
            res["methods"][f"G_{tag}_s{seed}"] = e
            if tag == "m9_tiles":
                pm = {m: proc["methods"][f"{m} seed {seed}"] for m in ("B", "BF")}
                ps = {m: float(np.mean([pm[m]["w1_iqr"][k] for k in SKEYS])) for m in pm}
                best = min(ps, key=ps.get)
                res["procedural"][f"s{seed}"] = {"S_B": ps, "best": best}
                rc = res[f"real_{tag}"]["building_count"]
                e["criteria"] = {"G1_better_than_procedural": e["S_B"] < ps[best],
                                 "G2_validity": e["validity"]["share_area_on_road"] < 0.01 and e["validity"]["overlap_pairs_gt1m2"] == 0,
                                 "G3_density": abs(e["median"]["building_count"] / rc - 1) <= 0.25 and e["w1_iqr"]["building_coverage"] <= 1.0}
                e["pass"] = all(e["criteria"].values())
                if seed == 0:
                    res["_layouts"] = (tiles, bs)
            print(f"G {tag} seed {seed}: S_B {e['S_B']:.3f} count {e['median']['building_count']:.0f} "
                  f"({secs:.0f}s) {e.get('criteria', '')}", flush=True)
    tiles, bs = res.pop("_layouts")
    res["inpainting"] = {}
    for seed in SEEDS:
        model, cities = load_model(seed, device)
        r, ex = inpainting(model, cities, device, seed)
        res["inpainting"][f"s{seed}"] = r
        print("inpainting", seed, json.dumps(r), flush=True)
        if seed == 0:
            fig, axes = plt.subplots(len(ex[0]), 4, figsize=(9, 2.3 * len(ex[0])), facecolor=SURFACE)
            for k in range(len(ex[0])):
                b, kb, km = ex[1][k]
                H, x = ex[4][k]
                for a, img, ttl in zip(axes[k], (b * 0.25 + x * 0.75, b * 0.25 + kb * 0.75 + km * 0.1, b * 0.25 + ex[2][k] * 0.75,
                                                  b * 0.25 + ex[3][k] * 0.75), ("real", "given (shaded side)", "inpainted", "unconditional")):
                    a.imshow(img, cmap="Greys", vmin=0, vmax=1)
                    a.set_title(ttl, fontsize=8)
                    a.axis("off")
            fig.tight_layout()
            fig.savefig(OUT / "inpainting.png", dpi=110, facecolor=SURFACE)
            plt.close(fig)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    # figure: real vs BlockGen on the 10 M9 tiles (one sample), streets + buildings
    n = len(tiles)
    fig, axes = plt.subplots(n, 2, figsize=(6.6, 3.3 * n), facecolor=SURFACE)
    for i, t in enumerate(tiles):
        draw_layout(axes[i, 0], t["graph"], real_b[t["sample_id"]], [], cw, f"{t['sample_id']} · real")
        draw_layout(axes[i, 1], t["graph"], bs[i * PER_TILE], [], cw, "BlockGen (real streets)")
    fig.tight_layout()
    fig.savefig(OUT / "blockgen_vs_real.png", dpi=100, facecolor=SURFACE)
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
