"""Milestone 18, D13: the final development round for the terrain potentials (docs/M18_LOG.md D13).

diag        A. Tile mode on the two development sites: NA-SMC on TNg_s0, terrain potentials off / on (frozen D12 TNg),
               inference seeds 1-3, the M14 tile sampler on the 16 grid windows of each site domain (each with its 256 m
               margin; the model sees the same GLO-30 terrain as in whole-town mode), merged without joining
               (site_benchmark.merge)                                         ->  runs/diag__<site>__<arm>__s<seed>.pkl
diag_report    Tile vs whole-town (the stored site runs), both cropped to the city limits (site_benchmark.crop) and
               measured alike (SiteField, tile_sums, pool): density and coverage per slope class, the flat-land excess
               E (0-10 % land, pooled) over the training-city reference, and D13's rule E_tile <= 0.5 E_whole
                                                                              ->  diag.json
fit         B. Joint fit on train_cal_tiles(96), seed 1: NA-SMC on TNg_s0 with M17 F1 + F4, the terrain parameters
               (start: frozen D12 TNg; D12 updates and clips) and lam_dead / lam_island (start: M14; M14 update, clip
               [0, 6]) matched to the real dead-end and island counts on the same tiles; 8 updates, iterations 0-8
               logged; D13 selection                     ->  fit.json, configs/m18_terrain_potentials_TNg_joint.json

    python scripts/m18_joint_round.py diag|diag_report|fit [--workers 6]  ->  outputs/m18/joint_round/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "joint_round"
JOINT = ROOT / "configs" / "m18_terrain_potentials_TNg_joint.json"
SITES = ("machu_picchu", "petra")
SEEDS = (1, 2, 3)
F1F4 = {"junction_reuse_m": 15.0, "degree_prior": 1.0}          # M17 F1 + F4 (site_benchmark.FIX)
ITERS, N_FIT, EPS, LAM_MAX = 8, 96, 0.05, 6.0
D_TOL, I_TOL = 0.10, 0.05                                     # M14 calibration tolerances (relative D, absolute I)


def setup():
    import torch
    torch.set_num_threads(2)


def sample(bb, ds, seed, over=None, crop=None):
    """NA-SMC tile sampler on terrain dicts ``ds``; ``over`` replaces / extends the method's NetConfig overrides."""
    from urbanmorph.experiments.backbones import load_backbone
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import net_preset, sample_net
    from urbanmorph.graphs.tc_potentials import Potentials
    model, names, norm, _ = load_backbone(bb, "cpu")
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")       # TNg: as m18_terrain_potentials
    m = method("NA-SMC")
    cfg = net_preset(m.preset, **{**m.overrides, **(over or {})})
    gs, _, _ = sample_net(model, ds, names, norm, pots, cfg, seed=seed, device="cpu", crop_tiles=crop, voc_log=None)
    return gs


# ---------------------------------------------------------------- A. diagnostic
def run_file(site, arm, seed):
    return OUT / "runs" / f"diag__{site}__{arm}__s{seed}.pkl"


def diag_one(job):
    site_name, arm, seed = job
    setup()
    import site_benchmark as sb
    from m18_terrain_potentials import frozen_params
    from urbanmorph.graphs.tc_global import window_tiles
    site = sb.Site(site_name)
    big = site.domain()
    n = int(site.L // sb.W)
    offs = [(c * sb.W, r * sb.W) for r in range(n) for c in range(n)]          # the order of window_tiles
    t0 = time.time()
    gs = sample("TNg_s0", window_tiles(big, site.L), seed, frozen_params("TNg") if arm == "on" else None)
    g = sb.merge(gs, offs, site.L)
    f = run_file(site_name, arm, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.with_suffix(".tmp").write_bytes(pickle.dumps({"graph": g, "windows": len(gs), "seconds": time.time() - t0}))
    f.with_suffix(".tmp").rename(f)
    return job, round(time.time() - t0)


def stage_diag(workers):
    from multiprocessing import Pool
    js = [(s, a, k) for s in SITES for a in ("off", "on") for k in SEEDS if not run_file(s, a, k).exists()]
    print(len(js), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for job, sec in p.imap_unordered(diag_one, js, chunksize=1):
            print(job, sec, "s", flush=True)
    return 0


def flat_density(cur, keys=("0-5", "5-10")):
    return sum(cur[k]["street_km"] for k in keys) / max(sum(cur[k]["land_km2"] for k in keys), 1e-9)


def five_way(g):
    if len(g.edges) == 0:
        return np.nan
    deg = np.bincount(g.edges.ravel(), minlength=len(g.nodes))
    return float(np.mean(deg[deg >= 3] >= 5)) if (deg >= 3).any() else np.nan


def stage_diag_report():
    import site_benchmark as sb
    from m18_terrain_potentials import SiteField
    from m18_terrain_reference import KEYS, pool, tile_sums
    ref = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())["reference"]
    ref_flat = sum(ref[k]["street_km"] for k in ("0-5", "5-10")) / sum(ref[k]["land_km2"] for k in ("0-5", "5-10"))
    rows = []
    for name in SITES:
        sb.init(name)
        site, big, mask = sb.G["site"], sb.G["big"], sb.G["mask"]
        F = SiteField(site, big, mask)
        for arm, ver in (("off", "TNg"), ("on", "TNg + TM")):
            for s in SEEDS:
                whole = pickle.loads((site.cache / f"{sb.key_of('NA-SMC', ver)}_s{s}.pkl").read_bytes())["graph"]
                tile = pickle.loads(run_file(name, arm, s).read_bytes())["graph"]
                for mode, g in (("whole-town", whole), ("tile", tile)):
                    gc = sb.crop(g, mask, site)
                    cur = pool([tile_sums(gc, F)])
                    rows.append({"site": name, "arm": arm, "seed": s, "mode": mode, "flat_density": flat_density(cur),
                                 "excess": flat_density(cur) - ref_flat, "km_per_km2": sb.city_stats(gc)["street_km_per_km2"],
                                 "five_way": five_way(gc),
                                 "density": [cur[k]["density"] for k in KEYS], "coverage": [cur[k]["coverage"] for k in KEYS]})
                    print(name, arm, s, mode, round(rows[-1]["flat_density"], 1), round(rows[-1]["km_per_km2"], 1), flush=True)
    res = {"reference_flat_density": ref_flat, "rows": rows, "rule": "mode explains the excess if mean E_tile <= 0.5 mean "
           "E_whole (over sites and seeds), for both arms (D13 A)", "arms": {}}
    for arm in ("off", "on"):
        E = {m: float(np.mean([r["excess"] for r in rows if r["arm"] == arm and r["mode"] == m])) for m in ("tile", "whole-town")}
        res["arms"][arm] = {"E_tile": E["tile"], "E_whole": E["whole-town"], "explains": bool(E["tile"] <= 0.5 * E["whole-town"])}
    res["mode_explains_excess"] = all(v["explains"] for v in res["arms"].values())
    atomic_json(OUT / "diag.json", res)
    print(json.dumps(res["arms"], indent=1), "mode explains:", res["mode_explains_excess"], flush=True)
    return 0


# ---------------------------------------------------------------- B. joint fit
def stage_fit():
    setup()
    from m14_calibrate import counts
    from m14_experiments import train_cal_tiles
    from m18_terrain_potentials import frozen_params
    from m18_terrain_reference import KEYS, Field, pool, score, tile_sums
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    lam0 = json.loads((ROOT / "configs" / "m14_lambda.json").read_text())
    tiles = train_cal_tiles(N_FIT)
    ds = [t["d"] for t in tiles]
    crop = model_inputs(load_backbone("TNg_s0", "cpu")[3], [t["sid"] for t in tiles], ds)
    F = [Field(d) for d in ds]
    real = np.array([counts(t["graph"]) for t in tiles], float).mean(0)
    p0 = frozen_params("TNg")
    tm = {"tm_len": list(p0["tm_len"]), "tm_ci": list(p0["tm_ci"]), "tm_steep": float(p0["tm_steep"])}
    centre = tuple(p0["tm_ci_centre"])
    lam = np.array([lam0["lam_dead"], lam0["lam_island"]], float)
    rshare = sum(ref[k]["share_gt15"] * ref[k]["street_km"] for k in KEYS) / sum(ref[k]["street_km"] for k in KEYS)
    log = []
    for it in range(ITERS + 1):
        t0 = time.time()
        par = {"tm_len": tuple(tm["tm_len"]), "tm_ci": tuple(tm["tm_ci"]), "tm_ci_centre": centre, "tm_steep": float(tm["tm_steep"]),
               "lam_dead": float(lam[0]), "lam_island": float(lam[1]), **F1F4}
        gs = sample("TNg_s0", ds, 1, par, crop)
        cur = pool([tile_sums(g, f) for g, f in zip(gs, F)])
        sc = score(cur, ref, sd)
        gen = np.array([counts(g) for g in gs], float).mean(0)
        share15 = sum(cur[k]["share_gt15"] * cur[k]["street_km"] for k in KEYS if np.isfinite(cur[k]["share_gt15"])) / \
            max(sum(cur[k]["street_km"] for k in KEYS), 1e-9)
        ok = bool(abs(gen[0] - real[0]) <= D_TOL * real[0] and abs(gen[1] - real[1]) <= I_TOL)
        log.append({"iter": it, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in par.items()}, "score": sc,
                    "curves": cur, "share_gt15": share15, "D": gen[0], "I": gen[1], "network_ok": ok,
                    "five_way": float(np.nanmean([five_way(g) for g in gs])), "seconds": round(time.time() - t0, 1)})
        atomic_json(OUT / "fit.json", {"tiles": N_FIT, "real_D": real[0], "real_I": real[1], "reference_share_gt15": rshare,
                                       "log": log})
        print(f"iter {it} score {sc:.3f} D {gen[0]:.2f}/{real[0]:.2f} I {gen[1]:.3f}/{real[1]:.3f} ok {ok} "
              f"5+ {100 * log[-1]['five_way']:.1f}% share15 {share15:.3f} lam {np.round(lam, 2)} "
              f"len {np.round(tm['tm_len'], 2)} ci {np.round(tm['tm_ci'], 2)} steep {tm['tm_steep']:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if it == ITERS:
            break
        for q, k in enumerate(KEYS):                                            # D12 updates and clips
            g, r = cur[k]["density"], ref[k]["density"]
            if np.isfinite(g) and np.isfinite(r) and g > 0 and r > 0:
                tm["tm_len"][q] = float(np.clip(tm["tm_len"][q] + np.log(g / r), -3, 6))
            g, r = cur[k]["contour_index"], ref[k]["contour_index"]
            if np.isfinite(g) and np.isfinite(r):
                tm["tm_ci"][q] = float(np.clip(tm["tm_ci"][q] + 20 * (g - r), -10, 10))
        if share15 > 0 and rshare > 0:
            tm["tm_steep"] = float(np.clip(tm["tm_steep"] + 2.0 * np.log(share15 / rshare), 0, 20))
        lam = np.clip(lam + np.log((gen + EPS) / (real + EPS)), 0, LAM_MAX)       # M14 update
    good = [r for r in log if r["network_ok"]]
    if good:
        best, rule = min(good, key=lambda r: r["score"]), "lowest training score among network-ok iterations"
    else:
        pen = lambda r: r["score"] + abs(np.log((r["D"] + EPS) / (real[0] + EPS))) + abs(np.log((r["I"] + EPS) / (real[1] + EPS)))
        best, rule = min(log, key=pen), "no network-ok iteration: lowest score + |log D ratio| + |log I ratio|"
    atomic_json(JOINT, {"frozen": time.strftime("%Y-%m-%d %H:%M"), "family": "TNg", "rule": rule + " (D13 B)",
                        "iteration": best["iter"], "score_training": best["score"], "D": best["D"], "I": best["I"],
                        "real_D": real[0], "real_I": real[1], "params": best["params"]})
    print("frozen iteration", best["iter"], rule, round(best["score"], 3), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["diag", "diag_report", "fit"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"diag": lambda: stage_diag(a.workers), "diag_report": stage_diag_report, "fit": stage_fit}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
