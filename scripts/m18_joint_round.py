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

eval        D. The frozen joint arm J (NA-SMC on TNg_s0-s2, inference seeds 1-2) on GEN2 and the 16 natural windows
                                                                              ->  runs/<where>__<backbone>__s<seed>__joint.json
report         J against off (TNg NA-SMC, nothing switched on; the stored runs) and, for information, against the D12
               potentials (TM): paired block bootstrap as D12, and D13's margins checked one by one  ->  report.json

whole_check    The whole-town runner below against a stored site run (TNg + D12 potentials, Petra, seed 1): bit-identical?
whole_fit   D14. Whole-town density control, cross-fitted: per-class offsets delta on tm_len (B + delta, clip [-3, 6]),
               fitted on one site (whole-town NA-SMC on TNg_s0, one torch thread, seed 1; 5 updates; classes with no
               measurable land take the nearest measurable class's delta; selection: lowest mean |log(gen / ref)|)
                                                                              ->  whole_fit_<site>.json
whole_eval     Each site with the OTHER site's delta, seeds 1-3              ->  runs/whole__<site>__cross__s<seed>.pkl
whole_report   Against off and D12's TNg + potentials (the stored site runs); D14's site margins  ->  whole_report.json
whole_pooled   For the fresh-city confirmation only: delta fitted on both sites pooled  ->  configs/m18_terrain_potentials_TNg_whole.json

    python scripts/m18_joint_round.py diag|diag_report|fit|eval|report|whole_check|whole_fit|whole_eval|whole_report|whole_pooled
        [--workers 6]  ->  outputs/m18/joint_round/
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


# ---------------------------------------------------------------- D. evaluation on GEN2 and the natural windows
def joint_params():
    p = json.loads(JOINT.read_text())["params"]
    return {k: tuple(v) if isinstance(v, list) else v for k, v in p.items()}


def eval_file(where, bb, s):
    return OUT / "runs" / f"{where}__{bb}__s{s}__joint.json"


def eval_one(job):
    where, bb, s = job
    setup()
    from m18_terrain_potentials import eval_tiles
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_to_json
    T = eval_tiles(where)
    ds = [t["d"] for t in T]
    crop = model_inputs(load_backbone(bb, "cpu")[3], [t["sid"] for t in T], ds)
    t0 = time.time()
    gs = sample(bb, ds, s, joint_params(), crop)
    atomic_json(eval_file(where, bb, s), {"where": where, "backbone": bb, "seed": s, "arm": "joint", "params": json.loads(
        JOINT.read_text())["params"], "sids": [t["sid"] for t in T], "graphs": [graph_to_json(g) for g in gs],
        "seconds": round(time.time() - t0, 1)})
    return job, round(time.time() - t0)


def stage_eval(workers):
    from multiprocessing import Pool
    js = [(w, f"TNg_s{b}", s) for w in ("gen2", "natural") for b in range(3) for s in (1, 2) if not eval_file(w, f"TNg_s{b}", s).exists()]
    print(len(js), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for job, sec in p.imap_unordered(eval_one, js, chunksize=1):
            print(job, sec, "s", flush=True)
    return 0


def junctions(g):
    if len(g.edges) == 0:
        return 0, 0
    deg = np.bincount(g.edges.ravel(), minlength=len(g.nodes))
    return int((deg >= 3).sum()), int((deg >= 5).sum())


def stage_report():
    from m18_terrain_potentials import (RUN_STATS, SEEDS as P_SEEDS, Stack, eval_tiles, load_graphs, paired, reference_runs,
                                        run_stats, tile_outcomes)
    from m18_terrain_reference import KEYS, Field
    from urbanmorph.experiments.graph_io import graph_from_json
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    rr = reference_runs()
    city_run = {c: run_stats(v) for c, v in rr.items()}
    run_sd = {k: float(np.nanstd([city_run[c][k] for c in city_run], ddof=1)) for k in RUN_STATS}
    res = {"params": json.loads(JOINT.read_text()), "where": {}, "margins": {}}
    for where in ("gen2", "natural"):
        T = eval_tiles(where)
        F = {t["sid"]: Field(t["d"]) for t in T}
        pc = {}
        tgt = tgt_runs = None
        out = {"tiles": len(T)}
        if where == "gen2":
            R = Stack([tile_outcomes(t["graph"], F[t["sid"]], t["sid"], pc) for t in T])
            st, _ = R.stats(np.ones(R.n), ref, sd)
            tgt, tgt_runs = {k: st[k] for k in RUN_STATS}, R.run_len
        units = {a: [] for a in ("J", "off", "TM")}
        jn = {a: [] for a in units}
        blocks = []
        for b in range(3):
            bb = f"TNg_s{b}"
            for s in P_SEEDS:
                gsets = {"J": [graph_from_json(x) for x in json.loads(eval_file(where, bb, s).read_text())["graphs"]],
                         "off": load_graphs(where, bb, s, "off", "TNg"), "TM": load_graphs(where, bb, s, "on", "TNg")}
                for a, gs in gsets.items():
                    units[a] += [tile_outcomes(g, F[t["sid"]], t["sid"], pc) for g, t in zip(gs, T)]
                    jn[a] += [junctions(g) for g in gs]
                blocks += [str(t["block"]) for t in T]
        for name, (a, b_) in (("J_minus_off", ("J", "off")), ("J_minus_TM", ("J", "TM"))):
            X = Stack(units[a] + units[b_])
            J3, J5 = (np.array([j[i] for j in jn[a] + jn[b_]], float) for i in (0, 1))

            def fn(w, X=X, J3=J3, J5=J5):
                st, cur = X.stats(w, ref, sd, tgt, tgt_runs, run_sd)
                return {**st, **{f"density_{k}": cur[k]["density"] for k in KEYS}, "five_way": float(w @ J5 / max(w @ J3, 1e-9))}
            n = len(units[a])
            for arm, w in ((a, np.r_[np.ones(n), np.zeros(n)]), (b_, np.r_[np.zeros(n), np.ones(n)])):
                out.setdefault(arm, fn(w))
            out[name] = paired(X, np.array(blocks), fn)
            print(where, name, {k: round(v["delta"], 3) for k, v in out[name].items() if k in ("terrain_realism", "geo_connected",
                  "coverage_20-30", "coverage_30-inf", "five_way", "density_0-5", "density_5-10")}, flush=True)
        res["where"][where] = out
        atomic_json(OUT / "report.json", res)
    # D13 D margins (sites: added once the whole-town arm exists)
    M = res["margins"]
    for where in ("gen2", "natural"):
        o = res["where"][where]
        M[f"{where}: realism CI below 0"] = o["J_minus_off"]["terrain_realism"]["hi"] < 0
        for k in ("0-5", "5-10"):
            M[f"{where}: flat density {k} within 1 SD"] = abs(o["J"][f"density_{k}"] - ref[k]["density"]) <= sd[k]["density"]
    g = res["where"]["gen2"]["J_minus_off"]
    for k in ("coverage_20-30", "coverage_30-inf", "geo_connected"):
        M[f"gen2: {k} lower CI >= -0.02"] = g[k]["lo"] >= -0.02
    M["gen2: 5+-way share <= 1.5 %"] = res["where"]["gen2"]["J"]["five_way"] <= 0.015
    M["gen2: climb gap CI below 0 (broader-improvement claim only, not a margin)"] = g["climb_run_score"]["hi"] < 0
    M = {k: bool(v) for k, v in M.items()}
    res["margins"] = M
    atomic_json(OUT / "report.json", res)
    print(json.dumps(M, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------- D14. whole-town density control
WHOLE = ROOT / "configs" / "m18_terrain_potentials_TNg_whole.json"
WHOLE_ITERS = 5


def whole_graph(site_name, over, seed):
    """One whole-town run: site_benchmark.generate_one's learned branch (TNg_s0, NA-SMC, no supplied roads), one thread."""
    import torch
    torch.set_num_threads(1)
    import site_benchmark as sb
    from m16_kaist_all import AnchoredGlobalNet
    from urbanmorph.experiments.backbones import load_backbone
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    site = sb.Site(site_name)
    big = site.domain()
    model, names, norm, _ = load_backbone("TNg_s0", "cpu")
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    cfg = net_preset(m.preset, **{**m.overrides, **over})
    sm = AnchoredGlobalNet(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
    sm.anchor = None
    return sm.run(seed)[0]


def stage_whole_check():
    import site_benchmark as sb
    from m18_terrain_potentials import frozen_params
    t0 = time.time()
    g = whole_graph("petra", frozen_params("TNg"), 1)
    ref = pickle.loads((sb.Site("petra").cache / f"{sb.key_of('NA-SMC', 'TNg + TM')}_s1.pkl").read_bytes())["graph"]
    same = g.nodes.shape == ref.nodes.shape and np.array_equal(g.nodes, ref.nodes) and np.array_equal(g.edges, ref.edges)
    atomic_json(OUT / "whole_check.json", {"identical": bool(same), "nodes": [len(g.nodes), len(ref.nodes)],
                                           "seconds": round(time.time() - t0)})
    print("identical:", same, len(g.nodes), len(ref.nodes), flush=True)
    return 0


def site_moments(names, graphs):
    """Pooled per-class curves of whole-town graphs (cropped to each site's limits) over one or more sites."""
    import site_benchmark as sb
    from m18_terrain_potentials import SiteField
    from m18_terrain_reference import pool, tile_sums
    sums = []
    for name, g in zip(names, graphs):
        site = sb.Site(name)
        big, mask = site.domain(), np.load(site.data / "city_mask.npz")["mask"]
        sums.append(tile_sums(sb.crop(g, mask, site), SiteField(site, big, mask)))
    return pool(sums)


def whole_one(job):
    name, over, seed = job
    return name, whole_graph(name, over, seed)


def fit_delta(names, tag, workers=2):
    """D14 iterative scaling of delta on the pooled moments of ``names`` (one whole-town run per site per iteration)."""
    from multiprocessing import Pool
    from m18_terrain_reference import KEYS
    ref = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())["reference"]
    B = joint_params()
    base = np.array(B["tm_len"], float)
    delta = np.zeros(5)
    log = []
    for it in range(WHOLE_ITERS + 1):
        t0 = time.time()
        over = {**B, "tm_len": tuple(np.clip(base + delta, -3, 6).tolist())}
        if len(names) == 1:                              # inside stage_whole_fit's (daemonic) pool: no nested pool
            res = dict([whole_one((names[0], over, 1))])
        else:
            with Pool(min(workers, len(names)), maxtasksperchild=1) as p:
                res = dict(p.map(whole_one, [(n, over, 1) for n in names]))
        cur = site_moments(names, [res[n] for n in names])
        meas = [q for q, k in enumerate(KEYS) if np.isfinite(cur[k]["density"]) and cur[k]["density"] > 0]
        gap = float(np.mean([abs(np.log(cur[KEYS[q]]["density"] / ref[KEYS[q]]["density"])) for q in meas]))
        log.append({"iter": it, "delta": delta.tolist(), "tm_len": list(over["tm_len"]), "gap": gap, "curves": cur,
                    "seconds": round(time.time() - t0)})
        atomic_json(OUT / f"whole_fit_{tag}.json", {"sites": list(names), "log": log})
        print(tag, "iter", it, "gap", round(gap, 3), "density", [round(cur[k]["density"], 1) for k in KEYS],
              "delta", np.round(delta, 2), f"{time.time() - t0:.0f}s", flush=True)
        if it == WHOLE_ITERS:
            break
        for q in meas:
            delta[q] += np.log(cur[KEYS[q]]["density"] / ref[KEYS[q]]["density"])
        for q in range(5):
            if q not in meas and meas:
                delta[q] = delta[min(meas, key=lambda m: abs(m - q))]
    best = min(log, key=lambda r: r["gap"])
    out = {"sites": list(names), "iteration": best["iter"], "gap": best["gap"], "delta": best["delta"], "tm_len": best["tm_len"],
           "rule": "lowest mean |log(generated / reference density)| over measurable classes (D14)"}
    atomic_json(OUT / f"whole_fit_{tag}.json", {"sites": list(names), "log": log, "selected": out})
    print(tag, "selected iteration", best["iter"], round(best["gap"], 3), flush=True)
    return out


def stage_whole_fit(workers):
    from multiprocessing import Pool
    with Pool(2) as p:                                   # the two cross-fits in parallel (one thread per run)
        p.starmap(fit_delta, [((n,), n, 1) for n in SITES])
    return 0


def cross_over(name):
    other = [s for s in SITES if s != name][0]
    sel = json.loads((OUT / f"whole_fit_{other}.json").read_text())["selected"]
    return {**joint_params(), "tm_len": tuple(sel["tm_len"])}, other


def whole_file(name, seed):
    return OUT / "runs" / f"whole__{name}__cross__s{seed}.pkl"


def whole_eval_one(job):
    name, seed = job
    t0 = time.time()
    over, other = cross_over(name)
    g = whole_graph(name, over, seed)
    f = whole_file(name, seed)
    f.with_suffix(".tmp").write_bytes(pickle.dumps({"graph": g, "delta_from": other, "params": over, "seconds": time.time() - t0}))
    f.with_suffix(".tmp").rename(f)
    return job, round(time.time() - t0)


def stage_whole_eval(workers):
    from multiprocessing import Pool
    js = [(n, s) for n in SITES for s in SEEDS if not whole_file(n, s).exists()]
    print(len(js), "jobs", flush=True)
    with Pool(workers, maxtasksperchild=1) as p:
        for job, sec in p.imap_unordered(whole_eval_one, js, chunksize=1):
            print(job, sec, "s", flush=True)
    return 0


def stage_whole_report():
    import site_benchmark as sb
    from m18_terrain_potentials import GEO_N, SiteField, geo_access, geo_points, geo_stats
    from m18_terrain_reference import KEYS, pool, score, tile_sums
    ref_all = json.loads((ROOT / "outputs" / "m18" / "terrain_reference" / "curves.json").read_text())
    ref, sd = ref_all["reference"], ref_all["between_city_sd"]
    res = {"sites": {}, "margins": {}}
    for name in SITES:
        sb.init(name)
        site, big, mask = sb.G["site"], sb.G["big"], sb.G["mask"]
        F = SiteField(site, big, mask)
        pts = geo_points(name, F, n=int(round(GEO_N * float(F.land.sum() * 16 / 1e6) / 0.262144)))
        rows = []
        for arm in ("J", "off", "TM"):
            for s in SEEDS:
                if arm == "J":
                    g = pickle.loads(whole_file(name, s).read_bytes())["graph"]
                else:
                    ver = "TNg" if arm == "off" else "TNg + TM"
                    g = pickle.loads((site.cache / f"{sb.key_of('NA-SMC', ver)}_s{s}.pkl").read_bytes())["graph"]
                gc = sb.crop(g, mask, site)
                cur = pool([tile_sums(gc, F)])
                cs = sb.city_stats(gc)
                rows.append({"arm": arm, "seed": s, "terrain_realism": score(cur, ref, sd), "TAS": sb.tas(gc)["TAS"],
                             "km_per_km2": cs["street_km_per_km2"], "largest_piece_share": cs["largest_piece_share"],
                             "five_way": five_way(gc), **geo_stats([geo_access(gc, *pts, closed=False)]),
                             **{f"density_{k}": cur[k]["density"] for k in KEYS}, **{f"coverage_{k}": cur[k]["coverage"] for k in KEYS}})
                print(name, arm, s, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        mean = {a: {k: float(np.nanmean([r[k] for r in rows if r["arm"] == a])) for k in rows[0] if k not in ("arm", "seed")}
                for a in ("J", "off", "TM")}
        res["sites"][name] = {"delta_from": cross_over(name)[1], "rows": rows, "mean": mean}
        M = res["margins"]
        for k in ("0-5", "5-10"):
            v = mean["J"][f"density_{k}"]
            M[f"{name}: flat density {k} within 1 SD"] = bool(abs(v - ref[k]["density"]) <= sd[k]["density"]) if np.isfinite(v) else "n/a"
        for k in ("largest_piece_share", "geo_connected"):
            M[f"{name}: {k} >= off - 0.05"] = bool(mean["J"][k] >= mean["off"][k] - 0.05)
        M[f"{name}: 5+-way share <= 1.5 %"] = bool(mean["J"]["five_way"] <= 0.015)
        atomic_json(OUT / "whole_report.json", res)
    print(json.dumps(res["margins"], indent=1), flush=True)
    return 0


def stage_whole_pooled():
    sel = fit_delta(SITES, "pooled", 2)
    atomic_json(WHOLE, {"frozen": time.strftime("%Y-%m-%d %H:%M"), "family": "TNg", "mode": "whole-town",
                        "rule": "D14 delta fitted on both development sites pooled (fresh-city confirmation only)",
                        "selected": sel, "params": {**{k: list(v) if isinstance(v, tuple) else v for k, v in joint_params().items()},
                                                    "tm_len": sel["tm_len"]}})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["diag", "diag_report", "fit", "eval", "report", "whole_check", "whole_fit", "whole_eval",
                                           "whole_report", "whole_pooled"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"diag": lambda: stage_diag(a.workers), "diag_report": stage_diag_report, "fit": stage_fit,
            "eval": lambda: stage_eval(a.workers), "report": stage_report, "whole_check": stage_whole_check,
            "whole_fit": lambda: stage_whole_fit(a.workers), "whole_eval": lambda: stage_whole_eval(a.workers),
            "whole_report": stage_whole_report, "whole_pooled": stage_whole_pooled}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
