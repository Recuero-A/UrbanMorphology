"""Milestone 20, D1: train the retrained backbones TNg-32, TNg-H, TNg-R and the TNg-H learning curve.

    python scripts/m20_train_tng.py --variant H --seed 0 --resume latest     # start, or continue after an interruption
    python scripts/m20_train_tng.py --variant 32 --seed 0 --smoke 1 --smoke-dir DIR   # pilot in DIR

Variants (configs/m20_tng.toml, docs/M20_LOG.md D1 and its implementation note):
* ``32``  TNg-32: the existing development tiles only (train 685, of which 473 with FABDEM inputs; Japan unchanged);
* ``H``   TNg-H: the existing tiles plus the train-H training windows (data/processed/m20/tng/towns_H.pkl);
* ``R``   TNg-R: the existing tiles plus the flat TNg-R windows (towns_R.pkl; scripts/m20_flat.py);
* ``H25`` / ``H50``: TNg-H on the first ceil(p x n) train-H settlements in the order sha256(lc_salt + id) (nested).

This runs scripts/m12_train.py's own ``main()`` unchanged (model, recipe, validation tiles and model selection on the
M12 val + val_t NLL, seeds, checkpointing, resume), as scripts/m18_train_tng.py does. The wrapper replaces:
* ``load_data``: FABDEM-32 m inputs and the norm of data/processed/m20/tng (scripts/m20_tng_data.py), the variant's
  added windows appended to the training list;
* ``make_samples`` (verbatim plus one key): every training decision carries ``w``, the D1 loss weight of the D3 slope
  class of its current node (FABDEM 32 m cell under the node), w_c = f_c^-1/2 / sum_c f_c^1/2 with f_c the share of the
  variant's training decisions (nodes) with a DEM in class c, so the mean weight over those decisions is 1; decisions
  without a DEM (Japanese tiles) get w = 1. Validation samples carry no ``w`` (unweighted NLL, as before);
* ``TerrainGraphTurtle``: ``pop_nll`` returns nll * w when the batch has ``w`` (identical otherwise).
Tiles are never repeated within an epoch (the D1 repeat cap is 1). The run config records
``dem_variant = "fabdem32_<variant>"`` (part of the checkpoint's config hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import m12_train as T  # noqa: E402
from m20_tng_data import CLASS_NAMES, K32, node_class  # noqa: E402

CFG = ROOT / "configs" / "m20_tng.toml"
BASE = ROOT / "configs" / "m6_roads.toml"
TRAINING = {"on": False, "w": None}


def lc_towns(ids, frac: float, salt: str):
    order = sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())
    return order[:math.ceil(frac * len(order))]


def loader(tc: dict, variant: str):
    data, fallback, graphs_p = ROOT / tc["data_dir"], ROOT / tc["fallback_dir"], ROOT / tc["graphs"]

    def load_data(dem_variant: str = ""):
        assert dem_variant == f"fabdem32_{variant}", dem_variant
        graphs = pickle.loads(graphs_p.read_bytes())
        assert set(graphs) <= {"train", "val", "val_t"}, set(graphs)        # development splits only
        terr, n_fab, n_none = {}, 0, 0
        for sp in graphs:
            for sid, _, _ in graphs[sp]:
                lid = dict(np.load(fallback / f"{sid}.npz"))
                f = data / "terrain_m12" / f"{sid}.npz"
                if lid["valid"].any():
                    d = dict(np.load(f))
                    assert np.array_equal(d["water"], lid["water"]) and np.array_equal(d["struct_label"], lid["struct_label"])
                    n_fab += 1
                else:
                    assert not f.exists(), f
                    d = {**lid, "cls32": np.full((K32, K32), -1, np.int8)}
                    n_none += 1
                terr[sid] = d
        added, towns = [], []
        if variant != "32":
            items = pickle.loads((data / f"towns_{variant[0] if variant[0] == 'R' else 'H'}.pkl").read_bytes())
            towns = sorted({t for _, t, _ in items})
            if variant in ("H25", "H50"):
                keep = set(lc_towns(towns, int(variant[1:]) / 100, tc["lc_salt"]))
                items = [x for x in items if x[1] in keep]
                towns = sorted(keep)
            for wid, town, g in items:
                assert wid not in terr, wid
                terr[wid] = dict(np.load(data / "towns" / f"{wid}.npz"))
                added.append((wid, town, g))
        graphs = {**graphs, "train": list(graphs["train"]) + added}
        # D1 loss weights: class shares over the training decisions (nodes) with a DEM
        cnt = np.zeros(5)
        for sid, _, g in graphs["train"]:
            c = node_class(terr[sid]["cls32"], g.nodes)
            np.add.at(cnt, c[c >= 0].astype(int), 1)
        f = cnt / cnt.sum()
        w = np.where(f > 0, f ** -tc["weight_power"], 0.0)
        w = w / (f * w).sum()
        TRAINING["w"] = w.astype(np.float32)
        print(f"[m20 TNg{variant}] terrain: {n_fab} existing tiles with FABDEM-32 m, {n_none} without a DEM "
              f"(unchanged M12 arrays); added {len(added)} windows from {len(towns)} settlements; train "
              f"{len(graphs['train'])} tiles", flush=True)
        print(f"[m20 TNg{variant}] decision shares by slope class " +
              ", ".join(f"{n}: {x:.4f}" for n, x in zip(CLASS_NAMES, f)) + " | weights " +
              ", ".join(f"{n}: {x:.3f}" for n, x in zip(CLASS_NAMES, w)), flush=True)
        return graphs, terr, json.loads((data / "norm.json").read_text())

    return load_data


def make_samples(g, d, rng, k, names, norm):
    """m12_train.make_samples verbatim, plus ``w`` (the D1 loss weight) on training decisions."""
    gk = T.encode(g, d["struct_label"]).transform(k)
    gj = T.jitter(gk, rng)
    tr = T.make_trace(gj, rng, turtle=True, clean_nodes=gk.nodes)
    pops, starts = T.pop_samples(tr, turtle=True), T.start_samples(tr)
    code = pops["target_class"]
    pops["target_class"], st = code % 4, code // 4
    pops["target_struct"] = np.where((st == 3) | (np.arange(code.shape[1])[None] >= pops["n_targets"][:, None]), -1, st)
    if names:
        lay = T.tile_layers(d, k, names, norm)
        pos = tr.g.nodes[tr.pop_node] if len(tr.pop_node) else np.zeros((0, 2))
        pops["tcrops"] = T.terrain_crops(lay, names, pos, T.headings(tr), norm)
        starts["tcanvas"] = np.repeat(T.tile_canvas(lay, names, norm)[None], len(starts["stop"]), 0)
    if TRAINING["on"]:
        from urbanmorph.graphs.terrain_sequence import transform_raster
        pos = tr.g.nodes[tr.pop_node] if len(tr.pop_node) else np.zeros((0, 2))
        c = node_class(transform_raster(d["cls32"], k), pos).astype(int)
        pops["w"] = np.where(c >= 0, TRAINING["w"][np.clip(c, 0, 4)], 1.0).astype(np.float32)
        assert len(pops["w"]) == len(pops["n_targets"])
    return pops, starts


_samples_for = T.samples_for


def samples_for(items, terr, rng, names, norm, syms=None):
    TRAINING["on"] = syms is None                       # m12_train: training chunks draw symmetries, validation fixes them
    try:
        return _samples_for(items, terr, rng, names, norm, syms)
    finally:
        TRAINING["on"] = False


class WeightedTGT(T.TerrainGraphTurtle):
    def pop_nll(self, batch):
        nll = super().pop_nll(batch)
        return nll * batch["w"] if "w" in batch else nll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume", metavar="latest|PATH", default=None)
    ap.add_argument("--ckpt-interval", type=float, default=None)
    ap.add_argument("--smoke", type=int, default=0, help="pilot epochs on 64 train / 16 val tiles (writes to --smoke-dir)")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg, base = tomllib.loads(CFG.read_text()), tomllib.loads(BASE.read_text())
    for k in ("model", "train", "sample"):
        assert cfg[k] == base[k], f"[{k}] of {CFG.name} differs from {BASE.name}: the TN recipe must be kept"
    tc = cfg["tng"]
    assert a.variant in tc["variants"], a.variant
    assert a.seed in (tc["lc_seeds"] if a.variant in ("H25", "H50") else tc["seeds"]), (a.variant, a.seed)
    name = tc["name"].format(variant=a.variant, seed=a.seed)
    T.CFG = CFG
    T.load_data = loader(tc, a.variant)
    T.make_samples = make_samples
    T.samples_for = samples_for
    T.TerrainGraphTurtle = WeightedTGT
    if a.smoke:
        assert a.smoke_dir, "--smoke needs --smoke-dir (pilot checkpoints never go to checkpoints/)"
        T.CKPT, T.RUNS = Path(a.smoke_dir) / "checkpoints", Path(a.smoke_dir) / "runs"
    else:
        T.CKPT, T.RUNS = ROOT / tc["checkpoints"], ROOT / tc["runs"]
        if (T.CKPT / name).exists() and not a.resume:
            sys.exit(f"{T.CKPT / name} exists: continue it with --resume latest (it is never restarted)")
    argv = ["m12_train.py", "--name", name, "--seed", str(a.seed), "--terrain", tc["terrain"],
            "--warmup", str(tc["warmup"]), "--dem-variant", f"fabdem32_{a.variant}"]
    if tc["lowpass_m"]:
        argv += ["--lowpass", str(tc["lowpass_m"])]
    if a.resume:
        argv += ["--resume", a.resume]
    if a.ckpt_interval:
        argv += ["--ckpt-interval", str(a.ckpt_interval)]
    if a.smoke:
        argv += ["--pilot", str(a.smoke)]
    print("[m20 TNg]", " ".join(argv[1:]), "| ckpt", T.CKPT / name, "| runs", T.RUNS / name, flush=True)
    sys.argv = argv
    return T.main()


if __name__ == "__main__":
    sys.exit(main())
