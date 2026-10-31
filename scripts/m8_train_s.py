"""Milestone 8, Experiment S: train the frontage model and the fallback placer with the real-shape library.

    python scripts/m8_shape_library.py                 # once: fit the library (train footprints)
    python scripts/m8_train_s.py --model frontage --seed 0
    python scripts/m8_train_s.py --model fallback --seed 0

Pre-registered (docs/MILESTONE_8.md §5). Differences from B1 / B2, and nothing else:
  * both: the building shape is a categorical choice among the library classes (100 real shapes x 4
    frame symmetries), labelled with each real footprint's nearest class;
  * fallback only: a random 0-60 % of the real frontage buildings is removed from the pre-placed set
    (the fallback must place them), and learning rate 2.5e-4 instead of 5e-4 (B2 seed 0 diverged).
Same data, epochs and protocol as B1 (40 epochs) and B2 (30 epochs); best validation epoch kept.
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m6_build_objects import load_objects  # noqa: E402
from m6_train_placer import batches, collate  # noqa: E402
from m8_train_b1 import DATA as EDGE_DATA  # noqa: E402
from m8_train_b2 import front_masks  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.placer_data import shape_from_unbounded, tile_samples_b2
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.models.frontage_net import FrontageNetLib
from urbanmorph.models.placer import ObjectPlacer
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CKPT = PROJECT_ROOT / "checkpoints" / "m8"
RUNS = PROJECT_ROOT / "outputs" / "m8" / "runs"
LIB = PROJECT_ROOT / "outputs" / "m8" / "shape_library" / "library.npz"
FRONT = {"epochs": 40, "batch": 256, "lr": 5e-4, "lr_min": 2e-5}
FALLBACK = {"model": {"width": 32, "hidden": 256}, "extra": {"neighbours": 8},
            "train": {"step_fraction": 0.5, "chunk_tiles": 24, "batch_size": 48, "lr": 2.5e-4, "lr_min": 1e-5,
                      "epochs": 30, "front_drop": 0.6}}
FKEYS = ("scalars", "stop", "gapd", "phi", "size", "shape_cls")


def cosine(ep, c):
    return c["lr_min"] + 0.5 * (c["lr"] - c["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / c["epochs"]))


def step(model, opt, loss):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()


# ---------------------------------------------------------------- frontage
def f_dev(d, idx, device):
    b = {k: torch.from_numpy(d[k][idx]).to(device) for k in FKEYS}
    b["crops"] = torch.from_numpy(np.unpackbits(d["crops"][idx], axis=-1, count=32).astype(bool)).to(device)
    return b


@torch.no_grad()
def f_eval(model, d, device):
    model.eval()
    tot, n = {}, len(d["stop"])
    for i in range(0, n, 1024):
        for k, v in model.nll(f_dev(d, np.arange(i, min(i + 1024, n)), device)).items():
            tot[k] = tot.get(k, 0.0) + v.sum().item()
    return {"nll_per_step": sum(tot.values()) / n, **{f"{k}_per_step": v / n for k, v in tot.items()}}


def train_frontage(seed, lib, device):
    data = pickle.loads(EDGE_DATA.read_bytes())
    for d in data.values():
        d["shape_cls"] = lib.nearest(shape_from_unbounded(d["shape"])).astype(np.int64)
    tr, va = data["train"], data["val"]
    model = FrontageNetLib(256, lib_logits=lib.unbounded(np.arange(lib.n_classes))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=FRONT["lr"])
    rng = np.random.default_rng(seed)
    name = f"frontage_lib_s{seed}"
    hist, best, best_ep, t0 = [], float("inf"), 0, time.time()
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    for ep in range(1, FRONT["epochs"] + 1):
        lr = cosine(ep, FRONT)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = rng.permutation(len(tr["stop"]))
        tot = n = 0
        for i in range(0, len(perm), FRONT["batch"]):
            idx = np.sort(perm[i:i + FRONT["batch"]])
            loss = sum(v.mean() for v in model.nll(f_dev(tr, idx, device)).values())
            step(model, opt, loss)
            tot += loss.item() * len(idx)
            n += len(idx)
        v = f_eval(model, va, device)
        hist.append({"epoch": ep, "lr": lr, "train": tot / n, **{f"val_{k}": x for k, x in v.items()}})
        print(f"ep {ep:3d} train {tot / n:7.3f} val {v['nll_per_step']:7.3f} " +
              " ".join(f"{k[:-9]} {x:.2f}" for k, x in v.items() if k != "nll_per_step") + f" ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_step"] < best:
            best, best_ep = v["nll_per_step"], ep
            (CKPT / name).mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep}, CKPT / name / "model.pt")
    return name, model, hist, {"best_epoch": best_ep, "best_val_nll_per_step": best}, t0


# ---------------------------------------------------------------- fallback
def with_cls(b, lib):
    b = dict(b)
    b["shape_cls"] = lib.nearest(shape_from_unbounded(b["shape"])).astype(np.int64)
    return b


def p_dev(b, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


@torch.no_grad()
def p_eval(model, tiles, lib, device):
    """Validation NLL with the frontage buildings complete (front_drop = 0), as in B2."""
    model.eval()
    tot, n_obj = 0.0, 0
    rng = np.random.default_rng(999)
    for t in tiles:
        for k in (0, 5):
            b = tile_samples_b2(t, k, rng)
            if not b:
                continue
            n_obj += int((1 - b["stop"]).sum())
            for mb in batches(b, 96):
                tot += sum(v.sum().item() for v in model.nll(p_dev(with_cls(mb, lib), device)).values())
    return tot / max(n_obj, 1)


def train_fallback(seed, lib, device):
    tc = FALLBACK["train"]
    data = load_objects()
    masks = front_masks(data["train"] + data["val"])
    for t in data["train"] + data["val"]:
        t["front_mask"] = masks[t["sample_id"]]
    model = ObjectPlacer(FALLBACK["model"]["width"], FALLBACK["model"]["hidden"], **FALLBACK["extra"],
                         shape_lib=lib.n_classes).to(device)
    model.lib_logits.copy_(torch.from_numpy(lib.unbounded(np.arange(lib.n_classes))))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    rng = np.random.default_rng(seed)
    name = f"fallback_lib_s{seed}"
    hist, best, best_ep, t0 = [], float("inf"), 0, time.time()
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    for ep in range(1, tc["epochs"] + 1):
        lr = cosine(ep, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = rng.permutation(len(data["train"]))
        tot = n = 0
        for c in range(0, len(order), tc["chunk_tiles"]):
            b = collate([tile_samples_b2(data["train"][i], int(rng.integers(8)), rng, tc["step_fraction"], tc["front_drop"])
                         for i in order[c:c + tc["chunk_tiles"]]])
            for mb in batches(b, tc["batch_size"], rng):
                terms = model.nll(p_dev(with_cls(mb, lib), device))
                step(model, opt, sum(v.mean() for v in terms.values()))
                tot += sum(v.sum().item() for v in terms.values())
                n += int((1 - mb["stop"]).sum())
        v = p_eval(model, data["val"], lib, device)
        hist.append({"epoch": ep, "lr": lr, "train_nll_per_object": tot / max(n, 1), "val_nll_per_object": v})
        print(f"ep {ep:3d} train {tot / max(n, 1):7.3f}/obj  val {v:7.3f}/obj  ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best, best_ep = v, ep
            (CKPT / name).mkdir(parents=True, exist_ok=True)
            cfg = {**FALLBACK, "extra": {**FALLBACK["extra"], "shape_lib": lib.n_classes}}
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, CKPT / name / "model.pt")
    return name, model, hist, {"best_epoch": best_ep, "best_val_nll_per_object": best}, t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["frontage", "fallback"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    device = pick_device("auto")
    lib = ShapeLibrary.load(LIB)
    name, model, hist, res, t0 = (train_frontage if a.model == "frontage" else train_fallback)(a.seed, lib, device)
    summ = {"name": name, "params": count_parameters(model), **res, "train_seconds": round(time.time() - t0, 1)}
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
