"""Milestone 6, Stages C + D: train the object placer on REAL street networks.

    python scripts/m6_build_roads.py && python scripts/m6_build_objects.py   # once
    python scripts/m6_train_placer.py --seed 0

Each epoch, every training tile gets a random symmetry and a fresh random object order, and a
random 20 % of its construction steps are used (uniform over steps, so dense Japanese tiles weigh
more, as they contain more buildings). Validation: all steps of 19 val tiles x 2 symmetries with a
fixed order; early stopping on NLL per object. Test tiles are never read.
"""
import argparse
import json
import math
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m6_build_objects import load_objects  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.placer_data import tile_samples
from urbanmorph.models.placer import ObjectPlacer
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m6_placer.toml"
CKPT = PROJECT_ROOT / "checkpoints" / "m6"
RUNS = PROJECT_ROOT / "outputs" / "m6" / "runs"
TERMS = ("stop", "location", "angle", "size", "shape")


def collate(samples):
    samples = [s for s in samples if s]
    return {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}


def batches(b, size, rng=None):
    n = len(b["stop"])
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield {k: v[idx[i:i + size]] for k, v in b.items()}


def to_dev(b, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, val_tiles, device):
    model.eval()
    tot = {k: 0.0 for k in TERMS}
    n_obj = n_tiles = 0
    rng = np.random.default_rng(999)
    for t in val_tiles:
        for k in (0, 5):
            b = tile_samples(t, k, rng)
            n_obj += int((1 - b["stop"]).sum())
            n_tiles += 1
            for mb in batches(b, 96):
                for name, v in model.nll(to_dev(mb, device)).items():
                    tot[name] += v.sum().item()
    return {"nll_per_object": sum(tot.values()) / max(n_obj, 1), **{f"{k}_per_object": v / max(n_obj, 1) for k, v in tot.items()},
            "nll_per_tile": sum(tot.values()) / n_tiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=None, help="timing runs only")
    a = ap.parse_args()
    cfg = tomllib.loads(CFG.read_text())
    tc = cfg["train"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = load_objects()
    train = data["train"][:a.max_tiles] if a.max_tiles else data["train"]
    model = ObjectPlacer(cfg["model"]["width"], cfg["model"]["hidden"]).to(device)
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    max_ep = a.max_epochs or tc["max_epochs"]
    rng = np.random.default_rng(a.seed)
    name = f"placer_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    best, best_ep, history = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        lr = tc["lr_min"] + 0.5 * (tc["lr"] - tc["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / max_ep))
        for gr in opt.param_groups:
            gr["lr"] = lr
        model.train()
        order = rng.permutation(len(train))
        tot, n = {k: 0.0 for k in TERMS}, 0
        for c in range(0, len(order), tc["chunk_tiles"]):
            b = collate([tile_samples(train[i], int(rng.integers(8)), rng, tc["step_fraction"])
                         for i in order[c:c + tc["chunk_tiles"]]])
            for mb in batches(b, tc["batch_size"], rng):
                terms = model.nll(to_dev(mb, device))
                loss = sum(v.mean() for v in terms.values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                for k, v in terms.items():
                    tot[k] += v.sum().item()
                n += int((1 - mb["stop"]).sum())
        v = evaluate(model, data["val"], device)
        history.append({"epoch": ep, "lr": lr, "train_nll_per_object": sum(tot.values()) / n,
                        **{f"val_{k}": x for k, x in v.items()}, "seconds": time.time() - t0})
        print(f"ep {ep:3d} lr {lr:.1e} train {sum(tot.values()) / n:7.3f}/obj  val {v['nll_per_object']:7.3f}/obj "
              f"(loc {v['location_per_object']:.2f} ang {v['angle_per_object']:.2f} size {v['size_per_object']:.2f} "
              f"shape {v['shape_per_object']:.2f} stop {v['stop_per_object']:.3f})  ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_object"] < best:
            best, best_ep = v["nll_per_object"], ep
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, CKPT / name / "model.pt")
        elif ep - best_ep >= tc["patience"]:
            break
    summary = {"name": name, "params": count_parameters(model), "best_epoch": best_ep, "epochs_run": ep,
               "best_val_nll_per_object": best, "train_seconds": round(time.time() - t0, 1),
               "n_train_tiles": len(train), "device": str(device)}
    (RUNS / name / "history.json").write_text(json.dumps(history, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
