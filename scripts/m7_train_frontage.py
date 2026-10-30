"""Milestone 7: train the frontage model on REAL street networks and buildings.

    python scripts/m7_train_frontage.py --build        # once: frontage sequences + crops (train + val)
    python scripts/m7_train_frontage.py --seed 0

Every step of every frontage of the 565 training tiles (no augmentation: inputs are in the street's
frame). Validation: the 19 validation tiles, early stopping on NLL per step. Test tiles never read.
"""
import argparse
import json
import math
import pickle
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
from urbanmorph.layout.frontage_data import crops_for, tile_frontage_samples
from urbanmorph.models.frontage_net import FrontageNet
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m7_frontage.toml"
DATA = PROJECT_ROOT / "data" / "processed" / "m7" / "frontage.pkl"
DATA_REL = PROJECT_ROOT / "data" / "processed" / "m7" / "frontage_relative.pkl"
CKPT = PROJECT_ROOT / "checkpoints" / "m7"
RUNS = PROJECT_ROOT / "outputs" / "m7" / "runs"
KEYS = ("scalars", "stop", "gapd", "phi", "size", "shape")
TERMS = ("stop", "gap_setback", "angle", "size", "shape")


def build(relative: bool):
    data = load_objects()
    out = {}
    t0 = time.time()
    for split in ("train", "val"):
        parts = []
        for t in data[split]:
            smp = tile_frontage_samples(t, relative=relative)
            if not len(smp["stop"]):
                continue
            crops = crops_for(smp)
            parts.append({**{k: smp[k] for k in KEYS}, "crops": np.packbits(crops, axis=-1), "tile": np.full(len(smp["stop"]), len(parts))})
        out[split] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        print(split, len(out[split]["stop"]), "steps,", int((1 - out[split]["stop"]).sum()), "buildings", f"({time.time() - t0:.0f}s)", flush=True)
    path = DATA_REL if relative else DATA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))


def to_dev(d, idx, device):
    b = {k: torch.from_numpy(d[k][idx]).to(device) for k in KEYS}
    b["crops"] = torch.from_numpy(np.unpackbits(d["crops"][idx], axis=-1, count=32).astype(bool)).to(device)
    return b


@torch.no_grad()
def evaluate(model, d, device):
    model.eval()
    tot = {k: 0.0 for k in TERMS}
    n = len(d["stop"])
    for i in range(0, n, 1024):
        for k, v in model.nll(to_dev(d, np.arange(i, min(i + 1024, n)), device)).items():
            tot[k] += v.sum().item()
    return {"nll_per_step": sum(tot.values()) / n, **{f"{k}_per_step": v / n for k, v in tot.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--relative", action="store_true", help="Milestone 7b: residuals w.r.t. the previous building")
    a = ap.parse_args()
    if a.build:
        build(a.relative)
        return 0
    cfg = tomllib.loads(CFG.read_text())
    tc = cfg["train"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = pickle.loads((DATA_REL if a.relative else DATA).read_bytes())
    tr, va = data["train"], data["val"]
    model = FrontageNet(cfg["model"]["hidden"]).to(device)
    print(f"params {count_parameters(model):,}  device {device}  train steps {len(tr['stop'])}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    max_ep = a.max_epochs or tc["max_epochs"]
    rng = np.random.default_rng(a.seed)
    name = f"frontage_rel_s{a.seed}" if a.relative else f"frontage_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    best, best_ep, history = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        lr = tc["lr_min"] + 0.5 * (tc["lr"] - tc["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / max_ep))
        for gr in opt.param_groups:
            gr["lr"] = lr
        model.train()
        perm = rng.permutation(len(tr["stop"]))
        tot, n = 0.0, 0
        for i in range(0, len(perm), tc["batch_size"]):
            idx = np.sort(perm[i:i + tc["batch_size"]])
            terms = model.nll(to_dev(tr, idx, device))
            loss = sum(v.mean() for v in terms.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        v = evaluate(model, va, device)
        history.append({"epoch": ep, "lr": lr, "train_nll_per_step": tot / n, **{f"val_{k}": x for k, x in v.items()},
                        "seconds": time.time() - t0})
        print(f"ep {ep:3d} lr {lr:.1e} train {tot / n:7.3f}/step  val {v['nll_per_step']:7.3f}/step "
              f"(stop {v['stop_per_step']:.3f} gap/setback {v['gap_setback_per_step']:.2f} ang {v['angle_per_step']:.2f} "
              f"size {v['size_per_step']:.2f} shape {v['shape_per_step']:.2f})  ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_step"] < best:
            best, best_ep = v["nll_per_step"], ep
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep, "relative": a.relative},
                       CKPT / name / "model.pt")
        elif ep - best_ep >= tc["patience"]:
            break
    summary = {"name": name, "params": count_parameters(model), "best_epoch": best_ep, "epochs_run": ep,
               "best_val_nll_per_step": best, "train_seconds": round(time.time() - t0, 1),
               "train_steps": int(len(tr["stop"])), "device": str(device)}
    (RUNS / name / "history.json").write_text(json.dumps(history, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
