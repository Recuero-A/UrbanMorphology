"""Milestone 8, Experiment B1: train the edge-parameterised frontage model (FrontageNetEdge).

    python scripts/m8_train_b1.py --build
    python scripts/m8_train_b1.py --seed 0

Same data, budget and protocol as Milestone 7's frontage model (40 epochs, best validation epoch kept,
no early stopping); only the parameterisation (edge gaps, front distance, rectangle flag) differs.
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
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.frontage_edge import crops_for, tile_edge_samples
from urbanmorph.models.frontage_net import FrontageNetEdge
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

DATA = PROJECT_ROOT / "data" / "processed" / "m8" / "frontage_edge.pkl"
CKPT = PROJECT_ROOT / "checkpoints" / "m8"
RUNS = PROJECT_ROOT / "outputs" / "m8" / "runs"
KEYS = ("scalars", "stop", "gapd", "phi", "size", "shape", "rect")
EPOCHS, BATCH, LR, LR_MIN = 40, 256, 5e-4, 2e-5


def build():
    data = load_objects()
    out = {}
    t0 = time.time()
    for split in ("train", "val"):
        parts = []
        for t in data[split]:
            smp = tile_edge_samples(t)
            if not len(smp["stop"]):
                continue
            parts.append({**{k: smp[k] for k in KEYS}, "crops": np.packbits(crops_for(smp), axis=-1)})
        out[split] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        s = out[split]
        obj = s["stop"] == 0
        print(split, len(s["stop"]), "steps,", int(obj.sum()), "buildings; rect share", round(float(s["rect"][obj].mean()), 3),
              "; edge gap < 0.5 m and >= 0:", round(float(((s["gapd"][obj, 0] >= 0) & (s["gapd"][obj, 0] < np.arcsinh(0.25))).mean()), 3),
              f"({time.time() - t0:.0f}s)", flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_bytes(pickle.dumps(out))


def to_dev(d, idx, device):
    b = {k: torch.from_numpy(d[k][idx]).to(device) for k in KEYS}
    b["crops"] = torch.from_numpy(np.unpackbits(d["crops"][idx], axis=-1, count=32).astype(bool)).to(device)
    return b


@torch.no_grad()
def evaluate(model, d, device):
    model.eval()
    tot, n = {}, len(d["stop"])
    for i in range(0, n, 1024):
        for k, v in model.nll(to_dev(d, np.arange(i, min(i + 1024, n)), device)).items():
            tot[k] = tot.get(k, 0.0) + v.sum().item()
    return {"nll_per_step": sum(tot.values()) / n, **{f"{k}_per_step": v / n for k, v in tot.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return 0
    set_seed(a.seed)
    device = pick_device("auto")
    data = pickle.loads(DATA.read_bytes())
    tr, va = data["train"], data["val"]
    model = FrontageNetEdge(256).to(device)
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(a.seed)
    name = f"frontage_edge_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (ep - 1) / EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = rng.permutation(len(tr["stop"]))
        tot = n = 0
        for i in range(0, len(perm), BATCH):
            idx = np.sort(perm[i:i + BATCH])
            loss = sum(v.mean() for v in model.nll(to_dev(tr, idx, device)).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        v = evaluate(model, va, device)
        hist.append({"epoch": ep, "lr": lr, "train": tot / n, **{f"val_{k}": x for k, x in v.items()}})
        print(f"ep {ep:3d} train {tot / n:7.3f} val {v['nll_per_step']:7.3f} " +
              " ".join(f"{k[:-9]} {x:.2f}" for k, x in v.items() if k != "nll_per_step") + f" ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_step"] < best:
            best, best_ep = v["nll_per_step"], ep
            torch.save({"model": model.state_dict(), "epoch": ep}, CKPT / name / "model.pt")
    summ = {"name": name, "params": count_parameters(model), "best_epoch": best_ep, "best_val_nll_per_step": best,
            "train_seconds": round(time.time() - t0, 1)}
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
