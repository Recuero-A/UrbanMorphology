"""Milestone 10: train BlockGen (docs/M10_BLOCK_GENERATION.md §4).

    python scripts/m10_block_data.py                 # once
    python scripts/m10_train_blockgen.py --seed 0    # -> checkpoints/m10/blockgen_s0/ (git-ignored), outputs/m10/runs/

Training windows only (validation windows give the validation loss; test tiles are never read).
Batches: windows sampled with probability ~ sqrt(block pixels) (large blocks are rare but carry most of the
tile's buildings); random flips of the canonical frame; known region: none (50 %), a half-plane through a
random point of the block (25 %) or a random rectangle (25 %). Loss weight 1 inside the block (dilated by
2 px), 0.05 outside. AdamW 2e-4, EMA 0.999 of the weights (the EMA is saved).
"""
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models.block_diffusion import BlockDiffusion
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

DATA = PROJECT_ROOT / "data" / "processed" / "m10" / "block_windows.pkl"
CKPT = PROJECT_ROOT / "checkpoints" / "m10"
RUNS = PROJECT_ROOT / "outputs" / "m10" / "runs"
H = 128


def load_windows():
    d = pickle.loads(DATA.read_bytes())
    unpack = lambda a: np.unpackbits(a, axis=1).reshape(-1, H, H)
    return d["rows"], unpack(d["block"]), unpack(d["building"]), d["cities"]


def random_known(block: np.ndarray, rng) -> np.ndarray:
    u = rng.random()
    if u < 0.5:
        return np.zeros_like(block, dtype=np.float32)
    if u < 0.75:
        ys, xs = np.nonzero(block)
        if not len(ys):
            return np.zeros_like(block, dtype=np.float32)
        k = rng.integers(len(ys))
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[:H, :H]
        return (((xx - xs[k]) * np.cos(a) + (yy - ys[k]) * np.sin(a)) > 0).astype(np.float32)
    h, w = rng.integers(16, 96, 2)
    y, x = rng.integers(0, H - h), rng.integers(0, H - w)
    m = np.zeros((H, H), np.float32)
    m[y:y + h, x:x + w] = 1
    return m


def make_batch(ids, blk, bld, city, rng, device):
    B, X, K = [], [], []
    for i in ids:
        b, x = blk[i].astype(np.float32), bld[i].astype(np.float32)
        if rng.random() < 0.5:
            b, x = b[:, ::-1], x[:, ::-1]
        if rng.random() < 0.5:
            b, x = b[::-1], x[::-1]
        B.append(b.copy())
        X.append(x.copy())
        K.append(random_known(b, rng))
    B, X, K = (torch.from_numpy(np.stack(a))[:, None].to(device) for a in (B, X, K))
    cond = torch.cat([B, X * K, K], 1)
    w = F.max_pool2d(B, 5, 1, 2)
    w = (0.05 + 0.95 * w) * (1 - 0.9 * K)                   # the known region is pasted back at sampling
    return X * 2 - 1, cond, torch.as_tensor(city[ids], device=device), w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    set_seed(a.seed)
    device = pick_device("auto")
    rows, blk, bld, cities = load_windows()
    split = np.array([r["split"] for r in rows])
    city = np.array([r["city"] for r in rows])
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    p = np.sqrt(blk[tr].reshape(len(tr), -1).sum(1).astype(float) + 1)
    p /= p.sum()
    model = BlockDiffusion(len(cities)).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    name = f"blockgen_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    print(f"{name}: params {count_parameters(model):,}  train windows {len(tr)}  device {device}", flush=True)
    rng = np.random.default_rng(a.seed)
    vrng = np.random.default_rng(999)
    vids = vrng.choice(va, size=min(256, len(va)), replace=False)
    vb = [make_batch(vids[i:i + 64], blk, bld, city, vrng, device) for i in range(0, len(vids), 64)]
    history, t0, run = [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        lr = 2e-4 * min(1.0, step / 500)
        for g in opt.param_groups:
            g["lr"] = lr
        x0, cond, c, w = make_batch(tr[rng.choice(len(tr), a.batch, p=p)], blk, bld, city, rng, device)
        loss = model.loss(x0, cond, c, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 1000 == 0 or step == a.steps:
            torch.manual_seed(0)
            with torch.no_grad():
                vl = float(np.mean([ema.loss(*b).item() for b in vb for _ in range(4)]))
            history.append({"step": step, "train_loss_ema": run, "val_loss": vl, "seconds": time.time() - t0})
            print(f"step {step:6d} train {run:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
            torch.save({"model": ema.state_dict(), "cities": cities, "step": step}, CKPT / name / "model.pt")
    summary = {"name": name, "params": count_parameters(model), "steps": a.steps, "batch": a.batch,
               "train_seconds": round(time.time() - t0, 1), "final_val_loss": history[-1]["val_loss"], "device": str(device)}
    (RUNS / name / "history.json").write_text(json.dumps(history, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
