"""Milestone 8, Experiment D: train the parcel mixture-density network on OBSERVED train parcels.

    python scripts/m8_train_d.py --seed 0

Rows: data/processed/m8/parcel_records.pkl (scripts/m8_parcel_records.py), split == "train" for
training and "val" for model selection (best validation epoch). Spain and the US only. 60 epochs,
Adam 1e-3 with cosine decay, batch 512; each epoch re-draws which neighbours are "known".
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m8_parcel_records import load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models.parcel_net import ParcelNet, features
from urbanmorph.models.train import set_seed

CKPT = PROJECT_ROOT / "checkpoints" / "m8"
RUNS = PROJECT_ROOT / "outputs" / "m8" / "runs"
EPOCHS, BATCH, LR, LR_MIN = 60, 512, 1e-3, 2e-5


def targets(r) -> dict:
    """Fixed targets and masks (everything except the neighbour features)."""
    nan0 = lambda v: np.nan_to_num(v.to_numpy(float), nan=1.0).astype(np.float32)
    lot_ok = (~r.interior & ~r.cut & (r.frontage_m > 1) & np.isfinite(r.depth_m)).to_numpy()
    one = lot_ok & (r.n_buildings == 1).to_numpy()
    built = ((r.coverage > 0) & (r.n_buildings >= 1)).to_numpy()
    return {"lot": np.log(np.c_[r.frontage_m.clip(lower=1).to_numpy(), np.nan_to_num(r.depth_m.to_numpy(), nan=1.0).clip(1)]).astype(np.float32),
            "m_lot": lot_ok.astype(np.float32), "built": built.astype(np.float32),
            "coverage": r.coverage.to_numpy(np.float32), "n_cls": (r.n_buildings.clip(1, 3) - 1).to_numpy(np.int64),
            "front": nan0(r.front_setback_m), "m_sb": (one & np.isfinite(r.front_setback_m.to_numpy())).astype(np.float32),
            "side": nan0(r.side_setback_m), "m_side": (one & np.isfinite(r.side_setback_m.to_numpy())).astype(np.float32),
            "rear": nan0(r.rear_setback_m), "m_rear": (one & np.isfinite(r.rear_setback_m.to_numpy())).astype(np.float32),
            "valid": (~r.cut).to_numpy()}


def neighbour_inputs(r, cov_all, rng):
    """Mean coverage of a random subset (fraction u ~ U(0, 1) per parcel) of each parcel's neighbours."""
    m, h = np.zeros(len(r), np.float32), np.zeros(len(r), np.float32)
    for i, nb in enumerate(r.nb.to_numpy()):
        if len(nb):
            keep = nb[rng.random(len(nb)) < rng.random()]
            if len(keep):
                m[i], h[i] = cov_all[keep].mean(), 1.0
    return m, h


def batch_dict(x, t, idx):
    b = {k: torch.from_numpy(v[idx]) for k, v in t.items() if k != "valid"}
    b["x"] = torch.from_numpy(x[idx])
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rec = load_records()
    cov_all = rec.coverage.to_numpy(np.float32)
    data = {}
    for split in ("train", "val"):
        r = rec[rec.split == split]
        data[split] = (r, targets(r))
    model = ParcelNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(a.seed)
    vr, vt = data["val"]
    vm, vh = neighbour_inputs(vr, cov_all, np.random.default_rng(999))
    vx = features(vr.country, vr.cls, vr.log_block_area, vr.interior, vm, vh)
    vidx = np.flatnonzero(vt["valid"])
    name = f"parcel_net_s{a.seed}"
    hist, best, best_ep, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, EPOCHS + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (ep - 1) / EPOCHS))
        for g in opt.param_groups:
            g["lr"] = lr
        tr, tt = data["train"]
        m, h = neighbour_inputs(tr, cov_all, rng)
        x = features(tr.country, tr.cls, tr.log_block_area, tr.interior, m, h)
        idx_all = rng.permutation(np.flatnonzero(tt["valid"]))
        model.train()
        tot = 0.0
        for i in range(0, len(idx_all), BATCH):
            terms = model.nll(batch_dict(x, tt, idx_all[i:i + BATCH]))
            loss = sum(v.mean() for v in terms.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx_all[i:i + BATCH])
        model.eval()
        with torch.no_grad():
            vterms = {k: v.mean().item() for k, v in model.nll(batch_dict(vx, vt, vidx)).items()}
        v = sum(vterms.values())
        hist.append({"epoch": ep, "lr": lr, "train": tot / len(idx_all), "val": v, **{f"val_{k}": x_ for k, x_ in vterms.items()}})
        print(f"ep {ep:3d} train {tot / len(idx_all):7.3f} val {v:7.3f} " + " ".join(f"{k} {x_:.2f}" for k, x_ in vterms.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best, best_ep = v, ep
            (CKPT / name).mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep}, CKPT / name / "model.pt")
    summ = {"name": name, "params": sum(p.numel() for p in model.parameters()), "best_epoch": best_ep,
            "best_val_nll_per_parcel": best, "train_seconds": round(time.time() - t0, 1)}
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
