"""Milestone 9: train the four BlockNets (type, side, zone, shape) on OBSERVED train parcels.

    python scripts/m9_train.py --seed 0     # -> checkpoints/m9/block_nets_s{seed}/model.pt (git-ignored)

Each net: Adam 1e-3 (cosine to 2e-5), batch 512, 40 epochs, best validation epoch kept (split == "val").
Zone: the neighbour feature is the mean green label of a random subset (fraction u ~ U(0, 1), re-drawn
every epoch) of each parcel's neighbours, as in sequential generation.
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models.block_net import BlockNets, N_MAX, block_x, shape_x, side_x, zone_x
from urbanmorph.models.train import set_seed

REC = PROJECT_ROOT / "data" / "processed" / "m9" / "records.pkl"
CKPT = PROJECT_ROOT / "checkpoints" / "m9"
RUNS = PROJECT_ROOT / "outputs" / "m9" / "runs"
EPOCHS, BATCH, LR, LR_MIN = 40, 512, 1e-3, 2e-5


def tensors(net, d, rng=None, green_all=None):
    if net == "type":
        return {"x": block_x(d.country, d.log_area, d.rect, d.compact, d.n_edges, d.cls), "y": d.type.to_numpy(np.int64)}
    if net == "side":
        bx = block_x(d.country, d.log_area, d.rect, d.compact, d.n_edges, d.cls)
        r = d.rear_ratio.to_numpy(np.float32)
        return {"x": side_x(bx, d.type, d.log_len, d.side_cls), "N": d.N.clip(upper=N_MAX).to_numpy(np.int64),
                "ratio": np.nan_to_num(r, nan=0.5), "m_ratio": np.isfinite(r).astype(np.float32)}
    if net == "zone":
        m, h = np.zeros(len(d), np.float32), np.zeros(len(d), np.float32)
        for i, nb in enumerate(d.nb.to_numpy()):
            if len(nb):
                keep = nb[rng.random(len(nb)) < rng.random()]
                if len(keep):
                    m[i], h[i] = green_all[keep].mean(), 1.0
        return {"x": zone_x(d.country, d.log_area, d.interior, d.btype, m, h), "y": d.green.to_numpy(np.float32)}
    return {"x": shape_x(d.country, d.log_w, d.log_d, d.coverage, d.interior, d.cls), "y": d.template.to_numpy(np.int64)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rec = pickle.loads(REC.read_bytes())
    green_all = rec["zoning"].green.to_numpy(np.float32)
    model = BlockNets()
    rng = np.random.default_rng(a.seed)
    summary, hist = {}, {}
    name = f"block_nets_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    best_state = {}
    for net, table in (("type", "blocks"), ("side", "sides"), ("zone", "zoning"), ("shape", "shapes")):
        d = rec[table]
        tr, va = d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)
        vt = {k: torch.from_numpy(v) for k, v in tensors(net, va, np.random.default_rng(999), green_all).items()}
        params = list(getattr(model, f"{net}_net").parameters())
        opt = torch.optim.Adam(params, lr=LR)
        best, best_ep, t0, h = float("inf"), 0, time.time(), []
        for ep in range(1, EPOCHS + 1):
            for g_ in opt.param_groups:
                g_["lr"] = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (ep - 1) / EPOCHS))
            tt = {k: torch.from_numpy(v) for k, v in tensors(net, tr, rng, green_all).items()}
            perm = torch.from_numpy(rng.permutation(len(tr)))
            tot = 0.0
            for i in range(0, len(tr), BATCH):
                b = {k: v[perm[i:i + BATCH]] for k, v in tt.items()}
                loss = model.nll(net, b).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                tot += loss.item() * len(b["x"])
            with torch.no_grad():
                v = model.nll(net, vt).mean().item()
            h.append({"epoch": ep, "train": tot / len(tr), "val": v})
            if v < best:
                best, best_ep = v, ep
                best_state[net] = {k: x.clone() for k, x in getattr(model, f"{net}_net").state_dict().items()}
        getattr(model, f"{net}_net").load_state_dict(best_state[net])
        summary[net] = {"train_rows": len(tr), "val_rows": len(va), "best_epoch": best_ep, "best_val_nll": best,
                        "seconds": round(time.time() - t0, 1)}
        hist[net] = h
        print(net, summary[net], flush=True)
    torch.save({"model": model.state_dict()}, CKPT / name / "model.pt")
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
