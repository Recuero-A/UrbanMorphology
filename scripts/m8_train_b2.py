"""Milestone 8, Experiment B2: neighbour-aware fallback placer (no parcels).

    python scripts/m8_train_b2.py --seed 0

Differences from the Milestone 6 placer (P2), and nothing else:
  1. the attribute head sees the 8 nearest already-placed buildings (offset, orientation, size, distance);
  2. it is trained for its actual role: the tile's real FRONTAGE buildings are already placed, and it
     learns to place the interior buildings (random order) and then the green spaces.
Same architecture otherwise, 8 symmetries, cosine LR, 30 epochs, best validation epoch kept.
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
from m6_train_placer import batches, collate, to_dev  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.frontage import assign
from urbanmorph.layout.objects import decode
from urbanmorph.layout.placer_data import tile_samples_b2
from urbanmorph.models.placer import ObjectPlacer
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CKPT = PROJECT_ROOT / "checkpoints" / "m8"
RUNS = PROJECT_ROOT / "outputs" / "m8" / "runs"
MASKS = PROJECT_ROOT / "data" / "processed" / "m8" / "front_masks.pkl"
CFG = {"model": {"width": 32, "hidden": 256}, "extra": {"neighbours": 8},
       "train": {"step_fraction": 0.5, "chunk_tiles": 24, "batch_size": 48, "lr": 5e-4, "lr_min": 2e-5, "epochs": 30}}


def front_masks(tiles):
    if MASKS.exists():
        return pickle.loads(MASKS.read_bytes())
    out = {}
    for t in tiles:
        o = t["objects"]
        b = np.flatnonzero(o["type"] == 0)
        polys = [decode(o["anchor"][i], o["theta"][i], o["length"][i], o["width"][i], o["ratios"][i]) for i in b]
        out[t["sample_id"]] = assign(polys, t["graph"], o["anchor"][b])[1] if len(b) else np.zeros(0, bool)
    MASKS.parent.mkdir(parents=True, exist_ok=True)
    MASKS.write_bytes(pickle.dumps(out))
    return out


@torch.no_grad()
def evaluate(model, tiles, device):
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
                tot += sum(v.sum().item() for v in model.nll(to_dev(mb, device)).values())
    return tot / max(n_obj, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tc = CFG["train"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = load_objects()
    masks = front_masks(data["train"] + data["val"])
    for t in data["train"] + data["val"]:
        t["front_mask"] = masks[t["sample_id"]]
    model = ObjectPlacer(CFG["model"]["width"], CFG["model"]["hidden"], **CFG["extra"]).to(device)
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    rng = np.random.default_rng(a.seed)
    name = f"fallback_nb_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, tc["epochs"] + 1):
        lr = tc["lr_min"] + 0.5 * (tc["lr"] - tc["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / tc["epochs"]))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = rng.permutation(len(data["train"]))
        tot = n = 0
        for c in range(0, len(order), tc["chunk_tiles"]):
            b = collate([tile_samples_b2(data["train"][i], int(rng.integers(8)), rng, tc["step_fraction"])
                         for i in order[c:c + tc["chunk_tiles"]]])
            for mb in batches(b, tc["batch_size"], rng):
                terms = model.nll(to_dev(mb, device))
                loss = sum(v.mean() for v in terms.values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += sum(v.sum().item() for v in terms.values())
                n += int((1 - mb["stop"]).sum())
        v = evaluate(model, data["val"], device)
        hist.append({"epoch": ep, "lr": lr, "train_nll_per_object": tot / max(n, 1), "val_nll_per_object": v})
        print(f"ep {ep:3d} train {tot / max(n, 1):7.3f}/obj  val {v:7.3f}/obj  ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best, best_ep = v, ep
            torch.save({"model": model.state_dict(), "config": CFG, "epoch": ep}, CKPT / name / "model.pt")
    summ = {"name": name, "params": count_parameters(model), "best_epoch": best_ep, "best_val_nll_per_object": best,
            "train_seconds": round(time.time() - t0, 1)}
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
