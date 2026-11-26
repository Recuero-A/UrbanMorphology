"""Milestone 6, Stage A: train the street-graph generator v2 (turtle frame + street classes).

    python scripts/m6_build_roads.py          # once
    python scripts/m6_train_roads.py --seed 0

Same data, splits, augmentation and validation protocol as Milestone 5 (scripts/m5_train_graph.py);
differences: turtle-frame inputs/steps, a street-class head, a cosine learning-rate schedule, and the
start head updated at every optimiser step. Test tiles are never read.
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

from m5_train_graph import collate, to_dev  # noqa: E402
from m6_build_roads import load_graphs  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.sequence import make_trace, pop_samples, start_samples
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m6_roads.toml"
CKPT = PROJECT_ROOT / "checkpoints" / "m6"
RUNS = PROJECT_ROOT / "outputs" / "m6" / "runs"


def traces_for(graphs, rng, turtle, symmetries=None):
    pops, starts = [], []
    for i, g in enumerate(graphs):
        k = int(rng.integers(8)) if symmetries is None else symmetries[i]
        tr = make_trace(g.transform(k), rng, turtle=turtle)
        pops.append(pop_samples(tr, turtle=turtle))
        starts.append(start_samples(tr))
    return collate(pops), collate(starts)


@torch.no_grad()
def evaluate(model, val, device, n_tiles) -> dict:
    model.eval()
    pops, starts = val
    pop_nll = sum(model.pop_nll(to_dev(pops, device, slice(i, i + 1024))).sum().item()
                  for i in range(0, len(pops["n_targets"]), 1024))
    start_nll = model.start_nll(to_dev(starts, device)).sum().item()
    return {"nll_per_tile": (pop_nll + start_nll) / n_tiles, "pop_nll_per_node": pop_nll / len(pops["n_targets"]),
            "start_nll_per_tile": start_nll / n_tiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = tomllib.loads(CFG.read_text())
    tc, mc = cfg["train"], cfg["model"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = load_graphs()
    train = [g for _, _, g in data["train"]]
    val_graphs = [g for _, _, g in data["val"] for _ in range(4)]
    val_sym = [k for _ in data["val"] for k in (0, 1, 4, 6)]
    val = traces_for(val_graphs, np.random.default_rng(999), mc["turtle"], val_sym)

    model = GraphTurtle(mc["hidden"], mc["n_mix"], turtle=mc["turtle"], n_classes=mc["n_classes"]).to(device)
    print(f"params {count_parameters(model):,}  device {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    max_ep = a.max_epochs or tc["max_epochs"]
    rng = np.random.default_rng(a.seed)
    name = f"roads_v2_s{a.seed}"
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
        tot, n_tot = 0.0, 0
        for c in range(0, len(order), tc["chunk_tiles"]):
            pops, starts = traces_for([train[i] for i in order[c:c + tc["chunk_tiles"]]], rng, mc["turtle"])
            perm = rng.permutation(len(pops["n_targets"]))
            n_start = len(starts["stop"])
            for i in range(0, len(perm), tc["batch_size"]):
                idx = perm[i:i + tc["batch_size"]]
                pl = model.pop_nll(to_dev(pops, device, idx)).mean()
                sl = model.start_nll(to_dev(starts, device, rng.integers(n_start, size=tc["start_batch"]))).mean()
                loss = pl + sl * n_start / len(perm)          # per-node weighting of the start head's terms
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += pl.item() * len(idx)
                n_tot += len(idx)
        v = evaluate(model, val, device, len(val_graphs))
        history.append({"epoch": ep, "lr": lr, "train_pop_nll_per_node": tot / n_tot,
                        **{f"val_{k}": x for k, x in v.items()}, "seconds": time.time() - t0})
        print(f"ep {ep:3d} lr {lr:.1e} train {tot / n_tot:+.3f}/node  val {v['pop_nll_per_node']:+.3f}/node  "
              f"start {v['start_nll_per_tile']:6.2f}/tile  {v['nll_per_tile']:8.1f}/tile  ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_tile"] < best:
            best, best_ep = v["nll_per_tile"], ep
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, CKPT / name / "model.pt")
        elif ep - best_ep >= tc["patience"]:
            break
    summary = {"name": name, "params": count_parameters(model), "best_epoch": best_ep, "epochs_run": ep,
               "best_val_nll_per_tile": best, "train_seconds": round(time.time() - t0, 1),
               "n_train_tiles": len(train), "n_val_traces": len(val_graphs), "device": str(device)}
    (RUNS / name / "history.json").write_text(json.dumps(history, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
