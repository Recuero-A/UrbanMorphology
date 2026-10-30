"""Milestone 5: train the NTG-inspired street-graph generator.

    python scripts/m5_build_graphs.py        # once
    python scripts/m5_train_graph.py --seed 0

Training data: street graphs of the 565 training tiles. Every epoch, each tile gets a random
symmetry of the square and random component start nodes (both are augmentations of the
construction order). Validation: 19 validation tiles x 4 symmetries with FIXED traces, used only
for early stopping. The test tiles are never read.
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m5_build_graphs import load_graphs
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.sequence import make_trace, pop_samples, start_samples
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m5_graph.toml"
CKPT = PROJECT_ROOT / "checkpoints" / "m5"
RUNS = PROJECT_ROOT / "outputs" / "m5" / "runs"


def collate(samples: list[dict]) -> dict:
    return {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}


def to_dev(b: dict, device, idx=None) -> dict:
    return {k: torch.from_numpy(v if idx is None else v[idx]).to(device) for k, v in b.items()}


def traces_for(graphs, rng, symmetries=None):
    pops, starts = [], []
    for i, g in enumerate(graphs):
        k = int(rng.integers(8)) if symmetries is None else symmetries[i]
        tr = make_trace(g.transform(k), rng)
        pops.append(pop_samples(tr))
        starts.append(start_samples(tr))
    return collate(pops), collate(starts)


@torch.no_grad()
def evaluate(model, val_chunks, device, n_tiles) -> dict:
    model.eval()
    pop_nll = start_nll = 0.0
    n_pops = 0
    for pops, starts in val_chunks:
        for i in range(0, len(pops["n_targets"]), 1024):
            sl = slice(i, i + 1024)
            pop_nll += model.pop_nll(to_dev(pops, device, sl)).sum().item()
        start_nll += model.start_nll(to_dev(starts, device)).sum().item()
        n_pops += len(pops["n_targets"])
    return {"nll_per_tile": (pop_nll + start_nll) / n_tiles, "pop_nll_per_node": pop_nll / n_pops,
            "start_nll_per_tile": start_nll / n_tiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = tomllib.loads(CFG.read_text())
    tc = cfg["train"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = load_graphs()
    train = [g for _, _, g in data["train"]]
    val = [g for _, _, g in data["val"]]
    vrng = np.random.default_rng(999)
    val_graphs = [g for g in val for _ in range(4)]
    val_sym = [k for _ in val for k in (0, 1, 4, 6)]
    val_chunks = [traces_for(val_graphs, vrng, val_sym)]

    model = GraphTurtle(cfg["model"]["hidden"], cfg["model"]["n_mix"]).to(device)
    print(f"params {count_parameters(model):,}  device {device}")
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    rng = np.random.default_rng(a.seed)
    name = f"graph_turtle_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    best, best_ep, history = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, (a.max_epochs or tc["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(train))
        tot, n_tot = 0.0, 0
        for c in range(0, len(order), tc["chunk_tiles"]):
            pops, starts = traces_for([train[i] for i in order[c:c + tc["chunk_tiles"]]], rng)
            perm = rng.permutation(len(pops["n_targets"]))
            for i in range(0, len(perm), tc["batch_size"]):
                loss = model.pop_nll(to_dev(pops, device, perm[i:i + tc["batch_size"]])).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(perm[i:i + tc["batch_size"]])
                n_tot += len(perm[i:i + tc["batch_size"]])
            sl = model.start_nll(to_dev(starts, device)).mean()
            opt.zero_grad()
            sl.backward()
            opt.step()
        v = evaluate(model, val_chunks, device, len(val_graphs))
        history.append({"epoch": ep, "train_pop_nll_per_node": tot / n_tot, **{f"val_{k}": x for k, x in v.items()},
                        "seconds": time.time() - t0})
        print(f"ep {ep:3d} train {tot / n_tot:+.3f}/node  val {v['pop_nll_per_node']:+.3f}/node  "
              f"{v['nll_per_tile']:8.1f}/tile  ({time.time() - t0:.0f}s)", flush=True)
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
