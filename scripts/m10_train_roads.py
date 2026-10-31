"""Milestone 10, Phase 2-3: train the street-generator variants of docs/M10_EVALUATION_PROTOCOL.md §3.

    python scripts/m10_train_roads.py --variant city   --seed 0     # B : one-hot city conditioning
    python scripts/m10_train_roads.py --variant noise  --seed 0     # N : denoising context noise
    python scripts/m10_train_roads.py --variant stage1 --seed 0     # C1: major + mid network only
    python scripts/m10_train_roads.py --variant stage2 --seed 0     # C2: local + minor given major + mid
    python scripts/m10_train_roads.py --variant stage2 --noise --seed 0   # CN stage 2 (likewise stage1)

Everything else is the Milestone 6 recipe (configs/m6_roads.toml, scripts/m6_train_roads.py): same data,
augmentation (random square symmetry), validation traces, schedule and early stopping. Test tiles are
never read. Checkpoints -> checkpoints/m10/<name>/ (git-ignored); history -> outputs/m10/runs/<name>/.
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
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.graphs.road_graph import RoadGraph, subgraph
from urbanmorph.graphs.sequence import make_trace, pop_samples, start_samples
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m6_roads.toml"
CKPT = PROJECT_ROOT / "checkpoints" / "m10"
RUNS = PROJECT_ROOT / "outputs" / "m10" / "runs"
NOISE_M = 2.5          # sigma of the context jitter (pre-registered, docs/M10_EVALUATION_PROTOCOL.md §3)
MAJOR = 1              # classes <= MAJOR (major, mid) form the stage-1 network


def city_table():
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & (idx.split == "train")]
    cities = sorted(idx.city.unique())
    return cities, dict(zip(idx.sample_id, idx.city))


def jitter(g: RoadGraph, rng) -> RoadGraph:
    p = g.nodes + rng.normal(0, NOISE_M, g.nodes.shape) * (~g.border)[:, None]
    return RoadGraph(np.clip(p, 0, 512.0), g.edges, g.border, g.edge_class)


def make_samples(g: RoadGraph, rng, variant: str, noise: bool, cond=None):
    if variant == "stage1":
        g = subgraph(g, g.edge_class <= MAJOR)
    given = (g.edge_class <= MAJOR) if variant == "stage2" else None
    gc = g
    if noise:
        g = jitter(g, rng)
    tr = make_trace(g, rng, turtle=True, given=given, clean_nodes=gc.nodes if noise else None)
    pops, starts = pop_samples(tr, turtle=True), start_samples(tr)
    if cond is not None:
        pops["cond"] = np.repeat(cond[None], len(pops["n_targets"]), 0)
        starts["cond"] = np.repeat(cond[None], len(starts["stop"]), 0)
    return pops, starts


def traces_for(graphs, conds, rng, variant, noise, symmetries=None):
    pops, starts = [], []
    for i, g in enumerate(graphs):
        k = int(rng.integers(8)) if symmetries is None else symmetries[i]
        p, s = make_samples(g.transform(k), rng, variant, noise, None if conds is None else conds[i])
        pops.append(p)
        starts.append(s)
    return collate(pops), collate(starts)


@torch.no_grad()
def evaluate(model, val, device, n_tiles) -> dict:
    model.eval()
    pops, starts = val
    n = len(pops["n_targets"])
    pop_nll = sum(model.pop_nll(to_dev(pops, device, slice(i, i + 1024))).sum().item() for i in range(0, n, 1024))
    start_nll = model.start_nll(to_dev(starts, device)).sum().item()
    return {"nll_per_tile": (pop_nll + start_nll) / n_tiles, "pop_nll_per_node": pop_nll / max(n, 1),
            "start_nll_per_tile": start_nll / n_tiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["city", "noise", "stage1", "stage2"], required=True)
    ap.add_argument("--noise", action="store_true", help="add context noise to stage1 / stage2 (CN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()
    noise = a.noise or a.variant == "noise"
    cfg = tomllib.loads(CFG.read_text())
    tc, mc = cfg["train"], cfg["model"]
    set_seed(a.seed)
    device = pick_device("auto")
    data = load_graphs()
    cities, city_of = city_table()
    onehot = lambda sid: np.eye(len(cities), dtype=np.float32)[cities.index(city_of_all[sid])]
    idx = load_index_with_expansion()
    city_of_all = dict(zip(idx.sample_id, idx.city))
    use_cond = a.variant == "city"
    train = [g for _, _, g in data["train"]]
    train_c = [onehot(sid) for sid, _, _ in data["train"]] if use_cond else None
    val_graphs = [g for _, _, g in data["val"] for _ in range(4)]
    val_c = [onehot(sid) for sid, _, _ in data["val"] for _ in range(4)] if use_cond else None
    val_sym = [k for _ in data["val"] for k in (0, 1, 4, 6)]
    val = traces_for(val_graphs, val_c, np.random.default_rng(999), a.variant, noise, val_sym)

    model = GraphTurtle(mc["hidden"], mc["n_mix"], turtle=True, n_classes=mc["n_classes"],
                        n_cond=len(cities) if use_cond else 0).to(device)
    name = {"city": "B_city", "noise": "N_noise", "stage1": "C1_major", "stage2": "C2_local"}[a.variant]
    if noise and a.variant in ("stage1", "stage2"):
        name += "_noise"
    name += f"_s{a.seed}"
    print(f"{name}: params {count_parameters(model):,}  device {device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    max_ep = a.max_epochs or tc["max_epochs"]
    rng = np.random.default_rng(a.seed)
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
            ids = order[c:c + tc["chunk_tiles"]]
            pops, starts = traces_for([train[i] for i in ids], None if train_c is None else [train_c[i] for i in ids],
                                      rng, a.variant, noise)
            perm = rng.permutation(len(pops["n_targets"]))
            n_start = len(starts["stop"])
            for i in range(0, len(perm), tc["batch_size"]):
                bi = perm[i:i + tc["batch_size"]]
                pl = model.pop_nll(to_dev(pops, device, bi)).mean()
                sl = model.start_nll(to_dev(starts, device, rng.integers(n_start, size=tc["start_batch"]))).mean()
                loss = pl + sl * n_start / len(perm)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += pl.item() * len(bi)
                n_tot += len(bi)
        v = evaluate(model, val, device, len(val_graphs))
        history.append({"epoch": ep, "lr": lr, "train_pop_nll_per_node": tot / max(n_tot, 1),
                        **{f"val_{k}": x for k, x in v.items()}, "seconds": time.time() - t0})
        print(f"ep {ep:3d} lr {lr:.1e} train {tot / max(n_tot, 1):+.3f}/node  val {v['pop_nll_per_node']:+.3f}/node  "
              f"start {v['start_nll_per_tile']:6.2f}/tile  {v['nll_per_tile']:8.1f}/tile  ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_tile"] < best:
            best, best_ep = v["nll_per_tile"], ep
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep, "variant": a.variant, "noise": noise,
                        "cities": cities if use_cond else None}, CKPT / name / "model.pt")
        elif ep - best_ep >= tc["patience"]:
            break
    summary = {"name": name, "variant": a.variant, "noise": noise, "params": count_parameters(model), "best_epoch": best_ep,
               "epochs_run": ep, "best_val_nll_per_tile": best, "train_seconds": round(time.time() - t0, 1),
               "n_train_tiles": len(train), "n_val_traces": len(val_graphs), "device": str(device)}
    (RUNS / name / "history.json").write_text(json.dumps(history, indent=1))
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
