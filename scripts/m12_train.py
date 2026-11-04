"""Milestone 12: train N0 (terrain-blind) and TN (terrain-conditioned) GraphTurtles on the 3D-aware graphs.

    python scripts/m12_train.py --name N0_s0 --seed 0                         # terrain-blind, structure head
    python scripts/m12_train.py --name TN_s0 --seed 0 --terrain elev,slope,water
    python scripts/m12_train.py --name TN_s0 --seed 0 --terrain elev,slope,water --resume latest
    python scripts/m12_train.py --name TN_s0 --pilot 2                        # 2 short epochs on 64 tiles

Recipe = GraphTurtle N (M10): denoising context noise (sigma 2.5 m, border nodes fixed), turtle frame, 8 random
symmetries per tile and epoch, chunks of 64 tiles, batch 512, start batch 128, Adam, cosine LR 1e-3 -> 5e-5 over
max_epochs = 60, 300-step linear warmup (mandatory), gradient clip 5, patience 10, model selection on validation
NLL (val + val_t, fixed symmetries 0 and 4). Every run trains on the same tiles in the same order (seeded), so N0
and TN differ only in their terrain inputs.

Data: data/processed/m12/graphs3d.pkl (streets, bridges, tunnels) and data/processed/m12/terrain/*.npz.
Checkpointing (src/urbanmorph/models/checkpoint.py): every --ckpt-interval s (default 300) after an optimiser
step, at the end of every epoch, and on Ctrl+C / SIGTERM; atomic, previous kept; ``--resume latest``. A mid-epoch
state stores the NumPy generator state at the start of the current chunk and the batch index, so the chunk is
rebuilt exactly and resumed training equals uninterrupted training (tests/test_checkpoint.py).
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

from m5_train_graph import collate, to_dev  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.road_graph import RoadGraph
from urbanmorph.graphs.sequence import heading_of, make_trace, pop_samples, start_samples
from urbanmorph.graphs.terrain_sequence import channel_list, terrain_crops, tile_canvas, tile_layers
from urbanmorph.models.checkpoint import Checkpointer, RunLog, add_resume_args
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle
from urbanmorph.models.train import pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m6_roads.toml"
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
CKPT = PROJECT_ROOT / "checkpoints" / "m12"
RUNS = PROJECT_ROOT / "outputs" / "m12" / "training"
NOISE_M = 2.5
VAL_SYM = (0, 4)


def load_data():
    graphs = pickle.loads(GRAPHS3D.read_bytes())
    terr = {sid: dict(np.load(TERRAIN / f"{sid}.npz")) for sp in graphs for sid, _, _ in graphs[sp]}
    return graphs, terr, json.loads((TERRAIN / "norm.json").read_text())


def encode(g: RoadGraph, label: np.ndarray) -> RoadGraph:
    """Street class and structure label in one edge code (class + 4 * label) so the unchanged trace builder
    carries both; label 3 = unknown."""
    return RoadGraph(g.nodes, g.edges, g.border, g.classes() + 4 * label.astype(int), g.levels(), g.structs())


def jitter(g: RoadGraph, rng) -> RoadGraph:
    p = g.nodes + rng.normal(0, NOISE_M, g.nodes.shape) * (~g.border)[:, None]
    return RoadGraph(np.clip(p, 0, 512.0), g.edges, g.border, g.edge_class, g.edge_level, g.edge_struct)


def headings(tr):
    ho = tr.head_override or {}
    return np.array([ho[u] if u in ho else heading_of(tr.g.nodes, u, tr.parent[u])[0] for u in tr.pop_node])


def make_samples(g, d, rng, k, names, norm):
    gk = encode(g, d["struct_label"]).transform(k)
    gj = jitter(gk, rng)
    tr = make_trace(gj, rng, turtle=True, clean_nodes=gk.nodes)
    pops, starts = pop_samples(tr, turtle=True), start_samples(tr)
    code = pops["target_class"]
    pops["target_class"], st = code % 4, code // 4
    pops["target_struct"] = np.where((st == 3) | (np.arange(code.shape[1])[None] >= pops["n_targets"][:, None]), -1, st)
    if names:
        lay = tile_layers(d, k, names, norm)
        pos = tr.g.nodes[tr.pop_node] if len(tr.pop_node) else np.zeros((0, 2))
        pops["tcrops"] = terrain_crops(lay, names, pos, headings(tr), norm)
        starts["tcanvas"] = np.repeat(tile_canvas(lay, names, norm)[None], len(starts["stop"]), 0)
    return pops, starts


def samples_for(items, terr, rng, names, norm, syms=None):
    pops, starts = [], []
    for i, (sid, g) in enumerate(items):
        k = int(rng.integers(8)) if syms is None else syms[i]
        p, s = make_samples(g, terr[sid], rng, k, names, norm)
        pops.append(p)
        starts.append(s)
    return collate(pops), collate(starts)


@torch.no_grad()
def evaluate(model, val, device, n_tiles):
    model.eval()
    pops, starts = val
    n = len(pops["n_targets"])
    pl = sum(model.pop_nll(to_dev(pops, device, slice(i, i + 1024))).sum().item() for i in range(0, n, 1024))
    m = len(starts["stop"])
    sl = sum(model.start_nll(to_dev(starts, device, slice(i, i + 512))).sum().item() for i in range(0, m, 512))
    return {"nll_per_tile": (pl + sl) / n_tiles, "pop_nll_per_node": pl / max(n, 1), "start_nll_per_tile": sl / n_tiles}


@torch.no_grad()
def decoder_health(model, val, device):
    model.eval()
    b = to_dev(val[0], device, slice(0, min(len(val[0]["n_targets"]), 2048)))
    ctx = model.context(b["crops"], b["scalars"], b["arms"], b["arm_mask"], tcrops=b.get("tcrops"))
    B = ctx.shape[0]
    x = torch.cat([torch.zeros(B, 1, 3 + model.n_classes, device=device), ctx.unsqueeze(1)], -1)
    x[:, 0, 2] = 1
    y, _ = model.gru(x, torch.tanh(model.h0(ctx)).unsqueeze(0))
    hid = model.out[1](model.out[0](y))[:, 0]
    return {"decoder_units_alive": float(((hid > 0).sum(0) > 0).float().mean()),
            "stop_logit_sd": float(model.out[2](hid)[:, 0].std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain", default="", help="comma list of elev,slope,water ('' = terrain-blind N0)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--pilot", type=int, default=0, help="pilot: this many epochs on 64 train / 16 val tiles")
    add_resume_args(ap)
    a = ap.parse_args()
    cfg = tomllib.loads(CFG.read_text())
    tc, mc = cfg["train"], cfg["model"]
    names = [n for n in a.terrain.split(",") if n]
    run_cfg = {"name": a.name, "seed": a.seed, "terrain": names, "warmup": a.warmup, "pilot": a.pilot,
               "train": tc, "model": mc, "noise_m": NOISE_M, "data": str(GRAPHS3D.name)}
    set_seed(a.seed)
    device = pick_device("auto")
    graphs, terr, norm = load_data()
    train = [(sid, g) for sid, _, g in graphs["train"]]
    val = [(sid, g) for sp in ("val", "val_t") for sid, _, g in graphs.get(sp, [])]
    if a.pilot:
        train, val = train[:64], val[:16]
    val_items = [x for x in val for _ in VAL_SYM]
    vsamp = samples_for(val_items, terr, np.random.default_rng(999), names, norm, [k for _ in val for k in VAL_SYM])
    model = TerrainGraphTurtle(mc["hidden"], mc["n_mix"], mc["n_classes"], n_terrain=len(channel_list(names))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    max_ep = a.pilot or a.max_epochs or tc["max_epochs"]
    rng = np.random.default_rng(a.seed)
    ck = Checkpointer(CKPT / a.name, config=run_cfg, interval_s=a.ckpt_interval, rngs={"data": rng})
    log = RunLog(RUNS / a.name / "log.jsonl")
    st = ck.resume(a.resume, model, opt) or {"epoch": 1, "chunk": 0, "batch": 0, "step": 0, "chunk_rng": None,
                                             "order": None, "best": float("inf"), "best_ep": 0, "history": [],
                                             "t_train": 0.0, "tot": 0.0, "n_tot": 0}
    log.write(event="resume" if a.resume and st["step"] else "start", step=st["step"], params=count_parameters(model),
              device=str(device), n_train=len(train), n_val=len(val), terrain=names)
    print(f"{a.name}: params {count_parameters(model):,}  terrain {names}  train {len(train)}  val {len(val)}", flush=True)
    t0 = time.time() - st["t_train"]
    stop = False
    for ep in range(st["epoch"], max_ep + 1):
        lr = tc["lr_min"] + 0.5 * (tc["lr"] - tc["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / max_ep))
        model.train()
        if st["order"] is None:
            st["order"] = rng.permutation(len(train)).tolist()
            st["tot"], st["n_tot"] = 0.0, 0
        order = st["order"]
        for c in range(st["chunk"], len(order), tc["chunk_tiles"]):
            if st["chunk_rng"] is None:
                st["chunk_rng"] = rng.bit_generator.state
            else:
                rng.bit_generator.state = st["chunk_rng"]
            pops, starts = samples_for([train[i] for i in order[c:c + tc["chunk_tiles"]]], terr, rng, names, norm)
            perm = rng.permutation(len(pops["n_targets"]))
            n_start = len(starts["stop"])
            sidx = rng.integers(n_start, size=(math.ceil(len(perm) / tc["batch_size"]), tc["start_batch"]))
            for bi_n, i in enumerate(range(st["batch"], len(perm), tc["batch_size"])):
                j = i // tc["batch_size"]
                for gr in opt.param_groups:
                    gr["lr"] = lr * min(1.0, (st["step"] + 1) / a.warmup) if a.warmup else lr
                bi = perm[i:i + tc["batch_size"]]
                pl = model.pop_nll(to_dev(pops, device, bi)).mean()
                sl = model.start_nll(to_dev(starts, device, sidx[j])).mean()
                loss = pl + sl * n_start / len(perm)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                st["step"] += 1
                st["batch"] = i + tc["batch_size"]
                st["tot"] += pl.item() * len(bi)
                st["n_tot"] += len(bi)
                if ck.due():
                    st["t_train"] = time.time() - t0
                    ck.save(st, model, opt)
                    if ck.stop_requested:
                        log.write(event="interrupted", step=st["step"], epoch=ep)
                        print("interrupted: checkpoint saved; continue with --resume latest", flush=True)
                        return 0
            st.update(batch=0, chunk=c + tc["chunk_tiles"], chunk_rng=None)
        v = evaluate(model, vsamp, device, len(val_items))
        row = {"epoch": ep, "lr": lr, "train_pop_nll_per_node": st["tot"] / max(st["n_tot"], 1),
               **{f"val_{k}": x for k, x in v.items()}, "seconds": time.time() - t0, "step": st["step"]}
        st["history"].append(row)
        log.write(**row)
        print(f"ep {ep:3d} lr {lr:.1e} train {row['train_pop_nll_per_node']:+.3f}/node  val {v['pop_nll_per_node']:+.3f}/node"
              f"  {v['nll_per_tile']:8.1f}/tile  ({time.time() - t0:.0f}s)", flush=True)
        if v["nll_per_tile"] < st["best"]:
            st["best"], st["best_ep"] = v["nll_per_tile"], ep
            torch.save({"model": model.state_dict(), "config": run_cfg, "epoch": ep, "norm": norm}, CKPT / a.name / "best.pt")
        elif ep - st["best_ep"] >= tc["patience"]:
            stop = True
        st.update(epoch=ep + 1, chunk=0, order=None)
        st["t_train"] = time.time() - t0
        ck.save(st, model, opt, tag="epoch")
        if stop:
            break
    model.load_state_dict(torch.load(CKPT / a.name / "best.pt", map_location=device)["model"])
    summary = {"name": a.name, "terrain": names, "seed": a.seed, **decoder_health(model, vsamp, device),
               "params": count_parameters(model), "best_epoch": st["best_ep"], "epochs_run": len(st["history"]),
               "best_val_nll_per_tile": st["best"], "train_seconds": round(time.time() - t0, 1),
               "n_train_tiles": len(train), "n_val_tiles": len(val), "device": str(device)}
    (RUNS / a.name).mkdir(parents=True, exist_ok=True)
    (RUNS / a.name / "history.json").write_text(json.dumps(st["history"], indent=1))
    (RUNS / a.name / "summary.json").write_text(json.dumps(summary, indent=1))
    log.write(event="done", **summary)
    print(json.dumps(summary, indent=1))
    ck.restore_signal_handlers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
