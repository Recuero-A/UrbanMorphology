"""Milestone 20, D1 freeze rule, condition 1 input: dev-H NLL per slope class of every retrained backbone.

    python scripts/m20_tng_devnll.py                      # every finished run under outputs/m20/tng/training
    python scripts/m20_tng_devnll.py --names TNgH_s0,TNg32_s0

Data: the 512 m windows of the 21 dev-H reference units (configs/m20_pool/dev_units.json; built by
scripts/m20_tng_data.py towns --set dev with the same FABDEM-32 m input recipe and the TNg-H norm). Samples are built
exactly as m12_train's validation samples (fixed symmetries 0 and 4, generator seeded 999), so every backbone is scored
on the same decisions. Each decision (pop) is assigned the D3 slope class of the FABDEM 32 m cell under its current
node. Output per class: decisions and the mean (unweighted) pop NLL per decision, overall and per unit, plus the start
NLL per window -> outputs/m20/tng/devnll/<name>.json. The comparison between backbones is left to the freeze rule.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import m12_train as T  # noqa: E402
from m5_train_graph import collate, to_dev  # noqa: E402
from m20_tng_data import CLASS_NAMES, node_class  # noqa: E402
from urbanmorph.graphs.terrain_sequence import channel_list, transform_raster  # noqa: E402
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle  # noqa: E402

DATA = ROOT / "data" / "processed" / "m20" / "tng"
RUNS = ROOT / "outputs" / "m20" / "tng" / "training"
CKPT = ROOT / "checkpoints" / "m20"
OUT = ROOT / "outputs" / "m20" / "tng" / "devnll"
NAMES = ["elev", "slope", "water"]


def dev_samples(norm):
    items = pickle.loads((DATA / "towns_dev.pkl").read_bytes())
    rng = np.random.default_rng(999)
    pops, starts, meta = [], [], []
    for wid, town, g in items:
        d = dict(np.load(DATA / "dev" / f"{wid}.npz"))
        for k in T.VAL_SYM:
            gk = T.encode(g, d["struct_label"]).transform(k)
            gj = T.jitter(gk, rng)
            tr = T.make_trace(gj, rng, turtle=True, clean_nodes=gk.nodes)
            p, s = T.pop_samples(tr, turtle=True), T.start_samples(tr)
            code = p["target_class"]
            p["target_class"], st = code % 4, code // 4
            p["target_struct"] = np.where((st == 3) | (np.arange(code.shape[1])[None] >= p["n_targets"][:, None]), -1, st)
            lay = T.tile_layers(d, k, NAMES, norm)
            pos = tr.g.nodes[tr.pop_node] if len(tr.pop_node) else np.zeros((0, 2))
            p["tcrops"] = T.terrain_crops(lay, NAMES, pos, T.headings(tr), norm)
            s["tcanvas"] = np.repeat(T.tile_canvas(lay, NAMES, norm)[None], len(s["stop"]), 0)
            cls = node_class(transform_raster(d["cls32"], k), pos).astype(int)
            pops.append(p)
            starts.append(s)
            meta.append((wid, town, cls, len(s["stop"])))
    return collate(pops), collate(starts), meta


@torch.no_grad()
def score(name, pops, starts, meta, device):
    ck = torch.load(CKPT / name / "best.pt", map_location=device)
    mc = ck["config"]["model"]
    model = TerrainGraphTurtle(mc["hidden"], mc["n_mix"], mc["n_classes"], n_terrain=len(channel_list(NAMES))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n = len(pops["n_targets"])
    pl = torch.cat([model.pop_nll(to_dev(pops, device, slice(i, i + 1024))).cpu() for i in range(0, n, 1024)]).numpy()
    m = len(starts["stop"])
    sl = torch.cat([model.start_nll(to_dev(starts, device, slice(i, i + 512))).cpu() for i in range(0, m, 512)]).numpy()
    cls = np.concatenate([c for _, _, c, _ in meta])
    town = np.concatenate([[t] * len(c) for _, t, c, _ in meta])
    assert len(cls) == n
    res = {"name": name, "epoch": ck["epoch"], "n_decisions": int(n), "n_windows": len(meta) // len(T.VAL_SYM),
           "pop_nll_per_decision": float(pl.mean()), "start_nll_per_window_sym": float(sl.sum() / len(meta)),
           "by_class": {}, "by_unit_class": {}}
    for c, cn in enumerate(CLASS_NAMES):
        s = cls == c
        res["by_class"][cn] = {"n": int(s.sum()), "pop_nll": float(pl[s].mean()) if s.any() else None}
    for t in sorted(set(town)):
        res["by_unit_class"][t] = {cn: {"n": int(((town == t) & (cls == c)).sum()),
                                        "pop_nll": float(pl[(town == t) & (cls == c)].mean()) if ((town == t) & (cls == c)).any() else None}
                                   for c, cn in enumerate(CLASS_NAMES)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    names = [x for x in a.names.split(",") if x] or sorted(p.name for p in RUNS.iterdir() if (p / "summary.json").exists())
    norm = json.loads((DATA / "norm.json").read_text())
    pops, starts, meta = dev_samples(norm)
    device = T.pick_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)
    for nm in names:
        r = score(nm, pops, starts, meta, device)
        (OUT / f"{nm}.json").write_text(json.dumps(r, indent=1))
        print(nm, "written", OUT / f"{nm}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
