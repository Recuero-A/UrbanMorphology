"""Milestone 14 experiment driver (docs/M14_EXPERIMENTAL_PROTOCOL.md): generation + metrics, resumable.

    python scripts/m14_experiments.py --plan configs/m14_plans/<plan>.json [--done marker]

A run = (backbone, sampler preset + overrides, tile set, inference seeds[, terrain variant]). Per inference seed and
tile: the M12 / M13 metrics (unchanged code), the M14 closed-window metrics (graphs/m14_metrics) and sampler
statistics; graphs are stored. Output outputs/m14/runs/<id>.json, rewritten atomically after every seed (resumable).
With "voc": true, decisions with roll-outs record context features and the value of computation (outputs/m14/voc/).

Presets: every M13 preset (A = "standard", "energy", "smc", ...) runs through the UNCHANGED M13 sampler; the M14
presets (graphs/m14_sampler.NET_PRESETS) run through NetSampler. "P" = the M13 procedural baseline; "repair": true
applies the M13 procedural connectivity repair after generation.

Tile sets (development data only; the sealed test splits are never read):
* devA / devB / gen  - as M13 (scripts/m13_experiments.tile_set);
* gen2               - the fresh M14 cities (scripts/m14_build_gen2.py), confirmatory only;
* train_cal          - 48 TRAINING tiles with a DEM drawn with seed 20261117 (lambda calibration only);
* <set>@<variant>    - terrain variants (distribution shift, model input AND rules / metrics use the variant):
                       ``x2`` elevation relief doubled about the tile mean, ``rough`` + 4 m-rms fractal noise.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import m13_experiments as ex  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.experiments.graph_io import graph_to_json
from urbanmorph.graphs.m13_metrics import diversity, tile_metrics_m13
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = PROJECT_ROOT / "outputs" / "m14"
RUNS = OUT / "runs"
CAL_SEED = 20261117


def train_cal_tiles(n=48):
    graphs = pickle.loads((PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    cand = []
    for sid, _, g in graphs["train"]:
        f = ex.TERRAIN / f"{sid}.npz"
        if f.exists():
            cand.append((sid, g))
    rng = np.random.default_rng(CAL_SEED)
    out = []
    for k in rng.permutation(len(cand)):
        sid, g = cand[k]
        d = dict(np.load(ex.TERRAIN / f"{sid}.npz"))
        if d["valid"].all():
            out.append({"sid": sid, "split": "train", "block": sid[:7], "graph": g, "d": d})
        if len(out) == n:
            break
    return out


def variant(d, name, sid):
    d = dict(d)
    z = d["elev"].astype(np.float64)
    if name == "x2":
        d["elev"] = (z.mean() + 2.0 * (z - z.mean())).astype(np.float32)
    elif name == "rough":
        r = np.random.default_rng(int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16))
        f = np.fft.fftfreq(z.shape[0])
        k = np.hypot(*np.meshgrid(f, f))
        spec = (r.normal(size=z.shape) + 1j * r.normal(size=z.shape)) / np.maximum(k, 1 / z.shape[0]) ** 1.5
        n = np.real(np.fft.ifft2(spec))
        d["elev"] = (z + 4.0 * (n - n.mean()) / n.std()).astype(np.float32)
    else:
        raise ValueError(name)
    return d


def tile_set(name):
    base, _, var = name.partition("@")
    if base == "gen2":
        from m14_build_gen2 import gen2_tiles  # noqa: E402
        T = gen2_tiles()
    elif base == "train_cal":
        T = train_cal_tiles()
    else:
        T = ex.tile_set(base)
    if var:
        T = [{**t, "d": variant(t["d"], var, t["sid"])} for t in T]
    return T


def real_rows(tiles, tag):
    """M12 + M13 + M14 metrics of the real graphs (cached)."""
    f = OUT / "real" / f"{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    rows = []
    for t in tiles:
        tf = TerrainField(terrain_of(t["d"]))
        rows.append({"sid": t["sid"], "block": t["block"], **ex.m12_rows(t["graph"], t["d"]),
                     **tile_metrics_m13(t["graph"], tf, np.random.default_rng(0)),
                     **tile_metrics_m14(t["graph"], tf, np.random.default_rng(0))})
    ex.atomic_json(f, rows)
    return rows


def run_id(spec):
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spec['tiles']}__{spec['backbone']}__{spec['preset']}__{h}"


def run(spec, device):
    rid = run_id(spec)
    path = RUNS / f"{rid}.json"
    res = json.loads(path.read_text()) if path.exists() else {"spec": spec, "id": rid, "seeds": {}}
    seeds = [int(s) for s in spec["seeds"]]
    todo = [s for s in seeds if str(s) not in res["seeds"]]
    if not todo:
        print("skip", rid, flush=True)
        return rid
    tiles = tile_set(spec["tiles"])
    real_rows(tiles, spec["tiles"])
    sids, ds = [t["sid"] for t in tiles], [t["d"] for t in tiles]
    tfs = [TerrainField(terrain_of(d)) for d in ds]
    if spec["preset"] == "P":
        from m12_evaluate import TH, density_pool, real_graphs
        from urbanmorph.graphs.procedural_terrain import sample_P
        pool = density_pool(real_graphs())
    else:
        model, names, norm, mcfg = ex.load_backbone(spec["backbone"], device)
        pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / f"{spec.get('potentials', 'TN')}.json")
        crop = ex.model_inputs(mcfg, sids, ds)
        is_net = spec["preset"] in NET_PRESETS
        cfg = net_preset(spec["preset"], **spec.get("overrides", {})) if is_net else preset(spec["preset"], **spec.get("overrides", {}))
    for s in todo:
        t0 = time.time()
        voc = [] if spec.get("voc") else None
        if spec["preset"] == "P":
            rng = np.random.default_rng(s)
            gs = [sample_P(terrain_of(d), rng, float(rng.choice(pool)), TH) for d in ds]
            st, info = [{} for _ in ds], {"seconds": time.time() - t0, "forward_passes": 0}
        elif is_net:
            gs, st, info = sample_net(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop, voc_log=voc)
        else:
            gs, st, info = sample_tc(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop)
        if spec.get("repair"):
            from m12_evaluate import TH
            from urbanmorph.graphs.repair import repair
            gs = [repair(g, terrain_of(d), TH)[0] for g, d in zip(gs, ds)]
        rng = np.random.default_rng(1000 + s)
        rows = []
        for t, g, tf, d, stt in zip(tiles, gs, tfs, ds, st):
            rows.append({"sid": t["sid"], "block": t["block"], **ex.m12_rows(g, d), **tile_metrics_m13(g, tf, rng),
                         **tile_metrics_m14(g, tf, rng),
                         "stats": {k: v for k, v in stt.items() if isinstance(v, (int, float))}})
        res["seeds"][str(s)] = {"rows": rows, "seconds": info["seconds"], "forward_passes": info["forward_passes"],
                                "mean_beta": info.get("mean_beta", 1.0),
                                "graphs": [graph_to_json(g) for g in gs]}
        if voc is not None:
            ex.atomic_json(OUT / "voc" / f"{rid}__s{s}.json", voc)
        res["peak_rss_mb"] = ex._rss_mb()
        ex.atomic_json(path, res)
        print(rid, "seed", s, f"{time.time() - t0:.0f}s", flush=True)
    from urbanmorph.graphs.road_graph import RoadGraph
    div = []
    for k, t in enumerate(tiles):
        gs = []
        for s in seeds:
            G = res["seeds"][str(s)]["graphs"][k]
            nodes = np.array(G["nodes"], float).reshape(-1, 2)
            gs.append(RoadGraph(nodes, np.array(G["edges"], int).reshape(-1, 2), np.zeros(len(nodes), bool)))
        div.append({"sid": t["sid"], **diversity(gs)})
    res["diversity"] = div
    ex.atomic_json(path, res)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--done")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps")
    ap.add_argument("--worker", default="0/1", help="k/n: run only the plan entries with index %% n == k")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else a.device
    torch.set_num_threads(a.threads)
    k, n = map(int, a.worker.split("/"))
    plan = json.loads(Path(a.plan).read_text())
    for j, spec in enumerate(plan["runs"]):
        if j % n == k:
            run(spec, device)
    if a.done:
        Path(a.done).parent.mkdir(parents=True, exist_ok=True)
        Path(a.done).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
