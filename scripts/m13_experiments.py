"""Milestone 13 experiment driver: generation + per-sample metrics, resumable (docs/M13_EVALUATION_PROTOCOL.md).

    python scripts/m13_experiments.py --plan configs/m13_plans/select.json      # one plan = a list of runs
    python scripts/m13_experiments.py --splits                                  # write the DEV-A / DEV-B split
    python scripts/m13_experiments.py --feature-norm                            # real training-tile feature norms

A run = (backbone checkpoint, sampler preset + overrides, tile set, inference seeds). For every inference seed the
sampler generates one network per tile (all tiles batched); per sample we store the M12 metrics (tile_metrics,
graph_metrics: S_T and S10 inputs, unchanged) and the M13 metrics (graphs/m13_metrics), plus compute statistics.
Output: outputs/m13/runs/<run_id>.json, rewritten atomically after every inference seed, so an interrupted run resumes
at the next seed. A run whose file lists all seeds is skipped.

Tile sets (development data only; the held-out test splits are never read):
* ``devA`` / ``devB``: the 52 development tiles with a DEM (val with DEM + val_t), split by GEOGRAPHIC BLOCK -
  one block per val city, 3 km blocks in San Diego - blocks shuffled with seed 20261106 and dealt alternately;
  devA = method selection and calibration, devB = confirmatory comparison;
* ``gen``: the generalisation cities (built by scripts/m13_build_gen_cities.py), confirmatory only.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m12_index import load_index_m12
from urbanmorph.graphs.m13_metrics import JOINT_KEYS, diversity, tile_metrics_m13
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import SamplerConfig, preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.graphs.terrain_sequence import channel_list, lowpass
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle

OUT = PROJECT_ROOT / "outputs" / "m13"
RUNS = OUT / "runs"
CK = PROJECT_ROOT / "checkpoints" / "m12"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
SE = PROJECT_ROOT / "data" / "processed" / "m13" / "terrain_se"
SPLITS = OUT / "splits.json"
NORM = PROJECT_ROOT / "configs" / "m13_feature_norm.json"
SPLIT_SEED = 20261106


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    os.replace(tmp, path)


def load_backbone(run, device):
    ck = torch.load(CK / run / "best.pt", map_location=device, weights_only=False)
    c = ck["config"]
    m = TerrainGraphTurtle(c["model"]["hidden"], c["model"]["n_mix"], c["model"]["n_classes"],
                           n_terrain=len(channel_list(c["terrain"]))).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, c["terrain"], ck["norm"], c


def model_inputs(cfg, sids, ds):
    """What the MODEL sees: low-passed or street-erased terrain; rules and metrics keep ``ds``."""
    if cfg.get("dem_variant"):
        f = cfg["dem_variant"].split("_")[1]
        return [{**d, "elev": np.load(SE / f / f"{s}.npz")["elev"]} for s, d in zip(sids, ds)]
    if cfg.get("lowpass_m"):
        return [lowpass(d, float(cfg["lowpass_m"])) for d in ds]
    return None


def dev_tiles():
    graphs = pickle.loads((PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    idx = load_index_m12().set_index("sample_id")
    out = []
    for sp in ("val", "val_t"):
        for sid, _, g in graphs[sp]:
            d = dict(np.load(TERRAIN / f"{sid}.npz"))
            if d["valid"].all():
                r = idx.loc[sid]
                block = r.city_code if sp == "val" else f"SAN_{int(r.xmin // 3000)}_{int(r.ymin // 3000)}"
                out.append({"sid": sid, "split": sp, "block": block, "graph": g, "d": d})
    return out


def make_splits():
    tiles = dev_tiles()
    blocks = sorted({t["block"] for t in tiles})
    rng = np.random.default_rng(SPLIT_SEED)
    order = [blocks[i] for i in rng.permutation(len(blocks))]
    assign = {b: ("devA" if k % 2 == 0 else "devB") for k, b in enumerate(order)}
    sets = {"devA": [t["sid"] for t in tiles if assign[t["block"]] == "devA"],
            "devB": [t["sid"] for t in tiles if assign[t["block"]] == "devB"]}
    atomic_json(SPLITS, {"seed": SPLIT_SEED, "blocks": assign, **sets,
                         "rule": "blocks = val city / 3 km San Diego cells; shuffled (seed), dealt alternately"})
    print({k: len(v) for k, v in sets.items()}, len(blocks), "blocks")


def tile_set(name):
    tiles = {t["sid"]: t for t in dev_tiles()}
    if name in ("devA", "devB"):
        return [tiles[s] for s in json.loads(SPLITS.read_text())[name]]
    if name == "dev":
        return list(tiles.values())
    if name == "gen":
        from m13_build_gen_cities import gen_tiles  # noqa: E402
        return gen_tiles()
    raise ValueError(name)


def real_rows(tiles, tag):
    """M13 metrics of the REAL graphs of a tile set (cached)."""
    f = OUT / "real" / f"{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    rng = np.random.default_rng(0)
    rows = []
    for t in tiles:
        m = tile_metrics_m13(t["graph"], TerrainField(terrain_of(t["d"])), rng)
        rows.append({"sid": t["sid"], "block": t["block"], **m12_rows(t["graph"], t["d"]), **m})
    atomic_json(f, rows)
    return rows


def m12_rows(g, d):
    from m10_street_diagnosis import KEYS  # noqa: E402
    from urbanmorph.graphs.road_graph import graph_metrics
    from urbanmorph.graphs.terrain_metrics import tile_metrics
    m = tile_metrics(g, terrain_of(d))
    m.update({f"s10_{k}": graph_metrics(g)[k] for k in KEYS})
    return {f"m12_{k}" if not k.startswith("s10_") else k: v for k, v in m.items()}


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
    if spec["preset"] == "P":
        from m12_evaluate import TH, density_pool, real_graphs
        from urbanmorph.graphs.procedural_terrain import sample_P
        pool = density_pool(real_graphs())
    else:
        model, names, norm, mcfg = load_backbone(spec["backbone"], device)
        pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / f"{spec['potentials']}.json")
        cfg = preset(spec["preset"], **spec.get("overrides", {}))
        crop = model_inputs(mcfg, sids, ds)
    tfs = [TerrainField(terrain_of(d)) for d in ds]
    for s in todo:
        t0 = time.time()
        if spec["preset"] == "P":
            rng = np.random.default_rng(s)
            gs = [sample_P(terrain_of(d), rng, float(rng.choice(pool)), TH) for d in ds]
            st, info = [{} for _ in ds], {"seconds": time.time() - t0, "forward_passes": 0}
        else:
            gs, st, info = sample_tc(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop)
        rep = [None] * len(gs)
        if spec.get("repair"):                                     # procedural repair: a labelled baseline
            from m12_evaluate import TH
            from urbanmorph.graphs.repair import repair
            out = [repair(g, terrain_of(d), TH) for g, d in zip(gs, ds)]
            gs, rep = [o[0] for o in out], [o[1] for o in out]
        rng = np.random.default_rng(1000 + s)
        rows = []
        for t, g, tf, d, stt, rp in zip(tiles, gs, tfs, ds, st, rep):
            m = tile_metrics_m13(g, tf, rng)
            rows.append({"sid": t["sid"], "block": t["block"], **m12_rows(g, d), **m,
                         "stats": {k: v for k, v in stt.items() if isinstance(v, (int, float))},
                         **({"repair": rp} if rp else {})})
        res["seeds"][str(s)] = {"rows": rows, "seconds": info["seconds"], "forward_passes": info["forward_passes"],
                                "graphs": [{"nodes": g.nodes.round(2).tolist(), "edges": g.edges.tolist(),
                                            "cls": np.asarray(g.classes()).tolist(), "st": np.asarray(g.structs()).tolist()}
                                           for g in gs]}
        res["peak_rss_mb"] = _rss_mb()
        atomic_json(path, res)
        print(rid, "seed", s, f"{time.time() - t0:.0f}s", flush=True)
    # diversity across the inference seeds of each tile
    from urbanmorph.graphs.road_graph import RoadGraph
    div = []
    for k, t in enumerate(tiles):
        gs = []
        for s in seeds:
            G = res["seeds"][str(s)]["graphs"][k]
            nodes = np.array(G["nodes"], float).reshape(-1, 2)
            edges = np.array(G["edges"], int).reshape(-1, 2)
            gs.append(RoadGraph(nodes, edges, np.zeros(len(nodes), bool)))
        div.append({"sid": t["sid"], **diversity(gs)})
    res["diversity"] = div
    atomic_json(path, res)
    return rid


def _rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def feature_norm():
    """Mean / std of the joint features over REAL training tiles with a DEM."""
    graphs = pickle.loads((PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    rng = np.random.default_rng(0)
    rows = []
    for sid, _, g in graphs["train"]:
        d = dict(np.load(TERRAIN / f"{sid}.npz"))
        if d["valid"].all():
            rows.append(tile_metrics_m13(g, TerrainField(terrain_of(d)), rng))
    X = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in JOINT_KEYS}
    norm = {"mean": {k: float(np.nanmean(v)) for k, v in X.items()}, "std": {k: float(np.nanstd(v)) for k, v in X.items()}}
    from urbanmorph.graphs.m13_metrics import feature_matrix
    Z = feature_matrix(rows, norm)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))[np.triu_indices(len(Z), 1)]
    atomic_json(NORM, {**norm, "bandwidth": float(np.median(D)), "n_tiles": len(rows),
                       "source": "real 3D-aware training graphs, train tiles with a DEM; MMD bandwidth = median "
                                 "pairwise distance of the normalised training features"})
    print("feature norm from", len(rows), "training tiles")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--feature-norm", action="store_true")
    ap.add_argument("--done", help="write this marker file when the whole plan finished")
    a = ap.parse_args()
    if a.splits:
        make_splits()
    if a.feature_norm:
        feature_norm()
    if a.plan:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        plan = json.loads(Path(a.plan).read_text())
        for spec in plan["runs"]:
            run(spec, device)
        if a.done:
            Path(a.done).parent.mkdir(parents=True, exist_ok=True)
            Path(a.done).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
