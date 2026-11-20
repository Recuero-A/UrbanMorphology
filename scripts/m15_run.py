"""Milestone 15 experiment runner: style guidance on top of the existing samplers (docs/M15_PROTOCOL.md).

    python scripts/m15_run.py --plan configs/m15_plans/<name>.json [--worker k/n] [--threads 2]

A run = (site set, backbone, method from urbanmorph.experiments.methods or TRACER / P, style preset, temperature,
grade ceiling, negative-control flag, inference seeds). For every seed all sites are generated in one batch and every
graph is scored on the TRUE terrain: M15 style / terrain metrics (graphs/style_metrics), M14 connectivity and coverage
(graphs/m14_metrics), the controllability check (field alignment), and the sampler's own log (candidates evaluated,
rejected by each hard rule, stops, forced stops, look-ahead steps, failures). Diversity across seeds is added per site.
Output: outputs/m15/runs/<run_id>.json, rewritten atomically after every seed (resumable); never overwritten by a
different specification (the id hashes the specification).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.experiments.backbones import load_backbone, model_inputs
from urbanmorph.experiments.graph_io import graph_from_json, graph_to_json
from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.experiments.methods import method
from urbanmorph.experiments.sites import site_set
from urbanmorph.graphs.m13_metrics import diversity
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.style import StyleField
from urbanmorph.graphs.style_metrics import field_alignment, morphology_row
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import SamplerConfig, preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = PROJECT_ROOT / "outputs" / "m15"
RUNS = OUT / "runs"
STYLE_CFG = PROJECT_ROOT / "configs" / "m15_style.json"
DECLARED_CENTRE = (256.0, 256.0)
_MODELS = {}


def run_id(spec: dict) -> str:
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    tag = spec["method"].replace("*", "star").replace("+", "-")
    return f"{spec['sites']}__{spec['backbone']}__{tag}__{spec.get('style', 'off')}__{h}"


def style_overrides(name: str) -> dict:
    if name in (None, "off"):
        return {}
    return dict(json.loads(STYLE_CFG.read_text())["styles"][name])


def sampler_config(spec: dict):
    m = method(spec["method"])
    ov = {**m.overrides, **style_overrides(spec.get("style")), **spec.get("overrides", {})}
    if spec.get("temperature") is not None:
        ov["temperature"] = spec["temperature"]
    if spec.get("grade_cap", "off") != "off":
        ov["grade_cap"] = spec["grade_cap"]
    if spec.get("start") == "centre":                  # protocol amendment 2: fixed starting point
        ov["fixed_start"] = "centre"
    is_net = m.preset in NET_PRESETS
    return m, (net_preset(m.preset, **ov) if is_net else preset(m.preset, **ov)), is_net


def generate(spec, shown, seed, device):
    """Graphs (one per site) and the sampler's per-site log, for one seed."""
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    if spec["method"] == "TRACER":
        from urbanmorph.graphs.field_tracer import trace
        cfg = SamplerConfig(**{**style_overrides(spec["style"]), **spec.get("overrides", {})})
        out = [trace(d, StyleField.build(cfg, d["site"], seed), pots, np.random.default_rng([seed, i]),
                     grade_cap=spec.get("grade_cap", "off"), **spec.get("tracer", {})) for i, d in enumerate(shown)]
        return [g for g, _ in out], [st for _, st in out], {"forward_passes": 0}
    if spec["method"] == "P":
        from m12_evaluate import TH, density_pool, real_graphs
        from urbanmorph.graphs.procedural_terrain import sample_P
        dens = float(np.median(density_pool(real_graphs())))
        rng = np.random.default_rng(seed)
        return [sample_P(terrain_of(d), rng, dens, TH) for d in shown], [{} for _ in shown], {"forward_passes": 0}
    if spec["backbone"] not in _MODELS:
        _MODELS[spec["backbone"]] = load_backbone(spec["backbone"], device)
    model, names, norm, mcfg = _MODELS[spec["backbone"]]
    m, cfg, is_net = sampler_config(spec)
    crop = model_inputs(mcfg, [d["site"] for d in shown], shown) if (mcfg.get("lowpass_m")) else None
    f = sample_net if is_net else sample_tc
    gs, st, info = f(model, shown, names, norm, pots, cfg, seed=seed, device=device, crop_tiles=crop)
    if m.extra.get("repair"):
        from m12_evaluate import TH
        from urbanmorph.graphs.repair import repair
        gs = [repair(g, terrain_of(d), TH)[0] for g, d in zip(gs, shown)]
    return gs, st, info


def run(spec: dict, device: str) -> str:
    rid = run_id(spec)
    path = RUNS / f"{rid}.json"
    res = json.loads(path.read_text()) if path.exists() else {"spec": spec, "id": rid, "seeds": {}}
    todo = [s for s in spec["seeds"] if str(s) not in res["seeds"]]
    true = site_set(spec["sites"])
    shown = site_set(spec["shown"]) if spec.get("shown") else true            # negative control: mismatched terrain
    assert [d["site"] for d in shown] == [d["site"] for d in true]
    style_cfg = SamplerConfig(**style_overrides(spec.get("style"))) if spec["method"] != "P" else SamplerConfig()
    for s in todo:
        t0 = time.time()
        gs, st, info = generate(spec, shown, s, device)
        rows = []
        for d, g, stt in zip(true, gs, st):
            tf = TerrainField(terrain_of(d))
            rng = np.random.default_rng([s, 15])
            dc = d.get("style_decl", {})
            sf = StyleField.build(replace(style_cfg, **dc) if dc else style_cfg, d["site"], s)
            centre = DECLARED_CENTRE if spec["sites"].startswith("syn") else None    # real: one estimation rule
            rows.append({"site": d["site"], **morphology_row(g, tf.t, centre=centre, rng=rng,
                                                              grade_span=spec.get("grade_span", 10.0)),
                         **{k: v for k, v in tile_metrics_m14(g, tf, rng).items() if not isinstance(v, (list, dict))},
                         "field_alignment": field_alignment(g, sf) if sf.weights else np.nan,
                         "edges": int(len(g.edges)), "bridges": int((np.asarray(g.structs()) == 1).sum()),
                         "empty": int(len(g.edges) == 0),
                         "log": {k: v for k, v in dict(stt).items() if isinstance(v, (int, float))}})
        res["seeds"][str(s)] = {"rows": rows, "seconds": time.time() - t0, "forward_passes": info.get("forward_passes"),
                                "graphs": [graph_to_json(g) for g in gs]}
        atomic_json(path, res)
        print(rid, "seed", s, f"{time.time() - t0:.0f}s", flush=True)
    seeds = sorted(res["seeds"], key=int)
    res["diversity"] = [{"site": d["site"], **diversity([graph_from_json(res["seeds"][s]["graphs"][k]) for s in seeds])}
                        for k, d in enumerate(true)]
    atomic_json(path, res)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--done")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    k, n = map(int, a.worker.split("/"))
    plan = json.loads(Path(a.plan).read_text())
    for j, spec in enumerate(plan["runs"]):
        if j % n == k:
            run(spec, "cpu")
    if a.done:
        Path(a.done).parent.mkdir(parents=True, exist_ok=True)
        Path(a.done).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
