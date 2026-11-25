"""Milestone 16 experiment runner (library form of scripts/m15_run.py, which stays unchanged for M15 replays).

A run = (site set, what the sampler is SHOWN, backbone, potentials, method, style preset, proposal mode, temperature,
grade ceiling, start, inference seeds). For every seed all sites are generated in one batch; every graph is scored on
the TRUE terrain with the M15 morphology / terrain metrics, the M14 connectivity metrics, the M16 independent network
outcomes (graphs/network_outcomes.py) and the sampler's own log (candidate sources, rejections, stops, look-ahead).

Spec keys (absent = default; the run id hashes the spec, so a stored run is never overwritten by another spec):
* ``sites``      site-set name (experiments/sites.py);
* ``shown``      what the sampler sees: absent = the true sites, ``"flatten"`` = the same sites with elevation set to
                 its tile median (water kept), or another site-set name with the same site keys (mismatch control);
* ``backbone``   checkpoint run under checkpoints/m12 (TN_s0, TNlp_s0, TNse_s0, N0_s0, ...);
* ``potentials`` configs/m13_potentials/<name>.json; default = the backbone family (TN, TNlp, TNse, N0), as in M13/M14
                 (scripts/m15_run.py always used TN, correct there because M15 ran TN backbones only);
* ``method``     a label of experiments/methods.py, or ``P`` / ``TRACER``;
* ``style``      a preset of configs/m15_style.json (heading reranking) or ``off``;
* ``proposals``  M16 style-aware proposals, a dict passed to SamplerConfig (graphs/style_proposals.py), e.g.
                 {"prop_mode": "style", "prop_m": 4};
* ``overrides``, ``temperature``, ``grade_cap``, ``start`` (``"centre"`` = fixed start), ``seeds``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.experiments.backbones import load_backbone, model_inputs
from urbanmorph.experiments.graph_io import graph_from_json, graph_to_json
from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.experiments.methods import method
from urbanmorph.experiments.sites import site_set
from urbanmorph.graphs.m13_metrics import diversity, tile_metrics_m13
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.network_outcomes import outcome_row
from urbanmorph.graphs.style import StyleField
from urbanmorph.graphs.style_metrics import field_alignment, morphology_row
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import SamplerConfig, preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of

STYLE_CFG = PROJECT_ROOT / "configs" / "m15_style.json"
DECLARED_CENTRE = (256.0, 256.0)
M13_KEYS = ("grade_p90_m13", "grade_p50_m13", "contour_align")     # the M13 terrain metrics (30 m sliding spans)
_MODELS: dict = {}


def run_id(spec: dict) -> str:
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    tag = spec["method"].replace("*", "star").replace("+", "-")
    mode = (spec.get("proposals") or {}).get("prop_mode", "")
    return f"{spec['sites']}__{spec['backbone']}__{tag}__{spec.get('style', 'off')}{'__' + mode if mode else ''}__{h}"


def style_overrides(name: str | None) -> dict:
    if name in (None, "off"):
        return {}
    return dict(json.loads(STYLE_CFG.read_text())["styles"][name])


def potentials_name(spec: dict) -> str:
    if spec.get("potentials"):
        return spec["potentials"]
    fam = spec["backbone"].split("_")[0]
    return fam if (PROJECT_ROOT / "configs" / "m13_potentials" / f"{fam}.json").exists() else "TN"


def sampler_config(spec: dict):
    m = method(spec["method"])
    ov = {**m.overrides, **style_overrides(spec.get("style")), **(spec.get("proposals") or {}),
          **spec.get("overrides", {})}
    if spec.get("temperature") is not None:
        ov["temperature"] = spec["temperature"]
    if spec.get("grade_cap", "off") != "off":
        ov["grade_cap"] = spec["grade_cap"]
    if spec.get("start") == "centre":
        ov["fixed_start"] = "centre"
    is_net = m.preset in NET_PRESETS
    return m, (net_preset(m.preset, **ov) if is_net else preset(m.preset, **ov)), is_net


def flatten(d: dict) -> dict:
    """The same site with its elevation replaced by the tile median (water and validity kept): the sampler is told
    the land is flat while rules and metrics keep the true terrain."""
    z = np.where(d["valid"], d["elev"], np.nan)
    return {**d, "elev": np.full_like(d["elev"], np.nanmedian(z) if np.isfinite(z).any() else 0.0)}


def shown_sites(spec: dict, true: list[dict]) -> list[dict]:
    s = spec.get("shown")
    if not s:
        return true
    if s == "flatten":
        return [flatten(d) for d in true]
    shown = site_set(s)
    assert [d["site"] for d in shown] == [d["site"] for d in true]
    return shown


def _crop_inputs(mcfg: dict, shown: list[dict]):
    """Model-only terrain (low-pass / street-erased). Synthetic sites carry no street imprint, so a street-erased
    backbone sees them unchanged."""
    if mcfg.get("dem_variant") and all(str(d["site"]).startswith(("eval_", "dev_", "conf_", "fix_", "demo_", "named_")) for d in shown):
        return None
    return model_inputs(mcfg, [d["site"] for d in shown], shown)


def generate(spec: dict, shown: list[dict], seed: int, device: str):
    """Graphs (one per site), the sampler's per-site log and run info, for one seed."""
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / f"{potentials_name(spec)}.json")
    if spec["method"] == "TRACER":
        from urbanmorph.graphs.field_tracer import trace
        cfg = SamplerConfig(**{**style_overrides(spec["style"]), **spec.get("overrides", {})})
        out = [trace(d, StyleField.build(cfg, d["site"], seed), pots, np.random.default_rng([seed, i]),
                     grade_cap=spec.get("grade_cap", "off"), **spec.get("tracer", {})) for i, d in enumerate(shown)]
        return [g for g, _ in out], [st for _, st in out], {"forward_passes": 0}
    if spec["method"] == "P":
        from urbanmorph.experiments.references import TH, median_density
        from urbanmorph.graphs.procedural_terrain import sample_P
        rng = np.random.default_rng(seed)
        return ([sample_P(terrain_of(d), rng, median_density(), TH) for d in shown], [{} for _ in shown],
                {"forward_passes": 0})
    if spec["backbone"] not in _MODELS:
        _MODELS[spec["backbone"]] = load_backbone(spec["backbone"], device)
    model, names, norm, mcfg = _MODELS[spec["backbone"]]
    m, cfg, is_net = sampler_config(spec)
    crop = _crop_inputs(mcfg, shown)
    f = sample_net if is_net else sample_tc
    gs, st, info = f(model, shown, names, norm, pots, cfg, seed=seed, device=device, crop_tiles=crop)
    if m.extra.get("repair"):
        from urbanmorph.experiments.references import TH
        from urbanmorph.graphs.repair import repair
        gs = [repair(g, terrain_of(d), TH)[0] for g, d in zip(gs, shown)]
    return gs, st, info


def score(spec: dict, d: dict, g, stt: dict, seed: int) -> dict:
    style_cfg = SamplerConfig(**style_overrides(spec.get("style"))) if spec["method"] != "P" else SamplerConfig()
    tf = TerrainField(terrain_of(d))
    rng = np.random.default_rng([seed, 15])
    dc = d.get("style_decl", {})
    sf = StyleField.build(replace(style_cfg, **dc) if dc else style_cfg, d["site"], seed)
    centre = d.get("centre", DECLARED_CENTRE if spec["sites"].startswith(("syn", "fix")) else None)
    return {"site": d["site"],
            **morphology_row(g, tf.t, centre=centre, rng=rng, grade_span=spec.get("grade_span", 10.0)),
            **{k: v for k, v in tile_metrics_m14(g, tf, rng).items() if not isinstance(v, (list, dict))},
            **outcome_row(g, centre=centre),
            **{k: v for k, v in tile_metrics_m13(g, tf, rng).items() if k in M13_KEYS},
            "field_alignment": field_alignment(g, sf) if sf.weights else np.nan,
            "edges": int(len(g.edges)), "bridges": int((np.asarray(g.structs()) == 1).sum()),
            "empty": int(len(g.edges) == 0),
            "log": {k: v for k, v in dict(stt).items() if isinstance(v, (int, float))}}


def run(spec: dict, out_dir: Path, device: str = "cpu") -> str:
    rid = run_id(spec)
    path = Path(out_dir) / f"{rid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text()) if path.exists() else {"spec": spec, "id": rid, "seeds": {}}
    if res["spec"] != spec:
        raise RuntimeError(f"{path} holds a different specification")
    todo = [s for s in spec["seeds"] if str(s) not in res["seeds"]]
    true = site_set(spec["sites"])
    shown = shown_sites(spec, true)
    for s in todo:
        t0 = time.time()
        gs, st, info = generate(spec, shown, s, device)
        events = [dict(stt).pop("_events", None) if isinstance(stt, dict) else None for stt in st]
        st = [{k: v for k, v in dict(stt).items() if k != "_events"} for stt in st]
        rows = [score(spec, d, g, stt, s) for d, g, stt in zip(true, gs, st)]
        res["seeds"][str(s)] = {"rows": rows, "seconds": time.time() - t0, "forward_passes": info.get("forward_passes"),
                                "graphs": [graph_to_json(g) for g in gs]}
        if any(e is not None for e in events):                   # M17 event logs (SamplerConfig.event_log)
            res["seeds"][str(s)]["events"] = events
        atomic_json(path, res)
        print(rid, "seed", s, f"{time.time() - t0:.0f}s", flush=True)
    seeds = sorted(res["seeds"], key=int)
    if "diversity" not in res or len(seeds) == len(spec["seeds"]):
        res["diversity"] = [{"site": d["site"], **diversity([graph_from_json(res["seeds"][x]["graphs"][k])
                                                             for x in seeds])} for k, d in enumerate(true)]
        atomic_json(path, res)
    return rid


def load_run(spec: dict, out_dir: Path) -> dict | None:
    """The stored run of exactly this spec, or None if absent or incomplete."""
    p = Path(out_dir) / f"{run_id(spec)}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if len(r["seeds"]) == len(spec["seeds"]) else None
