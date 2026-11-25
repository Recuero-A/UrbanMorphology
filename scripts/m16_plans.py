"""Milestone 16 run plans (docs/M16_PROTOCOL.md). Writes configs/m16_plans/<name>.json.

    python scripts/m16_plans.py replay     # part 1: milestone replays and the terrain-regression diagnosis
    python scripts/m16_plans.py dev        # part 4: development sweep of the proposal settings (syn_dev, dev seeds)
    python scripts/m16_plans.py confirm    # part 5: confirmatory runs (fresh sites; written after the freeze)
    python scripts/m16_plans.py named      # recurring comparisons on the named real sites (after the freeze)
    python scripts/m16_plans.py sweeps     # secondary: temperature and stop bias (after the confirmatory runs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "configs" / "m16_plans"

M14 = ["A", "M*", "M*+repair", "P", "M13-SMC", "NA-myopic", "NA-fixed", "NA-adaptive", "NA-random", "NA-SMC"]
REPLAY_SEEDS = [101, 102, 103, 104, 105]


def replay() -> list[dict]:
    runs = []
    # R1 historical configurations: each milestone's own loop and start rule (learned start head), demo landscapes
    for m in ["M12-loop", "M12-HR"] + M14:
        runs.append({"sites": "demo_scen", "backbone": "TN_s0", "method": m, "style": "off", "seeds": REPLAY_SEEDS})
    # R2 matched: same sites, seeds, fixed start, budget and evaluator; one component at a time
    for sites in ("syn_eval", "demo_scen"):
        base = {"sites": sites, "backbone": "TN_s0", "style": "off", "start": "centre", "seeds": REPLAY_SEEDS}
        for m in M14:
            runs.append({**base, "method": m})
        runs.append({**base, "method": "M*", "grade_cap": "real_p99"})          # the M12 hard grade limit, alone
        runs.append({**base, "method": "NA-fixed", "grade_cap": "real_p99"})
        for bb in ("TNlp_s0", "TNse_s0"):                                       # terrain-input controls
            for m in ("A", "M*", "NA-fixed"):
                runs.append({**base, "backbone": bb, "method": m})
        for m in ("A", "M*", "NA-fixed"):                                       # what the sampler is shown
            runs.append({**base, "method": m, "shown": "flatten"})
            if sites == "syn_eval":
                runs.append({**base, "sites": "syn_eval_nonflat", "method": m, "shown": "syn_eval_mismatch"})
    return runs


STYLES = ("grid", "organic", "radial", "culdesac")
SAMPLERS = ("M*", "NA-fixed")
DEV_SEEDS = [201, 202, 203, 204, 205]
CONF_SEEDS = list(range(301, 321))


def topology() -> dict:
    t = json.loads((ROOT / "configs" / "m16_topology.json").read_text())
    return {k: t[k] for k in ("grid_spacing_m", "ring_spacing_m", "branch_len_m", "term_depth", "n_spokes", "topo_continue")}


def proposals(mode: str, alpha: float, **kw) -> dict:
    return {"prop_mode": mode, "prop_alpha": alpha, **topology(), **kw}


def dev() -> list[dict]:
    runs = []
    for m in SAMPLERS:
        base = {"sites": "syn_dev", "backbone": "TN_s0", "method": m, "start": "centre", "seeds": DEV_SEEDS}
        runs.append({**base, "style": "off"})
        for st in STYLES:
            runs.append({**base, "style": st})
            for mode in ("style", "style_topo"):
                for al in (0.5, 0.8):
                    runs.append({**base, "style": st, "proposals": proposals(mode, al)})
            if m == "M*":                                  # adaptive trigger: its firing rate sets the random control
                for al in (0.5, 0.8):
                    runs.append({**base, "style": st, "proposals": proposals("style_topo", al, prop_trigger="deficit:2")})
    return runs


def dev_stat(spec, num: str, den: str) -> float:
    """sum(num) / sum(den) over a stored dev run's logs."""
    from urbanmorph.experiments.runner import load_run
    r = load_run(spec, ROOT / "outputs" / "m16" / "runs")
    rows = [x for v in r["seeds"].values() for x in v["rows"]]
    return sum(x["log"].get(num, 0) for x in rows) / max(sum(x["log"].get(den, 0) for x in rows), 1)


def confirm() -> list[dict]:
    """Confirmatory runs on the fresh syn_conf sites, seeds 301-320 (docs/M16_PROTOCOL.md §3); alphas from
    configs/m16_frozen.json (the development rule), matched-compute pool size and random-trigger rate from dev runs."""
    import math
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    al = lambda m, st, mode: fz[m][st][mode]["prop_alpha"]
    dv = {json.dumps(s, sort_keys=True): s for s in dev()}
    runs = []
    for m in SAMPLERS:
        base = {"sites": "syn_conf", "backbone": "TN_s0", "method": m, "start": "centre", "seeds": CONF_SEEDS}
        runs.append({**base, "style": "off"})
        for st in STYLES:
            runs.append({**base, "style": st})
            for mode in ("style", "style_topo"):
                runs.append({**base, "style": st, "proposals": proposals(mode, al(m, st, mode))})
            topo = proposals("style_topo", al(m, st, "style_topo"))
            dspec = {"sites": "syn_dev", "backbone": "TN_s0", "method": m, "start": "centre", "seeds": DEV_SEEDS,
                     "style": st, "proposals": topo}
            extra = dev_stat(dspec, "cand_style", "prop_decisions")
            runs.append({**base, "style": st, "overrides": {"K": 8 + math.ceil(extra)}, "control": "matched_pool"})
            runs.append({**base, "style": st, "proposals": {**topo, "prop_rule_only": True}})
            runs.append({**base, "style": st, "proposals": {**topo, "prop_weight": "is"}})
            for shown, sites in (("flatten", "syn_conf"), ("syn_conf_mismatch", "syn_conf_nonflat")):
                runs.append({**base, "sites": sites, "style": st, "proposals": topo, "shown": shown})
            if m == "M*":
                dtrig = {**dspec, "proposals": proposals("style_topo", al(m, st, "style_topo"), prop_trigger="deficit:2")}
                rate = 1 - dev_stat(dtrig, "prop_skipped", "prop_decisions")
                runs.append({**base, "style": st, "proposals": {**topo, "prop_trigger": "deficit:2"}})
                runs.append({**base, "style": st, "proposals": {**topo, "prop_trigger": f"random:{rate:.4f}"}})
        for shown, sites in (("flatten", "syn_conf"), ("syn_conf_mismatch", "syn_conf_nonflat")):
            runs.append({**base, "sites": sites, "style": "off", "shown": shown})
        cds = proposals("style_topo", al(m, "culdesac", "style_topo"))
        if m == "NA-fixed":
            runs.append({**base, "style": "culdesac", "proposals": {**cds, "dead_exempt_local": True}})
    for m in SAMPLERS:                                    # amendment 1: secondary backbone TNse_s0, same alphas
        base = {"sites": "syn_conf", "backbone": "TNse_s0", "method": m, "start": "centre", "seeds": CONF_SEEDS}
        runs.append({**base, "style": "off"})
        for st in STYLES:
            runs.append({**base, "style": st, "proposals": proposals("style_topo", al(m, st, "style_topo"))})
    for m, ref in (("M13-SMC", "M*"), ("NA-SMC", "NA-fixed")):
        base = {"sites": "syn_conf", "backbone": "TN_s0", "method": m, "start": "centre", "seeds": CONF_SEEDS}
        runs.append({**base, "style": "off"})
        for st in STYLES:
            runs.append({**base, "style": st, "proposals": proposals("style_topo", al(ref, st, "style_topo"))})
    return runs


def named() -> list[dict]:
    """Recurring comparisons on the named real sites (GLO-30 DSM for all four; MDT05 DTM for Madrid and El Escorial):
    style off (TN and TNlp), rerank and proposals + topology for every style, M* and NA-fixed, seeds 1-5."""
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    runs = []
    for sites in ("named_glo30", "named_mdt05"):
        for m in SAMPLERS:
            base = {"sites": sites, "backbone": "TN_s0", "method": m, "start": "centre", "seeds": [1, 2, 3, 4, 5],
                    "grade_span": 30.0 if sites == "named_glo30" else 10.0}
            runs.append({**base, "style": "off"})
            runs.append({**base, "backbone": "TNlp_s0", "style": "off"})
            for st in STYLES:
                runs.append({**base, "style": st})
                runs.append({**base, "style": st, "proposals": proposals("style_topo", fz[m][st]["style_topo"]["prop_alpha"])})
    return runs


def sweeps() -> list[dict]:
    """Secondary, after the confirmatory runs: displacement temperature and stop bias, M* grid and cul-de-sac."""
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    runs = []
    for st in ("grid", "culdesac"):
        base = {"sites": "syn_conf", "backbone": "TN_s0", "method": "M*", "start": "centre", "seeds": CONF_SEEDS,
                "style": st, "proposals": proposals("style_topo", fz["M*"][st]["style_topo"]["prop_alpha"])}
        for t in (0.35, 0.8):
            runs.append({**base, "temperature": t})
        for b in (-0.5, 0.5):
            runs.append({**base, "overrides": {"stop_bias": b + (0.25 if st == "culdesac" else 0.0)}})
    return runs


def main() -> int:
    name = sys.argv[1]
    runs = {"replay": replay, "dev": dev, "confirm": confirm, "sweeps": sweeps, "named": named}[name]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"protocol": "docs/M16_PROTOCOL.md", "runs": runs}, indent=1))
    print(name, len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
