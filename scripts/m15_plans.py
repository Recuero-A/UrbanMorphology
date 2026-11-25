"""Milestone 15 run plans, generated from the frozen protocol (docs/M15_PROTOCOL.md §2).

    python scripts/m15_plans.py eval      # synthetic interventions, evaluation sites, seeds 101-110
    python scripts/m15_plans.py real      # real-city pilot, development metros (written by the benchmark stage)
    python scripts/m15_plans.py explore   # EXPLORATORY: length-matched tracer (added after results were seen)

Specs are ordered core first. Each plan is written to configs/m15_plans/<name>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "configs" / "m15_plans"
EVAL_SEEDS = list(range(101, 111))
STYLES = ["off", "grid", "organic", "radial", "culdesac"]


def spec(method, style="off", sites="syn_eval", bb="TN_s0", seeds=EVAL_SEEDS, **kw):
    return {"sites": sites, "backbone": bb, "method": method, "style": style, "seeds": seeds, **kw}


def eval_plan() -> list[dict]:
    runs = []
    for m in ("M*", "NA-fixed"):                                             # core 2 x 2 (x every style)
        runs += [spec(m, st) for st in STYLES]
    runs += [spec(m) for m in ("A", "NA-myopic", "NA-SMC", "M13-SMC", "P")]  # existing comparisons, style off
    runs += [spec("TRACER", st) for st in ("grid", "organic", "radial")]     # external baseline
    runs += [spec("M*", st, overrides={"soft": False}) for st in STYLES[1:]]  # morphology-only
    for m in ("M*", "NA-fixed"):                                             # hard-mask control
        runs += [spec(m, st, grade_cap="real_p99") for st in STYLES[1:]]
    for m in ("M*", "NA-fixed"):                                             # negative control (mismatched terrain)
        runs += [spec(m, st, sites="syn_eval_nonflat", shown="syn_eval_mismatch") for st in ("off", "grid")]
    runs += [spec("TRACER", "grid", sites="syn_eval_nonflat", shown="syn_eval_mismatch")]
    runs += [spec("NA-fixed", "grid", grade_cap=c) for c in ("0.12", "0.08")]   # ceiling sensitivity (real_p99 above)
    runs += [spec("M*", "grid", overrides={"style_norm": False})]            # ablation: un-normalised style factor
    for m in ("M*", "NA-fixed"):                                             # temperature, separately
        for st in ("off", "grid"):
            runs += [spec(m, st, temperature=t) for t in (0.8, 1.0)]
    for m in ("M*", "NA-fixed"):                                             # second backbone (secondary)
        runs += [spec(m, st, bb="TN_s1") for st in ("off", "grid")]
    for r in runs:                       # amendment 2: fixed starting point (tile centre / nearest dry point) for all
        r["start"] = "centre"
    return runs


def real_plan() -> list[dict]:
    """Real-city pilot on development metros (protocol §5); GLO-30 grades on 30 m sub-spans."""
    seeds, g30 = [101, 102, 103], {"grade_span": 30.0}
    runs = [spec(m, "off", sites="real_dev", seeds=seeds, **g30) for m in ("M*", "NA-fixed", "A", "NA-SMC", "P")]   # Task A
    runs += [spec(m, "grid", sites="real_dev_gridB", seeds=seeds, **g30) for m in ("M*", "NA-fixed")]             # Task B
    runs += [spec(m, "off", sites="real_dev_spain_mdt05", seeds=seeds, grade_span=10.0) for m in ("M*", "NA-fixed")]  # DEM
    return runs


def explore_plan() -> list[dict]:
    """EXPLORATORY, added after the evaluation results were seen (not part of the protocol's outcomes): the tracer
    with a length budget of 2 km, about the samplers' achieved street length on the sloped sites, so the exposure
    comparison is made at matched length."""
    return [spec("TRACER", st, tracer={"max_km": 2.0}, start="centre") for st in ("grid", "organic", "radial")]


def main() -> int:
    name = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {"eval": eval_plan, "real": real_plan, "explore": explore_plan}[name]()
    (OUT / f"{name}.json").write_text(json.dumps({"protocol": "docs/M15_PROTOCOL.md", "runs": runs}, indent=1))
    print(name, len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
