"""Milestone 15: apply the pre-written style-strength selection rule (configs/m15_style.json "selection_rule") to the
development sweep (configs/m15_plans/dev_tune.json) and freeze the strengths.

    python scripts/m15_dev_select.py [--freeze]

Prints the sweep table; with --freeze writes the chosen strengths into configs/m15_style.json (status "frozen") and the
table to outputs/m15/dev_select.md. Development sites and seeds only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from m15_run import RUNS, run_id
from urbanmorph.config import PROJECT_ROOT

CFG = PROJECT_ROOT / "configs" / "m15_style.json"
PLAN = PROJECT_ROOT / "configs" / "m15_plans" / "dev_tune.json"


def mean(res, key):
    return float(np.nanmean([r[key] for s in res["seeds"].values() for r in s["rows"]]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    plan = json.loads(PLAN.read_text())["runs"]
    res = {}
    for spec in plan:
        r = json.loads((RUNS / f"{run_id(spec)}.json").read_text())
        res[(spec["style"], json.dumps(spec.get("overrides", {}), sort_keys=True))] = r
    off = res[("off", "{}")]
    base = {k: mean(off, k) for k in ("street_km", "deadend_per_km", "loops_per_km2")}
    md = ["# M15 development sweep (syn_dev, TN_s0, M*, seeds 1-3)", "",
          f"Style off: street_km {base['street_km']:.2f}, dead ends / km {base['deadend_per_km']:.2f}, "
          f"loops / km2 {base['loops_per_km2']:.1f}", "",
          "| Style | Setting | field alignment | gain | street_km | orth R4 | radial+tang. | dead ends / km | loops / km2 |",
          "|---|---|---|---|---|---|---|---|---|"]
    chosen = {}
    for st in ("grid", "organic", "radial"):
        rows = []
        for k in (0.5, 1, 2, 4, 8):
            r = res[(st, json.dumps({"style_kappa": k}, sort_keys=True))]
            al = mean(r, "field_alignment")
            # alignment of the style-off networks against THIS style's field (same sites and seeds)
            rows.append((k, al, mean(r, "street_km"), mean(r, "orth_order"), mean(r, "radial_share") + mean(r, "tangential_share"),
                         mean(r, "deadend_per_km"), mean(r, "loops_per_km2")))
        al_off = off_alignment(off, st)
        gains = [x[1] - al_off for x in rows]
        gmax = max(gains)
        ok = [x for x, g in zip(rows, gains) if g >= 0.9 * gmax and x[2] >= 0.8 * base["street_km"]]
        flag = ""
        if not ok:
            ok = [x for x, g in zip(rows, gains) if g >= 0.9 * gmax]
            flag = " (length condition not met: flagged)"
        chosen[st] = {"style_kappa": ok[0][0], "flag": flag.strip()}
        for x, g in zip(rows, gains):
            mark = " **chosen**" if x[0] == ok[0][0] else ""
            md.append(f"| {st} | kappa {x[0]}{mark} | {x[1]:.3f} | {g:+.3f} | {x[2]:.2f} | {x[3]:.2f} | {x[4]:.2f} | {x[5]:.2f} | {x[6]:.1f} |")
        md.append(f"| {st} | off (alignment vs this field) | {al_off:.3f} | — | {base['street_km']:.2f} | | | | |")
    best = None
    for b, l in ((0.25, 2), (0.5, 2), (0.5, 4), (1.0, 4)):
        r = res[("culdesac", json.dumps({"stop_bias": b, "style_loop": l}, sort_keys=True))]
        km, de, lo = mean(r, "street_km"), mean(r, "deadend_per_km"), mean(r, "loops_per_km2")
        ok = km >= 0.6 * base["street_km"] and de >= 1.25 * base["deadend_per_km"]
        if ok and (best is None or lo < best[2] - 1e-12):
            best = (b, l, lo)
        md.append(f"| culdesac | stop_bias {b}, loop {l}{' (eligible)' if ok else ''} | | | {km:.2f} | {mean(r, 'orth_order'):.2f} | | {de:.2f} | {lo:.1f} |")
    chosen["culdesac"] = {"stop_bias": best[0], "style_loop": best[1], "flag": ""} if best else {"flag": "no eligible setting"}
    md += ["", "Chosen (rule in configs/m15_style.json): " + json.dumps(chosen)]
    print("\n".join(md))
    if a.freeze:
        cfg = json.loads(CFG.read_text())
        for st in ("grid", "organic", "radial"):
            cfg["styles"][st]["style_kappa"] = chosen[st]["style_kappa"]
        if best:
            cfg["styles"]["culdesac"].update(stop_bias=best[0], style_loop=best[1])
        cfg["status"] = "frozen (scripts/m15_dev_select.py --freeze; development sweep only)"
        cfg["dev_selection"] = chosen
        CFG.write_text(json.dumps(cfg, indent=1))
        (PROJECT_ROOT / "outputs" / "m15" / "dev_select.md").write_text("\n".join(md) + "\n")
    return 0


def off_alignment(off, st):
    """Field alignment of the style-OFF dev networks against the given style's field (the gain's reference)."""
    from urbanmorph.experiments.graph_io import graph_from_json
    from urbanmorph.graphs.style import StyleField
    from urbanmorph.graphs.style_metrics import field_alignment
    from urbanmorph.graphs.tc_sampler import SamplerConfig
    from m15_run import style_overrides
    cfg = SamplerConfig(**style_overrides(st))
    vals = []
    for s, sd in off["seeds"].items():
        for row, G in zip(sd["rows"], sd["graphs"]):
            vals.append(field_alignment(graph_from_json(G), StyleField.build(cfg, row["site"], int(s))))
    return float(np.nanmean(vals))


if __name__ == "__main__":
    sys.exit(main())
