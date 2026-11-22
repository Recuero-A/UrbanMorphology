"""Milestone 17 run plans (docs/M17_PROTOCOL.md). Writes configs/m17_plans/<name>.json; runs go to outputs/m17/runs.

    python scripts/m17_plans.py diagnose     # step 2: replay + event logs, reward and density interventions (dev)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "configs" / "m17_plans"
DEV_SITES = ("named_glo30", "named_mdt05")          # the recurring terrain pack (development and regression)
DEV_SEEDS = list(range(1, 11))
REPLAY = ["A", "M*", "M13-SMC", "NA-myopic", "NA-fixed", "NA-adaptive", "NA-random", "NA-SMC"]


def style(st):
    from m16_plans import proposals
    fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
    return proposals("style_topo", fz["M*"][st]["style_topo"]["prop_alpha"])


def diagnose() -> list[dict]:
    from urbanmorph.experiments.methods import lam
    runs = []
    for sites in DEV_SITES:
        base = {"sites": sites, "backbone": "TN_s0", "start": "centre", "seeds": DEV_SEEDS,
                "grade_span": 30.0 if sites == "named_glo30" else 10.0}
        ev = {"event_log": True}
        for m in REPLAY:
            runs.append({**base, "method": m, "style": "off", "overrides": ev})
        runs.append({**base, "method": "M12-HR", "style": "off"})                 # legacy loop: no event log
        for st in ("organic", "radial"):
            runs.append({**base, "method": "M*", "style": st, "proposals": style(st), "overrides": ev})
        for m in ("NA-fixed", "NA-SMC"):                                       # H2: the dead-end reward, scaled
            for mult in (0.0, 0.5, 2.0):
                runs.append({**base, "method": m, "style": "off", "overrides": {**ev, **lam(mult)}})
        for m in ("M*", "NA-SMC"):                                             # H3: density budget (stop bias)
            for b in (-0.5, 0.5, 1.0):
                runs.append({**base, "method": m, "style": "off", "overrides": {**ev, "stop_bias": b}})
        for st in ("organic", "radial"):                                       # H3: style at reduced density
            for b in (1.0, 2.0):
                runs.append({**base, "method": "M*", "style": st, "proposals": style(st),
                             "overrides": {**ev, "stop_bias": b}})
    return runs


def main() -> int:
    name = sys.argv[1]
    runs = {"diagnose": diagnose}[name]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"protocol": "docs/M17_PROTOCOL.md", "runs": runs}, indent=1))
    print(name, len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
