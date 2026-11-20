"""Milestone 16 run plans (docs/M16_PROTOCOL.md). Writes configs/m16_plans/<name>.json.

    python scripts/m16_plans.py replay     # part 1: milestone replays and the terrain-regression diagnosis
    python scripts/m16_plans.py dev        # part 4: development sweep of the proposal settings (syn_dev, dev seeds)
    python scripts/m16_plans.py confirm    # part 5: confirmatory runs (fresh sites; written after the freeze)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def main() -> int:
    name = sys.argv[1]
    runs = {"replay": replay}[name]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"protocol": "docs/M16_PROTOCOL.md", "runs": runs}, indent=1))
    print(name, len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
