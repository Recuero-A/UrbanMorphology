"""Milestone 13 confirmatory experiments, built from the frozen selection (outputs/m13/selection.json).

    python scripts/m13_confirm.py --tier tn       --done outputs/m13/done/tn       # H1, H2, H3 (TN, 3 training seeds)
    python scripts/m13_confirm.py --tier leak     --done outputs/m13/done/leak     # H1-lp, H1-se
    python scripts/m13_confirm.py --tier dm       --done outputs/m13/done/dm       # density-matched A and M*
    python scripts/m13_confirm.py --tier ablate   --done outputs/m13/done/ablate   # §9 ablations, N0
    python scripts/m13_confirm.py --tier gen      --done outputs/m13/done/gen      # H4 (generalisation cities)

Every run uses DEV-B (or GEN) and inference seeds 1-4 (docs/M13_EVALUATION_PROTOCOL.md §3). Nothing here is tuned:
configurations come from the selection file; the density-matching stop bias is calibrated on DEV-A.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import m13_analyze as an  # noqa: E402
import m13_experiments as ex  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m13"
SEEDS = [1, 2, 3, 4]


def sel():
    return json.loads((OUT / "selection.json").read_text())


def best_of(prefix_preset):
    """The best S1 configuration of a method family by the selection rule's S_T among eligible ones."""
    S = sel()["summaries"]
    cand = [s for s in S.values() if s["spec"]["preset"] == prefix_preset]
    prem_a = sel()["premature_A"]
    ok = [s for s in cand if abs(s["density_ratio"] - 1) <= 0.15 and s["premature_share"] <= prem_a + 0.05] or cand
    return min(ok, key=lambda s: s["S_T"])["spec"].get("overrides", {})


def method_table():
    s = sel()
    H = s["H_star"] or 2
    e = s["E_star"]["overrides"]
    return {
        "A0": ("legacy", {}),
        "A": ("standard", {}),
        "B": ("topk", best_of("topk")),
        "C": ("fulltemp", best_of("fulltemp")),
        "D": ("filter", best_of("filter")),
        "E": ("energy", e),
        "softonly": ("softonly", best_of("softonly")),
        "ELA": ("energy_la", {**e, "horizon": H, "rollout_m": 4}),
        "F": ("beam", {**{k: v for k, v in e.items() if k in ("K", "lam_ratio")}, "particles": 4}),
        "G": ("smc", {**{k: v for k, v in e.items() if k in ("K", "lam_ratio")}, "particles": 8}),
        "GLA": ("smc_la", {**{k: v for k, v in e.items() if k in ("K", "lam_ratio")}, "particles": 4, "horizon": H, "rollout_m": 4}),
        "M": (s["M_star"]["preset"], s["M_star"]["overrides"]),
    }


def spec(tiles, backbone, pots, preset, ov, seeds=SEEDS, **extra):
    return {"tiles": tiles, "backbone": backbone, "potentials": pots, "pot_version": "v2", "preset": preset, "seeds": seeds,
            **({"overrides": ov} if ov else {}), **extra}


def calibrate_stop_bias(backbone, pots, preset, ov, device):
    """Bisection on DEV-A (inference seeds 1-2): mean density within +-5 % of real; at most 6 runs."""
    lo, hi, b, hist = -2.0, 2.0, 0.0, []
    for _ in range(6):
        s = spec("devA", backbone, pots, preset, {**ov, "stop_bias": round(b, 4)}, seeds=[1, 2])
        rid = ex.run(s, device)
        r = an.summarize(json.loads((ex.RUNS / f"{rid}.json").read_text()))
        hist.append({"bias": b, "density_ratio": r["density_ratio"]})
        if abs(r["density_ratio"] - 1) <= 0.05:
            break
        if r["density_ratio"] > 1:          # too dense -> stop more
            lo = b
        else:
            hi = b
        b = 0.5 * (lo + hi)
    return b, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["tn", "leak", "dm", "ablate", "gen"])
    ap.add_argument("--done")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    M = method_table()
    runs = []
    if a.tier == "tn":
        for sd in (0, 1, 2):
            for k in ("A0", "A", "B", "C", "D", "E", "softonly", "ELA", "F", "G", "GLA", "M"):
                runs.append(spec("devB", f"TN_s{sd}", "TN", *M[k]))
            runs.append(spec("devB", f"TN_s{sd}", "TN", *M["M"], repair=True))
        runs.append(spec("devB", "TN_s0", "TN", "P", {}))
    elif a.tier == "leak":
        for fam in ("TNlp", "TNse"):
            for sd in (0, 1, 2):
                for k in ("A", "E", "ELA", "M"):
                    runs.append(spec("devB", f"{fam}_s{sd}", fam, *M[k]))
    elif a.tier == "dm":
        cal = {}
        for k in ("A", "M"):
            b, hist = calibrate_stop_bias("TN_s0", "TN", *M[k], device)
            cal[k] = {"stop_bias": b, "bisection": hist}
            for sd in (0, 1, 2):
                runs.append(spec("devB", f"TN_s{sd}", "TN", M[k][0], {**M[k][1], "stop_bias": round(b, 4)}))
        (OUT / "density_calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    elif a.tier == "ablate":
        pre, ov = M["M"]
        for sd in (0, 1, 2):
            runs.append(spec("devB", f"N0_s{sd}", "N0", "standard", {}))
            runs.append(spec("devB", f"N0_s{sd}", "N0", pre, ov))
        runs.append(spec("devB", "TN_s0", "TN", pre, {**ov, "lam_infra": 0.0}))
        runs.append(spec("devB", "TN_s0", "TN", pre, {**ov, "scheduler": "fifo"}))
        runs.append(spec("devB", "TN_s0", "TN", pre, {**ov, "scheduler": "stochastic"}))
        runs.append(spec("devB", "TN_s0", "TN", "energy_la", {**M["E"][1], "horizon": 1, "rollout_m": 4}))
    elif a.tier == "gen":
        for fam in ("TN", "TNlp", "TNse"):
            for sd in (0, 1, 2):
                for k in ("A", "M"):
                    runs.append(spec("gen", f"{fam}_s{sd}", fam, *M[k]))
    for s in runs:
        ex.run(s, device)
    (OUT / "plans").mkdir(parents=True, exist_ok=True)
    (OUT / "plans" / f"{a.tier}.json").write_text(json.dumps({"runs": runs}, indent=1))
    if a.done:
        Path(a.done).parent.mkdir(parents=True, exist_ok=True)
        Path(a.done).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
