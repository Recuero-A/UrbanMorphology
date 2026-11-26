"""Milestone 14 density control (docs/M14_EXPERIMENTAL_PROTOCOL.md §7.1): per method, the stop-logit bias that brings
the mean street density to the real density on DEV-A (TN_s0, inference seeds 1-2), by bisection on [-1.5, 1.5]
(8 steps, stop when |ratio - 1| <= 0.05). Development data only. Methods: A, M*, NA-myopic, NA-SMC (NA-fixed and
NA-adaptive are not density-matched: cost; documented).

    python scripts/m14_density_calibrate.py <label>     # one method (run several in parallel)
    -> outputs/m14/density/<label>.json; configs/m14_density.json is assembled by --collect
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import m13_experiments as ex  # noqa: E402
import m14_experiments as mx  # noqa: E402
from m14_plans import methods  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset, sample_tc

OUT = PROJECT_ROOT / "outputs" / "m14" / "density"
FZ = json.loads((PROJECT_ROOT / "configs" / "m14_frozen.json").read_text())


def density(label, bias, T, model, names, norm, pots):
    _, p, ov, _ = [m for m in methods(FZ) if m[0] == label][0]
    ov = {**ov, "stop_bias": bias}
    cfg = net_preset(p, **ov) if p in NET_PRESETS else preset(p, **ov)
    dens = []
    for s in (1, 2):
        f = sample_net if p in NET_PRESETS else sample_tc
        gs, _, _ = f(model, [t["d"] for t in T], names, norm, pots, cfg, seed=s, device="cpu")
        dens += [float(np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T).sum() / 1000 / 0.262144)
                 if len(g.edges) else 0.0 for g in gs]
    return float(np.mean(dens))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "--collect":
        res = {f.stem: json.loads(f.read_text())["bias"] for f in OUT.glob("*.json")}
        (PROJECT_ROOT / "configs" / "m14_density.json").write_text(json.dumps({"stop_bias": res,
            "rule": "protocol §7.1: DEV-A, TN_s0, seeds 1-2, bisection to density ratio 1 +- 0.05"}, indent=1))
        print(res)
        return 0
    label = sys.argv[1]
    torch.set_num_threads(2)
    T = mx.tile_set("devA")
    real = float(np.mean([r["m12_street_km_per_km2"] for r in mx.real_rows(T, "devA")]))
    model, names, norm, _ = ex.load_backbone("TN_s0", "cpu")
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    lo, hi, log = -1.5, 1.5, []
    b = 0.0
    for _ in range(8):
        r = density(label, b, T, model, names, norm, pots) / real
        log.append({"bias": b, "ratio": r})
        print(label, b, round(r, 3), flush=True)
        if abs(r - 1) <= 0.05:
            break
        # a positive bias raises the stop probability -> lower density
        lo, hi = (b, hi) if r > 1 else (lo, b)
        b = 0.5 * (lo + hi)
    best = min(log, key=lambda x: abs(x["ratio"] - 1))
    (OUT / f"{label}.json").write_text(json.dumps({"bias": best["bias"], "ratio": best["ratio"], "log": log}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
