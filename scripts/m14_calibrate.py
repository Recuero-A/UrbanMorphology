"""Milestone 14: moment-matching calibration of the network-potential weights on TRAINING tiles (protocol §3).

    python scripts/m14_calibrate.py      # -> configs/m14_lambda.json, outputs/m14/calibration/log.json

Target (formulation §3): pi_lam ∝ p_theta 1[F] exp(lam_r phi_r - lam_D D(x) - lam_I I(x)). The I-projection of the
model onto {q : E_q[D] = D_real, E_q[I] = I_real} is this exponential tilt, with lam the solution of the moment
equations. We solve them by generalized iterative scaling (multiplicative moment matching):
    lam_k <- clip(lam_k + log((E_gen[S_k] + eps) / (E_real[S_k] + eps)), 0, LAM_MAX),   S = (D, I), eps = 0.05,
with E_gen estimated by the na_fixed sampler (roll-outs at every decision, H = 2) on the 48 train_cal tiles (training
geography only), backbone TN_s0, inference seeds 1-2. Stop when both moments are within 10 % (D) / 0.05 per tile (I)
or after 6 iterations; the last iterate is kept. The single-chain sampler does not sample pi_lam exactly
(formulation §4), so lam is the value that matches the moments UNDER THE IMPLEMENTED SAMPLER.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.road_graph import is_border

OUT = PROJECT_ROOT / "outputs" / "m14" / "calibration"
EPS, LAM_MAX, ITERS = 0.05, 6.0, 6


def counts(g):
    if not len(g.edges):
        return 0, 0
    from urbanmorph.graphs.m14_metrics import closed_components
    deg = g.degree()
    D = int(((deg == 1) & ~is_border(g.nodes)).sum())
    lab, L, _ = closed_components(g)
    cl = np.bincount(lab[g.edges[:, 0]], weights=L, minlength=lab.max() + 1)
    n = len(g.nodes)
    return D, int(sum(1 for c in range(len(cl)) if c != lab[n] and cl[c] >= 40))


def one(args):
    lam_d, lam_i, seed = args
    import torch
    torch.set_num_threads(3)
    import m13_experiments as ex
    import m14_experiments as mx
    from urbanmorph.graphs.m14_sampler import net_preset, sample_net
    from urbanmorph.graphs.tc_potentials import Potentials
    T = mx.tile_set("train_cal")
    m, names, norm, _ = ex.load_backbone("TN_s0", "cpu")
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    gs, _, info = sample_net(m, [t["d"] for t in T], names, norm, pots,
                             net_preset("na_fixed", lam_dead=lam_d, lam_island=lam_i), seed=seed, device="cpu")
    return [counts(g) for g in gs], info["seconds"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import m14_experiments as mx
    real = np.array([counts(t["graph"]) for t in mx.tile_set("train_cal")], float)
    target = real.mean(0)
    lam = np.zeros(2)
    log = {"target": {"D": target[0], "I": target[1]}, "iters": []}
    for it in range(ITERS + 1):
        with Pool(2) as pool:
            out = pool.map(one, [(lam[0], lam[1], s) for s in (1, 2)])
        gen = np.array([c for o, _ in out for c in o], float).mean(0)
        log["iters"].append({"lam_dead": lam[0], "lam_island": lam[1], "gen_D": gen[0], "gen_I": gen[1],
                             "seconds": sum(sec for _, sec in out)})
        print(log["iters"][-1], "target", target, flush=True)
        (OUT / "log.json").write_text(json.dumps(log, indent=1, default=float))
        ok = abs(gen[0] / target[0] - 1) <= 0.10 and abs(gen[1] - target[1]) <= 0.05
        if ok or it == ITERS:
            break
        lam = np.clip(lam + np.log((gen + EPS) / (target + EPS)), 0, LAM_MAX)
    res = {"lam_dead": float(lam[0]), "lam_island": float(lam[1]), "converged": bool(ok), "iterations": it,
           "sampler": "na_fixed (H=2, rollout_m=4, K=8, lam_ratio=0.5), TN_s0, train_cal 48 tiles, seeds 1-2",
           "target_real": {"D_per_tile": target[0], "I_per_tile": target[1]}, "final_gen": {"D": gen[0], "I": gen[1]}}
    (PROJECT_ROOT / "configs" / "m14_lambda.json").write_text(json.dumps(res, indent=1, default=float))
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
