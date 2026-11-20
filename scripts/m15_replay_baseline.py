"""Milestone 15 audit: deterministic replay of the archived M14 baselines.

Re-runs M* and NA-fixed (GEN2, backbone TN_s0, inference seed 1) with the frozen M14 specification and compares every
generated graph with the graph stored in the M14 run file (node coordinates to 0.01 m, edges, classes, structures).
Writes outputs/m15/audit/<--out> (default replay.json). A mismatch is reported, never silently accepted.

    python scripts/m15_replay_baseline.py [--labels M* NA-fixed] [--seed 1] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import m13_experiments as ex  # noqa: E402
import m14_experiments as mx  # noqa: E402
from m14_report import get  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import preset, sample_tc

OUT = PROJECT_ROOT / "outputs" / "m15" / "audit"


def same(G, g) -> bool:
    return (np.allclose(np.array(G["nodes"], float).reshape(-1, 2), g.nodes.round(2), atol=0.011)
            and np.array_equal(np.array(G["edges"], int).reshape(-1, 2), g.edges)
            and np.array_equal(np.array(G["cls"], int), np.asarray(g.classes()))
            and np.array_equal(np.array(G["st"], int), np.asarray(g.structs())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["M*", "NA-fixed"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="replay.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = mx.tile_set("gen2")
    sids, ds = [t["sid"] for t in tiles], [t["d"] for t in tiles]
    model, names, norm, mcfg = ex.load_backbone("TN_s0", "cpu")
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    crop = ex.model_inputs(mcfg, sids, ds)
    out = {}
    for lab in a.labels:
        run = get("gen2", "TN_s0", lab)
        spec = run["spec"]
        is_net = spec["preset"] in NET_PRESETS
        cfg = net_preset(spec["preset"], **spec["overrides"]) if is_net else preset(spec["preset"], **spec["overrides"])
        t0 = time.time()
        f = sample_net if is_net else sample_tc
        gs, _, _ = f(model, ds, names, norm, pots, cfg, seed=a.seed, device="cpu", crop_tiles=crop)
        stored = run["seeds"][str(a.seed)]["graphs"]
        eq = [same(G, g) for G, g in zip(stored, gs)]
        out[lab] = {"run_id": run["id"], "spec": spec, "tiles": len(eq), "identical": int(sum(eq)),
                    "mismatched_sids": [s for s, e in zip(sids, eq) if not e], "seconds": round(time.time() - t0, 1)}
        print(lab, out[lab]["identical"], "/", len(eq), "identical", flush=True)
    (OUT / a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
