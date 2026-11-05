"""Milestone 13: one traced M* run on the steepest DEV-B tile, for the method figures (figs 2-6 of m13_report.py).

    python scripts/m13_trace.py      # -> outputs/m13/trace.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import m13_experiments as ex  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.sequence import crops_turtle, heading_of
from urbanmorph.graphs.tc_potentials import Potentials
from urbanmorph.graphs.tc_sampler import TCSampler, preset

OUT = PROJECT_ROOT / "outputs" / "m13"


def main() -> int:
    sel = json.loads((OUT / "selection.json").read_text())
    ms = sel["M_star"]
    ov = dict(ms["overrides"])
    if not ov.get("horizon"):                                       # the figure shows look-ahead even if M* has none
        ov.update(horizon=sel.get("H_star") or 2, rollout_m=4)
    pre = ms["preset"] if ms["preset"] != "energy" else "energy_la"
    tiles = ex.tile_set("devB")
    real = json.loads((OUT / "real" / "devB.json").read_text())
    k = int(np.argmax([r["slope_mean_ctx"] for r in real]))
    t = tiles[k]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, names, norm, mcfg = ex.load_backbone("TN_s0", device)
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    cfg = preset(pre if pre in ("energy_la", "smc_la") else "energy_la", **{k2: v for k2, v in ov.items() if k2 != "particles"}, particles=1)
    sm = TCSampler(model, [t["d"]], names, norm, pots, cfg, device)
    sm.trace = []
    sm.run(seed=1)
    s = sm.states[0][0]
    u = next(i for i in range(1, s.n) if s.parent[i] >= 0)
    hd, _ = heading_of(s.nodes, u, s.parent[u])
    crop = crops_turtle(s.edge_map, s.created_map, s.popped_map, s.nodes[u], hd, [s.t])[0].astype(float)
    tcrop = sm.pol.crop_extra(0, s.nodes[u], hd).astype(float)
    (OUT / "trace.json").write_text(json.dumps({"sid": t["sid"], "config": cfg.__dict__, "decisions": sm.trace,
                                                "crop": np.concatenate([crop, tcrop]).tolist(),
                                                "tile": {"elev": t["d"]["elev"].tolist(), "water": t["d"]["water"].tolist()}}, default=float))
    print("traced", len(sm.trace), "decisions on", t["sid"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
