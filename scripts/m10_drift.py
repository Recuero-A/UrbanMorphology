"""Milestone 10: exposure-bias measure D2 (scripts/m10_street_diagnosis.py) for the unconditioned street models.

    python scripts/m10_drift.py A N A_w        # -> outputs/m10/diagnosis/drift.json

Median snap deviation of raw sampled turn angles by pop index, teacher-forced (real prefix, 19 validation tiles)
vs free-running (own prefix, 256 tiles, sampling seed 123), seed 0 of each model. The free-running minus
teacher-forced gap at late pops is the exposure-bias effect. Test tiles are never read.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from m6_build_roads import load_graphs  # noqa: E402
from m10_eval_streets import CK, load_model  # noqa: E402
from m10_street_diagnosis import OUT, T_BINS, snap_dev, teacher_forced  # noqa: E402
from urbanmorph.graphs.generate import sample_graphs

PATHS = {"A": CK / "m6" / "roads_v2_s0", "N": CK / "m10" / "N_noise_s0", "A_w": CK / "m10" / "A_w_s0"}


def drift(model, val) -> dict:
    real_th, tf_th, tf_t = teacher_forced(model, val)
    log = []
    sample_graphs(model, 256, device="cpu", seed=123, step_log=log)
    fr = np.array([(t, st, dx, dy) for _, t, st, dx, dy in log], float)
    fr = fr[fr[:, 1] < 0.5]
    fr_th, fr_t = np.degrees(np.arctan2(fr[:, 3], fr[:, 2])), fr[:, 0]
    out = {}
    for lo, hi in zip(T_BINS[:-1], T_BINS[1:]):
        mr, mf = (tf_t >= lo) & (tf_t < hi), (fr_t >= lo) & (fr_t < hi)
        out[f"pop {lo}-{hi if hi < 10 ** 6 else 'end'}"] = {
            "real": float(np.median(snap_dev(real_th[mr]))) if mr.any() else None,
            "teacher_forced": float(np.median(snap_dev(tf_th[mr]))) if mr.any() else None,
            "free_running": float(np.median(snap_dev(fr_th[mf]))) if mf.any() else None}
    return out


def main() -> int:
    val = [g for _, _, g in load_graphs()["val"]]
    f = OUT / "drift.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for name in sys.argv[1:]:
        res[name] = drift(load_model(PATHS[name]), val)
        print(name, json.dumps(res[name]), flush=True)
    f.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
