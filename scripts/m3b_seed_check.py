"""Milestone 3b: repeat selected experiments with another seed to estimate run-to-run variation.

    python scripts/m3b_seed_check.py B_spatial16_exp C2_spatial16_roadx3 --seed 1
Trains <name>_seed<k> with the identical config except the seed, then evaluates with m3b_evaluate.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3b_experiments import merged_configs  # noqa: E402
from urbanmorph.models.train import CHECKPOINT_DIR, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfgs = {c["experiment"]["name"]: c for c in merged_configs()}
    out = []
    for n in a.names:
        cfg = cfgs[n]
        cfg["experiment"]["seed"] = a.seed
        cfg["experiment"]["name"] = f"{n}_seed{a.seed}"
        if not (CHECKPOINT_DIR / cfg["experiment"]["name"] / "best.pt").exists():
            s = train(cfg, quiet=True)
            print(f"{cfg['experiment']['name']}: best epoch {s['best_epoch']}, {s['train_seconds']} s", flush=True)
        out.append(cfg["experiment"]["name"])
    return subprocess.call([sys.executable, str(Path(__file__).parent / "m3b_evaluate.py"), *out])


if __name__ == "__main__":
    sys.exit(main())
