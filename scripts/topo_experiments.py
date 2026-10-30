"""Road-topology experiment: C2 + lambda * soft-clDice (roads only).

    python scripts/topo_experiments.py --lambdas 0.1 0.3 1 3 --seeds 0     # pilot
    python scripts/topo_experiments.py --lambdas 0.3 --seeds 1               # repeat a chosen strength

Every run = the frozen C2 config (configs/m3b_experiments.toml, C2_spatial16_roadx3:
spatial latent 16x8x8, beta 0.5, sqrt pos-weights, road BCE x3, 565 train tiles,
300 epochs, identical seeds) + ONLY loss.cldice_weight = lambda.
Run names: T_cl<lambda>_s<seed>. The baseline (lambda = 0) is the existing C2 checkpoints
C2_spatial16_roadx3 (seed 0) and C2_spatial16_roadx3_seed1 (seed 1); training is
deterministic, so retraining them would reproduce them bit for bit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3b_experiments import merged_configs  # noqa: E402
from urbanmorph.models.train import CHECKPOINT_DIR, train  # noqa: E402

BASE = "C2_spatial16_roadx3"


def run_name(lam: float, seed: int) -> str:
    return f"T_cl{lam:g}_s{seed}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    for seed in a.seeds:
        for lam in a.lambdas:
            cfg = {c["experiment"]["name"]: c for c in merged_configs()}[BASE]
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["name"] = run_name(lam, seed)
            cfg["experiment"]["description"] = f"C2 + {lam:g} x soft-clDice on roads (seed {seed})"
            cfg["loss"]["cldice_weight"] = lam
            if (CHECKPOINT_DIR / cfg["experiment"]["name"] / "best.pt").exists():
                print(f"{cfg['experiment']['name']}: exists, skipped", flush=True)
                continue
            s = train(cfg, quiet=True)
            print(f"{cfg['experiment']['name']}: best epoch {s['best_epoch']}/{s['epochs_run']}, {s['train_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
