"""Milestone 3b: run the controlled experiments sequentially, then evaluate all of them.

    python scripts/m3b_experiments.py              # train A, B, C1, C2 (skips runs that already have a checkpoint)
    python scripts/m3b_experiments.py --only B_spatial16_exp
    python scripts/m3b_experiments.py --dry-run    # print the merged configs

Configs: configs/m3b_experiments.toml (overrides of configs/m3_vae.toml).
The M3 baseline (vae_z32_b0.5) is NOT retrained; it is re-evaluated with the same protocol.
"""

import argparse
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models.train import CHECKPOINT_DIR, load_config, train

EXPERIMENTS = PROJECT_ROOT / "configs" / "m3b_experiments.toml"
BASELINE = "vae_z32_b0.5"


def merged_configs() -> list[dict]:
    spec = tomllib.loads(EXPERIMENTS.read_text())
    base = load_config(PROJECT_ROOT / spec["base_config"])
    out = []
    for exp in spec["experiments"]:
        cfg = copy.deepcopy(base)
        for section in (spec.get("common", {}), exp):
            for key, value in section.items():
                if isinstance(value, dict):
                    cfg.setdefault(key, {}).update(value)
        cfg["experiment"]["name"] = exp["name"]
        cfg["experiment"]["description"] = exp["description"]
        cfg["_experiments_file"] = str(EXPERIMENTS.relative_to(PROJECT_ROOT))
        out.append(cfg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    cfgs = [c for c in merged_configs() if not args.only or c["experiment"]["name"] == args.only]
    for cfg in cfgs:
        name = cfg["experiment"]["name"]
        if args.dry_run:
            print(json.dumps({k: cfg[k] for k in ("experiment", "data", "model", "loss")}, indent=1))
            continue
        if (CHECKPOINT_DIR / name / "best.pt").exists():
            print(f"=== {name}: checkpoint exists, skipping training", flush=True)
            continue
        print(f"=== {name}: {cfg['experiment']['description']}", flush=True)
        s = train(cfg, quiet=True)
        print(f"    best epoch {s['best_epoch']}/{s['epochs_run']}, {s['train_seconds']} s, {s['n_params']:,} params, "
              f"{s['n_train']} train tiles", flush=True)
    if args.dry_run or args.no_eval:
        return 0
    names = [BASELINE] + [c["experiment"]["name"] for c in merged_configs()]
    return subprocess.call([sys.executable, str(PROJECT_ROOT / "scripts" / "m3b_evaluate.py"), *names])


if __name__ == "__main__":
    sys.exit(main())
