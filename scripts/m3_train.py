"""Milestone 3: train the VAE (or the deterministic AE baseline) from a config.

    python scripts/m3_train.py --config configs/m3_vae.toml                 # VAE
    python scripts/m3_train.py --config configs/m3_vae.toml --ae            # AE baseline (same net, no KL)
    python scripts/m3_train.py --config configs/m3_vae.toml --smoke         # 2-epoch smoke test
    python scripts/m3_train.py --config configs/m3_vae.toml --epochs 60     # short run

Checkpoints: checkpoints/m3/<name>/best.pt (git-ignored); history: outputs/m3/runs/<name>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.models.train import load_config, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3_vae.toml")
    ap.add_argument("--ae", action="store_true", help="train the deterministic autoencoder baseline")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, name suffix _smoke")
    ap.add_argument("--epochs", type=int, help="override max_epochs")
    ap.add_argument("--name", help="override experiment name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.ae:
        cfg["model"]["variational"] = False
        cfg["experiment"]["name"] = cfg["experiment"]["name"].replace("vae", "ae")
    if args.name:
        cfg["experiment"]["name"] = args.name
    epochs = args.epochs
    if args.smoke:
        cfg["experiment"]["name"] += "_smoke"
        epochs = 2
    if epochs:
        cfg["train"]["max_epochs"] = epochs
    s = train(cfg)
    print(f"\nbest val loss {s['best_val_loss']:.1f} at epoch {s['best_epoch']} / {s['epochs_run']} "
          f"({s['train_seconds']} s on {s['device']}); {s['n_params']:,} params; pos_weight {[round(w, 2) for w in s['pos_weight']]}")
    print(f"checkpoint {s['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
