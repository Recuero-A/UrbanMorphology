"""Milestone 3: small validation-only sweep (latent size x beta) plus AE baselines.

    python scripts/m3_sweep.py

Runs VAE latent {16, 32} x beta {0.1, 0.5, 1.0} and AE latent {16, 32}, each up to
max_epochs with early stopping. Model selection uses the VALIDATION split only:
VAEs are compared by validation negative ELBO at beta = 1 (reconstruction + KL),
because each run's own loss depends on its beta and is not comparable across runs.
Writes outputs/m3/sweep.csv and prints the selected VAE.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.losses import kl_divergence
from urbanmorph.models.train import CHECKPOINT_DIR, load_checkpoint, load_config, pick_device, train

OUT = PROJECT_ROOT / "outputs" / "m3"


def runs(base: dict):
    for z in (16, 32):
        for beta in (0.1, 0.5, 1.0):
            cfg = copy.deepcopy(base)
            cfg["model"].update(latent_dim=z, variational=True)
            cfg["loss"]["beta"] = beta
            cfg["experiment"]["name"] = f"vae_z{z}_b{beta}"
            yield cfg
        cfg = copy.deepcopy(base)
        cfg["model"].update(latent_dim=z, variational=False)
        cfg["experiment"]["name"] = f"ae_z{z}"
        yield cfg


def main() -> int:
    base = load_config(PROJECT_ROOT / "configs" / "m3_vae.toml")
    device = pick_device("auto")
    val = LayoutDataset("val")
    rows = []
    for cfg in runs(base):
        name = cfg["experiment"]["name"]
        print(f"=== {name}", flush=True)
        s = train(cfg, quiet=True)
        model, ck = load_checkpoint(CHECKPOINT_DIR / name / "best.pt", device)
        probs, mu = ev.predict(model, val.x, device)
        m = ev.reconstruction_metrics(probs, val.x.numpy(), cfg["eval"]["threshold"])
        hist = pd.read_json(s["run_dir"] / "history.json")
        best = hist.loc[hist["epoch"] == s["best_epoch"]].iloc[0]
        rows.append({
            "name": name, "variational": cfg["model"]["variational"], "latent_dim": cfg["model"]["latent_dim"],
            "beta": cfg["loss"]["beta"] if cfg["model"]["variational"] else 0.0,
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "seconds": s["train_seconds"],
            "train_recon": best["train_recon"], "val_recon": best["val_recon"], "val_kl": best["val_kl"],
            "val_neg_elbo_beta1": best["val_recon"] + best["val_kl"],
            **{f"val_iou_{c}": m[c]["iou"] for c in ev.CHANNELS},
            **{f"val_bce_{c}": m[c]["bce"] for c in ev.CHANNELS},
        })
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sweep.csv", index=False)
    vaes = df[df.variational]
    sel = vaes.loc[vaes["val_neg_elbo_beta1"].idxmin()]
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nSelected VAE (lowest validation -ELBO at beta=1): {sel['name']}; matching AE baseline: ae_z{int(sel['latent_dim'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
