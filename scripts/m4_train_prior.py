"""Milestone 4: train the Gated PixelCNN prior on codes of the FROZEN C2 encoder.

    python scripts/m4_train_prior.py --seed 0

Training data: q(z|x) of 565 training tiles x 8 symmetries (4,520 grids). Every epoch draws a
fresh z = mu + sigma * eps, so the prior learns the aggregate posterior the decoder was trained
on. Validation: 19 validation tiles x 8 symmetries with FIXED noise; used only for early stopping.
The VAE is never updated; reconstruction losses are untouched.
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.models.data import LayoutDataset, dihedral
from urbanmorph.models.prior import GatedPixelCNNPrior
from urbanmorph.models.train import load_checkpoint, pick_device, set_seed
from urbanmorph.models.vae import count_parameters

CFG = PROJECT_ROOT / "configs" / "m4_prior.toml"
CKPT = PROJECT_ROOT / "checkpoints" / "m4"
RUNS = PROJECT_ROOT / "outputs" / "m4" / "runs"


@torch.no_grad()
def encode_all(vae, x, device, batch=128):
    mus, lvs = [], []
    for k in range(8):
        xk = dihedral(x, k)
        for i in range(0, len(xk), batch):
            mu, lv = vae.encode(xk[i:i + batch].to(device))
            mus.append(mu.cpu())
            lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def load_codes(device):
    cfg = tomllib.loads(CFG.read_text())
    vae, _ = load_checkpoint(PROJECT_ROOT / cfg["base"]["vae_checkpoint"], device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    train = LayoutDataset("train", index=load_index_with_expansion())
    val = LayoutDataset("val")
    return cfg, vae, encode_all(vae, train.x, device), encode_all(vae, val.x, device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = pick_device("auto")
    set_seed(a.seed)
    cfg, vae, (mu_tr, lv_tr), (mu_va, lv_va) = load_codes(device)
    pc, tc = cfg["prior"], cfg["train"]
    prior = GatedPixelCNNPrior(latent_channels=mu_tr.shape[1], hidden=pc["hidden"], n_layers=pc["n_layers"],
                               n_mix=pc["n_mix"]).to(device)
    prior.set_normalisation(mu_tr.mean(dim=(0, 2, 3)).to(device), mu_tr.std(dim=(0, 2, 3)).to(device))
    g = torch.Generator().manual_seed(1000)
    z_va = (mu_va + (0.5 * lv_va).exp() * torch.randn(mu_va.shape, generator=g)).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=tc["lr"])
    n, D = len(mu_tr), mu_tr[0].numel()
    best, best_state, waited, hist, t0 = float("inf"), None, 0, [], time.time()
    gen = torch.Generator().manual_seed(a.seed)
    for epoch in range(1, tc["max_epochs"] + 1):
        prior.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, tc["batch_size"]):
            idx = perm[i:i + tc["batch_size"]]
            z = (mu_tr[idx] + (0.5 * lv_tr[idx]).exp() * torch.randn(mu_tr[idx].shape, generator=gen)).to(device)
            loss = -prior.log_prob(z).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        prior.eval()
        with torch.no_grad():
            va = -prior.log_prob(z_va).mean().item()
        hist.append({"epoch": epoch, "train_nll_per_dim": tot / n / D, "val_nll_per_dim": va / D,
                     "seconds": round(time.time() - t0, 1)})
        if va < best:
            best, waited = va, 0
            best_state = {k: v.detach().clone() for k, v in prior.state_dict().items()}
        else:
            waited += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"ep {epoch:3d} train {tot / n / D:+.3f} val {va / D:+.3f} nats/dim", flush=True)
        if waited >= tc["patience"]:
            break
    prior.load_state_dict(best_state)
    name = f"prior_pixelcnn_s{a.seed}"
    (CKPT / name).mkdir(parents=True, exist_ok=True)
    torch.save({"state": prior.state_dict(), "config": cfg, "seed": a.seed}, CKPT / name / "prior.pt")
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    (RUNS / name / "history.json").write_text(json.dumps(hist, indent=1))
    best_ep = min(hist, key=lambda r: r["val_nll_per_dim"])["epoch"]
    summary = {"name": name, "params": count_parameters(prior), "best_epoch": best_ep, "epochs_run": len(hist),
               "best_val_nll_per_dim": best / D, "train_seconds": round(time.time() - t0, 1),
               "n_train_grids": n, "n_val_grids": len(mu_va), "device": str(device)}
    (RUNS / name / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
