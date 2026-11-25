"""Training loop with validation, early stopping, checkpoints and history.

Model selection uses the VALIDATION split only; the test split is never read here.
"""

from __future__ import annotations

import copy
import json
import platform
import random
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.losses import beta_at, combined_reconstruction, kl_divergence, pos_weight_from_frequencies
from urbanmorph.models.vae import ConvVAE, SpatialVAE, count_parameters

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "m3"
RUNS_DIR = PROJECT_ROOT / "outputs" / "m3" / "runs"


def load_config(path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@dataclass
class History:
    epochs: list = field(default_factory=list)

    def log(self, **row) -> None:
        self.epochs.append(row)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.epochs, indent=1))


def build_model(cfg: dict):
    m = cfg["model"]
    if m.get("type", "global") == "spatial":
        return SpatialVAE(latent_channels=m["latent_channels"])
    return ConvVAE(latent_dim=m["latent_dim"], variational=m["variational"])


def save_checkpoint(path: Path, model: ConvVAE, cfg: dict, epoch: int, val_loss: float, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss,
                "extra": extra or {}}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[ConvVAE, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def run_epoch(model, loader, device, pos_weight, beta: float, optimizer=None, loss_cfg: dict | None = None) -> dict:
    """One pass; trains if an optimizer is given. Returns mean per-sample losses."""
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "recon_roads": 0.0, "recon_buildings": 0.0, "recon_green": 0.0}
    n = 0
    with torch.set_grad_enabled(train):
        for x, _ in loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            lc = loss_cfg or {}
            recon, per_ch = combined_reconstruction(logits, x, pos_weight, lc.get("channel_weights"), lc.get("dice_weight", 0.0),
                                                    lc.get("cldice_weight", 0.0))
            kl = kl_divergence(mu, logvar) if model.variational else torch.zeros((), device=device)
            loss = recon + beta * kl
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = x.shape[0]
            n += b
            for k, v in (("loss", loss), ("recon", recon), ("kl", kl), ("recon_roads", per_ch[0]),
                         ("recon_buildings", per_ch[1]), ("recon_green", per_ch[2])):
                tot[k] += float(v.detach()) * b
    return {k: v / n for k, v in tot.items()}


def train(cfg: dict, max_epochs: int | None = None, quiet: bool = False) -> dict:
    """Train one model from a config. Returns a summary (paths, best epoch, losses)."""
    name = cfg["experiment"]["name"]
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = pick_device(cfg["train"]["device"])

    train_index = load_index_with_expansion() if cfg["data"].get("include_expansion_m3b") else None
    train_ds = LayoutDataset("train", index=train_index, augment=cfg["data"]["augment"], seed=seed)
    val_ds = LayoutDataset("val", augment=False)  # always the original, unchanged validation split
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    pos_frac = train_ds.pos_fraction()
    pos_weight = pos_weight_from_frequencies(pos_frac, cfg["loss"]["pos_weight"])
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    beta, warm = cfg["loss"]["beta"], cfg["loss"]["kl_warmup_epochs"]
    max_epochs = max_epochs or cfg["train"]["max_epochs"]
    patience = cfg["train"]["patience"]
    ckpt_path = CHECKPOINT_DIR / name / "best.pt"
    hist = History()
    best, best_epoch, best_state, waited = float("inf"), -1, None, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        b = beta_at(epoch, beta, warm) if model.variational else 0.0
        tr = run_epoch(model, train_dl, device, pos_weight, b, opt, cfg["loss"])
        # validation always uses the FULL beta so values are comparable across epochs
        va = run_epoch(model, val_dl, device, pos_weight, beta if model.variational else 0.0, loss_cfg=cfg["loss"])
        hist.log(epoch=epoch, beta=b, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
                 seconds=round(time.time() - t0, 1))
        improved = va["loss"] < best
        if improved:
            best, best_epoch, waited = va["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch > warm:
            waited += 1
        if not quiet and (epoch == 1 or epoch % 10 == 0 or improved and epoch > warm):
            print(f"ep {epoch:4d} beta {b:.2f} | train {tr['loss']:9.1f} (rec {tr['recon']:9.1f} kl {tr['kl']:6.1f}) | "
                  f"val {va['loss']:9.1f} (rec {va['recon']:9.1f} kl {va['kl']:6.1f}){' *' if improved else ''}", flush=True)
        if waited >= patience:
            if not quiet:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    meta = {"pos_fraction_train": pos_frac.tolist(), "pos_weight": pos_weight.flatten().tolist(),
            "n_params": count_parameters(model), "device": str(device), "torch": torch.__version__,
            "python": platform.python_version(), "n_train": len(train_ds), "n_val": len(val_ds),
            "epochs_run": len(hist.epochs), "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1)}
    save_checkpoint(ckpt_path, model, cfg, best_epoch, best, meta)
    run_dir = RUNS_DIR / name
    hist.to_json(run_dir / "history.json")
    (run_dir / "summary.json").write_text(json.dumps({"config": cfg, **meta, "best_val_loss": best,
                                                      "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT))}, indent=2))
    return {"checkpoint": ckpt_path, "run_dir": run_dir, "best_val_loss": best, **meta}
