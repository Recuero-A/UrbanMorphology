"""VAE objective: L = reconstruction + beta * KL.

Reconstruction: binary cross-entropy between the decoder's Bernoulli
probabilities and the binary layout, SUMMED over channels and pixels and
averaged over the batch. Summing (not averaging) over the 49,152 pixels keeps
it on the same scale as the KL term, which is a sum over latent dimensions:
together they form the (negative) evidence lower bound, the ELBO.

KL: closed form for a diagonal Gaussian posterior q(z|x) = N(mu, sigma^2)
against the standard normal prior p(z) = N(0, I):
    KL = -1/2 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pos_weight_from_frequencies(pos_fraction, mode: str) -> torch.Tensor:
    """Per-channel weight for positive pixels, shape (3, 1, 1)."""
    p = np.asarray(pos_fraction, dtype=np.float64)
    ratio = (1 - p) / np.clip(p, 1e-6, None)
    if mode == "none":
        w = np.ones_like(p)
    elif mode == "sqrt":
        w = np.sqrt(ratio)
    elif mode == "full":
        w = ratio
    else:
        raise ValueError(mode)
    return torch.tensor(w, dtype=torch.float32).view(-1, 1, 1)


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None):
    """Returns (loss per sample averaged over batch, per-channel sums averaged over batch)."""
    pw = pos_weight.to(logits.device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    per_channel = bce.sum(dim=(2, 3)).mean(dim=0)  # (3,)
    return per_channel.sum(), per_channel


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def beta_at(epoch: int, beta: float, warmup_epochs: int) -> float:
    """Linear KL warm-up: 0 at epoch 0 -> beta after ``warmup_epochs``."""
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)
