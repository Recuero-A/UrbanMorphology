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


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice coefficient per sample and channel, shape (B, 3).

    Dice = (2 * sum(p * t) + eps) / (sum(p) + sum(t) + eps). Unlike BCE it is an
    overlap ratio, so a thin structure is not swamped by the many easy background
    pixels. It does NOT know about connectivity: a road broken into pieces can still
    score a high Dice.
    """
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    return (2 * inter + eps) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)


def combined_reconstruction(logits, target, pos_weight=None, channel_weights=None, dice_weight: float = 0.0,
                            cldice_weight: float = 0.0):
    """Weighted BCE (summed over pixels) + optional Dice term, both per channel.

    ``channel_weights`` multiplies each channel's BCE sum; the Dice term is scaled by
    the number of pixels per channel (H*W) so that ``dice_weight = 1`` makes it the
    same order of magnitude as the BCE sum. Returns (total, per-channel totals).
    """
    total, per_ch = reconstruction_loss(logits, target, pos_weight)
    if channel_weights is not None:
        per_ch = per_ch * torch.as_tensor(channel_weights, dtype=per_ch.dtype, device=per_ch.device)
    if dice_weight:
        hw = target.shape[2] * target.shape[3]
        per_ch = per_ch + dice_weight * hw * (1 - soft_dice(logits, target)).mean(dim=0)
    if cldice_weight:
        # soft-clDice on the ROAD channel only (index 0); buildings and green are unchanged.
        from urbanmorph.models.cldice import soft_cldice_loss
        hw = target.shape[2] * target.shape[3]
        road = cldice_weight * hw * soft_cldice_loss(torch.sigmoid(logits[:, :1]), target[:, :1]).mean()
        per_ch = per_ch + torch.stack([road, road.new_zeros(()), road.new_zeros(())])
    return per_ch.sum(), per_ch
