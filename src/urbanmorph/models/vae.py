"""Small convolutional VAE (and its deterministic autoencoder twin) for (3, 128, 128) layouts.

Encoder:  128 -> 64 -> 32 -> 16 -> 8 spatial resolution with four stride-2
          conv blocks (32, 64, 128, 256 channels), then two linear heads for
          the posterior mean mu and log-variance log sigma^2.
Decoder:  linear -> (256, 8, 8), four blocks of nearest-neighbour upsampling
          + 3x3 convolution (avoids the checkerboard artefacts of transposed
          convolutions), then a 1x1 conv to 3 channels of LOGITS. Probabilities
          are sigmoid(logits); the loss uses the logits directly because
          BCE-with-logits is numerically more stable than BCE on probabilities.
Norm:     GroupNorm (independent of batch size; batches here are small).
"""

from __future__ import annotations

import torch
from torch import nn


def _down(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, widths=(32, 64, 128, 256), variational: bool = True):
        super().__init__()
        chans = (3, *widths)
        self.features = nn.Sequential(*[_down(chans[i], chans[i + 1]) for i in range(len(widths))])
        flat = widths[-1] * 8 * 8
        self.mu = nn.Linear(flat, latent_dim)
        self.logvar = nn.Linear(flat, latent_dim) if variational else None

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(1)
        mu = self.mu(h)
        logvar = self.logvar(h) if self.logvar is not None else None
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, widths=(256, 128, 64, 32)):
        super().__init__()
        self.widths = widths
        self.fc = nn.Linear(latent_dim, widths[0] * 8 * 8)
        self.blocks = nn.Sequential(*[_up(widths[i], widths[min(i + 1, len(widths) - 1)]) for i in range(len(widths))])
        self.out = nn.Conv2d(widths[-1], 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.widths[0], 8, 8)
        return self.out(self.blocks(h))  # logits, (B, 3, 128, 128)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + sigma * eps, eps ~ N(0, I).

    Sampling z directly is not differentiable w.r.t. mu and sigma; writing it
    as a deterministic function of (mu, sigma) plus external noise eps lets
    gradients flow through mu and sigma (the "reparameterisation trick").
    """
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, variational: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.variational = variational
        self.encoder = Encoder(latent_dim, variational=variational)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        if self.variational and self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu  # deterministic AE, or the posterior mean at evaluation time
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
        """Decode z ~ N(0, I) into probabilities (B, 3, 128, 128)."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, generator=generator).to(device)
        return torch.sigmoid(self.decoder(z))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
