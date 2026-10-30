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


class SpatialVAE(nn.Module):
    """VAE whose latent is a (C, 8, 8) feature map instead of one global vector (Milestone 3b).

    Differences from ``ConvVAE``:
    * The encoder trunk is identical, but instead of flattening (256, 8, 8) into a
      vector, 1x1 convolutions produce mu and log sigma^2 *per position*, so each
      latent position describes one 64 m x 64 m patch (with a larger receptive field).
    * No large fully connected layers: geometry does not have to pass through 32 numbers.
    * Prior: N(0, I) over all C*8*8 elements, i.e. FACTORISED over positions. The prior
      therefore knows nothing about how neighbouring patches fit together; samples can
      be locally plausible but globally incoherent. A learned spatial prior would be needed
      to fix that.
    """

    def __init__(self, latent_channels: int = 16, widths=(32, 64, 128, 256)):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_shape = (latent_channels, 8, 8)
        self.latent_dim = latent_channels * 64
        self.variational = True
        chans = (3, *widths)
        self.encoder_features = nn.Sequential(*[_down(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.to_mu = nn.Conv2d(widths[-1], latent_channels, 1)
        self.to_logvar = nn.Conv2d(widths[-1], latent_channels, 1)
        up = tuple(reversed(widths))
        self.from_z = nn.Sequential(nn.Conv2d(latent_channels, up[0], 3, padding=1), nn.GroupNorm(8, up[0]), nn.SiLU())
        self.blocks = nn.Sequential(*[_up(up[i], up[min(i + 1, len(up) - 1)]) for i in range(len(up))])
        self.out = nn.Conv2d(up[-1], 3, kernel_size=1)

    def encode(self, x):
        h = self.encoder_features(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        return self.out(self.blocks(self.from_z(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar) if self.training else mu
        # return flattened mu/logvar so the shared losses (sum over dims) apply unchanged
        return self.decode(z), mu.flatten(1), logvar.flatten(1)

    @property
    def decoder(self):  # same call signature as ConvVAE.decoder: flat z -> logits
        return lambda z: self.decode(z.view(-1, *self.latent_shape))

    @torch.no_grad()
    def sample(self, n: int, device=None, generator: torch.Generator | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n, *self.latent_shape, generator=generator).to(device)
        return torch.sigmoid(self.decode(z))
