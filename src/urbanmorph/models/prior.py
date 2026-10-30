"""Learned autoregressive prior over the continuous (16, 8, 8) latent grid of the frozen C2 VAE (Milestone 4).

Probability model
-----------------
Raster order over the 64 positions (row by row, left to right):

    p(z) = prod_{i=1..64} p(z_i | z_<i),     z_i in R^16 (the latent vector at position i)

Each conditional is a mixture of K diagonal Gaussians over the 16-dim vector,

    p(z_i | z_<i) = sum_k pi_k * prod_d N(z_id ; m_kd, s_kd^2),

whose parameters (pi, m, s) are predicted from z_<i. This is a valid CONTINUOUS density,
so the loss is the exact negative log-likelihood. (A categorical PixelCNN softmax would only
be valid for discrete codes, as in VQ-VAE.)

Network: Gated PixelCNN (van den Oord et al., 2016) with a vertical stack (rows above) and a
horizontal stack (same row, to the left), which removes the "blind spot" of plain masked
convolutions. With 6 layers, vertical kernel (2 x 5) and horizontal kernel (1 x 2), every
position sees ALL 63 earlier positions of an 8 x 8 grid and none of the later ones (tested
with the Jacobian in tests/test_prior.py).

Codes are standardised per channel (training statistics) before modelling.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def down_shift(x: torch.Tensor) -> torch.Tensor:
    """Row i receives row i-1 (row 0 receives zeros)."""
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


class GatedLayer(nn.Module):
    def __init__(self, c_in: int, c: int, first: bool, kv: int = 5):
        super().__init__()
        self.first = first
        self.kv = kv
        self.v_conv = nn.Conv2d(c_in, 2 * c, kernel_size=(2, kv))       # rows i-1, i (padded on top)
        self.h_conv = nn.Conv2d(c_in, 2 * c, kernel_size=(1, 2))        # cols j-1, j (or j-2, j-1 for the first layer)
        self.v_to_h = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        self.h_out = nn.Conv2d(c, c, kernel_size=1)
        self.res = (c_in == c) and not first

    @staticmethod
    def gate(x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, v, h):
        p = self.kv // 2
        v_feat = self.v_conv(F.pad(v, (p, p, 1, 0)))                   # depends on rows <= i
        if self.first:
            h_feat = self.h_conv(F.pad(h, (2, 0, 0, 0)))[:, :, :, :-1]   # cols j-2, j-1: excludes the current position
        else:
            h_feat = self.h_conv(F.pad(h, (1, 0, 0, 0)))                 # cols j-1, j
        h_feat = h_feat + self.v_to_h(down_shift(v_feat))               # rows < i only reach h via the shift
        v_new = self.gate(v_feat)
        h_new = self.h_out(self.gate(h_feat))
        if self.res:
            h_new = h_new + h
        return v_new, h_new


class GatedPixelCNNPrior(nn.Module):
    def __init__(self, latent_channels: int = 16, hidden: int = 32, n_layers: int = 6, n_mix: int = 5):
        super().__init__()
        self.D, self.K = latent_channels, n_mix
        self.layers = nn.ModuleList([GatedLayer(latent_channels if i == 0 else hidden, hidden, first=(i == 0))
                                     for i in range(n_layers)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, n_mix * (1 + 2 * latent_channels), 1))
        self.register_buffer("mean", torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, latent_channels, 1, 1))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.view(1, -1, 1, 1))
        self.std.copy_(std.view(1, -1, 1, 1))

    def params(self, z_norm: torch.Tensor):
        """Mixture parameters for every position, each computed only from earlier positions."""
        v = h = z_norm
        for layer in self.layers:
            v, h = layer(v, h)
        out = self.head(h)  # (B, K*(1+2D), H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.K, 1 + 2 * self.D, H, W)
        logit_pi = out[:, :, 0]                                   # (B, K, H, W)
        mu = out[:, :, 1:1 + self.D]                              # (B, K, D, H, W)
        log_s = out[:, :, 1 + self.D:].clamp(-7.0, 5.0)           # (B, K, D, H, W)
        return logit_pi, mu, log_s

    def log_prob_normalised(self, z_norm: torch.Tensor) -> torch.Tensor:
        """log p(z_norm), summed over positions and channels, per sample (B,)."""
        logit_pi, mu, log_s = self.params(z_norm)
        x = z_norm.unsqueeze(1)                                   # (B, 1, D, H, W)
        comp = (-0.5 * ((x - mu) / log_s.exp()) ** 2 - log_s - 0.5 * math.log(2 * math.pi)).sum(dim=2)  # (B, K, H, W)
        lp = torch.logsumexp(F.log_softmax(logit_pi, dim=1) + comp, dim=1)    # (B, H, W)
        return lp.sum(dim=(1, 2))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z) in the ORIGINAL latent units (includes the Jacobian of the standardisation)."""
        z_norm = (z - self.mean) / self.std
        return self.log_prob_normalised(z_norm) - torch.log(self.std).sum() * z.shape[2] * z.shape[3]

    @torch.no_grad()
    def sample(self, n: int, H: int = 8, W: int = 8, generator: torch.Generator | None = None,
               device=None) -> torch.Tensor:
        """Ancestral sampling in raster order; returns z in original units, (n, D, H, W)."""
        device = device or self.mean.device
        z = torch.zeros(n, self.D, H, W, device=device)
        for i in range(H):
            for j in range(W):
                logit_pi, mu, log_s = self.params(z)
                probs = F.softmax(logit_pi[:, :, i, j], dim=1).cpu()
                k = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(device)
                idx = torch.arange(n, device=device)
                m = mu[idx, k, :, i, j]
                s = log_s[idx, k, :, i, j].exp()
                eps = torch.randn(n, self.D, generator=generator).to(device)
                z[:, :, i, j] = m + s * eps
        return z * self.std + self.mean
