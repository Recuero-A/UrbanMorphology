"""Frontage model: the next building along a street side, relative to the previous ones (Milestone 7).

Per step, from a street-aligned 32 x 32 crop and 20 scalars (layout/frontage_data.py):
  * STOP logit (no more buildings on this frontage);
  * (log(gap + 1 m), log setback): mixture of 8 bivariate Gaussians, so spacing and setback are learned
    jointly and relative to the previous building (rows, rhythm, building lines);
  * orientation relative to the street: mixture of 6 axial von Mises, conditioned on gap / setback;
  * log size: mixture of 6 bivariate Gaussians, conditioned on gap / setback;
  * shape: mixture of 4 diagonal Gaussians over the 20 ray logits, conditioned on size.
Exact negative log-likelihood of this parameterisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.layout.frontage_data import N_FSCAL
from urbanmorph.models.graph_gen import mixture_log_prob, mixture_params, mixture_sample
from urbanmorph.models.placer import N_SHAPE, ObjectPlacer


class FrontageNet(nn.Module):
    def __init__(self, hidden: int = 256, k_gap: int = 8, k_angle: int = 6, k_size: int = 6, k_shape: int = 4):
        super().__init__()
        self.kg, self.ka, self.ks, self.kh = k_gap, k_angle, k_size, k_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 16, hidden), nn.SiLU())
        self.mlp = nn.Sequential(nn.Linear(hidden + N_FSCAL, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.stop = nn.Linear(hidden, 1)
        self.gap_head = nn.Linear(hidden, 6 * k_gap)
        self.angle_head = nn.Linear(hidden + 2, 4 * k_angle)
        self.size_head = nn.Linear(hidden + 2, 6 * k_size)
        self.shape_head = nn.Linear(hidden + 1, k_shape * (1 + 2 * N_SHAPE))

    def trunk(self, crops, scal):
        return self.mlp(torch.cat([self.cnn(crops.float()), scal], 1))

    def angle_params(self, h, gapd):
        a = self.angle_head(torch.cat([h, gapd], 1)).view(-1, 4, self.ka)
        return a[:, 0], torch.atan2(a[:, 1], a[:, 2] + 1e-6), F.softplus(a[:, 3]) + 1e-3

    def size_params(self, h, gapd):
        return mixture_params(self.size_head(torch.cat([h, gapd], 1)), self.ks)

    def shape_params(self, h, size):
        o = self.shape_head(torch.cat([h, size[:, :1] - size[:, 1:]], 1)).view(-1, self.kh, 1 + 2 * N_SHAPE)
        return o[..., 0], o[..., 1:1 + N_SHAPE], o[..., 1 + N_SHAPE:].clamp(-5, 3)

    def nll(self, b) -> dict:
        h = self.trunk(b["crops"], b["scalars"])
        obj = 1 - b["stop"]
        gp = mixture_params(self.gap_head(h), self.kg)
        return {"stop": F.binary_cross_entropy_with_logits(self.stop(h).squeeze(1), b["stop"], reduction="none"),
                "gap_setback": -mixture_log_prob(b["gapd"], *gp) * obj,
                "angle": -ObjectPlacer.angle_log_prob(b["phi"], *self.angle_params(h, b["gapd"])) * obj,
                "size": -mixture_log_prob(b["size"], *self.size_params(h, b["gapd"])) * obj,
                "shape": -ObjectPlacer.shape_log_prob(b["shape"], *self.shape_params(h, b["size"])) * obj}

    @torch.no_grad()
    def sample(self, crops, scal, generator=None):
        import math
        h = self.trunk(crops, scal)
        stop = (torch.rand(len(h), generator=generator) < torch.sigmoid(self.stop(h).squeeze(1)).cpu()).numpy()
        gapd = mixture_sample(*mixture_params(self.gap_head(h), self.kg), generator=generator)
        al, am, ak = self.angle_params(h, gapd)
        k = torch.multinomial(F.softmax(al, -1).cpu(), 1, generator=generator).squeeze(1).to(h.device)
        i = torch.arange(len(k), device=h.device)
        phi = torch.distributions.VonMises(am[i, k].cpu(), ak[i, k].cpu()).sample() / 2
        phi = (phi + math.pi / 2) % math.pi - math.pi / 2
        size = mixture_sample(*self.size_params(h, gapd), generator=generator)
        sl, sm, ss = self.shape_params(h, size)
        k2 = torch.multinomial(F.softmax(sl, -1).cpu(), 1, generator=generator).squeeze(1).to(h.device)
        shape = sm[i, k2] + ss[i, k2].exp() * torch.randn(len(k2), N_SHAPE, generator=generator).to(h.device)
        return stop, gapd.cpu().numpy(), phi.numpy(), size.cpu().numpy(), shape.cpu().numpy()


class FrontageNetEdge(FrontageNet):
    """Milestone 8 B1: same network; 'gapd' now means (asinh(edge gap / 2 m), log(front distance + 1 m)),
    plus a Bernoulli head 'the footprint is exactly its minimum rotated rectangle' (the shape density is
    then only used for non-rectangular footprints)."""

    def __init__(self, hidden: int = 256, **kw):
        super().__init__(hidden, **kw)
        self.rect_head = nn.Linear(hidden + 2, 1)

    def rect_logit(self, h, size):
        return self.rect_head(torch.cat([h, size], 1)).squeeze(1)

    def nll(self, b) -> dict:
        h = self.trunk(b["crops"], b["scalars"])
        obj = 1 - b["stop"]
        gp = mixture_params(self.gap_head(h), self.kg)
        rect = b["rect"]
        return {"stop": F.binary_cross_entropy_with_logits(self.stop(h).squeeze(1), b["stop"], reduction="none"),
                "gap_front": -mixture_log_prob(b["gapd"], *gp) * obj,
                "angle": -ObjectPlacer.angle_log_prob(b["phi"], *self.angle_params(h, b["gapd"])) * obj,
                "size": -mixture_log_prob(b["size"], *self.size_params(h, b["gapd"])) * obj,
                "rect": F.binary_cross_entropy_with_logits(self.rect_logit(h, b["size"]), rect, reduction="none") * obj,
                "shape": -ObjectPlacer.shape_log_prob(b["shape"], *self.shape_params(h, b["size"])) * obj * (1 - rect)}

    @torch.no_grad()
    def sample(self, crops, scal, generator=None):
        stop, gapd, phi, size, shape = super().sample(crops, scal, generator)
        h = self.trunk(crops, scal)
        rect = (torch.rand(len(h), generator=generator) <
                torch.sigmoid(self.rect_logit(h, torch.from_numpy(size).to(h.device))).cpu()).numpy()
        shape = np.where(rect[:, None], 10.0, shape)        # logit 10 -> ray fraction 1: exactly the rectangle
        return stop, gapd, phi, size, shape
