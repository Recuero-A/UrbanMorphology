"""Object placer: buildings and green spaces around a street network (Milestone 6, Stages C and D).

Autoregressive over objects. For each step:
  * WHERE: a small U-Net over the 128 x 128 canvas (+ 2 coordinate channels) gives a logit per
    4 m pixel; the anchor is uniform inside the chosen pixel. A STOP logit (end of the phase) comes
    from the pooled bottleneck.
  * WHAT: from a 32 x 32 crop rotated to the nearest street, the U-Net feature at the chosen pixel and
    10 scalars, an MLP predicts
      - orientation phi relative to the street: mixture of 6 axial von Mises (period pi);
      - size (log length, log width): mixture of 6 bivariate Gaussians;
      - shape (20 ray fractions, logit scale): mixture of 4 diagonal Gaussians.
Loss = exact negative log-likelihood of this parameterisation (location in metres^2, angle in
radians, size and shape in their log / logit units).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.layout.placer_data import CROP, N, N_SCAL
from urbanmorph.models.graph_gen import mixture_log_prob, mixture_params, mixture_sample

PIXEL_AREA_M2 = 16.0
N_SHAPE = 20


def _block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())


class ObjectPlacer(nn.Module):
    def __init__(self, width: int = 32, hidden: int = 256, k_angle: int = 6, k_size: int = 6, k_shape: int = 4,
                 neighbours: int = 0, shape_lib: int = 0):
        super().__init__()
        self.neighbours = neighbours            # Milestone 8 B2: > 0 adds a set encoder over nearby placed buildings
        self.shape_lib = shape_lib              # Milestone 8 S: > 0 = building shapes are one of this many library classes
        w = width
        self.ka, self.ks, self.kh = k_angle, k_size, k_shape
        self.e1, self.e2, self.e3, self.e4 = _block(6, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = _block(8 * w, 2 * w), _block(4 * w, w), _block(2 * w, w)
        self.loc = nn.Conv2d(w, 1, 1)
        self.stop = nn.Sequential(nn.Linear(8 * w + 3, 128), nn.SiLU(), nn.Linear(128, 1))
        self.crop_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),          # 16
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),          # 8
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),          # 4
            nn.Flatten(), nn.Linear(64 * 16, hidden), nn.SiLU())
        nb_dim = 64 if neighbours else 0
        if neighbours:
            from urbanmorph.layout.placer_data import NB_FEAT
            self.nb_mlp = nn.Sequential(nn.Linear(NB_FEAT, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.attr = nn.Sequential(nn.Linear(hidden + w + N_SCAL + nb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.angle_head = nn.Linear(hidden, 4 * k_angle)
        self.size_head = nn.Linear(hidden + 1, 6 * k_size)
        self.shape_head = nn.Linear(hidden + 2, k_shape * (1 + 2 * N_SHAPE))        # green (and buildings without a library)
        if shape_lib:
            self.lib_head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.SiLU(), nn.Linear(hidden, shape_lib))
            self.register_buffer("lib_logits", torch.zeros(shape_lib, N_SHAPE))   # set from the library before training
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])

    # ---------------------------------------------------------------- where
    def unet(self, canvas):
        x = torch.cat([canvas.float(), self.coords.expand(len(canvas), -1, -1, -1)], 1)
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool2d(h1, 2))
        h3 = self.e3(F.avg_pool2d(h2, 2))
        h4 = self.e4(F.avg_pool2d(h3, 2))
        u3 = self.d3(torch.cat([F.interpolate(h4, scale_factor=2), h3], 1))
        u2 = self.d2(torch.cat([F.interpolate(u3, scale_factor=2), h2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, scale_factor=2), h1], 1))
        pooled = torch.cat([h4.mean((2, 3)), h4.amax((2, 3))], 1)
        return self.loc(u1).flatten(1), pooled, u1

    def where(self, canvas, phase, counts):
        loc, pooled, feat = self.unet(canvas)
        stop = self.stop(torch.cat([pooled, phase[:, None], counts], 1)).squeeze(1)
        return stop, loc, feat

    # ---------------------------------------------------------------- what
    def what(self, crops, feat_at, scal, nb=None, nb_mask=None):
        parts = [self.crop_cnn(crops.float()), feat_at, scal]
        if self.neighbours:
            parts.append((self.nb_mlp(nb) * nb_mask.unsqueeze(-1).float()).sum(1))
        h = self.attr(torch.cat(parts, 1))
        a = self.angle_head(h).view(-1, 4, self.ka)
        angle = (a[:, 0], torch.atan2(a[:, 1], a[:, 2] + 1e-6), F.softplus(a[:, 3]) + 1e-3)   # logits, mu (of 2 phi), kappa
        return h, angle

    def size_params(self, h, phase):
        return mixture_params(self.size_head(torch.cat([h, phase[:, None]], 1)), self.ks)

    def shape_params(self, h, phase, size):
        o = self.shape_head(torch.cat([h, phase[:, None], size[:, :1] - size[:, 1:]], 1)).view(-1, self.kh, 1 + 2 * N_SHAPE)
        return o[..., 0], o[..., 1:1 + N_SHAPE], o[..., 1 + N_SHAPE:].clamp(-5, 3)

    @staticmethod
    def angle_log_prob(phi, logits, mu, kappa):
        """Axial von Mises mixture on phi in [-pi/2, pi/2): density of 2 phi, times 2 (change of variables)."""
        x = 2 * phi[:, None]
        lp = kappa * torch.cos(x - mu) - math.log(2 * math.pi) - (torch.log(torch.special.i0e(kappa)) + kappa)
        return torch.logsumexp(F.log_softmax(logits, -1) + lp, -1) + math.log(2.0)

    @staticmethod
    def shape_log_prob(s, logits, mu, log_s):
        comp = (-0.5 * ((s[:, None] - mu) / log_s.exp()) ** 2 - log_s - 0.5 * math.log(2 * math.pi)).sum(-1)
        return torch.logsumexp(F.log_softmax(logits, -1) + comp, -1)

    def nll(self, b) -> dict:
        """Per-sample NLL terms (stop for every step; where + what only for object steps)."""
        stop_logit, loc, feat = self.where(b["canvas"], b["phase"], b["counts"])
        is_obj = 1 - b["stop"]
        nll_stop = F.binary_cross_entropy_with_logits(stop_logit, b["stop"], reduction="none")
        nll_loc = (F.cross_entropy(loc, b["cell"], reduction="none") + math.log(PIXEL_AREA_M2)) * is_obj
        feat_at = feat.flatten(2).gather(2, b["cell"][:, None, None].expand(-1, feat.shape[1], 1)).squeeze(2)
        h, (al, am, ak) = self.what(b["crops"], feat_at, b["scalars"], b.get("nb"), b.get("nb_mask"))
        nll_angle = -self.angle_log_prob(b["phi"], al, am, ak) * is_obj
        sp = self.size_params(h, b["phase"])
        nll_size = -mixture_log_prob(b["size"], *sp) * is_obj
        nll_shape = -self.shape_log_prob(b["shape"], *self.shape_params(h, b["phase"], b["size"]))
        if self.shape_lib:
            ce = F.cross_entropy(self.lib_head(torch.cat([h, b["size"]], 1)), b["shape_cls"], reduction="none")
            nll_shape = torch.where(b["phase"] < 0.5, ce, nll_shape)
        nll_shape = nll_shape * is_obj
        return {"stop": nll_stop, "location": nll_loc, "angle": nll_angle, "size": nll_size, "shape": nll_shape}

    # ---------------------------------------------------------------- sampling helpers
    @torch.no_grad()
    def sample_what(self, crops, feat_at, scal, phase, generator=None, nb=None, nb_mask=None):
        h, (al, am, ak) = self.what(crops, feat_at, scal, nb, nb_mask)
        k = torch.multinomial(F.softmax(al, -1).cpu(), 1, generator=generator).squeeze(1).to(al.device)
        i = torch.arange(len(k), device=al.device)
        vm = torch.distributions.VonMises(am[i, k].cpu(), ak[i, k].cpu())
        phi = (vm.sample() / 2)
        phi = (phi + math.pi / 2) % math.pi - math.pi / 2
        size = mixture_sample(*self.size_params(h, phase), generator=generator)
        sl, sm, ss = self.shape_params(h, phase, size)
        k2 = torch.multinomial(F.softmax(sl, -1).cpu(), 1, generator=generator).squeeze(1).to(sl.device)
        eps = torch.randn(len(k2), N_SHAPE, generator=generator).to(sl.device)
        shape = sm[i, k2] + ss[i, k2].exp() * eps
        if self.shape_lib:
            c = torch.multinomial(F.softmax(self.lib_head(torch.cat([h, size], 1)), -1).cpu(), 1, generator=generator).squeeze(1)
            shape = torch.where(phase[:, None] < 0.5, self.lib_logits[c.to(sl.device)], shape)
        return phi.numpy(), size.cpu().numpy(), shape.cpu().numpy()
