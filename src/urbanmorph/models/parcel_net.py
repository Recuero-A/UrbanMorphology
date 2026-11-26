"""Milestone 8, Experiment D: parcel attributes from a mixture-density network (MDN).

Input (per parcel, 10 values): country one-hot (2), nearest street class one-hot (4), standardised log
area of the containing block, interior flag, and the neighbour information available during
sequential generation: mean coverage of the KNOWN neighbours, has-known flag. In training a random
fraction u ~ U(0, 1) of each parcel's neighbours is "known" (as in generation, where parcels are
filled in random order).
Outputs (exact negative log-likelihood):
  * lot (street parcels): (log frontage width, log depth), mixture of 8 bivariate Gaussians;
  * built: Bernoulli (at least one building);
  * coverage | built: logit(coverage) clipped to [0.01, 0.99], mixture of 6 Gaussians;
  * buildings | built: categorical {1, 2, 3+};
  * setbacks (one-building street parcels): front and side = P(< 0.5 m) + mixture of 4 Gaussians on
    log metres; rear = mixture of 4 Gaussians on log(rear + 0.5 m).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.models.graph_gen import mixture_log_prob, mixture_params, mixture_sample

N_IN = 10
ZERO_M = 0.5


def features(country, cls, log_block_area, interior, nb_mean, has_nb) -> np.ndarray:
    country = np.atleast_1d(country)
    n = len(country)
    x = np.zeros((n, N_IN), np.float32)
    x[np.arange(n), (np.asarray(country) == "united_states").astype(int)] = 1
    x[np.arange(n), 2 + np.asarray(cls, int)] = 1
    x[:, 6] = (np.asarray(log_block_area) - 9.3) / 1.1
    x[:, 7] = np.asarray(interior, float)
    x[:, 8] = np.nan_to_num(np.asarray(nb_mean, float))
    x[:, 9] = np.asarray(has_nb, float)
    return x


def mix1_params(o, k):
    return o[:, :k], o[:, k:2 * k], o[:, 2 * k:3 * k].clamp(-5, 3)


def mix1_log_prob(x, logits, mu, log_s):
    comp = -0.5 * ((x[:, None] - mu) / log_s.exp()) ** 2 - log_s - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(F.log_softmax(logits, -1) + comp, -1)


def mix1_sample(logits, mu, log_s, generator=None):
    k = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=generator).squeeze(1)
    i = torch.arange(len(k))
    return (mu.cpu()[i, k] + log_s.cpu()[i, k].exp() * torch.randn(len(k), generator=generator))


class ParcelNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(N_IN, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.lot = nn.Linear(hidden, 6 * 8)
        self.built = nn.Linear(hidden, 1)
        self.cov = nn.Linear(hidden, 3 * 6)
        self.nb = nn.Linear(hidden, 3)
        self.front = nn.Linear(hidden, 1 + 3 * 4)
        self.side = nn.Linear(hidden, 1 + 3 * 4)
        self.rear = nn.Linear(hidden, 3 * 4)

    def _zero_mix(self, o, y, m):
        """P(< ZERO_M) + mixture on log y; y in metres, m = mask of rows with a target."""
        z = (y < ZERO_M).float()
        lp_z = -F.binary_cross_entropy_with_logits(o[:, 0], z, reduction="none")
        lp_c = mix1_log_prob(torch.log(y.clamp(min=ZERO_M)), *mix1_params(o[:, 1:], 4))
        return -(lp_z + (1 - z) * lp_c) * m

    def nll(self, b) -> dict:
        h = self.trunk(b["x"])
        built = b["built"]
        cov = torch.logit(b["coverage"].clamp(0.01, 0.99))
        return {"lot": -mixture_log_prob(b["lot"], *mixture_params(self.lot(h), 8)) * b["m_lot"],
                "built": F.binary_cross_entropy_with_logits(self.built(h).squeeze(1), built, reduction="none"),
                "coverage": -mix1_log_prob(cov, *mix1_params(self.cov(h), 6)) * built,
                "buildings": F.cross_entropy(self.nb(h), b["n_cls"], reduction="none") * built,
                "front": self._zero_mix(self.front(h), b["front"], b["m_sb"]),
                "side": self._zero_mix(self.side(h), b["side"], b["m_side"]),
                "rear": -mix1_log_prob(torch.log(b["rear"] + 0.5), *mix1_params(self.rear(h), 4)) * b["m_rear"]}

    # ---------------------------------------------------------------- sampling (one parcel at a time)
    @torch.no_grad()
    def sample_lot(self, x, g):
        return mixture_sample(*mixture_params(self.lot(self.trunk(x)), 8), generator=g).exp()[0].numpy()

    @torch.no_grad()
    def sample_coverage(self, x, g):
        h = self.trunk(x)
        if torch.rand(1, generator=g).item() >= torch.sigmoid(self.built(h)).item():
            return 0.0, 0
        c = torch.sigmoid(mix1_sample(*mix1_params(self.cov(h), 6), generator=g)).item()
        n = int(torch.multinomial(F.softmax(self.nb(h), -1), 1, generator=g).item()) + 1
        return c, n

    @torch.no_grad()
    def sample_setbacks(self, x, g):
        h = self.trunk(x)
        out = []
        for head in (self.front, self.side, self.side):
            o = head(h)
            if torch.rand(1, generator=g).item() < torch.sigmoid(o[:, 0]).item():
                out.append(0.0)
            else:
                out.append(float(mix1_sample(*mix1_params(o[:, 1:], 4), generator=g).exp().item()))
        out.append(max(float(mix1_sample(*mix1_params(self.rear(h), 4), generator=g).exp().item()) - 0.5, 0.0))
        return tuple(out)


class NetSampler:
    """layout/parcel_gen sampler interface backed by ParcelNet; lot jitter as in C (measured)."""

    def __init__(self, model, jitter: dict, seed=0):
        self.model = model.eval()
        self.g = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self._jitter = jitter

    def _x(self, country, ctx, nb_known=()):
        nb = np.asarray(nb_known, float)
        return torch.from_numpy(features([country], [ctx["cls"]], [ctx["log_block_area"]], [ctx["interior"]],
                                         [nb.mean() if len(nb) else 0.0], [len(nb) > 0]))

    def lot(self, country, ctx):
        w, d = self.model.sample_lot(self._x(country, ctx), self.g)
        return float(w), float(d)

    def jitter(self, country):
        return self._jitter[country]

    def coverage(self, country, ctx, nb_known):
        return self.model.sample_coverage(self._x(country, ctx, nb_known), self.g)

    def setbacks(self, country, ctx):
        return self.model.sample_setbacks(self._x(country, ctx), self.g)
