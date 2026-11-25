"""Milestone 9: the four small networks of tidy subdivision (all MLPs, exact likelihoods).

* type:   block features (10) -> block type {none, single, few, rows} (categorical);
* side:   block features + type + side length + side street class (19) -> number of street lots N in
          0 .. N_MAX (categorical) and rear ratio = (lot rear distance from the edge) / (block depth),
          mixture of 4 Gaussians on logit(ratio clipped to [0.02, 0.98]);
* zone:   parcel features (10, incl. the mean green label of already-assigned neighbours) -> green (Bernoulli);
* shape:  lot features (10) -> template class (layout/templates.py; categorical).
Inputs are standardised with fixed constants (below); trained on OBSERVED train parcels
(scripts/m9_records.py, scripts/m9_train.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.layout.templates import N_TEMPLATES
from urbanmorph.models.parcel_net import mix1_log_prob, mix1_params, mix1_sample

N_MAX = 20
TYPES = ("none", "single", "few", "rows")


def _onehot(i, k):
    i = np.asarray(i, int)
    out = np.zeros((len(i), k), np.float32)
    out[np.arange(len(i)), np.clip(i, 0, k - 1)] = 1
    return out


def _country(c):
    return _onehot((np.asarray(c) == "united_states").astype(int), 2)


def block_x(country, log_area, rect, compact, n_edges, cls):
    return np.c_[_country(country), (np.asarray(log_area) - 8.0) / 1.5, rect, compact, np.asarray(n_edges) / 10.0,
                 _onehot(cls, 4)].astype(np.float32)


def side_x(bx, btype, log_len, side_cls):
    return np.c_[bx, _onehot(btype, 4), (np.asarray(log_len) - 3.0), _onehot(side_cls, 4)].astype(np.float32)


def zone_x(country, log_area, interior, btype, nb_green, has_nb):
    return np.c_[_country(country), (np.asarray(log_area) - 5.5) / 1.2, np.asarray(interior, float), _onehot(btype, 4),
                 nb_green, np.asarray(has_nb, float)].astype(np.float32)


def shape_x(country, log_w, log_d, coverage, interior, cls):
    return np.c_[_country(country), (np.asarray(log_w) - 2.5) / 0.6, (np.asarray(log_d) - 3.0) / 0.6, coverage,
                 np.asarray(interior, float), _onehot(cls, 4)].astype(np.float32)


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class BlockNets(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.type_net = mlp(10, 4, hidden)
        self.side_net = mlp(19, N_MAX + 1 + 3 * 4, hidden)
        self.zone_net = mlp(10, 1, hidden)
        self.shape_net = mlp(10, N_TEMPLATES, hidden)

    # ---------------------------------------------------------------- likelihoods (per row)
    def nll(self, net, b) -> torch.Tensor:
        if net == "type":
            return F.cross_entropy(self.type_net(b["x"]), b["y"], reduction="none")
        if net == "side":
            o = self.side_net(b["x"])
            nll_n = F.cross_entropy(o[:, :N_MAX + 1], b["N"], reduction="none")
            r = torch.logit(b["ratio"].clamp(0.02, 0.98))
            nll_r = -mix1_log_prob(r, *mix1_params(o[:, N_MAX + 1:], 4)) * b["m_ratio"]
            return nll_n + nll_r
        if net == "zone":
            return F.binary_cross_entropy_with_logits(self.zone_net(b["x"]).squeeze(1), b["y"], reduction="none")
        return F.cross_entropy(self.shape_net(b["x"]), b["y"], reduction="none")

    # ---------------------------------------------------------------- sampling (one row at a time)
    @torch.no_grad()
    def sample_type(self, x, g) -> int:
        return int(torch.multinomial(F.softmax(self.type_net(torch.from_numpy(x)), -1), 1, generator=g).item())

    @torch.no_grad()
    def sample_side(self, x, g) -> tuple[int, float]:
        o = self.side_net(torch.from_numpy(x))
        n = int(torch.multinomial(F.softmax(o[:, :N_MAX + 1], -1), 1, generator=g).item())
        r = float(torch.sigmoid(mix1_sample(*mix1_params(o[:, N_MAX + 1:], 4), generator=g)).item())
        return n, r

    @torch.no_grad()
    def p_green(self, x) -> float:
        return float(torch.sigmoid(self.zone_net(torch.from_numpy(x))).item())

    @torch.no_grad()
    def sample_shape(self, x, g) -> int:
        return int(torch.multinomial(F.softmax(self.shape_net(torch.from_numpy(x)), -1), 1, generator=g).item())
