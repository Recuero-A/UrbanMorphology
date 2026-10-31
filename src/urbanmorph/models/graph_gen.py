"""NTG-inspired street-graph generator (Milestone 5).

Inspired by Neural Turtle Graphics (Chu et al., ICCV 2019): roads grow from a queue of nodes,
and for each popped node a recurrent decoder emits new street steps one at a time. Simplified
here: the "encoder" of NTG (RNN over incoming paths) is replaced by a CNN over the local canvas of
roads already drawn plus a set encoder over the node's existing edges ("arms").

Probability model of one tile (exact log-likelihood, every factor normalised):

    log p(G) = sum_components [ log p(continue | canvas) + log p(start location | canvas) ]
             + log p(STOP | final canvas)
             + sum_pops sum_k [ log p(not stop) + log p(step_k | context, steps_<k) ] + log p(STOP)

* step_k = displacement (dx, dy) / 30 m: mixture of K bivariate Gaussians with correlation
  (no preferred angle or length is imposed).
* start location: categorical over 32 x 32 cells of 16 m, uniform inside a cell (a valid density).

Milestone 6 options (defaults reproduce Milestone 5 exactly):
* ``turtle=True``: inputs and steps in the frame of the incoming street (see sequence.py); one extra
  scalar (start-node flag). The mixture then describes turn angle and length jointly; a rotation
  has Jacobian 1, so its log-likelihood is directly comparable with the absolute-frame model.
* ``n_classes=4``: every step also predicts the street class of the new edge (categorical), and the
  previous step's class is fed back to the GRU.

Milestone 10 option: ``n_cond`` > 0 adds a conditioning vector (e.g. a one-hot city) to the step context
and to the start head's pooled features (batch key "cond"; default 0 = Milestone 6 exactly).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.graphs.sequence import CELL, GRID, MAX_ARMS, MAX_T, N_SCALARS, N_SCALARS_TURTLE

N_CELLS = (GRID // CELL) ** 2
CELL_AREA_M2 = (CELL * 4.0) ** 2


def mixture_params(out: torch.Tensor, K: int):
    logit_pi, mx, my, lsx, lsy, rho = out.split(K, dim=-1)
    return (logit_pi, torch.stack([mx, my], -1), torch.stack([lsx.clamp(-5, 3), lsy.clamp(-5, 3)], -1),
            0.95 * torch.tanh(rho))


def mixture_log_prob(x: torch.Tensor, logit_pi, mu, log_s, rho) -> torch.Tensor:
    """log density of x (..., 2) under the mixture of bivariate Gaussians (in the 30 m units)."""
    z = (x.unsqueeze(-2) - mu) / log_s.exp()                       # (..., K, 2)
    one_m = 1 - rho ** 2
    quad = (z[..., 0] ** 2 + z[..., 1] ** 2 - 2 * rho * z[..., 0] * z[..., 1]) / one_m
    comp = -math.log(2 * math.pi) - log_s.sum(-1) - 0.5 * torch.log(one_m) - 0.5 * quad
    return torch.logsumexp(F.log_softmax(logit_pi, -1) + comp, -1)


def mixture_sample(logit_pi, mu, log_s, rho, generator=None) -> torch.Tensor:
    k = torch.multinomial(F.softmax(logit_pi, -1).cpu(), 1, generator=generator).squeeze(-1).to(mu.device)
    idx = torch.arange(len(k), device=mu.device)
    m, s, r = mu[idx, k], log_s[idx, k].exp(), rho[idx, k]
    e = torch.randn(len(k), 2, generator=generator).to(mu.device)
    z0 = e[:, 0]
    z1 = r * e[:, 0] + torch.sqrt(1 - r ** 2) * e[:, 1]
    return m + s * torch.stack([z0, z1], -1)


class GraphTurtle(nn.Module):
    def __init__(self, hidden: int = 256, n_mix: int = 10, turtle: bool = False, n_classes: int = 0, n_cond: int = 0):
        super().__init__()
        self.K = n_mix
        self.turtle = turtle
        self.n_classes = n_classes
        self.n_cond = n_cond
        n_scal = N_SCALARS_TURTLE if turtle else N_SCALARS
        self.crop_cnn = nn.Sequential(                                 # 3 x 48 x 48
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),      # 24
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),      # 12
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),      # 6
            nn.Flatten(), nn.Linear(64 * 6 * 6, hidden), nn.ReLU())
        self.arm_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(hidden + 64 + n_scal + n_cond, hidden), nn.ReLU())
        self.h0 = nn.Linear(hidden, hidden)
        self.gru = nn.GRU(3 + n_classes + hidden, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1 + 6 * n_mix + n_classes))
        # start head over the whole tile (1 x 128 x 128)
        self.start_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),      # 64
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),      # 32
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.start_loc = nn.Conv2d(32, 1, 1)
        self.start_stop = nn.Sequential(nn.Linear(64 + 1 + n_cond, 64), nn.ReLU(), nn.Linear(64, 1))
        if n_cond:
            self.start_cond = nn.Linear(n_cond, 32)                    # per-channel shift of the start features

    # ---------------------------------------------------------------- per-pop step decoder
    def context(self, crops, scalars, arms, arm_mask, cond=None):
        c = self.crop_cnn(crops.float())
        a = (self.arm_mlp(arms) * arm_mask.unsqueeze(-1).float()).sum(1)
        parts = [c, a, scalars] + ([cond.float()] if self.n_cond else [])
        return self.ctx(torch.cat(parts, -1))

    def decode_steps(self, ctx, prev, h=None, prev_class=None):
        """prev: (B, T, 2) previous displacement per step (zeros at step 0); prev_class: (B, T) previous
        step's street class (-1 = none). Returns stop logits, mixture, h (and class logits if n_classes)."""
        B, T, _ = prev.shape
        first = torch.zeros(B, T, 1, device=prev.device)
        if h is None:
            h = torch.tanh(self.h0(ctx)).unsqueeze(0)
            first[:, 0] = 1
        parts = [prev, first]
        if self.n_classes:
            pc = prev_class if prev_class is not None else torch.full((B, T), -1, device=prev.device)
            parts.append(F.one_hot(pc.clamp(min=0), self.n_classes).float() * (pc >= 0).unsqueeze(-1).float())
        x = torch.cat(parts + [ctx.unsqueeze(1).expand(B, T, ctx.shape[-1])], -1)
        y, h = self.gru(x, h)
        o = self.out(y)
        mix = mixture_params(o[..., 1:1 + 6 * self.K], self.K)
        if self.n_classes:
            return o[..., 0], mix, h, o[..., 1 + 6 * self.K:]
        return o[..., 0], mix, h

    def pop_nll(self, batch) -> torch.Tensor:
        """Negative log-likelihood of each pop's step sequence + STOP, (B,)."""
        ctx = self.context(batch["crops"], batch["scalars"], batch["arms"], batch["arm_mask"], batch.get("cond"))
        tgt, n = batch["targets"], batch["n_targets"]
        B = tgt.shape[0]
        prev = torch.cat([torch.zeros(B, 1, 2, device=tgt.device), tgt], 1)        # T = MAX_T + 1
        if self.n_classes:
            tc = batch["target_class"]
            prev_c = torch.cat([torch.full((B, 1), -1, device=tgt.device, dtype=tc.dtype), tc], 1)
            stop_logit, (lp, mu, ls, rho), _, cls_logit = self.decode_steps(ctx, prev, prev_class=prev_c)
        else:
            stop_logit, (lp, mu, ls, rho), _ = self.decode_steps(ctx, prev)
        steps = torch.arange(MAX_T + 1, device=tgt.device)[None]
        is_step = steps < n[:, None]
        is_stop = steps == n[:, None]
        tgt_pad = torch.cat([tgt, torch.zeros(B, 1, 2, device=tgt.device)], 1)
        ll_disp = mixture_log_prob(tgt_pad, lp, mu, ls, rho)
        nll = (is_step * (F.softplus(stop_logit) - ll_disp)).sum(1) + (is_stop * F.softplus(-stop_logit)).sum(1)
        if self.n_classes:
            tcp = torch.cat([tc, torch.zeros(B, 1, device=tgt.device, dtype=tc.dtype)], 1)
            ce = F.cross_entropy(cls_logit.reshape(-1, self.n_classes), tcp.reshape(-1), reduction="none").view(B, -1)
            nll = nll + (is_step * ce).sum(1)
        return nll

    # ---------------------------------------------------------------- start head
    def start_outputs(self, canvas, ncomp, cond=None):
        f = self.start_cnn(canvas.float())
        if self.n_cond:
            f = torch.relu(f + self.start_cond(cond.float())[:, :, None, None])
        loc = self.start_loc(f).flatten(1)                                        # (B, 1024)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3)), ncomp[:, None] / 10]
                           + ([cond.float()] if self.n_cond else []), -1)
        return self.start_stop(pooled).squeeze(-1), loc

    def start_nll(self, batch) -> torch.Tensor:
        stop_logit, loc = self.start_outputs(batch["canvas"], batch["ncomp"], batch.get("cond"))
        s = batch["stop"]
        loc_nll = -F.log_softmax(loc, -1).gather(1, batch["cell"][:, None]).squeeze(1) + math.log(CELL_AREA_M2 / 900)
        return s * F.softplus(-stop_logit) + (1 - s) * (F.softplus(stop_logit) + loc_nll)
