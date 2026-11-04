"""TerrainGraphTurtle (Milestone 12): GraphTurtle with terrain channels and a structure head.

Unchanged from GraphTurtle N: queue expansion, CNN crop encoder in the turtle frame, GRU step decoder, the
10-component step mixture, stop and street class, the start head. Added:
* ``n_terrain`` extra crop channels (graphs/terrain_sequence.py) concatenated to the 3 street channels, and the same
  channels over the whole tile for the start head;
* a STRUCTURE head on every emitted step: ground / bridge / tunnel, trained from OSM tags (struct_label; unknown
  labels are masked). It is a learned decision; feasibility rules may veto it at generation, never set it.
Batch keys: ``tcrops`` (B, n_terrain, 48, 48), ``tcanvas`` (B, n_terrain, 128, 128), ``target_struct`` (B, T) with
-1 = no label. With n_terrain = 0 the model is terrain-blind (N0) but keeps the structure head, so N0 and TN differ
only in their terrain inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from urbanmorph.graphs.sequence import MAX_T
from urbanmorph.models.graph_gen import GraphTurtle, mixture_log_prob, mixture_params

N_STRUCT = 3


class TerrainGraphTurtle(GraphTurtle):
    def __init__(self, hidden: int = 256, n_mix: int = 10, n_classes: int = 4, n_terrain: int = 0,
                 n_struct: int = N_STRUCT):
        super().__init__(hidden, n_mix, turtle=True, n_classes=n_classes)
        self.n_terrain, self.n_struct = n_terrain, n_struct
        if n_terrain:
            self.crop_cnn[0] = nn.Conv2d(3 + n_terrain, 32, 3, padding=1)
            self.start_cnn[0] = nn.Conv2d(1 + n_terrain, 16, 3, padding=1)
        self.struct_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_struct)) if n_struct else None
        self.last_struct_logit = None

    def context(self, crops, scalars, arms, arm_mask, cond=None, tcrops=None):
        x = crops.float()
        if self.n_terrain:
            x = torch.cat([x, tcrops.float()], 1)
        c = self.crop_cnn(x)
        a = (self.arm_mlp(arms) * arm_mask.unsqueeze(-1).float()).sum(1)
        return self.ctx(torch.cat([c, a, scalars], -1))

    def decode_steps(self, ctx, prev, h=None, prev_class=None):
        B, T, _ = prev.shape
        first = torch.zeros(B, T, 1, device=prev.device)
        if h is None:
            h = torch.tanh(self.h0(ctx)).unsqueeze(0)
            first[:, 0] = 1
        pc = prev_class if prev_class is not None else torch.full((B, T), -1, device=prev.device)
        parts = [prev, first, F.one_hot(pc.clamp(min=0), self.n_classes).float() * (pc >= 0).unsqueeze(-1).float()]
        x = torch.cat(parts + [ctx.unsqueeze(1).expand(B, T, ctx.shape[-1])], -1)
        y, h = self.gru(x, h)
        o = self.out(y)
        mix = mixture_params(o[..., 1:1 + 6 * self.K], self.K)
        self.last_struct_logit = self.struct_head(y) if self.struct_head is not None else None
        return o[..., 0], mix, h, o[..., 1 + 6 * self.K:]

    def pop_nll(self, batch) -> torch.Tensor:
        ctx = self.context(batch["crops"], batch["scalars"], batch["arms"], batch["arm_mask"], tcrops=batch.get("tcrops"))
        tgt, n = batch["targets"], batch["n_targets"]
        B = tgt.shape[0]
        prev = torch.cat([torch.zeros(B, 1, 2, device=tgt.device), tgt], 1)
        tc = batch["target_class"]
        prev_c = torch.cat([torch.full((B, 1), -1, device=tgt.device, dtype=tc.dtype), tc], 1)
        stop_logit, (lp, mu, ls, rho), _, cls_logit = self.decode_steps(ctx, prev, prev_class=prev_c)
        steps = torch.arange(MAX_T + 1, device=tgt.device)[None]
        is_step, is_stop = steps < n[:, None], steps == n[:, None]
        tgt_pad = torch.cat([tgt, torch.zeros(B, 1, 2, device=tgt.device)], 1)
        ll = mixture_log_prob(tgt_pad, lp, mu, ls, rho)
        nll = (is_step * (F.softplus(stop_logit) - ll)).sum(1) + (is_stop * F.softplus(-stop_logit)).sum(1)
        tcp = torch.cat([tc, torch.zeros(B, 1, device=tgt.device, dtype=tc.dtype)], 1)
        ce = F.cross_entropy(cls_logit.reshape(-1, self.n_classes), tcp.reshape(-1), reduction="none").view(B, -1)
        nll = nll + (is_step * ce).sum(1)
        if self.struct_head is not None and "target_struct" in batch:
            ts = torch.cat([batch["target_struct"], torch.full((B, 1), -1, device=tgt.device, dtype=tc.dtype)], 1)
            m = is_step & (ts >= 0)
            sce = F.cross_entropy(self.last_struct_logit.reshape(-1, self.n_struct), ts.clamp(min=0).reshape(-1),
                                  reduction="none").view(B, -1)
            nll = nll + (m * sce).sum(1)
        return nll

    def start_outputs(self, canvas, ncomp, cond=None, tcanvas=None):
        x = canvas.float()
        if self.n_terrain:
            x = torch.cat([x, tcanvas.float()], 1)
        f = self.start_cnn(x)
        loc = self.start_loc(f).flatten(1)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3)), ncomp[:, None] / 10], -1)
        return self.start_stop(pooled).squeeze(-1), loc

    def start_nll(self, batch) -> torch.Tensor:
        import math
        from urbanmorph.models.graph_gen import CELL_AREA_M2
        stop_logit, loc = self.start_outputs(batch["canvas"], batch["ncomp"], tcanvas=batch.get("tcanvas"))
        s = batch["stop"]
        loc_nll = -F.log_softmax(loc, -1).gather(1, batch["cell"][:, None]).squeeze(1) + math.log(CELL_AREA_M2 / 900)
        return s * F.softplus(-stop_logit) + (1 - s) * (F.softplus(stop_logit) + loc_nll)
