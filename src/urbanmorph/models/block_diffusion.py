"""Milestone 10: BlockGen, a small conditional DDPM over the building mask of one block window
(docs/M10_BLOCK_GENERATION.md §4).

x0 = building mask in {-1, +1} (128 x 128, 2 m pixels, canonical block frame). Conditioning channels:
block mask, known buildings, known-region mask (inpainting / outpainting); a learned city embedding is added
to the time embedding. epsilon-prediction, cosine noise schedule (Nichol & Dhariwal 2021), T = 1000 steps;
sampling with DDIM (Song et al. 2021) over a subsequence of steps. The known region is pasted back after
every step (a hard constraint), and pixels outside the block are set to "no building" at the end.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

T_STEPS = 1000


def cosine_alphas_bar(T: int = T_STEPS, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab[1:].float()                                   # alpha_bar at t = 1..T (index t-1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.e(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class BlockUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 128), n_cities: int = 9, emb: int = 128, in_cond: int = 3):
        super().__init__()
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.city = nn.Embedding(n_cities, emb)
        self.inp = nn.Conv2d(1 + in_cond, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        c = ch[0]
        skips = [c]
        for i, co in enumerate(ch):
            self.down.append(nn.ModuleList([ResBlock(c, co, emb), ResBlock(co, co, emb),
                                            nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity()]))
            c = co
            skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList([ResBlock(c + skips.pop(), co, emb), ResBlock(co, co, emb),
                                          nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity()]))
            c = co
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, cond, city):
        e = self.t_mlp(timestep_embedding(t, self.emb)) + self.city(city)
        h = self.inp(torch.cat([x, cond], 1))
        hs = []
        for r1, r2, dn in self.down:
            h = r2(r1(h, e), e)
            hs.append(h)
            h = dn(h)
        for m in self.mid:
            h = m(h, e)
        for r1, r2, upm in self.up:
            h = r2(r1(torch.cat([h, hs.pop()], 1), e), e)
            h = upm(h)
        return self.out(h)


class BlockDiffusion(nn.Module):
    def __init__(self, n_cities: int = 9, ch=(32, 64, 128, 128)):
        super().__init__()
        self.net = BlockUNet(ch, n_cities)
        self.register_buffer("ab", cosine_alphas_bar())

    def loss(self, x0, cond, city, weight):
        """x0 (B,1,H,W) in {-1,1}; weight (B,1,H,W): per-pixel loss weight."""
        B = x0.shape[0]
        t = torch.randint(0, T_STEPS, (B,), device=x0.device)
        ab = self.ab[t][:, None, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = self.net(xt, t, cond, city)
        return ((pred - eps) ** 2 * weight).sum() / weight.sum()

    @torch.no_grad()
    def sample(self, cond, city, steps: int = 50, eta: float = 1.0, generator=None):
        """cond = (block, known buildings in {0,1}, known mask); returns the {0,1} building mask (B,1,H,W)."""
        block, known_b, known_m = cond[:, :1], cond[:, 1:2], cond[:, 2:3]
        x0_known = known_b * 2 - 1
        dev = cond.device
        x = torch.randn(cond.shape[0], 1, *cond.shape[2:], generator=generator).to(dev)
        ts = torch.linspace(T_STEPS - 1, 0, steps).long().tolist()
        for i, t in enumerate(ts):
            ab = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            # hard constraint: known region follows the forward process of the known x0
            noise_k = torch.randn(x.shape, generator=generator).to(dev)
            x = known_m * (ab.sqrt() * x0_known + (1 - ab).sqrt() * noise_k) + (1 - known_m) * x
            eps = self.net(x, torch.full((x.shape[0],), t, device=dev), cond, city)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
            dir_ = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_ + sigma * torch.randn(x.shape, generator=generator).to(dev)
        x = known_m * x0_known + (1 - known_m) * x
        return ((x > 0).float() * block)
