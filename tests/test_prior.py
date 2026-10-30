"""Milestone 4: autoregressive prior - masks (no future access, no blind spot), density, sampling."""

import math

import pytest
import torch

from urbanmorph.models.prior import GatedPixelCNNPrior


def jacobian_dependence(model, D=4, H=8, W=8):
    """dep[p, q] = True if the mixture parameters at position p depend on the input at position q."""
    torch.manual_seed(0)
    z = torch.randn(1, D, H, W, requires_grad=True)
    logit_pi, mu, log_s = model.params(z)
    dep = torch.zeros(H * W, H * W, dtype=torch.bool)
    for p in range(H * W):
        i, j = divmod(p, W)
        out = logit_pi[0, :, i, j].sum() + mu[0, :, :, i, j].sum() + log_s[0, :, :, i, j].sum()
        g, = torch.autograd.grad(out, z, retain_graph=True)
        dep[p] = (g[0].abs().sum(0) > 0).flatten()
    return dep


def test_no_future_access_and_no_blind_spot():
    # float64 + default initialisation: large artificial weights would saturate tanh/sigmoid and
    # make gradients underflow to exactly 0, which looks like a (false) blind spot.
    torch.manual_seed(0)
    model = GatedPixelCNNPrior(latent_channels=4, hidden=16, n_layers=6, n_mix=2).double()
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dep = jacobian_dependence(model)
    finally:
        torch.set_default_dtype(old)
    n = dep.shape[0]
    future = torch.triu(torch.ones(n, n, dtype=torch.bool))       # q >= p (including itself)
    past = ~future
    assert not (dep & future).any(), "a position sees itself or a future position"
    assert dep[past].all(), "blind spot: some earlier position is not visible"


def test_log_prob_matches_single_gaussian_by_hand():
    torch.manual_seed(1)
    model = GatedPixelCNNPrior(latent_channels=3, hidden=8, n_layers=2, n_mix=1)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.zero_()          # pi = 1, mean 0, log-scale 0 -> standard normal everywhere
    z = torch.randn(2, 3, 8, 8)
    expected = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(dim=(1, 2, 3))
    assert torch.allclose(model.log_prob(z), expected, atol=1e-4)
    # standardisation is accounted for with its Jacobian: scaling the data by 2 shifts log p by -log 2 per element
    model.set_normalisation(torch.zeros(3), torch.full((3,), 2.0))
    expected2 = (-0.5 * (z / 2) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(2.0)).sum(dim=(1, 2, 3))
    assert torch.allclose(model.log_prob(z), expected2, atol=1e-4)


def test_training_step_lowers_nll_and_sampling_shapes():
    torch.manual_seed(2)
    model = GatedPixelCNNPrior(latent_channels=4, hidden=16, n_layers=6, n_mix=3)
    # data with strong spatial structure: every position equals the first one
    base = torch.randn(64, 4, 1, 1)
    data = base.expand(64, 4, 8, 8).contiguous() + 0.05 * torch.randn(64, 4, 8, 8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = -model.log_prob(data).mean().item()
    for _ in range(150):
        loss = -model.log_prob(data).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 100          # learns that later positions copy earlier ones
    g = torch.Generator().manual_seed(0)
    s = model.sample(5, generator=g)
    assert s.shape == (5, 4, 8, 8) and torch.isfinite(s).all()
    # samples reproduce the learned dependence: positions are close to the first position
    assert (s - s[:, :, :1, :1]).abs().mean() < (torch.randn(5, 4, 8, 8)).abs().mean()
