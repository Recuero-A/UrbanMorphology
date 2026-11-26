"""soft-clDice: equivalence with the official implementation, empty masks, gradients, topology sensitivity."""

import pytest
import torch
import torch.nn.functional as F

from urbanmorph.models.cldice import soft_cldice_loss, soft_skel
from urbanmorph.models.losses import combined_reconstruction, soft_dice
from urbanmorph.models.vae import SpatialVAE


# --- reference: official clDice PyTorch code (github.com/jocpae/clDice, cldice_loss/pytorch, MIT licence,
# (c) 2021 J. C. Paetzold and S. Shit), copied verbatim in behaviour for comparison.
def _ref_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _ref_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _ref_skel(img, num_iter=10):
    img1 = _ref_dilate(_ref_erode(img))
    skel = F.relu(img - img1)
    for _ in range(num_iter):
        img = _ref_erode(img)
        img1 = _ref_dilate(_ref_erode(img))
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def _ref_cldice(y_true, y_pred, smooth=1.0):
    skel_pred, skel_true = _ref_skel(y_pred), _ref_skel(y_true)
    tprec = (torch.sum(skel_pred * y_true) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * y_pred) + smooth) / (torch.sum(skel_true) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def road(width=4, gap=False, size=64):
    m = torch.zeros(1, 1, size, size)
    m[0, 0, 30:30 + width, 4:60] = 1
    if gap:
        m[0, 0, :, 30:36] = 0
    return m


def test_matches_official_implementation_for_batch_of_one():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        pred = torch.rand(1, 1, 64, 64, generator=g)
        true = (torch.rand(1, 1, 64, 64, generator=g) > 0.8).float()
        assert torch.allclose(soft_skel(pred), _ref_skel(pred))
        assert soft_cldice_loss(pred, true).item() == pytest.approx(_ref_cldice(true, pred).item(), rel=1e-6)


def test_per_tile_averaging_differs_from_batch_sum_only_by_design():
    a, b = road(), road(width=8)
    batch_pred = torch.cat([a, b * 0.5])
    batch_true = torch.cat([a, b])
    per_tile = soft_cldice_loss(batch_pred, batch_true)
    assert per_tile.shape == (2,)
    assert per_tile[0].item() == pytest.approx(_ref_cldice(a, a).item())


def test_empty_masks_are_safe():
    empty = torch.zeros(2, 1, 32, 32)
    assert torch.allclose(soft_cldice_loss(empty, empty), torch.zeros(2))  # nothing to find, nothing predicted
    hallucinated = soft_cldice_loss(road(size=32) * 0.9 + 0.0, torch.zeros(1, 1, 32, 32))
    assert hallucinated.item() > 0.5                                           # predicting roads where none exist costs
    missed = soft_cldice_loss(torch.zeros(1, 1, 64, 64), road())
    assert missed.item() > 0.5                                                 # missing all roads costs
    assert torch.isfinite(hallucinated) and torch.isfinite(missed)


def test_skeleton_is_thin_centreline():
    bar = road(width=6)
    s = (soft_skel(bar) > 0.5).float()
    assert s.sum() > 40                       # follows the bar along its length
    assert s.sum(dim=2).max() <= 2            # at most ~1-2 pixels across the bar


def test_broken_road_costs_more_under_cldice_than_under_dice():
    true = road(width=4)
    broken = road(width=4, gap=True)
    thinner = road(width=4).clone()
    thinner[0, 0, 30, 4:28] = 0               # remove exactly as many pixels as the gap (24), road stays connected
    removed_gap = (true - broken).sum().item()
    removed_thin = (true - thinner).sum().item()
    assert abs(removed_gap - removed_thin) <= 0.25 * removed_gap      # similar pixel error
    dice_gap = 1 - soft_dice(torch.logit(broken.clamp(1e-4, 1 - 1e-4)), true)[0, 0]
    dice_thin = 1 - soft_dice(torch.logit(thinner.clamp(1e-4, 1 - 1e-4)), true)[0, 0]
    assert abs(dice_gap - dice_thin) < 0.02                            # Dice barely distinguishes them
    assert soft_cldice_loss(broken, true).item() > soft_cldice_loss(thinner, true).item()  # clDice does


def test_roads_only_and_gradients_reach_the_encoder():
    torch.manual_seed(0)
    m = SpatialVAE(8)
    x = (torch.rand(2, 3, 128, 128) > 0.8).float()
    logits, mu, logvar = m(x)
    base, base_ch = combined_reconstruction(logits, x)
    total, ch = combined_reconstruction(logits, x, cldice_weight=1.0)
    assert ch[0] > base_ch[0]
    assert torch.allclose(ch[1:], base_ch[1:])          # buildings and green untouched
    (ch[0] - base_ch[0]).backward()
    enc_grad = sum(p.grad.abs().sum() for p in m.encoder_features.parameters() if p.grad is not None)
    assert enc_grad > 0                                   # the clDice term alone moves the encoder
