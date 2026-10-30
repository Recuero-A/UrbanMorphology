"""Soft-clDice (Shit et al., CVPR 2021) for the road channel.

Reference: S. Shit, J. C. Paetzold, et al., "clDice - A Novel Topology-Preserving
Loss Function for Tubular Structure Segmentation", CVPR 2021, arXiv:2003.07311.
The soft skeleton below reproduces the authors' official PyTorch implementation
(github.com/jocpae/clDice, cldice_loss/pytorch/soft_skeleton.py, MIT licence)
operation for operation:

    soft_erode  = min( -maxpool(-x, 3x1), -maxpool(-x, 1x3) )   (morphological erosion, cross element)
    soft_dilate = maxpool(x, 3x3)
    soft_open   = dilate(erode(x))
    skel_0      = relu(x - open(x))
    repeat k times:  x = erode(x);  delta = relu(x - open(x));  skel += relu(delta - skel*delta)

clDice compares SKELETONS with VOLUMES (masks):
    Tprec = (|S_pred . V_true| + 1) / (|S_pred| + 1)   - how much of the predicted centreline lies on true roads
    Tsens = (|S_true . V_pred| + 1) / (|S_true| + 1)   - how much of the true centreline is covered by the prediction
    clDice = 2 Tprec Tsens / (Tprec + Tsens);  loss = 1 - clDice

Why it targets connectivity: a gap in a predicted road removes a piece of the TRUE
centreline from the prediction (lower Tsens), and a spurious predicted branch adds
centreline outside true roads (lower Tprec), regardless of how few pixels are involved.

Deliberate deviation from the official code: the sums run PER TILE (then averaged
over the batch) instead of over the whole batch, so tiles with long road networks do
not dominate. With batch size 1 the two are identical (tested).
The +1 smoothing makes empty masks safe: empty prediction vs empty target -> loss 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """Differentiable skeleton of a (B, C, H, W) probability map (official recipe, 10 iterations)."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor, num_iter: int = 10, smooth: float = 1.0) -> torch.Tensor:
    """Per-tile soft-clDice loss, shape (B,). ``pred`` are probabilities, ``target`` binary, both (B, 1, H, W)."""
    skel_pred = soft_skel(pred, num_iter)
    skel_true = soft_skel(target, num_iter)
    dims = (1, 2, 3)
    tprec = ((skel_pred * target).sum(dims) + smooth) / (skel_pred.sum(dims) + smooth)
    tsens = ((skel_true * pred).sum(dims) + smooth) / (skel_true.sum(dims) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
