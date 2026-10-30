"""Evaluation of reconstructions, samples and latent space.

Raster metrics (BCE, IoU, coverage) measure pixel agreement. They do NOT
establish urban realism: a blurry average city can score decent IoU while
having disconnected streets. Hence the extra structural checks (road
connectivity), diversity, and a memorisation check (nearest training tile).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from urbanmorph.models.data import dihedral

CHANNELS = ("roads", "buildings", "green")
EPS = 1e-7


@torch.no_grad()
def predict(model, x: torch.Tensor, device, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic reconstruction from the posterior mean. Returns (probs, mu)."""
    model.eval()
    probs, mus = [], []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        logits, mu, _ = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        mus.append(mu.cpu().numpy())
    return np.concatenate(probs), np.concatenate(mus)


@torch.no_grad()
def decode(model, z: np.ndarray, device) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model.decoder(torch.from_numpy(z).float().to(device))).cpu().numpy()


def channel_bce(probs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unweighted mean BCE per channel (nats per pixel), shape (3,)."""
    p = np.clip(probs, EPS, 1 - EPS)
    bce = -(target * np.log(p) + (1 - target) * np.log(1 - p))
    return bce.mean(axis=(0, 2, 3))


def iou_per_tile(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """IoU per tile and channel, shape (N, 3). Empty prediction AND empty target counts as 1."""
    p, t = pred.astype(bool), target.astype(bool)
    inter = (p & t).sum(axis=(2, 3))
    union = (p | t).sum(axis=(2, 3))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def coverage_errors(pred_mask: np.ndarray, target: np.ndarray) -> dict:
    pc, tc = pred_mask.mean(axis=(2, 3)), target.mean(axis=(2, 3))
    return {"abs": np.abs(pc - tc).mean(axis=0), "bias": (pc - tc).mean(axis=0)}


def reconstruction_metrics(probs: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    mask = probs >= threshold
    ious = iou_per_tile(mask, target)
    cov = coverage_errors(mask, target)
    out = {}
    for c, name in enumerate(CHANNELS):
        out[name] = {"bce": float(channel_bce(probs, target)[c]), "iou": float(ious[:, c].mean()),
                     "coverage_abs_error": float(cov["abs"][c]), "coverage_bias": float(cov["bias"][c])}
    return out


def trivial_baseline(train: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    """Predict each channel's mean training coverage everywhere (a constant probability)."""
    p = train.mean(axis=(0, 2, 3))
    probs = np.broadcast_to(p.reshape(1, 3, 1, 1), target.shape).astype(np.float64)
    out = reconstruction_metrics(probs, target, threshold)
    # coverage error of predicting the mean coverage directly (fairer than a thresholded mask)
    tc = target.mean(axis=(2, 3))
    for c, name in enumerate(CHANNELS):
        out[name]["coverage_abs_error_mean_prediction"] = float(np.abs(tc[:, c] - p[c]).mean())
    return out


# ------------------------------------------------------------- road connectivity
def label_components(mask: np.ndarray) -> np.ndarray:
    """8-connected component labels of a 2-D boolean mask (0 = background)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    nxt = 0
    for i, j in zip(*np.nonzero(mask)):
        if labels[i, j]:
            continue
        nxt += 1
        labels[i, j] = nxt
        q = deque([(i, j)])
        while q:
            a, b = q.popleft()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    y, x = a + da, b + db
                    if 0 <= y < h and 0 <= x < w and mask[y, x] and not labels[y, x]:
                        labels[y, x] = nxt
                        q.append((y, x))
    return labels


def road_connectivity(road_masks: np.ndarray, min_size: int = 10) -> dict:
    """Fragmentation of road masks: number of components and share of road pixels in the largest one.

    Real street networks inside a 512 m window are mostly one connected piece;
    many small fragments indicate broken streets.
    """
    n_comp, largest, small = [], [], []
    for m in road_masks.astype(bool):
        lab = label_components(m)
        sizes = np.bincount(lab.ravel())[1:]
        if sizes.size == 0:
            n_comp.append(0), largest.append(np.nan), small.append(0)
            continue
        n_comp.append(int((sizes >= min_size).sum()))
        small.append(int((sizes < min_size).sum()))
        largest.append(float(sizes.max() / sizes.sum()))
    return {"components_median": float(np.median(n_comp)), "largest_component_share_median": float(np.nanmedian(largest)),
            "tiny_fragments_median": float(np.median(small))}


# ------------------------------------------------------------- diversity / memorisation
def mean_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean-over-channels IoU between every tile in a (N) and b (M): (N, M). Inputs boolean."""
    a2 = a.reshape(len(a), 3, -1).astype(np.float32)
    b2 = b.reshape(len(b), 3, -1).astype(np.float32)
    inter = np.einsum("ncp,mcp->nmc", a2, b2)
    sa, sb = a2.sum(-1), b2.sum(-1)
    union = sa[:, None, :] + sb[None, :, :] - inter
    iou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return iou.mean(-1)


def diversity(masks: np.ndarray) -> float:
    """Mean pairwise layout distance (1 - mean channel IoU)."""
    m = mean_iou_matrix(masks, masks)
    iu = np.triu_indices(len(masks), 1)
    return float((1 - m[iu]).mean())


def nearest_train_similarity(masks: np.ndarray, train: np.ndarray) -> np.ndarray:
    """For each mask, the highest mean IoU with any training tile under any of the 8 symmetries."""
    t = torch.from_numpy(train.astype(np.uint8))
    variants = np.concatenate([dihedral(t, k).numpy() for k in range(8)]).astype(bool)
    return mean_iou_matrix(masks.astype(bool), variants).max(axis=1)


# ------------------------------------------------------------- latent space
def knn_accuracy(train_z, train_y, test_z, test_y, k: int = 5) -> float:
    mu, sd = train_z.mean(0), train_z.std(0) + 1e-8
    a, b = (train_z - mu) / sd, (test_z - mu) / sd
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    train_y = np.asarray(train_y)
    pred = []
    for row in nn:
        vals, counts = np.unique(train_y[row], return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return float((np.asarray(pred) == np.asarray(test_y)).mean())


def ridge_r2(train_z, train_t, test_z, test_t, lam: float = 1.0) -> float:
    """R^2 on held-out tiles of a ridge regression from latent means to one scalar metric."""
    mu, sd = train_z.mean(0), train_z.std(0) + 1e-8
    a = np.hstack([(train_z - mu) / sd, np.ones((len(train_z), 1))])
    b = np.hstack([(test_z - mu) / sd, np.ones((len(test_z), 1))])
    reg = lam * np.eye(a.shape[1])
    reg[-1, -1] = 0
    w = np.linalg.solve(a.T @ a + reg, a.T @ train_t)
    pred = b @ w
    ss_res = ((test_t - pred) ** 2).sum()
    ss_tot = ((test_t - test_t.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)
